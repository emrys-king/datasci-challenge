
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

PARAM_NAMES = {
    "norm": ["loc", "scale"],
    "laplace": ["loc", "scale"],
    "cauchy": ["loc", "scale"],
    "t": ["df", "loc", "scale"],
    "logistic": ["loc", "scale"],
    "hypsecant": ["loc", "scale"],
    "gennorm": ["shape", "loc", "scale"],
}


def escape_latex(s):
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def _pluralize(word):
    if word.endswith("y") and len(word) > 1 and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


# ----------------------------------------------------------------
# Core data computation
# ----------------------------------------------------------------
def load_run(summary_csv, all_fits_csv):
    return pd.read_csv(summary_csv), pd.read_csv(all_fits_csv)


def compute_tally(summary_df):
    """Returns (tally: Series dist -> count, sorted descending; total: int; group_col: str)."""
    group_col = summary_df.columns[0]
    tally = summary_df["best_dist"].value_counts()
    return tally, len(summary_df), group_col


def compute_param_stats(summary_df, all_fits_df, min_count=1):
    """
    For each distribution that won at least min_count groups: mean and
    variance of each fitted parameter, computed only across the groups
    where that distribution actually won (not everywhere it was fit as a
    candidate).

    Returns: OrderedDict dist_name -> {"n": int, "params": OrderedDict param_name -> (mean, var)}
    """
    import json
    group_col = summary_df.columns[0]
    tally, _, _ = compute_tally(summary_df)
    common_dists = tally[tally >= min_count].index.tolist()

    out = OrderedDict()
    for dist_name in common_dists:
        winning_groups = summary_df.loc[summary_df["best_dist"] == dist_name, group_col].tolist()
        sub = all_fits_df[(all_fits_df["dist"] == dist_name)
                           & (all_fits_df[group_col].isin(winning_groups))]
        param_arr = np.array(sub["params"].apply(json.loads).tolist())
        names = PARAM_NAMES.get(dist_name, [f"p{i}" for i in range(param_arr.shape[1])])
        n = param_arr.shape[0]

        params = OrderedDict()
        for i, pname in enumerate(names):
            vals = param_arr[:, i]
            mean = float(np.mean(vals))
            var = float(np.var(vals, ddof=1)) if n > 1 else None
            params[pname] = (mean, var)
        out[dist_name] = {"n": n, "params": params}
    return out


# ----------------------------------------------------------------
# Single-run table builders
# ----------------------------------------------------------------
def make_country_fit_table(summary_df, caption="Best-fit ramp-rate distribution by country",
                            label="tab:country-fits"):
    """One row per country/group. Not used by default (see make_distribution_tally_table)."""
    group_col = summary_df.columns[0]
    df = summary_df.sort_values(group_col).reset_index(drop=True)
    has_p0 = "p0_mean" in df.columns
    cols = "lccc" + ("c" if has_p0 else "")

    lines = [r"\begin{table}", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule"]
    header = f"{group_col.capitalize()} & Distribution & AIC & KS $p$-value"
    if has_p0:
        header += r" & $P(\text{ramp}=0)$"
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        line = (f"{escape_latex(row[group_col])} & {escape_latex(row['best_dist'])} & "
                f"{row['AIC']:.1f} & {row['ks_pvalue']:.3f}")
        if has_p0:
            line += f" & {row['p0_mean']:.1%}".replace("%", r"\%")
        lines.append(line + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def make_distribution_tally_table(summary_df, caption="Winning distribution tally across countries",
                                   label="tab:dist-tally"):
    tally, total, group_col = compute_tally(summary_df)
    group_plural = _pluralize(group_col)

    lines = [r"\begin{table}", r"\centering", r"\small",
             r"\begin{tabular}{lrr}", r"\toprule",
             rf"Distribution & Count & \% of {group_plural} \\", r"\midrule"]
    for dist_name, count in tally.items():
        pct = 100 * count / total
        lines.append(f"{escape_latex(dist_name)} & {count} & {pct:.0f}\\% \\\\")
    lines += [r"\midrule", rf"Total & {total} & 100\% \\",
              r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def make_param_summary_table(summary_df, all_fits_df, min_count=1,
                              caption="Fitted parameters of the most common best-fit distributions",
                              label="tab:param-summary"):
    stats = compute_param_stats(summary_df, all_fits_df, min_count=min_count)

    lines = [r"\begin{table}", r"\centering", r"\small",
             r"\begin{tabular}{llrr}", r"\toprule",
             r"Distribution & Parameter & Mean & Variance \\", r"\midrule"]
    for dist_name, info in stats.items():
        dist_label = rf"{escape_latex(dist_name)} ($n={info['n']}$)"
        for i, (pname, (mean, var)) in enumerate(info["params"].items()):
            var_str = f"{var:.4g}" if var is not None else "--"
            lines.append(f"{dist_label if i == 0 else ''} & {escape_latex(pname)} & "
                         f"{mean:.4g} & {var_str} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------
# Multi-run (combined) table builders
# ----------------------------------------------------------------
def make_combined_tally_table(runs, caption="Winning distribution tally across runs",
                               label="tab:dist-tally-combined"):
    """runs: OrderedDict label -> (summary_df, all_fits_df)"""
    per_run = {label: compute_tally(summary_df) for label, (summary_df, _) in runs.items()}

    all_dists = []
    for tally, _, _ in per_run.values():
        for d in tally.index:
            if d not in all_dists:
                all_dists.append(d)
    all_dists.sort(key=lambda d: -sum(t.get(d, 0) for t, _, _ in per_run.values()))

    labels = list(runs.keys())
    cols = "l" + "r" * len(labels)
    lines = [r"\begin{table}", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             "Distribution & " + " & ".join(escape_latex(l) for l in labels) + r" \\",
             r"\midrule"]
    for d in all_dists:
        cells = []
        for l in labels:
            tally, total, _ = per_run[l]
            count = int(tally.get(d, 0))
            pct = 100 * count / total if total else 0
            cells.append(f"{count} ({pct:.0f}\\%)" if count else "--")
        lines.append(f"{escape_latex(d)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Total & " + " & ".join(str(per_run[l][1]) for l in labels) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def make_combined_param_table(runs, min_count=1,
                               caption="Fitted parameters across runs (mean (variance))",
                               label="tab:param-summary-combined"):
    """runs: OrderedDict label -> (summary_df, all_fits_df)"""
    per_run = {label: compute_param_stats(summary_df, all_fits_df, min_count=min_count)
               for label, (summary_df, all_fits_df) in runs.items()}

    all_dists = []
    for stats in per_run.values():
        for d in stats:
            if d not in all_dists:
                all_dists.append(d)
    all_dists.sort(key=lambda d: -sum(per_run[l][d]["n"] for l in runs if d in per_run[l]))

    labels = list(runs.keys())
    cols = "ll" + "r" * len(labels)
    lines = [r"\begin{table}", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             "Distribution & Parameter & " + " & ".join(escape_latex(l) for l in labels) + r" \\",
             r"\midrule"]

    for d in all_dists:
        n_by_run = [str(per_run[l][d]["n"]) if d in per_run[l] else "0" for l in labels]
        dist_label = rf"{escape_latex(d)} ($n=$ {'/'.join(n_by_run)})"

        all_param_names = []
        for l in labels:
            if d in per_run[l]:
                for pname in per_run[l][d]["params"]:
                    if pname not in all_param_names:
                        all_param_names.append(pname)

        for i, pname in enumerate(all_param_names):
            cells = []
            for l in labels:
                params = per_run[l].get(d, {}).get("params", {})
                if pname in params:
                    mean, var = params[pname]
                    var_str = f"{var:.4g}" if var is not None else "--"
                    cells.append(f"{mean:.4g} ({var_str})")
                else:
                    cells.append("--")
            lines.append(f"{dist_label if i == 0 else ''} & {escape_latex(pname)} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Tables for one run")
    p_single.add_argument("summary_csv")
    p_single.add_argument("all_fits_csv")
    p_single.add_argument("-o", "--out", default="ramp_tables.tex")
    p_single.add_argument("--min-count", type=int, default=1)

    p_combine = sub.add_parser("combine", help="Combined tables across multiple runs")
    p_combine.add_argument("--run", nargs=3, action="append", required=True,
                            metavar=("LABEL", "SUMMARY_CSV", "ALL_FITS_CSV"))
    p_combine.add_argument("-o", "--out", default="combined_tables.tex")
    p_combine.add_argument("--min-count", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "single":
        summary_df, all_fits_df = load_run(args.summary_csv, args.all_fits_csv)
        table1 = make_distribution_tally_table(summary_df)
        table2 = make_param_summary_table(summary_df, all_fits_df, min_count=args.min_count)
    else:
        runs = OrderedDict()
        for label, summary_csv, all_fits_csv in args.run:
            runs[label] = load_run(summary_csv, all_fits_csv)
        table1 = make_combined_tally_table(runs)
        table2 = make_combined_param_table(runs, min_count=args.min_count)

    with open(args.out, "w") as f:
        f.write("% Requires \\usepackage{booktabs} in your preamble\n\n")
        f.write(table1 + "\n\n" + table2 + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()