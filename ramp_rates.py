import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALUE_COLUMNS = ["solar_scaled_MWh", "wind_scaled_MWh", "demand_MWh",
                  "supply_scaled_MWh", "residual_MWh"]


def _infer_fmt(path, fmt):
    return fmt or path.suffix.lstrip(".").lower()


def load_panel(path, fmt=None):
    """
    Loads the full long-format panel (every bus, every timestamp) into a
    single DataFrame, sorted by (ID, Time).

    fmt: "parquet" or "csv"; inferred from the file extension if None.
    """
    path = Path(path)
    fmt = _infer_fmt(path, fmt)
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported fmt: {fmt!r} (use 'csv' or 'parquet')")

    df["Time"] = pd.to_datetime(df["Time"])
    a_s = 0.13  # solar matches (a_s x 100)% of the average yearly demand across EU
    a_w = 0.17  # wind matches (a_w x 100)% of the average yearly demand across EU
    dataset_residual = df.assign(
        solar_scaled_MWh=lambda df: a_s * df["solar_MWh"],
        wind_scaled_MWh=lambda df: a_w * df["wind_MWh"],
        supply_scaled_MWh=lambda df: df["solar_scaled_MWh"] + df["wind_scaled_MWh"],
        residual_MWh=lambda df: df["demand_MWh"] - df["supply_scaled_MWh"],
        lat = lambda df: df["latitude"],
        long = lambda df: df['longitude'],
        country = lambda df: df["country"]
    )[
        [
            "Time",
            "ID",
            "solar_scaled_MWh",
            "wind_scaled_MWh",
            "demand_MWh",
            "supply_scaled_MWh",
            "residual_MWh",
            "latitude",
            "longitude",
            "country"
        ]
    ]
    return dataset_residual.sort_values(["ID", "Time"]).reset_index(drop=True)


def load_edges(path, fmt=None):
    """Loads the node-to-node edge list. Not used by the ramp-rate fitting
    itself -- handy later if you want to group/restrict buses by topology
    (e.g. only neighbors of a given node, or by connected component)."""
    path = Path(path)
    fmt = _infer_fmt(path, fmt)
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_bus_series(panel, bus_id, column="wind_scaled_MWh"):
    """
    Single bus, single value column, sorted by time.
    Returns: timestamps (pd.Series[datetime]), values (np.ndarray)
    """
    sub = panel[panel["ID"] == str(bus_id)].sort_values("Time")
    if sub.empty:
        raise KeyError(f"bus_id {bus_id!r} not found "
                        f"(e.g. available IDs: {panel['ID'].unique()[:6]})")
    return sub["Time"].reset_index(drop=True), sub[column].values.astype(float)


def to_wide(panel, column="wind_scaled_MWh", bus_ids=None):
    """
    Pivots the long panel into a time x bus_id wide DataFrame for one
    value column -- this is the shape characterize_network() expects.

    bus_ids: optional list to restrict to a subset; None = every bus in the panel.
    """
    sub = panel if bus_ids is None else panel[panel["ID"].isin({str(b) for b in bus_ids})]
    wide = sub.pivot(index="Time", columns="ID", values=column).sort_index()
    return wide


# ----------------------------------------------------------------
# 2. RAMP RATES
# ----------------------------------------------------------------
def compute_ramp_rates(series, order=1):
    """Simple order-step differences: delta(t) = P(t) - P(t-order)."""
    series = np.asarray(series, dtype=float)
    return series[order:] - series[:-order]


def compute_ramp_rates_multi(df, order=1):
    """
    df: time x bus DataFrame (e.g. from load_data_multi).
    Returns: ramps_per_bus (dict: bus_id -> 1D ramp array),
             pooled (1D array, every bus's ramps concatenated together)
    """
    ramps_per_bus = {col: compute_ramp_rates(df[col].values, order=order) for col in df.columns}
    pooled = np.concatenate(list(ramps_per_bus.values()))
    return ramps_per_bus, pooled


def get_bus_groups(panel, group_col="country"):
    """
    Maps each bus ID to a static attribute column (default: country).
    Assumes the column is constant per bus across all timestamps -- true
    for country/latitude/longitude in this panel, but this will raise a
    clear error if that assumption doesn't hold for whatever column you pass.

    Returns: pd.Series indexed by ID, values = group label.
    """
    g = panel[["ID", group_col]].drop_duplicates()
    dupes = g["ID"][g["ID"].duplicated(keep=False)].unique()
    if len(dupes) > 0:
        raise ValueError(f"{group_col!r} isn't constant per bus ID -- e.g. bus(es) "
                         f"{list(dupes[:5])} have multiple {group_col} values. "
                         f"Can't use it as a static group.")
    return g.set_index("ID")[group_col]


def compute_ramp_rates_by_group(df, bus_to_group, order=1):
    """
    df: time x bus wide DataFrame.
    bus_to_group: Series or dict mapping bus_id -> group label (e.g. from
                  get_bus_groups(panel, "country")).

    Pools every bus's ramps within the same group into one array -- this is
    the ramp-rate analog of characterize_network's "pooled" level, but
    scoped to a group (e.g. country) rather than the whole network.

    Returns: dict group_label -> pooled 1D ramp array
    """
    ramps_per_bus, _ = compute_ramp_rates_multi(df, order=order)
    groups = {}
    for bus_id, ramps in ramps_per_bus.items():
        g = bus_to_group[bus_id] if isinstance(bus_to_group, dict) else bus_to_group.loc[bus_id]
        groups.setdefault(g, []).append(ramps)
    return {g: np.concatenate(arrs) for g, arrs in groups.items()}


# ----------------------------------------------------------------
# 3. FIT CANDIDATE DISTRIBUTIONS
# ----------------------------------------------------------------
CANDIDATE_DISTS = {
    "norm": stats.norm,
    "laplace": stats.laplace,
    "cauchy": stats.cauchy,
    "t": stats.t,
    "logistic": stats.logistic,
    "hypsecant": stats.hypsecant,
    "gennorm": stats.gennorm,
}


def fit_distributions(ramps, dists=None):
    """
    MLE-fits each candidate distribution to `ramps`, and returns a DataFrame
    ranked by AIC (ascending = better), with BIC, a KS test p-value (higher
    = fewer grounds to reject "data ~ this distribution"), and the fitted
    params as a JSON string (so the table round-trips through CSV/parquet).
    """
    dists = dists or CANDIDATE_DISTS
    n = len(ramps)
    rows = []
    fitted_params = {}

    for name, dist in dists.items():
        try:
            # Heavy-tailed candidates (esp. hypsecant) can hit float64 overflow
            # in cosh()/exp() when evaluated at extreme outlier ramps -- the
            # result correctly saturates to ~0 density / 0-or-1 cdf, but numpy
            # prints a RuntimeWarning about it. Suppress the print; the math
            # is already right.
            with np.errstate(over="ignore", invalid="ignore"):
                params = dist.fit(ramps)
        except Exception as e:
            rows.append({"dist": name, "k": np.nan, "loglik": np.nan,
                         "AIC": np.inf, "BIC": np.inf, "ks_stat": np.nan,
                         "ks_pvalue": np.nan, "params": None, "error": str(e)})
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            loglik = np.sum(dist.logpdf(ramps, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
            # NOTE: pass dist.cdf (the callable) rather than `name` (a string).
            # Some scipy versions take an internal fast-path when given a known
            # distribution *name* that calls the underlying scipy.special
            # function directly and mishandles loc/scale args (raises something
            # like "ndtr() takes from 1 to 2 positional arguments but 3 were
            # given"). Passing the bound .cdf method forces the generic,
            # correct code path on every scipy version.
            ks_stat, ks_p = stats.kstest(ramps, dist.cdf, args=params)

        fitted_params[name] = params
        rows.append({"dist": name, "k": k, "loglik": loglik,
                     "AIC": aic, "BIC": bic, "ks_stat": ks_stat,
                     "ks_pvalue": ks_p, "params": json.dumps(list(params)),
                     "error": None})

    results = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return results, fitted_params


def fit_zero_inflated(ramps, zero_tol=0.0, dists=None):
    """
    For data with a genuine point mass at (near) zero -- e.g. solar ramps
    across many night-to-night hours where output is 0 both before and
    after -- fits candidate distributions only to the *nonzero* ramps,
    after separately reporting what fraction were exactly (or near-) zero.

    Without this, a continuous distribution's MLE scale parameter can
    collapse toward zero to chase the zero spike (it's maximizing
    likelihood on the bulk, which is dominated by that spike) at the cost
    of describing the actual nonzero variability. Symptoms: an absurdly
    negative AIC, and a QQ-plot whose theoretical-quantile axis collapses
    to a tiny range around zero while the data axis spans the real range.

    Returns: p0 (float, fraction of ramps within zero_tol of 0),
             results, fitted_params (as from fit_distributions, but fit
             only on the nonzero ramps), ramps_nonzero (array actually used)
    """
    ramps = np.asarray(ramps, dtype=float)
    is_zero = np.abs(ramps) <= zero_tol
    p0 = float(is_zero.mean())
    ramps_nonzero = ramps[~is_zero]
    if len(ramps_nonzero) < 10:
        raise ValueError(f"Only {len(ramps_nonzero)} nonzero ramps after removing "
                         f"{p0:.1%} zero mass -- too few to fit a continuous distribution.")
    results, fitted_params = fit_distributions(ramps_nonzero, dists=dists)
    return p0, results, fitted_params, ramps_nonzero


def _bus_scale(ramps, method="std"):
    """Scale statistic used to normalize a bus's ramps before pooling
    across buses of different physical size (e.g. wind farm capacity)."""
    if method == "std":
        return float(np.std(ramps))
    elif method == "mad":
        return float(stats.median_abs_deviation(ramps, scale="normal"))
    raise ValueError(f"Unknown normalize method: {method!r} (use 'std' or 'mad')")


def prepare_ramps_for_pooling(ramps_per_bus, zero_inflated=False, zero_tol=0.0, normalize=None):
    """
    Shared prep step used before pooling ramps across buses (whole-network
    or by-group):
      - if zero_inflated: strips each bus's exact-zero ramps (within
        zero_tol) first, so pooling/fitting isn't dominated by a structural
        point mass (e.g. solar ramps at night)
      - if normalize ("std" or "mad"): rescales each bus's (already
        zero-stripped, if applicable) ramps by its own scale statistic, so
        buses of very different physical size don't distort the pooled
        shape

    Buses with too little data (or a degenerate/zero scale) after prep are
    dropped with a printed warning rather than silently corrupting the pool.

    Returns: prepped_ramps_per_bus (dict), p0_per_bus (dict or None),
             scale_per_bus (dict or None)
    """
    prepped = {}
    p0_per_bus = {} if zero_inflated else None
    scale_per_bus = {} if normalize else None

    for bus_id, ramps in ramps_per_bus.items():
        r = np.asarray(ramps, dtype=float)

        if zero_inflated:
            is_zero = np.abs(r) <= zero_tol
            p0 = float(is_zero.mean())
            p0_per_bus[bus_id] = p0
            r = r[~is_zero]
            if len(r) < 10:
                print(f"  WARNING: bus {bus_id} has only {len(r)} nonzero ramps "
                      f"after removing {p0:.1%} zero mass -- excluding from pool")
                continue

        if normalize:
            s = _bus_scale(r, method=normalize)
            scale_per_bus[bus_id] = s
            if not np.isfinite(s) or s <= 0:
                print(f"  WARNING: bus {bus_id} has degenerate scale ({s}) -- excluding from pool")
                continue
            r = r / s

        prepped[bus_id] = r

    return prepped, p0_per_bus, scale_per_bus


# ----------------------------------------------------------------
# 4. DIAGNOSTIC PLOTS
# ----------------------------------------------------------------
def _robust_xlim(data, lower_pct=0.5, upper_pct=99.5, pad_frac=0.05):
    """
    Percentile-based display range for plotting only (fitting always uses
    the full data, unaffected by this). A single extreme outlier -- or a
    data-quality bug that's orders of magnitude too large -- otherwise
    forces matplotlib to compute axis ticks across an astronomical range,
    which can crash the text renderer (FreeType raster overflow) rather
    than just look bad. Clipping the *view* to the bulk of the
    distribution sidesteps that regardless of how extreme the outlier is.
    """
    lo, hi = np.percentile(data, [lower_pct, upper_pct])
    if hi <= lo:  # degenerate (near-constant) data
        lo, hi = data.min(), data.max()
        if hi <= lo:
            return lo - 1, hi + 1
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def plot_diagnostics(ramps, results, fitted_params, label, top_n=3, out_path=None, p0=None, xlabel=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xlim = _robust_xlim(ramps)

    # --- left: histogram + top-N fitted PDFs ---
    ax = axes[0]
    ax.hist(ramps, bins=60, range=xlim, density=True, alpha=0.4, color="0.5", label="empirical")
    x = np.linspace(xlim[0], xlim[1], 1000)
    colors = plt.cm.tab10.colors
    with np.errstate(over="ignore", invalid="ignore"):
        for i, row in results.head(top_n).iterrows():
            name = row["dist"]
            dist = CANDIDATE_DISTS[name]
            params = fitted_params[name]
            ax.plot(x, dist.pdf(x, *params), color=colors[i], lw=2,
                    label=f"{name} (AIC={row['AIC']:.0f})")
    ax.set_xlim(xlim)
    title = f"{label}: ramp-rate distribution"
    if p0 is not None:
        title += f"  (P(ramp=0)={p0:.1%} excluded from fit)"
    ax.set_title(title)
    ax.set_xlabel(xlabel or "ramp (MWh / step, view clipped to 0.5-99.5 pct)")

    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # --- right: QQ-plot for the single best fit ---
    ax2 = axes[1]
    best_name = results.iloc[0]["dist"]
    best_dist = CANDIDATE_DISTS[best_name]
    best_params = fitted_params[best_name]
    with np.errstate(over="ignore", invalid="ignore"):
        stats.probplot(ramps, dist=best_dist, sparams=best_params, plot=ax2)
    ax2.set_ylim(xlim)  # same clipping logic; outlier points just fall outside view
    ax2.set_title(f"QQ-plot vs best fit: {best_name}")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_per_bus_grid(ramps_per_bus, per_bus_fits, label, ncols=6, out_path=None, key_name="bus", max_panels=60):
    """Small-multiples: one panel per bus (or group), empirical hist + its own best fit."""
    bus_ids = list(ramps_per_bus.keys())
    n_total = len(bus_ids)

    if n_total > max_panels:
        # deterministic, evenly-spaced sample across the sorted bus list
        # (not just the first max_panels), so the panels shown are
        # representative rather than an arbitrary prefix
        bus_ids = sorted(bus_ids)
        idx = np.linspace(0, n_total - 1, max_panels).round().astype(int)
        bus_ids = [bus_ids[i] for i in sorted(set(idx))]
        print(f"\n{label}: per-{key_name} grid capped at {len(bus_ids)} of {n_total} "
              f"{key_name} entries (evenly sampled) -- see the saved CSV for every {key_name}.")


    n = len(bus_ids)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)

    for i, bus_id in enumerate(bus_ids):
        ax = axes[i // ncols][i % ncols]
        ramps = ramps_per_bus[bus_id]
        results, fitted_params = per_bus_fits[bus_id]
        best = results.iloc[0]
        xlim = _robust_xlim(ramps)
        ax.hist(ramps, bins=40, range=xlim, density=True, alpha=0.4, color="0.5")
        dist = CANDIDATE_DISTS[best["dist"]]
        x = np.linspace(xlim[0], xlim[1], 300)
        with np.errstate(over="ignore", invalid="ignore"):
            y = dist.pdf(x, *fitted_params[best["dist"]])
        ax.plot(x, y, color="C0", lw=2)
        ax.set_xlim(xlim)
        ax.set_title(f"{key_name} {bus_id}: {best['dist']} (AIC={best['AIC']:.0f})", fontsize=9)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{label}: per-{key_name} best fit")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig

def plot_network_summary(per_bus_summary, extremes, label, out_path=None):
    """
    Scales to any number of buses (unlike plot_per_bus_grid, which caps at
    max_panels for readability). At-a-glance view across the whole network:
      (a) which distribution wins most often (bar chart of best_dist counts)
      (b) how good those winning fits actually are (KS p-value histogram --
          mass near 0 means even the "best" candidate is a poor fit for a
          lot of buses, not just a runner-up problem)
      (c) spread of ramp severity across buses (max |ramp| per bus) --
          useful for spotting whether a handful of buses are driving most
          of the tail risk, or whether it's spread evenly across the network
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- (a) winning distribution counts ---
    ax = axes[0]
    counts = per_bus_summary["best_dist"].value_counts().sort_values(ascending=False)
    ax.bar(counts.index, counts.values, color="C0")
    ax.set_title("winning distribution")
    ax.set_ylabel(f"# of {len(per_bus_summary)} buses")
    ax.tick_params(axis="x", rotation=40)

    # --- (b) goodness-of-fit spread ---
    ax = axes[1]
    ax.hist(per_bus_summary["ks_pvalue"].dropna(), bins=30, color="C1", alpha=0.8)
    ax.axvline(0.05, color="red", ls="--", lw=1, label="p=0.05")
    ax.set_title("goodness-of-fit (best dist per bus)")
    ax.set_xlabel("KS test p-value")
    ax.set_ylabel("# of buses")
    ax.legend(fontsize=8)

    # --- (c) ramp severity spread ---
    ax = axes[2]
    ax.hist(extremes["abs_max"], bins=30, color="C2", alpha=0.8)
    ax.set_title("max |ramp| per bus")
    ax.set_xlabel("abs_max ramp (MWh / step)")
    ax.set_ylabel("# of buses")

    fig.suptitle(f"{label}: network-wide summary ({len(per_bus_summary)} buses)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig

def plot_group_overlay(group_ramps, group_fits, label, out_path=None):
    """
    Overlays every group's best-fit PDF on one shared axis -- lets you see
    directly whether e.g. one country's ramp behavior (peak sharpness, tail
    weight) differs visibly from another's, rather than eyeballing separate
    per-panel subplots.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_ramps = np.concatenate(list(group_ramps.values()))
    xlim = _robust_xlim(all_ramps)
    x = np.linspace(xlim[0], xlim[1], 1000)
    colors = plt.cm.tab10.colors

    for i, (group, ramps) in enumerate(group_ramps.items()):
        results, fitted_params = group_fits[group]
        best = results.iloc[0]
        dist = CANDIDATE_DISTS[best["dist"]]
        with np.errstate(over="ignore", invalid="ignore"):
            y = dist.pdf(x, *fitted_params[best["dist"]])
        ax.plot(x, y, color=colors[i % len(colors)], lw=2,
                label=f"{group} ({best['dist']}, n={len(ramps)})")

    ax.set_xlim(xlim)
    ax.set_title(f"{label}: best-fit comparison across groups")
    ax.set_xlabel("ramp (MWh / step)")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def report_ramp_extremes(ramps_per_bus, label, n_top=5, key_name="bus_id"):
    """
    Prints each bus's (or group's) raw min/max ramp and flags any non-finite
    values. Run before plotting so an extreme outlier (possibly a real
    data-quality bug -- e.g. a unit error or bad sensor reading -- rather
    than a genuine physical ramp) shows up as a number you can look up,
    not just a crash.
    """
    rows = []
    for bus_id, ramps in ramps_per_bus.items():
        n_nonfinite = int((~np.isfinite(ramps)).sum())
        rows.append({key_name: bus_id, "min": np.nanmin(ramps), "max": np.nanmax(ramps),
                     "abs_max": np.nanmax(np.abs(ramps)), "n_nonfinite": n_nonfinite})
    extremes = pd.DataFrame(rows).sort_values("abs_max", ascending=False).reset_index(drop=True)

    if extremes["n_nonfinite"].sum() > 0:
        print(f"\n!!! {label}: non-finite ramp values found (inf/-inf/nan) -- "
              f"these will break fitting/plotting. Fix the source data first:")
        print(extremes[extremes["n_nonfinite"] > 0].to_string(index=False))

    print(f"\n{label}: entries with the most extreme ramps, by {key_name} (top {n_top} by |value|):")
    print(extremes.head(n_top).to_string(index=False))
    return extremes


# ----------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------
def characterize(series, label, out_path=None, top_n=3, zero_inflated=False, zero_tol=0.0):
    ramps = compute_ramp_rates(series)
    if zero_inflated:
        p0, results, fitted_params, ramps_fit = fit_zero_inflated(ramps, zero_tol=zero_tol)
        print(f"\n=== {label} (zero-inflated: P(ramp=0)={p0:.1%}) ===")
    else:
        p0 = None
        ramps_fit = ramps
        results, fitted_params = fit_distributions(ramps_fit)
        print(f"\n=== {label} ===")

    print(results[["dist", "k", "AIC", "BIC", "ks_stat", "ks_pvalue"]].to_string(index=False))
    plot_diagnostics(ramps_fit, results, fitted_params, label, top_n=top_n, out_path=out_path, p0=p0)
    return results, fitted_params


def characterize_network(series_df, label, top_n=3, out_dir=None, zero_inflated=False, zero_tol=0.0, pool_normalize=None,
                         plots=("pooled", "grid", "summary")):

    ramps_per_bus, _ = compute_ramp_rates_multi(series_df)
    extremes = report_ramp_extremes(ramps_per_bus, label)
    fit_ramps_per_bus, p0_per_bus, _ = prepare_ramps_for_pooling(
        ramps_per_bus, zero_inflated=zero_inflated, zero_tol=zero_tol, normalize=None)

    # --- per bus ---
    per_bus_rows = []
    per_bus_fits = {}
    for bus_id, ramps in fit_ramps_per_bus.items():
        results, fitted_params = fit_distributions(ramps)
        per_bus_fits[bus_id] = (results, fitted_params)
        best = results.iloc[0]
        row = {"bus_id": bus_id, "n": len(ramps), "best_dist": best["dist"],
               "AIC": best["AIC"], "ks_stat": best["ks_stat"], "ks_pvalue": best["ks_pvalue"]}
        if zero_inflated:
            row["p0"] = p0_per_bus[bus_id]
        per_bus_rows.append(row)

    per_bus_summary = pd.DataFrame(per_bus_rows).sort_values("bus_id").reset_index(drop=True)

    print(f"\n=== {label}: per-bus best fits ===")
    print(per_bus_summary.to_string(index=False))
    print(f"\n{label}: best-distribution counts across buses:")
    print(per_bus_summary["best_dist"].value_counts().to_string())

    # --- pooled ---
    pool_ramps_per_bus, _, scale_per_bus = prepare_ramps_for_pooling(
        ramps_per_bus, zero_inflated=zero_inflated, zero_tol=zero_tol, normalize=pool_normalize)
    pooled = np.concatenate(list(pool_ramps_per_bus.values()))
    if pool_normalize:
        print(f"\n{label}: pooling normalized by per-bus {pool_normalize} "
              f"(scale range across buses: {min(scale_per_bus.values()):.3g} "
              f"to {max(scale_per_bus.values()):.3g})")
        pool_xlabel = f"ramp / bus's own {pool_normalize} (unitless)"
    else:
        pool_xlabel = None

    overall_p0 = None
    if zero_inflated:
        raw_pooled = np.concatenate(list(ramps_per_bus.values()))
        overall_p0 = float(np.mean(np.abs(raw_pooled) <= zero_tol))


    pooled_results, pooled_fitted = fit_distributions(pooled)
    print(f"\n=== {label}: pooled (all buses combined, n={len(pooled)}) ===")
    print(pooled_results[["dist", "k", "AIC", "BIC", "ks_stat", "ks_pvalue"]].to_string(index=False))

    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    pooled_out = f"{out_dir}/{slug}_pooled_ramp_fit.png" if out_dir else None
    grid_out = f"{out_dir}/{slug}_per_bus_grid.png" if out_dir else None
    summary_out = f"{out_dir}/{slug}_summary_overview.png" if out_dir else None

    if "pooled" in plots:
        plot_diagnostics(pooled, pooled_results, pooled_fitted, f"{label} (pooled, all buses)",
                          top_n=top_n, out_path=pooled_out, p0=overall_p0, xlabel=pool_xlabel)
    if "grid" in plots:
        plot_per_bus_grid(fit_ramps_per_bus, per_bus_fits, label, out_path=grid_out)
    if "summary" in plots:
        plot_network_summary(per_bus_summary, extremes, label, out_path=summary_out)


    # --- save results to CSV (same out_dir as the plots) ---
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        per_bus_summary.to_csv(f"{out_dir}/{slug}_per_bus_summary.csv", index=False)

        # full per-bus results: every candidate distribution for every bus,
        # not just the winner -- lets you revisit the runner-up fits later
        # without re-running anything.
        per_bus_all = pd.concat(
            [results.assign(bus_id=bus_id) for bus_id, (results, _) in per_bus_fits.items()],
            ignore_index=True,
        )
        cols = ["bus_id"] + [c for c in per_bus_all.columns if c != "bus_id"]
        per_bus_all[cols].to_csv(f"{out_dir}/{slug}_per_bus_all_fits.csv", index=False)

        pooled_results.to_csv(f"{out_dir}/{slug}_pooled_fit_results.csv", index=False)

        saved_msg = (f"\nSaved: {slug}_per_bus_summary.csv, {slug}_per_bus_all_fits.csv, "
                     f"{slug}_pooled_fit_results.csv")
        if scale_per_bus:
            pd.Series(scale_per_bus, name="pool_scale").rename_axis("bus_id").reset_index().to_csv(
                f"{out_dir}/{slug}_pool_scales.csv", index=False)
            saved_msg += f", {slug}_pool_scales.csv"
        print(saved_msg + f" -> {out_dir}")


    return per_bus_summary, pooled_results, per_bus_fits


def characterize_by_group(panel, column, group_col="country", bus_ids=None, top_n=3, out_dir=None,
                          zero_inflated=False, zero_tol=0.0, normalize=None, plots=("grid", "overlay")):
    wide = to_wide(panel, column, bus_ids=bus_ids)
    bus_to_group = get_bus_groups(panel, group_col)
    ramps_per_bus, _ = compute_ramp_rates_multi(wide)
    
    prepped, p0_per_bus, scale_per_bus = prepare_ramps_for_pooling(
        ramps_per_bus, zero_inflated=zero_inflated, zero_tol=zero_tol, normalize=normalize)

    group_ramps = {}
    for bus_id, ramps in prepped.items():
        g = bus_to_group.loc[bus_id]
        group_ramps.setdefault(g, []).append(ramps)
    group_ramps = {g: np.concatenate(arrs) for g, arrs in group_ramps.items()}

    if normalize:
        print(f"\n{label}: pooling normalized by per-bus {normalize} "
              f"(scale range across buses: {min(scale_per_bus.values()):.3g} "
              f"to {max(scale_per_bus.values()):.3g})")


    label = f"{column} by {group_col}"
    report_ramp_extremes(group_ramps, label, key_name=group_col)

    n_buses_per_group = bus_to_group.value_counts()
    rows = []
    group_fits = {}
    for group, ramps in group_ramps.items():
        results, fitted_params = fit_distributions(ramps)
        group_fits[group] = (results, fitted_params)
        best = results.iloc[0]
        row = {group_col: group, "n_buses": int(n_buses_per_group.get(group, 0)),
               "n_ramps": len(ramps), "best_dist": best["dist"], "AIC": best["AIC"],
               "ks_stat": best["ks_stat"], "ks_pvalue": best["ks_pvalue"]}
        rows.append(row)

    group_summary = pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)

    if zero_inflated:
        group_p0 = {}
        for bus_id, p0 in p0_per_bus.items():
            g = bus_to_group.loc[bus_id]
            group_p0.setdefault(g, []).append(p0)
        group_summary["p0_mean"] = group_summary[group_col].map(
            lambda g: float(np.mean(group_p0.get(g, [np.nan]))))



    print(f"\n=== {label}: best fits ===")
    print(group_summary.to_string(index=False))

    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    grid_out = f"{out_dir}/{slug}_grid.png" if out_dir else None
    overlay_out = f"{out_dir}/{slug}_overlay.png" if out_dir else None

    if "grid" in plots:
        plot_per_bus_grid(group_ramps, group_fits, label, out_path=grid_out, key_name=group_col)
    if "overlay" in plots:
        plot_group_overlay(group_ramps, group_fits, label, out_path=overlay_out)


    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        group_summary.to_csv(f"{out_dir}/{slug}_summary.csv", index=False)

        group_all = pd.concat(
            [results.assign(**{group_col: group}) for group, (results, _) in group_fits.items()],
            ignore_index=True,
        )
        cols = [group_col] + [c for c in group_all.columns if c != group_col]
        group_all[cols].to_csv(f"{out_dir}/{slug}_all_fits.csv", index=False)

        saved_msg = f"\nSaved: {slug}_summary.csv, {slug}_all_fits.csv"
        if scale_per_bus:
            pd.Series(scale_per_bus, name="scale").rename_axis("bus_id").reset_index().to_csv(
                f"{out_dir}/{slug}_bus_scales.csv", index=False)
            saved_msg += f", {slug}_bus_scales.csv"
        print(saved_msg + f" -> {out_dir}")


    return group_summary, group_fits, group_ramps


if __name__ == "__main__":
    panel = load_panel("/Users/emrysking/Documents/github-projects/datasci-challenge/dataset/renewables-dataset.parquet")

    # -- single-bus usage --
    # bus_id = panel["ID"].iloc[0]
    # _, residual = get_bus_series(panel, bus_id, "residual_MWh")

    # characterize(residual, "Residual load", out_path="/Users/emrysking/Documents/github-projects/datasci-challenge/residual_ramp_fit.png")

    # -- multi-bus (network-wide) usage: per-bus fits + pooled fit --
    # residual_all = to_wide(panel, "residual_MWh")

    
    # characterize_network(residual_all, "Residual (network)", out_dir="/Users/emrysking/Documents/github-projects/datasci-challenge")

    # -- grouped by country: pooled ramps within each country --
    characterize_by_group(panel, "residual_MWh", group_col="country", out_dir="/Users/emrysking/Documents/github-projects/datasci-challenge/1317")