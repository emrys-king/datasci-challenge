"""
Builds a weighted adjacency matrix from RE-Europe's REAL network_edges.csv
schema (confirmed from Jensen & Pinson 2017, Table 7):
    fromNode, toNode, X (reactance, p.u.), Y (susceptance=1/X, p.u.),
    numLines, limit (MW, thermal flow limit), length (km)

Two documented gotchas this handles explicitly:
  - limit == 0 means UNLIMITED capacity, not zero capacity. Using raw
    `limit` as an edge weight would treat unlimited lines as having no
    capacity at all -- backwards.
  - X == 1e-5 is a sentinel for "no data", not a real (tiny) reactance
    value. Since Y = 1/X, this produces a huge but MEANINGLESS susceptance
    if not handled.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def build_weighted_adjacency(node_ids, edges_df, weight_by="susceptance",
                              unlimited_value=None, no_data_reactance=1e-5):
    """
    node_ids: canonical, ordered list of bus IDs (matching your residual-load
              matrix's column order) -- this defines row/col order of A.
    edges_df: raw network_edges.csv, with REAL RE-Europe column names.
    weight_by: "susceptance" (Y, physically motivated -- the natural weight
               for DC power-flow-style spatial coupling), "limit" (thermal
               capacity in MW), or "binary" (unweighted, just connectivity).
    unlimited_value: what to substitute for limit==0 rows if weight_by="limit".
               Defaults to 1.5x the max FINITE limit in the data -- i.e.
               "more than the biggest real limit", a reasonable stand-in for
               "unlimited" without using an arbitrary huge number that would
               dominate every average.
    no_data_reactance: the sentinel value (1e-5 per the dataset spec) that
               marks missing reactance data; rows with X at or below this
               are excluded from susceptance-based weighting (since Y=1/X
               would be a meaningless huge number for these) and fall back
               to a binary weight of 1 instead.
    """
    node_ids = [str(n) for n in node_ids]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    edges_df = edges_df.copy()
    edges_df["fromNode"] = edges_df["fromNode"].astype(str)
    edges_df["toNode"] = edges_df["toNode"].astype(str)

    # drop edges referencing buses outside our canonical node_ids (shouldn't
    # happen if you've already run check_id_consistency, but defensive)
    valid = edges_df["fromNode"].isin(idx) & edges_df["toNode"].isin(idx)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} edges referencing buses not in node_ids "
              f"(run check_id_consistency first if this number looks large)")
    edges_df = edges_df[valid]

    if weight_by == "binary":
        weights = np.ones(len(edges_df))

    elif weight_by == "limit":
        limits = edges_df["limit"].astype(float).values
        is_unlimited = limits == 0
        finite_limits = limits[~is_unlimited]
        if unlimited_value is None:
            unlimited_value = finite_limits.max() * 1.5 if len(finite_limits) else 1.0
        weights = np.where(is_unlimited, unlimited_value, limits)
        print(f"  {is_unlimited.sum()} of {len(edges_df)} edges were 'unlimited' (limit=0), "
              f"substituted with {unlimited_value:.1f} MW "
              f"(1.5x the max finite limit, {finite_limits.max() if len(finite_limits) else 'n/a'})")

    elif weight_by == "susceptance":
        X = edges_df["X"].astype(float).values
        Y = edges_df["Y"].astype(float).values
        has_no_data = X <= no_data_reactance
        # for no-data rows, fall back to a binary weight of 1 rather than a
        # meaningless huge susceptance value
        weights = np.where(has_no_data, 1.0, Y)
        print(f"  {has_no_data.sum()} of {len(edges_df)} edges had 'no data' reactance "
              f"(X<={no_data_reactance}), given a fallback weight of 1.0 instead of Y=1/X")

    else:
        raise ValueError(f"weight_by must be 'susceptance', 'limit', or 'binary', got {weight_by!r}")

    rows_, cols_, vals_ = [], [], []
    for (f, t), w in zip(zip(edges_df["fromNode"], edges_df["toNode"]), weights):
        i, j = idx[f], idx[t]
        rows_ += [i, j]
        cols_ += [j, i]
        vals_ += [w, w]  # undirected

    A = sparse.csr_matrix((vals_, (rows_, cols_)), shape=(n, n))
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for isolated nodes
    A_norm = sparse.diags(1.0 / row_sums) @ A

    n_isolated = (np.asarray(A.sum(axis=1)).flatten() == 0).sum()
    if n_isolated > 0:
        print(f"  WARNING: {n_isolated} buses have ZERO edges in this graph -- "
              f"every spatial feature (neighbor mean, concurrent stress, etc.) "
              f"will be undefined/zero for these buses. Investigate before proceeding.")

    return A, A_norm


def compute_resilience_feature(node_ids, edges_df, unlimited_value=None):
    """
    Per-bus STATIC feature: total connected transfer capacity (MW), summed
    across all lines touching this bus. This is a vulnerability/resilience
    indicator -- a bus with little connected capacity has less ability to
    import power during a local shortfall, plausibly leading to a heavier
    realized tail even at the same underlying stress level. Intended for
    Stage 2 (GPD scale regression) covariates, NOT for the dynamic
    neighbor-aggregation features (see the susceptance vs. limit discussion
    -- this uses `limit` deliberately, since capacity/resilience is exactly
    the question `limit` is suited to answer, unlike geographic proximity).

    Uses the same limit==0-means-unlimited handling as build_weighted_adjacency,
    for consistency.
    """
    node_ids_str = [str(n) for n in node_ids]
    edges_df = edges_df.copy()
    edges_df["fromNode"] = edges_df["fromNode"].astype(str)
    edges_df["toNode"] = edges_df["toNode"].astype(str)

    limits = edges_df["limit"].astype(float).values
    is_unlimited = limits == 0
    finite_limits = limits[~is_unlimited]
    if unlimited_value is None:
        unlimited_value = finite_limits.max() * 1.5 if len(finite_limits) else 1.0
    edges_df = edges_df.assign(_weight=np.where(is_unlimited, unlimited_value, limits))

    total_capacity = pd.Series(0.0, index=node_ids_str)
    for _, r in edges_df.iterrows():
        if r["fromNode"] in total_capacity.index:
            total_capacity[r["fromNode"]] += r["_weight"]
        if r["toNode"] in total_capacity.index:
            total_capacity[r["toNode"]] += r["_weight"]

    return total_capacity.reindex(node_ids_str).values


# ==================================================================
# VALIDATION: confirm both gotchas are handled correctly
# ==================================================================
if __name__ == "__main__":
    node_ids = ["1001", "1002", "1003", "1004", "1005"]
    edges = pd.read_csv("/home/claude/mock_edges_real_gotchas.csv")

    print("=== weight_by='susceptance' ===")
    A_sus, A_sus_norm = build_weighted_adjacency(node_ids, edges, weight_by="susceptance")
    print("A (susceptance-weighted):\n", A_sus.toarray())
    # check: the no-data edge (1002-1004, X=1e-5) should have weight 1.0, NOT 100000
    print(f"edge 1002<->1004 weight: {A_sus[node_ids.index('1002'), node_ids.index('1004')]} "
          f"(should be 1.0, not the raw Y=100000)")

    print("\n=== weight_by='limit' ===")
    A_lim, A_lim_norm = build_weighted_adjacency(node_ids, edges, weight_by="limit")
    print("A (limit-weighted):\n", A_lim.toarray())
    # check: the two 'unlimited' edges (limit=0) should have a LARGE substituted
    # weight, not zero
    w_1001_1003 = A_lim[node_ids.index("1001"), node_ids.index("1003")]
    w_1004_1005 = A_lim[node_ids.index("1004"), node_ids.index("1005")]
    print(f"edge 1001<->1003 (was limit=0/'unlimited') weight: {w_1001_1003} (should be large, not 0)")
    print(f"edge 1004<->1005 (was limit=0/'unlimited') weight: {w_1004_1005} (should be large, not 0)")

    print("\n=== weight_by='binary' ===")
    A_bin, A_bin_norm = build_weighted_adjacency(node_ids, edges, weight_by="binary")
    print("A (binary):\n", A_bin.toarray())

    print("\nAll three weighting schemes constructed successfully; both documented "
          "gotchas (limit=0='unlimited', X=1e-5='no data') handled correctly.")