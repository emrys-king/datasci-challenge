"""
Lead-lag spatial features for SHORT-HORIZON (e.g. 1-hour-ahead) forecasting.
------------------------------------------------------------------------------
For short horizons, the single best spatial feature is often: "what is a
geographically/electrically upwind neighbor doing RIGHT NOW" -- since
weather fronts advect across a grid at finite speed, a neighbor's CURRENT
state can be a leading indicator of your own NEAR-FUTURE state.

This is NOT leakage: it only uses data at t0 (now, already observed for
every bus), to predict a DIFFERENT bus's target at t0+1. What would be
leakage is using a neighbor's value AT the target time t0+1.

This module empirically discovers, for each pair of connected/nearby buses,
which one leads the other and by how much (via lagged cross-correlation),
then builds a "best leading neighbor" feature using only <= t0 information.
"""

import numpy as np
import pandas as pd


def lead_lag_correlation(x_leader_candidate, x_target, max_lag=6):
    """
    Tests whether x_leader_candidate at time (t - lag) predicts x_target at
    time t, for lag = 0..max_lag. Returns the lag with the highest
    correlation and that correlation's strength.

    A large positive best_lag means x_leader_candidate tends to show what
    x_target will look like `best_lag` hours later -- i.e. it LEADS.
    """
    best_lag, best_corr = 0, -np.inf
    n = len(x_target)
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = x_leader_candidate, x_target
        else:
            a, b = x_leader_candidate[:-lag], x_target[lag:]
        if len(a) < 10:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def discover_lead_lag_structure(values_matrix, neighbor_pairs, max_lag=6):
    """
    For each (i, j) pair in neighbor_pairs (i's actual grid/geographic
    neighbors), determines whether j leads i, and by how much, using ONLY
    the TRAINING portion of values_matrix (fit this once on train data,
    never on test data, same as any other learned hyperparameter).

    Returns a DataFrame: bus, neighbor, lag (hours neighbor leads bus by),
    strength (correlation at that lag).
    """
    results = []
    for i, j in neighbor_pairs:
        lag, corr = lead_lag_correlation(values_matrix[:, j], values_matrix[:, i], max_lag)
        results.append({"bus": i, "neighbor": j, "lead_lag_hours": lag, "correlation": corr})
    return pd.DataFrame(results)


def build_leading_neighbor_feature(values_matrix, lead_lag_df, node_idx_map=None):
    """
    For each bus, picks its STRONGEST leading neighbor (highest correlation
    among candidates with lag > 0, i.e. a genuine leading relationship, not
    lag=0 which is just contemporaneous correlation) and builds a feature:
    that neighbor's value at (t0 - 0), i.e. its CURRENT value, shifted
    forward by `lead_lag_hours` to align with bus i's target at t0+1.

    Concretely: if neighbor j leads bus i by L hours, then j's value at
    time t0 is informative about bus i's value at t0+L. For a 1-hour-ahead
    target, the relevant alignment is j's value at (t0 - (L-1)) predicting
    i's value at t0+1 -- but the simplest and most robust version for a
    1-hour horizon is just to use j's MOST RECENT value (at t0) whenever
    L>=1, since that's already "ahead" of bus i's own current state.
    """
    n_time, n_buses = values_matrix.shape
    leading_feature = np.full((n_time, n_buses), np.nan)
    chosen_neighbor = {}

    for bus in lead_lag_df["bus"].unique():
        candidates = lead_lag_df[(lead_lag_df["bus"] == bus) & (lead_lag_df["lead_lag_hours"] > 0)]
        if candidates.empty:
            continue
        best = candidates.loc[candidates["correlation"].idxmax()]
        neighbor = int(best["neighbor"])
        chosen_neighbor[bus] = (neighbor, best["lead_lag_hours"], best["correlation"])
        leading_feature[:, bus] = values_matrix[:, neighbor]  # neighbor's value AT t0 (safe, already observed)

    return leading_feature, chosen_neighbor


# ==================================================================
# VALIDATION against mock data with a KNOWN true advection lag
# ==================================================================
if __name__ == "__main__":
    values = np.load("/home/claude/mock_advection_values.npy")
    n_time, n_buses = values.shape
    train = values[: int(n_time * 0.7)]  # fit lead-lag structure on train only

    # buses are in a line 0-1-2-3 (west to east); true lag between
    # adjacent buses is 2 hours (bus b sees the front `b*2` hours after bus 0)
    neighbor_pairs = [(1, 0), (2, 1), (3, 2), (0, 1), (1, 2), (2, 3)]  # both directions, as if unsure

    lead_lag_df = discover_lead_lag_structure(train, neighbor_pairs, max_lag=6)
    print("Discovered lead-lag structure (fit on train data only):")
    print(lead_lag_df.to_string(index=False))

    print("\nTrue structure: each bus b's disturbance arrives 2h after bus (b-1)'s, "
          "so we expect (1,0)->lag~2, (2,1)->lag~2, (3,2)->lag~2, and the reverse "
          "pairs (0,1),(1,2),(2,3) to show near-zero or weaker correlation at lag>0 "
          "since the earlier bus does NOT lag the later one.")

    leading_feature, chosen = build_leading_neighbor_feature(values, lead_lag_df)
    print("\nChosen leading neighbor per bus (bus: (neighbor, lag_hours, correlation)):")
    for b, info in chosen.items():
        print(f"  bus {b}: leading neighbor={info[0]}, discovered lag={info[1]}h, corr={info[2]:.3f}")

    print("\nleading_feature shape:", leading_feature.shape,
          " NaN columns (buses with no usable leading neighbor):",
          np.isnan(leading_feature).all(axis=0).sum())