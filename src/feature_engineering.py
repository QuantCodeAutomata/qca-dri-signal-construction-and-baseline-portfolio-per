"""
Phase B: Feature Matrix Construction for the DRI Signal.

Builds 42-dimensional feature vectors: [R^C (21 chronological), R^R (21 rank-ordered)]
with within-calendar-date cross-sectional demeaning.

Custom — Context7 found no library equivalent for this specific DRI feature construction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def cross_sectional_demean_daily_returns(
    daily_rets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply cross-sectional demeaning per calendar date.

    For each calendar date in the sample, compute the cross-sectional mean
    return across all eligible stocks present on that date, then subtract.

    r_tilde_{i,t,d} = r_{i,t,d} - mean_d(r_{j,t,d} for all j with data on d)

    Args:
        daily_rets_df: DataFrame with columns [permno, date, ret].

    Returns:
        DataFrame with added column 'ret_demeaned'.
    """
    # Using groupby('date').transform('mean') — straightforward pandas
    df = daily_rets_df.copy()
    df["ret_demeaned"] = (
        df["ret"]
        - df.groupby("date")["ret"].transform("mean")
    )
    return df


def build_chronological_vector(demeaned_returns_21: np.ndarray) -> np.ndarray:
    """
    Construct R^C: 21-dimensional chronological return vector.

    The vector preserves the time ordering: index 0 = d=1 = most recent
    trading day, index 20 = d=21 = oldest trading day in the 21-day window.

    Args:
        demeaned_returns_21: Array of shape (21,) with demeaned returns,
            d=1 (most recent) at index 0.

    Returns:
        21-dim float32 array (identity operation — input already in correct order).
    """
    assert len(demeaned_returns_21) == 21, "Expected 21 returns"
    return demeaned_returns_21.astype(np.float32)


def build_rank_vector(demeaned_returns_21: np.ndarray) -> np.ndarray:
    """
    Construct R^R: 21-dimensional rank-ordered return vector.

    Sorts the 21 demeaned daily returns in descending order:
    r^(1) >= r^(2) >= ... >= r^(21), where k=1 is the maximum return
    and k=21 is the minimum return (MAX and MIN effects).

    Args:
        demeaned_returns_21: Array of shape (21,) with demeaned returns.

    Returns:
        21-dim float32 array sorted in descending order.
    """
    # Using np.sort with descending order — Context7 confirmed numpy approach
    assert len(demeaned_returns_21) == 21, "Expected 21 returns"
    return np.sort(demeaned_returns_21)[::-1].astype(np.float32)


def build_interaction_features(demeaned_returns_21: np.ndarray) -> np.ndarray:
    """
    Build 441-dimensional interaction features z^(d,k) for exp_2.

    For each day d and rank position k:
    z^(d,k) = r_tilde_d  if rank(r_tilde_d) == k  else 0

    This creates a sparse 21×21 matrix where exactly one entry per row d
    is nonzero (the entry at the rank position of r_tilde_d), flattened
    row-major to a 441-dimensional vector.

    Args:
        demeaned_returns_21: Array of shape (21,), d=1 at index 0.

    Returns:
        441-dim float32 interaction feature vector.
    """
    assert len(demeaned_returns_21) == 21, "Expected 21 returns"

    # Rank positions: argsort gives indices that would sort in ascending order.
    # We want descending ranks: rank 0 = largest (k=1), rank 20 = smallest (k=21).
    # argsort of ascending sort → position i has rank = n - 1 - ascending_rank_of_i
    ascending_ranks = np.argsort(np.argsort(demeaned_returns_21))  # 0=smallest
    descending_ranks = 20 - ascending_ranks  # 0=largest (k=1)

    interaction_matrix = np.zeros((21, 21), dtype=np.float32)
    for d in range(21):
        k = descending_ranks[d]
        interaction_matrix[d, k] = demeaned_returns_21[d]

    return interaction_matrix.flatten()


def _demean_window_returns(
    daily_df: pd.DataFrame,
    permnos_in_month: np.ndarray,
    month_end_date: pd.Timestamp,
) -> dict:
    """
    Demean the 21-day daily returns for all stocks in a given month.

    Returns dict: {permno: demeaned_returns_21} for each stock that has
    exactly 21 non-missing returns in the window.

    Args:
        daily_df: Full daily returns DataFrame [permno, date, ret].
        permnos_in_month: Array of permno values eligible in this month.
        month_end_date: Last trading day of the month.

    Returns:
        dict mapping permno → demeaned returns array (d=1 first, shape (21,))
    """
    # Get all trading days up to and including month_end_date
    all_dates = pd.to_datetime(
        daily_df[daily_df["date"] <= month_end_date]["date"].unique()
    )
    all_dates = sorted(all_dates)

    if len(all_dates) < 21:
        return {}

    # Last 21 trading days
    last_21_dates = all_dates[-21:]
    date_set = set(last_21_dates)

    # Filter daily data: only eligible stocks on last 21 trading days
    mask = (
        daily_df["permno"].isin(permnos_in_month)
        & daily_df["date"].isin(date_set)
    )
    window_data = daily_df[mask].copy()

    if window_data.empty:
        return {}

    # Cross-sectional demeaning per calendar date
    window_data["ret_demeaned"] = (
        window_data["ret"]
        - window_data.groupby("date")["ret"].transform("mean")
    )

    # Build per-stock demeaned return arrays
    result = {}
    date_to_idx = {d: i for i, d in enumerate(last_21_dates)}

    for perm in permnos_in_month:
        stock_data = window_data[window_data["permno"] == perm].sort_values("date")

        if len(stock_data) != 21:
            continue
        if stock_data["ret"].isna().any():
            continue
        # Verify all 21 required dates are present (type-safe via nunique)
        if stock_data["date"].nunique() != 21:
            continue

        # Order: oldest first in the DataFrame (sorted by date)
        # We want d=1 (most recent) at index 0
        demeaned = stock_data["ret_demeaned"].values[::-1].astype(np.float32)
        result[perm] = demeaned

    return result


def build_feature_matrix(
    eligible_universe: pd.DataFrame,
    daily_df: pd.DataFrame,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the full 42-dim feature matrix for all eligible stock-months.

    For each eligible (permno, month-end) pair:
    1. Retrieve 21 demeaned daily returns from the last 21 trading days
    2. Construct R^C (21-dim chronological) and R^R (21-dim rank-ordered)
    3. Concatenate to form 42-dim row vector: X_i = [R^C_i, R^R_i]
    4. Response y_i = next-month excess return

    Cross-sectional demeaning is done within each month's eligible
    cross-section, consistent with the paper's methodology.

    Args:
        eligible_universe: DataFrame with permno, date, next_ret_excess, me, exchcd.
            Must have column 'next_ret_excess' = next-month excess return.
        daily_df: Full CRSP daily DataFrame [permno, date, ret].
        is_training: Flag (currently unused, for future extensions).

    Returns:
        (meta_df, X, y) where:
        - meta_df: DataFrame [permno, date, me, exchcd] (N rows)
        - X: float32 array of shape (N, 42)
        - y: float32 array of shape (N,)
    """
    # Group by month-end date to process one month at a time
    months = sorted(eligible_universe["date"].unique())

    meta_rows = []
    X_rows = []
    y_rows = []

    for month_end in months:
        month_stocks = eligible_universe[
            eligible_universe["date"] == month_end
        ]
        permnos_this_month = month_stocks["permno"].values

        # Demean within this month's eligible cross-section
        demeaned_dict = _demean_window_returns(
            daily_df, permnos_this_month, pd.Timestamp(month_end)
        )

        for _, row in month_stocks.iterrows():
            perm = row["permno"]
            if perm not in demeaned_dict:
                continue

            demeaned_21 = demeaned_dict[perm]

            R_C = build_chronological_vector(demeaned_21)
            R_R = build_rank_vector(demeaned_21)
            feature_vec = np.concatenate([R_C, R_R])  # 42-dim

            meta_rows.append({
                "permno": perm,
                "date": month_end,
                "me": row["me"],
                "exchcd": row.get("exchcd", np.nan),
            })
            X_rows.append(feature_vec)
            y_rows.append(row["next_ret_excess"])

    if not X_rows:
        meta_df = pd.DataFrame(columns=["permno", "date", "me", "exchcd"])
        return meta_df, np.empty((0, 42), dtype=np.float32), np.empty(0, dtype=np.float32)

    meta_df = pd.DataFrame(meta_rows)
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    assert X.shape[1] == 42, f"Expected 42 features, got {X.shape[1]}"
    return meta_df, X, y


def build_augmented_feature_matrix(
    eligible_universe: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build augmented 462-dim feature matrix for exp_2 interaction analysis.

    Combines 21 chronological features with 441 interaction terms:
    X_aug = [R^C (21), Z^I (441)] = 462-dim vector

    The 441 interaction terms z^(d,k) capture the joint time-rank position
    of each daily return, as specified in the paper's augmented model.

    Args:
        eligible_universe: Same format as build_feature_matrix.
        daily_df: CRSP daily DataFrame.

    Returns:
        (meta_df, X_aug, y) where X_aug has shape (N, 462).
    """
    months = sorted(eligible_universe["date"].unique())

    meta_rows = []
    X_rows = []
    y_rows = []

    for month_end in months:
        month_stocks = eligible_universe[
            eligible_universe["date"] == month_end
        ]
        permnos_this_month = month_stocks["permno"].values
        demeaned_dict = _demean_window_returns(
            daily_df, permnos_this_month, pd.Timestamp(month_end)
        )

        for _, row in month_stocks.iterrows():
            perm = row["permno"]
            if perm not in demeaned_dict:
                continue

            demeaned_21 = demeaned_dict[perm]

            R_C = build_chronological_vector(demeaned_21)
            Z_I = build_interaction_features(demeaned_21)
            feature_vec = np.concatenate([R_C, Z_I])  # 21 + 441 = 462-dim

            meta_rows.append({
                "permno": perm,
                "date": month_end,
                "me": row["me"],
                "exchcd": row.get("exchcd", np.nan),
            })
            X_rows.append(feature_vec)
            y_rows.append(row["next_ret_excess"])

    if not X_rows:
        meta_df = pd.DataFrame(columns=["permno", "date", "me", "exchcd"])
        return meta_df, np.empty((0, 462), dtype=np.float32), np.empty(0, dtype=np.float32)

    meta_df = pd.DataFrame(meta_rows)
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    assert X.shape[1] == 462, f"Expected 462 features, got {X.shape[1]}"
    return meta_df, X, y
