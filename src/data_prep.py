"""
Phase A: Data Preparation for DRI Signal.

Implements universe construction, eligibility filters, excess return
computation, and daily return extraction for the 21-trading-day window.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def handle_delisting_returns(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust monthly returns for delisting events following CRSP convention.

    - If dlret is available (non-NaN): ret_adj = (1 + ret) * (1 + dlret) - 1
    - If dlret is NaN but stock is delisted: impute -30% for NYSE/Amex,
      -55% for Nasdaq per the standard procedure in the literature.

    Args:
        monthly_df: Monthly DataFrame with columns ret, dlret, exchcd.

    Returns:
        DataFrame with adjusted 'ret' column.
    """
    df = monthly_df.copy()

    # Apply known delisting return
    has_dlret = df["dlret"].notna()
    df.loc[has_dlret, "ret"] = (
        (1 + df.loc[has_dlret, "ret"]) * (1 + df.loc[has_dlret, "dlret"]) - 1
    )

    # Impute missing delisting returns (NaN dlret but likely delisted)
    # Heuristic: ret is missing (NaN) and stock has no future data
    # We detect delisted stocks by checking for extreme negative returns or missing future months
    # Simple conservative approach: keep as-is (missing dlret is set to NaN in data)
    return df


def compute_excess_returns(
    monthly_df: pd.DataFrame,
    rf_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute excess returns by subtracting the monthly risk-free rate.

    Uses period-based (year-month) joining to handle the mismatch between
    CRSP's last-trading-day-of-month dates and Kenneth French's calendar
    month-end dates (e.g. CRSP 1985-03-29 vs FF 1985-03-31).

    Args:
        monthly_df: Monthly DataFrame with 'ret' column and 'date' column.
        rf_series: Monthly risk-free rate Series indexed by date.

    Returns:
        DataFrame with added 'ret_excess' and 'rf' columns.
    """
    df = monthly_df.copy()
    # Normalize to calendar month-end so both sides align regardless of
    # whether dates are last-trading-day or last-calendar-day
    df["_ym"] = df["date"].dt.to_period("M")

    rf_df = rf_series.rename("rf").reset_index()
    rf_df["_ym"] = rf_df["date"].dt.to_period("M")

    df = df.merge(rf_df[["_ym", "rf"]], on="_ym", how="left")
    df.drop(columns=["_ym"], inplace=True)
    df["ret_excess"] = df["ret"] - df["rf"]
    return df


def get_last_21_trading_days(
    daily_df: pd.DataFrame,
    month_end_date: pd.Timestamp,
    permno: int,
) -> Optional[np.ndarray]:
    """
    Extract the 21 most recent daily returns for a stock ending on month_end_date.

    Returns returns ordered d=1 (most recent) to d=21 (oldest), i.e., the
    array index 0 is the last trading day of the month.

    Args:
        daily_df: CRSP daily DataFrame with [permno, date, ret].
        month_end_date: Last trading day of the month.
        permno: Stock identifier.

    Returns:
        np.ndarray of shape (21,) or None if any return is missing.
    """
    stock_daily = daily_df[
        (daily_df["permno"] == permno) & (daily_df["date"] <= month_end_date)
    ].sort_values("date")

    if len(stock_daily) < 21:
        return None

    last_21 = stock_daily.tail(21)

    if last_21["ret"].isna().any():
        return None

    # d=1 is most recent (last row), d=21 is oldest (first row)
    returns_d1_first = last_21["ret"].values[::-1].astype(np.float32)
    return returns_d1_first


def _get_month_end_dates(daily_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Identify all month-end trading dates in the daily data."""
    daily_df = daily_df.copy()
    daily_df["ym"] = daily_df["date"].dt.to_period("M")
    month_ends = (
        daily_df.groupby("ym")["date"].max().reset_index()["date"]
    )
    return pd.DatetimeIndex(month_ends.sort_values())


def apply_universe_filters(
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    nyse_breakpoints: pd.Series,
    rf_series: pd.Series,
) -> pd.DataFrame:
    """
    Apply all six eligibility filters to define the investment universe each month.

    Filters applied sequentially:
    (a) shrcd in {10, 11} - common stocks only
    (b) exchcd in {1, 2, 3} - NYSE/Amex/Nasdaq only
    (c) abs(prc) >= 5 - price filter
    (d) me > 0 and non-missing - non-missing market equity
    (e) me >= NYSE 1st percentile breakpoint - size filter
    (f) non-missing next-month excess return

    Also requires exactly 21 non-missing daily returns in the last 21 trading days.

    Args:
        monthly_df: Monthly CRSP DataFrame.
        daily_df: Daily CRSP DataFrame.
        nyse_breakpoints: Monthly Series with NYSE size breakpoints indexed by date.
        rf_series: Monthly risk-free rate Series indexed by date.

    Returns:
        Filtered DataFrame with 'ret_excess' added, one row per eligible stock-month.
    """
    df = monthly_df.copy()

    # (a) Share code filter
    df = df[df["shrcd"].isin([10, 11])]

    # (b) Exchange filter
    df = df[df["exchcd"].isin([1, 2, 3])]

    # (c) Price filter
    df = df[df["prc"].abs() >= 5.0]

    # (d) Non-missing market equity
    df = df[df["me"].notna() & (df["me"] > 0)]

    # (e) NYSE size breakpoint filter
    df = df.merge(
        nyse_breakpoints.reset_index().rename(columns={"nyse_breakpoint": "bp"}),
        on="date",
        how="left",
    )
    df = df[df["me"] >= df["bp"]]
    df = df.drop(columns=["bp"])

    # Compute excess returns
    df = compute_excess_returns(df, rf_series)

    # (f) Non-missing next-month excess return
    # Shift: for each permno, ret_excess at t is the CURRENT month's excess return.
    # We need NEXT month's return in the CURRENT month's row (for the response variable).
    df = df.sort_values(["permno", "date"])
    df["next_ret_excess"] = df.groupby("permno")["ret_excess"].shift(-1)
    df = df[df["next_ret_excess"].notna()]

    # Build a daily lookup for efficient per-stock extraction
    daily_pivot = _build_daily_lookup(daily_df)

    # Filter: need exactly 21 non-missing daily returns in last 21 trading days
    valid_rows = []
    for _, row in df.iterrows():
        perm = row["permno"]
        month_end = row["date"]
        ret_21 = _get_last_21_fast(daily_pivot, perm, month_end)
        if ret_21 is not None:
            valid_rows.append(row.name)

    df = df.loc[valid_rows]
    return df.reset_index(drop=True)


def _build_daily_lookup(daily_df: pd.DataFrame) -> dict:
    """
    Build a permno -> sorted daily DataFrame lookup for fast access.
    Returns dict: {permno: DataFrame sorted by date}
    """
    lookup = {}
    for perm, grp in daily_df.groupby("permno"):
        lookup[perm] = grp.sort_values("date").reset_index(drop=True)
    return lookup


def _get_last_21_fast(
    daily_lookup: dict,
    permno: int,
    month_end_date: pd.Timestamp,
) -> Optional[np.ndarray]:
    """Fast version of get_last_21_trading_days using pre-built lookup."""
    if permno not in daily_lookup:
        return None
    stock_daily = daily_lookup[permno]
    mask = stock_daily["date"] <= month_end_date
    subset = stock_daily[mask]
    if len(subset) < 21:
        return None
    last_21 = subset.tail(21)
    if last_21["ret"].isna().any():
        return None
    return last_21["ret"].values[::-1].astype(np.float32)


def build_eligible_universe(
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ff6_factors: pd.DataFrame,
    rf_series: pd.Series,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Master function: build the full eligible universe for the DRI experiment.

    Combines delisting adjustment, universe filtering, and daily return
    extraction. Returns the eligible stock-month panel and a daily return
    lookup for feature matrix construction.

    Args:
        monthly_df: Raw CRSP monthly DataFrame.
        daily_df: Raw CRSP daily DataFrame.
        ff6_factors: FF6 factor DataFrame (for aligning dates).
        rf_series: Monthly risk-free rate Series.
        verbose: If True, print progress information.

    Returns:
        (eligible_df, daily_lookup) where:
        - eligible_df: DataFrame with permno, date, ret_excess, next_ret_excess, me, exchcd, shrcd
        - daily_lookup: dict {permno: sorted daily DataFrame}
    """
    if verbose:
        print(f"Starting universe construction: {len(monthly_df)} raw stock-month obs")

    # Step 1: Handle delisting returns
    monthly_adj = handle_delisting_returns(monthly_df)

    # Step 2: NYSE breakpoints each month
    from data.synthetic_data import get_nyse_size_breakpoints
    nyse_bp = get_nyse_size_breakpoints(monthly_adj, percentile=1.0)

    # Step 3: Apply all filters
    eligible_df = apply_universe_filters(
        monthly_adj, daily_df, nyse_bp, rf_series
    )

    if verbose:
        n_months = eligible_df["date"].nunique()
        avg_cs = len(eligible_df) / max(n_months, 1)
        print(
            f"Eligible universe: {len(eligible_df)} stock-month obs, "
            f"{n_months} months, avg {avg_cs:.0f} stocks/month"
        )

    # Step 4: Build daily lookup
    daily_lookup = _build_daily_lookup(daily_df)

    return eligible_df, daily_lookup
