"""
Phase E: Decile Portfolio Formation.

Forms value-weighted (VW) and equal-weighted (EW) decile portfolios sorted
on the DRI signal each month. Computes High-Low (H-L) spreads.

Custom — Context7 found no library equivalent for this exact paper methodology.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def assign_decile_ranks(
    signal_df: pd.DataFrame,
    signal_col: str = "DRI",
) -> pd.DataFrame:
    """
    Assign decile ranks (1=lowest, 10=highest) to stocks each month.

    Uses all-stock breakpoints within each month: decile 1 = bottom 10%
    of DRI signal, decile 10 = top 10%.

    Args:
        signal_df: DataFrame with columns [permno, date, signal_col].
        signal_col: Name of the signal column to rank.

    Returns:
        Input DataFrame with added 'decile' column (integer 1–10).
    """
    df = signal_df.copy()
    decile_parts: list[pd.Series] = []

    for _date, group in df.groupby("date"):
        labels = None
        try:
            labels = pd.qcut(
                group[signal_col],
                q=10,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            pass

        # Fall back when qcut returns all-NaN (constant signal) or raises
        if labels is None or labels.isna().all():
            n = len(group)
            labels = pd.Series(
                np.ceil(
                    group[signal_col].rank(method="first") / n * 10
                ).clip(1, 10).astype(int),
                index=group.index,
            )
        else:
            # qcut labels=False gives 0–9; shift to 1–10
            labels = labels + 1

        decile_parts.append(labels.rename("decile"))

    df["decile"] = pd.concat(decile_parts).astype("Int64")
    return df


def compute_vw_portfolio_returns(
    signal_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    signal_col: str = "DRI",
) -> pd.DataFrame:
    """
    Compute value-weighted decile portfolio returns.

    For decile p in month t:
    R^VW_{p,t+1} = sum_i(w_{i,t} * r^m_{i,t+1})
    where w_{i,t} = ME_{i,t} / sum_{j in p}(ME_{j,t})

    The portfolio return in month t is the return of stocks that were
    SORTED into decile p at end of month t-1, holding for month t.
    The variable 'next_ret_excess' in eligible_df is the NEXT month's return.

    Args:
        signal_df: DataFrame with [permno, date, signal_col, decile].
        eligible_df: DataFrame with [permno, date, next_ret_excess, me].
        signal_col: Signal column name (unused here, carried for consistency).

    Returns:
        DataFrame with columns: date, decile, ret_vw
        (one row per date-decile pair)
    """
    # Merge signal/decile info with returns and market cap
    merged = signal_df[["permno", "date", "decile"]].merge(
        eligible_df[["permno", "date", "next_ret_excess", "me"]],
        on=["permno", "date"],
        how="inner",
    )
    merged = merged.dropna(subset=["next_ret_excess", "me", "decile"])

    # Vectorised VW: weight = ME_i / sum(ME) within each (date, decile) group
    grp_total_me = merged.groupby(["date", "decile"])["me"].transform("sum")
    # Zero-ME groups → weight = 0; avoids division by zero
    merged = merged[grp_total_me > 0].copy()
    grp_total_me = grp_total_me[grp_total_me > 0]
    merged["_weight"] = merged["me"] / grp_total_me
    merged["_wret"] = merged["_weight"] * merged["next_ret_excess"]
    result = (
        merged.groupby(["date", "decile"])["_wret"]
        .sum()
        .reset_index()
        .rename(columns={"_wret": "ret_vw"})
    )
    return result


def compute_ew_portfolio_returns(
    signal_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    signal_col: str = "DRI",
) -> pd.DataFrame:
    """
    Compute equal-weighted decile portfolio returns.

    R^EW_{p,t+1} = (1/N_p) * sum_{i in p}(r^m_{i,t+1})

    Args:
        signal_df: DataFrame with [permno, date, signal_col, decile].
        eligible_df: DataFrame with [permno, date, next_ret_excess].
        signal_col: Signal column name.

    Returns:
        DataFrame with columns: date, decile, ret_ew
    """
    merged = signal_df[["permno", "date", "decile"]].merge(
        eligible_df[["permno", "date", "next_ret_excess"]],
        on=["permno", "date"],
        how="inner",
    )
    merged = merged.dropna(subset=["next_ret_excess", "decile"])

    result = (
        merged.groupby(["date", "decile"])["next_ret_excess"]
        .mean()
        .reset_index(name="ret_ew")
    )
    return result


def compute_hl_spread(
    portfolio_df: pd.DataFrame,
    return_col: str = "ret_vw",
) -> pd.Series:
    """
    Compute the High-Low (H-L) spread as Decile 10 - Decile 1 return.

    Args:
        portfolio_df: DataFrame with columns [date, decile, return_col].
            Must contain deciles 1 and 10 for each date.
        return_col: Column name of portfolio returns.

    Returns:
        Series indexed by date with H-L spread values.
    """
    pivot = portfolio_df.pivot(index="date", columns="decile", values=return_col)

    if 10 not in pivot.columns or 1 not in pivot.columns:
        return pd.Series(dtype=float)

    hl = pivot[10] - pivot[1]
    hl.name = "HL_" + return_col
    return hl.dropna()


def build_decile_portfolios(
    signal_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    signal_col: str = "DRI",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Master function: assign deciles, compute VW and EW returns, H-L spreads.

    Args:
        signal_df: DataFrame with [permno, date, signal_col].
        eligible_df: DataFrame with [permno, date, next_ret_excess, me].
        signal_col: Column name of the DRI signal.

    Returns:
        (vw_returns, ew_returns, hl_vw, hl_ew) where:
        - vw_returns: DataFrame [date, decile, ret_vw]
        - ew_returns: DataFrame [date, decile, ret_ew]
        - hl_vw: Series of H-L VW returns indexed by date
        - hl_ew: Series of H-L EW returns indexed by date
    """
    # Step 1: Assign deciles
    ranked_df = assign_decile_ranks(signal_df, signal_col=signal_col)

    # Step 2: Compute portfolio returns
    vw_returns = compute_vw_portfolio_returns(ranked_df, eligible_df, signal_col)
    ew_returns = compute_ew_portfolio_returns(ranked_df, eligible_df, signal_col)

    # Step 3: H-L spreads
    hl_vw = compute_hl_spread(vw_returns, return_col="ret_vw")
    hl_ew = compute_hl_spread(ew_returns, return_col="ret_ew")

    return vw_returns, ew_returns, hl_vw, hl_ew


def pivot_portfolio_returns(
    returns_df: pd.DataFrame,
    return_col: str,
) -> pd.DataFrame:
    """
    Pivot portfolio returns from long to wide format (date × decile).

    Args:
        returns_df: Long DataFrame with [date, decile, return_col].
        return_col: Name of the return column.

    Returns:
        Wide DataFrame with deciles as columns (1–10), dates as rows.
    """
    wide = returns_df.pivot(index="date", columns="decile", values=return_col)
    wide.columns.name = None
    return wide
