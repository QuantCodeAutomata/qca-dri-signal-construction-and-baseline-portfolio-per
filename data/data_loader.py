"""
Data loader using Massive API for real data or synthetic data for testing.
Provides a unified interface for CRSP daily/monthly data, FF6 factors,
and risk-free rates.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional

try:
    from massive import RESTClient
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False

from data.synthetic_data import (
    generate_synthetic_crsp_data,
    generate_ff6_factors,
    get_nyse_size_breakpoints,
)

# Module-level cache to avoid regenerating synthetic data multiple times
_SYNTHETIC_CACHE: dict = {}


def _get_synthetic_data(
    start_date: str,
    end_date: str,
    n_stocks: int = 150,
    seed: int = 42,
) -> tuple:
    """Return cached synthetic (daily_df, monthly_df) pair."""
    cache_key = (start_date, end_date, n_stocks, seed)
    if cache_key not in _SYNTHETIC_CACHE:
        daily_df, monthly_df = generate_synthetic_crsp_data(
            n_stocks=n_stocks,
            start_date=start_date,
            end_date=end_date,
            seed=seed,
        )
        _SYNTHETIC_CACHE[cache_key] = (daily_df, monthly_df)
    return _SYNTHETIC_CACHE[cache_key]


def load_crsp_daily(
    start_date: str = "1985-01-01",
    end_date: str = "2010-12-31",
    use_synthetic: bool = True,
    n_stocks: int = 150,
    seed: int = 42,
    *,
    tickers: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load CRSP daily stock data.

    For real data (use_synthetic=False), fetches daily OHLCV bars via the
    Massive API and returns a CRSP-compatible DataFrame. For testing and
    demonstration, synthetic data is used by default.

    Args:
        tickers: List of ticker symbols (only used when use_synthetic=False).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        use_synthetic: If True, use synthetic data generator.
        n_stocks: Number of stocks for synthetic data.
        seed: Random seed for synthetic data.

    Returns:
        DataFrame with columns: permno, date, ret, shrcd, exchcd
    """
    if not use_synthetic and MASSIVE_AVAILABLE and tickers:
        api_key = os.getenv("MASSIVE_TOKEN", "")
        client = RESTClient(api_key=api_key)
        records = []
        for ticker in tickers:
            aggs = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000,
            ):
                aggs.append(a)
            for agg in aggs:
                # Compute daily return from open/close
                if hasattr(agg, "open") and agg.open and agg.close:
                    ret = (agg.close - agg.open) / agg.open
                else:
                    ret = np.nan
                records.append({
                    "permno": hash(ticker) % 100000,
                    "date": pd.Timestamp(agg.timestamp, unit="ms"),
                    "ret": ret,
                    "shrcd": 10,
                    "exchcd": 1,
                })
        if records:
            return pd.DataFrame(records)

    daily_df, _ = _get_synthetic_data(start_date, end_date, n_stocks, seed)
    return daily_df


def load_crsp_monthly(
    start_date: str = "1985-01-01",
    end_date: str = "2010-12-31",
    use_synthetic: bool = True,
    n_stocks: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load CRSP monthly stock data.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        use_synthetic: If True, use synthetic data.
        n_stocks: Number of stocks for synthetic data.
        seed: Random seed for synthetic data.

    Returns:
        DataFrame with columns: permno, date, ret, prc, shrout, me,
                                 shrcd, exchcd, dlret
    """
    if not use_synthetic and MASSIVE_AVAILABLE:
        raise NotImplementedError(
            "Real CRSP monthly data requires WRDS access. Use use_synthetic=True."
        )

    _, monthly_df = _get_synthetic_data(start_date, end_date, n_stocks, seed)
    return monthly_df


def load_ff6_factors(
    start_date: str = "1985-01-01",
    end_date: str = "2010-12-31",
    use_synthetic: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load Fama-French 6 factors (monthly).

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        use_synthetic: If True, generate synthetic factors.
        seed: Random seed for synthetic data.

    Returns:
        DataFrame indexed by date with columns:
        MKT_RF, SMB, HML, RMW, CMA, UMD, RF
    """
    if not use_synthetic:
        try:
            import pandas_datareader.data as web
            ff = web.DataReader(
                "F-F_Research_Data_5_Factors_2x3",
                "famafrench",
                start=start_date,
                end=end_date,
            )
            mom = web.DataReader(
                "F-F_Momentum_Factor",
                "famafrench",
                start=start_date,
                end=end_date,
            )
            factors = ff[0] / 100.0
            factors["UMD"] = mom[0].iloc[:, 0] / 100.0
            factors.index = pd.to_datetime(factors.index.to_timestamp())
            return factors
        except Exception:
            pass  # Fall through to synthetic

    # Generate monthly date range
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq="ME")
    return generate_ff6_factors(date_range=monthly_dates, seed=seed)


def load_risk_free_rate(
    start_date: str = "1985-01-01",
    end_date: str = "2010-12-31",
    use_synthetic: bool = True,
    seed: int = 42,
) -> pd.Series:
    """
    Load monthly risk-free rate (one-month T-bill).

    Args:
        start_date: Start date string.
        end_date: End date string.
        use_synthetic: If True, use synthetic risk-free rate.
        seed: Random seed.

    Returns:
        Series indexed by date with monthly risk-free rate values.
    """
    ff6 = load_ff6_factors(
        start_date=start_date,
        end_date=end_date,
        use_synthetic=use_synthetic,
        seed=seed,
    )
    return ff6["RF"].rename("rf")


def clear_cache() -> None:
    """Clear the synthetic data cache."""
    global _SYNTHETIC_CACHE
    _SYNTHETIC_CACHE = {}
