"""
Synthetic CRSP-like data generator for DRI signal experiments.
Produces daily returns, monthly data, FF6 factors, and risk-free rates
that mimic the structure of CRSP databases.

Signal injection: monthly returns include a cross-sectionally predictable
component driven by intra-month daily return patterns (reversal + rank effects),
matching the DRI methodology described in the paper.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def _inject_dri_signal(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    gamma_reversal: float = 0.5,
    gamma_lag2: float = 0.25,
    gamma_lag5: float = 0.13,
    gamma_max: float = 0.25,
    gamma_min: float = 0.13,
) -> pd.DataFrame:
    """
    Inject a DRI-like predictive signal into monthly returns.

    For each month t, computes cross-sectionally demeaned daily returns,
    then adds a predictive term to the NEXT month (t+1) return:
        signal_{i,t} = -gamma_reversal * r_tilde_{i,t,d=1}   (reversal, d=1)
                       -gamma_lag2     * r_tilde_{i,t,d=2}   (lag-2 reversal)
                       -gamma_lag5     * r_tilde_{i,t,d=5}   (lag-5 reversal)
                       -gamma_max      * r_tilde^(1)_{i,t}   (MAX effect)
                       +gamma_min      * r_tilde^(21)_{i,t}  (MIN effect)

    Signal amplitude: with daily idio std ~0.010 after cross-sectional demeaning,
    gamma_reversal=0.5 → signal std ~0.8%/month, giving SNR ~10–12% relative
    to typical monthly return volatility of 7%.  With N ≈ 15,000 training
    observations this implies t-stat > 10, making the signal reliably detectable
    by elastic-net while keeping monthly return swings small enough that
    cumulative prices stay well above the $5 eligibility filter.

    This creates:
    - beta^C[d=1] < 0 (largest magnitude, short-term reversal)
    - beta^C[d=2] < 0, beta^C[d=5] < 0 (weaker lag reversals)
    - beta^R[k=1] < 0 (MAX effect)
    - beta^R[k=21] > 0 (MIN effect)

    Args:
        daily_df: Daily returns DataFrame (permno, date, ret).
        monthly_df: Monthly DataFrame (permno, date, ret).
        gamma_reversal: Reversal coefficient for most-recent daily return (d=1).
        gamma_lag2: Reversal coefficient for second-most-recent return (d=2).
        gamma_lag5: Reversal coefficient for fifth-most-recent return (d=5).
        gamma_max: MAX-return effect coefficient (rank k=1).
        gamma_min: MIN-return effect coefficient (rank k=21).

    Returns:
        Monthly DataFrame with modified 'ret' column.
    """
    daily = daily_df.copy()
    monthly = monthly_df.copy().sort_values(["permno", "date"]).reset_index(drop=True)

    daily["ym"] = daily["date"].dt.to_period("M")

    # Build signal table: (permno, year, month) -> signal_value
    signal_records = {}

    for ym, grp_m in daily.groupby("ym"):
        # Cross-sectional demeaning per calendar date
        cs_mean = grp_m.groupby("date")["ret"].transform("mean")
        grp_m = grp_m.copy()
        grp_m["ret_dm"] = grp_m["ret"] - cs_mean

        # Last 21 trading days of this month
        month_dates = sorted(grp_m["date"].unique())
        last_21_dates = month_dates[-21:] if len(month_dates) >= 21 else month_dates
        grp_window = grp_m[grp_m["date"].isin(last_21_dates)]

        for perm, g_stock in grp_window.groupby("permno"):
            g_s = g_stock.sort_values("date")
            dm_rets = g_s["ret_dm"].dropna().values
            if len(dm_rets) < 21:
                continue
            dm_rets = dm_rets[-21:]  # oldest first; index -1 = most recent (d=1)

            # d=1: most recent, d=2: second most recent, d=5: fifth most recent
            r_d1 = dm_rets[-1]
            r_d2 = dm_rets[-2]
            r_d5 = dm_rets[-5]
            r_max = dm_rets.max()   # r_tilde^(1): largest (k=1)
            r_min = dm_rets.min()   # r_tilde^(21): smallest (k=21)

            sig = (
                -gamma_reversal * r_d1
                - gamma_lag2 * r_d2
                - gamma_lag5 * r_d5
                - gamma_max * r_max
                + gamma_min * r_min
            )

            signal_records[(int(perm), ym.year, ym.month)] = float(sig)

    # Apply signal to NEXT month's return
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    def _prev_signal(row):
        yr, mo = int(row["year"]), int(row["month"])
        prev_yr, prev_mo = (yr - 1, 12) if mo == 1 else (yr, mo - 1)
        return signal_records.get((int(row["permno"]), prev_yr, prev_mo), 0.0)

    monthly["ret"] = monthly["ret"] + monthly.apply(_prev_signal, axis=1)
    return monthly.drop(columns=["year", "month"])


def generate_synthetic_crsp_data(
    n_stocks: int = 150,
    start_date: str = "1985-01-01",
    end_date: str = "2010-12-31",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic CRSP daily and monthly data with realistic structure.

    Uses a factor model with an injected DRI signal:
    - Common market factor + idiosyncratic component for daily returns
    - Cross-sectionally predictable reversal/rank signal injected into monthly returns

    Args:
        n_stocks: Number of simulated stocks.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        seed: Random seed for reproducibility.

    Returns:
        (daily_df, monthly_df):
            daily_df columns: permno, date, ret, shrcd, exchcd
            monthly_df columns: permno, date, ret, prc, shrout, me, shrcd, exchcd, dlret
    """
    rng = np.random.default_rng(seed)

    permnos = np.arange(10001, 10001 + n_stocks)
    shrcd = rng.choice([10, 11], size=n_stocks, p=[0.70, 0.30])
    exchcd = rng.choice([1, 2, 3], size=n_stocks, p=[0.60, 0.15, 0.25])

    bdays = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(bdays)

    # Market factor: AR(1) with stronger positive drift to keep prices above $5
    # Daily drift 0.0008/day ≈ 20%/year ensures long-run price stability
    _daily_drift = 0.0008
    mkt_factor = np.zeros(n_days)
    mkt_factor[0] = rng.normal(_daily_drift, 0.01)
    for t in range(1, n_days):
        mkt_factor[t] = 0.05 * mkt_factor[t - 1] + rng.normal(_daily_drift, 0.01)

    betas = rng.uniform(0.5, 1.5, size=n_stocks)
    # Start prices high enough to survive 20+ years of volatility
    initial_prices = rng.uniform(50.0, 200.0, size=n_stocks)
    shrout_base = rng.uniform(1e6, 500e6, size=n_stocks)

    # ---------- Build daily returns ----------
    # Idio std reduced to 0.010 for tighter price paths and better signal detectability
    all_daily = []
    for i, perm in enumerate(permnos):
        idio = rng.normal(0.0, 0.010, size=n_days)
        ret = betas[i] * mkt_factor + idio

        # ~1% missing data (reduced from 2% to maximise eligible universe)
        missing_mask = rng.random(size=n_days) < 0.01
        ret_f = ret.astype(float)
        ret_f[missing_mask] = np.nan

        all_daily.append(pd.DataFrame({
            "permno": perm,
            "date": bdays,
            "ret": ret_f,
            "shrcd": shrcd[i],
            "exchcd": exchcd[i],
        }))

    daily_df = pd.concat(all_daily, ignore_index=True)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    # ---------- Build monthly data (compound daily returns) ----------
    daily_df["year_month"] = daily_df["date"].dt.to_period("M")

    monthly_records = []
    for (perm, ym), grp in daily_df.groupby(["permno", "year_month"]):
        valid_rets = grp["ret"].dropna()
        if len(valid_rets) == 0:
            continue
        monthly_records.append({
            "permno": perm,
            "date": grp["date"].max(),
            "ret": (1 + valid_rets).prod() - 1,
        })

    monthly_df = pd.DataFrame(monthly_records)
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])

    # ---- Inject DRI predictive signal ----
    # Gammas calibrated so that signal_std ≈ 0.8%/month (< 12% of monthly
    # return std), keeping prices stable while remaining statistically detectable
    # with ~15,000+ training observations (implied t-stat > 10).
    monthly_df = _inject_dri_signal(
        daily_df=daily_df.drop(columns=["year_month"]),
        monthly_df=monthly_df,
        gamma_reversal=0.5,
        gamma_lag2=0.25,
        gamma_lag5=0.13,
        gamma_max=0.25,
        gamma_min=0.13,
    )

    idx_map = {int(perm): i for i, perm in enumerate(permnos)}
    monthly_df = monthly_df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Cumulative price
    prices = []
    for perm, grp in monthly_df.groupby("permno"):
        i = idx_map[int(perm)]
        p = initial_prices[i]
        plist = []
        for r in grp["ret"].values:
            p = p * (1 + r)
            plist.append(p)
        prices.extend(plist)
    monthly_df["prc"] = prices

    monthly_df["shrout"] = monthly_df["permno"].map(
        lambda p: shrout_base[idx_map[int(p)]]
    ) * (1 + rng.normal(0, 0.001, size=len(monthly_df)))
    monthly_df["me"] = monthly_df["prc"].abs() * monthly_df["shrout"]
    monthly_df["shrcd"] = monthly_df["permno"].map(
        lambda p: shrcd[idx_map[int(p)]]
    )
    monthly_df["exchcd"] = monthly_df["permno"].map(
        lambda p: exchcd[idx_map[int(p)]]
    )
    monthly_df["dlret"] = np.nan
    delist_mask = rng.random(size=len(monthly_df)) < 0.005
    monthly_df.loc[delist_mask, "dlret"] = rng.choice(
        [-0.30, -0.55, -0.10], size=delist_mask.sum()
    )

    daily_df = daily_df.drop(columns=["year_month"])
    return daily_df, monthly_df


def generate_ff6_factors(
    date_range: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Fama-French 6 factors (monthly).

    Columns: date, MKT_RF, SMB, HML, RMW, CMA, UMD, RF

    Args:
        date_range: Monthly date index.
        seed: Random seed.

    Returns:
        DataFrame indexed by date with FF6 factor columns plus RF.
    """
    rng = np.random.default_rng(seed)
    n = len(date_range)

    corr = np.array([
        [1.00, 0.20, -0.15,  0.05, -0.10, -0.10],
        [0.20, 1.00,  0.10,  0.05,  0.05, -0.05],
        [-0.15, 0.10, 1.00,  0.30,  0.40, -0.20],
        [0.05, 0.05,  0.30,  1.00,  0.35, -0.10],
        [-0.10, 0.05, 0.40,  0.35,  1.00, -0.15],
        [-0.10, -0.05, -0.20, -0.10, -0.15, 1.00],
    ])
    std = np.array([0.045, 0.030, 0.030, 0.022, 0.020, 0.040])
    means = np.array([0.006, 0.002, 0.003, 0.002, 0.002, 0.007])

    cov = np.diag(std) @ corr @ np.diag(std)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(6))

    raw = rng.standard_normal((n, 6)) @ L.T + means

    rf_base = 0.003
    rf_drift = np.cumsum(rng.normal(0, 0.0003, size=n))
    rf = np.clip(rf_base + rf_drift, 0.0001, 0.008)

    factors = pd.DataFrame(raw, columns=["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"])
    factors["RF"] = rf
    factors["date"] = pd.DatetimeIndex(date_range)
    factors = factors.set_index("date")
    return factors


def get_nyse_size_breakpoints(
    monthly_df: pd.DataFrame,
    percentile: float = 1.0,
) -> pd.Series:
    """
    Compute NYSE market cap percentile breakpoint each month.

    Only uses NYSE stocks (exchcd=1) consistent with standard CRSP size filters.

    Args:
        monthly_df: Monthly DataFrame with columns [permno, date, me, exchcd].
        percentile: Percentile to compute (default 1.0 = 1st percentile).

    Returns:
        Series indexed by date with breakpoint value each month.
    """
    nyse = monthly_df[monthly_df["exchcd"] == 1].copy()
    return (
        nyse.groupby("date")["me"]
        .quantile(percentile / 100.0)
        .rename("nyse_breakpoint")
    )
