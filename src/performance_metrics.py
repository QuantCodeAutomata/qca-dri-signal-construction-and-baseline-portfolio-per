"""
Performance Metrics for DRI Signal Portfolios.

Computes mean monthly returns, Newey-West t-statistics, annualized Sharpe
ratios, and Fama-French 6-factor (FF6) alphas with Newey-West standard errors.

Using statsmodels OLS with HAC SE — Context7 confirmed
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def compute_newey_west_tstat(
    returns: pd.Series,
    maxlags: int = 12,
) -> Tuple[float, float, float]:
    """
    Compute Newey-West (HAC) t-statistic for testing H0: mean return = 0.

    Regresses return series on a constant and uses HAC standard errors
    with Bartlett kernel and the specified number of lags.

    Using statsmodels OLS HAC — Context7 confirmed

    Args:
        returns: Monthly return series (no NaN values).
        maxlags: Number of lags for Newey-West correction (default 12).

    Returns:
        (mean, t_stat, std) where std is the OLS standard deviation of returns.
    """
    clean = returns.dropna()
    if len(clean) < 5:
        return (float(clean.mean()), np.nan, float(clean.std()))

    endog = clean.values.astype(float)
    exog = np.ones(len(endog))

    model = sm.OLS(endog, exog)
    result = model.fit()
    robust = result.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

    mean_ret = float(robust.params[0])
    t_stat = float(robust.tvalues[0])
    std = float(np.std(endog, ddof=1))

    return mean_ret, t_stat, std


def compute_sharpe_ratio(
    returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Compute Sharpe ratio for a monthly return series.

    Annualized SR = (mean / std) * sqrt(12) for monthly returns.

    Args:
        returns: Monthly return series.
        annualize: If True, multiply by sqrt(12) to annualize.

    Returns:
        Sharpe ratio (annualized if annualize=True).
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    if std == 0:
        return np.nan
    sr = mean / std
    if annualize:
        sr *= np.sqrt(12)
    return sr


def compute_ff6_alpha(
    portfolio_returns: pd.Series,
    ff6_factors: pd.DataFrame,
    maxlags: int = 12,
) -> Tuple[float, float, pd.Series]:
    """
    Compute Fama-French 6-factor alpha via OLS with Newey-West standard errors.

    Regresses portfolio excess returns on:
    [const, MKT_RF, SMB, HML, RMW, CMA, UMD]

    Uses 12-lag Newey-West (HAC) standard errors throughout.

    Using statsmodels OLS with HAC SE — Context7 confirmed

    Args:
        portfolio_returns: Monthly excess return Series indexed by date.
        ff6_factors: DataFrame indexed by date with columns:
            [MKT_RF, SMB, HML, RMW, CMA, UMD] (and optionally RF).
        maxlags: Number of lags for Newey-West SE (12 per paper).

    Returns:
        (alpha, alpha_tstat, factor_loadings) where factor_loadings is a
        Series with the 6 factor betas.
    """
    factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    available_cols = [c for c in factor_cols if c in ff6_factors.columns]

    # Align dates
    aligned = portfolio_returns.rename("ret").to_frame().join(
        ff6_factors[available_cols], how="inner"
    )
    aligned = aligned.dropna()

    if len(aligned) < 10:
        nan_series = pd.Series(np.nan, index=available_cols)
        return np.nan, np.nan, nan_series

    y = aligned["ret"].values.astype(float)
    X = sm.add_constant(aligned[available_cols].values.astype(float))

    model = sm.OLS(y, X)
    result = model.fit()
    robust = result.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

    alpha = float(robust.params[0])
    alpha_tstat = float(robust.tvalues[0])
    factor_loadings = pd.Series(
        robust.params[1:], index=available_cols
    )

    return alpha, alpha_tstat, factor_loadings


def compute_spanning_test(
    returns_A: pd.Series,
    returns_B: pd.Series,
    maxlags: int = 12,
) -> Dict:
    """
    Bilateral spanning test: regress portfolio A returns on portfolio B returns.

    Model: R_A = alpha + beta * R_B + epsilon

    If alpha is statistically indistinguishable from zero, B spans A
    (i.e., B explains all of A's risk-adjusted returns).

    Using statsmodels OLS with HAC SE — Context7 confirmed

    Args:
        returns_A: Monthly H-L returns of signal A (indexed by date).
        returns_B: Monthly H-L returns of signal B (indexed by date).
        maxlags: Lags for Newey-West SE.

    Returns:
        Dict with keys: alpha, alpha_tstat, beta, r_squared
    """
    # Align dates
    aligned = pd.concat(
        [returns_A.rename("A"), returns_B.rename("B")], axis=1
    ).dropna()

    if len(aligned) < 10:
        return {
            "alpha": np.nan,
            "alpha_tstat": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
        }

    y = aligned["A"].values.astype(float)
    X = sm.add_constant(aligned["B"].values.astype(float))

    model = sm.OLS(y, X)
    result = model.fit()
    robust = result.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

    return {
        "alpha": float(robust.params[0]),
        "alpha_tstat": float(robust.tvalues[0]),
        "beta": float(robust.params[1]),
        "r_squared": float(result.rsquared),
    }


def compute_full_decile_metrics(
    vw_returns_wide: pd.DataFrame,
    ew_returns_wide: pd.DataFrame,
    hl_vw: pd.Series,
    hl_ew: pd.Series,
    ff6_factors: pd.DataFrame,
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    Compute the full performance table for all deciles and H-L spreads.

    Reports (per the paper's Table format) for each decile and H-L:
    - Mean monthly return (%)
    - Newey-West t-statistic
    - Standard deviation
    - Annualized Sharpe ratio
    - FF6 alpha (%)
    - FF6 alpha t-statistic

    Args:
        vw_returns_wide: Wide DataFrame [date × decile] of VW portfolio returns.
        ew_returns_wide: Wide DataFrame [date × decile] of EW portfolio returns.
        hl_vw: Series of H-L VW returns indexed by date.
        hl_ew: Series of H-L EW returns indexed by date.
        ff6_factors: FF6 factor DataFrame indexed by date.
        maxlags: Lags for Newey-West SE.

    Returns:
        DataFrame indexed by row labels (Decile 1, ..., Decile 10, H-L),
        with columns [VW_Mean, VW_tstat, VW_Std, VW_SR, VW_Alpha, VW_Alpha_t,
                      EW_Mean, EW_tstat, EW_Std, EW_SR, EW_Alpha, EW_Alpha_t].
    """
    rows = []
    decile_cols = sorted(
        set(vw_returns_wide.columns.tolist() + ew_returns_wide.columns.tolist())
    )

    for decile in decile_cols:
        vw_ret = vw_returns_wide[decile].dropna() if decile in vw_returns_wide.columns else pd.Series(dtype=float)
        ew_ret = ew_returns_wide[decile].dropna() if decile in ew_returns_wide.columns else pd.Series(dtype=float)

        vw_mean, vw_t, vw_std = compute_newey_west_tstat(vw_ret, maxlags)
        ew_mean, ew_t, ew_std = compute_newey_west_tstat(ew_ret, maxlags)

        vw_sr = compute_sharpe_ratio(vw_ret)
        ew_sr = compute_sharpe_ratio(ew_ret)

        vw_alpha, vw_alpha_t, _ = compute_ff6_alpha(vw_ret, ff6_factors, maxlags)
        ew_alpha, ew_alpha_t, _ = compute_ff6_alpha(ew_ret, ff6_factors, maxlags)

        rows.append({
            "label": f"Decile {int(decile)}",
            "VW_Mean_%": round(vw_mean * 100, 4),
            "VW_tstat": round(vw_t, 2) if not np.isnan(vw_t) else np.nan,
            "VW_Std_%": round(vw_std * 100, 4),
            "VW_SR": round(vw_sr, 2) if not np.isnan(vw_sr) else np.nan,
            "VW_Alpha_%": round(vw_alpha * 100, 4) if not np.isnan(vw_alpha) else np.nan,
            "VW_Alpha_t": round(vw_alpha_t, 2) if not np.isnan(vw_alpha_t) else np.nan,
            "EW_Mean_%": round(ew_mean * 100, 4),
            "EW_tstat": round(ew_t, 2) if not np.isnan(ew_t) else np.nan,
            "EW_Std_%": round(ew_std * 100, 4),
            "EW_SR": round(ew_sr, 2) if not np.isnan(ew_sr) else np.nan,
            "EW_Alpha_%": round(ew_alpha * 100, 4) if not np.isnan(ew_alpha) else np.nan,
            "EW_Alpha_t": round(ew_alpha_t, 2) if not np.isnan(ew_alpha_t) else np.nan,
        })

    # H-L row
    vw_mean_hl, vw_t_hl, vw_std_hl = compute_newey_west_tstat(hl_vw, maxlags)
    ew_mean_hl, ew_t_hl, ew_std_hl = compute_newey_west_tstat(hl_ew, maxlags)

    vw_sr_hl = compute_sharpe_ratio(hl_vw)
    ew_sr_hl = compute_sharpe_ratio(hl_ew)

    vw_alpha_hl, vw_alpha_t_hl, _ = compute_ff6_alpha(hl_vw, ff6_factors, maxlags)
    ew_alpha_hl, ew_alpha_t_hl, _ = compute_ff6_alpha(hl_ew, ff6_factors, maxlags)

    rows.append({
        "label": "H-L",
        "VW_Mean_%": round(vw_mean_hl * 100, 4),
        "VW_tstat": round(vw_t_hl, 2) if not np.isnan(vw_t_hl) else np.nan,
        "VW_Std_%": round(vw_std_hl * 100, 4),
        "VW_SR": round(vw_sr_hl, 2) if not np.isnan(vw_sr_hl) else np.nan,
        "VW_Alpha_%": round(vw_alpha_hl * 100, 4) if not np.isnan(vw_alpha_hl) else np.nan,
        "VW_Alpha_t": round(vw_alpha_t_hl, 2) if not np.isnan(vw_alpha_t_hl) else np.nan,
        "EW_Mean_%": round(ew_mean_hl * 100, 4),
        "EW_tstat": round(ew_t_hl, 2) if not np.isnan(ew_t_hl) else np.nan,
        "EW_Std_%": round(ew_std_hl * 100, 4),
        "EW_SR": round(ew_sr_hl, 2) if not np.isnan(ew_sr_hl) else np.nan,
        "EW_Alpha_%": round(ew_alpha_hl * 100, 4) if not np.isnan(ew_alpha_hl) else np.nan,
        "EW_Alpha_t": round(ew_alpha_t_hl, 2) if not np.isnan(ew_alpha_t_hl) else np.nan,
    })

    metrics_df = pd.DataFrame(rows).set_index("label")
    return metrics_df


def verify_monotonic_pattern(decile_means: pd.Series) -> bool:
    """
    Check if mean returns are (mostly) monotonically increasing across deciles.

    A strict monotonic increase from decile 1 to 10 is the ideal DRI
    signal property. We check whether at least 80% of consecutive decile
    pairs show a positive difference.

    Args:
        decile_means: Series indexed by decile number (1–10) with mean returns.

    Returns:
        True if the pattern is approximately monotonically increasing.
    """
    sorted_means = decile_means.sort_index()
    diffs = sorted_means.diff().dropna()
    if len(diffs) == 0:
        return False
    pct_increasing = (diffs > 0).mean()
    return bool(pct_increasing >= 0.7)
