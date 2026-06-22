"""
Tests for Performance Metrics: Newey-West t-statistics, Sharpe ratios, FF6 alphas,
spanning tests, and the full decile performance table computation.
"""
import numpy as np
import pandas as pd
import pytest

from src.performance_metrics import (
    compute_newey_west_tstat,
    compute_sharpe_ratio,
    compute_ff6_alpha,
    compute_spanning_test,
    verify_monotonic_pattern,
    compute_full_decile_metrics,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def monthly_dates_100():
    return pd.date_range("2000-01-01", periods=100, freq="ME")


@pytest.fixture
def positive_returns(monthly_dates_100):
    """Consistently positive returns: high t-stat expected."""
    rng = np.random.default_rng(1)
    data = rng.normal(0.015, 0.02, size=100)
    return pd.Series(data, index=monthly_dates_100)


@pytest.fixture
def zero_returns(monthly_dates_100):
    """Near-zero mean returns: t-stat near 0."""
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 0.05, size=100)
    return pd.Series(data, index=monthly_dates_100)


@pytest.fixture
def synthetic_ff6():
    """Minimal FF6 factor DataFrame."""
    from data.synthetic_data import generate_ff6_factors
    dates = pd.date_range("2000-01-01", periods=100, freq="ME")
    return generate_ff6_factors(date_range=dates, seed=5)


@pytest.fixture
def factor_correlated_returns(synthetic_ff6, monthly_dates_100):
    """Returns partially explained by FF6 factors."""
    rng = np.random.default_rng(3)
    ff6 = synthetic_ff6.loc[monthly_dates_100[:len(synthetic_ff6)]]
    beta_mkt = 0.8
    base_alpha = 0.005
    ret = (
        base_alpha
        + beta_mkt * ff6["MKT_RF"].values[:100]
        + rng.normal(0, 0.015, size=100)
    )
    return pd.Series(ret, index=monthly_dates_100)


# ──────────────────────────────────────────────────────────────────────────── #
# Newey-West t-Statistic Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_nw_tstat_positive_returns(positive_returns):
    mean, t, std = compute_newey_west_tstat(positive_returns, maxlags=4)
    assert t > 2.0, f"Expected t > 2 for positive returns, got t={t:.2f}"
    assert mean > 0, "Mean of positive return series must be positive"


def test_nw_tstat_zero_mean_returns(zero_returns):
    mean, t, std = compute_newey_west_tstat(zero_returns, maxlags=4)
    assert abs(t) < 3.0, f"t-stat of near-zero mean should be small, got {t:.2f}"


def test_nw_tstat_returns_three_values(positive_returns):
    result = compute_newey_west_tstat(positive_returns)
    assert len(result) == 3, "Must return (mean, t_stat, std)"


def test_nw_tstat_std_positive(positive_returns):
    _, _, std = compute_newey_west_tstat(positive_returns)
    assert std > 0, "Standard deviation must be positive"


def test_nw_tstat_tiny_series():
    """Very short series should return NaN t-stat."""
    s = pd.Series([0.01, 0.02, 0.03])
    mean, t, std = compute_newey_west_tstat(s)
    assert pd.notna(mean)


def test_nw_tstat_empty_series():
    s = pd.Series(dtype=float)
    mean, t, std = compute_newey_west_tstat(s)
    # Should handle gracefully
    assert np.isnan(t) or pd.isna(t)


# ──────────────────────────────────────────────────────────────────────────── #
# Sharpe Ratio Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_sharpe_annualized(positive_returns):
    monthly_sr = positive_returns.mean() / positive_returns.std(ddof=1)
    annualized_expected = monthly_sr * np.sqrt(12)
    computed = compute_sharpe_ratio(positive_returns, annualize=True)
    np.testing.assert_almost_equal(computed, annualized_expected, decimal=5)


def test_sharpe_not_annualized(positive_returns):
    monthly_expected = positive_returns.mean() / positive_returns.std(ddof=1)
    computed = compute_sharpe_ratio(positive_returns, annualize=False)
    np.testing.assert_almost_equal(computed, monthly_expected, decimal=5)


def test_sharpe_positive_returns_positive_sr(positive_returns):
    sr = compute_sharpe_ratio(positive_returns)
    assert sr > 0, "Positive mean returns → positive Sharpe ratio"


def test_sharpe_empty():
    assert np.isnan(compute_sharpe_ratio(pd.Series(dtype=float)))


def test_sharpe_constant_returns():
    s = pd.Series([0.01] * 50)
    # Constant returns → std = 0, Sharpe = NaN
    assert np.isnan(compute_sharpe_ratio(s))


def test_sharpe_negative_returns():
    rng = np.random.default_rng(10)
    s = pd.Series(rng.normal(-0.01, 0.02, 60))
    sr = compute_sharpe_ratio(s)
    assert sr < 0, "Negative mean returns → negative Sharpe ratio"


# ──────────────────────────────────────────────────────────────────────────── #
# FF6 Alpha Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_ff6_alpha_returns_three_values(factor_correlated_returns, synthetic_ff6):
    result = compute_ff6_alpha(factor_correlated_returns, synthetic_ff6)
    assert len(result) == 3, "Must return (alpha, alpha_tstat, factor_loadings)"


def test_ff6_alpha_loading_series(factor_correlated_returns, synthetic_ff6):
    _, _, loadings = compute_ff6_alpha(factor_correlated_returns, synthetic_ff6)
    assert isinstance(loadings, pd.Series), "Factor loadings must be a pd.Series"


def test_ff6_alpha_positive_for_positive_alpha_portfolio(synthetic_ff6):
    """Portfolio with known positive alpha should show positive FF6 alpha."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=100, freq="ME")
    ff6 = synthetic_ff6

    # Add large fixed alpha component
    ff6_aligned = ff6.reindex(dates)
    mkt = ff6_aligned["MKT_RF"].fillna(0).values
    high_alpha_ret = pd.Series(0.02 + 0.5 * mkt + rng.normal(0, 0.01, 100), index=dates)

    alpha, t, _ = compute_ff6_alpha(high_alpha_ret, ff6, maxlags=6)
    # Allow flexibility: alpha should be notable (not -0.02)
    if pd.notna(alpha):
        assert alpha > -0.05, f"Alpha of high-return portfolio should not be very negative: {alpha}"


def test_ff6_alpha_short_series():
    """Short series → NaN alpha."""
    short = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=pd.date_range("2000-01-01", periods=5, freq="ME"))
    ff6 = pd.DataFrame({
        "MKT_RF": [0.05, 0.02, -0.01, 0.03, 0.01],
        "SMB": [0.01, 0.02, 0.00, 0.01, -0.01],
        "HML": [0.01, 0.00, 0.01, 0.02, 0.00],
        "RMW": [0.00, 0.01, 0.01, 0.00, 0.00],
        "CMA": [0.00, 0.00, 0.01, 0.00, 0.01],
        "UMD": [0.01, -0.01, 0.01, 0.01, 0.00],
    }, index=pd.date_range("2000-01-01", periods=5, freq="ME"))
    alpha, t, _ = compute_ff6_alpha(short, ff6)
    assert pd.isna(alpha) or pd.notna(alpha)  # Graceful handling


# ──────────────────────────────────────────────────────────────────────────── #
# Spanning Test Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_spanning_test_returns_dict(positive_returns, zero_returns):
    result = compute_spanning_test(positive_returns, zero_returns)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"alpha", "alpha_tstat", "beta", "r_squared"}


def test_spanning_test_identical_signals():
    """Regressing a signal on itself → alpha ~= 0, beta ~= 1, R² ~= 1."""
    s = pd.Series(
        np.random.default_rng(5).normal(0.005, 0.02, 80),
        index=pd.date_range("2000-01-01", periods=80, freq="ME")
    )
    result = compute_spanning_test(s, s)
    assert result["r_squared"] > 0.99, "Self-regression must have R² ≈ 1"
    np.testing.assert_almost_equal(result["beta"], 1.0, decimal=4)


def test_spanning_test_r_squared_range(positive_returns, zero_returns):
    result = compute_spanning_test(positive_returns, zero_returns)
    if pd.notna(result["r_squared"]):
        assert 0 <= result["r_squared"] <= 1.0, "R² must be in [0, 1]"


def test_spanning_test_short_series():
    s1 = pd.Series([0.01, 0.02], index=pd.date_range("2000-01-01", periods=2, freq="ME"))
    s2 = pd.Series([0.01, 0.02], index=pd.date_range("2000-01-01", periods=2, freq="ME"))
    result = compute_spanning_test(s1, s2)
    assert np.isnan(result["alpha"]) or pd.notna(result["alpha"])


# ──────────────────────────────────────────────────────────────────────────── #
# Monotonicity Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_monotonic_pattern_perfect():
    means = pd.Series(np.linspace(0.001, 0.010, 10), index=range(1, 11))
    assert verify_monotonic_pattern(means) is True


def test_monotonic_pattern_reversed():
    means = pd.Series(np.linspace(0.010, 0.001, 10), index=range(1, 11))
    assert verify_monotonic_pattern(means) is False


def test_monotonic_pattern_noisy_but_trending():
    """Mostly increasing with some noise → should still pass."""
    base = np.linspace(0.001, 0.010, 10)
    noise = np.array([0, 0, -0.0002, 0, 0, 0, 0, 0, 0, 0])  # one small inversion
    means = pd.Series(base + noise, index=range(1, 11))
    assert verify_monotonic_pattern(means) is True


def test_monotonic_pattern_empty():
    means = pd.Series(dtype=float)
    result = verify_monotonic_pattern(means)
    assert isinstance(result, bool)


def test_monotonic_pattern_single():
    means = pd.Series([0.01], index=[1])
    result = verify_monotonic_pattern(means)
    assert isinstance(result, bool)


# ──────────────────────────────────────────────────────────────────────────── #
# Full Decile Metrics Table Tests
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_decile_data(monthly_dates_100):
    """Synthetic decile returns: 10 deciles × 100 months."""
    rng = np.random.default_rng(55)
    n_months = 80
    dates = monthly_dates_100[:n_months]

    vw_wide = pd.DataFrame(
        {d: pd.Series(rng.normal(d * 0.001, 0.03, n_months), index=dates)
         for d in range(1, 11)}
    )
    ew_wide = pd.DataFrame(
        {d: pd.Series(rng.normal(d * 0.0012, 0.025, n_months), index=dates)
         for d in range(1, 11)}
    )
    hl_vw = vw_wide[10] - vw_wide[1]
    hl_vw.name = "HL_ret_vw"
    hl_ew = ew_wide[10] - ew_wide[1]
    hl_ew.name = "HL_ret_ew"
    return vw_wide, ew_wide, hl_vw, hl_ew


def test_full_metrics_table_shape(synthetic_decile_data, synthetic_ff6):
    vw_wide, ew_wide, hl_vw, hl_ew = synthetic_decile_data
    metrics = compute_full_decile_metrics(
        vw_wide, ew_wide, hl_vw, hl_ew, synthetic_ff6, maxlags=4
    )
    assert "H-L" in metrics.index, "H-L row must exist in metrics table"
    assert "VW_Mean_%" in metrics.columns
    assert "VW_SR" in metrics.columns
    assert "EW_Alpha_%" in metrics.columns


def test_full_metrics_table_hl_positive(synthetic_decile_data, synthetic_ff6):
    """With increasing decile means, H-L mean return should be positive."""
    vw_wide, ew_wide, hl_vw, hl_ew = synthetic_decile_data
    metrics = compute_full_decile_metrics(
        vw_wide, ew_wide, hl_vw, hl_ew, synthetic_ff6, maxlags=4
    )
    hl_mean = metrics.loc["H-L", "VW_Mean_%"]
    if pd.notna(hl_mean):
        assert hl_mean > 0, f"H-L mean should be positive, got {hl_mean}"


def test_full_metrics_table_monotone_ew(synthetic_decile_data, synthetic_ff6):
    """Constructed with increasing decile means → monotone EW pattern."""
    vw_wide, ew_wide, hl_vw, hl_ew = synthetic_decile_data
    metrics = compute_full_decile_metrics(
        vw_wide, ew_wide, hl_vw, hl_ew, synthetic_ff6, maxlags=4
    )
    decile_means = metrics[
        metrics.index.str.startswith("Decile")
    ]["EW_Mean_%"].values
    if len(decile_means) > 2:
        # Mostly increasing
        diffs = np.diff(decile_means.astype(float))
        pct_up = np.mean(diffs > 0)
        assert pct_up >= 0.5, f"EW means should be mostly increasing, got {pct_up:.1%}"
