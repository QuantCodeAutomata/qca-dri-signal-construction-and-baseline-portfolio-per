"""
Tests for Phase E: Decile Portfolio Formation.

Validates decile assignment, VW/EW return computation, H-L spread, and
the full build_decile_portfolios pipeline with synthetic data.
"""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_construction import (
    assign_decile_ranks,
    compute_vw_portfolio_returns,
    compute_ew_portfolio_returns,
    compute_hl_spread,
    build_decile_portfolios,
    pivot_portfolio_returns,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def simple_signal_df():
    """50 stocks in one month with known DRI scores."""
    n = 50
    rng = np.random.default_rng(77)
    date = pd.Timestamp("2001-01-31")
    return pd.DataFrame({
        "permno": range(1, n + 1),
        "date": date,
        "DRI": np.linspace(-1, 1, n),  # known linear DRI scores
    })


@pytest.fixture
def simple_eligible_df():
    """50 stocks with known returns and ME for one month."""
    n = 50
    rng = np.random.default_rng(88)
    date = pd.Timestamp("2001-01-31")
    return pd.DataFrame({
        "permno": range(1, n + 1),
        "date": date,
        "next_ret_excess": rng.normal(0.005, 0.03, n),
        "me": rng.uniform(1e8, 1e10, n),
    })


@pytest.fixture
def multi_month_signal_df():
    """100 stocks in each of 12 months."""
    n_stocks = 100
    n_months = 12
    rng = np.random.default_rng(11)
    records = []
    dates = pd.date_range("2001-01-31", periods=n_months, freq="ME")
    for date in dates:
        for perm in range(1, n_stocks + 1):
            records.append({
                "permno": perm,
                "date": date,
                "DRI": rng.normal(0, 1),
            })
    return pd.DataFrame(records)


@pytest.fixture
def multi_month_eligible_df():
    """100 stocks in each of 12 months with returns and ME."""
    n_stocks = 100
    n_months = 12
    rng = np.random.default_rng(22)
    records = []
    dates = pd.date_range("2001-01-31", periods=n_months, freq="ME")
    for date in dates:
        for perm in range(1, n_stocks + 1):
            records.append({
                "permno": perm,
                "date": date,
                "next_ret_excess": rng.normal(0.002, 0.04),
                "me": rng.uniform(5e7, 5e9),
            })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────── #
# Decile Assignment Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_decile_ranks_range(simple_signal_df):
    ranked = assign_decile_ranks(simple_signal_df)
    deciles = ranked["decile"].dropna().values
    assert np.all(deciles >= 1), "All deciles must be >= 1"
    assert np.all(deciles <= 10), "All deciles must be <= 10"


def test_decile_ranks_all_ten_deciles(simple_signal_df):
    """With 50 stocks, all 10 deciles should be populated."""
    ranked = assign_decile_ranks(simple_signal_df)
    unique_deciles = set(ranked["decile"].dropna().astype(int).values)
    assert len(unique_deciles) == 10, \
        f"Expected 10 unique deciles, got {len(unique_deciles)}"


def test_decile_ranks_linear_signal(simple_signal_df):
    """With linear DRI scores, highest DRI stocks should be in decile 10."""
    ranked = assign_decile_ranks(simple_signal_df)
    # The stock with highest DRI (permno=50) should be in decile 10
    top_stock = ranked[ranked["permno"] == 50]["decile"].values[0]
    assert top_stock == 10, f"Highest DRI stock must be in decile 10, got {top_stock}"
    # The stock with lowest DRI (permno=1) should be in decile 1
    bottom_stock = ranked[ranked["permno"] == 1]["decile"].values[0]
    assert bottom_stock == 1, f"Lowest DRI stock must be in decile 1, got {bottom_stock}"


def test_decile_ranks_multi_month(multi_month_signal_df):
    """Decile assignment must be done independently each month."""
    ranked = assign_decile_ranks(multi_month_signal_df)
    for date, group in ranked.groupby("date"):
        deciles = group["decile"].dropna().astype(int).values
        assert np.all(deciles >= 1) and np.all(deciles <= 10), \
            f"Month {date}: deciles out of range"


# ──────────────────────────────────────────────────────────────────────────── #
# VW Portfolio Return Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_vw_returns_shape(simple_signal_df, simple_eligible_df):
    ranked = assign_decile_ranks(simple_signal_df)
    vw_returns = compute_vw_portfolio_returns(ranked, simple_eligible_df)
    assert "ret_vw" in vw_returns.columns
    assert "decile" in vw_returns.columns
    assert "date" in vw_returns.columns


def test_vw_returns_ten_deciles(simple_signal_df, simple_eligible_df):
    ranked = assign_decile_ranks(simple_signal_df)
    vw_returns = compute_vw_portfolio_returns(ranked, simple_eligible_df)
    unique_deciles = vw_returns["decile"].nunique()
    assert unique_deciles == 10, f"Expected 10 decile portfolios, got {unique_deciles}"


def test_vw_weights_sum_to_one(simple_signal_df, simple_eligible_df):
    """VW return = ME-weighted average, so weights sum to 1 within each decile."""
    ranked = assign_decile_ranks(simple_signal_df)
    # Manually compute expected VW return for decile 5
    merged = ranked.merge(simple_eligible_df, on=["permno", "date"])
    d5 = merged[merged["decile"] == 5]
    w = d5["me"] / d5["me"].sum()
    expected = float((w * d5["next_ret_excess"]).sum())

    vw_returns = compute_vw_portfolio_returns(ranked, simple_eligible_df)
    computed = float(vw_returns[vw_returns["decile"] == 5]["ret_vw"].values[0])
    np.testing.assert_almost_equal(computed, expected, decimal=8)


# ──────────────────────────────────────────────────────────────────────────── #
# EW Portfolio Return Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_ew_returns_shape(simple_signal_df, simple_eligible_df):
    ranked = assign_decile_ranks(simple_signal_df)
    ew_returns = compute_ew_portfolio_returns(ranked, simple_eligible_df)
    assert "ret_ew" in ew_returns.columns


def test_ew_return_equals_simple_mean(simple_signal_df, simple_eligible_df):
    """EW return = simple average of stock returns within each decile."""
    ranked = assign_decile_ranks(simple_signal_df)
    merged = ranked.merge(simple_eligible_df, on=["permno", "date"])
    d3 = merged[merged["decile"] == 3]
    expected = float(d3["next_ret_excess"].mean())

    ew_returns = compute_ew_portfolio_returns(ranked, simple_eligible_df)
    computed = float(ew_returns[ew_returns["decile"] == 3]["ret_ew"].values[0])
    np.testing.assert_almost_equal(computed, expected, decimal=8)


# ──────────────────────────────────────────────────────────────────────────── #
# H-L Spread Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_hl_spread_definition(multi_month_signal_df, multi_month_eligible_df):
    """H-L = D10 return - D1 return for each month."""
    ranked = assign_decile_ranks(multi_month_signal_df)
    vw_returns = compute_vw_portfolio_returns(ranked, multi_month_eligible_df)

    hl = compute_hl_spread(vw_returns, return_col="ret_vw")

    # Verify for each date
    for date in hl.index:
        d10 = float(vw_returns[(vw_returns["date"] == date) & (vw_returns["decile"] == 10)]["ret_vw"].iloc[0])
        d1 = float(vw_returns[(vw_returns["date"] == date) & (vw_returns["decile"] == 1)]["ret_vw"].iloc[0])
        expected = d10 - d1
        np.testing.assert_almost_equal(
            float(hl.loc[date]), expected, decimal=8,
            err_msg=f"H-L mismatch at {date}"
        )


def test_hl_spread_length(multi_month_signal_df, multi_month_eligible_df):
    ranked = assign_decile_ranks(multi_month_signal_df)
    vw_returns = compute_vw_portfolio_returns(ranked, multi_month_eligible_df)
    hl = compute_hl_spread(vw_returns, return_col="ret_vw")
    n_months = multi_month_signal_df["date"].nunique()
    assert len(hl) == n_months, \
        f"H-L spread must have one entry per month, got {len(hl)}"


def test_hl_spread_empty_when_missing_deciles():
    df = pd.DataFrame({"date": ["2000-01-01"], "decile": [5], "ret_vw": [0.01]})
    hl = compute_hl_spread(df, return_col="ret_vw")
    assert hl.empty, "H-L must be empty when deciles 1 or 10 are missing"


# ──────────────────────────────────────────────────────────────────────────── #
# Full Pipeline Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_build_decile_portfolios_returns_four(multi_month_signal_df, multi_month_eligible_df):
    vw, ew, hl_vw, hl_ew = build_decile_portfolios(
        signal_df=multi_month_signal_df,
        eligible_df=multi_month_eligible_df,
        signal_col="DRI",
    )
    assert isinstance(vw, pd.DataFrame)
    assert isinstance(ew, pd.DataFrame)
    assert isinstance(hl_vw, pd.Series)
    assert isinstance(hl_ew, pd.Series)


def test_build_decile_portfolios_hl_length(multi_month_signal_df, multi_month_eligible_df):
    _, _, hl_vw, hl_ew = build_decile_portfolios(
        multi_month_signal_df, multi_month_eligible_df
    )
    n_months = multi_month_signal_df["date"].nunique()
    assert len(hl_vw) == n_months, "VW H-L must have one entry per month"
    assert len(hl_ew) == n_months, "EW H-L must have one entry per month"


def test_pivot_portfolio_returns_shape(multi_month_signal_df, multi_month_eligible_df):
    vw, _, _, _ = build_decile_portfolios(multi_month_signal_df, multi_month_eligible_df)
    wide = pivot_portfolio_returns(vw, "ret_vw")
    assert wide.shape[1] == 10, "Wide format must have 10 decile columns"
    n_months = multi_month_signal_df["date"].nunique()
    assert wide.shape[0] == n_months, "Wide format must have one row per month"


# ──────────────────────────────────────────────────────────────────────────── #
# Mathematical Property Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_vw_ew_difference_with_varying_ME(simple_signal_df, simple_eligible_df):
    """
    VW and EW returns should differ when ME is non-uniform
    (as in real data where large-cap stocks dominate VW portfolios).
    """
    ranked = assign_decile_ranks(simple_signal_df)
    vw_returns = compute_vw_portfolio_returns(ranked, simple_eligible_df)
    ew_returns = compute_ew_portfolio_returns(ranked, simple_eligible_df)

    vw_vals = vw_returns["ret_vw"].values
    ew_vals = ew_returns["ret_ew"].values

    # With random ME, VW ≠ EW in at least some deciles
    assert not np.allclose(vw_vals, ew_vals, atol=1e-10), \
        "VW and EW returns should differ with non-uniform ME"


def test_decile_coverage_all_stocks_assigned(simple_signal_df):
    """Every stock must be assigned a decile (no NaN deciles with complete data)."""
    ranked = assign_decile_ranks(simple_signal_df)
    assert ranked["decile"].isna().sum() == 0, \
        "No stock should have a NaN decile with clean data"
