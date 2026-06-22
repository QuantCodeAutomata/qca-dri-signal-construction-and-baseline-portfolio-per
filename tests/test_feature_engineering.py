"""
Tests for Phase B: Feature Matrix Construction.

Validates chronological vectors, rank vectors, interaction features,
cross-sectional demeaning, and the full build_feature_matrix pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_chronological_vector,
    build_rank_vector,
    build_interaction_features,
    cross_sectional_demean_daily_returns,
    build_feature_matrix,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def sample_returns_21():
    """21 known daily demeaned returns for deterministic tests."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, size=21).astype(np.float32)


@pytest.fixture
def known_returns_21():
    """Explicitly known returns to test rank ordering."""
    return np.array([0.05, -0.03, 0.01, 0.08, -0.01, 0.02, -0.07,
                     0.04, 0.00, -0.02, 0.03, 0.06, -0.04, 0.01, -0.05,
                     0.02, 0.09, -0.08, 0.03, -0.01, 0.01], dtype=np.float32)


@pytest.fixture
def small_daily_df():
    """Small CRSP-like daily DataFrame with 3 stocks, 25 days."""
    dates = pd.bdate_range("2000-01-01", periods=25)
    permnos = [1, 2, 3]
    records = []
    rng = np.random.default_rng(99)
    for perm in permnos:
        rets = rng.normal(0.001, 0.015, size=25)
        for i, d in enumerate(dates):
            records.append({"permno": perm, "date": d, "ret": rets[i]})
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────── #
# Chronological Vector Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_chronological_vector_shape(sample_returns_21):
    R_C = build_chronological_vector(sample_returns_21)
    assert R_C.shape == (21,), "R^C must be 21-dimensional"


def test_chronological_vector_preserves_order(known_returns_21):
    """R^C must preserve the time ordering (d=1 first = most recent)."""
    R_C = build_chronological_vector(known_returns_21)
    np.testing.assert_array_equal(
        R_C, known_returns_21, err_msg="R^C must equal input in same order"
    )


def test_chronological_vector_dtype(sample_returns_21):
    R_C = build_chronological_vector(sample_returns_21)
    assert R_C.dtype == np.float32, "R^C must be float32"


def test_chronological_vector_wrong_length():
    with pytest.raises(AssertionError):
        build_chronological_vector(np.ones(20))


# ──────────────────────────────────────────────────────────────────────────── #
# Rank Vector Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_rank_vector_shape(sample_returns_21):
    R_R = build_rank_vector(sample_returns_21)
    assert R_R.shape == (21,), "R^R must be 21-dimensional"


def test_rank_vector_descending_order(known_returns_21):
    """R^R must be sorted descending: r^(1) >= r^(2) >= ... >= r^(21)."""
    R_R = build_rank_vector(known_returns_21)
    assert all(R_R[i] >= R_R[i + 1] for i in range(20)), \
        "R^R must be monotonically non-increasing (descending sort)"


def test_rank_vector_same_elements(known_returns_21):
    """R^R must contain exactly the same values as R^C, just reordered."""
    R_C = build_chronological_vector(known_returns_21)
    R_R = build_rank_vector(known_returns_21)
    np.testing.assert_array_almost_equal(
        np.sort(R_C)[::-1], R_R,
        err_msg="Sorted R^C must equal R^R"
    )


def test_rank_vector_max_first(known_returns_21):
    """First element of R^R must be the maximum return (MAX effect position)."""
    R_R = build_rank_vector(known_returns_21)
    assert R_R[0] == known_returns_21.max(), "k=1 must be maximum return"


def test_rank_vector_min_last(known_returns_21):
    """Last element of R^R must be the minimum return (MIN effect position)."""
    R_R = build_rank_vector(known_returns_21)
    assert R_R[-1] == known_returns_21.min(), "k=21 must be minimum return"


def test_rank_vector_dtype(sample_returns_21):
    R_R = build_rank_vector(sample_returns_21)
    assert R_R.dtype == np.float32, "R^R must be float32"


def test_rank_vector_wrong_length():
    with pytest.raises(AssertionError):
        build_rank_vector(np.ones(22))


# ──────────────────────────────────────────────────────────────────────────── #
# Interaction Feature Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_interaction_features_shape(sample_returns_21):
    Z_I = build_interaction_features(sample_returns_21)
    assert Z_I.shape == (441,), "Interaction vector must be 441-dimensional"


def test_interaction_features_sparsity(sample_returns_21):
    """Each row d has exactly one non-zero entry (its rank position)."""
    Z_I = build_interaction_features(sample_returns_21)
    Z_matrix = Z_I.reshape(21, 21)
    for d in range(21):
        n_nonzero = np.sum(Z_matrix[d, :] != 0)
        assert n_nonzero == 1, \
            f"Row d={d} must have exactly 1 nonzero entry, found {n_nonzero}"


def test_interaction_features_values_match_returns(sample_returns_21):
    """The non-zero value at row d must equal the return for day d."""
    Z_I = build_interaction_features(sample_returns_21)
    Z_matrix = Z_I.reshape(21, 21)
    for d in range(21):
        nonzero_vals = Z_matrix[d, Z_matrix[d, :] != 0]
        assert len(nonzero_vals) == 1
        np.testing.assert_almost_equal(
            nonzero_vals[0], sample_returns_21[d],
            decimal=5, err_msg=f"Interaction value at d={d} must match return"
        )


def test_interaction_features_dtype(sample_returns_21):
    Z_I = build_interaction_features(sample_returns_21)
    assert Z_I.dtype == np.float32, "Interaction vector must be float32"


# ──────────────────────────────────────────────────────────────────────────── #
# Cross-Sectional Demeaning Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_demeaning_zero_cross_section_mean(small_daily_df):
    """After demeaning, cross-sectional mean per date should be near zero."""
    demeaned_df = cross_sectional_demean_daily_returns(small_daily_df)
    cs_means = demeaned_df.groupby("date")["ret_demeaned"].mean()
    np.testing.assert_allclose(
        cs_means.values, 0.0, atol=1e-10,
        err_msg="Cross-sectional mean after demeaning must be zero"
    )


def test_demeaning_preserves_row_count(small_daily_df):
    demeaned_df = cross_sectional_demean_daily_returns(small_daily_df)
    assert len(demeaned_df) == len(small_daily_df), \
        "Demeaning must not change the number of rows"


def test_demeaning_original_ret_unchanged(small_daily_df):
    """Original 'ret' column must remain unmodified after demeaning."""
    original_rets = small_daily_df["ret"].values.copy()
    demeaned_df = cross_sectional_demean_daily_returns(small_daily_df)
    np.testing.assert_array_equal(
        demeaned_df["ret"].values, original_rets,
        err_msg="Original 'ret' column must not be modified"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Full Feature Matrix Tests
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_eligible_and_daily():
    """Small synthetic eligible universe and daily data for pipeline tests."""
    from data.synthetic_data import generate_synthetic_crsp_data
    from data.data_loader import load_ff6_factors, load_risk_free_rate

    daily_df, monthly_df = generate_synthetic_crsp_data(
        n_stocks=20, start_date="2000-01-01", end_date="2002-12-31", seed=7
    )
    rf_series = load_risk_free_rate(
        start_date="2000-01-01", end_date="2002-12-31", seed=7
    )

    from data.synthetic_data import get_nyse_size_breakpoints
    from src.data_prep import compute_excess_returns

    # Build a minimal eligible universe (no strict 21-day filter for speed)
    df = monthly_df[monthly_df["shrcd"].isin([10, 11]) & monthly_df["exchcd"].isin([1, 2, 3])].copy()
    df = df[df["me"].notna() & (df["me"] > 0)]
    df = compute_excess_returns(df, rf_series)
    df = df.sort_values(["permno", "date"])
    df["next_ret_excess"] = df.groupby("permno")["ret_excess"].shift(-1)
    df = df[df["next_ret_excess"].notna()].reset_index(drop=True)

    return df, daily_df


def test_feature_matrix_dimensions(synthetic_eligible_and_daily):
    eligible_df, daily_df = synthetic_eligible_and_daily
    if eligible_df.empty:
        pytest.skip("No eligible observations after filtering")
    meta_df, X, y = build_feature_matrix(eligible_df, daily_df)
    if X.shape[0] > 0:
        assert X.shape[1] == 42, f"Expected 42 features, got {X.shape[1]}"


def test_feature_matrix_no_nans(synthetic_eligible_and_daily):
    eligible_df, daily_df = synthetic_eligible_and_daily
    meta_df, X, y = build_feature_matrix(eligible_df, daily_df)
    if X.shape[0] > 0:
        assert not np.isnan(X).any(), "Feature matrix must not contain NaN"
        assert not np.isnan(y).any(), "Response vector must not contain NaN"


def test_feature_matrix_meta_alignment(synthetic_eligible_and_daily):
    eligible_df, daily_df = synthetic_eligible_and_daily
    meta_df, X, y = build_feature_matrix(eligible_df, daily_df)
    assert len(meta_df) == X.shape[0] == len(y), \
        "meta_df, X, and y must have same number of rows"


def test_feature_matrix_rc_rr_decomposition(synthetic_eligible_and_daily):
    """First 21 columns are R^C; last 21 are R^R (sorted descending of R^C)."""
    eligible_df, daily_df = synthetic_eligible_and_daily
    meta_df, X, y = build_feature_matrix(eligible_df, daily_df)
    if X.shape[0] == 0:
        pytest.skip("No feature matrix rows")
    for i in range(min(5, X.shape[0])):
        r_c = X[i, :21]
        r_r = X[i, 21:]
        r_r_expected = np.sort(r_c)[::-1]
        np.testing.assert_array_almost_equal(
            r_r, r_r_expected, decimal=5,
            err_msg=f"Row {i}: R^R must be descending sort of R^C"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Edge Case Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_rank_vector_all_equal():
    """Rank vector of all-equal returns must be all the same value."""
    returns = np.ones(21, dtype=np.float32) * 0.005
    R_R = build_rank_vector(returns)
    assert np.allclose(R_R, 0.005), "Rank vector of constant returns should be constant"


def test_rank_vector_extreme_values():
    """Rank vector with extreme values: check max at index 0, min at index 20."""
    returns = np.array([-100.0] + [0.0] * 19 + [100.0], dtype=np.float32)
    R_R = build_rank_vector(returns)
    assert R_R[0] == 100.0
    assert R_R[-1] == -100.0


def test_interaction_features_all_zero():
    """Interaction features of all-zero returns should be zero everywhere."""
    returns = np.zeros(21, dtype=np.float32)
    Z_I = build_interaction_features(returns)
    assert np.all(Z_I == 0.0), "All-zero returns → all-zero interaction features"
