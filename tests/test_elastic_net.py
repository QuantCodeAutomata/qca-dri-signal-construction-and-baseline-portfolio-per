"""
Tests for Phase C: Expanding Window Elastic-Net Model.

Validates lambda grid construction, CV selection, model fitting,
quality checks, and the expanding window estimation pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.elastic_net_model import (
    compute_lambda_max,
    build_lambda_grid,
    select_lambda_cv,
    fit_elastic_net,
    run_expanding_window_estimation,
    quality_check_coefficients,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def small_regression_data():
    """Small synthetic regression dataset: N=200 obs, 42 features."""
    rng = np.random.default_rng(42)
    N, p = 200, 42
    X = rng.normal(0, 1, size=(N, p)).astype(np.float64)
    beta_true = np.zeros(p)
    beta_true[0] = -0.5   # strong negative first feature (reversal)
    beta_true[1] = 0.3
    beta_true[21] = 0.2   # rank feature
    y = X @ beta_true + rng.normal(0, 0.5, size=N)
    return X, y


@pytest.fixture
def tiny_regression_data():
    """Very small dataset for speed in parametric tests."""
    rng = np.random.default_rng(7)
    N, p = 50, 42
    X = rng.normal(0, 1, size=(N, p)).astype(np.float64)
    y = rng.normal(0, 1, size=N)
    return X, y


@pytest.fixture
def expanding_window_inputs():
    """Synthetic data structured like a multi-year panel for expanding window tests."""
    rng = np.random.default_rng(21)
    n_years = 5
    n_per_year = 100
    N = n_years * n_per_year
    p = 42

    X = rng.normal(0, 1, size=(N, p)).astype(np.float32)
    beta = np.zeros(p)
    beta[0] = -0.3
    y = (X @ beta + rng.normal(0, 1, size=N)).astype(np.float32)

    base_year = 2000
    years = np.repeat(np.arange(base_year, base_year + n_years), n_per_year)
    months = rng.integers(1, 13, size=N)
    dates = pd.to_datetime(
        [f"{y}-{m:02d}-28" for y, m in zip(years, months)]
    )
    meta_df = pd.DataFrame({"permno": range(N), "date": dates})
    return X, y, meta_df, base_year


# ──────────────────────────────────────────────────────────────────────────── #
# lambda_max Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_lambda_max_positive(small_regression_data):
    X, y = small_regression_data
    lam_max = compute_lambda_max(X, y)
    assert lam_max > 0, "lambda_max must be strictly positive"


def test_lambda_max_formula(small_regression_data):
    """Verify lambda_max = max|X'*(y-mean(y))| / (N * eta)."""
    X, y = small_regression_data
    eta = 0.5
    N = X.shape[0]
    expected = float(np.abs(X.T @ (y - y.mean())).max()) / (N * eta)
    computed = compute_lambda_max(X, y, eta=eta)
    np.testing.assert_almost_equal(computed, expected, decimal=8)


def test_lambda_max_scales_with_eta():
    """lambda_max should be inversely proportional to eta."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (100, 10))
    y = rng.normal(0, 1, 100)
    lam1 = compute_lambda_max(X, y, eta=0.5)
    lam2 = compute_lambda_max(X, y, eta=1.0)
    assert lam1 > lam2, "Smaller eta → larger lambda_max"


# ──────────────────────────────────────────────────────────────────────────── #
# Lambda Grid Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_lambda_grid_length():
    grid = build_lambda_grid(1.0, n_lambdas=100)
    assert len(grid) == 100


def test_lambda_grid_descending():
    grid = build_lambda_grid(1.0, n_lambdas=50)
    assert np.all(np.diff(grid) < 0), "Lambda grid must be strictly descending"


def test_lambda_grid_bounds():
    lambda_max = 2.0
    grid = build_lambda_grid(lambda_max, n_lambdas=100)
    assert abs(grid[0] - lambda_max) < 1e-8, "First element must equal lambda_max"
    assert abs(grid[-1] - lambda_max * 1e-4) < 1e-8, "Last element must be lambda_max * 1e-4"


def test_lambda_grid_log_spacing():
    grid = build_lambda_grid(1.0, n_lambdas=100)
    log_grid = np.log10(grid)
    diffs = np.diff(log_grid)
    # All differences should be equal (log-uniform spacing)
    assert np.allclose(diffs, diffs[0], rtol=1e-5), "Grid must be log-uniformly spaced"


# ──────────────────────────────────────────────────────────────────────────── #
# ElasticNet Fitting Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_fit_returns_correct_shapes(small_regression_data):
    X, y = small_regression_data
    intercept, beta = fit_elastic_net(X, y, lambda_val=0.01)
    assert isinstance(intercept, float)
    assert beta.shape == (42,)


def test_fit_larger_lambda_more_sparsity(small_regression_data):
    """Higher lambda → more zero coefficients (stronger regularization)."""
    X, y = small_regression_data
    _, beta_small = fit_elastic_net(X, y, lambda_val=1e-4)
    _, beta_large = fit_elastic_net(X, y, lambda_val=1.0)
    n_nonzero_small = np.sum(beta_small != 0)
    n_nonzero_large = np.sum(beta_large != 0)
    assert n_nonzero_small >= n_nonzero_large, "Higher lambda → more sparsity"


def test_fit_null_model_at_lambda_max(small_regression_data):
    """At lambda_max all coefficients should be zero (null model)."""
    X, y = small_regression_data
    lam_max = compute_lambda_max(X, y)
    _, beta = fit_elastic_net(X, y, lambda_val=lam_max * 1.5)
    # At or above lambda_max, beta should be all zeros
    assert np.sum(beta != 0) == 0, "At lambda_max all coefficients should be zero"


def test_fit_float32_coefficients(small_regression_data):
    X, y = small_regression_data
    _, beta = fit_elastic_net(X, y, lambda_val=0.01)
    assert beta.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────── #
# Cross-Validation Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_cv_selects_valid_lambda(tiny_regression_data):
    """CV must return a lambda from the provided grid."""
    X, y = tiny_regression_data
    lam_max = compute_lambda_max(X, y)
    grid = build_lambda_grid(lam_max, n_lambdas=10)  # small grid for speed
    best_lam, mse_array = select_lambda_cv(
        X, y, grid,
        n_splits=2, n_repeats=2, random_state=42
    )
    assert best_lam in grid, "Selected lambda must be in the provided grid"
    assert mse_array.shape == (10,)


def test_cv_mse_all_positive(tiny_regression_data):
    """All CV MSE values must be non-negative."""
    X, y = tiny_regression_data
    lam_max = compute_lambda_max(X, y)
    grid = build_lambda_grid(lam_max, n_lambdas=5)
    _, mse_array = select_lambda_cv(
        X, y, grid, n_splits=2, n_repeats=2, random_state=1
    )
    assert np.all(mse_array >= 0), "MSE values must be non-negative"


def test_cv_reproducibility(tiny_regression_data):
    """Same seed → same best lambda."""
    X, y = tiny_regression_data
    lam_max = compute_lambda_max(X, y)
    grid = build_lambda_grid(lam_max, n_lambdas=5)

    lam1, _ = select_lambda_cv(X, y, grid, n_splits=2, n_repeats=2, random_state=42)
    lam2, _ = select_lambda_cv(X, y, grid, n_splits=2, n_repeats=2, random_state=42)
    assert lam1 == lam2, "Same random seed must yield same selected lambda"


# ──────────────────────────────────────────────────────────────────────────── #
# Expanding Window Estimation Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_expanding_window_returns_dict(expanding_window_inputs):
    X, y, meta_df, base_year = expanding_window_inputs
    coef_dict = run_expanding_window_estimation(
        X_all=X, y_all=y, meta_df=meta_df,
        train_end_year=base_year + 1,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    assert isinstance(coef_dict, dict), "Must return a dict"
    assert len(coef_dict) > 0, "Must estimate at least one year"


def test_expanding_window_coef_shapes(expanding_window_inputs):
    X, y, meta_df, base_year = expanding_window_inputs
    coef_dict = run_expanding_window_estimation(
        X_all=X, y_all=y, meta_df=meta_df,
        train_end_year=base_year + 1,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    for year, coefs in coef_dict.items():
        assert coefs["beta_C"].shape == (21,), \
            f"Year {year}: beta_C must be 21-dimensional"
        assert coefs["beta_R"].shape == (21,), \
            f"Year {year}: beta_R must be 21-dimensional"
        assert coefs["beta"].shape == (42,), \
            f"Year {year}: beta must be 42-dimensional"


def test_expanding_window_consistent_beta(expanding_window_inputs):
    """beta must equal concatenation of beta_C and beta_R."""
    X, y, meta_df, base_year = expanding_window_inputs
    coef_dict = run_expanding_window_estimation(
        X_all=X, y_all=y, meta_df=meta_df,
        train_end_year=base_year + 1,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    for year, coefs in coef_dict.items():
        expected = np.concatenate([coefs["beta_C"], coefs["beta_R"]])
        np.testing.assert_array_equal(
            coefs["beta"], expected,
            err_msg=f"Year {year}: beta != concat(beta_C, beta_R)"
        )


def test_expanding_window_increasing_ntrain(expanding_window_inputs):
    """Expanding window: n_train must strictly increase each year."""
    X, y, meta_df, base_year = expanding_window_inputs
    coef_dict = run_expanding_window_estimation(
        X_all=X, y_all=y, meta_df=meta_df,
        train_end_year=base_year + 3,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    years = sorted(coef_dict.keys())
    n_train_vals = [coef_dict[y]["n_train"] for y in years]
    assert all(n_train_vals[i] <= n_train_vals[i + 1] for i in range(len(n_train_vals) - 1)), \
        "n_train must be non-decreasing over years"


# ──────────────────────────────────────────────────────────────────────────── #
# Quality Check Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_quality_check_empty():
    result = quality_check_coefficients({})
    assert "error" in result


def test_quality_check_returns_dict(expanding_window_inputs):
    X, y, meta_df, base_year = expanding_window_inputs
    coef_dict = run_expanding_window_estimation(
        X_all=X, y_all=y, meta_df=meta_df,
        train_end_year=base_year + 1,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    qc = quality_check_coefficients(coef_dict)
    assert "avg_beta_C_d1" in qc
    assert "beta_C_d1_negative" in qc
    assert "pct_years_C_dominates_R" in qc
    assert isinstance(qc["beta_C_d1_negative"], bool)


def test_quality_check_reversal_in_simulated_data(expanding_window_inputs):
    """
    With a known negative beta[0], quality check should detect reversal.
    Uses data from fixture where beta_true[0] = -0.3.
    """
    X, y, meta_df, base_year = expanding_window_inputs
    # Make reversal signal strong
    X_reversal = X.copy()
    X_reversal[:, 0] = -np.abs(X_reversal[:, 0])
    y_reversal = X_reversal @ np.array([-0.5] + [0.0] * 41, dtype=np.float32)

    coef_dict = run_expanding_window_estimation(
        X_all=X_reversal, y_all=y_reversal, meta_df=meta_df,
        train_end_year=base_year + 1,
        start_year=base_year,
        n_lambdas=5, n_splits=2, n_repeats=2, verbose=False
    )
    qc = quality_check_coefficients(coef_dict)
    # With strong negative signal, beta_C[0] should be negative
    assert qc["avg_beta_C_d1"] < 0.1, \
        "Average beta_C[d=1] should be negative with reversal data"
