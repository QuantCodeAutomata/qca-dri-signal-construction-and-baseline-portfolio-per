"""
Tests for Phase D: DRI Signal Computation.

Validates DRI formula, masked component signals, individual estimation signals,
and edge cases in signal computation.
"""
import numpy as np
import pandas as pd
import pytest

from src.signal_computation import (
    compute_dri_signal,
    compute_masked_signals,
    compute_individual_signals,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_coef_dict():
    """Known coefficient dict for 3 years."""
    rng = np.random.default_rng(10)
    coef_dict = {}
    for year in [2000, 2001, 2002]:
        beta_C = rng.normal(0, 0.1, 21).astype(np.float32)
        beta_R = rng.normal(0, 0.1, 21).astype(np.float32)
        beta = np.concatenate([beta_C, beta_R]).astype(np.float32)
        coef_dict[year] = {
            "intercept": 0.001,
            "beta_C": beta_C,
            "beta_R": beta_R,
            "beta": beta,
            "lambda": 0.01,
            "n_train": 1000,
        }
    return coef_dict


@pytest.fixture
def synthetic_feature_data(synthetic_coef_dict):
    """Synthetic X (N=300, 42) and meta_df covering 3 test years."""
    rng = np.random.default_rng(20)
    n_stocks_per_month = 30
    test_years = [2001, 2002, 2003]
    records = []
    X_rows = []

    for year in test_years:
        for month in range(1, 5):  # 4 months per year
            date = pd.Timestamp(f"{year}-{month:02d}-28")
            for perm in range(1, n_stocks_per_month + 1):
                records.append({
                    "permno": perm,
                    "date": date,
                    "me": rng.uniform(1e8, 1e10),
                    "exchcd": 1,
                })
                X_rows.append(rng.normal(0, 0.01, 42).astype(np.float32))

    meta_df = pd.DataFrame(records)
    X = np.array(X_rows, dtype=np.float32)
    return X, meta_df


# ──────────────────────────────────────────────────────────────────────────── #
# DRI Signal Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_dri_signal_returns_dataframe(synthetic_coef_dict, synthetic_feature_data):
    X, meta_df = synthetic_feature_data
    result = compute_dri_signal(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    assert isinstance(result, pd.DataFrame)


def test_dri_signal_columns(synthetic_coef_dict, synthetic_feature_data):
    X, meta_df = synthetic_feature_data
    result = compute_dri_signal(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    assert "permno" in result.columns
    assert "date" in result.columns
    assert "DRI" in result.columns


def test_dri_signal_formula(synthetic_coef_dict, synthetic_feature_data):
    """
    DRI_{i,t} = X[i,0:21] @ beta_C_Y + X[i,21:42] @ beta_R_Y.
    Verify against manual computation for a specific row.
    """
    X, meta_df = synthetic_feature_data

    result = compute_dri_signal(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )

    if result.empty:
        pytest.skip("No DRI values computed")

    # Get first row's DRI value
    row = result.iloc[0]
    date = row["date"]
    perm = row["permno"]

    # Find the corresponding X row
    meta_idx = meta_df[
        (meta_df["permno"] == perm) & (meta_df["date"] == date)
    ].index

    if len(meta_idx) == 0:
        pytest.skip("Cannot find matching meta row")

    idx = meta_idx[0]

    # Use coefficients from the preceding year
    year = pd.Timestamp(date).year
    coef_year = year - 1
    if coef_year not in synthetic_coef_dict:
        pytest.skip("No coefficients for preceding year")

    beta_C = synthetic_coef_dict[coef_year]["beta_C"]
    beta_R = synthetic_coef_dict[coef_year]["beta_R"]

    # Find position of idx in meta_df for test observations
    test_meta = meta_df[meta_df["date"].dt.year >= 2001]
    if idx not in test_meta.index:
        pytest.skip("Index not in test subset")

    loc = test_meta.index.get_loc(idx)
    X_test = X[meta_df["date"].dt.year >= 2001]
    expected_dri = float(X_test[loc, :21] @ beta_C + X_test[loc, 21:] @ beta_R)

    np.testing.assert_almost_equal(
        float(row["DRI"]), expected_dri, decimal=5,
        err_msg="DRI does not match manual formula computation"
    )


def test_dri_signal_test_period_only(synthetic_coef_dict, synthetic_feature_data):
    """DRI must only be computed for the test period (not training period)."""
    X, meta_df = synthetic_feature_data
    result = compute_dri_signal(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2002
    )
    if not result.empty:
        min_year = pd.to_datetime(result["date"]).dt.year.min()
        assert min_year >= 2002, \
            f"DRI should only include test period (>=2002), got {min_year}"


def test_dri_no_nans(synthetic_coef_dict, synthetic_feature_data):
    X, meta_df = synthetic_feature_data
    result = compute_dri_signal(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    if not result.empty:
        assert not result["DRI"].isna().any(), "DRI signal must not contain NaN"


# ──────────────────────────────────────────────────────────────────────────── #
# Masked Signals Tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_masked_signals_columns(synthetic_coef_dict, synthetic_feature_data):
    X, meta_df = synthetic_feature_data
    result = compute_masked_signals(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    assert "DRI" in result.columns
    assert "DRI_C" in result.columns
    assert "DRI_R" in result.columns


def test_masked_signals_additivity(synthetic_coef_dict, synthetic_feature_data):
    """DRI = DRI_C + DRI_R (perfect additivity of linear signal)."""
    X, meta_df = synthetic_feature_data
    result = compute_masked_signals(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    if result.empty:
        pytest.skip("No masked signals computed")

    dri_sum = result["DRI_C"] + result["DRI_R"]
    np.testing.assert_array_almost_equal(
        result["DRI"].values, dri_sum.values, decimal=5,
        err_msg="DRI must equal DRI_C + DRI_R"
    )


def test_masked_signals_c_dominates_r_with_reversal(synthetic_feature_data):
    """With strong first-day reversal, chronological component should vary more."""
    X, meta_df = synthetic_feature_data

    # Build coefficient dict with strong reversal in chronological component
    rng = np.random.default_rng(100)
    coef_dict_reversal = {}
    for year in [2000, 2001, 2002]:
        beta_C = np.zeros(21, dtype=np.float32)
        beta_C[0] = -1.0  # strong reversal
        beta_R = np.zeros(21, dtype=np.float32) + 0.01  # weak rank signal
        coef_dict_reversal[year] = {
            "intercept": 0.0,
            "beta_C": beta_C,
            "beta_R": beta_R,
            "beta": np.concatenate([beta_C, beta_R]),
        }

    result = compute_masked_signals(
        X=X, meta_df=meta_df, coefficients_dict=coef_dict_reversal,
        test_start_year=2001
    )

    if not result.empty:
        std_C = result["DRI_C"].std()
        std_R = result["DRI_R"].std()
        assert std_C > std_R, \
            "With strong reversal, chronological component should have higher variance"


# ──────────────────────────────────────────────────────────────────────────── #
# Individual Signals Tests
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def individual_coef_dicts():
    """Separate coefficient dicts for C and R models."""
    rng = np.random.default_rng(30)
    coef_C = {}
    coef_R = {}
    for year in [2000, 2001, 2002]:
        b_C = rng.normal(0, 0.1, 21).astype(np.float32)
        b_R = rng.normal(0, 0.1, 21).astype(np.float32)
        coef_C[year] = {"beta": b_C, "intercept": 0.001}
        coef_R[year] = {"beta": b_R, "intercept": 0.001}
    return coef_C, coef_R


def test_individual_signals_columns(
    individual_coef_dicts, synthetic_feature_data
):
    X, meta_df = synthetic_feature_data
    coef_C, coef_R = individual_coef_dicts
    result = compute_individual_signals(
        X_C=X[:, :21],
        X_R=X[:, 21:],
        meta_df=meta_df,
        coefficients_C_dict=coef_C,
        coefficients_R_dict=coef_R,
        test_start_year=2001,
    )
    assert "DRI_C_indiv" in result.columns
    assert "DRI_R_indiv" in result.columns


def test_individual_signals_formula(
    individual_coef_dicts, synthetic_feature_data
):
    """DRI_C_indiv = X[:,0:21] @ beta_C (from individual model)."""
    X, meta_df = synthetic_feature_data
    coef_C, coef_R = individual_coef_dicts
    result = compute_individual_signals(
        X_C=X[:, :21],
        X_R=X[:, 21:],
        meta_df=meta_df,
        coefficients_C_dict=coef_C,
        coefficients_R_dict=coef_R,
        test_start_year=2001,
    )
    if result.empty:
        pytest.skip("No individual signals computed")

    # Spot-check first row
    row = result.iloc[0]
    perm, date = row["permno"], row["date"]
    year = pd.Timestamp(date).year
    coef_year = year - 1

    if coef_year not in coef_C:
        pytest.skip("Coefficient year not found")

    # Find X row
    test_meta = meta_df[meta_df["date"].dt.year >= 2001]
    m_idx = test_meta[
        (test_meta["permno"] == perm) & (test_meta["date"] == date)
    ].index
    if len(m_idx) == 0:
        pytest.skip("Row not found in meta_df")

    loc = list(test_meta.index).index(m_idx[0])
    X_C_test = X[:, :21][meta_df["date"].dt.year >= 2001]

    expected = float(X_C_test[loc] @ coef_C[coef_year]["beta"])
    np.testing.assert_almost_equal(
        float(row["DRI_C_indiv"]), expected, decimal=5
    )


def test_individual_vs_joint_signals_differ(
    individual_coef_dicts, synthetic_coef_dict, synthetic_feature_data
):
    """Individual model signals should generally differ from joint model signals."""
    X, meta_df = synthetic_feature_data
    coef_C, coef_R = individual_coef_dicts

    joint = compute_masked_signals(
        X=X, meta_df=meta_df, coefficients_dict=synthetic_coef_dict,
        test_start_year=2001
    )
    indiv = compute_individual_signals(
        X_C=X[:, :21], X_R=X[:, 21:],
        meta_df=meta_df,
        coefficients_C_dict=coef_C,
        coefficients_R_dict=coef_R,
        test_start_year=2001,
    )

    if joint.empty or indiv.empty:
        pytest.skip("Empty signal DataFrames")

    # They should differ since coefficients are different
    common = pd.merge(
        joint[["permno", "date", "DRI_C"]],
        indiv[["permno", "date", "DRI_C_indiv"]],
        on=["permno", "date"]
    )
    if len(common) > 0:
        diff = (common["DRI_C"] - common["DRI_C_indiv"]).abs().mean()
        assert diff > 0, "Joint and individual signals should differ (different coefficients)"
