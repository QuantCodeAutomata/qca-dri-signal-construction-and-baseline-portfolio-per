"""
Phase D: DRI Signal Computation.

Computes the out-of-sample DRI signal for each eligible stock-month
in the test period using annually estimated elastic-net coefficients.

DRI_{i,t} = R^C_{i,t}' * beta^C_Y + R^R_{i,t}' * beta^R_Y
           (for month t in year Y+1, using coefficients from year Y)

Custom — Context7 found no library equivalent (paper DRI signal formula).
"""
import numpy as np
import pandas as pd
from typing import Dict


def compute_dri_signal(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    coefficients_dict: Dict,
    test_start_year: int = 1995,
) -> pd.DataFrame:
    """
    Compute the DRI signal for all eligible stocks in the test period.

    For each stock i in each month t of year Y+1:
    DRI_{i,t} = X[i, 0:21] @ beta^C_Y + X[i, 21:42] @ beta^R_Y

    The intercept is NOT included in the signal — only the predictive
    component of the feature vector matters for cross-sectional ranking.

    All 12 months of year Y+1 use the SAME coefficient vector estimated
    with data through December Y (no intra-year updates).

    Custom — Context7 found no library equivalent (paper DRI signal formula).

    Args:
        X: Feature matrix, shape (N_total, p) where p=42 for baseline.
        meta_df: Metadata DataFrame with 'permno' and 'date' columns
            (same row order as X).
        coefficients_dict: Annual coefficient dict from
            run_expanding_window_estimation. Keys are training years Y,
            values contain 'beta_C' (21-dim) and 'beta_R' (21-dim).
        test_start_year: First year to compute DRI signal (uses coefficients
            from test_start_year - 1).

    Returns:
        DataFrame with columns: permno, date, DRI
    """
    meta = meta_df.copy()
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    meta = meta.reset_index(drop=True)

    # Only compute for test period rows
    test_mask = meta["year"] >= test_start_year
    meta_test = meta[test_mask].copy()
    X_test = X[test_mask.values]

    signals = []
    n_features = X.shape[1]
    half = n_features // 2

    for i, row in meta_test.iterrows():
        signal_year = row["year"]
        # Use coefficients from the preceding year
        coef_year = signal_year - 1

        if coef_year not in coefficients_dict:
            # Fall back to the most recent available year
            available_years = [y for y in coefficients_dict.keys() if y < signal_year]
            if not available_years:
                continue
            coef_year = max(available_years)

        coefs = coefficients_dict[coef_year]
        beta_C = coefs["beta_C"]  # 21-dim
        beta_R = coefs["beta_R"]  # 21-dim

        # DRI = R^C' * beta^C + R^R' * beta^R
        x_row = X_test[meta_test.index.get_loc(i)]
        dri = float(x_row[:half] @ beta_C + x_row[half:] @ beta_R)

        signals.append({
            "permno": row["permno"],
            "date": row["date"],
            "DRI": dri,
        })

    return pd.DataFrame(signals)


def compute_masked_signals(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    coefficients_dict: Dict,
    test_start_year: int = 1995,
) -> pd.DataFrame:
    """
    Compute the full DRI signal and its chronological/rank-only components.

    Method 1 (Joint Estimation with Masking):
    - DRI^C = X[:, 0:21] @ beta^C  (chronological component only)
    - DRI^R = X[:, 21:42] @ beta^R  (rank component only)
    - DRI   = DRI^C + DRI^R          (full signal)

    This uses the JOINTLY estimated 42-dim coefficient vector, with mask
    vectors m^C = [1]*21 + [0]*21 and m^R = [0]*21 + [1]*21 applied to
    isolate each component.

    Custom — Context7 found no library equivalent (paper methodology).

    Args:
        X: Feature matrix, shape (N_total, 42).
        meta_df: Metadata with permno, date.
        coefficients_dict: Annual coefficients from baseline 42-feature model.
        test_start_year: First year of test period.

    Returns:
        DataFrame with columns: permno, date, DRI, DRI_C, DRI_R
    """
    meta = meta_df.copy()
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    meta = meta.reset_index(drop=True)

    test_mask = meta["year"] >= test_start_year
    meta_test = meta[test_mask].copy()
    X_test = X[test_mask.values]

    # Assign the coefficient year (preceding calendar year) for each row
    meta_test = meta_test.copy()
    meta_test["coef_year"] = meta_test["year"] - 1

    sorted_coef_years = sorted(coefficients_dict.keys())

    def _resolve_coef_year(signal_yr: int) -> int:
        """Return the most recent available coefficient year < signal_yr."""
        candidates = [y for y in sorted_coef_years if y < signal_yr]
        return max(candidates) if candidates else -1

    # Resolve missing coef years (fall-back to latest available)
    coef_year_avail = set(coefficients_dict.keys())
    meta_test["coef_year"] = meta_test["coef_year"].apply(
        lambda cy: cy if cy in coef_year_avail else _resolve_coef_year(cy + 1)
    )

    # Drop rows where no coefficient year is available
    meta_test = meta_test[meta_test["coef_year"] >= 0].copy()
    if meta_test.empty:
        return pd.DataFrame(columns=["permno", "date", "DRI", "DRI_C", "DRI_R"])

    # Re-align X_test with potentially filtered meta_test
    X_test_df = pd.DataFrame(X_test, index=test_mask[test_mask].index)

    result_parts = []
    for coef_yr, grp in meta_test.groupby("coef_year"):
        coefs = coefficients_dict[coef_yr]
        beta_C = coefs["beta_C"].astype(np.float64)  # (21,)
        beta_R = coefs["beta_R"].astype(np.float64)  # (21,)

        # Subset feature rows by original index
        X_grp = X_test_df.loc[grp.index].values  # (n_grp, 42)
        dri_c = X_grp[:, :21] @ beta_C            # (n_grp,)
        dri_r = X_grp[:, 21:42] @ beta_R          # (n_grp,)

        part = pd.DataFrame({
            "permno": grp["permno"].values,
            "date": grp["date"].values,
            "DRI": dri_c + dri_r,
            "DRI_C": dri_c,
            "DRI_R": dri_r,
        })
        result_parts.append(part)

    return pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame(
        columns=["permno", "date", "DRI", "DRI_C", "DRI_R"]
    )


def compute_individual_signals(
    X_C: np.ndarray,
    X_R: np.ndarray,
    meta_df: pd.DataFrame,
    coefficients_C_dict: Dict,
    coefficients_R_dict: Dict,
    test_start_year: int = 1995,
) -> pd.DataFrame:
    """
    Compute DRI component signals from individually estimated models (exp_2 Method 2).

    Uses separate elastic-net models: one trained only on chronological
    features (X_C = X[:, 0:21]) and one trained only on rank features
    (X_R = X[:, 21:42]), each with independent lambda selection.

    Args:
        X_C: Chronological features, shape (N_total, 21) = X_all[:, 0:21].
        X_R: Rank features, shape (N_total, 21) = X_all[:, 21:42].
        meta_df: Metadata DataFrame with permno, date.
        coefficients_C_dict: Annual coefficients from chronological-only model.
        coefficients_R_dict: Annual coefficients from rank-only model.
        test_start_year: First year of test period.

    Returns:
        DataFrame with columns: permno, date, DRI_C_indiv, DRI_R_indiv
    """
    meta = meta_df.copy()
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    meta = meta.reset_index(drop=True)

    test_mask = meta["year"] >= test_start_year
    meta_test = meta[test_mask].copy()
    X_C_test = X_C[test_mask.values]
    X_R_test = X_R[test_mask.values]

    signals = []
    test_idx_list = list(meta_test.index)

    for loc, i in enumerate(test_idx_list):
        row = meta_test.loc[i]
        signal_year = row["year"]
        coef_year = signal_year - 1

        # Chronological individual model
        coef_year_C = coef_year
        if coef_year_C not in coefficients_C_dict:
            available = [y for y in coefficients_C_dict.keys() if y < signal_year]
            if not available:
                continue
            coef_year_C = max(available)

        # Rank individual model
        coef_year_R = coef_year
        if coef_year_R not in coefficients_R_dict:
            available = [y for y in coefficients_R_dict.keys() if y < signal_year]
            if not available:
                continue
            coef_year_R = max(available)

        beta_C_indiv = coefficients_C_dict[coef_year_C]["beta"]
        beta_R_indiv = coefficients_R_dict[coef_year_R]["beta"]

        dri_c = float(X_C_test[loc] @ beta_C_indiv)
        dri_r = float(X_R_test[loc] @ beta_R_indiv)

        signals.append({
            "permno": row["permno"],
            "date": row["date"],
            "DRI_C_indiv": dri_c,
            "DRI_R_indiv": dri_r,
        })

    return pd.DataFrame(signals)
