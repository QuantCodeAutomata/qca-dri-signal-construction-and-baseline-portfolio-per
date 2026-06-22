"""
Exp 2: Component Decomposition — Chronological vs. Rank-Ordered Signals.

Implements both joint estimation masking (Method 1) and individual estimation
(Method 2) to isolate and compare the chronological and rank-ordered components
of the DRI signal. Also implements spanning tests and interaction augmentation.

Custom — Context7 found no library equivalent for this DRI decomposition methodology.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from src.elastic_net_model import run_expanding_window_estimation
from src.signal_computation import (
    compute_masked_signals,
    compute_individual_signals,
)
from src.portfolio_construction import build_decile_portfolios, pivot_portfolio_returns
from src.performance_metrics import (
    compute_newey_west_tstat,
    compute_sharpe_ratio,
    compute_ff6_alpha,
    compute_spanning_test,
)


# --------------------------------------------------------------------------- #
# Method 2 — Individual Estimation
# --------------------------------------------------------------------------- #

def run_individual_elastic_net(
    X_component: np.ndarray,
    y_all: np.ndarray,
    meta_df: pd.DataFrame,
    train_end_year: int = 1994,
    start_year: int = 1985,
    eta: float = 0.5,
    n_lambdas: int = 100,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    label: str = "component",
) -> Dict[int, Dict]:
    """
    Run expanding window elastic-net using only one component (C or R).

    This is the same procedure as the full 42-feature model but applied to
    either the 21 chronological features or the 21 rank features independently.

    Args:
        X_component: Feature matrix of shape (N, 21) — either X[:,0:21] or X[:,21:42].
        y_all: Response vector (N,).
        meta_df: Metadata with 'date' column.
        train_end_year: Last training year.
        start_year: First year in the sample.
        eta: Elastic-net l1_ratio.
        n_lambdas: Lambda grid size.
        n_splits: CV folds.
        n_repeats: CV repetitions.
        random_state: Seed for reproducibility.
        verbose: Print progress.
        label: Component label for print statements.

    Returns:
        Annual coefficient dict (same structure as full model but with
        21-dim beta instead of 42-dim).
    """
    if verbose:
        print(f"\nRunning individual elastic-net for {label} features...")

    coef_dict = run_expanding_window_estimation(
        X_all=X_component,
        y_all=y_all,
        meta_df=meta_df,
        train_end_year=train_end_year,
        start_year=start_year,
        eta=eta,
        n_lambdas=n_lambdas,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        verbose=verbose,
    )
    return coef_dict


# --------------------------------------------------------------------------- #
# Spanning Test Matrix
# --------------------------------------------------------------------------- #

def build_spanning_test_matrix(
    signal_returns_dict: Dict[str, pd.Series],
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    Build bilateral spanning test matrix for all signal pairs.

    Entry (row=A, col=B) = alpha when regressing H-L(A) on H-L(B).
    A small, insignificant alpha means B spans A.

    Args:
        signal_returns_dict: Dict mapping signal name → H-L VW return Series.
        maxlags: Lags for Newey-West SE.

    Returns:
        Multi-index DataFrame with (alpha, t-stat, beta, R²) for each pair.
    """
    signals = list(signal_returns_dict.keys())
    rows = []

    for name_A in signals:
        for name_B in signals:
            if name_A == name_B:
                rows.append({
                    "tested": name_A,
                    "benchmark": name_B,
                    "alpha": np.nan,
                    "alpha_tstat": np.nan,
                    "beta": np.nan,
                    "r_squared": np.nan,
                })
                continue

            result = compute_spanning_test(
                signal_returns_dict[name_A],
                signal_returns_dict[name_B],
                maxlags=maxlags,
            )
            rows.append({
                "tested": name_A,
                "benchmark": name_B,
                **result,
            })

    df = pd.DataFrame(rows)
    return df.set_index(["tested", "benchmark"])


# --------------------------------------------------------------------------- #
# Interaction Augmentation (exp_2 INT section)
# --------------------------------------------------------------------------- #

def run_interaction_augmented_model(
    X_aug: np.ndarray,
    y_all: np.ndarray,
    meta_df: pd.DataFrame,
    train_end_year: int = 1994,
    start_year: int = 1985,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """
    Run the interaction-augmented model with 462-dim features (exp_2 INT).

    Feature vector: [R^C (21), Z^I (441)] where Z^I contains the 441
    interaction terms z^(d,k) = r_tilde_d * 1{rank(r_tilde_d) = k}.

    The DRI^aug signal uses only the chronological and interaction components:
    DRI^aug = R^C' * beta^C_aug + Z^I' * beta^I_aug

    Args:
        X_aug: Augmented feature matrix (N, 462).
        y_all: Response vector (N,).
        meta_df: Metadata with 'date' column.
        train_end_year: Last training year.
        start_year: First year of sample.
        verbose: Print progress.

    Returns:
        Annual coefficient dict.
    """
    assert X_aug.shape[1] == 462, f"Expected 462 features, got {X_aug.shape[1]}"

    if verbose:
        print("\nRunning interaction-augmented model (462 features)...")

    coef_dict = run_expanding_window_estimation(
        X_all=X_aug,
        y_all=y_all,
        meta_df=meta_df,
        train_end_year=train_end_year,
        start_year=start_year,
        verbose=verbose,
    )
    return coef_dict


def compute_augmented_dri_signal(
    X_aug: np.ndarray,
    meta_df: pd.DataFrame,
    coefficients_aug_dict: Dict,
    test_start_year: int = 1995,
) -> pd.DataFrame:
    """
    Compute DRI^aug from 462-dim feature model.

    DRI^aug_{i,t} = X_aug[i, 0:21] @ beta_C_aug + X_aug[i, 21:462] @ beta_I_aug

    Note: beta stored as full 462-dim vector from run_expanding_window_estimation.

    Args:
        X_aug: Augmented feature matrix (N, 462).
        meta_df: Metadata with permno, date.
        coefficients_aug_dict: From run_interaction_augmented_model.
        test_start_year: First test year.

    Returns:
        DataFrame with columns: permno, date, DRI_aug
    """
    meta = meta_df.copy()
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    meta = meta.reset_index(drop=True)

    test_mask = meta["year"] >= test_start_year
    meta_test = meta[test_mask].copy()
    X_test = X_aug[test_mask.values]

    signals = []
    test_idx_list = list(meta_test.index)

    for loc, i in enumerate(test_idx_list):
        row = meta_test.loc[i]
        signal_year = row["year"]
        coef_year = signal_year - 1

        if coef_year not in coefficients_aug_dict:
            available = [y for y in coefficients_aug_dict.keys() if y < signal_year]
            if not available:
                continue
            coef_year = max(available)

        coefs = coefficients_aug_dict[coef_year]
        beta_full = coefs["beta"]  # 462-dim

        dri_aug = float(X_test[loc] @ beta_full)

        signals.append({
            "permno": row["permno"],
            "date": row["date"],
            "DRI_aug": dri_aug,
        })

    return pd.DataFrame(signals)


# --------------------------------------------------------------------------- #
# Full Exp 2 Performance Table
# --------------------------------------------------------------------------- #

def build_component_performance_table(
    signals_dict: Dict[str, Tuple[pd.Series, pd.Series]],
    ff6_factors: pd.DataFrame,
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    Build Table 3-style performance summary for all signal variants.

    For each signal variant, reports VW and EW H-L:
    Mean, t-stat, SR, Alpha, Alpha t-stat.

    Args:
        signals_dict: Dict mapping signal name → (hl_vw, hl_ew) Series pair.
        ff6_factors: FF6 factor DataFrame.
        maxlags: Lags for Newey-West SE.

    Returns:
        Performance table DataFrame with signal variants as rows.
    """
    rows = []

    for name, (hl_vw, hl_ew) in signals_dict.items():
        vw_mean, vw_t, vw_std = compute_newey_west_tstat(hl_vw, maxlags)
        ew_mean, ew_t, ew_std = compute_newey_west_tstat(hl_ew, maxlags)

        vw_sr = compute_sharpe_ratio(hl_vw)
        ew_sr = compute_sharpe_ratio(hl_ew)

        vw_alpha, vw_alpha_t, _ = compute_ff6_alpha(hl_vw, ff6_factors, maxlags)
        ew_alpha, ew_alpha_t, _ = compute_ff6_alpha(hl_ew, ff6_factors, maxlags)

        rows.append({
            "Signal": name,
            "VW_Mean_%": round(vw_mean * 100, 4),
            "VW_tstat": round(vw_t, 2) if pd.notna(vw_t) else np.nan,
            "VW_SR": round(vw_sr, 2) if pd.notna(vw_sr) else np.nan,
            "VW_Alpha_%": round(vw_alpha * 100, 4) if pd.notna(vw_alpha) else np.nan,
            "VW_Alpha_t": round(vw_alpha_t, 2) if pd.notna(vw_alpha_t) else np.nan,
            "EW_Mean_%": round(ew_mean * 100, 4),
            "EW_tstat": round(ew_t, 2) if pd.notna(ew_t) else np.nan,
            "EW_SR": round(ew_sr, 2) if pd.notna(ew_sr) else np.nan,
            "EW_Alpha_%": round(ew_alpha * 100, 4) if pd.notna(ew_alpha) else np.nan,
            "EW_Alpha_t": round(ew_alpha_t, 2) if pd.notna(ew_alpha_t) else np.nan,
        })

    return pd.DataFrame(rows).set_index("Signal")
