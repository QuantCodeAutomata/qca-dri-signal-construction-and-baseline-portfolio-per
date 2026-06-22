"""
Experiment 2: Component Decomposition — Chronological vs. Rank-Ordered Signals.

Decomposes the DRI signal into chronological and rank-ordered components using:
  Method 1 (Joint Estimation + Masking)
  Method 2 (Individual Estimation)

Conducts bilateral spanning tests to establish chronological dominance.
Also evaluates the interaction-augmented 462-feature model.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_ff6_factors
from src.elastic_net_model import run_expanding_window_estimation
from src.signal_computation import compute_masked_signals, compute_individual_signals
from src.feature_engineering import build_augmented_feature_matrix
from src.portfolio_construction import build_decile_portfolios
from src.performance_metrics import compute_newey_west_tstat, compute_sharpe_ratio
from src.component_decomposition import (
    run_individual_elastic_net,
    build_spanning_test_matrix,
    build_component_performance_table,
    compute_augmented_dri_signal,
    run_interaction_augmented_model,
)

CFG = {
    "start_date": "1985-01-01",
    "end_date": "2005-12-31",
    "seed": 42,
    "train_end_year": 1994,
    "test_start_year": 1995,
    "eta": 0.5,
    "n_lambdas": 100,
    "n_splits": 5,
    "n_repeats": 5,
    "random_state": 42,
    "maxlags": 12,
    "run_augmented": False,  # 462-feature model; set True to enable (slow)
    "n_stocks": 200,
    "results_dir": os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    ),
}


def run_exp2(exp1_results: dict, cfg: dict = CFG, verbose: bool = True) -> dict:
    """
    Run Experiment 2 using the feature matrix and coefficients from Experiment 1.

    Args:
        exp1_results: Return dict from exp1.run_exp1().
        cfg: Configuration dict.
        verbose: Print progress.

    Returns:
        Dict with all Exp 2 result DataFrames.
    """
    os.makedirs(cfg["results_dir"], exist_ok=True)
    t0 = time.time()

    X = exp1_results["X"]
    y = exp1_results["y"]
    meta_df = exp1_results["meta_df"]
    coefficients_dict = exp1_results["coefficients_dict"]
    eligible_df = exp1_results["eligible_df"]
    ff6_factors = exp1_results["ff6_factors"]

    if verbose:
        print("\n=== EXP 2: Component Decomposition ===")

    # ─────────────────────────────────────── #
    # Method 1: Joint Estimation with Masking
    # ─────────────────────────────────────── #
    if verbose:
        print("\n--- Method 1: Joint Estimation + Masking ---")

    signals_joint = compute_masked_signals(
        X=X,
        meta_df=meta_df,
        coefficients_dict=coefficients_dict,
        test_start_year=cfg["test_start_year"],
    )

    test_eligible = eligible_df[
        eligible_df["date"].dt.year >= cfg["test_start_year"]
    ].copy()

    # Full DRI portfolios
    _, _, hl_vw_joint, hl_ew_joint = build_decile_portfolios(
        signal_df=signals_joint.rename(columns={"DRI": "DRI"})[["permno", "date", "DRI"]],
        eligible_df=test_eligible,
        signal_col="DRI",
    )

    # Chronological-only portfolios (Method 1)
    _, _, hl_vw_C1, hl_ew_C1 = build_decile_portfolios(
        signal_df=signals_joint[["permno", "date", "DRI_C"]].rename(columns={"DRI_C": "signal"}),
        eligible_df=test_eligible,
        signal_col="signal",
    )

    # Rank-only portfolios (Method 1)
    _, _, hl_vw_R1, hl_ew_R1 = build_decile_portfolios(
        signal_df=signals_joint[["permno", "date", "DRI_R"]].rename(columns={"DRI_R": "signal"}),
        eligible_df=test_eligible,
        signal_col="signal",
    )

    if verbose:
        print(f"  VW H-L Combined: {hl_vw_joint.mean()*100:.4f}%")
        print(f"  VW H-L Chron (M1): {hl_vw_C1.mean()*100:.4f}%")
        print(f"  VW H-L Rank  (M1): {hl_vw_R1.mean()*100:.4f}%")

    # ─────────────────────────────────────── #
    # Method 2: Individual Estimation
    # ─────────────────────────────────────── #
    if verbose:
        print("\n--- Method 2: Individual Estimation ---")

    X_C = X[:, :21]  # Chronological features only
    X_R = X[:, 21:]  # Rank features only

    coef_C = run_individual_elastic_net(
        X_component=X_C,
        y_all=y,
        meta_df=meta_df,
        train_end_year=cfg["train_end_year"],
        start_year=int(cfg["start_date"][:4]),
        eta=cfg["eta"],
        n_lambdas=cfg["n_lambdas"],
        n_splits=cfg["n_splits"],
        n_repeats=cfg["n_repeats"],
        random_state=cfg["random_state"],
        verbose=verbose,
        label="Chronological",
    )

    coef_R = run_individual_elastic_net(
        X_component=X_R,
        y_all=y,
        meta_df=meta_df,
        train_end_year=cfg["train_end_year"],
        start_year=int(cfg["start_date"][:4]),
        eta=cfg["eta"],
        n_lambdas=cfg["n_lambdas"],
        n_splits=cfg["n_splits"],
        n_repeats=cfg["n_repeats"],
        random_state=cfg["random_state"],
        verbose=verbose,
        label="Rank",
    )

    signals_indiv = compute_individual_signals(
        X_C=X_C,
        X_R=X_R,
        meta_df=meta_df,
        coefficients_C_dict=coef_C,
        coefficients_R_dict=coef_R,
        test_start_year=cfg["test_start_year"],
    )

    _, _, hl_vw_C2, hl_ew_C2 = build_decile_portfolios(
        signal_df=signals_indiv[["permno", "date", "DRI_C_indiv"]].rename(
            columns={"DRI_C_indiv": "signal"}
        ),
        eligible_df=test_eligible,
        signal_col="signal",
    )

    _, _, hl_vw_R2, hl_ew_R2 = build_decile_portfolios(
        signal_df=signals_indiv[["permno", "date", "DRI_R_indiv"]].rename(
            columns={"DRI_R_indiv": "signal"}
        ),
        eligible_df=test_eligible,
        signal_col="signal",
    )

    if verbose:
        print(f"  VW H-L Chron (M2): {hl_vw_C2.mean()*100:.4f}%")
        print(f"  VW H-L Rank  (M2): {hl_vw_R2.mean()*100:.4f}%")

    # ─────────────────────────────────────── #
    # Interaction Augmentation (Optional)
    # ─────────────────────────────────────── #
    hl_vw_aug = pd.Series(dtype=float)
    hl_ew_aug = pd.Series(dtype=float)

    if cfg.get("run_augmented", False):
        if verbose:
            print("\n--- Interaction Augmentation (462 features) ---")
        try:
            meta_aug, X_aug, y_aug = build_augmented_feature_matrix(
                eligible_universe=eligible_df,
                daily_df=exp1_results.get("daily_df", _load_daily(cfg)),
            )
            coef_aug = run_interaction_augmented_model(
                X_aug=X_aug,
                y_all=y_aug,
                meta_df=meta_aug,
                train_end_year=cfg["train_end_year"],
                start_year=int(cfg["start_date"][:4]),
                verbose=verbose,
            )
            signals_aug = compute_augmented_dri_signal(
                X_aug=X_aug,
                meta_df=meta_aug,
                coefficients_aug_dict=coef_aug,
                test_start_year=cfg["test_start_year"],
            )
            _, _, hl_vw_aug, hl_ew_aug = build_decile_portfolios(
                signal_df=signals_aug[["permno", "date", "DRI_aug"]].rename(
                    columns={"DRI_aug": "signal"}
                ),
                eligible_df=test_eligible,
                signal_col="signal",
            )
            if verbose:
                print(f"  VW H-L Aug: {hl_vw_aug.mean()*100:.4f}%")
        except Exception as e:
            if verbose:
                print(f"  Augmented model skipped ({e})")

    # ─────────────────────────────────────── #
    # Component Performance Table
    # ─────────────────────────────────────── #
    signals_for_table = {
        "Combined (Joint)": (hl_vw_joint, hl_ew_joint),
        "Chron-only (M1)": (hl_vw_C1, hl_ew_C1),
        "Rank-only (M1)": (hl_vw_R1, hl_ew_R1),
        "Chron-only (M2)": (hl_vw_C2, hl_ew_C2),
        "Rank-only (M2)": (hl_vw_R2, hl_ew_R2),
    }
    if not hl_vw_aug.empty:
        signals_for_table["Interaction-aug"] = (hl_vw_aug, hl_ew_aug)

    perf_table = build_component_performance_table(
        signals_dict=signals_for_table,
        ff6_factors=ff6_factors,
        maxlags=cfg["maxlags"],
    )

    if verbose:
        print("\n--- Component Performance Table ---")
        print(perf_table.to_string())

    # ─────────────────────────────────────── #
    # Spanning Tests
    # ─────────────────────────────────────── #
    spanning_signals = {
        "Combined": hl_vw_joint,
        "Chron (M1)": hl_vw_C1,
        "Rank (M1)": hl_vw_R1,
        "Chron (M2)": hl_vw_C2,
        "Rank (M2)": hl_vw_R2,
    }
    spanning_df = build_spanning_test_matrix(
        signal_returns_dict=spanning_signals,
        maxlags=cfg["maxlags"],
    )

    if verbose:
        print("\n--- Spanning Test Matrix (alpha of row | column) ---")
        alpha_pivot = spanning_df["alpha"].unstack("benchmark")
        print(alpha_pivot.to_string())

    # ─────────────────────────────────────── #
    # Save Results
    # ─────────────────────────────────────── #
    _save_exp2_results(
        perf_table=perf_table,
        spanning_df=spanning_df,
        signals_for_table=signals_for_table,
        cfg=cfg,
        elapsed=time.time() - t0,
    )

    return {
        "perf_table": perf_table,
        "spanning_df": spanning_df,
        "hl_vw_joint": hl_vw_joint,
        "hl_ew_joint": hl_ew_joint,
        "hl_vw_C1": hl_vw_C1,
        "hl_ew_C1": hl_ew_C1,
        "hl_vw_R1": hl_vw_R1,
        "hl_ew_R1": hl_ew_R1,
        "hl_vw_C2": hl_vw_C2,
        "hl_ew_C2": hl_ew_C2,
        "hl_vw_R2": hl_vw_R2,
        "hl_ew_R2": hl_ew_R2,
        "hl_vw_aug": hl_vw_aug,
        "hl_ew_aug": hl_ew_aug,
    }


def _load_daily(cfg: dict):
    """Fallback: reload daily data if not cached in exp1_results."""
    from data.data_loader import load_crsp_daily
    return load_crsp_daily(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        n_stocks=cfg.get("n_stocks", 200),
        seed=cfg["seed"],
    )


def _save_exp2_results(
    perf_table: pd.DataFrame,
    spanning_df: pd.DataFrame,
    signals_for_table: dict,
    cfg: dict,
    elapsed: float,
) -> None:
    """Save Experiment 2 results."""
    results_dir = cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # CSVs
    perf_table.to_csv(os.path.join(results_dir, "exp2_component_performance.csv"))
    spanning_df.to_csv(os.path.join(results_dir, "exp2_spanning_tests.csv"))

    # Plot: cumulative H-L for all components
    _plot_component_hl_curves(signals_for_table, results_dir)

    # Plot: spanning test heatmap
    _plot_spanning_heatmap(spanning_df, results_dir)

    # Append to RESULTS.md
    md_path = os.path.join(results_dir, "RESULTS.md")
    mode = "a" if os.path.isfile(md_path) else "w"
    with open(md_path, mode) as f:
        f.write("---\n\n")
        f.write("## Experiment 2: Component Decomposition\n\n")
        f.write(
            f"**Runtime:** {elapsed:.1f} seconds\n\n"
            "### Component Performance Table\n\n"
        )
        f.write(perf_table.to_markdown() + "\n\n")
        f.write("### Spanning Test Alpha Matrix\n\n")
        if not spanning_df.empty:
            try:
                alpha_pivot = spanning_df["alpha"].unstack("benchmark") * 100
                f.write(alpha_pivot.round(4).to_markdown() + "\n\n")
            except Exception:
                f.write(spanning_df.to_markdown() + "\n\n")
        f.write("### Saved Artifacts\n\n")
        f.write("- `exp2_component_performance.csv`\n")
        f.write("- `exp2_spanning_tests.csv`\n")
        f.write("- `exp2_component_hl_curves.png`\n")
        f.write("- `exp2_spanning_heatmap.png`\n\n")

    print(f"\nExp 2 results saved to: {results_dir}")


def _plot_component_hl_curves(
    signals_for_table: dict,
    results_dir: str,
) -> None:
    """Cumulative H-L return curves for all signal components."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("tab10", len(signals_for_table))

    for (name, (hl_vw, _)), color in zip(signals_for_table.items(), colors):
        if hl_vw.empty:
            continue
        cum = (1 + hl_vw).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=color, linewidth=1.3)

    ax.set_title("Cumulative H-L Returns by DRI Component (VW)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp2_component_hl_curves.png"), dpi=150)
    plt.close()


def _plot_spanning_heatmap(
    spanning_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Heatmap of spanning test alpha t-statistics."""
    if spanning_df.empty:
        return
    try:
        tstat_pivot = spanning_df["alpha_tstat"].unstack("benchmark")
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = max(abs(tstat_pivot.values[~np.isnan(tstat_pivot.values)]).max(), 1)
    sns.heatmap(
        tstat_pivot.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Spanning Test Alpha t-Statistics\n(row = tested, col = benchmark)", fontsize=12)
    ax.set_xlabel("Benchmark Signal")
    ax.set_ylabel("Tested Signal")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp2_spanning_heatmap.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    from exp.exp1_dri_signal import run_exp1
    print("Running Experiment 1 first (required for Exp 2)...")
    exp1_res = run_exp1(verbose=True)
    exp2_res = run_exp2(exp1_results=exp1_res, verbose=True)
    print("\nExperiment 2 complete.")
