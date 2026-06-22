"""
Experiment 1: DRI Signal Construction and Baseline Portfolio Performance.

Constructs the 42-dim Daily Return Information (DRI) feature matrix,
estimates elastic-net models in an expanding window, computes the out-of-sample
DRI signal, and evaluates decile portfolio performance.

Pipeline:
    Phase A — Data Preparation
    Phase B — Feature Matrix Construction
    Phase C — Expanding Window Elastic-Net
    Phase D — DRI Signal Computation
    Phase E — Portfolio Formation and Performance Evaluation
"""
import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import (
    load_crsp_daily,
    load_crsp_monthly,
    load_ff6_factors,
    load_risk_free_rate,
)
from data.synthetic_data import get_nyse_size_breakpoints
from src.data_prep import handle_delisting_returns, compute_excess_returns
from src.feature_engineering import build_feature_matrix
from src.elastic_net_model import (
    run_expanding_window_estimation,
    quality_check_coefficients,
)
from src.signal_computation import compute_masked_signals
from src.portfolio_construction import (
    build_decile_portfolios,
    pivot_portfolio_returns,
)
from src.performance_metrics import (
    compute_full_decile_metrics,
    verify_monotonic_pattern,
    compute_newey_west_tstat,
    compute_sharpe_ratio,
    compute_ff6_alpha,
)

# ──────────────────────────────────────────────────────────────────────────── #
# Configuration
# ──────────────────────────────────────────────────────────────────────────── #
CFG = {
    "start_date": "1985-01-01",
    "end_date": "2005-12-31",
    "n_stocks": 200,
    "seed": 42,
    "train_end_year": 1994,
    "test_start_year": 1995,
    "eta": 0.5,
    "n_lambdas": 100,
    "n_splits": 5,
    "n_repeats": 5,
    "random_state": 42,
    "maxlags": 12,
    "results_dir": os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    ),
}


# ──────────────────────────────────────────────────────────────────────────── #
# Helper: apply universe filters inline
# ──────────────────────────────────────────────────────────────────────────── #

def _build_eligible_universe(
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    rf_series: pd.Series,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Inline universe construction with all six filters plus 21-day daily check.

    Returns eligible stock-month DataFrame with 'next_ret_excess' column.
    """
    df = monthly_df.copy()

    # (a)+(b) share code and exchange filters
    df = df[df["shrcd"].isin([10, 11]) & df["exchcd"].isin([1, 2, 3])]

    # (c) price filter
    df = df[df["prc"].abs() >= 5.0]

    # (d) non-missing ME
    df = df[df["me"].notna() & (df["me"] > 0)]

    # (e) NYSE size breakpoint
    nyse_bp = get_nyse_size_breakpoints(monthly_df, percentile=1.0)
    df = df.merge(
        nyse_bp.reset_index().rename(columns={"nyse_breakpoint": "bp"}),
        on="date", how="left"
    )
    df = df[df["me"] >= df["bp"]].drop(columns=["bp"])

    # Excess returns
    df = compute_excess_returns(df, rf_series)

    # (f) non-missing next-month excess return
    df = df.sort_values(["permno", "date"])
    df["next_ret_excess"] = df.groupby("permno")["ret_excess"].shift(-1)
    df = df[df["next_ret_excess"].notna()].reset_index(drop=True)

    # Build per-permno daily lookup for 21-day window check
    daily_by_perm = {
        perm: grp.sort_values("date").reset_index(drop=True)
        for perm, grp in daily_df.groupby("permno")
    }

    # Filter: exactly 21 non-missing daily returns in last 21 trading days
    valid_rows = []
    months = sorted(df["date"].unique())
    for month_end in months:
        month_stocks = df[df["date"] == month_end]
        # Last 21 trading days up to this month-end
        all_dates_up_to = pd.to_datetime(
            daily_df[daily_df["date"] <= month_end]["date"].unique()
        )
        if len(all_dates_up_to) < 21:
            continue
        last_21_dates = set(sorted(all_dates_up_to)[-21:])

        for _, row in month_stocks.iterrows():
            perm = row["permno"]
            if perm not in daily_by_perm:
                continue
            stock_daily = daily_by_perm[perm]
            mask = stock_daily["date"].isin(last_21_dates)
            window = stock_daily[mask]
            if len(window) < 21 or window["ret"].isna().any():
                continue
            valid_rows.append(row.name)

    df = df.loc[valid_rows].reset_index(drop=True)

    if verbose:
        n_months = df["date"].nunique()
        avg_cs = len(df) / max(n_months, 1)
        print(
            f"Eligible universe: {len(df):,} stock-months, "
            f"{n_months} months, avg {avg_cs:.0f} stocks/month"
        )
    return df


# ──────────────────────────────────────────────────────────────────────────── #
# Main Experiment Runner
# ──────────────────────────────────────────────────────────────────────────── #

def run_exp1(cfg: dict = CFG, verbose: bool = True) -> dict:
    """
    Run Experiment 1 end-to-end and return result metrics.

    Args:
        cfg: Configuration dict.
        verbose: Print progress.

    Returns:
        Dict with result DataFrames and key metrics.
    """
    os.makedirs(cfg["results_dir"], exist_ok=True)
    t0 = time.time()

    # ──────────────────────────────────────── #
    # Phase A: Data Loading
    # ──────────────────────────────────────── #
    if verbose:
        print("\n=== PHASE A: Data Loading ===")

    daily_df = load_crsp_daily(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        n_stocks=cfg["n_stocks"],
        seed=cfg["seed"],
    )
    monthly_df = load_crsp_monthly(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        n_stocks=cfg["n_stocks"],
        seed=cfg["seed"],
    )
    ff6_factors = load_ff6_factors(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        seed=cfg["seed"],
    )
    rf_series = load_risk_free_rate(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        seed=cfg["seed"],
    )

    if verbose:
        print(
            f"  Daily: {len(daily_df):,} rows  |  "
            f"Monthly: {len(monthly_df):,} rows  |  "
            f"Months: {monthly_df['date'].nunique()}"
        )

    # Delisting adjustment
    monthly_df = handle_delisting_returns(monthly_df)

    # Build eligible universe
    eligible_df = _build_eligible_universe(
        monthly_df, daily_df, rf_series, verbose=verbose
    )

    if len(eligible_df) < 100:
        raise RuntimeError("Insufficient eligible stock-months after filtering.")

    # ──────────────────────────────────────── #
    # Phase B: Feature Matrix Construction
    # ──────────────────────────────────────── #
    if verbose:
        print("\n=== PHASE B: Feature Matrix Construction ===")

    meta_df, X, y = build_feature_matrix(
        eligible_universe=eligible_df,
        daily_df=daily_df,
        is_training=True,
    )

    if verbose:
        print(
            f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features  |  "
            f"Response: {y.shape[0]:,} obs"
        )

    assert X.shape[1] == 42, f"Expected 42 features, got {X.shape[1]}"

    # ──────────────────────────────────────── #
    # Phase C: Expanding Window Elastic-Net
    # ──────────────────────────────────────── #
    if verbose:
        print("\n=== PHASE C: Expanding Window Elastic-Net Estimation ===")

    coefficients_dict = run_expanding_window_estimation(
        X_all=X,
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
    )

    if verbose:
        print(f"  Estimated {len(coefficients_dict)} annual coefficient sets.")

    # Quality checks
    qc = quality_check_coefficients(coefficients_dict)
    if verbose and "avg_beta_C_d1" in qc:
        print(f"  QC: avg beta_C[d=1] = {qc['avg_beta_C_d1']:.4f} "
              f"(negative: {qc['beta_C_d1_negative']})")
        print(f"  QC: {qc['pct_years_C_dominates_R']*100:.0f}% years chron dominates rank")
    elif verbose:
        print(f"  QC: {qc.get('error', 'no data')}")

    # ──────────────────────────────────────── #
    # Phase D: DRI Signal Computation
    # ──────────────────────────────────────── #
    if verbose:
        print("\n=== PHASE D: DRI Signal Computation ===")

    signal_df = compute_masked_signals(
        X=X,
        meta_df=meta_df,
        coefficients_dict=coefficients_dict,
        test_start_year=cfg["test_start_year"],
    )

    if verbose:
        n_test_months = signal_df["date"].nunique()
        avg_cs = len(signal_df) / max(n_test_months, 1)
        print(
            f"  DRI signal: {len(signal_df):,} stock-months, "
            f"{n_test_months} months, avg {avg_cs:.0f} stocks/month"
        )

    if signal_df.empty:
        raise RuntimeError("Empty DRI signal DataFrame. Check test_start_year config.")

    # ──────────────────────────────────────── #
    # Phase E: Portfolio Formation & Performance
    # ──────────────────────────────────────── #
    if verbose:
        print("\n=== PHASE E: Portfolio Formation and Performance ===")

    # Align eligible_df to test period only (for portfolio returns)
    test_eligible = eligible_df[
        eligible_df["date"].dt.year >= cfg["test_start_year"]
    ].copy()

    vw_returns, ew_returns, hl_vw, hl_ew = build_decile_portfolios(
        signal_df=signal_df,
        eligible_df=test_eligible,
        signal_col="DRI",
    )

    if verbose:
        print(f"  VW H-L: {len(hl_vw)} months, mean={hl_vw.mean()*100:.3f}%")
        print(f"  EW H-L: {len(hl_ew)} months, mean={hl_ew.mean()*100:.3f}%")

    # Wide format
    vw_wide = pivot_portfolio_returns(vw_returns, "ret_vw")
    ew_wide = pivot_portfolio_returns(ew_returns, "ret_ew")

    # Performance metrics table
    metrics_df = compute_full_decile_metrics(
        vw_returns_wide=vw_wide,
        ew_returns_wide=ew_wide,
        hl_vw=hl_vw,
        hl_ew=hl_ew,
        ff6_factors=ff6_factors,
        maxlags=cfg["maxlags"],
    )

    if verbose:
        print("\n--- Performance Summary (VW H-L) ---")
        hl_row = metrics_df.loc["H-L"] if "H-L" in metrics_df.index else None
        if hl_row is not None:
            print(f"  VW H-L mean:  {hl_row['VW_Mean_%']:.4f}%")
            print(f"  VW H-L t:     {hl_row['VW_tstat']:.2f}")
            print(f"  VW H-L SR:    {hl_row['VW_SR']:.2f}")
            print(f"  VW H-L alpha: {hl_row['VW_Alpha_%']:.4f}%  (t={hl_row['VW_Alpha_t']:.2f})")
            print(f"  EW H-L mean:  {hl_row['EW_Mean_%']:.4f}%")

    # Monotonicity check
    decile_vw_means = vw_wide.mean()
    is_monotone = verify_monotonic_pattern(decile_vw_means)
    if verbose:
        print(f"\n  Monotonic pattern: {is_monotone}")

    # ──────────────────────────────────────── #
    # Save Results
    # ──────────────────────────────────────── #
    _save_exp1_results(
        metrics_df=metrics_df,
        vw_wide=vw_wide,
        ew_wide=ew_wide,
        hl_vw=hl_vw,
        hl_ew=hl_ew,
        coefficients_dict=coefficients_dict,
        qc=qc,
        is_monotone=is_monotone,
        cfg=cfg,
        elapsed=time.time() - t0,
    )

    return {
        "metrics_df": metrics_df,
        "vw_wide": vw_wide,
        "ew_wide": ew_wide,
        "hl_vw": hl_vw,
        "hl_ew": hl_ew,
        "coefficients_dict": coefficients_dict,
        "signal_df": signal_df,
        "eligible_df": eligible_df,
        "X": X,
        "y": y,
        "meta_df": meta_df,
        "ff6_factors": ff6_factors,
        "qc": qc,
        "is_monotone": is_monotone,
    }


# ──────────────────────────────────────────────────────────────────────────── #
# Saving & Plotting
# ──────────────────────────────────────────────────────────────────────────── #

def _save_exp1_results(
    metrics_df: pd.DataFrame,
    vw_wide: pd.DataFrame,
    ew_wide: pd.DataFrame,
    hl_vw: pd.Series,
    hl_ew: pd.Series,
    coefficients_dict: dict,
    qc: dict,
    is_monotone: bool,
    cfg: dict,
    elapsed: float,
) -> None:
    """Save all experiment 1 results to the results/ directory."""
    results_dir = cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. CSV metrics table
    metrics_df.to_csv(os.path.join(results_dir, "exp1_decile_metrics.csv"))

    # ── 2. Plot: Decile mean returns (VW)
    _plot_decile_returns(vw_wide, ew_wide, results_dir)

    # ── 3. Plot: Cumulative H-L return
    _plot_cumulative_hl(hl_vw, hl_ew, results_dir)

    # ── 4. Plot: Coefficient heatmap
    _plot_coefficient_heatmap(coefficients_dict, results_dir)

    # ── 5. Plot: H-L rolling Sharpe
    _plot_rolling_sharpe(hl_vw, hl_ew, results_dir)

    # ── 6. Append to RESULTS.md
    hl_row = metrics_df.loc["H-L"] if "H-L" in metrics_df.index else pd.Series()
    _write_results_md(
        metrics_df=metrics_df,
        hl_row=hl_row,
        qc=qc,
        is_monotone=is_monotone,
        cfg=cfg,
        elapsed=elapsed,
        results_dir=results_dir,
    )

    print(f"\nResults saved to: {results_dir}")


def _plot_decile_returns(
    vw_wide: pd.DataFrame,
    ew_wide: pd.DataFrame,
    results_dir: str,
) -> None:
    """Bar chart of mean monthly return per decile (VW and EW)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    deciles = sorted([c for c in vw_wide.columns if isinstance(c, (int, float))])
    vw_means = [vw_wide[d].mean() * 100 for d in deciles]
    ew_means = [ew_wide[d].mean() * 100 if d in ew_wide.columns else np.nan for d in deciles]

    colors = sns.color_palette("coolwarm", len(deciles))

    ax = axes[0]
    ax.bar([str(d) for d in deciles], vw_means, color=colors)
    ax.set_title("Value-Weighted Decile Mean Monthly Returns", fontsize=13)
    ax.set_xlabel("Decile (1=Low DRI, 10=High DRI)")
    ax.set_ylabel("Mean Monthly Return (%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax = axes[1]
    ax.bar([str(d) for d in deciles], ew_means, color=colors)
    ax.set_title("Equal-Weighted Decile Mean Monthly Returns", fontsize=13)
    ax.set_xlabel("Decile (1=Low DRI, 10=High DRI)")
    ax.set_ylabel("Mean Monthly Return (%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_decile_returns.png"), dpi=150)
    plt.close()


def _plot_cumulative_hl(
    hl_vw: pd.Series,
    hl_ew: pd.Series,
    results_dir: str,
) -> None:
    """Cumulative H-L return plot for VW and EW portfolios."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_vw = (1 + hl_vw).cumprod()
    cum_ew = (1 + hl_ew).cumprod()

    ax.plot(cum_vw.index, cum_vw.values, label="VW H-L", color="steelblue", linewidth=1.5)
    ax.plot(cum_ew.index, cum_ew.values, label="EW H-L", color="darkorange", linewidth=1.5, linestyle="--")
    ax.set_title("Cumulative Return of DRI H-L Portfolio", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_cumulative_hl.png"), dpi=150)
    plt.close()


def _plot_coefficient_heatmap(
    coefficients_dict: dict,
    results_dir: str,
) -> None:
    """Heatmap of chronological and rank coefficient vectors over years."""
    years = sorted(coefficients_dict.keys())
    if not years:
        return

    beta_C_matrix = np.array([coefficients_dict[y]["beta_C"] for y in years])
    beta_R_matrix = np.array([coefficients_dict[y]["beta_R"] for y in years])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(years) * 0.4 + 2)))

    for ax, mat, title in zip(
        axes,
        [beta_C_matrix, beta_R_matrix],
        ["beta^C (Chronological, d=1…21)", "beta^R (Rank-Ordered, k=1…21)"],
    ):
        vmax = np.abs(mat).max()
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Feature Index (1=most recent/largest)")
        ax.set_ylabel("Training Year")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_coefficient_heatmap.png"), dpi=150)
    plt.close()


def _plot_rolling_sharpe(
    hl_vw: pd.Series,
    hl_ew: pd.Series,
    results_dir: str,
    window: int = 36,
) -> None:
    """Rolling Sharpe ratio of H-L portfolio over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    def rolling_sr(s: pd.Series, w: int) -> pd.Series:
        rm = s.rolling(w).mean()
        rs = s.rolling(w).std(ddof=1)
        return (rm / rs * np.sqrt(12)).rename(s.name)

    sr_vw = rolling_sr(hl_vw, window)
    sr_ew = rolling_sr(hl_ew, window)

    ax.plot(sr_vw.index, sr_vw.values, label=f"VW H-L ({window}m rolling SR)", color="steelblue")
    ax.plot(sr_ew.index, sr_ew.values, label=f"EW H-L ({window}m rolling SR)", color="darkorange", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling Annualized Sharpe Ratio of DRI H-L Portfolio", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Annualized SR ({window}-month window)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_rolling_sharpe.png"), dpi=150)
    plt.close()


def _write_results_md(
    metrics_df: pd.DataFrame,
    hl_row: pd.Series,
    qc: dict,
    is_monotone: bool,
    cfg: dict,
    elapsed: float,
    results_dir: str,
) -> None:
    """Write/append experiment 1 results to RESULTS.md."""
    md_path = os.path.join(results_dir, "RESULTS.md")
    exists = os.path.isfile(md_path)

    lines = []
    if not exists:
        lines += [
            "# DRI Signal Construction — Experimental Results\n\n",
            "Generated by quantitative finance DRI signal research pipeline.\n\n",
        ]

    lines += [
        "---\n\n",
        "## Experiment 1: DRI Signal Construction and Baseline Portfolio Performance\n\n",
        f"**Configuration:** n_stocks={cfg['n_stocks']}, "
        f"start={cfg['start_date']}, end={cfg['end_date']}, "
        f"train_end_year={cfg['train_end_year']}, test_start_year={cfg['test_start_year']}\n\n",
        f"**Runtime:** {elapsed:.1f} seconds\n\n",
        "### Key Headline Statistics (H-L Portfolio)\n\n",
    ]

    if not hl_row.empty:
        lines += [
            "| Metric | VW | EW |\n",
            "|--------|----:|----:|\n",
            f"| Mean Monthly Return (%) | {hl_row.get('VW_Mean_%', 'n/a'):.4f} | {hl_row.get('EW_Mean_%', 'n/a'):.4f} |\n",
            f"| Newey-West t-stat | {hl_row.get('VW_tstat', 'n/a')} | {hl_row.get('EW_tstat', 'n/a')} |\n",
            f"| Annualized Sharpe Ratio | {hl_row.get('VW_SR', 'n/a')} | {hl_row.get('EW_SR', 'n/a')} |\n",
            f"| FF6 Alpha (%) | {hl_row.get('VW_Alpha_%', 'n/a')} | {hl_row.get('EW_Alpha_%', 'n/a')} |\n",
            f"| FF6 Alpha t-stat | {hl_row.get('VW_Alpha_t', 'n/a')} | {hl_row.get('EW_Alpha_t', 'n/a')} |\n\n",
        ]
    else:
        lines.append("*(H-L row not found in metrics table)*\n\n")

    lines += [
        "### Quality Check\n\n",
        f"- Average beta_C[d=1] = {qc.get('avg_beta_C_d1', 'n/a'):.4f} "
        f"(negative: {qc.get('beta_C_d1_negative', 'n/a')})\n",
        f"- % years where chron dominates rank: "
        f"{qc.get('pct_years_C_dominates_R', 0)*100:.0f}%\n",
        f"- Monotonic return pattern across deciles: {is_monotone}\n\n",
        "### Full Decile Performance Table\n\n",
    ]

    lines.append(metrics_df.to_markdown() + "\n\n")
    lines += [
        "### Saved Artifacts\n\n",
        "- `exp1_decile_metrics.csv` — full decile performance table\n",
        "- `exp1_decile_returns.png` — bar chart of mean returns per decile\n",
        "- `exp1_cumulative_hl.png` — cumulative H-L return\n",
        "- `exp1_coefficient_heatmap.png` — annual coefficient heatmap\n",
        "- `exp1_rolling_sharpe.png` — rolling Sharpe ratio\n\n",
    ]

    mode = "a" if exists else "w"
    with open(md_path, mode) as f:
        f.writelines(lines)


# ──────────────────────────────────────────────────────────────────────────── #
# Entry Point
# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    results = run_exp1(verbose=True)
    print("\nExperiment 1 complete.")
    print(f"Metrics shape: {results['metrics_df'].shape}")
