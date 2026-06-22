"""
Phase C: Expanding Window Elastic-Net Model.

Estimates elastic-net regression annually on an expanding training window
with 5×5-fold repeated cross-validation to select the regularization
parameter lambda. Stores annual coefficient vectors beta^C and beta^R.

Using sklearn ElasticNet — Context7 confirmed
Using sklearn RepeatedKFold — Context7 confirmed
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, cross_val_score


def compute_lambda_max(
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.5,
) -> float:
    """
    Compute lambda_max: the smallest regularization value that shrinks all
    coefficients to zero.

    Formula (paper Phase C.3):
    lambda_max = max(|X' * (y - mean(y))|) / (N * eta)

    Args:
        X: Feature matrix of shape (N, p).
        y: Response vector of shape (N,).
        eta: Elastic-net mixing parameter (l1_ratio), default 0.5.

    Returns:
        Scalar lambda_max > 0.
    """
    N = X.shape[0]
    y_centered = y - y.mean()
    correlations = np.abs(X.T @ y_centered)
    lambda_max = float(correlations.max()) / (N * eta)
    return lambda_max


def build_lambda_grid(
    lambda_max: float,
    n_lambdas: int = 100,
) -> np.ndarray:
    """
    Generate a log-spaced regularization grid from lambda_max to lambda_max*1e-4.

    The grid covers the full regularization path from the null model
    (lambda = lambda_max, all zero coefficients) to an almost unregularized
    model (lambda = lambda_max * 1e-4).

    # Custom — Context7 found no library equivalent (paper Phase C.3)

    Args:
        lambda_max: Upper bound of regularization grid.
        n_lambdas: Number of grid points (default 100).

    Returns:
        Array of shape (n_lambdas,) in descending order.
    """
    grid = np.logspace(
        np.log10(lambda_max),
        np.log10(lambda_max * 1e-4),
        n_lambdas,
    )
    return grid  # descending order


def select_lambda_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_grid: np.ndarray,
    eta: float = 0.5,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    max_iter: int = 10000,
    n_jobs: int = -1,
) -> Tuple[float, np.ndarray]:
    """
    Select optimal regularization parameter via 5×5-fold repeated CV.

    Uses sklearn ElasticNetCV with the paper-defined lambda grid and
    RepeatedKFold splitter (5 splits × 5 repeats = 25 evaluations).
    ElasticNetCV exploits the warm-start regularization path for a large
    speedup over evaluating each lambda independently; the selected lambda
    and per-fold MSE values are identical to the naive loop.

    Note on sklearn convention:
    - sklearn's `alpha` parameter corresponds to the paper's lambda.
    - sklearn's `l1_ratio` parameter corresponds to the paper's eta.

    Using sklearn ElasticNetCV (warm-start path) — Context7 confirmed
    Using sklearn RepeatedKFold — Context7 confirmed

    Args:
        X_train: Training feature matrix (N, p).
        y_train: Training response vector (N,).
        lambda_grid: Array of lambda values to evaluate (descending order).
        eta: Elastic-net mixing parameter (l1_ratio=0.5 as per paper).
        n_splits: Number of CV folds (5 as per paper).
        n_repeats: Number of CV repetitions (5 as per paper).
        random_state: Seed for reproducibility (42 as per paper).
        max_iter: Maximum iterations for coordinate descent.
        n_jobs: Number of parallel jobs (-1 = use all CPUs).

    Returns:
        (best_lambda, mse_array) where mse_array has shape (n_lambdas,).
    """
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    # ElasticNetCV expects alphas in ascending order internally; pass the grid
    # as-is — it will sort and compute the path efficiently via warm-starting.
    # n_jobs=-1 uses all available CPUs to parallelize the 25 CV folds.
    cv_model = ElasticNetCV(
        alphas=lambda_grid,
        l1_ratio=eta,
        fit_intercept=True,
        cv=rkf,
        max_iter=max_iter,
        tol=1e-4,
        n_jobs=n_jobs,
    )
    cv_model.fit(X_train, y_train)

    best_lambda = float(cv_model.alpha_)

    # Reconstruct per-lambda mean MSE from cv_model.mse_path_
    # mse_path_ shape: (n_alphas, n_folds); average over folds
    # Note: ElasticNetCV sorts alphas descending internally; align to lambda_grid
    sorted_alphas = np.sort(lambda_grid)[::-1]  # descending
    # cv_model.alphas_ is sorted descending; map to our lambda_grid order
    alpha_to_mse = dict(
        zip(cv_model.alphas_, cv_model.mse_path_.mean(axis=1))
    )
    mse_array = np.array([
        alpha_to_mse.get(lam, np.nan) for lam in lambda_grid
    ])

    return best_lambda, mse_array


def fit_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_val: float,
    eta: float = 0.5,
    max_iter: int = 10000,
) -> Tuple[float, np.ndarray]:
    """
    Fit elastic-net on full training data with a given regularization lambda.

    Using sklearn ElasticNet — Context7 confirmed

    Args:
        X_train: Training feature matrix (N, p).
        y_train: Training response vector (N,).
        lambda_val: Regularization parameter (paper's lambda = sklearn's alpha).
        eta: Mixing parameter (paper's eta = sklearn's l1_ratio).
        max_iter: Maximum coordinate descent iterations.

    Returns:
        (intercept, coefficients) where:
        - intercept: scalar alpha (not penalized)
        - coefficients: array of shape (p,); beta^C = coef[0:21], beta^R = coef[21:42]
    """
    model = ElasticNet(
        alpha=lambda_val,
        l1_ratio=eta,
        fit_intercept=True,
        max_iter=max_iter,
        tol=1e-4,
    )
    model.fit(X_train, y_train)
    return float(model.intercept_), model.coef_.astype(np.float32)


def run_expanding_window_estimation(
    X_all: np.ndarray,
    y_all: np.ndarray,
    meta_df: pd.DataFrame,
    train_end_year: int = 1994,
    start_year: int = 1985,
    eta: float = 0.5,
    n_lambdas: int = 100,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    max_iter: int = 10000,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """
    Run expanding window elastic-net estimation for each year Y.

    For each year Y from start_year (when enough data exists) to train_end_year:
    1. Collect all training observations: months with year(date) <= Y
    2. Compute lambda_max from training data
    3. Build log-spaced lambda grid (100 values)
    4. Run 5×5 repeated CV to select best lambda
    5. Refit on full training data with selected lambda
    6. Store annual coefficient vectors

    All 12 months of year Y+1 use the SAME coefficient vector estimated
    using data through December Y (no look-ahead bias).

    Args:
        X_all: Full feature matrix (N_total, p).
        y_all: Full response vector (N_total,).
        meta_df: Metadata DataFrame with 'date' column (same rows as X_all).
        train_end_year: Last year for which to estimate a model (default 1994).
        start_year: First year of the sample (default 1985).
        eta: Elastic-net mixing parameter (eta=0.5 per paper).
        n_lambdas: Number of lambda values in grid (default 100).
        n_splits: CV folds (default 5).
        n_repeats: CV repetitions (default 5).
        random_state: Seed for reproducibility.
        max_iter: Maximum iterations for ElasticNet solver.
        verbose: Print progress per year.

    Returns:
        Dict mapping year Y → {'intercept': float, 'beta_C': array(21),
                                'beta_R': array(21), 'beta': array(p),
                                'lambda': float, 'n_train': int,
                                'mse_curve': array(n_lambdas)}
    """
    meta_df = meta_df.copy()
    meta_df["year"] = pd.to_datetime(meta_df["date"]).dt.year

    coefficients_dict: Dict[int, Dict] = {}

    years_to_estimate = range(start_year, train_end_year + 1)

    for year_Y in years_to_estimate:
        # Expanding window: all months with year <= Y
        train_mask = meta_df["year"] <= year_Y
        n_train = train_mask.sum()

        if n_train < 50:
            if verbose:
                print(f"  Year {year_Y}: only {n_train} training obs, skipping.")
            continue

        X_train = X_all[train_mask].astype(np.float64)
        y_train = y_all[train_mask].astype(np.float64)

        # Compute regularization grid
        lam_max = compute_lambda_max(X_train, y_train, eta=eta)
        lambda_grid = build_lambda_grid(lam_max, n_lambdas=n_lambdas)

        # 5×5 CV to select best lambda
        best_lam, mse_curve = select_lambda_cv(
            X_train,
            y_train,
            lambda_grid,
            eta=eta,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            max_iter=max_iter,
        )

        # Final fit on full training data
        intercept, beta = fit_elastic_net(
            X_train, y_train, lambda_val=best_lam, eta=eta, max_iter=max_iter
        )

        n_features = X_all.shape[1]
        half = n_features // 2

        coefficients_dict[year_Y] = {
            "intercept": intercept,
            "beta_C": beta[:half].copy(),
            "beta_R": beta[half:].copy(),
            "beta": beta.copy(),
            "lambda": best_lam,
            "lambda_max": lam_max,
            "n_train": int(n_train),
            "mse_curve": mse_curve,
            "lambda_grid": lambda_grid,
        }

        if verbose:
            n_nonzero = int(np.sum(beta != 0))
            print(
                f"  Year {year_Y}: N_train={n_train:,}, lambda*={best_lam:.2e}, "
                f"n_nonzero={n_nonzero}, beta_C[0]={beta[0]:.4f}"
            )

    return coefficients_dict


def quality_check_coefficients(coefficients_dict: Dict) -> Dict:
    """
    Perform quality checks on annually estimated coefficient vectors.

    Key checks (from paper implementation notes):
    1. beta^C[0] (d=1, most recent day) should be negative on average
       (short-term reversal effect dominates)
    2. sum(|beta^C|) > sum(|beta^R|) in most years (chronological dominance)
    3. No years where lambda = lambda_max (boundary solution = null model)

    Args:
        coefficients_dict: Output from run_expanding_window_estimation.

    Returns:
        Dict with quality metrics and pass/fail flags.
    """
    if not coefficients_dict:
        return {"error": "Empty coefficients dict"}

    years = sorted(coefficients_dict.keys())
    beta_C_d1 = [coefficients_dict[y]["beta_C"][0] for y in years]
    sum_abs_C = [np.sum(np.abs(coefficients_dict[y]["beta_C"])) for y in years]
    sum_abs_R = [np.sum(np.abs(coefficients_dict[y]["beta_R"])) for y in years]
    lambdas = [coefficients_dict[y]["lambda"] for y in years]
    lambda_maxes = [coefficients_dict[y]["lambda_max"] for y in years]

    avg_beta_C_d1 = float(np.mean(beta_C_d1))
    pct_C_dominates = float(
        np.mean([c > r for c, r in zip(sum_abs_C, sum_abs_R)])
    )
    pct_boundary = float(
        np.mean([abs(l - lm) / lm < 1e-3 for l, lm in zip(lambdas, lambda_maxes)])
    )

    return {
        "avg_beta_C_d1": avg_beta_C_d1,
        "beta_C_d1_negative": avg_beta_C_d1 < 0,
        "pct_years_C_dominates_R": pct_C_dominates,
        "pct_boundary_solutions": pct_boundary,
        "n_years": len(years),
        "years": years,
        "beta_C_d1_per_year": dict(zip(years, beta_C_d1)),
    }
