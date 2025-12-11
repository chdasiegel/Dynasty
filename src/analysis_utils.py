import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# scikit-learn imports are done lazily inside the functions that need them
# to avoid import-time errors when sklearn is not available or when static
# analyzers cannot resolve sklearn sources.


# ---- Model Spaces ----
def model_spaces(random_state: int):

    """
    Return a list of (tag, estimator, param_grid) for all supported models.

    Notes:
    - RF: avoid 'auto' (deprecated); use None/'sqrt'/'log2'.
    - HGB: uses 'max_iter' instead of 'n_estimators'.
    - Ridge/BayesianRidge don't expose feature_importances_; SHAP/perm importance still ok.
    """
    try:
        from importlib import import_module
        ensemble = import_module("sklearn.ensemble")
        linear = import_module("sklearn.linear_model")
        GradientBoostingRegressor = getattr(ensemble, "GradientBoostingRegressor")
        HistGradientBoostingRegressor = getattr(ensemble, "HistGradientBoostingRegressor")
        RandomForestRegressor = getattr(ensemble, "RandomForestRegressor")
        Ridge = getattr(linear, "Ridge")
        BayesianRidge = getattr(linear, "BayesianRidge")
    except Exception as e:
        raise ImportError(
            "scikit-learn is required to construct model spaces; please install it "
            "(e.g. `pip install scikit-learn`)."
        ) from e

    return [
        # Gradient Boosting (trees)
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 3],
        }),

        # HistGradientBoosting (fast tree boosting)
        ("HGB", HistGradientBoostingRegressor(random_state=random_state), {
            "max_iter": [400, 600, 800, 1000, 1200],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "max_depth": [2, 3, 4, None],
            "l2_regularization": [0.0, 0.1, 1.0],
            "max_leaf_nodes": [15, 31, 63],
        }),

        # Random Forest
        ("RF", RandomForestRegressor(random_state=random_state, n_jobs=-1), {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "max_depth": [None, 6, 10, 14],
            "min_samples_leaf": [1, 2, 3],
            "max_features": [None, "sqrt", "log2"],
        }),

        # Ridge Regression (linear, L2)
        ("Ridge", Ridge(), {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
            "max_iter": [1000, 2000],
            # 'random_state' is not a Ridge param; leave it out
        }),

        # Bayesian Ridge (linear, Bayesian)
        ("BR",  BayesianRidge(), {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "alpha_2": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
            "lambda_2": [1e-6, 1e-5, 1e-4],
            # You can add 'n_iter' if you want longer fits: [300, 600, 1000]
        }),
    ]


# ---- Pareto Chart ----
def save_pareto_chart(summary: pd.DataFrame, position: str, out_dir: Path) -> Path:
    """Save RMSE (best RMSE per n_subsets) vs avg runtime (seconds)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = (summary.groupby("n_subsets")
               .agg(best_RMSE=("best_test_RMSE","min"),
                    avg_runtime_s=("runtime_sec","mean"),
                    runs=("best_test_RMSE","count"))
               .reset_index())
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.scatter(agg["avg_runtime_s"], agg["best_RMSE"], s=80, alpha=0.9)
    for _, row in agg.iterrows():
        ax.annotate(int(row["n_subsets"]),
                    (row["avg_runtime_s"], row["best_RMSE"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ordered = agg.sort_values("n_subsets")
    ax.plot(ordered["avg_runtime_s"], ordered["best_RMSE"], linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Average runtime per setting (seconds)")
    ax.set_ylabel("Best Test RMSE (across seeds)")
    ax.set_title(f"{position.upper()}: RMSE vs Runtime by N_SUBSETS (Pareto view)")
    ax.grid(alpha=0.2)
    png_path = out_dir / f"{position.lower()}_rmse_vs_runtime.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path

# ---- SHAP ----

def compute_shap(best_est, Xtr_arr, Xte_arr):
    """Return (sv, base_value) if SHAP works; else (None, 0.s0)."""
    try:
        from importlib import import_module
        shap = import_module("shap")
    except Exception:
        return None, 0.0
    try:
        explainer = shap.TreeExplainer(best_est)
        sv = explainer.shap_values(Xte_arr)
        base_val = getattr(explainer, "expected_value", 0.0)
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.asarray(base_val).ravel()[0])
        return np.asarray(sv), float(base_val)
    except Exception as e:
        warnings.warn(f"SHAP failed: {e}")
        return None, 0.0

# ---- Feature Caps ----

def save_importances(
    out_dir: Path,
    pos: str,
    seed: int,
    n_subsets: int,
    feature_names: list[str],
    feature_kinds: dict[str, str],
    model,
    shap_values: np.ndarray | None = None,
    X_te_arr: np.ndarray | None = None,
    y_te: np.ndarray | None = None,
    do_permutation: bool = False,
    draft_cap_importance_cap: float | None = None,
    breakout_age_importance_cap: float | None = None,
    draft_age_importance_cap: float | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    def to_df_with_kind(scores: pd.Series, colname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # cap selected features if caps provided
        s = scores.copy()
        def imp_cap(name: str, cap: float | None):
            nonlocal s
            if cap is None or name not in s.index:
                return
            s.loc[name] = min(float(s.loc[name]), float(cap))
        imp_cap("Draft Capital", draft_cap_importance_cap)
        imp_cap("Breakout Age", breakout_age_importance_cap)
        imp_cap("Draft Age", draft_age_importance_cap)

        df = pd.DataFrame({
            "feature": s.index,
            colname: s.values,
            "kind": [feature_kinds.get(f, "base") for f in s.index],
        })
        df_sorted = df.sort_values(colname, ascending=False).reset_index(drop=True)
        split = (df.groupby("kind")[colname].sum()
                   .reindex(["base", "interaction"])
                   .rename_axis("kind")
                   .reset_index())
        return df_sorted, split

    # Tree FI
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names, name="tree_feature_importance")
        per_df, split_df = to_df_with_kind(fi, "tree_feature_importance")
        p1 = out_dir / f"{pos.lower()}_tree_importances_seed{seed}_subs{n_subsets}.csv"
        p1s = out_dir / f"{pos.lower()}_tree_importances_split_seed{seed}_subs{n_subsets}.csv"
        per_df.to_csv(p1, index=False); split_df.to_csv(p1s, index=False)
        paths["tree_importances_csv"] = str(p1); paths["tree_importances_split_csv"] = str(p1s)

    # SHAP mean |v|
    if shap_values is not None:
        # Compute mean absolute SHAP values across all test samples
        shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
        shap_ser = pd.Series(shap_mean_abs, index=feature_names, name="shap_mean_abs")
        per_df, split_df = to_df_with_kind(shap_ser, "shap_mean_abs")
        p2 = out_dir / f"{pos.lower()}_shap_importances_seed{seed}_subs{n_subsets}.csv"
        p2s = out_dir / f"{pos.lower()}_shap_importances_split_seed{seed}_subs{n_subsets}.csv"
        per_df.to_csv(p2, index=False); split_df.to_csv(p2s, index=False)
        paths["shap_importances_csv"] = str(p2); paths["shap_importances_split_csv"] = str(p2s)

    # Permutation
    if do_permutation and X_te_arr is not None and y_te is not None:
        try:
            from importlib import import_module
            inspection = import_module("sklearn.inspection")
            permutation_importance = getattr(inspection, "permutation_importance")
        except Exception as e:
            warnings.warn(f"scikit-learn permutation_importance not available: {e}")
        else:
            r = permutation_importance(model, X_te_arr, y_te, scoring="r2", n_repeats=10, random_state=0, n_jobs=-1)
            perm_ser = pd.Series(r.importances_mean, index=feature_names, name="perm_importance_mean")
            per_df, split_df = to_df_with_kind(perm_ser, "perm_importance_mean")
            p3 = out_dir / f"{pos.lower()}_perm_importances_seed{seed}_subs{n_subsets}.csv"
            p3s = out_dir / f"{pos.lower()}_perm_importances_split_seed{seed}_subs{n_subsets}.csv"
            per_df.to_csv(p3, index=False); split_df.to_csv(p3s, index=False)
            paths["permutation_importances_csv"] = str(p3); paths["permutation_importances_split_csv"] = str(p3s)

    return paths