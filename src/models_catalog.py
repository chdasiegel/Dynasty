# src.models_catalog.py
# =====================================================
# Catalog of model functions
# =====================================================



    """
    ----Example Use----
    spaces = model_spaces(42, include=("GB","HGB"), shrink=True)

for tag, model, grid in spaces:
    print("Model:", tag)
    print("Estimator object:", model)
    print("Parameter grid:", grid)
    """

from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor, HistGradientBoostingRegressor
)


# ----Model Spaces----

def model_spaces(random_state: int,
                 include=("GB","RF","ET","HGB"),
                 shrink=False):
    """
    Return a list of (tag, estimator, param_grid) tuples you can feed into CV.
    
    ----Args----
    random_state : int         #Seed for deterministic behavior.
    include : tuple[str]       #Which models to include: any of {"GB","RF","ET","HGB"}.
    shrink : bool              #If True, uses a smaller grid (faster search).

    ----Returns----
    list[tuple[str, estimator, dict]]
    """
    out = []

    if "GB" in include:
        gb = GradientBoostingRegressor(random_state=random_state)
        gb_grid = (
            {
                "n_estimators": [400, 600, 800, 1000, 1200],
                "learning_rate": [0.03, 0.05, 0.07, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_leaf": [1, 2, 3],
            } if not shrink else
            {
                "n_estimators": [600, 900],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3],
                "subsample": [0.9, 1.0],
                "min_samples_leaf": [1, 2],
            }
        )
        out.append(("GB", gb, gb_grid))

    if "RF" in include:
        rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        rf_grid = (
            {
                "n_estimators": [600, 900, 1200, 1500],
                "max_depth": [None, 12, 16, 20],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "max_features": ["sqrt", 0.7, 0.9, 1.0],
            } if not shrink else
            {
                "n_estimators": [800, 1200],
                "max_depth": [None, 16],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.9],
            }
        )
        out.append(("RF", rf, rf_grid))

    if "ET" in include:
        et = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
        et_grid = (
            {
                "n_estimators": [600, 900, 1200, 1500],
                "max_depth": [None, 12, 16, 20],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "max_features": ["sqrt", 0.7, 0.9, 1.0],
            } if not shrink else
            {
                "n_estimators": [800, 1200],
                "max_depth": [None, 16],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.9],
            }
        )
        out.append(("ET", et, et_grid))

    if "HGB" in include:
        hgb = HistGradientBoostingRegressor(random_state=random_state)
        hgb_grid = (
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_depth": [3, 6, 9],
                "l2_regularization": [0.0, 0.1, 0.3, 0.5],
                "max_bins": [128, 255],
            } if not shrink else
            {
                "learning_rate": [0.05, 0.1],
                "max_depth": [6],
                "l2_regularization": [0.0, 0.1],
                "max_bins": [255],
            }
        )
        out.append(("HGB", hgb, hgb_grid))

    return out