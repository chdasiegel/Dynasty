import gradio as gr
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

"""
Dynasty QB Predictor — new prediction from your rankings.

- A prepopulated list of QBs is shown with default rankings (editable).
- User adjusts the rank column (1 = best, 2 = second, ...), then clicks Run model.
- The app fits a new model using (features, your_rank) and returns the
  model ranking and standardized score.
"""

# These files must live next to app.py in the Space repo.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "QB_model.pkl"
FEATURE_CSV = APP_DIR / "master_list_with_ranks.csv"
# Default list: rank order from this CSV (columns: rank, player_name or player-name).
QB_RANKS_CSV = APP_DIR / "qb_ranks.csv"


def _load_model_bundle():
    """Load model, feature list, and optional imputer from QB_model.pkl."""
    with open(str(MODEL_PATH), "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("best_model") or bundle.get("estimator")
        feature_names = (
            bundle.get("feature_names")
            or bundle.get("features")
            or bundle.get("feature_cols")
        )
        imputer = bundle.get("imputer")
    else:
        # Fallback: bundle itself is the model
        model = bundle
        feature_names = getattr(bundle, "feature_names_in_", None)
        imputer = None

    if model is None:
        raise ValueError("QB_model.pkl did not contain a 'model' object.")
    if feature_names is None:
        raise ValueError(
            "QB_model.pkl did not contain a feature list "
            "(expected 'feature_names', 'features', or 'feature_cols')."
        )

    return model, list(feature_names), imputer


def _load_qb_features(feature_cols: list[str]) -> pd.DataFrame:
    """
    Load the static feature table and build a QB-only view with a normalized
    player key and exactly the columns the model expects.
    """
    df = pd.read_csv(str(FEATURE_CSV))
    df.columns = [c.strip() for c in df.columns]

    # Detect name + position columns
    if "player_name" not in df.columns:
        raise ValueError(
            f"{FEATURE_CSV} is missing required 'player_name' column."
        )
    if "pos" not in df.columns:
        raise ValueError(f"{FEATURE_CSV} is missing required 'pos' column.")

    # Restrict to QBs
    qb_df = df[df["pos"].str.upper() == "QB"].copy()

    # Normalize join key
    qb_df["player_key"] = qb_df["player_name"].astype(str).str.strip().str.lower()

    # Ensure all required feature columns exist (case-insensitive match)
    available_cols_lower = {c.lower(): c for c in qb_df.columns}
    missing = []
    feature_col_map = {}
    for feat in feature_cols:
        if feat in qb_df.columns:
            feature_col_map[feat] = feat
        elif feat.lower() in available_cols_lower:
            feature_col_map[feat] = available_cols_lower[feat.lower()]
        else:
            missing.append(feat)
    
    if missing:
        raise ValueError(
            f"Static feature table is missing required columns used by the model: {', '.join(missing[:10])}"
            + (f" (and {len(missing)-10} more)" if len(missing) > 10 else "")
            + f"\nAvailable columns: {', '.join(list(qb_df.columns)[:20])}..."
        )
    
    # Use mapped column names
    return qb_df, feature_col_map


try:
    _model, FEATURE_COLS, IMPUTER = _load_model_bundle()
    print(f"✓ Loaded feature config with {len(FEATURE_COLS)} features: {FEATURE_COLS[:5]}...")
    QB_FEATURES, FEATURE_COL_MAP = _load_qb_features(FEATURE_COLS)
    print(f"✓ Loaded {len(QB_FEATURES)} QB records from {FEATURE_CSV}")
    FEATURE_COLS_ACTUAL = [FEATURE_COL_MAP.get(f, f) for f in FEATURE_COLS]
    # Prepopulated default rankings: from qb_ranks.csv if present, else alphabetical
    _all_qb_names = QB_FEATURES["player_name"].drop_duplicates()
    if QB_RANKS_CSV.exists():
        _def = pd.read_csv(str(QB_RANKS_CSV))
        _def.columns = [c.strip().replace("-", "_") for c in _def.columns]
        if "rank" in _def.columns and "player_name" in _def.columns:
            _def = _def.sort_values("rank").drop_duplicates(subset=["player_name"], keep="first")
            _qb_names = _def["player_name"].astype(str).str.strip()
            _qb_names = _qb_names[_qb_names.str.len() > 0].reset_index(drop=True)
            if len(_qb_names) > 0:
                DEFAULT_RANKINGS = pd.DataFrame({
                    "rank": range(1, len(_qb_names) + 1),
                    "player_name": _qb_names,
                })
            else:
                _qb_names = _all_qb_names.sort_values().reset_index(drop=True)
                DEFAULT_RANKINGS = pd.DataFrame({"rank": range(1, len(_qb_names) + 1), "player_name": _qb_names})
        else:
            _qb_names = _all_qb_names.sort_values().reset_index(drop=True)
            DEFAULT_RANKINGS = pd.DataFrame({"rank": range(1, len(_qb_names) + 1), "player_name": _qb_names})
    else:
        _qb_names = _all_qb_names.sort_values().reset_index(drop=True)
        DEFAULT_RANKINGS = pd.DataFrame({"rank": range(1, len(_qb_names) + 1), "player_name": _qb_names})
    # For UI: one name per line (order = rank)
    DEFAULT_LIST_LINES = "\n".join(DEFAULT_RANKINGS["player_name"].astype(str).tolist())
except Exception as e:
    print(f"ERROR during startup: {e}")
    import traceback
    traceback.print_exc()
    raise


def _list_to_ranks(rankings_input):
    """Convert list input to ranks DataFrame. Items can be strings ('1. Name' or 'Name') or [rank, name]."""
    if not rankings_input or (isinstance(rankings_input, (list, tuple)) and len(rankings_input) == 0):
        return None
    rows = []
    for i, item in enumerate(rankings_input):
        if isinstance(item, (list, tuple)):
            name = str(item[-1]).strip() if len(item) else ""
        else:
            s = str(item).strip()
            if ". " in s and s.split(". ", 1)[0].replace(".", "").isdigit():
                name = s.split(". ", 1)[1].strip()
            else:
                name = s
        if name and not name.startswith("["):
            rows.append({"rank": i + 1, "player_name": name})
    return pd.DataFrame(rows) if rows else None


def _textarea_to_ranks(text: str):
    """Parse textarea: one player name per line; order = rank."""
    if not text or not str(text).strip():
        return None
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return pd.DataFrame({
        "rank": range(1, len(lines) + 1),
        "player_name": lines,
    })


def run_model(rankings_data):
    """
    Accept textarea (one name per line), list, or DataFrame. Fit a new model to (features, your_rank)
    and return the model ranking.
    """
    # Textarea: one player per line, order = rank
    if isinstance(rankings_data, str):
        ranks = _textarea_to_ranks(rankings_data)
        if ranks is None or ranks.empty:
            raise ValueError("Rankings are empty. Enter one player name per line; order = your ranking.")
        player_col = "player_name"
    # List: e.g. ["1. Jayden Daniels", "2. Trevor Lawrence"]
    elif isinstance(rankings_data, (list, tuple)):
        ranks = _list_to_ranks(rankings_data)
        if ranks is None or ranks.empty:
            raise ValueError("Rankings list is empty. Use the text box below, one name per line.")
        player_col = "player_name"
    elif isinstance(rankings_data, pd.DataFrame):
        ranks = rankings_data.copy()
        ranks.columns = [str(c).strip() for c in ranks.columns]
        # Normalize "Rank" -> "rank" for internal use
        if "Rank" in ranks.columns and "rank" not in ranks.columns:
            ranks["rank"] = range(1, len(ranks) + 1)  # row order = rank
        player_col = None
        for cand in ["player_name", "Player Name", "player", "Player"]:
            if cand in ranks.columns:
                player_col = cand
                break
        if player_col is None:
            for c in ranks.columns:
                if c.lower() != "rank":
                    player_col = c
                    break
        if player_col is None:
            raise ValueError("Table needs a player name column (e.g. Player).")
        if "rank" not in ranks.columns:
            ranks["rank"] = range(1, len(ranks) + 1)
        ranks["rank"] = pd.to_numeric(ranks["rank"], errors="coerce").ffill().fillna(1).astype(int)
    elif isinstance(rankings_data, dict) and "data" in rankings_data:
        ranks = pd.DataFrame(
            rankings_data["data"],
            columns=rankings_data.get("headers", ["rank", "player_name"]),
        )
        ranks.columns = [str(c).strip() for c in ranks.columns]
        if "Rank" in ranks.columns and "rank" not in ranks.columns:
            ranks["rank"] = range(1, len(ranks) + 1)
        player_col = "Player" if "Player" in ranks.columns else "player_name"
    else:
        ranks = pd.DataFrame(rankings_data)
        ranks.columns = [str(c).strip() for c in ranks.columns]
        # Gradio may pass list of lists -> columns 0,1: treat as Rank, Player
        if len(ranks.columns) >= 2 and ranks.columns[0].isdigit():
            ranks = ranks.rename(columns={ranks.columns[0]: "rank", ranks.columns[1]: "Player"})
            ranks["rank"] = range(1, len(ranks) + 1)
            player_col = "Player"
        else:
            player_col = "Player" if "Player" in ranks.columns else "player_name" if "player_name" in ranks.columns else ranks.columns[0]
    if ranks.empty or len(ranks) == 0:
        raise ValueError("Rankings are empty. Use the prepopulated list and drag to reorder.")

    # Normalize for join
    ranks["player_key"] = ranks[player_col].astype(str).str.strip().str.lower()

    # Join static QB feature table
    merged = ranks.merge(
        QB_FEATURES[["player_key"] + FEATURE_COLS_ACTUAL],
        on="player_key",
        how="left",
        suffixes=("", "_feat"),
    )

    # Track which players had no feature match (for display only; we still score them)
    merged["missing_features"] = merged[FEATURE_COLS_ACTUAL].isna().any(axis=1)

    # Predict for every row: fill missing features so no player is skipped
    X = merged[FEATURE_COLS_ACTUAL].copy()
    for col in FEATURE_COLS_ACTUAL:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if IMPUTER is not None:
        X_arr = IMPUTER.transform(X)
    else:
        # Fill NaN with column median from static QB table so model always gets valid input
        X_arr = X.values.copy()
        for j in range(X_arr.shape[1]):
            col_vals = X_arr[:, j]
            finite = col_vals[np.isfinite(col_vals)]
            fill = np.median(finite) if len(finite) > 0 else 0.0
            X_arr[~np.isfinite(col_vals), j] = fill

    # Use your submitted rankings as the target: rank 1 = best → target = -1, etc.
    y = -np.asarray(merged["rank"], dtype=float)

    if len(y) < 2:
        raise ValueError("Need at least 2 ranked players to fit a model.")

    # Fit a new model to (features, your_rank) so the prediction is based on your rankings
    fit_model = GradientBoostingRegressor(
        n_estimators=min(80, max(20, len(y) * 2)),
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    fit_model.fit(X_arr, y)
    preds = fit_model.predict(X_arr)
    merged["model_score"] = preds

    # Model ranking: order players by model score (higher = better), assign 1, 2, 3, ...
    merged = merged.sort_values("model_score", ascending=False).reset_index(drop=True)
    merged["model_rank"] = range(1, len(merged) + 1)

    # Scale model_score to 0–100 (no negative values)
    s = merged["model_score"]
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        merged["model_score_scaled"] = (s - s_min) / (s_max - s_min) * 100.0
    else:
        merged["model_score_scaled"] = 50.0
    merged["model_score_scaled"] = merged["model_score_scaled"].round(2)

    merged = merged.rename(columns={"rank": "your_rank", player_col: "Player", "model_score_scaled": "Score"})
    merged = merged.sort_values("model_rank").reset_index(drop=True)
    display_cols = ["model_rank", "Player", "Score", "your_rank", "missing_features"]
    return merged[display_cols]


# --- UI: Text box only — one name per line, order = rank ---
with gr.Blocks(title="Dynasty QB Predictor", css=".gradio-container { max-width: 720px; }") as demo:
    gr.Markdown("## Dynasty QB Predictor")
    gr.Markdown(
        "Enter your QB ranking below — **one player per line**.\n\n"
        "Order = your rank: first line = #1, second = #2, etc.\n\n"
        "To reorder: cut and paste lines (select a line, cut it, click where you want it, paste)."
    )
    rank_input = gr.Textbox(
        value=DEFAULT_LIST_LINES,
        label="Your rankings (one name per line)",
        placeholder="Patrick Mahomes\nJosh Allen\nLamar Jackson\n...",
        lines=22,
        max_lines=400,
        interactive=True,
    )
    run_btn = gr.Button("Run model", variant="primary")
    out_table = gr.Dataframe(label="Model ranking", interactive=False)

    run_btn.click(fn=run_model, inputs=rank_input, outputs=out_table)

if __name__ == "__main__":
    # Explicit server args for Hugging Face Spaces / Docker (avoids asyncio fd -1 errors)
    demo.launch(server_name="0.0.0.0", server_port=7860)