# src/visuals/plot_feature_scatter_batch.py
from __future__ import annotations

import itertools
import importlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---- Useage ----
"""
from src.visuals.plot_feature_scatter_batch import main

# Generate up to 20 plots, 6 per PDF page, for RB
main(position="RB", max_plots=20, cols=3, rows=2)
"""

# ---- defaults ----
try:
    _HERE = Path(__file__).resolve()
    _PROJECT_ROOT = _HERE.parents[2]  # <project>/src/visuals/... -> up 2
except NameError:
    _PROJECT_ROOT = Path.cwd()

def _default_csv_for_position(pos: str) -> Path:
    return _PROJECT_ROOT / "data" / "Bakery" / pos / f"Bakery_{pos}_Overall.csv"

def _default_out_for_position(pos: str) -> Path:
    return _PROJECT_ROOT / "tests" / "feature_scatter" / pos

_DEFAULT_CFG = "src.features_config"  # module exposing BASE_FEATURES and ALIASES

# ---- helpers ----
def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _find_col(df: pd.DataFrame, aliases: Dict[str, List[str]], canonical: str) -> Optional[str]:
    cands = aliases.get(canonical, [canonical])
    norm = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in cands:
        key = cand.lower().replace(" ", "")
        if key in norm:
            return norm[key]
    return None

def _label_name(row: pd.Series) -> str:
    name = str(row.get("Player") or row.get("player") or "").strip()
    if not name:
        return "Unknown"
    parts = name.split()
    return parts[0][0] + ". " + parts[-1] if len(parts) >= 2 else parts[0]

def _select_label_indices(df: pd.DataFrame, n: int, rule: str, xcol: str, ycol: str) -> pd.Index:
    if len(df) == 0:
        return df.index
    if n >= len(df) or rule == "all":
        return df.index
    if rule == "extremes":
        n1 = n // 2
        n2 = n - n1
        i1 = df.nsmallest(max(1, n1), xcol).index
        i2 = df.nlargest (max(1, n2), ycol).index
        return pd.Index(i1).union(i2)
    return df.sample(n=min(n, len(df)), random_state=42).index

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_config(module_path: str):
    mod = importlib.import_module(module_path)
    base_features = getattr(mod, "BASE_FEATURES")
    aliases = getattr(mod, "ALIASES")
    return list(base_features), aliases

def _build_pairs(
    base_features: Sequence[str],
    x: Optional[Sequence[str]],
    y: Optional[Sequence[str]],
    pairs: Optional[Sequence[Tuple[str, str]]],
    max_plots: Optional[int],
) -> List[Tuple[str, str]]:
    if pairs:
        out = list(dict.fromkeys([(a, b) for a, b in pairs if a != b]))
    elif x and y:
        out = [(a, b) for a in x for b in y if a != b]
    else:
        out = []
        for a, b in itertools.permutations(base_features, 2):
            if a != b:
                out.append((a, b))
        out = list(dict.fromkeys(out))
    if max_plots is not None:
        out = out[: max(0, int(max_plots))]
    return out

def _safe_slug(text: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_]+', '_', text)
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def _linreg_and_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (slope, intercept, r2) using simple linear regression.
    r2 computed as correlation^2 (safe when x,y > 1 sample).
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r2 = float(r * r)
    return float(m), float(b), r2

# Optional “visual inversion” for lower-is-better (purely visual)
_INVERT_X_VISUALLY = {"40 Time", "Draft Capital"}
_INVERT_Y_VISUALLY = set()

# ----main-----
def main(
    position: str = "RB",
    csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    config_module: str = _DEFAULT_CFG,
    *,
    x: Optional[Sequence[str]] = None,
    y: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    max_plots: Optional[int] = 24,
    max_labels: int = 35,
    label_rule: str = "extremes",
    dropna_any: bool = True,
    # NEW: page layout + output
    cols: int = 2,
    rows: int = 2,
    pdf_name: Optional[str] = None,     # default auto name like RB_feature_scatters.pdf
    also_pngs: bool = False,            # if True, also write individual PNGs
) -> List[Path]:
    """
    Generate scatter plots (with slope & R²) into a single multi-page PDF (and optional PNGs).

    Each subplot shows a least-squares fit line; annotation includes slope (m) and R².
    Draft Capital rows equal to zero are dropped whenever Draft Capital is on X or Y.
    """
    pos = position.upper()
    csv_path = Path(csv) if csv is not None else _default_csv_for_position(pos)
    out_path = Path(out_dir) if out_dir is not None else _default_out_for_position(pos)
    _safe_mkdir(out_path)

    # PDF file path
    if pdf_name is None:
        pdf_path = out_path / f"{pos}_feature_scatters.pdf"
    else:
        pdf_path = out_path / pdf_name

    base_features, aliases = _load_config(config_module)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    name_col = "Player" if "Player" in df.columns else ("player" if "player" in df.columns else None)
    requested_pairs = _build_pairs(base_features, x, y, pairs, max_plots)

    written_pngs: List[Path] = []
    per_page = max(1, cols * rows)
    total = len(requested_pairs)
    n_pages = math.ceil(total / per_page) if total else 0

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(n_pages):
            subset = requested_pairs[page_idx * per_page : (page_idx + 1) * per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 4.6), constrained_layout=True)
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            for ax, pair in itertools.zip_longest(axes, subset, fillvalue=None):
                if pair is None:
                    # empty slot on last page
                    ax.axis("off")
                    continue

                cx, cy = pair
                col_x = _find_col(df, aliases, cx)
                col_y = _find_col(df, aliases, cy)
                if not col_x or not col_y:
                    ax.set_visible(False)
                    continue

                plot_df = df[[col_x, col_y] + ([name_col] if name_col else [])].copy()
                plot_df[col_x] = _coerce_numeric(plot_df[col_x])
                plot_df[col_y] = _coerce_numeric(plot_df[col_y])

                # drop Draft Capital == 0 if used
                if cx == "Draft Capital":
                    plot_df = plot_df[plot_df[col_x] != 0]
                if cy == "Draft Capital":
                    plot_df = plot_df[plot_df[col_y] != 0]

                if dropna_any:
                    plot_df = plot_df.dropna(subset=[col_x, col_y])
                if plot_df.empty:
                    ax.set_visible(False)
                    continue

                # scatter
                ax.scatter(plot_df[col_x], plot_df[col_y], s=28, alpha=0.9)
                ax.set_xlabel(cx)
                ax.set_ylabel(cy)
                ax.set_title(f"{pos}: {cx} vs {cy}  (n={len(plot_df)})", fontsize=11)

                # optional visual inversion
                if cx in _INVERT_X_VISUALLY:
                    ax.invert_xaxis()
                if cy in _INVERT_Y_VISUALLY:
                    ax.invert_yaxis()

                # regression line + stats
                xvals = plot_df[col_x].to_numpy(dtype=float)
                yvals = plot_df[col_y].to_numpy(dtype=float)
                m, b, r2 = _linreg_and_stats(xvals, yvals)
                if np.isfinite(m) and np.isfinite(b):
                    xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
                    ys = m * xs + b
                    ax.plot(xs, ys, linewidth=2)

                # annotate slope & R² in upper-left corner of axes
                txt = f"m = {m:.4f}   R² = {r2:.3f}" if np.isfinite(r2) else "m = n/a   R² = n/a"
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))

                # labels (subset)
                lab_idx = _select_label_indices(plot_df, n=max_labels, rule=label_rule, xcol=col_x, ycol=col_y)
                for _, row in plot_df.loc[lab_idx].iterrows():
                    label = _label_name(row) if name_col else ""
                    if label:
                        ax.annotate(
                            label,
                            (row[col_x], row[col_y]),
                            xytext=(0, 0), textcoords="offset points",
                            fontsize=8, ha="left", va="center",
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                            clip_on=True
                        )

                ax.grid(alpha=0.15)

                # optional individual PNGs
                if also_pngs:
                    fname = f"{pos}_{_safe_slug(cx)}_vs_{_safe_slug(cy)}.png"
                    png_path = out_path / fname
                    fig_png = plt.figure(figsize=(6.5, 5.0))
                    ax_png = fig_png.add_subplot(111)
                    ax_png.scatter(plot_df[col_x], plot_df[col_y], s=28, alpha=0.9)
                    ax_png.set_xlabel(cx); ax_png.set_ylabel(cy)
                    ax_png.set_title(f"{pos}: {cx} vs {cy}  (n={len(plot_df)})")
                    if cx in _INVERT_X_VISUALLY: ax_png.invert_xaxis()
                    if cy in _INVERT_Y_VISUALLY: ax_png.invert_yaxis()
                    if np.isfinite(m) and np.isfinite(b):
                        xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
                        ys = m * xs + b
                        ax_png.plot(xs, ys, linewidth=2)
                    txt2 = f"m = {m:.4f}   R² = {r2:.3f}" if np.isfinite(r2) else "m = n/a   R² = n/a"
                    ax_png.text(0.02, 0.98, txt2, transform=ax_png.transAxes, va="top", ha="left",
                                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8",alpha=0.9))
                    ax_png.grid(alpha=0.15)
                    fig_png.savefig(png_path, dpi=240)
                    plt.close(fig_png)
                    written_pngs.append(png_path)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"[{pos}] Wrote multi-page PDF → {pdf_path}")
    if also_pngs:
        print(f"[{pos}] Also wrote {len(written_pngs)} PNGs to {out_path}")
    return [pdf_path] + written_pngs

# ----CLI----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Batch feature scatter plots into a single PDF.")
    ap.add_argument("--position", type=str, default="RB", help="Position (RB, WR, QB, TE)")
    ap.add_argument("--csv", type=str, default=None, help="Optional path to CSV")
    ap.add_argument("--out-dir", type=str, default=None, help="Folder to save outputs")
    ap.add_argument("--config-module", type=str, default=_DEFAULT_CFG, help="Module with BASE_FEATURES & ALIASES")
    ap.add_argument("--x", nargs="*", type=str, default=None, help="List of canonical features to use on X")
    ap.add_argument("--y", nargs="*", type=str, default=None, help="List of canonical features to use on Y")
    ap.add_argument("--pairs", nargs="*", type=str, default=None,
                    help='Explicit pairs like: --pairs "DOM++:YPC" "ELU:YCO/A"')
    ap.add_argument("--max-plots", type=int, default=24, help="Cap the number of plots")
    ap.add_argument("--max-labels", type=int, default=35, help="Max labels per subplot")
    ap.add_argument("--label-rule", type=str, default="extremes",
                    choices=["all","extremes","random"], help="Which points to annotate")
    ap.add_argument("--cols", type=int, default=2, help="Subplots per row (PDF pages)")
    ap.add_argument("--rows", type=int, default=2, help="Subplots per column (PDF pages)")
    ap.add_argument("--pdf-name", type=str, default=None, help="Output PDF filename (default auto)")
    ap.add_argument("--also-pngs", action="store_true", help="Also save individual PNGs")
    args = ap.parse_args()

    parsed_pairs = None
    if args.pairs:
        parsed_pairs = []
        for s in args.pairs:
            if ":" in s:
                a, b = s.split(":", 1)
                parsed_pairs.append((a.strip(), b.strip()))

    main(
        position=args.position,
        csv=args.csv,
        out_dir=args.out_dir,
        config_module=args.config_module,
        x=args.x,
        y=args.y,
        pairs=parsed_pairs,
        max_plots=args.max_plots,
        max_labels=args.max_labels,
        label_rule=args.label_rule,
        cols=args.cols,
        rows=args.rows,
        pdf_name=args.pdf_name,
        also_pngs=args.also_pngs,
    )
