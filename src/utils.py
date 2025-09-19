# src/utils.py
from __future__ import annotations
import re
from typing import Iterable, Optional, Sequence, Dict
import pandas as pd
import sys

pd.set_option("display.max_columns", None)

# ---- Name cleaning ----
def clean_player_name(player_name):
    """Remove punctuation, suffixes, and extra spaces from player names."""
    if not isinstance(player_name, str):
        return player_name
    player_name = re.sub(r"[^\w\s'-]", '', player_name)  # Remove punctuation
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    player_name = re.sub(pattern, '', player_name, flags=re.IGNORECASE)
    return ' '.join(player_name.split())



def strip_name_marks(s: object) -> object:
    """Strip common marks like '*' from names."""
    if not isinstance(s, str):
        return s
    return s.replace("*", "")





# ---- Height parsing ----
def height_to_inches(ht: object) -> Optional[float]:
    """Convert 5-11, 5/11, '5 11', or 5'11 to inches. Ignores date-like strings."""
    if ht is None or (isinstance(ht, float) and pd.isna(ht)):
        return None
    ht = str(ht).strip()

    for pat in (r"^(\d+)[-/](\d+)$", r"^(\d+)\s+(\d+)$", r"^(\d+)'\s*(\d+)$"):
        m = re.match(pat, ht)
        if m:
            feet, inches = map(int, m.groups())
            return float(feet * 12 + inches)

    # reject date-like tokens such as 05/31 or May/31
    if re.match(r"^\d{1,2}[-/]\d{1,2}$", ht):
        return None
    if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/]\d{1,2}$", ht, flags=re.IGNORECASE):
        return None
    return None



# ---- CSV IO ----
def safe_read_csv(path: str, *, encoding: str = "latin1", dtype="str", parse_dates: bool = False) -> pd.DataFrame | None:
    """Read CSV with consistent defaults, returning None on failure."""
    try:
        return pd.read_csv(path, encoding=encoding, dtype=dtype, parse_dates=parse_dates)
    except Exception:
        return None





# ---- DataFrame utilities ----
def reorder_after(df: pd.DataFrame, move_col: str, after_col: str) -> pd.DataFrame:
    """Return df with `move_col` placed immediately after `after_col` (if both exist)."""
    if move_col not in df.columns or after_col not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove(move_col)
    idx = cols.index(after_col)
    cols = cols[: idx + 1] + [move_col] + cols[idx + 1 :]
    return df[cols]

def filter_positions(df: pd.DataFrame, positions: Sequence[str]) -> pd.DataFrame:
    """Filter df to rows where Pos is in positions, if column exists."""
    if "Pos" in df.columns:
        return df[df["Pos"].isin(list(positions))].copy()
    return df

def parse_drafted_column(df: pd.DataFrame, col: str = "Drafted (tm/rnd/yr)") -> pd.DataFrame:
    """Split drafted column into Draft_Team/Round/Pick if present."""
    if col not in df.columns:
        return df
    split = df[col].str.extract(
        r"^(.*?)\s*/\s*(\d+(?:st|nd|rd|th))\s*/\s*(\d+(?:st|nd|rd|th)\s+pick)"
    )
    df["Draft_Team"] = split[0].str.strip()
    df["Draft_Round"] = split[1].str.strip()
    df["Draft_Pick"]  = split[2].str.strip()
    return df.drop(columns=[col])

def group_to_player_dict(df: pd.DataFrame, player_col: str = "player") -> Dict[str, pd.DataFrame]:
    """Group by player_col -> dict of player: DataFrame with index reset."""
    return {name: g.reset_index(drop=True) for name, g in df.groupby(player_col)}

# ---- Log ----

def log(message: str, level: str = "info", verbose: bool = True, stream=sys.stdout):
    """
    Tiny logger for consistent printing.
    - message: the text to log
    - level: "info", "warn", "error", "success"
    - verbose: if False, suppress all logs
    - stream: defaults to stdout (use sys.stderr for errors if needed)
    """
    if not verbose:
        return

    prefix_map = {
        "info": "ℹ️ ",
        "warn": "⚠️ ",
        "error": "❌ ",
        "success": "✅ ",
    }
    prefix = prefix_map.get(level.lower(), "")
    print(f"{prefix}{message}", file=stream)








    