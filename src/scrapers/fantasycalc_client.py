# fantasycalc_client.py
"""
FantasyCalc client + CLI.

Depends on:
- requests
- pandas

Usage:
    from src.scrapers.fantasycalc_client import (
        get_player_value, search_players,
        get_rankings_df, save_current_rankings
    )

    # Look up one player
    row = get_player_value("Breece Hall")
    print(row)

    # Search for possible name matches
    print(search_players("Harrison"))

    # Get full rankings as a DataFrame
    df = get_rankings_df(dynasty=True, num_qbs=1, teams=12, ppr=1.0)
    print(df.head())

    # Save CSV snapshot(s) to Market_Value/
    path = save_current_rankings(dynasty=True, num_qbs=1, teams=12, ppr=1.0)
    print("Saved:", path)


Usage (CLI):
    python fantasycalc_client.py "Breece Hall"
    python fantasycalc_client.py "CJ Stroud" --qbs 2
    python fantasycalc_client.py "Justin Jefferson" --redraft
    python fantasycalc_client.py --search "Harrison"
    python fantasycalc_client.py --save
"""

from __future__ import annotations

import time
import json
import pathlib
import difflib
from typing import Dict, Any, List, Optional

import requests
import pandas as pd

# ---------- Config ----------
DEFAULT_BASE = "https://api.fantasycalc.com"
CACHE_DIR = pathlib.Path("Market_Value")
CACHE_PATH = CACHE_DIR / ".fc_cache.json"
CACHE_TTL_SECS = 60 * 60 * 2  # 2 hours
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _endpoint(dynasty: bool, num_qbs: int, teams: int, ppr: float) -> str:
    return (
        f"{DEFAULT_BASE}/values/current"
        f"?isDynasty={'true' if dynasty else 'false'}"
        f"&numQbs={num_qbs}"
        f"&numTeams={teams}"
        f"&ppr={ppr}"
    )


def _load_cache() -> Optional[Dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return None
    return None


def _save_cache(payload: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(payload))
    except Exception:
        pass


def _fetch_payload(dynasty: bool, num_qbs: int, teams: int, ppr: float) -> List[Dict[str, Any]]:
    cache = _load_cache()
    key = _endpoint(dynasty, num_qbs, teams, ppr)
    now = time.time()
    if cache and cache.get("key") == key and (now - cache.get("ts", 0) < CACHE_TTL_SECS):
        return cache["data"]

    resp = requests.get(key, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    _save_cache({"key": key, "ts": now, "data": data})
    return data


def _build_name_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in rows:
        name = (r.get("player") or {}).get("name")
        if isinstance(name, str):
            idx[name.lower()] = r
    return idx


def search_players(query: str, *, dynasty: bool = True, num_qbs: int = 1, teams: int = 12, ppr: float = 1.0,
                   limit: int = 10) -> List[str]:
    rows = _fetch_payload(dynasty, num_qbs, teams, ppr)
    names = [(r.get("player") or {}).get("name") for r in rows if (r.get("player") or {}).get("name")]
    names = [n for n in names if isinstance(n, str)]
    q = query.lower()
    contains = [n for n in names if q in n.lower()]
    if contains:
        return contains[:limit]
    return difflib.get_close_matches(query, names, n=limit, cutoff=0.6)


def get_player_value(
    name: str,
    *,
    dynasty: bool = True,
    num_qbs: int = 1,
    teams: int = 12,
    ppr: float = 1.0,
) -> Optional[Dict[str, Any]]:
    rows = _fetch_payload(dynasty, num_qbs, teams, ppr)
    idx = _build_name_index(rows)
    row = idx.get(name.lower())
    if row is None:
        candidates = difflib.get_close_matches(name.lower(), list(idx.keys()), n=1, cutoff=0.7)
        if candidates:
            row = idx[candidates[0]]
    return row


def _settings_slug(dynasty: bool, num_qbs: int, teams: int, ppr: float) -> str:
    mode = "dynasty" if dynasty else "redraft"
    ppr_str = str(ppr).rstrip("0").rstrip(".") if isinstance(ppr, float) else str(ppr)
    return f"{mode}_{num_qbs}QB_{teams}_PPR{ppr_str}"


def get_rankings_df(*, dynasty: bool = True, num_qbs: int = 1, teams: int = 12, ppr: float = 1.0) -> pd.DataFrame:
    rows = _fetch_payload(dynasty, num_qbs, teams, ppr)
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows, sep="_")
    colmap = {
        "player.name": "name",
        "player.position": "position",
        "player.maybeTeam": "team",
        "player.id": "player_id",
    }
    # json_normalize with sep="_" yields keys like "player_name"
    for k in list(colmap.keys()):
        underscore = k.replace(".", "_")
        if underscore in df.columns:
            df.rename(columns={underscore: colmap[k]}, inplace=True)

    preferred = ["name", "position", "team", "player_id", "value", "overallRank", "positionRank", "trend30Day"]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        df = df[cols]

    if "overallRank" in df.columns:
        df = df.sort_values("overallRank", ascending=True, kind="stable")
    elif "value" in df.columns:
        df = df.sort_values("value", ascending=False, kind="stable")

    df.reset_index(drop=True, inplace=True)
    return df


def save_current_rankings(*, dynasty: bool = True, num_qbs: int = 1, teams: int = 12, ppr: float = 1.0,
                          timestamped: bool = True) -> pathlib.Path:
    df = get_rankings_df(dynasty=dynasty, num_qbs=num_qbs, teams=teams, ppr=ppr)
    slug = _settings_slug(dynasty, num_qbs, teams, ppr)

    latest_path = CACHE_DIR / "latest.csv"
    df.to_csv(latest_path, index=False)

    if timestamped:
        ts = time.strftime("%Y-%m-%d_%H%M")
        snap_path = CACHE_DIR / f"{slug}_{ts}.csv"
        df.to_csv(snap_path, index=False)
        return snap_path

    return latest_path


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FantasyCalc CLI")
    parser.add_argument("player", nargs="?", help="Player name to look up (optional)")
    parser.add_argument("--dynasty", action="store_true", default=True, help="Dynasty mode (default true)")
    parser.add_argument("--redraft", action="store_true", help="Use redraft instead of dynasty")
    parser.add_argument("--qbs", type=int, default=1, help="Number of starting QBs (1 or 2)")
    parser.add_argument("--teams", type=int, default=12, help="League size (default 12)")
    parser.add_argument("--ppr", type=float, default=1.0, help="PPR (default 1.0)")
    parser.add_argument("--save", action="store_true", help="Save full rankings CSV into Market_Value/")
    parser.add_argument("--search", help="Fuzzy search names (prints matches)")

    args = parser.parse_args()
    dynasty_mode = False if args.redraft else args.dynasty

    if args.search:
        print("\n".join(search_players(args.search, dynasty=dynasty_mode, num_qbs=args.qbs, teams=args.teams, ppr=args.ppr)))
    if args.player:
        row = get_player_value(args.player, dynasty=dynasty_mode, num_qbs=args.qbs, teams=args.teams, ppr=args.ppr)
        if row:
            print({
                "name": row["player"]["name"],
                "team": row["player"].get("maybeTeam"),
                "pos": row["player"]["position"],
                "value": row["value"],
                "overallRank": row["overallRank"],
                "positionRank": row["positionRank"],
                "trend30Day": row.get("trend30Day"),
            })
        else:
            print("❌ Player not found")
    if args.save:
        out = save_current_rankings(dynasty=dynasty_mode, num_qbs=args.qbs, teams=args.teams, ppr=args.ppr, timestamped=True)
        print("✅ Saved snapshot:", out)
