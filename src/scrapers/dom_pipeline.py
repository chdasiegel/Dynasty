#!/usr/bin/env python3
# =============================================================================
# dom_pipeline.py
# =============================================================================
# HOW TO RUN (Terminal)
# ---------------------
# export BEARER_TOKEN="your_cfbd_api_key"
# python dom_pipeline.py --start 2000 --end 2025 --positions WR,TE,RB --skip-master
#
# HOW TO RUN (Jupyter from Dynasty/notebooks)
# -------------------------------------------
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
# from scrapers import dom_pipeline as dp
#
# os.environ["BEARER_TOKEN"] = "your_cfbd_api_key_here"
#
# # Example 1: Run all positions (WR, TE, RB, QB)
# df = dp.run_dom_pipeline(start=2000, end=2025)
#
# # Example 2: WR + TE only
# df = dp.run_dom_pipeline(start=2010, end=2024, positions=["WR", "TE"])
#
# # Example 3: RB/QB only, skipping master CSV
# df = dp.run_dom_pipeline(start=2015, end=2024, positions=["RB", "QB"], skip_master=True)
#
# df.head()
#
# Output files automatically save to Dynasty/data/scraper/
# =============================================================================

import os
import time
import argparse
import pandas as pd
import cfbd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Dynamically resolve the repo root (so script works anywhere inside /Dynasty)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUT_DIR = os.path.join(REPO_ROOT, "data", "scraper")

ALL_POSITIONS = {"WR", "TE", "RB", "QB"}

CONF_MULTIPLIERS = {
    "SEC": 1.00,
    "BIG TEN": 1.00,
    "BIG 10": 1.00,
    "BIG 12": 0.95,
    "ACC": 0.95,
    "PAC-12": 0.95,
    "PAC 12": 0.95,
    "AAC": 0.85,
    "MOUNTAIN WEST": 0.76,
    "SUN BELT": 0.76,
    "MAC": 0.70,
    "C-USA": 0.70,
    "CUSA": 0.70,
    "DII": 0.50,
    "OTHER": 0.60,
    "FCS": 0.60,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_api_client():
    """Return authenticated CFBD client using BEARER_TOKEN."""
    token = os.getenv("BEARER_TOKEN")
    if not token:
        raise EnvironmentError("Missing BEARER_TOKEN environment variable for CFBD API.")
    configuration = cfbd.Configuration(access_token=token)
    return cfbd.ApiClient(configuration)

def fetch_player_season_stats(year: int) -> pd.DataFrame:
    """Fetch player season stats."""
    with _get_api_client() as api_client:
        api = cfbd.StatsApi(api_client)
        data = api.get_player_season_stats(year=year)
        if not data:
            return pd.DataFrame()
        df = pd.json_normalize([d.to_dict() for d in data], sep=".")
        df["season"] = year
        return df

def fetch_team_season_stats(year: int) -> pd.DataFrame:
    """Fetch team season stats."""
    with _get_api_client() as api_client:
        api = cfbd.StatsApi(api_client)
        data = api.get_team_season_stats(year=year)
        if not data:
            return pd.DataFrame()
        df = pd.json_normalize([d.to_dict() for d in data], sep=".")
        df["season"] = year
        return df

def flatten_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten CFBD team season stats JSON into wide format for passing + rushing."""
    rows = []
    for _, r in df.iterrows():
        base = {"team": r.get("team"), "conference": r.get("conference"), "season": r.get("season")}
        cats = r.get("categories", [])
        if isinstance(cats, list):
            for cat in cats:
                cname = cat.get("name", "")
                for st in cat.get("stats", []):
                    key = f"{cname}.{st.get('stat')}"
                    base[key] = st.get("value")
        rows.append(base)
    dfw = pd.DataFrame(rows)
    dfw = dfw.rename(columns={
        "passing.passingYards": "team_receiving_yards",
        "passing.passingTds": "team_receiving_tds",
        "rushing.rushingYards": "team_rushing_yards",
        "rushing.rushingTds": "team_rushing_tds",
    })
    keep = ["season", "team", "conference", "team_receiving_yards",
            "team_receiving_tds", "team_rushing_yards", "team_rushing_tds"]
    return dfw[keep]

def compute_dom_metrics(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Compute DOM and RDOM metrics."""
    keep_cols = [
        "player", "team", "season", "conference", "position",
        "receivingYards", "receivingTds", "rushingYards", "rushingTds"
    ]
    df = players[keep_cols].copy()
    df = df.rename(columns={
        "receivingYards": "rec_yards",
        "receivingTds": "rec_tds",
        "rushingYards": "rush_yards",
        "rushingTds": "rush_tds",
        "position": "pos"
    })

    merged = pd.merge(df, teams, on=["team", "season", "conference"], how="left")

    # Receiving DOM
    merged["rec_yards_share"] = merged["rec_yards"] / merged["team_receiving_yards"]
    merged["rec_tds_share"] = merged["rec_tds"] / merged["team_receiving_tds"]
    merged["DOM"] = 0.5 * (merged["rec_yards_share"].fillna(0) + merged["rec_tds_share"].fillna(0))

    # Rushing DOM
    merged["rush_yards_share"] = merged["rush_yards"] / merged["team_rushing_yards"]
    merged["rush_tds_share"] = merged["rush_tds"] / merged["team_rushing_tds"]
    merged["RDOM"] = 0.5 * (merged["rush_yards_share"].fillna(0) + merged["rush_tds_share"].fillna(0))

    # Conference multiplier
    merged["conf_upper"] = merged["conference"].astype(str).str.upper().str.strip()
    merged["multiplier"] = merged["conf_upper"].map(CONF_MULTIPLIERS).fillna(0.6)
    merged["DOM+"] = merged["DOM"] * merged["multiplier"]
    merged["RDOM+"] = merged["RDOM"] * merged["multiplier"]

    return merged

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------
def run_dom_pipeline(start: int = 2000, end: int = 2025, positions=None, skip_master=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    if positions is None:
        positions = {"WR", "TE", "RB", "QB"}
    else:
        positions = {p.strip().upper() for p in positions}

    player_frames, team_frames = [], []

    for yr in range(start, end + 1):
        try:
            p = fetch_player_season_stats(yr)
            t = fetch_team_season_stats(yr)
            if not p.empty:
                player_frames.append(p)
            if not t.empty:
                team_frames.append(t)
            time.sleep(0.25)
        except Exception as e:
            print(f"[WARN] {yr}: {e}")

    if not player_frames or not team_frames:
        raise SystemExit("No data retrieved — check CFBD API key or network.")

    players = pd.concat(player_frames, ignore_index=True)
    teams = pd.concat(team_frames, ignore_index=True)
    team_flat = flatten_team_stats(teams)

    players["position"] = players["position"].astype(str).str.upper().str.strip()
    players = players[players["position"].isin({"WR", "TE", "RB", "QB"})].copy()

    dom_df = compute_dom_metrics(players, team_flat)

    # Filter to requested positions
    dom_df = dom_df[dom_df["pos"].isin(positions)]

    all_subframes = []
    for pos in positions:
        sub = dom_df[dom_df["pos"] == pos].copy()

        # Receiving DOM
        if pos in {"WR", "TE", "RB"}:
            out_dom = os.path.join(OUT_DIR, f"{pos}_DOM_{start}_{end}.csv")
            sub_dom = sub[[
                "player","team","season","conference","pos",
                "rec_yards","rec_tds","team_receiving_yards","team_receiving_tds",
                "rec_yards_share","rec_tds_share","DOM","DOM+"
            ]]
            sub_dom.to_csv(out_dom, index=False)
            print(f"[✓] Wrote {len(sub_dom)} rows to {out_dom}")
            all_subframes.append(sub_dom.assign(metric="DOM"))

        # Rushing DOM
        if pos in {"RB", "QB"}:
            out_rdom = os.path.join(OUT_DIR, f"{pos}_RDOM_{start}_{end}.csv")
            sub_rdom = sub[[
                "player","team","season","conference","pos",
                "rush_yards","rush_tds","team_rushing_yards","team_rushing_tds",
                "rush_yards_share","rush_tds_share","RDOM","RDOM+"
            ]]
            sub_rdom.to_csv(out_rdom, index=False)
            print(f"[✓] Wrote {len(sub_rdom)} rows to {out_rdom}")
            all_subframes.append(sub_rdom.assign(metric="RDOM"))

    # Save master combined CSV
    if not skip_master and all_subframes:
        master = pd.concat(all_subframes, ignore_index=True)
        out_master = os.path.join(OUT_DIR, f"DOM_master_{start}_{end}.csv")
        master.to_csv(out_master, index=False)
        print(f"[★] Master file written: {out_master}")
    elif skip_master:
        print("[ℹ] Skipped master file generation per user flag.")

    print("✅ DOM/RDOM pipeline completed successfully.")
    return dom_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument(
        "--positions",
        type=str,
        default="WR,TE,RB,QB",
        help="Comma-separated list of positions to include (default: WR,TE,RB,QB)"
    )
    ap.add_argument(
        "--skip-master",
        action="store_true",
        help="Skip saving master CSV file (default: False)"
    )
    args = ap.parse_args()
    pos_list = [p.strip() for p in args.positions.split(",") if p.strip()]
    run_dom_pipeline(start=args.start, end=args.end, positions=pos_list, skip_master=args.skip_master)


if __name__ == "__main__":
    main()
