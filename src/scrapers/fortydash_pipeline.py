#!/usr/bin/env python3
# =============================================================================
# fortydash_pipeline.py
# =============================================================================
# HOW TO RUN (Terminal)
# ---------------------

# 1) Run the pipeline (2000–2025). Output defaults to Dynasty/data/scraper/40yd_time_2000_2025.csv:
#    python fortydash_pipeline.py --start 2000 --end 2025
#    # (optional custom path override)
#    python fortydash_pipeline.py --start 2000 --end 2025 --out /absolute/or/relative/path.csv
#
# HOW TO RUN (Jupyter Notebook)
# -----------------------------
# !pip install pandas numpy requests lxml scikit-learn beautifulsoup4
# import fortydash_pipeline as p
# df = p.run_pipeline(start=2000, end=2025)  # writes to Dynasty/data/scraper/40yd_time_2000_2025.csv
# df.head()
#
# If this file lives inside a package (e.g., src/scrapers/fortydash_pipeline.py):
# from src.scrapers import fortydash_pipeline as p
# df = p.run_pipeline(start=2000, end=2025)
#
# What this pipeline does
# -----------------------
# • Scrapes Combine 40-yard dash results from Pro-Football-Reference (PFR) for each year.
# • Supplements with NFL.com Combine Tracker (recent years) and BNB Football Pro Day pages.
# • Merges sources with priority: Combine (PFR/NFL) > Pro Day (BNB).
# • Imputes missing 40 times via ElasticNet regression, then position-era medians.
# • Adds flags: is_estimate, estimate_method, confidence.
# • Filters final output to positions: QB, RB, WR, TE only.
# =============================================================================

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; 40yd-pipeline/1.0)"}
TARGET_POS = {"QB", "RB", "WR", "TE"}

# Pro-Football-Reference combine page template
PFR_YEAR_URL = "https://www.pro-football-reference.com/draft/{year}-combine.htm"

# NFL.com tracker (best-effort: site is JS-heavy; we heuristically parse visible text)
NFL_TRACKER_40_ALL = (
    "https://www.nfl.com/combine/tracker/live-results/40-yard-dash/all-positions/all-colleges/"
)
NFL_TRACKER_TOP_40_BY_YEAR = (
    "https://www.nfl.com/combine/tracker/top-performers/40-yard-dash/all-positions/{year}/"
)

# BNB Football Pro Day pages (pattern used for recent years)
BNB_PRODAY_URL_TMPL = "https://bnbfootball.com/complete-pro-day-results-{year}/"


# ---------------------------------------------------------------------------------
# Repo-aware output path helpers (works in both scripts & notebooks)
# ---------------------------------------------------------------------------------
def _find_dynasty_root() -> Path:
    """
    Try to locate the 'Dynasty' repo root, even when running inside a notebook
    where __file__ may not exist. Falls back to CWD if 'Dynasty' not found.
    """
    try:
        here = Path(__file__).resolve()
    except NameError:
        # In Jupyter, __file__ is not defined; use the notebook's CWD
        here = Path.cwd().resolve()

    # If we're already inside a folder named Dynasty, that's our root
    if here.name == "Dynasty":
        return here

    # Search upward for a directory named 'Dynasty'
    for p in [here] + list(here.parents):
        if p.name == "Dynasty":
            return p

    # Fallback: assume current working directory is the project root
    return Path.cwd().resolve()


_DYNASTY_ROOT = _find_dynasty_root()
_DEFAULT_OUT_DIR = _DYNASTY_ROOT / "data" / "scraper"
_DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def _normalize_name(name: str) -> str:
    """Basic name normalizer to improve cross-source matching."""
    if not isinstance(name, str):
        return ""
    # remove accents, lower, strip, collapse spaces, strip punctuation other than hyphen/space
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-\.']", " ", s)  # keep word chars, spaces, hyphen, dot, apostrophe
    s = re.sub(r"\s+", " ", s)
    return s


def _height_to_inches(ht: str | float | None) -> Optional[float]:
    if ht is None or (isinstance(ht, float) and np.isnan(ht)):
        return None
    if isinstance(ht, (int, float)):
        return float(ht)
    s = str(ht).strip()
    # PFR uses e.g. 6-1 for 6 ft 1 in
    m = re.match(r"^\s*(\d+)\s*[-']\s*(\d+)\s*\"?\s*$", s)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    # sometimes "6-01" or "601" style
    m2 = re.match(r"^\s*(\d)\s*[-]?\s*(\d{2})\s*$", s)
    if m2:
        return int(m2.group(1)) * 12 + int(m2.group(2))
    return None


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            val = float(x)
            return val if np.isfinite(val) else None
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "n/a", "nan", "none", "--"}:
            return None
        # Remove any non-numeric except dot
        s = re.sub(r"[^0-9.]+", "", s)
        return float(s) if s else None
    except Exception:
        return None


# ---------------------------------------------------------------------------------
# PFR scraping
# ---------------------------------------------------------------------------------
def fetch_pfr_year(year: int) -> pd.DataFrame:
    """Scrape a single year's PFR combine table and return normalized DataFrame."""
    url = PFR_YEAR_URL.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # PFR tables can be wrapped in HTML comments; un-comment any that contain tables
    for c in soup.find_all(string=lambda t: isinstance(t, (type(soup.string))) and str(t).strip().startswith("<!--")):
        txt = str(c)
        if "<table" in txt:
            frag = BeautifulSoup(txt, "lxml")
            for tbl in frag.find_all("table"):
                if soup.body:
                    soup.body.append(tbl)

    html_str = str(soup)
    df_list = pd.read_html(StringIO(html_str)) if soup else []
    if not df_list:
        return pd.DataFrame(columns=["player","year","pos","height_in","weight_lb","forty_s","source"])

    best = None
    for df in df_list:
        cols = [str(c).strip().lower() for c in df.columns]
        if any("player" in c for c in cols) and any(("40" in c and "yd" in c) or ("40_yard" in c) for c in cols):
            best = df
            break
    if best is None:
        return pd.DataFrame(columns=["player","year","pos","height_in","weight_lb","forty_s","source"])

    # Normalize columns
    best.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in best.columns]
    colmap = {
        "player": "player",
        "pos": "pos",
        "position": "pos",
        "ht": "height",
        "height": "height",
        "wt": "weight",
        "weight": "weight",
        "40yd": "forty",
        "_40yd": "forty",
        "40_yd": "forty",
        "40_yard_dash": "forty",
        "forty": "forty",
        "year": "year",
    }
    best = best.rename(columns={c: colmap.get(c, c) for c in best.columns})
    if "year" not in best.columns:
        best["year"] = year

    # Coerce types
    best["player"] = best.get("player").astype(str)
    best["pos"] = best.get("pos").astype(str).str.upper().str.strip()
    best["height_in"] = best.get("height").apply(_height_to_inches) if "height" in best.columns else np.nan
    best["weight_lb"] = best.get("weight").apply(_to_float) if "weight" in best.columns else np.nan
    forty_col = next((c for c in ["forty", "40yd", "_40yd", "40_yard_dash", "40_yd"] if c in best.columns), None)
    best["forty_s"] = best[forty_col].apply(_to_float) if forty_col else np.nan

    best["source"] = f"PFR:{year}"
    best["player_norm"] = best["player"].map(_normalize_name)
    out = best[["player", "player_norm", "year", "pos", "height_in", "weight_lb", "forty_s", "source"]].copy()
    out = out[out["player"].str.lower() != "player"]
    return out.reset_index(drop=True)


def fetch_pfr_range(start: int, end: int, sleep: float = 1.0) -> pd.DataFrame:
    frames = []
    for y in range(start, end + 1):
        try:
            dfy = fetch_pfr_year(y)
            if not dfy.empty:
                frames.append(dfy)
        except Exception as e:
            print(f"[WARN] PFR {y}: {e}")
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame(columns=["player","player_norm","year","pos","height_in","weight_lb","forty_s","source"])
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(by=["player_norm", "year", "forty_s"], inplace=True, na_position="last")
    df = df.drop_duplicates(subset=["player_norm", "year", "pos"], keep="first")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# NFL.com tracker (best-effort)
# ---------------------------------------------------------------------------------
def fetch_nfl_tracker_year(year: int) -> pd.DataFrame:
    """
    Attempts to parse NFL.com tracker pages for a year.
    Produces rows with player, year, possibly missing pos/size (to be enriched from PFR).
    """
    rows = []

    # 1) Top performers by year (static content)
    url_year = NFL_TRACKER_TOP_40_BY_YEAR.format(year=year)
    try:
        r = requests.get(url_year, headers=HEADERS, timeout=20)
        if r.ok:
            soup = BeautifulSoup(r.text, "lxml")
            for node in soup.find_all(text=re.compile(r"\bSeconds\b", re.I)):
                t = node.parent.get_text(" ", strip=True)
                m = re.search(r"([A-Z][a-zA-Z\.\-\' ]+)\s+.*?\b(\d\.\d{2})\s+Seconds", t)
                if m:
                    player = m.group(1).strip()
                    forty = float(m.group(2))
                    rows.append({
                        "player": player,
                        "player_norm": _normalize_name(player),
                        "year": year,
                        "pos": None,
                        "height_in": np.nan,
                        "weight_lb": np.nan,
                        "forty_s": forty,
                        "source": f"NFL:{year}:top"
                    })
    except Exception as e:
        print(f"[WARN] NFL top {year}: {e}")

    # 2) Live results (often current year)
    try:
        r2 = requests.get(NFL_TRACKER_40_ALL, headers=HEADERS, timeout=20)
        if r2.ok:
            soup2 = BeautifulSoup(r2.text, "lxml")
            for node in soup2.find_all(text=re.compile(r"\bSeconds\b", re.I)):
                t = node.parent.get_text(" ", strip=True)
                m2 = re.search(r"([A-Z][a-zA-Z\.\-\' ]+)\s+\((\d{4})\).*?\b(\d\.\d{2})\s+Seconds", t)
                if m2:
                    name = m2.group(1).strip()
                    yr = int(m2.group(2))
                    if yr == year:
                        forty = float(m2.group(3))
                        rows.append({
                            "player": name,
                            "player_norm": _normalize_name(name),
                            "year": year,
                            "pos": None,
                            "height_in": np.nan,
                            "weight_lb": np.nan,
                            "forty_s": forty,
                            "source": f"NFL:{year}:live"
                        })
    except Exception as e:
        print(f"[WARN] NFL live: {e}")

    if not rows:
        return pd.DataFrame(columns=["player","player_norm","year","pos","height_in","weight_lb","forty_s","source"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["player_norm", "year"])
    return df


# ---------------------------------------------------------------------------------
# Pro Day (BNB Football)
# ---------------------------------------------------------------------------------
def fetch_bnb_proday(year: int) -> pd.DataFrame:
    # Attempt years in a reasonable window (pattern known for recent years)
    url = BNB_PRODAY_URL_TMPL.format(year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if not r.ok:
            return pd.DataFrame(columns=["player","player_norm","year","pos","height_in","weight_lb","forty_s","source"])
        tables = pd.read_html(r.text)
        frames = []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("40" in c for c in cols) and any(("name" in c) or ("player" in c) for c in cols):
                df = t.copy()
                df.columns = [re.sub(r"\W+","_",str(c).strip().lower()).strip("_") for c in df.columns]
                name_col = "name" if "name" in df.columns else "player" if "player" in df.columns else None
                forty_col = next((c for c in df.columns if c.startswith("40")), None)
                pos_col = "position" if "position" in df.columns else "pos" if "pos" in df.columns else None
                if name_col and forty_col:
                    out = pd.DataFrame({
                        "player": df[name_col].astype(str),
                        "player_norm": df[name_col].astype(str).map(_normalize_name),
                        "year": year,
                        "pos": df[pos_col].astype(str).str.upper().str.strip() if pos_col else None,
                        "height_in": np.nan,
                        "weight_lb": np.nan,
                        "forty_s": df[forty_col].apply(_to_float),
                        "source": f"BNB_ProDay:{year}"
                    })
                    frames.append(out)
        if frames:
            dfo = pd.concat(frames, ignore_index=True)
            dfo = dfo.dropna(subset=["player"]).drop_duplicates(subset=["player_norm", "year"])
            return dfo
    except Exception as e:
        print(f"[WARN] BNB {year}: {e}")
    return pd.DataFrame(columns=["player","player_norm","year","pos","height_in","weight_lb","forty_s","source"])


# ---------------------------------------------------------------------------------
# Imputation & Merge
# ---------------------------------------------------------------------------------
def fit_imputer(df: pd.DataFrame) -> Pipeline:
    train = df.dropna(subset=["forty_s"]).copy()
    if train.empty:
        raise ValueError("No rows with measured forty_s to train on.")
    X = train[["year", "height_in", "weight_lb", "pos"]]
    y = train["forty_s"].values
    pre = ColumnTransformer([
        ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         ["year", "height_in", "weight_lb"]),
        ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         ["pos"]),
    ])
    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], max_iter=5000, cv=5, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Overloaded: build-and-apply model internally for simplicity
    train = df.dropna(subset=["forty_s"]).copy()
    out = df.copy()
    if train.empty:
        # If absolutely no measured rows (unlikely), return df unchanged
        out["is_estimate"] = True
        out["estimate_method"] = "unknown"
        out["confidence"] = 0.55
        return out

    X_tr = train[["year", "height_in", "weight_lb", "pos"]]
    y_tr = train["forty_s"].values
    pre = ColumnTransformer([
        ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         ["year", "height_in", "weight_lb"]),
        ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         ["pos"]),
    ])
    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], max_iter=5000, cv=5, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)

    if "is_estimate" not in out.columns:
        out["is_estimate"] = False
    if "estimate_method" not in out.columns:
        out["estimate_method"] = None
    if "confidence" not in out.columns:
        out["confidence"] = np.where(out["source"].str.startswith(("PFR", "NFL")), 0.95,
                              np.where(out["source"].str.startswith("BNB_ProDay"), 0.85, 0.60))

    miss = out["forty_s"].isna()
    if miss.any():
        Xmiss = out.loc[miss, ["year", "height_in", "weight_lb", "pos"]]
        out.loc[miss, "forty_s"] = pipe.predict(Xmiss)
        out.loc[miss, "is_estimate"] = True
        out.loc[miss, "estimate_method"] = "elasticnet_regression"
        out.loc[miss, "confidence"] = 0.60  # imputed baseline

    # Fallback: position-era median for any rows still missing
    still = out["forty_s"].isna()
    if still.any():
        out["era"] = pd.cut(out["year"], bins=[1995, 2004, 2013, 2020, 2026],
                            labels=["2000s", "2005-2013", "2014-2020", "2021-2025"], include_lowest=True)
        med = out.dropna(subset=["forty_s"]).groupby(["pos", "era"])["forty_s"].median().rename("pos_era_median")
        out = out.merge(med, left_on=["pos", "era"], right_index=True, how="left")
        mask = still & out["pos_era_median"].notna()
        out.loc[mask, "forty_s"] = out.loc[mask, "pos_era_median"]
        out.loc[mask, "is_estimate"] = True
        out.loc[mask, "estimate_method"] = "pos_era_median"
        out.loc[mask, "confidence"] = 0.55
        out.drop(columns=["pos_era_median", "era"], inplace=True, errors="ignore")
    return out


def merge_sources(pfr: pd.DataFrame, nfl_df: pd.DataFrame, proday: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PFR, NFL.com, and Pro Day. Priority: Combine (PFR/NFL) > Pro Day.
    For NFL.com rows, enrich pos/height/weight by exact name+year match to PFR.
    """
    frames = []
    if not pfr.empty:
        frames.append(pfr.assign(priority=1))               # combine official
    if not nfl_df.empty:
        frames.append(nfl_df.assign(priority=1))            # combine official
    if not proday.empty:
        frames.append(proday.assign(priority=2))            # pro day

    if not frames:
        return pd.DataFrame(columns=["player","player_norm","year","pos","height_in","weight_lb","forty_s","source"])

    allrows = pd.concat(frames, ignore_index=True)

    # Enrich NFL.com rows from PFR (same year, same normalized name)
    if not nfl_df.empty and not pfr.empty:
        key_cols = ["player_norm", "year"]
        enrich_cols = ["pos", "height_in", "weight_lb"]
        enriched = allrows.merge(
            pfr[key_cols + enrich_cols].rename(columns={c: f"pfr_{c}" for c in enrich_cols}),
            on=key_cols, how="left"
        )
        # fill only if missing
        for c in enrich_cols:
            allrows[c] = np.where(
                allrows[c].isna() | (allrows[c].astype(str).isin(["None", "nan", "" ])),
                enriched[f"pfr_{c}"],
                allrows[c]
            )

    # Choose best per (player_norm, year, pos) by priority, preferring non-null forty first
    allrows["null_forty"] = allrows["forty_s"].isna()
    allrows.sort_values(by=["player_norm", "year", "pos", "priority", "null_forty"], inplace=True)
    best = allrows.drop_duplicates(subset=["player_norm", "year", "pos"], keep="first").copy()
    best.drop(columns=["priority", "null_forty"], inplace=True, errors="ignore")

    # Initialize confidence (official combine high, pro day medium)
    best["confidence"] = np.where(best["source"].str.startswith(("PFR", "NFL")), 0.95,
                           np.where(best["source"].str.startswith("BNB_ProDay"), 0.85, 0.60))
    best["is_estimate"] = False
    best["estimate_method"] = None
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------------
def run_pipeline(start: int = 2000, end: int = 2025, out_path: str | None = None,
                 sleep_pfr: float = 1.0) -> pd.DataFrame:
    # Base PFR
    pfr = fetch_pfr_range(start, end, sleep=sleep_pfr)

    # NFL tracker (recent years most useful)
    nfl_frames = []
    for yr in range(max(2021, start), end + 1):
        try:
            dfy = fetch_nfl_tracker_year(yr)
            if not dfy.empty:
                nfl_frames.append(dfy)
        except Exception as e:
            print(f"[WARN] NFL tracker {yr}: {e}")
        time.sleep(0.4)
    nfl_df = pd.concat(nfl_frames, ignore_index=True) if nfl_frames else pd.DataFrame(
        columns=["player", "player_norm", "year", "pos", "height_in", "weight_lb", "forty_s", "source"]
    )

    # Pro Day (attempt a wider range; pattern likely exists for recent years)
    proday_frames = []
    for yr in range(start, end + 1):
        try:
            dpp = fetch_bnb_proday(yr)
            if not dpp.empty:
                proday_frames.append(dpp)
        except Exception as e:
            print(f"[WARN] Pro Day {yr}: {e}")
        time.sleep(0.3)
    proday = pd.concat(proday_frames, ignore_index=True) if proday_frames else pd.DataFrame(
        columns=["player", "player_norm", "year", "pos", "height_in", "weight_lb", "forty_s", "source"]
    )

    # Merge & impute
    merged = merge_sources(pfr, nfl_df, proday)
    if merged.empty:
        raise SystemExit("No data collected; check network or site changes.")

    final = impute_missing(merged)

    # Filter to target positions after enrichment/imputation
    final["pos"] = final["pos"].astype(str).str.upper().str.strip()
    final = final[final["pos"].isin(TARGET_POS)].copy()

    # Reorder
    cols = ["player", "year", "pos", "height_in", "weight_lb", "forty_s",
            "is_estimate", "estimate_method", "confidence", "source"]
    final = final[cols].sort_values(["year", "pos", "player"]).reset_index(drop=True)

    # Determine output path
    if out_path is None:
        filename = f"40yd_time_{start}_{end}.csv"
        out_path = _DEFAULT_OUT_DIR / filename
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"Wrote {len(final):,} rows to {out_path}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional custom path. Default is Dynasty/data/scraper/40yd_time_{start}_{end}.csv",
    )
    args = ap.parse_args()
    run_pipeline(start=args.start, end=args.end, out_path=args.out)


if __name__ == "__main__":
    main()
