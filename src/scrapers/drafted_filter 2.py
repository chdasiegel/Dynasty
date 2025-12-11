#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add a 'drafted' flag (and draft metadata) to a player list by scraping Pro-Football-Reference.

Input  : CSV with 'player_name' or 'player'
Output : SAME rows + drafted metadata (drafted, draft_year, draft_team, draft_round, draft_pick)

NEW:
 - Separate first-name and last-name similarity thresholds.
 - Keep all rows (no dropping).
 - Ignore candidates drafted before a given year (e.g. --min-draft-year 2000).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ------------------ Tunables & Paths ------------------ #

BASE_SLEEP   = 1.2
JITTER       = (0.4, 0.9)
BACKOFF_MULT = 2.0
MAX_BACKOFF  = 60

CACHE_PATH    = Path("cache/pfr_draft_cache.json")
PFR_INDEX_DIR = Path("cache/pfr_index")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

for p in (CACHE_PATH.parent, PFR_INDEX_DIR):
    p.mkdir(parents=True, exist_ok=True)

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

# ------------------ Cache ------------------ #

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

# ------------------ Name Cleaning ------------------ #

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_for_tokens(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(r"[.,'’`\"]", " ", s)
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_name(name: str):
    s = normalize_for_tokens(name)
    if not s:
        return None, None, []
    parts = s.split()
    if parts and parts[-1] in _SUFFIXES and len(parts) >= 2:
        parts = parts[:-1]
    if not parts:
        return None, None, []
    first = parts[0]
    last  = parts[-1] if len(parts) > 1 else None
    rest  = parts[1:-1]
    return first, last, rest

def last_name_initial(full_name: str) -> Optional[str]:
    _, last, _ = tokenize_name(full_name)
    return last[0].upper() if last else None

def clean_player_name(name: str) -> str:
    return str(name).replace("*", "").strip()

def normalize_fullname(name: str) -> str:
    first, last, rest = tokenize_name(name)
    parts = [p for p in [first] + rest + [last] if p]
    return " ".join(parts) if parts else ""

# ------------------ Name Similarity ------------------ #

def first_name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def last_name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def full_name_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def candidate_score(target: str, cand: str):
    tf, tl, _ = tokenize_name(target)
    cf, cl, _ = tokenize_name(cand)

    if not tl or not cl:
        return 0.0, 0.0, 0.0, False

    f_sim = first_name_similarity(tf, cf)
    l_sim = last_name_similarity(tl, cl)
    full_sim = full_name_ratio(normalize_fullname(target), normalize_fullname(cand))

    # Hard reject if last names mismatch heavily
    if l_sim < 0.60:
        return 0.0, f_sim, l_sim, False

    score = 0.60 * l_sim + 0.25 * f_sim + 0.15 * full_sim
    return score, f_sim, l_sim, True

# ------------------ HTTP ------------------ #

def sleep_with_jitter(base=BASE_SLEEP):
    time.sleep(base + random.uniform(*JITTER))

def req_with_backoff(url: str, debug: bool = False):
    delay = BASE_SLEEP
    while True:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                delay = min(MAX_BACKOFF, delay * BACKOFF_MULT)
                if debug:
                    print(f"[429] Backing off {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        except requests.RequestException:
            delay = min(MAX_BACKOFF, delay * BACKOFF_MULT)
            time.sleep(delay)

# ------------------ PFR Index ------------------ #

def load_pfr_index_letter(letter: str, debug: bool = False) -> BeautifulSoup:
    letter = letter.upper()
    file = PFR_INDEX_DIR / f"{letter}.html"
    if file.exists():
        return BeautifulSoup(file.read_text(), "lxml")
    url = f"https://www.pro-football-reference.com/players/{letter}/"
    sleep_with_jitter()
    r = req_with_backoff(url, debug)
    file.write_text(r.text)
    return BeautifulSoup(r.text, "lxml")

def pfr_candidates_by_score(
    target_name: str,
    debug: bool,
    first_thresh: float,
    last_thresh: float,
):
    ini = last_name_initial(target_name)
    if not ini:
        return []

    soup = load_pfr_index_letter(ini, debug)
    out = []

    # let first-name floor be a bit looser than the main threshold
    loose_first_floor = max(0.50, first_thresh * 0.6)

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not re.match(r"^/players/[A-Z]/[^/]+\.htm$", href):
            continue

        cand_text = (a.get_text() or "").strip()
        score, f_sim, l_sim, ok = candidate_score(target_name, cand_text)
        if not ok:
            continue
        if l_sim < last_thresh:
            continue
        if f_sim < loose_first_floor:
            continue
        if score < 0.60:
            continue

        url = "https://www.pro-football-reference.com" + href
        out.append((score, url, cand_text, f_sim, l_sim))

    out.sort(key=lambda x: x[0], reverse=True)
    return out

# ------------------ Draft Parsing ------------------ #

@dataclass
class DraftInfo:
    drafted: bool
    year: Optional[int] = None
    team: Optional[str] = None
    round: Optional[int] = None
    pick: Optional[int] = None
    url: Optional[str] = None

def parse_draft_block(text: str) -> DraftInfo:
    txt = text.strip()
    if "Undrafted" in txt:
        return DraftInfo(drafted=False)

    team = None
    year = None
    rnd = None
    pick = None

    m_year = re.search(r"(\d{4})\s+NFL Draft", txt)
    if m_year:
        year = int(m_year.group(1))

    m_team = re.search(r"Draft:\s*([^,]+?)\s+in the", txt)
    if m_team:
        team = m_team.group(1).strip()

    m_round = re.search(r"in the\s+(\d+)(?:st|nd|rd|th)\s+round", txt)
    if m_round:
        rnd = int(m_round.group(1))

    m_pick = re.search(r"\((\d+)(?:st|nd|rd|th)\s+overall\)", txt)
    if m_pick:
        pick = int(m_pick.group(1))

    if year is None and team is None:
        return DraftInfo(drafted=False)

    return DraftInfo(True, year, team, rnd, pick)

def extract_draft_info_from_page(soup: BeautifulSoup) -> DraftInfo:
    for lab in soup.select("strong"):
        if lab.get_text(strip=True).rstrip(":").lower() == "draft":
            return parse_draft_block(lab.parent.get_text(" ", strip=True))

    text = soup.get_text(" ", strip=True)
    m = re.search(r"Draft:\s*(.+?)(?:Born:|College:|High School:|$)", text)
    return parse_draft_block(m.group(0)) if m else DraftInfo(False)

# ------------------ Candidate Evaluation ------------------ #

def try_candidate_url_for_draft(url: str, min_draft_year: int, debug: bool = False) -> DraftInfo:
    sleep_with_jitter()
    r = req_with_backoff(url, debug)
    soup = BeautifulSoup(r.text, "lxml")
    info = extract_draft_info_from_page(soup)
    info.url = url

    # Reject players drafted before cutoff year
    if info.drafted and info.year is not None and info.year < min_draft_year:
        if debug:
            print(f"    ❌ Rejected: drafted in {info.year} < {min_draft_year}")
        return DraftInfo(drafted=False)

    return info

def fetch_draft_info_for_name(
    name: str,
    first_thresh: float,
    last_thresh: float,
    min_draft_year: int,
    debug: bool = False
) -> DraftInfo:

    cands = pfr_candidates_by_score(name, debug, first_thresh, last_thresh)

    for score, url, cand, f_sim, l_sim in cands:
        if debug:
            print(f"    Trying {cand} (score={score:.2f}, first={f_sim:.2f}, last={l_sim:.2f})")
        info = try_candidate_url_for_draft(url, min_draft_year, debug)
        if info.drafted:
            return info

    return DraftInfo(drafted=False)

# ------------------ CLI MAIN ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="Input CSV with 'player' or 'player_name'.")
    ap.add_argument("--out", dest="outp", required=True, help="Output CSV with drafted flags added.")

    ap.add_argument("--first-name-thresh", type=float, default=0.85,
                    help="Minimum first-name similarity (default 0.85)")
    ap.add_argument("--last-name-thresh", type=float, default=0.97,
                    help="Minimum last-name similarity (default 0.97)")
    ap.add_argument("--min-draft-year", type=int, default=2000,
                    help="Ignore any candidate drafted before this year (default 2000)")

    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of unique players to process (for testing).")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Robust name column detection
    if "player_name" in df.columns:
        name_col = "player_name"
    elif "player" in df.columns:
        name_col = "player"
    else:
        raise SystemExit(f"Input CSV must contain 'player' or 'player_name'. Columns: {df.columns.tolist()}")

    df["player_name_clean"] = df[name_col].map(clean_player_name)
    names = (
        df["player_name_clean"]
        .dropna()
        .astype(str)
        .pipe(lambda s: s[s.ne("")])
        .drop_duplicates()
        .tolist()
    )

    if args.limit:
        names = names[:args.limit]

    cache = load_cache()
    results: List[Dict] = []

    print(f"Processing {len(names)} unique names...")

    for i, nm in enumerate(names, 1):
        if args.debug:
            print(f"[{i}/{len(names)}] {nm}")

        if nm in cache:
            info = DraftInfo(**cache[nm])
        else:
            info = fetch_draft_info_for_name(
                nm,
                first_thresh=args.first_name_thresh,
                last_thresh=args.last_name_thresh,
                min_draft_year=args.min_draft_year,
                debug=args.debug
            )
            cache[nm] = info.__dict__
            save_cache(cache)

        results.append({
            "player_name_clean": nm,
            "drafted": info.drafted,
            "draft_year": info.year,
            "draft_team": info.team,
            "draft_round": info.round,
            "draft_pick": info.pick,
            "draft_source_url": info.url,
        })

    df_draft = pd.DataFrame(results)

    merged = df.merge(df_draft, on="player_name_clean", how="left")
    merged["drafted"] = merged["drafted"].fillna(False)
    merged.drop(columns=["player_name_clean"], inplace=True)

    out_path = Path(args.outp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    drafted_count = int(merged["drafted"].sum())
    print(f"\n✅ Done → {out_path}")
    print(f"Drafted players: {drafted_count}/{len(merged)} = {drafted_count/len(merged)*100:.2f}%")

if __name__ == "__main__":
    main()
