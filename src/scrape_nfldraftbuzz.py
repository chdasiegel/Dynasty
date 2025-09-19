#!/usr/bin/env python3
"""
Scrape player metrics from NFLDraftBuzz profiles and save to CSV.

Usage examples:
  # From a list of profile URLs (one per line)
  python scrape_nfldraftbuzz.py --urls-file urls.txt --out players.csv

  # Crawl by year and position, then scrape each profile
  python scrape_nfldraftbuzz.py --year 2025 --position RB --out rb_2025.csv

  # Crawl all positions for a draft year
  python scrape_nfldraftbuzz.py --year 2025 --out draft_2025.csv
"""

from __future__ import annotations

import argparse
import re
import time
import sys
from typing import Dict, List, Optional, Tuple
import csv
import random

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------
# Config (A: robust session + headers) + (B: slower politeness)
# --------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = (1.0, 2.0)  # (B) slightly slower, friendlier delays

BASE = "https://www.nfldraftbuzz.com"

# one shared session with retries
_SESSION = requests.Session()
_RETRY = Retry(
    total=5,                            # up to 5 tries
    backoff_factor=0.6,                 # 0.6, 1.2, 1.8, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD", "OPTIONS"],
    raise_on_status=False,
)
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))

# Positions commonly used on the site (uppercase)
ALL_POSITIONS = [
    "QB","RB","WR","TE","OT","IOL","C","G","T",
    "EDGE","DL","IDL","LB","CB","S","K","P","LS"
]

# --------------------------
# Utilities
# --------------------------
def sleep_polite() -> None:
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))

def fetch(url: str) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and polite headers; return BeautifulSoup or None."""
    try:
        headers = dict(HEADERS)
        # light referer sometimes helps pass simple anti-bot checks
        headers.setdefault("Referer", "https://www.nfldraftbuzz.com/draftboard/2025/QB")
        r = _SESSION.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            # Helpful debug so we can see why parse_profile returns Nones
            print(f"[fetch] {r.status_code} {r.reason} for {url}")
            return None
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"[fetch] exception for {url}: {e}")
        return None

def textnorm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s.strip()) if s else ""

def to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

# Height conversion: "5'10\"" or "5-10" -> inches; also returns meters if present
def parse_height(raw: str) -> Tuple[Optional[int], Optional[float]]:
    s = raw.replace("–","-").replace("—","-")
    m = re.search(r"(\d+)[\'-]?\s*(\d{1,2})", s)
    inches = None
    if m:
        feet = int(m.group(1)); inch = int(m.group(2))
        inches = feet * 12 + inch
    mm = re.search(r"(\d\.\d{2})m", raw)
    meters = float(mm.group(1)) if mm else None
    return inches, meters

def parse_weight(raw: str) -> Tuple[Optional[int], Optional[float]]:
    lbm = re.search(r"(\d{2,3})\s?lb", raw, re.I)
    kgm = re.search(r"(\d{2,3}\.?\d*)\s?kg", raw, re.I)
    lbs = int(lbm.group(1)) if lbm else None
    kgs = float(kgm.group(1)) if kgm else None
    return lbs, kgs

def extract_value_after_label(block: BeautifulSoup, labels: List[str]) -> Optional[str]:
    # 1) dt/dd structure
    for dt in block.select("dt"):
        key = textnorm(dt.get_text(":"))
        for lab in labels:
            if lab.lower() in key.lower():
                dd = dt.find_next("dd")
                if dd:
                    return textnorm(dd.get_text(" "))
    # 2) table rows
    for tr in block.select("tr"):
        tds = [textnorm(td.get_text(" ")) for td in tr.select("th,td")]
        if len(tds) >= 2:
            for lab in labels:
                if lab.lower() in tds[0].lower():
                    return tds[1]
    # 3) generic "Label: value"
    for node in block.find_all(string=True):
        s = textnorm(node)
        for lab in labels:
            if s.lower().startswith(lab.lower()+":"):
                return s.split(":",1)[1].strip()
    return None

# --------------------------
# Crawlers
# --------------------------
def discover_profiles_by_year(year: int, positions: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    pos_list = positions or ALL_POSITIONS
    for pos in pos_list:
        board_url = f"{BASE}/draftboard/{year}/{pos}"
        soup = fetch(board_url)
        sleep_polite()
        if not soup:
            continue
        for a in soup.select("a[href*='/Player/']"):
            href = a.get("href")
            if href and href.startswith("/Player/"):
                urls.append(BASE + href)
    # Dedup while preserving order
    seen, deduped = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); deduped.append(u)
    return deduped

def load_urls_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# --------------------------
# Profile Parser
# --------------------------
def parse_profile(url: str) -> Dict[str, Optional[str]]:
    soup = fetch(url)
    sleep_polite()
    row: Dict[str, Optional[str]] = {
        "source_url": url,
        "player": None, "position": None, "college": None, "class": None,
        "height_raw": None, "height_in": None, "height_m": None,
        "weight_raw": None, "weight_lb": None, "weight_kg": None,
        "forty": None, "ten_split": None, "shuttle": None, "three_cone": None,
        "vertical": None, "broad": None, "bench": None,
        "hand_size": None, "arm_length": None, "wingspan": None,
        "age": None, "dob": None, "class_year": None,
        "draft_projection": None, "ras": None
    }
    if not soup:
        return row

    # name
    h1 = soup.select_one("h1, .player-header h1, .page-title h1")
    if h1:
        row["player"] = textnorm(h1.get_text(" "))
    if not row["player"]:
        og = soup.select_one("meta[property='og:title']")
        if og and og.get("content"):
            row["player"] = textnorm(og["content"])

    # common containers
    containers = soup.select(
        ".player-card, .player-profile, .player-bio, .measurements, "
        ".player-info, .card, .grid, .content"
    ) or [soup]

    LABELS = {
        "position": ["Position"],
        "college": ["College", "School"],
        "class": ["Class", "Year"],
        "height": ["Height"],
        "weight": ["Weight"],
        "forty": ["40 Yard", "40-yard", "40 yard", "40 time", "40"],
        "ten_split": ["10 Yard", "10 Split", "10-yard"],
        "shuttle": ["Shuttle", "20 Shuttle", "20-yard shuttle"],
        "three_cone": ["3 Cone", "Three Cone", "3-cone", "3 Cone Drill"],
        "vertical": ["Vertical", "Vertical Jump"],
        "broad": ["Broad", "Broad Jump"],
        "bench": ["Bench", "Bench Press"],
        "hand_size": ["Hand Size", "Hands"],
        "arm_length": ["Arm Length", "Arms"],
        "wingspan": ["Wingspan", "Span"],
        "age": ["Age"],
        "dob": ["DOB", "Born", "Date of Birth"],
        "class_year": ["Draft Class", "Draft Year"],
        "draft_projection": ["Projection", "Draft Projection"],
        "ras": ["RAS", "Relative Athletic Score"]
    }

    for cont in containers:
        if not row["position"]:
            row["position"] = extract_value_after_label(cont, LABELS["position"]) or row["position"]
        if not row["college"]:
            row["college"] = extract_value_after_label(cont, LABELS["college"]) or row["college"]
        if not row["class"]:
            row["class"] = extract_value_after_label(cont, LABELS["class"]) or row["class"]

        if not row["height_raw"]:
            row["height_raw"] = extract_value_after_label(cont, LABELS["height"]) or row["height_raw"]
        if not row["weight_raw"]:
            row["weight_raw"] = extract_value_after_label(cont, LABELS["weight"]) or row["weight_raw"]

        for key in ["forty","ten_split","shuttle","three_cone","vertical","broad","bench",
                    "hand_size","arm_length","wingspan","age","dob","class_year",
                    "draft_projection","ras"]:
            if not row[key]:
                row[key] = extract_value_after_label(cont, LABELS[key]) or row[key]

    # normalize height/weight
    if row["height_raw"]:
        h_in, h_m = parse_height(row["height_raw"])
        row["height_in"] = str(h_in) if h_in is not None else None
        row["height_m"]  = f"{h_m:.2f}" if h_m is not None else None
    if row["weight_raw"]:
        w_lb, w_kg = parse_weight(row["weight_raw"])
        row["weight_lb"] = str(w_lb) if w_lb is not None else None
        row["weight_kg"] = f"{w_kg:.1f}" if w_kg is not None else None

    # clean numerics
    for k in ["forty","ten_split","shuttle","three_cone","vertical","broad","bench",
              "hand_size","arm_length","wingspan","age","ras"]:
        if row[k]:
            m = re.search(r"(\d+\.\d+|\d+)", row[k])
            row[k] = m.group(1) if m else row[k]

    return row

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", help="Path to text file of NFLDraftBuzz profile URLs (one per line).")
    ap.add_argument("--year", type=int, help="Draft year to crawl (e.g., 2025).")
    ap.add_argument("--position", help="Position to crawl (e.g., RB). If omitted with --year, all positions are crawled.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    urls: List[str] = []
    if args.urls_file:
        urls = load_urls_file(args.urls_file)

    if args.year:
        positions = [args.position] if args.position else None
        discovered = discover_profiles_by_year(args.year, positions)
        urls = list(dict.fromkeys((urls or []) + discovered))  # merge + dedup

    if not urls:
        print("No URLs to scrape. Provide --urls-file and/or --year [--position].", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Optional[str]]] = []
    for i, u in enumerate(urls, 1):
        sys.stdout.write(f"[{i}/{len(urls)}] {u}\n")
        sys.stdout.flush()
        row = parse_profile(u)
        rows.append(row)

    fieldnames = [
        "player","position","college","class","class_year","draft_projection",
        "height_raw","height_in","height_m","weight_raw","weight_lb","weight_kg",
        "forty","ten_split","shuttle","three_cone","vertical","broad","bench",
        "hand_size","arm_length","wingspan","age","dob","ras","source_url"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved {len(rows)} players -> {args.out}")

if __name__ == "__main__":
    main()
