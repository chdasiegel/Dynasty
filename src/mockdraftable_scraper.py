#!/usr/bin/env python3
"""
MockDraftable scraper
- Given a list of player names, crawl MockDraftable search pages to find profile URLs.
- Scrape each profile's measurables and return a tidy pandas DataFrame.

Requires: cloudscraper, bs4, pandas
"""

from __future__ import annotations

import re
import time
import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

BASE = "https://www.mockdraftable.com"

# ---- Polite scraping config ----
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = (1.2, 2.4)  # randomized delay
RETRY_ATTEMPTS = 3

# Rotate a few realistic desktop UAs
UA_POOL = [
    # Chrome (Win)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome (Mac)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Edge (Win)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

# One global scraper (Cloudflare-friendly)
_SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)

def _headers(referer: Optional[str] = None) -> dict:
    h = {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
    }
    if referer:
        h["Referer"] = referer
    return h

def _sleep():
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))

def _get_html(url: str, referer: Optional[str] = None) -> Optional[str]:
    """GET with basic retries + polite delay; return HTML or None."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = _SCRAPER.get(url, headers=_headers(referer), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code == 200 and r.text and len(r.text) > 500:
                return r.text
            # soft backoff
            _sleep()
        except Exception:
            _sleep()
    return None

# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Basic text normalize (remove weird whitespace, normalize unicode)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_non_digits(s: str) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def _parse_height_to_inches(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*(?:'|ft|-)\s*(\d{1,2})", t)
    if m:
        feet, inch = int(m.group(1)), int(m.group(2))
        return feet * 12 + inch
    # Sometimes shows as e.g. 6-2.5
    m2 = re.search(r"(\d+)-(\d{1,2})(?:\.(\d))?", t)
    if m2:
        feet = int(m2.group(1))
        inch = int(m2.group(2))
        frac = m2.group(3)
        inches = feet * 12 + inch + (0.5 if frac and frac == "5" else 0)
        return inches
    return None

def _clean_measure_key(k: str) -> str:
    """Normalize row label keys to consistent column names."""
    k = _norm(k).lower()
    # map common variants
    mapping = {
        "height": "height",
        "weight": "weight",
        "hand size": "hand_size",
        "hand": "hand_size",
        "arm length": "arm_length",
        "arm": "arm_length",
        "wingspan": "wingspan",

        "40 yard dash": "forty",
        "40-yard dash": "forty",
        "40": "forty",
        "40y": "forty",
        "10 yard split": "ten_split",
        "10 split": "ten_split",
        "10": "ten_split",
        "short shuttle": "short_shuttle",
        "20 yard shuttle": "short_shuttle",
        "20 shuttle": "short_shuttle",
        "3 cone": "three_cone",
        "3-cone": "three_cone",
        "three cone": "three_cone",
        "vertical": "vertical",
        "vertical jump": "vertical",
        "broad": "broad",
        "broad jump": "broad",
        "bench": "bench",
        "bench press": "bench",
        "ras": "ras",
    }
    # try exact first
    if k in mapping:
        return mapping[k]
    # try contains
    for pat, val in mapping.items():
        if pat in k:
            return val
    return k.replace(" ", "_").replace("-", "_")

@dataclass
class PlayerIndexItem:
    name: str
    href: str  # relative path to profile, e.g., /players/abcd-efgh
    position: Optional[str] = None
    year: Optional[str] = None
    college: Optional[str] = None

# ---------- Search-page crawling ----------
def _collect_search_index(base_search_url: str, max_pages: int = 10) -> Dict[str, PlayerIndexItem]:
    """
    Crawl paginated search pages (adding &page=2..N if not present) and
    build a dict: normalized_name -> PlayerIndexItem
    """
    index: Dict[str, PlayerIndexItem] = {}
    # detect if page param present; otherwise add &page=
    def with_page(url: str, page: int) -> str:
        if "page=" in url:
            return re.sub(r"(?:[?&]page=)\d+", f"&page={page}", url)
        return url + ("&" if "?" in url else "?") + f"page={page}"

    referer = base_search_url
    for page in range(1, max_pages + 1):
        url = with_page(base_search_url, page)
        html = _get_html(url, referer=referer)
        _sleep()
        if not html:
            # likely end of pages or blocked
            break
        soup = BeautifulSoup(html, "html.parser")

        # Heuristic: result "cards" or table rows with player links
        # anchor hrefs that look like player pages:
        # e.g., /players/xxxxxxxx (often lower-case, hyphenated)
        for a in soup.select("a[href^='/players/'], a[href^='/player/']"):
            href = a.get("href") or ""
            if not href or not href.startswith("/"):
                continue
            # Try to extract name nearest to the link (link text or nearby header)
            name_text = _norm(a.get_text(" "))
            if not name_text or len(name_text) < 3:
                # sometimes name is in parent container
                parent = a.find_parent(["div", "li", "tr", "article"])
                if parent:
                    h = parent.select_one("h2, h3, .name, .player-name, strong")
                    if h:
                        name_text = _norm(h.get_text(" "))
            if not name_text:
                continue

            # capture position/year/college if present nearby
            pos = None; year = None; college = None
            parent = a.find_parent(["div", "li", "tr", "article"])
            if parent:
                # common places for meta text
                meta = parent.get_text(" ")
                mm = re.search(r"\b(WR|RB|QB|TE|CB|S|LB|EDGE|DL|IDL|OT|IOL|G|T|C|K|P)\b", meta)
                if mm: pos = mm.group(1)
                y = re.search(r"\b(19|20)\d{2}\b", meta)
                if y: year = y.group(0)
                col = re.search(r"College:\s*([A-Za-z .'-]+)", meta)
                if col: college = _norm(col.group(1))

            idx_key = _norm(name_text).lower()
            if idx_key not in index:
                index[idx_key] = PlayerIndexItem(name=name_text, href=href, position=pos, year=year, college=college)

        # simple termination: if page yielded no new entries, stop
        if page > 1 and len(index) == 0:
            break
        referer = url

    return index

def _resolve_names_to_urls(
    player_names: List[str],
    search_base_urls: List[str],
    max_pages_per_base: int = 10,
) -> Dict[str, str]:
    """
    Build a unified search index from the given search pages, then
    map each requested player name -> absolute profile URL (best match).
    """
    # Build combined index
    combined: Dict[str, PlayerIndexItem] = {}
    for base in search_base_urls:
        index = _collect_search_index(base, max_pages=max_pages_per_base)
        for k, v in index.items():
            if k not in combined:
                combined[k] = v

    # Match requested names
    out: Dict[str, str] = {}
    for name in player_names:
        key = _norm(name).lower()
        if key in combined:
            out[name] = BASE + combined[key].href
            continue
        # try loose match (ignore punctuation / hyphens)
        alt = re.sub(r"[^a-z0-9]+", "", key)
        best = None
        for k, v in combined.items():
            k_alt = re.sub(r"[^a-z0-9]+", "", k)
            if k_alt == alt:
                best = v
                break
        if best:
            out[name] = BASE + best.href
        # else leave unresolved; we'll report later
    return out

# ---------- Profile scraping ----------
def _parse_profile(url: str) -> Dict[str, Optional[str]]:
    """Fetch a player's profile page and parse measurables into a dict."""
    # warm up on the root to set cookies
    _get_html(BASE)
    html = _get_html(url, referer=BASE + "/search")
    _sleep()
    if not html:
        return {"source_url": url}

    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, Optional[str]] = {"source_url": url}

    # Name / header
    h = soup.select_one("h1, h2, .player-name, .header h1")
    if h:
        data["player"] = _norm(h.get_text(" "))

    # Position / year / college (best-effort)
    # Often present in a header block or info list
    header_text = ""
    header_block = soup.select_one(".player-header, .header, .summary, header, .bio, .profile-header")
    if header_block:
        header_text = _norm(header_block.get_text(" "))

    mm_pos = re.search(r"\b(QB|RB|WR|TE|CB|S|LB|EDGE|DL|IDL|OT|IOL|G|T|C|K|P)\b", header_text)
    if mm_pos: data["position"] = mm_pos.group(1)
    mm_year = re.search(r"\b(19|20)\d{2}\b", header_text)
    if mm_year: data["year"] = mm_year.group(0)
    mm_col = re.search(r"College:\s*([A-Za-z .'-]+)", header_text)
    if mm_col: data["college"] = _norm(mm_col.group(1))

    # Look for a measurables table by scanning for likely labels in table rows
    tables = soup.select("table")
    labels_seen = set()
    for tbl in tables:
        for tr in tbl.select("tr"):
            tds = [ _norm(td.get_text(" ")) for td in tr.select("th,td") ]
            if len(tds) < 2:
                continue
            key_raw, val_raw = tds[0], tds[1]
            # Identify if this looks like a measurement row by seeing
            # a key label we recognize
            key = _clean_measure_key(key_raw)
            if key in {
                "height","weight","hand_size","arm_length","wingspan",
                "forty","ten_split","short_shuttle","three_cone",
                "vertical","broad","bench","ras"
            }:
                labels_seen.add(key)
                data[key] = val_raw

    # Normalize numeric fields
    # Height → inches; Weight → lb; hand/arm/wingspan inches; timed drills seconds; jumps inches; bench reps int
    if "height" in data:
        hi = _parse_height_to_inches(data["height"])
        if hi is not None:
            data["height_in"] = hi

    if "weight" in data:
        data["weight_lb"] = _strip_non_digits(data["weight"])

    for k in ["hand_size","arm_length","wingspan","vertical","broad"]:
        if k in data:
            data[k + "_in"] = _strip_non_digits(data[k])

    for k in ["forty","ten_split","short_shuttle","three_cone"]:
        if k in data:
            data[k + "_s"] = _strip_non_digits(data[k])

    if "bench" in data:
        data["bench_reps"] = _strip_non_digits(data["bench"])

    return data

# ---------- Public API ----------
def build_mockdraftable_dataframe(
    player_names: List[str],
    search_base_urls: List[str],
    max_pages_per_base: int = 10,
) -> pd.DataFrame:
    """
    1) Crawl the given search pages (with pagination) to build a name->URL map.
    2) Resolve the provided player_names to profile URLs.
    3) Scrape each profile and return a tidy DataFrame.
    """
    # Step 1/2: Resolve names
    name_to_url = _resolve_names_to_urls(player_names, search_base_urls, max_pages_per_base=max_pages_per_base)

    unresolved = [n for n in player_names if n not in name_to_url]
    if unresolved:
        print(f"[warn] Unresolved names (not found in search pages): {unresolved}")

    rows: List[Dict[str, Optional[str]]] = []
    for name, url in name_to_url.items():
        rec = _parse_profile(url)
        rec["requested_name"] = name
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # A tidy set of preferred column order
    preferred = [
        "requested_name","player","position","year","college","source_url",
        "height","height_in",
        "weight","weight_lb",
        "hand_size","hand_size_in",
        "arm_length","arm_length_in",
        "wingspan","wingspan_in",
        "forty","forty_s",
        "ten_split","ten_split_s",
        "short_shuttle","short_shuttle_s",
        "three_cone","three_cone_s",
        "vertical","vertical_in",
        "broad","broad_in",
        "bench","bench_reps",
        "ras",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]
