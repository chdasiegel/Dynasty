# src/mockdraftable_scraper.py

from __future__ import annotations

import re
import time
import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

BASE = "https://www.mockdraftable.com"

# ---- Polite scraping config ----
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = (1.2, 2.4)  # randomized delay
RETRY_ATTEMPTS = 3

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

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
    for _ in range(RETRY_ATTEMPTS):
        try:
            r = _SCRAPER.get(url, headers=_headers(referer), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code == 200 and r.text and len(r.text) > 500:
                return r.text
            _sleep()
        except Exception:
            _sleep()
    return None

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()

def _strip_num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def _parse_height_in(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*(?:'|ft|-)\s*(\d{1,2})", t)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m2 = re.search(r"(\d+)-(\d{1,2})(?:\.(\d))?", t)
    if m2:
        feet, inch = int(m2.group(1)), int(m2.group(2))
        frac = m2.group(3)
        return feet * 12 + inch + (0.5 if frac == "5" else 0.0)
    return None

def _clean_key(k: str) -> str:
    k = _norm(k).lower()
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
    if k in mapping:
        return mapping[k]
    for pat, val in mapping.items():
        if pat in k:
            return val
    return k.replace(" ", "_").replace("-", "_")

@dataclass
class PlayerIndexItem:
    name: str
    href: str
    position: Optional[str] = None
    year: Optional[str] = None
    college: Optional[str] = None

def _collect_search_index(base_search_url: str, max_pages: int = 10) -> Dict[str, PlayerIndexItem]:
    index: Dict[str, PlayerIndexItem] = {}

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
            break
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.select("a[href^='/players/'], a[href^='/player/']"):
            href = a.get("href") or ""
            if not href.startswith("/"):
                continue
            name_text = _norm(a.get_text(" "))
            if not name_text or len(name_text) < 2:
                parent = a.find_parent(["div", "li", "tr", "article"])
                if parent:
                    h = parent.select_one("h1,h2,h3,.name,.player-name,strong")
                    if h:
                        name_text = _norm(h.get_text(" "))
            if not name_text:
                continue

            pos = year = college = None
            parent = a.find_parent(["div", "li", "tr", "article"])
            if parent:
                meta = _norm(parent.get_text(" "))
                mm = re.search(r"\b(QB|RB|WR|TE|CB|S|LB|EDGE|DL|IDL|OT|IOL|G|T|C|K|P|ATH)\b", meta)
                if mm: pos = mm.group(1)
                y = re.search(r"\b(19|20)\d{2}\b", meta)
                if y: year = y.group(0)
                col = re.search(r"College:\s*([A-Za-z .'-]+)", meta)
                if col: college = _norm(col.group(1))

            key = _norm(name_text).lower()
            if key not in index:
                index[key] = PlayerIndexItem(name=name_text, href=href, position=pos, year=year, college=college)
        referer = url

    return index

def _search_query_page(name: str) -> Optional[str]:
    """Fallback: use the site's search query page to find the first profile link for a given name."""
    q = re.sub(r"\s+", "+", _norm(name))
    url = f"{BASE}/search?q={q}"
    html = _get_html(url, referer=BASE)
    _sleep()
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    # Look for the first player link in results
    a = soup.select_one("a[href^='/players/'], a[href^='/player/']")
    if not a:
        return None
    href = a.get("href") or ""
    return BASE + href if href.startswith("/") else None

def _resolve_names_to_urls(
    player_names: List[str],
    search_base_urls: List[str],
    max_pages_per_base: int = 10,
) -> Dict[str, str]:
    combined: Dict[str, PlayerIndexItem] = {}
    for base in search_base_urls:
        idx = _collect_search_index(base, max_pages=max_pages_per_base)
        for k, v in idx.items():
            if k not in combined:
                combined[k] = v

    out: Dict[str, str] = {}
    unresolved: List[str] = []

    for name in player_names:
        key = _norm(name).lower()
        if key in combined:
            out[name] = BASE + combined[key].href
            continue
        # try loose alt key
        alt = re.sub(r"[^a-z0-9]+", "", key)
        best = None
        for k, v in combined.items():
            if re.sub(r"[^a-z0-9]+", "", k) == alt:
                best = v
                break
        if best:
            out[name] = BASE + best.href
        else:
            unresolved.append(name)

    # Fallback: per-player query page
    for name in unresolved:
        url = _search_query_page(name)
        if url:
            out[name] = url

    still_missing = [n for n in player_names if n not in out]
    if still_missing:
        print(f"[warn] Still unresolved after query fallback: {still_missing}")

    return out

def _parse_profile(url: str) -> Dict[str, Optional[str]]:
    _get_html(BASE)
    html = _get_html(url, referer=BASE + "/search")
    _sleep()
    if not html:
        return {"source_url": url}

    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, Optional[str]] = {"source_url": url}

    h = soup.select_one("h1, h2, .player-name, .header h1")
    if h:
        data["player"] = _norm(h.get_text(" "))

    header_block = soup.select_one(".player-header, .header, .summary, header, .bio, .profile-header")
    header_text = _norm(header_block.get_text(" ")) if header_block else ""

    mm_pos = re.search(r"\b(QB|RB|WR|TE|CB|S|LB|EDGE|DL|IDL|OT|IOL|G|T|C|K|P|ATH)\b", header_text)
    if mm_pos: data["position"] = mm_pos.group(1)
    mm_year = re.search(r"\b(19|20)\d{2}\b", header_text)
    if mm_year: data["year"] = mm_year.group(0)
    mm_col = re.search(r"College:\s*([A-Za-z .'-]+)", header_text)
    if mm_col: data["college"] = _norm(mm_col.group(1))

    # scan tables for key/value rows
    for tbl in soup.select("table"):
        for tr in tbl.select("tr"):
            cells = [ _norm(td.get_text(" ")) for td in tr.select("th,td") ]
            if len(cells) < 2:
                continue
            key = _clean_key(cells[0])
            val = cells[1]
            if key in {
                "height","weight","hand_size","arm_length","wingspan",
                "forty","ten_split","short_shuttle","three_cone",
                "vertical","broad","bench","ras"
            }:
                data[key] = val

    # normalize numeric variants
    if "height" in data:
        hi = _parse_height_in(data["height"])
        if hi is not None:
            data["height_in"] = hi

    if "weight" in data:
        data["weight_lb"] = _strip_num(data["weight"])

    for k in ["hand_size","arm_length","wingspan","vertical","broad"]:
        if k in data:
            data[k + "_in"] = _strip_num(data[k])

    for k in ["forty","ten_split","short_shuttle","three_cone"]:
        if k in data:
            data[k + "_s"] = _strip_num(data[k])

    if "bench" in data:
        data["bench_reps"] = _strip_num(data["bench"])

    return data

def build_mockdraftable_dataframe(
    player_names: List[str],
    search_base_urls: List[str],
    max_pages_per_base: int = 10,
) -> pd.DataFrame:
    name_to_url = _resolve_names_to_urls(player_names, search_base_urls, max_pages_per_base=max_pages_per_base)

    unresolved = [n for n in player_names if n not in name_to_url]
    if unresolved:
        print(f"[warn] Unresolved names (not found in search pages): {unresolved}")

    rows = []
    for name, url in name_to_url.items():
        rec = _parse_profile(url)
        rec["requested_name"] = name
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
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
