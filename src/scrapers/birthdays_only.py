#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
birthdays_only.py

Scrape accurate player birthdays from Sports-Reference for all skill-position
players who were drafted (from skill_draftees_2000_2025.csv).

This script:
- Uses Pro-Football-Reference *letter index pages* (avoids 429 blocks)
- Falls back to College Football Reference search
- Performs strict last-name matching + first-name scoring + nicknames
- Extracts birthdates from multiple SR formats
- Verifies the page is skill-position (QB/RB/WR/TE/FB)
- Uses name thresholds, min-year filtering, cache, jitter, and exponential backoff
- Designed for large datasets

USAGE:
    python3 birthdays_only.py
    python3 birthdays_only.py --debug
    python3 birthdays_only.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple, List, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ================================
# DEFAULT PATHS FOR YOUR PROJECT
# ================================
BASE_DIR = Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")
DEFAULT_IN = BASE_DIR / "data" / "scraper" / "skill_draftees_2000_2025.csv"
DEFAULT_OUT = BASE_DIR / "data" / "processed" / "skill_draftee_birthdays.csv"

# ================================
# CACHE / RATE LIMIT SETTINGS
# ================================
BASE_SLEEP   = 1.2
JITTER       = (0.4, 0.9)
BACKOFF_MULT = 2.0
MAX_BACKOFF  = 60

CACHE_PATH    = Path("cache/sr_birthdays_cache.json")
PFR_INDEX_DIR = Path("cache/pfr_index")
DEBUG_DIR     = Path("debug_html")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

for p in (CACHE_PATH.parent, PFR_INDEX_DIR, DEBUG_DIR):
    p.mkdir(parents=True, exist_ok=True)

_SUFFIXES = {"jr","jr.","sr","sr.","ii","iii","iv","v"}

# Nickname expansion map
_NICK = {
    "bill": "william","billy": "william","will": "william","liam": "william",
    "bob": "robert","bobby": "robert","rob": "robert","robbie": "robert","bert": "robert",
    "rich": "richard","rick": "richard","ricky": "richard","dick": "richard",
    "mike": "michael","mikey": "michael",
    "alex": "alexander","sasha": "alexander",
    "andy": "andrew","drew": "andrew",
    "tony": "anthony",
    "chris": "christopher","kit": "christopher",
    "dan": "daniel","danny": "daniel",
    "kate": "katherine","katie": "katherine","kat": "katherine",
    "liz": "elizabeth","lizzy": "elizabeth","beth": "elizabeth",
    "jamie": "james","jim": "james","jimmy": "james",
    "nick": "nicholas","nico": "nicholas",
    "pat": "patrick",
    "sam": "samuel",
    "tom": "thomas","tommy": "thomas",
    "ty": "tyler"
}

# ================================
# GENERIC TEXT UTILITIES
# ================================
def sleep_with_jitter(base=BASE_SLEEP):
    time.sleep(base + random.uniform(*JITTER))

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))

def clean_player_name(name: str) -> str:
    return re.sub(r"\*+", "", str(name)).strip()

def normalize_for_tokens(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(r"[.,'’`\"]", " ", s)
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_name(name: str):
    s = normalize_for_tokens(name)
    if not s:
        return None,None,[]
    parts = s.split()
    if parts and parts[-1] in _SUFFIXES and len(parts)>=2:
        parts = parts[:-1]
    if not parts:
        return None,None,[]
    first = parts[0]
    last  = parts[-1] if len(parts)>1 else None
    rest  = [p for p in parts[1:-1]
             if not re.fullmatch(r"[a-z]", p)]
    return first,last,rest

def last_name_initial(full_name: str) -> Optional[str]:
    _, last, _ = tokenize_name(full_name)
    return last[0].upper() if last else None

def req_with_backoff(url: str, debug: bool=False):
    delay = BASE_SLEEP
    while True:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = getattr(e.response,"status_code",None)
            if code == 429:
                delay = min(MAX_BACKOFF, delay*BACKOFF_MULT)
                if debug:
                    print(f"[429] backoff {delay:.1f}s for {url}")
                time.sleep(delay)
                continue
            raise
        except requests.RequestException:
            delay = min(MAX_BACKOFF, delay*BACKOFF_MULT)
            time.sleep(delay)

def get_soup(text: str) -> BeautifulSoup:
    return BeautifulSoup(text, "lxml")

def parse_iso_date(text: str) -> Optional[str]:
    text = (text or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    from datetime import datetime
    for fmt in ("%B %d, %Y","%b %d, %Y","%Y/%m/%d","%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def year_from_iso(iso: Optional[str]) -> Optional[int]:
    if not iso or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso):
        return None
    try:
        return int(iso[:4])
    except:
        return None

# ================================
# POSITION EXTRACTION
# ================================
DEFAULT_ALLOWED_SKILL = {"QB","RB","FB","WR","TE"}
NONSKILL_HINTS = {
    "OL","T","G","C","OT","OG","OC","LT","LG","RT","RG",
    "DL","DE","DT","NT","IDL",
    "LB","EDGE",
    "DB","CB","S","FS","SS",
    "K","P","LS"
}

def extract_birthdate_from_player_page(soup: BeautifulSoup):
    el = soup.select_one('[itemprop="birthDate"]')
    if el:
        iso = el.get("data-birth") or el.get_text(strip=True)
        iso = parse_iso_date(iso)
        if iso: return iso

    el = soup.select_one("#necro-birth")
    if el:
        iso = el.get("data-birth") or el.get_text(strip=True)
        iso = parse_iso_date(iso)
        if iso: return iso

    el = soup.select_one("span[data-birth]")
    if el:
        iso = parse_iso_date(el.get("data-birth"))
        if iso: return iso

    text = soup.get_text(" ", strip=True)
    m = re.search(
        r"Born:\s*("
        r"[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}"
        r"|\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}/\d{1,2}/\d{4}"
        r")",
        text
    )
    return parse_iso_date(m.group(1)) if m else None

def extract_positions_from_page(soup: BeautifulSoup) -> Set[str]:
    tokens=set()
    for lab in soup.select("strong"):
        if lab.get_text(strip=True).rstrip(":").lower()=="position":
            txt = lab.parent.get_text(" ", strip=True)
            m   = re.search("Position:\s*([A-Za-z0-9/ ,\\-\u00A0]+)",txt)
            if m:
                c = m.group(1)
                tokens |= set(re.findall(r"[A-Za-z]{1,4}", c))
    if not tokens:
        text = soup.get_text(" ",strip=True)
        m = re.search("Position:\s*([A-Za-z0-9/ ,\\-\u00A0]+)",text)
        if m:
            c = m.group(1)
            tokens |= set(re.findall(r"[A-Za-z]{1,4}", c))

    out=set()
    for t in tokens:
        u=t.upper()
        if u in {"HB","TB"}: u="RB"
        if u in {"SE","FL","SL"}: u="WR"
        out.add(u)
    return out

# ================================
# NAME MATCHING
# ================================
def normalize_fullname(name: str) -> str:
    first,last,rest = tokenize_name(name)
    if not first and not last:
        return ""
    parts=[p for p in [first]+rest+[last] if p]
    return " ".join(parts)

def canonical_first(fn: Optional[str]):
    if not fn: return fn
    f = re.sub(r"[.\s]","", fn.lower())
    return _NICK.get(f,f)

def first_name_score(target_first, cand_first):
    if not target_first or not cand_first:
        return 0.0
    tf = canonical_first(target_first)
    cf = canonical_first(cand_first)
    if tf==cf: return 1.0
    if len(tf)==1 and cf.startswith(tf): return 0.95
    if len(cf)==1 and tf.startswith(cf): return 0.95
    return SequenceMatcher(None, tf, cf).ratio()

def full_name_ratio(t,c):
    return SequenceMatcher(None,t,c).ratio()

def candidate_score(target,cand):
    t_first,t_last,_ = tokenize_name(target)
    c_first,c_last,_ = tokenize_name(cand)
    if not t_last or not c_last: return 0.0,False
    if t_last!=c_last: return 0.0,False
    fn=first_name_score(t_first,c_first)
    full=full_name_ratio(normalize_fullname(target), normalize_fullname(cand))
    score=0.75*fn + 0.25*full
    return score,True

# ================================
# FIND CANDIDATES — PFR INDEX
# ================================
def load_pfr_index_letter(letter: str, debug=False):
    letter = letter.upper()
    fp = PFR_INDEX_DIR / f"{letter}.html"
    if fp.exists():
        return get_soup(fp.read_text())
    url=f"https://www.pro-football-reference.com/players/{letter}/"
    sleep_with_jitter()
    r=req_with_backoff(url, debug=debug)
    fp.write_text(r.text)
    return get_soup(r.text)

def pfr_candidates_by_score(target_name, debug, name_thresh):
    ini = last_name_initial(target_name)
    if not ini: return []
    soup = load_pfr_index_letter(ini, debug)
    out=[]
    for a in soup.select("a[href]"):
        href=a.get("href","")
        if not re.match(r"^/players/[A-Z]/[^/]+\.htm$", href):
            continue
        cand_text=(a.get_text() or "").strip()
        score,ok=candidate_score(target_name,cand_text)
        if not ok or score<0.60:
            continue
        out.append((score,"https://www.pro-football-reference.com"+href,cand_text))
    out.sort(key=lambda t: t[0], reverse=True)
    return out

# ================================
# FIND CANDIDATES — CFB SEARCH
# ================================
def cfb_search_url(name):
    q=requests.utils.quote(name)
    return f"https://www.sports-reference.com/cfb/search/search.fcgi?search={q}"

def cfb_candidates_by_score(target, debug, name_thresh):
    sleep_with_jitter()
    r=req_with_backoff(cfb_search_url(target), debug=debug)
    soup=get_soup(r.text)
    if re.match(r"^https?://www\.sports-reference\.com/cfb/players/[a-z0-9-]+\.html$", r.url):
        return [(1.0, r.url, target)]
    out=[]
    for a in soup.select("a[href]"):
        href=a.get("href","")
        if not re.match(r"^/cfb/players/[a-z0-9-]+\.html$", href):
            continue
        cand=(a.get_text() or "").strip()
        score,ok=candidate_score(target,cand)
        if ok and score>=0.60:
            out.append((score,"https://www.sports-reference.com"+href,cand))
    out.sort(key=lambda t: t[0], reverse=True)
    return out

# ================================
# VERIFY CANDIDATE PAGE
# ================================
def try_candidate_url(url, min_year, allowed_skill, relax_skill, debug=False):
    sleep_with_jitter()
    r=req_with_backoff(url, debug=debug)
    soup=get_soup(r.text)

    pos=extract_positions_from_page(soup)
    if pos:
        if pos & NONSKILL_HINTS:
            return None
        if not (pos & allowed_skill):
            return None
    else:
        if not relax_skill:
            return None

    bd=extract_birthdate_from_player_page(soup)
    if not bd:
        return None
    y=year_from_iso(bd)
    if not y or y<min_year:
        return None
    return bd

# ================================
# MAIN BIRTHDAY FETCH LOGIC
# ================================
def fetch_birthdate(name, min_year, name_thresh, allowed_skill, relax_skill, debug=False):
    # PFR index (preferred)
    for score,url,cand in pfr_candidates_by_score(name, debug, name_thresh):
        if score<name_thresh: break
        bd=try_candidate_url(url, min_year, allowed_skill, relax_skill, debug)
        if bd: return bd,url

    # CFB fallback
    for score,url,cand in cfb_candidates_by_score(name, debug, name_thresh):
        if score<name_thresh: break
        bd=try_candidate_url(url, min_year, allowed_skill, relax_skill, debug)
        if bd: return bd,url

    return None,None

# ================================
# CLI + MAIN WORKFLOW
# ================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN),
                    help=f"Input CSV (default {DEFAULT_IN})")
    ap.add_argument("--out", dest="outp", default=str(DEFAULT_OUT),
                    help=f"Output CSV (default {DEFAULT_OUT})")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-year", type=int, default=1975)
    ap.add_argument("--name-thresh", type=float, default=0.92)
    ap.add_argument("--allowed-skill", type=str,
                    default="QB,RB,HB,TB,FB,WR,SE,FL,TE")
    ap.add_argument("--strict-skill", action="store_true",
                    help="Require explicit skill position.")
    ap.add_argument("--relax-skill", action="store_true",
                    help="Allow unknown position pages.")
    args=ap.parse_args()

    relax_skill = not args.strict_skill or args.relax_skill

    allowed_skill=set(u.strip().upper()
                      for u in args.allowed_skill.split(",") if u.strip())
    mapped=set()
    for u in list(allowed_skill):
        if u in {"HB","TB"}: mapped.add("RB")
        if u in {"SE","FL","SL"}: mapped.add("WR")
    allowed_skill |= mapped
    allowed_skill |= DEFAULT_ALLOWED_SKILL

    df=pd.read_csv(args.inp)
    name_col="player_name" if "player_name" in df.columns else (
        "player" if "player" in df.columns else None)
    if not name_col:
        raise SystemExit("Input CSV must contain 'player_name' or 'player'.")

    names=(df[name_col].dropna().astype(str).map(clean_player_name)
           .pipe(lambda s: s[s.ne("")]).drop_duplicates())

    if args.limit:
        names=names.head(args.limit)

    names=names.sort_values().tolist()

    cache=load_cache()
    out_rows=[]

    for i,name in enumerate(names,1):
        if args.debug:
            print(f"[{i}/{len(names)}] {name}")

        cached=cache.get(name,{})
        bd_cached=cached.get("birth_date")
        if bd_cached:
            y=year_from_iso(bd_cached)
            if y and y>=args.min_year:
                out_rows.append((name, bd_cached, cached.get("source_url")))
                continue

        bd,src = fetch_birthdate(
            name,
            min_year=args.min_year,
            name_thresh=args.name_thresh,
            allowed_skill=allowed_skill,
            relax_skill=relax_skill,
            debug=args.debug
        )

        out_rows.append((name, bd, src))
        if bd and year_from_iso(bd)>=args.min_year:
            cache[name]={"birth_date":bd,"source_url":src}
            save_cache(cache)

    out=pd.DataFrame(out_rows,
                     columns=["player_name","birth_date","birthdate_source_url"])
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outp, index=False)

    missing=int(out["birth_date"].isna().sum())
    print(f"\nDONE. Saved {args.outp}")
    print(f"Missing birthdays: {missing} / {len(out)}")

if __name__=="__main__":
    main()
