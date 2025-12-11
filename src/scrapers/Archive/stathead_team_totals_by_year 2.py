#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/scrapers/stathead_team_totals_by_year.py

Hybrid approach with configurable Stathead "Sort By":
  • For each year, try Stathead once for PASSING and once for RUSHING, with order_by=<your key>.
  • If Stathead yields nothing, fall back to scraping public SR team pages (no login).
  • Save ONE CSV per year for passing and ONE for rushing.

Outputs:
  /Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_passing/team_passing_{year}.csv
  /Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_rushing/team_rushing_{year}.csv
"""

from __future__ import annotations
import os, re, time, io, random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# --------------------------- Config ---------------------------
BASE_SR = "https://www.sports-reference.com"
BASE_STATHEAD = "https://www.sports-reference.com/stathead/football/cfb/team-season-finder.cgi"

# Be realistic to reduce “are you a bot?” flags on the public SR pages
HEADERS_PUBLIC = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.sports-reference.com/cfb/",
}

HEADERS_STATHEAD = {
    # UA can be the same; this is for the Stathead finder endpoint
    "User-Agent": HEADERS_PUBLIC["User-Agent"],
}

PASS_DIR = "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_passing"
RUSH_DIR = "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_rushing"

# Column candidates seen across years
PASS_CANDS = {
    "cmp": ["pass_cmp","cmp","completions","cmp_pass"],
    "att": ["pass_att","att_pass","att (pass)","att"],
    "yds": ["pass_yds","yds_pass","passing_yards","pass yds","yds (pass)","yds"],
    "td" : ["pass_td","td_pass","passing_td","pass tds","td","tds"],
    "int": ["int","ints","interceptions"],
    "team": ["team","school","school_name","tm","team_name_abbr"],
    "conf": ["conf","conference"],
}
RUSH_CANDS = {
    "att": ["rush_att","att_rush","att (rush)","rush att"],
    "yds": ["rush_yds","yds_rush","rushing_yards","rush yds","yds (rush)"],
    "td" : ["rush_td","td_rush","rushing_td","rush tds"],
    "team": ["team","school","school_name","tm","team_name_abbr"],
    "conf": ["conf","conference"],
}

# --------------------------- Utilities ---------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def snake(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", "_", _re.sub(r"[^\w\s]", "", str(s))).lower()

def normalize_text(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", str(s or "")).strip()

def pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for c in cands:
        for k, orig in low.items():
            if c.lower() in k: return orig
    return None

# --------------------------- Stathead (with configurable sort) ---------------------------
def _fetch_stathead(year: int, mode: str, order_by: str, session: requests.Session, debug: bool=False) -> pd.DataFrame:
    """
    mode: "passing" or "rushing"
    order_by: any Stathead column key, e.g. "team_name_abbr", "pass_cmp", "rush_att", "pass_yds", ...
    Applies a simple >0 filter so the table returns rows.
    """
    if mode == "passing":
        params = {
            "request": 1,
            "year_min": year,
            "year_max": year,
            "order_by": order_by,
            "cstat[1]": "pass_cmp",  # minimal filter to ensure results
            "ccomp[1]": "gt",
            "cval[1]": 0,
        }
    else:
        params = {
            "request": 1,
            "year_min": year,
            "year_max": year,
            "order_by": order_by,
            "cstat[2]": "rush_att",
            "ccomp[2]": "gt",
            "cval[2]": 0,
        }

    r = session.get(BASE_STATHEAD, headers=HEADERS_STATHEAD, params=params, timeout=40)
    if debug:
        print(f"    [stathead:{mode}] status={r.status_code} order_by={order_by}")

    if r.status_code != 200 or not r.text:
        return pd.DataFrame()

    try:
        tables = pd.read_html(r.text)
    except ValueError:
        return pd.DataFrame()
    if not tables:
        return pd.DataFrame()

    df = max(tables, key=lambda d: d.shape[1]).copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if x and str(x) != "nan"]).strip() for tup in df.columns]
    df.columns = [snake(c) for c in df.columns]

    # drop self-header rows
    for col in ["team","school","school_name","team_name_abbr"]:
        if col in df.columns:
            df = df[df[col].astype(str).str.lower().ne(col)]

    if mode == "passing":
        team_col = pick(df, PASS_CANDS["team"]) or df.columns[0]
        conf_col = pick(df, PASS_CANDS["conf"])
        cmp_col  = pick(df, PASS_CANDS["cmp"])
        att_col  = pick(df, PASS_CANDS["att"])
        yds_col  = pick(df, PASS_CANDS["yds"])
        td_col   = pick(df, PASS_CANDS["td"])
        int_col  = pick(df, PASS_CANDS["int"])

        keep = [team_col] + [c for c in [conf_col, cmp_col, att_col, yds_col, td_col, int_col] if c]
        out = df[keep].rename(columns={
            team_col:"team",
            conf_col:"conference" if conf_col else "conference",
            cmp_col:"pass_cmp",
            att_col:"pass_att",
            yds_col:"pass_yds",
            td_col:"pass_td",
            int_col:"int"
        })
        for c in ["pass_cmp","pass_att","pass_yds","pass_td","int"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out["year"] = year
        for c in ["conference","pass_cmp","pass_att","pass_yds","pass_td","int"]:
            if c not in out.columns: out[c] = pd.NA
        return out[["year","team","conference","pass_cmp","pass_att","pass_yds","pass_td","int"]]

    else:
        team_col = pick(df, RUSH_CANDS["team"]) or df.columns[0]
        conf_col = pick(df, RUSH_CANDS["conf"])
        att_col  = pick(df, RUSH_CANDS["att"])
        yds_col  = pick(df, RUSH_CANDS["yds"])
        td_col   = pick(df, RUSH_CANDS["td"])

        keep = [team_col] + [c for c in [conf_col, att_col, yds_col, td_col] if c]
        out = df[keep].rename(columns={
            team_col:"team",
            conf_col:"conference" if conf_col else "conference",
            att_col:"rush_att",
            yds_col:"rush_yds",
            td_col:"rush_td",
        })
        for c in ["rush_att","rush_yds","rush_td"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out["year"] = year
        for c in ["conference","rush_att","rush_yds","rush_td"]:
            if c not in out.columns: out[c] = pd.NA
        return out[["year","team","conference","rush_att","rush_yds","rush_td"]]

# --------------------------- Public SR fallback (by team) ---------------------------
def _request_with_retry(url: str, session: requests.Session, max_tries: int=4, base_sleep: float=0.6) -> Optional[requests.Response]:
    for attempt in range(1, max_tries+1):
        try:
            r = session.get(url, headers=HEADERS_PUBLIC, timeout=35)
            if r.status_code == 200 and r.text:
                return r
            if r.status_code in (403, 429, 503):
                time.sleep(base_sleep * attempt + random.uniform(0.1, 0.4))
            else:
                time.sleep(0.2)
        except requests.RequestException:
            time.sleep(base_sleep * attempt)
    return None

def _discover_team_urls(year: int, session: requests.Session) -> List[str]:
    idx = f"{BASE_SR}/cfb/years/{year}.html"
    r = _request_with_retry(idx, session)
    if not r: 
        return []
    soup = BeautifulSoup(r.text, "lxml")
    urls = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if re.match(rf"^/cfb/schools/[^/]+/{year}\.html$", href):
            urls.add(BASE_SR + href)
    for com in soup.find_all(string=lambda t: isinstance(t, Comment)):
        s = str(com)
        for m in re.finditer(rf'href="(/cfb/schools/[^/]+/{year}\.html)"', s):
            urls.add(BASE_SR + m.group(1))
    return sorted(urls)

def _parse_team_page(url: str, year: int, session: requests.Session) -> Tuple[pd.DataFrame, str, str]:
    r = _request_with_retry(url, session)
    if not r:
        return pd.DataFrame(), "", ""
    soup = BeautifulSoup(r.text, "lxml")
    h1 = soup.find("h1")
    team_name = normalize_text(h1.get_text()) if h1 else url.split("/")[-2].replace("-", " ").title()
    conference = ""
    meta = soup.select_one("#meta") or soup.select_one("div#info")
    if meta:
        txt = normalize_text(meta.get_text(" "))
        m = re.search(r"Conference:\s*([A-Za-z0-9\-\s]+)", txt, flags=re.I)
        if m: conference = normalize_text(m.group(1))

    blocks = [r.text]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        s = str(c)
        if "<table" in s and any(k in s.lower() for k in ["team offense","offense","passing","rushing"]):
            blocks.append(s)

    best = None
    for block in blocks:
        try:
            dfs = pd.read_html(block)
        except ValueError:
            continue
        for d in dfs:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = ["_".join([str(x) for x in tup if x and str(x) != "nan"]).strip() for tup in d.columns]
            d.columns = [snake(c) for c in d.columns]
            if any(w in "_".join(d.columns) for w in ["pass","rush","cmp","yds"]):
                best = d; break
        if best is not None: break

    return (best.copy() if best is not None else pd.DataFrame()), team_name, conference

def _extract_pass_rush(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    d = df.copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="ignore")
    # prefer max pass yards row if present
    p_yds_col = pick(d, PASS_CANDS["yds"])
    if p_yds_col and pd.api.types.is_numeric_dtype(d[p_yds_col]):
        d = d.sort_values(p_yds_col, ascending=False)
    row = d.iloc[0] if len(d) else pd.Series(dtype="float64")

    p_cmp = pick(d, PASS_CANDS["cmp"])
    p_att = pick(d, PASS_CANDS["att"])
    p_td  = pick(d, PASS_CANDS["td"])
    p_int = pick(d, PASS_CANDS["int"])

    p = pd.Series(dtype="float64")
    if any([p_cmp, p_att, p_yds_col, p_td, p_int]):
        p = pd.Series({
            "pass_cmp": pd.to_numeric(row.get(p_cmp), errors="coerce") if p_cmp else None,
            "pass_att": pd.to_numeric(row.get(p_att), errors="coerce") if p_att else None,
            "pass_yds": pd.to_numeric(row.get(p_yds_col), errors="coerce") if p_yds_col else None,
            "pass_td":  pd.to_numeric(row.get(p_td), errors="coerce") if p_td else None,
            "int":      pd.to_numeric(row.get(p_int), errors="coerce") if p_int else None,
        })

    r_att = pick(d, RUSH_CANDS["att"])
    r_yds = pick(d, RUSH_CANDS["yds"])
    r_td  = pick(d, RUSH_CANDS["td"])

    r = pd.Series(dtype="float64")
    if any([r_att, r_yds, r_td]):
        r = pd.Series({
            "rush_att": pd.to_numeric(row.get(r_att), errors="coerce") if r_att else None,
            "rush_yds": pd.to_numeric(row.get(r_yds), errors="coerce") if r_yds else None,
            "rush_td":  pd.to_numeric(row.get(r_td), errors="coerce") if r_td else None,
        })

    return p, r

# --------------------------- Orchestrator ---------------------------
def run_team_totals_with_sort(
    start: int=2000,
    end: int=2024,
    sort_key_pass: str="pass_cmp",
    sort_key_rush: str="rush_att",
    sleep: float=0.35,
    write_csv: bool=True,
    debug: bool=False,
) -> None:
    """
    Primary runner with configurable Stathead 'Sort By'.
    Tries Stathead first (order_by=sort_key_*). If empty, falls back to public SR by-team scrape.
    """
    ensure_dir(PASS_DIR); ensure_dir(RUSH_DIR)

    with requests.Session() as sh_session, requests.Session() as sr_session:
        # (Optional) If you have a logged-in Stathead cookie, you can uncomment:
        # sh_session.headers.update({"Cookie": os.getenv("STATHEAD_COOKIE","")})
        sh_session.headers.update(HEADERS_STATHEAD)
        sr_session.headers.update(HEADERS_PUBLIC)

        for year in range(start, end+1):
            print(f"Year {year} … Stathead order_by(pass='{sort_key_pass}', rush='{sort_key_rush}')")
            # --- Stathead pass/rush ---
            p_df = _fetch_stathead(year, "passing", sort_key_pass, sh_session, debug=debug)
            r_df = _fetch_stathead(year, "rushing", sort_key_rush, sh_session, debug=debug)

            # --- Fallback to public SR if needed ---
            if (p_df is None or p_df.empty) or (r_df is None or r_df.empty):
                if debug: print("  [fallback] Using public SR team pages …")
                team_urls = _discover_team_urls(year, sr_session)
                passing_rows, rushing_rows = [], []
                for i, url in enumerate(team_urls, 1):
                    df, team_name, conf = _parse_team_page(url, year, sr_session)
                    if df.empty:
                        time.sleep(sleep); continue
                    p, r = _extract_pass_rush(df)
                    if not p.empty and p.notna().any():
                        row = {"year": year, "team": team_name, "conference": conf}
                        row.update({k: (None if pd.isna(v) else v) for k, v in p.to_dict().items()})
                        passing_rows.append(row)
                    if not r.empty and r.notna().any():
                        row = {"year": year, "team": team_name, "conference": conf}
                        row.update({k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()})
                        rushing_rows.append(row)
                    if i % 20 == 0: print(f"    … {i}/{len(team_urls)} processed")
                    time.sleep(sleep)
                if p_df is None or p_df.empty:
                    p_df = pd.DataFrame(passing_rows, columns=["year","team","conference","pass_cmp","pass_att","pass_yds","pass_td","int"])
                if r_df is None or r_df.empty:
                    r_df = pd.DataFrame(rushing_rows, columns=["year","team","conference","rush_att","rush_yds","rush_td"])

            # --- Write per-year CSVs ---
            pass_path = os.path.join(PASS_DIR, f"team_passing_{year}.csv")
            rush_path = os.path.join(RUSH_DIR, f"team_rushing_{year}.csv")

            if write_csv:
                if not p_df.empty:
                    p_df.to_csv(pass_path, index=False)
                    print(f"  ✓ Passing {len(p_df):,} rows → {pass_path}")
                else:
                    print("  ⚠ Passing empty")
                if not r_df.empty:
                    r_df.to_csv(rush_path, index=False)
                    print(f"  ✓ Rushing {len(r_df):,} rows → {rush_path}")
                else:
                    print("  ⚠ Rushing empty")

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--sort-key-pass", type=str, default="pass_cmp",
                    help="Stathead order_by for passing (e.g., team_name_abbr, pass_cmp, pass_yds …)")
    ap.add_argument("--sort-key-rush", type=str, default="rush_att",
                    help="Stathead order_by for rushing (e.g., team_name_abbr, rush_att, rush_yds …)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    run_team_totals_with_sort(
        start=args.start, end=args.end,
        sort_key_pass=args.sort_key_pass, sort_key_rush=args.sort_key_rush,
        sleep=args.sleep, write_csv=not args.no_save, debug=args.debug
    )
