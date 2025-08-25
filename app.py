# app.py
# Trading Hub: Insider Buys (SEC Form 4) + Auto COT (CFTC) + Prices + Market Scan
# Tip: put SEC_USER_AGENT in Streamlit secrets, or type one in sidebar (name + email).

import io
import time
import json
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

st.set_page_config(page_title="Trading Hub", page_icon="📊", layout="wide")

# ───────────────────────── Sidebar / Global Settings ─────────────────────────
st.sidebar.header("⚙️ Settings")

DEFAULT_UA = st.secrets.get("SEC_USER_AGENT", "")
ua = st.sidebar.text_input("SEC User-Agent (name + email)", value=DEFAULT_UA or "Your Name you@email.com")
if not ua or "@" not in ua:
    st.sidebar.warning("Enter a valid User-Agent (must include an email).")

lookback_days = st.sidebar.slider("Insider lookback (days)", 30, 365, 180, step=15)
sec_delay = st.sidebar.slider("SEC polite delay (sec/request)", 0.12, 0.50, 0.20, step=0.02)

st.sidebar.markdown("---")
st.sidebar.subheader("COT settings")
cot_window_weeks = st.sidebar.slider("COT Index lookback (weeks)", 52, 260, 156, step=4)
cot_thresh = st.sidebar.slider("Washed-out threshold (percentile)", 5, 40, 20, step=1)
price_for_cot = st.sidebar.text_input("Default price symbol for COT (e.g., GC=F, CL=F)", value="GC=F")

# Helpful mapping for quick price defaults by market name keywords
PRICE_HINTS = [
    ("GOLD", "GC=F"),
    ("CRUDE OIL", "CL=F"),
    ("WTI", "CL=F"),
    ("NATURAL GAS", "NG=F"),
    ("COPPER", "HG=F"),
    ("SILVER", "SI=F"),
    ("PLATINUM", "PL=F"),
    ("PALLADIUM", "PA=F"),
    ("CORN", "ZC=F"),
    ("SOYBEANS", "ZS=F"),
    ("WHEAT", "ZW=F"),
    ("LIVE CATTLE", "LE=F"),
    ("LEAN HOGS", "HE=F"),
    ("S&P", "^GSPC"),
    ("NASDAQ-100", "^NDX"),
]

def guess_price_symbol(market_name: str, fallback: str) -> str:
    up = (market_name or "").upper()
    for key, sym in PRICE_HINTS:
        if key in up:
            return sym
    return fallback

# ───────────────────────────── SEC / Insider utilities ───────────────────────
@st.cache_data(show_spinner=False, ttl=24*3600)
def load_company_tickers_json(user_agent: str) -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    data = r.json()
    rows = [{"cik": str(v["cik_str"]).zfill(10), "ticker": v["ticker"].upper(), "title": v["title"]}
            for _, v in data.items()]
    return pd.DataFrame(rows)

def to_cik_map(df_map: pd.DataFrame) -> dict:
    return {row["ticker"]: row["cik"] for _, row in df_map.iterrows()}

@st.cache_data(show_spinner=False, ttl=6*3600)
def get_submissions_json(cik: str, user_agent: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.json()

def accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")

@st.cache_data(show_spinner=False, ttl=6*3600)
def fetch_form4_xml(cik: str, accession: str, primary_doc: str, user_agent: str) -> str:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"
    r = requests.get(base, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.text

def parse_form4_buys(xml_text: str) -> list:
    """Return list of dicts for open-market BUY transactions (code 'P')."""
    res = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return res

    def tag_endswith(elem, suffix):
        return elem.tag.lower().endswith(suffix)

    for ndt in root.iter():
        if tag_endswith(ndt, "nonderivativetable"):
            for txn in ndt:
                tcode = None; shares = None; price = None; tdate = None
                for ch in txn.iter():
                    if tag_endswith(ch, "transactioncode"):
                        tcode = (ch.text or "").strip()
                    if tag_endswith(ch, "transactionshares"):
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                try: shares = float(vv.text)
                                except: pass
                    if tag_endswith(ch, "transactionpricepershare"):
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                try: price = float(vv.text)
                                except: pass
                    if tag_endswith(ch, "transactiondate"):
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                tdate = vv.text
                if tcode == "P" and shares and price:
                    res.append({"date": tdate, "shares": shares, "price": price, "value": shares * price})
    return res

def polite_sleep(s: float):
    try: time.sleep(s)
    except: pass

def scan_insider_buys(tickers: list, user_agent: str, lookback_days: int, delay: float) -> pd.DataFrame:
    m = load_company_tickers_json(user_agent)
    cm = to_cik_map(m)
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).date()
    out_rows = []

    for tk in [t.strip().upper() for t in tickers if t.strip()]:
        cik = cm.get(tk)
        if not cik:
            out_rows.append({"Ticker": tk, "CIK": None, "BuyUSD_lookback": 0.0, "Buys": 0,
                             "LatestBuyDate": None, "Note": "Ticker not in SEC map"})
            continue

        try:
            sub = get_submissions_json(cik, user_agent)
        except Exception as e:
            out_rows.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": 0.0, "Buys": 0,
                             "LatestBuyDate": None, "Note": f"submissions error: {e}"})
            polite_sleep(delay); continue

        forms = sub.get("filings", {}).get("recent", {})
        form_list = forms.get("form", [])
        acc_list = forms.get("accessionNumber", [])
        pdoc_list = forms.get("primaryDocument", [])
        fdate_list = forms.get("filingDate", [])

        total_usd = 0.0; buys = 0; last_date = None

        for form, acc, pdoc, fdate in zip(form_list, acc_list, pdoc_list, fdate_list):
            if form not in ("4", "4/A"): continue
            try: filed = datetime.strptime(fdate, "%Y-%m-%d").date()
            except: filed = None
            if filed and filed < cutoff: continue

            try:
                xml_text = fetch_form4_xml(cik, acc, pdoc, user_agent)
                for t in parse_form4_buys(xml_text):
                    try: t_d = datetime.strptime(t.get("date",""), "%Y-%m-%d").date()
                    except: t_d = filed
                    if t_d and t_d >= cutoff:
                        total_usd += float(t["value"]); buys += 1
                        if (not last_date) or (t_d > last_date): last_date = t_d
                polite_sleep(delay)
            except Exception:
                polite_sleep(delay); continue

        out_rows.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": total_usd, "Buys": buys,
                         "LatestBuyDate": last_date.isoformat() if last_date else None, "Note": ""})

    return pd.DataFrame(out_rows).sort_values("BuyUSD_lookback", ascending=False).reset_index(drop=True)

# ───────────────────────────── Prices / yfinance ─────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*3600)
def yf_prices(symbol: str, period: str="1y", interval: str="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]:"Date"}, inplace=True)
    return df

def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("Date").resample("W-FRI").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return w.reset_index()

# ───────────────────────────── COT (auto from CFTC) ──────────────────────────
# We try a couple of official endpoints (CSV/TXT). The Disaggregated Futures+Options Combined dataset is ideal.
CFTC_ENDPOINTS = [
    "https://www.cftc.gov/dea/newcot/DisaggComFutOpt.csv",  # primary
    "https://www.cftc.gov/dea/newcot/DisaggComFut.csv",     # futures-only fallback
    "https://www.cftc.gov/dea/newcot/DisaggComFutOpt.txt",  # sometimes served as txt
]

COT_COL_SYNONYMS = {
    "date": ["Report_Date_as_YYYY-MM-DD", "Report_Date", "Report_Date_as_YYYY_MM_DD"],
    "market": ["Market_and_Exchange_Names", "Market_and_Exchange_Name"],
    "code": ["CFTC_Contract_Market_Code", "CFTC_Contract_Market_Code_"],
    "mm_long": ["Mgr_Positions_Long_All", "Money_Manager_Long_All", "Money_Manager_Long_All_Positions"],
    "mm_short": ["Mgr_Positions_Short_All", "Money_Manager_Short_All", "Money_Manager_Short_All_Positions"],
    "oi": ["Open_Interest_All", "Open_Interest_All_All"],
}

def pick_col(cols, keys):
    lc = {c.lower(): c for c in cols}
    for k in keys:
        if k.lower() in lc:
            return lc[k.lower()]
    # fuzzy match
    for k in keys:
        for c in cols:
            if c.lower().startswith(k.lower()):
                return c
    return None

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cftc_disagg() -> pd.DataFrame:
    last_err = None
    for url in CFTC_ENDPOINTS:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            try:
                df = pd.read_csv(io.StringIO(r.text))
            except Exception:
                df = pd.read_csv(io.StringIO(r.text), sep=";")
            # Normalize columns (strip spaces)
            df.rename(columns=lambda x: str(x).strip().replace(" ", "_"), inplace=True)
            # Try to coerce date
            date_col = pick_col(df.columns, COT_COL_SYNONYMS["date"])
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CFTC fetch failed. Last error: {last_err}")

def standardize_cot(df: pd.DataFrame) -> pd.DataFrame:
    date_col   = pick_col(df.columns, COT_COL_SYNONYMS["date"])
    mkt_col    = pick_col(df.columns, COT_COL_SYNONYMS["market"])
    code_col   = pick_col(df.columns, COT_COL_SYNONYMS["code"])
    mm_long_c  = pick_col(df.columns, COT_COL_SYNONYMS["mm_long"])
    mm_short_c = pick_col(df.columns, COT_COL_SYNONYMS["mm_short"])
    oi_c       = pick_col(df.columns, COT_COL_SYNONYMS["oi"])

    needed = [date_col, mkt_col, code_col, mm_long_c, mm_short_c]
    if any(c is None for c in needed):
        raise ValueError("COT: could not detect required columns from CFTC file.")

    out = df[[date_col, mkt_col, code_col, mm_long_c, mm_short_c]].copy()
    out.columns = ["Report_Date", "Market", "Market_Code", "MM_LONG", "MM_SHORT"]
    out["Report_Date"] = pd.to_datetime(out["Report_Date"], errors="coerce")
    out = out.dropna(subset=["Report_Date"])
    out["MM_NET"] = pd.to_numeric(out["MM_LONG"], errors="coerce") - pd.to_numeric(out["MM_SHORT"], errors="coerce")
    if oi_c and oi_c in df.columns:
        out["Open_Interest"] = pd.to_numeric(df[oi_c], errors="coerce")
    else:
        out["Open_Interest"] = np.nan
    out = out.sort_values(["Market", "Report_Date"]).reset_index(drop=True)
    return out

def cot_index(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window, min_periods=4).min()
    hi = series.rolling(window, min_periods=4).max()
    return (series - lo) / (hi - lo + 1e-12)

def compute_cot_signals(df_std: pd.DataFrame, mkt: str, price_symbol_hint: str, window: int, pctl: float):
    mdf = df_std[df_std["Market"] == mkt].copy()
    if mdf.empty:
        return None, None
    mdf["COT_Idx"] = cot_index(mdf["MM_NET"], window)
    mdf["Washed"]  = (mdf["MM_NET"] < 0) | (mdf["COT_Idx"] <= (pctl/100.0))
    mdf["TurnUp"]  = mdf["MM_NET"].diff() > 0

    # price series for 10W SMA
    px_symbol = guess_price_symbol(mkt, price_symbol_hint or price_for_cot)
    px = yf_prices(px_symbol, period="5y", interval="1d")
    wpx = weekly_from_daily(px)
    wpx["SMA10W"] = wpx["Close"].rolling(10).mean()

    merged = pd.merge_asof(
        wpx.sort_values("Date"), mdf.sort_values("Report_Date"),
        left_on="Date", right_on="Report_Date", direction="backward"
    )
    merged["PriceUp"] = merged["Close"] > merged["SMA10W"]
    merged["Trigger"] = merged["Washed"] & merged["TurnUp"] & merged["PriceUp"]

    latest = merged.iloc[-1] if len(merged) else None
    return merged, latest

# ───────────────────────────── Market (S&P 500) list ─────────────────────────
@st.cache_data(show_spinner=False, ttl=24*3600)
def load_sp500_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return [str(t).upper().strip().replace(".", "-") for t in df["Symbol"].tolist()]

def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ───────────────────────────────────── UI ─────────────────────────────────────
st.title("📊 Trading Hub — Real Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏦 Insider (watchlist)",
    "📝 COT (auto from CFTC)",
    "📈 Prices",
    "🌐 Market Scan (S&P 500)"
])

# ------------------------------ Tab 1: Insider -------------------------------
with tab1:
    st.subheader("Insider Buys from SEC Form 4 (open-market 'P')")
    tickers_input = st.text_input("Enter comma-separated tickers", "AAPL,MSFT,NVDA")
    if st.button("Scan insiders", use_container_width=True):
        if not ua or "@" not in ua:
            st.error("Please enter a valid User-Agent (with email) in the sidebar.")
        else:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Contacting SEC…"):
                df = scan_insider_buys(tickers, ua, lookback_days, sec_delay)
            st.success("Done.")
            st.dataframe(df, use_container_width=True)
            if len(df):
                top = df.iloc[0]
                st.caption(f"Top: {top['Ticker']} — ${top['BuyUSD_lookback']:,.0f} over last {lookback_days} days; buys: {int(top['Buys'])}")

# ------------------------------ Tab 2: COT ----------------------------------
with tab2:
    st.subheader("Weekly COT — Auto fetch from CFTC (Disaggregated, Futures+Options Combined)")
    try:
        with st.spinner("Fetching CFTC Disaggregated data…"):
            raw = fetch_cftc_disagg()
        df_std = standardize_cot(raw)

        # Market selector with search
        search = st.text_input("Search market", "GOLD")
        markets = sorted(df_std["Market"].unique())
        if search:
            markets = [m for m in markets if search.upper() in m.upper()]
        mkt = st.selectbox("Choose market", markets)

        price_hint = st.text_input("Override price symbol (optional)", value=guess_price_symbol(mkt, price_for_cot))
        merged, latest = compute_cot_signals(df_std, mkt, price_hint, cot_window_weeks, cot_thresh)

        if merged is not None:
            st.dataframe(merged.tail(20), use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            last = merged.iloc[-1]
            c1.metric("MM Net", f"{int(last['MM_NET']):,}")
            c2.metric("COT Index", f"{(last['COT_Idx']*100):.1f}%")
            c3.metric("Price vs 10W", "Above" if last["PriceUp"] else "Below")
            c4.metric("Trigger", "✅" if last["Trigger"] else "—")
            st.line_chart(merged.set_index("Date")[["MM_NET"]])
            st.line_chart(merged.set_index("Date")[["Close","SMA10W"]])
        else:
            st.warning("No rows for that market.")
    except Exception as e:
        st.error(f"COT auto-fetch error: {e}")
        st.info("If issues persist, you can temporarily use the older upload approach we had earlier.")

# ------------------------------ Tab 3: Prices -------------------------------
with tab3:
    st.subheader("Quick price chart with SMA")
    px_ticker = st.text_input("Ticker (e.g., AAPL, GC=F, CL=F)", value="AAPL", key="px_ticker")
    period = st.selectbox("Period", ["6mo","1y","2y","5y"], index=1)
    if st.button("Load prices", key="load_px"):
        dfp = yf_prices(px_ticker, period=period, interval="1d")

        # Robust Date / Close normalization
        if "Date" not in dfp.columns:
            dfp.reset_index(inplace=True)
            for cand in ("index","Datetime","datetime","date"):
                if cand in dfp.columns:
                    dfp.rename(columns={cand:"Date"}, inplace=True)
                    break
        if "Close" not in dfp.columns and "close" in dfp.columns:
            dfp.rename(columns={"close":"Close"}, inplace=True)

        if "Date" not in dfp.columns:
            st.error("No Date column found in price data.")
        elif "Close" not in dfp.columns:
            st.error("No Close price found in price data.")
        else:
            dfp["SMA50"]  = dfp["Close"].rolling(50).mean()
            dfp["SMA200"] = dfp["Close"].rolling(200).mean()
            st.line_chart(dfp.set_index("Date")[["Close","SMA50","SMA200"]], use_container_width=True)
            st.dataframe(dfp.tail(20), use_container_width=True)

# --------------------------- Tab 4: Market Scan -----------------------------
with tab4:
    st.subheader("S&P 500 Insider Buy Leaderboard (batch-scanned)")
    st.caption("Loads S&P 500 from Wikipedia, scans in batches to respect SEC rate limits.")
    colA, colB = st.columns(2)
    batch_size = colA.slider("Batch size (tickers per batch)", 20, 200, 60, step=10)
    max_batches = colB.slider("How many batches to run now", 1, 20, 3, step=1)
    if st.button("Run S&P 500 Scan", use_container_width=True):
        if not ua or "@" not in ua:
            st.error("Please enter a valid User-Agent (with email) in the sidebar.")
        else:
            with st.spinner("Fetching S&P 500 list…"):
                sp = load_sp500_tickers()

            totals = []
            batches_run = 0
            for group in (g for g in (list(batch(sp, batch_size))[:max_batches])):
                df = scan_insider_buys(group, ua, lookback_days, sec_delay)
                totals.append(df)
                batches_run += 1
                st.write(f"Completed batch {batches_run}: scanned {len(group)} tickers")

            if totals:
                big = pd.concat(totals, ignore_index=True).sort_values("BuyUSD_lookback", ascending=False)
                st.success("Done.")
                st.dataframe(big.head(20), use_container_width=True)
                st.caption(f"Scanned {min(batch_size*max_batches, len(sp))} of {len(sp)} tickers "
                           f"({batches_run} batches). Increase 'max batches' and run again to cover more.")
            else:
                st.warning("No results (check connection or lower rate limits).")

st.markdown("---")
st.caption("Be polite to the SEC: ≤10 req/sec, real User-Agent. CFTC data is fetched directly; triggers: washed-out & turn up & price > 10W.")
