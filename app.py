# app.py
# Streamlit Trading Hub: Insider Buys (SEC Form 4) + Prices + COT
# - Live insider buys via data.sec.gov (parses form4.xml)
# - Prices via yfinance
# - COT scanner: upload CFTC/Tradingster CSV OR fetch from a CSV URL
# Notes:
#   • Use a real User-Agent (name + email) to avoid SEC rate limiting.
#   • Keep watchlists modest (e.g., 5–50 tickers) to stay polite with SEC.

import time
import io
import re
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

# ─────────────────────────────── Settings / Sidebar ───────────────────────────────
st.sidebar.header("⚙️ Settings")

# SEC requires a real User-Agent; let user type one if secrets not set
DEFAULT_UA = st.secrets.get("SEC_USER_AGENT", "")
ua = st.sidebar.text_input("SEC User-Agent (name + email)", value=DEFAULT_UA or "YourName your@email.com")
if not ua or "@" not in ua:
    st.sidebar.warning("Enter a valid User-Agent (include an email) so SEC allows requests.")

scan_mode = st.sidebar.selectbox("Insider scan mode", ["Single ticker", "Watchlist (comma-separated)"])
tickers_input = st.sidebar.text_area("Ticker(s) (e.g., AAPL or AAPL,MSFT,NVDA)", value="AAPL")

lookback_days = st.sidebar.slider("Insider lookback (days)", 30, 365, 180, step=15)
sec_delay = st.sidebar.slider("SEC polite delay (seconds/request)", 0.12, 0.50, 0.20, step=0.02)

st.sidebar.markdown("---")
st.sidebar.subheader("COT data")
cot_mode = st.sidebar.radio("COT source", ["Upload CSV", "Fetch from CSV URL"])
cot_url = st.sidebar.text_input("If 'Fetch from CSV URL', paste URL here", value="")
price_for_cot = st.sidebar.text_input("Price symbol for 10W SMA (e.g., GC=F, CL=F, ES=F)", value="GC=F")
cot_window_weeks = st.sidebar.slider("COT Index lookback (weeks)", 52, 260, 156, step=4)
cot_thresh = st.sidebar.slider("Washed-out threshold (percentile)", 5, 40, 20, step=1)
time_stop_w = st.sidebar.slider("Time stop (weeks, 0=off)", 0, 26, 12, step=1)

# ─────────────────────────────── Utilities ───────────────────────────────
@st.cache_data(show_spinner=False, ttl=24*3600)
def load_company_tickers_json(user_agent: str) -> pd.DataFrame:
    # Official mapping file published by SEC (includes cik_str, ticker, title)
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    data = r.json()
    rows = []
    for _, v in data.items():
        rows.append({"cik": str(v["cik_str"]).zfill(10), "ticker": v["ticker"].upper(), "title": v["title"]})
    return pd.DataFrame(rows)

def to_cik_map(df_map: pd.DataFrame) -> dict:
    return {row["ticker"]: row["cik"] for _, row in df_map.iterrows()}

@st.cache_data(show_spinner=False, ttl=24*3600)
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
    """
    Returns a list of dicts for open-market BUY transactions (transactionCode == 'P')
    from the <nonDerivativeTable>.
    """
    res = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return res

    # Namespaces sometimes present; ignore by matching localname
    def tag_endswith(elem, suffix):
        return elem.tag.lower().endswith(suffix)

    for ndt in root.iter():
        if tag_endswith(ndt, "nonderivativetable"):
            for txn in ndt:
                # Only consider entries with transactionCode 'P'
                tcode = None
                shares = None
                price = None
                tdate = None
                for ch in txn.iter():
                    if tag_endswith(ch, "transactioncode"):
                        tcode = (ch.text or "").strip()
                    if tag_endswith(ch, "transactionshares"):
                        # usually inside <value>
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                try:
                                    shares = float(vv.text)
                                except Exception:
                                    pass
                    if tag_endswith(ch, "transactionpricepershare"):
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                try:
                                    price = float(vv.text)
                                except Exception:
                                    pass
                    if tag_endswith(ch, "transactiondate"):
                        for vv in ch:
                            if tag_endswith(vv, "value"):
                                tdate = vv.text
                if tcode == "P" and shares and price:
                    res.append({"date": tdate, "shares": shares, "price": price, "value": shares * price})
    return res

def polite_sleep(s: float):
    try:
        time.sleep(s)
    except Exception:
        pass

# ─────────────────────────────── Insider Buys Scanner ───────────────────────────────
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

        # Pull submissions index to list recent 4/4A filings
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

        total_usd = 0.0
        buys = 0
        last_date = None

        # Iterate newest → oldest
        for form, acc, pdoc, fdate in zip(form_list, acc_list, pdoc_list, fdate_list):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = datetime.strptime(fdate, "%Y-%m-%d").date()
            except Exception:
                filed = None
            if filed and filed < cutoff:
                # older than lookback window; skip remaining
                continue
            try:
                xml_text = fetch_form4_xml(cik, acc, pdoc, user_agent)
                txns = parse_form4_buys(xml_text)
                for t in txns:
                    # Use transaction date for latest flag
                    try:
                        t_d = datetime.strptime(t.get("date",""), "%Y-%m-%d").date()
                    except Exception:
                        t_d = filed
                    if t_d and t_d >= cutoff:
                        total_usd += float(t["value"])
                        buys += 1
                        if (not last_date) or (t_d > last_date):
                            last_date = t_d
                polite_sleep(delay)
            except Exception:
                polite_sleep(delay)
                continue

        out_rows.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": total_usd, "Buys": buys,
                         "LatestBuyDate": last_date.isoformat() if last_date else None,
                         "Note": ""})

    df = pd.DataFrame(out_rows).sort_values("BuyUSD_lookback", ascending=False).reset_index(drop=True)
    return df

# ─────────────────────────────── Price helpers ───────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*3600)
def yf_prices(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.reset_index()
    if "Date" not in df.columns:
        # yfinance sometimes names the index differently
        df.rename(columns={df.columns[0]:"Date"}, inplace=True)
    return df

def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("Date").resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return w.reset_index()

# ─────────────────────────────── COT Scanner ───────────────────────────────
def cot_index(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window).min()
    hi = series.rolling(window).max()
    return (series - lo) / (hi - lo + 1e-12)

def load_cot_from_csv_file(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Date"])
    return df

def load_cot_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    return pd.read_csv(buf, parse_dates=["Date"])

def prepare_cot(df: pd.DataFrame, price_symbol: str, window: int) -> pd.DataFrame:
    """
    Expects a CSV with columns at least:
      Date, Managed Money Longs, Managed Money Shorts   (or 'MM_LONG','MM_SHORT')
    """
    cols = {c.lower(): c for c in df.columns}
    # flexible column detection
    def pick(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_long = pick("Managed Money Longs", "MM_LONG", "managed_money_longs")
    c_short= pick("Managed Money Shorts","MM_SHORT","managed_money_shorts")

    if not c_long or not c_short or "Date" not in df.columns:
        raise ValueError("COT CSV must include Date, Managed Money Longs, Managed Money Shorts columns")

    cotw = (df[["Date", c_long, c_short]]
            .rename(columns={c_long:"MM_LONG", c_short:"MM_SHORT"})
            .sort_values("Date").reset_index(drop=True))
    cotw["MM_NET"] = cotw["MM_LONG"] - cotw["MM_SHORT"]

    # weekly price + 10W SMA
    px = yf_prices(price_symbol, period="5y", interval="1d")
    wpx = weekly_from_daily(px)
    wpx["SMA10W"] = wpx["Close"].rolling(10).mean()

    merged = pd.merge_asof(wpx.sort_values("Date"), cotw.sort_values("Date"), on="Date", direction="backward")
    merged["COT_Idx"] = cot_index(merged["MM_NET"], window)  # 0..1
    merged["Washed"]  = (merged["MM_NET"] < 0) | (merged["COT_Idx"] <= (cot_thresh/100.0))
    merged["TurnUp"]  = merged["MM_NET"].diff() > 0
    merged["PriceUp"] = merged["Close"] > merged["SMA10W"]
    merged["Trigger"] = merged["Washed"] & merged["TurnUp"] & merged["PriceUp"]
    return merged

# ─────────────────────────────── UI / Tabs ───────────────────────────────
st.title("📊 Trading Hub — Real Data")

tab1, tab2, tab3 = st.tabs(["🏦 Insider Buys (SEC Form 4)", "📝 COT Scanner", "📈 Prices & SMA"])

# ── Tab 1: Insider Buys ────────────────────────────────────────────────────
with tab1:
    st.subheader("Largest insider buys over your lookback window")
    if scan_mode == "Single ticker":
        tickers = [tickers_input.strip()]
    else:
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    if st.button("Scan insiders", use_container_width=True):
        if not ua or "@" not in ua:
            st.error("Please enter a valid User-Agent with an email in the sidebar.")
        else:
            with st.spinner("Contacting SEC…"):
                df = scan_insider_buys(tickers, ua, lookback_days, sec_delay)
            st.success("Done.")
            st.dataframe(df, use_container_width=True)
            top = df.iloc[0] if len(df)>0 else None
            if top is not None:
                st.caption(f"Top: {top['Ticker']} — ${top['BuyUSD_lookback']:,.0f} over last {lookback_days} days; buys: {int(top['Buys'])}")

    st.markdown("""
**Tips**
- Keep lists reasonable (e.g., 5–50 tickers) to respect SEC’s request limits.
- Only **open-market buys** (transactionCode `P`) are counted. Grants/awards are ignored.
- You can re-run anytime; results are cached briefly for speed.
""")

# ── Tab 2: COT Scanner ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Weekly COT extremes + turn + price filter")
    uploaded = None
    if cot_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload COT CSV (must include Date, Managed Money Longs, Managed Money Shorts)", type=["csv"])
    else:
        if st.button("Fetch COT CSV from URL"):
            pass

    df_cot = None
    try:
        if cot_mode == "Upload CSV" and uploaded is not None:
            df_cot = load_cot_from_csv_file(uploaded)
        elif cot_mode == "Fetch from CSV URL" and cot_url.strip():
            df_cot = load_cot_from_url(cot_url.strip())
        if df_cot is not None:
            merged = prepare_cot(df_cot, price_for_cot, cot_window_weeks)
            st.dataframe(merged.tail(20), use_container_width=True)
            last = merged.iloc[-1]
            colA, colB, colC, colD = st.columns(4)
            colA.metric("MM Net", f"{int(last['MM_NET']):,}")
            colB.metric("COT Index", f"{last['COT_Idx']*100:.1f}%")
            colC.metric("Price vs 10W", "Above" if last["PriceUp"] else "Below")
            colD.metric("Trigger", "✅" if last["Trigger"] else "—")
            st.line_chart(merged.set_index("Date")[["MM_NET"]])
            st.line_chart(merged.set_index("Date")[["Close","SMA10W"]])
            st.caption("Trigger rule: Washed-out (COT index <= threshold or net<0) AND turning up WoW AND price > 10W SMA.")
        else:
            st.info("Upload a COT CSV or paste a CSV URL, then the app will compute signals.")
    except Exception as e:
        st.error(f"COT error: {e}")

# ── Tab 3: Prices ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Quick price chart with SMA")
    px_ticker = st.text_input("Ticker (e.g., AAPL, GC=F, CL=F)", value="AAPL", key="px_ticker")
    period = st.selectbox("Period", ["6mo","1y","2y","5y"], index=1)
    if st.button("Load prices", key="load_px"):
        dfp = yf_prices(px_ticker, period=period, interval="1d")
        if len(dfp) == 0:
            st.error("No data returned for that ticker.")
        else:
            dfp["SMA50"] = dfp["Close"].rolling(50).mean()
            dfp["SMA200"] = dfp["Close"].rolling(200).mean()
            st.line_chart(dfp.set_index("Date")[["Close","SMA50","SMA200"]])
            st.dataframe(dfp.tail(20), use_container_width=True)

st.markdown("---")
st.caption("Be nice to the SEC: keep requests <=10/sec, use a valid User-Agent, and avoid scanning the whole market at once.")
