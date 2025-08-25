# app.py
# Trading Hub: Insider Buys (SEC Form 4) + Auto COT (CFTC Socrata) + Prices + S&P500 Scan
# Robust to environments without Streamlit: falls back to a lightweight shim so the module imports & tests run.
# To use full UI, install: streamlit, pandas, numpy, yfinance, requests, lxml, html5lib
# Streamlit secret recommended:  SEC_USER_AGENT = Your Name <you@example.com>

from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
import io, time, sys
import pandas as pd
import numpy as np
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Optional Streamlit import with graceful fallback
# ──────────────────────────────────────────────────────────────────────────────
HAS_STREAMLIT = True
try:
    import streamlit as st  # type: ignore
except Exception:
    HAS_STREAMLIT = False

    # Minimal shim for Streamlit API used below so this file can import & run tests
    class _DummyCache:
        def __call__(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

    class _DummySt:
        def __init__(self):
            self.sidebar = self
            self.secrets = {}
        def cache_data(self, *args, **kwargs):
            return _DummyCache()(*args, **kwargs)
        def set_page_config(self, **kwargs):
            pass
        def header(self, *a, **k):
            print(*a)
        def subheader(self, *a, **k):
            print(*a)
        def markdown(self, *a, **k):
            pass
        def text_input(self, label, value="", key=None):
            return value
        def slider(self, label, min_value=None, max_value=None, value=None, step=None):
            return value
        def warning(self, *a, **k):
            print("WARNING:", *a)
        def error(self, *a, **k):
            print("ERROR:", *a)
        def info(self, *a, **k):
            print("INFO:", *a)
        def success(self, *a, **k):
            print("SUCCESS:", *a)
        def title(self, *a, **k):
            print(*a)
        def tabs(self, labels):
            # Return list of shim objects; context manager not required here
            return [self for _ in labels]
        def button(self, label, use_container_width=False, key=None):
            return False
        def dataframe(self, df, use_container_width=False):
            try:
                print(df.tail(5))
            except Exception:
                print(df)
        def metric(self, label, value):
            print(f"{label}: {value}")
        def line_chart(self, df, use_container_width=False):
            pass
        def caption(self, *a, **k):
            print(*a)
        def columns(self, n):
            return [self for _ in range(n)]
        def spinner(self, text):
            class _Ctx:
                def __enter__(self):
                    print(text)
                def __exit__(self, exc_type, exc, tb):
                    return False
            return _Ctx()

    st = _DummySt()  # type: ignore

# yfinance import (optional) with fallback that raises a clearer error when used
try:
    import yfinance as yf
except Exception:
    class _YF:
        def download(self, *args, **kwargs):
            raise RuntimeError("yfinance is not installed in this environment; price features unavailable.")
    yf = _YF()  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# UI boilerplate (safe under shim)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trading Hub", page_icon="📊", layout="wide")

# ───────── Sidebar ─────────
st.sidebar.header("⚙️ Settings")
DEFAULT_UA = getattr(st, "secrets", {}).get("SEC_USER_AGENT", "") if hasattr(st, "secrets") else ""
ua = st.sidebar.text_input("User-Agent (name + email)", value=DEFAULT_UA or "Your Name <you@email.com>")
if not ua or "@" not in ua:
    st.sidebar.warning("Enter a valid User‑Agent (must include an email).")

lookback_days = st.sidebar.slider("Insider lookback (days)", 30, 365, 180, step=15)
sec_delay      = st.sidebar.slider("SEC polite delay (sec/request)", 0.12, 0.50, 0.20, step=0.02)

st.sidebar.markdown("---")
st.sidebar.subheader("COT settings")
cot_window_weeks = st.sidebar.slider("COT Index window (weeks)", 52, 260, 156, step=4)
cot_thresh       = st.sidebar.slider("Washed‑out threshold (%)", 5, 40, 20, step=1)
price_for_cot    = st.sidebar.text_input("Default price for COT (e.g., GC=F, CL=F)", value="GC=F")

PRICE_HINTS = [
    ("GOLD", "GC=F"), ("CRUDE OIL", "CL=F"), ("WTI", "CL=F"), ("NATURAL GAS", "NG=F"),
    ("COPPER", "HG=F"), ("SILVER", "SI=F"), ("PLATINUM", "PL=F"), ("PALLADIUM", "PA=F"),
    ("CORN", "ZC=F"), ("SOYBEANS", "ZS=F"), ("WHEAT", "ZW=F"),
    ("LIVE CATTLE", "LE=F"), ("LEAN HOGS", "HE=F"),
    ("S&P", "^GSPC"), ("NASDAQ-100", "^NDX")
]

def guess_price_symbol(market_name: str, fallback: str) -> str:
    up = (market_name or "").upper()
    for key, sym in PRICE_HINTS:
        if key in up:
            return sym
    return fallback

# ───────── Utility helpers ─────────

def polite_sleep(s):
    try:
        time.sleep(s)
    except Exception:
        pass

@st.cache_data(ttl=6*3600, show_spinner=False)
def yf_prices(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = pd.DataFrame(df).reset_index()
    if "Date" not in df.columns and len(df.columns) > 0:
        first_col = df.columns[0]
        df.rename(columns={first_col: "Date"}, inplace=True)
    return df


def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("weekly_from_daily: 'Date' column missing")
    df['Date'] = pd.to_datetime(df['Date'])
    agg = df.set_index("Date").resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    return agg.reset_index()


def cot_index(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window, min_periods=4).min()
    hi = series.rolling(window, min_periods=4).max()
    return (series - lo) / (hi - lo + 1e-12)

# ───────── SEC insiders ─────────

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_company_tickers_json(user_agent: str) -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    rows = []
    for _, v in r.json().items():
        rows.append({"cik": str(v.get("cik_str")).zfill(10), "ticker": str(v.get("ticker", "")).upper(), "title": v.get("title")})
    return pd.DataFrame(rows)


def to_cik_map(df_map: pd.DataFrame) -> dict:
    return {r["ticker"]: r["cik"] for _, r in df_map.iterrows()}


@st.cache_data(ttl=6*3600, show_spinner=False)
def get_submissions_json(cik: str, user_agent: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.json()


def accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")


@st.cache_data(ttl=6*3600, show_spinner=False)
def fetch_form4_xml(cik: str, accession: str, primary_doc: str, user_agent: str) -> str:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"
    r = requests.get(base, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.text


def parse_form4_buys(xml_text: str) -> list:
    res = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return res
    def tag_endswith(e, suf): return e.tag.lower().endswith(suf)
    for ndt in root.iter():
        if tag_endswith(ndt, "nonderivativetable"):
            for txn in ndt:
                tcode = shares = price = tdate = None
                for ch in txn.iter():
                    if tag_endswith(ch, "transactioncode"): tcode = (ch.text or "").strip()
                    if tag_endswith(ch, "transactionshares"):
                        for vv in ch:
                            if str(vv.tag).lower().endswith("value"):
                                try: shares = float(vv.text)
                                except Exception: pass
                    if tag_endswith(ch, "transactionpricepershare"):
                        for vv in ch:
                            if str(vv.tag).lower().endswith("value"):
                                try: price = float(vv.text)
                                except Exception: pass
                    if tag_endswith(ch, "transactiondate"):
                        for vv in ch:
                            if str(vv.tag).lower().endswith("value"): tdate = vv.text
                if tcode == "P" and shares and price:
                    res.append({"date": tdate, "shares": shares, "price": price, "value": shares * price})
    return res


def scan_insider_buys(tickers, user_agent, lookback_days, delay) -> pd.DataFrame:
    m = load_company_tickers_json(user_agent)
    cm = to_cik_map(m)
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).date()
    out = []
    for tk in [t.strip().upper() for t in tickers if t.strip()]:
        cik = cm.get(tk)
        if not cik:
            out.append({"Ticker": tk, "CIK": None, "BuyUSD_lookback": 0.0, "Buys": 0, "LatestBuyDate": None, "Note": "Ticker not in SEC map"})
            continue
        try:
            sub = get_submissions_json(cik, user_agent)
        except Exception as e:
            out.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": 0.0, "Buys": 0, "LatestBuyDate": None, "Note": f"submissions error: {e}"})
            polite_sleep(delay)
            continue
        forms = sub.get("filings", {}).get("recent", {})
        total_usd, buys, last_date = 0.0, 0, None
        for form, acc, pdoc, fdate in zip(forms.get("form", []), forms.get("accessionNumber", []), forms.get("primaryDocument", []), forms.get("filingDate", [])):
            if form not in ("4", "4/A"): continue
            try: filed = datetime.strptime(fdate, "%Y-%m-%d").date()
            except Exception: filed = None
            if filed and filed < cutoff: continue
            try:
                xml_text = fetch_form4_xml(cik, acc, pdoc, user_agent)
                for t in parse_form4_buys(xml_text):
                    try: t_d = datetime.strptime(t.get("date", ""), "%Y-%m-%d").date()
                    except Exception: t_d = filed
                    if t_d and t_d >= cutoff:
                        total_usd += float(t["value"])
                        buys += 1
                        if (not last_date) or (t_d > last_date): last_date = t_d
                polite_sleep(delay)
            except Exception:
                polite_sleep(delay)
                continue
        out.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": total_usd, "Buys": buys, "LatestBuyDate": last_date.isoformat() if last_date else None, "Note": ""})
    return pd.DataFrame(out).sort_values("BuyUSD_lookback", ascending=False).reset_index(drop=True)

# ───────── COT (auto via CFTC Public Reporting / Socrata) ─────────

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_cftc_disagg(user_agent: str) -> pd.DataFrame:
    base = "https://publicreporting.cftc.gov/resource/gr4m-cvuh.csv"
    params = {"$limit": 500000, "$order": "report_date_as_yyyy_mm_dd ASC"}
    
    # **FIX**: Use a more comprehensive, browser-like header to avoid 403 Forbidden errors.
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://publicreporting.cftc.gov/data/",
        "Connection": "keep-alive",
    }
    r = requests.get(base, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}
    def need(name):
        if name not in lc: raise RuntimeError(f"CFTC columns changed: missing {name}")
        return lc[name]
    date_c, mkt_c, long_c, short_c = need("report_date_as_yyyy_mm_dd"), need("market_and_exchange_names"), need("mgr_positions_long_all"), need("mgr_positions_short_all")
    out = df[[date_c, mkt_c, long_c, short_c]].copy()
    out.rename(columns={date_c: "Date", mkt_c: "Market", long_c: "MM_LONG", short_c: "MM_SHORT"}, inplace=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["MM_LONG", "MM_SHORT"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def prepare_cot_auto(cftc_df: pd.DataFrame, market_filter: str, price_symbol: str, window: int, pctl: float) -> pd.DataFrame:
    sub = cftc_df[cftc_df["Market"].str.contains(market_filter, case=False, na=False)].copy()
    if sub.empty: raise ValueError("No rows found for that market filter.")
    sub = sub[["Date", "Market", "MM_LONG", "MM_SHORT"]].sort_values("Date").reset_index(drop=True)
    sub["MM_NET"] = sub["MM_LONG"] - sub["MM_SHORT"]
    px = yf_prices(price_symbol, period="10y", interval="1d")
    wpx = weekly_from_daily(px)
    wpx["SMA10W"] = wpx["Close"].rolling(10).mean()
    merged = pd.merge_asof(wpx.sort_values("Date"), sub.sort_values("Date"), on="Date", direction="backward")
    merged["COT_Idx"] = cot_index(merged["MM_NET"], window)
    merged["Washed"]  = (merged["MM_NET"] < 0) | (merged["COT_Idx"] <= (pctl/100.0))
    merged["TurnUp"]  = merged["MM_NET"].diff() > 0
    merged["PriceUp"] = merged["Close"] > merged["SMA10W"]
    merged["Trigger"] = merged["Washed"] & merged["TurnUp"] & merged["PriceUp"]
    return merged

# ───────── S&P500 list (used in Streamlit tab) ─────────

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_sp500_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]["Symbol"].str.upper().str.replace(".", "-", regex=False).tolist()

def batch(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

# ───────── UI (only when Streamlit is installed) ─────────

if HAS_STREAMLIT:
    st.title("📊 Trading Hub — Real Data")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏦 Insider (watchlist)",
        "📝 COT (auto from CFTC)",
        "🌊 COT Washout Strategy",
        "📈 Prices",
        "🌐 Market Scan (S&P 500)",
    ])

    # Tab 1 — Insider watchlist
    with tab1:
        st.subheader("SEC Form 4 open‑market buys ('P')")
        tickers_input = st.text_input("Enter comma‑separated tickers", "AAPL,MSFT,NVDA")
        if st.button("Scan insiders", use_container_width=True):
            if not ua or "@" not in ua: st.error("Enter a valid User‑Agent in the sidebar.")
            else:
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                with st.spinner("Contacting SEC…"):
                    df = scan_insider_buys(tickers, ua, lookback_days, sec_delay)
                st.success("Done.")
                st.dataframe(df, use_container_width=True)
                if len(df):
                    top = df.iloc[0]
                    st.caption(f"Top: {top['Ticker']} — ${top['BuyUSD_lookback']:,.0f} in {lookback_days}d; buys: {int(top['Buys'])}")

    # Tab 2 — COT auto (Socrata)
    with tab2:
        st.subheader("Disaggregated Futures+Options Combined (CFTC Public Reporting)")
        try:
            with st.spinner("Fetching CFTC Public Reporting dataset…"):
                if not ua or "@" not in ua:
                    st.error("Enter a valid User-Agent in the sidebar to fetch CFTC data.")
                    cftc_df = pd.DataFrame()
                else:
                    cftc_df = fetch_cftc_disagg(ua)
            if not cftc_df.empty:
                search = st.text_input("Search market", "GOLD")
                markets = sorted(cftc_df["Market"].dropna().unique())
                if search: markets = [m for m in markets if search.upper() in m.upper()]
                if not markets: st.warning("No markets match your search.")
                else:
                    mkt_filter = st.selectbox("Choose/Filter market (substring)", markets, index=0)
                    price_hint = st.text_input("Override price symbol (optional)", value=guess_price_symbol(mkt_filter, price_for_cot))
                    if st.button("Run COT scan", use_container_width=True):
                        merged = prepare_cot_auto(cftc_df, mkt_filter, price_hint, cot_window_weeks, cot_thresh)
                        st.dataframe(merged.tail(20), use_container_width=True)
                        last = merged.iloc[-1]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MM Net", f"{int(last['MM_NET']):,}")
                        c2.metric("COT Index", f"{last['COT_Idx']*100:.1f}%")
                        c3.metric("Price vs 10W", "Above" if last["PriceUp"] else "Below")
                        c4.metric("Trigger", "✅" if last["Trigger"] else "—")
                        st.line_chart(merged.set_index("Date")[["MM_NET"]])
                        st.line_chart(merged.set_index("Date")[["Close", "SMA10W"]])
        except Exception as e:
            st.error(f"COT error: {e}")

    # Tab 3 - COT Washout Strategy
    with tab3:
        st.subheader("COT Washout Strategy Scanner")
        st.markdown("This tool scans all markets to find those where Managed Money sentiment is extremely bearish (washed out) but the price has started to show strength. This can be a powerful contrarian buy signal.")
        if st.button("Scan All Markets for Washouts", use_container_width=True, key="washout_scan"):
            if not ua or "@" not in ua: st.error("Enter a valid User-Agent in the sidebar.")
            else:
                with st.spinner("Fetching latest CFTC data for all markets..."):
                    all_cot = fetch_cftc_disagg(ua)
                
                results = []
                markets = all_cot['Market'].unique()
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, market in enumerate(markets):
                    market_df = all_cot[all_cot['Market'] == market].copy()
                    if len(market_df) < cot_window_weeks: continue
                    
                    market_df['MM_NET'] = market_df['MM_LONG'] - market_df['MM_SHORT']
                    market_df['COT_Idx'] = cot_index(market_df['MM_NET'], cot_window_weeks)
                    
                    latest = market_df.iloc[-1]
                    previous = market_df.iloc[-2]

                    is_washed_out = (latest['MM_NET'] < 0) or (latest['COT_Idx'] <= (cot_thresh / 100.0))
                    is_turning_up = latest['MM_NET'] > previous['MM_NET']

                    if is_washed_out and is_turning_up:
                        status_text.text(f"Found potential washout in {market}. Checking price...")
                        try:
                            price_sym = guess_price_symbol(market, price_for_cot)
                            px = yf_prices(price_sym, period="1y", interval="1d")
                            wpx = weekly_from_daily(px)
                            if len(wpx) < 10: continue
                            wpx["SMA10W"] = wpx["Close"].rolling(10).mean()
                            latest_price = wpx.iloc[-1]
                            
                            if latest_price['Close'] > latest_price['SMA10W']:
                                results.append({
                                    "Market": market,
                                    "Report_Date": latest['Date'].date(),
                                    "MM_Net": f"{latest['MM_NET']:,.0f}",
                                    "COT_Index": f"{latest['COT_Idx']*100:.1f}%",
                                    "Price_Symbol": price_sym,
                                    "Status": "✅ Triggered"
                                })
                        except Exception:
                            continue # Skip if price check fails
                    
                    progress_bar.progress((i + 1) / len(markets))
                
                status_text.text("Scan complete.")
                st.success(f"Found {len(results)} markets triggering the washout signal.")
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # Tab 4 — Prices
    with tab4:
        st.subheader("Quick price chart with SMA")
        px_ticker = st.text_input("Ticker (e.g., AAPL, GC=F, CL=F)", value="AAPL", key="px_ticker")
        period = st.selectbox("Period", ["6mo","1y","2y","5y","10y"], index=2)
        if st.button("Load prices", key="load_px"):
            dfp = yf_prices(px_ticker, period=period, interval="1d")
            if "Date" not in dfp.columns:
                dfp = dfp.reset_index()
                for cand in ("index", "Datetime", "datetime", "date"):
                    if cand in dfp.columns:
                        dfp.rename(columns={cand: "Date"}, inplace=True)
                        break
            if "Close" not in dfp.columns and "close" in dfp.columns:
                dfp.rename(columns={"close": "Close"}, inplace=True)
            if "Date" in dfp.columns and "Close" in dfp.columns:
                dfp["SMA50"] = dfp["Close"].rolling(50).mean()
                dfp["SMA200"] = dfp["Close"].rolling(200).mean()
                st.line_chart(dfp.set_index("Date")[["Close","SMA50","SMA200"]], use_container_width=True)
                st.dataframe(dfp.tail(20), use_container_width=True)
            else:
                st.error("Price data missing Date/Close columns.")

    # Tab 5 — S&P 500 Market Scan
    with tab5:
        st.subheader("S&P 500 Insider Buy Leaderboard (batch‑scanned)")
        st.caption("Loads S&P 500 from Wikipedia, scans in batches to respect SEC limits.")
        colA, colB = st.columns(2)
        batch_size = colA.slider("Batch size", 20, 200, 60, step=10)
        max_batches = colB.slider("Batches to run now", 1, 20, 3, step=1)
        if st.button("Run S&P 500 scan", use_container_width=True):
            if not ua or "@" not in ua: st.error("Enter a valid User‑Agent with email in the sidebar.")
            else:
                with st.spinner("Loading S&P 500 list…"): sp = load_sp500_tickers()
                results, batches = [], 0
                for grp in batch(sp, batch_size):
                    if batches >= max_batches: break
                    df = scan_insider_buys(grp, ua, lookback_days, sec_delay)
                    results.append(df)
                    batches += 1
                    st.write(f"Completed batch {batches} (scanned {len(grp)} tickers)")
                if results:
                    big = pd.concat(results, ignore_index=True).sort_values("BuyUSD_lookback", ascending=False)
                    st.success("Done.")
                    st.dataframe(big.head(20), use_container_width=True)
                else:
                    st.warning("No results. Try more batches or a larger delay.")

# ───────── Tests (run when executed without Streamlit UI) ─────────

import unittest

class TestTradingHubCore(unittest.TestCase):
    def test_guess_price_symbol(self):
        self.assertEqual(guess_price_symbol("GOLD - COMMODITY EXCHANGE INC.", "X"), "GC=F")
        self.assertEqual(guess_price_symbol("Unknown Market", "CL=F"), "CL=F")

    def test_cot_index_bounds(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7])
        idx = cot_index(s, window=5)
        self.assertTrue(((idx >= 0) & (idx <= 1)).fillna(True).all())

    def test_prepare_cot_auto_with_dummy_prices(self):
        dates = pd.date_range("2023-01-06", periods=12, freq="W-FRI")
        cftc = pd.DataFrame({ "Date": dates, "Market": ["GOLD - COMMODITY EXCHANGE INC."] * len(dates), "MM_LONG": np.linspace(200_000, 220_000, len(dates)), "MM_SHORT": np.linspace(210_000, 200_000, len(dates)) })
        def _fake_prices(symbol: str, period="10y", interval="1d"):
            d = pd.date_range(dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=7), freq="D")
            close = np.linspace(1800, 1900, len(d))
            return pd.DataFrame({"Date": d, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 0})
        globals()["yf_prices"] = _fake_prices
        merged = prepare_cot_auto(cftc, "GOLD", "GC=F", window=6, pctl=20)
        for col in ["MM_NET", "COT_Idx", "Washed", "TurnUp", "PriceUp", "Trigger"]: self.assertIn(col, merged.columns)
        self.assertTrue(len(merged) >= len(dates) - 1)

if __name__ == "__main__":
    if not HAS_STREAMLIT:
        unittest.main(argv=[sys.argv[0]], exit=False)
        print("\nStreamlit not detected. Core logic tests executed.\nTo use the full UI, install 'streamlit' and run: streamlit run app.py")
