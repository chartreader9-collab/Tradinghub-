# app.py
# Trading Hub: Insider Buys (SEC Form 4) + Auto COT (CFTC) + Prices + S&P500 Scan
# Put SEC_USER_AGENT in Secrets or enter in the sidebar (name + email).

import io, time, requests, pandas as pd, numpy as np, streamlit as st, yfinance as yf
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

st.set_page_config(page_title="Trading Hub", page_icon="📊", layout="wide")

# ───────── Sidebar ─────────
st.sidebar.header("⚙️ Settings")
DEFAULT_UA = st.secrets.get("SEC_USER_AGENT", "")
ua = st.sidebar.text_input("SEC User-Agent (name + email)", value=DEFAULT_UA or "Your Name <you@email.com>")
if not ua or "@" not in ua:
    st.sidebar.warning("Enter a valid User‑Agent (must include an email).")

lookback_days = st.sidebar.slider("Insider lookback (days)", 30, 365, 180, step=15)
sec_delay     = st.sidebar.slider("SEC polite delay (sec/request)", 0.12, 0.50, 0.20, step=0.02)

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
        if key in up: return sym
    return fallback

def polite_sleep(s): 
    try: time.sleep(s)
    except: pass

# ───────── Prices ─────────
@st.cache_data(ttl=6*3600, show_spinner=False)
def yf_prices(symbol: str, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.reset_index()
    if "Date" not in df.columns: df.rename(columns={df.columns[0]:"Date"}, inplace=True)
    return df

def weekly_from_daily(df: pd.DataFrame):
    return df.set_index("Date").resample("W-FRI").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna().reset_index()

def cot_index(series: pd.Series, window: int):
    lo = series.rolling(window, min_periods=4).min()
    hi = series.rolling(window, min_periods=4).max()
    return (series - lo) / (hi - lo + 1e-12)

# ───────── SEC insiders ─────────
@st.cache_data(ttl=24*3600, show_spinner=False)
def load_company_tickers_json(user_agent: str):
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    rows=[]
    for _, v in r.json().items():
        rows.append({"cik": str(v["cik_str"]).zfill(10), "ticker": v["ticker"].upper(), "title": v["title"]})
    return pd.DataFrame(rows)

def to_cik_map(df_map): return {r["ticker"]: r["cik"] for _, r in df_map.iterrows()}

@st.cache_data(ttl=6*3600, show_spinner=False)
def get_submissions_json(cik: str, user_agent: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.json()

def accession_no_dashes(acc): return acc.replace("-", "")

@st.cache_data(ttl=6*3600, show_spinner=False)
def fetch_form4_xml(cik: str, accession: str, primary_doc: str, user_agent: str):
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"
    r = requests.get(base, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.text

def parse_form4_buys(xml_text: str):
    res=[]
    try: root = ET.fromstring(xml_text)
    except Exception: return res
    def tag_endswith(e, suf): return e.tag.lower().endswith(suf)
    for ndt in root.iter():
        if tag_endswith(ndt, "nonderivativetable"):
            for txn in ndt:
                tcode=shares=price=None; tdate=None
                for ch in txn.iter():
                    if tag_endswith(ch, "transactioncode"): tcode=(ch.text or "").strip()
                    if tag_endswith(ch, "transactionshares"):
                        for vv in ch:
                            if vv.tag.lower().endswith("value"):
                                try: shares=float(vv.text)
                                except: pass
                    if tag_endswith(ch, "transactionpricepershare"):
                        for vv in ch:
                            if vv.tag.lower().endswith("value"):
                                try: price=float(vv.text)
                                except: pass
                    if tag_endswith(ch, "transactiondate"):
                        for vv in ch:
                            if vv.tag.lower().endswith("value"): tdate=vv.text
                if tcode=="P" and shares and price:
                    res.append({"date": tdate, "shares": shares, "price": price, "value": shares*price})
    return res

def scan_insider_buys(tickers, user_agent, lookback_days, delay):
    m = load_company_tickers_json(user_agent); cm = to_cik_map(m)
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).date()
    out=[]
    for tk in [t.strip().upper() for t in tickers if t.strip()]:
        cik = cm.get(tk)
        if not cik:
            out.append({"Ticker": tk, "CIK": None, "BuyUSD_lookback": 0.0, "Buys": 0,
                        "LatestBuyDate": None, "Note":"Ticker not in SEC map"})
            continue
        try: sub = get_submissions_json(cik, user_agent)
        except Exception as e:
            out.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": 0.0, "Buys": 0,
                        "LatestBuyDate": None, "Note": f"submissions error: {e}"})
            polite_sleep(delay); continue

        forms = sub.get("filings", {}).get("recent", {})
        total_usd=0.0; buys=0; last_date=None
        for form, acc, pdoc, fdate in zip(forms.get("form",[]), forms.get("accessionNumber",[]),
                                          forms.get("primaryDocument",[]), forms.get("filingDate",[])):
            if form not in ("4","4/A"): continue
            try: filed = datetime.strptime(fdate,"%Y-%m-%d").date()
            except: filed = None
            if filed and filed < cutoff: continue
            try:
                xml_text = fetch_form4_xml(cik, acc, pdoc, user_agent)
                for t in parse_form4_buys(xml_text):
                    try: t_d = datetime.strptime(t.get("date",""), "%Y-%m-%d").date()
                    except: t_d = filed
                    if t_d and t_d >= cutoff:
                        total_usd += float(t["value"]); buys += 1
                        if (not last_date) or (t_d > last_date): last_date=t_d
                polite_sleep(delay)
            except Exception:
                polite_sleep(delay); continue
        out.append({"Ticker": tk, "CIK": cik, "BuyUSD_lookback": total_usd, "Buys": buys,
                    "LatestBuyDate": last_date.isoformat() if last_date else None, "Note": ""})
    return pd.DataFrame(out).sort_values("BuyUSD_lookback", ascending=False).reset_index(drop=True)

# ───────── COT (auto from CFTC) ─────────
CFTC_ENDPOINTS = [
    "https://www.cftc.gov/dea/newcot/DisaggComFutOpt.csv",
    "https://www.cftc.gov/dea/newcot/DisaggComFut.csv",
    "https://www.cftc.gov/dea/newcot/DisaggComFutOpt.txt",
]
COT_COLS = {
    "date": ["Report_Date_as_YYYY-MM-DD","Report_Date","Report_Date_as_YYYY_MM_DD"],
    "market": ["Market_and_Exchange_Names","Market_and_Exchange_Name"],
    "mm_long": ["Mgr_Positions_Long_All","Money_Manager_Long_All","Money_Manager_Long_All_Positions"],
    "mm_short":["Mgr_Positions_Short_All","Money_Manager_Short_All","Money_Manager_Short_All_Positions"]
}

def pick_col(cols, names):
    lc = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lc: return lc[n.lower()]
    for n in names:
        for c in cols:
            if c.lower().startswith(n.lower()): return c
    return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_cftc_disagg():
    last_err=None
    for url in CFTC_ENDPOINTS:
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            try: df = pd.read_csv(io.StringIO(r.text))
            except Exception: df = pd.read_csv(io.StringIO(r.text), sep=";")
            df.rename(columns=lambda x: str(x).strip().replace(" ","_"), inplace=True)
            dcol = pick_col(df.columns, COT_COLS["date"]);  mcol = pick_col(df.columns, COT_COLS["market"])
            lcol = pick_col(df.columns, COT_COLS["mm_long"]); scol = pick_col(df.columns, COT_COLS["mm_short"])
            if not all([dcol,mcol,lcol,scol]): continue
            df = df[[dcol,mcol,lcol,scol]].rename(columns={dcol:"Date", mcol:"Market", lcol:"MM_LONG", scol:"MM_SHORT"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            return df
        except Exception as e:
            last_err=e; continue
    raise RuntimeError(f"CFTC fetch failed. Last error: {last_err}")

def prepare_cot_auto(cftc_df: pd.DataFrame, market_filter: str, price_symbol: str, window: int, pctl: float):
    sub = cftc_df[cftc_df["Market"].str.contains(market_filter, case=False, na=False)].copy()
    if sub.empty: raise ValueError("No rows found for that market filter.")
    sub = sub[["Date","Market","MM_LONG","MM_SHORT"]].sort_values("Date").reset_index(drop=True)
    sub["MM_NET"] = pd.to_numeric(sub["MM_LONG"], errors="coerce") - pd.to_numeric(sub["MM_SHORT"], errors="coerce")
    px = yf_prices(price_symbol, period="10y", interval="1d")
    wpx = weekly_from_daily(px); wpx["SMA10W"] = wpx["Close"].rolling(10).mean()
    merged = pd.merge_asof(wpx.sort_values("Date"), sub.sort_values("Date"), on="Date", direction="backward")
    merged["COT_Idx"] = cot_index(merged["MM_NET"], window)
    merged["Washed"]  = (merged["MM_NET"] < 0) | (merged["COT_Idx"] <= (pctl/100.0))
    merged["TurnUp"]  = merged["MM_NET"].diff() > 0
    merged["PriceUp"] = merged["Close"] > merged["SMA10W"]
    merged["Trigger"] = merged["Washed"] & merged["TurnUp"] & merged["PriceUp"]
    return merged

# ───────── S&P 500 list ─────────
@st.cache_data(ttl=24*3600, show_spinner=False)
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]["Symbol"].str.upper().str.replace(".","-", regex=False).tolist()

def batch(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

# ───────── UI ─────────
st.title("📊 Trading Hub — Real Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏦 Insider (watchlist)",
    "📝 COT (auto from CFTC)",
    "📈 Prices",
    "🌐 Market Scan (S&P 500)"
])

# Tab 1 — Insider watchlist
with tab1:
    st.subheader("SEC Form 4 open‑market buys ('P')")
    tickers_input = st.text_input("Enter comma‑separated tickers", "AAPL,MSFT,NVDA")
    if st.button("Scan insiders", use_container_width=True):
        if not ua or "@" not in ua:
            st.error("Enter a valid User‑Agent with an email in the sidebar.")
        else:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Contacting SEC…"):
                df = scan_insider_buys(tickers, ua, lookback_days, sec_delay)
            st.success("Done.")
            st.dataframe(df, use_container_width=True)
            if len(df):
                top = df.iloc[0]
                st.caption(f"Top: {top['Ticker']} — ${top['BuyUSD_lookback']:,.0f} in {lookback_days}d; buys: {int(top['Buys'])}")

# Tab 2 — COT auto
with tab2:
    st.subheader("Disaggregated Futures+Options Combined (CFTC)")
    try:
        with st.spinner("Fetching CFTC disaggregated file…"):
            cftc_df = fetch_cftc_disagg()
        search = st.text_input("Search market", "GOLD")
        markets = sorted(cftc_df["Market"].dropna().unique())
        if search: markets = [m for m in markets if search.upper() in m.upper()]
        mkt_filter = st.selectbox("Choose/Filter market (substring)", markets, index=0) if markets else st.text_input("Market filter")
        price_hint = st.text_input("Override price symbol (optional)", value=guess_price_symbol(mkt_filter, price_for_cot))
        if st.button("Run COT scan", use_container_width=True):
            merged = prepare_cot_auto(cftc_df, mkt_filter, price_hint, cot_window_weeks, cot_thresh)
            st.dataframe(merged.tail(20), use_container_width=True)
            last = merged.iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("MM Net", f"{int(last['MM_NET']):,}")
            c2.metric("COT Index", f"{last['COT_Idx']*100:.1f}%")
            c3.metric("Price vs 10W", "Above" if last["PriceUp"] else "Below")
            c4.metric("Trigger", "✅" if last["Trigger"] else "—")
            st.line_chart(merged.set_index("Date")[["MM_NET"]])
            st.line_chart(merged.set_index("Date")[["Close","SMA10W"]])
    except Exception as e:
        st.error(f"COT error: {e}")
        st.info("If CFTC shifts files, try again later or we can add more fallbacks.")

# Tab 3 — Prices
with tab3:
    st.subheader("Quick price chart with SMA")
    px_ticker = st.text_input("Ticker (e.g., AAPL, GC=F, CL=F)", value="AAPL", key="px_ticker")
    period = st.selectbox("Period", ["6mo","1y","2y","5y","10y"], index=2)
    if st.button("Load prices", key="load_px"):
        dfp = yf_prices(px_ticker, period=period, interval="1d")
        if "Date" not in dfp.columns:
            dfp.reset_index(inplace=True)
            for cand in ("index","Datetime","datetime","date"):
                if cand in dfp.columns: 
                    dfp.rename(columns={cand:"Date"}, inplace=True); break
        if "Close" not in dfp.columns and "close" in dfp.columns:
            dfp.rename(columns={"close":"Close"}, inplace=True)
        if "Date" in dfp.columns and "Close" in dfp.columns:
            dfp["SMA50"] = dfp["Close"].rolling(50).mean()
            dfp["SMA200"]= dfp["Close"].rolling(200).mean()
            st.line_chart(dfp.set_index("Date")[["Close","SMA50","SMA200"]], use_container_width=True)
            st.dataframe(dfp.tail(20), use_container_width=True)
        else:
            st.error("Price data missing Date/Close columns.")

# Tab 4 — S&P500 Market Scan
with tab4:
    st.subheader("S&P 500 Insider Buy Leaderboard (batch‑scanned)")
    st.caption("Loads S&P 500 from Wikipedia, scans in batches to respect SEC limits.")
    colA, colB = st.columns(2)
    batch_size = colA.slider("Batch size", 20, 200, 60, step=10)
    max_batches = colB.slider("Batches to run now", 1, 20, 3, step=1)
    if st.button("Run S&P 500 scan", use_container_width=True):
        if not ua or "@" not in ua:
            st.error("Enter a valid User‑Agent with email in the sidebar.")
        else:
            with st.spinner("Loading S&P 500 list…"):
                sp = load_sp500_tickers()
            results=[]; batches=0
            for grp in batch(sp, batch_size):
                if batches >= max_batches: break
                df = scan_insider_buys(grp, ua, lookback_days, sec_delay)
                results.append(df); batches += 1
                st.write(f"Completed batch {batches} (scanned {len(grp)} tickers)")
            if results:
                big = pd.concat(results, ignore_index=True).sort_values("BuyUSD_lookback", ascending=False)
                st.success("Done.")
                st.dataframe(big.head(20), use_container_width=True)
            else:
                st.warning("No results. Try more batches or a larger delay.")
