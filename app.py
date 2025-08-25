# 30‑Week EMA + COT Extremes Scanner — Streamlit
# Run: streamlit run app.py

from __future__ import annotations
import os, math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

import yfinance as yf

from utils import weekly_ema, crossups_above, weeks_since, percent_from
from cot_data import fetch_cot_timeseries, compute_net_and_pct_oi, percentile_rank_last, zscore, DISAGG_GROUPS, TFF_GROUPS

# --------------------------------
# App config
# --------------------------------
st.set_page_config(page_title="EMA + COT Scanner", page_icon="📊", layout="wide")
st.title("📊 30‑Week EMA + COT Positioning Scanner")

st.caption(
    "Weekly price trends (30‑week EMA) plus Commitments of Traders (COT) positioning extremes. "
    "Data via Yahoo Finance and CFTC Socrata APIs. Educational use only."
)

# --------------------------------
# Load ticker mapping
# --------------------------------
@st.cache_data
def load_mapping() -> pd.DataFrame:
    try:
        df = pd.read_csv("ticker_map.csv")
        for col in ["yahoo","category","dataset","market_and_exchange_names"]:
            if col not in df.columns: raise ValueError("Bad mapping file")
        return df
    except Exception as e:
        st.warning("Couldn't load ticker_map.csv; mapping features will be limited.")
        return pd.DataFrame(columns=["yahoo","category","dataset","market_and_exchange_names"])

MAP = load_mapping()

DEFAULT_UNIVERSES: Dict[str, List[str]] = {
    "US Equity Index & Sectors (ETFs)": ["SPY","QQQ","IWM","DIA"],
    "Bonds (ETFs)": ["TLT","IEF","IEI","SHY"],
    "Commodities (ETFs)": ["GLD","SLV","USO","UNG","CPER","PPLT","PALL","CORN","SOYB","WEAT","CANE","JO"],
    "FX Majors (Yahoo)": ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","UUP"],
    "Crypto (Yahoo)": ["BTC-USD","ETH-USD"],
}

HELP_TEXT = (
    "• Scanner uses weekly bars (1‑week) to compute EMA.\n"
    "• COT datasets: **Disaggregated (commodities)** and **TFF (financial futures)**.\n"
    "• Extremes: percentile rank of **net % open interest** for a trader group.\n"
)

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("Settings")
st.sidebar.caption(HELP_TEXT)

buckets = st.sidebar.multiselect(
    "Asset buckets",
    options=list(DEFAULT_UNIVERSES.keys()),
    default=list(DEFAULT_UNIVERSES.keys()),
)

user_raw = st.sidebar.text_area("Optional: extra tickers (comma/space/newline)")
def _parse_tickers(s: str) -> List[str]:
    import re
    if not s: return []
    parts = [p.strip().upper() for p in re.split(r"[\s,;]+", s) if p.strip()]
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

years_hist = st.sidebar.slider("Years of price history (yfinance)", 3, 25, 10, 1)
ema_span = st.sidebar.number_input("EMA span (weeks)", 5, 60, 30, 1)

# COT controls
st.sidebar.subheader("COT Extremes")
cot_lookback_years = st.sidebar.slider("COT lookback (years)", 1, 15, 5, 1)
cot_group_disagg = st.sidebar.selectbox("Commodity group (Disaggregated)", list(DISAGG_GROUPS.keys()), index=0)
cot_group_tff = st.sidebar.selectbox("Financial group (TFF)", list(TFF_GROUPS.keys()), index=2)  # Leveraged Funds by default
pctl_hi = st.sidebar.slider("High extreme (≥ percentile)", 70, 99, 90, 1)
pctl_lo = st.sidebar.slider("Low extreme (≤ percentile)", 1, 30, 10, 1)

run_btn = st.sidebar.button("Scan Now 🚀")

# --------------------------------
# Helper functions
# --------------------------------
@st.cache_data(ttl=60*60*8, show_spinner=False)
def fetch_weekly(ticker: str, years: int = 10) -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=365 * years)).date()
    df = yf.download(ticker, interval="1wk", start=str(start), auto_adjust=False, progress=False)
    if df is None or df.empty: return pd.DataFrame()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    out = df[["Close"]].dropna().copy()
    out.index = pd.to_datetime(out.index)
    return out

def analyze_price(t: str) -> Optional[Dict]:
    try:
        w = fetch_weekly(t, years_hist)
        if w.empty: return None
        w["EMA"] = weekly_ema(w["Close"], span=ema_span)
        w["Above"] = w["Close"] > w["EMA"]
        w["CrossUp"] = crossups_above(w["Close"], w["EMA"])
        last_idx = w.index[-1]
        last_close = float(w["Close"].iloc[-1])
        last_ema = float(w["EMA"].iloc[-1])
        last_above = bool(w["Above"].iloc[-1])
        dist_pct = percent_from(last_close, last_ema)
        cross_dates = list(w.index[w["CrossUp"]])
        last_cross = cross_dates[-1] if cross_dates else None
        wk_since = weeks_since(w.index, last_cross) if last_cross is not None else None
        return {"ticker": t, "raw": w, "last_date": last_idx, "last_close": last_close, "ema": last_ema,
                "above": last_above, "dist_pct": dist_pct, "last_cross": last_cross, "weeks_since_cross": wk_since}
    except Exception:
        return None

def build_table(results: List[Dict]) -> pd.DataFrame:
    if not results: return pd.DataFrame()
    df = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Date": pd.to_datetime(r["last_date"]).date(),
        "Close": r["last_close"],
        "EMA30": r["ema"],
        "% from EMA": r["dist_pct"],
        "Above EMA?": r["above"],
        "Last CrossUp": (pd.to_datetime(r["last_cross"]).date() if r["last_cross"] is not None else None),
        "Weeks Since CrossUp": r["weeks_since_cross"],
    } for r in results])
    df.sort_values(by=["Above EMA?", "% from EMA"], ascending=[False, False], inplace=True)
    return df

def cot_for_ticker(t: str, mapping: pd.DataFrame, lookback_years: int, group_disagg: str, group_tff: str) -> Optional[Dict]:
    row = mapping[mapping["yahoo"].str.upper() == t.upper()].head(1)
    if row.empty: return None
    row = row.iloc[0]
    dataset_key = row["dataset"]
    market = row["market_and_exchange_names"]
    start_date = (datetime.utcnow() - timedelta(days=int(365*lookback_years*1.05))).date().isoformat()

    df = fetch_cot_timeseries(market, dataset_key=dataset_key, start_date=start_date)
    if df.empty: return None
    group = group_disagg if dataset_key.startswith("disagg") else group_tff
    ts = compute_net_and_pct_oi(df, dataset_key, group)
    pctl = percentile_rank_last(ts["net_pct_oi"])
    z = zscore(ts["net_pct_oi"])
    latest = ts.dropna().iloc[-1]
    direction = "Bullish (high net)" if pctl >= 50 else "Bearish (low net)"
    return {"ticker": t, "market": market, "dataset": dataset_key, "group": group,
            "latest_date": pd.to_datetime(latest["date"]).date(),
            "net": float(latest["net"]), "net_pct_oi": float(latest["net_pct_oi"]), "pctl": float(pctl),
            "zscore": float(z), "series": ts}

def plot_price_and_cot(ticker: str, price: pd.DataFrame, ema_span: int, cot: Optional[Dict]):
    if cot is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price["Close"], mode="lines", name="Weekly Close"))
        fig.add_trace(go.Scatter(x=price.index, y=price["EMA"], mode="lines", name=f"EMA{ema_span}"))
        st.plotly_chart(fig, use_container_width=True)
        return

    ts = cot["series"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price.index, y=price["Close"], mode="lines", name="Weekly Close"), secondary_y=True)
    fig.add_trace(go.Scatter(x=price.index, y=price["EMA"], mode="lines", name=f"EMA{ema_span}"), secondary_y=True)
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["net_pct_oi"], mode="lines", name=f"COT Net % OI ({cot['group']})"), secondary_y=False)
    # Mark extremes
    hi = ts["net_pct_oi"].quantile(pctl_hi/100.0)
    lo = ts["net_pct_oi"].quantile(pctl_lo/100.0)
    fig.add_hline(y=hi, line_dash="dot", annotation_text=f"{pctl_hi}th pct", secondary_y=False)
    fig.add_hline(y=lo, line_dash="dot", annotation_text=f"{pctl_lo}th pct", secondary_y=False)
    fig.update_layout(title=f"{ticker} — Price & COT ({cot['market']})", hovermode="x unified")
    fig.update_yaxes(title_text="Net % Open Interest", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Main
# --------------------------------
if run_btn:
    # Build universe
    tickers = []
    for b in buckets: tickers.extend(DEFAULT_UNIVERSES.get(b, []))
    tickers.extend(_parse_tickers(user_raw))
    # Dedup preserve order
    uniq, seen = [], set()
    for t in tickers:
        if t and t not in seen: uniq.append(t); seen.add(t)
    tickers = uniq

    if not tickers:
        st.info("No tickers selected."); st.stop()

    st.success(f"Scanning {len(tickers)} symbols…")

    # PRICE SCAN (parallel)
    results, errors = [], []
    max_workers = min(16, max(2, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_price, t): t for t in tickers}
        prog = st.progress(0)
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                r = f.result()
                if r is not None: results.append(r)
                else: errors.append(t)
            except Exception:
                errors.append(t)
            done += 1
            prog.progress(done / len(tickers))

    if not results:
        st.error("No price data found.")
        st.stop()

    # PRICE TABLES
    table = build_table(results)

    with st.expander("Crossed ABOVE EMA recently"):
        st.caption("Filtered by 'Weeks Since CrossUp' within chosen window via the chart header.")
        st.dataframe(table.sort_values(["Weeks Since CrossUp","% from EMA"], ascending=[True, False]), use_container_width=True)

    # COT EXTREMES
    st.subheader("COT Positioning Extremes (Net % OI percentile)")
    cot_rows = []
    cot_details: Dict[str, Dict] = {}
    for r in results:
        t = r["ticker"]
        info = cot_for_ticker(t, MAP, cot_lookback_years, cot_group_disagg, cot_group_tff)
        if info is None: 
            continue
        cot_details[t] = info
        extreme = "High" if info["pctl"] >= pctl_hi else ("Low" if info["pctl"] <= pctl_lo else "")
        cat = MAP.loc[MAP["yahoo"].str.upper()==t.upper(), "category"].head(1).values
        cat = cat[0] if len(cat) else ""
        cot_rows.append({
            "Ticker": t, "Category": cat, "Market": info["market"], "Dataset": info["dataset"],
            "Group": info["group"], "Date": info["latest_date"], "Net % OI": round(info["net_pct_oi"], 2),
            "Pctile": round(info["pctl"],1), "Z-Score": round(info["zscore"],2), "Extreme": extreme
        })
    cot_df = pd.DataFrame(cot_rows)
    if not cot_df.empty:
        # Cluster summary
        st.dataframe(cot_df.sort_values(["Extreme","Pctile"], ascending=[True, False]), use_container_width=True)
        with st.expander("Extreme clusters"):
            if "Extreme" in cot_df.columns:
                clusters = (cot_df[cot_df["Extreme"]!=""]
                            .groupby(["Category","Extreme"])
                            .size().reset_index(name="Count")
                            .sort_values(["Count"], ascending=False))
                st.dataframe(clusters, use_container_width=True)
    else:
        st.info("No COT data matched this universe (edit ticker_map.csv to add mappings).")

    # DETAIL / CHART
    st.subheader("Chart a symbol (Price + COT)")
    choice = st.selectbox("Pick a ticker", options=[r["ticker"] for r in results])
    if choice:
        sel = next((r for r in results if r["ticker"] == choice), None)
        if sel is not None:
            st.markdown(
                f"**{choice}** — Last Close **{sel['last_close']:.2f}** | EMA{ema_span}: **{sel['ema']:.2f}** | Dist: **{sel['dist_pct']:.2f}%**"
            )
            cot_info = cot_details.get(choice)
            plot_price_and_cot(choice, sel["raw"], ema_span, cot_info)

    if errors:
        with st.expander("Symbols with errors / insufficient data"):
            st.write(", ".join(errors))

else:
    st.info("Configure your universe in the left sidebar and click **Scan Now**. "
            "Edit **ticker_map.csv** to wire your tickers to CFTC markets.")
