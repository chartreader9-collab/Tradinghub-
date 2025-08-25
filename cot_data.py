# cot_data.py
from __future__ import annotations
import os, time, math
from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

SODA_BASE = "https://publicreporting.cftc.gov/resource"

# Dataset IDs (Socrata)
DATASET_IDS = {
    "disagg_combined": "kh3c-gbw2",  # Disaggregated - Futures & Options Combined
    "disagg_futures": "72hh-3qpy",   # Disaggregated - Futures Only
    "tff_combined": "udgc-27he",     # Traders in Financial Futures - Combined
    "tff_futures": "gpe5-46if",      # TFF - Futures Only
    "legacy_combined": "jun7-fc8e",  # Legacy Combined (sometimes useful for names)
}

# Mapping from friendly group names to column stems in each dataset family
DISAGG_GROUPS = {
    "Managed Money": "managed_money",
    "Producer/Merchant": "producer_merchant_processor_user",
    "Swap Dealers": "swap_dealer",
    "Other Reportables": "other_reportable",
}

TFF_GROUPS = {
    "Dealer/Intermediary": "dealer_intermediary",
    "Asset Manager/Institutional": "asset_manager_institutional",
    "Leveraged Funds": "leveraged_funds",
    "Other Reportables": "other_reportable",
}

def _auth_headers() -> Dict[str,str]:
    token = os.getenv("CFTC_SODA_APP_TOKEN") or os.getenv("SODA_APP_TOKEN") or os.getenv("APP_TOKEN")
    return {"X-App-Token": token} if token else {}

class CFTCError(Exception): pass

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.8, min=1, max=10),
       retry=retry_if_exception_type((requests.RequestException, CFTCError)))
def _socrata_query(dataset: str, params: Dict[str, str]) -> pd.DataFrame:
    url = f"{SODA_BASE}/{dataset}.json"
    headers = _auth_headers()
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code == 429:
        # rate limited
        raise CFTCError("Rate limited (429)")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise CFTCError("Unexpected response")
    df = pd.DataFrame(data)
    return df

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # normalize date column
    if "report_date_as_yyyy_mm_dd" in df.columns:
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    elif "report_date" in df.columns:
        df["date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
    else:
        raise CFTCError("No date column found in dataset")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def fetch_cot_timeseries(
    market_and_exchange_names: str,
    dataset_key: str = "disagg_combined",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    select_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch COT timeseries for a given market name using Socrata API.
    `dataset_key` must be one of DATASET_IDS.keys().
    """
    ds = DATASET_IDS[dataset_key]
    where_clauses = [f"upper(market_and_exchange_names) = upper('{market_and_exchange_names.replace(\"'\",\"''\")}')"]
    if start_date:
        where_clauses.append(f"report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00.000'")
    if end_date:
        where_clauses.append(f"report_date_as_yyyy_mm_dd <= '{end_date}T23:59:59.999'")
    where = " AND ".join(where_clauses)

    params = {
        "$where": where,
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": "500000"
    }
    if select_columns:
        params["$select"] = ", ".join(select_columns)

    df = _socrata_query(ds, params)
    df = _ensure_ts(df)
    return df

def compute_net_and_pct_oi(df: pd.DataFrame, dataset_key: str, group_label: str) -> pd.DataFrame:
    out = df.copy()
    if dataset_key.startswith("disagg"):
        stem = DISAGG_GROUPS[group_label]
    else:
        stem = TFF_GROUPS[group_label]
    long_col = f"{stem}_long_all"
    short_col = f"{stem}_short_all"
    if long_col not in out.columns or short_col not in out.columns:
        # If columns weren't selected, re-fetch only necessary columns to be lighter
        need = ["market_and_exchange_names", "open_interest_all", long_col, short_col, "report_date_as_yyyy_mm_dd"]
        back = fetch_cot_timeseries(str(out.get("market_and_exchange_names", [""])[0]), dataset_key, select_columns=need)
        out = back

    out[long_col] = pd.to_numeric(out[long_col], errors="coerce")
    out[short_col] = pd.to_numeric(out[short_col], errors="coerce")
    out["open_interest_all"] = pd.to_numeric(out["open_interest_all"], errors="coerce")
    out["net"] = out[long_col] - out[short_col]
    out["net_pct_oi"] = np.where(out["open_interest_all"] > 0, 100.0 * out["net"] / out["open_interest_all"], np.nan)
    return out[["date","market_and_exchange_names","open_interest_all","net","net_pct_oi"]].dropna()

def percentile_rank_last(series: pd.Series, lookback_weeks: Optional[int] = None) -> float:
    """Percentile rank of the last value within the lookback window."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    if lookback_weeks is not None and lookback_weeks > 0 and len(s) > lookback_weeks:
        s = s.iloc[-lookback_weeks:]
    # rank last value among values
    last = s.iloc[-1]
    rank = (s <= last).sum() / float(len(s)) * 100.0
    return float(rank)

def zscore(series: pd.Series, lookback_weeks: Optional[int] = None) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    if lookback_weeks is not None and lookback_weeks > 0 and len(s) > lookback_weeks:
        s = s.iloc[-lookback_weeks:]
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or math.isclose(sd, 0.0):
        return float("nan")
    return float((s.iloc[-1] - mu) / sd)
