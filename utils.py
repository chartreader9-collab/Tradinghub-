# utils.py
from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def weekly_ema(close: pd.Series, span: int = 30) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()

def crossups_above(close: pd.Series, ema: pd.Series) -> pd.Series:
    above = close > ema
    return above & (~above.shift(1).fillna(False))

def weeks_since(date_index: pd.DatetimeIndex, last_date: pd.Timestamp) -> int | None:
    if last_date is None: 
        return None
    return int((date_index[-1] - last_date).days // 7)

def percent_from(a: float, b: float) -> float:
    if b == 0 or pd.isna(b): return float('nan')
    return (a / b - 1.0) * 100.0
