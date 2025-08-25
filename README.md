# EMA + COT Scanner (Streamlit)

Scan asset classes for:
1) **Price trend** — weekly **30‑EMA** cross‑ups and status
2) **Positioning** — **CFTC COT** extremes by percentile of **net % open interest**

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Optional: set a Socrata app token
Socrata (CFTC) rate‑limits anonymous requests. Create a token and export it for smoother pulls:

- macOS/Linux:
  ```bash
  export CFTC_SODA_APP_TOKEN=YOUR_TOKEN
  ```
- Windows (Powershell):
  ```powershell
  setx CFTC_SODA_APP_TOKEN YOUR_TOKEN
  ```

## Files
- `app.py` – Streamlit app combining EMA scanner + COT extremes + charts
- `cot_data.py` – CFTC fetch + transforms (Disaggregated & TFF datasets)
- `utils.py` – EMA / cross‑up helpers
- `ticker_map.csv` – Mapping from Yahoo tickers → CFTC `market_and_exchange_names` and dataset family (edit this to fit your universe)
- `.streamlit/config.toml` – dark theme
- `requirements.txt`

## How COT extremes are calculated
- We query CFTC **Disaggregated Combined** (commodities) and **TFF Combined** (financials) via the CFTC Socrata API datasets.
- For each market we compute **net = long − short** for a chosen trader group (e.g., Managed Money, Leveraged Funds).
- We normalize by open interest: **net % OI = 100 × net / open_interest_all**.
- The **percentile** for the latest value is computed versus the selected lookback window.

## Sources
- CFTC Commitments of Traders PRE (dataset pages): Disaggregated Combined `kh3c-gbw2`, TFF All `udgc-27he`.
- Naming examples of `market_and_exchange_names` come from current CFTC viewable pages (e.g., EURO FX, GOLD, S&P 500 Consolidated).

## Notes
- Some ETFs don’t map cleanly to a single futures contract (e.g., DBC/DBA/LQD/HYG). Edit or remove those.
- If a mapping is missing, add a row to `ticker_map.csv` with the exact `market_and_exchange_names` string used by CFTC.
- This is for **education** only, not trading advice.
