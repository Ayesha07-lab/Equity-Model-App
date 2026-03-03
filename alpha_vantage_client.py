import os
import time
import requests
import pandas as pd


BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageClient:
    def __init__(self, api_key: str | None = None, throttle_seconds: float = 12.0):
        # Free tier is rate limited; default throttle avoids constant 429s.
        self.api_key = (api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")).strip()
        self.throttle_seconds = throttle_seconds
        self._last_call_ts = 0.0

    def _throttle(self):
        if self.throttle_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)
        self._last_call_ts = time.time()

    def _get(self, params: dict) -> dict:
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is missing. Set ALPHAVANTAGE_API_KEY in .env.")
        params = dict(params)
        params["apikey"] = self.api_key

        self._throttle()
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
        if isinstance(data, dict) and "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
        return data

    def company_overview(self, symbol: str) -> dict:
        return self._get({"function": "OVERVIEW", "symbol": symbol})

    def income_statement_annual(self, symbol: str) -> pd.DataFrame:
        data = self._get({"function": "INCOME_STATEMENT", "symbol": symbol})
        rows = data.get("annualReports", [])
        return pd.DataFrame(rows)

    def balance_sheet_annual(self, symbol: str) -> pd.DataFrame:
        data = self._get({"function": "BALANCE_SHEET", "symbol": symbol})
        rows = data.get("annualReports", [])
        return pd.DataFrame(rows)

    def cash_flow_annual(self, symbol: str) -> pd.DataFrame:
        data = self._get({"function": "CASH_FLOW", "symbol": symbol})
        rows = data.get("annualReports", [])
        return pd.DataFrame(rows)