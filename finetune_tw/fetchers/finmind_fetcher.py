from __future__ import annotations
import requests
import pandas as pd

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_symbol_finmind(
    symbol: str, start: str, end: str, token: str
) -> pd.DataFrame | None:
    try:
        resp = requests.get(
            FINMIND_URL,
            params={
                "dataset": "TaiwanStockPrice",
                "data_id": symbol,
                "start_date": start,
                "end_date": end,
                "token": token,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("msg") != "success" or not data.get("data"):
            return None
        df = pd.DataFrame(data["data"])
        df = df.rename(columns={
            "max": "high", "min": "low",
            "Trading_Volume": "volume", "Trading_money": "amount",
        })
        return df[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    except Exception:
        return None
