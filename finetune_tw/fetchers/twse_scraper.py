from __future__ import annotations
import time
import requests
import pandas as pd

TWSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
_req_times: list[float] = []


def _rate_limit() -> None:
    now = time.time()
    _req_times[:] = [t for t in _req_times if now - t < 5.0]
    if len(_req_times) >= 3:
        wait = 5.0 - (now - _req_times[0]) + 0.1
        if wait > 0:
            time.sleep(wait)
    _req_times.append(time.time())


def fetch_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    _rate_limit()
    try:
        resp = requests.get(
            TWSE_URL,
            params={"response": "json", "date": f"{year}{month:02d}01", "stockNo": symbol},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("stat") != "OK" or not data.get("data"):
            return None
        rows = []
        for row in data["data"]:
            try:
                y, m, d = row[0].split("/")
                ad_date = f"{int(y) + 1911}-{m}-{d}"
                rows.append({
                    "date": ad_date,
                    "volume": float(row[1].replace(",", "")),
                    "amount": float(row[2].replace(",", "")),
                    "open": float(row[3].replace(",", "")),
                    "high": float(row[4].replace(",", "")),
                    "low": float(row[5].replace(",", "")),
                    "close": float(row[6].replace(",", "")),
                })
            except (ValueError, IndexError):
                continue
        if not rows:
            return None
        return pd.DataFrame(rows)[
            ["date", "open", "high", "low", "close", "volume", "amount"]
        ]
    except Exception:
        return None


def fetch_symbol_twse(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch all months in [start, end] for a 4-digit TWSE symbol (without .TW suffix)."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    cur = start_dt.replace(day=1)
    while cur <= end_dt:
        df = fetch_month(symbol, cur.year, cur.month)
        if df is not None:
            frames.append(df)
        cur = (cur + pd.offsets.MonthEnd(1)) + pd.offsets.Day(1)
    if not frames:
        return None
    result = pd.concat(frames).drop_duplicates("date")
    result = result[(result["date"] >= start) & (result["date"] <= end)]
    return result.reset_index(drop=True)
