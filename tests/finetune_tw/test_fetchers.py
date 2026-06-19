from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from finetune_tw.fetchers.yfinance_fetcher import fetch_symbol, get_twse_symbol_list


def _mock_history(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="Asia/Taipei")
    return pd.DataFrame({
        "Open": [100.0] * n, "High": [101.0] * n,
        "Low": [99.0] * n, "Close": [100.5] * n, "Volume": [1_000_000] * n,
    }, index=idx)


def test_fetch_symbol_returns_standard_columns():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(5)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 5


def test_fetch_symbol_date_format():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert df["date"].iloc[0] == "2024-01-01"


def test_fetch_symbol_amount_is_turnover_proxy():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    expected = 1_000_000 * 100.125
    assert (df["amount"] - expected).abs().max() < 1e-6


def test_fetch_symbol_returns_none_on_empty():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_symbol("9999.TW", start="2024-01-01")
    assert result is None


def test_fetch_symbol_returns_none_on_exception():
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", side_effect=Exception("network error")):
        result = fetch_symbol("2330.TW", start="2024-01-01")
    assert result is None


def test_get_twse_symbol_list_parses_response():
    mock_json = [{"Code": "2330", "Name": "台積電"}, {"Code": "2317", "Name": "鴻海"}]
    with patch("finetune_tw.fetchers.yfinance_fetcher.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.raise_for_status = MagicMock()
        symbols = get_twse_symbol_list()
    assert "2330.TW" in symbols
    assert "2317.TW" in symbols


from finetune_tw.fetchers.twse_scraper import fetch_month, fetch_symbol_twse

TWSE_SAMPLE_RESPONSE = {
    "stat": "OK",
    "data": [
        ["113/01/02", "10,000", "1,000,000", "580.00", "585.00", "578.00", "582.00", "2.00", "100"],
        ["113/01/03", "12,000", "1,200,000", "582.00", "588.00", "580.00", "586.00", "4.00", "120"],
    ],
}


def test_twse_fetch_month_standard_columns():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = TWSE_SAMPLE_RESPONSE
        df = fetch_month("2330", 2024, 1)
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]


def test_twse_fetch_month_roc_date_conversion():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = TWSE_SAMPLE_RESPONSE
        df = fetch_month("2330", 2024, 1)
    assert df["date"].iloc[0] == "2024-01-02"


def test_twse_fetch_month_returns_none_on_bad_stat():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"stat": "查無資料"}
        result = fetch_month("9999", 2024, 1)
    assert result is None


from finetune_tw.fetchers.finmind_fetcher import fetch_symbol_finmind

FINMIND_RESPONSE = {
    "msg": "success",
    "data": [
        {"date": "2024-01-02", "open": 580.0, "max": 585.0,
         "min": 578.0, "close": 582.0, "Trading_Volume": 10000, "Trading_money": 5800000},
    ],
}


def test_finmind_returns_standard_columns():
    with patch("finetune_tw.fetchers.finmind_fetcher.requests.get") as mock_get:
        mock_get.return_value.json.return_value = FINMIND_RESPONSE
        mock_get.return_value.raise_for_status = MagicMock()
        df = fetch_symbol_finmind("2330", "2024-01-01", "2024-01-31", token="test_token")
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
