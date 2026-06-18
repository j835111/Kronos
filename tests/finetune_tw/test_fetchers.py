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


def test_fetch_symbol_amount_is_zero():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert (df["amount"] == 0.0).all()


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
