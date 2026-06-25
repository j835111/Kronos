import pytest
import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, query_symbol, upsert_prices
from finetune_tw.signal_today import get_signals_for_date


_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _make_price_frame(start: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            "amount": [100000.0] * len(closes),
        }
    )


class _FakePredictor:
    def _predict_from_frames(self, df_list, y_timestamp_list, pred_len):
        out = []
        for df, y_ts in zip(df_list, y_timestamp_list):
            last_close = float(df["close"].iloc[-1])
            close_path = [last_close * (1.0 + 0.01 * (i + 1)) for i in range(pred_len)]
            out.append(
                pd.DataFrame(
                    {
                        "open": close_path,
                        "high": close_path,
                        "low": close_path,
                        "close": close_path,
                        "volume": [0.0] * pred_len,
                        "amount": [0.0] * pred_len,
                    },
                    index=y_ts,
                )
            )
        return out

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, sample_count, verbose):
        return self._predict_from_frames(df_list, y_timestamp_list, pred_len)

    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        means = [float(df["close"].iloc[-1]) for df in df_list]
        return df_list, x_timestamp_list, y_timestamp_list, means, means, y_timestamp_list

    def predict_prepared_batch(self, df_list, x_timestamp_list, y_timestamp_list, means, stds, y_index_list, pred_len, T, top_k, top_p, sample_count, verbose):
        return self._predict_from_frames(df_list, y_timestamp_list, pred_len)


class _PreparedOnlyPredictor(_FakePredictor):
    def __init__(self):
        self.prepared_called = False

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("legacy predict_batch path should not be used")

    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        self.prepared_called = True
        return super().prepare_batch_inputs(df_list, x_timestamp_list, y_timestamp_list, pred_len)


class _OrderCapturingPreparedPredictor(_PreparedOnlyPredictor):
    def __init__(self):
        super().__init__()
        self.captured_last_closes = []

    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        self.captured_last_closes = [float(df["close"].iloc[-1]) for df in df_list]
        return super().prepare_batch_inputs(df_list, x_timestamp_list, y_timestamp_list, pred_len)


class _LegacyOnlyPredictor:
    def __init__(self):
        self.legacy_called = False

    def _predict_from_frames(self, df_list, y_timestamp_list, pred_len):
        out = []
        for df, y_ts in zip(df_list, y_timestamp_list):
            last_close = float(df["close"].iloc[-1])
            close_path = [last_close * (1.0 + 0.01 * (i + 1)) for i in range(pred_len)]
            out.append(
                pd.DataFrame(
                    {
                        "open": close_path,
                        "high": close_path,
                        "low": close_path,
                        "close": close_path,
                        "volume": [0.0] * pred_len,
                        "amount": [0.0] * pred_len,
                    },
                    index=y_ts,
                )
            )
        return out

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, sample_count, verbose):
        self.legacy_called = True
        return self._predict_from_frames(df_list, y_timestamp_list, pred_len)


def _legacy_get_signals_for_date(predictor, cfg, rebal_date, hold_days, symbols):
    batch_syms, batch_dfs, batch_xts, batch_yts = [], [], [], []
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    y_ts = pd.date_range(rebal_date, periods=hold_days, freq="B")

    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
        if len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:]
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        batch_syms.append(sym)
        batch_dfs.append(ctx_df)
        batch_xts.append(pd.to_datetime(ctx["date"]).reset_index(drop=True))
        batch_yts.append(pd.Series(y_ts))

    signals = {}
    preds = predictor.predict_batch(
        df_list=batch_dfs,
        x_timestamp_list=batch_xts,
        y_timestamp_list=batch_yts,
        pred_len=hold_days,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    for sym, pred, ctx_df in zip(batch_syms, preds, batch_dfs):
        last_close = float(ctx_df["close"].iloc[-1])
        signals[sym] = float(pred["close"].iloc[hold_days - 1]) / last_close - 1.0
    return signals


def test_get_signals_for_date_matches_legacy_path_exactly(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))
    upsert_prices(db, "1301", _make_price_frame("2024-01-01", [30, 31]))  # insufficient lookback

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _FakePredictor()
    rebal_date = pd.Timestamp("2024-01-08")
    symbols = ["1101", "1216", "1301"]

    expected = _legacy_get_signals_for_date(predictor, cfg, rebal_date, 3, symbols)
    actual = get_signals_for_date(predictor, cfg, rebal_date, 3, symbols)

    assert actual == expected
    assert actual == pytest.approx({"1101": 0.03, "1216": 0.03})


def test_get_signals_for_date_prefers_prepared_batch_api(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _PreparedOnlyPredictor()

    actual = get_signals_for_date(
        predictor,
        cfg,
        pd.Timestamp("2024-01-08"),
        3,
        ["1101", "1216"],
    )

    assert predictor.prepared_called is True
    assert actual == pytest.approx({"1101": 0.03, "1216": 0.03})


def test_get_signals_for_date_preserves_caller_symbol_order_into_prepared_batch(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _OrderCapturingPreparedPredictor()

    actual = get_signals_for_date(
        predictor,
        cfg,
        pd.Timestamp("2024-01-08"),
        3,
        ["1216", "1101"],
    )

    assert predictor.prepared_called is True
    assert predictor.captured_last_closes == [25.0, 15.0]
    assert list(actual) == ["1216", "1101"]
    assert actual == pytest.approx({"1216": 0.03, "1101": 0.03})


def test_get_signals_for_date_falls_back_to_legacy_predict_batch_when_prepared_api_absent(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _LegacyOnlyPredictor()

    actual = get_signals_for_date(
        predictor,
        cfg,
        pd.Timestamp("2024-01-08"),
        3,
        ["1101", "1216"],
    )

    assert predictor.legacy_called is True
    assert actual == pytest.approx({"1101": 0.03, "1216": 0.03})
