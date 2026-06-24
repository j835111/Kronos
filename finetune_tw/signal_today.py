"""
signal_today.py — 輸出今日 Kronos 選股訊號

用法:
  python -m finetune_tw.signal_today --config finetune_tw/configs/config_tw_daily_rtx6000.yaml
  python -m finetune_tw.signal_today --config ... --date 2026-06-20    # 指定日期
  python -m finetune_tw.signal_today --config ... --top_k 10 --hold_days 3
  python -m finetune_tw.signal_today --config ... --holdings 2330,2317  # 目前持倉（計算換股）
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from finetune_tw.config import Config
from finetune_tw.db import list_symbols, get_last_date, query_symbols_window
from finetune_tw.backtest import (
    build_model_specs,
    load_predictor_from_spec,
    rank_stocks,
)


_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _last_trading_day(db_path: str, benchmark: str = "^TWII") -> str:
    """DB 中最新的交易日期（用大盤當基準）。"""
    d = get_last_date(db_path, benchmark)
    if d is None:
        # fallback：找任意股票的最新日期
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM daily_prices"
            ).fetchone()
        d = row[0] if row else None
    if d is None:
        raise RuntimeError("DB 中找不到任何交易日，請先執行 download_data。")
    return d


def _load_signal_contexts(
    cfg: Config,
    rebal_date: pd.Timestamp,
    hold_days: int,
    symbols: list[str],
) -> list[tuple[str, pd.DataFrame, pd.Series, pd.Series]]:
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (
        rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    y_ts = pd.Series(pd.date_range(rebal_date, periods=hold_days, freq="B"))

    rows = query_symbols_window(
        cfg.db_path,
        symbols,
        start=lookback_start,
        end=rebal_str,
    )
    if rows.empty:
        return []

    grouped = {sym: grp.reset_index(drop=True) for sym, grp in rows.groupby("symbol", sort=False)}
    contexts = []
    for sym in symbols:
        df = grouped.get(sym)
        if df is None or len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:].reset_index(drop=True)
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        contexts.append((sym, ctx_df, x_ts, y_ts.copy()))
    return contexts


def get_signals_for_date(
    predictor,
    cfg: Config,
    rebal_date: pd.Timestamp,
    hold_days: int,
    symbols: list[str],
) -> dict[str, float]:
    """對單一 rebal_date 執行推論，回傳 {sym: hold_days 預測報酬率}。"""
    BATCH_SIZE = 64
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    contexts = _load_signal_contexts(cfg, rebal_date, hold_days, symbols)
    batch_syms = [sym for sym, _, _, _ in contexts]
    batch_dfs = [ctx_df for _, ctx_df, _, _ in contexts]
    batch_xts = [x_ts for _, _, x_ts, _ in contexts]
    batch_yts = [y_ts for _, _, _, y_ts in contexts]

    print(f"  推論 {len(batch_syms)} 支股票（日期 {rebal_str}）...")
    sys.stdout.flush()

    signals: dict[str, float] = {}
    has_prepared_batch_api = (
        hasattr(predictor, "prepare_batch_inputs")
        and hasattr(predictor, "predict_prepared_batch")
    )
    with torch.no_grad():
        for b in range(0, len(batch_syms), BATCH_SIZE):
            df_slice = batch_dfs[b : b + BATCH_SIZE]
            xt_slice = batch_xts[b : b + BATCH_SIZE]
            yt_slice = batch_yts[b : b + BATCH_SIZE]
            if has_prepared_batch_api:
                prepared = predictor.prepare_batch_inputs(
                    df_list=df_slice,
                    x_timestamp_list=xt_slice,
                    y_timestamp_list=yt_slice,
                    pred_len=hold_days,
                )
                preds = predictor.predict_prepared_batch(
                    *prepared,
                    pred_len=hold_days,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
            else:
                preds = predictor.predict_batch(
                    df_list=df_slice,
                    x_timestamp_list=xt_slice,
                    y_timestamp_list=yt_slice,
                    pred_len=hold_days,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
            for sym, pred, ctx_df in zip(
                batch_syms[b : b + BATCH_SIZE],
                preds,
                batch_dfs[b : b + BATCH_SIZE],
            ):
                if pred is not None and len(pred) >= hold_days:
                    last_close = ctx_df["close"].iloc[-1]
                    ret = float(pred["close"].iloc[hold_days - 1]) / last_close - 1.0
                    signals[sym] = ret

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="輸出今日 Kronos 選股訊號")
    parser.add_argument("--config", required=True, help="YAML config 路徑")
    parser.add_argument("--model", default="round0",
                        choices=["pretrained", "round0", "round1", "round2"],
                        help="使用哪個模型版本")
    parser.add_argument("--date", default=None,
                        help="指定交易日（YYYY-MM-DD），預設為 DB 最新日期")
    parser.add_argument("--top_k", type=int, default=None,
                        help="持股數（預設讀 config.top_k）")
    parser.add_argument("--hold_days", type=int, default=None,
                        help="持有天數（預設讀 config.hold_days）")
    parser.add_argument("--holdings", default="",
                        help="目前已持有的股票代碼，逗號分隔（用於顯示換股建議）")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    top_k     = args.top_k     or cfg.top_k
    hold_days = args.hold_days or cfg.hold_days

    # 決定訊號日期
    if args.date:
        rebal_date = pd.Timestamp(args.date)
    else:
        latest = _last_trading_day(cfg.db_path, cfg.benchmark_symbol)
        rebal_date = pd.Timestamp(latest)

    print(f"\n=== Kronos 選股訊號 ===")
    print(f"  模型：{args.model}  |  top_k={top_k}  |  hold_days={hold_days}")
    print(f"  訊號日：{rebal_date.date()}  （預測 +{hold_days} 個交易日後的收盤報酬）")
    print()

    # 載入模型
    specs = build_model_specs(cfg)
    if args.model not in specs:
        print(f"未知模型 key: {args.model}，可用: {list(specs)}")
        sys.exit(1)
    predictor = load_predictor_from_spec(specs[args.model], cfg)

    # 取 symbol 清單（排除大盤指數）
    all_symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]

    # 推論
    signals = get_signals_for_date(predictor, cfg, rebal_date, hold_days, all_symbols)

    if not signals:
        print("警告：沒有取得任何訊號，請確認 DB 資料已更新至今日。")
        sys.exit(1)

    # 選 top_k
    top_set = rank_stocks(signals, top_k=top_k, threshold=cfg.min_signal_threshold)

    # 排序輸出
    ranked = sorted(top_set, key=lambda s: signals[s], reverse=True)

    print(f"\n【選股結果】預測報酬 top {top_k}（{rebal_date.date()} 訊號）")
    print(f"{'排名':>4}  {'代碼':>8}  {'預測 +{:d}日報酬':>12}".format(hold_days))
    print("-" * 32)
    for rank, sym in enumerate(ranked, 1):
        ret_pct = signals[sym] * 100
        print(f"  {rank:>2}   {sym:>8}   {ret_pct:>+8.2f}%")

    # 換股建議
    if args.holdings:
        current = set(args.holdings.split(","))
        to_sell = current - top_set
        to_buy  = top_set - current
        hold    = current & top_set
        print(f"\n【換股建議】（目前持倉: {sorted(current)}）")
        if hold:
            print(f"  繼續持有：{sorted(hold)}")
        if to_sell:
            print(f"  賣出：    {sorted(to_sell)}")
        if to_buy:
            print(f"  買入：    {sorted(to_buy)}")
        if not to_sell and not to_buy:
            print("  持倉無需調整。")

    print()


if __name__ == "__main__":
    main()
