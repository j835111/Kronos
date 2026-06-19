"""
python finetune_tw/backtest.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: fine-tuned predictor at outputs/{exp_name}/predictor/best_model/
          fine-tuned tokenizer at outputs/{exp_name}/tokenizer/best_model/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols


# ── Pure helper functions (testable without a model) ────────────────────────

def compute_metrics(daily_returns: pd.Series) -> dict:
    ann_ret = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    cum = (1 + daily_returns).cumprod()
    max_dd = ((cum.cummax() - cum) / cum.cummax()).max()
    return {"annualised_return": ann_ret, "sharpe": sharpe, "max_drawdown": max_dd}


def rank_stocks(signals: dict[str, float], top_k: int) -> set[str]:
    sorted_syms = sorted(signals, key=signals.__getitem__, reverse=True)
    return set(sorted_syms[:top_k])


def build_portfolio_returns(
    price_data: dict[str, pd.Series],
    holdings_sequence: list[set[str]],
    rebalance_dates: pd.Index,
) -> pd.Series:
    ret_list = []
    for date, holdings in zip(rebalance_dates, holdings_sequence):
        period_returns = []
        for sym in holdings:
            if sym not in price_data:
                continue
            series = price_data[sym]
            if date not in series.index:
                continue
            pos = series.index.get_loc(date)
            if pos + 1 >= len(series):
                continue
            r = series.iloc[pos + 1] / series.iloc[pos] - 1
            period_returns.append(r)
        ret_list.append(float(np.mean(period_returns)) if period_returns else 0.0)
    return pd.Series(ret_list, index=rebalance_dates)


# ── Main backtest loop ──────────────────────────────────────────────────────

def run_backtest(cfg: Config) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tok_path  = Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model"
    pred_path = Path(cfg.output_dir) / cfg.exp_name / "predictor" / "best_model"

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
    model     = Kronos.from_pretrained(str(pred_path))
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    tokenizer.eval(); model.eval()

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())

    # Pre-load close prices for all symbols over the test period
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=test_end)
        if len(df) > 0:
            idx = pd.DatetimeIndex(df["date"])
            close_prices[sym] = pd.Series(df["close"].values, index=idx)

    # Build rebalance dates
    all_dates = pd.bdate_range(cfg.test_start_date, test_end)
    rebalance_dates = all_dates[::cfg.hold_days]

    holdings_sequence: list[set[str]] = []
    for rebal_date in rebalance_dates:
        signals: dict[str, float] = {}
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        for sym in symbols:
            df = query_symbol(cfg.db_path, sym,
                              end=rebal_str)
            if len(df) < cfg.lookback_window:
                continue
            ctx = df.iloc[-cfg.lookback_window:]
            x_ts = pd.to_datetime(ctx["date"])
            y_ts = pd.date_range(rebal_date, periods=cfg.pred_len, freq="B")
            with torch.no_grad():
                pred = predictor.predict(
                    df=ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
                    x_timestamp=x_ts.reset_index(drop=True),
                    y_timestamp=pd.Series(y_ts),
                    pred_len=cfg.pred_len,
                    T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
            if pred is not None and len(pred) >= cfg.pred_len:
                signals[sym] = pred["close"].iloc[-1] / ctx["close"].iloc[-1] - 1

        holdings_sequence.append(rank_stocks(signals, cfg.top_k))

    strategy_returns = build_portfolio_returns(close_prices, holdings_sequence, rebalance_dates[:-1])

    # Benchmark returns
    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol,
                         start=cfg.test_start_date, end=test_end)
    bm_close = pd.Series(bm_df["close"].values,
                         index=pd.DatetimeIndex(bm_df["date"]))
    bm_returns = bm_close.pct_change().dropna().reindex(strategy_returns.index).fillna(0)

    metrics = compute_metrics(strategy_returns)
    bm_metrics = compute_metrics(bm_returns)

    print(f"\n=== Backtest Results ({cfg.test_start_date} → {test_end}) ===")
    print(f"Strategy  — Ann. Return: {metrics['annualised_return']:.2%}  "
          f"Sharpe: {metrics['sharpe']:.2f}  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"Benchmark — Ann. Return: {bm_metrics['annualised_return']:.2%}  "
          f"Sharpe: {bm_metrics['sharpe']:.2f}  Max DD: {bm_metrics['max_drawdown']:.2%}")

    # Plot
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bm    = (1 + bm_returns).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(cum_strat.index, cum_strat.values, label="Kronos-TW Strategy")
    plt.plot(cum_bm.index,    cum_bm.values,    label=cfg.benchmark_symbol, linestyle="--")
    plt.title("Cumulative Return: Strategy vs Benchmark")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.tight_layout()
    out_path = Path(cfg.output_dir) / cfg.exp_name / "backtest_result.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--top_k",    type=int,   default=None)
    parser.add_argument("--hold_days", type=int,  default=None)
    parser.add_argument("--pred_len",  type=int,  default=None)
    parser.add_argument("--test_start", default=None)
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.top_k:      cfg.top_k = args.top_k
    if args.hold_days:  cfg.hold_days = args.hold_days
    if args.pred_len:   cfg.pred_len = args.pred_len
    if args.test_start: cfg.test_start_date = args.test_start
    run_backtest(cfg)


if __name__ == "__main__":
    main()
