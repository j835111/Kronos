"""
Usage:
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml --update
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml --source twse
"""
from __future__ import annotations
import argparse
from datetime import date
from tqdm import tqdm

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices, get_last_date
from finetune_tw.fetchers.yfinance_fetcher import fetch_symbol, get_twse_symbol_list
from finetune_tw.fetchers.twse_scraper import fetch_symbol_twse

BENCHMARK_SYMBOL = "^TWII"


def download(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    update_only: bool = False,
    finmind_token: str | None = None,
) -> None:
    init_db(db_path)
    for sym in tqdm(symbols, desc=f"Downloading [{source}]"):
        effective_start = start
        if update_only:
            last = get_last_date(db_path, sym)
            if last:
                effective_start = last  # re-fetch last known date to catch amendments

        df = None
        if source in ("yfinance", "auto"):
            df = fetch_symbol(sym, start=effective_start, end=end)
        if df is None and source in ("twse", "auto"):
            # Benchmark symbol is not available in TWSE; skip TWSE fallback for it
            if sym != BENCHMARK_SYMBOL:
                bare = sym.replace(".TW", "")
                df = fetch_symbol_twse(bare, effective_start, end)
        if df is None and source == "finmind":
            if finmind_token:
                from finetune_tw.fetchers.finmind_fetcher import fetch_symbol_finmind
                bare = sym.replace(".TW", "")
                df = fetch_symbol_finmind(bare, effective_start, end, token=finmind_token)
        if df is not None and not df.empty:
            upsert_prices(db_path, sym, df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--source", choices=["yfinance", "twse", "auto", "finmind"], default="auto")
    parser.add_argument("--update", action="store_true", help="Only fetch missing dates")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--finmind-token", default=None, help="FinMind API token")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    symbols = get_twse_symbol_list()
    # Also add benchmark
    symbols = [cfg.benchmark_symbol] + symbols

    download(
        db_path=cfg.db_path,
        symbols=symbols,
        start=args.start,
        end=args.end,
        source=args.source,
        update_only=args.update,
        finmind_token=args.finmind_token,
    )
    print(f"Done. DB: {cfg.db_path}")


if __name__ == "__main__":
    main()
