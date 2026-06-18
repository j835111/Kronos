# Taiwan Stock Fine-tuning Design

**Date:** 2026-06-19  
**Status:** Approved

## Overview

Build a new `finetune_tw/` module to fine-tune Kronos-base on Taiwan Stock Exchange (TWSE) daily K-line data for all ~1700 listed stocks (2015–2026). The module supports both forecasting and backtesting, and is designed to run on Google Colab free tier (Tesla T4, single GPU, ephemeral storage).

## Requirements

| Dimension | Decision |
|-----------|----------|
| Model | Kronos-base (102.3M params, context 512) |
| Frequency | Daily (日線) |
| Scope | All ~1700 TWSE listed stocks |
| Date range | 2015-01-01 – present (2026) |
| Runtime | Google Colab free tier (T4 16GB, single GPU) |
| Output | Fine-tuned model + backtesting report |

## §1 Data Layer

### Storage: SQLite

All market data is stored in a single `finetune_tw/data/tw_stocks.db` file on Google Drive. Two tables:

```sql
CREATE TABLE stocks (
    symbol     TEXT PRIMARY KEY,  -- e.g. '2330.TW'
    name       TEXT,
    first_date TEXT,
    last_date  TEXT
);

CREATE TABLE daily_prices (
    symbol  TEXT,
    date    TEXT,         -- 'YYYY-MM-DD'
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    amount  REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX idx_date ON daily_prices(date);
```

The TWSE benchmark index (`^TWII`) is stored as a regular symbol for backtesting comparison.

> `amount`（成交額）在 yfinance 與 TWSE 爬蟲中可能不提供；缺失時填 0，與 `finetune_csv` 規格一致。

### Data Sources (multi-source, fallback chain)

| Source | Role | Constraint |
|--------|------|------------|
| yfinance | Primary bulk download (`.TW` suffix) | Free, no hard rate limit for daily |
| TWSE scraper | Fallback for yfinance gaps + incremental updates | ≤3 req/5s |
| FinMind | Optional supplement within free quota | Paid above free tier |

**`download_data.py` behaviour:**
- `--source yfinance` (default): batch download all symbols, auto-fallback to TWSE scraper on failure
- `--source twse`: TWSE scraper only, respects rate limit with `time.sleep`
- `--source finmind`: FinMind API only
- `--update`: only fetch dates newer than `last_date` per symbol (incremental)
- On completion, updates `stocks.first_date` / `stocks.last_date`

## §2 Module Structure & Training

### Directory layout

```
finetune_tw/
├── config.py                  # Config dataclass
├── download_data.py           # Multi-source downloader
├── dataset.py                 # MultiStockDataset (SQLite → windows)
├── train_tokenizer.py         # Tokenizer fine-tuning (single GPU + AMP)
├── train_predictor.py         # Predictor fine-tuning (single GPU + AMP)
├── backtest.py                # Backtesting script
├── colab_setup.ipynb          # Colab entry point: install deps, mount Drive, launch training
├── data/
│   └── tw_stocks.db
├── outputs/                   # Checkpoints + logs (symlinked to Drive on Colab)
├── fetchers/
│   ├── __init__.py
│   ├── yfinance_fetcher.py
│   ├── twse_scraper.py
│   └── finmind_fetcher.py
└── configs/
    └── config_tw_daily.yaml
```

### MultiStockDataset

- At init: loads all symbols and their date indices from SQLite
- `__getitem__`: randomly selects a symbol → randomly selects a start index → returns a window of `lookback_window + predict_window` rows as a tensor
- Windows never cross stock boundaries (each stock is an isolated sequence)
- Missing values (trading halts, etc.) are excluded from the valid index pool

### Training pipeline

Two-stage, identical to `finetune_csv/` logic but adapted for single-GPU Colab:

1. `train_tokenizer.py` — fine-tunes the BSQ tokenizer to Taiwan market distribution
2. `train_predictor.py` — fine-tunes Kronos-base autoregressive model

Both scripts:
- Use `torch.device('cuda:0')` directly (no `torchrun`)
- Enable `torch.cuda.amp` (mixed precision fp16) for memory efficiency
- Save a checkpoint every N steps (configurable `save_steps`)
- On startup, auto-detect the latest checkpoint in `outputs/` and resume from it
- Write training loss to a CSV log (viewable from Drive between sessions)

### Google Drive integration (Colab)

`colab_setup.ipynb` handles:
1. `from google.colab import drive; drive.mount('/content/drive')`
2. Symlink `finetune_tw/data/` and `finetune_tw/outputs/` → Drive paths
3. `pip install -r requirements.txt`
4. Launch training cells

### T4 memory budget (Kronos-base, fp16)

| Component | Est. VRAM |
|-----------|-----------|
| Model weights | ~400 MB |
| Gradients + optimizer state | ~1.2 GB |
| Activations (batch=16, lookback=90) | ~2–3 GB |
| **Total** | **~5–6 GB** |

Well within T4's 16 GB. Batch size tunable upward if needed.

### Train / Val / Test split

| Split | Date range |
|-------|-----------|
| Train | 2015-01-01 – 2023-12-31 |
| Validation | 2024-01-01 – 2024-06-30 |
| Test (backtest) | 2024-07-01 – present |

## §3 Backtesting

Pure pandas implementation; no qlib dependency.

### Strategy

Long-only, equal-weight, top-K cross-sectional momentum on model signal:

```
For each rebalance date T:
  1. Load last `lookback_window` days of daily prices for all active stocks
  2. Run inference → predicted close price at T + pred_len
  3. Signal = predicted_close / current_close - 1
  4. Rank all stocks by signal descending
  5. Buy top_k stocks at T+1 open (equal weight)
  6. Hold for hold_days, then rebalance
```

### CLI

```bash
python finetune_tw/backtest.py \
  --top_k 20 \
  --hold_days 5 \
  --pred_len 10 \
  --test_start 2024-07-01
```

### Output

- `backtest_result.png`: cumulative return curve (strategy vs `^TWII`)
- Console: annualised return, Sharpe ratio, max drawdown

### Limitations (by design)

- No transaction costs or slippage modelled
- Long-only, no leverage
- Simplified backtesting; not production-ready

## Out of Scope

- Tick or intraday data
- Short selling / options
- Risk factor neutralisation
- Multi-GPU training
