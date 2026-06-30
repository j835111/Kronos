# PRD — Kronos Round 4 Fine-tuning
> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## 問題陳述

Round 0（Sharpe 1.356）是目前最佳結果，Rounds 1-3 全部退步。根本原因：
1. **Catastrophic forgetting**：全參數微調破壞 pretrained temporal representations（val_loss 單調上升）
2. **Early stopping signal noise**：open-to-open IC-IR@h5=0.023，SNR 不足以可靠選 checkpoint
3. **Warmup 太短**：3% warmup 導致 IC 不穩定觸發 patience=3 過早 stop

Round 4 目標：Sharpe ≥ 1.5 / MaxDD < 20% / 預算 $5.94 A40。

---

## 成功指標

| 指標 | 目標 | 說明 |
|------|------|------|
| Sharpe | ≥ 1.5 | open/open v2，top_k=10，hold=5d |
| MaxDD | < 20% | 測試期 2024-07-01 起 |
| Ann Return | > 30% | |
| val_loss | ≤ 3.50 | 不應高於 Round 3（3.644）|
| Close IC-IR@h1 | ≥ 0.65 | Early stop 健康指標 |

---

## 技術方案

### 程式碼更改

#### C1：FPT Selective Freeze（必做）

**檔案**: `finetune_tw/train_predictor.py`  
**位置**: model load 後（約 line 443）

```python
# 凍結 self_attn + ffn，只訓練 LayerNorm + head
for name, param in model.named_parameters():
    freeze = any(k in name for k in ['self_attn', 'ffn'])
    param.requires_grad_(not freeze)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M ({100*n_trainable/n_total:.1f}%)")
```

**同時**：optimizer 只傳 trainable params（可以提高 lr）：
```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.predictor_lr,
    ...
)
```

---

#### C2：Close-to-Close IC Early Stopping（必做）

**檔案**: `finetune_tw/train_predictor.py`  
**Step A**：擴展 `actual_open_cache` → `actual_price_cache`（同時快取 open + close）

在 line 540 附近，把：
```python
actual_open_cache = {}
for sym in val_universe:
    df = query_symbol(...)
    if len(df):
        actual_open_cache[sym] = pd.Series(df["open"].values, index=...)
```

改為：
```python
actual_price_cache = {}
for sym in val_universe:
    df = query_symbol(...)
    if len(df):
        actual_price_cache[sym] = {
            "open":  pd.Series(df["open"].values,  index=pd.DatetimeIndex(df["date"])),
            "close": pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"])),
        }
```

**Step B**：`_actual_open_lookup` 改為支援 close：
```python
def _actual_price_lookup(cfg, cache, sym, last_ctx_date, n, col="open"):
    price_series = cache.get(sym, {}).get(col)
    if price_series is None:
        return [float("nan")] * n
    ...  # 同原邏輯，但使用 price_series
```

**Step C**：`_run_validation_metrics` 加參數 `use_close=False`，當 `use_close=True` 時傳 close lookup。

**Step D**：呼叫端 (line 625) 改為：
```python
use_close = getattr(cfg, "ic_use_close", False)
target_h  = getattr(cfg, "ic_target_horizon", 5)
actual_fn = lambda sym, last, n: _actual_price_lookup(
    cfg, actual_price_cache, sym, last, n, col="close" if use_close else "open"
)
val_ic, ic_ir = _run_validation_metrics(
    cfg=cfg,
    predict_batch_fn=predict_fn,
    actual_lookup=actual_fn,
    val_universe=val_universe,
    val_dates=val_dates,
    prepared_batch_predict_fn=prepared_predict_fn,
    contexts_by_date=validation_contexts,
    target_horizon=target_h,
)
```

**Step E**：log 欄位名更新：
```python
log_path.write_text("epoch,step,train_loss,val_loss,val_ic,ic_ir\n")
```

---

#### C3：Warmup 延長（必做）

**檔案**: `finetune_tw/train_predictor.py`  
**位置**: line 519

```python
# Before:
pct_start=0.03, div_factor=10,
# After:
pct_start=0.08, div_factor=25,
```

---

### Config 更改（`config_tw_daily_a40.yaml`）

```yaml
# ── 起點 ──
hf_revision: "round-0"       # 從 Round 0 起點（已有台股 domain adaptation）
hf_revision_out: "round-4"

# ── IC Early Stop ──
ic_use_close: true            # 新 flag — close-to-close IC（SNR 28x）
ic_target_horizon: 1          # 新 flag — Label Horizon Paradox h1

# ── Validation 大小 ──
ic_val_symbols: 150           # 500 → 150（加速 3x）
ic_val_dates: 30              # 40 → 30（速度/統計力折衷）

# ── Training ──
basemodel_epochs: 20
early_stop_patience: 5        # 3 → 5（配合延長 warmup）

# ── Backtest ──
top_k: 10
hold_days: 5
```

---

## 風險與緩解

| 風險 | 可能性 | 緩解 |
|------|--------|------|
| FPT freeze 不夠靈活，台股特化不足 | MEDIUM | 若 IC-IR < 0.50，嘗試解凍最後 2 個 block |
| T=30 dates SE=0.20 仍偏高 | MEDIUM | 若前 5 epoch IC-IR 波動 > 0.15，增加 ic_val_dates=60 |
| Round 0 起點再次退步 | LOW | M1（freeze）直接阻止 catastrophic forgetting |
| A40 超時 | LOW | 估算 4.7hr，$5.94 預算內 |

---

## DECISION NEEDED

> 以下兩點需要你做決定：

**D1 — Early stop horizon**  
- h1 close IC-IR（推薦）：高 SNR（0.64），可靠 checkpoint 選擇，但理論上與 hold=5d 不對齊
- h5 close IC-IR：理論對齊 Grinold-Kahn，但 h5 close IC-IR 未知（只知道 open@h5=0.023）

**D2 — ic_val_dates 30 vs 60**  
- 30（預設）：SE=0.200，速度快，budget 安全
- 60：SE=0.142，統計力足夠，但每 epoch 多 2 倍推論時間

---

## 驗收條件

1. `train_log.csv` 顯示 `ic_ir` 欄位在訓練期間波動 < 0.15（穩定）
2. `best_model` 對應 epoch 的 `ic_ir_close_h1 ≥ 0.65`
3. backtest Sharpe ≥ 1.5，MaxDD < 20%
4. `val_loss` 不超過 Round 0 的 3.644

---

## 實作順序

1. `train_predictor.py`：C1（freeze）→ C2（close IC）→ C3（warmup）
2. `config_tw_daily_a40.yaml`：加入新 flags
3. 本地 smoke test（`--max_steps 100` 確認 freeze 正確、close IC 有值）
4. 推送 git → RunPod 開 A40 訓練
5. 監控 `train_log.csv` 前 5 epoch，確認 IC-IR 穩定
