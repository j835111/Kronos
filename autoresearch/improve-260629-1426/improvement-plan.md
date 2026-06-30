# Improvement Plan — Kronos Round 4 Fine-tuning
> Synthesized from 8 research iterations. Branch: research/pretrained-finetune-round4

---

## 目標
- Sharpe ≥ 1.5（目前最佳 Round 0: 1.356）
- MaxDD < 20%（目前 35%）
- 預算：RunPod A40 ≤ $5.94（~6hr）

---

## Must-have（3 項，必做）

### M1 — FPT Selective Freeze（凍結 self_attn + ffn）
**信心**: HIGH | **複雜度**: LOW（10 行程式碼）  
凍結所有 TransformerBlock 的 `self_attn` 和 `ffn` weights，只訓練 LayerNorm（`norm1`、`norm2`、`model.norm`）和 head（`DualHead`）。Trainable params 降至 ~5-7M（102M 的 5-7%）。

直接解決 Round 2/3 val_loss 單調上升的 catastrophic forgetting 根本原因。

```python
# train_predictor.py — model load 後插入
for name, param in model.named_parameters():
    freeze = any(k in name for k in ['self_attn', 'ffn'])
    param.requires_grad_(not freeze)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M")
```

---

### M2 — 切換 Close-to-Close IC Early Stopping
**信心**: HIGH | **複雜度**: MEDIUM（~30 行）  
IC 驗證從 open-to-open（IC-IR=0.023）切換至 close-to-close（IC-IR=0.64），SNR 提升 28 倍。Early stop 才能可靠選出最佳 checkpoint。

需要：(1) `train_predictor.py` 的 `actual_open_cache` 擴展為同時快取 close；(2) `_run_validation_metrics` 加 `ic_use_close` 參數；(3) config 加 `ic_use_close: true`。

---

### M3 — Label Horizon h1 + 延長 Warmup
**信心**: HIGH | **複雜度**: LOW（config + 1 行）  
(a) `target_horizon=1`（Label Horizon Paradox — h1 SNR 高 4x）  
(b) `pct_start=0.08, div_factor=25`（延長 warmup，防止前幾 epoch IC 低谷觸發 patience）

---

## Nice-to-have（2 項，時間允許）

### N1 — 增加 ic_val_dates 至 60
SE(ICIR@h1=0.64) = sqrt(1.2048/60) = 0.142（adequate）。目前 30 dates SE=0.200（marginal）。若 A40 驗證 30 dates 快（< 5 min/epoch），可升級至 60。

### N2 — LambdaRankIC loss（arXiv 2605.00501）
取代目前 broken 的 soft Pearson ranking loss。需要擴展 DataLoader 攜帶 h1 close return、實作 pairwise lambda-gradient。複雜度高，列為可選 bonus。

---

## Skip

- **Tokenizer retrain**：節省 ~1hr A40 budget，收益不確定，skip
- **LoRA**：需新依賴，M1 frozen fine-tune 已達同等效果
- **更換 pretrained 起點**（NeoQuasar/Kronos-base raw）：Round 0 已有台股 domain adaptation，從 Round 0 起點更佳

---

## Config 對照表

| 參數 | Round 0/3 | Round 4 | 說明 |
|------|-----------|---------|------|
| `hf_revision` | `null` / `round-0` | `round-0` | 從 Round 0 起點 |
| `hf_revision_out` | `round-3` | `round-4` | |
| `ic_val_symbols` | 500 | 150 | 減少 validation 推論時間 |
| `ic_val_dates` | 40 | 30 | 速度/統計力折衷（SE≈0.20） |
| `ic_use_close` | — | `true` | close-to-close IC（SNR 28x） |
| `ic_target_horizon` | — | `1` | Label Horizon Paradox |
| `early_stop_patience` | 3 | 5 | 配合延長 warmup |
| `basemodel_epochs` | 20 | 20 | 保持不變 |
| warmup `pct_start` | 0.03 | 0.08 | hardcoded in train_predictor.py |
| warmup `div_factor` | 10 | 25 | hardcoded in train_predictor.py |

---

## DECISION NEEDED

1. **h1 vs h5 early stopping**：Grinold-Kahn 理論支持 h5 配對 hold=5d；Label Horizon Paradox empirical 支持 h1。**建議 h1 close**（open-to-open h5 IC-IR=0.023 根本不能選 checkpoint；close h1 IC-IR=0.64 可以）。
2. **ic_val_dates 30 vs 60**：30 快但統計力邊緣（SE=0.20）；60 統計力足夠（SE=0.14）但推論時間多 2x。若 A40 每 epoch 驗證 30 dates < 3 min，直接升 60。

---

## 預算估計（A40 $0.89/hr）

| 步驟 | 時間 | 費用 |
|------|------|------|
| Tokenizer | Skip | $0 |
| Predictor 訓練（20 epoch × ~8 min/epoch） | 2.7hr | $2.40 |
| IC 驗證（150×30，每 epoch ~3 min） | 1hr | $0.89 |
| Backtest | 30 min | $0.44 |
| Buffer | — | $1 |
| **Total** | **~5hr** | **≈$4.73** |

**在 $5.94 預算內。**
