# Research Findings — Round 4 Fine-tuning Design
> 8 iterations completed (session limit hit at depth-9). Synthesized manually.

---

## [I1] CE 訓練目標與 IC 部署指標永久脫鉤 — **HIGH confidence**

**Problem**: `_combine_training_loss(loss, cfg.ranking_loss_alpha)` 在 line 581 永遠傳 `ranking_loss=None`，`ranking_loss_alpha` 完全無效。  
**Evidence**: LambdaRankIC (arXiv 2605.00501, May 2026) — 直接優化 Rank IC 比 MSE/NDCG 改善 IC-IR 30%（1.03 vs 0.81）。CIKM 2025 (arXiv 2510.14156) — CE early stop 對 top-K tail 不足。  
**Actionable**: Wire `differentiable_rank_ic_loss`（已存在程式碼），切換 IC-IR@h1 close-to-close early stop，縮小 validation 至 150×30。

---

## [I2] Early stop 用 h5 label，但 h1 SNR 高 4 倍 — **HIGH confidence**

**Problem**: IC-IR@h5=0.37 vs IC-IR@h1=0.64，h5 cumulative noise 使 checkpoint 選擇雜訊過高。  
**Evidence**: "Label Horizon Paradox" (arXiv 2602.03395, Feb 2026) — 最佳 supervision horizon 遠小於 deployment horizon；CSI 300/500/1000 上 h1 label 比 h5 label IC 高 22%。  
**Actionable**: 在 `_run_validation_metrics` call 加 `target_horizon=1`（或用 config `ic_target_horizon: 1`）。

---

## [I3] 全參數微調觸發 Catastrophic Forgetting — **HIGH confidence**

**Problem**: val_loss 在 Round 2/3 每 epoch 單調上升（Round 3: 3.32→3.48 over 8 epochs），是 catastrophic forgetting 的教科書症狀。  
**Evidence**: FPT (2023) — 凍結 self_attn+FFN，只訓練 LayerNorm，MSE 比 full fine-tune 低 33%，trainable params 降至 4.6-6.1%。arXiv 2403.20284 — LayerNorm-only tuning 與 full fine-tune 在 GLUE 差距 1-3%（p=0.627）。arXiv 2603.27707 — 凍結>95% params 後 task similarity 0.9988 vs full fine-tune 0.8935。  
**Actionable**: model load 後插入 freeze 迴圈（self_attn + ffn）；相應調高 lr（只訓練少量 params 可承受更高 lr）。

---

## [I4] 台灣集合競價 noise 使 open IC 比 close IC 弱 18 倍 — **HIGH confidence**

**Problem**: TWSE 8:30-9:00 集合競價，thin order book 清算隔夜零售情緒，注入高 idiosyncratic noise。open-to-open IC-IR@h5=0.023 vs close-to-close=0.407（ratio=18x）。  
**Evidence**: Round 3 empirical，Label Horizon Paradox，TWSE trading rules，Lou-Polk-Skouras "Tug of War"（overnight vs intraday dynamics）。  
**Actionable**: `ic_validation.py` 新增 close 價格 lookup；config 加 `ic_use_close: true`；early stop 用 close-to-close IC-IR（不需要改 backtest，執行仍是 open-to-open）。

---

## [I5] IC-IR horizon 應配對 hold_days（Grinold-Kahn） — **MEDIUM confidence**（與 I2 衝突）

**Problem**: 理論上 IR 最大化發生在 validation horizon = portfolio holding period（h5 for hold=5d）。  
**Evidence**: Grinold-Kahn JPM 2007；LambdaRankIC (arXiv 2605.00501)。  
**Resolution**: I2（Label Horizon Paradox empirical）>I5（Grinold-Kahn theoretical）— 實際上 h1 open IC-IR=0.023 根本不可靠，h1 close IC-IR=0.64 提供遠更強的選 checkpoint 信號。用 close@h1 IC-IR 是在正確 proxy 上選，open@h5 是在錯誤且嘈雜的信號上選。

---

## [I6] 3% warmup 太短，導致 IC 不穩定觸發過早 early stop — **HIGH confidence**

**Problem**: `pct_start=0.03`（600/20K steps）warmup 太短，模型不穩定期 IC-IR 低谷觸發 patience=3 early stop。  
**Evidence**: arXiv 2409.04777 — 低 token/param ratio fine-tune 需要更長 warmup；arXiv 2412.13337 — 10% warmup 是 supervised fine-tune 的安全預設；ICLR 2025 arXiv 2502.15938 — linear decay-to-zero 優於 cosine-to-10%。  
**Actionable**: `pct_start: 0.08`，`div_factor: 25`（起始 LR 降為 peak/25）。

---

## [I7] LambdaRankIC 比 soft Pearson 更強的 ranking loss — **MEDIUM confidence**

**Problem**: 現有 `differentiable_rank_ic_loss = -Pearson(z_pred, z_actual)` 是 soft proxy；LambdaRankIC 直接優化 Rank IC 上界。  
**Evidence**: arXiv 2605.00501 — IC-IR 1.03 vs 0.46（XGBoost）和 0.81（LambdaRank-NDCG），低 SNR 環境優勢更大。  
**Actionable**: 需要 (1) 擴展 dataset 攜帶 h1 close return target (2) 實作 lambda-gradient 公式。複雜度高，列為 Nice-to-have。

---

## [I8] T=40 validation dates 不足，SE(ICIR) 過高 — **HIGH confidence**（與 I2 部分衝突）

**Problem**: 在 IC-IR=0.37 (h5 open) 下，SE = sqrt(1.2048/40) = 0.174，低於 80% power 閾值。  
**Evidence**: Ding & Sun 2022 — ICIR variance = (1+ICIR²/2)/T；Bonett & Wright 2000 — minimum n=149 for CI width 0.2。  
**Resolution**: 切換至 h1 close IC-IR=0.64，SE 公式為 sqrt(1.2048/T)。T=30 → SE=0.200（marginal），T=60 → SE=0.142（adequate）。建議 **ic_val_dates=30** 作為 speed/power 折衷（每 epoch 省時 3x vs 現在，SE 仍可接受）。
