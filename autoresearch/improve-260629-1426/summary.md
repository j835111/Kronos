# AutoResearch Summary — Round 4 Fine-tuning Design
> 2026-06-29 | Branch: research/pretrained-finetune-round4

## Research Stats
- Total iterations: 8（session limit at depth-9，synthesized manually）
- Insights: 5 new + 3 extension
- Coverage: training-objective, label-horizon, finetuning-strategy, ranking-loss, open-price-prediction, warmup, IC-validation-size, horizon-matching
- Status: PARTIAL（7/15 depth iterations failed），core findings captured

## Key Findings

| # | Finding | Confidence | Impact |
|---|---------|------------|--------|
| I1 | `ranking_loss_alpha` 是 dead code，ranking loss 從未計算 | HIGH | Medium |
| I2 | Label Horizon Paradox — h1 SNR 比 h5 高 4x | HIGH | **High** |
| I3 | Full fine-tune = catastrophic forgetting（FPT 解法） | HIGH | **High** |
| I4 | TWSE 集合競價 noise 使 open IC 弱 18x | HIGH | **High** |
| I5 | Grinold-Kahn h5 原則（與 I2 衝突，I2 勝出） | MEDIUM | Low |
| I6 | 3% warmup 太短，觸發過早 early stop | HIGH | Medium |
| I7 | LambdaRankIC 優於 soft Pearson | MEDIUM | Medium |
| I8 | T=40 validation dates SE=0.174，統計力不足 | HIGH | Medium |

## Decision Made

**起點**：Round 0（`j835111/kronos-tw-finetune@round-0`），保留台股 domain adaptation，加 FPT freeze 防止退步。

**三個 Must-have**：
1. M1 FPT freeze（10 行程式碼）
2. M2 close-to-close IC early stopping（~30 行）
3. M3 h1 label + 延長 warmup（config + 1 行）

## Output Files
- `research-findings.md` — 8 個 insights 完整說明
- `improvement-plan.md` — 優先順序 + config 對照表
- `prd-round4-finetune.md` — 完整 PRD（程式碼片段 + 驗收條件）
