# E2 质量锚点 — wave 2 观测与验收小节

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: b32064c
- experiment_date: 2026-07-17（观测访问日期同为 2026-07-17）
- agent_wave: gate2-wave2
- 预注册: `validation/prereg/e2_predictions.md`（wave 1 commit `03f15d9`；§8 由本节回填）
- 预测全量（机器可读）: `validation/e2_quality/ac_predictions.json`
- 证据包: `validation/e2_quality/observed/`（原始 WandB 拉取、42-run 扫描、SmolLM3 部分曲线）

## 1. 观测值与出处（全部 tier T2：实验室官方发布的训练日志/评测记录）

| anchor_id | predicted_loss [CI] | observed_final_loss | 口径 | 出处（2026-07-17 访问） |
|---|---|---|---|---|
| e2_olmo2_7b | 2.001578 [1.941531, 2.061625] | **2.230520** | stage-1 末段最后记录值（step 928646 = 3.895e12 tokens）；末 20 log 点均值 2.2289±0.0108，逐点 σ≈0.011 | Ai2 官方 WandB `ai2-llm/OLMo-2-1124-7B` run `uwjy7cji`（OLMo-2-1124-7B-stage-1-run-11）；论文 arXiv:2501.00656「WandB Logs」脚注 → https://api.wandb.ai/links/ai2-llm/fjn0v0ec |
| e2_pythia_1p4b | 2.301203 [2.231941, 2.370465] | **1.977952** | 终点 checkpoint（step 143000 = 299,892,736,000 tokens）train split 评测 loss；valid 1.960450 / test 1.966468 | EleutherAI 官方 WandB `eleutherai/pythia` run `vd5ogsc6`；README（github.com/EleutherAI/pythia 第 73 行）指认该项目为官方 loss 曲线存档；旁证 run `3bj0rp1k`（bs512 复训、同 299.9B tokens）1.965073 |
| e2_pythia_12b | 2.150205 [2.085698, 2.214711] | **1.757044** | 同口径；valid 1.741183 / test 1.741121 | 同项目 run `kg5ni1dl`；独立旁证 run `3hssg7up` 1.757780（互差 7e-4） |
| e2_smollm3_3b | 2.095947 [1.927055, 2.264839] | — | **注销**：无官方终点观测（理由见 §5） | — |

观测方法注记：
- 三者的观测都不是论文正文表格数字（三篇文献均未发表终点训练 loss 数值），而是**官方训练日志存档中的终点值**——OLMo-2 为训练流最后记录点；Pythia 为官方在终点 checkpoint 上对 train/valid/test 三个 split 的评测 run（train split 值即「训练分布 loss」，与预注册口径「公开 token 数处的训练分布 loss」一致）。
- Pythia 项目内有 600+ 个 hostname 命名的 run。已按 `num_layers`/`hidden_size`/`train_data_paths`/`_step` 全量扫描并留存 42 个含终点 loss 三元组的 run（`observed/pythia_eval_sweep_scan.json`），排除去重版（deduped）、非 1024 batch、早停（71.5k/133k steps）与 2022 年旧版（v0）run，避免挑选。
- OLMo-2 论文图 2 的 OLMo 2 曲线仅覆盖前 ~2.5e12 tokens（x 轴与 OLMo-0424 的 61 万步对齐），若误当终点会读出 ~2.28-2.3 的偏高值；本报告使用官方日志的真实终点 step 928646。

## 2. 点误差（全部公开，不设硬阈值）

| anchor_id | pred − obs | abs_rel_err_pct | 方向 |
|---|---|---|---|
| e2_olmo2_7b | −0.228942 | 10.26% | 预测偏低 |
| e2_pythia_1p4b | +0.323251 | 16.34% | 预测偏高 |
| e2_pythia_12b | +0.393161 | 22.38% | 预测偏高 |
| e2_smollm3_3b | — | — | 注销 |

## 3. 区间覆盖率（核心指标，目标 ≥80% 预校准）

- stock 变体：**0/3**（OLMo-2 观测高于上限；两个 Pythia 观测低于下限）。
- pre2024 变体：**0/3**（预测与 stock 逐位相同，覆盖率必然相同）。
- 按预注册 4 锚点口径：**0/4**。**结论：覆盖率目标未达成**（{0,25,50,75,100}% 步进中落在 0%）。
- 对比：wave 1 的不确定性预算由 spine 3.0%（域内）主导；实测偏差 10-22%，为预算的 3.4-7.5 倍。

## 4. 排序正确性（目标 Kendall τ ≥ 0.9）

- Pythia 族内：预测序 1.4B > 12B，实测序 1.4B(1.977952) > 12B(1.757044) ⇒ **τ = +1.0，达标**（n=2 仅能取 ±1，粒度局限已按 §6-3 记录）。
- 跨族补充证据（3 个可观测锚点）：实测序 Pythia-12B < Pythia-1.4B < OLMo-2-7B；预测序为 OLMo-2-7B < Pythia-12B < Pythia-1.4B。跨族排序不一致（OLMo-2 预测最低、实测最高），与符号分裂的误差结构一致。

## 5. 误差归因（taxonomy）

1. **跨语料校准失配（主因，双向）**：predicted_loss 由 Chinchilla spine（MassiveText 标定）主导；`data_quality`/`effective_data` 两项对四个锚点全为 0，即 mix 差异完全在账本外。实测显示该差异巨大且双向：
   - Pile（GPT-NeoX tokenizer，0.87 截断标准版）比 spine 预期低 16-22%——Pythia 双锚点同向高估；
   - OLMo-mix-1124（DCLM 为主）比 spine 预期高 10%——OLMo-2 低估。论文亦自述「OLMo 2 的 overall training loss 更高是因为训练数据变化」（arXiv:2501.00656 图 2 注）。
2. **符号分裂不可单点修正**：低估与高估并存 ⇒ 不存在全局偏置能同时修复；需要 mix 级 data term（或按语料族分组的 spine）。
3. **族内一致性**：两个 Pythia 高估同向，幅度随 N 增大（16.3% → 22.4%），族内排序仍正确；提示 spine 的 N 指数在 Pile 上偏陡而方向未错。
4. **OLMo-2 token 口径**：wave 1 用 model card「4 Trillion」；论文为 3.9T stage-1 + 3×50B stage-2（soup）。观测取 stage-1 终点（3.895e12），与 4.0e12 锚点差 2.6%，相对 10% 缺口可忽略；stage-2 为不同 mix 且终点是 soup，不构成同一训练分布 loss，不作为观测。
5. **SmolLM3 注销**：官方未发布终点 loss（详单见预注册 §8.1）。该锚点本已是 out-of-regime（11.2e12 > D_MAX=5e12 → 8.058% unc），并叠有 tied-embedding 账本 +8.6% N 的已知 AC 局限。唯一官方曲线材料为主 run 前 7.24e11 tokens（末段 lm_loss≈2.24-2.28，逐点噪声大），不满足锚点 token 口径，仅作轨迹参考：官方主 run 在 0.72T 处的 loss 高于预测终点 2.0959，与「继续训练继续降」的方向相容，但不构成观测。
6. **不确定度预算缺维度**：3.0% spine unc 由拟合域内残差标定，不含跨语料转移分量 ⇒ 覆盖率 0/3 是结构性（under-dispersed）而非运气问题。校准（recalibration）若只做尺度放大，需要的因子 ≥7.5×（22.38%/3.0%）；更合理的路径是引入 mix/data 项而非单纯放宽 CI。

## 6. 全量 vs 截断先验

见 `validation/e2_quality/full_vs_truncated.md`。结论：dense 四锚点上两变体预测逐位相同，覆盖率无差异（各 0/3）；本切片对截断效应不可分。

## 7. 复现路径（wave 2 观测侧）

```bash
# OLMo-2（匿名 GraphQL 读取公开项目）
curl -X POST https://api.wandb.ai/graphql -H 'Content-Type: application/json' -d '{"query":"...","variables":{"entity":"ai2-llm","project":"OLMo-2-1124-7B","run":"uwjy7cji"}}'
# Pythia 全量扫描逻辑与 42-run 结果：observed/pythia_eval_sweep_scan.json
# 逐项原始字段（summaryMetrics/config/tail 统计）：observed/wandb_evidence.json
```

## 8. 结论

- 覆盖率（核心指标）：**0/3（0/4），目标 ≥80% 未达成**。
- 排序：Pythia 族内 **τ=+1.0 达标**（n=2 粒度受限）；跨族排序不一致。
- 误差结构：符号分裂（OLMo-2 −10.3%；Pythia +16.3%/+22.4%），主因是 Chinchilla spine 的跨语料失配，且不确定度预算不含该维度。
- Blocker：SmolLM3 官方训练日志截至 2026-07-17 未发布（blog 承诺「将分享」）——若日后发布，可直接补观测重算覆盖率（分母回到 4）。
