# E2 全量 vs 截断（pre2024）先验对比 — wave 2 终评

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: b32064c
- experiment_date: 2026-07-17
- agent_wave: gate2-wave2
- 先验文件: `validation/e2_quality/priors_pre2024.yaml`（SHA256 `e59cbdee29a5658a2ba5dac44589a64d645c29de43a9639803c7e8341e06e2a0`，构建脚本 `build_priors_pre2024.py`，确定性已验证）
- 变体约定: stock = `AC_QUALITY_DEFAULTS` 置空；pre2024 = 上述 overlay（仅保留 2024-01-01 前有公开锚定的 term 值，去锚定值归零）

## 1. 预测侧：dense 四锚点零差异（wave 1 事实，复核引用）

| anchor_id | stock predicted_loss | pre2024 predicted_loss | Δloss | Δunc |
|---|---|---|---|---|
| e2_olmo2_7b | 2.001578 | 2.001578 | 0 | 0 |
| e2_pythia_1p4b | 2.301203 | 2.301203 | 0 | 0 |
| e2_pythia_12b | 2.150205 | 2.150205 | 0 | 0 |
| e2_smollm3_3b | 2.095947 | 2.095947 | 0 | 0 |

机制：截断只触及三个 term——`state_residual`（hybrid/Jamba）、`effective_capacity`/`moe_residual`（MoE）、`mtp_residual`（MTP）。四锚点全部为 dense、非 MoE、非 MTP、非 hybrid，三个 term 在两种先验下恒等于 0；决定预测的保留 term（spine、architecture_residual、vocab_residual、risk、context_utility）在两变体中逐项相同。

## 2. Overlay 确实被消费的探针（wave 1 记录，非空转证据）

- hybrid/Jamba 臂：`state_residual` 值 0.00033 → 0.0，predicted_loss 2.0410 → 2.0741；
- DeepSeekMoE 臂：`effective_capacity` 值 −0.00597 → 0.0（N_eff 2.20B → 1.92B = N_active），predicted_loss 2.1645 → 2.1775。

即 overlay 在「term 会触发」的架构上生效；零差异是 dense 切片的结构属性，不是 overlay 未加载。

## 3. 观测侧：两变体覆盖率无差异（wave 2 新增）

| anchor_id | observed | stock CI 命中 | pre2024 CI 命中 |
|---|---|---|---|
| e2_olmo2_7b | 2.230520 | 否 | 否 |
| e2_pythia_1p4b | 1.977952 | 否 | 否 |
| e2_pythia_12b | 1.757044 | 否 | 否 |
| e2_smollm3_3b | 注销（无官方观测） | — | — |

预测逐位相同 ⇒ 覆盖率必然相同：**两变体各 0/3（按 4 锚点口径各 0/4）**。观测值的引入不产生任何可分性——这在逻辑上先于观测即可知，此处仅作登记确认。

## 4. 结论与建议

1. **本切片结论**：E2 dense 切片上「全量 vs 截断」差异恒为零；这本身是 dense 架构族的时间稳健性证据（2024 年前的锚定已足以复现 stock 预测），与预注册 §3 的预期一致。
2. **截断效应在本实验中不可检验**：要检验截断是否损伤 2024+ 架构的预测力，需要三个被去锚定 term 会触发的锚点——hybrid/state-space（如 Jamba）、MoE（如 DeepSeekMoE/Qwen3MoE）、MTP（如 DeepSeek-V3）。建议把该检验路由到 E3 retrodiction 的 hybrid/MoE 臂（其配置已含这些架构），或未来增补 E2 的 hybrid/MoE 锚点后重跑本对比。
3. 覆盖率失败（0/3）与先验变体无关：两个变体共享同一条 Chinchilla spine 与相同的 vocab/architecture 残差，误差主因（跨语料失配，见 `e2_section.md` §5）对两变体同等地成立。
