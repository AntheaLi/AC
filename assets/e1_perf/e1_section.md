# E1 性能锚点 · 报告章节（吞吐/显存/训练效率保真度）

> ac_version 0.4.0 · quality_model_version effective_capacity_v2 ·
> git_commit c170cda · experiment_date 2026-07-17 · agent wave "gate2-wave1"
> （wave-2 追加：训练侧 T2 锚点观测与对照）
>
> 预注册：`validation/prereg/e1_predictions.md`（wave-1 commit `b2da20d`；
> §5A 为 wave-2 追加，wave-1 预测数字未改动）。
> 机器可读记录：`validation/e1_perf/anchors.fragment.json`（23 条）。

---

## 1. 方法

**锚点设计**：6 个开放权重模型 × 3 服务指标（decode TBT p50、TTFT p95、峰值显存），
Llama-3.3-70B 按计划取 TP=4/8 两档，共 21 条 T1 锚点；另加 2 条训练侧 T2 锚点
（Llama-3.1-405B MFU、DeepSeek-V3 H800 卡时）。对标 Vidur（MLSys 2024）口径，
计划 §2.3 阈值：预校准中位绝对相对误差 ≤ 15%。

**纪律**：全部 AC 预测在任何实测之前运行并预注册（wave 1）；架构事实全部 T2
（官方 HF config / Meta 官方 GitHub / arXiv 论文，逐字段 provenance，见
`validation/e1_perf/model_configs/*.provenance.json`）；测量协议（vLLM 0.25.1
钉版、并发/prompt/输出长度与 AC `--serving-batch/--context-length/--prompt-len`
1:1 对齐、KV 池按官方架构参数手算固定）在实测前锁定。

**AC 侧设置**：h100（`hardware_specs/h100_sxm.json`）/ h800（子任务 C 的
`h800.json`）；服务锚点 workload = chat 预设 + batch 1 / context 2560 / prompt 2048；
训练锚点 workload = training 预设（ctx 8192 for 405B，4096 for DSv3）。
全部预测命令与原始输出存于 `validation/e1_perf/ac_runs/`。

## 2. AC 预测总表（wave-1 预注册，未校准出厂 priors）

| 锚点 | TP/EP | decode TBT p50 (ms) | TTFT p95 (ms) | peak mem (GiB/GPU) |
|------|-------|--------------------:|--------------:|-------------------:|
| mistral_7b | 1/1 | 8.2165 | 80.1686 | 13.8897 |
| llama31_8b | 1/1 | 8.2165 | 80.1686 | 15.3584 |
| qwen3_8b | 1/1 | 8.4418 | 81.2370 | 15.6939 |
| qwen3_32b | 2/1 | 19.2123 | 159.0602 | 30.8793 |
| llama33_70b | 4/1 | 21.8367 | 167.7227 | 33.1115 |
| llama33_70b | 8/1 | 14.2336 | 100.7443 | 16.5558 |
| gpt_oss_120b | 8/8 | 3.4264 | 25.6073 | 3.8543 |

训练侧（隐含 MFU，AC canonical 口径 6·N_active·TPS/GPU ÷ datasheet bf16 峰值）：

| 锚点 | 硬件 | AC 隐含 MFU | AC 隐含 TFLOPs/GPU |
|------|------|------------:|-------------------:|
| llama31_405b | h100 | 26.24% | 259.5 |
| deepseek_v3 (k=3) | h800 | 9.13% | 90.3 |

## 3. T1 服务锚点状态：AWAITING MEASUREMENT

21 条服务锚点 `observed = null`、`status = "awaiting_t1_measurement"`。
**Owner runbook**：在租赁 H100 SXM×8 节点上执行
`validation/e1_perf/run_benchmarks.sh`（自动安装 vllm==0.25.1、逐锚点
serve+bench+显存采样，产物落 `validation/e1_perf/raw_logs/`；Llama 两个 gated
仓库需预设 `HF_TOKEN`），随后 `python3 validation/e1_perf/parse_logs.py`
解析为 anchors 记录（预测域预填、observed 回填、误差自动计算；
解析器已经合成样本自测）。协议细节与全部工程决定见预注册 §2。

## 4. T2 训练侧对照（wave-2 实测填充）

| 锚点 | AC 预测 | T2 观测 | abs_rel_err | 来源 |
|------|--------:|--------:|------------:|------|
| llama31_405b 训练 MFU (%) | 26.24 | 40.5（区间 38–43 中点） | **35.2%**（对最近边 30.9%） | arXiv:2407.21783v3 §3.3.2 Table 4 |
| deepseek_v3 训练隐含 MFU (%) | 9.13 | 34.64（区间 33.09–34.64） | **73.6%** | arXiv:2412.19437 §1 Table 1 |

换算口径（全部写入 provenance）：观测侧 DSv3 以 "180K H800 GPU-h/万亿 tokens"
→ 1543.2 tok/s/GPU → 6·37e9 FLOPs/token ÷ 989 TF = 34.64%；Llama 直接引用
论文报告 MFU。两侧 MFU 分母均为 datasheet bf16 dense 峰值（与 AC 配置精度字段
一致）；DSv3 实为 FP8 混合精度训练，fp8 峰值口径下观测 17.3% / AC 4.6%（notes）。

## 5. 误差分类学条目（超出 family band 的锚点）

两条 T2 锚点均远超 15% 带（对照计划 §2.3 与簇 B family-band 机制），按纪律
**不删除**，归因如下：

**E-05（roofline/效率表标定；dense 训练 MFU 系统性低估）**
- 证据：llama31_405b 26.24% vs 论文 38–43%（err 30.9–35.2%）。
- 归因：AC 有效峰值约定（NVIDIA 上 peak_flops_tf ≈ 50% datasheet，即 h100 bf16
  495 TF）× 训练效率表合成后，稳态隐含 MFU 上限落在 ~26%；论文报告的 38–43%
  对应 495 TF 有效峰的 76–86%，超出出厂效率表允许区间。PP=14-vs-16 偏差
  敏感度 <1%（PP=18 复跑 25.98%），非因。
- 路径：属 `ac-auto-calibrate fit` 可吸收项（用本锚点 GPU-hours 数据拟合训练
  效率桶）；校准前维持"排序信号"定位。

**E-01+E-05 复合（spec 编码缺口 × MoE 训练 roofline 假设）**
- 证据：deepseek_v3 9.13% vs 34.64%（err 73.6%）；k=0/k=3 两形态同值，
  说明与 dense-prefix 无关，瓶颈在 MoE 训练项本身。
- 归因（候选，按贡献预期排序）：(i) AC MoE all-to-all 训练通信不重叠，
  而 DeepSeek DualPipe/DeepEP 实现近全重叠（roofline 假设）；
  (ii) capacity_factor 1.25 路由填充与 load-balance 惩罚计入每步
  （口径差）；(iii) moe_mla 效率桶（compute 0.42–0.46）较 dense 桶更低
  （效率表标定）；(iv) AC 按 bf16 记账，DSv3 实际 FP8 混合精度
  （spec 编码缺口）；(v) AC 自报 training_memory 157 GiB/GPU > 80GB，
  TP8×EP8×PP1 布局相对真实 2048 卡训练欠配（协议口径，已记录）。
- 路径：(i)(iii) 走校准；(ii) 评估是否应将 capacity 填充从稳态步时中剥离
  （若确认，回流第一道开 issue）；(iv) 待 fp8 精度字段进入 schema 后重跑。

**已预注册、待 T1 验证的分类条目**（不影响本节 T2 结论）：gpt-oss-120b 服务
显存锚点的 bf16-vs-MXFP4 编码差（预期大幅高估）；mistral/llama31-8b 同形对照
（词表不敏感性直接测量）；vLLM 框架开销口径（CUDA context/cudagraph/KV 预留）。

## 6. 已知边界

- 训练侧锚点为 T2（论文/官方报告），非自测；DSv3 观测隐含 MFU 依赖
  6·N_active·T 换算约定（N_active=37B 取自报告）。
- AC 训练预测使用出厂 priors；`training_tps` 口径为 per-replica（TP×PP×CP），
  per-GPU 换算按 cli_compile 的 canonical 公式（6·N_active·TPS_per_GPU/峰值）。
- Llama-405B AC 配置以 PP=14 近似论文 PP=16（schema 整除约束；敏感度 <1%）。
- 服务锚点的观测值在 owner 执行 runbook 前不存在；本节数字不构成对
  vLLM 实测值的任何预期声明之外的内容（预期声明以预注册 §1.2/§4 为准）。
