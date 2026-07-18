# 算模 AC — Architecture Compiler 

<p align="center">
  <a href="https://github.com/AntheaLi/AC/actions/workflows/ci.yml"><img src="https://github.com/AntheaLi/AC/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/archc/"><img src="https://img.shields.io/pypi/v/archc.svg" alt="PyPI"></a>
  <a href="https://github.com/AntheaLi/AC/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/archc/"><img src="https://img.shields.io/pypi/status/archc.svg" alt="Status"></a>
  <a href="https://antheali.github.io/ac-demo/"><img src="https://img.shields.io/badge/demo-live-6f42c1" alt="Live Demo"></a>
</p>

![算模AC - ArchCalc](assets/image.png)

AC is a compiler for model architecture design under hardware constraints. It can take hardware platform, param count, training tokens, serving workload, and an optional basemodel, 
AC optimizes for multi-objective Pareto-improving model architectures and architecture deltas. It currently supports greenfield generation, 
basemodel modification, and modifier eval.
The goal is to make architecture design less like folklore and more like multi-objective Pareto optimization.
It serves as a quantitative **pre-flight** check, it does NOT replace training. 

Check out demo here: [ac-demo](https://antheali.github.io/ac-demo/)

> **TL;DR (60s)** — Give AC a hardware target + budget (`ac-compile --hardware h100 --params 7 --tokens 2 …`); get back a Pareto-optimal architecture, the binding constraint that decided it, and the price of relaxing it. Why trust it: an [anchor validation study](asset/anchor_report.md) pre-registered predictions before comparing them to the public record — decision retrodiction **4 hit + 1 partial of 5** (DeepSeek-V3 MLA, Llama-3 GQA-8, Mistral SWA, first-K-dense, EP=72-on-NVL72), ranking fidelity **Kendall τ = 1.0**, with every miss published in an error taxonomy instead of deleted. Absolute numbers are priors until you calibrate — [read the 3 lines below](#calibration-vs-ordinal-use). Run it in 30s: `pip install archc && ac-compile --hardware h100 --params 7 --tokens 2 --context 8192 --serving-tbt 50 --serving-batch 32 --tp 8 --pp 1 --dp 8`

Three composable use cases, one shared config format:

- **Greenfield**  Given compute -> architecture | `ac-compile --hardware H --params N --tokens T …` 

- **Modifier**  Given compute + a base architecture -> modifier | `ac-compile --baseline-config CONF …` 

- **Delta influence**  Given compute + base + delta -> influence |  `ac-delta-eval --baseline-config CONF …`

Also check out the [AC-Harness](https://github.com/AntheaLi/AC-harness), which is a loop scaffold 
built to automate the process. It can also exist as a thin layer that sits beside existing training, eval, and benchmarking stack. 

---

## Install

```bash
# from PyPI (recommended)
pip install archc

# or, from a source checkout: editable install from the repo root
pip install -e .

# or run directly without installing (from the repo root)
python ac/cli_compile.py --help
```

After install the console scripts are on your `PATH`:

```bash
ac-compile        --help    # greenfield / modifier entry
```

No external runtime dependencies beyond Python ≥ 3.10 and PyYAML.

---


## Calibration vs. Ordinal Use

AC is a **forward proxy**, not a measurement system. Without calibration, AC is **ordinal**:

- **Pareto ranking.** Given identical priors, two candidates'
  *relative* ordering on the (loss, TBT, memory, TPS) frontier is much
  more robust than either's absolute number. Use the Pareto CSV.
- **Binding-axis identification.** The 10-axis stress vector and shadow
  prices tell you which constraint would actually move quality if
  loosened. That's a structural answer; it doesn't depend on a tight
  loss calibration.

The current coefficients are calibrated to public ablations, used as priors and documented in
`ac/quality_defaults.yaml`. The uncalibrated starter mode can be different from your private internal traces. 
The absolute loss numbers (and the TPS/TBT predictions) can be **biased** relative to what your stack
actually produces.


Calibrate, if you want **absolute numerals**:

- Run `ac-auto-calibrate fit --measurements <your_traces>.jsonl` against
  ≥12 measured runs spanning the architecture families you care about
  (see the Auto-calibration section for the gates).
- Pass `AC_QUALITY_DEFAULTS` and `AC_HARDWARE_SPEC_DIR` from the
  resulting pack into `ac-compile`. The emitted config will then carry
  a `confidence_envelope` block and `calibration_warnings` will name
  any gates the pack didn't pass.
- Keep separate packs per cluster / kernel / datamix / training recipe
  — one global pack will mask all the interesting variance.

For any decision that depends on absolute loss (e.g. "is 7B at 2T
better than 13B at 1T at our budget?"), an uncalibrated AC can be 
misleading. For comparative decisions ("does adding MLA relieve the
binding axis without spending >1% loss on a frontier already at TP=8?"),
AC's structural answer is the value it adds.


---

## Quickstart

```bash
# 1) Greenfield: 7B dense on H100
ac-compile \
  --hardware h100 --params 7 --tokens 2 --context 8192 \
  --serving-tbt 50 --serving-batch 32 --tp 8 --pp 1 --dp 8 \
  --output-config out/mistral_arch.json \
  --output-justification out/mistral_arch.md \
  --output-pareto out/mistral_pareto.csv

# 2) Modifier: Pareto-front delta against Mistral-7B
ac-compile \
  --baseline-config configs/mistral_7b.json \
  --hardware h100 --tp-options 4,8 \
  --quality-risk-budget-pct 1.0 --allow-quality-spending \
  --out out/mistral_modifier

# 3) Delta influence: what does GQA(group_size=8) do to Mistral-7B at 32k?
ac-delta-eval \
  --baseline-config configs/mistral_7b.json \
  --hardware h100 --tp 8 --workload long_context \
  --apply swap_attention_to_gqa --apply-args group_size=8 \
  --out out/mistral_delta_gqa
```

---

## Greenfield

Search the full architecture lattice for the Pareto-front winner under the
given hardware and constraints. No baseline config required.

```
ac-compile [OPTIONS]

required
  --hardware {h100, b200, gb200_nvl72, h800, tpu_v5p, tpu_v5e, trainium2, trn2, trainium3, trn3}
  --params  N        (billions; supports "7" or "7B")
  --tokens  N        (trillions; supports "2" or "2T")

```

### ac-compile args 

<details>

```
workload
  --context           int       sequence length (default 8192)
  --prompt-len        int       prefill length override
  --output-len        int       expected generation length (default 512)
  --concurrency       int       concurrent serving requests (default 256)
  --scheduler         {continuous, static, chunked}

selection
  --objective-profile  research_quality | loss_only | balanced | quality |
                       latency | serving_cost | training_cost
                       default: research_quality

serving budgets
  --serving-tbt   ms      time-between-tokens p95
  --serving-ttft  ms      time-to-first-token p95
  --serving-batch int     batch size (default 32)

parallelism
  --tp / --pp / --dp   degrees (defaults 8/1/8)
  --training-cluster-gpus  minimum cluster size; derives candidate-specific
                           DP across TP/PP/CP search and rounds DP for EP
  --cp / --cp-method   context parallel: ring | ulysses
  --cp-options         comma-list of CP degrees to sweep

architecture sweeps
  --allow-state            enable state/hybrid candidates
  --state-type             mamba2 | mamba | s4 | s5 | s6
                           | gla | kda | deltanet | gated_deltanet
                           | rwkv7 | retnet | linear_attention
                           | parallel_heads | moh | hydra
                           | sliding_window
  --placement-strategy     first_periodic_last, interleaved, periodic
  --state-precision        bf16 | fp16 | fp32

  --allow-moe              enable MoE FFN candidates
  --max-total-params-b     MoE memory ceiling (total params billions)
  --moe-n-experts          comma-list (e.g. "64,128,256")
  --moe-top-k              comma-list
  --moe-granularity        comma-list of expert-granularity targets
                           (1.0 = coarse Mixtral-style; 0.25 =
                           DeepSeek-V3-style fine-grained experts)
  --ep-topology            single_axis | cross_axis
  --dense-ffn-layers       comma-list of first-K-dense layer counts
  --ep-options             comma-list of expert-parallel degrees

  --allow-mla              enable Multi-head Latent Attention candidates
  --mla-kv-latent          comma-list of c_kv options (default 512)
  --mla-q-latent           comma-list of c_q  options (default 1536)

  --allow-local-global     enable local:global attention interleave sweep
                           (GPT-OSS / Gemma-2 / Llama-4 pattern): a fraction
                           of layers use sliding-window attention, the rest
                           stay global (full/GQA, or MLA with --allow-mla)
  --local-windows          comma-list of local windows (default "1024,4096")
  --local-global-ratios    comma-list of local:global ratios
                           (default "1:1,3:1,7:1")

  --allow-mtp              enable Multi-Token Prediction
  --mtp-depths             comma-list (e.g. "0,1,2")

  --allow-rope-scaling             enable per-method RoPE sweep
  --rope-original-max-position     pretrain context (default 8192)
  --rope-scaling-methods           comma-list: yarn,ntk,longrope,pi,none

evaluated attention transforms (scored by quality + throughput models)
  --nsa                            require Native Sparse Attention on every candidate
    --nsa-compress-block-size      (default 64)
    --nsa-compress-block-stride    (default 16)
    --nsa-select-block-size        (default 64)
    --nsa-select-top-k             (default 16)
    --nsa-window-size              (default 512)

  --yoco                           require YOCO KV sharing on every candidate
    --yoco-n-self-attn-layers      (default 1)
    --yoco-share-pattern           single_source (the calibrated YOCO topology)

compressed / indexer attention sweeps (Wave 32)
  --allow-csa                      Compressed Sparse Attention candidates
    --csa-block-sizes              comma-list (default 64,128)
    --csa-top-k-blocks             comma-list
    --csa-compression-dim          (default 64)
  --allow-indexshare               DSA-style bucketed lightning-indexer candidates
    --indexshare-buckets           comma-list (default 64,128)
    --indexshare-top-k             comma-list (default 4,8)
    --indexshare-index-dim         (default 64)
  --allow-msa                      multi-scale attention candidates
    --msa-windows                  comma-list (default 512,1024)
    --msa-dilated-top-k            comma-list
    --msa-global-top-k             comma-list

precision search
  --precision-modes        comma-list: bf16, fp8_ffn, fp8, fp4, mxfp4, mxfp6
  --kv-dtypes              comma-list: bf16, int8, fp8, int4, fp4

outputs (paths)
  --output-config           default arch.json
  --output-justification    default arch.md
  --output-pareto           default pareto.csv
  --output-shadow-prices    default shadow_prices.json
  --output-assumptions      default not written
  --output-model-card       default not written
  --output-implementation   generated PyTorch architecture scaffold
  --implementation-class-name  class name for --output-implementation
  --no-shadow-prices        skip the shadow-price pass (faster)
  --max-candidates          optional greenfield cap after candidate dedupe
  --progress-every          print evaluation progress every N candidates
                            (default auto: every 1000 on large searches)
  --quiet                   suppress progress logs
```
</details>

Training memory and DP communication assume FSDP/ZeRO-3: weights,
gradients, and optimizer state are sharded across DP ranks. The training
replica is `TP x PP x CP`; EP overlays DP for training but expands a serving
instance. Compare topology choices with the emitted per-GPU, per-replica,
and aggregate TPS fields rather than the legacy `training_tps` field alone.

Alternatively you can also use yaml to pass in args (more details below):

```bash
ac-compile --recipe configs/recipes/<YOUR RECIPE>.yaml
``` 


### Output and example

One row per Pareto-frontier candidate, sorted by the same uncertainty-aware
tiebreak the picker uses, so `rank=1` always agrees with the row that has
`selected=True`. Loss columns:

- `predicted_loss` — point estimate from the quality model spine + residuals.
- `loss_ci_low`, `loss_ci_high` — symmetric uncertainty band around
  `predicted_loss` (half-width = `uncertainty_total_pct/100 × predicted_loss`).
  Populated for every row. Two rows whose `[loss_ci_low, loss_ci_high]`
  intervals overlap are *quality-equivalent within modeled uncertainty*;
  prefer the one that dominates on the throughput/memory axes.
- `uncertainty_total_pct` — quality-model total relative uncertainty (%).
  When `auto-calibrate` runs against your lab traces and writes a
  `quality_overrides.json` pack, this column is the scaled, post-calibration
  uncertainty.

Treat the loss column as a *ranking* signal rather than a forecast unless
you have run `ac-auto-calibrate` against your lab's measurements.
```

#### example: MAI-Thinking-1-ish

```bash
ac-compile \
  --hardware b200 --params 35 --tokens 8 --context 131072 \
  --serving-tbt 60 --serving-batch 8 \
  --tp 8 --pp 4 --dp 4 \
  --allow-moe --moe-n-experts 256 --moe-top-k 8 \
  --allow-mla --mla-kv-latent 512 --mla-q-latent 1536 \
  --allow-mtp --mtp-depths 0,1 \
  --allow-rope-scaling --rope-original-max-position 32768 \
                       --rope-scaling-methods longrope \
  --cp 4 --max-total-params-b 800 \
  --output-config out/mai_arch.json --no-shadow-prices
```



## Modifier

Holds the architecture *family* fixed (uses the baseline as anchor) and
searches the local Pareto-frontier of modifications around it.

```
ac-compile --baseline-config PATH [OPTIONS]

required
  --baseline-config  PATH    JSON config emitted by greenfield or any existing model stripped in the format 
```


### modifer args 

<details>
 
```
scoring
  --allow-quality-spending   allow non-zero loss-proxy delta
  --quality-risk-budget-pct  max loss-proxy %-delta (default 1.0)
  --top-modifications        rows to render in reports (default 8)

parallelism sweep
  --tp-options       comma-list (e.g. "4,8")

workload
  --context, --serving-tbt, --serving-ttft, --serving-batch, --prompt-len

output
  --out DIR     destination for config.json + baseline_delta.md +
                pareto.csv + shadow_prices.md + justification.md +
                assumptions.md + model_card.md
```

</details>

Modifier mode writes **fixed file names** into the `--out` directory; the
greenfield `--output-config` / `--output-*` path flags do not apply here
and are ignored with a stderr warning.

Modifier mode preserves the baseline's architecture family exactly (including
state/MoE/MLA/compressed attention, MTP, CP, RoPE, NSA, YOCO, and local/global
layout) while searching nearby depth, KV-head, FFN/expert-width, precision,
KV-cache, and TP choices. Greenfield-only family-enabling flags do not add a
new family to an existing baseline; use delta-eval for an explicit family
transition or greenfield mode for a broad family search.


## Delta influence

Quantitative effect of one (or a chain of) named transformations against a
specific baseline architecture. Outputs a one-page Markdown report + JSON
+ 3-row Pareto CSV.

```
ac-delta-eval --baseline-config PATH --apply NAME [OPTIONS]

required
  --baseline-config  PATH    JSON config (greenfield output OR hand-written)
  --apply            NAME    one of REGISTRY (repeatable)
    --apply-args  k=v        args for the most recent --apply (repeatable)

```

### Delta args 

<details>
 
```
baseline / hw
  --hardware    h100 | b200 | tpu_v5p | trainium2 | trn2 | trainium3 | trn3
  --tp / --pp / --dp

workload preset (preset = chat | batched | long_context | training)
  --workload          PRESET   default chat
  --serving-batch     int      override the preset
  --context-length    int      override the preset
  --prompt-len        int

other
  --no-pareto        skip Pareto-position classification (faster)
  --json             emit JSON only (no Markdown / CSV)
  --stdout           print Markdown to stdout instead of writing files
  --out      DIR     destination directory
```

</details>


Available delta names (REGISTRY):

| Name | Effect | Legal --apply-args keys |
|---|---|---|
| `swap_attention_to_gqa` | Set `n_kv_heads = n_heads / group_size` | `group_size` |
| `swap_attention_to_mla` | Replace full attention with MLA at `latent_dim` | `latent_dim` |
| `swap_attention_to_swa` | Sliding-window attention at `window_size` | `window_size` |
| `interleave_local_attention` | Local:global interleave — `ratio` of layers become SWA at `window`, rest stay global | `ratio` (e.g. `"3:1"`), `window` |
| `add_state_layers` | Replace a fraction of attention with a state mixer | `ratio` (e.g. `"1:3"`), `state_type` |
| `densify_first_k` | Convert the first K MoE layers back to dense | `k` |
| `change_moe_topology` | Reshape an MoE block | `n_experts`, `top_k` |
| `change_precision_per_component` | Weight, activation, and KV precision | `weight`, `activation`, `kv` |
| `change_parallelism` | Swap TP / PP / EP / CP degrees | `tp`, `pp`, `ep`, `cp` |
| `scale_d_model` | Shift `d_model` by `delta`, aligned to `align` | `delta`, `align` |
| `scale_n_layers` | Shift `n_layers` by `delta` | `delta` |

Unknown delta names or kwarg keys fail fast with a structured error
before any evaluation runs.

#### Sequence example

```bash
# What if we run both MLA *and* add state layers on GPT-OSS-120B?
ac-delta-eval \
  --baseline-config configs/gpt_oss_120b.json \
  --hardware h100 --tp 8 --workload chat \
  --apply swap_attention_to_mla --apply-args latent_dim=256 \
  --apply add_state_layers      --apply-args ratio=1:3 \
  --out out/gpt_oss_mla_state
```

The deltas compose left-to-right; the report describes the *cumulative*
effect on the metric panel, stress vector, quality decomposition, and
Pareto position.

---

## Base-model and CLI arg config

### Input base model config format

Schema version 0.3. JSON. One `layer_configs` entry per uniform layer
band. A first-K-dense MoE config uses two entries (first K layers dense,
rest MoE). See `configs/mistral_7b.json` for the dense reference and
`configs/{gpt_oss_120b, mai_thinking_1}.json` for MoE and MoE+MLA.

#### Base model config format 

<details>

```jsonc
{
  "schema_version": "0.3",
  "metadata": {
    "model_name": "your-model",
    "source_note": "free-form provenance"
  },
  "parallelism": {
    "tensor_parallel":   8,
    "pipeline_parallel": 1,
    "data_parallel":     8,
    "expert_parallel":   8,    // MoE memory: per-rank expert count = n_experts / ep
    "context_parallel":  1,    // splits sequence axis
    "cp_method":         "ring"
  },
  "architecture": {
    "d_model": 4096,           // MUST equal n_heads × d_head; the loader rejects mismatches
    "n_layers": 32,
    "vocab_size": 32000,
    "positional_encoding": {
      "type": "rope",
      "base": 1000000,
      "scaling": {                // optional; "none" = unmodified RoPE
        "method": "yarn",         // yarn | ntk | longrope | pi | none
        "factor": 4.0,
        "original_max_position": 8192
      }
    },
    "mtp": {                      // optional Multi-Token Prediction
      "n_predict_depths": 1,
      "depth_n_layers": 1,
      "share_embeddings": true,
      "train_loss_weight": 0.3,
      "inference_mode": "drop"
    },
    "layer_configs": [
      {
        "layer_idx": [0, 1, /* … */, 31],
        "type": "transformer_block",
        "attention": {
          "type": "full",         // full | mla | nsa
          "n_heads": 32,
          "n_kv_heads": 8,
          "d_head": 128,
          "rope": true,
          "kv_cache_bits": 16,
          "precision": {"qk": "bf16", "v": "bf16", "output": "bf16"}
        },
        "ffn": {
          "type": "swiglu",       // swiglu | moe
          "ffn_dim": 14336,
          "precision": "bf16"
        },
        "normalization": {"type": "rmsnorm", "eps": 1e-5, "precision": "bf16"},
        "residual_dtype": "bf16",
        "state": null             // or {"type": "mamba2|gla|kda|...", "d_state": 64, "n_heads": 72, "d_head": 64}
      }
    ]
  }
}
```

The baseline loader threads `parallelism.expert_parallel` and
`parallelism.context_parallel` into the candidate, which is required for
MoE configs to evaluate correctly. **If you hand-write an MoE config and
forget `expert_parallel`, the throughput model will place all experts on
every rank and the quality model will return its INFEASIBLE marker.**

</details>


### Alternative using YAML or TOML for recipe configurations instead of flags:

```bash
ac-compile --recipe configs/recipes/h100_dense_7b.yaml \
  --override params=70 \
  --output-config out/arch.json
```

Key commands:

* `--recipe PATH`: Load a saved configuration.
* `--override KEY=VALUE`: Modify individual recipe values.
* `--print-recipe PATH`: Save the resolved configuration from a run.
* `ac-compile config show`: Preview the resolved config, output paths, and warnings without running a search.
* `ac-compile init TEMPLATE --out PATH`: Create a recipe from a built-in template.
* `--help-group GROUP`: Show help for one flag group, such as `serving`, `moe`, `precision`, or `recipe`.

Example templates are available in `configs/recipes/`:

```text
h100_dense_7b.yaml
b200_moe_mla_long_ctx.yaml
delta_mistral_gqa_long_ctx.yaml
```

`ac-delta-eval` also supports inline delta arguments:

```bash
ac-delta-eval --apply 'swap_attention_to_mla{latent_dim=256,heads=8}'
```

---

## Other Layers 

### Stress diagnostic layer

`ac-stress` gives you the 10-axis stress vector for any architecture: HBM bandwidth, KV footprint, tensor-core utilization, SRAM tile fit, all-reduce pressure, all-to-all pressure, training memory, and more. ac-stress transition ranks every named architectural change by binding-axis relief. The justification output names the constraint explicitly — "Selected MLA because HBM-BW-decode is binding at 0.94; MLA relieves to 0.46. Cost: +0.008 attention residual" — not just the change.


### Auto-calibration

Use `ac-auto-calibrate` to fit lab-local uncertainty and hardware-efficiency
overlays from measured runs. It accepts JSON, JSONL, or CSV rows with flexible
field names.

#### Minimal data needed for auto calibration

<details>

<summary> Minimal row </summary> 

```json
{
  "id": "h100_mistral_7b_decode",
  "hardware": "h100",
  "architecture_family": "dense_gqa",
  "model_type": "dense",
  "active_params_b": 7.2,
  "total_params_b": 7.2,
  "training_tokens": 2.0,
  "context_length": 8192,
  "predicted_loss": 2.03,
  "observed_loss": 2.08,
  "predicted_uncertainty_total_pct": 3.1,
  "eval_scores": {
    "mmlu_pro": 0.421,
    "gpqa": 0.311
  },
  "predicted_evals": {
    "mmlu_pro": 0.409,
    "gpqa": 0.298
  },
  "predicted_training_tps": 11800,
  "observed_training_tps": 10400,
  "predicted_serving_tbt_ms": 6.2,
  "observed_serving_tbt_ms": 7.1,
  "predicted_prefill_time_ms": 34.0,
  "observed_prefill_time_ms": 39.0
}
```

</details>


Fit a pack:

```bash
ac-auto-calibrate fit \
  --measurements lab_measurements.jsonl \
  --out out/lab_calibration \
  --target-coverage 0.90 \
  --min-quality-rows 12 \
  --min-eval-rows 12 \
  --min-eval-families 3 \
  --min-hardware-rows 3 \
  --max-hardware-scatter-p90-pct 15
```

Two backend switches expose the calibration surface:

- `--backend {ridge, hierarchical}` — `ridge` (default) runs the current
  fitter. `hierarchical` is a stubbed posterior backend that exits with a
  "not yet implemented" message.
- `--public-anchor-gate {on, off}` — release gate. When `on` (default),
  the fitter runs the public-model predictive-accuracy audit at
  post-calibration tolerances; any failing entry demotes the pack from
  `production_ready` to `experimental` and writes a per-model breakdown
  to `public_anchor_report.md` alongside the pack.

An editable starter file is included at
`examples/lab_measurements.example.jsonl`.

Outputs:

```
out/lab_calibration/
  calibration_pack.json      full fit summary
  quality_overrides.json     overlay for quality uncertainty calibration
  hardware_specs/*.json      copied + tuned hardware specs
  report.md                  human-readable calibration report
  public_anchor_report.md    anchor pass/fail table (when gate is on)
```

<details>
 
<summary> Use the pack without editing source files </summary>

```bash
AC_QUALITY_DEFAULTS=out/lab_calibration/quality_overrides.json \
AC_HARDWARE_SPEC_DIR=out/lab_calibration/hardware_specs \
ac-compile --hardware h100 --params 7 --tokens 2 ...
```

Quality calibration scales uncertainty intervals; it does not bias-correct the
loss point estimate. Hardware calibration adjusts `training_system_efficiency`,
`decode_system_efficiency`, and `prefill_system_efficiency` from median
observed/predicted ratios.

When rows include `eval_scores`, the fitter also writes ridge eval models with
held-out architecture-family CV. The overlay marks the pack as
`production_ready` only when the configured sample gates pass; otherwise compile
outputs carry `metadata.predicted.calibration_warnings`. Greenfield configs also
include:

```json
{
  "confidence_envelope": {
    "loss_low": 1.91,
    "loss_high": 2.11,
    "robust_to_loss_uncertainty": false,
    "contending_candidates": 7
  },
  "eval_predictions": {
    "mmlu_pro": {
      "score": 0.438,
      "uncertainty": 0.021,
      "heldout_family_rmse": 0.019
    }
  }
}
```

Keep separate packs for materially different clusters, kernels, schedulers,
recipes, and datamixes.

</details>

#### Zero-compute calibration

Two subcommands sharpen decisions without any training runs:

- **`ac-auto-calibrate fit-pairs`** — fits per-residual-term scale
  factors from a corpus of published paired ablations (Waleffe/Jamba
  hybrids, Ainslie GQA/MQA, DeepSeek MLA/MoE/MTP, Gemma-2 and Mistral
  locality, YaRN/PI) and emits a coverage audit naming every term with
  zero constraining pairs. Add lab pairs in the same format to sharpen.
  Cross-paper scales are confounded by datamix/tokenizer — treat as
  priors, not lab truth.

    ```bash
    ac-auto-calibrate fit-pairs --out out/pairfit
    ```

- **`ac-auto-calibrate plan-ladder`** — generates (never runs) the
  cheapest paired-run ladder that resolves a named architecture
  decision: scores both arms at target scale, computes the paired
  sigma, and proposes scaled-down paired runs priced by AC's own
  throughput model. Emits `plan.md`, `plan.json`, and a
  measurement-template JSONL that feeds straight back into
  `ac-auto-calibrate fit`.

    ```bash
    ac-auto-calibrate plan-ladder --arm-a dense --arm-b moe \
      --params 13 --tokens 2 --context 8192 --out out/plan
    ```

### Cost estimates (optional layer)

`--cost-usd` appends a `cost_estimate_usd` block (training_total /
serving_per_1m_tokens / annual_serving_at_load) to any emitted config —
greenfield and modifier. Pure-add: Pareto ranking and every existing
field are unchanged, and with the flag off the output is byte-identical.

```bash
ac-compile --hardware h100 --params 7 --tokens 2 ... --cost-usd --price-tier on_demand
```

Price books live in `ac/pricing_specs/*.json` (on-demand / reserved-1y /
spot, TDP, PUE) and are designed to be hot-updated — they are **list
prices** with per-file provenance and access dates, not quotes. Demo:
[`validation/e_pricing/`](validation/e_pricing/).

---

## Implementation export

Greenfield runs can also emit a standalone PyTorch module scaffold from the
selected AC schema config:

```bash
ac-compile \
  --hardware h100 --params 7 --tokens 2 --context 8192 \
  --serving-tbt 50 --serving-batch 32 --tp 8 --pp 1 --dp 8 \
  --output-config out/arch.json \
  --output-implementation out/ac_model.py \
  --implementation-class-name ACModel
```

The generated file embeds the config as `AC_CONFIG` and defines an
`nn.Module` class with dense/GQA/MLA attention, SwiGLU, MoE, RMSNorm, and
state-block adapter slots. It uses PyTorch reference paths by default, tries
`flash-attn` for attention when available, and lets labs provide their own
installed kernels:

```python
from ac_model import ACModel

model = ACModel(component_overrides={
    "attention:nsa": my_native_sparse_attention_forward,
    "state:gla": lambda d_model, config: MyFlaGlaBlock(d_model, **config),
    "state:mamba2": lambda d_model, config: MyMamba2Block(d_model, **config),
})
```

This artifact is meant as integration glue and shape-faithful reference code.
For production pretraining, replace attention/state/MoE kernels with the lab's
own FlashAttention, FLA, Mamba, expert-parallel, quantization, and checkpointing
components.

---

### Supported components

#### Hardware targets

| Target | Peak BF16 / FP8 / FP4 (TF) | HBM | Interconnect | Tile path |
|---|---|---:|---|---|
| **NVIDIA H100 SXM** | 990 / 1980 / — | 80 GB | NVLink 4 (900 GB/s) | wmma 16×16 |
| **NVIDIA B200** | 2 250 / 4 500 / 9 000 (MXFP4) | 192 GB | NVLink 5 (1.8 TB/s) | wmma + MX |
| **NVIDIA GB200 NVL72** (rack-scale, 72× B200) | 2 250 / 4 500 / 9 000 (MXFP4) | 192 GB ×72 | NVLink 5 domain = 72 GPUs (1.8 TB/s per GPU, 130 TB/s rack); IB scale-out 400 Gb/s per GPU | wmma + MX |
| **NVIDIA H800 SXM** (H100 export SKU) | 990 / 1980 / — | 80 GB | NVLink 4 reduced (400 GB/s) | wmma 16×16 |
| **TPU v5p** | 459 BF16 / — / — | 95 GB | ICI mesh | MXU 128×128 |
| **TPU v5e** | 197 BF16 / — / — | 16 GB | ICI mesh | MXU 128×128 |
| **AWS Trainium 2** | 650 / 1 300 / — | 96 GB | NeuronLink v3 (1.28 TB/s) | NCv3 128×128 |
| **AWS Trainium 3** | 1 300 / 2 600 / 5 200 (MX) | 192 GB | NeuronLink v4 (2.4 TB/s) | NCv4 + FP4 |

`gb200_nvl72` and `h800` are *system-level* targets: the per-chip numbers are
identical to B200 / H100 (same silicon); what changes is the fabric.
`gb200_nvl72` models the full rack as one NVLink domain
(`nvlink_domain_size = 72`), which is what makes rack-scale expert
parallelism (EP up to 72) priceable — see
[`validation/c_nvl72/report.md`](validation/c_nvl72/report.md) (EP=72 on the
Pareto frontier for a 1T-class 288-expert MoE on NVL72 vs mandatory spill +
a −21% training all-to-all tax on single-node H100). `h800` models the
export-restricted H100 SKU whose NVLink is capped at 400 GB/s. Rack power,
cooling, and failure rates are intentionally not modeled (see Roadmap).

The numbers above are the **vendor datasheet** dense Tensor-Core peaks.
The `peak_flops_tf` field inside `ac/hardware_specs/*.json` is an *effective*
per-precision baseline (typically ~50% of the datasheet peak for NVIDIA
targets, equal to the datasheet for TPU and Trainium) that composes with
`calibration.efficiency_multipliers` to recover measured production
throughput. The `_peak_flops_tf_convention` field at the top of each
NVIDIA spec explains this; the `notes.peak_flops_source` field cites the
datasheet. If you fork a spec, keep both fields in sync.

#### attention + cache

| Mechanism | `attention.type` | Greenfield flag | Delta name | Source |
|---|---|---|---|---|
| Full / MHA / GQA / MQA | `full` | (default; n_kv_heads sweeps) | `swap_attention_to_gqa` | — |
| **MLA** (Multi-head Latent Attention) | `mla` | `--allow-mla --mla-kv-latent --mla-q-latent` | `swap_attention_to_mla` | DeepSeek-V2/V3 |
| **NSA** (Native Sparse Attention) | `nsa` | `--nsa --nsa-{compress,select,window}-*` | — | DeepSeek 2025 |
| **CSA** (Compressed Sparse Attention) | `csa` | `--allow-csa --csa-*` | — | block-compressed KV + top-k blocks |
| **IndexShare** (bucketed lightning indexer) | `indexshare` | `--allow-indexshare --indexshare-*` | — | DSA-style shared top-k buckets |
| **MSA** (multi-scale attention) | `msa` | `--allow-msa --msa-*` | — | local + dilated + global top-k |
| **SWA** (Sliding Window Attention) | `full` + window | (via state-hybrid `--state-type sliding_window`) | `swap_attention_to_swa` | Mistral / Longformer |
| **YOCO** (You Only Cache Once) | `architecture.yoco` | `--yoco --yoco-n-self-attn-layers --yoco-share-pattern` | — | Sun et al. 2024 (Microsoft) |

#### FFN families

| Family | `ffn.type` | Greenfield flag | Delta name |
|---|---|---|---|
| **SwiGLU dense** | `swiglu` | default | — |
| **MoE** (top-k softmax router, optional shared expert, capacity factor) | `moe` | `--allow-moe --moe-n-experts --moe-top-k --ep-options` | `change_moe_topology` |
| **First-K-dense MoE prefix** (DeepSeek-V3 / Qwen3-MoE) | `moe` + 2 layer_configs | `--dense-ffn-layers` | `densify_first_k` |

#### State / hybrid families

Hybrid layers replace a fraction of attention with a state mixer; the
family controls which residual-quality term fires.

| Family | `--state-type` aliases | Residual family | Source |
|---|---|---|---|
| **Mamba-2** / Mamba / S4 / S5 / S6 | `mamba2`, `mamba`, `s4`, `s5`, `s6` | `mamba_sequential` | Gu & Dao 2024 |
| **GLA** / **KDA** / DeltaNet / Gated DeltaNet | `gla`, `kda`, `deltanet`, `gated_deltanet` | `gated_delta_or_kda_linear` | Yang 2024 / Kimi 2024 |
| Parallel-heads (MoH / Hydra) | `parallel_heads`, `moh`, `hydra` | `parallel_hybrid_heads` | Jin 2024 |
| Sliding-window / local recurrent | `swa`, `sliding_window`, `local_recurrent` | `recurrent_local_attention` | Beltagy 2020 |

Placement: `--placement-strategy first_periodic_last,interleaved,periodic`.
State sizing (`d_state`) is SRAM-derived per hardware target.
The CLI accepts the alias spellings above but emits canonical schema names
where needed, e.g. `swa` / `local_recurrent` become `sliding_window`,
`delta_net` becomes `deltanet`, and `gated_delta` becomes `gated_deltanet`.

#### Parallelism axes

| Axis | Schema field | Greenfield flag |
|---|---|---|
| Tensor (TP) | `parallelism.tensor_parallel` | `--tp` |
| Pipeline (PP) | `parallelism.pipeline_parallel` | `--pp` |
| Data (DP) | `parallelism.data_parallel` | `--dp` |
| Fixed training cluster | derived candidate-specific DP | `--training-cluster-gpus` |
| **Expert (EP)** | `parallelism.expert_parallel` | `--ep-options` |
| **Context (CP)** — Ring / Ulysses | `parallelism.context_parallel`, `cp_method` | `--cp --cp-method --cp-options` |

#### Positional encoding

| Method | `positional_encoding.scaling.method` | Multiplier on long-ctx degradation | Source |
|---|---|---:|---|
| None | `none` | 1.00 | baseline |
| **PI** (Position Interpolation) | `pi` | 0.85 | Chen 2023 |
| **NTK**-aware | `ntk` | 0.65 | NousResearch 2023 |
| **YaRN** | `yarn` | 0.45 | Peng 2024 |
| **LongRoPE** | `longrope` | 0.40 | Ding 2024 |

Enabled via `--allow-rope-scaling --rope-original-max-position N
--rope-scaling-methods …`. Beyond the trained extension range the
multiplier snaps to 1.0.

#### Precision

| Format | Weights / FFN | KV cache | Hardware (peak path) |
|---|:---:|:---:|---|
| BF16 / FP16 | ✓ | ✓ (16-bit) | all |
| **FP8** (E4M3 / E5M2) | ✓ | ✓ (8-bit) | H100, B200, Trn2, Trn3 |
| INT8 | — | ✓ (8-bit) | all (KV only) |
| **FP4** (E2M1) | ✓ | ✓ (4-bit) | B200, Trn3 |
| INT4 | — | ✓ (4-bit) | all (KV only) |
| **MXFP4** (OCP microscaling) | ✓ | — | B200, Trn3 |
| **MXFP6** | ✓ | — | B200, Trn3 |

Greenfield: `--precision-modes bf16,fp8_ffn,fp8,fp4,mxfp4,mxfp6` and
`--kv-dtypes bf16,fp8,int8,fp4,int4`. Hardware-specific filtering applies:
FP4/MX modes are available on B200 and Trainium 3.

### Other architectural primitives

| Primitive | Schema location | Greenfield flag | Source |
|---|---|---|---|
| **MTP** (Multi-Token Prediction) | `architecture.mtp` | `--allow-mtp --mtp-depths` | DeepSeek-V3 §2.2 |
| **2:4 structured sparsity** | `sparsity_2_4` per component | (post-search; quality-model only) | NVIDIA H100/B200 |
| **RMSNorm** | `normalization.type = rmsnorm` | default | Zhang & Sennrich 2019 |

#### Delta REGISTRY

| Name | Effect | Legal `--apply-args` |
|---|---|---|
| `swap_attention_to_gqa` | n_kv_heads ← n_heads / group_size | `group_size` |
| `swap_attention_to_mla` | full → MLA at `latent_dim` | `latent_dim` |
| `swap_attention_to_swa` | full → sliding window | `window_size` |
| `interleave_local_attention` | local:global interleave | `ratio`, `window` |
| `add_state_layers` | replace fraction of attention with a state mixer | `ratio`, `state_type` |
| `densify_first_k` | first K MoE layers → dense | `k` |
| `change_moe_topology` | reshape an MoE block | `n_experts`, `top_k` |
| `change_precision_per_component` | weight, activation, and KV precision | `weight`, `activation`, `kv` |
| `change_parallelism` | swap TP / PP / EP / CP | `tp`, `pp`, `ep`, `cp` |
| `scale_d_model` | shift `d_model`, aligned to `align` | `delta`, `align` |
| `scale_n_layers` | shift `n_layers` | `delta` |

##### given reference architectures 

```
Llama-2-{7B, 13B, 70B}   Llama-3-{8B, 70B}   Mistral-7B   Gemma-2-9B
Qwen3-{8B, 32B}   DeepSeek-V3   Kimi-K2.5   GLM-5.1
GPT-OSS-120B   MAI-Base-1
```


---

## Roadmap to boundaries

Future plans to our current boundaries with needs.

- **Datamix dimension to quality spline** Cross-corpus absolute-loss
  error is 10–22% with intervals too narrow to cover it (E2: 0/3 CI
  coverage — published, not hidden). Use AC ordinally, or fit
  per-datamix calibration packs (`ac-auto-calibrate fit`) as designed.
- **Training-efficiency bucket runs ~10 pp low** against reported
  frontier-lab MFU (E1 T2 anchors). Same fix: lab-local calibration.
- **Optimizer-agnostic**: assumes AdamW-family scaling; Muon-class
  optimizers shift the exponents and are not modeled.
- **Hardware coverage**: no TPU v6/v7, AMD MI300/MI355, Ascend 910B/C.
- **EP granularity**: EP must divide `n_experts`.
- Data recipes, tokenizers, and post-training effects are out of scope
  for the current stage (stated boundaries, not oversights).

---

## Repository layout

```
.
├── README.md
├── pyproject.toml
├── ac/                              ← the Python package
│   ├── __init__.py
│   ├── cli_compile.py               greenfield + modifier point
│   ├── cli_delta_eval.py            delta influence entry point
│   ├── cli_stress.py                stress / quality / transition inspection
│   ├── cli_recipe.py                --recipe / --override / --print-recipe /
│   │                                --help-group / `config show` / `init`
│   ├── cli_matrix18b.py             budget-matched matrix + scenario Pareto
│   ├── cli_trust_audit.py           public-model anchors + trust audit
│   │
│   ├── lattice_engine.py            tile-aligned architecture lattice + KNOWN_ARCHITECTURES
│   ├── throughput_model.py          roofline throughput + MoE all-to-all + state-hybrid + MLA
│   ├── quality_model.py             modular scaling-law backbone + residual hooks
│   ├── auto_calibrate.py            local calibration pack fitter
│   ├── penalties.py                 quality-side penalty primitives
│   ├── sram_derivation.py           SRAM-derived state-block sizing
│   ├── schema.py                    schema 0.3 emit/validate
│   │
│   ├── optimizer.py                 candidate enumeration + Pareto search
│   ├── baseline.py                  base-config ingestion
│   ├── modifier.py                  baseline-aware local Pareto search
│   ├── baseline_delta.py            modifier report generation
│   ├── justification.py             prose justification + model card + assumptions
│   ├── shadow_prices.py             dual-variable interpretation
│   │
│   ├── stress.py                    10-axis StressVector
│   ├── quality_stress.py            7-axis QualityStressVector
│   ├── delta_engine.py              named transformation engine
│   ├── transition.py                pre/post stress diff
│   ├── rank.py                      transition ranking
│   ├── justify_transition.py        transition justifier
│   ├── optimizer_bridge.py          glue: CandidateArch ↔ ArchConfig
│   │
│   ├── evaluator.py                 capability-3 evaluator
│   ├── pareto_position.py           6-class Pareto verdict
│   ├── report.py                    delta-eval Markdown / JSON / CSV renderer
│   │
│   ├── deltas/                      10 named transformations
│   │   ├── base.py
│   │   ├── swap_attention_to_{gqa,mla,swa}.py
│   │   ├── add_state_layers.py
│   │   ├── densify_first_k.py
│   │   ├── change_moe_topology.py
│   │   ├── change_parallelism.py
│   │   ├── change_precision_per_component.py
│   │   ├── scale_{d_model,n_layers}.py
│   │   └── __init__.py              exports REGISTRY
│   │
│   ├── hardware_specs/              h100, b200, tpu_v5p, tpu_v5e, trainium2, trainium3
│   ├── calibration/                 h100, b200, tpu_v5p calibration jsons
│   └── quality_defaults.yaml        modular scaling-law constants
│
└── configs/                         reference base-model configs + shipped recipes
    ├── mistral_7b.json              dense + GQA
    ├── gpt_oss_120b.json            MoE 128 × top-4
    ├── mai_thinking_1.json          MoE + MLA + MTP + LongRoPE
    └── recipes/                     shipped --recipe bundles
        ├── h100_dense_7b.yaml
        ├── b200_moe_mla_long_ctx.yaml
        └── delta_mistral_gqa_long_ctx.yaml
```

---

## License

Apache-2.0.
