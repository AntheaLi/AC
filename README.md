# 算模 AC — Architecture Compiler - v0

![算模AC - ArchCalc](assets/image.png)

AC is a hardware-aware architecture compiler that turns a compute budget, hardware target, and optional base model into multi-objective Pareto-front model architectures and architecture deltas. It serves as a quantitative **pre-flight** check, it does NOT replace training. 

> AC is a compiler for model architecture design under real hardware constraints. Given a target hardware platform, parameter budget, training tokens, serving workload, or an existing baseline architecture, AC searches for Pareto-improving architectures and local modifiers across choices like width/depth, attention layout, GQA/KV configuration, precision policy, MoE structure, and hybrid attention/state ratios. It supports greenfield architecture search, baseline-aware local modification, and delta influence evaluation, making it useful both for designing new models and for understanding whether a proposed architecture change actually improves the quality–latency–memory tradeoff.

check out demo here: [ac-demo](https://antheali.github.io/ac-demo/)

Three composable capabilities, one shared config format:

- **Greenfield**  Given compute -> architecture | `ac-compile --hardware H --params N --tokens T …` 

- **Modifier**  Given compute + a base architecture -> modifier? `ac-compile --baseline-config CONF …` 

- **Delta influence**  Given compute + base + delta -> influence. `ac-delta-eval --baseline-config CONF …`

Also check out the [AC-Harness](https://github.com/AntheaLi/AC-harness), which is a loop scaffold 
built to automate the process. It can also exist as a thin layer that sits beside existing training, eval, and benchmarking stack. 

---

## What's new

This release lands a daily-use ergonomics pass on top of the v0.3 core, plus
several correctness fixes that came out of an internal review.

### Bug fixes

- **`rank=1` now always agrees with `selected=True`.** `pareto.csv` rows are
  sorted with the same uncertainty-aware tiebreak the optimum picker uses
  (`build_display_sort_key` in `optimizer.py` — for `research_quality` /
  `loss_only` profiles: `loss_bucket → memory_gb → serving_tbt_ms → -training_tps
  → adj_loss → total_params`; other profiles add an `objective_score` bucket
  ahead of the lexicographic tail), so the row at the top of the CSV is the
  row the picker actually chose. Previously the CSV emitted rows in
  Pareto-discovery order, which routinely put a high-uncertainty lower-loss
  candidate at `rank=1` while `selected=True` appeared 10+ rows down.
- **Output-path basename inheritance is consistent.** When only
  `--output-config out/foo.json` is passed, the sibling outputs now inherit
  both the directory **and** the basename: `out/foo.md`,
  `out/foo_pareto.csv`, `out/foo_shadow_prices.json`. When `--output-config`
  is left at its default (`arch.json`), the historical short names
  (`pareto.csv`, `shadow_prices.json`) are still used so existing globbers
  keep working.
- **`Optimal:` log line is arch-aware.** On MoE+MLA runs the line previously
  printed `kv=<n_kv_heads>` (meaningless under MLA) and never mentioned MoE
  topology or MTP. The new format names what was actually picked, e.g.:

  ```
  [arch-compiler] Optimal: d=6144 L=89 attn=mla(kv_latent=512,q_latent=1536)
                          ffn=16384 moe=128xtop8(ep=8) mtp=1 prec=fp8 kv_bits=16
  ```

  Dense runs collapse to the head count and KV head count:

  ```
  [arch-compiler] Optimal: d=4608 L=32 attn=full h=72 kv=8 ffn=12288 prec=fp8 kv_bits=8
  ```

- **`training_mem` axis no longer prints `binding (117% — over) [inactive
  for decode]`.** A phase-inactive axis (e.g. `training_mem` during decode)
  now reports as `inactive (binding 117% if active)`. The contradiction
  came from `binding_axes` including out-of-phase axes; that's now scoped
  to the in-phase set.
- **`contending_family.members` capped at 5 in emitted JSON.** Previously
  the inline list could carry 30+ candidate dicts, bloating the config for
  downstream tooling. The top 5 stay inline; when the envelope is
  non-robust, the full top-32 view is written to
  `<config-stem>_contending_family.json` next to the config. Override the
  inline cap with `AC_CONTENDING_FAMILY_INLINE=N` for forensics (validated
  end-to-end — wiring the env var was a real fix in this release).
- **README banner image typo fixed** (`image.1png` → `image.png`).

### New ergonomics features

- **`--recipe PATH` + `--override KEY=VALUE`.** Bundle the 8–14 flags users
  keep retyping into a checked-in `.yaml` / `.toml` recipe; replay with
  one line and tweak with overrides on top. See [Recipes](#recipes) below.
- **`--print-recipe PATH`.** Snapshot the resolved flag set of a successful
  run as a yaml recipe you can commit and replay. Closes the "I ran
  something last week and forgot which flags" loop. List-valued flags
  (`kv_dtypes`, `precision_modes`) are written back in their CLI-input
  alias form (`bf16,int8`, not `[16, 8]`) so the snapshot round-trips
  cleanly through `--recipe`. Output paths are intentionally omitted from
  the snapshot — supply them per-run on replay so the recipe stays portable
  across machines and run directories.
- **`--help-group GROUP`.** Filtered help. `ac-compile --help-group serving`
  prints just the serving-budget flags instead of the ~70-flag flat list.
  Groups: `workload`, `selection`, `serving`, `parallelism`, `precision`,
  `outputs`, `moe`, `mla`, `rope`, `stamps`, `search`, `recipe`,
  `architecture`.
- **`ac-compile config show`.** Resolve a recipe + flag set without
  searching: prints the resolved namespace, an `inferred_output_paths` block
  listing the sibling files the run will write (`<stem>.md`,
  `<stem>_pareto.csv`, `<stem>_shadow_prices.json`, and the contending-family
  sidecar if the envelope ends up non-robust), and any sleeping warnings
  (`--tp` and `--tp-options` both set, `--objective-profile latency` /
  `serving_cost` without `--serving-tbt` / `--serving-ttft`, hardware not
  calibrated, missing `--params` and `--baseline-config`, etc.). Saves the
  12s greenfield round-trip when an axis is misconfigured.
- **Deprecated aliases hidden from `--help`.** `--batch-size`,
  `--tbt-p95-ms`, `--ttft-p95-ms`, `--param-band`, and the
  `trn2` / `trn3` hardware short forms still parse, but they no longer
  clutter `--help` output. The README's flag tables only show the canonical
  name for each option.
- **Grouped `--apply` syntax for `ac-delta-eval`.** `--apply
  swap_attention_to_mla:latent_dim=256` or
  `--apply 'swap_attention_to_mla{latent_dim=256,heads=8}'` replace the
  two-flag `--apply NAME --apply-args k=v` form. The legacy form still
  parses (hidden from help) so existing scripts keep working, but the
  "args belong to the closest preceding --apply" positional rule is no
  longer the only way to attach kwargs to a delta.

---

## Recipes

Recipes bundle 8–14 flags into a single checked-in artifact so a one-line
invocation is enough to reproduce (or evolve) a run. The recipe format is
flat key/value YAML or TOML where each key is an argparse `dest` name with
underscores (`tp`, `serving_batch`, `allow_moe`, `moe_n_experts`, …).
List values become comma-joined strings; booleans become flag
presence/absence; `null` falls back to the argparse default.

Shipped under `configs/recipes/`:

```
configs/recipes/
  h100_dense_7b.yaml             greenfield 7B dense on H100
  b200_moe_mla_long_ctx.yaml     MoE + MLA + LongRoPE on B200 at 65K context
  delta_mistral_gqa_long_ctx.yaml  modifier-mode GQA + long context on Mistral
```

Drive a run from a recipe with one line:

```bash
ac-compile --recipe configs/recipes/h100_dense_7b.yaml \
  --output-config out/h100_7b/arch.json
```

Scale or change one knob without copying the file:

```bash
ac-compile --recipe configs/recipes/b200_moe_mla_long_ctx.yaml \
  --override params=70 --override context=131072 \
  --output-config out/b200_70b/arch.json
```

Snapshot a successful ad-hoc invocation for later replay:

```bash
ac-compile --hardware h100 --params 7 --tokens 2 \
  --serving-tbt 50 --serving-batch 32 --tp 8 \
  --print-recipe out/last_run.yaml
# later
ac-compile --recipe out/last_run.yaml --output-config out/replay/arch.json
```

Inspect what a recipe + override stack resolves to before searching:

```bash
ac-compile config show \
  --recipe configs/recipes/b200_moe_mla_long_ctx.yaml \
  --override params=70 \
  --output-config out/b200_70b/arch.json
```

`config show` prints the full resolved namespace, the inferred output
paths (highlighting any inherited from `--output-config`), and a list of
warnings — e.g. `--tp=4` not in `--tp-options=2,8`, an uncalibrated
hardware target, or `--objective-profile=latency` without a TBT budget.
It does not run the search, so it costs no minutes.

For filtered help on one section of the CLI, use `--help-group`:

```bash
ac-compile --help-group moe       # just the MoE sweep flags
ac-compile --help-group precision # just the precision/KV flags
ac-compile --help-group recipe    # --recipe / --override / --print-recipe / --help-group
```

To bootstrap a checked-in recipe from a known-good starting point, use
`ac-compile init`:

```bash
ac-compile init                                # list available templates
ac-compile init h100_dense_7b --out recipe.yaml
ac-compile init b200_moe_mla_long_ctx --out recipes/b200.yaml
ac-compile init delta_mistral_gqa_long_ctx --out recipes/delta.yaml
```

The init templates mirror the shipped `configs/recipes/*.yaml` bundles and
are intended as the "I want to vary three knobs on top of an existing
working setup" starting point.

---

## Read this first: rank, don't predict

AC is a **forward proxy**, not a measurement system. Internally it is:

1. a roofline + tile-efficiency throughput model with one
   `*_system_efficiency` scalar per phase per hardware target, and
2. a Hoffmann-style scaling-law spine
   (`L = E + A/N^α + B/D^β`) plus a stack of additive residuals for
   architecture shape, precision, MoE, state/hybrid, risk, and data
   quality.

The default coefficients are documented (and labelled) as priors in
`ac/quality_defaults.yaml`. The shape-law refit ships with a small dense
fit set; the MoE / state residuals are calibrated to public ablations,
not your lab's traces. **As shipped, the absolute loss numbers (and the
TPS/TBT predictions) will be biased relative to what your stack
actually produces.**

What AC *is* good for, without calibration:

- **Pareto ranking.** Given identical priors, two candidates'
  *relative* ordering on the (loss, TBT, memory, TPS) frontier is much
  more robust than either's absolute number. Use the Pareto CSV.
- **Binding-axis identification.** The 10-axis stress vector and shadow
  prices tell you which constraint would actually move quality if
  loosened. That's a structural answer; it doesn't depend on a tight
  loss calibration.

What AC needs **before you trust the absolute numerals**:

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
better than 13B at 1T at our budget?"), assume an uncalibrated AC will
mislead you. For comparative decisions ("does adding MLA relieve the
binding axis without spending >1% loss on a frontier already at TP=8?"),
AC's structural answer is the value it adds.

---

## Install

```bash
# editable install from the repo
pip install -e .

# or run directly without installing
python ac/cli_compile.py --help
python ac/cli_delta_eval.py --help
python ac/cli_stress.py --help
python ac/auto_calibrate.py --help
```

After install the console scripts are on your `PATH`:

```bash
ac-compile        --help
ac-delta-eval     --help
ac-stress         --help
ac-auto-calibrate --help
```

No external runtime dependencies beyond Python ≥ 3.10 and PyYAML.

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

### Greenfield

Search the full architecture lattice for the Pareto-front winner under the
given hardware and constraints. No baseline config required.

```
ac-compile [OPTIONS]

required
  --hardware {h100, b200, tpu_v5p, tpu_v5e, trainium2, trainium3}
                     (`trn2` / `trn3` still parse as aliases for
                     `trainium2` / `trainium3` but are no longer
                     advertised in --help)
  --params  N        (billions; supports "7" or "7B")
  --tokens  N        (trillions; supports "2" or "2T")

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
  --cp / --cp-method   context parallel: ring | ulysses
  --cp-options         comma-list of CP degrees to sweep

architecture sweeps
  --allow-state            enable state/hybrid candidates
  --state-type             mamba2 | mamba | gla | kda | gated_deltanet
                           | deltanet | rwkv7 | retnet | swa
                           | sliding_window | linear_attention
  --placement-strategy     first_periodic_last, interleaved, periodic
  --state-precision        bf16 | fp16 | fp32

  --allow-moe              enable MoE FFN candidates
  --max-total-params-b     MoE memory ceiling (total params billions)
  --moe-n-experts          comma-list (e.g. "64,128,256")
  --moe-top-k              comma-list
  --dense-ffn-layers       comma-list of first-K-dense layer counts
  --ep-options             comma-list of expert-parallel degrees

  --allow-mla              enable Multi-head Latent Attention candidates
  --mla-kv-latent          comma-list of c_kv options (default 512)
  --mla-q-latent           comma-list of c_q  options (default 1536)

  --allow-mtp              enable Multi-Token Prediction
  --mtp-depths             comma-list (e.g. "0,1,2")

  --allow-rope-scaling             enable per-method RoPE sweep
  --rope-original-max-position     pretrain context (default 8192)
  --rope-scaling-methods           comma-list: yarn,ntk,longrope,pi,none

architecture stamps (post-search emission; optimizer does not sweep)
  --nsa                            emit Native Sparse Attention block
    --nsa-compress-block-size      (default 64)
    --nsa-compress-block-stride    (default 16)
    --nsa-select-block-size        (default 64)
    --nsa-select-top-k             (default 16)
    --nsa-window-size              (default 512)

  --yoco                           emit YOCO sharing block
    --yoco-n-self-attn-layers      (default 1)
    --yoco-share-pattern           single_source | block_shared

precision search
  --precision-modes        comma-list: bf16, fp8_ffn, fp8, fp4, mxfp4, mxfp6
  --kv-dtypes              comma-list: bf16, int8, fp8, int4, fp4

outputs (paths)
  --output-config           default arch.json
  --output-justification    default arch.md       (basename inherits from
                                                   --output-config when it
                                                   is overridden)
  --output-pareto           default pareto.csv    (inherits as
                                                   <stem>_pareto.csv)
  --output-shadow-prices    default shadow_prices.json (inherits as
                                                   <stem>_shadow_prices.json)
  --output-assumptions      default not written
  --output-model-card       default not written
  --output-implementation   generated PyTorch architecture scaffold
  --implementation-class-name  class name for --output-implementation
  --no-shadow-prices        skip the shadow-price pass (faster)
  --max-candidates          optional greenfield cap after candidate dedupe
  --progress-every          print evaluation progress every N candidates
  --quiet                   suppress progress logs

recipes (see Recipes section)
  --recipe         PATH     load flags from a .yaml / .toml recipe
  --override       KEY=VAL  override one recipe key (repeatable)
  --print-recipe   PATH     snapshot resolved flags to PATH after run
  --help-group     NAME     print help for one group and exit
                            (workload | selection | serving | parallelism |
                            precision | outputs | moe | mla | rope | stamps |
                            search | recipe | architecture)
```

When the envelope around the picked candidate is **not robust** (i.e.
several other candidates' loss CI bands overlap the optimum's), a
`<output-config-stem>_contending_family.json` sidecar is written next to
the config carrying up to 32 contender rows. The inline
`metadata.predicted.confidence_envelope.contending_family.members` list
is capped at 5 rows so the emitted config stays small; override with
`AC_CONTENDING_FAMILY_INLINE=N` for forensics runs.

The accompanying stderr `WARNING:` line varies based on whether a lab
calibration pack is loaded:

- **No pack loaded** (no `AC_QUALITY_DEFAULTS`, or the path doesn't
  resolve): "rank-1 is not robust to **uncalibrated** quality-model
  uncertainty". This is the expected state out of the box and the right
  cue to run `ac-auto-calibrate fit` if you care about absolute loss.
- **Pack loaded**: "rank-1 is not robust to **the calibrated**
  quality-model uncertainty band". The envelope is wide because the
  lab-fit uncertainty is genuinely wide for this run, not because
  calibration is missing — re-fit on more rows in the relevant
  architecture family if you want a tighter band.

#### pareto.csv columns

One row per Pareto-frontier candidate, sorted by the same uncertainty-aware
tiebreak the picker uses (`objective_score → predicted_loss →
prefill_time_ms → serving_tbt_ms → -training_tps`), so `rank=1` always
agrees with the row carrying `selected=True`. This was a real bug in
earlier releases — the rank column was emitted in the optimizer's internal
Pareto-discovery order, which routinely disagreed with the picker. Loss
columns:

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

This is a wide MoE+MLA+MTP sweep on a long context window; the full
unconstrained search can take several minutes. For a first-look run we
cap the candidate budget with `--max-candidates 200`; drop the cap for
a thorough sweep.

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
  --cp 4 --max-total-params-b 800 --max-candidates 200 \
  --output-config out/mai_arch.json --no-shadow-prices
```

---

### Modifier

Holds the architecture *family* fixed (uses the baseline as anchor) and
searches the local Pareto-frontier of modifications around it.

```
ac-compile --baseline-config PATH [OPTIONS]

required
  --baseline-config  PATH    JSON config emitted by greenfield or any existing model stripped in the format 

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

The modifier and greenfield share all other flags (precision, parallelism,
state, MoE, MLA, MTP, CP, RoPE, NSA, YOCO).

---

### Delta influence

Quantitative effect of one (or a chain of) named transformations against a
specific baseline architecture. Outputs a one-page Markdown report + JSON
+ 3-row Pareto CSV.

```
ac-delta-eval --baseline-config PATH --apply NAME [OPTIONS]

required
  --baseline-config  PATH    JSON config (greenfield output OR hand-written)
  --apply            NAME[:k=v,k=v]
                             one of REGISTRY (repeatable). Inline kwargs
                             after a colon (or inside braces) replace the
                             legacy --apply-args two-flag form, which is
                             still accepted but hidden from --help.
                             Examples:
                               --apply swap_attention_to_mla:latent_dim=256
                               --apply 'swap_attention_to_mla{latent_dim=256,heads=8}'

baseline / hw
  --hardware    h100 | b200 | tpu_v5p | tpu_v5e | trainium2 | trainium3
                             (`trn2` / `trn3` accepted as aliases)
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

Available delta names (REGISTRY):

| Name | Effect | Legal kwargs (inline `--apply NAME:k=v` or `--apply-args k=v`) |
|---|---|---|
| `swap_attention_to_gqa` | Set `n_kv_heads = n_heads / group_size` | `group_size` |
| `swap_attention_to_mla` | Replace full attention with MLA at `latent_dim` | `latent_dim` |
| `swap_attention_to_swa` | Sliding-window attention at `window_size` | `window_size` |
| `add_state_layers` | Replace a fraction of attention with a state mixer | `ratio` (e.g. `"1:3"`), `state_type` |
| `densify_first_k` | Convert the first K MoE layers back to dense | `k` |
| `change_moe_topology` | Reshape an MoE block | `n_experts`, `top_k` |
| `change_precision_per_component` | Per-component weight / KV precision | `weight`, `kv` |
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
  --apply swap_attention_to_mla:latent_dim=256 \
  --apply add_state_layers:ratio=1:3 \
  --out out/gpt_oss_mla_state
```

The legacy two-flag form `--apply NAME --apply-args k=v` still parses for
back-compat (hidden from `--help`). The inline `NAME:k=v,k=v` and
`NAME{k=v,k=v}` syntaxes attach kwargs to a specific delta without relying
on the "closest preceding --apply" positional binding rule.

The deltas compose left-to-right; the report describes the *cumulative*
effect on the metric panel, stress vector, quality decomposition, and
Pareto position.

---

### Base-model config format

Schema version 0.3. JSON. One `layer_configs` entry per uniform layer
band. A first-K-dense MoE config uses two entries (first K layers dense,
rest MoE). See `configs/mistral_7b.json` for the dense reference and
`configs/{gpt_oss_120b, mai_thinking_1}.json` for MoE and MoE+MLA.

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
    "d_model": 4096,           // MUST equal n_heads × d_head (see Caveats)
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



---

### Stress diagnostic layer

`ac-stress` gives you the 10-axis stress vector for any architecture: HBM bandwidth, KV footprint, tensor-core utilization, SRAM tile fit, all-reduce pressure, all-to-all pressure, training memory, and more. ac-stress transition ranks every named architectural change by binding-axis relief. The justification output names the constraint explicitly — "Selected MLA because HBM-BW-decode is binding at 0.94; MLA relieves to 0.46. Cost: +0.008 attention residual" — not just the change.

---

### Auto-calibration

Use `ac-auto-calibrate` to fit lab-local uncertainty and hardware-efficiency
overlays from measured runs. It accepts JSON, JSONL, or CSV rows with flexible
field names.

Minimal row:

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

An editable starter file is included at
`examples/lab_measurements.example.jsonl`.

Outputs:

```
out/lab_calibration/
  calibration_pack.json      full fit summary
  quality_overrides.json     overlay for quality uncertainty calibration
  hardware_specs/*.json      copied + tuned hardware specs
  report.md                  human-readable calibration report
```

Use the pack without editing source files:

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

The table below lists the **effective per-precision compute baseline** AC's
roofline model uses (BF16 / FP8 / FP4 TFLOP/s), not the vendor's marketed
dense-Tensor-Core peak. The effective values are deliberately set so
that, after AC composes its `efficiency_multipliers` and the per-family
table in `throughput_model._DEFAULT_EFFICIENCY_TABLE`, end-to-end TPS /
TBT predictions track public traces (NeMo / Megatron-LM / vLLM on H100;
Blackwell early-access on B200). Vendor datasheets quote ~2× these
numbers; the divergence is intentional and is documented in each
`ac/hardware_specs/*.json` under `_peak_flops_tf_convention`. Override
with `ac-auto-calibrate` when you have lab traces.

| Target | Effective BF16 / FP8 / FP4 (TF) | Calibration table | HBM | Interconnect | Tile path |
|---|---|---|---:|---|---|
| **NVIDIA H100 SXM** | 495 / 990 / — | shipped | 80 GB | NVLink 4 (900 GB/s) | wmma 16×16 |
| **NVIDIA B200** | 1 125 / 2 250 / 4 500 (MXFP4) | shipped | 192 GB | NVLink 5 (1.8 TB/s) | wmma + MX |
| **TPU v5p** | 459 / — / — | shipped | 95 GB | ICI mesh | MXU 128×128 |
| **TPU v5e** | 197 / — / — | **not shipped** | 16 GB | ICI mesh | MXU 128×128 |
| **AWS Trainium 2** | 667 / 1 334 / — | **not shipped** | 96 GB | NeuronLink v3 (1.28 TB/s) | NCv3 128×128 |
| **AWS Trainium 3** | 1 300 / 2 600 / 5 200 (MX) | **not shipped** | 192 GB | NeuronLink v4 (2.4 TB/s) | NCv4 + FP4 |

Hardware targets marked **"not shipped"** above run with AC's default
efficiency multipliers; the CLI emits a one-shot `WARNING` at startup
when no calibration table is found. Use `ac-auto-calibrate fit
--measurements <traces>.jsonl` to fit lab-local efficiency, or treat
absolute TPS / TBT / loss numbers as uncalibrated priors and use AC for
ranking only.

#### attention + cache

| Mechanism | `attention.type` | Greenfield flag | Delta name | Source |
|---|---|---|---|---|
| Full / MHA / GQA / MQA | `full` | (default; n_kv_heads sweeps) | `swap_attention_to_gqa` | — |
| **MLA** (Multi-head Latent Attention) | `mla` | `--allow-mla --mla-kv-latent --mla-q-latent` | `swap_attention_to_mla` | DeepSeek-V2/V3 |
| **NSA** (Native Sparse Attention) | `nsa` | `--nsa --nsa-{compress,select,window}-*` | — | DeepSeek 2025 |
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

#### Parallelism axes

| Axis | Schema field | Greenfield flag |
|---|---|---|
| Tensor (TP) | `parallelism.tensor_parallel` | `--tp` |
| Pipeline (PP) | `parallelism.pipeline_parallel` | `--pp` |
| Data (DP) | `parallelism.data_parallel` | `--dp` |
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
| `add_state_layers` | replace fraction of attention with a state mixer | `ratio`, `state_type` |
| `densify_first_k` | first K MoE layers → dense | `k` |
| `change_moe_topology` | reshape an MoE block | `n_experts`, `top_k` |
| `change_precision_per_component` | per-component weight / KV precision | `weight`, `kv` |
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
└── configs/                         reference base-model configs +
    │                                shipped recipes
    ├── mistral_7b.json              dense + GQA
    ├── gpt_oss_120b.json            MoE 128 × top-4
    ├── mai_thinking_1.json          MoE + MLA + MTP + LongRoPE
    └── recipes/                     shipped --recipe bundles
        ├── h100_dense_7b.yaml             greenfield 7B dense on H100
        ├── b200_moe_mla_long_ctx.yaml     MoE + MLA + LongRoPE on B200
        └── delta_mistral_gqa_long_ctx.yaml  modifier-mode GQA + long ctx
```

---

## License

Apache-2.0.
