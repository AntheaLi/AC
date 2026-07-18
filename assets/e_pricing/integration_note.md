# Task E integration note — wiring `ac.pricing` into `ac-compile`

> ac_version 0.4.0 | quality_model_version effective_capacity_v2 |
> git_commit c170cda | experiment_date 2026-07-17 | agent wave gate2-wave1
>
> Audience: the Gate-2 assembly orchestrator. Task E is pure-add; this note
> is the ONLY wiring needed. No existing numeric field is changed; with the
> flag off (default), compile output is byte-identical to today (golden
> snapshot safe).

## 1. What exists now

- `ac/pricing.py` — public API:
  `attach_cost_block(result_dict, hardware_target, workload_dict=None, *, price_tier="on_demand") -> dict`.
  Returns a **deep copy** of the emitted config dict with
  `metadata.predicted.cost_estimate_usd` added (top-level fallback for
  non-config dicts). Never mutates input. Missing/broken spec or
  no-usable-price → `warnings.warn(...)` + block omitted (never raises).
- `ac/pricing_specs/{h100,b200,tpu_v5p,tpu_v5e,trainium2,trainium3}.json`
  — hot-updatable price books (`schema_version: pricing_v1`,
  on-demand / reserved_1y / spot $/accelerator-hour, TDP, utilization,
  PUE, provenance). Dir override: env `AC_PRICING_SPEC_DIR`.
- `tests/test_pricing.py` — 12 tests, all green (full suite 808 passed,
  1 skipped).
- `validation/e_pricing/run_demo.py` — end-to-end reference for the exact
  post-processing the CLI should do inline.

## 2. Where to call it in `ac/cli_compile.py`

### 2a. Import (top of file, in the existing try/except import block, ~lines 43–56)

```python
from .pricing import attach_cost_block          # in the `try:` branch
# ...
from pricing import attach_cost_block           # in the `except ImportError:` branch
```

### 2b. Greenfield path — `main()`, at current lines 2141–2143

Current code:

```python
    # 1. JSON config
    config = result_to_config(result)
    ensure_parent_dir(args.output_config)
    save_config(config, args.output_config)
```

Insert between `result_to_config` and `ensure_parent_dir`:

```python
    if getattr(args, "cost_usd", False) and config is not None:
        config = attach_cost_block(
            config,
            args.hardware,
            {
                "training_tokens": constraints.training_tokens,
                "prompt_len": constraints.prompt_len,
                "output_len": constraints.output_len,
                "serving_batch": constraints.serving_batch,
            },
            price_tier=getattr(args, "price_tier", "on_demand"),
        )
```

Do it **before** the first `save_config` so the later re-save that appends
CLI warnings (current line ~2231) keeps the block.

### 2c. Modifier path — `_run_modifier_mode()`, current line ~2414

Current code: `save_config(modifier_result_to_config(result), paths["config"])`.
Same insertion: assign to a variable, conditionally attach, then save. The
modifier config reuses `result_to_config`, so the same call works verbatim
(use `result.constraints` for the workload dict).

## 3. Flag suggestions

```python
p.add_argument("--cost-usd", action="store_true",
               help="Attach a cost_estimate_usd block (training_total / "
                    "serving_per_1m_tokens / annual_serving_at_load) to the "
                    "emitted config. Pure-add; list prices from "
                    "ac/pricing_specs/. Default off.")
p.add_argument("--price-tier", default="on_demand",
               choices=["on_demand", "reserved_1y", "spot"],
               help="Price tier for --cost-usd (default: on_demand). "
                    "Falls back to on_demand, then reference_estimate, "
                    "when a tier is not published for the target.")
```

Naming rationale: `--cost-usd` matches the plan ("dollar units"); the tier
flag is orthogonal and future-proof against hot-updated spec files.

## 4. Objective-profile semantics (per plan §5)

- **Existing profiles unchanged.** Do NOT touch
  `ac/optimizer.py:OBJECTIVE_PROFILES` weights; `serving_cost` /
  `training_cost` keep their current proxy-unit semantics and Pareto
  ranking. The cost block is an appended column, not an objective input.
- **Dollar units only when `*_cost` profiles are selected:** suggested
  UX-only addition — when `args.objective_profile in ("serving_cost",
  "training_cost")` and `--cost-usd` is set, extend the `_format_optimal_line`
  log line (current line ~128) with the matching USD figure
  (`serving_per_1m_tokens` for serving_cost, `training_total` for
  training_cost). Ranking logic stays byte-identical.
- Making USD a true optimization objective would require per-candidate
  price evaluation inside the optimizer loop — explicitly out of Task E
  scope (would violate the pure-add / zero-regression gate).

## 5. Packaging (one line, at assembly)

Editable installs (today's setup) resolve `ac/pricing_specs/` from the
source tree, but wheel/sdist builds only ship declared package data. Add
`"pricing_specs/*.json"` to `[tool.setuptools.package-data]` in
`pyproject.toml` (next to `hardware_specs/*.json`) so the price books ship.
Left out of Task E scope to keep the wave pure-add-only.

## 6. Golden-snapshot guarantee

With `--cost-usd` off, nothing changes: `ac/pricing.py` is imported but
`attach_cost_block` is never called, and `result_to_config` is untouched.
Verified in this wave by running `scripts/golden_snapshot.sh` into a
scratch dir and diffing against `out/golden`: **zero diff** (see
`validation/e_pricing/snapshot_check_diff.txt`).
