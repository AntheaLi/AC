"""
Delta-eval CLI.

    delta-eval \\
        --baseline-config configs/mistral_7b_like.json \\
        --hardware h100 --tp 8 \\
        --apply swap_attention_to_gqa --apply-args group_size=8 \\
        --out outputs/mistral_gqa_eval/

Supports:
    --apply REPEAT      — multiple --apply NAME flags compose into a sequence
    --apply-args k=v    — args for the most recent --apply (repeatable)
    --workload PRESET   — chat | batched | long_context | training
    --serving-batch N   — explicit override (overrides preset)
    --context-length N  — explicit override
    --no-pareto         — skip the Pareto-position computation (cheap mode)
    --json              — dump JSON only (no Markdown)
    --stdout            — print Markdown to stdout instead of writing files

Outputs (under --out):
    evaluation.json     — full DeltaEvaluation serialized
    report.md           — one-screen Markdown report
    pareto.csv          — 3-row Pareto CSV (baseline / candidate / delta)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from optimizer import DeploymentConstraints  # noqa: E402
from baseline import load_baseline_model  # noqa: E402

from evaluator import (  # noqa: E402
    DeltaEvaluation,
    evaluate_delta,
    evaluate_delta_sequence,
)
from report import (  # noqa: E402
    render_markdown,
    render_markdown_multi,
    render_json,
    render_pareto_csv,
)


# =============================================================================
# Workload presets (multiplicative scaling of constraints)
# =============================================================================

_WORKLOAD_PRESETS = {
    "chat": {
        "serving_batch": 1,
        "context_length": 2048,
        "prompt_len": 2048,
        "serving_tbt_ms": 50.0,
    },
    "batched": {
        "serving_batch": 64,
        "context_length": 4096,
        "prompt_len": 4096,
        "serving_tbt_ms": 50.0,
    },
    "long_context": {
        "serving_batch": 8,
        "context_length": 32768,
        "prompt_len": 32768,
        "serving_tbt_ms": 80.0,
    },
    "training": {
        "serving_batch": 4,
        "context_length": 4096,
        "prompt_len": 4096,
        "serving_tbt_ms": 200.0,
    },
}

VALID_HARDWARE = ["h100", "b200", "tpu_v5p", "tpu_v5e",
                  "trainium2", "trn2", "trainium3", "trn3"]


def _positive_int(v: str) -> int:
    out = int(v)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return out


def _non_negative_float(v: str) -> float:
    out = float(v)
    if out < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return out


def _coerce_arg_value(v: str) -> Any:
    """Coerce a CLI arg string into int / float / bool / str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


# =============================================================================
# Custom argparse: collect --apply NAME [--apply-args k=v]... groups
# =============================================================================

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="delta-eval",
        description="Evaluate the Pareto-front and metric influence of a "
                    "delta transformation against a baseline architecture.",
    )
    parser.add_argument("--baseline-config", required=True,
                        help="Path to a compiler-schema JSON baseline.")
    parser.add_argument("--hardware", "--hw", dest="hardware", default="h100",
                        choices=VALID_HARDWARE)
    # Default None = "not explicitly set". Resolution order (see main):
    # explicit CLI flag > baseline config's parallelism block > 8/1/8.
    # A concrete argparse default here would be indistinguishable from an
    # explicit flag, which used to make --tp silently lose to the config.
    parser.add_argument("--tp", type=_positive_int, default=None,
                        help="Tensor-parallel degree. Default: the baseline "
                             "config's parallelism.tensor_parallel, else 8.")
    parser.add_argument("--pp", type=_positive_int, default=None,
                        help="Pipeline-parallel degree. Default: the baseline "
                             "config's parallelism.pipeline_parallel, else 1.")
    parser.add_argument("--dp", type=_positive_int, default=None,
                        help="Data-parallel degree. Default: the baseline "
                             "config's parallelism.data_parallel, else 8.")
    parser.add_argument("--workload", default="chat",
                        choices=list(_WORKLOAD_PRESETS.keys()))
    parser.add_argument("--serving-batch", type=_positive_int, default=None,
                        help="Override workload preset.")
    parser.add_argument("--context-length", type=_positive_int, default=None,
                        help="Override workload preset.")
    parser.add_argument("--prompt-len", type=_positive_int, default=None)
    parser.add_argument("--target-params-b", type=_non_negative_float, default=0.0,
                        help="0 means inferred from baseline.")
    parser.add_argument("--training-tokens", type=_positive_int,
                        default=20_000_000_000_000)
    parser.add_argument("--unique-training-tokens", type=_positive_int,
                        default=None)
    parser.add_argument("--pretraining-context-length", type=_positive_int,
                        default=8192)
    parser.add_argument(
        "--quality-model",
        choices=["effective_capacity_v2", "legacy_residual_v1"],
        default="effective_capacity_v2",
    )
    parser.add_argument("--apply", action="append", required=True,
                        metavar="DELTA_NAME[:k=v,k=v]",
                        help="A transformation name with optional inline "
                             "kwargs. Examples: "
                             "`--apply swap_attention_to_mla:latent_dim=256` "
                             "or `--apply 'swap_attention_to_mla{latent_dim=256,d_rope=64}'`. "
                             "Repeat for sequences.")
    # Deprecated two-flag form: --apply NAME --apply-args k=v ...
    # Kept for back-compat, hidden from --help. See FEEDBACK item #12.
    parser.add_argument("--apply-args", action="append", default=[],
                        metavar="k=v", help=argparse.SUPPRESS)
    parser.add_argument("--no-pareto", action="store_true",
                        help="Skip Pareto-position classification.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: "
                             "outputs/delta_eval_<baseline>_<delta>/")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON only (no Markdown / CSV).")
    parser.add_argument("--stdout", action="store_true",
                        help="Print Markdown to stdout instead of writing files.")
    return parser.parse_args(argv)


def _parse_inline_args(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Parse an inline --apply value: NAME, NAME:k=v,k=v, or NAME{k=v,k=v}.

    Eliminates the "args bound to the closest preceding --apply" rule that
    was easy to get wrong in shell history (FEEDBACK item #12).
    """
    s = (spec or "").strip()
    if not s:
        return "", {}
    if s.endswith("}") and "{" in s:
        name, _, body = s[:-1].partition("{")
        return name.strip(), _parse_kv_body(body)
    if ":" in s:
        name, _, body = s.partition(":")
        return name.strip(), _parse_kv_body(body)
    return s, {}


def _parse_kv_body(body: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for chunk in body.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, _, v = chunk.partition("=")
        out[k.strip()] = _coerce_arg_value(v.strip())
    return out


def _build_delta_groups(argv: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Walk argv to build (name, kwargs) groups from --apply / --apply-args.

    Two syntaxes are supported:

        --apply NAME[:k=v,k=v]                 (inline form, preferred)
        --apply NAME --apply-args k=v ...      (legacy two-flag form)

    The legacy form remains for back-compat; the inline form removes the
    positional binding (FEEDBACK item #12).
    """
    groups: List[Tuple[str, Dict[str, Any]]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--apply" or tok.startswith("--apply="):
            if tok == "--apply":
                raw = argv[i + 1] if i + 1 < len(argv) else ""
                i += 2
            else:
                raw = tok.split("=", 1)[1]
                i += 1
            name, kw = _parse_inline_args(raw)
            groups.append((name, kw))
            continue
        if tok == "--apply-args":
            if not groups:
                i += 2
                continue
            kv = argv[i + 1] if i + 1 < len(argv) else ""
            # Reject empty / malformed --apply-args. Previously
            # `--apply-args ""` (or any value without `=`) was silently
            # swallowed by the `if "=" in kv:` branch below, so users got
            # no signal that their key-value pair never made it into the
            # delta. Fail fast in the same style as bogus-key validation.
            if not kv.strip() or "=" not in kv:
                last_delta = groups[-1][0] if groups else "<unknown>"
                print(
                    f"ERROR: --apply {last_delta}: --apply-args value {kv!r} "
                    "is not a valid `key=value` pair. Pass one key=value per "
                    "--apply-args, e.g. `--apply-args group_size=8`.",
                    file=sys.stderr,
                )
                sys.exit(2)
            # Fix #2: detect the legacy comma-separated form `k1=v1,k2=v2`
            # and emit an actionable error instead of letting the value
            # fall through as a string and surfacing later as a misleading
            # "k must be an integer". This is the most common migration
            # failure from v1-release / ac-cli-release-v1.
            k, _, v = kv.partition("=")
            if "," in v and "=" in v:
                print(
                    f"ERROR: --apply-args value {kv!r} looks like the "
                    "legacy comma-separated form `k1=v1,k2=v2`. "
                    "Repeat --apply-args once per key instead: "
                    f"`--apply-args {k}=<val> --apply-args "
                    f"{v.split(',', 1)[1].split('=', 1)[0]}=<val>`.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not k.strip():
                last_delta = groups[-1][0] if groups else "<unknown>"
                print(
                    f"ERROR: --apply {last_delta}: --apply-args value {kv!r} "
                    "has an empty key. Pass `key=value`.",
                    file=sys.stderr,
                )
                sys.exit(2)
            groups[-1][1][k] = _coerce_arg_value(v)
            i += 2
            continue
        i += 1
    return groups


def _validate_delta_groups(groups: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Validate --apply names + --apply-args keys against the delta REGISTRY
    *before* trying to evaluate. Emits a useful, actionable error instead of
    a late TypeError from the delta's `apply()` method.
    """
    import inspect
    # Lazy import — keeps the cli importable even when v1-stress isn't on path
    # at module import time.
    from deltas import REGISTRY
    errors: List[str] = []
    for name, args in groups:
        if name not in REGISTRY:
            errors.append(
                f"--apply {name!r} is not a known transformation. "
                f"Available: {', '.join(sorted(REGISTRY.keys()))}"
            )
            continue
        sig = inspect.signature(REGISTRY[name]().apply)
        # arg 0 is `arch` — strip it. Keep everything else as legal kwargs.
        legal = [p for p in sig.parameters.keys() if p != "arch"]
        legal_set = set(legal)
        for k in args.keys():
            if k not in legal_set:
                errors.append(
                    f"--apply {name}: unknown --apply-args key {k!r}. "
                    f"Legal keys for this delta: {', '.join(legal) or '(none)'}"
                )
        errors.extend(_validate_delta_arg_values(name, args))
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


def _is_int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_delta_arg_values(name: str, args: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    def require_int(key: str, minimum: int = None) -> None:
        if key not in args:
            return
        value = args[key]
        if not _is_int_value(value):
            errors.append(f"--apply {name}: {key} must be an integer")
            return
        if minimum is not None and value < minimum:
            errors.append(f"--apply {name}: {key} must be >= {minimum}")

    def require_float_gt_zero(key: str) -> None:
        if key not in args:
            return
        value = args[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"--apply {name}: {key} must be numeric")
            return
        if float(value) <= 0:
            errors.append(f"--apply {name}: {key} must be > 0")

    if name == "swap_attention_to_gqa":
        require_int("group_size", 1)
    elif name == "swap_attention_to_mla":
        require_int("latent_dim", 16)
    elif name == "swap_attention_to_swa":
        require_int("window_size", 64)
    elif name == "add_state_layers":
        require_int("d_state", 1)
        ratio = args.get("ratio")
        if ratio is not None and ratio != "all":
            try:
                left, right = str(ratio).split(":")
                if int(left) <= 0 or int(right) <= 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"--apply {name}: ratio must be 'all' or positive A:B integers"
                )
    elif name == "densify_first_k":
        require_int("k", 0)
    elif name == "change_moe_topology":
        require_int("n_experts", 1)
        require_int("top_k", 1)
        require_int("expert_dim", 1)
        require_float_gt_zero("capacity_factor")
    elif name == "change_parallelism":
        for key in ("tp", "pp", "ep", "cp", "dp"):
            require_int(key, 1)
    elif name == "scale_d_model":
        require_int("delta")
        require_int("align", 1)
    elif name == "scale_n_layers":
        require_int("delta")
    elif name == "change_precision_per_component":
        valid = {"bf16", "fp16", "fp32", "fp8", "fp4", "int8", "int4", "mxfp4", "mxfp6"}
        for key in ("kv", "weight", "activation"):
            if key in args and str(args[key]).lower() not in valid:
                errors.append(
                    f"--apply {name}: {key} must be one of {', '.join(sorted(valid))}"
                )
    return errors


# =============================================================================
# Main
# =============================================================================

def main(argv: List[str] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = _parse_args(argv)

    # Wave 28: fail fast on a broken AC_QUALITY_DEFAULTS /
    # AC_HARDWARE_SPEC_DIR. Previously the evaluator's per-candidate
    # exception handler swallowed the real error and this CLI exited 0
    # with "delta was infeasible against baseline" — misattributing an
    # environment typo to the delta being evaluated.
    from quality_model import validate_calibration_environment
    try:
        validate_calibration_environment(getattr(args, "hardware", None))
    except Exception as e:
        print(
            f"ERROR: invalid calibration environment: {e}\n"
            "Fix or unset AC_QUALITY_DEFAULTS / AC_HARDWARE_SPEC_DIR and "
            "re-run. AC does not fall back to different constants.",
            file=sys.stderr,
        )
        return 2

    delta_groups = _build_delta_groups(argv)
    if not delta_groups:
        print("No --apply DELTA_NAME provided.", file=sys.stderr)
        return 2

    # Validate delta names + kwargs early so users get a useful error before
    # we load and evaluate anything.
    _validate_delta_groups(delta_groups)

    # Fix #3: warn on non-power-of-two parallelism (typo guard).
    def _is_pow2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0
    for label, val in (("--tp", args.tp), ("--pp", args.pp), ("--dp", args.dp)):
        if val is not None and not _is_pow2(val):
            print(
                f"WARNING: {label}={val} is not a power of two. Most accelerator "
                "nodes use power-of-two parallelism; double-check this is intended.",
                file=sys.stderr,
            )

    # Warn once if the chosen hardware has no calibration table.
    try:
        from throughput_model import warn_if_uncalibrated
        warn_if_uncalibrated(args.hardware)
    except Exception:
        pass

    # Direct config-validity checks. EP=1 is a legal TP-sharded MoE plan
    # (Wave 21); the throughput/memory models price the larger resident
    # expert set directly, so no special rejection belongs here.
    #
    # Wave 7a.2 adds validators that
    # of which used to be caught via the same memory/quality→INFEASIBLE
    # side-channel and now silently produce TBT-inflated outputs without
    # the side-channel:
    #
    #   2. attention.type=="mla" but no kv_latent_dim — malformed MLA spec.
    #   3. attention.type=="nsa" but missing NSA fields.
    #   4. MoE with expert_parallel > NVLink domain — inter-node EP is
    #      50-100× slower than intra-node, optimizer never picks it.
    #   5. tensor_parallel > NVLink domain — same reasoning.
    #   6. params.kind says hybrid but no state-typed layer present —
    #      malformed hybrid (state_config declared but state_layers <= 0).
    def _emit_infeasible(reason: str) -> None:
        """Mirror the existing MoE-no-EP branch: write a stub
        evaluation.json so test_delta_eval_* can read structured failure."""
        import json as _json_w
        out_dir = args.out
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "evaluation.json"), "w") as _ef:
                _json_w.dump({
                    "feasible": False,
                    "reason_if_infeasible": reason,
                    "metrics": {},
                }, _ef)
        except OSError:
            pass

    def _nvlink_cap_for(hw_name: str) -> int:
        """Return the NVLink-domain size the search uses for this hw.

        Mirrors throughput_model._nvlink_domain_size. We load the hardware
        spec; if that fails (e.g. uncalibrated alias) we fall back to the
        family default — 8 for Hopper/B200 default SKU, 16 for TPUs /
        Trainium-2, 32 for Trainium-3."""
        try:
            from throughput_model import load_hardware, _nvlink_domain_size
            return int(_nvlink_domain_size(load_hardware(hw_name)))
        except Exception:
            fallback = {
                "h100": 8, "b200": 8,
                "tpu_v5e": 8, "tpu_v5p": 16,
                "trainium2": 16, "trn2": 16,
                "trainium3": 32, "trn3": 32,
            }
            return fallback.get(hw_name, 8)

    try:
        import json as _json
        with open(args.baseline_config) as _f:
            _raw = _json.load(_f)
        _meta = _raw.get("metadata", {}) if isinstance(_raw, dict) else {}
        _par = _raw.get("parallelism", {}) if isinstance(_raw, dict) else {}
        _arch = _raw.get("architecture", {}) if isinstance(_raw, dict) else {}
        _layers = _arch.get("layer_configs", []) or []
        _has_moe = any(
            (lc.get("ffn", {}) or {}).get("type", "").lower() == "moe"
            for lc in _layers
        )
        _ep = int(_par.get("expert_parallel", 0) or 0)
        _tp_cfg = int(_par.get("tensor_parallel", 0) or 0)
        _nvlink_cap = _nvlink_cap_for(args.hardware)

        # Validator 1 (Wave 7a.2): hybrid metadata without state layers.
        # Detect via metadata.params.kind containing "hybrid", combined with
        # zero layer_configs that carry an actual `state` block. We look at
        # the *params.kind* string rather than at a separate state_config
        # field because schema-v2 keeps state on the per-layer block; the
        # only top-level signal is the params.kind label.
        _kind = str((_meta.get("params") or {}).get("kind", "")).lower()
        if "hybrid" in _kind or "state" in _kind:
            _n_state = sum(
                1 for lc in _layers
                if (lc.get("state") is not None) or
                   str(lc.get("type", "")).lower() == "state_block"
            )
            if _n_state <= 0:
                print(
                    f"ERROR: baseline metadata declares params.kind='{_kind}' "
                    "but no layer_configs carry a state block. Malformed "
                    "hybrid: state_config declared but state_layers=0. Either "
                    "fix params.kind to 'dense'/'moe' or add state layers.",
                    file=sys.stderr,
                )
                _emit_infeasible(
                    f"Hybrid metadata declares params.kind='{_kind}' but "
                    "layer_configs contain zero state layers. INFEASIBLE: "
                    "malformed hybrid spec (state_layers <= 0)."
                )
                return 2

        # Validator 3 (Wave 7a.2): MLA layer missing kv_latent_dim.
        # The optimizer's enumerator always emits an MLA candidate with
        # kv_latent_dim set; a baseline JSON that types itself "mla" but
        # forgets kv_latent_dim used to flow through the MHA-fallback path
        # and silently produce numbers as if MLA were never enabled. Reject.
        for _lc in _layers:
            _attn = _lc.get("attention") or {}
            if str(_attn.get("type", "")).lower() == "mla":
                if not _attn.get("kv_latent_dim"):
                    print(
                        "ERROR: layer declares attention.type='mla' but is "
                        "missing kv_latent_dim. Malformed MLA spec; the "
                        "optimizer never emits this shape. Add kv_latent_dim "
                        "(DeepSeek-V2/V3 default 512) to the layer's "
                        "attention block.",
                        file=sys.stderr,
                    )
                    _emit_infeasible(
                        "MLA layer missing kv_latent_dim. INFEASIBLE: "
                        "MLA requires kv_latent_dim."
                    )
                    return 2

        # Validator 4 (Wave 7a.2): NSA layer missing required NSA fields.
        # The required-fields list mirrors schema._validate_nsa_fields (the
        # schema constructor at emit time enforces this; the validator
        # mirrors the check at the CLI boundary so an externally-authored
        # baseline can't slip through).
        _required_nsa = ("nsa_compress_block_size", "nsa_select_top_k",
                         "nsa_window_size")
        for _lc in _layers:
            _attn = _lc.get("attention") or {}
            if str(_attn.get("type", "")).lower() == "nsa":
                _missing = [f for f in _required_nsa if _attn.get(f) is None]
                if _missing:
                    print(
                        f"ERROR: layer declares attention.type='nsa' but is "
                        f"missing required NSA fields: {sorted(_missing)}. "
                        "Add the block parameters (compress_block_size, "
                        "select_top_k, window_size) — defaults are documented "
                        "in schema.NSA_DEFAULTS.",
                        file=sys.stderr,
                    )
                    _emit_infeasible(
                        f"NSA layer missing fields {sorted(_missing)}. "
                        "INFEASIBLE: NSA spec incomplete."
                    )
                    return 2

        # Cross-node EP is legal but expensive. The throughput model switches
        # all-to-all to inter-node bandwidth beyond the local fabric domain.
        if _has_moe and _ep > _nvlink_cap:
            print(
                f"WARNING: MoE expert_parallel={_ep} exceeds the {args.hardware} "
                f"NVLink-domain cap of {_nvlink_cap}. The all-to-all would "
                f"use inter-node fabric; AC will include that delay.",
                file=sys.stderr,
            )

        # Cross-node TP is also legal. `_allreduce_cost` applies inter-node
        # bandwidth and the larger collective launch-latency floor.
        if _tp_cfg > _nvlink_cap:
            print(
                f"WARNING: tensor_parallel={_tp_cfg} exceeds the {args.hardware} "
                f"NVLink-domain cap of {_nvlink_cap}. Tensor-parallel "
                f"all-reduce will use inter-node bandwidth; AC will include "
                f"that delay.",
                file=sys.stderr,
            )
    except (OSError, _json.JSONDecodeError):
        # If we can't parse the raw config, fall through; load_baseline_model
        # below will produce a more specific error.
        pass

    # 1) Load baseline. Done AFTER the raw-JSON validators so the latter run
    # first — they reject configurations that load_baseline_model's schema
    # validator would also reject, but with a structured evaluation.json
    # output rather than a bare CLI error. This matches the spec's intent:
    # a malformed baseline produces a feasibility=False record downstream
    # tools can read, not just an exit-2 ERROR.
    try:
        bm = load_baseline_model(args.baseline_config)
    except ValueError as exc:
        print(f"ERROR: Could not load baseline config: {exc}", file=sys.stderr)
        return 2

    # Resolve parallelism: explicit CLI flag > baseline config > 8/1/8.
    # Then stamp the resolved degrees onto BOTH the candidate and the
    # constraints. evaluate_candidate prefers the candidate's own
    # tp_degree/pp_degree when set, so stamping only the constraints would
    # let a config-declared degree silently override an explicit CLI flag
    # (the pre-fix behaviour: `--tp 4` against a tensor_parallel=8 config
    # evaluated at TP=8 while the report claimed TP=4).
    _cfg_par = (bm.config.get("parallelism") or {}) if isinstance(bm.config, dict) else {}
    resolved_tp = int(args.tp if args.tp is not None
                      else (getattr(bm.candidate, "tp_degree", 0) or 8))
    resolved_pp = int(args.pp if args.pp is not None
                      else (getattr(bm.candidate, "pp_degree", 0) or 1))
    resolved_dp = int(args.dp if args.dp is not None
                      else (_cfg_par.get("data_parallel") or 8))
    bm.candidate.tp_degree = resolved_tp
    bm.candidate.pp_degree = resolved_pp
    args.tp, args.pp, args.dp = resolved_tp, resolved_pp, resolved_dp

    # 2) Build DeploymentConstraints, layering preset + explicit overrides
    preset = _WORKLOAD_PRESETS[args.workload]
    # Wave 19 (P0-2): prompt length must FOLLOW an overridden context.
    # Every preset ties prompt_len == context_length, but the old resolution
    # (`args.prompt_len or preset["prompt_len"]`) kept the preset's prompt
    # when only --context-length was overridden — so `long_context
    # --context-length 131072` silently evaluated a 32k prefill (TTFT
    # bit-identical from 16k to 131k), and at --context-length 16384 the
    # 32k prompt exceeded the context window entirely.
    resolved_context = int(args.context_length or preset["context_length"])
    if args.prompt_len is not None:
        resolved_prompt = int(args.prompt_len)
    elif args.context_length is not None:
        resolved_prompt = resolved_context
    else:
        resolved_prompt = int(preset["prompt_len"])
    if resolved_prompt > resolved_context:
        print(
            f"ERROR: prompt_len ({resolved_prompt}) exceeds context_length "
            f"({resolved_context}). A prompt cannot be longer than the "
            "context window; pass a consistent --prompt-len / "
            "--context-length pair.",
            file=sys.stderr,
        )
        return 2
    try:
        constraints = DeploymentConstraints(
            target_params_b=(args.target_params_b
                             if args.target_params_b > 0
                             else bm.candidate.total_params / 1e9),
            training_tokens=args.training_tokens,
            unique_training_tokens=args.unique_training_tokens,
            pretraining_context_length=args.pretraining_context_length,
            quality_model_version=args.quality_model,
            context_length=resolved_context,
            prompt_len=resolved_prompt,
            serving_tbt_ms=preset["serving_tbt_ms"],
            serving_ttft_ms=2000.0,
            serving_batch=args.serving_batch or preset["serving_batch"],
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 3) Evaluate. Single delta vs sequence.
    if len(delta_groups) == 1:
        name, dargs = delta_groups[0]
        ev = evaluate_delta(
            bm.candidate, args.hardware, constraints,
            delta_name=name, delta_args=dargs,
            baseline_name=bm.name,
            include_pareto=not args.no_pareto,
        )
        results = [ev]
    else:
        ev = evaluate_delta_sequence(
            bm.candidate, args.hardware, constraints,
            deltas=delta_groups,
            baseline_name=bm.name,
            include_pareto=not args.no_pareto,
        )
        results = [ev]

    # Fix #3: stamp the resolved workload onto every evaluation so reports
    # and JSON make it unambiguous what assumptions the predictions used.
    resolved = {
        "workload_preset": args.workload,
        "serving_batch": int(constraints.serving_batch),
        "context_length": int(constraints.context_length),
        "prompt_len": int(constraints.prompt_len or constraints.context_length),
        "serving_tbt_ms_budget": float(constraints.serving_tbt_ms),
        "serving_ttft_ms_budget": float(constraints.serving_ttft_ms or 0.0) or None,
        "tp": int(constraints.tp),
        "pp": int(constraints.pp),
        "dp": int(constraints.dp),
        # EP/CP have no CLI flag here; they come from the baseline config
        # (change_parallelism deltas surface overrides in field_changes).
        "ep": int(getattr(bm.candidate, "ep_degree", 1) or 1),
        "cp": int(getattr(bm.candidate, "cp_degree", 1) or 1),
    }
    for ev in results:
        ev.resolved_workload = resolved

    # 4) Output
    if args.stdout:
        if args.json:
            for ev in results:
                print(render_json(ev))
        else:
            print(render_markdown_multi(results)
                   if len(results) > 1
                   else render_markdown(results[0]))
        return 0

    out_dir = args.out
    if not out_dir:
        slug = "+".join(g[0] for g in delta_groups)
        out_dir = os.path.join(
            "outputs", f"delta_eval_{bm.name}_{slug}")
    os.makedirs(out_dir, exist_ok=True)

    # evaluation.json
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        if len(results) > 1:
            f.write(json.dumps([ev.as_dict() for ev in results],
                                indent=2, default=str))
        else:
            f.write(render_json(results[0]))

    if not args.json:
        # report.md
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            if len(results) > 1:
                f.write(render_markdown_multi(results))
            else:
                f.write(render_markdown(results[0]))
        # pareto.csv
        with open(os.path.join(out_dir, "pareto.csv"), "w") as f:
            f.write(render_pareto_csv(results[0]))

    print(f"Wrote evaluation to: {out_dir}")

    # Fix #11: surface infeasibility to stderr so users notice when a delta
    # was a precondition-failed no-op (e.g. densify_first_k on a dense
    # baseline). Previously the CLI exited 0 with "Wrote evaluation to: …"
    # and the user had to open the report to see "Infeasible".
    infeasible = [ev for ev in results if not ev.feasible]
    if infeasible:
        for ev in infeasible:
            print(
                f"WARNING: delta `{ev.delta_name}` was infeasible against "
                f"baseline `{ev.baseline_name}`: {ev.reason_if_infeasible}",
                file=sys.stderr,
            )
        # Exit non-zero so automation pipelines notice that the delta
        # evaluation hit infeasibility (e.g. the INFEASIBLE-sentinel
        # leak from a malformed MoE baseline). The output files are
        # still written so the user can inspect what happened.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
