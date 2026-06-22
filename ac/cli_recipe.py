"""Recipe + override + help-group helpers shared by every AC CLI.

The motivation, captured in one paragraph so future contributors don't
re-litigate it: the greenfield / modifier / delta-eval invocations carry
8–14 flags each, the flag names are dense, and the differences between
"my H100 dense 7B run" and "my B200 MoE+MLA long-context run" are
exactly the flag bundle. We make that bundle a first-class artifact —
the `recipe` — which can be checked into a repo next to a research
note, replayed with `--recipe path.yaml`, and tweaked with `--override
key=value`. The recipe format is intentionally just "the same flag
names you'd type at the CLI, written as YAML keys". No new DSL, no new
field names, no new validation. The `--override` flag is last-word-wins
so you can scale up a recipe with `--override params=70` without
copying the file.

Conventions:
 - Recipe values map onto argparse `dest` names (`tp`, `serving_batch`,
   `allow_moe`, `moe_n_experts`, ...) — i.e. flag names with dashes
   replaced by underscores and the leading `--` stripped. We also accept
   the dash form (`--tp`, `serving-batch`) and normalize.
 - `null` / `None` values in a recipe remove the flag (use the
   argparse default).
 - Lists in YAML become comma-joined strings (because the underlying
   argparse types expect strings like `4,8`). Booleans become flag
   presence/absence.
 - `--override key=value` may be repeated; each `key` follows the same
   normalization rules as recipe keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_RECIPE_FLAG = "--recipe"
_OVERRIDE_FLAG = "--override"
_PRINT_RECIPE_FLAG = "--print-recipe"
_HELP_GROUP_FLAG = "--help-group"


def _normalize_key(key: str) -> str:
    """Accept --tp, tp, serving-batch, serving_batch — all map to the
    argparse `dest` form (snake_case, no leading dashes).
    """
    k = key.strip()
    if k.startswith("--"):
        k = k[2:]
    if k.startswith("-"):
        k = k[1:]
    return k.replace("-", "_")


def _yaml_load(path: str) -> Dict[str, Any]:
    """Load a recipe file. YAML preferred (PyYAML is already in the
    pyproject), JSON accepted as a fallback. Failures raise SystemExit
    with a structured message instead of a stack trace.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"ERROR: recipe loading needs PyYAML installed ({e!r})."
        )
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: recipe file not found: {path}")
    except Exception as e:
        raise SystemExit(f"ERROR: could not parse recipe {path!r}: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(
            f"ERROR: recipe {path!r} must be a YAML/JSON mapping, "
            f"got {type(data).__name__}."
        )
    # Accept two formats interchangeably:
    #   1. Flat mapping ({hardware: h100, params: 7, ...})
    #   2. Wrapped ({flags: {hardware: h100, ...}, description: "..."})
    # Form 2 leaves room for top-level metadata (description, owner,
    # provenance) without colliding with parser dest names.
    if "flags" in data and isinstance(data["flags"], dict):
        return data["flags"]
    return data


def _stringify(value: Any) -> Optional[str]:
    """Render a recipe value into the string form argparse expects.

    Lists/tuples get comma-joined. Booleans control flag presence (the
    caller handles that). None means "skip" (use the parser default).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None if not value else ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _kv_to_argv(key: str, value: Any) -> List[str]:
    """Convert a single (key, value) into argv tokens.

    Booleans → store_true (just the flag); other types → `--key`, `value`.
    """
    norm = _normalize_key(key)
    flag = "--" + norm.replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    s = _stringify(value)
    if s is None:
        return []
    return [flag, s]


def _split_override(token: str) -> Tuple[str, str]:
    if "=" not in token:
        raise SystemExit(
            f"ERROR: --override expects key=value, got {token!r}."
        )
    k, v = token.split("=", 1)
    return _normalize_key(k), v


def _extract_meta_flags(
    argv: Sequence[str],
) -> Tuple[Optional[str], List[str], Optional[str], Optional[str], List[str]]:
    """Pull out --recipe, --override, --print-recipe, --help-group
    from argv before argparse sees them. Returns
    (recipe_path, overrides, print_recipe_path, help_group, remaining_argv).
    """
    recipe_path: Optional[str] = None
    overrides: List[str] = []
    print_recipe_path: Optional[str] = None
    help_group: Optional[str] = None
    remaining: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == _RECIPE_FLAG and i + 1 < len(argv):
            recipe_path = argv[i + 1]; i += 2; continue
        if tok.startswith(_RECIPE_FLAG + "="):
            recipe_path = tok.split("=", 1)[1]; i += 1; continue
        if tok == _OVERRIDE_FLAG and i + 1 < len(argv):
            overrides.append(argv[i + 1]); i += 2; continue
        if tok.startswith(_OVERRIDE_FLAG + "="):
            overrides.append(tok.split("=", 1)[1]); i += 1; continue
        if tok == _PRINT_RECIPE_FLAG and i + 1 < len(argv):
            print_recipe_path = argv[i + 1]; i += 2; continue
        if tok.startswith(_PRINT_RECIPE_FLAG + "="):
            print_recipe_path = tok.split("=", 1)[1]; i += 1; continue
        if tok == _HELP_GROUP_FLAG and i + 1 < len(argv):
            help_group = argv[i + 1]; i += 2; continue
        if tok.startswith(_HELP_GROUP_FLAG + "="):
            help_group = tok.split("=", 1)[1]; i += 1; continue
        remaining.append(tok); i += 1
    return recipe_path, overrides, print_recipe_path, help_group, remaining


def expand_argv(argv: Sequence[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Expand --recipe + --override tokens into a flat argv argparse can
    consume.

    Order of precedence (last wins under argparse):
      1. recipe values (lowest)
      2. user-supplied CLI flags (middle)
      3. --override key=value tokens (highest)

    Returns (expanded_argv, print_recipe_path, help_group).
    """
    recipe_path, overrides, print_recipe_path, help_group, remaining = (
        _extract_meta_flags(argv)
    )
    out: List[str] = []
    if recipe_path:
        data = _yaml_load(recipe_path)
        for k, v in data.items():
            out.extend(_kv_to_argv(k, v))
    out.extend(remaining)
    for ov in overrides:
        k, v = _split_override(ov)
        # Booleans expressed as override: --override allow_moe=true
        if v.lower() in ("true", "yes", "on"):
            out.extend(_kv_to_argv(k, True))
        elif v.lower() in ("false", "no", "off"):
            # No-op for store_true flags. We can't express "turn off"
            # for a store_true flag through argparse without surgery,
            # so we simply omit the flag and let the parser default win.
            pass
        else:
            out.extend(["--" + k.replace("_", "-"), v])
    return out, print_recipe_path, help_group


# -----------------------------------------------------------------------------
# Help-group filter + recipe snapshot
# -----------------------------------------------------------------------------


_LOGICAL_GROUPS: Dict[str, List[str]] = {
    # name → flag-name prefixes (with leading "--")
    "hardware": ["--hardware", "--params", "--tokens"],
    "workload": [
        "--context", "--vocab-size", "--prompt-len", "--output-len",
        "--concurrency", "--scheduler", "--param-tolerance",
    ],
    "serving": ["--serving-"],
    "parallelism": ["--tp", "--tp-options", "--pp", "--dp", "--num-gpus",
                    "--cp", "--cp-method", "--cp-options"],
    "selection": ["--objective-profile"],
    "precision": ["--precision-modes", "--kv-dtypes"],
    "state": ["--allow-state", "--state-", "--placement-strategy"],
    "moe": [
        "--allow-moe", "--moe-", "--max-total-params-b",
        "--dense-ffn-layers", "--ep-options",
    ],
    "mla": ["--allow-mla", "--mla-"],
    "mtp": ["--allow-mtp", "--mtp-"],
    "rope": ["--allow-rope-scaling", "--rope-"],
    "nsa": ["--nsa"],
    "yoco": ["--yoco"],
    "modifier": [
        "--baseline-config", "--out", "--quality-risk-budget-pct",
        "--allow-quality-spending", "--top-modifications",
    ],
    "outputs": ["--output-", "--implementation-class-name"],
    "recipe": ["--recipe", "--override", "--print-recipe", "--help-group"],
    "misc": [
        "--no-shadow-prices", "--max-candidates", "--progress-every",
        "--quiet",
    ],
}


def render_group_help(parser: argparse.ArgumentParser, group_name: str) -> str:
    """Return help text for the named logical group.

    Logical groups are flag-name-prefix bundles (see _LOGICAL_GROUPS),
    not argparse argument groups. We use prefixes so we don't have to
    rewire every existing `add_argument` call into a separate group.
    Unknown names list the available group names instead of failing.
    """
    needle = (group_name or "").strip().lower()
    prefixes = _LOGICAL_GROUPS.get(needle)
    if prefixes is None:
        return (
            f"(no logical group matched {group_name!r}; available: "
            + ", ".join(sorted(_LOGICAL_GROUPS.keys()))
            + ")\n"
        )
    # Collect actions whose long option matches any prefix.
    matched = []
    for action in parser._actions:  # type: ignore[attr-defined]
        if not action.option_strings:
            continue
        opt = action.option_strings[0]
        if any(opt == pfx or opt.startswith(pfx) for pfx in prefixes):
            matched.append(action)
    if not matched:
        return f"(group {group_name!r} resolved no flags)\n"
    out = [f"usage: {parser.prog} [--help-group {group_name} flags]", ""]
    out.append(f"flags in group `{group_name}`:")
    for a in matched:
        flag = ", ".join(a.option_strings)
        helptxt = (a.help or "").replace("\n", " ")
        if a.default not in (None, False, [], "", argparse.SUPPRESS):
            helptxt = f"{helptxt} (default: {a.default})"
        out.append(f"  {flag}")
        if helptxt:
            out.append(f"      {helptxt}")
    return "\n".join(out) + "\n"


_SNAPSHOT_SKIP = {
    # Meta flags — these are intercepted by expand_argv and re-supplying
    # them in the snapshot would shadow the recipe semantics.
    "recipe", "override", "help_group", "print_recipe",
    # Internal stash from parse_args.
    "_print_recipe_path", "_defaults_snapshot",
    # Always-on / output-controlling flags that should be decided per
    # replay, not frozen into the recipe.
    "quiet", "progress_every",
    # Output paths are run-specific — freezing them into the recipe
    # makes the recipe machine-bound and would clobber the original
    # outputs on replay. The caller supplies them at replay time.
    "output_config", "output_justification", "output_pareto",
    "output_shadow_prices", "output_assumptions", "output_model_card",
    "output_implementation", "implementation_class_name",
    "out",
}


# Reverse maps for round-trip safety. The CLI parsers normalize input
# strings into internal representations (kv_dtypes "bf16,int8" → [16, 8];
# precision_modes "bf16,fp8_ffn" → ["all_bf16", "ffn_fp8"]). Without
# these reverse maps the snapshot would write the post-parse form and
# the next --recipe load would error out.
_KV_DTYPE_INT_TO_INPUT = {16: "bf16", 8: "int8", 4: "int4"}
_PRECISION_MODE_CANONICAL_TO_INPUT = {
    "all_bf16": "bf16",
    "ffn_fp8": "fp8_ffn",
    "all_fp8": "fp8",
    "ffn_fp4": "fp4_ffn",
    "all_fp4": "fp4",
    "ffn_mxfp4": "mxfp4_ffn",
    "all_mxfp4": "mxfp4",
    "ffn_mxfp6": "mxfp6_ffn",
    "all_mxfp6": "mxfp6",
}


def _denormalize_for_snapshot(key: str, value: Any) -> Any:
    """Convert a parsed argparse value back to the string the CLI parser
    would have accepted as input. This is what makes
    `--print-recipe → --recipe` actually round-trip.
    """
    if key == "kv_dtypes" and isinstance(value, (list, tuple)):
        return ",".join(_KV_DTYPE_INT_TO_INPUT.get(int(v), str(v)) for v in value)
    if key == "precision_modes" and isinstance(value, (list, tuple)):
        return ",".join(
            _PRECISION_MODE_CANONICAL_TO_INPUT.get(str(v), str(v)) for v in value
        )
    return value


def snapshot_recipe(args: argparse.Namespace, path: str) -> None:
    """Write the resolved args namespace to a YAML recipe file.

    Skips None-valued fields, the parser's own default values, and the
    meta flags listed in _SNAPSHOT_SKIP. Output paths are excluded so
    snapshots are machine-portable. List-valued args (kv_dtypes,
    precision_modes) are de-normalized back to the input alias form so
    the snapshot is replayable. The result is the minimal recipe that,
    when replayed with `--recipe <path>` plus per-run output paths,
    reproduces the run. We deliberately exclude parser defaults so
    future AC versions that ship different defaults don't get locked
    into stale recipes.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"ERROR: --print-recipe needs PyYAML ({e!r}).")
    payload: Dict[str, Any] = {}
    # Build a defaults table by re-invoking the same logic the parser
    # used. We rely on the public attribute set; if a key isn't in the
    # parser's defaults (e.g. injected by main), we keep it.
    defaults_attr = getattr(args, "_defaults_snapshot", None) or {}
    for k, v in sorted(vars(args).items()):
        if v is None or k in _SNAPSHOT_SKIP:
            continue
        if k in defaults_attr and defaults_attr[k] == v:
            continue
        payload[k] = _denormalize_for_snapshot(k, v)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "# AC recipe — replay with: ac-compile --recipe <this-file>\n"
            "# Override any field with: --override key=value\n"
            "# Output paths are intentionally omitted; pass\n"
            "# `--output-config out/<run>/arch.json` (etc.) on replay.\n"
        )
        yaml.safe_dump({"flags": payload}, f, sort_keys=True)


# -----------------------------------------------------------------------------
# Subcommand entry points
# -----------------------------------------------------------------------------


_STARTER_RECIPE_TEMPLATES = {
    "h100_dense_7b": {
        "hardware": "h100",
        "params": 7,
        "tokens": 2,
        "context": 8192,
        "serving_tbt": 50,
        "serving_batch": 32,
        "tp": 8,
        "pp": 1,
        "dp": 8,
        "objective_profile": "research_quality",
    },
    "b200_moe_mla_long_ctx": {
        "hardware": "b200",
        "params": 35,
        "tokens": 8,
        "context": 131072,
        "serving_tbt": 60,
        "serving_batch": 8,
        "tp": 8,
        "pp": 4,
        "dp": 4,
        "cp": 4,
        "allow_moe": True,
        "moe_n_experts": "256",
        "moe_top_k": "8",
        "allow_mla": True,
        "mla_kv_latent": "512",
        "mla_q_latent": "1536",
        "allow_mtp": True,
        "mtp_depths": "0,1",
        "allow_rope_scaling": True,
        "rope_original_max_position": 32768,
        "rope_scaling_methods": "longrope",
        "max_total_params_b": 800,
        "max_candidates": 200,
    },
    "delta_mistral_gqa_long_ctx": {
        "baseline_config": "configs/mistral_7b.json",
        "hardware": "h100",
        "tp": 8,
        "workload": "long_context",
        "apply": "swap_attention_to_gqa:group_size=8",
    },
}


def run_init(prog: str, argv: Sequence[str]) -> int:
    """`ac-compile init <name>` writes a starter recipe.

    Without a name, lists available templates. We deliberately keep
    this non-interactive — the spec asked for "starter recipe" and an
    interactive prompt adds dependencies (terminal handling) without
    much value over editing the resulting YAML.
    """
    args = list(argv)
    if not args or args[0] in ("--help", "-h"):
        print(f"usage: {prog} <template> [--out PATH]")
        print()
        print("available templates:")
        for k in _STARTER_RECIPE_TEMPLATES:
            print(f"  {k}")
        return 0
    name = args[0]
    out_path = "recipe.yaml"
    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 < len(args):
            out_path = args[idx + 1]
    if name not in _STARTER_RECIPE_TEMPLATES:
        print(
            f"ERROR: unknown starter recipe {name!r}. Try one of: "
            + ", ".join(sorted(_STARTER_RECIPE_TEMPLATES))
        )
        return 2
    try:
        import yaml  # type: ignore
    except Exception as e:
        print(f"ERROR: starter recipes need PyYAML ({e!r}).")
        return 2
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# AC starter recipe: {name}\n")
        f.write(f"# Replay: {prog.split()[0]} --recipe {out_path}\n")
        yaml.safe_dump(_STARTER_RECIPE_TEMPLATES[name], f, sort_keys=True)
    print(f"Wrote {out_path}")
    return 0


def run_config_show(
    prog: str,
    build_parser,
    argv: Sequence[str],
    infer_paths=None,
) -> int:
    """`ac-compile config show ...` resolves args (recipe + overrides
    included) without running the search, then prints what would
    happen: flag values, resolved output paths, and warnings AC would
    emit (untrained hardware, --tp / --tp-options conflicts, etc.).

    `infer_paths` is an optional callable `args -> dict[str, str]`
    that resolves the sibling output paths the real run would derive
    from --output-config. Without it the caller still gets the raw
    --output-config value but no sibling-path preview.
    """
    expanded, _, _ = expand_argv(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(expanded)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    inferred = {}
    if infer_paths is not None:
        try:
            inferred = infer_paths(args)
        except Exception as e:
            # Path inference must not crash the preview. Surface the
            # failure in the payload so users can spot a misconfigured
            # output dir without losing the rest of the report.
            inferred = {"_error": f"path inference failed: {e!r}"}
    payload: Dict[str, Any] = {
        "command": prog,
        "resolved_args": {
            k: v for k, v in sorted(vars(args).items()) if v is not None
        },
        "inferred_output_paths": inferred,
        "warnings": _config_show_warnings(args),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _config_show_warnings(args: argparse.Namespace) -> List[str]:
    warnings: List[str] = []
    tp_opts = getattr(args, "tp_options", None)
    tp = getattr(args, "tp", None)
    if tp_opts and tp:
        warnings.append(
            "Both --tp and --tp-options are set; modifier mode uses "
            "--tp-options and ignores --tp. Remove one to silence."
        )
    hw = getattr(args, "hardware", None)
    if hw in {"tpu_v5e", "trainium2", "trn2", "trainium3", "trn3"}:
        warnings.append(
            f"Hardware {hw!r} ships without a calibration table; "
            "absolute TPS/TBT/loss will be uncalibrated priors. Run "
            "ac-auto-calibrate fit to ground them."
        )
    if getattr(args, "params", None) is None and getattr(
        args, "baseline_config", None
    ) is None:
        warnings.append(
            "Neither --params nor --baseline-config is set; the run "
            "will fail at parse time."
        )
    # A latency-flavoured objective profile with no TBT budget will
    # silently fall back to the default profile weights — the optimizer
    # has nothing to optimize against. Catch this in the preview so the
    # user doesn't burn a multi-minute greenfield run before noticing.
    profile = getattr(args, "objective_profile", None)
    serving_tbt = getattr(args, "serving_tbt", None)
    serving_ttft = getattr(args, "serving_ttft", None)
    if profile in {"latency", "serving_cost"} and not (serving_tbt or serving_ttft):
        warnings.append(
            f"--objective-profile={profile} but no --serving-tbt / "
            "--serving-ttft is set; the latency objective has nothing "
            "to optimize against. Set a serving budget or pick a "
            "different --objective-profile."
        )
    return warnings
