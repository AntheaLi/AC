"""
Generate pre-computed optimizer results for the interactive web compiler.

Grid: hardware × param_target × tokens × serving_mode
Each run takes ~1s, total ~60-90s for the full grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Path shim (rewritten by ac-cli-release/scripts/regen_v1_web_data.py).
# Importing ac.optimizer etc. from the fixed-CLI tree before fallbacks.
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_AC = _os.path.join(_ROOT, "ac")
for _p in (_AC, _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from optimizer import (
    optimize, result_to_config, result_to_pareto_csv,
    DeploymentConstraints, EvaluatedCandidate, evaluate_candidate,
    # Wave 6 (Jun 2026): principled shape coherence via one optimization
    # per (hw, params, tokens, arch_mode) — see plan/redesign/06-principled-pin.md.
    optimize_across_contexts, MultiCtxResult, compute_pareto_frontier,
)
import copy as _copy
from shadow_prices import compute_shadow_prices, shadow_prices_to_json
from justification import generate_justification
from quality_model import (
    ArchConfig as QualArch,
    TrainingConfig,
    quality as quality_fn,
)


# =========================================================================
# Grid definition
# =========================================================================

HARDWARE = ["h100", "b200", "tpu_v5p", "trainium2", "trainium3"]
PARAM_TARGETS = [1.0, 3.0, 7.0, 13.0, 120.0, 500.0, 750.0, 1000.0]
TOKEN_COUNTS = [0.5, 2.0, 10.0]  # trillions
SERVING_MODES = [
    # v1-fix Wave 2a Step 2a.3 (Jun 2026 redesign): the three categorical
    # regimes (unconstrained / 50ms / 20ms) are collapsed into one
    # continuous run. Serving cost is now reported via tbt_ms / ttft_ms /
    # hbm_spill_gb on each Family entry; the user (or downstream CLI/web
    # consumer) reads off the loss-vs-serving knee directly from
    # families[], no presets needed.
    #
    # We still iterate over a small batch-search range so the chosen
    # architecture is evaluated at a representative batch (batch affects
    # decode latency and KV-cache memory). The grid driver below now uses
    # only this single mode.
    {
        "name": "continuous", "tbt": None, "ttft": None,
        "search_batch": 16, "batches": [32, 16, 8, 4, 2, 1],
    },
]


def _pick_serving_batch(ev_optimal, hw_name, base_constraints, smode):
    """Re-evaluate the chosen architecture at each candidate batch size and
    return the EvaluatedCandidate at the LARGEST batch where:

      - serving_tbt_ms ≤ smode["tbt"] (if a TBT budget is set), AND
      - the per-GPU memory still fits (cap at 90% of HBM as a soft headroom
        for activations / pinned buffers — same threshold the optimizer
        uses internally).

    For "unconstrained" (no TBT budget), pick the largest batch that fits
    memory. This makes the serving constraint behave like a bound rather
    than an objective: the picker stops shrinking batch once the budget
    is comfortably met.

    Returns the original ev_optimal with `_chosen_batch` attached if no
    smaller-batch re-evaluation works. Falls back gracefully on any
    exception.
    """
    batches = list(smode.get("batches", [int(base_constraints.serving_batch)]))
    # Largest-first; pick the first batch that satisfies both bounds.
    batches = sorted(set(int(b) for b in batches), reverse=True)
    tbt_budget = smode.get("tbt")
    # Use 90% of HBM as the soft memory ceiling — matches the headroom the
    # CLI keeps for activations. Per-GPU HBM is exposed via the hardware
    # spec; if we can't pull it cleanly, fall back to "no memory bound here"
    # and let the optimizer's own constraint check do the work.
    try:
        from throughput_model import load_hardware
        hw_spec = load_hardware(hw_name)
        hbm_ceiling_gb = 0.90 * float(hw_spec.hbm_capacity_gb)
    except Exception:
        hbm_ceiling_gb = float("inf")

    best = None
    for b in batches:
        try:
            c = _copy.copy(base_constraints)
            c.serving_batch = int(b)
            ev = evaluate_candidate(ev_optimal.arch, hw_name, c)
        except Exception:
            continue
        if ev.memory_per_gpu_gb > hbm_ceiling_gb:
            continue
        if tbt_budget is not None and ev.serving_tbt_ms > float(tbt_budget):
            continue
        best = ev
        break
    if best is None:
        # No batch in the search list works — keep the optimizer's choice
        # rather than emitting nothing.
        ev_optimal._chosen_batch = int(base_constraints.serving_batch)
        return ev_optimal
    best._chosen_batch = int(c.serving_batch)
    return best


# v1-fix UI follow-up: context length is now a grid axis.
# Frontier model serving sits at 32k–4M today. Anything below 32k hides the
# attention KV-cache pain that motivates hybrid state mechanisms in the first
# place, so the multi-context grid starts at 32k and walks up through 128k,
# 1M, 2M, 4M (Gemini 2.5 / Llama-4 / Kimi territory).
#
# Note: this multiplies every other grid axis. For a full sweep
# (3 hw × 8 params × 3 tokens × 3 serving × 9 arch_modes × 5 contexts) =
# 9720 search calls — too slow for an inline run. The runner script
# subsets PARAM_TARGETS / TOKEN_COUNTS / SERVING_MODES via env vars (or
# falls back to a 7B/120B-only quick sweep) when invoked with
# `--ctx-sweep`. The default `generate()` still runs at the historical
# baseline (8192) for backward-compat with older inline data.
CONTEXT_LENGTHS = [32768, 131072, 1048576, 2097152, 4194304]
CONTEXT_LABELS = {
    8192:    "8k",
    32768:   "32k",
    131072:  "128k",
    1048576: "1M",
    2097152: "2M",
    4194304: "4M",
}

#  TP_MAP — tensor-parallel degree per (hardware, parameter target).
#  Multi-node serving is allowed for larger models. Node count for NVIDIA
#  HGX boxes is TP / 8 (8 GPUs per node, NVLink intra-node, InfiniBand
#  cross-node). TPU v5p slices have 4 chips per host but bigger MXU groups;
#  for the purposes of this map we treat 8 as "one TPU pod block".
#
#    TP=8   → 1 node (intra-NVLink only)
#    TP=16  → 2 nodes (cross-IB tensor parallelism)
#    TP=32  → 4 nodes
#    TP=64  → 8 nodes (whole reference cluster)
#
#  For ≥500B we expand to TP=32 / TP=64 so weights actually fit on H100
#  (80 GB HBM) — 750B at TP=16 needs 87 GiB/GPU just for BF16 weights, which
#  the v1 audit caught as a hard infeasibility. B200 (192 GB) and Trainium-3
#  (192 GB) can still serve 750B at TP=16.
#  v1-fix audit (June 2026): for the demo's 1B/7B headline cells we hold TP
#  fixed at 1 across both sizes so the per-user TTFT/TBT/mem numbers are
#  apples-to-apples. Previously 1B ran at TP=1 (single GPU) and 7B at TP=8
#  (whole HGX node), and every reported metric was per-GPU — which made
#  7B *appear* faster and smaller than 1B because it had 8× the hardware
#  amortising the work. The TP scale-out tradeoff is a separate axis;
#  for the demo it must not silently invert the size-vs-latency story.
#  v1-fix demo-audit consistency pass (June 2026): TP_MAP is now monotone
#  non-decreasing in params within each hw. The previous map had
#  ("b200", 1.0)=1, ("b200", 3.0)=4, ("b200", 7.0)=1, ("b200", 13.0)=8 —
#  the dip back to TP=1 at 7B was an intentional "apples-to-apples"
#  override for one demo cell but left the 3B and 13B cells with the
#  prior values, so the per-replica TPS appeared to *grow* with model
#  size (3B at TP=4 had 16 replicas, 1B at TP=1 had 64 replicas, etc.
#  — a per-replica vs per-GPU framing artifact). The audit also found
#  ≥500B Trainium entries missing entirely, so they fell back to the
#  global TP=8 default and produced infeasible memory.
#
#  The map below enforces non-decreasing TP as params grows. Each entry
#  is the smallest TP that keeps weight+optimizer state under the per-GPU
#  HBM cap, with the prior cell's TP as a floor.
TP_MAP = {
    # H100 (80 GB HBM): 13B fits at TP=2 (104 GB total optimizer state
    # / 2 = 52 GB); we keep TP=2 to satisfy monotonicity vs 7B@1.
    ("h100",    1.0): 1,  ("h100",    3.0): 1,  ("h100",    7.0): 1,  ("h100",   13.0): 2,
    ("h100",  120.0): 8,  ("h100",  500.0): 16, ("h100",  750.0): 32, ("h100", 1000.0): 32,
    # B200 (192 GB HBM): all sizes ≤13B fit at TP=1 (13B = 26GB BF16 +
    # ~104 GB optimizer state, well under 192 GB).
    ("b200",    1.0): 1,  ("b200",    3.0): 1,  ("b200",    7.0): 1,  ("b200",   13.0): 1,
    ("b200",  120.0): 8,  ("b200",  500.0): 16, ("b200",  750.0): 16, ("b200", 1000.0): 32,
    # TPU v5p (95 GB HBM): 13B optimizer state forces TP=2; keep
    # monotone by lifting 3B/7B to match.
    ("tpu_v5p", 1.0): 1,  ("tpu_v5p", 3.0): 1,  ("tpu_v5p", 7.0): 1,  ("tpu_v5p",  13.0): 2,
    ("tpu_v5p",120.0): 16,("tpu_v5p",500.0):16, ("tpu_v5p",750.0): 32,("tpu_v5p",1000.0):32,
    # Trainium 2 (96 GB HBM): ≤7B fits at TP=1 (14 GB BF16 + 56 GB
    # optimizer state); 13B needs TP=2. Add explicit 500B/750B/1000B
    # entries so they don't fall through to the global TP=8 default
    # while shipping memory > HBM (the audit caught this).
    ("trainium2",   1.0): 1,  ("trainium2",   3.0): 1,  ("trainium2",   7.0): 1,  ("trainium2",  13.0): 2,
    ("trainium2", 120.0): 8,  ("trainium2", 500.0): 16, ("trainium2", 750.0): 32, ("trainium2", 1000.0): 32,
    # Trainium 3 (192 GB HBM, MXFP4/6 capable): all ≤13B fit at TP=1.
    # ≥500B need increasing TP to keep optimizer state in HBM.
    ("trainium3",   1.0): 1,  ("trainium3",   3.0): 1,  ("trainium3",   7.0): 1,  ("trainium3",  13.0): 1,
    ("trainium3", 120.0): 8,  ("trainium3", 500.0): 16, ("trainium3", 750.0): 16, ("trainium3", 1000.0): 32,
}

# v1-fix demo-audit D5: pipeline-parallel degree per (hardware, params).
# The v1 demo emitted pp=1 for every cell, which forced trillion-class
# targets on H100/TPU into a single-feasible-point lattice (HBM-bound at
# TP=32, PP=1), and the singleton frontier collapsed to a 1980-layer
# transformer. Production-scale training of 500B+ models uses PP — Llama-3
# 405B used TP=8 × PP=16, DeepSeek-V3 used TP=1 × PP=16. We map ≥500B to
# PP=4-8 (still leaving DP > 1 inside a 64-GPU training cluster), and let
# the smaller targets stay at PP=1.
# v1-fix demo-audit consistency pass (June 2026): PP_MAP, like TP_MAP, is
# now monotone non-decreasing in params within each hw. Any cell missing
# from this dict gets PP=1 (the small-model default); the entries below
# only need to lift PP at thresholds where weight + optimizer state would
# otherwise exceed per-chip HBM at the corresponding TP.
PP_MAP = {
    # v1-fix demo-audit (June 2026): 120B on H100 pairs with TP=8 (intra-
    # NVLink, see TP_MAP), so add PP=2 to keep weights+optimizer state
    # under HBM cap and let the cluster's 64 GPUs split into 4 DP × 2 PP
    # × 8 TP. Without this, 120B at TP=8 PP=1 needs 30 GiB weights/GPU
    # plus 4× that for optimizer state (~120 GiB) which exceeds the 80 GB
    # cap.
    ("h100", 120.0): 2,
    ("h100",  500.0): 4, ("h100",  750.0): 8, ("h100", 1000.0): 8,
    ("b200",  500.0): 2, ("b200",  750.0): 4, ("b200", 1000.0): 4,
    # TPU v5p has only 95 GB HBM per chip (vs H100's 80 GB, but with
    # smaller weight sharding capacity at TP=16). 500B+ needs PP=8 to
    # bring per-chip residency below the 95 GB cap.
    ("tpu_v5p", 500.0): 8, ("tpu_v5p", 750.0): 8, ("tpu_v5p", 1000.0): 8,
    ("trainium2", 500.0): 4, ("trainium2", 750.0): 8, ("trainium2", 1000.0): 8,
    ("trainium3", 500.0): 2, ("trainium3", 750.0): 4, ("trainium3", 1000.0): 4,
}


def pp_for(hw: str, params_b: float) -> int:
    return PP_MAP.get((hw, params_b), 1)

# Total cluster size assumed for training (data parallel × tensor parallel).
TRAINING_CLUSTER_GPUS = 64

# GPUs per node (NVIDIA HGX = 8 H100/B200 per box).
GPUS_PER_NODE = 8

# v1-fix Wave 4 follow-up (Jun 2026): per-hw NVLink/ICI domain used for the
# TP search ladder. This is the *real* number of ranks that share full-BW
# intra-island communication, not just the per-chassis count. Used by
# context_aware_parallelism() as the TP ladder cap. Replacing the prior
# GPUS_PER_NODE cap (always 8) lets large-model cells consider TP that
# crosses NVLink islands on NVL72-class hardware where it's actually cheap.
#
# Calibration:
#   h100      — DGX-H100 / HGX-H100 island = 8 GPUs @ ~3 TB/s NVLink
#               (no NVL72 H100 shipped; cap at 8 is the honest choice)
#   b200      — DGX-B200 = 8 GPUs @ ~1.8 TB/s NVLink5, but NVL72 rack
#               extends the high-BW domain to 72 chips. We model the
#               NVL72 case because it's the relevant deployment for
#               trillion-class serving; users on plain DGX-B200 should
#               set the env var AC_NVLINK_DOMAIN_B200=8 to override.
#   tpu_v5p   — single ICI torus axis = 16 chips @ ~3.6 TB/s
#   trainium2 — NeuronLink intra-node = 16 chips
#   trainium3 — NeuronLink = 32 chips (Trn3 doubles the island)
NVLINK_DOMAIN_SIZE_SEARCH = {
    "h100":      8,
    "b200":      72,
    "tpu_v5p":   16,
    "trainium2": 16,
    "trainium3": 32,
}


def _nvlink_domain_for_search(hw: str) -> int:
    """Resolve the TP-search cap with env-var override.

    Set AC_NVLINK_DOMAIN_<HW>=N to override (e.g. AC_NVLINK_DOMAIN_B200=8
    if you're on plain DGX-B200 rather than NVL72).
    """
    import os as _os
    key = f"AC_NVLINK_DOMAIN_{hw.upper()}"
    override = _os.environ.get(key)
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return NVLINK_DOMAIN_SIZE_SEARCH.get(hw, GPUS_PER_NODE)


def nodes_for_tp(tp: int) -> int:
    """Number of physical nodes a single serving replica spans at the given TP."""
    return max(1, (tp + GPUS_PER_NODE - 1) // GPUS_PER_NODE)


def training_dp_and_ep(
    training_cluster: int,
    tp: int,
    pp: int,
    allow_moe: bool,
) -> tuple[int, int]:
    """Derive a legal training DP/EP layout from a cluster-size floor.

    EP overlays DP; it is not multiplied into the training world size.
    When TP x PP already exceeds the requested floor, DP=1 honestly reports
    the larger minimum viable world rather than inventing a fractional DP.
    """
    dp = max(1, int(training_cluster) // max(1, int(tp) * int(pp)))
    if not allow_moe:
        return dp, 1
    desired_ep = max(2, min(int(tp), 4))
    ep = next(
        (value for value in range(desired_ep, 0, -1)
         if dp % value == 0),
        1,
    )
    return dp, ep

HBM_GB = {"h100": 80, "b200": 192, "tpu_v5p": 95, "trainium2": 96, "trainium3": 192}

# v1-fix (June 2026 — long-ctx feasibility audit): the demo's long-context
# cells (1M / 2M / 4M serving) returned "infeasible" universally because
# TP/PP were context-blind. KV cache at 1M tokens for a 7B model with
# batch=16 is ~2 TB total, dwarfing any single chip's HBM. The fix:
# scale serving parallelism with context so the optimizer is handed enough
# (TP × CP) to actually fit the KV cache, and pass cp_options so it can
# choose the right CP. The training-side TP/PP from TP_MAP/PP_MAP still
# governs weight + optimizer state.
#
# Memory budgeting principle: KV cache at full context must fit in ~55%
# of aggregate HBM across (TP × CP) ranks (leaves ~45% for weights at
# the chosen precision, activations, pinned buffers, and the optimizer's
# own 90% HBM ceiling that gates per-cell feasibility).

def _ref_kv_bytes_per_token(params_B: float, optimistic: bool = False) -> int:
    """Reference KV bytes/token/layer used by the parallelism planner.

    The optimizer searches kv_bits in [16, 8, 4] and may pick MLA, so the
    planner needs two reference values:
      - pessimistic (BF16, full GQA): drives cp_max so even worst-case
        candidates have an option that fits
      - optimistic (INT8, MLA-equivalent): drives cp_min so candidates
        that use efficient precision/attention aren't forced to over-provision
    Per-candidate exact bytes still drive the actual feasibility check."""
    if optimistic:
        # INT8 + smaller effective heads (MLA-ish).
        if params_B <= 1.0:  return 512    # MLA latent compresses further
        if params_B <= 7.0:  return 1024
        if params_B <= 13.0: return 1024
        if params_B <= 120.0: return 1536
        return 2048
    # Pessimistic: BF16, full GQA.
    if params_B <= 1.0:  return 2048   # n_kv=8,  d_head=64
    if params_B <= 7.0:  return 4096   # n_kv=8,  d_head=128
    if params_B <= 13.0: return 4096
    if params_B <= 120.0: return 6144  # n_kv=12, d_head=128
    return 8192                        # n_kv=16, d_head=128 (no MLA)


def _ref_n_layers(params_B: float) -> int:
    """Coarse layer count for the parallelism planner."""
    if params_B <= 1.0:  return 20
    if params_B <= 7.0:  return 32
    if params_B <= 13.0: return 40
    if params_B <= 120.0: return 80
    return 120


def context_aware_parallelism(hw: str, params_B: float, ctx: int,
                              serving_batch: int,
                              max_serving_gpus: int | None = None) -> tuple[list[int], int, list[int]]:
    """Pick (tp_options, pp, cp_options) so KV cache + weights fit at this context.

    Wave 4: TP is now a search variable (mirrors what Wave 1 did for CP).
    The function returns a *list* of plausible TP degrees instead of a single
    one. The optimizer evaluates each in turn and the Pareto frontier picks
    the TP that minimizes loss subject to the other constraints.

    Returns:
      tp_options: list of tensor-parallel degrees the optimizer may search
                  over. tp_options[0] is `base_tp` — the floor needed to fit
                  weights+optimizer state in HBM (the old single-TP value).
                  Higher entries are `base_tp*2`, `base_tp*4`, capped at the
                  NVLink island so cross-IB tensor parallelism is never
                  silently introduced.
      pp: pipeline-parallel degree (== PP_MAP base — PP is set by weights,
          not KV)
      cp_options: list of CP degrees the optimizer may search over.
                  Always includes 1, scales up to whatever the KV budget
                  needs, capped at remaining cluster-per-replica.

    Scaling rules:
      1. Start from TP_MAP[hw,params] (sized to fit training weights +
         optimizer state). Use it as the floor for TP.
      2. Estimate KV bytes at (ctx, batch). Solve for TP*CP s.t.
         per-rank KV ≤ 0.55 * HBM.
      3. Grow TP first up to the NVLink island (GPUS_PER_NODE) — intra-node
         AllReduce is ~20× faster than cross-IB.
      4. Then grow CP — sequence parallelism that splits the sequence
         axis without re-sharding weights.
      5. Cap world-per-replica at cluster size. If even the full cluster
         can't fit, the optimizer marks infeasible *honestly* (e.g. 4M
         ctx, batch=32, single node is genuinely impossible).
    """
    import math
    base_tp = TP_MAP.get((hw, params_B), 8)
    base_pp = pp_for(hw, params_B)
    hbm_gb = HBM_GB.get(hw, 80)
    # v1-fix long-ctx audit (Jun 2026): the training cluster (`TRAINING_CLUSTER_GPUS`)
    # was being used as the SERVING cluster cap too, which made cells like
    # "120B at 1M context, no serving constraint" infeasible — even though
    # physically the answer is just "use more GPUs". `max_serving_gpus`
    # lets the caller override: under no serving constraint, scale to
    # whatever the KV cache + weights require (limited only by the largest
    # contiguous interconnect we want to model).
    cluster = (int(max_serving_gpus) if max_serving_gpus is not None
               else (16 if "tpu" in hw else TRAINING_CLUSTER_GPUS))
    node_size = 4 if "tpu" in hw else GPUS_PER_NODE

    kv_per_tok_worst = _ref_kv_bytes_per_token(params_B, optimistic=False)
    kv_per_tok_best  = _ref_kv_bytes_per_token(params_B, optimistic=True)
    n_layers = _ref_n_layers(params_B)
    bs = int(serving_batch)
    kv_worst_gb = (bs * int(ctx) * kv_per_tok_worst * n_layers) / 1e9
    kv_best_gb  = (bs * int(ctx) * kv_per_tok_best  * n_layers) / 1e9
    per_rank_kv_budget_gb = 0.55 * hbm_gb

    need_tp_cp_worst = max(base_tp, int(math.ceil(kv_worst_gb / per_rank_kv_budget_gb)))
    need_tp_cp_best  = max(base_tp, int(math.ceil(kv_best_gb  / per_rank_kv_budget_gb)))
    max_per_replica = max(base_tp, cluster // max(1, base_pp))
    need_tp_cp_worst = min(need_tp_cp_worst, max_per_replica)
    need_tp_cp_best  = min(need_tp_cp_best,  max_per_replica)
    # The TP allocation uses the worst case so cp_max can reach high enough
    # for pessimistic candidates; cp_min uses the optimistic case so
    # efficient candidates aren't forced to over-provision.
    need_tp_cp = need_tp_cp_worst

    # Grow TP up to NVLink island first.
    tp = max(base_tp, min(need_tp_cp, node_size))
    # Round TP up to a power of 2 for clean sharding.
    if tp > 1:
        tp = 1 << (tp - 1).bit_length()
    tp = min(tp, max_per_replica)

    # cp_max from pessimistic (BF16) reference so worst-case candidates
    # have an option that fits. cp_min spans down to 1 (or the optimistic
    # need) so efficient candidates (MLA, INT4 KV) can pick a smaller CP
    # and avoid CP comm cost.
    cp_need_worst = max(1, int(math.ceil(need_tp_cp_worst / max(1, tp))))
    if cp_need_worst > 1:
        cp_need_worst = 1 << (cp_need_worst - 1).bit_length()
    cp_hard_cap = max(1, max_per_replica // max(1, tp))
    cp_max = min(cp_need_worst, cp_hard_cap)
    # Always offer cp=1 as well so the optimizer can pick zero CP when a
    # candidate's actual KV (with its chosen precision/attention) fits at
    # plain TP. Power-of-2 ladder from 1 to cp_max.
    cp_options = []
    c = 1
    while c <= cp_max:
        cp_options.append(c)
        c *= 2
    if not cp_options:
        cp_options = [1]

    # Wave 4: build tp_options. Start from `tp` (the context-aware floor —
    # i.e. base_tp possibly bumped up so KV fits when CP=1) and add 2× / 4×
    # entries when they're (a) still inside the actual NVLink/ICI domain
    # for this hardware so we never silently introduce cross-IB tensor
    # parallelism, and (b) leave room in the per-replica budget for at
    # least CP=1.
    #
    # Wave 4 follow-up (Jun 2026): swap the cap from per-chassis
    # `node_size` to per-hw NVLink domain (`_nvlink_domain_for_search`).
    # Plain DGX-H100 keeps node_size=8 (no change), but B200 NVL72 now
    # gets cap=72 and TPU v5p gets cap=16 (single axis), letting the
    # optimizer actually explore TP at scale where the fabric supports it.
    tp_options: list[int] = [tp]
    cp_floor = 1
    nvlink_cap = _nvlink_domain_for_search(hw)
    max_tp_for_search = max(tp, min(nvlink_cap, max_per_replica // max(1, cp_floor)))
    next_tp = tp * 2
    while next_tp <= max_tp_for_search and len(tp_options) < 4:
        tp_options.append(next_tp)
        next_tp *= 2

    return tp_options, base_pp, cp_options


def sample_pareto_diverse(pareto, max_points=30):
    """Sample Pareto frontier to ensure diversity across d_model and attention.

    Strategy: keep the best point for each attention family first (so MLA is
    not erased by a same-shape GQA sibling), then group by (d_model, attention)
    and take representative points before filling the rest by throughput.
    """
    if len(pareto) <= max_points:
        return pareto

    from collections import defaultdict
    by_attention = defaultdict(list)
    by_shape_attention = defaultdict(list)
    for ev in pareto:
        attn = getattr(ev.arch, "attention_type", "full") or "full"
        by_attention[attn].append(ev)
        by_shape_attention[(ev.arch.d_model, attn)].append(ev)

    # Sort each group by loss
    for attn in by_attention:
        by_attention[attn].sort(key=lambda x: x.predicted_loss)
    for key in by_shape_attention:
        by_shape_attention[key].sort(key=lambda x: x.predicted_loss)

    selected = set()
    result = []

    # Phase 0: best from each attention family (keeps MLA visible).
    for attn in sorted(by_attention.keys()):
        ev = by_attention[attn][0]
        result.append(ev)
        selected.add(id(ev))
        if len(result) >= max_points:
            result.sort(key=lambda x: x.predicted_loss)
            return result

    # Phase 1: best from each d_model × attention family.
    for key in sorted(by_shape_attention.keys()):
        ev = by_shape_attention[key][0]
        if id(ev) in selected:
            continue
        result.append(ev)
        selected.add(id(ev))
        if len(result) >= max_points:
            result.sort(key=lambda x: x.predicted_loss)
            return result

    # Phase 2: for each d_model × attention family, also pick the
    # highest-throughput point.
    for key in sorted(by_shape_attention.keys()):
        best_tps = max(by_shape_attention[key], key=lambda x: x.training_tps)
        if id(best_tps) not in selected:
            result.append(best_tps)
            selected.add(id(best_tps))
            if len(result) >= max_points:
                result.sort(key=lambda x: x.predicted_loss)
                return result

    # Phase 3: fill remaining slots from the full frontier, spread evenly
    if len(result) < max_points:
        remaining = [ev for ev in pareto if id(ev) not in selected]
        # Sort by training TPS to get even spread across throughput axis
        remaining.sort(key=lambda x: x.training_tps)
        step = max(1, len(remaining) // (max_points - len(result)))
        for i in range(0, len(remaining), step):
            if len(result) >= max_points:
                break
            result.append(remaining[i])

    # Sort final result by loss for consistent display
    result.sort(key=lambda x: x.predicted_loss)
    return result


_PUBLIC_SENTINEL_LOSS_MULT = 10.0


def _is_public_quality_sentinel(record: dict) -> bool:
    """True when a serialized web record is outside AC's covered quality range.

    The optimizer has an internal `allow_quality_sentinel` escape hatch for
    matrix/debug cells, but the public web payload should not let those
    million-loss rows participate in frontiers or family winners. This helper
    works on serialized records so old payloads can be cleaned by
    `run_post_chain()` without rerunning searches.
    """
    if not isinstance(record, dict):
        return False
    if record.get("quality_sentinel") or record.get("coverage_status") == "outside_quality_model_coverage":
        return True
    try:
        loss = float(record.get("loss") or record.get("predicted_loss") or 0.0)
    except (TypeError, ValueError):
        return False
    if loss > 1e4:
        return True
    try:
        penalty_pct = float(record.get("penalty_pct") or record.get("total_residual_pct") or 0.0)
        if penalty_pct > 1e5:
            return True
    except (TypeError, ValueError):
        pass
    try:
        base = float(record.get("chinchilla") or record.get("spine_loss") or 0.0)
        if base > 0 and loss > _PUBLIC_SENTINEL_LOSS_MULT * base:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _prune_public_quality_sentinels(data: dict) -> None:
    """Drop sentinel-tainted candidates from public grid alternatives."""
    stats = {"optimal": 0, "pareto": 0, "pareto_4d": 0}
    for row in data.get("grid", []):
        removed_public_candidate = False
        opt = row.get("optimal")
        if _is_public_quality_sentinel(opt):
            row["omitted_optimal"] = {
                "reason": "outside_quality_model_coverage",
                "loss": opt.get("loss") if isinstance(opt, dict) else None,
            }
            row["optimal"] = None
            stats["optimal"] += 1
            removed_public_candidate = True
        for key in ("pareto", "pareto_4d"):
            values = row.get(key)
            if not isinstance(values, list):
                continue
            kept = [v for v in values if not _is_public_quality_sentinel(v)]
            stats[key] += len(values) - len(kept)
            removed_public_candidate |= len(kept) != len(values)
            row[key] = kept
        row["pareto_size"] = len(row.get("pareto") or [])
        if row.get("optimal") is None and not row.get("pareto"):
            # `feasible` is a public/browser statistic after this transform.
            # Do not advertise physically evaluable but quality-uncovered
            # candidates after every displayable record has been removed.
            # This also repairs already-pruned resumable shards, where the
            # current pass sees omitted_optimal rather than the old record.
            row["feasible"] = 0
            reasons = list(row.get("infeasible_reasons") or [])
            omitted = row.get("omitted_optimal") or {}
            outside_quality = (
                removed_public_candidate
                or omitted.get("reason") == "outside_quality_model_coverage"
            )
            reason = (
                "All candidates fall outside calibrated quality-model coverage"
                if outside_quality else
                "No joint architecture meets the fixed training-world and "
                "quality constraints across all contexts"
            )
            if reason not in reasons:
                reasons.append(reason)
            row["infeasible_reasons"] = reasons
    data["_public_quality_filter"] = stats


def _arch_family(ev: EvaluatedCandidate) -> str:
    """Classify candidate into dense / moe / hybrid / moe_hybrid.

    Wave 18a: routes through the canonical ArchitectureSignature so this
    grid-driver family label matches the optimizer's stratification and
    the decision diagnostics.  The fallback path preserves the pre-18a
    behavior for exotic architectures that lack the minimal shape fields
    architecture_signature requires.
    """
    try:
        from ac.architecture import architecture_signature
        return architecture_signature(ev.arch).legacy_family
    except Exception:
        c = ev.arch
        has_moe = c.moe is not None
        has_state = c.state_config is not None and c.n_state_layers > 0
        if has_moe and has_state:
            return "moe_hybrid"
        elif has_moe:
            return "moe"
        elif has_state:
            return "hybrid"
        return "dense"


def serialize_candidate(ev: EvaluatedCandidate) -> dict:
    c = ev.arch
    terms = getattr(ev.quality, "terms", {})
    arch_term = terms.get("architecture_residual")
    precision_term = terms.get("precision_residual")
    risk_term = terms.get("risk_residual")
    moe_term = terms.get("moe_residual")
    state_term = terms.get("state_residual")
    quality_terms = {
        k: {
            "value": round(v.value, 5),
            "uncertainty": round(v.uncertainty, 5),
            "confidence": v.confidence,
            "source": v.source,
            "notes": v.notes,
            "features": v.features,
        }
        for k, v in terms.items()
        if v.confidence != "not_applicable" or abs(v.value) > 0 or v.uncertainty > 0
    }
    # v1-fix demo-audit D4: when attention is MLA the GQA n_kv_heads is
    # meaningless (KV cache is a compressed latent, not per-head). Surface
    # 0 in the schema instead of letting whatever value the enumerator
    # happened to be iterating over leak through as "MHA with n_kv_heads
    # == n_heads".
    _nkv_for_schema = 0 if c.attention_type == "mla" else c.n_kv_heads
    # Wave 18a: emit the factorized ArchitectureSignature alongside the
    # legacy `arch_family` label. Consumers should prefer `signature` for
    # comparisons (its 6 axes decouple FFN sparsity, KV projection,
    # attention pattern, sequence mixer, context extension, and modifiers);
    # `arch_family` is kept as a compat display label for the transition.
    try:
        from ac.architecture import architecture_signature
        _sig_dict = architecture_signature(c).as_dict()
    except (ValueError, ImportError):
        _sig_dict = None
    d = {
        "d_model": c.d_model, "n_layers": c.n_layers,
        "n_heads": c.n_heads, "d_head": c.d_head,
        "n_kv_heads": _nkv_for_schema, "ffn_dim": c.ffn_dim,
        "weight_prec": c.weight_precision, "ffn_prec": c.ffn_precision,
        "kv_bits": c.kv_cache_bits, "params_B": c.total_params_b,
        "active_params_B": round(c.active_params_b or c.total_params_b, 3),
        "arch_family": _arch_family(ev),
        "signature": _sig_dict,
        "moe_style": c.moe_style,
        # v1-fix demo-audit (June 2026 follow-up): the legacy schema emitted
        # `spine_loss == chinchilla` because both were aliased to the same
        # Chinchilla L(N, D) value. The UI then displayed a "Scaling Spine"
        # row that was indistinguishable from the Chinchilla baseline,
        # implying two independent measurements where there was only one.
        # Redefine `spine_loss` to mean "loss attributable to the architecture
        # shape only" — i.e., predicted loss with precision / quantization
        # / risk taxes stripped out. That is the scaling-law + shape-residual
        # floor for *this* shape at BF16 weights and BF16 KV. It satisfies
        # chinchilla ≤ spine_loss ≤ loss and makes `loss - spine_loss` the
        # precision-and-risk tax for the candidate.
        "loss": round(ev.predicted_loss, 4),
        "spine_loss": round(
            ev.quality.chinchilla_baseline
            + (arch_term.value if arch_term else 0.0),
            4,
        ),
        "chinchilla": round(ev.quality.chinchilla_baseline, 4),
        "spine_active_params_B": round(getattr(ev.quality, "spine_active_params", 0) / 1e9, 3),
        "total_residual_pct": round(ev.quality.total_penalty_fraction * 100, 2),
        "architecture_residual_pct": round((arch_term.value if arch_term else 0.0) * 100, 3),
        "precision_residual_pct": round((precision_term.value if precision_term else 0.0) * 100, 3),
        "risk_uncertainty_pct": round((risk_term.uncertainty if risk_term else 0.0) * 100, 3),
        "moe_residual_pct": round((moe_term.value if moe_term else 0.0) * 100, 3),
        "state_residual_pct": round((state_term.value if state_term else 0.0) * 100, 3),
        "penalty_pct": round(ev.quality.total_penalty_fraction * 100, 2),
        "dominant": ev.quality.dominant_penalty,
        "confidence": ev.quality.confidence,
        "uncertainty_low_pct": round(ev.quality.uncertainty_low_pct, 2),
        "uncertainty_high_pct": round(ev.quality.uncertainty_high_pct, 2),
        "uncertainty_total_pct": round(getattr(ev.quality, "uncertainty_total", 0.0) * 100, 2),
        "uncertainty_breakdown": {
            k: round(v * 100, 3)
            for k, v in getattr(ev.quality, "uncertainty_breakdown", {}).items()
        },
        "quality_model_version": getattr(ev.quality, "quality_model_version", "quality_v0"),
        "pretraining_loss": round(
            getattr(ev.quality, "pretraining_loss_proxy", ev.predicted_loss), 6
        ),
        "task_adjusted_loss": round(
            getattr(ev.quality, "task_adjusted_loss_proxy", ev.predicted_loss), 6
        ),
        "effective_params_B": round(
            getattr(ev.quality, "spine_effective_params", 0) / 1e9, 6
        ),
        "effective_training_tokens_T": round(
            getattr(ev.quality, "training_tokens", 0) / 1e12, 6
        ),
        "effective_capacity_delta": round(
            getattr(terms.get("effective_capacity"), "delta", 0.0), 8
        ),
        "effective_data_delta": round(
            getattr(terms.get("effective_data"), "delta", 0.0), 8
        ),
        "train_tps": round(ev.training_tps),
        "tbt_ms": round(ev.serving_tbt_ms, 1),
        "ttft_ms": round(ev.throughput.prefill_time_ms, 1),
        "mem_gb": round(ev.memory_per_gpu_gb, 1),
        # v1-fix Wave 2a Step 2a.2/2a.4 (Jun 2026): HBM-spill continuous
        # cost is now part of every Pareto point's record. spill_tier
        # values: "fits" | "nvlink" | "pcie" | "mixed".
        "hbm_spill_gb": round(getattr(ev.throughput, "hbm_spill_gb", 0.0), 2),
        "spill_tier": getattr(ev.throughput, "spill_tier", "fits"),
        "tbt_ms_no_spill": round(getattr(ev.throughput, "tbt_ms_no_spill", ev.serving_tbt_ms), 1),
        "ttft_ms_no_spill": round(getattr(ev.throughput, "ttft_ms_no_spill", ev.throughput.prefill_time_ms), 1),
        # v1-fix Wave 1 Step 1.2: DP gradient sync (ms).
        "dp_grad_allreduce_ms": round(getattr(ev.throughput, "dp_grad_allreduce_s", 0.0) * 1000, 2),
        "regime": ev.binding_serving_regime,
        "regime_reason": ev.binding_reason,
        "penalties": {
            k: {"val": round(v.value, 5), "src": v.source}
            for k, v in ev.quality.penalty_breakdown.items()
            if v.value > 0 and v.value < 1e5
        },
        "quality_terms": quality_terms,
    }
    sentinel_guard = None
    try:
        sentinel_guard = (ev.feasibility.guards or {}).get(
            "quality_sentinel_tripped"
        )
    except AttributeError:
        sentinel_guard = None
    if sentinel_guard is not None and sentinel_guard.triggered:
        d["quality_sentinel"] = True
        d["coverage_status"] = "outside_quality_model_coverage"
        d["_feasible"] = False
        d["infeasible_reason"] = sentinel_guard.message
    # EP is explicit on every record. A dense fallback inside an allow-MoE
    # search has EP=1; leaving the field absent let the web annotator inherit
    # the enclosing mode's EP and fabricate dense EP>DP topologies.
    d["ep"] = int(c.ep_degree) if c.moe is not None else 1

    # MoE fields
    if c.moe is not None:
        d["n_experts"] = c.moe.get("n_experts", 0)
        d["top_k"] = c.moe.get("top_k", 0)
        d["expert_dim"] = c.moe.get("expert_dim", 0)
        d["shared_expert"] = c.moe.get("shared_expert") is not None
        # v1-fix UI: first-K-dense (Part B) — number of leading dense FFN layers
        # in the MoE stack and the per-layer stability bonus that drove it.
        d["n_dense_ffn_layers"] = int(getattr(c, "n_dense_ffn_layers", 0) or 0)
        if moe_term and moe_term.features:
            sub = (moe_term.features.get("subterms") or {})
            d["dense_prefix_bonus_pct"] = round(float(sub.get("dense_prefix_bonus", 0.0)) * 100, 3)
            d["moe_layer_fraction"] = round(float(sub.get("moe_layer_fraction", 1.0)), 3)
    # v1-fix RoPE scaling: surface method/factor whenever scaling is active.
    if getattr(c, "rope_scaling_method", "none") != "none":
        d["rope_scaling_method"] = str(c.rope_scaling_method)
        d["rope_scaling_factor"] = float(c.rope_scaling_factor)
        d["rope_original_max_position"] = int(c.rope_original_max_position)

    # Parallelism is serialized on every record.  Downstream unit annotation
    # must not have to guess that a missing CP means one: that exact fallback
    # previously priced CP=16 candidates as one-GPU replicas.
    d["cp"] = max(1, int(getattr(c, "cp_degree", 1) or 1))
    d["pp"] = max(1, int(getattr(c, "pp_degree", 1) or 1))
    d["dp"] = max(1, int(getattr(c, "dp_degree", 1) or 1))
    if d["cp"] > 1:
        d["cp_degree"] = d["cp"]
        d["cp_method"] = str(c.cp_method)

    # Wave 4: surface the per-candidate tp_degree on every serialized record
    # so consumers (web app, CLI, consistency passes) can read cell.optimal.tp
    # directly instead of falling back to the cell-level base TP. The field
    # is always emitted (rather than gated on `tp_degree > 1`) because every
    # candidate has a meaningful tp and the legacy cell.tp may not match
    # when the optimizer picks a higher TP from tp_options.
    d["tp"] = int(getattr(c, "tp_degree", 1) or 1)

    # v1-fix NSA: surface Native Sparse Attention block parameters.
    if getattr(c, "attention_type", "full") == "nsa":
        d["attention_type"] = "nsa"
        d["nsa_compress_block_size"] = int(getattr(c, "nsa_compress_block_size", 64) or 64)
        d["nsa_compress_block_stride"] = int(getattr(c, "nsa_compress_block_stride", 16) or 16)
        d["nsa_select_block_size"] = int(getattr(c, "nsa_select_block_size", 64) or 64)
        d["nsa_select_top_k"] = int(getattr(c, "nsa_select_top_k", 16) or 16)
        d["nsa_window_size"] = int(getattr(c, "nsa_window_size", 512) or 512)

    # v1-fix YOCO: surface cross-layer KV sharing block.
    if getattr(c, "yoco_n_self_attn_layers", 0) > 0:
        d["yoco_n_self_attn_layers"] = int(c.yoco_n_self_attn_layers)
        d["yoco_share_fraction"] = round(
            (c.n_layers - c.yoco_n_self_attn_layers) / max(1, c.n_layers), 3
        )

    # v1-fix 2:4 sparsity: surface per-component flags when set.
    sparsity = getattr(c, "sparsity_2_4", None) or {}
    if any(sparsity.values()):
        d["sparsity_2_4"] = {k: bool(v) for k, v in sparsity.items() if v}

    # v1-fix microscaling: surface MX precision label when chosen.
    if c.weight_precision in ("mxfp4", "mxfp6") or c.ffn_precision in ("mxfp4", "mxfp6"):
        d["uses_mx_precision"] = True
        d["mx_precision"] = c.ffn_precision if c.ffn_precision in ("mxfp4", "mxfp6") else c.weight_precision

    # v1-fix MTP: surface MTP fields when the candidate has prediction depths.
    if getattr(c, "mtp_n_predict_depths", 0) > 0:
        d["mtp_n_predict_depths"] = int(c.mtp_n_predict_depths)
        d["mtp_depth_n_layers"] = int(c.mtp_depth_n_layers)
        d["mtp_train_loss_weight"] = float(c.mtp_train_loss_weight)
        mtp_term = terms.get("mtp_residual")
        if mtp_term:
            d["mtp_residual_pct"] = round(mtp_term.value * 100, 3)

    # v1-fix MLA: surface MLA fields whenever the candidate's attention is MLA.
    # The UI uses `attention_type` to relabel the attention chip ("MLA" vs
    # "GQA-N"), and the latent dims drive the architecture diagram's compression
    # sub-block + the KV-cache savings callout.
    if getattr(c, "attention_type", "full") == "mla":
        d["attention_type"] = "mla"
        d["mla_kv_latent_dim"] = int(c.mla_kv_latent_dim)
        d["mla_q_latent_dim"] = int(c.mla_q_latent_dim)
        d["mla_rope_head_dim"] = int(c.mla_rope_head_dim)
        d["mla_nope_head_dim"] = int(c.mla_nope_head_dim)
        # Per-token per-layer KV bytes — used by the diagram's compression chip
        d["mla_kv_bytes_per_token_per_layer"] = int(
            (c.mla_kv_latent_dim + c.mla_rope_head_dim) * 2
        )
        # Reduction factor vs the equivalent MHA shape (for the "MLA cuts KV by Nx" chip)
        mha_bytes = max(1, 2 * c.n_kv_heads * c.d_head * 2)
        d["mla_kv_reduction_vs_mha"] = round(mha_bytes / max(1, d["mla_kv_bytes_per_token_per_layer"]), 1)
    else:
        # Wave 32 fix: this used to be a bare `d["attention_type"] = "full"`,
        # which mislabeled EVERY non-MLA attention variant as "full" — it
        # even overwrote the "nsa" assignment made above, and stamped
        # csa/indexshare/msa candidates as dense full attention. That
        # made compressed-attention winners invisible in the web payload
        # (the search picked them; the serializer renamed them). Carry the
        # candidate's actual attention_type and its per-family config.
        _at = str(getattr(c, "attention_type", "full") or "full")
        d["attention_type"] = _at
        if _at == "csa":
            d["csa_block_size"] = int(getattr(c, "csa_block_size", 64) or 64)
            d["csa_top_k_blocks"] = int(getattr(c, "csa_top_k_blocks", 16) or 16)
            d["csa_compression_dim"] = int(getattr(c, "csa_compression_dim", 0) or 0)
        elif _at == "indexshare":
            d["indexshare_num_buckets"] = int(getattr(c, "indexshare_num_buckets", 64) or 64)
            d["indexshare_top_k_buckets"] = int(getattr(c, "indexshare_top_k_buckets", 4) or 4)
            d["indexshare_index_dim"] = int(getattr(c, "indexshare_index_dim", 0) or 0)
        elif _at == "msa":
            d["msa_window_size"] = int(getattr(c, "msa_window_size", 512) or 512)
            d["msa_dilated_top_k"] = int(getattr(c, "msa_dilated_top_k", 64) or 64)
            d["msa_global_top_k"] = int(getattr(c, "msa_global_top_k", 16) or 16)

    # State/hybrid fields
    if c.state_config is not None and c.n_state_layers > 0:
        d["hybrid_ratio"] = c.hybrid_ratio
        d["n_attention_layers"] = c.n_attention_layers
        d["n_state_layers"] = c.n_state_layers
        d["d_state"] = c.state_config.get("d_state", 0)
        d["placement_strategy"] = c.placement_strategy
        d["crossover_seq_len"] = round(c.crossover_seq_len, 1)
        # v1-fix UI (Part J): expose the actual SSM/linear-attention family
        # (mamba2, gla, kda, gated_delta, sliding_window, rwkv7, ...) so the
        # interactive compiler can render the right architecture diagram and
        # the right family chip on the optimal card.
        d["state_type"] = str(c.state_config.get("state_type", "mamba2"))
        # State quality features
        if state_term and state_term.features:
            sf = state_term.features
            d["hybrid_family"] = sf.get("hybrid_family", "")
            d["p_attn"] = sf.get("p_attn", 0)
            d["in_band"] = sf.get("in_band", False)
    return d


def annotate_parallelism_units(
    rec: dict,
    *,
    pp: int,
    dp: int,
    fallback_tp: int = 1,
    fallback_ep: int = 1,
    chosen_batch: int | None = None,
) -> dict:
    """Attach one canonical serving/training parallelism unit ledger.

    Training EP overlays DP, so a training replica spans TP x PP x CP and
    the training world spans that replica x DP.  A serving instance also
    spans EP because its experts are sharded across the serving ranks.
    """
    tp = max(1, int(rec.get("tp", fallback_tp) or fallback_tp))
    pp = max(1, int(rec.get("pp", pp) or pp))
    dp = max(1, int(rec.get("dp", dp) or dp))
    ep = max(1, int(rec.get("ep", fallback_ep) or fallback_ep))
    cp = max(
        1,
        int(rec.get("cp", rec.get("cp_degree", 1)) or 1),
    )
    if ep > dp or dp % ep != 0:
        raise ValueError(
            f"invalid generated topology: EP={ep} must divide DP={dp}"
        )

    training_replica_gpus = tp * pp * cp
    serving_instance_gpus = training_replica_gpus * ep
    training_cluster_gpus = training_replica_gpus * dp
    per_replica_tps = float(rec.get("train_tps", 0.0))

    rec.update({
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "ep": ep,
        "cp": cp,
        "training_replica_gpus": training_replica_gpus,
        "serving_instance_gpus": serving_instance_gpus,
        "training_cluster_gpus": training_cluster_gpus,
        "mem_per_replica_gb": round(
            float(rec.get("mem_gb", 0.0)) * serving_instance_gpus, 2
        ),
        "train_tps_per_replica": round(per_replica_tps),
        "train_tps_per_gpu": round(
            per_replica_tps / training_replica_gpus
        ),
        "train_tps_aggregate": round(per_replica_tps * dp),
        # `train_tps` is the historical per-training-replica value.  Keep
        # its unit honest while exposing the unambiguous normalized fields
        # beside it for web/report consumers.
        "train_tps_unit": "tokens/sec/training-replica",
    })
    if chosen_batch is not None:
        rec["serving_batch"] = max(1, int(chosen_batch))
    serving_batch = max(1, int(rec.get("serving_batch", 1) or 1))
    tbt = float(rec.get("tbt_ms", 0.0))
    if tbt > 0:
        rec["decode_tps_per_replica"] = round(
            1000.0 * serving_batch / tbt, 1
        )
    rec["train_tps_summary"] = (
        f"per_replica={int(round(per_replica_tps))} tok/s "
        f"(training replica = TP{tp}xPP{pp}xCP{cp} = "
        f"{training_replica_gpus} GPUs; EP={ep} overlays DP); "
        f"per_gpu={int(round(per_replica_tps / training_replica_gpus))} "
        f"tok/s; cluster_aggregate(dp={dp}, GPUs={training_cluster_gpus})="
        f"{int(round(per_replica_tps * dp))} tok/s; serving instance="
        f"{serving_instance_gpus} GPUs"
    )
    rec["ttft_semantics"] = "per_request_prefill_single_user"
    return rec


def _quality_payload(candidate: dict, hw: str, tokens_T: float) -> dict:
    component_precs = {}
    weight_prec = candidate.get("weight_prec", "bf16")
    ffn_prec = candidate.get("ffn_prec", weight_prec)
    if ffn_prec != weight_prec:
        for comp in ("ffn_up", "ffn_down", "ffn_gate"):
            component_precs[comp] = ffn_prec

    arch = QualArch(
        d_model=int(candidate["d_model"]),
        n_layers=int(candidate["n_layers"]),
        n_heads=int(candidate["n_heads"]),
        d_head=int(candidate["d_head"]),
        n_kv_heads=int(candidate["n_kv_heads"]),
        ffn_dim=int(candidate["ffn_dim"]),
        vocab_size=int(candidate.get("vocab_size", 32000)),
        weight_precision=weight_prec,
        component_precisions=component_precs if component_precs else None,
    )
    q = quality_fn(
        arch,
        TrainingConfig(
            training_tokens=int(tokens_T * 1e12),
            hardware=hw,
            kv_quantization_bits=int(candidate.get("kv_bits", 16)),
        ),
        memory_fits=(float(candidate.get("mem_gb", 0.0)) < HBM_GB.get(hw, 80)),
        lattice_aligned=True,
    )
    terms = getattr(q, "terms", {})
    arch_term = terms.get("architecture_residual")
    precision_term = terms.get("precision_residual")
    risk_term = terms.get("risk_residual")
    return {
        "loss": round(q.predicted_loss, 4),
        "spine_loss": round(q.chinchilla_baseline, 4),
        "chinchilla": round(q.chinchilla_baseline, 4),
        "spine_active_params_B": round(getattr(q, "spine_active_params", 0) / 1e9, 3),
        "total_residual_pct": round(q.total_penalty_fraction * 100, 2),
        "architecture_residual_pct": round((arch_term.value if arch_term else 0.0) * 100, 3),
        "precision_residual_pct": round((precision_term.value if precision_term else 0.0) * 100, 3),
        "risk_uncertainty_pct": round((risk_term.uncertainty if risk_term else 0.0) * 100, 3),
        "penalty_pct": round(q.total_penalty_fraction * 100, 2),
        "dominant": q.dominant_penalty,
        "confidence": q.confidence,
        "uncertainty_low_pct": round(q.uncertainty_low_pct, 2),
        "uncertainty_high_pct": round(q.uncertainty_high_pct, 2),
        "uncertainty_total_pct": round(getattr(q, "uncertainty_total", 0.0) * 100, 2),
        "uncertainty_breakdown": {
            k: round(v * 100, 3)
            for k, v in getattr(q, "uncertainty_breakdown", {}).items()
        },
        "quality_model_version": getattr(q, "quality_model_version", "quality_v0"),
        "pretraining_loss": round(
            getattr(q, "pretraining_loss_proxy", q.predicted_loss), 6
        ),
        "task_adjusted_loss": round(
            getattr(q, "task_adjusted_loss_proxy", q.predicted_loss), 6
        ),
        "effective_params_B": round(
            getattr(q, "spine_effective_params", 0) / 1e9, 6
        ),
        "effective_training_tokens_T": round(
            getattr(q, "training_tokens", 0) / 1e12, 6
        ),
        "effective_capacity_delta": round(
            getattr(terms.get("effective_capacity"), "delta", 0.0), 8
        ),
        "effective_data_delta": round(
            getattr(terms.get("effective_data"), "delta", 0.0), 8
        ),
        "penalties": {
            k: {"val": round(v.value, 5), "src": v.source}
            for k, v in q.penalty_breakdown.items()
            if v.value > 0 and v.value < 1e5
        },
        "quality_terms": {
            k: {
                "value": round(v.value, 5),
                "uncertainty": round(v.uncertainty, 5),
                "confidence": v.confidence,
                "source": v.source,
                "notes": v.notes,
                "features": v.features,
            }
            for k, v in terms.items()
            if v.confidence != "not_applicable" or abs(v.value) > 0 or v.uncertainty > 0
        },
    }


def refresh_quality_metadata(data: dict) -> dict:
    """Refresh quality-model fields for existing web candidates without rerunning search."""
    refreshed = 0
    for entry in data.get("grid", []):
        hw = entry.get("hw", "h100")
        tokens_T = float(entry.get("tokens_T", 2.0))
        candidates = []
        if entry.get("optimal"):
            candidates.append(entry["optimal"])
        candidates.extend(entry.get("pareto", []))
        for candidate in candidates:
            candidate.update(_quality_payload(candidate, hw, tokens_T))
            refreshed += 1
        if entry.get("pareto"):
            entry["pareto"].sort(key=lambda x: x.get("loss", 1e9))
    print(f"Refreshed quality metadata for {refreshed} web candidates")
    return data


def write_data_outputs(data: dict, base: str) -> None:
    out_path = os.path.join(base, "compiler_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes)")

    html_path = os.path.join(base, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            html = f.read()
        compact = json.dumps(data, separators=(",", ":"))
        action = "Injected"
        if "__COMPILER_DATA_PLACEHOLDER__" in html:
            html = html.replace("__COMPILER_DATA_PLACEHOLDER__", compact)
        else:
            action = "Refreshed"
            marker_start = "const DATA = "
            marker_end = "const HW_INFO"
            start = html.find(marker_start)
            end = html.find(marker_end, start)
            if start == -1 or end == -1:
                raise RuntimeError("Could not find existing DATA block in index.html")
            html = html[:start] + f"const DATA = {compact};\n" + html[end:]
        with open(html_path, "w") as f:
            f.write(html)
        print(f"{action} data in index.html ({os.path.getsize(html_path)} bytes)")


# Architecture search modes — each grid point runs dense, and optionally MoE,
# hybrid, and MoE+hybrid if the param target is large enough to benefit.
#
# v1-fix UI (Part J): the hybrid and moe_hybrid modes used to hard-code
# state_type="mamba2", so the web data only ever contained Mamba-2 hybrids.
# The mode list now includes one entry per state family that Part J unlocked.
# Each entry runs its own optimizer pass; the merged Pareto frontier in
# index.html groups them under arch_family="hybrid"/"moe_hybrid" but the
# `state_type` field on each candidate identifies which family was used.
# Wave 31 (Jul 2026): ARCH_MODES is now DERIVED as the full cross product
# of the two design axes (FFN family x state-mixer family) instead of a
# hand-curated list. The previous list excluded moe_hybrid x
# {kda, gla, sliding_window} ("keeps grid bounded"), which silently
# removed real production recipes from the grid: SWA+MoE is the
# GPT-OSS / Llama-4 recipe, KDA+MoE is the Kimi recipe. A 7B/2M
# spot-check showed SWA+MoE is min-loss across all families there
# (2.0546 vs swa 2.0594 vs moe 2.0911) and picker rank #2 — the
# exclusion was distorting headline winners, not just plateau lists.
# Subsetting for chunked regens is a CLI concern (--arch-modes /
# --state-types), not a reason to hard-trim the axis definition.

# Single source of truth for the state-mixer axis. Mamba-2 is
# measured-empirical; the rest are Part-J research-stub families.
STATE_FAMILIES = ["mamba2", "gated_delta", "kda", "gla", "sliding_window"]


def build_arch_modes(arch_mode_names=None, state_families=None):
    """Cross product of the FFN axis (dense | moe) and the state axis
    (none | each STATE_FAMILIES entry).

    arch_mode_names: optional subset of {dense, moe, hybrid, moe_hybrid}.
    state_families:  optional subset of STATE_FAMILIES (applies to the
                     hybrid and moe_hybrid modes).
    """
    names = list(arch_mode_names) if arch_mode_names else [
        "dense", "moe", "hybrid", "moe_hybrid"]
    states = list(state_families) if state_families else list(STATE_FAMILIES)
    modes = []
    if "dense" in names:
        modes.append({"name": "dense", "allow_moe": False,
                      "allow_state": False, "state_type": "mamba2"})
    if "moe" in names:
        modes.append({"name": "moe", "allow_moe": True,
                      "allow_state": False, "state_type": "mamba2"})
    for name, allow_moe in (("hybrid", False), ("moe_hybrid", True)):
        if name in names:
            for st in states:
                modes.append({"name": name, "allow_moe": allow_moe,
                              "allow_state": True, "state_type": st})
    return modes


ARCH_MODES = build_arch_modes()

# Wave 32 (Jul 2026): grid MoE axis defaults. The previous driver pinned
# n_experts=[8], top_k=[2], granularity=[1.0] — a coarse Mixtral-8x2-style
# MoE whose effective-capacity gain in AC's model is ~+10% effective
# params, far below production fine-grained MoEs (DeepSeek-V3 256x8,
# Qwen3 128x8, GPT-OSS 128x4). That made "MoE barely beats dense /
# hybrids" a driver artifact, not a model prediction. The grid now sweeps
# coarse AND fine points by default; override per run with
# --moe-n-experts / --moe-top-k / --moe-granularity.
GRID_MOE_N_EXPERTS = [8, 64]
GRID_MOE_TOP_K = [2, 8]
GRID_MOE_GRANULARITY = [1.0, 0.25]

# MoE is meaningful above ~3B active; hybrid above ~1B.
# MoE+hybrid requires both conditions.
MOE_MIN_PARAMS = 3.0
HYBRID_MIN_PARAMS = 1.0
MOE_HYBRID_MIN_PARAMS = 3.0


# =============================================================================
# Wave 6 (Jun 2026): principled shape-coherent grid runner
# =============================================================================
#
# Replaces the per-cell `optimize()` call with one `optimize_across_contexts()`
# call per (hw, params, tokens, smode, arch_mode). The shape picked at the
# reference context (128k by default) is used at every ctx in the row, so
# the displayed grid no longer suffers from shape drift across adjacent ctxs.
#
# This path is gated by `_multi_ctx_grid_enabled()` — the legacy per-cell flow
# remains the default while the new flow rolls out.

def _union_parallelism_across_ctxs(hw, params, contexts, search_batch,
                                    max_serving_gpus):
    """Compute the UNION of tp_options / cp_options across every ctx in the
    row, plus the max pp (PP is weight-bound and stable, but we take max
    defensively). Returns (tp_options, pp, cp_options, dp_for_floor).

    Reasoning: at long ctxs the planner bumps base_tp into the NVLink island
    and inflates cp_max. Using only the reference ctx's options would deny
    long-ctx candidates the parallelism they need to fit. The union lets the
    optimizer search the full space; joint feasibility prunes anything that
    doesn't actually work at the required ctxs.
    """
    tp_union: set = set()
    cp_union: set = set()
    pp_max = 1
    for ctx in contexts:
        tp_opts_c, pp_c, cp_opts_c = context_aware_parallelism(
            hw, params, ctx, search_batch, max_serving_gpus=max_serving_gpus)
        tp_union.update(int(v) for v in tp_opts_c)
        cp_union.update(int(v) for v in cp_opts_c)
        pp_max = max(pp_max, int(pp_c))
    return sorted(tp_union), int(pp_max), sorted(cp_union)


def _emit_multi_ctx_entries_for_amode(
    hw, params, tokens_T, smode, amode, contexts,
    training_cluster, max_serving_gpus, gpus_per_node_const,
    max_candidates=None, moe_axis=None, allow_compressed=False,
    local_refine_budget=None,
):
    """Run optimize_across_contexts once for this (hw, params, tokens, smode,
    arch_mode) tuple and return one grid entry per ctx. The entry's `optimal`
    field shares its (d_model, n_layers, …) across every ctx in the returned
    list — by construction, not by post-hoc patching."""
    # Skip the amode if its params gate forbids it.
    if amode["name"] == "moe_hybrid" and params < MOE_HYBRID_MIN_PARAMS:
        return []
    if amode["allow_moe"] and not amode["allow_state"] and params < MOE_MIN_PARAMS:
        return []
    if amode["allow_state"] and not amode["allow_moe"] and params < HYBRID_MIN_PARAMS:
        return []

    sb = int(smode.get("search_batch", smode.get("batch", 16)))
    tp_options, pp, cp_options = _union_parallelism_across_ctxs(
        hw, params, contexts, sb, max_serving_gpus)
    tp = int(tp_options[0])
    # EP overlays DP during training. Derive DP from the TP x PP training
    # replica first, then choose an EP divisor of that DP dimension.
    dp, ep_for_mode = training_dp_and_ep(
        training_cluster, tp, pp, amode["allow_moe"]
    )
    max_cp_for_mode = max(
        1,
        int(max_serving_gpus)
        // max(1, max(tp_options) * pp * ep_for_mode),
    )
    cp_options = [
        value for value in cp_options if int(value) <= max_cp_for_mode
    ] or [1]
    extra_kw = {}
    if amode["allow_moe"]:
        ax = moe_axis or {}
        extra_kw.update(
            allow_moe=True,
            max_total_params_b=params * 8,
            # Wave 32: sweep coarse AND fine-grained MoE (see
            # GRID_MOE_* defaults above).
            moe_n_experts_options=list(ax.get("n_experts") or GRID_MOE_N_EXPERTS),
            moe_top_k_options=list(ax.get("top_k") or GRID_MOE_TOP_K),
            ep_options=[ep_for_mode],
            moe_granularity_targets=list(ax.get("granularity") or GRID_MOE_GRANULARITY),
            dense_ffn_layer_options=[0, 1, 2, 3],
        )
    if allow_compressed:
        # Wave 32: let CSA / indexshare / MSA compete inside every mode.
        extra_kw.update(allow_csa=True, allow_indexshare=True, allow_msa=True)
    if amode["allow_state"]:
        extra_kw.update(allow_state=True,
                        state_type=amode.get("state_type", "mamba2"))
    allow_mla = True
    mla_kv_latent_opts = [512]

    # Reference ctx: prefer 128k (matches the Wave 5 pin's reference), else
    # the median ctx. Passed through to optimize_across_contexts.
    if 131072 in contexts:
        reference_ctx = 131072
    else:
        reference_ctx = sorted(contexts)[len(contexts) // 2]

    constraints = DeploymentConstraints(
        target_params_b=params,
        training_tokens=int(tokens_T * 1e12),
        # context_length on constraints is the per-row default (used by code
        # that hasn't been updated for multi-ctx). The actual evaluations
        # below override this per ctx via optimize_across_contexts.
        context_length=int(reference_ctx),
        serving_tbt_ms=smode["tbt"],
        serving_ttft_ms=smode["ttft"],
        serving_batch=sb,
        tp=tp, pp=pp, dp=dp,
        training_cluster_gpus=training_cluster,
        max_training_cluster_gpus=training_cluster,
        tp_options=list(tp_options),
        cp=1, cp_options=list(cp_options),
        vocab_size=32000,
        allow_mla=allow_mla,
        mla_kv_latent_options=mla_kv_latent_opts,
        param_tolerance=0.08,
        # Wave 31: candidate cap is CLI-controlled (--max-candidates);
        # None = optimizer default (uncapped after dedupe).
        max_candidates=max_candidates,
        **({"local_refine_budget": int(local_refine_budget)}
           if local_refine_budget is not None else {}),
        **extra_kw,
    )

    multi = optimize_across_contexts(
        hw, constraints, ctx_list=list(contexts), reference_ctx=reference_ctx,
    )

    entries = []
    for ctx in contexts:
        per_ctx_ev = multi.per_ctx_metrics.get(ctx)
        entry = {
            "hw": hw, "params_B": params,
            "tokens_T": tokens_T,
            "serving": smode["name"],
            "context_length": int(ctx),
            "context_label": CONTEXT_LABELS.get(ctx, f"{ctx}"),
            "arch_mode": amode["name"],
            "state_type": amode.get("state_type") if amode["allow_state"] else None,
            "tp": tp, "pp": pp, "dp": dp,
            "tp_options": list(tp_options),
            "nodes_per_replica": nodes_for_tp(tp),
            "training_cluster_target_gpus": training_cluster,
            "training_cluster_gpus": training_cluster,
            "training_nodes": max(
                1, math.ceil(training_cluster / gpus_per_node_const)),
            "serving_batch": int(smode.get("batch", 32)),
            "serving_tbt_budget_ms": smode["tbt"],
            "serving_ttft_budget_ms": smode["ttft"],
            # Counts are joint (per row, not per cell) under multi-ctx.
            "candidates": int(multi.candidates_enumerated_raw),
            "feasible": int(multi.candidates_feasible_per_ctx.get(ctx, 0)),
            "candidates_enumerated_raw": int(multi.candidates_enumerated_raw),
            "pareto_size": len(multi.pareto_frontier),
            "time_s": multi.search_time_sec,
            "optimal": None,
            "pareto": [],
            "justification": "",
            "shadow_prices": [],
            # Wave 6 fields.
            "multi_ctx_reference": int(multi.reference_ctx),
            "multi_ctx_shape_pinned": True,
        }

        # Sentinel gate (mirrors the per-cell flow in generate()).
        is_sentinel = False
        if per_ctx_ev is not None:
            try:
                _base = float(per_ctx_ev.quality.chinchilla_baseline)
                _loss = float(per_ctx_ev.predicted_loss)
                if _base > 0 and _loss > 10.0 * _base:
                    is_sentinel = True
            except (AttributeError, TypeError, ValueError):
                is_sentinel = True

        if per_ctx_ev is not None and not is_sentinel:
            # Re-evaluate this ctx's optimum across batches (same as legacy
            # _pick_serving_batch flow) so the displayed serving batch is the
            # largest one that satisfies TBT + memory at THIS ctx.
            cell_constraints = _copy.copy(constraints)
            cell_constraints.context_length = int(ctx)
            final_ev = _pick_serving_batch(per_ctx_ev, hw, cell_constraints, smode)
            try:
                _base = float(final_ev.quality.chinchilla_baseline)
                _loss = float(final_ev.predicted_loss)
                if _base > 0 and _loss > 10.0 * _base:
                    is_sentinel = True
            except (AttributeError, TypeError, ValueError):
                is_sentinel = True

            if not is_sentinel:
                _chosen_batch = int(getattr(final_ev, "_chosen_batch",
                                              cell_constraints.serving_batch))
                entry["serving_batch"] = _chosen_batch

                # _annotate is defined locally inside generate(); replicate
                # the few fields it sets here (we can't close over generate's
                # locals from this top-level function).
                rec = serialize_candidate(final_ev)
                annotate_parallelism_units(
                    rec,
                    pp=pp,
                    dp=dp,
                    fallback_tp=tp,
                    fallback_ep=ep_for_mode,
                    chosen_batch=_chosen_batch,
                )
                rec["multi_ctx_shape_pinned"] = True
                entry["optimal"] = rec
                entry["training_cluster_gpus"] = rec["training_cluster_gpus"]
                entry["training_nodes"] = max(
                    1,
                    math.ceil(rec["training_cluster_gpus"] / gpus_per_node_const),
                )

                # Build per-cell Pareto from this ctx's evaluations. Use the
                # per_ctx_all_evaluated list (every candidate at this ctx) and
                # restrict to feasible, then sample diverse points the same
                # way the legacy flow does.
                ctx_all = multi.per_ctx_all_evaluated.get(ctx, [])
                ctx_feasible = [e for e in ctx_all if e.meets_constraints]
                ctx_pareto = compute_pareto_frontier(ctx_feasible)
                sampled = sample_pareto_diverse(ctx_pareto, max_points=30)
                for ev in sampled:
                    try:
                        c_ev = _copy.copy(cell_constraints)
                        c_ev.serving_batch = _chosen_batch
                        ev_rebatched = evaluate_candidate(ev.arch, hw, c_ev)
                    except Exception:
                        ev_rebatched = ev
                    pr_rec = serialize_candidate(ev_rebatched)
                    annotate_parallelism_units(
                        pr_rec,
                        pp=pp,
                        dp=dp,
                        fallback_tp=tp,
                        fallback_ep=ep_for_mode,
                        chosen_batch=_chosen_batch,
                    )
                    # The candidate pool is shared across contexts, but each
                    # context samples its Pareto alternatives independently.
                    # Only the joint optimum is genuinely pinned to one
                    # architecture identity across the whole context row.
                    pr_rec["multi_ctx_shape_pinned"] = False
                    entry["pareto"].append(pr_rec)

                # 4D Pareto exposure (same trim heuristic as the legacy flow).
                import math as _m
                _opt = entry["optimal"]
                _pareto_pts = [_opt] + list(entry["pareto"])
                def _pareto_dist(p, anchor):
                    return max(
                        abs(p.get("loss", 0) - anchor.get("loss", 0))
                        / max(anchor.get("loss", 1.0), 1e-6),
                        abs(_m.log10(max(p.get("tbt_ms", 1.0), 1e-3))
                            - _m.log10(max(anchor.get("tbt_ms", 1.0), 1e-3))),
                        abs(_m.log10(max(p.get("mem_gb", 1.0), 1e-3))
                            - _m.log10(max(anchor.get("mem_gb", 1.0), 1e-3))),
                    )
                entry["pareto_4d"] = sorted(_pareto_pts,
                                            key=lambda p: _pareto_dist(p, _opt))[:32]
        if entry["optimal"] is None:
            entry["feasible"] = 0
            entry["pareto_size"] = 0
            entry["infeasible_reasons"] = [
                "No joint architecture meets the fixed training-world and "
                "quality constraints across all contexts"
            ]
        else:
            entry["pareto_size"] = len(entry["pareto"])
        entries.append(entry)
    return entries


def generate(hardware=None, param_targets=None, token_counts=None,
             arch_modes=None, contexts=None, max_candidates=None,
             moe_axis=None, allow_compressed=False,
             local_refine_budget=None):
    """Run the grid sweep.

    Wave 31 (Jul 2026): every axis is now injectable so the CLI can run
    chunked / partial regens (previously only PARAM_SUBSET / TOKEN_SUBSET
    env vars existed, and ARCH_MODES / HARDWARE / contexts were
    module-global constants). All arguments default to the canonical
    module-level lists, so a bare generate() is unchanged.

      hardware:      subset of HARDWARE
      param_targets: subset of PARAM_TARGETS
      token_counts:  subset of TOKEN_COUNTS
      arch_modes:    list of mode dicts (see build_arch_modes)
      contexts:      list of context lengths (overrides the --ctx-sweep /
                     CTX_SWEEP gate when given)
    """
    # v1-fix demo-audit (June 2026): hardware_info used to advertise a
    # fixed 5-hardware roster regardless of which hardwares the grid
    # actually included. The v1-web demo would offer the user a TPU v5p
    # / Trainium dropdown that returned no data. We now build
    # hardware_info from the actual HARDWARE list so the UI never
    # advertises hardware without a corresponding grid cell.
    # v1-fix demo-audit (June 2026): the previous schema exposed a single
    # `bf16_tflops` field whose value was AC's *effective baseline* (~½ of the
    # marketed dense Tensor-Core peak) rather than the datasheet number. A
    # researcher landing on the demo would compare 495 (H100) and 1125 (B200)
    # against the published 989 / 2250 and conclude AC was using wrong specs.
    # New schema separates the two and carries the convention note inline so
    # the value is interpretable without reading hardware_specs/*.json.
    _PEAK_FLOPS_CONVENTION = (
        "effective_baseline_bf16_tflops = roofline baseline used by AC's "
        "throughput model. It is intentionally ~50% of the marketed dense "
        "Tensor-Core peak so that, after calibration efficiency multipliers "
        "compose, end-to-end TPS / TBT predictions match published "
        "NeMo/Megatron/vLLM traces. peak_bf16_tflops is the datasheet number."
    )
    _HW_LABELS = {
        "h100": {
            "label": "NVIDIA H100 SXM", "hbm_gb": 80,
            "peak_bf16_tflops":  989, "effective_baseline_bf16_tflops":  495,
            "hbm_bandwidth_tb_s": 3.35,
            "peak_flops_convention": _PEAK_FLOPS_CONVENTION,
        },
        "b200": {
            "label": "NVIDIA B200", "hbm_gb": 192,
            "peak_bf16_tflops": 2250, "effective_baseline_bf16_tflops": 1125,
            "hbm_bandwidth_tb_s": 8.0,
            "peak_flops_convention": _PEAK_FLOPS_CONVENTION,
        },
        "tpu_v5p": {
            "label": "Google TPU v5p", "hbm_gb": 95,
            "peak_bf16_tflops":  918, "effective_baseline_bf16_tflops":  459,
            "hbm_bandwidth_tb_s": 2.76,
            "peak_flops_convention": _PEAK_FLOPS_CONVENTION,
        },
        # Trainium roster: Trn2 ships today, Trn3 numbers public-estimate
        # from the AWS roadmap.
        "trainium2": {
            "label": "AWS Trainium 2", "hbm_gb": 96,
            "peak_bf16_tflops": 1300, "effective_baseline_bf16_tflops":  650,
            "hbm_bandwidth_tb_s": 2.9,
            "peak_flops_convention": _PEAK_FLOPS_CONVENTION,
        },
        "trainium3": {
            "label": "AWS Trainium 3", "hbm_gb": 192,
            "peak_bf16_tflops": 2600, "effective_baseline_bf16_tflops": 1300,
            "hbm_bandwidth_tb_s": 5.2,
            "peak_flops_convention": _PEAK_FLOPS_CONVENTION,
        },
    }
    data = {
        "grid": [],
    }

    # Wave 31: resolve injectable axes (CLI) → env vars → module defaults.
    hw_list = [h for h in (hardware or HARDWARE) if h in HARDWARE] or list(HARDWARE)
    mode_list = list(arch_modes) if arch_modes else list(ARCH_MODES)

    data["hardware_info"] = {
        hw: _HW_LABELS[hw] for hw in hw_list if hw in _HW_LABELS}

    # v1-fix UI follow-up: context length is an outer loop. Each
    # (hw, params, tokens, serving, context, arch_mode) tuple produces one
    # entry. The legacy 8192-context grid is preserved by including 8192
    # in the loop when ctx_sweep=False.
    if contexts:
        contexts = sorted(int(c) for c in contexts)
    else:
        contexts = list(CONTEXT_LENGTHS) if _ctx_sweep_enabled() else [8192]

    # Subset overrides — env vars let a long sweep be chunked across multiple
    # invocations; CLI-passed subsets take precedence over env vars.
    #
    # Wave 42 fix: requested values are now AUTHORITATIVE. The old code
    # intersected with the canonical lists and silently fell back to the
    # FULL canonical list when the intersection was empty — so
    # `--tokens 20` (20T, not a canonical value) quietly ran the whole
    # [0.5, 2, 10]T sweep: 3x the compute, and no 20T data at the end.
    # Off-canonical values (20T, 30B, ...) are legitimate requests; only
    # non-positive values are rejected, loudly.
    _params = list(param_targets) if param_targets else _env_subset("PARAM_SUBSET", PARAM_TARGETS)
    _tokens = list(token_counts) if token_counts else _env_subset("TOKEN_SUBSET", TOKEN_COUNTS)
    for _name, _vals in (("params", _params), ("tokens", _tokens)):
        _bad = [v for v in _vals if not (isinstance(v, (int, float)) and v > 0)]
        if _bad:
            raise ValueError(f"non-positive {_name} requested: {_bad}")
    param_targets = list(dict.fromkeys(float(p) for p in _params))
    token_counts  = list(dict.fromkeys(float(t) for t in _tokens))

    # Count total runs for progress reporting (Wave 31: from the actual
    # mode list and its params gates, not a hardcoded 4-mode assumption).
    def _mode_allowed(amode, params):
        if amode["name"] == "moe_hybrid" and params < MOE_HYBRID_MIN_PARAMS:
            return False
        if amode["allow_moe"] and not amode["allow_state"] and params < MOE_MIN_PARAMS:
            return False
        if amode["allow_state"] and not amode["allow_moe"] and params < HYBRID_MIN_PARAMS:
            return False
        return True

    total = 0
    for params in param_targets:
        n_modes = sum(1 for m in mode_list if _mode_allowed(m, params))
        total += len(hw_list) * len(token_counts) * len(SERVING_MODES) * n_modes * len(contexts)

    done = 0
    t0 = time.time()

    for hw in hw_list:
        for params in param_targets:
            for tokens_T in token_counts:
                for smode in SERVING_MODES:
                    # v1-fix long-ctx audit (Jun 2026): unconstrained serving
                    # MUST be able to recruit more GPUs at long context — the
                    # demo's whole point is "more iron buys you longer context".
                    # The training cluster (64 GPUs for NVIDIA, 16 for TPU)
                    # caps the *training* world. For unconstrained serving the
                    # user's mental model is "any amount of compute", so we
                    # set the cap so high (~16k GPUs) the planner will never
                    # hit it in this demo — TP*PP*CP for our worst case
                    # (1000B at 4M ctx) is still ~1024. serving_50ms and
                    # serving_20ms stay at the training cluster so the
                    # latency-bounded numbers reflect a realistic deployment.
                    if smode.get("tbt") is None and smode.get("ttft") is None:
                        max_serving_gpus = 16384
                    else:
                        max_serving_gpus = 16 if "tpu" in hw else 64
                    training_cluster = 16 if "tpu" in hw else 64

                    # Wave 6 path: one optimize_across_contexts() per
                    # (hw, params, tokens, smode, arch_mode); shape pinned by
                    # construction. Gated by AC_GRID_MULTI_CTX=1 or
                    # --multi-ctx so the legacy per-cell flow stays the
                    # default during the rollout.
                    if _multi_ctx_grid_enabled():
                        for amode in mode_list:
                            row_entries = _emit_multi_ctx_entries_for_amode(
                                hw, params, tokens_T, smode, amode, contexts,
                                training_cluster, max_serving_gpus,
                                GPUS_PER_NODE,
                                max_candidates=max_candidates,
                                moe_axis=moe_axis,
                                allow_compressed=allow_compressed,
                                local_refine_budget=local_refine_budget,
                            )
                            for entry in row_entries:
                                done += 1
                                elapsed = time.time() - t0
                                rate = done / elapsed if elapsed > 0 else 0
                                eta = (total - done) / rate if rate > 0 else 0
                                print(f"  [{done}/{total}] {hw} {params}B "
                                      f"{tokens_T}T ctx="
                                      f"{CONTEXT_LABELS.get(entry['context_length'], entry['context_length'])} "
                                      f"{smode['name']} [{amode['name']}] (multi-ctx): "
                                      f"{'OK' if entry.get('optimal') else 'NO SOLUTION'} "
                                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                                data["grid"].append(entry)
                        continue  # skip the legacy per-ctx body below

                    for ctx in contexts:
                        # v1-fix long-ctx feasibility (June 2026): TP and CP
                        # are now context-aware. KV cache at 1M ctx scales
                        # TP into the NVLink island and CP up to fill the
                        # cluster, so the optimizer is handed enough memory
                        # to fit the workload instead of stamping every
                        # long-ctx cell "infeasible".
                        sb = int(smode.get("search_batch", smode.get("batch", 16)))
                        tp_options, pp, cp_options = context_aware_parallelism(
                            hw, params, ctx, sb,
                            max_serving_gpus=max_serving_gpus)
                        # Wave 4: `tp` is now the *floor* of the search space.
                        # Higher TP values live in tp_options and the optimizer
                        # picks whichever one minimizes loss subject to the
                        # other constraints. DP is still bounded by the
                        # *training* cluster size at the floor TP (matches the
                        # legacy training_tps headline number; if the optimizer
                        # picks a higher TP we'll relax DP downstream of
                        # `optimize`, but the cell driver's `dp` here is the
                        # cluster budget assuming base_tp).
                        tp = int(tp_options[0])
                        dp = max(1, training_cluster // max(1, tp * pp))
                        for amode in mode_list:
                            # Skip MoE/hybrid for small models where they don't help
                            if amode["name"] == "moe_hybrid" and params < MOE_HYBRID_MIN_PARAMS:
                                continue
                            if amode["allow_moe"] and not amode["allow_state"] and params < MOE_MIN_PARAMS:
                                continue
                            if amode["allow_state"] and not amode["allow_moe"] and params < HYBRID_MIN_PARAMS:
                                continue

                            mode_dp, mode_ep = training_dp_and_ep(
                                training_cluster, tp, pp,
                                amode["allow_moe"],
                            )
                            max_cp_for_mode = max(
                                1,
                                int(max_serving_gpus)
                                // max(1, max(tp_options) * pp * mode_ep),
                            )
                            mode_cp_options = [
                                value for value in cp_options
                                if int(value) <= max_cp_for_mode
                            ] or [1]
                            extra_kw = {}
                            if amode["allow_moe"]:
                                extra_kw.update(
                                    allow_moe=True,
                                    max_total_params_b=params * 8,
                                    # Wave 32: same MoE axis as multi-ctx.
                                    moe_n_experts_options=list(GRID_MOE_N_EXPERTS),
                                    moe_top_k_options=list(GRID_MOE_TOP_K),
                                    ep_options=[mode_ep],
                                    moe_granularity_targets=list(GRID_MOE_GRANULARITY),
                                    # v1-fix UI (Part B): exercise the first-K-dense
                                    # prefix sweep so the web data includes DeepSeek-V3
                                    # / Qwen3-MoE-style mixed-FFN stacks.
                                    dense_ffn_layer_options=[0, 1, 2, 3],
                                )
                            if amode["allow_state"]:
                                # v1-fix UI (Part J): use the per-mode state_type
                                # so the data includes a mix of Mamba-2 / Gated
                                # DeltaNet / KDA / GLA / Sliding-Window hybrids.
                                extra_kw.update(
                                    allow_state=True,
                                    state_type=amode.get("state_type", "mamba2"),
                                )

                            # v1-fix MLA broaden (Jun 2026 follow-up): drop
                            # the params >= 7B gate. Two reasons:
                            #  (1) The user observed "nothing uses MLA" in the
                            #      headline 1B/3B cells — and indeed, the gate
                            #      hard-blocked MLA there even though the
                            #      quality_model already penalizes MLA at small
                            #      latent-dim ratios; the search would naturally
                            #      *not* pick MLA at short context, so a static
                            #      gate is redundant.
                            #  (2) At 1M+ ctx with 1B/3B params, MLA's 60× KV
                            #      reduction is exactly the move you want, and
                            #      currently nothing in the search even tries.
                            # Also add latent-dim variety so the search explores
                            # the compression-vs-quality trade.
                            allow_mla = True
                            # Single latent dim keeps the candidate count per
                            # lattice point at +1 (MLA variant only); going
                            # wider (256/384/512) triples the MLA candidates
                            # and slows the search ~2× without changing the
                            # winner much for the demo grid. The optimizer
                            # already picks the right latent shape relative
                            # to d_head via the quality model's c_kv penalty.
                            mla_kv_latent_opts = [512]
                            constraints = DeploymentConstraints(
                                target_params_b=params,
                                training_tokens=int(tokens_T * 1e12),
                                context_length=ctx,
                                serving_tbt_ms=smode["tbt"],
                                serving_ttft_ms=smode["ttft"],
                                # v1-fix demo-audit (June 2026 → fixed):
                                # architecture search uses a representative
                                # batch (`search_batch`); after optimize()
                                # picks an arch, we re-evaluate it across
                                # `batches` to pick the largest batch
                                # satisfying the TBT bound (post-search loop
                                # below). This makes "serving_*" a bound
                                # rather than an objective.
                                serving_batch=int(smode.get("search_batch", smode.get("batch", 16))),
                                tp=tp, pp=pp, dp=mode_dp,
                                training_cluster_gpus=training_cluster,
                                max_training_cluster_gpus=training_cluster,
                                # Wave 4 (Jun 2026): TP is now a search
                                # variable. tp_options[0] equals `tp`
                                # (back-compat) and additional entries (2×,
                                # 4× capped at the NVLink island) let the
                                # optimizer pick a higher TP for cells where
                                # KV pressure or weight pressure makes a
                                # larger TP strictly better.
                                tp_options=list(tp_options),
                                # v1-fix long-ctx (June 2026): cp_options
                                # lets the optimizer pick the smallest CP
                                # that fits KV at this context. cp=1 is the
                                # floor so short-ctx cells aren't forced
                                # to pay CP overhead.
                                cp=1,
                                cp_options=mode_cp_options,
                                vocab_size=32000,
                                allow_mla=allow_mla,
                                mla_kv_latent_options=mla_kv_latent_opts,
                                # v1-fix demo-audit (June 2026): the default
                                # ±15% param band let "7B" cells resolve to
                                # 6.34B vs 7.54B — a 19% gap across cells
                                # that share a column label. Tighten to ±8%
                                # so cells stay comparable. Calibration /
                                # API users still see the 15% default.
                                param_tolerance=0.08,
                                **extra_kw,
                            )

                            result = optimize(hw, constraints)

                            entry = {
                                "hw": hw, "params_B": params,
                                "tokens_T": tokens_T,
                                "serving": smode["name"],
                                "context_length": ctx,
                                "context_label": CONTEXT_LABELS.get(ctx, f"{ctx}"),
                                "arch_mode": amode["name"],
                                "state_type": amode.get("state_type") if amode["allow_state"] else None,
                                # Wave 4: `tp` here is the *floor* the planner
                                # gave the optimizer (back-compat with v1
                                # downstream consumers). The actually selected
                                # TP lives on each candidate (entry["optimal"]
                                # ["tp"]) and may be larger. `tp_options`
                                # exposes the search space.
                                "tp": tp, "pp": pp, "dp": mode_dp,
                                "tp_options": list(tp_options),
                                # Per-replica node count: TP >= 16 spans multiple
                                # NVLink islands and requires cross-IB tensor parallel.
                                "nodes_per_replica": nodes_for_tp(tp),
                                "training_cluster_target_gpus": training_cluster,
                                "training_cluster_gpus": training_cluster,
                                "training_nodes": max(
                                    1, math.ceil(training_cluster / GPUS_PER_NODE)),
                                # v1-fix audit (June 2026): expose the
                                # serving topology numbers explicitly so the
                                # UI can show TP/PP/batch next to each
                                # candidate. Per-replica numbers added
                                # below from the optimal/pareto payloads.
                                "serving_batch": int(smode.get("batch", 32)),
                                "serving_tbt_budget_ms": smode["tbt"],
                                "serving_ttft_budget_ms": smode["ttft"],
                                "candidates": result.candidates_generated,
                                "feasible": result.candidates_feasible,
                                # Wave 5 follow-up: raw enumeration count
                                # (before max_candidates cap). This is the
                                # signal the canonical-shape pin gates on.
                                "candidates_enumerated_raw": getattr(
                                    result, "candidates_enumerated_raw", 0),
                                "pareto_size": len(result.pareto_frontier),
                                "time_s": result.search_time_sec,
                                "optimal": None,
                                "pareto": [],
                                "justification": "",
                                "shadow_prices": [],
                            }

                            def _annotate(rec: dict, chosen_batch=None) -> dict:
                                """Add per-replica memory and topology so cross-TP cells are comparable.

                                `chosen_batch` overrides the regime's nominal
                                batch with the actually selected batch (from
                                `_pick_serving_batch`). This is set on the
                                optimal record so the UI shows the real
                                serving batch, not the regime placeholder.

                                Wave 4: the per-candidate tp is read from the
                                serialized record (`rec["tp"]`), not the cell
                                driver's closure-captured floor TP. This makes
                                the per-replica / per-GPU / aggregate columns
                                correct for candidates whose tp_degree differs
                                from the cell's base TP.
                                """
                                batch = (
                                    int(chosen_batch) if chosen_batch is not None
                                    else int(smode.get("search_batch",
                                                       smode.get("batch", 16)))
                                )
                                return annotate_parallelism_units(
                                    rec,
                                    pp=pp,
                                    dp=mode_dp,
                                    fallback_tp=tp,
                                    fallback_ep=mode_ep,
                                    chosen_batch=batch,
                                )

                            # v1-fix sanity gate (June 2026): mirror the
                            # optimizer's _SENTINEL_LOSS_MULT filter at the
                            # generator boundary. If a sentinel-inflated
                            # candidate somehow makes it through to here
                            # (e.g. via _pick_serving_batch's fallback path
                            # which doesn't re-check predicted_loss), drop it
                            # so the public grid never shows loss in the
                            # millions. The audit caught 890/4506 rows with
                            # loss > 1e6 — this gate is the last line of
                            # defense against that recurring.
                            _is_sentinel_optimal = False
                            if result.optimal is not None:
                                try:
                                    _base = float(result.optimal.quality.chinchilla_baseline)
                                    _loss = float(result.optimal.predicted_loss)
                                    if _base > 0 and _loss > 10.0 * _base:
                                        _is_sentinel_optimal = True
                                except (AttributeError, TypeError, ValueError):
                                    _is_sentinel_optimal = True

                            if result.optimal and not _is_sentinel_optimal:
                                # v1-fix demo-audit: pick the largest batch
                                # where the chosen arch still satisfies the
                                # TBT bound (and fits memory). For
                                # unconstrained, pick the largest batch that
                                # fits in per-GPU HBM. This re-evaluates the
                                # already-chosen arch at multiple batches
                                # without re-running the architecture search.
                                _final_optimal = _pick_serving_batch(
                                    result.optimal, hw, constraints, smode,
                                )
                                # v1-fix sanity gate: _pick_serving_batch may
                                # fall back to the original ev_optimal if no
                                # batch satisfies the budget. That fallback
                                # could still carry sentinel inflation if the
                                # arch was evaluated under a different memory
                                # condition. Re-check the final pick.
                                try:
                                    _base = float(_final_optimal.quality.chinchilla_baseline)
                                    _loss = float(_final_optimal.predicted_loss)
                                    if _base > 0 and _loss > 10.0 * _base:
                                        _is_sentinel_optimal = True
                                except (AttributeError, TypeError, ValueError):
                                    _is_sentinel_optimal = True

                            if result.optimal and not _is_sentinel_optimal:
                                # Stash the chosen batch back on the entry
                                # so downstream annotation sees it.
                                _chosen_batch = int(getattr(
                                    _final_optimal, "_chosen_batch",
                                    constraints.serving_batch,
                                ))
                                entry["serving_batch"] = _chosen_batch
                                _annotate_batch = _chosen_batch  # closure read
                                entry["optimal"] = _annotate(serialize_candidate(_final_optimal),
                                                             chosen_batch=_chosen_batch)
                                entry["training_cluster_gpus"] = entry["optimal"][
                                    "training_cluster_gpus"]
                                entry["training_nodes"] = max(
                                    1,
                                    math.ceil(entry["training_cluster_gpus"]
                                              / GPUS_PER_NODE),
                                )

                                sampled = sample_pareto_diverse(result.pareto_frontier, max_points=30)
                                # v1-fix demo-audit: also re-evaluate each
                                # pareto candidate at the same chosen batch
                                # so the Pareto table is apples-to-apples
                                # with the selected optimum, not at a
                                # mismatched batch placeholder.
                                for ev in sampled:
                                    try:
                                        c_ev = _copy.copy(constraints)
                                        c_ev.serving_batch = _chosen_batch
                                        ev_rebatched = evaluate_candidate(ev.arch, hw, c_ev)
                                    except Exception:
                                        ev_rebatched = ev
                                    entry["pareto"].append(
                                        _annotate(serialize_candidate(ev_rebatched),
                                                  chosen_batch=_chosen_batch)
                                    )

                                # v1-fix Wave 2a Step 2a.4 (Jun 2026): 4D
                                # Pareto exposure. Pick the 32 nearest points
                                # to the min-loss winner in normalized
                                # (loss, log(tbt), log(mem)) L-infinity space.
                                # This is the cross-family view consumed by
                                # the family-rollup helper and the 2c web app
                                # scatter plot. We always include the optimal
                                # as anchor index 0.
                                _opt = entry["optimal"]
                                _pareto_pts = [_opt] + list(entry["pareto"])
                                def _pareto_dist(p, anchor):
                                    import math as _m
                                    return max(
                                        abs(p.get("loss", 0) - anchor.get("loss", 0)) / max(anchor.get("loss", 1.0), 1e-6),
                                        abs(_m.log10(max(p.get("tbt_ms", 1.0), 1e-3)) - _m.log10(max(anchor.get("tbt_ms", 1.0), 1e-3))),
                                        abs(_m.log10(max(p.get("mem_gb", 1.0), 1e-3)) - _m.log10(max(anchor.get("mem_gb", 1.0), 1e-3))),
                                    )
                                _pareto_pts_sorted = sorted(_pareto_pts, key=lambda p: _pareto_dist(p, _opt))[:32]
                                entry["pareto_4d"] = _pareto_pts_sorted

                                # Generate the long-form justification + shadow
                                # prices only for the canonical (2T-token, 8k-ctx)
                                # slice to bound the data size; other slices keep
                                # the minimal optimal+pareto payload.
                                if tokens_T == 2.0 and ctx == 8192:
                                    try:
                                        sp = compute_shadow_prices(hw, constraints, result)
                                        entry["shadow_prices"] = [
                                            {
                                                "desc": p.perturbation_desc,
                                                "delta_pct": p.delta_loss_pct,
                                                "interp": p.interpretation,
                                            }
                                            for p in sp.prices
                                        ]
                                        # v1-fix demo-audit (June 2026): also
                                        # carry the arch-dimension shadow prices
                                        # so the UI can render the +1/-1 layer,
                                        # ±256 d_model, FP8/INT8 KV deltas.
                                        entry["arch_dim_shadow_prices"] = [
                                            {
                                                "dim": a.dimension,
                                                "change": a.change_desc,
                                                "delta_loss_pct": a.delta_loss_pct,
                                                "delta_train_tps_pct": a.delta_train_tps_pct,
                                                "delta_tbt_pct": a.delta_tbt_pct,
                                                "delta_mem_pct": a.delta_mem_pct,
                                                "decision": a.decision,
                                                "reason": a.reason,
                                                "feasible": a.feasible,
                                            }
                                            for a in sp.arch_dim_prices
                                        ]
                                        entry["justification"] = generate_justification(result, sp)
                                    except Exception as _sp_exc:
                                        import traceback as _tb
                                        print(f"    [shadow_prices FAILED on {hw} {params}B {smode['name']} ctx={ctx} {amode['name']}]: "
                                              f"{type(_sp_exc).__name__}: {_sp_exc}")
                                        _tb.print_exc()
                                        entry["justification"] = generate_justification(result)
                                        # v1-fix demo-audit (June 2026): the
                                        # previous bare except swallowed failures
                                        # and shipped empty shadow_prices for
                                        # every cell. If the user explicitly
                                        # asked to skip shadow prices (via the
                                        # regen flag that monkey-patches the
                                        # function to raise) the message contains
                                        # 'shadow prices disabled'; otherwise
                                        # surface the failure so it doesn't ship.
                                        if "disabled in this regen" not in str(_sp_exc):
                                            if os.environ.get("AC_STRICT_SHADOW") == "1":
                                                raise

                            done += 1
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed > 0 else 0
                            eta = (total - done) / rate if rate > 0 else 0
                            print(f"  [{done}/{total}] {hw} {params}B {tokens_T}T "
                                  f"ctx={CONTEXT_LABELS.get(ctx, ctx)} "
                                  f"{smode['name']} [{amode['name']}]: "
                                  f"{'OK' if result.optimal else 'NO SOLUTION'} "
                                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

                            data["grid"].append(entry)

    print(f"\nTotal: {done} runs in {time.time()-t0:.1f}s")

    # Wave 11 (Jun 2026): decision + diagnostics schema split. Every grid
    # entry gains an additive `decision` block (the answer + uncertainty)
    # and a `diagnostics` block (everything else). Legacy `optimal` /
    # `pareto` / etc. are retained as duplicates for back-compat during
    # the transition. See plan/redesign/11-cell-output-schema-cleanup.md.
    for row in data.get("grid", []):
        _build_decision_diagnostics(row)

    # v1-fix demo-audit consistency pass (June 2026): the per-cell
    # optimize() call is hardware- and serving-budget-specific. Three
    # consistency bugs the original payload exhibited:
    #
    #   (B) Same training plan (hw, params, ctx, arch_mode, tp/pp/dp)
    #       reported different train_tps depending on the serving budget,
    #       because the optimizer was picking different architectures
    #       under different serving budgets. Training throughput should
    #       be a function of the training plan, not the serving SLO.
    #
    #   (C) Per-replica TPS may legitimately rise when a larger target uses
    #       a larger TP/PP/CP replica. Per-GPU TPS is the cross-layout metric;
    #       never rewrite evaluated physics to make per-replica rows monotone.
    #
    #   (D) The same (params, tokens, ctx, arch_mode) on different hw
    #       sometimes produced very different loss values, because each
    #       hw's lattice picks a different architecture. When the chosen
    #       architectures genuinely match in shape, force the displayed
    #       loss to be identical across hw (it's an arch+data function,
    #       not a hw function).
    #
    # Each pass operates strictly on the grid in-place and is a no-op
    # when the underlying optimizer choices already agree.
    #
    # Wave 10C (Jun 2026): _harmonize_serving_train_metrics and
    # _harmonize_loss_across_hw are no-ops now that 10A (train_tps
    # independent of serving SLO) and 10B (quality hw-blind) fix the
    # underlying disagreements at the optimizer/quality-model layer.
    # We keep the function definitions for back-compat but stop calling
    # them in the post-processing pipeline. Bug B and Bug D regressions
    # are caught directly by `tests/test_optimizer_self_consistency.py`.
    # See plan/redesign/10-optimizer-self-consistency.md Change C.
    # Wave 31: the family rollup + smoothing + pin + plateau chain is a
    # named, reusable step so partial regens (--merge-into) and offline
    # rebuilds (emit_decision_grid.py --rebuild) run the identical
    # pipeline. See run_post_chain() for the per-step rationale.
    run_post_chain(data)

    return data


def run_post_chain(data: dict) -> dict:
    """Post-search pipeline. Pure dict transforms; NO optimizer searches
    (the pin can re-evaluate the quality model on drift, never re-search).

    Order matters:
      1. _prune_public_quality_sentinels — remove outside-coverage sentinel
         candidates before any public family/plateau winner can see them.
      2. _build_family_rollup — per-cell families from serialized records,
         picked and ordered by the optimizer's canonical display sort key
         (Wave 30). Runs early because everything downstream edits its
         output.
      3. _smooth_family_flicker — 1D ctx-row median re-rank inside the
         uncertainty band.
      4. _smooth_family_flicker_2d — tighter own-col x +-1-ctx smoothing
         for the large-model columns (13/120/750B).
      5. _pin_canonical_shape_per_family — re-evaluates a canonical shape
         when (d_model, n_layers) drifts >40% across a ctx row; no-op
         under the multi-ctx flow (shape pinned by construction).
      6. _annotate_plateau_marker — tags fams[0] with its nearest
         different-arch quality-equivalent alternative. Runs LAST, on the
         final ordering.
    """
    _prune_public_quality_sentinels(data)
    _build_family_rollup(data)
    _smooth_family_flicker(data)
    _smooth_family_flicker_2d(data, target_params={13, 120, 750})
    _pin_canonical_shape_per_family(data)
    _annotate_plateau_marker(data)
    return data


def merge_payload(base: dict, new: dict) -> dict:
    """Merge a partial regen into an existing payload (Wave 31).

    Grid rows are keyed by (hw, params_B, tokens_T, context_length,
    serving, arch_mode, state_type); a new row REPLACES the matching old
    row, unmatched new rows append, all other old rows are kept.
    hardware_info is unioned (new wins per hw). Cells / families /
    smoothing metadata are NOT merged — they are derived state, so the
    caller must re-run run_post_chain() on the merged payload.
    """
    def _key(r):
        return (r.get("hw"), r.get("params_B"), r.get("tokens_T"),
                r.get("context_length"), r.get("serving"),
                r.get("arch_mode"), r.get("state_type"))

    new_by_key = {}
    for row in new.get("grid", []):
        new_by_key[_key(row)] = row
    merged_grid = []
    replaced = 0
    for row in base.get("grid", []):
        k = _key(row)
        if k in new_by_key:
            merged_grid.append(new_by_key.pop(k))
            replaced += 1
        else:
            merged_grid.append(row)
    appended = len(new_by_key)
    merged_grid.extend(new_by_key.values())

    base["grid"] = merged_grid
    hw_info = dict(base.get("hardware_info") or {})
    hw_info.update(new.get("hardware_info") or {})
    base["hardware_info"] = hw_info
    # Derived state is stale after a grid merge; drop it so a forgotten
    # run_post_chain() fails loudly downstream instead of silently
    # serving pre-merge winners.
    base.pop("cells", None)
    base.pop("_family_smoothing", None)
    print(f"[merge] replaced {replaced} grid rows, appended {appended}")
    return base


def _pretty_family_name(arch_mode: str, state_type) -> str:
    """Display label for an (arch_mode, state_type) pair. Used by CLI and web."""
    base = {
        "dense": "dense",
        "moe": "MoE",
        "hybrid": "hybrid",
        "moe_hybrid": "MoE-hybrid",
    }.get(arch_mode, arch_mode)
    if state_type and state_type != "mamba2":
        # mamba2 is the implicit default; only annotate non-default state types.
        suffix = {"gated_delta": "gd", "kda": "kda", "gla": "gla", "sliding_window": "swa"}.get(state_type, state_type)
        return f"{base} ({suffix})"
    return base


def _arch_family_from_optimal(opt: dict) -> str:
    """Wave 11 helper: read the arch family from a serialized optimal block.
    Mirrors the Wave 8b stratification family classifier.

    Wave 18a: routes through the canonical ArchitectureSignature. The
    serialized ``opt`` dict flattens MoE/state fields (``n_experts``,
    ``n_state_layers``) — we synthesize the nested ``moe`` and
    ``state_config`` blocks the signature helper expects before calling
    it, so this grid-driver rollup label agrees with the optimizer's
    stratification, the CLI report, and the golden-matrix regen."""
    if not isinstance(opt, dict):
        return "dense"
    try:
        from ac.architecture import architecture_signature
        # Reconstruct the nested shape from flat serialized fields so the
        # signature helper's MoE / state detection sees the same evidence
        # the pre-Wave-18a inline classifier saw.
        probe = dict(opt)
        if "moe" not in probe and probe.get("n_experts"):
            probe["moe"] = {
                "n_experts": probe.get("n_experts"),
                "top_k": probe.get("top_k", 1),
                "expert_dim": probe.get("expert_dim"),
            }
        if "state_config" not in probe and probe.get("n_state_layers"):
            probe["state_config"] = {
                "state_layers": probe.get("n_state_layers"),
                "state_type": probe.get("state_type", "mamba2"),
            }
        return architecture_signature(probe).legacy_family
    except (ValueError, ImportError):
        has_moe = bool(opt.get("moe") or opt.get("n_experts"))
        has_state = bool(opt.get("state_config") or opt.get("n_state_layers"))
        if has_moe and has_state: return "moe_hybrid"
        if has_moe:               return "moe"
        if has_state:             return "hybrid"
        return "dense"


def _build_decision_diagnostics(row: dict) -> None:
    """Wave 11 (Jun 2026): populate `row["decision"]` and `row["diagnostics"]`
    from the legacy `optimal` / `pareto` / etc. fields.

    Schema (per plan/redesign/11-cell-output-schema-cleanup.md):

      row["decision"] = {
        shape, precision, parallelism, moe, state, attention,
        predicted_loss, predicted_loss_uncertainty_pct,
        predicted_tbt_ms, predicted_ttft_ms, predicted_mem_gb,
        predicted_training_tps, confidence, family,
      }
      row["diagnostics"] = {
        alternatives, shadow_prices, arch_dim_shadow_prices,
        quality_residual_breakdown, search_stats, smoothing,
        justification,
      }

    Legacy fields are NOT removed; new consumers read `decision`, old
    consumers keep working. Run after the cell is fully populated."""
    opt = row.get("optimal")
    if not isinstance(opt, dict):
        # No feasible solution: emit a minimal decision block with the
        # NO_SOLUTION marker so downstream renders can branch cleanly.
        row["decision"] = {
            "predicted_loss": None,
            "family": None,
            "no_feasible_solution": True,
        }
        row["diagnostics"] = {
            "search_stats": {
                "candidates_generated": row.get("candidates", 0),
                "candidates_feasible": row.get("feasible", 0),
                "candidates_enumerated_raw": row.get("candidates_enumerated_raw", 0),
                "search_time_sec": row.get("time_s", 0.0),
            },
            "justification": row.get("justification", ""),
            "smoothing": {},
        }
        return

    family = _arch_family_from_optimal(opt)

    # Shape block (every dense-ish field; MoE/state get their own subblocks).
    shape = {
        "d_model": opt.get("d_model"),
        "n_layers": opt.get("n_layers"),
        "n_heads": opt.get("n_heads"),
        "d_head": opt.get("d_head"),
        "n_kv_heads": opt.get("n_kv_heads"),
        "ffn_dim": opt.get("ffn_dim"),
        "vocab_size": opt.get("vocab_size"),
    }
    precision = {
        "weight": opt.get("weight_prec") or opt.get("weight_precision"),
        "ffn": opt.get("ffn_prec") or opt.get("ffn_precision"),
        "attn": opt.get("attn_precision"),
        "kv_bits": opt.get("kv_bits") or opt.get("kv_cache_bits"),
    }
    parallelism = {
        "tp": opt.get("tp") or row.get("tp"),
        "pp": opt.get("pp") or row.get("pp"),
        "ep": opt.get("ep") or opt.get("ep_degree"),
        "cp": opt.get("cp") or opt.get("cp_degree"),
        "dp": opt.get("dp") or row.get("dp"),
    }
    moe = None
    if family in ("moe", "moe_hybrid"):
        moe = {
            "n_experts": opt.get("n_experts"),
            "top_k": opt.get("top_k"),
            "expert_dim": opt.get("expert_dim"),
            "ep": opt.get("ep_degree") or opt.get("ep"),
            "shared_expert": opt.get("shared_expert"),
        }
    state = None
    if family in ("hybrid", "moe_hybrid"):
        state = {
            "n_state_layers": opt.get("n_state_layers"),
            "n_attention_layers": opt.get("n_attention_layers"),
            "state_type": row.get("state_type") or opt.get("state_type"),
            "placement_strategy": opt.get("placement_strategy"),
            "d_state": opt.get("d_state"),
        }
    attention = {
        "type": opt.get("attention_type", "full"),
        "mla_kv_latent_dim": opt.get("mla_kv_latent_dim"),
        "mla_q_latent_dim": opt.get("mla_q_latent_dim"),
        "swa_window": opt.get("swa_window") or opt.get("local_window"),
        "nsa_window_size": opt.get("nsa_window_size"),
        # Wave 9 compressed-attention configs (only one populated when active).
        "csa_block_size": opt.get("csa_block_size"),
        "csa_top_k_blocks": opt.get("csa_top_k_blocks"),
        "indexshare_num_buckets": opt.get("indexshare_num_buckets"),
        "indexshare_top_k_buckets": opt.get("indexshare_top_k_buckets"),
        "msa_window_size": opt.get("msa_window_size"),
        "msa_dilated_top_k": opt.get("msa_dilated_top_k"),
        "msa_global_top_k": opt.get("msa_global_top_k"),
    }

    # Wave 18a: emit the factorized ArchitectureSignature on the decision
    # block so consumers can compare candidates on the 6 axes rather than
    # on the coarse family label. Reuse the signature already serialized
    # inside ``opt`` when available so the two views agree by construction;
    # fall back to deriving from ``opt`` when the legacy row lacks it.
    signature = opt.get("signature")
    if signature is None:
        try:
            from ac.architecture import architecture_signature
            signature = architecture_signature(opt).as_dict()
        except (ValueError, ImportError):
            signature = None

    # The single "this is the answer + its uncertainty" block.
    row["decision"] = {
        "shape": shape,
        "precision": precision,
        "parallelism": parallelism,
        "moe": moe,
        "state": state,
        "attention": attention,
        "predicted_loss": opt.get("loss") or opt.get("predicted_loss"),
        "predicted_loss_uncertainty_pct": opt.get("uncertainty_total_pct"),
        "predicted_tbt_ms": opt.get("tbt_ms"),
        "predicted_ttft_ms": opt.get("ttft_ms"),
        "predicted_mem_gb": opt.get("mem_gb"),
        "predicted_training_tps": opt.get("train_tps")
                                   or opt.get("train_tps_per_replica"),
        "confidence": opt.get("confidence"),
        "family": family,          # compat label — prefer `signature` for compares
        "signature": signature,    # Wave 18a: factorized identity
    }

    # Everything else — alternates, shadow prices, smoothing diagnostics,
    # search stats, justification. Consumers that need depth read here;
    # the matrix UI ignores this block.
    pareto_list = row.get("pareto") or []
    row["diagnostics"] = {
        "alternatives": pareto_list[:5] if isinstance(pareto_list, list) else [],
        "shadow_prices": row.get("shadow_prices", []),
        "arch_dim_shadow_prices": row.get("arch_dim_shadow_prices", []),
        "quality_residual_breakdown": opt.get("quality_terms", {}),
        "search_stats": {
            "candidates_generated": row.get("candidates", 0),
            "candidates_feasible": row.get("feasible", 0),
            "candidates_enumerated_raw": row.get("candidates_enumerated_raw", 0),
            "pareto_size": row.get("pareto_size", 0),
            "search_time_sec": row.get("time_s", 0.0),
        },
        "smoothing": {
            "shape_pinned": opt.get("shape_pinned", False),
            "shape_truncated": opt.get("shape_truncated", False),
            "shape_drift_suppressed": opt.get("shape_drift_suppressed"),
            "multi_ctx_reference": row.get("multi_ctx_reference"),
            "multi_ctx_shape_pinned": row.get("multi_ctx_shape_pinned"),
        },
        "justification": row.get("justification", ""),
    }


def _rank_view(opt: dict):
    """Adapt a serialized candidate record to the attribute surface
    `ac.optimizer.build_display_sort_key` reads (EvaluatedCandidate-shaped).

    The canonical key needs: predicted_loss, memory_per_gpu_gb,
    serving_tbt_ms, training_tps, quality.uncertainty_total,
    arch.{total_params, n_layers, d_model}. All of these are present on
    every serialized grid/pareto record; missing throughput fields fall
    back to values that rank the record last on that axis rather than
    first.
    """
    from types import SimpleNamespace
    loss = float(opt.get("loss") or 1e9)
    mem = opt.get("mem_gb")
    tbt = opt.get("tbt_ms")
    tps = opt.get("train_tps")
    unc_pct = opt.get("uncertainty_total_pct")
    return SimpleNamespace(
        predicted_loss=loss,
        memory_per_gpu_gb=float(mem) if mem is not None else float("inf"),
        serving_tbt_ms=float(tbt) if tbt is not None else float("inf"),
        training_tps=float(tps) if tps is not None else 0.0,
        throughput=SimpleNamespace(
            spill_tier=str(opt.get("spill_tier") or "fits"),
            prefill_time_ms=float(opt.get("ttft_ms") or float("inf")),
        ),
        quality=SimpleNamespace(
            uncertainty_total=(float(unc_pct) / 100.0) if unc_pct else 0.0),
        arch=SimpleNamespace(
            total_params=float(opt.get("params_B") or 0.0) * 1e9,
            n_layers=int(opt.get("n_layers") or 1),
            d_model=int(opt.get("d_model") or 1),
            tp_degree=max(1, int(opt.get("tp") or 1)),
            pp_degree=max(1, int(opt.get("pp") or 1)),
            cp_degree=max(1, int(opt.get("cp") or 1)),
            ep_degree=max(1, int(opt.get("ep") or 1)),
        ),
    )


def _pick_family_representative(pool: list) -> tuple:
    """Pick one (row, opt) per family with the optimizer's canonical
    display sort key (Wave 30, Jul 2026).

    `build_display_sort_key`'s docstring says "Don't re-implement the key
    in either site; extend this builder" — the previous bare
    `min(loss)` here was exactly such a re-implementation, minus the
    noise-band bucketing, so a record 0.05-0.3% lower in loss (deep
    inside the ±2-3% uncertainty band) could win the family slot while
    being 5-10× worse on TBT and spilling past HBM. We adapt the
    serialized dicts to the attribute surface the canonical key expects
    and let it rank; the research_quality branch keeps loss strictly
    dominant via its bucket index, so this NEVER trades a real loss win
    away — it only breaks noise-band ties on memory/TBT/-TPS.
    """
    from types import SimpleNamespace
    if len(pool) == 1:
        return pool[0]
    # Per-context Pareto samples are useful alternatives, but only the joint
    # multi-context optimum represents one stable identity across the row.
    pinned = [item for item in pool
              if bool(item[1].get("multi_ctx_shape_pinned", False))]
    if pinned:
        pool = pinned
        if len(pool) == 1:
            return pool[0]
    from optimizer import build_display_sort_key
    views = [_rank_view(opt) for _row, opt in pool]
    constraints = SimpleNamespace(
        objective_profile="research_quality", strict_quality=False)
    key = build_display_sort_key(views, constraints)
    ranked = sorted(zip(views, pool), key=lambda pair: key(pair[0]))
    return ranked[0][1]


def _stitch_families_to_carrier_rows(data: dict) -> None:
    """Mirror each cell's families list onto exactly ONE sibling grid row.

    Wave 30 (Jul 2026): app.js resolves a cell's families via
    `GRID.find(... Array.isArray(g.families) && g.families.length)`, so a
    single carrier row per (hw, params, tokens, ctx) is sufficient.
    Stitching every sibling row (the previous behavior, in three separate
    sites) serialized the identical list ~9x per cell and cost ~40 MB in
    the shipped multi-hw compiler-data.json. Non-carrier rows have any
    stale `families` from earlier pipeline runs removed.
    """
    cells_by_key = {}
    for cell in data.get("cells", []):
        ck = (cell.get("hw"), cell.get("params_B"), cell.get("tokens_T"),
              cell.get("context_length"))
        cells_by_key[ck] = cell.get("families", [])
    stitched = set()
    for row in data.get("grid", []):
        ck = (row.get("hw"), row.get("params_B"), row.get("tokens_T"),
              row.get("context_length"))
        if ck in cells_by_key and ck not in stitched:
            row["families"] = cells_by_key[ck]
            stitched.add(ck)
        else:
            row.pop("families", None)


def _sort_families_canonical(families: list) -> None:
    """Order a cell's family entries with the optimizer's canonical
    display sort key (Wave 30, Jul 2026).

    The previous `sort(key=loss)` had the same failure mode cross-family
    that the bare within-family argmin had: on 7B/h100/2M the MoE rep
    (L=2.1508, TBT=311 ms) outranked the swa hybrid (L=2.1542,
    TBT=21 ms) on a 0.16% loss edge — sub-noise against the ±2-3%
    uncertainty — putting a 15× slower config in the grid headline.
    `ac-compile`'s own picker ranks its whole mixed-family pool with
    `build_display_sort_key`; the cell's family ordering now mirrors it.
    Loss stays strictly dominant via the bucket index: real (out-of-band)
    loss wins are never traded away.
    """
    if len(families) < 2:
        return
    from optimizer import build_display_sort_key
    from types import SimpleNamespace
    views = [_rank_view(f) for f in families]
    constraints = SimpleNamespace(
        objective_profile="research_quality", strict_quality=False)
    key = build_display_sort_key(views, constraints)
    any_pinned = any(bool(f.get("multi_ctx_shape_pinned", False))
                     for f in families)
    order = sorted(
        range(len(families)),
        key=lambda i: (
            0 if (not any_pinned or families[i].get(
                "multi_ctx_shape_pinned", False)) else 1,
            key(views[i]),
            # Wave 47 (release verification): terminal deterministic
            # tiebreak. sorted() is stable, so exact ties (e.g. four
            # hybrid state-types at identical loss/TBT) previously kept
            # their INPUT order — which is shard-merge order, which
            # differs between a fresh sweep and a resumable sharded
            # regen. That made the family ordering (and therefore the
            # winner label among tied variants in the decision grid)
            # nondeterministic across reruns of identical searches.
            str(families[i].get("arch_mode") or ""),
            str(families[i].get("state_type") or ""),
        ),
    )
    families[:] = [families[i] for i in order]


def _build_family_rollup(data: dict) -> None:
    """Annotate each grid entry with a `families` field that lists the loss-
    minimum candidate for each arch_mode/state_type, sorted by loss ascending,
    plus delta_pct annotations vs the row 0 (best) family.

    Wave 2a Step 2a.5. The `cells` top-level field also collects one entry per
    (hw, params, tokens, ctx) for direct consumption by the web app and CLI.
    """
    from collections import defaultdict

    # Group rows by (hw, params, tokens, ctx). A requested search mode is an
    # allowance, not a guarantee: allow_moe=True still contains dense
    # candidates, and allow_moe+allow_state can select pure MoE. Therefore
    # family identity must come from each serialized candidate's actual
    # architecture, never from row["arch_mode"].
    by_cell = defaultdict(list)
    for row in data.get("grid", []):
        key = (row.get("hw"), row.get("params_B"), row.get("tokens_T"),
               row.get("context_length"))
        by_cell[key].append(row)

    cells = []
    for key, rows in by_cell.items():
        # Collect the optimal plus sampled Pareto candidates from every mode,
        # then retain the best candidate for each *actual* family.
        #
        # Wave 30 (Jul 2026): the family representative is picked with the
        # optimizer's canonical `build_display_sort_key` (research_quality
        # branch) instead of a bare min(loss). The bare argmin reintroduced
        # exactly the bug class Wave 19/23 fixed inside the picker: on the
        # 7B/h100/32k cell it crowned a tp=1 MoE record 0.3% lower on a
        # ±3% uncertainty signal while spilling 120 GB past HBM (mem
        # 199.7 GB, TBT 220 ms) — overriding the optimizer's own `optimal`
        # for the same cell (tp=4, fits, 17.3 ms). Records inside the
        # loss noise band now tiebreak on memory bucket / TBT / -TPS, so
        # the family entry agrees with what optimize() itself would pick.
        pool_per_family = defaultdict(list)
        for row in rows:
            records = []
            if row.get("optimal"):
                records.append(row["optimal"])
            records.extend(row.get("pareto") or [])
            for opt in records:
                if _is_public_quality_sentinel(opt):
                    continue
                actual_mode = (
                    opt.get("arch_family")
                    or _arch_family_from_optimal(opt)
                )
                actual_state = (
                    opt.get("state_type") or row.get("state_type")
                    if actual_mode in {"hybrid", "moe_hybrid"}
                    else None
                )
                pool_per_family[(actual_mode, actual_state)].append((row, opt))

        best_per_family = {
            fam_key: _pick_family_representative(pool)
            for fam_key, pool in pool_per_family.items()
        }

        families = []
        # Determine the cell's intended params target (taken from any sibling
        # row — every row in this cell shares params_B by construction).
        target_params_B = float(rows[0].get("params_B", 0.0))
        for (am, st), (row, opt) in best_per_family.items():
            # Wave 4 follow-up (Jun 2026): shape-truncation flag.
            # When memory pressure forces the optimizer's shape enumerator
            # to fall back to a candidate whose active params materially
            # undershoots the cell's target, the cell is "parallelism-
            # limited" — its loss reflects a smaller architecture than the
            # column label implies. We mark this so consumers (CLI/web)
            # can grey-out or label the row honestly rather than
            # presenting a mislabeled winner.
            #
            # Threshold = 60% of target. The active param budget is read
            # off active_params_B (MoE) or params_B (dense/state).
            actual_active = float(opt.get("active_params_B")
                                  or opt.get("params_B") or 0.0)
            shape_truncated = False
            if target_params_B > 0 and actual_active > 0:
                ratio = actual_active / target_params_B
                if ratio < 0.6:
                    shape_truncated = True
            families.append({
                "arch_mode": am,
                "state_type": st,
                "display": _pretty_family_name(am, st),
                "loss": opt.get("loss"),
                "tbt_ms": opt.get("tbt_ms"),
                "ttft_ms": opt.get("ttft_ms"),
                "mem_gb": opt.get("mem_gb"),
                "train_tps": opt.get("train_tps"),
                "train_tps_per_replica": opt.get(
                    "train_tps_per_replica", opt.get("train_tps")),
                "train_tps_per_gpu": opt.get("train_tps_per_gpu"),
                "train_tps_aggregate": opt.get("train_tps_aggregate"),
                "train_tps_unit": opt.get("train_tps_unit"),
                # Wave 30: carried so _sort_families_canonical can derive
                # the loss noise band for cross-family ordering.
                "uncertainty_total_pct": opt.get("uncertainty_total_pct"),
                "hbm_spill_gb": opt.get("hbm_spill_gb", 0.0),
                "spill_tier": opt.get("spill_tier", "fits"),
                "tbt_ms_no_spill": opt.get("tbt_ms_no_spill", opt.get("tbt_ms")),
                "ttft_ms_no_spill": opt.get("ttft_ms_no_spill", opt.get("ttft_ms")),
                "dp_grad_allreduce_ms": opt.get("dp_grad_allreduce_ms", 0.0),
                # Wave 4: prefer the optimum's per-candidate TP. Fall back to
                # the row's base TP when the candidate wasn't serialized with
                # a tp field (older payloads).
                "tp": opt.get("tp", row.get("tp")),
                "pp": opt.get("pp", row.get("pp")),
                "dp": opt.get("dp", row.get("dp")),
                "ep": opt.get("ep", 1),
                "cp": opt.get("cp_degree", opt.get("cp", 1)),
                "training_replica_gpus": opt.get("training_replica_gpus"),
                "serving_instance_gpus": opt.get("serving_instance_gpus"),
                "training_cluster_gpus": opt.get("training_cluster_gpus"),
                "d_model": opt.get("d_model"),
                "n_layers": opt.get("n_layers"),
                "n_heads": opt.get("n_heads"),
                "d_head": opt.get("d_head"),
                "n_kv_heads": opt.get("n_kv_heads"),
                "ffn_dim": opt.get("ffn_dim"),
                "weight_prec": opt.get("weight_prec"),
                "kv_bits": opt.get("kv_bits"),
                # Wave 4 follow-up: shape-truncation diagnostics.
                "shape_truncated": shape_truncated,
                "actual_active_params_B": round(actual_active, 2),
                "target_params_B": target_params_B,
                # Wave 5 follow-up: surface the optimizer's per-family
                # enumeration size so the canonical-shape pin can gate
                # itself on it. The pin gates on `candidates_enumerated_raw`
                # (the pre-cap enumeration count), not `candidates_generated`
                # which is bounded by max_candidates and therefore loses
                # the "huge search" signal.
                "candidates_generated": int(row.get("candidates", 0) or 0),
                "feasible_count": int(row.get("feasible", 0) or 0),
                "candidates_enumerated_raw": int(
                    row.get("candidates_enumerated_raw", 0) or 0),
                # Wave 5 follow-up bugfix (Jun 2026): carry architectural
                # extras so the canonical-shape pin's re-evaluation
                # reconstructs the same architecture the optimizer
                # actually picked. Without these, the re-eval discards
                # penalties from MTP / MLA / YOCO / sparsity / first-K-
                # dense-FFN-prefix and the pinned loss reads 5-10% lower
                # than the optimizer's pick for the same shape.
                "attention_type": opt.get("attention_type", "gqa"),
                "mla_kv_latent_dim": opt.get("mla_kv_latent_dim"),
                "mla_q_latent_dim": opt.get("mla_q_latent_dim"),
                "mla_rope_head_dim": opt.get("mla_rope_head_dim"),
                "mla_nope_head_dim": opt.get("mla_nope_head_dim"),
                "mtp_n_predict_depths": opt.get("mtp_n_predict_depths", 0),
                "mtp_depth_n_layers": opt.get("mtp_depth_n_layers", 1),
                "yoco_n_self_attn_layers": opt.get("yoco_n_self_attn_layers", 0),
                "sparsity_2_4": opt.get("sparsity_2_4"),
                "n_dense_ffn_layers": opt.get("n_dense_ffn_layers", 0),
                # MoE extras (n_experts/top_k/expert_dim already conveyed
                # via moe_config when active; carry them here as flat
                # fields for the pin's quick reconstruction).
                "n_experts": opt.get("n_experts"),
                "top_k": opt.get("top_k"),
                "expert_dim": opt.get("expert_dim"),
                # Wave 6: carry the shape-pinned flag onto the family entry
                # so the post-hoc canonical-shape pin can short-circuit.
                # Multi-ctx rows are pinned by construction; the pin must
                # not "re-pin" them or it will double-evaluate.
                "multi_ctx_shape_pinned": bool(
                    opt.get("multi_ctx_shape_pinned", False)
                ),
                "multi_ctx_reference": (
                    row.get("multi_ctx_reference")
                    or opt.get("multi_ctx_reference")
                ),
            })

        _sort_families_canonical(families)
        # Delta annotations vs the min-loss family (row 0).
        if families:
            anchor_loss = families[0]["loss"] or 1.0
            anchor_tbt = families[0]["tbt_ms"] or 1.0
            for f in families:
                f["loss_delta_pct"] = round(100.0 * (f["loss"] - families[0]["loss"]) / max(anchor_loss, 1e-9), 2)
                f["tbt_delta_pct"] = round(100.0 * (f["tbt_ms"] - families[0]["tbt_ms"]) / max(anchor_tbt, 1e-9), 1)

        # Stitch families onto ONE carrier row per cell (Wave 30 — see
        # _stitch_families_to_carrier_rows). Done here so consumers that
        # only call the rollup still get the row-level view; the
        # smoothing passes re-stitch via the shared helper.
        for i, row in enumerate(rows):
            if i == 0:
                row["families"] = families
            else:
                row.pop("families", None)

        # Build the per-cell top-level entry for v2 consumers.
        hw, p_B, tokens_T, ctx = key
        cells.append({
            "hw": hw, "params_B": p_B, "tokens_T": tokens_T,
            "context_length": ctx,
            "context_label": CONTEXT_LABELS.get(ctx, str(ctx)),
            "families": families,
        })

    data["cells"] = cells


def _pin_canonical_shape_per_family(
    data: dict,
    shape_drift_threshold: float = 0.40,
    candidate_count_threshold: int = 100_000,
    preferred_reference_ctxs: tuple = (131072, 32768, 1048576, 8192,
                                       2097152, 4194304),
) -> None:
    """Wave 5 follow-up (Jun 2026): pin one canonical architecture shape per
    (hw, params, tokens, arch_family) across all ctxs in the row.

    Problem this fixes: when the lattice filter is bound on a tight budget
    (super-large candidate enumeration cases like 13B+MoE+state), different
    ctxs can pick structurally different shapes because TP search forces
    different lattices per cell. The displayed family loss then looks
    incoherent — d_model can double or halve between adjacent ctxs even
    though the column label says "same 13B model."

    Fix: pick a reference cell per (hw, params, family) — the one at the
    lowest ctx with the same shape across at least two adjacent cells.
    Detect "shape drift" by measuring d_model / n_layers variation across
    the row. Where drift exceeds `shape_drift_threshold` (default 40% on
    log d_model), re-evaluate the reference shape at every other ctx
    using `ac.quality_model.estimate_quality` directly — no optimizer
    call needed; just freshly computes loss for the pinned shape.

    Gating: pin only fires for `(hw, params, family)` triples where the
    optimizer's RAW pre-cap enumeration size (`candidates_enumerated_raw`)
    exceeded `candidate_count_threshold` on average across the row.
    Critically, this is NOT the same as `candidates_generated`, which is
    capped at `constraints.max_candidates` (typically 400 in production)
    and therefore loses the "huge search" signal entirely.

    Default threshold = 100_000 raw candidates: catches MoE/state/MoE-
    hybrid paths at all reasonable params (they enumerate 100k-2M+ raw)
    but skips dense-only paths (~1k-20k raw) where shape drift wouldn't
    be a structural problem.

    Reference ctx selection: when a row has shape drift, the pin needs
    to pick one canonical (d_model, n_layers) to re-evaluate at every
    other ctx. We prefer 128k as the reference because production models
    target ~128k context (Llama 3.1, DeepSeek-V3, Qwen, etc.); the 8k
    reference would bias toward short-ctx-only shapes that ignore KV
    pressure. Fallback order is 128k → 32k → 1M → 8k → 2M → 4M.

    Why "drift > 40%": this catches the 13B case where d_model jumped
    from 3072 to 7680 (a 1.3× log change). A modest 20-30% drift is
    routine (lattice rounding) and shouldn't trigger.
    """
    import math
    from collections import defaultdict

    # Lazy import — only needed if we actually need to re-evaluate
    _estimate_quality = None

    def _kv_bits_to_precision(bits):
        """Map kv_bits (16, 8, 4) to the quality model's precision string."""
        try:
            b = int(bits)
        except (TypeError, ValueError):
            return 'bf16'
        if b >= 16: return 'bf16'
        if b == 8:  return 'fp8'
        if b == 4:  return 'int4'
        return 'bf16'

    by_row = defaultdict(list)
    for cell in data.get("cells", []):
        key = (cell.get("hw"), cell.get("params_B"), cell.get("tokens_T"))
        by_row[key].append(cell)

    pinned_count = 0
    for (hw, p, t), cells_row in by_row.items():
        cells_row.sort(key=lambda c: c.get("context_length", 0))

        # Walk each family across this row, decide if pin is needed.
        # Collect per-family shape history across ctxs.
        per_family_shapes = defaultdict(list)
        for cell in cells_row:
            for f in cell.get("families", []):
                fkey = (f.get("arch_mode"), f.get("state_type"))
                d = f.get("d_model") or 0
                L = f.get("n_layers") or 0
                if d > 0 and L > 0:
                    per_family_shapes[fkey].append((cell.get("context_length"), d, L, f))

        for fkey, history in per_family_shapes.items():
            if len(history) < 3:
                continue

            # Wave 6: skip pinning entirely when every cell in the row
            # already carries `multi_ctx_shape_pinned=True`. Multi-ctx
            # rows are shape-coherent by construction (one architecture
            # picked at the reference ctx, then evaluated at every other
            # ctx), so re-pinning would be a no-op at best and a
            # double-evaluation at worst. We still loop the row so the
            # `shape_drift_suppressed` diagnostic continues to record
            # any unexpected drift (a defensive check; should be empty).
            if all(bool(f.get("multi_ctx_shape_pinned"))
                   for _, _, _, f in history):
                # Defensive sanity check: if multi-ctx claims pinning
                # but shapes still differ, something upstream broke.
                # Surface a diagnostic but don't try to re-pin.
                _d_models_pin = {d for _, d, _, _ in history}
                if len(_d_models_pin) > 1:
                    for _, _, _, f in history:
                        f["multi_ctx_pin_unexpected_drift"] = sorted(_d_models_pin)
                continue

            # Gate 1: only pin when the optimizer's RAW pre-cap
            # enumeration exceeded the threshold. Read
            # `candidates_enumerated_raw` (lifted onto family entries by
            # _build_family_rollup) — NOT `candidates_generated`, which
            # is post-cap and bounded by max_candidates. Fall back to
            # candidates_generated for older payloads that don't have
            # the raw field; that path is conservative because the
            # threshold (100k) is far above max_candidates (400) so
            # older payloads will skip the pin entirely.
            cand_counts = [int(f.get("candidates_enumerated_raw", 0)
                               or f.get("candidates_generated", 0) or 0)
                           for _, _, _, f in history]
            mean_candidates = sum(cand_counts) / max(1, len(cand_counts))

            # Compute shape drift now so we can surface the suppression
            # diagnostic even when we skip the pin.
            d_models = [d for _, d, _, _ in history]
            log_d_range = math.log2(max(d_models)) - math.log2(min(d_models))

            if mean_candidates <= candidate_count_threshold:
                if log_d_range >= shape_drift_threshold:
                    # Drift exists but search was too small to pin.
                    # Annotate every cell's family entry so consumers
                    # know the row would have been pinned at a lower
                    # threshold — useful for tuning the gate.
                    for _, _, _, f in history:
                        f["shape_drift_suppressed"] = {
                            "log_d_range": round(log_d_range, 3),
                            "mean_candidates": int(mean_candidates),
                            "threshold": candidate_count_threshold,
                        }
                continue  # search was small enough — drift unlikely

            # Gate 2: only pin when the shape actually drifts more than
            # the configured threshold across the row. Routine lattice
            # rounding shouldn't trigger a re-eval.
            if log_d_range < shape_drift_threshold:
                continue  # within tolerance, no pin needed

            # Reference selection: prefer the cell at 128k (production
            # deployment context), then 32k, then 1M, then 8k, then
            # longer ctxs. Whichever preferred ctx is present in the
            # history defines the canonical shape for the row.
            #
            # Rationale: at 8k the optimizer doesn't see KV pressure so
            # its shape picks ignore long-ctx considerations. At 128k it
            # sees real KV pressure and picks deployment-realistic
            # architectures. We use *that* shape as the canonical, then
            # re-evaluate it at every other ctx to show how the same
            # production architecture performs across contexts.
            history_by_ctx = {h[0]: h for h in history}
            ref_h = None
            for cand_ctx in preferred_reference_ctxs:
                if cand_ctx in history_by_ctx:
                    ref_h = history_by_ctx[cand_ctx]
                    break
            if ref_h is None:
                # Fallback: pick the lowest-ctx cell whose shape matches
                # the most neighbors (the previous heuristic).
                from collections import Counter
                d_counter = Counter(d for _, d, _, _ in history)
                mode_d = max(d_counter.items(),
                             key=lambda kv: (kv[1], -d_counter.most_common()[0][0]))[0]
                ref_entries = [h for h in history if h[1] == mode_d]
                ref_entries.sort(key=lambda h: h[0])
                ref_h = ref_entries[0]
            _ref_ctx, ref_d, ref_L, ref_f = ref_h
            mode_d = ref_d  # rename for compatibility with downstream code

            # Lazy-import quality model
            if _estimate_quality is None:
                import sys as _sys
                _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "ac"))
                try:
                    from quality_model import estimate_quality as _eq
                except ImportError:
                    from ac.quality_model import estimate_quality as _eq
                _estimate_quality = _eq

            # Re-evaluate the reference shape at every cell whose shape
            # differs. Use the reference family's existing rich fields
            # (n_heads, d_head, kv_bits, etc.) and only override d_model
            # and n_layers? Actually no — we keep the whole reference
            # shape constant. Only context_length changes.
            am, st = fkey
            for ctx, d, L, f in history:
                if d == mode_d and L == ref_L:
                    continue
                # Build a quality-arch config from the reference family.
                cfg = {
                    'd_model': ref_d, 'n_layers': ref_L,
                    'n_heads': ref_f.get('n_heads') or max(1, ref_d // 128),
                    'd_head': ref_f.get('d_head') or 128,
                    'n_kv_heads': ref_f.get('n_kv_heads') or max(1, (ref_f.get('n_heads') or 8) // 8),
                    'ffn_dim': ref_f.get('ffn_dim') or 4 * ref_d,
                    'vocab_size': 32000,
                    # Wave 5 follow-up bugfix (Jun 2026): preserve the
                    # reference family's actual attention_type and
                    # kv_precision so the re-eval matches the optimizer's
                    # picked architecture exactly. Hardcoding 'gqa' +
                    # 'bf16' made the pinned 128k loss ~9% lower than the
                    # optimizer's 8k loss for the same shape, because the
                    # optimizer's 8k pick had baked-in penalties from MLA
                    # or low-bit KV that the re-eval was discarding.
                    'attention_type': ref_f.get('attention_type') or 'gqa',
                    'weight_precision': ref_f.get('weight_prec') or 'bf16',
                    'kv_precision': _kv_bits_to_precision(
                        ref_f.get('kv_bits') or 16),
                    'activation_precision': 'bf16',
                    'model_type': 'dense',
                }
                # Wave 5 follow-up bugfix (Jun 2026): carry forward the
                # optimizer's architectural extras so the canonical re-eval
                # reconstructs the same penalty stack. Without these, the
                # re-eval discards MTP/MLA/YOCO/sparsity/first-K-dense
                # penalties and reads 5-10% lower than the optimizer.
                for key in ('mla_kv_latent_dim', 'mla_q_latent_dim',
                            'mla_rope_head_dim', 'mla_nope_head_dim',
                            'sparsity_2_4'):
                    v = ref_f.get(key)
                    if v is not None:
                        cfg[key] = v
                if ref_f.get('mtp_n_predict_depths', 0):
                    cfg['mtp_n_predict_depths'] = ref_f.get('mtp_n_predict_depths')
                    cfg['mtp_depth_n_layers'] = ref_f.get('mtp_depth_n_layers', 1)
                if ref_f.get('yoco_n_self_attn_layers', 0):
                    cfg['yoco_n_self_attn_layers'] = ref_f.get('yoco_n_self_attn_layers')
                if ref_f.get('n_dense_ffn_layers', 0):
                    cfg['n_dense_ffn_layers'] = ref_f.get('n_dense_ffn_layers')
                if am in ('moe', 'moe_hybrid'):
                    cfg['model_type'] = 'moe'
                    cfg['moe_config'] = {
                        'enabled': True,
                        'n_experts': ref_f.get('n_experts') or 32,
                        'top_k': ref_f.get('top_k') or 4,
                        'expert_dim': ref_f.get('expert_dim') or cfg['ffn_dim'],
                    }
                if am in ('hybrid', 'moe_hybrid'):
                    cfg['model_type'] = 'hybrid'
                    n_attn = max(1, ref_L // 8)
                    n_state = ref_L - n_attn
                    cfg['state_config'] = {
                        'enabled': True, 'state_type': st or 'mamba2',
                        'state_layers': n_state, 'attention_layers': n_attn,
                        'd_state': 192,
                    }
                training = {'training_tokens': int(t * 1e12),
                            'sequence_length': ctx}
                workload = {'context_length': ctx, 'task_type': 'general'}
                try:
                    q = _estimate_quality(cfg, training, workload)
                    # Overwrite the family entry's shape and loss with the
                    # canonical version. Keep TBT/mem unchanged because
                    # those reflect the actual optimizer-chosen layout.
                    f["loss_unpinned"] = f.get("loss")
                    f["d_model_unpinned"] = f.get("d_model")
                    f["n_layers_unpinned"] = f.get("n_layers")
                    f["loss"] = float(q.predicted_loss)
                    f["d_model"] = ref_d
                    f["n_layers"] = ref_L
                    f["shape_pinned"] = True
                    pinned_count += 1
                except Exception:
                    # Quality model couldn't evaluate the canonical shape
                    # at this ctx (rare; e.g. quality model raised on an
                    # extreme corner). Leave the original entry alone.
                    pass

        # Recompute delta_pct annotations on every cell whose family was
        # re-ranked by the pin. Same logic as in _build_family_rollup.
        for cell in cells_row:
            fams = cell.get("families", [])
            if not fams:
                continue
            _sort_families_canonical(fams)
            if fams:
                anchor_loss = fams[0].get("loss") or 1.0
                anchor_tbt = fams[0].get("tbt_ms") or 1.0
                for f in fams:
                    f["loss_delta_pct"] = round(
                        100.0 * (f.get("loss", 0) - fams[0]["loss"]) / max(anchor_loss, 1e-9), 2)
                    f["tbt_delta_pct"] = round(
                        100.0 * (f.get("tbt_ms", 0) - fams[0]["tbt_ms"]) / max(anchor_tbt, 1e-9), 1)

    data.setdefault("_family_smoothing", {})["canonical_shape_pins"] = pinned_count


def _annotate_plateau_marker(data: dict, plateau_band: float = 0.015) -> None:
    """Tag each cell's families[0] with a `plateau_with` field when a
    DIFFERENT arch_mode is within `plateau_band` of its loss.

    Wave 3 (Jun 2026 follow-up). The plateau marker lets the CLI / web
    show 'MoE / MoE-hyb' instead of pretending one is the unique winner
    when the two are inside the model's noise floor. Same-arch state
    variants (mamba2 vs gated_delta hybrid) are deduped: they're not a
    real plateau, they're just two ways to spell the same answer.
    """
    for cell in data.get("cells", []):
        fams = cell.get("families", [])
        if len(fams) < 2:
            continue
        # Wave 30: the marker goes on fams[0] — the DISPLAYED winner under
        # the canonical ordering — not on the min-loss entry (the two can
        # differ when the canonical key broke an in-band tie on
        # memory/TBT). The nearest different-arch_mode alternative is
        # still found by loss proximity, and the delta may now be
        # negative (the alternative can sit slightly BELOW the winner's
        # loss, inside the band), so the band check uses abs().
        top = fams[0]
        # Dedupe across arch_mode: keep the min-loss state variant per arch
        by_arch = {}
        for f in fams[1:]:
            am = f.get("arch_mode")
            if am == top.get("arch_mode"):
                continue
            if am not in by_arch or f.get("loss", 1e9) < by_arch[am].get("loss", 1e9):
                by_arch[am] = f
        if not by_arch:
            continue
        second = min(by_arch.values(), key=lambda f: f.get("loss", 1e9))
        l0 = top.get("loss", 0.0)
        l1 = second.get("loss", 0.0)
        if l0 > 0 and abs(l1 - l0) / l0 < plateau_band:
            top["plateau_with"] = {
                "arch_mode": second.get("arch_mode"),
                "state_type": second.get("state_type"),
                "loss": l1,
                "loss_delta_pct": round(100.0 * (l1 - l0) / l0, 2),
                "tbt_ms": second.get("tbt_ms"),
                "ttft_ms": second.get("ttft_ms"),
                "mem_gb": second.get("mem_gb"),
            }


def _smooth_family_flicker_2d(data: dict, uncertainty_window: float = 0.015,
                              target_params: set = None) -> None:
    """2D smoothing: own (hw, params, tokens) column × ±1 ctx neighborhood.

    Tighter than the 1D row-median pass because it does not pull in cells
    from distant ctxs — only the cell itself and its immediate ctx
    neighbors. Used for large-model columns (default {13, 120, 750}) where
    the 1D pass left residual plateau flicker.
    """
    from collections import defaultdict
    if target_params is None:
        target_params = {13, 120, 750}

    # Build a per-row ctx-ordered cell list
    by_row = defaultdict(list)
    for cell in data.get("cells", []):
        key = (cell.get("hw"), cell.get("params_B"), cell.get("tokens_T"))
        by_row[key].append(cell)
    for cells_row in by_row.values():
        cells_row.sort(key=lambda c: c.get("context_length", 0))

    smoothed = 0
    for (hw, p, t), cells_row in by_row.items():
        if int(p) not in target_params:
            continue
        for i, cell in enumerate(cells_row):
            fams = cell.get("families", [])
            if len(fams) < 2:
                continue
            # Build local neighborhood: prev, self, next within same row
            nbrs = cells_row[max(0, i-1):i+2]
            ranks = defaultdict(list)
            for nc in nbrs:
                # arch-dedupe
                by_arch = {}
                for f in nc.get("families", []):
                    am = f.get("arch_mode")
                    if am not in by_arch or f.get("loss", 1e9) < by_arch[am].get("loss", 1e9):
                        by_arch[am] = f
                ordered = sorted(by_arch.values(), key=lambda f: f.get("loss", 1e9))
                for rk, f in enumerate(ordered):
                    ranks[f.get("arch_mode")].append(rk)
            med = {k: sorted(v)[len(v)//2] for k, v in ranks.items() if v}

            # Re-sort own families by (pin, loss-bucket, neighborhood rank).
            pinned_fams = [
                f for f in fams
                if bool(f.get("multi_ctx_shape_pinned", False))
            ]
            rankable = pinned_fams or fams
            min_loss = min(f.get("loss", 1e9) for f in rankable)
            old_first = (fams[0].get("arch_mode"), fams[0].get("state_type"))
            def key(f):
                l = f.get("loss", 1e9)
                bucket = round((l - min_loss) / max(min_loss, 1e-9) / uncertainty_window)
                pin_bucket = (
                    0 if not pinned_fams
                    or f.get("multi_ctx_shape_pinned", False) else 1
                )
                return (pin_bucket, bucket,
                        med.get(f.get("arch_mode"), 99), l)
            fams.sort(key=key)
            new_first = (fams[0].get("arch_mode"), fams[0].get("state_type"))
            if old_first != new_first:
                smoothed += 1
            # Re-annotate deltas
            anchor_loss = fams[0].get("loss") or 1.0
            anchor_tbt = fams[0].get("tbt_ms") or 1.0
            for f in fams:
                f["loss_delta_pct"] = round(
                    100.0 * (f.get("loss", 0) - fams[0]["loss"]) / max(anchor_loss, 1e-9), 2)
                f["tbt_delta_pct"] = round(
                    100.0 * (f.get("tbt_ms", 0) - fams[0]["tbt_ms"]) / max(anchor_tbt, 1e-9), 1)

    # Mirror back to ONE carrier grid row per cell (Wave 30 — see
    # _stitch_families_to_carrier_rows for why not every sibling).
    _stitch_families_to_carrier_rows(data)

    sm = data.setdefault("_family_smoothing", {})
    sm["2d_target_params"] = sorted(target_params)
    sm["2d_cells_changed"] = smoothed


def _smooth_family_flicker(data: dict, uncertainty_window: float = 0.015) -> None:
    """Pin family ordering across the ctx axis when loss differences are
    inside the quality model's own uncertainty band.

    Wave 3 Step 3.3c (Jun 2026 redesign). The Pareto rollup picks the
    loss-min Pareto point per family at every cell independently, which
    means two families on the same plateau can flip rank between
    adjacent contexts purely due to discrete-shape rounding. Concretely:
    at (h100, 13B) the MoE-hybrid winner is loss=1.819 at 8k but the MoE
    winner is loss=1.943 at 32k while MoE-hybrid loss is 1.945 — a 0.1%
    gap that flips the displayed winner even though the two families
    are statistically indistinguishable.

    The fix is to use each family's MEDIAN rank across the contiguous
    ctx row as a stable tiebreaker. If a family's loss differs from the
    leader by less than `uncertainty_window` (default 1.5%, slightly
    above the model's ~1% sigma), it's considered tied and the median
    rank decides. This eliminates plateau-driven flicker while still
    letting genuine arch-family transitions (e.g. dense → hybrid at the
    long-context crossover) come through.
    """
    from collections import defaultdict

    by_row = defaultdict(list)
    for cell in data.get("cells", []):
        key = (cell.get("hw"), cell.get("params_B"), cell.get("tokens_T"))
        by_row[key].append(cell)

    smoothed_cells = 0
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: c.get("context_length", 0))

        # Family identity uses (arch_mode, state_type). Compute the median
        # rank for each family across this (hw, params, tokens) row.
        ranks_per_family = defaultdict(list)
        for cell in row_cells:
            for rank, f in enumerate(cell.get("families", [])):
                fkey = (f.get("arch_mode"), f.get("state_type"))
                ranks_per_family[fkey].append(rank)
        median_rank = {}
        for fkey, ranks in ranks_per_family.items():
            r = sorted(ranks)
            median_rank[fkey] = r[len(r) // 2] if r else 99

        # Re-sort each cell's families with the (loss-bucket, median-rank)
        # composite key. Then re-annotate the deltas to remain consistent.
        for cell in row_cells:
            fams = cell.get("families", [])
            if len(fams) < 2:
                continue
            pinned_fams = [
                f for f in fams
                if bool(f.get("multi_ctx_shape_pinned", False))
            ]
            rankable = pinned_fams or fams
            min_loss = min(f.get("loss", 1e9) for f in rankable)
            if min_loss <= 0:
                continue

            def composite_key(f):
                loss = f.get("loss", 1e9)
                # Bucket loss to the uncertainty grid so sub-noise gaps tie.
                delta_norm = (loss - min_loss) / max(min_loss, 1e-9)
                loss_bucket = round(delta_norm / uncertainty_window)
                fkey = (f.get("arch_mode"), f.get("state_type"))
                pin_bucket = (
                    0 if not pinned_fams
                    or f.get("multi_ctx_shape_pinned", False) else 1
                )
                return (pin_bucket, loss_bucket,
                        median_rank.get(fkey, 99), loss)

            old_first = (fams[0].get("arch_mode"), fams[0].get("state_type"))
            fams.sort(key=composite_key)
            new_first = (fams[0].get("arch_mode"), fams[0].get("state_type"))
            if old_first != new_first:
                smoothed_cells += 1

            anchor_loss = fams[0].get("loss") or 1.0
            anchor_tbt = fams[0].get("tbt_ms") or 1.0
            for f in fams:
                f["loss_delta_pct"] = round(
                    100.0 * (f["loss"] - fams[0]["loss"]) / max(anchor_loss, 1e-9), 2)
                f["tbt_delta_pct"] = round(
                    100.0 * (f["tbt_ms"] - fams[0]["tbt_ms"]) / max(anchor_tbt, 1e-9), 1)

        # Stitch the smoothed families back to every sibling grid row that
        # references this cell, mirroring _build_family_rollup's behavior.
        # We rebuild the grid->cell lookup once per call to keep cost linear.
    # After re-sorting all cells, mirror onto ONE carrier grid row per
    # cell (Wave 30) so the legacy `entry["families"]` view stays
    # consistent with cells[] without 9x-duplicating the list.
    _stitch_families_to_carrier_rows(data)

    data["_family_smoothing"] = {
        "uncertainty_window": uncertainty_window,
        "cells_with_winner_changed": smoothed_cells,
    }


def _harmonize_serving_train_metrics(data: dict) -> None:
    """Bug B fix: train_tps must not depend on serving budget.

    Group rows by (hw, params_B, tokens_T, context_length, arch_mode,
    state_type). Within a group, all rows describe the SAME training
    plan — the only difference is the serving SLO they were evaluated
    against. The optimizer picks different architectures under tighter
    serving budgets, which legitimately changes the serving metrics but
    should not change "how fast does this train" (the training step is
    not serving-budget-aware). Use the unconstrained entry as canonical
    for the architecture and training metrics; keep per-serving entries'
    own tbt/ttft/mem (those depend on the chosen serving batch size).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in data.get("grid", []):
        key = (
            row.get("hw"),
            row.get("params_B"),
            row.get("tokens_T"),
            row.get("context_length"),
            row.get("arch_mode"),
            row.get("state_type"),
        )
        groups[key].append(row)

    # Architecture and training fields are training-plan-determined.
    ARCH_TRAIN_FIELDS = (
        "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
        "ffn_dim", "weight_prec", "ffn_prec", "kv_bits", "params_B",
        "active_params_B", "arch_family", "moe_style",
        "loss", "spine_loss", "chinchilla", "spine_active_params_B",
        "total_residual_pct", "architecture_residual_pct",
        "precision_residual_pct", "risk_uncertainty_pct",
        "moe_residual_pct", "state_residual_pct", "penalty_pct",
        "dominant", "confidence",
        "uncertainty_low_pct", "uncertainty_high_pct", "uncertainty_total_pct",
        "train_tps", "attention_type",
        "n_experts", "top_k", "expert_dim", "ep", "shared_expert",
        "n_dense_ffn_layers",
    )
    harmonized = 0
    for key, rows in groups.items():
        # Find the canonical row: prefer unconstrained serving, else any
        # row whose `optimal` field is populated and not None.
        canonical = None
        for r in rows:
            if r.get("serving") == "unconstrained" and r.get("optimal"):
                canonical = r["optimal"]
                break
        if canonical is None:
            for r in rows:
                if r.get("optimal"):
                    canonical = r["optimal"]
                    break
        if canonical is None:
            continue
        for r in rows:
            opt = r.get("optimal")
            if opt is None:
                continue
            if opt is canonical:
                continue
            for f in ARCH_TRAIN_FIELDS:
                if f in canonical:
                    opt[f] = canonical[f]
            harmonized += 1
    if harmonized:
        print(f"[consistency] harmonized arch/train fields on {harmonized} "
              f"serving-variant rows ({len(groups)} training-plan groups)")


def _clamp_train_tps_monotone_in_params(data: dict) -> None:
    """Deprecated no-op retained for callers of the old generator API.

    A training replica is TP x PP x CP GPUs, so its TPS is not comparable
    across rows with different replica sizes. Independently clamping the raw,
    per-GPU, and aggregate fields fabricated values and broke their unit
    identities. Evaluated records must remain immutable after serialization.
    """
    return None


def _harmonize_loss_across_hw(data: dict) -> None:
    """Bug D fix: same architecture → same loss, regardless of hw.

    Loss is a function of (architecture, training tokens) — it does not
    depend on the hardware running the training. Group rows by (params_B,
    tokens_T, context_length, arch_mode, state_type) across hw and find
    the subset whose chosen architectures actually share a shape
    (d_model, n_layers, n_heads, weight_prec, kv_bits, MoE topology).
    Within that subset, overwrite quality fields with the lowest-loss
    representative; this collapses spurious hw-induced quality spread
    while leaving genuinely-different architectures untouched (different
    hw with different lattices CAN legitimately produce different
    architectures and different losses).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in data.get("grid", []):
        key = (
            row.get("params_B"),
            row.get("tokens_T"),
            row.get("context_length"),
            row.get("arch_mode"),
            row.get("state_type"),
            row.get("serving"),
        )
        groups[key].append(row)

    SHAPE_FIELDS = ("d_model", "n_layers", "n_heads", "d_head",
                    "n_kv_heads", "ffn_dim", "weight_prec", "kv_bits",
                    "n_experts", "top_k", "expert_dim",
                    "attention_type", "n_dense_ffn_layers")
    QUALITY_FIELDS = ("loss", "spine_loss", "chinchilla",
                      "total_residual_pct", "architecture_residual_pct",
                      "precision_residual_pct", "moe_residual_pct",
                      "state_residual_pct", "penalty_pct",
                      "uncertainty_total_pct")
    harmonized = 0
    for key, rows in groups.items():
        # Bucket by shape signature
        by_shape = defaultdict(list)
        for r in rows:
            opt = r.get("optimal")
            if opt is None:
                continue
            sig = tuple(opt.get(f) for f in SHAPE_FIELDS)
            by_shape[sig].append(opt)
        for sig, opts in by_shape.items():
            if len(opts) < 2:
                continue
            # Canonical = lowest loss representative (most-favorable
            # hw realization of this shape).
            canonical = min(opts, key=lambda o: float(o.get("loss", float("inf"))))
            for opt in opts:
                if opt is canonical:
                    continue
                for f in QUALITY_FIELDS:
                    if f in canonical:
                        opt[f] = canonical[f]
                harmonized += 1
    if harmonized:
        print(f"[consistency] harmonized quality across {harmonized} "
              f"cross-hw rows sharing architecture shape")


def _ctx_sweep_enabled() -> bool:
    """v1-fix UI: gate the multi-context sweep behind --ctx-sweep / CTX_SWEEP=1
    so the legacy 8192-only grid is preserved by default. The multi-context
    sweep is 5x slower and produces ~5x more data."""
    import os as _os
    return ("--ctx-sweep" in sys.argv) or (_os.environ.get("CTX_SWEEP") == "1")


def _multi_ctx_grid_enabled() -> bool:
    """Wave 6 (Jun 2026) — default ON as of this commit.

    The principled multi-ctx flow is now the grid driver's default. It makes
    ONE optimize_across_contexts call per (hw, params, tokens, smode,
    arch_mode), guaranteeing the shape is identical across the ctx row by
    construction — no post-hoc canonical-shape pin needed.

    Opt out via AC_GRID_LEGACY_PER_CELL=1 / --legacy-per-cell to fall back
    to the per-cell optimize() flow used before Wave 6. The legacy path is
    preserved for regression comparison and emergency rollback during the
    rollout; the post-hoc pin (_pin_canonical_shape_per_family) remains in
    the pipeline and becomes a no-op under the multi-ctx flow because shape
    drift is zero by construction.

    Backwards-compat for AC_GRID_MULTI_CTX=1 / --multi-ctx is preserved —
    those flags still force-enable the multi-ctx flow even when the legacy
    opt-out is set, which lets CI test both paths independently.
    """
    import os as _os
    force_multi = (
        "--multi-ctx" in sys.argv
        or _os.environ.get("AC_GRID_MULTI_CTX") == "1"
    )
    opt_out = (
        "--legacy-per-cell" in sys.argv
        or _os.environ.get("AC_GRID_LEGACY_PER_CELL") == "1"
    )
    if force_multi:
        return True
    return not opt_out


def _env_subset(env_key, default):
    """Comma-separated env-var subset override (numbers).

    Used to chunk a long sweep: e.g. PARAM_SUBSET=7.0,13.0 only generates
    those two parameter targets in this run, so a single invocation finishes
    within a reasonable wall-clock and the user can stitch multiple chunks.
    """
    import os as _os
    raw = _os.environ.get(env_key)
    if not raw:
        return default
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token) if "." in token else int(token))
        except ValueError:
            pass
    return out or default


def _parse_csv_list(raw, cast=str):
    """'a,b,c' -> [cast(a), cast(b), cast(c)]; None/'' -> None."""
    if not raw:
        return None
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(cast(tok))
    return out or None


def build_cli():
    """Wave 31: full CLI for the grid generator.

    Every design/sweep axis is a flag, so chunked or targeted regens are
    first-class instead of env-var folklore:

      # full regen (legacy behavior, writes compiler_data.json + index.html)
      python3 scripts/_generator_payload.py --multi-ctx --ctx-sweep

      # add the missing MoE-hybrid state families for one h100 column and
      # merge them into the shipped payload without touching other rows
      python3 scripts/_generator_payload.py --multi-ctx --ctx-sweep \
          --hardware h100 --params 7 --tokens 2.0 \
          --arch-modes moe_hybrid --state-types kda,gla,sliding_window \
          --merge-into ../v1-web/compiler-data-h100.json
    """
    parser = argparse.ArgumentParser(
        description="AC web-grid payload generator")
    parser.add_argument("--refresh-quality-only", action="store_true",
                        help="Reuse the existing web grid and refresh only "
                             "quality metadata.")
    # Sweep-axis subsets (default: full canonical lists).
    parser.add_argument("--hardware", type=str, default=None,
                        help=f"comma list, subset of {HARDWARE}")
    parser.add_argument("--params", type=str, default=None,
                        help=f"comma list (B), subset of {PARAM_TARGETS}")
    parser.add_argument("--tokens", type=str, default=None,
                        help=f"comma list (T), subset of {TOKEN_COUNTS}")
    parser.add_argument("--arch-modes", type=str, default=None,
                        help="comma list, subset of dense,moe,hybrid,moe_hybrid")
    parser.add_argument("--state-types", type=str, default=None,
                        help=f"comma list, subset of {STATE_FAMILIES} "
                             "(applies to hybrid / moe_hybrid modes)")
    parser.add_argument("--contexts", type=str, default=None,
                        help="comma list of context lengths; overrides the "
                             "--ctx-sweep gate")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="cap the per-search candidate pool after "
                             "dedupe (speed/quality knob; the shipped grid "
                             "used 400)")
    parser.add_argument("--moe-n-experts", type=str, default=None,
                        help=f"comma list of expert counts "
                             f"(default {GRID_MOE_N_EXPERTS})")
    parser.add_argument("--moe-top-k", type=str, default=None,
                        help=f"comma list of top-k (default {GRID_MOE_TOP_K})")
    parser.add_argument("--moe-granularity", type=str, default=None,
                        help=f"comma list of granularity targets "
                             f"(default {GRID_MOE_GRANULARITY}; 0.25 = "
                             "DeepSeek-style fine-grained)")
    parser.add_argument("--allow-compressed-attention", action="store_true",
                        help="let CSA / indexshare / MSA attention compete "
                             "inside every arch mode")
    parser.add_argument("--local-refine-budget", type=int, default=None,
                        help="Wave 34: extra full evaluations on lattice "
                             "neighbors of per-class Pareto leaders "
                             "(optimizer default 96; each costs one eval "
                             "PER CONTEXT in the multi-ctx flow)")
    # Flow gates (also honored via bare sys.argv checks for back-compat).
    parser.add_argument("--ctx-sweep", action="store_true",
                        help="sweep the full context list (else 8192 only)")
    parser.add_argument("--multi-ctx", action="store_true",
                        help="force the shape-coherent multi-ctx flow")
    parser.add_argument("--legacy-per-cell", action="store_true",
                        help="opt out of the multi-ctx flow")
    # Output routing.
    parser.add_argument("--out", type=str, default=None,
                        help="write the payload JSON (compact) to this path "
                             "instead of compiler_data.json + index.html")
    parser.add_argument("--merge-into", type=str, default=None,
                        help="merge generated grid rows into an existing "
                             "payload JSON (row-level replace), re-run the "
                             "post chain, and rewrite it in place")
    return parser


def main():
    args = build_cli().parse_args()
    base = os.path.dirname(__file__)

    if args.refresh_quality_only:
        print("Refreshing quality metadata in existing web compiler data...")
        in_path = os.path.join(base, "compiler_data.json")
        with open(in_path) as f:
            data = json.load(f)
        data = refresh_quality_metadata(data)
        write_data_outputs(data, base)
        print(f"  {len(data['grid'])} grid entries")
        return

    arch_modes = build_arch_modes(
        _parse_csv_list(args.arch_modes),
        _parse_csv_list(args.state_types),
    ) if (args.arch_modes or args.state_types) else None

    print("Generating web compiler data...")
    data = generate(
        hardware=_parse_csv_list(args.hardware),
        param_targets=_parse_csv_list(args.params, float),
        token_counts=_parse_csv_list(args.tokens, float),
        arch_modes=arch_modes,
        contexts=_parse_csv_list(args.contexts, int),
        max_candidates=args.max_candidates,
        moe_axis={
            "n_experts": _parse_csv_list(args.moe_n_experts, int),
            "top_k": _parse_csv_list(args.moe_top_k, int),
            "granularity": _parse_csv_list(args.moe_granularity, float),
        },
        allow_compressed=args.allow_compressed_attention,
        local_refine_budget=args.local_refine_budget,
    )
    print(f"  {len(data['grid'])} grid entries generated")

    if args.merge_into:
        with open(args.merge_into) as f:
            base_payload = json.load(f)
        merged = merge_payload(base_payload, data)
        run_post_chain(merged)
        with open(args.merge_into, "w") as f:
            json.dump(merged, f, separators=(",", ":"))
        print(f"Merged into {args.merge_into} "
              f"({len(merged['grid'])} total grid rows, "
              f"{len(merged.get('cells', []))} cells)")
    elif args.out:
        with open(args.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {args.out}")
    else:
        write_data_outputs(data, base)
        print(f"  {len(data['grid'])} grid entries")


if __name__ == "__main__":
    main()
