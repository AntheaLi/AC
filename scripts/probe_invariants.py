"""Wave 37: property-based invariant audit for AC's quality + throughput models.

Every check encodes a physical or function-class invariant that must hold
REGARDLESS of calibration. Violations are reported with values; this script
is diagnostic (the fixes get pinned in tests/test_wave37_fixes.py).

Usage: python3 scripts/probe_invariants.py [--section quality|throughput|all]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "ac"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optimizer import (  # noqa: E402
    DeploymentConstraints, evaluate_candidate, generate_candidates,
    generate_moe_candidates,
)

VIOLATIONS = []
CHECKS = [0]


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        VIOLATIONS.append((name, detail))
        print(f"  VIOLATION {name}: {detail}")


def _constraints(ctx=8192, tokens=2.0, **kw):
    base = dict(target_params_b=7.0, training_tokens=int(tokens * 1e12),
                context_length=ctx, tp=4, pp=1, dp=8,
                serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=16,
                vocab_size=32000, param_tolerance=0.10)
    base.update(kw)
    return DeploymentConstraints(**base)


def _dense_base(c):
    pool = generate_candidates("h100", c)
    full = [x for x in pool if x.attention_type == "full"
            and x.n_kv_heads not in (None, 0)]
    return full[len(full) // 3]


def _loss(base, c, **kw):
    return evaluate_candidate(replace(base, **kw) if kw else base,
                              "h100", c).predicted_loss


def _ev(base, c, **kw):
    return evaluate_candidate(replace(base, **kw) if kw else base, "h100", c)


def probe_quality():
    print("== quality-model invariants ==")
    c = _constraints()
    base = _dense_base(c)

    # --- Q1: KV-cache quantization: fewer bits => loss must not DECREASE.
    prev = None
    for bits in (16, 8, 4):
        L = _loss(base, c, kv_cache_bits=bits)
        if prev is not None:
            check("Q1.kv_bits_monotone", L >= prev - 1e-9,
                  f"kv{bits} loss {L:.5f} < kv{prev_bits} loss {prev:.5f}")
        prev, prev_bits = L, bits

    # --- Q2: weight/ffn precision: fp8 must not beat bf16 at same shape.
    L_bf16 = _loss(base, c)
    L_fp8 = _loss(base, c, weight_precision="fp8", ffn_precision="fp8")
    check("Q2.precision_monotone", L_fp8 >= L_bf16 - 1e-9,
          f"fp8 {L_fp8:.5f} < bf16 {L_bf16:.5f}")

    # --- Q3: training tokens: more data must not hurt.
    c4 = _constraints(tokens=4.0)
    check("Q3.tokens_monotone", _loss(base, c4) <= _loss(base, c) + 1e-9,
          f"4T {_loss(base, c4):.5f} > 2T {_loss(base, c):.5f}")

    # --- Q4: MSA pattern budget: bigger window/top-k must not hurt.
    msa_small = _loss(base, c, attention_type="msa", msa_window_size=256,
                      msa_dilated_top_k=32, msa_global_top_k=8)
    msa_big = _loss(base, c, attention_type="msa", msa_window_size=2048,
                    msa_dilated_top_k=128, msa_global_top_k=32)
    check("Q4.msa_budget_monotone", msa_big <= msa_small + 1e-9,
          f"bigger msa budget {msa_big:.5f} > smaller {msa_small:.5f}")

    # --- Q5: CSA top-k blocks: more blocks attended must not hurt.
    csa_lo = _loss(base, c, attention_type="csa", csa_block_size=64,
                   csa_top_k_blocks=8, csa_compression_dim=64)
    csa_hi = _loss(base, c, attention_type="csa", csa_block_size=64,
                   csa_top_k_blocks=32, csa_compression_dim=64)
    check("Q5.csa_topk_monotone", csa_hi <= csa_lo + 1e-9,
          f"top32 {csa_hi:.5f} > top8 {csa_lo:.5f}")

    # --- Q6: IndexShare coverage: more buckets attended must not hurt.
    idx_lo = _loss(base, c, attention_type="indexshare",
                   indexshare_num_buckets=64, indexshare_top_k_buckets=2,
                   indexshare_index_dim=64)
    idx_hi = _loss(base, c, attention_type="indexshare",
                   indexshare_num_buckets=64, indexshare_top_k_buckets=16,
                   indexshare_index_dim=64)
    check("Q6.indexshare_topk_monotone", idx_hi <= idx_lo + 1e-9,
          f"top16 {idx_hi:.5f} > top2 {idx_lo:.5f}")

    # --- Q7: function-class: every sparse type >= full at 8k, same shape.
    L_full = _loss(base, c)
    for at, kw in (
        ("msa", dict(msa_window_size=512, msa_dilated_top_k=64,
                     msa_global_top_k=16)),
        ("csa", dict(csa_block_size=64, csa_top_k_blocks=16,
                     csa_compression_dim=64)),
        ("indexshare", dict(indexshare_num_buckets=64,
                            indexshare_top_k_buckets=4,
                            indexshare_index_dim=64)),
        ("nsa", dict(nsa_compress_block_size=64, nsa_compress_block_stride=16,
                     nsa_select_block_size=64, nsa_select_top_k=16,
                     nsa_window_size=512)),
    ):
        L = _loss(base, c, attention_type=at, **kw)
        check(f"Q7.{at}_ge_full_at_8k", L >= L_full - 1e-9,
              f"{at} {L:.5f} < full {L_full:.5f}")

    # --- Q8: YOCO: sharing must cost quality; more self layers = closer
    # to full = lower loss.
    L_y1 = _loss(base, c, yoco_n_self_attn_layers=1)
    L_y4 = _loss(base, c, yoco_n_self_attn_layers=4)
    check("Q8a.yoco_costs_quality", L_y1 >= L_full - 1e-9,
          f"yoco1 {L_y1:.5f} < full {L_full:.5f}")
    check("Q8b.yoco_monotone_in_k", L_y4 <= L_y1 + 1e-9,
          f"yoco4 {L_y4:.5f} > yoco1 {L_y1:.5f}")

    # --- Q9: GQA heads: more KV heads (superset function class + more
    # params) must not hurt, comparing real generated shapes.
    c_pool = _constraints()
    pool = generate_candidates("h100", c_pool)
    by_kv = {}
    for x in pool:
        if x.attention_type != "full":
            continue
        key = (x.d_model, x.n_layers, x.n_heads, x.d_head, x.ffn_dim,
               x.kv_cache_bits, x.weight_precision)
        by_kv.setdefault(key, {})[x.n_kv_heads] = x
    tested = 0
    for key, variants in by_kv.items():
        kvs = sorted(variants)
        if len(kvs) < 2 or tested >= 3:
            continue
        losses = {kv: _loss(variants[kv], c_pool) for kv in kvs}
        for lo, hi in zip(kvs, kvs[1:]):
            check("Q9.gqa_kv_heads_monotone",
                  losses[hi] <= losses[lo] + 1e-9,
                  f"shape {key[:2]} kv{hi} {losses[hi]:.5f} > "
                  f"kv{lo} {losses[lo]:.5f}")
        tested += 1

    # --- Q10: 2:4 sparsity must cost quality.
    L_sp = _loss(base, c, sparsity_2_4={"ffn_up": True, "ffn_down": True,
                                        "ffn_gate": True})
    check("Q10.sparsity_costs_quality", L_sp >= L_full - 1e-9,
          f"2:4 {L_sp:.5f} < full {L_full:.5f}")

    # --- Q11: context length: with vanilla RoPE, longer ctx must not
    # IMPROVE loss (degradation + utility must net >= 0 vs 8k... utility
    # is a modeling choice; enforce only that 4M is not better than 8k
    # by more than the utility bound of 0).
    L_1m = _loss(base, _constraints(ctx=1048576))
    check("Q11.vanilla_rope_longctx_not_better", L_1m >= L_full - 1e-9,
          f"1M {L_1m:.5f} < 8k {L_full:.5f} under vanilla RoPE")


def probe_throughput():
    print("== throughput-model invariants ==")
    c = _constraints()
    base = _dense_base(c)

    ATTN = [
        ("full", {}),
        ("msa", dict(msa_window_size=512, msa_dilated_top_k=64,
                     msa_global_top_k=16)),
        ("csa", dict(csa_block_size=64, csa_top_k_blocks=16,
                     csa_compression_dim=64)),
        ("indexshare", dict(indexshare_num_buckets=64,
                            indexshare_top_k_buckets=4,
                            indexshare_index_dim=64)),
        ("nsa", dict(nsa_compress_block_size=64, nsa_compress_block_stride=16,
                     nsa_select_block_size=64, nsa_select_top_k=16,
                     nsa_window_size=512)),
    ]

    # --- T1: prefill time strictly increases with S (every attention type).
    for at, kw in ATTN:
        prev = None
        for ctx in (8192, 65536, 524288):
            ev = _ev(base, _constraints(ctx=ctx), attention_type=at, **kw)
            t = ev.throughput.prefill_time_ms
            if prev is not None:
                check(f"T1.prefill_monotone_S.{at}", t > prev * 0.999,
                      f"S={ctx} prefill {t:.1f}ms <= S={prev_ctx} {prev:.1f}ms")
            prev, prev_ctx = t, ctx

    # --- T2: sparse prefill <= full prefill at same shape and S=512k.
    cL = _constraints(ctx=524288)
    full_pref = _ev(base, cL).throughput.prefill_time_ms
    for at, kw in ATTN[1:]:
        t = _ev(base, cL, attention_type=at, **kw).throughput.prefill_time_ms
        check(f"T2.{at}_prefill_le_full", t <= full_pref * 1.001,
              f"{at} {t:.1f}ms > full {full_pref:.1f}ms at 512k")

    # --- T3: decode TBT non-decreasing in context for KV-carrying types.
    for at, kw in ATTN:
        prev = None
        for ctx in (8192, 131072, 1048576):
            ev = _ev(base, _constraints(ctx=ctx), attention_type=at, **kw)
            t = ev.serving_tbt_ms
            if prev is not None:
                check(f"T3.tbt_nondecreasing_ctx.{at}", t >= prev * 0.98,
                      f"ctx={ctx} TBT {t:.2f} < ctx={prev_ctx} TBT {prev:.2f}")
            prev, prev_ctx = t, ctx

    # --- T4: memory non-decreasing in batch.
    for b in ((8, 32),):
        m_lo = _ev(base, _constraints(serving_batch=b[0])).memory_per_gpu_gb
        m_hi = _ev(base, _constraints(serving_batch=b[1])).memory_per_gpu_gb
        check("T4.memory_monotone_batch", m_hi >= m_lo - 1e-6,
              f"batch{b[1]} {m_hi:.2f}GB < batch{b[0]} {m_lo:.2f}GB")

    # --- T5: memory non-decreasing in context (KV growth), full attention.
    m8 = _ev(base, _constraints(ctx=8192)).memory_per_gpu_gb
    m128 = _ev(base, _constraints(ctx=131072)).memory_per_gpu_gb
    check("T5.memory_monotone_ctx", m128 >= m8 - 1e-6,
          f"128k {m128:.2f}GB < 8k {m8:.2f}GB")

    # --- T6: YOCO reduces KV memory, never increases decode TBT beyond full.
    ev_full = _ev(base, c)
    ev_yoco = _ev(base, c, yoco_n_self_attn_layers=1)
    check("T6a.yoco_reduces_memory",
          ev_yoco.memory_per_gpu_gb <= ev_full.memory_per_gpu_gb + 1e-6,
          f"yoco {ev_yoco.memory_per_gpu_gb:.2f}GB > "
          f"full {ev_full.memory_per_gpu_gb:.2f}GB")
    check("T6b.yoco_tbt_not_worse",
          ev_yoco.serving_tbt_ms <= ev_full.serving_tbt_ms * 1.05,
          f"yoco TBT {ev_yoco.serving_tbt_ms:.2f} >> "
          f"full {ev_full.serving_tbt_ms:.2f}")

    # --- T7: TP sharding: TBT at TP=8 must not exceed TP=2 by more than
    # allreduce slack; memory per GPU must shrink.
    for at, kw in ATTN:
        e2 = _ev(base, _constraints(tp=2), attention_type=at,
                 tp_degree=2, **kw)
        e8 = _ev(base, _constraints(tp=8), attention_type=at,
                 tp_degree=8, **kw)
        check(f"T7a.tp_memory_shrinks.{at}",
              e8.memory_per_gpu_gb <= e2.memory_per_gpu_gb * 1.02,
              f"tp8 {e8.memory_per_gpu_gb:.2f}GB > tp2 "
              f"{e2.memory_per_gpu_gb:.2f}GB")
        check(f"T7b.tp_tbt_not_worse_4x.{at}",
              e8.serving_tbt_ms <= e2.serving_tbt_ms * 4.0,
              f"tp8 TBT {e8.serving_tbt_ms:.2f} vs tp2 "
              f"{e2.serving_tbt_ms:.2f}")

    # --- T8: MoE memory per GPU shrinks with EP.
    cm = _constraints(allow_moe=True, max_total_params_b=56,
                      moe_n_experts_options=[64], moe_top_k_options=[8],
                      moe_granularity_targets=[0.25], ep_options=[2])
    mpool = generate_moe_candidates("h100", cm)
    mbase = [x for x in mpool if x.attention_type == "full"][len(mpool) // 4]
    e1 = _ev(mbase, cm, ep_degree=1)
    e4 = _ev(mbase, cm, ep_degree=4)
    check("T8.moe_memory_shrinks_with_ep",
          e4.memory_per_gpu_gb <= e1.memory_per_gpu_gb + 1e-6,
          f"ep4 {e4.memory_per_gpu_gb:.2f}GB > ep1 "
          f"{e1.memory_per_gpu_gb:.2f}GB")

    # --- T9: quality must be invariant to parallelism (pure deployment).
    L_tp2 = _ev(base, _constraints(tp=2), tp_degree=2).predicted_loss
    L_tp8 = _ev(base, _constraints(tp=8), tp_degree=8).predicted_loss
    check("T9.loss_invariant_to_tp", abs(L_tp2 - L_tp8) < 1e-9,
          f"tp2 {L_tp2:.6f} != tp8 {L_tp8:.6f}")
    L_ep = _ev(mbase, cm, ep_degree=1).predicted_loss
    L_ep4 = _ev(mbase, cm, ep_degree=4).predicted_loss
    check("T9b.loss_invariant_to_ep", abs(L_ep - L_ep4) < 1e-9,
          f"ep1 {L_ep:.6f} != ep4 {L_ep4:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["quality", "throughput", "round2", "round3", "round4", "round5", "all"])
    args = ap.parse_args()
    if args.section in ("quality", "all"):
        probe_quality()
    if args.section in ("throughput", "all"):
        probe_throughput()
    if args.section in ("round2", "all"):
        probe_round2()
    if args.section in ("round3", "all"):
        probe_round3()
    if args.section in ("round4", "all"):
        probe_round4()
    if args.section in ("round5", "all"):
        probe_round5()
    print(f"\n{CHECKS[0]} checks, {len(VIOLATIONS)} violations")
    for name, detail in VIOLATIONS:
        print(f"  - {name}: {detail}")
    sys.exit(1 if VIOLATIONS else 0)




def probe_round2():
    """Deeper axes: interleave, MLA latents, MTP, batch/PP/CP, ledger, KV bytes.

    Deliberately NOT probed as an invariant: state-hybrid vs full at short
    ctx — hybrids matching or beating pure transformers at matched params
    is measured-empirical (Waleffe, Jamba), so hybrid <= full is NOT a
    function-class violation, unlike sparse-readout attention.
    """
    print("== round-2 invariants ==")
    c = _constraints()
    base = _dense_base(c)

    # --- R1: local:global interleave — bigger local window must not hurt
    # quality; and interleave must not beat full at 8k (windowed subset).
    L_full = _loss(base, c)
    n_local = max(1, base.n_layers // 2)
    L_w1k = _loss(base, c, n_local_attn_layers=n_local, swa_window=1024)
    L_w4k = _loss(base, c, n_local_attn_layers=n_local, swa_window=4096)
    check("R1a.interleave_window_monotone", L_w4k <= L_w1k + 1e-9,
          f"w4096 {L_w4k:.5f} > w1024 {L_w1k:.5f}")
    check("R1b.interleave_ge_full_at_8k", L_w1k >= L_full - 1e-9,
          f"interleave {L_w1k:.5f} < full {L_full:.5f} at 8k")

    # --- R2: MLA latent dim — bigger c_kv must not hurt.
    mla_kw = dict(attention_type="mla", mla_q_latent_dim=1536,
                  mla_rope_head_dim=64, mla_nope_head_dim=128)
    L_256 = _loss(base, c, mla_kv_latent_dim=256, **mla_kw)
    L_512 = _loss(base, c, mla_kv_latent_dim=512, **mla_kw)
    check("R2.mla_latent_monotone", L_512 <= L_256 + 1e-9,
          f"c_kv512 {L_512:.5f} > c_kv256 {L_256:.5f}")

    # --- R3: MTP effect bounded (report-only sign, hard-fail only if |d|>2%).
    L_mtp = _loss(base, c, mtp_n_predict_depths=1)
    check("R3.mtp_effect_bounded", abs(L_mtp - L_full) / L_full < 0.02,
          f"mtp1 {L_mtp:.5f} vs full {L_full:.5f}")
    print(f"  (info) MTP depth-1 delta: {(L_mtp - L_full) / L_full * 100:+.3f}%")

    # --- R4: per-request decode TBT non-decreasing in batch.
    t8 = _ev(base, _constraints(serving_batch=8)).serving_tbt_ms
    t64 = _ev(base, _constraints(serving_batch=64)).serving_tbt_ms
    check("R4.tbt_nondecreasing_batch", t64 >= t8 * 0.98,
          f"batch64 TBT {t64:.2f} < batch8 TBT {t8:.2f}")

    # --- R5: PP does not blow up prefill (bubble-bounded).
    e_pp1 = _ev(base, _constraints(pp=1))
    e_pp2 = _ev(base, _constraints(pp=2))
    check("R5.pp_prefill_bounded",
          e_pp2.throughput.prefill_time_ms
          <= e_pp1.throughput.prefill_time_ms * 2.0,
          f"pp2 {e_pp2.throughput.prefill_time_ms:.1f}ms > 2x pp1 "
          f"{e_pp1.throughput.prefill_time_ms:.1f}ms")

    # --- R6: CP reduces (or holds) long-ctx prefill.
    cL1 = _constraints(ctx=524288, cp=1)
    cL4 = _constraints(ctx=524288, cp=4)
    p1 = _ev(base, cL1, cp_degree=1).throughput.prefill_time_ms
    p4 = _ev(base, cL4, cp_degree=4).throughput.prefill_time_ms
    check("R6.cp_reduces_prefill", p4 <= p1 * 1.05,
          f"cp4 {p4:.1f}ms > cp1 {p1:.1f}ms at 512k")

    # --- R7: parameter-ledger identities.
    check("R7a.dense_total_eq_active",
          abs(base.total_params_b
              - float(getattr(base, "active_params_b", None)
                      or base.total_params_b)) < 0.05,
          f"dense total {base.total_params_b} != active")
    cm = _constraints(allow_moe=True, max_total_params_b=56,
                      moe_n_experts_options=[64], moe_top_k_options=[8],
                      moe_granularity_targets=[0.25], ep_options=[2])
    mpool = generate_moe_candidates("h100", cm)
    mbase = [x for x in mpool if x.attention_type == "full"][len(mpool) // 4]
    ev_m = _ev(mbase, cm)
    check("R7b.moe_total_gt_active",
          ev_m.arch.total_params_b
          > float(getattr(ev_m.quality, "n_active_params", 0) or 0) / 1e9
          or ev_m.arch.total_params_b > 7.0 * 1.5,
          f"moe total {ev_m.arch.total_params_b}B not > active")

    # --- R8: per-token KV bytes — sparse <= dense; int4 < bf16.
    from throughput_model import ArchConfig as TArch
    common = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=131072,
                  precision="bf16")
    dense_b = TArch(kv_precision="bf16", attention_type="full",
                    **common).kv_bytes_per_token_per_layer(131072)
    int4_b = TArch(kv_precision="int4", attention_type="full",
                   **common).kv_bytes_per_token_per_layer(131072)
    check("R8a.kv_bytes_int4_lt_bf16", int4_b < dense_b,
          f"int4 {int4_b} >= bf16 {dense_b}")
    for at, kw in (
        ("msa", dict(msa_window_size=512, msa_dilated_top_k=64,
                     msa_global_top_k=16)),
        ("csa", dict(csa_block_size=64, csa_top_k_blocks=16,
                     csa_compression_dim=64)),
        ("indexshare", dict(indexshare_num_buckets=64,
                            indexshare_top_k_buckets=4,
                            indexshare_index_dim=64)),
    ):
        b = TArch(kv_precision="bf16", attention_type=at, **common,
                  **kw).kv_bytes_per_token_per_layer(131072)
        check(f"R8b.kv_bytes_{at}_le_dense", b <= dense_b,
              f"{at} {b} > dense {dense_b}")



def probe_round3():
    """Robustness + consistency corners where past bugs hid."""
    print("== round-3 invariants ==")
    import math

    # --- X1: extreme inputs must not produce NaN/inf/nonpositive outputs.
    c = _constraints()
    base = _dense_base(c)
    extremes = [
        dict(ctx=4194304),                      # 4M ctx
        dict(ctx=2048),                          # tiny ctx
        dict(serving_batch=1),
        dict(serving_batch=256),
        dict(tokens=0.1),
        dict(tokens=30.0),
    ]
    for kw in extremes:
        ev = _ev(base, _constraints(**kw))
        vals = dict(loss=ev.predicted_loss, tbt=ev.serving_tbt_ms,
                    ttft=ev.throughput.prefill_time_ms,
                    mem=ev.memory_per_gpu_gb, tps=ev.training_tps)
        for k, v in vals.items():
            check(f"X1.finite_positive.{k}.{kw}",
                  v is not None and math.isfinite(float(v)) and float(v) > 0,
                  f"{k}={v} at {kw}")

    # --- X2: tp > n_kv_heads must still be finite and not FASTER than
    # tp == n_kv_heads by more than the shard limit (KV can't shard past
    # its head count).
    e_kv = _ev(base, _constraints(tp=base.n_kv_heads),
               tp_degree=base.n_kv_heads)
    e_big = _ev(base, _constraints(tp=base.n_kv_heads * 2),
                tp_degree=base.n_kv_heads * 2)
    check("X2.tp_past_kv_heads_finite",
          math.isfinite(e_big.serving_tbt_ms) and e_big.serving_tbt_ms > 0,
          f"tbt={e_big.serving_tbt_ms}")

    # --- X3: rank-1 == selected under declared budgets (desync class).
    from optimizer import optimize, build_display_sort_key
    cb = _constraints(ctx=131072, serving_tbt_ms=30.0, tp=8,
                      max_candidates=120, local_refine_budget=0)
    r = optimize("h100", cb)
    if r.optimal is not None and r.pareto_frontier:
        key = build_display_sort_key(r.pareto_frontier, cb)
        rank1 = sorted(r.pareto_frontier, key=key)[0]
        check("X3.rank1_is_selected",
              rank1.arch is r.optimal.arch
              or abs(rank1.predicted_loss - r.optimal.predicted_loss) < 1e-9,
              f"rank1 loss {rank1.predicted_loss:.5f} != selected "
              f"{r.optimal.predicted_loss:.5f}")
        check("X3b.selected_meets_declared_budget",
              r.optimal.serving_tbt_ms <= 30.0 + 1e-6
              or all(e.serving_tbt_ms > 30.0 for e in r.pareto_frontier),
              f"selected TBT {r.optimal.serving_tbt_ms:.1f} > 30ms budget "
              f"while budget-meeting candidates exist")

    # --- X4: loss must be deterministic (two evaluations agree).
    L1 = _loss(base, c)
    L2 = _loss(base, c)
    check("X4.deterministic", L1 == L2, f"{L1} != {L2}")

def probe_round4():
    """Per-architecture ctx monotonicity for EVERY attention type.

    Generalizes Q11: at a fixed shape (and vanilla RoPE), a longer
    context is a strictly harder task — predicted loss must be
    non-decreasing in ctx for full, mla, nsa, csa, indexshare, and msa
    alike. Wave 42 was caught by this: IndexShare's loss was ctx-FLAT
    (scale-free coverage prior with no indexer-routing risk), so past
    ~1.5M it undercut every ctx-penalized family and the displayed row
    loss DROPPED from 1M to 2M."""
    print("== round-4 invariants ==")
    c = _constraints()
    base = _dense_base(c)
    variants = [
        ("full", {}),
        ("mla", dict(mla_kv_latent_dim=512, mla_q_latent_dim=1536,
                     mla_rope_head_dim=64, mla_nope_head_dim=128)),
        ("nsa", dict(nsa_compress_block_size=64, nsa_compress_block_stride=16,
                     nsa_select_block_size=64, nsa_select_top_k=16,
                     nsa_window_size=512)),
        ("csa", dict(csa_block_size=64, csa_top_k_blocks=16,
                     csa_compression_dim=64)),
        ("indexshare", dict(indexshare_num_buckets=64,
                            indexshare_top_k_buckets=4,
                            indexshare_index_dim=64)),
        ("msa", dict(msa_window_size=512, msa_dilated_top_k=64,
                     msa_global_top_k=16)),
    ]
    for at, kw in variants:
        prev = None
        for ctx in (8192, 131072, 1048576, 2097152):
            L = _loss(base, _constraints(ctx=ctx), attention_type=at, **kw)
            if prev is not None:
                check(f"R4.ctx_monotone.{at}", L >= prev - 1e-9,
                      f"ctx={ctx} loss {L:.5f} < ctx={prev_ctx} {prev:.5f}")
            prev, prev_ctx = L, ctx

def probe_round5():
    """Cross-hardware invariance, config round-trip identity, train/prefill
    coupling — the remaining unprobed surfaces."""
    print("== round-5 invariants ==")
    import math
    c = _constraints()
    base = _dense_base(c)

    # --- H1: predicted loss is hardware-invariant (quality ⊥ hardware).
    variants = [
        ("full", {}),
        ("mla", dict(mla_kv_latent_dim=512, mla_q_latent_dim=1536,
                     mla_rope_head_dim=64, mla_nope_head_dim=128)),
        ("msa", dict(msa_window_size=512, msa_dilated_top_k=64,
                     msa_global_top_k=16)),
    ]
    for at, kw in variants:
        cand = replace(base, attention_type=at, **kw) if kw else base
        try:
            l_h = evaluate_candidate(cand, "h100", c).predicted_loss
            l_b = evaluate_candidate(cand, "b200", c).predicted_loss
            check(f"H1.loss_hw_invariant.{at}", abs(l_h - l_b) < 1e-9,
                  f"h100 {l_h:.6f} != b200 {l_b:.6f}")
        except Exception as e:
            check(f"H1.loss_hw_invariant.{at}", False, f"raised: {e}")

    # --- H2: b200 must not be SLOWER than h100 at the same config
    # (2.3x flops, 2.4x bandwidth: prefill and decode should both win).
    ev_h = _ev(base, c)
    ev_b = evaluate_candidate(base, "b200", c)
    check("H2a.b200_prefill_faster",
          ev_b.throughput.prefill_time_ms <= ev_h.throughput.prefill_time_ms
          * 1.01,
          f"b200 prefill {ev_b.throughput.prefill_time_ms:.1f}ms > h100 "
          f"{ev_h.throughput.prefill_time_ms:.1f}ms")
    check("H2b.b200_tbt_faster",
          ev_b.serving_tbt_ms <= ev_h.serving_tbt_ms * 1.01,
          f"b200 TBT {ev_b.serving_tbt_ms:.2f}ms > h100 "
          f"{ev_h.serving_tbt_ms:.2f}ms")
    check("H2c.b200_train_tps_higher",
          ev_b.training_tps >= ev_h.training_tps * 0.99,
          f"b200 tps {ev_b.training_tps:.0f} < h100 {ev_h.training_tps:.0f}")

    # --- H3: config round-trip identity. Emit the schema config for an
    # evaluated candidate, load it through the baseline loader, re-evaluate
    # the loaded candidate, and the loss must match the direct evaluation
    # (loader/bridge must not drop any quality-relevant field).
    import json as _json
    import tempfile
    from types import SimpleNamespace
    from optimizer import result_to_config
    from baseline import load_baseline_model
    for at, kw in variants:
        cand = replace(base, attention_type=at, **kw) if kw else base
        try:
            ev = evaluate_candidate(cand, "h100", c)
            fake_result = SimpleNamespace(
                optimal=ev, constraints=c, hardware="h100",
                pareto_frontier=[ev], all_evaluated=[ev],
                candidates_generated=1, candidates_feasible=1,
                candidates_evaluated=1,
                search_time_sec=0.0, binding_constraints=[],
                shadow_prices=None, contending_family=None,
                candidates_enumerated_raw=1)
            cfg = result_to_config(fake_result)
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".json", delete=False) as f:
                _json.dump(cfg, f)
                path = f.name
            bm = load_baseline_model(path)
            ev2 = evaluate_candidate(bm.candidate, "h100", c)
            check(f"H3.roundtrip_loss.{at}",
                  abs(ev2.predicted_loss - ev.predicted_loss)
                  < 5e-4 * ev.predicted_loss,
                  f"direct {ev.predicted_loss:.5f} vs roundtrip "
                  f"{ev2.predicted_loss:.5f}")
        except Exception as e:
            check(f"H3.roundtrip_loss.{at}", False,
                  f"round-trip raised: {type(e).__name__}: {e}")

    # --- H4: training/prefill coupling. prefill_time_ms is PER-REQUEST
    # (single-user TTFT semantics), so the per-request prefill token rate
    # is ctx / t. Training processes fwd+bwd (+recompute); its token rate
    # must sit below the per-request prefill rate but within a physical band.
    prefill_tps = (c.context_length
                   / max(1e-9, ev_h.throughput.prefill_time_ms / 1e3))
    ratio = ev_h.training_tps / max(1e-9, prefill_tps)
    check("H4.train_vs_prefill_band", 0.05 <= ratio <= 1.0,
          f"training_tps/prefill_tps ratio {ratio:.3f} outside "
          f"[0.05, 1.0] (train {ev_h.training_tps:.0f}, per-request "
          f"prefill {prefill_tps:.0f} tok/s)")


if __name__ == "__main__":
    main()
