"""Wave 44 follow-up: fill `arch_dim_prices` on the 20T web payload.

The multi-ctx grid flow never computed the architecture-dimension
perturbation table (the legacy per-cell flow gated it on the retired
2T/8k anchor slice), so the regenerated 20T payload shipped with the
demo's "Architecture Dimension Perturbation" tab empty everywhere.

This filler reconstructs each 8k-context anchor row's winning candidate
from the raw shard files (richer `optimal` dicts than the slimmed web
payload), re-evaluates it, and — only when the re-evaluated loss matches
the stored loss within `--tolerance` (default 2%), i.e. the
reconstruction is faithful — computes the ~8 single-dimension
perturbations via `shadow_prices.compute_arch_dim_shadow_prices` and
injects them into v1-web/compiler-data.json under the `arch_dim_prices`
key app.js reads. Rows that fail reconstruction keep no table and the
demo falls back to the nearest sibling with data (same UX as the old
payload's partial coverage).

Usage:
  python3 scripts/fill_arch_dim_prices.py [--budget 30] [--payload PATH]
Repeat until it prints ALL-DONE (resumable; progress in the work dir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "ac"), _ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optimizer import (  # noqa: E402
    CandidateArch, DeploymentConstraints, evaluate_candidate)
from shadow_prices import compute_arch_dim_shadow_prices  # noqa: E402

WORK = os.environ.get(
    "AC_REGEN20T_DIR", os.path.join(os.path.dirname(_ROOT), ".regen20t"))
SHARD_DIR = os.path.join(WORK, "shards")
PROGRESS = os.path.join(WORK, "arch_dim_fill.json")
ANCHOR_CTX = 8192


def _row_key(r):
    return (r.get("hw"), r.get("params_B"), r.get("tokens_T"),
            r.get("context_length"), r.get("serving"),
            r.get("arch_mode"), r.get("state_type"))


def _rebuild_candidate(o, decision):
    """CandidateArch from a shard row's optimal dict (+ decision.shape)."""
    shape = (decision or {}).get("shape") or {}
    vocab = int(o.get("vocab_size") or shape.get("vocab_size") or 32000)
    kw = dict(
        d_model=int(o["d_model"]), n_layers=int(o["n_layers"]),
        n_heads=int(o["n_heads"]), d_head=int(o["d_head"]),
        n_kv_heads=int(o.get("n_kv_heads") or o["n_heads"]),
        ffn_dim=int(o["ffn_dim"]), vocab_size=vocab,
        weight_precision=o.get("weight_prec", "bf16"),
        ffn_precision=o.get("ffn_prec", "bf16"),
        kv_cache_bits=int(o.get("kv_bits", 16)),
    )
    if o.get("n_experts"):
        kw["moe"] = {
            "n_experts": int(o["n_experts"]),
            "top_k": int(o.get("top_k") or 2),
            "expert_dim": int(o.get("expert_dim") or o["ffn_dim"]),
        }
        # `shared_expert` in the shard optimal is a bool flag; the schema
        # wants a dict. Shard rows don't serialize the shared dim, so use
        # expert_dim as the conventional shared size — the loss self-check
        # below rejects the row if that guess is materially wrong.
        if o.get("shared_expert"):
            kw["moe"]["shared_expert"] = {
                "ffn_dim": int(o.get("shared_expert_dim")
                               or o.get("expert_dim") or o["ffn_dim"])}
        if o.get("n_dense_ffn_layers"):
            kw["n_dense_ffn_layers"] = int(o["n_dense_ffn_layers"])
        if o.get("ep"):
            kw["ep_degree"] = int(o["ep"])
    at = o.get("attention_type") or "full"
    if at == "mla" and o.get("mla_kv_latent_dim"):
        kw.update(attention_type="mla",
                  mla_kv_latent_dim=int(o["mla_kv_latent_dim"]),
                  mla_q_latent_dim=int(o.get("mla_q_latent_dim") or 1536),
                  mla_rope_head_dim=int(o.get("mla_rope_head_dim") or 64),
                  mla_nope_head_dim=int(o.get("mla_nope_head_dim") or 128))
    elif at not in ("full", "gqa", "mha"):
        # Compressed / windowed variants (msa, csa, swa, ...) carry
        # per-variant params the shard row may not serialize completely;
        # attempt the label and let the loss self-check decide.
        kw["attention_type"] = at
        for k in ("msa_window_size", "msa_dilated_top_k", "msa_global_top_k",
                  "csa_block_size", "csa_top_k_blocks", "csa_compression_dim",
                  "nsa_compress_block_size", "nsa_compress_block_stride",
                  "nsa_select_block_size", "nsa_select_top_k",
                  "nsa_window_size"):
            if o.get(k) is not None:
                kw[k] = int(o[k])
    if o.get("swa_window"):
        kw["swa_window"] = int(o["swa_window"])
    if o.get("n_local_attn_layers"):
        kw["n_local_attn_layers"] = int(o["n_local_attn_layers"])
    return CandidateArch(**kw)


def _constraints_for(row):
    return DeploymentConstraints(
        target_params_b=float(row["params_B"]),
        training_tokens=int(float(row["tokens_T"]) * 1e12),
        context_length=int(row["context_length"]),
        serving_tbt_ms=float(row.get("serving_tbt_budget_ms") or 50.0),
        serving_ttft_ms=float(row.get("serving_ttft_budget_ms") or 2000.0),
        serving_batch=int(row.get("serving_batch") or 32),
        tp=int(row.get("tp") or 8), pp=int(row.get("pp") or 1),
        dp=int(row.get("dp") or 8),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--payload", default=os.path.join(
        os.path.dirname(_ROOT), "v1-web", "compiler-data.json"))
    args = ap.parse_args()

    prog = {}
    if os.path.exists(PROGRESS):
        prog = json.load(open(PROGRESS))

    # Anchor rows come from the raw shards (full optimal dicts).
    todo = []
    for fn in sorted(os.listdir(SHARD_DIR)):
        if not fn.endswith(".json"):
            continue
        data = json.load(open(os.path.join(SHARD_DIR, fn)))
        for r in data.get("grid", []):
            if r.get("context_length") != ANCHOR_CTX:
                continue
            if not r.get("optimal"):
                continue
            todo.append(r)

    t0 = time.time()
    results = {k: v for k, v in prog.items()}
    ran = 0
    for row in todo:
        key = json.dumps(_row_key(row))
        if key in results:
            continue
        if time.time() - t0 > args.budget:
            break
        o = row["optimal"]
        try:
            cand = _rebuild_candidate(o, row.get("decision"))
            con = _constraints_for(row)
            hw = row["hw"]
            base_ev = evaluate_candidate(cand, hw, con)
            stored = float(o.get("loss") or 0.0)
            rel = (abs(base_ev.predicted_loss - stored) / stored
                   if stored > 0 else 1.0)
            if rel > args.tolerance:
                results[key] = {"status": "skip",
                                "reason": f"reconstruction drift {rel:.3f}"}
            else:
                prices = compute_arch_dim_shadow_prices(
                    hw, con, SimpleNamespace(optimal=base_ev))
                results[key] = {"status": "ok", "prices": [
                    {"dimension": a.dimension, "change": a.change_desc,
                     "base_value": a.base_value,
                     "perturbed_value": a.perturbed_value,
                     "delta_loss_pct": a.delta_loss_pct,
                     "delta_train_tps_pct": a.delta_train_tps_pct,
                     "delta_tbt_pct": a.delta_tbt_pct,
                     "delta_mem_pct": a.delta_mem_pct,
                     "decision": a.decision, "reason": a.reason,
                     "feasible": a.feasible} for a in prices]}
        except Exception as exc:  # keep going; row just stays uncovered
            results[key] = {"status": "skip",
                            "reason": f"{type(exc).__name__}: {exc}"[:200]}
        ran += 1
        json.dump(results, open(PROGRESS, "w"))

    done = len(results)
    print(f"PROGRESS {done}/{len(todo)} anchors ({ran} this window)")
    if done < len(todo):
        return 0

    # All anchors processed: inject into the payload and report coverage.
    payload = json.load(open(args.payload))
    filled = skipped = 0
    by_key = {json.dumps(_row_key(r)): r for r in payload.get("grid", [])}
    for key, rec in results.items():
        row = by_key.get(key)
        if row is None:
            continue
        if rec["status"] == "ok" and rec.get("prices"):
            row["arch_dim_prices"] = rec["prices"]
            filled += 1
        else:
            skipped += 1
    json.dump(payload, open(args.payload, "w"), separators=(",", ":"))
    print(f"ALL-DONE filled={filled} skipped={skipped} -> {args.payload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
