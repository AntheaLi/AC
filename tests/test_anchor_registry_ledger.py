"""Wave 19 (loop finding L2): ledger gate for the public-anchor registry.

The trust audit is only as honest as its reference data. Three anchor
entries shipped with FABRICATED architectures (qwen3-235b-a22b computed
626B total for a declared 235B; kimi-k2 1414B for 1000B; jamba 621B for
398B) — the qwen entry alone explained most of its +239% decode-TBT
"model error". Every anchor's architecture block must reproduce its own
declared parameter counts, the same gate shipped configs already pass.

Hybrid (state-layer) anchors are exempt from the ACTIVE check: the flat
ledger below does not book mamba/state params (noted in the entry).
"""

import json
import os
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

REGISTRY = os.path.join(os.path.dirname(__file__), "fixtures",
                        "public_model_anchors_v1.json")
TOL = 0.10  # publication rounding + embedding-convention slack


def _flat_ledger(a):
    d, L = a["d_model"], a["n_layers"]
    h, dh, kv, V = a["n_heads"], a["d_head"], a["n_kv_heads"], a["vocab_size"]
    attn = d * dh * h + 2 * d * dh * kv + dh * h * d
    if a.get("attention_type") == "mla":
        c_kv = a.get("mla_kv_latent_dim", 512)
        c_q = a.get("mla_q_latent_dim", 1536)
        attn = (d * (c_kv + 64) + c_kv * h * dh * 2
                + d * c_q + c_q * h * (dh + 64) + h * dh * d)
    moe = a.get("moe")
    if moe:
        e, n, k = moe["expert_dim"], moe["n_experts"], moe["top_k"]
        shared = 0
        if isinstance(moe.get("shared_expert"), dict):
            shared = int(moe["shared_expert"].get("ffn_dim", 0) or 0)
        ffn_tot = n * 3 * d * e + 3 * d * shared
        ffn_act = k * 3 * d * e + 3 * d * shared
    else:
        ffn_tot = ffn_act = 3 * d * a["ffn_dim"]
    emb = 2 * V * d
    return ((attn + ffn_tot) * L + emb) / 1e9, ((attn + ffn_act) * L + emb) / 1e9


def _entries():
    with open(REGISTRY) as f:
        return json.load(f)["entries"]


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e["id"])
def test_anchor_arch_reproduces_declared_params(entry):
    arch = entry["arch"]
    total_l, active_l = _flat_ledger(arch)
    decl_t = entry.get("total_params_b")
    decl_a = entry.get("active_params_b")
    is_hybrid = bool(arch.get("state_config"))
    if decl_t:
        rel = abs(total_l - decl_t) / decl_t
        assert rel <= TOL, (
            f"{entry['id']}: declared total {decl_t}B but arch computes "
            f"{total_l:.0f}B ({rel*100:.0f}% off) — fabricated anchor "
            "architectures corrupt every audit metric derived from them")
    if decl_a and not is_hybrid:
        rel = abs(active_l - decl_a) / decl_a
        assert rel <= 0.15, (
            f"{entry['id']}: declared active {decl_a}B but arch computes "
            f"{active_l:.1f}B ({rel*100:.0f}% off)")
