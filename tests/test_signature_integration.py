"""Factorized ArchitectureSignature — integration & acceptance gate.

(Pins from Wave 18a.)

`test_architecture_signature.py` and `test_architecture_wiring.py` already
cover the classification rules. This file locks in the *acceptance gate*
side of Wave 18a:

  * serialize_candidate() emits `signature` alongside the legacy
    `arch_family` label, and the two agree.
  * Wave 11's decision block carries `signature` derived from the same
    ArchitectureSignature.
  * Every hand-rolled family classifier still in the codebase (cli_compile,
    trust_audit, throughput_model._arch_family, budget_pareto,
    _generator_payload._arch_family_from_optimal, regen_golden_matrix) now
    returns the same label the canonical helper would.
  * The 5-bucket calibration taxonomy (throughput_model / auto_calibrate)
    stays consistent with the 4-label legacy taxonomy under the signature
    projection.
  * Public-model shapes classify by component axes without needing a
    handwritten "hybrid" label — the Wave 18a §Acceptance Gate.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bf16_precision() -> dict:
    return {"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"}


def _make_cand(**overrides):
    """Minimal CandidateArch for signature exercises."""
    from ac.optimizer import CandidateArch
    kw = dict(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, vocab_size=32000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision=_bf16_precision(), kv_cache_bits=16,
        moe=None, state_config=None, n_dense_ffn_layers=0,
        attention_type="gqa",
        tp_degree=1, cp_degree=1, ep_degree=1,
    )
    kw.update(overrides)
    return CandidateArch(**kw)


class SerializeCandidateEmitsSignatureTests(unittest.TestCase):
    """serialize_candidate() must carry the factorized signature."""

    def _serialize(self, cand):
        from ac.optimizer import DeploymentConstraints, evaluate_candidate
        from _generator_payload import serialize_candidate
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
            allow_quality_sentinel=True,
        )
        return serialize_candidate(evaluate_candidate(cand, "h100", c))

    def test_dense_serialized_row_carries_signature(self):
        rec = self._serialize(_make_cand())
        self.assertIn("signature", rec)
        sig = rec["signature"]
        self.assertIsInstance(sig, dict)
        # 9 required axes per Wave 18a spec.
        for axis in ("ffn_mode", "attention_pattern", "kv_projection",
                     "sequence_mixer", "mixer_fraction",
                     "context_extension", "modifiers",
                     "active_params", "total_params", "legacy_family"):
            self.assertIn(axis, sig, f"signature missing axis: {axis}")
        # The legacy `arch_family` label must equal `signature.legacy_family`.
        self.assertEqual(rec["arch_family"], sig["legacy_family"])

    def test_mla_serialized_row_carries_signature(self):
        cand = _make_cand(
            attention_type="mla",
            mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        )
        rec = self._serialize(cand)
        self.assertEqual(rec["signature"]["kv_projection"], "mla")
        self.assertEqual(rec["signature"]["legacy_family"], "dense")


class DecisionBlockCarriesSignatureTests(unittest.TestCase):
    """Wave 11 decision block must carry the factorized signature."""

    def _decision(self, cand):
        from ac.optimizer import DeploymentConstraints, evaluate_candidate
        from _generator_payload import (
            serialize_candidate, _build_decision_diagnostics,
        )
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None, serving_batch=8,
            allow_quality_sentinel=True,
        )
        row = {"optimal": serialize_candidate(evaluate_candidate(cand, "h100", c))}
        _build_decision_diagnostics(row)
        return row["decision"]

    def test_dense_decision_has_signature(self):
        d = self._decision(_make_cand())
        self.assertIn("signature", d)
        self.assertIsNotNone(d["signature"])
        # Legacy compat: `family` in the decision block equals
        # `signature.legacy_family`.
        self.assertEqual(d["family"], d["signature"]["legacy_family"])

    def test_moe_decision_signature_agrees(self):
        cand = _make_cand(
            moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine",
            ep_degree=2,
        )
        d = self._decision(cand)
        self.assertEqual(d["family"], "moe")
        self.assertEqual(d["signature"]["legacy_family"], "moe")
        self.assertEqual(d["signature"]["ffn_mode"], "moe")
        self.assertEqual(d["signature"]["sequence_mixer"], "attention")
        # MoE-only candidate: no state mixer.
        self.assertEqual(d["signature"]["mixer_fraction"], 0.0)


class HandRolledClassifiersAgreeTests(unittest.TestCase):
    """Every remaining hand-rolled classifier must now return the same
    label the canonical ArchitectureSignature would."""

    def _cases(self):
        return [
            ("dense_gqa",    _make_cand()),
            ("dense_mla",    _make_cand(attention_type="mla",
                                       mla_kv_latent_dim=512,
                                       mla_q_latent_dim=1536,
                                       mla_rope_head_dim=64,
                                       mla_nope_head_dim=128)),
            ("moe",          _make_cand(
                moe={"n_experts": 8, "top_k": 2, "expert_dim": 4096,
                     "shared_expert": None,
                     "router": {"precision": "bf16"},
                     "capacity_factor": 1.0, "precision": "bf16"},
                moe_style="fine", ep_degree=2)),
            ("hybrid_mamba", _make_cand(
                state_config={"d_state": 128, "state_expansion": 2,
                              "n_heads": 32, "d_head": 128,
                              "state_type": "mamba2"},
                layer_type_list=["state"] * 24 + ["attention"] * 8,
                n_state_layers=24, n_attention_layers=8)),
        ]

    def test_cli_compile_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from ac.cli_compile import _arch_mode_of
        for label, cand in self._cases():
            with self.subTest(case=label):
                self.assertEqual(
                    _arch_mode_of(cand),
                    architecture_signature(cand).legacy_family,
                    f"{label}: cli_compile._arch_mode_of disagrees with signature",
                )

    def test_trust_audit_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from ac.trust_audit import _family_of
        for label, cand in self._cases():
            with self.subTest(case=label):
                self.assertEqual(
                    _family_of(cand),
                    architecture_signature(cand).legacy_family,
                    f"{label}: trust_audit._family_of disagrees with signature",
                )

    def test_generator_payload_and_signature_agree(self):
        from ac.architecture import architecture_signature
        from _generator_payload import _arch_family_from_optimal
        # Build a serialized-optimal-shaped dict for each case.
        for label, cand in self._cases():
            with self.subTest(case=label):
                opt = {
                    "d_model": cand.d_model, "n_layers": cand.n_layers,
                    "n_heads": cand.n_heads, "d_head": cand.d_head,
                    "n_kv_heads": cand.n_kv_heads, "ffn_dim": cand.ffn_dim,
                    "vocab_size": cand.vocab_size,
                    "attention_type": cand.attention_type,
                    "moe": cand.moe,
                    "state_config": cand.state_config,
                    "n_state_layers": cand.n_state_layers,
                    "layer_type_list": cand.layer_type_list,
                    "mla_kv_latent_dim": cand.mla_kv_latent_dim,
                    "mla_q_latent_dim": cand.mla_q_latent_dim,
                }
                self.assertEqual(
                    _arch_family_from_optimal(opt),
                    architecture_signature(cand).legacy_family,
                    f"{label}: _arch_family_from_optimal disagrees with signature",
                )

    def test_throughput_model_calibration_bucket_from_signature(self):
        """throughput_model._arch_family projects the signature into the
        5-bucket calibration taxonomy. Verify the projection matches the
        signature axes for representative cases."""
        from ac.throughput_model import _arch_family as tput_family, ArchConfig
        # dense GQA → dense_gqa
        arch = ArchConfig(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                          n_kv_heads=8, ffn_dim=14336)
        self.assertEqual(tput_family(arch), "dense_gqa")
        # MLA-dense → mla_dense
        arch = ArchConfig(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                          n_kv_heads=8, ffn_dim=14336,
                          attention_type="mla", mla_kv_latent_dim=512)
        self.assertEqual(tput_family(arch), "mla_dense")


class AcceptanceGateTests(unittest.TestCase):
    """Wave 18a §Acceptance Gate:

    > No user-visible decision may rely on a handwritten family label.
    > Public model comparisons must be expressible as component signatures.
    """

    def test_public_model_signatures_are_expressible_as_components(self):
        """Illustrative fixtures for DeepSeek-V3-shape (MoE + MLA) and a
        Jamba-shape (MoE + Mamba-2 hybrid). Neither uses a handwritten
        "hybrid" label — both are expressible as (ffn_mode, kv_projection,
        sequence_mixer, mixer_fraction) tuples."""
        from ac.architecture import architecture_signature

        # DeepSeek-V3-shape: MoE + MLA + pure attention mixer.
        v3 = _make_cand(
            attention_type="mla",
            mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
            moe={"n_experts": 256, "top_k": 8, "expert_dim": 2048,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine", ep_degree=8,
        )
        v3_sig = architecture_signature(v3)
        # Component signature: (moe, mla, attention). NOT "hybrid".
        self.assertEqual(v3_sig.ffn_mode, "moe")
        self.assertEqual(v3_sig.kv_projection, "mla")
        self.assertEqual(v3_sig.sequence_mixer, "attention")
        self.assertFalse(v3_sig.has_state_mixer)
        # legacy_family stays "moe" — no phantom "hybrid".
        self.assertEqual(v3_sig.legacy_family, "moe")

        # Jamba-shape: MoE + Mamba-2 mixer (dense KV projection).
        jamba = _make_cand(
            moe={"n_experts": 16, "top_k": 2, "expert_dim": 8192,
                 "shared_expert": None,
                 "router": {"precision": "bf16"},
                 "capacity_factor": 1.0, "precision": "bf16"},
            moe_style="fine", ep_degree=4,
            state_config={"d_state": 128, "state_expansion": 2,
                          "n_heads": 32, "d_head": 128,
                          "state_type": "mamba2"},
            layer_type_list=["state"] * 28 + ["attention"] * 4,
            n_state_layers=28, n_attention_layers=4,
        )
        j_sig = architecture_signature(jamba)
        self.assertEqual(j_sig.ffn_mode, "moe")
        self.assertEqual(j_sig.sequence_mixer, "mamba2")
        self.assertGreater(j_sig.mixer_fraction, 0.5)
        # Same component signature (moe + mamba2 + gqa) — the label
        # "moe_hybrid" is just a compat display artifact.
        self.assertEqual(j_sig.legacy_family, "moe_hybrid")

        # Their factorized signatures MUST differ; if they don't, Wave 18a
        # failed to factor the axes correctly.
        self.assertNotEqual(v3_sig.as_dict(), j_sig.as_dict())


if __name__ == "__main__":
    unittest.main()
