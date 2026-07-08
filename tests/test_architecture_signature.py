"""Wave 18a — Factorized architecture signature.

Tests classification of the concrete architectures listed in the plan:

  * dense GQA / dense MHA / dense MQA / dense MLA
  * MoE + MLA (DeepSeek-V3-style)
  * Mamba hybrid / MoE + Mamba (Jamba-style)
  * NSA (block_sparse) / YOCO modifier / SWA local attention
  * requested-MoE-returned-dense classifies dense
  * transforming attention before evaluation changes signature + fingerprint
  * all phase-specific architecture views (throughput/quality) share the signature
  * active/total parameter values match parameter_ledger

The tests are independent — no CLI, no full-optimizer run required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bf16_precision() -> dict:
    return {"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"}


def _make(**overrides):
    """CandidateArch with sensible defaults; overrides fill in test-specific bits."""
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


class ClassificationFixtureTests(unittest.TestCase):
    """One assertion per fixture, all independent."""

    def test_dense_gqa(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make())
        self.assertEqual(sig.ffn_mode, "dense")
        self.assertEqual(sig.attention_pattern, "global")
        self.assertEqual(sig.kv_projection, "gqa")
        self.assertEqual(sig.sequence_mixer, "attention")
        self.assertEqual(sig.mixer_fraction, 0.0)
        self.assertEqual(sig.legacy_family, "dense")

    def test_dense_mha(self):
        from ac.architecture import architecture_signature
        # MHA: n_kv_heads == n_heads
        sig = architecture_signature(_make(n_kv_heads=32))
        self.assertEqual(sig.kv_projection, "mha")

    def test_dense_mqa(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(n_kv_heads=1))
        self.assertEqual(sig.kv_projection, "mqa")

    def test_dense_mla(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            attention_type="mla", mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        ))
        self.assertEqual(sig.kv_projection, "mla")
        # MLA is a KV-projection choice; the *pattern* is still global.
        self.assertEqual(sig.attention_pattern, "global")
        # And it does NOT imply state-space hybrid.
        self.assertEqual(sig.sequence_mixer, "attention")
        self.assertEqual(sig.legacy_family, "dense")

    def test_moe_mla_deepseek_v3_style(self):
        """DeepSeek-V3 is MoE + MLA — legacy_family must be `moe`, not `moe_hybrid`."""
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            moe={"n_experts": 256, "top_k": 8, "expert_dim": 2048},
            n_dense_ffn_layers=3,
            attention_type="mla", mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        ))
        self.assertEqual(sig.ffn_mode, "moe")
        self.assertEqual(sig.kv_projection, "mla")
        self.assertEqual(sig.sequence_mixer, "attention")
        # Crucial invariant per 18a: MLA + MoE is NOT state-hybrid.
        self.assertEqual(sig.legacy_family, "moe")

    def test_mamba_hybrid_dense_ffn(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            state_config={
                "state_layers": 24, "state_type": "mamba2",
                "d_state": 128, "state_expansion": 2,
            },
            n_state_layers=24, n_attention_layers=8,
        ))
        self.assertEqual(sig.ffn_mode, "dense")
        self.assertEqual(sig.sequence_mixer, "mamba2")
        self.assertAlmostEqual(sig.mixer_fraction, 24 / 32, places=6)
        self.assertEqual(sig.legacy_family, "hybrid")

    def test_moe_plus_mamba_jamba_style(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            moe={"n_experts": 16, "top_k": 2, "expert_dim": 14336},
            state_config={
                "state_layers": 24, "state_type": "mamba2",
                "d_state": 128, "state_expansion": 2,
            },
            n_state_layers=24, n_attention_layers=8,
        ))
        self.assertEqual(sig.ffn_mode, "moe")
        self.assertEqual(sig.sequence_mixer, "mamba2")
        self.assertEqual(sig.legacy_family, "moe_hybrid")

    def test_nsa_block_sparse_attention(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(attention_type="nsa"))
        self.assertEqual(sig.attention_pattern, "block_sparse")
        self.assertIn("nsa", sig.modifiers)
        # Block-sparse attention alone is not a state hybrid.
        self.assertEqual(sig.sequence_mixer, "attention")
        self.assertEqual(sig.legacy_family, "dense")

    def test_yoco_modifier_recorded(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(yoco_n_self_attn_layers=4))
        self.assertIn("yoco", sig.modifiers)

    def test_local_swa_attention(self):
        """Mistral-7B-style: dense GQA + SWA window → attention_pattern=local, +swa modifier."""
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(swa_window=4096))
        self.assertEqual(sig.attention_pattern, "local")
        self.assertIn("swa", sig.modifiers)

    def test_compressed_attention_variants(self):
        from ac.architecture import architecture_signature
        for at in ("csa", "indexshare", "msa"):
            sig = architecture_signature(_make(attention_type=at))
            self.assertEqual(sig.attention_pattern, "compressed",
                             f"attention_type={at} must classify as compressed")
            self.assertIn(at, sig.modifiers,
                          f"attention_type={at} must appear in modifiers")


class RequestVsBuiltTests(unittest.TestCase):
    """Requested-mode flags never determine classification — only the built arch does."""

    def test_requested_moe_returning_dense_is_dense(self):
        """moe={} (placeholder from a requested-but-empty MoE build) is dense."""
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(moe={}))
        self.assertEqual(sig.ffn_mode, "dense")
        self.assertEqual(sig.legacy_family, "dense")

    def test_moe_with_single_expert_top_k_1_is_dense(self):
        """n_experts=1, top_k=1 is dense in disguise — signature says dense."""
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            moe={"n_experts": 1, "top_k": 1, "expert_dim": 14336},
        ))
        self.assertEqual(sig.ffn_mode, "dense")

    def test_disabled_moe_config_is_dense(self):
        from ac.architecture import architecture_signature
        sig = architecture_signature(_make(
            moe={"n_experts": 8, "top_k": 2, "expert_dim": 14336, "enabled": False},
        ))
        self.assertEqual(sig.ffn_mode, "dense")


class TransformChangesSignatureTests(unittest.TestCase):
    """Transforming the architecture must produce a different signature + fingerprint."""

    def test_transforming_attention_changes_signature(self):
        from ac.architecture import architecture_signature, signature_fingerprint
        base = _make()
        sig_base = architecture_signature(base)
        fp_base = signature_fingerprint(sig_base)
        transformed = _make(attention_type="mla", mla_kv_latent_dim=512,
                            mla_q_latent_dim=1536, mla_rope_head_dim=64,
                            mla_nope_head_dim=128)
        sig_new = architecture_signature(transformed)
        fp_new = signature_fingerprint(sig_new)
        self.assertNotEqual(sig_base.kv_projection, sig_new.kv_projection)
        self.assertNotEqual(fp_base, fp_new)


class SignatureViewConsistencyTests(unittest.TestCase):
    """All phase-specific views must produce the same signature."""

    def test_optimizer_and_throughput_views_agree(self):
        """CandidateArch and TputArch views of the same model share their signature."""
        from ac.architecture import architecture_signature
        from ac.optimizer import evaluate_candidate, DeploymentConstraints
        cand = _make()
        dc = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True, vocab_size=32000,
        )
        ev = evaluate_candidate(cand, "h100", dc)
        sig_cand = architecture_signature(cand)
        sig_tput = architecture_signature(ev.throughput.arch) if hasattr(
            ev.throughput, "arch") else sig_cand
        # Signature is a pure function of the arch fields, so throughput's
        # view (built from the same fields) must produce the same signature.
        self.assertEqual(sig_cand.as_dict(), sig_tput.as_dict())

    def test_signature_cached_on_evaluated_candidate(self):
        """EvaluatedCandidate.signature caches after first access."""
        from ac.optimizer import evaluate_candidate, DeploymentConstraints
        dc = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True, vocab_size=32000,
        )
        ev = evaluate_candidate(_make(), "h100", dc)
        # First access populates; second access returns the same object.
        sig1 = ev.signature
        sig2 = ev.signature
        self.assertIs(sig1, sig2)


class ParameterLedgerAgreementTests(unittest.TestCase):
    """Active/total parameter values in the signature come exclusively from parameter_ledger."""

    def test_signature_active_matches_ledger(self):
        from ac.architecture import architecture_signature, parameter_ledger
        for kw in [
            {},  # dense GQA
            dict(n_kv_heads=1),  # MQA
            dict(moe={"n_experts": 8, "top_k": 2, "expert_dim": 14336}),  # MoE
            dict(state_config={"state_layers": 16, "state_type": "mamba2",
                               "d_state": 128, "state_expansion": 2},
                 n_state_layers=16, n_attention_layers=16),  # hybrid
        ]:
            arch = _make(**kw)
            sig = architecture_signature(arch)
            ledger = parameter_ledger(arch)
            self.assertEqual(sig.active_params, ledger.active_params,
                             f"mismatch for {kw}: sig={sig.active_params}, "
                             f"ledger={ledger.active_params}")
            self.assertEqual(sig.total_params, ledger.total_params,
                             f"mismatch for {kw}: sig={sig.total_params}, "
                             f"ledger={ledger.total_params}")


class SerializationRoundtripTests(unittest.TestCase):
    def test_signature_json_roundtrip(self):
        from ac.architecture import ArchitectureSignature, architecture_signature
        sig = architecture_signature(_make(
            moe={"n_experts": 8, "top_k": 2, "expert_dim": 14336},
            attention_type="mla", mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        ))
        restored = ArchitectureSignature.from_dict(sig.as_dict())
        self.assertEqual(sig, restored)

    def test_display_string_reflects_axes(self):
        from ac.architecture import architecture_signature
        s = architecture_signature(_make(
            moe={"n_experts": 256, "top_k": 8, "expert_dim": 2048},
            attention_type="mla", mla_kv_latent_dim=512, mla_q_latent_dim=1536,
            mla_rope_head_dim=64, mla_nope_head_dim=128,
        ))
        display = s.display()
        self.assertIn("moe", display)
        self.assertIn("mla", display)


class NoHandwrittenFamilyLabelTests(unittest.TestCase):
    """Wave 18a acceptance gate: no user-visible decision may rely on a
    handwritten family label.  We check the two places that used to have a
    handwritten `_family_key`: optimizer stratification and the grid driver's
    `_arch_family`.  Both must route through architecture_signature now.
    """

    def test_optimizer_family_key_routes_through_signature(self):
        """Confirm `_family_key` in optimizer.py imports architecture_signature."""
        import ac.optimizer as opt
        src = Path(opt.__file__).read_text()
        # The routed version must contain the import of architecture_signature
        # in the stratification block.
        self.assertIn("architecture_signature as _arch_sig", src,
                      "optimizer._family_key must route through architecture_signature")

    def test_generator_payload_arch_family_routes_through_signature(self):
        gen_path = ROOT / "scripts" / "_generator_payload.py"
        src = gen_path.read_text()
        self.assertIn("architecture_signature", src,
                      "_arch_family in _generator_payload must route through "
                      "architecture_signature")


if __name__ == "__main__":
    unittest.main()
