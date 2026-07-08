"""Stress vector and shadow-price diagnostics.

(Pins from Waves 18h, 19, 21, and 29.)

  * The stress model is MLA-aware: an MLA arch's KV traffic uses the
    compressed latent (unsharded), not the baseline GQA formula.
  * KV-precision deltas are visible to the transition quality panel
    (kv int4 shows a precision_loss delta; the justification does not
    claim "negligible").
  * `ac-stress` runs on every shipped config (parallelism threaded), and
    the schema adapter preserves interleave / MLA / MTP / CP fields.
  * MoE decode weight traffic is activation-aware (distinct-experts
    bound), mirroring the throughput fix.
  * Shadow-price perturbation re-runs are stride-sample capped with the
    BASE re-optimized under the same cap, and carry a sampling_note.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
_AC_DIR = os.path.join(REPO, "ac")
if _AC_DIR not in sys.path:
    sys.path.insert(0, _AC_DIR)

from ac.throughput_model import ArchConfig as TArch  # noqa: E402
from ac.stress import Workload, compute_throughput_stress, _kv_cache_bytes_total  # noqa: E402
from ac.delta_engine import apply_transitions  # noqa: E402


def test_stress_kv_bytes_mla_aware():
    common = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=32768)
    gqa = TArch(**common)
    mla = TArch(**common, attention_type="mla", mla_kv_latent_dim=512,
                mla_rope_head_dim=64)
    kv_len = 32768
    gqa_tp8 = _kv_cache_bytes_total(gqa, kv_len, tp_degree=8)
    mla_tp8 = _kv_cache_bytes_total(mla, kv_len, tp_degree=8)
    # MLA latent per token per layer: (512+64)*2 bytes, unsharded -> 1152 B.
    expect_mla = 1152 * kv_len * 32
    assert abs(mla_tp8 - expect_mla) / expect_mla < 0.01
    # At TP=8 the sharded GQA-8 cache is SMALLER than the unsharded latent —
    # exactly the case the old GQA-only formula got backwards.
    assert mla_tp8 > gqa_tp8
    # And unsharded (TP=1) MLA must be far smaller than unsharded GQA.
    assert _kv_cache_bytes_total(mla, kv_len, 1) < 0.3 * _kv_cache_bytes_total(gqa, kv_len, 1)


def test_kv_precision_delta_visible_in_transition_quality():
    baseline = TArch(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                     n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=2048,
                     kv_precision="bf16")
    transitions = apply_transitions(
        baseline,
        [("change_precision_per_component", {"kv": "int4"})],
        hardware="h100", tp_degree=8,
    )
    t = transitions[0]
    assert t.feasible
    delta_prec = t.delta_quality.get("precision_loss", 0.0)
    assert delta_prec > 1e-4, (
        "int4 KV swap must surface a precision_loss delta in the transition "
        f"quality panel (got {delta_prec}); the summary previously claimed "
        "'negligible' while the metric table showed +1.2% loss")
    try:
        from ac.justify_transition import justify
    except ImportError:
        from justify_transition import justify
    text = justify(t)
    assert "negligible" not in text.lower()


class TestStressOnShippedConfigs:
    @pytest.mark.parametrize("cfg", sorted(
        glob.glob(os.path.join(REPO, "configs", "*.json"))),
        ids=lambda p: os.path.basename(p))
    def test_ac_stress_runs_on_shipped_config(self, cfg):
        out = subprocess.run(
            [sys.executable, os.path.join(REPO, "ac", "cli_stress.py"),
             "stress", "--baseline-config", cfg, "--hardware", "h100",
             "--phase", "decode"],
            capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, (
            f"ac-stress failed on shipped config {os.path.basename(cfg)}: "
            f"{out.stderr[-400:]}")
        assert "hbm_bw_decode" in out.stdout

    def test_stress_schema_adapter_preserves_gpt_oss_interleave(self):
        from ac.cli_stress import _archconfig_from_schema_v03

        with open(os.path.join(REPO, "configs", "gpt_oss_120b.json")) as f:
            cfg = json.load(f)
        arch = _archconfig_from_schema_v03(cfg, batch=1, seq=2048)
        assert arch.vocab_size == 200000
        assert arch.n_local_attn_layers == 18
        assert arch.local_window == 128
        assert arch.layer_type_list.count("local_attention") == 18
        assert arch.moe_config["n_experts"] == 128

    def test_stress_schema_adapter_preserves_mai_mla_mtp_cp(self):
        from ac.cli_stress import _archconfig_from_schema_v03

        with open(os.path.join(REPO, "configs", "mai_thinking_1.json")) as f:
            cfg = json.load(f)
        arch = _archconfig_from_schema_v03(cfg, batch=1, seq=2048)
        assert arch.vocab_size == 152064
        assert arch.attention_type == "mla"
        assert arch.mla_kv_latent_dim == 512
        assert arch.mla_q_latent_dim == 1536
        assert arch.mtp_n_predict_depths == 1
        assert arch.cp_degree == 4


class TestMoeDecodeStressTraffic:
    def test_moe_decode_bandwidth_not_charged_all_resident_experts(self):
        def _moe_arch(batch=1):
            return TArch(
                d_model=2880, n_layers=36, n_heads=64, d_head=64, n_kv_heads=8,
                ffn_dim=2880, batch_size=batch, seq_len=2048,
                moe_config={"n_experts": 128, "top_k": 4, "expert_dim": 2880},
            )

        arch = _moe_arch(batch=1)
        sv = compute_throughput_stress(
            arch, "h100",
            workload=Workload(batch_size=1, prefill_seq_len=2048,
                              decode_kv_len=2048, phase="decode"),
            tp_degree=8, ep_degree=8, arch_name="stress-moe",
        )
        # At b=1 top-4-of-128, decode streams only the touched expert
        # slice; the old accounting charged all resident experts and
        # reported 2.1x ("violated") on this shape.
        assert sv.hbm_bw_decode < 1.0
        # Monotone in batch: more distinct experts touched at b=64.
        sv64 = compute_throughput_stress(
            _moe_arch(batch=64), "h100",
            workload=Workload(batch_size=64, prefill_seq_len=2048,
                              decode_kv_len=2048, phase="decode"),
            tp_degree=8, ep_degree=8, arch_name="stress-moe64",
        )
        assert sv64.intermediates.get(
            "decode_weight_bytes", sv64.intermediates["weight_bytes"]) > \
            sv.intermediates.get("decode_weight_bytes", 0.0)


class TestShadowPriceCap:
    """Perturbation re-runs are capped with a like-for-like base."""

    @classmethod
    def setup_class(cls):
        from optimizer import optimize, DeploymentConstraints
        cls._DeploymentConstraints = DeploymentConstraints
        cls._optimize = optimize
        cls._constraints = DeploymentConstraints(
            target_params_b=1.0,
            training_tokens=int(1e12),
            tp=8, tp_options=[8],
            max_candidates=120,
        )
        cls._base = optimize("h100", cls._constraints)
        assert cls._base.optimal is not None

    def test_small_capped_search_gets_no_note(self):
        from shadow_prices import compute_shadow_prices
        rep = compute_shadow_prices("h100", self._constraints, self._base)
        assert rep.sampling_note == ""
        assert rep.prices

    def test_uncapped_search_is_capped_with_note_and_consistent_base(self):
        import shadow_prices as sp
        import copy
        cons = copy.deepcopy(self._constraints)
        cons.max_candidates = 0  # what the CLI default (unbounded) passes
        old_cap = sp.SHADOW_PERTURBATION_CANDIDATE_CAP
        sp.SHADOW_PERTURBATION_CANDIDATE_CAP = 80
        try:
            rep = sp.compute_shadow_prices("h100", cons, self._base)
        finally:
            sp.SHADOW_PERTURBATION_CANDIDATE_CAP = old_cap
        assert "stride-sampled at 80" in rep.sampling_note
        assert "uncapped" in rep.sampling_note
        assert rep.prices
        # Like-for-like: every price's original_loss equals the CAPPED
        # base loss (deltas never mix uncapped and capped optima).
        for p in rep.prices:
            assert abs(p.original_loss - rep.base_loss) < 1e-4
        # JSON carries the note.
        from shadow_prices import shadow_prices_to_json
        d = shadow_prices_to_json(rep)
        assert "sampling_note" in d
