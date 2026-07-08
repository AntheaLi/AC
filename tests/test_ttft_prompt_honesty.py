"""TTFT serving-stack floor and prompt-length honesty.

(Pins from Waves 19 and 29.)

  * `--context-length` overrides cascade to `prompt_len`; prompt > ctx is
    a hard error; TTFT is monotone in context.
  * TTFT carries a calibratable serving-stack floor (tokenize + scheduler
    admission + sampler + detokenize; EXCLUDES load-dependent queueing).
    The floor scales with prompt length, is inside the reported figure,
    and calibrating it to zero removes exactly the attributed overhead.
"""

import os
import re
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
_AC_DIR = os.path.join(REPO, "ac")
if _AC_DIR not in sys.path:
    sys.path.insert(0, _AC_DIR)


class TestPromptLength:
    def _delta_cmd(self, extra):
        return [sys.executable, os.path.join(REPO, "ac", "cli_delta_eval.py"),
                "--baseline-config", os.path.join(REPO, "configs", "mistral_7b.json"),
                "--hardware", "h100", "--tp", "8", "--workload", "long_context",
                "--apply", "swap_attention_to_gqa:group_size=8",
                "--stdout"] + extra

    def test_context_override_cascades_to_prompt(self):
        out = subprocess.run(self._delta_cmd(["--context-length", "16384"]),
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr
        assert "prompt_len: 16384" in out.stdout

    def test_prompt_exceeding_context_errors(self):
        out = subprocess.run(
            self._delta_cmd(["--context-length", "16384",
                             "--prompt-len", "32768"]),
            capture_output=True, text=True, cwd=REPO)
        assert out.returncode != 0
        assert "exceeds context_length" in out.stderr

    def test_ttft_monotone_in_context(self):
        ttfts = []
        for cl in ("16384", "65536", "131072"):
            out = subprocess.run(self._delta_cmd(["--context-length", cl]),
                                 capture_output=True, text=True, cwd=REPO)
            assert out.returncode == 0, out.stderr
            m = re.search(r"Prefill / TTFT \(ms\) \| ([0-9.]+)", out.stdout)
            assert m, "TTFT row missing"
            ttfts.append(float(m.group(1)))
        assert ttfts[0] < ttfts[1] < ttfts[2], (
            f"TTFT must grow with context; got {ttfts}")


class TestTtftServingFloor:
    """TTFT includes an attributable, calibratable serving-stack floor
    that excludes queueing."""

    def _arch(self, seq_len=8192):
        from throughput_model import ArchConfig
        return ArchConfig(
            d_model=4096, n_layers=32, n_heads=32, d_head=128,
            n_kv_heads=8, ffn_dim=14336, batch_size=8, seq_len=seq_len,
        )

    def test_floor_present_and_scales_with_prompt(self):
        from throughput_model import (
            throughput,
            DEFAULT_TTFT_FIXED_OVERHEAD_MS,
            DEFAULT_TTFT_PER_PROMPT_TOKEN_US,
        )
        r1k = throughput(self._arch(), "h100", tp_degree=8,
                         prefill_seq_len=1024)
        r8k = throughput(self._arch(), "h100", tp_degree=8,
                         prefill_seq_len=8192)
        exp_1k = (DEFAULT_TTFT_FIXED_OVERHEAD_MS
                  + DEFAULT_TTFT_PER_PROMPT_TOKEN_US * 1024 / 1000.0)
        exp_8k = (DEFAULT_TTFT_FIXED_OVERHEAD_MS
                  + DEFAULT_TTFT_PER_PROMPT_TOKEN_US * 8192 / 1000.0)
        assert abs(r1k.ttft_serving_overhead_ms - exp_1k) < 1e-3
        assert abs(r8k.ttft_serving_overhead_ms - exp_8k) < 1e-3
        # The floor is INSIDE the reported prefill/TTFT figure.
        assert r1k.prefill_time_ms > r1k.ttft_serving_overhead_ms
        # TTFT remains monotone in prompt length with the floor applied.
        assert r8k.prefill_time_ms > r1k.prefill_time_ms

    def test_floor_is_calibratable_to_zero(self):
        import throughput_model as tm
        real = tm.load_hardware

        def _zero_floor(name):
            hw = real(name)
            hw.calibration = dict(hw.calibration)
            hw.calibration["ttft_serving_overhead"] = {
                "fixed_ms": 0.0, "per_prompt_token_us": 0.0}
            return hw

        tm.load_hardware = _zero_floor
        try:
            r = tm.throughput(self._arch(), "h100", tp_degree=8,
                              prefill_seq_len=1024)
        finally:
            tm.load_hardware = real
        assert r.ttft_serving_overhead_ms == 0.0
        r_default = tm.throughput(self._arch(), "h100", tp_degree=8,
                                  prefill_seq_len=1024)
        assert abs(
            (r_default.prefill_time_ms - r.prefill_time_ms)
            - r_default.ttft_serving_overhead_ms) < 1e-3
