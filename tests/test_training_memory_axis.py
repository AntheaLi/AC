"""Wave 14 — training_memory_per_gpu_gb as Pareto axis.

Tests:
  * is_dominated checks training_memory in addition to inference memory
  * a candidate with lower training_mem can Pareto-dominate one with same
    inference cost
  * back-compat: candidates without training_memory_per_gpu_gb (default 0)
    don't break the dominance test
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mock_ev(loss, tps, tbt, ttft, mem, train_mem, total_params):
    """Build a mock EvaluatedCandidate with the fields is_dominated reads."""
    ev = MagicMock()
    ev.predicted_loss = loss
    ev.training_tps = tps
    ev.serving_tbt_ms = tbt
    ev.throughput.prefill_time_ms = ttft
    ev.serving_request_latency_ms = ttft + tbt * 512
    ev.memory_per_gpu_gb = mem
    ev.throughput.training_memory_per_gpu_gb = train_mem
    ev.arch.total_params = total_params
    return ev


class TrainingMemoryParetoAxisTests(unittest.TestCase):

    def test_lower_training_mem_pareto_dominates(self):
        """Two candidates identical on all axes except training_mem;
        the lower-training-mem one must dominate."""
        from ac.optimizer import is_dominated
        a = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=120, total_params=70e9)
        b = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=60, total_params=70e9)
        # b dominates a (b has lower train_mem)
        self.assertTrue(is_dominated(a, b),
                        "candidate with higher training_mem must be Pareto-dominated")
        self.assertFalse(is_dominated(b, a),
                         "candidate with lower training_mem cannot be dominated by it")

    def test_higher_training_mem_not_auto_dominated_if_better_elsewhere(self):
        """A candidate with higher training_mem but lower loss is NOT
        dominated — it's Pareto-incomparable."""
        from ac.optimizer import is_dominated
        a = _mock_ev(loss=1.75, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=120, total_params=70e9)
        b = _mock_ev(loss=1.80, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=60, total_params=70e9)
        # Neither dominates: a has lower loss, b has lower train_mem
        self.assertFalse(is_dominated(a, b))
        self.assertFalse(is_dominated(b, a))

    def test_back_compat_zero_training_mem(self):
        """Candidates without training_memory_per_gpu_gb computed
        (default 0) don't break the dominance test."""
        from ac.optimizer import is_dominated
        a = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=0.0, total_params=70e9)
        b = _mock_ev(loss=1.7, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=0.0, total_params=70e9)
        # b dominates a (lower loss, all else equal including train_mem=0)
        self.assertTrue(is_dominated(a, b))

    def test_missing_training_mem_attribute_handled(self):
        """If throughput object doesn't have training_memory_per_gpu_gb
        attribute at all, the dominance test still runs (defaults to 0)."""
        from ac.optimizer import is_dominated
        a = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                     train_mem=120, total_params=70e9)
        # Construct b without the training_memory_per_gpu_gb attribute
        b = MagicMock()
        b.predicted_loss = 1.7
        b.training_tps = 1000
        b.serving_tbt_ms = 20
        b.throughput.prefill_time_ms = 200
        b.memory_per_gpu_gb = 40
        # Don't set training_memory_per_gpu_gb — should default to 0
        delattr(b.throughput, 'training_memory_per_gpu_gb')
        b.throughput.training_memory_per_gpu_gb = MagicMock(
            side_effect=AttributeError)
        b.arch.total_params = 70e9
        # Even with the missing attribute, getattr fallback to 0 keeps
        # the test running.
        try:
            is_dominated(a, b)
        except AttributeError:
            self.fail("is_dominated should handle missing training_memory_per_gpu_gb")

    def test_training_cost_objective_prices_training_memory(self):
        from ac.optimizer import _objective_score

        high = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                        train_mem=120, total_params=70e9)
        low = _mock_ev(loss=1.8, tps=1000, tbt=20, ttft=200, mem=40,
                       train_mem=60, total_params=70e9)
        pool = [high, low]
        self.assertLess(
            _objective_score(low, pool, "training_cost"),
            _objective_score(high, pool, "training_cost"),
        )


if __name__ == "__main__":
    unittest.main()
