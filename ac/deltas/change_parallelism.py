"""change_parallelism — modify (TP, PP, EP, DP).

Parallelism is not stored on ArchConfig; it's a runtime argument to
throughput(). The delta engine carries this transformation as a sidecar
hint that the stress computer reads to pass tp_degree/ep_degree through.
"""

from .base import Transformation, _copy_arch


class ChangeParallelism(Transformation):
    name = "change_parallelism"
    expected_stress_signature = {
        "all_reduce": "varies",
        "all_to_all": "varies",
        "training_mem": "decrease_with_more_tp",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, tp: int = None, pp: int = None, ep: int = None,
              dp: int = None):
        out = _copy_arch(arch)
        # Store as sidecar attrs the delta engine reads. Doesn't affect
        # the arch shape, just the parallelism degrees used during stress
        # computation for the candidate.
        if tp is not None:
            out._tp_override = max(1, int(tp))  # type: ignore[attr-defined]
        if pp is not None:
            out._pp_override = max(1, int(pp))  # type: ignore[attr-defined]
        if ep is not None:
            out._ep_override = max(1, int(ep))  # type: ignore[attr-defined]
        if dp is not None:
            out._dp_override = max(1, int(dp))  # type: ignore[attr-defined]
        return out
