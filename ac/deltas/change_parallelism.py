"""change_parallelism — modify (TP, PP, EP, CP, DP).

Parallelism is not stored on ArchConfig; it's a runtime argument to
throughput(). The delta engine carries this transformation as a sidecar
hint that the stress computer reads to pass tp_degree/ep_degree through.
"""

from .base import Transformation, _copy_arch, _record_applied


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
              cp: int = None, dp: int = None):
        out = _copy_arch(arch)
        # Store as sidecar attrs the delta engine reads. Doesn't affect
        # the arch shape, just the parallelism degrees used during stress
        # computation for the candidate.
        if tp is not None:
            if int(tp) <= 0:
                raise ValueError("tp must be >= 1")
            out._tp_override = int(tp)  # type: ignore[attr-defined]
        if pp is not None:
            if int(pp) <= 0:
                raise ValueError("pp must be >= 1")
            out._pp_override = int(pp)  # type: ignore[attr-defined]
        if ep is not None:
            if int(ep) <= 0:
                raise ValueError("ep must be >= 1")
            out._ep_override = int(ep)  # type: ignore[attr-defined]
        if cp is not None:
            if int(cp) <= 0:
                raise ValueError("cp must be >= 1")
            out._cp_override = int(cp)  # type: ignore[attr-defined]
            # CP is read directly by stress/throughput before the evaluator
            # reverse bridge runs.
            out.cp_degree = int(cp)
        if dp is not None:
            if int(dp) <= 0:
                raise ValueError("dp must be >= 1")
            out._dp_override = int(dp)  # type: ignore[attr-defined]
        _record_applied(out, self.name)
        return out
