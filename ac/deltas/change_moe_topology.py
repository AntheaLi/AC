"""change_moe_topology — modify (n_experts, top_k, expert_dim) for MoE."""

from .base import Transformation, _copy_arch, _record_applied


class ChangeMoeTopology(Transformation):
    name = "change_moe_topology"
    expected_stress_signature = {
        "all_to_all": "varies",
        "hbm_capacity": "varies",
        "moe_residual": "varies",
    }

    def precondition(self, arch):
        if arch.moe_config is None:
            return False, "no moe_config to modify"
        return True, ""

    def apply(self, arch, n_experts: int = None, top_k: int = None,
              expert_dim: int = None, capacity_factor: float = None):
        out = _copy_arch(arch)
        cfg = dict(arch.moe_config)
        if n_experts is not None:
            if n_experts < 1:
                raise ValueError("n_experts must be >= 1")
            cfg["n_experts"] = int(n_experts)
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be >= 1")
            cfg["top_k"] = int(top_k)
        if expert_dim is not None:
            if expert_dim < 1:
                raise ValueError("expert_dim must be >= 1")
            cfg["expert_dim"] = int(expert_dim)
        if capacity_factor is not None:
            if capacity_factor <= 0:
                raise ValueError("capacity_factor must be > 0")
            cfg["capacity_factor"] = float(capacity_factor)
        if int(cfg["top_k"]) > int(cfg["n_experts"]):
            raise ValueError(
                f"top_k={cfg['top_k']} must be <= n_experts={cfg['n_experts']}")
        out.moe_config = cfg
        _record_applied(out, self.name)
        return out
