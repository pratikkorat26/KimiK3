"""
kimi_k3/training/muon.py — Per-Head Muon optimizer (hybrid Muon + AdamW).

Muon (Jordan et al. 2024) replaces the raw momentum update of a 2-D weight
matrix with its orthogonalized polar factor, approximated by a quintic
Newton–Schulz iteration. Kimi K2/K3 use a **Per-Head** variant: the fused
attention projections are orthogonalized per head rather than as one big matrix,
so heads don't mix through the orthogonalization.

This module implements a single hybrid optimizer (the Keller-Jordan
`MuonWithAuxAdam` pattern) so it exposes one `param_groups` list and works with
both the manual LR schedule in `trainer.py` and HF's `LambdaLR`:

  - Muon governs the 2-D "hidden" matmul weights: attention Q/K/V/gate/output
    projections (per-head split) and their low-rank down-projections, plus all
    MoE / FFN Linear weights (whole-matrix).
  - AdamW governs everything else: token embeddings, the (shared) lm_head, all
    RMSNorm gains, depthwise conv weights, 1-D scale/bias params, and the router.

Newton–Schulz runs in float32 and has no CUDA-only ops, so this trains on
CPU / MPS as well as GPU.

Shapes:
    per-head row-split weight:  (H*d_out, in)  → view (H, d_out, in), NS per head
    per-head col-split weight:  (out, H*d_in)  → view (H, out, d_in), NS per head
"""

from __future__ import annotations

import torch

from ..attention.gated_mla import GatedMLA
from ..attention.kda import KimiDeltaAttention

# Quintic Newton–Schulz coefficients (Keller Jordan, "Muon").
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def zeropower_via_newtonschulz5(
    grad: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Orthogonalize the last two dims of `grad` via quintic Newton–Schulz.

    Accepts a 2-D matrix or a batch (..., m, n) — per-head slices orthogonalize
    in one batched call. Returns a tensor of the same shape whose non-zero
    singular values are pushed toward 1 (the semi-orthogonal polar factor).
    """
    if grad.ndim < 2:
        raise ValueError(f"expected a matrix or batch of matrices, got ndim={grad.ndim}")
    x = grad.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:  # iterate on the smaller dimension
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        a = x @ x.mT
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    if transposed:
        x = x.mT
    return x.to(grad.dtype)


def _orthogonalize(
    grad: torch.Tensor,
    head_split: tuple[int, int] | None,
    steps: int,
) -> torch.Tensor:
    """Newton–Schulz on `grad`, per head when `head_split=(num_heads, axis)`."""
    if head_split is None:
        return zeropower_via_newtonschulz5(grad, steps)
    num_heads, axis = head_split
    out, inner = grad.shape
    if axis == 0:  # rows are head-major: (H*d_out, in)
        blocks = grad.view(num_heads, out // num_heads, inner)
        ortho = zeropower_via_newtonschulz5(blocks, steps)
        return ortho.reshape(out, inner)
    # axis == 1: columns are head-major: (out, H*d_in)
    blocks = grad.view(out, num_heads, inner // num_heads).permute(1, 0, 2).contiguous()
    ortho = zeropower_via_newtonschulz5(blocks, steps)
    return ortho.permute(1, 0, 2).reshape(out, inner)


# Per-head attention projections: attr name → split axis (0 = rows, 1 = cols).
_KDA_HEAD_PROJ = {
    "w_q": 0,
    "w_k": 0,
    "w_v": 0,
    "w_alpha_up": 0,
    "w_gate": 0,
    "w_gate_up": 0,
    "w_o": 1,
}
_KDA_WHOLE_PROJ = ("w_alpha_down", "w_beta", "w_gate_down")
_MLA_HEAD_PROJ = {"w_q_up": 0, "w_kv_up": 0, "w_gate": 0, "w_o": 1}
_MLA_WHOLE_PROJ = ("w_q_down", "w_kv_down")


def collect_muon_param_groups(
    model: torch.nn.Module,
) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Partition parameters into Muon specs and AdamW params (exhaustive, disjoint).

    Returns:
      muon_specs: list of {"param": p, "head_split": (H, axis) | None}
      adamw_params: everything else (embeddings, lm_head, norms, conv, 1-D, router)
    """
    muon_specs: list[dict] = []
    claimed: set[int] = set()

    def claim_head(module: torch.nn.Module, attr: str, num_heads: int, axis: int) -> None:
        layer = getattr(module, attr, None)
        if layer is None:
            return
        param = layer.weight
        muon_specs.append({"param": param, "head_split": (num_heads, axis)})
        claimed.add(id(param))

    def claim_whole(module: torch.nn.Module, attr: str) -> None:
        layer = getattr(module, attr, None)
        if layer is None:
            return
        param = layer.weight
        muon_specs.append({"param": param, "head_split": None})
        claimed.add(id(param))

    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            num_heads = module.cfg.num_heads
            for attr, axis in _KDA_HEAD_PROJ.items():
                claim_head(module, attr, num_heads, axis)
            for attr in _KDA_WHOLE_PROJ:
                claim_whole(module, attr)
        elif isinstance(module, GatedMLA):
            num_heads = module.n_heads
            for attr, axis in _MLA_HEAD_PROJ.items():
                claim_head(module, attr, num_heads, axis)
            for attr in _MLA_WHOLE_PROJ:
                claim_whole(module, attr)

    # Default pass: any unclaimed 2-D weight that is not an embedding / head /
    # router goes to whole-matrix Muon (MoE + FFN + MTP projections); everything
    # else (1-D params, 3-D conv weights, embeddings, lm_head, router) → AdamW.
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in claimed:
            continue
        excluded = (
            name.endswith("embed_tokens.weight")
            or name.endswith("lm_head.weight")
            or "router" in name
        )
        if param.ndim == 2 and not excluded:
            muon_specs.append({"param": param, "head_split": None})
            claimed.add(id(param))
        else:
            adamw_params.append(param)
    return muon_specs, adamw_params


class Muon(torch.optim.Optimizer):
    """Hybrid optimizer: Newton–Schulz (per-head) on Muon groups, AdamW on the rest.

    Each param group carries a boolean ``use_muon``. Muon groups additionally use
    ``momentum`` and ``ns_steps``; AdamW groups use ``betas`` and ``eps``. Build
    one with :func:`build_muon_optimizer`.
    """

    def __init__(self, param_groups: list[dict], head_splits: dict[int, tuple[int, int] | None]):
        super().__init__(param_groups, defaults={})
        self._head_splits = head_splits

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group.get("weight_decay", 0.0)
            if group["use_muon"]:
                self._muon_step(group, lr, weight_decay)
            else:
                self._adamw_step(group, lr, weight_decay)
        return loss

    def _muon_step(self, group: dict, lr: float, weight_decay: float) -> None:
        momentum = group["momentum"]
        ns_steps = group["ns_steps"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(grad)
                state["momentum_buffer"] = buf
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum)  # Nesterov
            ortho = _orthogonalize(update, self._head_splits.get(id(param)), ns_steps)
            # Canonical Muon scale: O has unit-ish singular values; correct for
            # the matrix aspect ratio so the RMS update magnitude tracks AdamW.
            scale = max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(ortho, alpha=-lr * scale)

    def _adamw_step(self, group: dict, lr: float, weight_decay: float) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_c1 = 1.0 - beta1 ** state["step"]
            bias_c2 = 1.0 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-lr / bias_c1)


def build_muon_optimizer(
    model: torch.nn.Module,
    *,
    muon_lr: float,
    adam_lr: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> Muon:
    """Build a hybrid Muon+AdamW optimizer for `model` (Per-Head Muon groups).

    ``initial_lr`` is stamped on each group so a proportional LR schedule scales
    the Muon and AdamW base rates together.
    """
    muon_specs, adamw_params = collect_muon_param_groups(model)
    muon_params = [spec["param"] for spec in muon_specs]
    head_splits = {id(spec["param"]): spec["head_split"] for spec in muon_specs}
    groups: list[dict] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "initial_lr": muon_lr,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
            }
        )
    if adamw_params:
        groups.append(
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": adam_lr,
                "initial_lr": adam_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        )
    return Muon(groups, head_splits)
