# Kimi K3 Architecture (this reference)

This document describes the architecture as implemented in `kimi_k3/`, and why
each piece is shaped the way it is. It is a **study reference**: readable math
over production kernels. The forward/decode path and tiny Hugging Face
checkpoint/resume integration are test-verified. The real 111M CUDA pilot is
still pending (see [verification.md](verification.md)).

## The stack, top to bottom

```
tokens (B, T)
  → embed_tokens                     (b_0 for AttnRes)
  → N × KimiK3Block:
       Block AttnRes → PreNorm → Attention (KDA or Gated MLA) → accumulate
       Block AttnRes → PreNorm → FFN (SiTU-GLU dense | Stable LatentMoE) → accumulate
  → final Block AttnRes depth-mix
  → RMSNorm → lm_head
logits (B, T, vocab)
```

Attention layers follow a **3 KDA : 1 Gated MLA** schedule, final layer MLA
(`config.py:attention_type`). This is the Kimi Linear hybrid.

## 1. Kimi Delta Attention (KDA) — the linear-attention backbone

`attention/kda.py`, core math in `attention/kda_ops.py`.

KDA is linear attention extending **Gated DeltaNet** with a **per-channel
(diagonal) decay gate** — every channel of the `d_k` state dimension forgets at
its own learned rate, versus GDN's single per-head scalar. The recurrence:

```
S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ      # state (d_k × d_v)
o_t = S_tᵀ q_t
```

- `Diag(α_t)` is the fine-grained per-channel decay. **K3 (report Eq. 5) uses a
  lower-bounded scaled sigmoid** `g = g_min·sigmoid(exp(A_h)·z)`, `g_min = −5`,
  `α = exp(g)` — deliberately **not** Kimi Linear's negative-softplus form; the
  bounded range keeps cumulative chunk decay inside bf16 so the kernel can use dense
  Tensor-Core tiles. `(I − β_t k_t k_tᵀ)` is the classical delta rule (erase-then-write).
- State `S` is **fixed size** (independent of sequence length) → O(1) memory,
  O(T) compute. This is what carries long context cheaply.
- Two equivalent implementations: `kda_recurrence` (per-token, the decode path
  and correctness reference) and `kda_chunkwise` (chunk-parallel DPLR/WY form
  for training/prefill). `tests/test_kda.py` asserts they agree numerically.
- Depthwise causal **ShortConv** + SiLU on q/k/v; L2-norm on q/k; **output gate**.

## 2. Gated MLA — the global-attention layers

`attention/gated_mla.py`.

Full softmax attention with DeepSeek-style **low-rank KV compression**
(`kv_latent_dim`) and a **query LoRA bottleneck** (`q_lora_rank`), plus a K3
output gate. It is O(T²) compute/memory — the expensive but exact-recall path,
used sparingly (1 in 4 layers).

## 3. Positional encoding — NoPE by design (do NOT add RoPE)

**There is no positional encoding on any attention layer.** KDA injects
positional structure implicitly through its decay gate, so the global MLA layers
can be **NoPE** too. This is a deliberate Kimi Linear / K3 finding: dropping RoPE
from the full-attention layers removes RoPE-extrapolation artifacts and is *why*
the model extends to long context cleanly.

The MLA config field `qk_shared_dim` (alias `qk_rope_head_dim`) is a
**DeepSeek-MLA legacy name** — those channels are used, but RoPE is **not**
applied to them. Do not "fix" this to RoPE. Consequently, extending context
64K → 256K is a **training/curriculum** matter (progressively longer sequences),
not a RoPE-theta / YaRN scaling knob. See [roadmap.md](roadmap.md).

## 4. Attention Residuals (AttnRes)

`residuals/attn_res.py`, `residuals/depth_history.py`.

Replaces additive residuals with a depth-wise **softmax mix over block-level
representations**: a layer can read from any earlier block, not just the previous
one. `use_interim_residual=True` falls back to standard additive residuals (an
ablation switch).

## 5. Stable LatentMoE

`moe/latent_moe.py`, `moe/router.py`, `moe/situ_glu.py`.

- **Routed path**: `w_down → latent → top-k SiTU-GLU experts → RMSNorm → w_up`.
- **Shared experts**: always-on full-width SiTU-GLU.
- **SiTU-GLU**: Sigmoid-Tanh-Unit gated FFN that **bounds activations**
  (`|h| ≤ β₁β₂`) — stability under extreme sparsity, replacing SwiGLU.
- **Quantile Balancing router**: sigmoid scores, top-k on `score + expert_bias`,
  with an **aux-loss-free** bias update (bias corrects the *next* batch).

## What's out of scope (vs. the 2.8T flagship)

Native vision (MoonViT-V2), MXFP4/MXFP8 quantization + QAT, multi-teacher
distillation, and MTP speculative decoding are **not** implemented. Per-Head
Muon and training-time MTP auxiliary heads are implemented as opt-in features.

## Verification status (against primary sources)

Each module was checked line-by-line against the source PDFs, read directly:

| Component | Source (verified real) | Verdict |
|---|---|---|
| KDA recurrence (Eq. 1) | Kimi Linear 2510.26692 / K3 2607.24653 Eq. 1 | ✅ exact |
| KDA chunkwise (Eqs. 6–9, Listing 8b) | Kimi Linear 2510.26692 | ✅ exact port |
| KDA decay gate (Eq. 5, scaled sigmoid) | K3 2607.24653 §2.1.1 | ✅ (fixed docstring + `A_h=0` init) |
| KDA full-rank output gate (Eq. 6) | K3 2607.24653 | ✅ |
| Gated MLA + NoPE (Eq. 7, §2.1.2) | K3 2607.24653 | ✅ |
| Hybrid 3:1, final layer MLA | K3 2607.24653 §2.1 | ✅ |
| Attention Residuals (Eqs. 8–10) | "Attention Residuals" 2603.15031 | ✅ |
| SiTU-GLU (Eq. 12, β₁=4, β₂=25) | K3 2607.24653 §2.3.2 | ✅ exact |
| Stable LatentMoE (Eq. 11, routed RMSNorm) | K3 2607.24653 §2.3 | ✅ (shared experts fused as one wide SiTU-GLU) |
| Quantile-Balancing router (Eq. 14) | K3 2607.24653 §2.3.3 | ✅ exact (quantile reset, not DeepSeek sign-step) |

All in-code arXiv citations resolve to real papers. Deliberate study simplifications:
the two shared experts are fused into one double-width SiTU-GLU, and the router uses an
exact quantile rather than K3's distributed histogram estimator.

See [kimi-k3-notes.md](kimi-k3-notes.md) for the research digest and sources.
