# Kimi K3 — research notes (as of July 2026)

Background digest for this project. Kimi K3 (Moonshot AI) is a **2.8T-param MoE**
(104B active, 1M context, native vision) — the production scale-up of the
**Kimi Linear** paper. We implement the *architecture*, scaled down; not the
flagship's size, vision, quantization, or training infrastructure.

## Flagship facts

| Property | Kimi K3 flagship | This reference (`kimi_1b_64k`) |
|---|---|---|
| Total params | 2.8T | ~0.97B |
| Active / token | 104B (~4%) | ~0.40B |
| Experts (active / total) | 16 / 896 | 8 / 64 |
| Context | 1M | 64K → 256K (target) |
| Attention | 3 KDA : 1 Gated MLA, NoPE | same schedule, NoPE |
| Vocab | 163,840 | 65,536 |
| Vision | MoonViT-V2 (native) | out of scope |
| Quantization | MXFP4 weights / MXFP8 acts, QAT | out of scope (fp32/bf16) |
| Optimizer | (per-head) Muon | out of scope (roadmap) |

## Core architecture ingredients

- **Kimi Delta Attention (KDA)** — linear attention extending Gated DeltaNet with
  fine-grained **channel-wise diagonal decay gating**; fixed-size recurrent state;
  chunkwise **DPLR / WY** parallel algorithm; ShortConv; output gating; **NoPE**.
- **Gated MLA** — low-rank-KV full attention, **NoPE** (position comes from KDA).
- **Attention Residuals** — depth-wise mix letting a layer read earlier layers.
- **Stable LatentMoE** — sigmoid router, grouped top-k, shared experts,
  **SiTU-GLU** activation, **Quantile Balancing** (aux-loss-free load balancing).
- **MTP** — extra next-n-predict heads (deferred; `kimi_k3/mtp/`).
- Flagship-only: MoonViT-V2 vision, MXFP4/8 QAT, Muon, multi-teacher distillation.

## Why NoPE matters for our context goal

Because there is no RoPE, the 64K → 256K extension is **training-driven** (a
progressive-length curriculum + synthetic long-context data), not a positional
scaling trick. KDA's linear O(T) / fixed-memory cost is what makes long context
affordable; the MLA layers are the O(T²) bottleneck. See
[architecture.md](architecture.md) §3 and [roadmap.md](roadmap.md).

## Sources (all verified to resolve to real papers)

- **Kimi K3: Open Frontier Intelligence** — arXiv:2607.24653 <https://arxiv.org/abs/2607.24653>
  (the K3 report: KDA scaled-sigmoid decay Eq. 5, full-rank gates Eqs. 6–7, SiTU-GLU
  Eq. 12, Stable LatentMoE Eq. 11, Quantile Balancing Eq. 13–14, Per-Head Muon)
- **Kimi Linear: An Expressive, Efficient Attention Architecture** — arXiv:2510.26692
  <https://arxiv.org/abs/2510.26692> (KDA recurrence Eq. 1, chunkwise DPLR/WY Eqs. 6–9,
  hybrid 3:1, NoPE)
- **Attention Residuals** — arXiv:2603.15031 (Kimi Team) — Block AttnRes softmax
  depth-mixing (distinct from Hyper-Connections / value-residual).
- **Kimi K3 overview / MXFP4** — <https://huggingface.co/blog/ResterChed/kimi-k3-model-overview-mxfp4-quantization-open-wei>
- **Kimi Linear config schema** — `configuration_kimi_linear.py`:
  <https://huggingface.co/hyper-accel/tiny-random-kimi-linear/blob/main/configuration_kimi_linear.py>
- **Kimi K2 technical report** — arXiv:2507.20534 <https://arxiv.org/abs/2507.20534>
  (MoE lineage, Muon optimizer, MTP)

> Kimi K3 postdates this assistant's training cutoff. The architecture claims here
> were verified line-by-line against the arXiv PDFs of the K3 (2607.24653), Kimi
> Linear (2510.26692), and Attention Residuals (2603.15031) reports — see the
> verification table in [architecture.md](architecture.md). Flagship *numbers* (e.g.
> the 2.8T scale) still trace to the report/model card, not prior knowledge.
