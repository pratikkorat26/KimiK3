# kimi_k3 — a minimal Kimi K3 architecture, from scratch

A readable, test-backed **PyTorch reference implementation of the Kimi K3
architecture**, built for study. The goal is to understand the model, then train
a small **~1B-total-param MoE** for agentic / browser-use-with-tools work at
**64K context (extendable to 256K)**.

This is an **architecture reference**: the forward and decode paths are complete
and verified; the tokenizer, data pipeline, and training loop are on the
[roadmap](docs/roadmap.md), not yet built.

## What's implemented

| Component | Where | Notes |
|---|---|---|
| **Kimi Delta Attention (KDA)** | `kimi_k3/attention/kda.py`, `kda_ops.py` | linear attention, per-channel decay gate, fixed-size state; chunk + recurrent forms |
| **Gated MLA** | `kimi_k3/attention/gated_mla.py` | low-rank-KV full attention, **NoPE** |
| **Hybrid schedule** | `kimi_k3/config.py` | 3 KDA : 1 Gated MLA, final layer MLA |
| **Attention Residuals** | `kimi_k3/residuals/` | depth-wise mix over blocks |
| **Stable LatentMoE** | `kimi_k3/moe/` | SiTU-GLU experts + Quantile-Balancing router |
| **Decode caches, sampling** | `kimi_k3/attention/cache.py`, `model.py` | KDA/MLA caches; greedy + top-k/top-p |

Deferred (stubs in the tree): `vision/`, `mtp/`, `tokenizer/`, `data/`, `training/`.

> **Positional encoding: NoPE by design.** KDA supplies position implicitly, so
> the MLA layers use no RoPE either — this is faithful to Kimi K3 and is why
> context extends cleanly. See [docs/architecture.md](docs/architecture.md) §3.

## Quickstart

```python
import torch
from kimi_k3 import KimiK3Model, ModelConfig

model = KimiK3Model(ModelConfig.tiny_hybrid()).eval()
tokens = torch.randint(0, 4096, (1, 16))
logits, cache = model(tokens, mode="chunk", use_cache=True)

out = model.generate(tokens, 8, do_sample=True, temperature=0.8, top_k=20)
```

### Presets (`ModelConfig`)

- `tiny()` / `tiny_hybrid()` — CPU unit-test scale.
- `small()` — ~100M, runnable K3-like model.
- `kimi_1b_64k()` — **~0.97B total** (~0.40B active), 64K context — the training target.

Configs also live as YAML in [`configs/`](configs/): `ModelConfig.from_yaml("configs/small.yaml")`.

## Scripts

```bash
python scripts/param_count.py kimi_1b_64k     # total & active params
python scripts/smoke_forward.py small         # forward + greedy/sampled generate
python scripts/bench_context.py --preset tiny --seq 65536   # 64K KDA-only forward
```

## Training (Phase 2)

The model is trainable end-to-end on CPU. The overfit gate proves the
forward→loss→backward→optimizer path; then train on a corpus and generate.

```bash
python scripts/overfit.py                       # loss → ~0 on a fixed batch (sanity gate)
python scripts/train.py --preset small --steps 200   # train on the bundled corpus → out/
python scripts/generate_text.py --ckpt out/ckpt_final.pt --prompt "Alice was"
```

- Tokenizer: byte-level by default (zero-dep); `--tokenizer gpt2` uses a BPE
  (`pip install -e '.[train]'` for tiktoken).
- Optimizer is AdamW; keep `--seq-len`/`--batch-size` small on CPU (the readable
  ops aren't throughput-optimized yet). See [docs/roadmap.md](docs/roadmap.md).

## Install & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check . && mypy kimi_k3
```

## Docs

- [docs/architecture.md](docs/architecture.md) — the architecture, with math and rationale.
- [docs/kimi-k3-notes.md](docs/kimi-k3-notes.md) — research digest + sources.
- [docs/roadmap.md](docs/roadmap.md) — path to 1B training and 256K context.
