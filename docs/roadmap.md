# Roadmap

Goal: from a verified architecture reference → a trainable **~1B-total MoE**
(`kimi_1b_64k`) for agentic / browser-use-with-tools work, at **64K context,
extendable to 256K**.

## Phase 1 — Architecture correctness + structure ✅ (this pass)

- Confirmed & documented **NoPE** (KDA supplies position; MLA is NoPE) — this is
  faithful to K3 and is why context extension is training-based, not RoPE scaling.
- Added `kimi_1b_64k` preset (~0.97B total, 64K ctx) + `scripts/param_count.py`.
- Added sampling (temperature / top-k / top-p / EOS) to `generate()`.
- File-based configs (`configs/*.yaml`), docs, dev tooling (ruff/mypy/CI),
  hot-path type hints, removed dead code.

## Phase 2 — Minimal trainability ✅ (in progress)

- **Tokenizer** (`kimi_k3/tokenizer/`): ✅ `ByteTokenizer` (zero-dep) + `TiktokenTokenizer`
  (GPT-2 BPE) behind a shared protocol. Training our own BPE for the 65,536 vocab is
  still future work.
- **Data** (`kimi_k3/data/`): ✅ fixed-length causal-LM packing + bundled tiny corpus.
  The progressive-length sampler (8K → 64K → 256K) remains a stub.
- **Training** (`kimi_k3/training/`): ✅ cross-entropy loss, `TrainConfig`, AdamW `Trainer`
  (warmup+cosine, grad-clip, checkpointing, overfit helper). Validated end-to-end: the
  overfit gate drives loss → ~0, and `scripts/train.py` learns on the tiny corpus.
- **Still open**: Per-Head Muon optimizer; MTP heads; throughput (fused KDA / batched MoE)
  before any longer run.

## Phase 3 — 1B pretraining

- Pretrain `kimi_1b_64k` with the progressive-length curriculum **8K → 64K**.
- Watch MoE load balance (Quantile Balancing bias), activation bounds (SiTU-GLU),
  and KDA state stability at long context.
- Eval harness (perplexity + a few long-context probes). Checkpoint/resume.
- *Optional:* Muon optimizer; MTP heads (`kimi_k3/mtp/`) for speculative decode.

## Phase 4 — 256K extension + agentic fine-tuning

- Extend to **256K** via a long-context curriculum + synthetic long-context data
  that genuinely requires attending across the window. **No RoPE knob** — this is
  continued training, leaning on KDA's linear-cost long memory. The MLA layers
  are the O(T²) bottleneck; budget compute accordingly.
- Agentic / tool-use fine-tuning (browser use, tool calling).
- 256K is the hard ceiling for this project — no further extension planned.

## Known limitations of the current reference

- Pure-PyTorch KDA ops use Python loops (readable, not fast); MoE dispatch loops
  over experts. Fine at study scale, not for real pretraining throughput — Phase 3
  will need fused/batched kernels or an existing linear-attention kernel library.
- Gated MLA is full O(T²) attention — a hybrid 64K forward needs GPU/kernels;
  `scripts/bench_context.py` demonstrates long context on the KDA-only path.
