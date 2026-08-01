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
- **Per-Head Muon** ✅ (`kimi_k3/training/muon.py`): a hybrid optimizer — quintic
  Newton–Schulz orthogonalization applied *per head* to the fused attention
  projections (and whole-matrix to MoE/FFN weights), with AdamW on embeddings /
  lm_head / norms / router. Opt-in via `TrainConfig.optimizer="muon"` (and the
  pretraining `OptimizerConfig.name`); AdamW stays the default. Validated by the
  overfit gate on CPU/MPS.
- **MTP heads** ✅ (`kimi_k3/mtp/heads.py`): sequential DeepSeek/K3-faithful
  next-n-predict heads (config `num_nextn_predict_layers`), teacher-forced on
  ground-truth future tokens and sharing the trunk embedding + lm_head, each
  refined by a lightweight SiTU-GLU FFN (deviation from a full block, for
  CPU-testability). Weighted auxiliary loss (`mtp_loss_weight`); training-only —
  speculative decoding is deferred.
- **Still open**: throughput (fused KDA / batched MoE) before any longer run.

## Phase 3 — 1B pretraining

- Pretrain `kimi_1b_64k` with the progressive-length curriculum **8K → 64K**.
- Watch MoE load balance (Quantile Balancing bias), activation bounds (SiTU-GLU),
  and KDA state stability at long context.
- Eval harness (perplexity + a few long-context probes). Checkpoint/resume.
- Per-Head Muon and MTP heads are implemented (see Phase 2); MTP's
  speculative-decode use in `generate()` remains optional/deferred.

## Phase 4 — 256K extension + agentic fine-tuning

- Extend to **256K** via a long-context curriculum + synthetic long-context data
  that genuinely requires attending across the window. **No RoPE knob** — this is
  continued training, leaning on KDA's linear-cost long memory. The MLA layers
  are the O(T²) bottleneck; budget compute accordingly.
- Agentic / tool-use fine-tuning (browser use, tool calling).
- 256K is the hard ceiling for this project — no further extension planned.

## Throughput (Phase 3a) — status

- KDA `kda_chunkwise` is now **vectorized** (masked-decay einsum + batched
  triangular solve; the transparent loop is kept as `kda_chunkwise_reference` and
  as the equivalence-test oracle). MPS is a safe auto device (triangular-solve has
  an MPS fallback), and the trainer logs tokens/sec; `scripts/bench_train.py` A/Bs
  vectorized vs loop across CPU/MPS.
- **Honest findings** (`small`, CPU): the vectorization is ~a wash on CPU (the loop
  bodies were already vectorized tensor ops); the real CPU lever is **chunk_size**
  (KDA cost ~O(T·C²)) — C=64→16 is ~1.4×, so `small` now defaults to C=32. KDA (not
  MoE) dominates the model's CPU time. The vectorization's payoff is on **MPS/GPU**
  (kernel-launch reduction) — measure with `bench_train.py --device mps`.
- Still open: a matmul-factored intra-chunk with the paper's 16-token secondary
  tiling (avoids the (C,C,d_k) intermediate for a genuine CPU win); batched MoE
  dispatch (small effect — MoE is ~4% of CPU time here); Triton/CUDA fused kernels.

## Known limitations of the current reference

- Gated MLA is full O(T²) attention — a hybrid 64K forward needs GPU/kernels;
  `scripts/bench_context.py` demonstrates long context on the KDA-only path.
