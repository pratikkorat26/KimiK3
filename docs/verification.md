# Verification status

This is the verification ledger for the local Kimi K3 implementation and the
100M pretraining pipeline. It separates automated coverage from checks that
still require the real datasets and a GPU training run.

Last updated: **2026-07-30**

## Evidence levels

- **Automated**: asserted by a repeatable test or static-analysis command.
- **Inspected**: checked manually or through a CLI without exercising training.
- **Limited benchmark**: exercised only for a narrow performance measurement.
- **Pending**: not yet run against the complete local pretraining pipeline.

## Verified

### Automated suite

The full suite completed with **115 passed and 1 skipped** out of 116 collected
tests. The skipped test was the MPS smoke test because MPS is unavailable in
the WSL/CUDA environment.

| Area | What the tests verify |
|---|---|
| Architecture | Preset shapes and parameter counts, reference K3 configuration, initialization, and hybrid layer schedule |
| KDA and Gated MLA | Output shapes, chunk/recurrent numerical agreement, cached continuation, causality, gradients, and malformed-cache rejection |
| Attention residuals and MoE | Residual mixing, block boundaries, router selection/balancing, expert forward/backward, and small-model construction |
| Model contracts | Token, cache, context-limit, dtype, shape, and configuration validation |
| Generation | Greedy and sampled decoding, reproducibility, top-k/top-p validation, EOS handling, and model-state restoration |
| Basic training | Causal language-model loss, fixed-batch overfitting, optimizer flow, and checkpoint save/load |
| Tokenizers and data | Byte-tokenizer round trips, tokenizer protocol behavior, sequence packing, labels, and batch shapes |
| Hugging Face adapter | Shifted-label loss, packed-mask enforcement, checkpointing hooks, safetensors save/load, and regular versus checkpointed output/gradient agreement |
| Pretraining pipeline | Token accounting, curriculum boundaries, project-local artifact paths, memory-mapped shard ranges, scheduler warmup/floor, BF16/checkpoint/logging arguments |

Static checks also completed successfully:

- `ruff check .`
- `mypy kimi_k3` across 37 source files

### Inspected checks

- `prepare_pretraining.py --help` and `train_pretraining.py --help` load and
  expose the expected safety flags.
- `training_status.py` loads and reports available `nvidia-smi` GPU metrics.
- Hugging Face metadata for all three configured dataset sources exposes the
  configured `text` field. The corpora themselves were not downloaded.
- The shipped pretraining configuration resolves to 500,006,912 training
  tokens, 5,000,000 validation tokens, and 15,259 optimizer steps.
- Parameter counting reports 111,148,812 total and 67,895,052 active
  parameters for `small`, and 973,874,016 total and 404,104,032 active
  parameters for `kimi_1b_64k`.

### Limited benchmark

The following preliminary CUDA benchmark completed on an RTX 4070 Laptop GPU:

```bash
python scripts/bench_train.py \
  --preset small \
  --device cuda \
  --batch-size 1 \
  --seq-len 128 \
  --steps 3
```

It measured approximately 450 tokens/second. This is **not** a pretraining
throughput result: the benchmark replaces the 32K vocabulary with 259 tokens
and does not execute an optimizer step. Do not use it to estimate the duration
of the configured 500M-token campaign.

## Pending verification

These items must remain unchecked until their commands have run successfully:

- [ ] Train the real 32K byte-level BPE from streamed source samples.
- [ ] Create all 500M training-token and 5M validation-token shards.
- [ ] Audit the generated manifest, checksums, provenance, source revisions,
      licenses, document counts, and token counts.
- [ ] Complete a real Hugging Face `Trainer.train()` optimizer step.
- [ ] Complete the 111M-parameter CUDA pilot and inspect loss, gradients,
      throughput, memory use, and router load.
- [ ] Confirm TensorBoard event creation and live terminal monitoring.
- [ ] Resume the pilot from a saved checkpoint and confirm optimizer,
      scheduler, RNG, and global-step continuity.
- [ ] Complete a curriculum boundary and verify the milestone copy and
      transition from 512 to 1024 tokens.
- [ ] Verify rolling-checkpoint pruning keeps three checkpoints without
      deleting milestones.
- [ ] Exercise the elapsed-time stop and confirm it saves without advancing
      to the next stage.
- [ ] Complete the full pretraining campaign and report held-out loss by
      domain and generated-sample quality.

No tokenizer training, dataset preparation, Hugging Face training step, 111M
pilot, full pretraining run, stage transition, or live TensorBoard event has
been verified yet.

## Rerun automated checks

Run these from WSL. They do not start pretraining:

```bash
cd /mnt/e/learning/KimiTraining/KimiK3
source .venv/bin/activate

pytest
ruff check .
mypy kimi_k3
```

The focused pretraining-contract checks are:

```bash
pytest -q tests/test_hf_integration.py tests/test_pretraining.py
```

## Validate the pending pipeline

The following sequence is intentionally opt-in. The first command downloads
and processes the configured datasets; the pilot commands update model
weights.

1. Prepare the tokenizer and token shards:

   ```bash
   python scripts/prepare_pretraining.py \
     --config configs/pretrain_100m.yaml \
     --step all \
     --confirm-download
   ```

2. Inspect the generated plan without training:

   ```bash
   python scripts/train_pretraining.py \
     --config configs/pretrain_100m.yaml \
     --stage all \
     --dry-run
   ```

3. Run a short CUDA pilot:

   ```bash
   python scripts/train_pretraining.py \
     --config configs/pretrain_100m.yaml \
     --stage 0 \
     --pilot-steps 10
   ```

4. Monitor the pilot from separate WSL terminals:

   ```bash
   tensorboard \
     --logdir .artifacts/runs/kimi-k3-100m-pretrain-pilot/tensorboard \
     --host 127.0.0.1 \
     --port 6006
   ```

   ```bash
   python scripts/training_status.py \
     --metrics .artifacts/runs/kimi-k3-100m-pretrain-pilot/metrics.jsonl \
     --watch 10
   ```

5. Test checkpoint resume separately. The 10-step pilot above does not reach
   the configured 250-step checkpoint interval, so it cannot validate resume.
   First run a 250-step pilot:

   ```bash
   python scripts/train_pretraining.py \
     --config configs/pretrain_100m.yaml \
     --stage 0 \
     --pilot-steps 250
   ```

   Confirm that
   `.artifacts/runs/kimi-k3-100m-pretrain-pilot/checkpoint-250/` exists. Then
   run 10 more steps:

   ```bash
   python scripts/train_pretraining.py \
     --config configs/pretrain_100m.yaml \
     --stage 0 \
     --pilot-steps 10
   ```

   Confirm that the second command loads `checkpoint-250` and finishes at
   global step 260 rather than restarting at zero.

Full training should start only after the pilot and resume checks pass. Its
command and operating procedure are in [pretraining.md](pretraining.md).

## Recording new evidence

When completing a pending item, add the following information beside it or in
a short dated note:

```text
Date:
Hardware:
Command:
Result:
Evidence path under .artifacts/:
```

Do not commit datasets, checkpoints, TensorBoard events, or large logs. They
remain under the ignored `.artifacts/` directory.
