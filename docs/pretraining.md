# Local 100M pretraining

This pipeline trains the `small` Kimi K3 preset (111.15M total parameters,
67.90M active per token) through Hugging Face `Trainer`. It does not launch
training implicitly. Large downloads and the full run both require explicit
confirmation flags.

For the exact list of completed checks and pending end-to-end validation, see
[`verification.md`](verification.md).

All caches and generated files are routed beneath `.artifacts/`. Smoke and
production campaigns use separate roots:

```text
.artifacts/
  smoke/                  isolated 1M-token readiness campaign
  hf/                     production Hugging Face caches
  tokenizer/v1/           production 32K byte-level BPE
  data/tokenized-v1/      production uint16 shards and provenance
  runs/                   production checkpoints and metrics
  tmp/                    production temporary files
```

## 1. Install

From WSL:

```bash
cd /mnt/e/learning/KimiTraining/KimiK3
source .venv/bin/activate
python -m pip install -e '.[dev,pretrain]'
```

## 2. Prepare the smoke campaign

Do this before downloading or training the production campaign. The smoke
configuration trains an 8K tokenizer from a 1MB sample and writes roughly 1M
training tokens beneath `.artifacts/smoke/`. It still uses the real 111M model.

```bash
python scripts/prepare_pretraining.py \
  --config configs/pretrain_100m_smoke.yaml \
  --step all \
  --confirm-download
```

Validate every prepared file before training:

```bash
python scripts/validate_pretraining_data.py \
  --config configs/pretrain_100m_smoke.yaml
```

## 3. Inspect and run the smoke pilot

Inspect the resolved plan without updating weights:

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m_smoke.yaml \
  --stage all \
  --dry-run
```

Run 10 optimizer steps:

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m_smoke.yaml \
  --stage 0 \
  --pilot-steps 10
```

The stage boundary creates `checkpoint-10`. Run the same command again to
confirm automatic resume reaches step 20. Inspect metrics and evaluate the
saved model:

```bash
python scripts/evaluate_pretraining.py \
  --config configs/pretrain_100m_smoke.yaml \
  --model-path .artifacts/smoke/runs/kimi-k3-100m-smoke-pilot/stage-0-model \
  --stage 0
```

Do not proceed if loss or gradients are non-finite, a checkpoint is missing,
resume restarts at zero, CUDA runs out of memory, or router load remains
collapsed.

## 4. Prepare production data

This streams the configured sources and writes 500M training tokens plus a
5M-token holdout. It does not retain complete raw corpora.

```bash
python scripts/prepare_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --step all \
  --confirm-download
```

The manifest records immutable source revisions, licenses, document and token
counts, shard paths, and SHA-256 checksums. Each source also has a compressed
provenance ledger.

```bash
python scripts/validate_pretraining_data.py \
  --config configs/pretrain_100m.yaml
```

Decode random documents from each source manually before accepting the corpus.
Declared licenses and provenance metadata still require human review.

## 5. Run the production pilot

Run 250 steps to exercise scheduled evaluation and checkpointing:

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --stage 0 \
  --pilot-steps 250
```

Use the measured tokens/second, peak VRAM, temperature, and validation loss to
estimate the full campaign. The reduced-vocabulary benchmark is not suitable
for that estimate.

## 6. Start or resume the full curriculum

Start all three stages:

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --stage all \
  --confirm-full-run
```

The command automatically resumes from the latest checkpoint in the run
directory. To select an exact checkpoint:

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --stage 1 \
  --resume-from .artifacts/runs/kimi-k3-100m-pretrain/milestones/stage-0-checkpoint-10681 \
  --confirm-full-run
```

AdamW is the default optimizer. Set `optimizer.name: muon` to use the opt-in
Per-Head Muon hybrid; its hidden-matrix groups use Muon while embeddings,
language-model heads, norms, and other excluded parameters use AdamW. The
global scheduler warms up linearly for 1% of all optimizer steps, then follows
cosine decay from `3e-4` to `3e-5`. Optimizer and scheduler state are restored
at every resume and carried across the 512/1024/2048-token curriculum stages.

Only the three newest rolling checkpoints are retained. Completed stage
boundaries are copied to `.artifacts/runs/kimi-k3-100m-pretrain/milestones/`
and are not pruned.

## Monitoring

For the smoke pilot, start TensorBoard in a second WSL terminal:

```bash
source .venv/bin/activate
tensorboard \
  --logdir .artifacts/smoke/runs/kimi-k3-100m-smoke-pilot/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

Open `http://127.0.0.1:6006`. TensorBoard shows loss, learning rate, gradient
norm, evaluation loss, and runtime.

For terminal monitoring with GPU temperature, utilization, memory, and power:

```bash
python scripts/training_status.py \
  --metrics .artifacts/smoke/runs/kimi-k3-100m-smoke-pilot/metrics.jsonl \
  --watch 10
```

For production, use `.artifacts/runs/kimi-k3-100m-pretrain/` for both paths.
The JSONL metrics record windowed tokens/second, tokens processed, CUDA memory,
and router min/mean and max/mean load ratios.

Training stops and saves a checkpoint after 30 cumulative days. Periodic
evaluation uses 131,072 held-out tokens per domain to bound overhead.
