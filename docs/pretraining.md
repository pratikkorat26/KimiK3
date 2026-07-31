# Local 100M pretraining

This pipeline trains the `small` Kimi K3 preset (111.15M total parameters,
67.90M active per token) through Hugging Face `Trainer`. It does not launch
training implicitly. Large downloads and the full run both require explicit
confirmation flags.

For the exact list of completed checks and pending end-to-end validation, see
[`verification.md`](verification.md).

All caches and generated files are routed beneath `.artifacts/`:

```text
.artifacts/
  hf/                     Hugging Face hub and dataset caches
  tokenizer/v1/           32K byte-level BPE
  data/tokenized-v1/      uint16 token shards and provenance manifest
  runs/                   checkpoints, TensorBoard events, and JSONL metrics
  tmp/                    temporary files
```

## 1. Install

From WSL:

```bash
cd /mnt/e/learning/KimiTraining/KimiK3
source .venv/bin/activate
python -m pip install -e '.[dev,pretrain]'
```

## 2. Prepare the tokenizer and shards

This streams the configured Hugging Face sources and writes 500M training
tokens plus a 5M-token holdout. It does not retain the complete raw corpora.

```bash
python scripts/prepare_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --step all \
  --confirm-download
```

The immutable source revisions, licenses, document counts, token counts, shard
paths, and SHA-256 checksums are recorded in
`.artifacts/data/tokenized-v1/manifest.json`.
Each source also has a compressed `provenance.jsonl.gz` ledger with document
hashes and available IDs, URLs, repository paths, and per-document licenses.

## 3. Inspect the run without training

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --stage all \
  --dry-run
```

## 4. Run a pilot

The pilot uses the real model and data but stops after 10 optimizer steps. Its
artifacts are isolated from the full campaign.

```bash
python scripts/train_pretraining.py \
  --config configs/pretrain_100m.yaml \
  --stage 0 \
  --pilot-steps 10
```

## 5. Start or resume the full curriculum

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

The optimizer uses AdamW. The global scheduler warms up linearly for 1% of all
optimizer steps, then follows cosine decay from `3e-4` to `3e-5`. Optimizer and
scheduler state are restored at every resume and carried across the
512/1024/2048-token curriculum stages.

Only the three newest rolling checkpoints are retained. Completed stage
boundaries are copied to `.artifacts/runs/kimi-k3-100m-pretrain/milestones/`
and are not pruned.

## Monitoring

Start TensorBoard in a second WSL terminal:

```bash
source .venv/bin/activate
tensorboard \
  --logdir .artifacts/runs/kimi-k3-100m-pretrain/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

Open `http://127.0.0.1:6006`. TensorBoard shows loss, learning rate, gradient
norm, evaluation loss, and runtime.

For terminal monitoring with GPU temperature, utilization, memory, and power:

```bash
python scripts/training_status.py --watch 10
```

The local `.artifacts/runs/kimi-k3-100m-pretrain/metrics.jsonl` additionally
records windowed tokens/second, tokens processed, CUDA memory, and router
min/mean and max/mean load ratios.

Training stops and saves a checkpoint after 30 cumulative days. Periodic
evaluation uses 131,072 held-out tokens per domain to bound overhead.
