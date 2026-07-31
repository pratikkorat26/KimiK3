#!/usr/bin/env python3
"""Print the latest local training metric and optional live GPU status."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def latest_metric(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def display(path: Path) -> None:
    metric = latest_metric(path)
    print(json.dumps(metric, indent=2, sort_keys=True))
    if shutil.which("nvidia-smi"):
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader",
            ],
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=".artifacts/runs/kimi-k3-100m-pretrain/metrics.jsonl",
    )
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    args = parser.parse_args()
    path = Path(args.metrics)
    while True:
        display(path)
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
