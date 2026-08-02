#!/usr/bin/env python3
"""Run or resume the staged Hugging Face pretraining campaign."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kimi_k3.pretraining.artifacts import configure_local_artifacts
from kimi_k3.pretraining.config import PretrainingConfig


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_100m.yaml")
    parser.add_argument("--stage", default="all", choices=("0", "1", "2", "all"))
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--pilot-steps",
        type=int,
        default=None,
        help="run only this many optimizer steps in the selected stage",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="required unless --pilot-steps is supplied",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = PretrainingConfig.from_yaml(args.config)
    paths = configure_local_artifacts(config.artifact_root)
    tokenizer_dir = paths.tokenizer / config.tokenizer.artifact_name
    manifest_path = paths.data / config.data.artifact_name / "manifest.json"
    selected_run_name = (
        f"{config.run_name}-pilot"
        if args.pilot_steps is not None
        else config.run_name
    )
    run_dir = paths.runs / selected_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not tokenizer_dir.exists() or not manifest_path.exists():
        raise SystemExit(
            "prepared tokenizer/shards are missing; run scripts/prepare_pretraining.py"
        )
    if args.pilot_steps is None and not args.confirm_full_run and not args.dry_run:
        raise SystemExit("full training requires --confirm-full-run")

    from kimi_k3.pretraining.callbacks import prune_checkpoints
    from kimi_k3.pretraining.data import validate_prepared_artifacts
    from kimi_k3.pretraining.trainer import (
        build_stage_trainer,
        stage_end_step,
        total_optimizer_steps,
    )

    try:
        validate_prepared_artifacts(
            config,
            tokenizer_dir=tokenizer_dir,
            manifest_path=manifest_path,
            verify_checksums=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"prepared artifact validation failed: {exc}") from exc

    resume = Path(args.resume_from).resolve() if args.resume_from else _latest_checkpoint(run_dir)
    if resume is not None and not resume.exists():
        raise SystemExit(f"resume checkpoint does not exist: {resume}")
    resume_step = (
        int(resume.name.rsplit("-", 1)[1])
        if resume is not None
        else 0
    )
    if args.stage == "all":
        stages = [
            index
            for index in range(len(config.curriculum))
            if stage_end_step(config, index) > resume_step
        ]
    else:
        stages = [int(args.stage)]
        if stages[0] > 0 and resume is None:
            raise SystemExit("stages 1 and 2 require --resume-from or an existing checkpoint")
        if resume_step >= stage_end_step(config, stages[0]):
            raise SystemExit(
                f"checkpoint step {resume_step} is already past stage {stages[0]}"
            )
    plan = {
        "total_optimizer_steps": total_optimizer_steps(config),
        "stages": [
            {
                "index": index,
                "name": config.curriculum[index].name,
                "end_step": stage_end_step(config, index),
            }
            for index in stages
        ],
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if not stages:
        print("all curriculum stages are already complete")
        return
    protected: set[int] = set()
    for stage_index in stages:
        natural_end = stage_end_step(config, stage_index)
        if args.pilot_steps is not None:
            current = (
                int(resume.name.rsplit("-", 1)[1])
                if resume is not None
                else 0
            )
            end_step = min(current + args.pilot_steps, natural_end)
        else:
            end_step = natural_end
        trainer = build_stage_trainer(
            config,
            stage_index=stage_index,
            run_dir=run_dir,
            tokenizer_dir=tokenizer_dir,
            manifest_path=manifest_path,
            max_steps_override=end_step,
        )
        trainer.train(
            resume_from_checkpoint=str(resume) if resume is not None else None
        )
        trainer.save_model(run_dir / f"stage-{stage_index}-model")
        resume = run_dir / f"checkpoint-{trainer.state.global_step}"
        if (
            args.pilot_steps is None
            and trainer.state.global_step >= natural_end
            and resume.exists()
        ):
            milestone = (
                run_dir
                / "milestones"
                / f"stage-{stage_index}-checkpoint-{trainer.state.global_step}"
            )
            shutil.copytree(resume, milestone, dirs_exist_ok=True)
        protected.add(trainer.state.global_step)
        prune_checkpoints(
            run_dir,
            config.runtime.keep_last_checkpoints,
            protected,
        )
        if args.pilot_steps is not None:
            break
        if trainer.state.global_step < natural_end:
            print(
                f"stage {stage_index} stopped at step {trainer.state.global_step}; "
                "not advancing to the next curriculum stage"
            )
            break


if __name__ == "__main__":
    main()
