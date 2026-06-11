#!/usr/bin/env python3
"""Launcher/scheduler for self-distillation comparison experiments.

Manages up to 4 independent single-GPU jobs on GPUs 0-3.
Supports baseline, tuned_baseline, and self_distill variants.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAIN_SCRIPT = REPO_ROOT / "isaaclab_fpo" / "scripts" / "train.py"
LOG_DIR = SCRIPT_DIR / "logs"
MANIFEST_PATH = SCRIPT_DIR / "run_manifest.json"

GPUS = [0, 1, 2, 3]
TASK = "Tracking-Flat-G1-v0"
WANDB_PROJECT = "fpo-self-distill-isaaclab"

DEFAULT_MOTION = str(REPO_ROOT / "whole_body_tracking_reference_data" / "walk1_subject1.npz")


def build_run_configs(
    seeds: list[int],
    max_iterations: int,
    num_envs: int | None,
    motion_file: str,
    wandb: bool,
    self_distill_iterations: int,
    self_distill_rollout_steps: int,
    self_distill_batch_size: int,
    self_distill_lr: float,
):
    """Build a list of run configurations for all variants and seeds."""
    motion_short = Path(motion_file).stem.split("_")[0]
    runs = []

    for seed in seeds:
        base_args = [
            "--task", TASK,
            "--seed", str(seed),
            "--max_iterations", str(max_iterations),
            "--headless",
            "--device", "cuda:0",
            "--motion_file", motion_file,
        ]
        if num_envs is not None:
            base_args += ["--num_envs", str(num_envs)]

        if wandb:
            base_args += ["--logger", "wandb", "--log_project_name", WANDB_PROJECT]

        # baseline
        bl_name = f"{motion_short}__baseline__seed{seed}__no_distill"
        runs.append({
            "name": bl_name,
            "variant": "baseline",
            "seed": seed,
            "motion_file": motion_file,
            "args": base_args + ["--run_name", bl_name],
        })

        # tuned_baseline (same config as baseline -- see CLAUDE.md)
        tb_name = f"{motion_short}__tuned_baseline__seed{seed}__matched"
        runs.append({
            "name": tb_name,
            "variant": "tuned_baseline",
            "seed": seed,
            "motion_file": motion_file,
            "args": base_args + ["--run_name", tb_name],
        })

        # self_distill
        sd_tag = f"d{self_distill_iterations}_r{self_distill_rollout_steps}_lr{self_distill_lr}"
        sd_name = f"{motion_short}__self_distill__seed{seed}__{sd_tag}"
        sd_args = base_args + [
            "--run_name", sd_name,
            "--self_distill",
            "--self_distill_iterations", str(self_distill_iterations),
            "--self_distill_rollout_steps", str(self_distill_rollout_steps),
            "--self_distill_batch_size", str(self_distill_batch_size),
            "--self_distill_lr", str(self_distill_lr),
        ]
        runs.append({
            "name": sd_name,
            "variant": "self_distill",
            "seed": seed,
            "motion_file": motion_file,
            "args": sd_args,
        })

    return runs


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(manifest: list[dict]):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def is_run_completed(manifest: list[dict], run_name: str) -> bool:
    for entry in manifest:
        if entry["name"] == run_name and entry.get("status") == "completed":
            return True
    return False


def get_free_gpu(active_procs: dict[int, subprocess.Popen]) -> int | None:
    """Return first GPU not currently running a job."""
    for gpu in GPUS:
        if gpu not in active_procs:
            return gpu
        if active_procs[gpu].poll() is not None:
            del active_procs[gpu]
            return gpu
    return None


def launch_run(run_cfg: dict, gpu_id: int, manifest: list[dict]) -> subprocess.Popen:
    """Launch a single training run on the given GPU."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{run_cfg['name']}.log"

    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "yes"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    args = list(run_cfg["args"])
    for i, a in enumerate(args):
        if a == "--device":
            args[i + 1] = "cuda:0"
            break

    cmd = [sys.executable, str(TRAIN_SCRIPT)] + args

    print(f"[GPU {gpu_id}] Launching: {run_cfg['name']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

    entry = {
        "name": run_cfg["name"],
        "variant": run_cfg["variant"],
        "seed": run_cfg["seed"],
        "motion_file": run_cfg["motion_file"],
        "gpu": gpu_id,
        "pid": proc.pid,
        "command": " ".join(cmd),
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "wandb_project": WANDB_PROJECT,
        "task": TASK,
    }
    manifest.append(entry)
    save_manifest(manifest)

    return proc


def main():
    parser = argparse.ArgumentParser(description="Launch self-distillation comparison experiments.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Max training iterations.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
    parser.add_argument("--motion_file", type=str, default=DEFAULT_MOTION, help="Path to reference motion NPZ.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--sd_iterations", type=int, default=100, help="Self-distillation iterations.")
    parser.add_argument("--sd_rollout_steps", type=int, default=8, help="Rollout steps for distillation.")
    parser.add_argument("--sd_batch_size", type=int, default=16384, help="Distillation batch size.")
    parser.add_argument("--sd_lr", type=float, default=3e-4, help="Distillation learning rate.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    parser.add_argument("--skip_completed", action="store_true", default=True, help="Skip completed runs.")
    args = parser.parse_args()

    runs = build_run_configs(
        seeds=args.seeds,
        max_iterations=args.max_iterations,
        num_envs=args.num_envs,
        motion_file=args.motion_file,
        wandb=args.wandb,
        self_distill_iterations=args.sd_iterations,
        self_distill_rollout_steps=args.sd_rollout_steps,
        self_distill_batch_size=args.sd_batch_size,
        self_distill_lr=args.sd_lr,
    )

    manifest = load_manifest()

    if args.skip_completed:
        runs = [r for r in runs if not is_run_completed(manifest, r["name"])]

    if not runs:
        print("All runs already completed!")
        return

    print(f"Queued {len(runs)} runs across GPUs {GPUS}")

    if args.dry_run:
        for r in runs:
            print(f"  [{r['variant']}] {r['name']}")
            print(f"    {' '.join([sys.executable, str(TRAIN_SCRIPT)] + r['args'])}")
        return

    active_procs: dict[int, subprocess.Popen] = {}
    run_queue = list(runs)

    while run_queue or active_procs:
        # Check completed processes
        for gpu_id in list(active_procs.keys()):
            proc = active_procs[gpu_id]
            ret = proc.poll()
            if ret is not None:
                status = "completed" if ret == 0 else f"failed(rc={ret})"
                for entry in manifest:
                    if entry.get("pid") == proc.pid:
                        entry["status"] = status
                        entry["end_time"] = datetime.now().isoformat()
                save_manifest(manifest)
                print(f"[GPU {gpu_id}] Finished with status: {status}")
                del active_procs[gpu_id]

        # Launch new jobs on free GPUs
        while run_queue:
            gpu = get_free_gpu(active_procs)
            if gpu is None:
                break
            run_cfg = run_queue.pop(0)
            proc = launch_run(run_cfg, gpu, manifest)
            active_procs[gpu] = proc

        if active_procs:
            time.sleep(10)

    print("\nAll runs finished!")
    save_manifest(manifest)


if __name__ == "__main__":
    main()
