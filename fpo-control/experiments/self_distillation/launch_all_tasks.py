#!/usr/bin/env python3
"""Comprehensive launcher for self-distillation experiments across ALL tasks.

Runs baseline + self-distill on:
  - G1 motion tracking with 7 motions (dance1_s1, dance1_s2, walk1, run1, fight1, jumps1, fallAndGetUp1)
  - FPO locomotion: Go2, H1, G1 velocity
  - 1 seed, 500 iterations each (enough to see convergence gap)
  - Uses all 8 GPUs for max throughput
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
MANIFEST_PATH = SCRIPT_DIR / "run_manifest_all_tasks.json"
MOTION_DIR = REPO_ROOT / "whole_body_tracking_reference_data"

GPUS = list(range(8))
PYTHON = str(REPO_ROOT / "thirdparty" / "miniconda3" / "envs" / "isaaclab_fpo" / "bin" / "python")
LD_LIB = str(REPO_ROOT / "thirdparty" / "miniconda3" / "envs" / "isaaclab_fpo" / "lib")

TRACKING_MOTIONS = {
    "dance1_s1": "dance1_subject1.npz",
    "dance1_s2": "dance1_subject2.npz",
    "walk1":     "walk1_subject1.npz",
    "run1":      "run1_subject2.npz",
    "fight1":    "fight1_subject2.npz",
    "jumps1":    "jumps1_subject1.npz",
    "fallGetUp": "fallAndGetUp1_subject1.npz",
}

LOCOMOTION_TASKS = [
    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Flat-G1-v0",
]


def build_all_runs(seed, max_iterations, sd_iterations, sd_rollout_steps, sd_batch_size, sd_lr):
    runs = []

    for motion_tag, motion_file in TRACKING_MOTIONS.items():
        motion_path = str(MOTION_DIR / motion_file)
        task = "Tracking-Flat-G1-v0"
        base_args = [
            "--task", task, "--seed", str(seed),
            "--max_iterations", str(max_iterations),
            "--headless", "--device", "cuda:0",
            "--motion_file", motion_path,
        ]

        bl_name = f"{motion_tag}__baseline__seed{seed}"
        runs.append({
            "name": bl_name, "variant": "baseline", "seed": seed,
            "task": task, "motion": motion_tag,
            "args": base_args + ["--run_name", bl_name],
        })

        sd_name = f"{motion_tag}__self_distill__seed{seed}__d{sd_iterations}"
        runs.append({
            "name": sd_name, "variant": "self_distill", "seed": seed,
            "task": task, "motion": motion_tag,
            "args": base_args + [
                "--run_name", sd_name,
                "--self_distill",
                "--self_distill_iterations", str(sd_iterations),
                "--self_distill_rollout_steps", str(sd_rollout_steps),
                "--self_distill_batch_size", str(sd_batch_size),
                "--self_distill_lr", str(sd_lr),
            ],
        })

    for task in LOCOMOTION_TASKS:
        task_short = task.replace("Isaac-Velocity-Flat-", "").replace("-v0", "").lower()
        base_args = [
            "--task", task, "--seed", str(seed),
            "--max_iterations", str(max_iterations),
            "--headless", "--device", "cuda:0",
        ]

        bl_name = f"loco_{task_short}__baseline__seed{seed}"
        runs.append({
            "name": bl_name, "variant": "baseline", "seed": seed,
            "task": task, "motion": None,
            "args": base_args + ["--run_name", bl_name],
        })

        sd_name = f"loco_{task_short}__self_distill__seed{seed}__d{sd_iterations}"
        runs.append({
            "name": sd_name, "variant": "self_distill", "seed": seed,
            "task": task, "motion": None,
            "args": base_args + [
                "--run_name", sd_name,
                "--self_distill",
                "--self_distill_iterations", str(sd_iterations),
                "--self_distill_rollout_steps", str(sd_rollout_steps),
                "--self_distill_batch_size", str(sd_batch_size),
                "--self_distill_lr", str(sd_lr),
            ],
        })

    return runs


def load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def is_run_completed(manifest, run_name):
    return any(e["name"] == run_name and e.get("status") == "completed" for e in manifest)


def launch_run(run_cfg, gpu_id, manifest):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{run_cfg['name']}.log"

    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + LD_LIB

    cmd = [PYTHON, str(TRAIN_SCRIPT)] + run_cfg["args"]

    print(f"  [GPU {gpu_id}] {run_cfg['name']}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                cwd=str(REPO_ROOT))

    entry = {
        "name": run_cfg["name"],
        "variant": run_cfg["variant"],
        "seed": run_cfg["seed"],
        "task": run_cfg["task"],
        "motion": run_cfg.get("motion"),
        "gpu": gpu_id,
        "pid": proc.pid,
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
        "status": "running",
    }
    manifest.append(entry)
    save_manifest(manifest)
    return proc


def main():
    parser = argparse.ArgumentParser(description="Launch comprehensive self-distillation experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--sd_iterations", type=int, default=100)
    parser.add_argument("--sd_rollout_steps", type=int, default=8)
    parser.add_argument("--sd_batch_size", type=int, default=16384)
    parser.add_argument("--sd_lr", type=float, default=3e-4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=GPUS)
    parser.add_argument("--stagger", type=int, default=30,
                        help="Seconds between launching consecutive jobs (Isaac Sim init conflicts)")
    args = parser.parse_args()

    runs = build_all_runs(
        seed=args.seed,
        max_iterations=args.max_iterations,
        sd_iterations=args.sd_iterations,
        sd_rollout_steps=args.sd_rollout_steps,
        sd_batch_size=args.sd_batch_size,
        sd_lr=args.sd_lr,
    )

    manifest = load_manifest()
    if args.skip_completed:
        runs = [r for r in runs if not is_run_completed(manifest, r["name"])]

    if not runs:
        print("All runs completed!")
        return

    print(f"Queued {len(runs)} runs across GPUs {args.gpus}")
    for r in runs:
        tag = f"[{r['variant']}] {r['task']}"
        if r.get("motion"):
            tag += f" ({r['motion']})"
        print(f"  {r['name']}  {tag}")

    if args.dry_run:
        return

    active: dict[int, tuple[subprocess.Popen, str]] = {}
    queue = list(runs)
    last_launch = 0.0

    while queue or active:
        for gpu_id in list(active.keys()):
            proc, name = active[gpu_id]
            ret = proc.poll()
            if ret is not None:
                status = "completed" if ret == 0 else f"failed(rc={ret})"
                for entry in manifest:
                    if entry.get("pid") == proc.pid:
                        entry["status"] = status
                        entry["end_time"] = datetime.now().isoformat()
                save_manifest(manifest)
                print(f"  [GPU {gpu_id}] Done: {name} -> {status}")
                del active[gpu_id]

        while queue:
            free_gpus = [g for g in args.gpus if g not in active]
            if not free_gpus:
                break
            elapsed = time.time() - last_launch
            if elapsed < args.stagger:
                time.sleep(args.stagger - elapsed)
            gpu = free_gpus[0]
            run_cfg = queue.pop(0)
            proc = launch_run(run_cfg, gpu, manifest)
            active[gpu] = (proc, run_cfg["name"])
            last_launch = time.time()

        if active:
            time.sleep(15)

    print(f"\nAll {len(runs)} runs finished!")
    save_manifest(manifest)


if __name__ == "__main__":
    main()
