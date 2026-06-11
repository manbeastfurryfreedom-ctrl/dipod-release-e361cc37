#!/usr/bin/env python3
"""Launch motion-tracking tasks × {baseline, self-distill} at 1000 iterations."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN = REPO / "isaaclab_fpo" / "scripts" / "train.py"
PYTHON = str(REPO / "thirdparty" / "miniconda3" / "envs" / "isaaclab_fpo" / "bin" / "python")
LD_LIB = str(REPO / "thirdparty" / "miniconda3" / "envs" / "isaaclab_fpo" / "lib")
LOG_DIR = Path(__file__).resolve().parent / "logs_1k"
MANIFEST = Path(__file__).resolve().parent / "manifest_1k.json"
MOTION_DIR = REPO / "whole_body_tracking_reference_data"

MOTIONS = {
    "dance1_s1":  "dance1_subject1.npz",
    "dance1_s2":  "dance1_subject2.npz",
    "walk1":      "walk1_subject1.npz",
    "run1":       "run1_subject2.npz",
    "fight1":     "fight1_subject2.npz",
    "jumps1":     "jumps1_subject1.npz",
    "fallGetUp":  "fallAndGetUp1_subject1.npz",
}

TASK = "Tracking-Flat-G1-v0"
SEED = 42
MAX_ITER = 1000
SD_ITER = 100
SD_ROLLOUT = 8
SD_BATCH = 16384
SD_LR = 3e-4
GPUS = list(range(8))
STAGGER = 30


def build_runs(motions, variants, seed, max_iterations, sd_iterations, sd_rollout, sd_batch, sd_lr):
    runs = []
    variants = set(variants)
    for tag in motions:
        npz = MOTIONS[tag]
        motion = str(MOTION_DIR / npz)
        base = ["--task", TASK, "--seed", str(seed), "--max_iterations", str(max_iterations),
                "--headless", "--device", "cuda:0", "--motion_file", motion]
        if "baseline" in variants:
            runs.append({"name": f"{tag}__baseline_1k",
                         "args": base + ["--run_name", f"{tag}__baseline_1k"],
                         "variant": "baseline", "motion": tag})
        if "self_distill" in variants:
            runs.append({"name": f"{tag}__sd_1k", "variant": "self_distill", "motion": tag,
                         "args": base + ["--run_name", f"{tag}__sd_1k",
                                         "--self_distill",
                                         "--self_distill_iterations", str(sd_iterations),
                                         "--self_distill_rollout_steps", str(sd_rollout),
                                         "--self_distill_batch_size", str(sd_batch),
                                         "--self_distill_lr", str(sd_lr)]})
    return runs


def load_manifest():
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            return json.load(f)
    return []


def save_manifest(m):
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Launch 1k G1 motion-tracking comparisons.")
    parser.add_argument("--motions", nargs="+", choices=sorted(MOTIONS), default=list(MOTIONS),
                        help="Motion tasks to launch.")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "self_distill"],
                        default=["baseline", "self_distill"],
                        help="Experiment variants to launch.")
    parser.add_argument("--gpus", nargs="+", type=int, default=GPUS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_iterations", type=int, default=MAX_ITER)
    parser.add_argument("--sd_iterations", type=int, default=SD_ITER)
    parser.add_argument("--sd_rollout_steps", type=int, default=SD_ROLLOUT)
    parser.add_argument("--sd_batch_size", type=int, default=SD_BATCH)
    parser.add_argument("--sd_lr", type=float, default=SD_LR)
    parser.add_argument("--stagger", type=int, default=STAGGER)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_runs(
        motions=args.motions,
        variants=args.variants,
        seed=args.seed,
        max_iterations=args.max_iterations,
        sd_iterations=args.sd_iterations,
        sd_rollout=args.sd_rollout_steps,
        sd_batch=args.sd_batch_size,
        sd_lr=args.sd_lr,
    )
    manifest = load_manifest()
    done = {e["name"] for e in manifest if e.get("status") == "completed"}
    runs = [r for r in runs if r["name"] not in done]

    if not runs:
        print("All runs already completed!")
        return

    print(f"Queued {len(runs)} runs across GPUs {args.gpus}")
    for r in runs:
        print(f"  [{r['variant']:14s}] {r['name']}")

    if args.dry_run:
        for r in runs:
            print("  " + " ".join([PYTHON, str(TRAIN)] + r["args"]))
        return

    active = {}
    queue = list(runs)
    last_launch = 0.0

    while queue or active:
        for gpu in list(active):
            proc, name = active[gpu]
            ret = proc.poll()
            if ret is not None:
                status = "completed" if ret == 0 else f"failed(rc={ret})"
                for e in manifest:
                    if e.get("pid") == proc.pid:
                        e["status"] = status
                        e["end_time"] = datetime.now().isoformat()
                save_manifest(manifest)
                print(f"  [GPU {gpu}] Done: {name} -> {status}")
                del active[gpu]

        while queue:
            free = [g for g in args.gpus if g not in active]
            if not free:
                break
            elapsed = time.time() - last_launch
            if elapsed < args.stagger:
                time.sleep(args.stagger - elapsed)

            gpu = free[0]
            cfg = queue.pop(0)
            log_file = LOG_DIR / f"{cfg['name']}.log"
            env = os.environ.copy()
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + LD_LIB
            cmd = [PYTHON, str(TRAIN)] + cfg["args"]

            print(f"  [GPU {gpu}] Launching: {cfg['name']}")
            with open(log_file, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                        env=env, cwd=str(REPO))
            active[gpu] = (proc, cfg["name"])
            manifest.append({
                "name": cfg["name"], "variant": cfg["variant"], "motion": cfg["motion"],
                "gpu": gpu, "pid": proc.pid, "log": str(log_file),
                "start_time": datetime.now().isoformat(), "status": "running",
            })
            save_manifest(manifest)
            last_launch = time.time()

        if active:
            time.sleep(15)

    print(f"\nAll done!")
    save_manifest(manifest)

    plotter = Path(__file__).resolve().parent / "plot_tracking_1k.py"
    if plotter.exists():
        subprocess.run([PYTHON, str(plotter)], cwd=str(REPO), check=False)


if __name__ == "__main__":
    main()
