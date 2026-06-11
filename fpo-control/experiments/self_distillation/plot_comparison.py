#!/usr/bin/env python3
"""Plot training curves comparing baseline, tuned_baseline, and self_distill variants.

Reads TensorBoard event files from local log directories or fetches from W&B.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"


def find_tb_event_files(log_root: str, run_name_pattern: str = "*") -> list[str]:
    """Find TensorBoard event files matching a run name pattern."""
    results = []
    for root, dirs, files in os.walk(log_root):
        for f in files:
            if f.startswith("events.out.tfevents"):
                if run_name_pattern == "*" or run_name_pattern in root:
                    results.append(os.path.join(root, f))
    return sorted(results)


def read_tb_scalar(event_file: str, tag: str) -> tuple[list[int], list[float]]:
    """Read a scalar tag from a TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed, trying tbparse...")
        return _read_tb_scalar_tbparse(event_file, tag)

    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def _read_tb_scalar_tbparse(event_file: str, tag: str) -> tuple[list[int], list[float]]:
    """Fallback using tbparse."""
    try:
        from tbparse import SummaryReader
        reader = SummaryReader(os.path.dirname(event_file))
        df = reader.scalars
        filtered = df[df["tag"] == tag]
        return filtered["step"].tolist(), filtered["value"].tolist()
    except ImportError:
        print("Neither tensorboard nor tbparse available. Cannot read events.")
        return [], []


def read_wandb_runs(project: str, filters: dict | None = None) -> dict:
    """Read runs from W&B project."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project, filters=filters or {})
        results = {}
        for run in runs:
            name = run.name
            history = run.history(keys=["Train/mean_reward", "_step"], pandas=True)
            if not history.empty:
                results[name] = {
                    "steps": history["_step"].tolist(),
                    "rewards": history["Train/mean_reward"].dropna().tolist(),
                    "config": dict(run.config),
                }
        return results
    except Exception as e:
        print(f"W&B fetch failed: {e}")
        return {}


def plot_comparison(
    data: dict[str, tuple[list[int], list[float]]],
    title: str,
    ylabel: str,
    output_path: str,
    window: int = 10,
    xlim: tuple[int, int] | None = None,
):
    """Plot comparison curves with smoothing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Saving raw data instead.")
        np.savez(output_path.replace(".png", ".npz"), **{k: np.array(v) for k, (s, v) in data.items()})
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    variant_colors = {
        "baseline": "#1f77b4",
        "tuned_baseline": "#ff7f0e",
        "self_distill": "#2ca02c",
    }

    for name, (steps, values) in sorted(data.items()):
        if not steps:
            continue

        variant = "baseline"
        for v in ["self_distill", "tuned_baseline", "baseline"]:
            if v in name:
                variant = v
                break

        color = variant_colors.get(variant, "#333333")

        arr = np.array(values)
        if len(arr) > window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(arr, kernel, mode="valid")
            plot_steps = np.array(steps[window - 1 :])
        else:
            smoothed = arr
            plot_steps = np.array(steps)

        ax.plot(plot_steps, smoothed, label=name, color=color, alpha=0.8)
        ax.fill_between(
            np.array(steps),
            arr - np.std(arr) * 0.1,
            arr + np.std(arr) * 0.1,
            alpha=0.1,
            color=color,
        )

    if xlim:
        ax.set_xlim(xlim)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot self-distillation comparison curves.")
    parser.add_argument("--source", choices=["wandb", "tensorboard"], default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="fpo-self-distill-isaaclab")
    parser.add_argument("--tb_log_root", type=str, default="logs/fpo_rsl_rl")
    parser.add_argument("--metric", type=str, default="Train/mean_reward")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size.")
    parser.add_argument("--xlim_min", type=int, default=None)
    parser.add_argument("--xlim_max", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output filename.")
    args = parser.parse_args()

    xlim = None
    if args.xlim_min is not None or args.xlim_max is not None:
        xlim = (args.xlim_min or 0, args.xlim_max or 1000)

    data = {}

    if args.source == "wandb":
        wandb_data = read_wandb_runs(args.wandb_project)
        for name, info in wandb_data.items():
            data[name] = (info["steps"][:len(info["rewards"])], info["rewards"])
    else:
        event_files = find_tb_event_files(args.tb_log_root)
        for ef in event_files:
            run_dir = os.path.basename(os.path.dirname(ef))
            steps, values = read_tb_scalar(ef, args.metric)
            if steps:
                data[run_dir] = (steps, values)

    if not data:
        print("No data found!")
        return

    output_name = args.output or f"comparison_{args.metric.replace('/', '_')}.png"
    output_path = str(PLOT_DIR / output_name)

    plot_comparison(
        data,
        title=f"Self-Distillation Comparison: {args.metric}",
        ylabel=args.metric,
        output_path=output_path,
        window=args.window,
        xlim=xlim,
    )


if __name__ == "__main__":
    main()
