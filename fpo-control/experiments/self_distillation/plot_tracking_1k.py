#!/usr/bin/env python3
"""Generate per-motion 1k baseline vs self-distillation plots from launcher logs."""

import argparse
import re
from pathlib import Path


MOTIONS = [
    "dance1_s1",
    "dance1_s2",
    "walk1",
    "run1",
    "fight1",
    "jumps1",
    "fallGetUp",
]
VARIANTS = {
    "baseline": "baseline_1k",
    "self_distill": "sd_1k",
}
COLORS = {
    "baseline": "#1f77b4",
    "self_distill": "#d62728",
}
LABELS = {
    "baseline": "FPO++",
    "self_distill": "FPO++ + initial self-distillation",
}
PATTERNS = {
    "reward": re.compile(r"Mean reward:\s*([-+0-9.eE]+)"),
    "ep_length": re.compile(r"Mean episode length:\s*([-+0-9.eE]+)"),
}


def plotting_deps():
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"Plotting requires numpy and matplotlib: {exc}") from exc
    return np, plt


def extract_metric(log_path: Path, metric: str) -> list[float]:
    values = []
    pattern = PATTERNS[metric]
    if not log_path.exists():
        return values
    with log_path.open(errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    return values


def smooth(values: list[float], window: int, np):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=int), arr
    if len(arr) <= window:
        return np.arange(1, len(arr) + 1), arr
    kernel = np.ones(window) / window
    return np.arange(window, len(arr) + 1), np.convolve(arr, kernel, mode="valid")


def load_runs(log_dir: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    runs = {motion: {} for motion in MOTIONS}
    for motion in MOTIONS:
        for variant, suffix in VARIANTS.items():
            log_path = log_dir / f"{motion}__{suffix}.log"
            runs[motion][variant] = {
                "reward": extract_metric(log_path, "reward"),
                "ep_length": extract_metric(log_path, "ep_length"),
            }
    return runs


def has_metric_data(runs: dict[str, dict[str, dict[str, list[float]]]], metric: str) -> bool:
    return any(runs[motion][variant][metric] for motion in MOTIONS for variant in VARIANTS)


def plot_motion(motion: str, variants: dict[str, dict[str, list[float]]], metric: str, output: Path, window: int):
    if not any(data[metric] for data in variants.values()):
        return False
    np, plt = plotting_deps()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, data in variants.items():
        values = data[metric]
        if not values:
            continue
        steps, smoothed = smooth(values, window, np)
        ax.plot(steps, smoothed, color=COLORS[variant], linewidth=2, label=LABELS[variant])
        ax.plot(np.arange(1, len(values) + 1), values, color=COLORS[variant], alpha=0.16, linewidth=0.8)

    ylabel = "Mean reward" if metric == "reward" else "Mean episode length"
    ax.set_title(motion)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_grid(runs: dict[str, dict[str, dict[str, list[float]]]], metric: str, output: Path, window: int):
    if not has_metric_data(runs, metric):
        return False
    np, plt = plotting_deps()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for idx, motion in enumerate(MOTIONS):
        ax = axes[idx // 4][idx % 4]
        for variant, data in runs[motion].items():
            values = data[metric]
            if not values:
                continue
            steps, smoothed = smooth(values, window, np)
            ax.plot(steps, smoothed, color=COLORS[variant], linewidth=2, label=LABELS[variant])
            ax.plot(np.arange(1, len(values) + 1), values, color=COLORS[variant], alpha=0.12, linewidth=0.7)
        ax.set_title(motion)
        ax.grid(True, alpha=0.25)
        if idx // 4 == 1:
            ax.set_xlabel("Training iteration")
        if idx % 4 == 0:
            ax.set_ylabel("Mean reward" if metric == "reward" else "Mean episode length")
    axes[1][3].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_advantage(runs: dict[str, dict[str, dict[str, list[float]]]], output: Path):
    motions = []
    advantages = []
    for motion in MOTIONS:
        baseline = runs[motion]["baseline"]["reward"]
        distill = runs[motion]["self_distill"]["reward"]
        if not baseline or not distill:
            continue
        min_len = min(len(baseline), len(distill))
        tail = max(1, min_len // 2)
        baseline_tail = baseline[min_len - tail:min_len]
        distill_tail = distill[min_len - tail:min_len]
        baseline_mean = sum(baseline_tail) / len(baseline_tail)
        distill_mean = sum(distill_tail) / len(distill_tail)
        advantages.append(distill_mean - baseline_mean)
        motions.append(motion)
    if not motions:
        return False
    _, plt = plotting_deps()

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in advantages]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(motions, advantages, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Reward advantage")
    ax.set_title("FPO++ + initial self-distillation minus FPO++")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot 1k motion-tracking self-distillation comparisons.")
    parser.add_argument("--log_dir", type=Path, default=Path(__file__).resolve().parent / "logs_1k")
    parser.add_argument("--plot_dir", type=Path, default=Path(__file__).resolve().parent / "plots_1k")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.log_dir)

    generated = []
    for motion, variants in runs.items():
        if plot_motion(motion, variants, "reward", args.plot_dir / f"{motion}.png", args.window):
            generated.append(args.plot_dir / f"{motion}.png")
        if plot_motion(motion, variants, "ep_length", args.plot_dir / f"{motion}_ep_length.png", args.window):
            generated.append(args.plot_dir / f"{motion}_ep_length.png")

    if plot_grid(runs, "reward", args.plot_dir / "all_tasks_grid.png", args.window):
        generated.append(args.plot_dir / "all_tasks_grid.png")
    if plot_grid(runs, "ep_length", args.plot_dir / "all_tasks_ep_length_grid.png", args.window):
        generated.append(args.plot_dir / "all_tasks_ep_length_grid.png")
    if plot_advantage(runs, args.plot_dir / "advantage_bar.png"):
        generated.append(args.plot_dir / "advantage_bar.png")

    if generated:
        print("Generated plots:")
        for path in generated:
            print(f"  {path}")
    else:
        print(f"No usable logs found in {args.log_dir}")


if __name__ == "__main__":
    main()
