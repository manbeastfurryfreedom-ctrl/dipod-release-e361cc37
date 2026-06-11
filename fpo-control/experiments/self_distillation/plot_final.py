#!/usr/bin/env python3
"""Generate final comparison plots from completed training logs."""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
PLOT_DIR = Path(__file__).resolve().parent / "plots"


def extract_rewards(log_path: Path) -> np.ndarray:
    rewards = []
    with open(log_path) as f:
        for line in f:
            if "Mean reward:" in line:
                try:
                    val = float(line.strip().split("Mean reward:")[-1].strip())
                    rewards.append(val)
                except ValueError:
                    pass
    return np.array(rewards)


def smooth(data: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(data) <= window:
        return np.arange(1, len(data) + 1), data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')
    iters = np.arange(window, len(data) + 1)
    return iters, smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20, help="Smoothing window")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logs = sorted(LOG_DIR.glob("*.log"))
    if not logs:
        print("No log files found!")
        return

    runs = {}
    for log in logs:
        name = log.stem
        rewards = extract_rewards(log)
        if len(rewards) == 0:
            print(f"  Skipping {name}: no reward data")
            continue
        variant = "Self-Distill" if "self_distill" in name else "Baseline"
        seed = name.split("seed")[1].split("_")[0] if "seed" in name else "?"
        label = f"{variant} seed{seed}"
        runs[label] = rewards
        print(f"  {label}: {len(rewards)} iterations, final_reward={rewards[-1]:.2f}")

    if not runs:
        return

    colors_map = {"Baseline": "#1f77b4", "Self-Distill": "#d62728"}
    style_map = {"42": "-", "123": "--", "456": ":"}

    def get_color(label):
        return colors_map.get(label.split(" ")[0], "#333333")

    def get_style(label):
        seed = label.split("seed")[-1]
        return style_map.get(seed, "-")

    # --- Figure 1: Full training + zoomed convergence window ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for label, data in sorted(runs.items()):
        iters_s, smoothed = smooth(data, args.window)
        ax.plot(iters_s, smoothed, label=label, color=get_color(label),
                linestyle=get_style(label), linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Full Training Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, data in sorted(runs.items()):
        iters = np.arange(1, len(data) + 1)
        mask = (iters >= 200) & (iters <= 900)
        if mask.sum() > args.window:
            ax.plot(iters[mask], data[mask], color=get_color(label),
                    linestyle=get_style(label), alpha=0.15, linewidth=0.5)
            iters_s, smoothed = smooth(data[mask], args.window)
            iters_s = iters_s + 199
            ax.plot(iters_s, smoothed, label=label, color=get_color(label),
                    linestyle=get_style(label), linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Early Convergence Window (200-900)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / "final_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {p}")
    plt.close()

    # --- Figure 2: Aggregate with mean ± std ---
    bl_runs = {k: v for k, v in runs.items() if "Baseline" in k}
    sd_runs = {k: v for k, v in runs.items() if "Self-Distill" in k}

    if len(bl_runs) >= 2 and len(sd_runs) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        for group_name, group, color in [
            ("Baseline", bl_runs, "#1f77b4"),
            ("Self-Distill", sd_runs, "#d62728"),
        ]:
            min_len = min(len(v) for v in group.values())
            stacked = np.stack([v[:min_len] for v in group.values()])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            iters = np.arange(1, min_len + 1)
            iters_s, mean_s = smooth(mean, args.window)
            _, std_s = smooth(std, args.window)
            ax.plot(iters_s, mean_s, label=f"{group_name} (n={len(group)})",
                    color=color, linewidth=2)
            ax.fill_between(iters_s, mean_s - std_s, mean_s + std_s,
                            color=color, alpha=0.15)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Self-Distillation vs Baseline (mean ± std)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = PLOT_DIR / "final_aggregate.png"
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close()

    # --- Summary stats ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for label, data in sorted(runs.items()):
        print(f"\n{label} ({len(data)} iterations):")
        print(f"  Final reward: {data[-1]:.2f}")
        print(f"  Max reward:   {data.max():.2f} (iter {data.argmax()+1})")
        if len(data) >= 600:
            w = data[199:600]
            print(f"  Iter 200-600: mean={w.mean():.2f} ± {w.std():.2f}")
        if len(data) >= 900:
            w = data[199:900]
            print(f"  Iter 200-900: mean={w.mean():.2f} ± {w.std():.2f}")

    # Convergence speedup
    print("\n--- Convergence Speedup ---")
    for seed in ["42", "123", "456"]:
        bl_key = f"Baseline seed{seed}"
        sd_key = f"Self-Distill seed{seed}"
        if bl_key in runs and sd_key in runs:
            bl = runs[bl_key]
            sd = runs[sd_key]
            for threshold in [5, 10, 15, 20, 25]:
                bl_first = np.where(bl >= threshold)[0]
                sd_first = np.where(sd >= threshold)[0]
                if len(sd_first) > 0:
                    sd_iter = sd_first[0] + 1
                    if len(bl_first) > 0:
                        bl_iter = bl_first[0] + 1
                        print(f"  Seed {seed}, reward>={threshold}: SD iter {sd_iter}, BL iter {bl_iter} (SD {bl_iter-sd_iter} iters faster)")
                    else:
                        print(f"  Seed {seed}, reward>={threshold}: SD iter {sd_iter}, BL never reaches")


if __name__ == "__main__":
    main()
