#!/usr/bin/env python3
"""Generate comprehensive comparison plots across ALL tasks and motions."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

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


def smooth(data, window=15):
    if len(data) <= window:
        return np.arange(1, len(data) + 1), data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')
    iters = np.arange(window, len(data) + 1)
    return iters, smoothed


def parse_log_name(name):
    """Return (task_group, variant) from log file name."""
    if name.endswith("__no_distill"):
        return None, None
    parts = name.split("__")
    task_group = parts[0]
    variant = "self_distill" if "self_distill" in name else "baseline"
    return task_group, variant


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logs = sorted(LOG_DIR.glob("*.log"))
    runs = {}
    for log in logs:
        name = log.stem
        if "launcher" in name or "no_distill" in name or "r8_lr" in name:
            continue
        data = extract_rewards(log)
        if len(data) < 10:
            continue
        task_group, variant = parse_log_name(name)
        if task_group is None:
            continue
        runs[name] = {"data": data, "group": task_group, "variant": variant}

    groups = defaultdict(dict)
    for name, info in runs.items():
        groups[info["group"]][info["variant"]] = info["data"]

    if not groups:
        print("No valid runs found!")
        return

    tracking_groups = {k: v for k, v in groups.items() if not k.startswith("loco_")}
    loco_groups = {k: v for k, v in groups.items() if k.startswith("loco_")}

    # --- Figure 1: All tracking tasks grid ---
    if tracking_groups:
        n = len(tracking_groups)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

        for idx, (group_name, variants) in enumerate(sorted(tracking_groups.items())):
            ax = axes[idx // cols][idx % cols]
            for variant, data in sorted(variants.items()):
                color = "#d62728" if variant == "self_distill" else "#1f77b4"
                label = "Self-Distill" if variant == "self_distill" else "Baseline"
                iters_s, smoothed = smooth(data)
                ax.plot(iters_s, smoothed, label=label, color=color, linewidth=2)
                ax.plot(np.arange(1, len(data)+1), data, color=color, alpha=0.15, linewidth=0.5)
            ax.set_title(group_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Reward')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(tracking_groups), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle("Self-Distillation vs Baseline: G1 Motion Tracking", fontsize=16, fontweight='bold')
        plt.tight_layout()
        p = PLOT_DIR / "all_tracking_tasks.png"
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close()

    # --- Figure 2: Locomotion tasks grid ---
    if loco_groups:
        n = len(loco_groups)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

        for idx, (group_name, variants) in enumerate(sorted(loco_groups.items())):
            ax = axes[0][idx]
            for variant, data in sorted(variants.items()):
                color = "#d62728" if variant == "self_distill" else "#1f77b4"
                label = "Self-Distill" if variant == "self_distill" else "Baseline"
                iters_s, smoothed = smooth(data)
                ax.plot(iters_s, smoothed, label=label, color=color, linewidth=2)
                ax.plot(np.arange(1, len(data)+1), data, color=color, alpha=0.15, linewidth=0.5)
            display = group_name.replace("loco_", "").replace("unitree-", "").upper()
            ax.set_title(display, fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Reward')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Self-Distillation vs Baseline: FPO Locomotion", fontsize=16, fontweight='bold')
        plt.tight_layout()
        p = PLOT_DIR / "all_locomotion_tasks.png"
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close()

    # --- Figure 3: Convergence advantage bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    group_names = []
    advantages = []
    colors_bar = []

    for group_name in sorted(groups.keys()):
        variants = groups[group_name]
        if "baseline" not in variants or "self_distill" not in variants:
            continue
        bl = variants["baseline"]
        sd = variants["self_distill"]
        min_len = min(len(bl), len(sd))
        if min_len < 50:
            continue

        # Mean reward advantage across overlapping range
        half = min_len // 2
        bl_mean = bl[half:min_len].mean()
        sd_mean = sd[half:min_len].mean()
        adv = sd_mean - bl_mean

        display = group_name.replace("loco_", "Loco:").replace("unitree-", "")
        group_names.append(display)
        advantages.append(adv)
        colors_bar.append("#2ca02c" if adv > 0 else "#d62728")

    if group_names:
        bars = ax.bar(range(len(group_names)), advantages, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('Reward Advantage (Self-Distill - Baseline)', fontsize=11)
        ax.set_title('Self-Distillation Advantage by Task (2nd half of training)', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, advantages):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        p = PLOT_DIR / "advantage_by_task.png"
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Task':<25} {'Variant':<15} {'Iters':<8} {'Final':<10} {'Max':<10} {'2ndHalf Mean':<12}")
    print("-" * 80)

    for group_name in sorted(groups.keys()):
        for variant in ["baseline", "self_distill"]:
            if variant not in groups[group_name]:
                continue
            data = groups[group_name][variant]
            half = len(data) // 2
            second_half_mean = data[half:].mean() if half > 0 else data.mean()
            display = group_name.replace("loco_", "Loco:")
            label = "SD" if variant == "self_distill" else "BL"
            print(f"{display:<25} {label:<15} {len(data):<8} {data[-1]:<10.2f} {data.max():<10.2f} {second_half_mean:<12.2f}")

    # Convergence speedup
    print("\n--- Convergence Speedup (iterations to reach threshold) ---")
    for group_name in sorted(groups.keys()):
        variants = groups[group_name]
        if "baseline" not in variants or "self_distill" not in variants:
            continue
        bl = variants["baseline"]
        sd = variants["self_distill"]
        display = group_name.replace("loco_", "Loco:")
        sd_final = sd[-1]
        thresholds = [sd_final * 0.25, sd_final * 0.5, sd_final * 0.75]
        for frac, thresh in zip([0.25, 0.5, 0.75], thresholds):
            bl_first = np.where(bl >= thresh)[0]
            sd_first = np.where(sd >= thresh)[0]
            if len(sd_first) > 0:
                sd_iter = sd_first[0] + 1
                if len(bl_first) > 0:
                    bl_iter = bl_first[0] + 1
                    print(f"  {display:<25} reward>={thresh:.1f} ({frac:.0%} of SD final): SD iter {sd_iter}, BL iter {bl_iter} (SD {bl_iter-sd_iter:+d} faster)")
                else:
                    print(f"  {display:<25} reward>={thresh:.1f} ({frac:.0%} of SD final): SD iter {sd_iter}, BL never reaches")


if __name__ == "__main__":
    main()
