"""Plot training and eval curves for FPO++ locomotion baselines.

Generates two figures:
  1. Training reward curves (1x4 grid, one per robot)
  2. PostEval reward curves (1x4 grid, one per robot)

Style adapted from arxiv_plotting/1_loco_plots.ipynb.

Usage:
    python results_plots/plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent

ROBOTS = ["go2", "spot", "h1", "g1"]
ROBOT_TITLES = {"go2": "Go2", "spot": "Spot", "h1": "H1", "g1": "G1"}
CUTOFFS = {"go2": 1500, "spot": 1500, "h1": 2000, "g1": 2000}

TRAIN_COLOR = "#3bc4ff"
EVAL_COLORS = {
    "zero": "#3bc4ff",
    "random": "#7bc47f",
}


def load_data(robot):
    path = RESULTS_DIR / f"{robot}_results.npz"
    return dict(np.load(path, allow_pickle=True))


def plot_training_curves():
    """1x4 grid of training reward curves."""
    fig = plt.figure(figsize=(14, 3))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3, bottom=0.18, top=0.88)

    for col, robot in enumerate(ROBOTS):
        data = load_data(robot)
        ax = fig.add_subplot(gs[0, col])

        steps = np.arange(1, len(data["train_rewards"]) + 1)
        rewards = data["train_rewards"]
        cutoff = CUTOFFS[robot]

        mask = steps <= cutoff
        steps = steps[mask]
        rewards = rewards[mask]

        ax.plot(steps, rewards, color=TRAIN_COLOR, linewidth=1.5, alpha=0.9)
        if "train_rewards_std" in data:
            std = data["train_rewards_std"][mask]
            ax.fill_between(steps, rewards - std, rewards + std, color=TRAIN_COLOR, alpha=0.15, edgecolor="none")
        ax.set_title(ROBOT_TITLES[robot], fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(0, cutoff)

        if col == 0:
            ax.set_ylabel("Episode Return", fontsize=10)

    fig.text(0.5, 0.04, "Training Iteration", ha="center", fontsize=10)
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "training_curves.svg", bbox_inches="tight")
    print(f"Saved training_curves.png/svg")
    plt.close(fig)


def plot_eval_curves():
    """1x4 grid of post-training eval reward curves."""
    fig = plt.figure(figsize=(14, 3))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3, bottom=0.18, top=0.88)

    for col, robot in enumerate(ROBOTS):
        data = load_data(robot)
        ax = fig.add_subplot(gs[0, col])

        eval_iters = data["eval_iters"]
        if len(eval_iters) == 0:
            ax.set_title(ROBOT_TITLES[robot], fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)
            continue

        cutoff = CUTOFFS[robot]
        mask = eval_iters <= cutoff

        for mode, color in EVAL_COLORS.items():
            mean_key = f"PostEval_{mode}/mean_reward"
            std_key = f"PostEval_{mode}/std_reward"

            if mean_key not in data:
                continue

            mean = data[mean_key][mask]
            iters = eval_iters[mask]

            # Filter NaNs
            valid = ~np.isnan(mean)
            if not valid.any():
                continue

            mean = mean[valid]
            iters_v = iters[valid]

            label = {"zero": "Zero noise", "random": "Random noise"}[mode]
            ax.plot(iters_v, mean, color=color, linewidth=1.5, alpha=0.9, label=label)

            if std_key in data:
                std = data[std_key][mask][valid]
                ax.fill_between(iters_v, mean - std, mean + std, color=color, alpha=0.15, edgecolor="none")

        ax.set_title(ROBOT_TITLES[robot], fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(0, cutoff)

        if col == 0:
            ax.set_ylabel("Episode Return", fontsize=10)
        if col == 3:
            ax.legend(fontsize=7, loc="lower right")

    fig.text(0.5, 0.04, "Training Iteration", ha="center", fontsize=10)
    fig.savefig(RESULTS_DIR / "eval_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "eval_curves.svg", bbox_inches="tight")
    print(f"Saved eval_curves.png/svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_training_curves()
    plot_eval_curves()
