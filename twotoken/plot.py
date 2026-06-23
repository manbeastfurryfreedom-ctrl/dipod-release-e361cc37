"""Plot the strengthened two-token diagnostic across all three evidence axes."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTDIR = sys.argv[1] if len(sys.argv) > 1 else ".openresearch/artifacts"


def load(name):
    p = os.path.join(ARTDIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


paper = load("paper_single.json")
robust = load("reward_robust.json")
ablation = load("beta_ablation.json")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# ---- Panel (0,0)+(0,1): paper single run, gap and alignment vs step ----------
if paper is not None:
    for key, hist in paper.items():
        beta = float(key.split("=")[1])
        lab = "FPO (beta=0)" if beta == 0.0 else f"FPO+DiPOD (beta={beta:g})"
        steps = [h["step"] for h in hist]
        axes[0, 0].plot(steps, [h["gap_AA"] for h in hist], label=lab, lw=2)
        axes[0, 1].plot(steps, [h["align"] for h in hist], label=lab, lw=2)
    axes[0, 0].set_title("Discrepancy on AA  $D^L(AA)=\\log\\pi-\\mathrm{ELBO}$\n(paper Fig. 2, exact)")
    axes[0, 0].set_ylabel("gap on AA")
    axes[0, 1].set_title("Gradient alignment: $\\cos(g_{\\mathrm{proxy}}, g_{\\mathrm{true}})$\n(second drift, exact)")
    axes[0, 1].set_ylabel("cosine similarity")
    axes[0, 1].set_ylim(0, 1.02)
    for ax in axes[0, :2]:
        ax.set_xlabel("training step"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
else:
    axes[0, 0].text(0.5, 0.5, "paper_single.json missing", ha="center", va="center")
    axes[0, 1].text(0.5, 0.5, "paper_single.json missing", ha="center", va="center")

# ---- Panel (0,2): reward robustness — per-reward final gap, FPO vs DiPOD ----
if robust is not None:
    by_beta = {}
    for row in robust:
        by_beta.setdefault(row["beta"], []).append(row)
    for beta, rows in sorted(by_beta.items()):
        lab = "FPO" if beta == 0.0 else f"DiPOD b={beta:g}"
        xs = np.arange(len(rows))
        axes[0, 2].scatter(xs, [r["final_gap_AA"] for r in rows], label=lab,
                           s=30, alpha=0.7)
    axes[0, 2].set_title("Reward robustness: final gap(AA) across random rewards\n(each dot = one reward)")
    axes[0, 2].set_xlabel("reward index"); axes[0, 2].set_ylabel("final gap(AA)")
    axes[0, 2].legend(fontsize=9); axes[0, 2].grid(alpha=0.3)
else:
    axes[0, 2].text(0.5, 0.5, "reward_robust.json missing", ha="center", va="center")

# ---- Panel (1,0): beta ablation — final gap(AA) mean+/-std vs beta ----------
if ablation is not None:
    betas = sorted([float(k.split("=")[1]) for k in ablation])
    means, stds = [], []
    for b in betas:
        vals = [r["final_gap_AA"] for r in ablation[f"beta={b}"]]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    axes[1, 0].errorbar(betas, means, yerr=stds, marker="o", lw=2, capsize=4)
    axes[1, 0].set_title("beta ablation: final gap(AA) mean$\\pm$std\n(paper App. E.2, n seeds)")
    axes[1, 0].set_xlabel("beta (DiPOD)"); axes[1, 0].set_ylabel("final gap(AA)")
    axes[1, 0].grid(alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, "beta_ablation.json missing", ha="center", va="center")

# ---- Panel (1,1): beta ablation — alignment mean+/-std vs beta --------------
if ablation is not None:
    means, stds = [], []
    for b in betas:
        vals = [r["final_align"] for r in ablation[f"beta={b}"]]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    axes[1, 1].errorbar(betas, means, yerr=stds, marker="o", lw=2, capsize=4)
    axes[1, 1].set_title("beta ablation: gradient alignment mean$\\pm$std")
    axes[1, 1].set_xlabel("beta (DiPOD)"); axes[1, 1].set_ylabel("final cos(proxy, true)")
    axes[1, 1].grid(alpha=0.3); axes[1, 1].set_ylim(0, 1.02)
else:
    axes[1, 1].text(0.5, 0.5, "beta_ablation.json missing", ha="center", va="center")

# ---- Panel (1,2): reward across beta (sanity: DiPOD doesn't hurt reward) -----
if ablation is not None:
    means, stds = [], []
    for b in betas:
        vals = [r["final_reward"] for r in ablation[f"beta={b}"]]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    axes[1, 2].errorbar(betas, means, yerr=stds, marker="o", lw=2, capsize=4)
    axes[1, 2].set_title("beta ablation: final reward mean$\\pm$std\n(DiPOD shouldn't hurt reward)")
    axes[1, 2].set_xlabel("beta (DiPOD)"); axes[1, 2].set_ylabel("final reward")
    axes[1, 2].grid(alpha=0.3)
else:
    axes[1, 2].text(0.5, 0.5, "beta_ablation.json missing", ha="center", va="center")

for ax in axes.flat:
    ax.set_xticks(ax.get_xticks()[::2] if len(ax.get_xticks()) > 6 else ax.get_xticks())

fig.suptitle("DiPOD two-token diagnostic: double-drift evidence (gap + gradient alignment + robustness)")
fig.tight_layout()
out = os.path.join(ARTDIR, "twotoken_diagnostic.png")
fig.savefig(out, dpi=130)
print("wrote", out)
