"""Plot the two-token diagnostic, reproducing the paper's Figure 2: the
ELBO-likelihood discrepancy on AA (and the on-policy mean gap) vs RL step, for
FPO (beta=0) vs FPO+DiPOD (beta>0). Writes a PNG into the artifacts dir."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTDIR = sys.argv[1] if len(sys.argv) > 1 else ".openresearch/artifacts"

with open(os.path.join(ARTDIR, "history.json")) as f:
    results = json.load(f)


def label_for(key):
    beta = float(key.split("=")[1])
    return "FPO (beta=0)" if beta == 0.0 else f"FPO+DiPOD (beta={beta:g})"


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

for key, hist in results.items():
    steps = [h["step"] for h in hist]
    lab = label_for(key)
    axes[0].plot(steps, [h["gap_AA"] for h in hist], label=lab, linewidth=2)
    axes[1].plot(steps, [h["gap"] for h in hist], label=lab, linewidth=2)
    axes[2].plot(steps, [h["reward"] for h in hist], label=lab, linewidth=2)

axes[0].set_title("Discrepancy on AA  $D^L_\\theta(AA)=\\log\\pi(AA)-\\mathrm{ELBO}(AA)$\n(paper Fig. 2)")
axes[0].set_xlabel("training step"); axes[0].set_ylabel("gap on AA")

axes[1].set_title("On-policy mean gap  $E_{a\\sim\\pi}[D^L_\\theta(a)]$")
axes[1].set_xlabel("training step"); axes[1].set_ylabel("mean gap")

axes[2].set_title("Expected reward")
axes[2].set_xlabel("training step"); axes[2].set_ylabel("E[R]")

for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Two-token diffusion diagnostic: FPO gap drifts up; DiPOD controls it")
fig.tight_layout()
out = os.path.join(ARTDIR, "twotoken_diagnostic.png")
fig.savefig(out, dpi=130)
print("wrote", out)
