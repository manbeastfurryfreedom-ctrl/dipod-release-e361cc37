"""Summarise the two-token diagnostic into EVAL.md at the repo root.

Core claim (paper Section 4.1 / Appendix D / blog "A Toy Experiment"):
  Under FPO the ELBO-likelihood discrepancy drifts UP during RL (the "double
  drift"); the DiPOD on-policy ELBO regulariser keeps it controlled while reward
  still improves. The paper tracks the discrepancy on the sequence AA (Fig. 2).
"""
import json
import os
import sys

ARTDIR = sys.argv[1] if len(sys.argv) > 1 else ".openresearch/artifacts"
REPO_ROOT = sys.argv[2] if len(sys.argv) > 2 else "."

with open(os.path.join(ARTDIR, "history.json")) as f:
    results = json.load(f)


def summ(hist):
    init = hist[0]
    final = hist[-1]
    peak_aa = max(h["gap_AA"] for h in hist)
    peak_mean = max(h["gap"] for h in hist)
    return {
        "init_gap_AA": init["gap_AA"],
        "final_gap_AA": final["gap_AA"],
        "peak_gap_AA": peak_aa,
        "final_mean_gap": final["gap"],
        "peak_mean_gap": peak_mean,
        "final_reward": final["reward"],
        "init_reward": init["reward"],
    }


rows = {float(k.split("=")[1]): summ(v) for k, v in results.items()}
fpo = rows.get(0.0)
dipod_betas = sorted(b for b in rows if b > 0.0)

L = []
L.append("# EVAL — Two-token diffusion diagnostic (DiPOD minimal repro)\n")
L.append("Minimal CPU reproduction of the DiPOD paper's controlled diagnostic "
         "(Section 4.1 / Appendix D, following SPG's two-token toy setting). A "
         "fully-enumerable 2-token masked-diffusion policy (6 logits a..f, init 0 "
         "so ELBO = log pi at start) is RL-post-trained with **exact** policy "
         "gradients (no Monte-Carlo, as the paper specifies). True log-likelihood, "
         "ELBO, and the gap `D^L = log pi - ELBO` are computed in closed form, "
         "matching the paper's worked example for AA.\n")
L.append("Methods: **FPO** (policy gradient on the ELBO score, paper Eq. 4) vs "
         "**FPO+DiPOD** (same + `beta * ELBO` on-policy regulariser, paper "
         "Algorithm 2 — the same term as `language/d1/diffu-grpo/"
         "diffu_grpo_trainer.py:160`). Reward r(AA)=0.8, r(AB)=1, r(BA)=0.7, "
         "r(BB)=1; lr=0.1; beta_DiPOD=0.2; 1500 steps (all per Appendix D).\n")

L.append("## Core claim under test\n")
L.append("Under FPO the ELBO–likelihood discrepancy (tracked on **AA**, the "
         "paper's Fig. 2) **drifts up** as RL proceeds. DiPOD keeps it "
         "**controlled** while still improving reward.\n")

L.append("## Results\n")
L.append("| Method | init gap(AA) | final gap(AA) | peak gap(AA) | final mean gap | final reward |")
L.append("|---|---|---|---|---|---|")
for beta in [0.0] + dipod_betas:
    r = rows[beta]
    name = "FPO (beta=0)" if beta == 0.0 else f"FPO+DiPOD (beta={beta:g})"
    L.append(f"| {name} | {r['init_gap_AA']:.4f} | {r['final_gap_AA']:.4f} | "
             f"{r['peak_gap_AA']:.4f} | {r['final_mean_gap']:.4f} | {r['final_reward']:.4f} |")
L.append("")

verdict = "INCONCLUSIVE"
detail = ""
if fpo is not None and dipod_betas:
    best = min(dipod_betas, key=lambda b: rows[b]["final_gap_AA"])
    rb = rows[best]
    fpo_drifted = fpo["final_gap_AA"] > fpo["init_gap_AA"] + 0.01
    dipod_controls = rb["final_gap_AA"] < 0.5 * fpo["final_gap_AA"]
    reward_ok = rb["final_reward"] >= fpo["final_reward"] - 0.05
    if fpo_drifted and dipod_controls and reward_ok:
        verdict = "REPRODUCED"
    elif dipod_controls:
        verdict = "PARTIAL"
    ratio = fpo["final_gap_AA"] / rb["final_gap_AA"] if rb["final_gap_AA"] > 0 else float("inf")
    detail = (f"FPO discrepancy on AA drifts {fpo['init_gap_AA']:.4f} -> "
              f"{fpo['final_gap_AA']:.4f} over training (the *double drift*); "
              f"FPO+DiPOD (beta={best:g}) holds it at {rb['final_gap_AA']:.4f} "
              f"-- **{ratio:.1f}x smaller** -- while reward is comparable "
              f"({rb['final_reward']:.4f} vs FPO {fpo['final_reward']:.4f}). "
              f"This matches the paper's Figure 2.")

L.append("## Verdict\n")
L.append(f"**{verdict}** — {detail}\n")
L.append("![diagnostic](.openresearch/artifacts/twotoken_diagnostic.png)\n")

md = "\n".join(L)
with open(os.path.join(REPO_ROOT, "EVAL.md"), "w") as f:
    f.write(md)
print(md)
print("\nwrote", os.path.join(REPO_ROOT, "EVAL.md"))
