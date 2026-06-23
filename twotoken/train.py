"""Two-token RL post-training: FPO vs FPO+DiPOD, EXACT policy gradients.

Reproduces the paper's Section 4.1 / Appendix D controlled diagnostic. The paper
states: "we directly calculate the policy gradient without invoking Monte Carlo
samples for gradient estimation." We do the same -- all expectations over the
four-sequence state space are exact.

FPO (paper Eq. 4) replaces the intractable log-likelihood score with the ELBO
score in the policy gradient:

    g_FPO(theta) = E_{a ~ pi_theta} [ A(a) * grad_theta ELBO_theta(a) ]

where A(a) = r(a) - E_{a' ~ pi_theta}[r(a')] is the (exact) advantage.

DiPOD (paper Algorithm 2 / blog "The Practical Version") adds the on-policy ELBO
regulariser (the same `beta * ELBO` term as in the release's
diffu_grpo_trainer.py:160), pushing the variational gap back down on the
distribution the policy actually visits:

    g_DiPOD(theta) = g_FPO(theta) + beta * E_{a ~ pi_theta} [ grad_theta ELBO_theta(a) ]

Gradient ascent:  theta <- theta + lr * g(theta).

Paper hyperparameters (Appendix D): lr = 0.1, beta_DiPOD = 0.2, 1500 steps,
params init to 0 (tight bound). Reward: r(AA)=0.8, r(AB)=1, r(BA)=0.7, r(BB)=1.
"""

import argparse
import json
import os
import numpy as np

import diffusion as D

# Paper's reward (Appendix D), "chosen for clarity of exposition".
REWARD = {"AA": 0.8, "AB": 1.0, "BA": 0.7, "BB": 1.0}


def expected_reward(theta):
    p = D.true_probs(theta)
    z = sum(p.values())
    return sum(p[s] * REWARD[s] for s in D.SEQS) / z


def _grad_elbo(theta, seq, h=1e-6):
    """Exact (finite-difference) grad of ELBO_theta(seq) wrt the 6 params."""
    g = np.zeros_like(theta)
    base = D.elbo_all
    for i in range(theta.size):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        g[i] = (base(tp)[seq] - base(tm)[seq]) / (2 * h)
    return g


def fpo_dipod_gradient(theta, beta):
    """Exact ascent direction: E_pi[A grad ELBO] + beta * E_pi[grad ELBO]."""
    probs = D.true_probs(theta)
    z = sum(probs.values())
    pi = {s: probs[s] / z for s in D.SEQS}
    baseline = sum(pi[s] * REWARD[s] for s in D.SEQS)   # E_pi[r]

    g = np.zeros_like(theta)
    for s in D.SEQS:
        adv = REWARD[s] - baseline
        gE = _grad_elbo(theta, s)
        g += pi[s] * (adv + beta) * gE                  # (A + beta) weight on grad ELBO
    return g


def train(beta, steps, lr, seed=0, init_noise=0.0):
    rng = np.random.default_rng(seed)
    theta = D.init_params()
    if init_noise > 0:
        theta = theta + rng.normal(0.0, init_noise, size=theta.shape)
    history = []
    for step in range(steps):
        g = fpo_dipod_gradient(theta, beta)
        theta = theta + lr * g
        gaps = D.gaps(theta)
        history.append({
            "step": step,
            "reward": float(expected_reward(theta)),
            "gap": float(D.mean_gap(theta)),     # on-policy E_pi[D^L]
            "gap_AA": float(gaps["AA"]),         # the paper's tracked "discrepancy on AA"
            "gap_AB": float(gaps["AB"]),
            "gap_BA": float(gaps["BA"]),
            "gap_BB": float(gaps["BB"]),
        })
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)      # paper: 1500
    ap.add_argument("--lr", type=float, default=0.1)        # paper: 0.1
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.2])  # FPO, FPO+DiPOD
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--init_noise", type=float, default=0.0)
    ap.add_argument("--outdir", default=".openresearch/artifacts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = {}
    for beta in args.betas:
        per_seed = [train(beta, args.steps, args.lr, seed=s, init_noise=args.init_noise)
                    for s in range(args.seeds)]
        steps = len(per_seed[0])
        avg = []
        for t in range(steps):
            row = {"step": t}
            for k in ["reward", "gap", "gap_AA", "gap_AB", "gap_BA", "gap_BB"]:
                row[k] = float(np.mean([per_seed[s][t][k] for s in range(args.seeds)]))
            avg.append(row)
        results[f"beta={beta}"] = avg

    out = os.path.join(args.outdir, "history.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)

    # quick console summary
    for key, hist in results.items():
        last = hist[-1]
        peak_aa = max(h["gap_AA"] for h in hist)
        print(f"{key:>10}  final gap_AA={last['gap_AA']:.4f}  peak gap_AA={peak_aa:.4f}  "
              f"final reward={last['reward']:.4f}")
    return results


if __name__ == "__main__":
    main()
