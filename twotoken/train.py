"""Two-token DiPOD diagnostic — STRENGTHENED EVIDENCE harness.

Builds on the verified minimal repro (same exact 6-parameter two-token model,
same FPO / FPO+DiPOD updates). Strengthens the evidence along three axes, all
computed EXACTLY (no Monte-Carlo, as the paper specifies):

  AXIS 1 — Gradient alignment / "second drift".
    The paper's second drift is that the FPO proxy gradient (the ELBO-score
    policy gradient) drifts away from the TRUE policy gradient (the log-pi-score
    policy gradient). In the real experiments this is unmeasurable because log pi
    is intractable; HERE it is exact. We track, at each step, the cosine between

        g_FPO    = E_{a~pi}[ A(a) * grad ELBO(a) ]          (proxy)
        g_true   = E_{a~pi}[ A(a) * grad log pi(a) ]         (true PG)

    DiPOD should keep alignment high; FPO should lose it. This is the direct
    mechanistic evidence for the double drift, which the paper argues but cannot
    measure at scale.

  AXIS 2 — Reward robustness.
    Run FPO and FPO+DiPOD across many RANDOM reward tables (uniform draws), not
    just the paper's hand-chosen reward. The drift is claimed to be a generic
    failure mode of ELBO-proxy RL, not an artefact of one reward; we test that
    directly by reporting, over rewards, the fraction in which DiPOD controls the
    gap and the alignment.

  AXIS 3 — beta ablation (paper Appendix E.2) + multiple inits with variance.
    Sweep beta in {0, 0.05, 0.1, 0.2, 0.5} and report mean +/- std over random
    reward+init seeds, reproducing the paper's beta ablation.

FPO update (paper Eq. 4):   g = E_pi[ A(a) grad ELBO(a) ].
DiPOD (Algorithm 2):        g = E_pi[ A(a) grad ELBO(a) ] + beta * E_pi[ grad ELBO(a) ].
Reward (paper Appendix D):  r(AA)=0.8, r(AB)=1, r(BA)=0.7, r(BB)=1.
Hyperparameters (Appendix D): lr=0.1, beta_DiPOD=0.2, 1500 steps.
"""

import argparse
import json
import os
import numpy as np

import diffusion as D

SEQS = D.SEQS

# Paper's reward (Appendix D).
PAPER_REWARD = {"AA": 0.8, "AB": 1.0, "BA": 0.7, "BB": 1.0}


# --------------------------------------------------------------------------- #
#  Core: one training run returning per-step measurements.
# --------------------------------------------------------------------------- #
def expected_reward(theta, reward):
    p = D.true_probs(theta)
    z = sum(p.values())
    return sum(p[s] * reward[s] for s in SEQS) / z


def _cos(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-15 or nv < 1e-15:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def policy_grad(theta, reward, score_fn, grad_seq_fn):
    """E_pi[ A(a) grad score(a) ],  A(a)=r(a)-E_pi[r]."""
    probs = D.true_probs(theta)
    z = sum(probs.values())
    pi = {s: probs[s] / z for s in SEQS}
    baseline = sum(pi[s] * reward[s] for s in SEQS)
    g = np.zeros_like(theta)
    for s in SEQS:
        adv = reward[s] - baseline
        g += pi[s] * adv * grad_seq_fn(theta, s)
    return g


def fpo_dipod_gradient(theta, reward, beta):
    """Ascent direction: E_pi[A grad ELBO] + beta * E_pi[grad ELBO]."""
    probs = D.true_probs(theta)
    z = sum(probs.values())
    pi = {s: probs[s] / z for s in SEQS}
    baseline = sum(pi[s] * reward[s] for s in SEQS)
    g = np.zeros_like(theta)
    for s in SEQS:
        adv = reward[s] - baseline
        gE = D.grad_elbo_seq(theta, s)
        g += pi[s] * (adv + beta) * gE
    return g


def train_once(reward, beta, steps, lr, init=None):
    theta = D.init_params() if init is None else init.copy()
    hist = []
    for step in range(steps):
        # ---- metrics BEFORE the step (measured at current theta) -------------
        gaps = D.gaps(theta)
        g_true = policy_grad(theta, reward, None, D.grad_logpi_seq)
        g_proxy = policy_grad(theta, reward, None, D.grad_elbo_seq)
        # proxy gradient that DiPOD would use (includes the beta*ELBO term). For
        # the alignment metric we compare the *PG-integrand proxy* (ELBO score)
        # vs true PG -- that is the "second drift" the paper defines.
        hist.append({
            "step": step,
            "reward": float(expected_reward(theta, reward)),
            "gap_AA": float(gaps["AA"]),
            "gap_mean": float(D.mean_gap(theta)),
            "align": _cos(g_proxy, g_true),     # cosine(proxy PG, true PG)
        })
        # ---- ascent step ----------------------------------------------------
        g = fpo_dipod_gradient(theta, reward, beta)
        theta = theta + lr * g
    # final-step metrics
    gaps = D.gaps(theta)
    g_true = policy_grad(theta, reward, None, D.grad_logpi_seq)
    g_proxy = policy_grad(theta, reward, None, D.grad_elbo_seq)
    hist.append({
        "step": steps,
        "reward": float(expected_reward(theta, reward)),
        "gap_AA": float(gaps["AA"]),
        "gap_mean": float(D.mean_gap(theta)),
        "align": _cos(g_proxy, g_true),
    })
    return hist


# --------------------------------------------------------------------------- #
#  Axis 2 / 3: random rewards and inits.
# --------------------------------------------------------------------------- #
def random_reward(rng):
    """Uniform[0,1) reward per sequence (kept non-constant, so RL has signal)."""
    while True:
        r = {s: float(rng.uniform(0.0, 1.0)) for s in SEQS}
        if len(set(round(v, 6) for v in r.values())) > 1:
            return r


def random_init(rng, scale=0.4):
    return rng.normal(0.0, scale, size=D.N_PARAMS)


# --------------------------------------------------------------------------- #
#  Drivers
# --------------------------------------------------------------------------- #
def run_paper_single(outdir):
    """Axis 0: the paper's exact single-run (one reward, one init, beta 0/0.2),
    with the new per-step alignment metric added. This is the headline Figure-2
    reproduction, now also reporting the second drift directly."""
    res = {}
    for beta in [0.0, 0.2]:
        res[f"beta={beta}"] = train_once(PAPER_REWARD, beta, 1500, 0.1)
    out = os.path.join(outdir, "paper_single.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print("wrote", out)
    return res


def run_reward_robust(outdir, n_rewards, betas, steps, lr, seed):
    """Axis 2: for many random rewards, run FPO and FPO+DiPOD and record the
    final gap / alignment / reward. Summarises how often DiPOD controls the drift
    across rewards (not just on the paper's one)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rewards):
        r = random_reward(rng)
        for beta in betas:
            h = train_once(r, beta, steps, lr)
            rows.append({
                "reward_i": i,
                "beta": beta,
                "final_gap_AA": h[-1]["gap_AA"],
                "final_gap_mean": h[-1]["gap_mean"],
                "final_align": h[-1]["align"],
                "final_reward": h[-1]["reward"],
            })
    out = os.path.join(outdir, "reward_robust.json")
    with open(out, "w") as f:
        json.dump(rows, f)
    print("wrote", out, f"({n_rewards} rewards x {len(betas)} betas = {len(rows)} runs)")
    return rows


def run_beta_ablation(outdir, betas, n_seeds, steps, lr, seed):
    """Axis 3: beta ablation with random reward+init seeds, mean +/- std."""
    rng = np.random.default_rng(seed)
    res = {f"beta={b}": [] for b in betas}
    for s in range(n_seeds):
        r = random_reward(rng)
        init = random_init(rng)
        for beta in betas:
            h = train_once(r, beta, steps, lr, init=init)
            res[f"beta={beta}"].append({
                "seed": s,
                "final_gap_AA": h[-1]["gap_AA"],
                "final_gap_mean": h[-1]["gap_mean"],
                "final_align": h[-1]["align"],
                "final_reward": h[-1]["reward"],
                "peak_gap_AA": max(x["gap_AA"] for x in h),
            })
    out = os.path.join(outdir, "beta_ablation.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print("wrote", out, f"({len(betas)} betas x {n_seeds} seeds)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n_rewards", type=int, default=20)
    ap.add_argument("--n_seeds", type=int, default=12)
    ap.add_argument("--ablation_betas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--robust_betas", type=float, nargs="+", default=[0.0, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".openresearch/artifacts")
    ap.add_argument("--axes", nargs="+", default=["paper", "robust", "ablation"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if "paper" in args.axes:
        print("=== AXIS 0: paper single run (with alignment) ===")
        run_paper_single(args.outdir)
    if "robust" in args.axes:
        print("=== AXIS 2: reward robustness ===")
        run_reward_robust(args.outdir, args.n_rewards, args.robust_betas,
                          args.steps, args.lr, args.seed)
    if "ablation" in args.axes:
        print("=== AXIS 3: beta ablation ===")
        run_beta_ablation(args.outdir, args.ablation_betas, args.n_seeds,
                          args.steps, args.lr, args.seed)
    print("=== all axes done ===")


if __name__ == "__main__":
    main()
