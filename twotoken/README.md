# Two-token diffusion diagnostic — minimal DiPOD reproduction

A self-contained, **CPU-only** reproduction of the DiPOD paper's core claim
(arXiv 2606.13795, Section 4.1 / Appendix D — the "two-token post-training"
controlled diagnostic, following SPG's toy setting).

## What it shows

A fully-enumerable two-token masked discrete-diffusion policy (vocab {A,B},
6 logits `a..f`, all initialised to 0 so the ELBO is tight at start) is
RL-post-trained on a fixed reward. The whole state space is the four sequences
{AA, AB, BA, BB}, so the true log-likelihood `log pi`, the `ELBO`, and the
variational gap `D^L = log pi - ELBO` are all computed **exactly** (no
Monte-Carlo — as the paper specifies). This makes the **double drift** directly
visible:

- **FPO** (policy gradient on the ELBO score, paper Eq. 4): the ELBO–likelihood
  discrepancy — tracked on AA, as in the paper's Figure 2 — **drifts up** as RL
  proceeds.
- **FPO+DiPOD** (FPO + a `beta * ELBO` on-policy regulariser, paper Algorithm 2;
  the same term as `language/d1/diffu-grpo/diffu_grpo_trainer.py:160`): the gap
  stays **controlled**, while reward is comparable.

## Run

```bash
bash twotoken/run.sh
```

Writes `EVAL.md` at the repo root and, under `.openresearch/artifacts/`,
`history.json` and `twotoken_diagnostic.png`.

## Files

- `diffusion.py` — the 6-parameter model; exact `log pi`, `ELBO`, gap (closed form).
- `train.py` — exact FPO / FPO+DiPOD policy-gradient training.
- `plot.py` — reproduces the Figure-2-style gap curves.
- `eval.py` — writes the quantitative `EVAL.md` verdict.

## Settings (paper Appendix D)

Reward `r(AA)=0.8, r(AB)=1, r(BA)=0.7, r(BB)=1`; `lr=0.1`; `beta_DiPOD=0.2`;
1500 steps.

## Scope

This is the **minimal** illustration of the paper's central mechanism. The full
language (LLaDA-8B on GSM8K/MATH500/Countdown/Sudoku) and control (IsaacLab
G1 motion tracking) experiments live under `language/` and `fpo-control/` and
need 8×H100 / Isaac Sim respectively.
