"""Two-token masked discrete-diffusion policy, parameterised EXACTLY as in the
DiPOD paper, Appendix D (which follows SPG's Appendix C.3 toy experiment).

Six real logits parameterise the model:

    a = logit pi(x1 = A | x = MA)      b = logit pi(x1 = A | x = MM)
    c = logit pi(x2 = A | x = AM)      d = logit pi(x2 = A | x = MM)
    e = logit pi(x1 = A | x = MB)      f = logit pi(x2 = A | x = BM)

with S(.) the sigmoid (inverse logit). Generation starts from MM and decodes the
two tokens in a uniformly random order. Because the whole state space is four
clean sequences {AA, AB, BA, BB}, the true log-likelihood log pi, the ELBO, and
the variational gap D^L = log pi - ELBO are all available in closed form, so the
drift is measured EXACTLY (no Monte-Carlo, matching the paper).

The paper's worked example (Appendix D), reproduced exactly by this module:
    log pi(AA) = log( 1/2 S(b)S(c) + 1/2 S(a)S(d) )
    ELBO(AA)   = 1/2 ( log S(a) + log S(b) + log S(c) + log S(d) )

Parameters are initialised to 0 (probability 0.5), so ELBO = log pi for every
output at init -- a perfectly tight, "pretrained" diffusion model. RL then moves
the policy and the bound can drift.
"""

import numpy as np

SEQS = ["AA", "AB", "BA", "BB"]
N_PARAMS = 6                      # a, b, c, d, e, f
IA, IB, IC, ID, IE, IF = range(6)


def init_params():
    """All logits 0  ->  every conditional prob 0.5  ->  ELBO == log pi (tight)."""
    return np.zeros(N_PARAMS, dtype=np.float64)


def _S(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logS(x):                     # numerically stable log sigmoid
    return -np.logaddexp(0.0, -x)


def _p_tok(param, is_A):
    """prob the token equals the realised value: S(param) if A else 1 - S(param)."""
    return _S(param) if is_A else _S(-param)


def _logp_tok(param, is_A):
    return _logS(param) if is_A else _logS(-param)


def true_logprobs(theta):
    """Exact log pi for the four clean sequences, marginalising the random
    decode order (order (1,2) and (2,1), each probability 1/2)."""
    a, b, c, d, e, f = theta
    out = {}
    for s in SEQS:
        A1 = s[0] == "A"
        A2 = s[1] == "A"
        # order (1,2): x1 | MM (param b), then x2 | (x1, M)  (param c if x1=A else f)
        p_12 = _p_tok(b, A1) * _p_tok(c if A1 else f, A2)
        # order (2,1): x2 | MM (param d), then x1 | (M, x2)  (param a if x2=A else e)
        p_21 = _p_tok(d, A2) * _p_tok(a if A2 else e, A1)
        out[s] = np.log(0.5 * p_12 + 0.5 * p_21)
    return out


def true_probs(theta):
    return {s: np.exp(lp) for s, lp in true_logprobs(theta).items()}


def elbo_all(theta):
    """Exact masked-diffusion ELBO for each clean sequence.

    ELBO(x1,x2) = 1/2 [ log p(x1|MM) + log p(x2|MM)
                        + log p(x1|M,x2) + log p(x2|x1,M) ]
    which reduces to the paper's ELBO(AA) = 1/2(logS(a)+logS(b)+logS(c)+logS(d)).
    """
    a, b, c, d, e, f = theta
    out = {}
    for s in SEQS:
        A1 = s[0] == "A"
        A2 = s[1] == "A"
        lp_x1_MM = _logp_tok(b, A1)                  # x1 | MM
        lp_x2_MM = _logp_tok(d, A2)                  # x2 | MM
        lp_x1_x2 = _logp_tok(a if A2 else e, A1)     # x1 | (M, x2)
        lp_x2_x1 = _logp_tok(c if A1 else f, A2)     # x2 | (x1, M)
        out[s] = 0.5 * (lp_x1_MM + lp_x2_MM + lp_x1_x2 + lp_x2_x1)
    return out


def gaps(theta):
    lp = true_logprobs(theta)
    eb = elbo_all(theta)
    return {s: lp[s] - eb[s] for s in SEQS}


def mean_gap(theta):
    """On-policy expected gap E_{a~pi}[D^L(a)]."""
    g = gaps(theta)
    p = true_probs(theta)
    z = sum(p.values())
    return sum(p[s] * g[s] for s in SEQS) / z
