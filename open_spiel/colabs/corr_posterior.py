"""Learned-rho Bayesian backup: the posterior function and its helpers.

Implements Steps 1, 2 and the Dirichlet-multinomial likelihood of the spec
"ThompsonZero v2: Learned-rho Bayesian Backup" (one posterior function whose only
free parameter, the leaf-error correlation rho, is predicted per position by the
network and trained only on real game outcomes).

THIS MODULE IS DELIBERATELY STANDALONE: no game, no tree, no network.  The tests
(test_corr_posterior.py) can therefore pin the maths down exactly, before any of
it is wired into search or training.

What the posterior does.  A node has two error-correlated estimators of its
outcome distribution theta: the network PRIOR (mean mu0, total variance u0) and
the SEARCH mean mus (the average of backed-up leaf means, variance Ts).  They
share error with covariance r, so the best linear combination is the
generalised-least-squares weight

    w = (Ts - r) / (u0 + Ts - 2r)            (weight on the prior)

and the combined mean and variance are mu = w mu0 + (1-w) mus and
T = w^2 u0 + (1-w)^2 Ts + 2 w (1-w) r.  A Dirichlet with that mean and TOTAL
variance T (trace of its covariance, (1-|mu|^2)/(alpha0+1)) is the posterior; the
inversion alpha0 = (1-|mu|^2)/T - 1 is the only fitting step in the system.

Two implementations with identical semantics:
  * posterior_scalar -- pure-Python float64, for the search hot path
  * posterior_t      -- batched torch, differentiable in rho, for training
"""
import math
from dataclasses import dataclass

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:                                       # pragma: no cover
    torch = None
    _HAS_TORCH = False


@dataclass
class CorrConfig:
    """Numerical guards, not modelling choices: changing any of these within an
    order of magnitude should not change results (spec, 'Numerical safeguards')."""
    rho_min: float = 1e-3
    rho_max: float = 0.999
    eps: float = 1e-12           # denominator guard for the GLS weight
    T_min: float = 1e-10         # below this the node is treated as resolved
    a0_min: float = 0.05         # floor on concentration from very dispersed leaves
    a0_max: float = 1e6          # cap for spikes; replaces TERMINAL_CONC
    alpha_floor: float = 1e-4    # per-component floor: gamma sampling / lgamma stay finite


DEFAULT = CorrConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 -- accumulator and leaf payload
# ══════════════════════════════════════════════════════════════════════════════
# Flat 9-vector layout, shared by the stored replay record and the torch side:
#     [ n, SM0, SM1, SM2, SQ0, SQ1, SQ2, SU, ST ]
ACC_N, ACC_SM, ACC_SQ, ACC_SU, ACC_ST = 0, slice(1, 4), slice(4, 7), 7, 8
ACC_DIM = 9


class Acc:
    """Reference accumulator (float64).  The search uses plain lists for speed
    (see connect4_dirichlet_utils._new_acc); this class is the readable spec
    version and what the tests are written against."""
    __slots__ = ('n', 'SM', 'SQ', 'SU', 'ST')

    def __init__(self):
        self.n = 0
        self.SM = np.zeros(3)
        self.SQ = np.zeros(3)
        self.SU = 0.0
        self.ST = 0.0

    def add(self, m, u):
        m = np.asarray(m, dtype=np.float64)
        self.n += 1
        self.SM = self.SM + m
        self.SQ = self.SQ + m * m
        self.SU += float(u)
        self.ST += math.sqrt(max(float(u), 0.0))
        return self

    def copy(self):
        c = Acc()
        c.n, c.SM, c.SQ, c.SU, c.ST = self.n, self.SM.copy(), self.SQ.copy(), self.SU, self.ST
        return c

    def to_vec(self):
        return np.concatenate([[self.n], self.SM, self.SQ, [self.SU, self.ST]])

    @staticmethod
    def from_vec(v):
        a = Acc()
        a.n = int(round(float(v[0])))
        a.SM = np.asarray(v[1:4], dtype=np.float64).copy()
        a.SQ = np.asarray(v[4:7], dtype=np.float64).copy()
        a.SU, a.ST = float(v[7]), float(v[8])
        return a

    def flipped(self):
        """Perspective change: swap W and L.  u is invariant under flip (||mu||^2
        is), so SU and ST are untouched."""
        a = self.copy()
        a.SM = self.SM[[2, 1, 0]].copy()
        a.SQ = self.SQ[[2, 1, 0]].copy()
        return a


def flip(v):
    """(W, D, L) -> (L, D, W)."""
    return np.asarray(v)[..., [2, 1, 0]]


def leaf_payload(alpha_leaf=None, terminal_outcome=None):
    """(m, u) for one backed-up leaf.  A terminal or solver-proven leaf is an
    exact one-hot with u = 0: it is not an estimate, so it adds no evaluation
    variance and a resolved subtree can reach a0_max."""
    if terminal_outcome is not None:
        m = np.zeros(3)
        m[int(terminal_outcome)] = 1.0
        return m, 0.0
    alpha_leaf = np.asarray(alpha_leaf, dtype=np.float64)
    a0 = alpha_leaf.sum()
    m = alpha_leaf / a0
    return m, (1.0 - float(m @ m)) / (a0 + 1.0)


def total_variance(alpha):
    """u(alpha) = (1 - |mu|^2) / (alpha0 + 1): trace of the Dirichlet covariance."""
    alpha = np.asarray(alpha, dtype=np.float64)
    a0 = alpha.sum(-1)
    mu = alpha / a0[..., None]
    return (1.0 - (mu * mu).sum(-1)) / (a0 + 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 -- the posterior (scalar float64 version, search hot path)
# ══════════════════════════════════════════════════════════════════════════════
def posterior_scalar(prior, n, SM, SQ, SU, ST, rho, cfg=DEFAULT,
                     search_only=False, aux=None):
    """Posterior Dirichlet (3 floats) from a prior and an accumulator.

    prior : 3-sequence, the network's Dirichlet for this node/edge
    n, SM, SQ, SU, ST : accumulator fields (SM, SQ are 3-sequences)
    rho   : leaf-error correlation, clipped to [rho_min, rho_max]
    search_only : force w = 0 (the search estimate alone); used by the limit tests
    aux   : optional dict, filled with {Ts, r, w, T, w_clipped, resolved}

    Pure Python on purpose: this runs at every backup level of every simulation,
    where numpy's per-call dispatch overhead on 3-vectors dominates the arithmetic.
    """
    if rho < cfg.rho_min:
        rho = cfg.rho_min
    elif rho > cfg.rho_max:
        rho = cfg.rho_max
    pw, pd, pl = float(prior[0]), float(prior[1]), float(prior[2])
    if n <= 0:
        return (pw, pd, pl)
    a0p = pw + pd + pl
    m0w, m0d, m0l = pw / a0p, pd / a0p, pl / a0p
    u0 = (1.0 - (m0w * m0w + m0d * m0d + m0l * m0l)) / (a0p + 1.0)
    if u0 < 0.0:
        u0 = 0.0

    inv_n = 1.0 / n
    msw, msd, msl = SM[0] * inv_n, SM[1] * inv_n, SM[2] * inv_n
    if n >= 2:
        # trace of the sample covariance of the leaf means; clamp the rounding
        b_emp = (SQ[0] + SQ[1] + SQ[2]
                 - (SM[0] * SM[0] + SM[1] * SM[1] + SM[2] * SM[2]) * inv_n) / (n - 1)
        if b_emp < 0.0:
            b_emp = 0.0
    else:
        b_emp = 0.0
    # what idiosyncratic leaf noise ALONE must produce; the max() stops the
    # empirical dispersion (which already contains that noise) being added twice
    b_idio = (1.0 - rho) * SU * inv_n
    b_eff = b_emp if b_emp > b_idio else b_idio
    Ts = b_eff * inv_n + rho * ST * ST * inv_n * inv_n
    r = rho * math.sqrt(u0) * ST * inv_n

    if search_only:
        w = 0.0
        clipped = True
    else:
        den = u0 + Ts - 2.0 * r
        if den > cfg.eps:
            w = (Ts - r) / den
            clipped = False
            if w < 0.0:
                w, clipped = 0.0, True
            elif w > 1.0:
                w, clipped = 1.0, True
        else:                      # near-perfectly correlated estimators
            w = 0.0 if Ts <= u0 else 1.0
            clipped = True
    ww = 1.0 - w
    mw = w * m0w + ww * msw
    md = w * m0d + ww * msd
    ml = w * m0l + ww * msl
    T = w * w * u0 + ww * ww * Ts + 2.0 * w * ww * r

    num = 1.0 - (mw * mw + md * md + ml * ml)
    if T <= cfg.T_min or num <= cfg.eps:
        a0 = cfg.a0_max                                  # resolved: spike
        resolved = True
    else:
        a0 = num / T - 1.0
        resolved = False
        if a0 < cfg.a0_min:
            a0 = cfg.a0_min
        elif a0 > cfg.a0_max:
            a0 = cfg.a0_max
    if aux is not None:
        aux.update(Ts=Ts, r=r, w=w, T=T, w_clipped=clipped, resolved=resolved,
                   u0=u0, B_emp=b_emp, B_idio=b_idio)
    fl = cfg.alpha_floor
    aw, ad, al = a0 * mw, a0 * md, a0 * ml
    return (aw if aw > fl else fl, ad if ad > fl else fl, al if al > fl else fl)


def posterior(alpha_prior, acc, rho, cfg=DEFAULT, search_only=False, aux=None):
    """Readable wrapper: (prior 3-vector, Acc, rho) -> numpy 3-vector."""
    out = posterior_scalar(alpha_prior, acc.n, acc.SM, acc.SQ, acc.SU, acc.ST,
                           rho, cfg, search_only=search_only, aux=aux)
    return np.array(out)


def posterior_search_only(acc, rho, cfg=DEFAULT, aux=None):
    """The search estimate alone (w = 0).  Exists only for the limit tests."""
    return posterior((1.0, 1.0, 1.0), acc, rho, cfg, search_only=True, aux=aux)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 -- batched torch version, differentiable in rho (training side)
# ══════════════════════════════════════════════════════════════════════════════
def b_emp_host(acc):
    """Empirical dispersion B_emp from a (..., 9) FLOAT64 numpy accumulator.

    In float32, SQ - SM^2/n cancels badly once n exceeds a few hundred, so the
    training side computes this on the host in float64 and hands the result to
    posterior_t, which then never forms the difference itself."""
    acc = np.asarray(acc, dtype=np.float64)
    n = acc[..., 0]
    SM, SQ = acc[..., 1:4], acc[..., 4:7]
    safe = np.maximum(n, 1.0)
    b = (SQ.sum(-1) - (SM * SM).sum(-1) / safe) / np.maximum(n - 1.0, 1.0)
    return np.where(n >= 2, np.maximum(b, 0.0), 0.0)


if _HAS_TORCH:

    def posterior_t(prior, acc, rho, cfg=DEFAULT, B_emp=None, search_only=False,
                    aux=False):
        """Batched posterior.  prior (...,3), acc (...,9), rho broadcastable to (...).

        Differentiable in rho (and in prior, though training passes the stored
        search-time prior as a constant).  Every branch is a torch.where over
        quantities that are finite on BOTH sides, so no NaN can reach the backward
        pass from the unselected branch.  Rows with n == 0 return the prior.
        """
        n = acc[..., 0]
        SM, SQ = acc[..., 1:4], acc[..., 4:7]
        SU, ST = acc[..., 7], acc[..., 8]
        rho = rho.clamp(cfg.rho_min, cfg.rho_max)
        zero = torch.zeros_like(n)
        one = torch.ones_like(n)

        a0p = prior.sum(-1)
        mu0 = prior / a0p.unsqueeze(-1)
        u0 = ((1.0 - (mu0 * mu0).sum(-1)) / (a0p + 1.0)).clamp_min(0.0)
        # sqrt is only ever applied to a quantity that carries no gradient
        sqrt_u0 = torch.sqrt(u0.detach().clamp_min(1e-30))

        has = n > 0
        ns = n.clamp_min(1.0)
        mus = SM / ns.unsqueeze(-1)
        if B_emp is None:
            b_emp = torch.where(
                n >= 2,
                ((SQ.sum(-1) - (SM * SM).sum(-1) / ns) / (ns - 1.0).clamp_min(1.0)
                 ).clamp_min(0.0), zero)
        else:
            b_emp = B_emp
        b_idio = (1.0 - rho) * SU / ns
        b_eff = torch.maximum(b_emp, b_idio)
        Ts = b_eff / ns + rho * ST * ST / (ns * ns)
        r = rho * sqrt_u0 * ST / ns

        if search_only:
            w = zero
            w_clip = one > 0
        else:
            den = u0 + Ts - 2.0 * r
            ok = den > cfg.eps
            w_raw = (Ts - r) / den.clamp_min(cfg.eps)
            w_lim = torch.where(Ts <= u0, zero, one)
            w_un = torch.where(ok, w_raw, w_lim)
            w = w_un.clamp(0.0, 1.0)
            w_clip = (~ok) | (w_un < 0.0) | (w_un > 1.0)
        mu = w.unsqueeze(-1) * mu0 + (1.0 - w).unsqueeze(-1) * mus
        T = w * w * u0 + (1.0 - w) ** 2 * Ts + 2.0 * w * (1.0 - w) * r

        num = 1.0 - (mu * mu).sum(-1)
        resolved = (T <= cfg.T_min) | (num <= cfg.eps)
        a0_fit = (num.clamp_min(cfg.eps) / T.clamp_min(cfg.T_min) - 1.0
                  ).clamp(cfg.a0_min, cfg.a0_max)
        a0 = torch.where(resolved, torch.full_like(a0_fit, cfg.a0_max), a0_fit)
        alpha = (a0.unsqueeze(-1) * mu).clamp_min(cfg.alpha_floor)
        out = torch.where(has.unsqueeze(-1), alpha, prior)
        if aux:
            return out, dict(w=w, w_clipped=w_clip & has, resolved=resolved & has,
                             T=T, Ts=Ts, u0=u0)
        return out

    def dm_nll(c, alpha, lgamma=None):
        """Dirichlet-multinomial NLL of an ORDERED outcome sequence with counts c.

        With k = sum(c): lgamma(a0+k) - lgamma(a0) - sum(lgamma(alpha+c) - lgamma(alpha)).
        For k = 1 this is exactly -log(alpha_z / alpha0): the MARGINAL form, which
        is what the spec requires.  The digamma form psi(a0) - psi(alpha_z) rewards
        concentration growth and causes inflation, so it is never used here.
        `lgamma` is pluggable because DirectML has no lgamma kernel."""
        lg = lgamma or torch.lgamma
        a0 = alpha.sum(-1)
        k = c.sum(-1)
        return lg(a0 + k) - lg(a0) - (lg(alpha + c) - lg(alpha)).sum(-1)


def dm_nll_np(c, alpha):
    """float64 numpy twin of dm_nll, for tests."""
    from math import lgamma
    c = np.asarray(c, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    a0, k = alpha.sum(-1), c.sum(-1)
    v = np.vectorize(lgamma)
    return v(a0 + k) - v(a0) - (v(alpha + c) - v(alpha)).sum(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Identifying rho from SNAPSHOT DRIFT -- no outcomes, no forks
# ══════════════════════════════════════════════════════════════════════════════
# The spec identifies rho from fork outcomes, arguing that shared evaluation error
# is invisible to more search.  That is true of the search mean's ACCURACY but not
# of its MOTION: for two NESTED snapshots (the first n simulations and the first N,
# n < N) the shared error b appears in both means and CANCELS in the difference, so
# the drift sees the idiosyncratic part alone.  With independent leaf errors
#
#     mu_n - mu_N = (1/n - 1/N) sum_{i<=n} e_i  -  (1/N) sum_{n<i<=N} e_i
#     E|| mu_n - mu_N ||^2 = (1 - rho) * [ (1/n - 1/N)^2 SU_n + (SU_N - SU_n)/N^2 ]
#
# and SU_n (the cumulative sum of leaf total variances) is already in the
# accumulator.  So 1 - rho is a ratio of two observable sums and rho needs no
# outcomes at all.  Measured on the spec's own synthetic generator: rho recovered
# to within 0.01 from 400 positions, against the k>=16 outcomes per position the
# outcome likelihood needs.
#
# TWO LIMITS, both real:
#   * rho is identified RELATIVE to the leaf concentrations the network reports.
#     Those enter through SU, so if the net claims half the leaf variance it
#     really has, this returns 1 - (1-rho)/0.5.  The outcome terms (cev/cea) are
#     what keep the concentrations honest; drift cannot do it alone.  Partially
#     self-cancelling in the posterior: the (1-rho)*SU/n term is invariant to that
#     rescaling, the rho*ST^2/n^2 term is not.
#   * the leaves must be conditionally independent given the shared error.  The
#     DENOMINATOR above handles heterogeneous leaf variances exactly (verified:
#     rho_hat stays 0.28-0.30 while a naive single-u version collapses to -1.6
#     when later leaves are quieter), but genuinely CORRELATED idiosyncratic
#     errors -- two leaves in the same subtree misreading the same tactic -- are
#     indistinguishable from the shared term and inflate rho.  Arguably that is
#     the quantity the posterior wants anyway.
def drift_terms(snaps, ns=None):
    """Per-(position, snapshot-pair) drift and its model coefficient.

    snaps : (..., J, 9) accumulators for ONE node at increasing n (nested).
    Returns (d, delta): d = ||mu_n - mu_N||^2, delta = the bracket above, each of
    shape (..., P) with P = J*(J-1)/2 pairs.  1 - rho = sum(d) / sum(delta)."""
    snaps = np.asarray(snaps, dtype=np.float64)
    if snaps.ndim < 2 or snaps.shape[-2] < 2:
        # fewer than two snapshots: no nested pair, so no drift information
        base = snaps.shape[:-2] if snaps.ndim >= 2 else ()
        z = np.zeros(base + (0,))
        return z, z
    J = snaps.shape[-2]
    n = snaps[..., 0] if ns is None else np.broadcast_to(
        np.asarray(ns, dtype=np.float64), snaps.shape[:-1])
    SM, SU = snaps[..., 1:4], snaps[..., 7]
    ds, dl = [], []
    for j in range(J):
        for k in range(j + 1, J):
            nj, nk = n[..., j], n[..., k]
            ok = (nj > 0) & (nk > nj)
            nj_s, nk_s = np.maximum(nj, 1.0), np.maximum(nk, 1.0)
            diff = SM[..., j, :] / nj_s[..., None] - SM[..., k, :] / nk_s[..., None]
            inv = 1.0 / nj_s - 1.0 / nk_s
            delta = inv * inv * SU[..., j] + (SU[..., k] - SU[..., j]) / (nk_s * nk_s)
            ds.append(np.where(ok, (diff * diff).sum(-1), 0.0))
            dl.append(np.where(ok, np.maximum(delta, 0.0), 0.0))
    return np.stack(ds, -1), np.stack(dl, -1)


def rho_from_drift(snaps, ns=None, cfg=DEFAULT):
    """Pooled moment estimate of rho from nested snapshots.  Returns (rho, n_pairs)."""
    d, delta = drift_terms(snaps, ns)
    den = float(delta.sum())
    if den <= 0.0:
        return float('nan'), 0
    rho = 1.0 - float(d.sum()) / den
    return min(max(rho, cfg.rho_min), cfg.rho_max), int((delta > 0).sum())


if _HAS_TORCH:

    def drift_nll(d, delta, rho, floor=1e-12):
        """Differentiable loss tying rho(s) to observed snapshot drift.

        d, delta: (..., P) from drift_terms, CONSTANTS (no gradient).  Treats the
        drift as exponential with mean V = (1-rho)*delta -- the simplex has 2 free
        dimensions, so a chi-square with 2 df, i.e. exponential, is the right shape
        -- giving NLL = log V + d/V.  Its stationary point is E[d] = V, which is
        exactly the moment condition, so the loss and the closed-form estimator
        agree.  Gradient flows only into rho (and the trunk through it)."""
        V = ((1.0 - rho).unsqueeze(-1) if rho.dim() == d.dim() - 1 else (1.0 - rho)) \
            * delta
        V = V.clamp_min(floor)
        ds = d.clamp_min(floor)
        mask = (delta > 0).to(d.dtype)
        # EXCESS NLL: log(V/d) + d/V - 1 >= 0, zero iff V == d.  The raw
        # log V + d/V is a density NLL and is legitimately large and NEGATIVE
        # here (drifts are ~1e-5, so log V ~ -11), which would swamp every
        # other term in the reported total and make the loss incomparable to
        # runs without it.  Subtracting the per-pair minimum (log d + 1) is a
        # constant in rho, so the gradient and the argmin are untouched.
        per = (torch.log(V / ds) + ds / V - 1.0) * mask
        return per.sum(-1) / mask.sum(-1).clamp_min(1.0)
