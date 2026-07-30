"""ThompsonZero-C4 — Dirichlet state/action value learning on Connect 4.

This module is the ENTIRE implementation; the companion notebook
(`connect4_dirichlet_training.ipynb`) holds only a Config, one call to
`run_training`, the plots, and the arena.  It is a radical simplification of the
chess "ThompsonZero-FULL" methodology (`chess_thompson_full_utils.py`) that keeps
the same idea — Thompson sampling over Dirichlet value beliefs — but throws the
policy head away entirely and replaces the recursive value backup with plain
distributional averaging.

═══ THE NETWORK ═══════════════════════════════════════════════════════════════
A ResNet trunk feeding TWO heads, each emitting a Dirichlet over the 3-way
outcome (win, draw, loss) FROM THE MOVER'S PERSPECTIVE:

  (1) STATE  head: 4 numbers  (p_win, p_draw, p_loss, c)   → V(s)   = Dir(c·p)
  (2) ACTION head: 4 numbers PER ACTION                    → Q(s,a) = Dir(c_a·p_a)

That is the whole network.  There is no policy head, no scalar value head, and
nothing derived — both beliefs are emitted directly.

═══ THE SEARCH ════════════════════════════════════════════════════════════════
Expanding a node stores V(s) and every Q(s,a).  Selection is pure Thompson
sampling: draw one (x_w, x_d, x_l) per legal action from that action's CURRENT
belief and descend the argmax of the score x_w − x_l (chess/Connect-4 utility
scale, +1/0/−1).  Descend until an unexpanded or terminal edge is hit, expand it,
then back up ONE Dirichlet — the freshly expanded leaf's own STATE belief V(leaf)
(or a proven-terminal spike) — flipping win↔loss at every ply.

Every node and every edge keeps a running EVIDENCE ACCUMULATOR of the Dirichlets
backed up through it, and its "observed" belief is that collection collapsed back
to a single Dirichlet.  The collapse is O(1) from four running sums
(n, Σm, Σm², Σv) — no recomputation, no per-visit history.  Three collapse rules
are implemented (see `AGG_*` and `observed_alpha`), selectable independently for
SEARCH and for TRAINING TARGETS, because the two roles want different things:

  'mixture'  moment-match the MIXTURE (1/n)Σ Dir(α_i).  Concentration measures
             how much the backed-up evaluations DISAGREE — a property of the
             position, not of the search budget.  Does NOT grow with visits.
  'mean'     moment-match the AVERAGE random variable (1/n)Σ X_i, X_i∼Dir(α_i).
             Var = ΣVar_i/n², so concentration ≈ n·α₀ — grows linearly with
             visits, which is what makes Thompson exploration anneal and the
             search converge.  Budget-dependent.
  'sum'      conjugate evidence accumulation, α = Σ α_i.  Concentration grows
             linearly; the mean is precision-weighted, so one confident leaf
             (e.g. a terminal spike) dominates.
  'additive' back up each leaf's MEAN PROBABILITY rather than its Dirichlet:
             α = Σ m_i, so α₀ is exactly the VISIT COUNT and the mean is the
             running average of leaf means.  One simulation is one soft-labelled
             observation, whatever the leaf's own confidence was, and the belief
             anneals at the textbook Thompson rate sd[v] ∝ 1/√n.  Disagreement
             needs no separate term: conflicting leaves pull the mean toward the
             middle, where M(1−M) — and hence the spread — is largest.
  'additive_mle'  same observations (leaf means), but fit the Dirichlet by
             MAXIMUM LIKELIHOOD rather than counting: solve
             ψ(α_j) = ψ(α₀) + mean_i log x_ij by Minka's fixed point, from the
             sufficient statistics (n, Σ log x).  α₀ then tracks
             min(n, intrinsic dispersion) — it grows while observations are
             scarce and saturates at the population value, so it is neither
             pinned like 'mixture' nor budget-locked like 'additive'.  Measured
             on real searched nodes the dispersion is only ~8-12, so it
             saturates within ~10 visits and lands near 'mixture' in magnitude.

Selection samples from  α_sel = α_net(s,a) + α_observed(s,a)  — the network's
belief is a PRIOR whose pseudo-counts search adds evidence to, so a well-searched
edge is sampled ever more tightly around what search actually found.  Proven
edges (MCTS-Solver overlay) override both with a terminal spike.  After the
configured number of simulations, ONE more Thompson sample at the root picks the
self-play move.

═══ THE LOSSES ════════════════════════════════════════════════════════════════
Five terms, all closed-form and differentiable:

  L_klv   KL( Ô(s)    ‖ V_θ(s)     )   observed state belief   → state head
  L_kla   KL( Ô(s,a)  ‖ Q_θ(s,a)   )   observed action beliefs → action head,
                                       scored only on edges search actually
                                       touched (visited or proven)
  L_cev   CE( z , p_θ(s)     )         game outcome, from s's mover's view
  L_cea   CE( z , p_θ(s,a*)  )         same, on the action actually played
  L_cons  KL( flip(V_θ(s'))  ‖ Q_θ(s,a*) )   self-consistency: taking a* must be
                                       worth what the resulting state is worth,
                                       one ply flipped (state side stop-gradient)

The CE terms are what ground everything in real results; the KL terms are what
distil search back into the net.  There is no z-mixing into the value targets
(the chess run's `z_mix`) — the CE terms do that job directly and better.

═══ WHAT WAS KEPT ═════════════════════════════════════════════════════════════
Every performance optimisation from the chess implementation: multiprocess
self-play workers behind one central batched GPU inference server, batched-leaf
waves with virtual loss and leaf de-duplication, O(1) incremental accumulators,
exact-vs-Gaussian selection sampling, subtree reuse between moves, fp16
observations, on-device Dirichlet KL with DirectML-friendly lgamma/digamma,
LerpFreeAdamW, torch.compile, checkpoint/resume — plus the whole logging and
two-tier evaluation scheme (self-erasing progress bar, per-eval diagnostic line
with weighted loss shares and concentration pred/target, search-free quick match,
MCTS running-Elo pool, arena duels).

What was dropped: the policy-prior head and its KL, the Stockfish supervised
bootstrap (chess-only), and `z_mix`.  The endgame-first curriculum and the
backward-restart pool are still here but default OFF — a Connect 4 game is at
most 42 plies, so full self-play from move 1 is already cheap and on-distribution.

Only the torch-dependent pieces need torch; the tree/search/target logic is pure
numpy and imports (and self-tests) without torch or a GPU.
"""

import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                   # numpy-only (tree unit tests)
    torch = None
    _HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  Outcome convention + Dirichlet primitives (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════
# Every belief is a Dirichlet over (win, draw, loss) FROM THE MOVER'S PERSPECTIVE.
# One ply up the tree the mover changes, so the belief FLIPS (win<->loss, draw
# fixed).  The scalar used for Thompson selection is v = p_win − p_loss ∈ [−1, 1],
# exactly Connect 4's own utility scale (+1 / 0 / −1).
_WIN, _DRAW, _LOSS = 0, 1, 2
_FLIP = np.array([_LOSS, _DRAW, _WIN])
_FLIP_TERM = np.array([_LOSS, _DRAW, _WIN], dtype=np.int8)

# A Dirichlet is defined only for α > 0, and an fp32 softmax/softplus can underflow
# to exactly 0 (which makes lgamma/digamma infinite and the loss NaN).  ALPHA_FLOOR
# is that positivity epsilon and nothing else — nine orders of magnitude below any
# concentration the net will ever learn, so it never shapes anything.
ALPHA_FLOOR = 1e-9

# TARGET Dirichlets get a real smoothing pseudo-count instead, because the two
# sides of KL(target ‖ prediction) respond very differently to a near-zero
# component: the target side goes through digamma (ψ(1e-9) = −1e9) while the
# prediction side goes through lgamma (mild, logarithmic).  Laplace-style
# smoothing of an unobserved count — a modelling choice, not a cap.
TARGET_EPS = 0.05

# A proven terminal is a point mass, and no Dirichlet represents one (its KL would
# be infinite), so a proof is asserted as a finite spike.  Both numbers say how
# hard a proof is asserted; neither bounds anything the network can output.
TERMINAL_CONC = 100.0
TERMINAL_EPS = 0.05

_SPIKE = np.full((3, 3), TERMINAL_EPS)
_SPIKE[_WIN, _WIN] = _SPIKE[_DRAW, _DRAW] = _SPIKE[_LOSS, _LOSS] = TERMINAL_CONC

# Evidence-collapse rules (see the module docstring).  Chosen independently for
# search and for training targets via Config.search_agg / Config.target_agg.
AGG_MIXTURE = 'mixture'
AGG_MEAN = 'mean'
AGG_SUM = 'sum'
AGG_ADDITIVE = 'additive'
AGG_ADDITIVE_MLE = 'additive_mle'
AGGREGATIONS = (AGG_MIXTURE, AGG_MEAN, AGG_SUM, AGG_ADDITIVE, AGG_ADDITIVE_MLE)

# ── 'additive_mle' tuning.  Speed is preferred to exactness here: the estimate
# is a search statistic, not a reported quantity, and it moves slowly.
_MLE_FLOOR = 1e-9      # clamp on x before log().  A single zero component would
                       # set L[j] = -inf permanently and NaN the node forever.
_MLE_GROWTH = 1.3      # refresh the fixed point when n reaches n*GROWTH, not
                       # every update: ~23 refreshes over a 400-sim search
                       # instead of 400, and MORE accurate than one step per
                       # update because each refresh iterates to near-convergence
_MLE_STEPS = 12        # Minka iterations per refresh
_MLE_NEWTON = 2        # Newton steps inside inverse-digamma (0.8% max rel err;
                       # 3 would give ~1e-5 for ~15% more cost)


# ── Scalar digamma / trigamma / inverse-digamma (pure Python, no numpy dispatch
# on the backup hot path).  Shift up to 6 by recurrence, then asymptotic series;
# accurate to ~1e-10 against scipy over x in [1e-3, 500].
_DIGAMMA_1 = -0.5772156649015329            # digamma(1) = -Euler-Mascheroni


def _digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x + f * (
        -1.0 / 12 + f * (1.0 / 120 + f * (-1.0 / 252 + f * (1.0 / 240))))


def _trigamma(x):
    r = 0.0
    while x < 6.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + (1.0 / x) * (1.0 + 0.5 / x + f * (
        1.0 / 6 + f * (-1.0 / 30 + f * (1.0 / 42 + f * (-1.0 / 30)))))


def _inv_digamma(y):
    """psi^-1(y) by Minka's initialiser plus a couple of Newton steps."""
    x = math.exp(y) + 0.5 if y >= -2.22 else -1.0 / (y - _DIGAMMA_1)
    for _ in range(_MLE_NEWTON):
        x -= (_digamma(x) - y) / _trigamma(x)
        if x < 1e-9:
            x = 1e-9
    return x


def _mle_refresh(acc):
    """One batch of Minka fixed-point steps from the sufficient statistics.
    The MLE solves  psi(alpha_j) = psi(alpha_0) + mean_i log x_ij,  so (n, L) is
    all the state needed and the iteration warm-starts from the last estimate."""
    inv_n = 1.0 / acc[0]
    L, A = acc[5], acc[6]
    g0, g1, g2 = L[0] * inv_n, L[1] * inv_n, L[2] * inv_n
    a0, a1, a2 = A[0], A[1], A[2]
    for _ in range(_MLE_STEPS):
        c = _digamma(a0 + a1 + a2)
        a0 = _inv_digamma(c + g0)
        a1 = _inv_digamma(c + g1)
        a2 = _inv_digamma(c + g2)
    A[0], A[1], A[2] = a0, a1, a2


def flip_alpha(alpha):
    """Reflect a (…,3) belief one ply up: swap the win and loss columns."""
    return np.ascontiguousarray(np.asarray(alpha)[..., _FLIP])


def dir_mean(alpha):
    """Mean of a Dirichlet (…,3)."""
    a = np.asarray(alpha, dtype=np.float64)
    return a / a.sum(-1, keepdims=True)


def dir_value(alpha):
    """Scalar value v = E[p_win] − E[p_loss] of a belief (…,3)."""
    m = dir_mean(alpha)
    return m[..., _WIN] - m[..., _LOSS]


def dir_value_mean_var(alpha):
    """(E[v], Var[v]) for v = p_win − p_loss under Dir(alpha), closed form.
    For a Dirichlet Cov(x_i,x_j) = −m_i m_j/(α₀+1) and Var(x_i) = m_i(1−m_i)/(α₀+1),
    so Var(x_w − x_l) = (m_w(1−m_w) + m_l(1−m_l) + 2 m_w m_l)/(α₀+1)."""
    a = np.asarray(alpha, dtype=np.float64)
    a0 = a.sum(-1)
    m = a / a0[..., None]
    mw, ml = m[..., _WIN], m[..., _LOSS]
    return mw - ml, (mw * (1 - mw) + ml * (1 - ml) + 2 * mw * ml) / (a0 + 1.0)


def _beta0_from_moments(M, V):
    """Minka moment-match: the single Dirichlet with mean M and total variance
    ΣV has concentration β₀ = ΣM(1−M)/ΣV − 1.  Scalars in, scalar out.

    The only guard is POSITIVITY: a mixture can be more dispersed than ANY single
    Dirichlet, which makes β₀ come out negative — that is an invalid distribution,
    not a cap, so it floors.  Nothing bounds it above."""
    mw, md, ml = M
    vw, vd, vl = V
    if vw < 1e-300: vw = 1e-300
    if vd < 1e-300: vd = 1e-300
    if vl < 1e-300: vl = 1e-300
    b0 = (mw * (1 - mw) + md * (1 - md) + ml * (1 - ml)) / (vw + vd + vl) - 1.0
    return b0 if b0 > ALPHA_FLOOR else ALPHA_FLOOR


def moment_match_mixture(weights, alphas):
    """Collapse Σ_k w_k · Dir(α_k) to one Dirichlet by matching mean + total
    variance (exact, closed form).  A single component round-trips to its own α₀.
    Used by the numpy-only tests and by any caller without accumulators."""
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
    a = np.asarray(alphas, dtype=np.float64)
    a0 = a.sum(1)
    m = a / a0[:, None]
    M = w @ m
    var_c = m * (1.0 - m) / (a0[:, None] + 1.0)
    ex2 = w @ (var_c + m * m)
    var_mix = np.maximum(ex2 - M * M, 1e-300)
    b0 = _beta0_from_moments(tuple(M), tuple(var_mix))
    return np.maximum(M * b0, ALPHA_FLOOR)


def observed_alpha(acc, mode):
    """Collapse one evidence accumulator to a single Dirichlet.

    `acc` is [n, SM(3), SQ(3), SV(3), SA(3)] — running sums over the Dirichlets
    D_i backed up through this node/edge, where m_i is D_i's mean, v_i its
    per-component variance m_i(1−m_i)/(α₀+1), and SA = Σ α_i:
        n  = count          SM = Σ m_i        SQ = Σ m_i²       SV = Σ v_i

    Every rule below is O(1) in the number of observations:
      'mixture'  the mixture (1/n)Σ Dir(α_i):  Var = (SV+SQ)/n − M²
                 → concentration reflects DISAGREEMENT between the backed-up
                   evaluations and is independent of how many there were.
      'mean'     the average draw (1/n)Σ X_i:  Var = SV/n²
                 → concentration ≈ n·α₀, growing linearly with visits.
      'sum'      conjugate evidence:           α = SA  (no moment matching)
      'additive_mle'  Dirichlet MLE of the same observations, from (n, Σ log x)
                 → α₀ ≈ min(n, dispersion of the leaf means).  Falls back to
                   'additive' below 3 observations, where the MLE is degenerate
                   (a single sample is fit perfectly by α₀ → ∞).
      'additive' soft counts:                   α = SM,  so α₀ == n exactly
                 → each simulation is ONE observation whose outcome label is the
                   leaf's mean probability vector.  The leaf's own concentration
                   is deliberately discarded: a confident leaf and a guess both
                   count once, and a proven-terminal spike contributes its mean
                   (≈ one unit on the proven corner) instead of TERMINAL_CONC
                   units.  This is the plain Dirichlet-categorical posterior, so
                   the spread shrinks as 1/√n and conflicting leaves stay
                   uncertain by pulling the mean toward the middle.

    Returns a (3,) float64 array, or None if there is no evidence yet."""
    n = acc[0]
    if n <= 0.0:
        return None
    if mode == AGG_ADDITIVE_MLE:
        # Below 3 observations the Dirichlet MLE is degenerate (one sample is
        # fit perfectly by alpha0 -> infinity), and acc[5] being untouched means
        # the statistics were never maintained.  Both fall back to 'additive',
        # which is where the MLE sits at small n anyway.
        if n >= 3.0 and acc[5][0] < 0.0:
            A = acc[6]
            return np.array([max(A[0], ALPHA_FLOOR), max(A[1], ALPHA_FLOOR),
                             max(A[2], ALPHA_FLOOR)])
        mode = AGG_ADDITIVE
    if mode == AGG_ADDITIVE:
        sm = acc[1]
        return np.array([max(sm[0], ALPHA_FLOOR), max(sm[1], ALPHA_FLOOR),
                         max(sm[2], ALPHA_FLOOR)])
    if mode == AGG_SUM:
        sa = acc[4]
        return np.array([max(sa[0], ALPHA_FLOOR), max(sa[1], ALPHA_FLOOR),
                         max(sa[2], ALPHA_FLOOR)])
    SM, SQ, SV = acc[1], acc[2], acc[3]
    M = (SM[0] / n, SM[1] / n, SM[2] / n)
    if mode == AGG_MEAN:
        nn2 = n * n
        V = (SV[0] / nn2, SV[1] / nn2, SV[2] / nn2)
    else:                                            # AGG_MIXTURE (default)
        V = ((SV[0] + SQ[0]) / n - M[0] * M[0],
             (SV[1] + SQ[1]) / n - M[1] * M[1],
             (SV[2] + SQ[2]) / n - M[2] * M[2])
    b0 = _beta0_from_moments(M, V)
    return np.array([max(M[0] * b0, ALPHA_FLOOR), max(M[1] * b0, ALPHA_FLOOR),
                     max(M[2] * b0, ALPHA_FLOOR)])


def _new_acc():
    """A fresh evidence accumulator, as plain Python floats.  Python lists, not
    numpy: these are mutated once per node per simulation, and numpy's per-op
    dispatch overhead on 3-vectors dwarfs the arithmetic.

        [0] n    [1] SM = sum m   [2] SQ = sum m^2   [3] SV = sum var
        [4] SA = sum alpha
        [5] L    = sum log m      — 'additive_mle' sufficient statistic
        [6] A    = current MLE estimate
        [7] next refresh threshold
    Slots 5-7 are only maintained when an 'additive_mle' rule is selected."""
    return [0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0]


def _payload(alpha):
    """Pre-compute one backed-up Dirichlet's contribution in BOTH perspectives.

    Returns (as_is, flipped); each is (m, m², v, α) as 3-tuples of Python floats.
    A simulation backs one Dirichlet up a whole path, alternating perspective at
    every ply, so both forms are built once per simulation and then just indexed
    — no per-ply array flipping."""
    aw, ad, al = (float(alpha[0]), float(alpha[1]), float(alpha[2]))
    a0 = aw + ad + al
    inv = 1.0 / (a0 + 1.0)
    mw, md, ml = aw / a0, ad / a0, al / a0
    m = (mw, md, ml)
    q = (mw * mw, md * md, ml * ml)
    v = (mw * (1.0 - mw) * inv, md * (1.0 - md) * inv, ml * (1.0 - ml) * inv)
    a = (aw, ad, al)
    fm = (ml, md, mw)
    fq = (q[2], q[1], q[0])
    fv = (v[2], v[1], v[0])
    fa = (al, ad, aw)
    return ((m, q, v, a), (fm, fq, fv, fa))


def _acc_add(acc, pl):
    """Fold one payload into an accumulator.  O(1), pure scalar."""
    m, q, v, a = pl
    n = acc[0] + 1.0
    acc[0] = n
    SM, SQ, SV, SA = acc[1], acc[2], acc[3], acc[4]
    SM[0] += m[0]; SM[1] += m[1]; SM[2] += m[2]
    SQ[0] += q[0]; SQ[1] += q[1]; SQ[2] += q[2]
    SV[0] += v[0]; SV[1] += v[1]; SV[2] += v[2]
    SA[0] += a[0]; SA[1] += a[1]; SA[2] += a[2]
    if _MLE_ACTIVE:
        # Accumulating the logs is cheap; the fixed point is not, so it only
        # runs on the geometric schedule (see _MLE_GROWTH).
        L = acc[5]
        x = m[0]; L[0] += math.log(x if x > _MLE_FLOOR else _MLE_FLOOR)
        x = m[1]; L[1] += math.log(x if x > _MLE_FLOOR else _MLE_FLOOR)
        x = m[2]; L[2] += math.log(x if x > _MLE_FLOOR else _MLE_FLOOR)
        if n >= acc[7]:
            acc[7] = max(n + 1.0, n * _MLE_GROWTH)
            _mle_refresh(acc)


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level search settings (set once from the Config; workers set their own)
# ══════════════════════════════════════════════════════════════════════════════
_SEARCH_AGG = AGG_MIXTURE      # evidence rule used for SELECTION
_TARGET_AGG = AGG_MIXTURE      # evidence rule used for TRAINING TARGETS
_GAUSSIAN_SELECT = False       # exact Dirichlet draw vs cached-moment Gaussian
_VIRTUAL_LOSS = 1.0            # penalty on v for an in-flight edge
_MLE_ACTIVE = False            # maintain the 'additive_mle' log-sums in _acc_add
                               # (skipped entirely when no rule needs them, so
                               # the default backup pays nothing for this)


def set_search(search_agg=None, target_agg=None, selection=None,
               virtual_loss=None):
    """Configure the tree for this PROCESS.  Self-play workers call it themselves
    from their cfg so a spawned process matches the parent exactly."""
    global _SEARCH_AGG, _TARGET_AGG, _GAUSSIAN_SELECT, _VIRTUAL_LOSS, _MLE_ACTIVE
    if search_agg is not None:
        if search_agg not in AGGREGATIONS:
            raise ValueError(f'search_agg must be one of {AGGREGATIONS}')
        _SEARCH_AGG = search_agg
    if target_agg is not None:
        if target_agg not in AGGREGATIONS:
            raise ValueError(f'target_agg must be one of {AGGREGATIONS}')
        _TARGET_AGG = target_agg
    if selection is not None:
        if selection not in ('dirichlet', 'gaussian'):
            raise ValueError("selection must be 'dirichlet' or 'gaussian'")
        _GAUSSIAN_SELECT = (selection == 'gaussian')
    if virtual_loss is not None:
        _VIRTUAL_LOSS = float(virtual_loss)
    _MLE_ACTIVE = AGG_ADDITIVE_MLE in (_SEARCH_AGG, _TARGET_AGG)


# ══════════════════════════════════════════════════════════════════════════════
#  The tree
# ══════════════════════════════════════════════════════════════════════════════

class _CNode:
    """One expanded state.

        v_alpha   (3,)    the network's STATE belief for this node (the payload
                          backed up when this node is the freshly expanded leaf)
        alpha_p   (k,3)   the network's per-action PRIOR beliefs Q(s,a)
        alpha_sel (k,3)   what selection actually samples:
                              α_p + observed(edge evidence)     — normal edge
                              terminal spike                    — proven edge
                          maintained INCREMENTALLY: a backup touches exactly one
                          edge, so it rewrites exactly one row (O(1), not O(k)).
        eacc      [k]     per-edge evidence accumulators (see observed_alpha)
        nacc              this node's own evidence accumulator (state target)
        term      (k,)    proven outcome (_WIN/_DRAW/_LOSS) or −1
        vloss     (k,)    in-flight selections, for within-wave diversification
    """
    __slots__ = ('player', 'legal', 'alpha_p', 'v_alpha', 'alpha_sel', 'term',
                 'vloss', 'children', 'obs', 'eacc', 'nacc', 'ev', 'sd')

    def __init__(self, player, legal, v3, vconf, p3, conf, obs=None):
        self.player = player
        self.legal = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        v = np.asarray(v3, dtype=np.float64).reshape(3)
        self.v_alpha = np.maximum(float(vconf) * v, ALPHA_FLOOR)
        p = np.asarray(p3, dtype=np.float64).reshape(k, 3)
        c = np.asarray(conf, dtype=np.float64).reshape(k)
        self.alpha_p = np.maximum(c[:, None] * p, ALPHA_FLOOR)
        self.alpha_sel = self.alpha_p.copy()          # no evidence yet
        self.term = np.full(k, -1, dtype=np.int8)
        self.vloss = np.zeros(k, dtype=np.int32)
        self.children = [None] * k
        self.obs = obs
        self.eacc = [_new_acc() for _ in range(k)]
        self.nacc = _new_acc()
        if _GAUSSIAN_SELECT:
            self.ev, self.sd = dir_value_mean_var(self.alpha_sel)
            self.sd = np.sqrt(self.sd)
        else:
            self.ev = self.sd = None

    # ── edge belief maintenance ───────────────────────────────────────────────
    def refresh_edge(self, idx):
        """Rewrite one edge's selection belief after its evidence changed."""
        if self.term[idx] >= 0:
            self.alpha_sel[idx] = _SPIKE[self.term[idx]]
        else:
            obs = observed_alpha(self.eacc[idx], _SEARCH_AGG)
            self.alpha_sel[idx] = (self.alpha_p[idx] if obs is None
                                   else self.alpha_p[idx] + obs)
        if _GAUSSIAN_SELECT and self.ev is not None:
            aw, ad, al = self.alpha_sel[idx].tolist()
            a0 = aw + ad + al
            mw, ml = aw / a0, al / a0
            self.ev[idx] = mw - ml
            self.sd[idx] = ((mw * (1 - mw) + ml * (1 - ml) + 2 * mw * ml)
                            / (a0 + 1.0)) ** 0.5

    def edge_target(self, idx):
        """The TRAINING target belief for one edge (proof if proven, else the
        observed evidence under `target_agg`).  None if never searched."""
        if self.term[idx] >= 0:
            return _SPIKE[self.term[idx]].copy()
        return observed_alpha(self.eacc[idx], _TARGET_AGG)

    def state_target(self):
        """The TRAINING target belief for this node's own state value."""
        solved = _node_solved_outcome(self)
        if solved is not None:
            return _SPIKE[solved].copy()
        obs = observed_alpha(self.nacc, _TARGET_AGG)
        return self.v_alpha.copy() if obs is None else obs

    def visits(self):
        """Per-edge evidence counts (the analogue of MCTS visit counts)."""
        return np.array([a[0] for a in self.eacc])


def _set_term(node, idx, outcome):
    """Mark edge idx proven and collapse its selection belief onto the proof."""
    node.term[idx] = outcome
    node.refresh_edge(idx)


def _dir_v_from_gammas(g, alpha, rng):
    """v = p_win − p_loss from per-row gamma draws `g` of Dir(alpha), handling
    degenerate rows exactly.

    With concentration unbounded BELOW the net can learn α₀ → 0, and there every
    gamma draw underflows to exactly 0, so the normaliser is 0 and v would be
    0/0 = NaN.  The exact limit of Dir(α) as α₀ → 0 is a point mass on corner i
    with probability α_i/α₀, so those rows draw a corner directly — no clamp, no
    bias, just the right distribution in the limit."""
    s = g.sum(1)
    bad = ~(s > 0.0)
    if bad.any():
        rows = np.nonzero(bad)[0]
        a = np.asarray(alpha, dtype=np.float64)[rows]
        a0 = a.sum(1, keepdims=True)
        p = np.where(a0 > 0.0, a / np.maximum(a0, 1e-300), 1.0 / a.shape[1])
        u = rng.random((len(rows), 1))
        corner = (p.cumsum(1) < u).sum(1).clip(0, a.shape[1] - 1)
        g = g.copy()
        g[rows] = 0.0
        g[rows, corner] = 1.0
        s = g.sum(1)
    return (g[:, _WIN] - g[:, _LOSS]) / s


def _sample_edge_values(node, rng, temp=1.0):
    """One Thompson draw of v = p_win − p_loss per legal edge from
    Dir(temp · α_sel) (temp > 1 sharpens toward the mean → argmax ≈ max), minus a
    virtual-loss penalty on in-flight edges.

    Exact Dirichlet sampling via 3 gamma draws per edge.  In 'gaussian' selection
    mode, instead draw v ∼ N(E[v], Var[v]) from the belief's cached closed-form
    moments — much cheaper, matches mean and variance exactly, drops skewness;
    harmless for an argmax, but A/B it before trusting it for strength."""
    if _GAUSSIAN_SELECT and node.ev is not None:
        sd = node.sd if temp == 1.0 else node.sd * (temp ** -0.5)
        v = node.ev + sd * rng.standard_normal(len(node.legal))
    else:
        a = node.alpha_sel if temp == 1.0 else node.alpha_sel * temp
        g = rng.standard_gamma(a)
        v = _dir_v_from_gammas(g, a, rng)
    if node.vloss.any():
        v = v - _VIRTUAL_LOSS * node.vloss
    return v


def _select_leaf(root, root_state, rng, temp=1.0):
    """Thompson-descend to an unexpanded or terminal edge, applying virtual loss.

    Returns (path, leaf_state_or_None, payloads_or_None, edge_or_None):
      terminal/proven edge → (path, None, payloads, None) — ready to back up now
      unexpanded edge      → (path, state, None, (node, idx)) — needs an NN eval
    `payloads` is always (as_is, flipped) in the DEEPEST EDGE's perspective at
    index 1 (see `_backup`)."""
    node, state, path = root, root_state.clone(), []
    while True:
        idx = int(_sample_edge_values(node, rng, temp).argmax())
        node.vloss[idx] += 1
        path.append((node, idx))
        if node.term[idx] >= 0:
            return path, None, _payload(flip_alpha(_SPIKE[node.term[idx]])), None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            r = state.returns()[node.player]                   # +1 / 0 / −1
            out = _WIN if r > 0 else (_LOSS if r < 0 else _DRAW)
            _set_term(node, idx, out)
            return path, None, _payload(flip_alpha(_SPIKE[out])), None
        child = node.children[idx]
        if child is None:
            return path, state, None, (node, idx)
        node = child


def _backup(path, payloads):
    """Fold one backed-up Dirichlet into every node and edge along the path.

    `payloads` = (as_is, flipped) from `_payload`.  Perspective alternates by
    ply, and — this is the useful symmetry — node n_i and the edge entering it
    from n_{i−1} share the SAME perspective, so one payload serves both.  The
    deepest edge on the path uses payloads[1] and it alternates upward from
    there.  O(depth) scalar work per simulation; the only array write is one
    refreshed row of `alpha_sel` per level."""
    sel = 1
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        pl = payloads[sel]
        _acc_add(node.eacc[idx], pl)
        _acc_add(node.nacc, pl)
        node.refresh_edge(idx)
        sel ^= 1


def _seed_leaf(leaf):
    """A freshly expanded node's first piece of evidence about itself is its own
    network belief, in its own perspective."""
    _acc_add(leaf.nacc, _payload(leaf.v_alpha)[0])


def _node_solved_outcome(node):
    """Proven outcome for node.player if this node is solved, else None.  WIN if
    any edge is a proven win; else (all edges proven) DRAW if any proven draw,
    else LOSS — the mover forces at least a draw over a loss."""
    t = node.term
    if (t == _WIN).any():
        return _WIN
    if (t >= 0).all():
        return _DRAW if (t == _DRAW).any() else _LOSS
    return None


def _propagate_solved(path, aux=None):
    """Walk leaf→root; when a node becomes fully solved, prove the parent edge
    entering it (flipped).  Emits an EXACT solver-labelled training sample per
    newly solved node.

    Each NODE is emitted at most once: proving the parent edge makes
    `_select_leaf` stop there instead of descending, so a solved node never
    appears on a later path (verified: 0 repeats across 11.5k emissions).  That
    is NOT per-POSITION de-duplication, and nothing else de-duplicates either.
    Connect 4 transposes heavily and every root search rebuilds the subtrees the
    previous move discarded, so the same board legitimately arrives as several
    distinct nodes.  Measured over 900 games (95.8k samples): 12% of solver rows
    are repeated boards, against 18% for ordinary self-play rows — openings
    recur far more than solved endgames do.  If that matters for your run, cap
    the per-game aux count or de-duplicate the replay buffer; the bigger lever
    is the solver SHARE of the buffer (~76%), not the repeats."""
    for k in range(len(path) - 1, 0, -1):
        node = path[k][0]
        out = _node_solved_outcome(node)
        if out is None:
            break
        parent, pidx = path[k - 1]
        if parent.term[pidx] >= 0:
            break
        _set_term(parent, pidx, int(_FLIP_TERM[out]))
        if aux is not None and node.obs is not None:
            t = make_target(node)
            t['solved'] = True
            t['z'] = np.float32(1.0 if out == _WIN else
                                (-1.0 if out == _LOSS else 0.0))
            aux.append(t)


def _backup_terminal(path, payloads, aux=None):
    _backup(path, payloads)
    _propagate_solved(path, aux)


def _descend(root, action):
    """Subtree reuse: the child under `action` becomes the next search's root.
    Returns None if that edge was never expanded."""
    if root is None:
        return None
    hit = np.nonzero(root.legal == action)[0]
    return root.children[int(hit[0])] if len(hit) else None


def root_pick(root, rng, thompson, temp=1.0):
    """Final move.  `thompson=True` → one more Thompson sample of the root's
    posterior beliefs (the self-play mover, as specified).  `thompson=False` →
    the posterior value-mean argmax, restricted to edges search actually touched
    so a confident-but-untested prior can't win the pick (evaluation/endgame)."""
    if thompson:
        v = _sample_edge_values(root, rng, temp)
    else:
        v = dir_value(root.alpha_sel)
        touched = (root.visits() > 0) | (root.term >= 0)
        if touched.any():
            v = np.where(touched, v, -np.inf)
    return int(root.legal[int(v.argmax())])


# ══════════════════════════════════════════════════════════════════════════════
#  Training targets
# ══════════════════════════════════════════════════════════════════════════════
# One sample per self-play move (plus solver-labelled aux samples):
#   obs       (D,) fp16   network input for this position (mover's perspective)
#   legal     (k,) int32  legal action ids
#   v_obs     (3,) fp32   observed STATE belief                    → L_klv
#   ev_idx    (m,) int32  legal-index of every SEARCHED edge
#   ev_alpha  (m,3) fp32  observed ACTION beliefs there            → L_kla
#   played    int         legal-index of the move actually played  → L_cea/L_cons
#   next_obs  (D,) fp16   observation AFTER that move, or None     → L_cons
#   next_term int         outcome if that move ENDED the game, in THIS
#                         position's mover's perspective, else −1
#   z         fp32        game outcome for this position's mover   → L_cev/L_cea
#   z_w       fp32        0 if the outcome was never observed
#   solved    bool        exact (solver-labelled) sample?

def make_target(root):
    """Searched root → training-target dict.  `played`/`next_obs`/`z` are filled
    in by the caller once the move is played and the game ends.

    Evidence edges are those search actually touched (any backed-up evidence, or
    a proof), so the action loss never grades the net against its own untouched
    prior.  Concentrations pass through as search found them — nothing is capped;
    `_floor_conc` only lifts a degenerate all-but-zero target to a usable total
    while PRESERVING its direction."""
    vis = root.visits()
    ev = (vis > 0) | (root.term >= 0)
    ev_idx = np.nonzero(ev)[0].astype(np.int32)
    rows = [root.edge_target(int(i)) for i in ev_idx]
    ev_alpha = (_floor_conc(np.asarray(rows, dtype=np.float64), 3.0 * TARGET_EPS)
                .astype(np.float32) if rows else np.zeros((0, 3), np.float32))
    v_obs = _floor_conc(root.state_target(), 3.0 * TARGET_EPS).astype(np.float32)
    return {'obs': root.obs, 'legal': root.legal.copy(), 'v_obs': v_obs,
            'ev_idx': ev_idx, 'ev_alpha': ev_alpha, 'played': -1,
            'next_obs': None, 'next_term': -1, 'z': np.float32(0.0),
            'z_w': np.float32(1.0), 'solved': False, 'player': int(root.player)}


def _floor_conc(alpha, min_total):
    """Raise each (…,3) Dirichlet's TOTAL to at least `min_total` by SCALING it,
    which preserves its mean — the direction is the information.

    An elementwise np.maximum(alpha, eps) looks equivalent but is not: it drags a
    low-concentration belief toward UNIFORM and so deletes exactly what the target
    encodes.  Moment matching legitimately returns a near-zero concentration
    whenever wildly disagreeing beliefs are mixed (the mixture is more dispersed
    than any single Dirichlet, so β₀ goes negative and floors), and an elementwise
    floor would turn those into an exact coin flip.  A genuinely all-zero row has
    no direction, so uniform is right only there."""
    a = np.asarray(alpha, dtype=np.float64)
    n = a.shape[-1]
    tot = a.sum(-1, keepdims=True)
    scaled = a * np.maximum(1.0, min_total / np.maximum(tot, 1e-300))
    return np.where(tot > 0.0, scaled, min_total / n)


def finish_episode(samples, returns):
    """Stamp the observed game outcome onto every sample of a finished game.
    `z` is the result from THAT position's mover's perspective — the target of
    both cross-entropy terms.  Solver-labelled samples already carry an exact
    proven z and are left alone."""
    for s in samples:
        if s['solved']:
            continue
        s['z'] = np.float32(returns[s['player']])
        s['z_w'] = np.float32(1.0)
    return samples


def mark_unfinished(samples):
    """A ply-capped game never observed its outcome: keep the KL targets, drop
    the cross-entropy ones.  (Connect 4 always terminates within 42 plies, so
    this only fires if a cap is set deliberately.)"""
    for s in samples:
        s['z_w'] = np.float32(0.0)
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Game wiring — observation tensor with a MOVER-RELATIVE perspective
# ══════════════════════════════════════════════════════════════════════════════
# Connect 4's default observation tensor is NOT perspective-relative: open_spiel
# builds it from absolute cell ownership and ignores the `player` argument
# entirely (connect_four.cc, ObservationTensor / StateToPlayer).  Feeding it
# straight to the net — the way the chess code legitimately can, because chess's
# tensor IS mover-relative — would train a network that cannot tell whose move it
# is, and every belief here is defined from the mover's perspective.
#
# So the perspective is applied here, by swapping the two ownership planes when
# player 1 is to move, giving a SELF-FIRST layout: plane 0 = mover's discs,
# plane 1 = opponent's, plane 2 = empty.  Doing it in Python rather than via the
# game's `egocentric_obs_tensor` parameter keeps this working on any open_spiel
# build, and `set_game` auto-detects a genuinely mover-relative tensor (so
# loading the egocentric variant on purpose does not double-flip).
_OBS_SHAPE = None
_NUM_ACTIONS = None
_OBS_DIM = None
_OBS_NEEDS_SWAP = True


def set_game(game):
    """Record the observation shape / action count and detect whether the game's
    own tensor is already mover-relative.  Call once per process before building
    a network or a tree."""
    global _OBS_SHAPE, _NUM_ACTIONS, _OBS_DIM, _OBS_NEEDS_SWAP
    _OBS_SHAPE = tuple(game.observation_tensor_shape())
    _NUM_ACTIONS = game.num_distinct_actions()
    _OBS_DIM = int(np.prod(_OBS_SHAPE))
    st = game.new_initial_state()
    st.apply_action(st.legal_actions()[0])
    same = np.array_equal(np.asarray(st.observation_tensor(0)),
                          np.asarray(st.observation_tensor(1)))
    _OBS_NEEDS_SWAP = bool(same)          # identical for both players → absolute
    return _OBS_SHAPE, _NUM_ACTIONS


def make_obs(state):
    """(D,) fp16 observation from the CURRENT MOVER's perspective."""
    if not _OBS_NEEDS_SWAP:
        return np.asarray(state.observation_tensor(state.current_player()),
                          dtype=np.float16)
    t = np.asarray(state.observation_tensor(0),
                   dtype=np.float16).reshape(_OBS_SHAPE)
    if state.current_player() == 1:
        t = t[[1, 0, 2]]
    return np.ascontiguousarray(t).reshape(-1)


def load_game(name='connect_four'):
    import pyspiel
    return pyspiel.load_game(name)


# ══════════════════════════════════════════════════════════════════════════════
#  Progress bar (self-erasing) + duration formatting
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_hms(s):
    """Compact duration: 42s / 7m12s / 1h03m."""
    if not (s == s) or s < 0 or s == float('inf'):
        return '--'
    s = int(s)
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s // 60}m{s % 60:02d}s'
    return f'{s // 3600}h{(s % 3600) // 60:02d}m'


class Progress:
    """One-line \\r progress bar that ERASES itself on close(), so what survives
    in the notebook output is only the eval lines — no scrollback spam.  Throttled
    by wall clock so a fast inner loop can call update() every episode."""

    def __init__(self, total, width=22, min_interval=0.4, enabled=True, label=''):
        self.total = max(int(total), 1)
        self.width = int(width)
        self.min_interval = float(min_interval)
        self.enabled = bool(enabled)
        self.label = label
        self.t0 = time.perf_counter()
        self._last = 0.0
        self._len = 0

    def reset(self, total=None, label=None):
        if total is not None:
            self.total = max(int(total), 1)
        if label is not None:
            self.label = label
        self.t0 = time.perf_counter()
        self._last = 0.0
        return self

    def update(self, done, extra=''):
        if not self.enabled:
            return
        now = time.perf_counter()
        if done < self.total and now - self._last < self.min_interval:
            return
        self._last = now
        frac = min(max(done / self.total, 0.0), 1.0)
        el = now - self.t0
        eta = el * (1.0 - frac) / frac if frac > 1e-9 else float('nan')
        fill = int(round(frac * self.width))
        bar = '█' * fill + '·' * (self.width - fill)
        msg = (f'\r  {self.label}[{bar}] {done}/{self.total}  '
               f'{_fmt_hms(el)} elapsed  ETA {_fmt_hms(eta)}'
               f'{("  " + extra) if extra else ""}')
        self._len = max(self._len, len(msg))
        try:
            sys.stdout.write(msg)
            sys.stdout.flush()
        except Exception:                    # never let display kill training
            self.enabled = False

    def close(self):
        if not self.enabled or not self._len:
            return
        try:
            sys.stdout.write('\r' + ' ' * self._len + '\r')
            sys.stdout.flush()
        except Exception:
            pass
        self._len = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration — the notebook's only surface
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """Every tunable, with the defaults this method was designed around.  The
    notebook constructs one of these and hands it to `run_training`."""

    # ── run / model ───────────────────────────────────────────────────────────
    num_episodes: int = 20_000        # budget; the LR schedule is decoupled below
    channels: int = 64                # trunk width
    num_blocks: int = 5               # residual blocks
    head_ch: int = 16                 # 1x1-conv width feeding both flat heads.
                                      # Connect 4 has 7 actions, so unlike chess
                                      # the action head is nearly free and this
                                      # is NOT the parameter-count lever.
    device_preference: str = 'auto'   # 'cpu' | 'cuda' | 'directml' | 'auto'
    checkpoint_dir: str = 'c4_dirichlet_ckpt'
    resume: bool = True
    seed: int = 0

    # ── evidence collapse (see observed_alpha) ────────────────────────────────
    # 'mixture' | 'mean' | 'sum' | 'additive' | 'additive_mle'
    search_agg: str = AGG_ADDITIVE
    target_agg: str = AGG_ADDITIVE
    # THE DEFAULT IS 'additive' for both.
    #
    # SEARCH wants concentration to GROW with evidence, so Thompson exploration
    # anneals.
    # 'additive' makes α₀ exactly the visit count — the plain
    # Dirichlet-categorical posterior, spread ∝ 1/√(n+1).  The alternatives both
    # fail here: 'mixture' reports how much the leaves DISAGREE, which does not
    # shrink with sampling (median 0.128 over real unsolved positions at 400
    # sims, i.e. pinned), while 'mean' and 'sum' assume the simulations are
    # independent — they are not, sharing a network and overlapping subtrees —
    # and hit ~140 after two visits, going greedy immediately.
    #
    # As a TARGET, 'additive' says "this belief rests on n observations", which
    # is a real and legible statement, and it removes the pathology that made
    # 'mixture' targets nearly inert: ~54% of unsolved state targets landed on
    # the _floor_conc floor, carrying a direction but no scale at all.  The
    # trade-off is budget dependence — α₀ IS the simulation count, so a
    # FAST_SIMS and a FULL_SIMS visit to the same position label it differently
    # and the network cannot see which it is getting.  Note the two rules give
    # the SAME target mean; they differ only in concentration.  See the second
    # A/B below, which isolates exactly that.
    #
    # 'additive_mle' is kept DISTINCT from 'additive' so the combinations can be
    # tested separately.  It fits the same observations by maximum likelihood
    # instead of counting them, which grounds the concentration in the actual
    # dispersion of the backed-up leaves rather than in the simulation budget.
    # Measured beforehand on real searched nodes at 400 sims: median α₀ 8.1 for
    # state nodes and 12.3 for action nodes, against 7.1 / 8.4 for 'mixture' and
    # 401 / 54 for 'additive'.  So it is a better-grounded estimate of what
    # 'mixture' approximates, NOT a way to keep 'additive' magnitudes — expect
    # it to need the KL weights raised (~2.5x) to preserve the distillation
    # pressure that won 'additive' its +200 Elo, since both rules produce the
    # same target MEAN and differ only in how hard the KL pulls on it.
    # Cost: 1.18 us per accumulator update against 0.69 for 'additive'.
    #
    # MEASURED, on target_agg (2000 episodes/arm, 70k params, one seed, search
    # fixed at 'mixture' in both).  Arm B is the documented pairing for 'mean'
    # targets — kl_normalize=True, plus w_klv raised 16x to restore the state
    # KL's share of the loss, which normalisation had cut from 17% to 1%:
    #     head to head, search-free (200 games)  mixture 69.5%  (+143 Elo)
    #     head to head, MCTS-64      (60 games)  mixture 60.8%  ( +76 Elo)
    #     vs a common anchor, search-free        mixture 53.3%, mean 28.3%
    #     vs random                              mixture 100%,  mean 91.2%
    #     outcome accuracy / NLL on held-out     mixture 67.4% / 0.677
    #       positions with known results         mean    62.2% / 0.872
    # 'mean' targets DO make the concentration channel live — predicted α₀ spans
    # 20x across positions (p10 0.84, p90 16.8) versus 2.6x (0.12 to 0.31) under
    # 'mixture'.  It just does not pay: ranking positions by predicted α₀, the
    # top-quartile minus bottom-quartile accuracy gap is +15 points under
    # 'mixture' against +12 (and non-monotonic) under 'mean'.  A compressed,
    # nearly-floored concentration still orders positions usefully, and the
    # per-observation normalisation that 'mean' requires costs more — it weakens
    # both KL terms, and the action KL is what Thompson selection reads.
    # Caveat: three flags differ between the arms, so this indicts the
    # CONFIGURATION, not 'mean' targets in isolation.
    #
    # STATUS: search_agg='additive' is a reasoned default, NOT yet an A/B'd one
    # — the arms above varied target_agg only.  NOTE THE COUPLING when testing.  Selection samples α_net + α_obs, and
    # under 'additive' α_obs grows by exactly 1 per visit — so the network's
    # concentration head now literally means "how many visits my opinion is
    # worth".  The head was trained under the old semantics and currently emits
    # ≈0.2 on ordinary positions and ≈35-44 on solved-looking ones, i.e. the
    # prior is ignored where guidance would help and swamps the first ~40
    # simulations where search would have settled it anyway.  Expect an A/B of
    # 'additive' search to be measuring that mis-scaling as much as the rule.
    # As a TARGET, α₀ == n makes the target budget-dependent (FAST=100 vs
    # FULL=400 label the same position 100 and 400), which is what sank the
    # 'mean' arm above.

    # ── search ────────────────────────────────────────────────────────────────
    selection: str = 'dirichlet'      # exact draw | 'gaussian' approximation
    virtual_loss: float = 1.0         # in-flight edge penalty on v ∈ [−1, 1]
    fast_sims: int = 100              # simulations for a `fast_prob` game
    full_sims: int = 400              # …and for the rest
    fast_prob: float = 0.75
    temp_threshold: int = 12          # plies of full-temperature Thompson moves
    late_temp: float = 8.0            # after that, concentration ×this (≈ greedy)

    # ── self-play ─────────────────────────────────────────────────────────────
    use_workers: bool = True          # N processes + one central batched server
    selfplay_workers: int = 0         # 0 → auto (cpu_count − 2, clamped)
    games_per_worker: int = 32        # THE lever on GPU batch size
    worker_wave: int = 8              # leaves per game per wave
    n_parallel_games: int = 16        # used only when use_workers=False
    wave_per_game: int = 8
    max_plies: int = 42               # Connect 4's own maximum
    pool_prob: float = 0.15           # frac. of games with one side a frozen
                                      # benchmark or a uniform-random mover
    random_pool_frac: float = 0.5     # …of those, frac. played vs random
    # OFF by default — both exist to make LONG games tractable and a 42-ply game
    # is already all endgame.  Kept because they cost nothing when disabled.
    curriculum: bool = False          # endgame-first: random opening backbone,
    curr_depth0: float = 8.0          # MCTS covers only the last `depth` plies
    curr_mcts_tail: int = 24
    curr_step: float = 4.0
    curr_val_thresh: float = 0.40     # push the frontier back once the STATE
                                      # cross-entropy drops below this, i.e. the
                                      # net predicts the outcome at the current
                                      # depth (ln 3 = 1.10 is chance)
    curr_max_depth: float = 42.0
    restart_prob: float = 0.0         # backward-restart curriculum
    restart_k_min: int = 2
    restart_k_max: int = 12
    restart_pool_cap: int = 128

    # ── training ──────────────────────────────────────────────────────────────
    batch_size: int = 512
    train_steps_per_ep: int = 8
    max_buffer: int = 150_000
    lr_peak: float = 2e-3
    lr_warmup_eps: int = 100
    lr_decay_eps: int = 8_000         # cosine horizon (decoupled from num_episodes)
    lr_min_factor: float = 0.10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # (klv, kla, cev, cea, cons).  The ACTION head is what Thompson selection
    # reads, so it carries the most weight; the consistency term is a regulariser
    # and is deliberately light.  Watch the printed `sh` shares: if one term sits
    # near 100% the others have effectively stopped training.
    loss_weights: tuple = (1.0, 2.0, 1.0, 1.0, 0.5)
    cons_frac: float = 1.0            # fraction of samples whose SUCCESSOR is
                                      # forwarded for the consistency term.  Each
                                      # one adds a row to a second forward block,
                                      # so 1.0 nearly doubles the training batch
                                      # (~90% of samples have a successor).  The
                                      # term stays unbiased when sub-sampled — it
                                      # is a weight-normalised mean — so 0.25
                                      # buys ~1.5x faster steps for a little
                                      # extra gradient variance on the lightest
                                      # of the five loss terms.
    kl_normalize: bool = False        # divide each Dirichlet KL by its target's
                                      # total concentration (a PER-OBSERVATION
                                      # KL).  Leave off for target_agg='mixture'
                                      # (bounded concentration); turn ON for
                                      # 'mean'/'sum', whose target concentration
                                      # scales with the simulation budget and
                                      # would otherwise swamp every other term.
                                      # It rescales the KLs by ~1/α₀ (measured:
                                      # their share of the loss drops from ~60%
                                      # to ~12%), so raise the two KL weights
                                      # when turning it on.

    # ── evaluation (two tiers) ────────────────────────────────────────────────
    quick_eval_every: int = 500       # cheap SEARCH-FREE pulse vs last checkpoint
    quick_eval_games: int = 40
    deep_eval_every: int = 2_500      # new checkpoint + MCTS running-Elo pool
    eval_sims: int = 128
    eval_games_per_pair: int = 4
    eval_last_n: int = 3
    eval_refresh_pairs: int = 10
    eval_opening_plies: int = 2
    eval_max_plies: int = 42
    eval_k_base: float = 32.0
    eval_k_halflife: float = 30.0
    eval_temp: float = 6.0
    start_elo: float = 1000.0
    eval_device: str = 'cpu'
    # ── Absolute strength against exactly-solved positions ────────────────────
    # Every other eval here is RELATIVE (Elo against this run's own earlier
    # checkpoints), which cannot tell an improving run from a collapsing one
    # whose opponents collapsed with it.  Point `solved_dir` at a directory of
    # Pascal Pons' test files and each deep eval reports the value head's MSE
    # against the exact outcome, PER CATEGORY (win/draw/loss) so a net that
    # answers "win" to everything cannot hide behind a win-skewed test set.
    # One batched forward over precomputed observations.  Off when unset.
    solved_dir: str = ''
    solved_every: int = 0        # 0 → every deep eval
    solved_n: int = 0            # positions per bucket; 0 → all of them

    def resolved_workers(self):
        if self.selfplay_workers > 0:
            return self.selfplay_workers
        return min(8, max(2, (os.cpu_count() or 8) - 2))


# ══════════════════════════════════════════════════════════════════════════════
#  Multiprocess self-play worker (top level so `spawn` can import it)
# ══════════════════════════════════════════════════════════════════════════════
# N CPU worker processes run the trees; a central inference-server THREAD in the
# parent batches all of their NN requests into one forward pass (see MPSelfPlayPool).
# The tree work is CPU-bound and serial within a process, so parallelising it
# across cores while the GPU serves big fused batches is the dominant speedup.
#
# Wire format (crosses pickling queues, so kept tiny):
#   request : (worker_id, net_id, obs (n, D) fp16, [legals int32, …])
#   response: [(v3 (3,) f32, vconf float, p3 (k,3) f32, conf (k,) f32), …]
#             — only the gathered legal entries.

def _mp_load_game(cfg):
    import pyspiel
    return pyspiel.load_game(cfg.get('game_name', 'connect_four'))


def _random_backbone(game, rng, depth, max_plies):
    """Endgame-first curriculum prefix: play a uniform-random game to its end
    (capped), then resume `depth` plies before that end, so MCTS self-play only
    covers the tail.  Returns (state, replayed_actions, resume_ply)."""
    st = game.new_initial_state()
    seq = []
    while not st.is_terminal() and len(seq) < max_plies:
        legal = st.legal_actions()
        a = int(legal[rng.integers(len(legal))])
        st.apply_action(a); seq.append(a)
    resume = max(0, len(seq) - int(round(depth)))
    state = game.new_initial_state()
    for a in seq[:resume]:
        state.apply_action(a)
    return state, seq[:resume], resume


def _restart_prefix(seq, rng, k_min, k_max):
    """Backward-restart curriculum: drop a random tail of a past game."""
    if len(seq) <= k_min:
        return []
    k = rng.integers(k_min, min(k_max, len(seq)) + 1)
    return list(seq[:len(seq) - k])


class _SlotEngine:
    """The self-play state machine shared by the single-process and multiprocess
    drivers: `n` concurrent games, each an independent tree, advanced in waves so
    every NN evaluation is batched across ALL of them.  Subclasses supply only
    how a batch of positions gets evaluated and how a finished game is reported.
    """

    def __init__(self, game, cfg, rng, checkpoint_dir=None):
        self.game, self.cfg, self.rng = game, cfg, rng
        self.checkpoint_dir = checkpoint_dir
        self.curr_depth = float(cfg['curr_depth0'])
        self._restart_pool = []

    def _push_seed(self, seq):
        if len(seq) >= 2:
            self._restart_pool.append(list(seq))
            if len(self._restart_pool) > self.cfg['restart_pool_cap']:
                del self._restart_pool[0]

    def _pool_labels(self):
        try:
            return [f[6:-3] for f in os.listdir(self.checkpoint_dir)
                    if f.startswith('bench_') and f.endswith('.pt')] \
                if self.checkpoint_dir else []
        except OSError:
            return []

    def new_game(self):
        cfg, rng = self.cfg, self.rng
        sims = cfg['fast_sims'] if rng.random() < cfg['fast_prob'] else cfg['full_sims']
        state, actions, resume = self.game.new_initial_state(), [], 0
        if cfg['curriculum']:
            state, actions, resume = _random_backbone(
                self.game, rng, self.curr_depth, cfg['max_plies'])
        elif self._restart_pool and rng.random() < cfg['restart_prob']:
            seq = self._restart_pool[rng.integers(len(self._restart_pool))]
            pref = _restart_prefix(seq, rng, cfg['restart_k_min'], cfg['restart_k_max'])
            st = self.game.new_initial_state(); ok = True
            for a in pref:
                if st.is_terminal() or a not in st.legal_actions():
                    ok = False; break
                st.apply_action(int(a))
            if ok and pref and not st.is_terminal():
                state, actions, resume = st, list(pref), len(pref)
        cap = (min(cfg['max_plies'], resume + cfg['curr_mcts_tail'])
               if cfg['curriculum'] else cfg['max_plies'])
        slot = {'state': state, 'hist': [], 'aux': [], 'actions': actions,
                'move': resume, 'resume': resume, 'sims': sims, 'cap': cap,
                'root': None, 'n': 0, 'pool': None}
        if cfg['pool_prob'] > 0 and rng.random() < cfg['pool_prob']:
            labels = self._pool_labels()
            label = ('random' if not labels or rng.random() < cfg['random_pool_frac']
                     else labels[rng.integers(len(labels))])
            slot['pool'] = {'label': label, 'side': int(rng.integers(2))}
        return slot

    def _temp(self, move):
        return 1.0 if move < self.cfg['temp_threshold'] else self.cfg['late_temp']

    def _record_move(self, s, root):
        """Append this position's target, Thompson-pick the move, apply it, and
        wire up the successor observation the consistency loss needs."""
        t = make_target(root)
        mm = s['move'] - s['resume']
        a = root_pick(root, self.rng, thompson=(mm < self.cfg['temp_threshold']),
                      temp=self._temp(mm))
        pidx = int(np.nonzero(root.legal == a)[0][0])
        t['played'] = pidx
        s['hist'].append(t)
        s['actions'].append(int(a))
        s['root'] = root.children[pidx]
        s['state'].apply_action(a)
        s['move'] += 1; s['n'] = 0
        st = s['state']
        if st.is_terminal():
            # The consistency target is the value of HAVING PLAYED a*, i.e. in
            # THIS node's mover's perspective — the same perspective Q(s,a*) is
            # in, and the same one the network branch reaches by flipping
            # V(s') one ply.  A terminal successor has no mover to flip from,
            # so the outcome is recorded unflipped.
            r = st.returns()[root.player]
            t['next_term'] = int(_WIN if r > 0 else (_LOSS if r < 0 else _DRAW))
        else:
            t['next_obs'] = make_obs(st)

    def _apply_pool_move(self, s, a):
        s['root'] = _descend(s['root'], a)
        s['state'].apply_action(a)
        s['actions'].append(int(a))
        s['move'] += 1


def mp_worker(worker_id, req_q, resp_q, pool_resp_q, episode_q, cfg):
    """One self-play worker process: runs `games_per_worker` trees in two
    pipelined halves so that while one half's NN request is in flight the other
    half is doing tree work."""
    game = _mp_load_game(cfg)
    set_game(game)
    set_search(cfg['search_agg'], cfg['target_agg'], cfg['selection'],
               cfg['virtual_loss'])
    rng = np.random.default_rng(cfg['seed'] + worker_id * 7919)
    eng = _SlotEngine(game, cfg, rng, cfg.get('checkpoint_dir'))
    curr_shared = cfg.get('curr_depth_shared')

    def _curr():
        return curr_shared.value if curr_shared is not None else cfg['curr_depth0']

    def finish_and_reset(i):
        s = slots[i]; st = s['state']
        if st.is_terminal():
            ret = st.returns()
            finish_episode(s['hist'], ret)
            result = 'draw' if ret[0] == 0.0 else 'decisive'
            if cfg['restart_prob'] > 0 and result == 'decisive':
                eng._push_seed(s['actions'])
        else:
            mark_unfinished(s['hist']); result = 'cutoff'
        episode_q.put((s['hist'] + s['aux'], len(s['aux']), result, int(s['move'])))
        eng.curr_depth = _curr()
        slots[i] = eng.new_game()

    slots = [eng.new_game() for _ in range(cfg['games_per_worker'])]
    mid = max(1, cfg['games_per_worker'] // 2)
    halves = [list(range(mid)), list(range(mid, cfg['games_per_worker']))]

    def collect(idxs):
        evals, pending, obs, legals = [], [], [], []
        seen = set()
        for i in idxs:
            s = slots[i]; st0 = s['state']; pool = s['pool']
            if pool is not None and st0.current_player() == pool['side']:
                continue
            if s['root'] is None:
                leg = st0.legal_actions()
                o = make_obs(st0)
                evals.append(('root', i, None, st0.current_player(), leg, o))
                obs.append(o); legals.append(np.asarray(leg, dtype=np.int32))
                continue
            if _node_solved_outcome(s['root']) is not None:
                continue
            wave = min(cfg['wave'], s['sims'] - s['n'])
            for _ in range(max(wave, 0)):
                path, st, payloads, edge = _select_leaf(
                    s['root'], st0, rng, eng._temp(s['move'] - s['resume']))
                if st is None:
                    _backup_terminal(path, payloads, s['aux'])
                    s['n'] += 1; continue
                node, idx = edge
                pending.append((i, path, node, idx))
                if (id(node), idx) not in seen:
                    seen.add((id(node), idx))
                    leg = st.legal_actions()
                    o = make_obs(st)
                    evals.append(('leaf', node, idx, st.current_player(), leg, o))
                    obs.append(o); legals.append(np.asarray(leg, dtype=np.int32))
        return evals, pending, obs, legals

    def apply_and_advance(idxs, evals, pending, resp):
        if evals:
            for e, (v3, vc, p3, cf) in zip(evals, resp):
                kind, a, b, player, leg, o = e
                nd = _CNode(player, leg, v3, vc, p3, cf, obs=o)
                _seed_leaf(nd)
                if kind == 'root':
                    slots[a]['root'] = nd
                else:
                    a.children[b] = nd
        for i, path, node, idx in pending:
            child = node.children[idx]
            _backup(path, _payload(child.v_alpha))
            slots[i]['n'] += 1
        for i in idxs:
            s = slots[i]
            if s['root'] is None:
                continue
            if s['n'] < s['sims'] and _node_solved_outcome(s['root']) is None:
                continue
            eng._record_move(s, s['root'])
            if s['state'].is_terminal() or s['move'] >= s['cap']:
                finish_and_reset(i)

    def resolve_pool_moves(idxs):
        for i in idxs:
            s = slots[i]; pool = s['pool']
            if pool is None:
                continue
            state = s['state']
            if state.current_player() != pool['side']:
                continue
            legal = state.legal_actions()
            if pool['label'] == 'random':
                a = int(legal[rng.integers(len(legal))])
            else:
                req_q.put((worker_id, pool['label'], make_obs(state)[None],
                           [np.asarray(legal, dtype=np.int32)]))
                (v3, vc, p3, cf), = pool_resp_q.get()
                alpha = np.maximum(np.asarray(cf)[:, None] * np.asarray(p3),
                                   ALPHA_FLOOR)
                a = int(legal[int(dir_value(alpha).argmax())])
            eng._apply_pool_move(s, a)
            if state.is_terminal() or s['move'] >= s['cap']:
                finish_and_reset(i)

    inflight = [None, None]
    while True:
        for h in (0, 1):
            if inflight[h] is not None:
                evals, pending, sent = inflight[h]
                resp = resp_q.get() if sent else None
                apply_and_advance(halves[h], evals, pending, resp)
                inflight[h] = None
            resolve_pool_moves(halves[h])
            evals, pending, obs, legals = collect(halves[h])
            sent = False
            if evals:
                req_q.put((worker_id, 'live', np.stack(obs), legals)); sent = True
            inflight[h] = (evals, pending, sent)


# ══════════════════════════════════════════════════════════════════════════════
#  Everything below needs torch (network, losses, bots, eval, checkpoints, driver)
# ══════════════════════════════════════════════════════════════════════════════
if _HAS_TORCH:

    # ── Device selection ──────────────────────────────────────────────────────
    def pick_device(pref='auto'):
        if pref in ('directml', 'auto'):
            try:
                import torch_directml
                try:
                    name = torch_directml.device_name(0)
                except Exception:
                    name = 'DirectML GPU'
                print(f'Using DirectML: {name}')
                return torch_directml.device(), 'directml'
            except Exception:
                if pref == 'directml':
                    print('DirectML requested but unavailable — falling back.')
        if pref in ('cuda', 'auto') and torch.cuda.is_available():
            return torch.device('cuda'), 'cuda'
        return torch.device('cpu'), 'cpu'

    def batch_to_tensor(obs_list, device):
        obs = np.asarray(obs_list, dtype=np.float32)
        return torch.from_numpy(obs.reshape(-1, *_OBS_SHAPE)).to(device)

    # ── DirectML-safe primitives (fused kernels lack DML backward) ────────────
    # Which fused kernels this device can actually run.  Every entry here has a
    # hand-rolled fallback below, written because DirectML lacked the kernel (or
    # its backward).  The fallbacks are correct but expand ONE kernel into eight
    # to thirty, and this workload is launch-bound rather than FLOP-bound — a
    # 64ch/5blk trunk on a 6x7 board runs at roughly 10% of an RX 5700 XT's fp32
    # peak — so which path is taken matters more than the arithmetic in it.
    # Probed once against the real device, forward AND backward, since DirectML
    # often implements one and not the other.
    _NATIVE = {'lgamma': False, 'softplus': False, 'group_norm': False}

    class _GroupNorm(nn.Module):
        def __init__(self, num_groups, num_channels, eps=1e-5):
            super().__init__()
            self.num_groups, self.eps = num_groups, eps
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        def forward(self, x):
            if _NATIVE['group_norm']:
                return F.group_norm(x, self.num_groups, self.weight, self.bias,
                                    self.eps)
            n, c = x.shape[0], x.shape[1]
            xg = x.reshape(n, self.num_groups, -1)
            mean = xg.mean(dim=2, keepdim=True)
            var = (xg - mean).pow(2).mean(dim=2, keepdim=True)
            xg = (xg - mean) / torch.sqrt(var + self.eps)
            return (xg.reshape(x.shape) * self.weight.view(1, c, 1, 1)
                    + self.bias.view(1, c, 1, 1))

    def _norm(channels):
        g = min(8, channels)
        while channels % g != 0:
            g -= 1
        return _GroupNorm(g, channels)

    def _softplus(x):
        if _NATIVE['softplus']:
            return F.softplus(x)
        return torch.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            mid = max(channels // reduction, 4)
            self.fc = nn.Sequential(nn.Linear(channels, mid),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(mid, channels * 2))

        def forward(self, x):
            s = x.mean(dim=(2, 3))
            scale, bias = self.fc(s).chunk(2, dim=1)
            return (x * torch.sigmoid(scale)[:, :, None, None]
                    + bias[:, :, None, None])

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                _norm(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                _norm(channels))
            self.se = SEBlock(channels)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.se(self.net(x)) + x)

    class C4DirichletNet(nn.Module):
        """Trunk → two Dirichlet heads.

        forward(x) → (v_logits (B,3), v_conf_raw (B,), a_logits (B,A,3),
                      a_conf_raw (B,A)).

        Softmax / softplus are DEFERRED so the action head's activation can be
        applied to the gathered LEGAL entries only.  The state belief is
        Dir(softplus(v_conf_raw) · softmax(v_logits)); the action belief for a is
        Dir(softplus(a_conf_raw[a]) · softmax(a_logits[a]))."""

        def __init__(self, channels=64, num_blocks=5, head_ch=16):
            super().__init__()
            self._sig = (channels, num_blocks, head_ch)
            in_ch = _OBS_SHAPE[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                _norm(channels), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[ResBlock(channels)
                                        for _ in range(num_blocks)])
            self.head = nn.Sequential(
                nn.Conv2d(channels, head_ch, 1, bias=False),
                _norm(head_ch), nn.ReLU(inplace=True), nn.Flatten())
            flat = head_ch * _OBS_SHAPE[1] * _OBS_SHAPE[2]
            self.v_out = nn.Linear(flat, 4)                  # 3 logits + 1 conc
            self.a_out = nn.Linear(flat, _NUM_ACTIONS * 4)   # …per action
            # Untrained net: exactly uniform outcome beliefs with WEAK confidence
            # (α₀ = softplus(1.4) ≈ 1.62), so search dominates the prior from
            # generation 0 instead of chasing a random initialisation.
            nn.init.zeros_(self.v_out.weight); nn.init.zeros_(self.a_out.weight)
            with torch.no_grad():
                self.v_out.bias.zero_(); self.v_out.bias[3] = 1.4
                self.a_out.bias.view(_NUM_ACTIONS, 4).zero_()
                self.a_out.bias.view(_NUM_ACTIONS, 4)[:, 3] = 1.4

        def forward(self, x):
            h = self.head(self.body(self.stem(x)))
            v = self.v_out(h)
            a = self.a_out(h).view(-1, _NUM_ACTIONS, 4)
            return v[:, :3], v[:, 3], a[..., :3], a[..., 3]

    # Above this, softplus(x) and x differ by log1p(exp(-x)) — 2e-9 at x=20,
    # far below fp32 resolution at that magnitude — so the two are the same
    # number and only one of them can overflow.
    _CONF_LINEAR = 20.0

    def _conf(raw):
        """RAW concentration output → α₀ > 0.  UNBOUNDED above: softplus only,
        with a positivity floor so an fp32 underflow to 0 can't NaN the KL.

        The transcendental part is evaluated on a CLAMPED input and the linear
        branch spliced back in, so no backend can overflow here no matter how
        its softplus is implemented.  This is not hypothetical: a kernel that
        computes softplus as log1p(exp(x)) — rather than PyTorch's
        threshold-at-20 form — returns +inf for x > 88.7 in fp32, and the
        `additive` rule drives targets to exactly that range (α₀ = visit count,
        so α₀ ≈ 89 at full_sims).  One +inf reaches lgamma as inf−inf = NaN in
        the KL, and inf×0 = NaN in the masked action mean; the next backward
        then writes NaN into every weight.  Clamping the input rather than the
        OUTPUT also keeps the gradient clean: torch.where alone would still
        differentiate the dead branch and multiply inf by zero."""
        safe = _softplus(raw.clamp(max=_CONF_LINEAR))
        return torch.where(raw > _CONF_LINEAR, raw, safe).clamp_min(ALPHA_FLOOR)

    # ── lgamma / digamma with a DirectML-friendly fallback ────────────────────
    # The closed-form Dirichlet KL's only expensive pieces are lgamma (the log
    # normaliser) and digamma (E[log x]).  Neither has a DirectML kernel, which
    # used to force the whole loss onto the CPU.  These elementary-op
    # approximations (8-step upward recurrence + Stirling/asymptotic series) run
    # entirely on the GPU: float32 error ~2e-4, and autograd through _lgamma_dml
    # reproduces digamma to ~1e-9.  Used only on DirectML.
    _LG_SHIFT = 8

    def _lgamma_dml(x):
        g = torch.zeros_like(x); xx = x
        for _ in range(_LG_SHIFT):
            g = g - torch.log(xx); xx = xx + 1.0
        inv = 1.0 / xx; inv2 = inv * inv
        return g + ((xx - 0.5) * torch.log(xx) - xx + 0.5 * math.log(2 * math.pi)
                    + inv * (1.0 / 12 - inv2 * (1.0 / 360 - inv2 * (1.0 / 1260))))

    def _digamma_dml(x):
        g = torch.zeros_like(x); xx = x
        for _ in range(_LG_SHIFT):
            g = g - 1.0 / xx; xx = xx + 1.0
        inv = 1.0 / xx; inv2 = inv * inv
        return g + (torch.log(xx) - 0.5 * inv
                    - inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 / 252.0)))

    _LG_APPROX = False

    # Magnitudes the loss actually feeds these ops.  α ranges from ALPHA_FLOOR
    # (a masked-out or underflowed component) up past TERMINAL_CONC, and with
    # the `additive` rule α₀ is the visit count — so ~full_sims, not O(1).  The
    # probe used to sample [0.5, 1.5] only, which every backend gets right; the
    # kernels that break do so in the tails, which is precisely where they were
    # never asked.
    _PROBE_MAGS = (1e-9, 1e-3, 0.5, 1.0, 20.0, 89.0, 200.0)
    # DirectML emits a UserWarning naming the operator and saying it will run on
    # the CPU instead.  "Correct" and "actually ran on the GPU" are different
    # questions, and the probe has to ask both: a silent CPU fallback returns the
    # right numbers, so it passes a correctness check, while copying to host and
    # back on every call — a full pipeline sync inside the loss, three times per
    # training step, on a workload that is already launch-bound.
    _FALLBACK_HINTS = ('not currently supported', 'fall back', 'fallback',
                       'will run on the cpu')

    def _fallback_warning(caught):
        """A short reason from the first 'this op is not supported, running on
        the CPU' warning in `caught`, or ''."""
        for w in caught:
            text = str(w.message)
            if any(h in text.lower() for h in _FALLBACK_HINTS):
                op = re.search(r"'([^']+)'", text)   # e.g. 'aten::lgamma.out'
                return (f'{op.group(1)} runs on the CPU' if op
                        else ' '.join(text.split())[:70])
        return ''

    def _probe(fn, device, shape=(3,)):
        """Does `fn` run ON THIS DEVICE, forward AND backward, and agree with the
        CPU ACROSS THE RANGE THE LOSS USES?  Returns (ok, reason-if-not).

        Correct-on-easy-inputs is not the question — a wrong kernel here is not
        caught by any test, it silently poisons a multi-hour run — so this sweeps
        the real dynamic range and demands finite, CPU-matching output and
        gradients at every magnitude, and rejects an operator the backend only
        implements by shipping the tensor to the CPU."""
        try:
            for mag in _PROBE_MAGS:
                ref_in = (torch.rand(shape).abs() + 0.5) * mag
                x = ref_in.to(device).requires_grad_(True)
                # Suppress while probing: this call is a deliberate test, so its
                # warning is an answer rather than something to print.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    y = fn(x)
                    y.sum().backward()
                why = _fallback_warning(caught)
                if why:
                    return False, f'no device kernel ({why})'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ref_out = fn(ref_in)   # CPU reference; its noise is not news
                if not torch.isfinite(y.detach().to('cpu')).all():
                    return False, f'non-finite output at x~{mag:g}'
                if not torch.isfinite(ref_out).all():
                    continue        # CPU itself has no finite answer here
                if not torch.allclose(y.detach().to('cpu'), ref_out,
                                      rtol=1e-3, atol=1e-3):
                    return False, f'disagrees with the CPU at x~{mag:g}'
                if x.grad is None or not torch.isfinite(x.grad.to('cpu')).all():
                    return False, f'bad gradient at x~{mag:g}'
            return True, ''
        except Exception as e:
            return False, f'{type(e).__name__}: {e}'

    def set_backend(backend, device=None):
        """Choose fused kernels where the device supports them, hand-rolled
        fallbacks where it does not.

        This used to flip only lgamma/digamma, and only for DirectML — so the
        softplus and GroupNorm fallbacks ran unconditionally and CUDA/CPU users
        paid for them too.  Each fallback is correct but expands ONE kernel into
        eight to thirty, and this workload is launch-bound rather than
        FLOP-bound, so the choice matters more than the arithmetic.  Probed
        against the real device because DirectML often implements a forward
        without its backward, and because it implements some operators only by
        copying the tensor to the CPU -- which is worse here than the
        hand-rolled version, since a host round-trip inside the loss
        synchronises the whole pipeline."""
        global _LG_APPROX
        if device is None:
            _LG_APPROX = (backend == 'directml')
            _NATIVE.update(lgamma=not _LG_APPROX, softplus=True, group_norm=True)
            return
        why = {}

        def probe(name, fn, **kw):
            ok, reason = _probe(fn, device, **kw)
            if not ok:
                why[name] = reason
            return ok

        # lgamma and digamma share one flag: the KL needs both, and _lgamma_dml
        # supplies digamma through its own autograd, so they stand or fall
        # together.  `&` not `and` so both are probed and both get reported.
        _NATIVE['lgamma'] = (probe('lgamma', torch.lgamma)
                             & probe('digamma', torch.digamma))
        _NATIVE['softplus'] = probe('softplus', F.softplus)
        _NATIVE['group_norm'] = probe('group_norm',
                                      lambda t: F.group_norm(t, 2),
                                      shape=(2, 4, 3, 3))
        _LG_APPROX = not _NATIVE['lgamma']
        if why:
            print(f'{backend}: using fallbacks for '
                  f'{sorted(k for k, v in _NATIVE.items() if not v)}')
            for k in sorted(why):
                print(f'  {k}: {why[k]}')
        else:
            print(f'{backend}: all fused kernels available')

    def _lg(x):
        return _lgamma_dml(x) if _LG_APPROX else torch.lgamma(x)

    def _dg(x):
        return _digamma_dml(x) if _LG_APPROX else torch.digamma(x)

    def _dir_kl_rows(a, b):
        """forward KL( Dir(a) ‖ Dir(b) ) per row; a, b: (N,3)."""
        a0, b0 = a.sum(-1), b.sum(-1)
        return (_lg(a0) - _lg(a).sum(-1) - _lg(b0) + _lg(b).sum(-1)
                + ((a - b) * (_dg(a) - _dg(a0).unsqueeze(-1))).sum(-1))

    def _kl_terms(target, pred, normalize):
        """KL rows, optionally divided by the TARGET's own total concentration.

        An unnormalised Dirichlet KL scales roughly linearly with α₀, so if the
        target's concentration tracks the simulation budget (target_agg='mean' or
        'sum'), playout randomisation alone swings the term several-fold and no
        fixed loss weight can balance it.  Dividing makes it a per-observation KL
        — the optimum is still target == prediction, but note it also flattens
        the strongest targets hardest: a proven-terminal spike (α₀ = 100) is
        divided by 100, so proofs stop dominating the gradient."""
        kl = _dir_kl_rows(target, pred)
        if normalize:
            kl = kl / target.sum(-1).clamp_min(1.0)
        return kl

    def full_loss(out, out2, meta, weights, kl_normalize=False):
        """All five losses on one batch.

        `out`  = network outputs for the batch positions              (B rows)
        `out2` = outputs for the SUCCESSOR positions the consistency
                 term needs, or None when nothing in the batch has one (B2 rows)
        Returns (total, parts-dict-of-floats)."""
        v_logits, v_conf_raw, a_logits, a_conf_raw = out
        act = meta['pad_act']                            # (B,K) long
        mask = meta['pad_mask']                          # (B,K) bool
        idx3 = act.unsqueeze(-1).expand(-1, -1, 3)
        # Activate only the gathered LEGAL entries.
        q_logp = F.log_softmax(a_logits.gather(1, idx3), dim=-1)   # (B,K,3)
        q_p3 = q_logp.exp()
        q_cf = _conf(a_conf_raw.gather(1, act))                    # (B,K)
        q_alpha = (q_cf.unsqueeze(-1) * q_p3).clamp_min(ALPHA_FLOOR)
        q_alpha = torch.where(mask.unsqueeze(-1), q_alpha,
                              torch.full_like(q_alpha, ALPHA_FLOOR))
        v_logp = F.log_softmax(v_logits, dim=-1)                   # (B,3)
        v_alpha = (_conf(v_conf_raw).unsqueeze(-1) * v_logp.exp()).clamp_min(ALPHA_FLOOR)

        # (1) state-belief KL against the observed state distribution
        L_klv = _kl_terms(meta['v_obs'], v_alpha, kl_normalize).mean()

        # (2) action-belief KL over SEARCHED edges only
        evm = meta['ev_mask'].reshape(-1).float()
        kla = _kl_terms(meta['ev_alpha'].reshape(-1, 3),
                        q_alpha.reshape(-1, 3), kl_normalize)
        L_kla = (kla * evm).sum() / evm.sum().clamp_min(1.0)

        # (3) state cross-entropy against the game result
        zw = meta['z_w']
        ce_v = -v_logp.gather(1, meta['z_idx'].unsqueeze(1)).squeeze(1)
        L_cev = (zw * ce_v).sum() / zw.sum().clamp_min(1.0)

        # (4) action cross-entropy on the move actually played.
        # NOTE when reading the logged `cev` vs `cea`: they are NOT measured on
        # the same positions.  Solver-labelled aux samples have no played action,
        # so `cea` covers self-play moves only, while `cev` also includes every
        # aux sample — and those are exact, near-terminal, and easy.  In a
        # Connect 4 run aux samples are the MAJORITY of the buffer (~69% measured
        # over 2000 episodes), which is most of why `cev` reads ~0.25 while `cea`
        # sits at ~0.70.  The gap is a difference in sample population, not
        # evidence that the action head is learning the outcome less well.
        pw = meta['played_w'] * zw
        pl_idx = meta['played'].clamp_min(0)
        q_played_logp = q_logp.gather(
            1, pl_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)   # (B,3)
        ce_a = -q_played_logp.gather(1, meta['z_idx'].unsqueeze(1)).squeeze(1)
        L_cea = (pw * ce_a).sum() / pw.sum().clamp_min(1.0)

        # (5) consistency: Q(s,a*) must match the resulting state's flipped value
        cw = meta['cons_w']
        q_played = q_alpha.gather(
            1, pl_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)   # (B,3)
        tgt = meta['cons_term']                                     # terminal spikes
        if out2 is not None:
            v2_logits, v2_conf_raw, _a2l, _a2c = out2
            v2 = (_conf(v2_conf_raw).unsqueeze(-1)
                  * F.softmax(v2_logits, dim=-1)).clamp_min(ALPHA_FLOOR)
            v2f = v2.detach()[:, [_LOSS, _DRAW, _WIN]]              # one ply up
            rows = v2f.index_select(0, meta['cons_row'].clamp_min(0))
            tgt = torch.where(meta['cons_isnn'].unsqueeze(-1), rows, tgt)
        cons = _kl_terms(tgt.detach(), q_played, kl_normalize)
        L_cons = (cw * cons).sum() / cw.sum().clamp_min(1.0)

        w_klv, w_kla, w_cev, w_cea, w_cons = weights
        total = (w_klv * L_klv + w_kla * L_kla + w_cev * L_cev
                 + w_cea * L_cea + w_cons * L_cons)
        # Every reported scalar is stacked and pulled back in ONE .cpu() call —
        # a separate .item() per number is a full pipeline sync on DirectML.
        with torch.no_grad():
            evc = evm.sum().clamp_min(1.0)
            qa_p = (q_alpha.reshape(-1, 3).sum(-1) * evm).sum() / evc
            qa_t = (meta['ev_alpha'].reshape(-1, 3).sum(-1) * evm).sum() / evc
            # The plain batch means are dominated by solver-labelled samples,
            # whose targets are a TERMINAL_CONC spike by construction: measured
            # at 63% of a buffer they contributed 95% of the predicted-alpha0
            # mass, so `cv_p` reads ~23 while ordinary positions sit at ~0.2 and
            # match their targets closely.  Report the ORDINARY-sample figures
            # too, or the diagnostic just tracks the solver share.
            uns = meta['unsolved']
            unsc = uns.sum().clamp_min(1.0)
            uv_p = (v_alpha.sum(1) * uns).sum() / unsc
            uv_t = (meta['v_obs'].sum(1) * uns).sum() / unsc
            uevm = evm * uns.unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)
            uevc = uevm.sum().clamp_min(1.0)
            ua_p = (q_alpha.reshape(-1, 3).sum(-1) * uevm).sum() / uevc
            ua_t = (meta['ev_alpha'].reshape(-1, 3).sum(-1) * uevm).sum() / uevc
            diag = torch.stack([
                total, L_klv, L_kla, L_cev, L_cea, L_cons,
                v_alpha.sum(1).mean(), meta['v_obs'].sum(1).mean(), qa_p, qa_t,
                uv_p, uv_t, ua_p, ua_t, uns.mean(),
            ]).to('cpu', copy=False).tolist()
        parts = dict(zip(('loss', 'klv', 'kla', 'cev', 'cea', 'cons',
                          'cv_p', 'cv_t', 'ca_p', 'ca_t',
                          'uv_p', 'uv_t', 'ua_p', 'ua_t', 'unsolved'), diag))
        return total, parts

    _Z_CORNER = {1: _WIN, 0: _DRAW, -1: _LOSS}

    def build_batch_meta(batch, device, cons_frac=1.0, rng=None):
        """Pad a list of sample dicts into fixed-(B,K) target tensors on `device`,
        and collect the successor observations the consistency term needs.
        Returns (meta, obs2) where obs2 may be an empty list."""
        B = len(batch)
        K = max(len(s['legal']) for s in batch)
        pad_act = np.zeros((B, K), np.int64)
        pad_mask = np.zeros((B, K), bool)
        ev_mask = np.zeros((B, K), bool)
        ev_alpha = np.full((B, K, 3), TARGET_EPS, np.float32)
        v_obs = np.empty((B, 3), np.float32)
        z_idx = np.zeros(B, np.int64)
        z_w = np.zeros(B, np.float32)
        unsolved = np.zeros(B, np.float32)
        played = np.zeros(B, np.int64)
        played_w = np.zeros(B, np.float32)
        cons_w = np.zeros(B, np.float32)
        cons_row = np.full(B, -1, np.int64)
        cons_isnn = np.zeros(B, bool)
        cons_term = np.full((B, 3), TERMINAL_EPS, np.float32)
        obs2 = []
        if cons_frac >= 1.0:
            keep_cons = np.ones(B, bool)
        else:
            r = rng if rng is not None else np.random
            keep_cons = r.random(B) < cons_frac
        for i, s in enumerate(batch):
            k = len(s['legal'])
            pad_act[i, :k] = s['legal']
            pad_mask[i, :k] = True
            if len(s['ev_idx']):
                ev_mask[i, s['ev_idx']] = True
                ev_alpha[i, s['ev_idx']] = s['ev_alpha']
            v_obs[i] = s['v_obs']
            unsolved[i] = 0.0 if s['solved'] else 1.0
            z_w[i] = s['z_w']
            z_idx[i] = _Z_CORNER[int(round(float(s['z'])))]
            p = int(s['played'])
            if p >= 0:
                played[i] = p
                played_w[i] = 1.0
                if s['next_obs'] is not None:
                    # Each of these appends a row to the SECOND forward block, so
                    # a full-coverage consistency term nearly doubles the batch.
                    # Sub-sampling keeps the term unbiased (the loss is a
                    # weight-normalised mean) at the cost of gradient variance,
                    # and buys back most of that compute.
                    if keep_cons[i]:
                        cons_w[i] = 1.0
                        cons_row[i] = len(obs2)
                        cons_isnn[i] = True
                        obs2.append(s['next_obs'])
                elif s['next_term'] >= 0:
                    cons_w[i] = 1.0          # terminal: free, no extra NN row
                    cons_term[i] = _SPIKE[s['next_term']]
        # ONE transfer per dtype, not fourteen.  DirectML allocates a D3D12
        # resource per host->device copy, and a long run makes hundreds of
        # thousands of them (14 per step x 8 steps x thousands of episodes);
        # past a few hundred thousand the heap fragments and the copy fails with
        # a bare "RuntimeError: The parameter is incorrect" from inside .to().
        # Packing by dtype and slicing on-device cuts the allocation count ~5x
        # and is faster everywhere else too.
        i64 = np.concatenate([pad_act.reshape(-1), z_idx, played, cons_row])
        f32 = np.concatenate([ev_alpha.reshape(-1), v_obs.reshape(-1), z_w,
                              played_w, cons_w, cons_term.reshape(-1), unsolved])
        b8 = np.concatenate([pad_mask.reshape(-1), ev_mask.reshape(-1),
                             cons_isnn])
        ti = torch.from_numpy(i64).to(device)
        tf = torch.from_numpy(f32).to(device)
        tb = torch.from_numpy(b8).to(device)
        BK = B * K
        meta = {
            'pad_act':   ti[:BK].view(B, K),
            'z_idx':     ti[BK:BK + B],
            'played':    ti[BK + B:BK + 2 * B],
            'cons_row':  ti[BK + 2 * B:BK + 3 * B],
            'ev_alpha':  tf[:BK * 3].view(B, K, 3),
            'v_obs':     tf[BK * 3:BK * 3 + B * 3].view(B, 3),
            'z_w':       tf[BK * 3 + B * 3:BK * 3 + B * 4],
            'played_w':  tf[BK * 3 + B * 4:BK * 3 + B * 5],
            'cons_w':    tf[BK * 3 + B * 5:BK * 3 + B * 6],
            'cons_term': tf[BK * 3 + B * 6:BK * 3 + B * 9].view(B, 3),
            'unsolved':  tf[BK * 3 + B * 9:BK * 3 + B * 10],
            'pad_mask':  tb[:BK].view(B, K),
            'ev_mask':   tb[BK:2 * BK].view(B, K),
            'cons_isnn': tb[2 * BK:2 * BK + B],
        }
        return meta, obs2

    def device_retry(fn, *a, **kw):
        """Run `fn`; on a device-level RuntimeError, release dead device tensors
        and try once more.

        DirectML allocates a D3D12 resource per host->device copy and fails with
        a bare "RuntimeError: The parameter is incorrect" once its heap
        fragments — observed after ~28k training steps of a long run, inside
        `.to(device)` rather than in any op.  A gc pass usually frees enough for
        the next allocation to succeed.  Genuine out-of-memory is re-raised
        immediately; retrying that would only stall."""
        try:
            return fn(*a, **kw)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                raise
            import gc
            gc.collect()
            time.sleep(0.25)
            return fn(*a, **kw)

    def train_step(network, optimizer, batch, device, weights, grad_clip=1.0,
                   model_lock=None, kl_normalize=False, with_consistency=True,
                   cons_frac=1.0, rng=None):
        """One optimiser step, loss computed entirely ON-DEVICE for every backend.

        The consistency term needs the successor position's state belief, so the
        batch positions and their successors are concatenated into ONE forward
        pass (never two) and split afterwards.  With `with_consistency=False`
        (or a zero weight) the successor rows are not built at all."""
        import contextlib
        lock = model_lock or contextlib.nullcontext()

        def _once():
            # Everything that touches the device lives in here, so a retry
            # redoes the whole step rather than resuming a half-built one.
            meta, obs2 = build_batch_meta(batch, device, cons_frac, rng)
            if not with_consistency:
                obs2 = []
                meta['cons_w'] = torch.zeros_like(meta['cons_w'])
            obs = [s['obs'] for s in batch]
            B = len(obs)
            x = batch_to_tensor(obs + obs2, device)
            with lock:
                v_logits, v_conf, a_logits, a_conf = network(x)
                out = (v_logits[:B], v_conf[:B], a_logits[:B], a_conf[:B])
                out2 = ((v_logits[B:], v_conf[B:], a_logits[B:], a_conf[B:])
                        if obs2 else None)
                optimizer.zero_grad()
                loss, parts = full_loss(out, out2, meta, weights, kl_normalize)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(network.parameters(),
                                                       grad_clip)
                # A single non-finite step is unrecoverable: clip_grad_norm_
                # turns a NaN norm into a NaN scale factor, AdamW writes NaN
                # into every weight and every moment, and the run is dead while
                # continuing to look alive.  Skipping the step instead costs one
                # batch.  `parts['loss']` is already on the host (full_loss
                # stacks the diagnostics into one transfer), so the only added
                # sync is the grad norm.
                ok = (math.isfinite(parts['loss'])
                      and bool(torch.isfinite(gnorm)))
                if ok:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                parts['nonfinite'] = 0.0 if ok else 1.0
            return parts['loss'], parts

        return device_retry(_once)

    # ── DirectML-safe AdamW (aten::lerp has no DML kernel) ────────────────────
    class LerpFreeAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=1e-2):
            super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                          weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for g in self.param_groups:
                lr, (b1, b2) = g['lr'], g['betas']
                eps, wd = g['eps'], g['weight_decay']
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st['step'] = 0
                        st['m'] = torch.zeros_like(p)
                        st['v'] = torch.zeros_like(p)
                    m, v = st['m'], st['v']
                    st['step'] += 1
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st['step']
                    bc2 = 1 - b2 ** st['step']
                    denom = (v / bc2).sqrt_().add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)
            return loss

    # ── Inference helpers ─────────────────────────────────────────────────────
    def nn_eval_states(network, device, states):
        """States → (v3 (B,3), vconf (B,), p3 (B,A,3), conf (B,A), obs16)."""
        obs16 = np.asarray([make_obs(s) for s in states], dtype=np.float16)
        x = batch_to_tensor(obs16, device)
        with torch.inference_mode():
            v_logits, v_conf_raw, a_logits, a_conf_raw = network(x)
            v3 = F.softmax(v_logits, dim=-1)
            vc = _conf(v_conf_raw)
            p3 = F.softmax(a_logits, dim=-1)
            cf = _conf(a_conf_raw)
        return (v3.cpu().numpy(), vc.cpu().numpy(), p3.cpu().numpy(),
                cf.cpu().numpy(), obs16)

    def expand_node(network, device, state):
        v3, vc, p3, cf, ob = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        node = _CNode(state.current_player(), leg, v3[0], float(vc[0]),
                      p3[0][leg], cf[0][leg], obs=ob[0])
        _seed_leaf(node)
        return node

    class C4MCTSBot:
        """mcts_search(state) → searched root node.  Batched-leaf waves; Thompson
        sampling's own randomness plus virtual loss diversifies a wave.  Reads
        weights live from `network`."""

        def __init__(self, game, network, device, max_simulations,
                     batch_size=16, temp=1.0, random_state=None):
            self.game, self.network, self.device = game, network, device
            self.max_simulations = max_simulations
            self.batch_size = batch_size
            self.temp = temp
            self._rng = random_state or np.random.default_rng()

        def mcts_search(self, state, root=None):
            if root is None:
                root = expand_node(self.network, self.device, state)
            sims = 0
            while sims < self.max_simulations:
                if _node_solved_outcome(root) is not None:
                    break
                wave = min(self.batch_size, self.max_simulations - sims)
                pending = []
                for _ in range(wave):
                    path, st, payloads, edge = _select_leaf(
                        root, state, self._rng, self.temp)
                    if st is None:
                        _backup_terminal(path, payloads, None)
                        sims += 1
                    else:
                        pending.append((path, st, edge))
                uniq = {}
                for path, st, (node, idx) in pending:
                    uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    entries = list(uniq.values())
                    v3, vc, p3, cf, ob = nn_eval_states(
                        self.network, self.device, [e[2] for e in entries])
                    for (node, idx, st), a, b, c, d, o in zip(
                            entries, v3, vc, p3, cf, ob):
                        leg = st.legal_actions()
                        child = _CNode(st.current_player(), leg, a, float(b),
                                       c[leg], d[leg], obs=o)
                        _seed_leaf(child)
                        node.children[idx] = child
                for path, st, (node, idx) in pending:
                    _backup(path, _payload(node.children[idx].v_alpha))
                    sims += 1
            return root

    def thompson_value_move(network, state, device, rng, temp=1.0):
        """Search-free move: ONE Thompson sample of the network's per-action value
        beliefs — exactly the in-tree selection rule applied to the raw head."""
        _v3, _vc, p3, cf, _o = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        a = np.maximum(cf[0][leg, None] * p3[0][leg], ALPHA_FLOOR) * temp
        g = rng.standard_gamma(a)
        v = _dir_v_from_gammas(g, a, rng)
        return int(leg[int(v.argmax())])

    def value_greedy_move(network, state, device):
        """Search-free move: argmax of the action head's posterior mean value."""
        _v3, _vc, p3, cf, _o = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        a = np.maximum(cf[0][leg, None] * p3[0][leg], ALPHA_FLOOR)
        return int(leg[int(dir_value(a).argmax())])

    def quick_match(net_a, net_b, game, n_games, device, rng=None,
                    opening_plies=2, max_plies=42, temp=1.0):
        """Search-free Thompson-value match, alternating colours.  net == None →
        uniform-random mover.  Returns (wins_a, draws, wins_b)."""
        rng = rng or np.random.default_rng()
        wa = d = wb = 0
        for i in range(n_games):
            a_side = i % 2
            state = game.new_initial_state()
            for _ in range(opening_plies):
                if state.is_terminal():
                    break
                leg = state.legal_actions()
                state.apply_action(int(leg[rng.integers(len(leg))]))
            ply = 0
            while not state.is_terminal() and ply < max_plies:
                net = net_a if state.current_player() == a_side else net_b
                if net is None:
                    leg = state.legal_actions()
                    mv = int(leg[rng.integers(len(leg))])
                else:
                    mv = thompson_value_move(net, state, device, rng, temp)
                state.apply_action(mv); ply += 1
            if not state.is_terminal():
                d += 1; continue
            r = state.returns()[a_side]
            wa += r > 0; wb += r < 0; d += r == 0
        return int(wa), int(d), int(wb)

    # ══════════════════════════════════════════════════════════════════════════
    #  Single-process parallel self-play (fallback / debugging)
    # ══════════════════════════════════════════════════════════════════════════
    class ParallelSelfPlay(_SlotEngine):
        """Runs `n_parallel` games at once, sharing ONE NN forward per leaf wave
        across all of them.  Weights are read live, so in-flight games pick up
        training updates.  Same .episodes()/.stats interface as MPSelfPlayPool."""

        def __init__(self, game, network, device, cfg, checkpoint_dir=None,
                     seed=None):
            super().__init__(game, cfg, np.random.default_rng(seed),
                             checkpoint_dir)
            self.network, self.device = network, device
            self.n_parallel = cfg['n_parallel']
            self.wave = cfg['wave_per_game']
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0}
            self.fwd_calls = self.fwd_rows = 0
            self._pool_nets = {}
            self.slots = [self.new_game() for _ in range(self.n_parallel)]

        def set_curr_depth(self, d):
            self.curr_depth = float(d)

        def _load_pool_net(self, label):
            net = self._pool_nets.get(label)
            if net is None:
                net = load_benchmark_net(self.checkpoint_dir, label,
                                         self.cfg['net_sig'])
                self._pool_nets[label] = net
            return net

        def _finish(self, i):
            s = self.slots[i]; st = s['state']
            if st.is_terminal():
                ret = st.returns()
                finish_episode(s['hist'], ret)
                result = 'draw' if ret[0] == 0.0 else 'decisive'
                if self.cfg['restart_prob'] > 0 and result == 'decisive':
                    self._push_seed(s['actions'])
            else:
                mark_unfinished(s['hist']); result = 'cutoff'
            self.last_aux = len(s['aux'])
            self.stats['games'] += 1
            self.stats['plies'] += int(s['move'])
            if result == 'draw':   self.stats['draw'] += 1
            if result == 'cutoff': self.stats['cutoff'] += 1
            data = s['hist'] + s['aux']
            self.slots[i] = self.new_game()
            return data

        def _resolve_pool_moves(self):
            done = []
            for i, s in enumerate(self.slots):
                pool, state = s['pool'], s['state']
                if pool is None or state.current_player() != pool['side']:
                    continue
                if pool['label'] == 'random':
                    leg = state.legal_actions()
                    a = int(leg[self.rng.integers(len(leg))])
                else:
                    a = value_greedy_move(self._load_pool_net(pool['label']),
                                          state, 'cpu')
                self._apply_pool_move(s, a)
                if state.is_terminal() or s['move'] >= s['cap']:
                    done.append(self._finish(i))
            return done

        def _step(self):
            done = self._resolve_pool_moves()
            pending, evals, seen = [], [], set()
            for i, s in enumerate(self.slots):
                pool = s['pool']
                if pool is not None and s['state'].current_player() == pool['side']:
                    continue
                if s['root'] is None:
                    evals.append(('root', i, None, s['state'])); continue
                if _node_solved_outcome(s['root']) is not None:
                    continue
                wave = min(self.wave, s['sims'] - s['n'])
                for _ in range(max(wave, 0)):
                    path, st, payloads, edge = _select_leaf(
                        s['root'], s['state'], self.rng,
                        self._temp(s['move'] - s['resume']))
                    if st is None:
                        _backup_terminal(path, payloads, s['aux'])
                        s['n'] += 1
                    else:
                        node, idx = edge
                        pending.append((i, path, node, idx))
                        if (id(node), idx) not in seen:
                            seen.add((id(node), idx))
                            evals.append(('leaf', node, idx, st))
            if evals:
                self.fwd_calls += 1; self.fwd_rows += len(evals)
                v3, vc, p3, cf, ob = nn_eval_states(
                    self.network, self.device, [e[3] for e in evals])
                for (kind, a, b, st), x1, x2, x3, x4, o in zip(
                        evals, v3, vc, p3, cf, ob):
                    leg = st.legal_actions()
                    node = _CNode(st.current_player(), leg, x1, float(x2),
                                  x3[leg], x4[leg], obs=o)
                    _seed_leaf(node)
                    if kind == 'root':
                        self.slots[a]['root'] = node
                    else:
                        a.children[b] = node
            for i, path, node, idx in pending:
                _backup(path, _payload(node.children[idx].v_alpha))
                self.slots[i]['n'] += 1
            for i, s in enumerate(self.slots):
                if s['root'] is None:
                    continue
                if (s['n'] < s['sims']
                        and _node_solved_outcome(s['root']) is None):
                    continue
                self._record_move(s, s['root'])
                if s['state'].is_terminal() or s['move'] >= s['cap']:
                    done.append(self._finish(i))
            return done

        def episodes(self):
            while True:
                for data in self._step():
                    yield data

        def shutdown(self):
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Multiprocess self-play pool: worker processes + central GPU inference server
    # ══════════════════════════════════════════════════════════════════════════
    import threading as _threading
    import queue as _queue
    import multiprocessing as _mp

    def _probe_gather(device):
        """Does index_select run on this device?  Lets the server gather only the
        legal entries ON-device so only those cross the bus."""
        if str(device) == 'cpu':
            return True
        try:
            x = torch.arange(12.0, device=device).reshape(4, 3)
            idx = torch.tensor([2, 0, 3], device=device)
            y = x.index_select(0, idx).cpu()
            return bool(torch.equal(y, torch.tensor([[6., 7., 8.], [0., 1., 2.],
                                                     [9., 10., 11.]])))
        except Exception:
            return False

    class MPSelfPlayPool:
        """`n_workers` CPU processes run the trees; an inference-server thread here
        batches ALL their NN requests into one forward pass on `device`.  Exposes
        the same .episodes()/.stats/.last_aux interface as ParallelSelfPlay, so
        the training loop is identical.  `lock` serialises model access between
        the server thread and training (DirectML is not thread-safe)."""

        def __init__(self, network, device, n_workers, cfg, batch_window_s=0.002,
                     checkpoint_dir=None, max_batch_rows=1024):
            self.network, self.device = network, device
            self.checkpoint_dir = checkpoint_dir
            self.net_sig = cfg['net_sig']
            self._pool_nets = {}
            self.lock = _threading.Lock()
            self._stop = _threading.Event()
            self.window, self.max_batch_rows = batch_window_s, max_batch_rows
            self._gather_ok = _probe_gather(device)
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0}
            self.fwd_calls = self.fwd_rows = 0
            ctx = _mp.get_context('spawn')
            self.req_q = ctx.Queue()
            self.episode_q = ctx.Queue(maxsize=64)
            self.resp_qs = [ctx.Queue() for _ in range(n_workers)]
            self.pool_resp_qs = [ctx.Queue() for _ in range(n_workers)]
            self._curr = ctx.Value('d', float(cfg.get('curr_depth0', 8.0)))
            cfg = dict(cfg); cfg['curr_depth_shared'] = self._curr
            # Initialise the autograd engine's device state from the MAIN thread
            # before any other thread touches the device (DirectML assert).
            if str(device) != 'cpu':
                _t = torch.zeros(4, device=device, requires_grad=True)
                (_t * 2.0).sum().backward()
            self.procs = [ctx.Process(target=mp_worker,
                                      args=(i, self.req_q, self.resp_qs[i],
                                            self.pool_resp_qs[i],
                                            self.episode_q, cfg),
                                      daemon=True) for i in range(n_workers)]
            self.server = None
            try:
                for p in self.procs:
                    p.start()
            except Exception:
                # A half-started pool would otherwise hang the interpreter on
                # exit (live queue feeder threads with no reader).  Tear the
                # started workers down and let the real error surface.  The
                # usual cause is running this from a plain script without an
                # `if __name__ == "__main__":` guard — 'spawn' re-imports the
                # main module, so an unguarded script recurses into itself.
                self.procs = [p for p in self.procs if p.is_alive()]
                self.shutdown()
                raise
            self.server = _threading.Thread(target=self._serve, daemon=True)
            self.server.start()

        def set_curr_depth(self, d):
            self._curr.value = float(d)

        @property
        def curr_depth(self):
            return self._curr.value

        def _get_net(self, net_id):
            if net_id == 'live':
                return self.network, self.device, True
            net = self._pool_nets.get(net_id)
            if net is None:
                try:
                    net = load_benchmark_net(self.checkpoint_dir, net_id,
                                             self.net_sig)
                except Exception as e:
                    # A worker picks its pool opponent by listing the checkpoint
                    # directory, so the file can vanish before we load it.  An
                    # exception here would kill the server thread and leave every
                    # worker blocked on a response that never comes, so fall back
                    # to the live net for this request instead.
                    print(f'pool net {net_id} unavailable ({e}) — using live net')
                    return self.network, self.device, True
                self._pool_nets[net_id] = net
            return net, 'cpu', False

        def _forward_gathered(self, net, dev, xin, flat):
            x = torch.from_numpy(xin).to(dev)
            v_logits, v_conf_raw, a_logits, a_conf_raw = net(x)
            v3 = F.softmax(v_logits, dim=-1)
            vc = _conf(v_conf_raw)
            if self._gather_ok and str(dev) != 'cpu':
                ft = torch.from_numpy(flat).to(dev)
                p = F.softmax(a_logits.reshape(-1, 3).index_select(0, ft),
                              dim=-1).cpu().numpy()
                c = _conf(a_conf_raw.reshape(-1).index_select(0, ft)).cpu().numpy()
            else:
                p = F.softmax(a_logits.reshape(-1, 3), dim=-1).cpu().numpy()[flat]
                c = _conf(a_conf_raw.reshape(-1)).cpu().numpy()[flat]
            return v3.cpu().numpy(), vc.cpu().numpy(), p, c

        def _serve(self):
            A = _NUM_ACTIONS
            while not self._stop.is_set():
                try:
                    reqs = [self.req_q.get(timeout=0.1)]
                except _queue.Empty:
                    continue
                rows = reqs[0][2].shape[0]
                deadline = time.monotonic() + self.window
                while time.monotonic() < deadline and rows < self.max_batch_rows:
                    try:
                        r = self.req_q.get_nowait(); reqs.append(r)
                        rows += r[2].shape[0]
                    except _queue.Empty:
                        time.sleep(0.0003)
                groups = {}
                for wid, net_id, obs, legals in reqs:
                    groups.setdefault(net_id, []).append((wid, obs, legals))
                for net_id, group in groups.items():
                    net, dev, needs_lock = self._get_net(net_id)
                    obs = np.concatenate([o for _, o, _ in group], axis=0)
                    xin = obs.reshape(-1, *_OBS_SHAPE).astype(np.float32)
                    if net_id == 'live':
                        self.fwd_calls += 1; self.fwd_rows += xin.shape[0]
                    row_legals = [l for _, _, ls in group for l in ls]
                    flat = np.concatenate([l.astype(np.int64) + r * A
                                           for r, l in enumerate(row_legals)])
                    offs = np.zeros(len(row_legals) + 1, dtype=np.int64)
                    np.cumsum([len(l) for l in row_legals], out=offs[1:])
                    import contextlib
                    ctxm = self.lock if needs_lock else contextlib.nullcontext()
                    with ctxm, torch.no_grad():
                        v3, vc, p, c = self._forward_gathered(net, dev, xin, flat)
                    tqs = self.resp_qs if net_id == 'live' else self.pool_resp_qs
                    ri = 0
                    for wid, o, ls in group:
                        out = []
                        for _ in ls:
                            a, b = offs[ri], offs[ri + 1]
                            out.append((v3[ri], float(vc[ri]), p[a:b], c[a:b]))
                            ri += 1
                        tqs[wid].put(out)

        def episodes(self):
            while True:
                samples, n_aux, result, plies = self.episode_q.get()
                self.last_aux = n_aux
                self.stats['games'] += 1
                self.stats['plies'] += plies
                if result == 'draw':   self.stats['draw'] += 1
                if result == 'cutoff': self.stats['cutoff'] += 1
                yield samples

        def shutdown(self):
            self._stop.set()
            try:
                if self.server is not None:
                    self.server.join(timeout=2.0)
            except Exception:
                pass
            for p in self.procs:
                p.terminate()
            for p in self.procs:
                p.join(timeout=2.0)
            for q in ([self.req_q, self.episode_q] + self.resp_qs
                      + self.pool_resp_qs):
                try:
                    q.close(); q.cancel_join_thread()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Sparse deep eval — running-Elo pool of checkpoints @ MCTS-N + random
    # ══════════════════════════════════════════════════════════════════════════
    # Every checkpoint enters a SINGLE Elo table, each rated at MCTS=`eval_sims`,
    # alongside a `random` mover.  Per new checkpoint the cost is FIXED
    # (independent of pool size): it plays `games_per_pair` games against each of
    # the last-N checkpoints + random, then `refresh_pairs` random pairs from the
    # whole pool play too (keeps old ratings mixing).  Elo K decays with the
    # number of games a pair has already played, so ratings settle.
    class EloPool:
        def __init__(self, game, device, eval_sims=128, k_base=32.0,
                     k_halflife=30.0, games_per_pair=4, last_n=3,
                     refresh_pairs=10, opening_plies=2, batch_size=16,
                     start_elo=1000.0, eval_temp=6.0, max_eval_plies=42, seed=0):
            self.game, self.device = game, device
            self.eval_sims = eval_sims
            self.eval_temp = eval_temp
            self.k_base, self.k_hl = k_base, k_halflife
            self.games_per_pair = games_per_pair
            self.last_n, self.refresh_pairs = last_n, refresh_pairs
            self.opening_plies = opening_plies
            self.max_eval_plies = max_eval_plies
            self.batch_size = batch_size
            self.start_elo = start_elo
            self.rng = np.random.default_rng(seed)
            self.players = ['random']
            self.nets = {'random': None}
            self.elo = {'random': start_elo}
            self.order = []
            self.pair_games = {}

        def _bot(self, label):
            return C4MCTSBot(self.game, self.nets[label], self.device,
                             self.eval_sims, batch_size=self.batch_size,
                             temp=self.eval_temp, random_state=self.rng)

        def _move(self, label, bot_cache, state):
            if self.nets[label] is None:
                leg = state.legal_actions()
                return int(leg[self.rng.integers(len(leg))])
            bot = bot_cache.setdefault(label, self._bot(label))
            root = bot.mcts_search(state)
            return root_pick(root, self.rng, thompson=False)

        def _play(self, a, b, bot_cache):
            """One game: `a` moves first.  A couple of random opening plies add
            variety.  Returns the first mover's result in {1, 0.5, 0}."""
            state = self.game.new_initial_state()
            for _ in range(self.opening_plies):
                if state.is_terminal():
                    break
                leg = state.legal_actions()
                state.apply_action(int(leg[self.rng.integers(len(leg))]))
            ply = 0
            while not state.is_terminal() and ply < self.max_eval_plies:
                lab = a if state.current_player() == 0 else b
                state.apply_action(self._move(lab, bot_cache, state))
                ply += 1
            if state.is_terminal():
                r = state.returns()[0]
                return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
            return 0.5

        def _update(self, a, b, sa):
            key = frozenset((a, b))
            n = self.pair_games.get(key, 0)
            k = self.k_base * self.k_hl / (self.k_hl + n)
            ea = 1.0 / (1.0 + 10 ** ((self.elo[b] - self.elo[a]) / 400.0))
            self.elo[a] += k * (sa - ea)
            self.elo[b] += k * ((1.0 - sa) - (1.0 - ea))
            self.pair_games[key] = n + 1

        def _match(self, a, b, bot_cache):
            for g in range(self.games_per_pair):
                w, x = (a, b) if g % 2 == 0 else (b, a)     # alternate who starts
                s_first = self._play(w, x, bot_cache)
                sa = s_first if w == a else 1.0 - s_first
                self._update(a, b, sa)

        def add_checkpoint(self, label, net):
            """Register a checkpoint (rated at MCTS-`eval_sims`), warm-start its
            Elo from the previous one, then run its fixed-cost eval."""
            self.nets[label] = net
            self.elo.setdefault(label, self.elo[self.order[-1]] if self.order
                                else self.start_elo)
            self.players.append(label)
            bot_cache = {}
            opponents = self.order[-self.last_n:] + ['random']
            for opp in opponents:
                self._match(label, opp, bot_cache)
            # Refresh random pairs across the pool — but only once it is big
            # enough that these are genuinely NEW pairings, otherwise it just
            # replays the pair that was played above.
            existing = self.order + ['random']
            n_new_pairs = (len(existing) * (len(existing) - 1) // 2
                           - len(opponents))
            if n_new_pairs > 0:
                seen = set()
                for _ in range(min(self.refresh_pairs, n_new_pairs)):
                    for _try in range(20):
                        a, b = self.rng.choice(len(existing), 2, replace=False)
                        key = frozenset((existing[a], existing[b]))
                        if key not in seen and label not in key:
                            seen.add(key); break
                    else:
                        break
                    self._match(existing[a], existing[b], bot_cache)
            self.order.append(label)
            return dict(self.elo)

    # ══════════════════════════════════════════════════════════════════════════
    #  Checkpointing
    # ══════════════════════════════════════════════════════════════════════════
    def _cpu_sd(net):
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    def cpu_clone(net, sig):
        c = C4DirichletNet(*sig)
        c.load_state_dict(_cpu_sd(net)); c.eval()
        return c

    # One definition of the history schema, so a rewound or back-filled
    # checkpoint has exactly the columns the plots and the eval line expect.
    _HIST_KEYS = ('ep', 'loss', 'klv', 'kla', 'cev', 'cea', 'cons', 'draw_pct',
                  'plies', 'buf', 'aux', 'elo', 'quick_ep', 'q_w', 'q_d', 'q_l',
                  'cv_p', 'cv_t', 'ca_p', 'ca_t', 'uv_p', 'uv_t', 'ua_p',
                  'ua_t', 'unsolved', 'solved', 'solved_ep')
    _HIST_QUICK = ('quick_ep', 'q_w', 'q_d', 'q_l')

    def _new_hist():
        return {k: [] for k in _HIST_KEYS}

    def _assert_finite_sd(sd, what):
        """Refuse to persist a poisoned network.

        A NaN net still plays (argmax of NaN picks index 0) and still trains, so
        nothing downstream notices — it just overwrites the last good weights
        and the resume checkpoint with garbage.  Failing the SAVE keeps the most
        recent healthy checkpoint on disk to restart from."""
        bad = [k for k, v in sd.items()
               if torch.is_tensor(v) and v.is_floating_point()
               and not torch.isfinite(v).all()]
        if bad:
            raise ValueError(
                f'refusing to save {what}: non-finite weights in {len(bad)} '
                f'tensor(s), e.g. {bad[:3]}. The most recent healthy checkpoint '
                f'on disk is untouched — restart from it.')

    def save_benchmark_net(checkpoint_dir, label, net):
        os.makedirs(checkpoint_dir, exist_ok=True)
        sd = _cpu_sd(net)
        _assert_finite_sd(sd, f'bench_{label}.pt')
        torch.save(sd, os.path.join(checkpoint_dir, f'bench_{label}.pt'))

    def load_benchmark_net(checkpoint_dir, label, sig):
        net = C4DirichletNet(*sig)
        net.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f'bench_{label}.pt'),
            map_location='cpu', weights_only=True))
        net.eval()
        return net

    def save_checkpoint(checkpoint_dir, ep, base_network, optimizer, scheduler,
                        elo_pool, hist, cfg=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        _assert_finite_sd(_cpu_sd(base_network), f'latest.pt at ep {ep}')
        blob = {'ep': ep, 'model': _cpu_sd(base_network),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict() if scheduler else None,
                'elo': elo_pool.elo, 'order': elo_pool.order,
                'pair_games': {tuple(sorted(k)): v
                               for k, v in elo_pool.pair_games.items()},
                'hist': hist, 'cfg': asdict(cfg) if cfg is not None else None}
        tmp = os.path.join(checkpoint_dir, 'latest.pt.tmp')
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(checkpoint_dir, 'latest.pt'))

    def load_checkpoint(checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'latest.pt')
        if not os.path.exists(path):
            return None
        try:
            blob = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            blob = torch.load(path, map_location='cpu')
        # `latest.pt` is a run blob; `bench_N.pt` is a bare state_dict.  Copying
        # one over the other is the obvious way to try to rewind a run, so say
        # what to do instead of failing later with KeyError: 'model'.
        if isinstance(blob, dict) and 'model' not in blob:
            if all(torch.is_tensor(v) for v in blob.values()):
                raise ValueError(
                    f'{path} is a bare network state_dict (a bench_N.pt), not a '
                    f'resumable checkpoint. To restart a run from generation N, '
                    f'use rewind_to_generation(checkpoint_dir, N) — it rebuilds '
                    f'latest.pt with the right episode counter and Elo pool.')
            raise ValueError(f'{path} has no "model" key and is not a '
                             f'state_dict; it is not a usable checkpoint.')
        return blob

    def rewind_to_generation(checkpoint_dir, gen, log=print):
        """Rebuild `latest.pt` so training resumes from the `bench_{gen}.pt`
        snapshot, discarding everything after it.

        Deleting latest.pt does NOT do this — it restarts from episode 1, since
        the benchmark snapshots are never read as a resume point.  Optimizer and
        scheduler state are dropped (they belong to the discarded episodes; the
        LR schedule is recomputed from `ep`), history is truncated at `gen`, and
        any Elo entry for a later generation is removed."""
        bench = os.path.join(checkpoint_dir, f'bench_{gen}.pt')
        if not os.path.exists(bench):
            raise FileNotFoundError(bench)
        sd = torch.load(bench, map_location='cpu', weights_only=True)
        _assert_finite_sd(sd, f'bench_{gen}.pt')

        path = os.path.join(checkpoint_dir, 'latest.pt')
        old = {}
        if os.path.exists(path):
            try:
                old = torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                old = {}
            if not isinstance(old, dict) or 'model' not in old:
                old = {}                      # a bench file copied into place

        prev = old.get('hist') or {}
        keep = sum(1 for e in prev.get('ep', []) if e <= gen)
        qkeep = sum(1 for e in prev.get('quick_ep', []) if e <= gen)
        hist = _new_hist()
        for k in _HIST_KEYS:
            v = prev.get(k)
            if isinstance(v, list):
                hist[k] = v[:qkeep if k in _HIST_QUICK else keep]

        # Drop only the generations being discarded.  'random' is a permanent
        # member of the pool and is NOT a generation label, so it has to survive
        # — the Elo update reads elo['random'] on every match.
        def _later(lb):
            return str(lb).isdigit() and int(lb) > gen
        order = [lb for lb in old.get('order', []) if not _later(lb)]
        elo = {k: v for k, v in (old.get('elo') or {}).items() if not _later(k)}
        pair = {tuple(sorted(k)): v
                for k, v in (old.get('pair_games') or {}).items()
                if not any(_later(p) for p in k)}

        blob = {'ep': int(gen), 'model': sd, 'optim': None, 'sched': None,
                'elo': elo, 'order': order, 'pair_games': pair,
                'hist': hist, 'cfg': old.get('cfg')}
        tmp = path + '.tmp'
        torch.save(blob, tmp)
        os.replace(tmp, path)
        log(f'rewound {checkpoint_dir} to generation {gen} '
            f'(optimizer state dropped, {len(order)} Elo entries kept)')
        return blob

    # ══════════════════════════════════════════════════════════════════════════
    #  Training driver — the notebook calls this and nothing else
    # ══════════════════════════════════════════════════════════════════════════
    def _worker_cfg(cfg, sig):
        return dict(
            seed=cfg.seed, game_name='connect_four', net_sig=sig,
            games_per_worker=cfg.games_per_worker, wave=cfg.worker_wave,
            n_parallel=cfg.n_parallel_games, wave_per_game=cfg.wave_per_game,
            fast_sims=cfg.fast_sims, full_sims=cfg.full_sims,
            fast_prob=cfg.fast_prob, temp_threshold=cfg.temp_threshold,
            late_temp=cfg.late_temp, max_plies=cfg.max_plies,
            pool_prob=cfg.pool_prob, random_pool_frac=cfg.random_pool_frac,
            checkpoint_dir=cfg.checkpoint_dir,
            restart_prob=cfg.restart_prob, restart_k_min=cfg.restart_k_min,
            restart_k_max=cfg.restart_k_max,
            restart_pool_cap=cfg.restart_pool_cap,
            curriculum=cfg.curriculum, curr_depth0=cfg.curr_depth0,
            curr_mcts_tail=cfg.curr_mcts_tail,
            search_agg=cfg.search_agg, target_agg=cfg.target_agg,
            selection=cfg.selection, virtual_loss=cfg.virtual_loss)

    def build_network(cfg, device):
        sig = (cfg.channels, cfg.num_blocks, cfg.head_ch)
        return C4DirichletNet(*sig).to(device), sig

    def _load_solved_probe(cfg, log):
        """Precomputed value probe for the absolute metric, or None when it is
        not configured.  Built once; each eval is then one forward pass."""
        if not cfg.solved_dir:
            return None
        try:
            import connect4_solved_eval as sev
            probe = sev.ValueProbe.build(cfg.solved_dir, limit=cfg.solved_n,
                                         log=log)
            return probe
        except Exception as e:
            log(f'solved-position eval DISABLED ({type(e).__name__}: {e})')
            return None

    def _solved_eval_line(probe, net, cfg):
        """Value MSE per difficulty for one checkpoint: a single batched forward
        over the precomputed observations, so this is cheap enough to run at
        every deep eval.  Returns (printable line, pooled MSE)."""
        import connect4_solved_eval as sev
        d = probe.mse_by_bucket(sev.thompson_value_fn(net, cfg.eval_device))
        return sev.line(d), d['all']

    def run_training(cfg, game=None, log=print):
        """Full self-play + training run.  Prints the same two-tier eval scheme
        as the chess notebook and returns the history dict."""
        import threading
        game = game or load_game()
        device, backend = pick_device(cfg.device_preference)
        set_game(game)
        set_search(cfg.search_agg, cfg.target_agg, cfg.selection,
                   cfg.virtual_loss)
        set_backend(backend, device)
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        base_network, sig = build_network(cfg, device)
        network = base_network
        if backend != 'directml' and hasattr(torch, 'compile'):
            import shutil
            if any(shutil.which(c) for c in ('g++', 'gcc', 'clang', 'cl')):
                try:
                    _c = torch.compile(base_network, dynamic=True)
                    with torch.no_grad():
                        _c(torch.zeros(1, *_OBS_SHAPE, device=device))
                    network = _c; log('torch.compile: enabled')
                except Exception as e:
                    log(f'torch.compile: disabled ({type(e).__name__}) — eager')

        optimizer = (LerpFreeAdamW if backend == 'directml' else torch.optim.AdamW)(
            network.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

        def _lr(ep):
            if ep < cfg.lr_warmup_eps:
                return ep / max(cfg.lr_warmup_eps, 1)
            f = min((ep - cfg.lr_warmup_eps)
                    / max(cfg.lr_decay_eps - cfg.lr_warmup_eps, 1), 1.0)
            return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                1 + np.cos(np.pi * f))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)

        wcfg = _worker_cfg(cfg, sig)
        if cfg.use_workers:
            n_w = cfg.resolved_workers()
            self_play = MPSelfPlayPool(network, device, n_w, wcfg,
                                       checkpoint_dir=cfg.checkpoint_dir)
            torch.set_num_threads(max(2, (os.cpu_count() or 8) - n_w))
            log(f'Self-play: {n_w} WORKER PROCESSES x {cfg.games_per_worker} '
                f'games -> central inference server on {device}')
        else:
            self_play = ParallelSelfPlay(game, network, device, wcfg,
                                         checkpoint_dir=cfg.checkpoint_dir,
                                         seed=cfg.seed)
            log(f'Self-play: SINGLE PROCESS, {cfg.n_parallel_games} '
                f'parallel games')
        episode_stream = self_play.episodes()
        model_lock = getattr(self_play, 'lock', None) or threading.Lock()

        elo_pool = EloPool(game, cfg.eval_device, eval_sims=cfg.eval_sims,
                           k_base=cfg.eval_k_base, k_halflife=cfg.eval_k_halflife,
                           games_per_pair=cfg.eval_games_per_pair,
                           last_n=cfg.eval_last_n,
                           refresh_pairs=cfg.eval_refresh_pairs,
                           opening_plies=cfg.eval_opening_plies,
                           start_elo=cfg.start_elo, eval_temp=cfg.eval_temp,
                           max_eval_plies=cfg.eval_max_plies, seed=cfg.seed)

        hist = _new_hist()
        replay_buffer, start_ep, aux_total = [], 1, 0

        ckpt = load_checkpoint(cfg.checkpoint_dir) if cfg.resume else None
        if ckpt is not None:
            base_network.load_state_dict(ckpt['model'])
            # A rewound checkpoint (rewind_to_generation) carries weights only:
            # the optimizer and scheduler state belonged to the episodes being
            # discarded, and the LR schedule is a function of `ep` anyway.
            if ckpt.get('optim'):
                optimizer.load_state_dict(ckpt['optim'])
            else:
                log('  (no optimizer state in the checkpoint — starting Adam '
                    'moments from zero)')
            if ckpt.get('sched'):
                scheduler.load_state_dict(ckpt['sched'])
            elo_pool.elo = dict(ckpt.get('elo') or {})
            # 'random' is the pool's anchor and every Elo update reads it; a
            # hand-edited or rewound checkpoint may not carry it.
            elo_pool.elo.setdefault('random', cfg.start_elo)
            elo_pool.order = list(ckpt.get('order') or [])
            elo_pool.pair_games = {frozenset(k): v
                                   for k, v in (ckpt.get('pair_games')
                                                or {}).items()}
            # A benchmark file can be missing (interrupted save, hand-cleaned
            # directory) — drop that rating rather than making the whole
            # checkpoint unloadable.
            kept = []
            for lb in list(elo_pool.order):
                try:
                    elo_pool.nets[lb] = load_benchmark_net(cfg.checkpoint_dir,
                                                           lb, sig)
                    kept.append(lb)
                except FileNotFoundError:
                    log(f'  (benchmark {lb} missing — dropped from the Elo pool)')
                    elo_pool.elo.pop(lb, None)
            elo_pool.order = kept
            elo_pool.players = ['random'] + list(kept)
            # A checkpoint written before a diagnostic was added has no series
            # for it; back-fill with NaN so the columns stay aligned and the
            # first eval after a resume doesn't KeyError.
            old = ckpt.get('hist') or {}
            n = len(old.get('ep', []))
            for k, v in hist.items():
                if k not in old:
                    old[k] = [float('nan')] * (n if k != 'elo' else 0)
            hist = old
            start_ep = ckpt['ep'] + 1
            log(f'resumed at ep {ckpt["ep"]}')

        solved_probe = _load_solved_probe(cfg, log)

        n_params = sum(p.numel() for p in base_network.parameters())
        log(f'device={device} backend={backend} | params={n_params:,} '
            f'| search_agg={cfg.search_agg} target_agg={cfg.target_agg} '
            f'| start_ep={start_ep}')

        if not elo_pool.order:      # seed the Elo pool (fresh run, or a resume
            label = str(start_ep - 1)   # whose benchmark files went missing)
            init = cpu_clone(base_network, sig)
            save_benchmark_net(cfg.checkpoint_dir, label, init)
            elo_pool.add_checkpoint(label, init)

        from collections import defaultdict
        prev = dict(self_play.stats)
        t_sp = t_tr = 0.0
        win_t0 = time.perf_counter(); win_games = self_play.stats['games']
        bar = Progress(cfg.quick_eval_every, label='ep ')
        # The bar's g/s is INSTANTANEOUS (a ~3s rolling window), not a cumulative
        # mean: a cumulative mean decays asymptotically toward the true rate,
        # which reads as a slow steady "decline" for thousands of episodes after
        # any one-off speed change (e.g. training switching on once the buffer
        # first fills) — misleading exactly when you want to spot a real slowdown.
        bar_mark = (time.perf_counter(), self_play.stats['games']); inst_gs = 0.0
        prev_fwd = (self_play.fwd_calls, self_play.fwd_rows)
        with_cons = cfg.loss_weights[4] > 0.0
        step_fails = 0
        step_rng = np.random.default_rng(cfg.seed + 991)
        try:
            for ep in range(start_ep, cfg.num_episodes + 1):
                network.eval()
                a = time.perf_counter()
                raw = next(episode_stream)     # blocks iff self-play can't keep up
                t_sp += time.perf_counter() - a
                aux_total += self_play.last_aux
                replay_buffer.extend(raw)
                if len(replay_buffer) > cfg.max_buffer:
                    del replay_buffer[:-cfg.max_buffer]

                network.train()
                accs = defaultdict(list)
                b = time.perf_counter()
                if len(replay_buffer) >= cfg.batch_size:
                    for _ in range(cfg.train_steps_per_ep):
                        batch = random.sample(replay_buffer, cfg.batch_size)
                        try:
                            _lv, parts = train_step(
                                network, optimizer, batch, device,
                                cfg.loss_weights, cfg.grad_clip,
                                model_lock=model_lock,
                                kl_normalize=cfg.kl_normalize,
                                with_consistency=with_cons,
                                cons_frac=cfg.cons_frac, rng=step_rng)
                        except RuntimeError as e:
                            # Losing one step in thousands costs nothing; losing
                            # a multi-hour run to a transient device fault costs
                            # a lot.  Give up only if it stops being transient.
                            step_fails += 1
                            log(f'  ! train step failed, skipped '
                                f'({type(e).__name__}: {e}) '
                                f'[{step_fails} consecutive]')
                            if step_fails >= 20:
                                raise
                            continue
                        step_fails = 0
                        for k, v in parts.items():
                            accs[k].append(v)
                    scheduler.step()
                t_tr += time.perf_counter() - b

                done = ep - (ep - 1) // cfg.quick_eval_every * cfg.quick_eval_every
                if ep % cfg.quick_eval_every != 0:
                    tn, gn = time.perf_counter(), self_play.stats['games']
                    if tn - bar_mark[0] >= 3.0:
                        inst_gs = (gn - bar_mark[1]) / (tn - bar_mark[0])
                        bar_mark = (tn, gn)
                    bar.update(done, extra=f'{inst_gs:.2f} g/s')
                    continue
                bar.close()              # wipe the bar before the eval line prints

                st = self_play.stats
                dg = max(st['games'] - prev['games'], 1)
                draw_pct = 100 * (st['draw'] - prev['draw']) / dg
                plies = (st['plies'] - prev['plies']) / dg
                prev = dict(st)
                ml = lambda k: float(np.mean(accs[k])) if accs[k] else float('nan')
                hist['ep'].append(ep); hist['loss'].append(ml('loss'))
                for k in ('klv', 'kla', 'cev', 'cea', 'cons',
                          'cv_p', 'cv_t', 'ca_p', 'ca_t',
                          'uv_p', 'uv_t', 'ua_p', 'ua_t', 'unsolved'):
                    hist[k].append(ml(k))
                hist['draw_pct'].append(draw_pct); hist['plies'].append(plies)
                hist['buf'].append(len(replay_buffer)); hist['aux'].append(aux_total)
                # Concentration (α₀) of predicted vs target Dirichlets per head.
                # Both populations, because the all-sample mean mostly reports
                # how much of the buffer the solver labelled (see full_loss).
                conc = (f'conc(pred/tgt) v {ml("cv_p"):.1f}/{ml("cv_t"):.1f} '
                        f'a {ml("ca_p"):.1f}/{ml("ca_t"):.1f} | unsolved '
                        f'({100*ml("unsolved"):.0f}% of batch) '
                        f'v {ml("uv_p"):.2f}/{ml("uv_t"):.2f} '
                        f'a {ml("ua_p"):.2f}/{ml("ua_t"):.2f}')
                # WEIGHTED share of the total loss per term — the number to tune
                # loss_weights by.  If one term sits near 100% the others have
                # effectively stopped training.
                names = ('klv', 'kla', 'cev', 'cea', 'cons')
                comp = {n: w * ml(n) for n, w in zip(names, cfg.loss_weights)}
                tot = sum(v for v in comp.values() if v == v) or float('nan')
                share = 'sh ' + ' '.join(f'{k} {100 * v / tot:.0f}%'
                                         for k, v in comp.items())
                # Steps whose loss or gradient was not finite and were therefore
                # NOT applied.  Loud on purpose: a silent NaN destroyed three
                # 4-hour arms once already.
                nf = float(np.sum(accs['nonfinite'])) if accs['nonfinite'] else 0.0
                if nf:
                    log(f'  ! {nf:.0f} of {len(accs["nonfinite"])} train steps '
                        f'had a non-finite loss/gradient and were SKIPPED — '
                        f'weights are intact, but something is wrong upstream.')
                curr_s = ''
                if cfg.curriculum:
                    cd = self_play.curr_depth
                    if cd < cfg.curr_max_depth and ml('cev') < cfg.curr_val_thresh:
                        self_play.set_curr_depth(
                            min(cfg.curr_max_depth, cd + cfg.curr_step))
                    curr_s = f' curr {self_play.curr_depth:.0f}'
                diag = (f'loss {ml("loss"):.2f} (klv {ml("klv"):.2f} '
                        f'kla {ml("kla"):.2f} cev {ml("cev"):.3f} '
                        f'cea {ml("cea"):.3f} cons {ml("cons"):.2f}) | {share} '
                        f'| {conc} | dr {draw_pct:.0f}% ply {plies:.0f}{curr_s} '
                        f'buf {len(replay_buffer) // 1000}k aux {aux_total} '
                        f'| lr {optimizer.param_groups[0]["lr"]:.2e}')
                # Perf: games/s over the window + where wall-time went + batch size.
                #   wait(sp) high → self-play is the bottleneck
                #   train high    → training steps dominate the GPU
                #   NNbatch small → GPU underfed: raise games_per_worker/workers
                wall = max(time.perf_counter() - win_t0, 1e-9)
                dg2 = self_play.stats['games'] - win_games
                dfc = self_play.fwd_calls - prev_fwd[0]
                dfr = self_play.fwd_rows - prev_fwd[1]
                perf = (f'{dg2 / wall:.2f} games/s | wait(sp) {100 * t_sp / wall:.0f}% '
                        f'train {100 * t_tr / wall:.0f}% '
                        f'| NNbatch {dfr / max(dfc, 1):.0f} '
                        f'({dfc / wall:.0f} fwd/s)')
                t_sp = t_tr = 0.0; win_t0 = time.perf_counter()
                win_games = self_play.stats['games']
                prev_fwd = (self_play.fwd_calls, self_play.fwd_rows)

                if ep % cfg.deep_eval_every == 0:
                    snap = cpu_clone(base_network, sig)
                    save_benchmark_net(cfg.checkpoint_dir, str(ep), snap)
                    elo = elo_pool.add_checkpoint(str(ep), snap)
                    hist['elo'].append(dict(elo))
                    ladder = '  '.join(f'{k}={v:.0f}' for k, v in
                                       sorted(elo.items(), key=lambda kv: -kv[1])[:6])
                    log(f'ep {ep:6d} | {diag}')
                    log(f'         DEEP Elo@{cfg.eval_sims}: {ladder}')
                    if solved_probe and (not cfg.solved_every
                                          or ep % cfg.solved_every == 0):
                        line, agg = _solved_eval_line(solved_probe, snap, cfg)
                        hist['solved'].append(agg)
                        hist['solved_ep'].append(ep)
                        log(f'         SOLVED {line}')
                else:
                    eval_net = cpu_clone(base_network, sig)
                    ref = elo_pool.order[-1]
                    w, d, l = quick_match(eval_net, elo_pool.nets[ref], game,
                                          cfg.quick_eval_games, cfg.eval_device,
                                          opening_plies=cfg.eval_opening_plies,
                                          max_plies=cfg.eval_max_plies)
                    hist['quick_ep'].append(ep); hist['q_w'].append(w)
                    hist['q_d'].append(d); hist['q_l'].append(l)
                    log(f'ep {ep:6d} | {diag} | vs {ref} (no-MCTS) W{w} D{d} L{l}')
                log(f'         perf: {perf}')
                save_checkpoint(cfg.checkpoint_dir, ep, base_network, optimizer,
                                scheduler, elo_pool, hist, cfg)
                bar.reset()              # fresh window/ETA for the next stretch
        finally:
            bar.close()
            self_play.shutdown()
        return hist

    # ══════════════════════════════════════════════════════════════════════════
    #  Plots + arena
    # ══════════════════════════════════════════════════════════════════════════
    def plot_history(hist, title='ThompsonZero-C4 (Connect 4)'):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); fig.suptitle(title)
        ax[0, 0].plot(hist['ep'], hist['klv'], label='state KL')
        ax[0, 0].plot(hist['ep'], hist['kla'], label='action KL')
        ax[0, 0].plot(hist['ep'], hist['cons'], label='consistency KL')
        ax[0, 0].set_title('Distributional losses'); ax[0, 0].legend(fontsize=8)
        ax[0, 1].plot(hist['ep'], hist['cev'], label='state CE')
        ax[0, 1].plot(hist['ep'], hist['cea'], label='action CE')
        ax[0, 1].set_title('Outcome cross-entropy'); ax[0, 1].legend(fontsize=8)
        ax[0, 2].plot(hist['ep'], hist['cv_p'], label='state pred')
        ax[0, 2].plot(hist['ep'], hist['cv_t'], label='state tgt')
        ax[0, 2].plot(hist['ep'], hist['ca_p'], label='action pred')
        ax[0, 2].plot(hist['ep'], hist['ca_t'], label='action tgt')
        ax[0, 2].set_title('Concentration α₀'); ax[0, 2].legend(fontsize=8)
        ax[1, 0].plot(hist['ep'], hist['draw_pct'])
        ax[1, 0].set_title('self-play draws %')
        ax[1, 1].plot(hist['ep'], hist['plies'])
        ax[1, 1].set_title('game length (plies)')
        if hist['quick_ep']:
            q = np.array(hist['quick_ep'])
            w = np.array(hist['q_w'], float); d = np.array(hist['q_d'], float)
            l = np.array(hist['q_l'], float)
            ax[1, 2].plot(q, (w + 0.5 * d) / np.maximum(w + d + l, 1))
            ax[1, 2].axhline(0.5, ls='--', c='k', lw=0.8)
            ax[1, 2].set_title('quick eval score vs last ckpt')
        for a in ax.ravel():
            a.set_xlabel('ep')
        fig.tight_layout()
        return fig

    def duel(cfg, label_a, label_b, sims_a=128, sims_b=128, n_games=20, seed=0,
             game=None, device='cpu'):
        """Pit two saved benchmarks head-to-head at chosen search budgets, or
        measure how much search adds by giving one side sims=0 (search-free).
        `label` may also be 'random'.  Returns (wins_a, draws, wins_b)."""
        game = game or load_game()
        set_game(game)
        sig = (cfg.channels, cfg.num_blocks, cfg.head_ch)
        rng = np.random.default_rng(seed)
        nets = {}
        for lb in (label_a, label_b):
            nets[lb] = (None if lb == 'random'
                        else load_benchmark_net(cfg.checkpoint_dir, lb, sig))

        def mv(lb, sims, state, cache):
            net = nets[lb]
            if net is None:
                leg = state.legal_actions()
                return int(leg[rng.integers(len(leg))])
            if sims <= 0:
                return value_greedy_move(net, state, device)
            bot = cache.setdefault((lb, sims), C4MCTSBot(
                game, net, device, sims, temp=cfg.eval_temp, random_state=rng))
            return root_pick(bot.mcts_search(state), rng, thompson=False)

        wa = d = wb = 0
        cache = {}
        for i in range(n_games):
            a_side = i % 2
            state = game.new_initial_state()
            for _ in range(cfg.eval_opening_plies):
                if state.is_terminal():
                    break
                leg = state.legal_actions()
                state.apply_action(int(leg[rng.integers(len(leg))]))
            while not state.is_terminal():
                if state.current_player() == a_side:
                    state.apply_action(mv(label_a, sims_a, state, cache))
                else:
                    state.apply_action(mv(label_b, sims_b, state, cache))
            r = state.returns()[a_side]
            wa += r > 0; wb += r < 0; d += r == 0
        return int(wa), int(d), int(wb)
