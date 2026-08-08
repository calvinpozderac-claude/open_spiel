"""Adaptive per-position search budgets.

A fixed simulation count spends the same effort on a position with one legal
reply as on a sharp tactical fork.  This module decides, per root, when more
search has stopped paying — and banks what it saves so the harder positions can
spend it.  It is deliberately engine-agnostic: nothing here imports torch,
pyspiel, or any of the three engines.  Each engine supplies a `RootProbe` and
gets the same rule.


WHY THE OBVIOUS RULE DOES NOT WORK
──────────────────────────────────
"Search until the root's belief is narrow enough" is a threshold on the visit
count wearing a disguise.  Every posterior in this project narrows as 1/√n
MECHANICALLY, whatever search finds:

  * Dirichlet, 'additive':  α₀ == n exactly, so sd ∝ 1/√n by construction.
  * Gaussian, `belief()`:   total variance divided by n, the standard error.

A confidence threshold on either is satisfied by patience alone.  Worse, it is
satisfied FASTEST in positions the network is already sure about and slowest in
the messy ones, which is close to the opposite of a useful allocation.

The same trap catches the second-most-obvious rule, "stop when the target stops
moving".  Observation n has weight 1/n, so the per-simulation drift shrinks like
1/n whether or not anything converged.


WHAT IS ACTUALLY n-ROBUST
─────────────────────────
The dispersion of the backed-up evaluations.  It does NOT go to zero with n; it
converges to how much the leaves under this position genuinely disagree.  Both
engines already expose it, because both needed it for something else:

  * Dirichlet:  AGG_MIXTURE — "concentration reflects DISAGREEMENT between the
                backed-up evaluations and is independent of how many there
                were" (observed_alpha's own docstring).
  * Gaussian:   `spread()` / `outcome_spread()` — the law of total variance
                WITHOUT the /n, added in the variance-collapse fix.

So the decision gate asks: under the disagreement belief, is the best action
separated from the rest?  If two moves' leaf evaluations genuinely overlap,
another thousand simulations will not separate them and the search should stop.
If they are separated, the search has established something real.  Neither
answer is reachable by simply waiting.

The drift gate is kept, but measured across a DOUBLING of the evidence rather
than a fixed number of simulations — "the target did not move when the evidence
doubled" is a fair question at any n, where "did not move in 20 simulations" is
not.  It is ANDed with the decision gate, so as it goes vacuous at large n the
n-robust gate is left in control.  That degradation is intentional.


THE THREE GATES
───────────────
Stop when the position is proven, or when ALL of:

  floor      n >= min_sims                     never judge on nothing
  decision   P(best) >= p_stop  OR  gap < eps  more search will not change the
                                               move, or will not change it in a
                                               way worth having
  drift      target moved < delta since n/2    more search will not change the
                                               TARGET either

held for `patience` consecutive checks, and always by n >= max_sims.


THE BUDGET POOL
───────────────
Simulations not spent on an easy position are banked and lent to hard ones.  A
position may spend at most `base + pool`, which makes the arm's total spend
bounded ABOVE by the fixed-budget baseline — so if dynamic search wins, it did
not win by using more compute.  It can still spend materially less, which would
be a different experiment, so `realized_mean()` goes into the log every eval
rather than being assumed to equal the nominal budget.

A consequence of the drift gate worth knowing: the rule cannot stop before it
has seen the target across one doubling, i.e. before the second probe.  That is
a second floor on top of `min_sims`, and it is deliberate — the first snapshot
is a measurement of nothing.
"""

import math

import numpy as np

# The value scale the gates are expressed in.  Both engines present the root's
# per-action value as v ∈ [−1, 1] (Dirichlet: p_win − p_loss; Gaussian: the
# normalised score differential), so one set of thresholds serves both.
V_SCALE = 2.0


_A = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)


def phi(z):
    """Standard normal CDF, vectorised, no scipy.

    Abramowitz & Stegun 26.2.17, |error| < 7.5e-8 — six orders below anything
    the gates read.  numpy has no erf and `np.vectorize(math.erf)` is a Python
    loop; this is called on a few hundred nodes per check, so it has to be
    array code.  scipy is optional in this project and cannot be assumed."""
    z = np.asarray(z, dtype=np.float64)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * a)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    upper = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi) * poly
    return np.where(z >= 0.0, 1.0 - upper, upper)


def _normal_quantiles(s):
    """`s` stratified standard-normal quantiles, Φ⁻¹((k+0.5)/s), by bisection.

    Runs once at import."""
    u = (np.arange(s) + 0.5) / s
    lo = np.full(s, -12.0)
    hi = np.full(s, 12.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        go = phi(mid) < u
        lo = np.where(go, mid, lo)
        hi = np.where(go, hi, mid)
    return 0.5 * (lo + hi)


# Nodes for the P(best) integral.  STRATIFIED QUANTILES, not Gauss-Hermite:
# when one action's spread is much tighter than the leader's the integrand is
# nearly a step in x, and 32-node Gauss-Hermite (a polynomial fit) was off by
# 0.021 on such cases.  Uniform strata in probability space is a Riemann sum in
# u = Φ(x), which does not care that the integrand is steep.
_Z = _normal_quantiles(512)


def p_best(q, sd, floor=1e-9):
    """P(the leading action is the best one) under independent normal beliefs.

    Condition on the leader's own draw and the rivals factorise:

        P = ∫ φ(x) Π_{j≠i} Φ((q_i + sd_i·x − q_j) / sd_j) dx

    The tempting shortcut — Π_j Φ((q_i−q_j)/√(sd_i²+sd_j²)) — is wrong by up to
    0.34 here, and wrong in the unhelpful direction: the comparisons all share
    the leader's draw, so they are positively correlated and the product
    UNDER-states.  A rule keyed to 0.95 would have gone on searching well past
    the point the decision was settled.  `test_p_best_against_monte_carlo` pins
    both the integral and that claim about the shortcut.

    Returns 1.0 for a single action: there is nothing to be uncertain between.
    """
    q = np.asarray(q, dtype=np.float64)
    sd = np.maximum(np.asarray(sd, dtype=np.float64), floor)
    if q.size <= 1:
        return 1.0
    i = int(q.argmax())
    j = np.arange(q.size) != i
    if q.size == 2:
        # One rival is a difference of two normals — closed form, exact, and
        # cheaper than the quadrature it would otherwise fall into.
        k = int(np.flatnonzero(j)[0])
        return float(phi((q[i] - q[k])
                         / math.sqrt(sd[i] ** 2 + sd[k] ** 2)))
    # (nodes, rivals): the leader's value at each node, standardised against
    # every rival.
    z = ((q[i] + sd[i] * _Z[:, None]) - q[j][None, :]) / sd[j][None, :]
    return float(phi(z).prod(1).mean())


def top_gap(q):
    """Value distance between the best action and the runner-up, in v units.

    Zero-or-one action means there is nothing to choose, which is the strongest
    possible form of indifference — reported as +inf so the gate opens."""
    q = np.asarray(q, dtype=np.float64)
    if q.size <= 1:
        return float('inf')
    p = np.partition(q, -2)
    return float(p[-1] - p[-2])


def drift(a, b):
    """How far one target vector moved from another: L1, so on a probability
    vector this is twice the total-variation distance.

    Length mismatch (the legal set cannot change, but a target's evidence set
    can) counts as maximal movement rather than raising."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        return float('inf')
    return float(np.abs(a - b).sum())


class RootProbe(object):
    """What an engine must be able to say about a root, in engine-free terms.

    `q`/`sd` are the per-action value mean and DISAGREEMENT standard deviation
    on the v ∈ [−1, 1] scale — the n-independent spread, never the 1/√n
    posterior (see the module docstring; passing the posterior silently turns
    the decision gate back into a visit-count threshold).

    `target` is any flat vector that moves when the training target moves.
    """

    __slots__ = ('n', 'solved', 'q', 'sd', 'target')

    def __init__(self, n, solved, q, sd, target):
        self.n = int(n)
        self.solved = bool(solved)
        self.q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.sd = np.asarray(sd, dtype=np.float64).reshape(-1)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1)


class StopRule(object):
    """The three gates, plus the geometric drift bookkeeping.

    One instance per root: `update(probe)` is called after each wave and returns
    True when the search should stop.  It is a pure function of the probes it
    has been shown, so a test can drive it without a tree."""

    def __init__(self, min_sims, max_sims, p_stop=0.95, eps_indiff=0.02,
                 delta=0.05, patience=2):
        self.min_sims = int(min_sims)
        self.max_sims = int(max_sims)
        self.p_stop = float(p_stop)
        self.eps_indiff = float(eps_indiff)
        self.delta = float(delta)
        self.patience = int(patience)
        self._snap_n = 0
        self._snap = None
        self._streak = 0
        self._drift = float('inf')
        self.reason = ''
        self.last = {}

    def _refresh_drift(self, probe):
        """Compare the target against the snapshot taken at half the evidence.

        The snapshot only advances once n has DOUBLED, which is what makes the
        comparison fair across n: 'the target held still while the evidence
        doubled' means the same thing at n=20 and at n=2000, where 'held still
        for 20 simulations' does not."""
        if self._snap is None:
            self._snap_n, self._snap = probe.n, probe.target.copy()
            return
        if probe.n >= 2 * max(self._snap_n, 1):
            self._drift = drift(probe.target, self._snap)
            self._snap_n, self._snap = probe.n, probe.target.copy()

    def update(self, probe):
        self._refresh_drift(probe)
        if probe.solved:
            self.reason = 'solved'
            return True
        if probe.n >= self.max_sims:
            self.reason = 'ceiling'
            return True
        if probe.n < self.min_sims:
            self._streak = 0
            return False

        pb = p_best(probe.q, probe.sd)
        gap = top_gap(probe.q)
        decided = pb >= self.p_stop or gap < self.eps_indiff
        stable = self._drift < self.delta
        self.last = {'p_best': pb, 'gap': gap, 'drift': self._drift,
                     'decided': decided, 'stable': stable}
        if decided and stable:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            self.reason = 'converged'
            return True
        return False


class BudgetPool(object):
    """Simulations saved on easy positions, lent to hard ones.

    The pool cannot go into debt, and that is structural rather than a choice:
    a position may spend at most `base + pool`, so `pool + base − used` is never
    negative.  The arm therefore spends at most what the fixed-budget baseline
    spends, and the A/B is one-sided in the safe direction — dynamic search
    winning cannot be explained by it having used more compute.

    What it CAN do is quietly spend much less, if the stop rule fires
    everywhere.  That would be a real result but it is not the same experiment,
    so `realized_mean()` is logged every eval and the reader can see which of
    the two happened instead of assuming."""

    def __init__(self, base, floor_frac=0.25, ceil_mult=4.0, pool_mult=8.0):
        if not 0.0 < floor_frac <= 1.0:
            raise ValueError('floor_frac must be in (0, 1]')
        if ceil_mult < 1.0:
            raise ValueError('ceil_mult must be at least 1')
        self.base = int(base)
        self.floor = max(1, int(round(base * floor_frac)))
        self.hard_ceiling = max(self.floor, int(round(base * ceil_mult)))
        self.cap = max(1, int(round(base * pool_mult)))
        self.pool = 0
        self.moves = 0
        self.spent = 0

    def allot(self):
        """(min_sims, max_sims) for the next position."""
        top = self.base + self.pool
        top = min(top, self.hard_ceiling)
        top = max(top, self.floor)
        return self.floor, int(top)

    def settle(self, used):
        used = int(used)
        self.moves += 1
        self.spent += used
        # max(…, 0) is belt and braces: `allot` already bounds `used` by
        # base + pool, so the sum cannot be negative unless a caller ignores the
        # ceiling it was handed.
        self.pool = int(min(max(self.pool + self.base - used, 0), self.cap))

    def realized_mean(self):
        """Simulations per move actually spent.  Logged, so 'compute-matched'
        is a measurement rather than a claim."""
        return self.spent / self.moves if self.moves else 0.0


class DynamicSearch(object):
    """One per self-play slot / bot: a pool plus the rule for the current root.

    `enabled=False` makes every method a pass-through that reports the fixed
    budget, so the call sites read the same either way and the fixed-budget arms
    keep bit-identical behaviour."""

    def __init__(self, base_sims, enabled=False, floor_frac=0.25,
                 ceil_mult=4.0, pool_mult=8.0, p_stop=0.95, eps_indiff=0.02,
                 delta=0.05, patience=2):
        self.enabled = bool(enabled)
        self.base = int(base_sims)
        self.params = dict(p_stop=p_stop, eps_indiff=eps_indiff, delta=delta,
                           patience=patience)
        self.pool = (BudgetPool(base_sims, floor_frac, ceil_mult, pool_mult)
                     if self.enabled else None)
        self.rule = None
        self.cap = int(base_sims)
        self.stops = {}

    def begin(self, base_sims=None):
        """Start a new position.  Returns the ceiling for this search."""
        if base_sims is not None and int(base_sims) != self.base:
            # fast_sims / full_sims are drawn per GAME, so the base can change
            # between positions only when a new game starts in this slot.
            self.base = int(base_sims)
            if self.enabled:
                p = self.pool
                self.pool = BudgetPool(self.base, p.floor / max(p.base, 1),
                                       p.hard_ceiling / max(p.base, 1),
                                       p.cap / max(p.base, 1))
                self.pool.pool, self.pool.moves = p.pool, p.moves
                self.pool.spent = p.spent
        if not self.enabled:
            self.cap = self.base
            self.rule = None
            return self.cap
        lo, hi = self.pool.allot()
        self.rule = StopRule(lo, hi, **self.params)
        self.cap = hi
        return hi

    def should_stop(self, probe):
        if not self.enabled or self.rule is None:
            return False
        stop = self.rule.update(probe)
        return stop

    def end(self, used):
        if self.enabled and self.pool is not None:
            self.pool.settle(used)
            r = self.rule.reason if self.rule else ''
            if r:
                self.stops[r] = self.stops.get(r, 0) + 1

    def summary(self):
        if not self.enabled or self.pool is None:
            return {}
        out = {'sims_mean': self.pool.realized_mean(),
               'sims_base': float(self.base),
               'pool': float(self.pool.pool)}
        tot = sum(self.stops.values()) or 1
        for k in ('solved', 'converged', 'ceiling'):
            out['stop_' + k] = self.stops.get(k, 0) / tot
        return out


def config_kwargs(cfg, base_sims, enabled=None):
    """Build DynamicSearch kwargs from any engine's Config or worker dict.

    Accepts either an object with attributes or a plain dict, because the three
    engines pass their settings to workers as dicts and keep them as dataclasses
    in-process."""
    def get(k, d):
        if isinstance(cfg, dict):
            return cfg.get(k, d)
        return getattr(cfg, k, d)
    return dict(
        base_sims=base_sims,
        enabled=get('dynamic_search', False) if enabled is None else enabled,
        floor_frac=get('ds_floor_frac', 0.25),
        ceil_mult=get('ds_ceil_mult', 4.0),
        pool_mult=get('ds_pool_mult', 8.0),
        p_stop=get('ds_p_stop', 0.95),
        eps_indiff=get('ds_eps_indiff', 0.02),
        delta=get('ds_delta', 0.05),
        patience=get('ds_patience', 2))
