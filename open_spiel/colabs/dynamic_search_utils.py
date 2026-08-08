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
  decision   P(best) >= p_stop, OR the gap    more search will not change the
             is SHOWN to be under eps          move, or will not change it in a
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

The indifference half of the decision gate is an EQUIVALENCE test, not a bare
`gap < eps` — see `indifferent`.  The bare form is the same trap from the other
side: early in training every action looks equal because nothing is known yet,
and a gap test fires everywhere, collapsing the search onto its floor exactly
when bootstrapping needs it most.  Measured before the fix: 20 simulations
against a nominal 64, on every position of an untrained run.

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

    Zero-or-one action means there is nothing to choose, so the gap is 0: the
    strongest possible form of indifference."""
    q = np.asarray(q, dtype=np.float64)
    if q.size <= 1:
        return 0.0
    p = np.partition(q, -2)
    return float(p[-1] - p[-2])


def indifferent(q, sd, eps, z=1.645, floor=1e-9):
    """Is the best action provably no better than the runner-up by more than
    `eps`?  An EQUIVALENCE test, not a failure to reject one.

    The naive form — `top_gap(q) < eps` — is the same trap as the naive
    confidence rule, from the other side.  Early in training every action looks
    equal because NOTHING is known, so a bare gap test fires on every position
    and the search collapses onto its floor exactly when bootstrapping needs it
    most: measured at 20 simulations against a nominal 64 on an untrained net,
    on every position played.

    Requiring `gap + z·sd(gap) < eps` instead asks whether the gap has been
    SHOWN to be small.  A wide belief cannot pass it, so an unsearched position
    keeps searching, and only a position whose leaves agree that the moves are
    close stops for this reason.

    Note this uses the disagreement spread, which does not shrink with n — so
    two moves whose leaves genuinely disagree a lot will never satisfy it and
    will run to the ceiling.  That is the honest answer: nothing established
    whether the choice matters."""
    q = np.asarray(q, dtype=np.float64)
    if q.size <= 1:
        return True
    sd = np.maximum(np.asarray(sd, dtype=np.float64), floor)
    order = np.argsort(q)
    i, k = int(order[-1]), int(order[-2])
    sd_gap = math.sqrt(sd[i] ** 2 + sd[k] ** 2)
    return bool((q[i] - q[k]) + z * sd_gap < eps)


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

    __slots__ = ('n', 'solved', 'q', 'sd', 'target', 'sd_post')

    def __init__(self, n, solved, q, sd, target, sd_post=None):
        self.n = int(n)
        self.solved = bool(solved)
        self.q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.sd = np.asarray(sd, dtype=np.float64).reshape(-1)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1)
        # The 1/sqrt(n) POSTERIOR spread.  Carried only so the naive rule can
        # be run as a measured control -- nothing else should read it.
        self.sd_post = (self.sd if sd_post is None
                        else np.asarray(sd_post, dtype=np.float64).reshape(-1))


# The rules, so they can be A/B'd against each other rather than argued about.
# The last two are the traps the module docstring describes, kept runnable so
# their cost is a measurement.
MODE_FULL = 'full'                 # decision AND drift, the real rule
MODE_DECISION = 'decision'         # decision gate only
MODE_DRIFT = 'drift'               # drift gate only
MODE_NAIVE_POSTERIOR = 'naive_posterior'   # P(best) on the 1/sqrt(n) posterior
MODE_NAIVE_GAP = 'naive_gap'       # bare gap < eps, no equivalence test
MODES = (MODE_FULL, MODE_DECISION, MODE_DRIFT,
         MODE_NAIVE_POSTERIOR, MODE_NAIVE_GAP)


class StopRule(object):
    """The three gates, plus the geometric drift bookkeeping.

    One instance per root: `update(probe)` is called after each wave and returns
    True when the search should stop.  It is a pure function of the probes it
    has been shown, so a test can drive it without a tree."""

    def __init__(self, min_sims, max_sims, p_stop=0.95, eps_indiff=0.02,
                 delta=0.05, patience=2, indiff_z=1.645, mode=MODE_FULL):
        self.min_sims = int(min_sims)
        self.max_sims = int(max_sims)
        self.p_stop = float(p_stop)
        self.eps_indiff = float(eps_indiff)
        self.delta = float(delta)
        self.patience = int(patience)
        self.indiff_z = float(indiff_z)
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        self._snap_n = 0
        self._snap = None
        self._streak = 0
        self._drift = float('inf')
        self.lent = 0
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

        gap = top_gap(probe.q)
        if self.mode == MODE_NAIVE_POSTERIOR:
            # The trap, run as a control: read the belief that narrows as
            # 1/sqrt(n) whatever the search found.
            pb = p_best(probe.q, probe.sd_post)
            moot = False
            decided, stable = pb >= self.p_stop, True
        elif self.mode == MODE_NAIVE_GAP:
            # The other trap: a small gap taken as indifference without asking
            # whether it was established.
            pb = p_best(probe.q, probe.sd)
            moot = gap < self.eps_indiff
            decided, stable = pb >= self.p_stop or moot, True
        else:
            pb = p_best(probe.q, probe.sd)
            moot = indifferent(probe.q, probe.sd, self.eps_indiff,
                               self.indiff_z)
            decided = pb >= self.p_stop or moot
            stable = self._drift < self.delta
            if self.mode == MODE_DECISION:
                stable = True
            elif self.mode == MODE_DRIFT:
                decided = True
        self.last = {'p_best': pb, 'gap': gap, 'drift': self._drift,
                     'moot': moot, 'decided': decided, 'stable': stable}
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

    Lending is DEBITED WHEN RESERVED, not when spent.  The self-play drivers
    hold several positions open at once, so a pool that is only debited on
    settle hands the same surplus to every position that looks at it before any
    of them finishes — measured at a 12% overspend against the baseline with
    four concurrent slots, which is exactly the confound the pool exists to
    prevent.  Reserving first makes the total spend at most `base × moves`
    however many searches are in flight.

    What the pool CAN do is spend much less, if the stop rule fires everywhere.
    That is a real result but not the same experiment, so `realized_mean()` is
    logged every eval and the reader can see which of the two happened instead
    of assuming."""

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
        self.outstanding = 0

    def reserve(self):
        """(min_sims, max_sims, lent) for one position, debiting `lent` now.

        The debit is the whole point — see the class docstring."""
        lent = max(min(self.pool, self.hard_ceiling - self.base), 0)
        self.pool -= lent
        self.outstanding += lent
        top = max(self.floor, min(self.base + lent, self.hard_ceiling))
        return self.floor, int(top), int(lent)

    def settle(self, used, lent=0):
        used = int(used)
        lent = int(lent)
        self.moves += 1
        self.spent += used
        self.outstanding -= lent
        # Return the unspent part of what this position was allowed.  max(…, 0)
        # is belt and braces: `reserve` already bounds `used` by base + lent.
        self.pool = int(min(max(self.pool + self.base + lent - used, 0),
                            self.cap))

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
                 delta=0.05, patience=2, indiff_z=1.645, mode=MODE_FULL):
        self.enabled = bool(enabled)
        self.base = int(base_sims)
        self.params = dict(p_stop=p_stop, eps_indiff=eps_indiff, delta=delta,
                           patience=patience, indiff_z=indiff_z, mode=mode)
        self.pool = (BudgetPool(base_sims, floor_frac, ceil_mult, pool_mult)
                     if self.enabled else None)
        self.rule = None
        self.cap = int(base_sims)
        self.stops = {}

    def open_position(self, base_sims=None):
        """Start one position: returns (rule_or_None, ceiling).

        The RULE is per-root and the POOL is shared, because the self-play
        drivers interleave many games in one engine — every slot needs its own
        convergence state, and they should all bank into and borrow from the
        same budget.  `begin`/`should_stop`/`end` below are the single-root
        form for the bots, which have exactly one search in flight."""
        cap = self._rebase(base_sims)
        if not self.enabled:
            return None, cap
        lo, hi, lent = self.pool.reserve()
        rule = StopRule(lo, hi, **self.params)
        rule.lent = lent
        return rule, hi

    def close_position(self, rule, used):
        if not self.enabled or self.pool is None:
            return
        self.pool.settle(used, getattr(rule, 'lent', 0))
        r = rule.reason if rule is not None else ''
        if r:
            self.stops[r] = self.stops.get(r, 0) + 1

    def _rebase(self, base_sims):
        """Adopt a new base budget, carrying the pool's running state over."""
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
                self.pool.spent, self.pool.outstanding = p.spent, p.outstanding
        return self.base if not self.enabled else self.base + self.pool.pool

    # ── single-root form, for the bots ────────────────────────────────────────
    def begin(self, base_sims=None):
        """Start a position and keep its rule internally.  Returns the ceiling."""
        self.rule, self.cap = self.open_position(base_sims)
        return self.cap

    def should_stop(self, probe):
        if not self.enabled or self.rule is None:
            return False
        return self.rule.update(probe)

    def end(self, used):
        self.close_position(self.rule, used)

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
        patience=get('ds_patience', 2),
        indiff_z=get('ds_indiff_z', 1.645),
        mode=get('ds_rule', MODE_FULL))
