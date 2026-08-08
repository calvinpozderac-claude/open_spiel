"""Self-tests for dynamic_search_utils.

Run:  python dynamic_search_tests.py

Pure numpy — no torch, no pyspiel, no engine.  That is the point of the module
being engine-agnostic: the rule can be driven directly and its failure modes
reproduced without running a search.
"""

import math
import sys

import numpy as np

import dynamic_search_utils as ds

_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def probe(n, q, sd, target=None, solved=False):
    q = np.asarray(q, float)
    return ds.RootProbe(n, solved, q, sd,
                        q if target is None else np.asarray(target, float))


# ══════════════════════════════════════════════════════════════════════════════
def test_p_best_against_monte_carlo():
    """The quadrature against sampling, and why the obvious shortcut is out.

    P(best) is the probability the leader wins every pairwise comparison at
    ONCE.  Those comparisons all share the leader's own draw, so they are
    positively correlated and the product of the marginals under-states — by up
    to 0.34 on these cases, which would have left a rule keyed to 0.95
    searching long past the point the decision was settled.  Conditioning on
    the leader's draw makes the rest factorise, and the remaining
    one-dimensional integral is quadrature."""
    print('\np_best: quadrature against Monte Carlo')
    rng = np.random.default_rng(0)
    worst, worst_prod = 0.0, 0.0
    prod_signed = []
    for _ in range(300):
        k = int(rng.integers(2, 9))
        q = rng.normal(0, 0.3, k)
        sd = rng.uniform(0.02, 0.4, k)
        mc = (rng.normal(q, sd, (40000, k)).argmax(1) == q.argmax()).mean()
        worst = max(worst, abs(ds.p_best(q, sd) - mc))
        # the shortcut this replaced
        i = int(q.argmax()); j = np.arange(k) != i
        pr = float(np.prod(ds.phi((q[i] - q[j])
                                  / np.sqrt(sd[i] ** 2 + sd[j] ** 2))))
        worst_prod = max(worst_prod, abs(pr - mc))
        prod_signed.append(pr - mc)
    check('the quadrature is within Monte Carlo error of the truth',
          worst < 0.01, f'{worst:.4f}')
    check('phi matches math.erf', max(
        abs(float(ds.phi(z)) - 0.5 * (1 + math.erf(z / math.sqrt(2.0))))
        for z in np.linspace(-8, 8, 2001)) < 1e-7)
    check('the product of the marginals is not', worst_prod > 0.2,
          f'{worst_prod:.4f}')
    check('and it under-states, which would have over-searched',
          np.mean(prod_signed) < 0, f'{np.mean(prod_signed):+.4f}')

    check('two actions are exact', abs(ds.p_best([0.5, 0.0], [0.1, 0.1])
                                       - ds.phi(0.5 / math.sqrt(0.02))) < 1e-9)
    check('a single action is certain', ds.p_best([0.3], [0.9]) == 1.0)
    check('no actions is certain', ds.p_best([], []) == 1.0)
    check('a tie is a coin flip', abs(ds.p_best([0.2, 0.2], [0.1, 0.1]) - 0.5)
          < 1e-9)
    check('a wider belief lowers it',
          ds.p_best([0.4, 0.0], [0.5, 0.5]) < ds.p_best([0.4, 0.0], [0.05, 0.05]))
    check('zero spread is not a divide by zero',
          np.isfinite(ds.p_best([0.4, 0.0], [0.0, 0.0])))


def test_the_pitfall():
    """THE regression test: a rule keyed on the posterior is a visit counter.

    Both engines' posteriors narrow as 1/√n whatever search finds, so a
    confidence threshold on one is satisfied by patience alone.  Here two
    actions are genuinely tied and the posterior is fed in; the naive rule fires
    on schedule and would fire at the same n no matter what the position was.
    The disagreement spread does not narrow, and the real rule keeps searching
    — which is correct, because no amount of search separates a tie."""
    print('\nThe pitfall: a posterior threshold is a visit-count threshold')
    q = np.array([0.30, 0.30])           # a genuine tie
    spread = np.array([0.25, 0.25])      # how much the leaves disagree
    naive_fires_at = None
    for n in range(1, 4001):
        post = spread / math.sqrt(n)     # the 1/sqrt(n) posterior
        edge = q + np.array([0.02, 0.0])
        if naive_fires_at is None and ds.p_best(edge, post) >= 0.95:
            naive_fires_at = n
    check('a posterior-based rule fires on a dead tie', naive_fires_at is not None,
          f'{naive_fires_at}')
    # ...and it fires at the same n for a position twice as contested, because
    # the only thing it is really reading is n.
    a = None
    b = None
    for n in range(1, 4001):
        if a is None and ds.p_best([0.32, 0.30], spread / math.sqrt(n)) >= 0.95:
            a = n
        if b is None and ds.p_best([0.32, 0.30], 2 * spread / math.sqrt(n)) >= 0.95:
            b = n
    check('and 4x the disagreement only costs it a factor of 4 in n, never a '
          'different answer',
          a is not None and b is not None and abs(b / a - 4.0) < 0.02,
          f'{a} vs {b}')

    # The real rule reads the disagreement, which does not move with n.
    rule = ds.StopRule(min_sims=8, max_sims=10 ** 6, p_stop=0.95,
                       eps_indiff=0.0, delta=1e9, patience=2)  # gate 2 off
    stopped = None
    for n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        if rule.update(probe(n, q + np.array([0.005, 0.0]), spread)):
            stopped = n
            break
    check('the disagreement-based rule does NOT stop on a tie it cannot resolve',
          stopped is None, f'stopped at {stopped}')

    # But it stops at once when the actions really are separated, at ANY n.
    # Separated: it stops as soon as it CAN, which is two probes after the
    # first -- one to take the drift snapshot, one to compare across the
    # doubling, and then `patience` of them.
    rule2 = ds.StopRule(min_sims=8, max_sims=10 ** 6, delta=1e9, patience=1)
    fires = [rule2.update(probe(n, [0.9, 0.0], [0.05, 0.05]))
             for n in (8, 16, 32)]
    check('and it stops on a separated position as soon as a doubling exists',
          fires == [False, True, True], f'{fires}')


def test_indifference():
    """A tie must not search forever — it must stop for the OTHER reason."""
    print('\nIndifference: an unresolvable tie is still a reason to stop')
    rule = ds.StopRule(min_sims=4, max_sims=10 ** 6, eps_indiff=0.02,
                       delta=1e9, patience=1)
    rule.update(probe(8, [0.4, 0.4], [0.003, 0.003]))      # drift snapshot
    check('a dead tie stops on indifference, not on confidence',
          rule.update(probe(16, [0.4, 0.4], [0.003, 0.003])))
    check('and says so', rule.reason == 'converged')
    # A gap above the band with tight beliefs also stops -- but for the OTHER
    # reason, and the rule records which.
    rule2 = ds.StopRule(min_sims=4, max_sims=10 ** 6, eps_indiff=0.02,
                        delta=1e9, patience=1)
    rule2.update(probe(8, [0.50, 0.40], [0.003, 0.003]))
    check('a gap above the band stops on confidence instead',
          rule2.update(probe(16, [0.50, 0.40], [0.003, 0.003]))
          and rule2.last['moot'] is False and rule2.last['p_best'] > 0.99,
          f'{rule2.last}')
    # Above the band AND unresolved: neither gate, so it keeps going.
    rule3 = ds.StopRule(min_sims=4, max_sims=10 ** 6, eps_indiff=0.02,
                        delta=1e9, patience=1)
    rule3.update(probe(8, [0.50, 0.40], [0.4, 0.4]))
    check('a gap above the band with wide beliefs keeps searching',
          not rule3.update(probe(16, [0.50, 0.40], [0.4, 0.4])),
          f'{rule3.last}')
    check('one action has no gap at all', ds.top_gap([0.5]) == 0.0)
    check('top_gap ignores the also-rans',
          abs(ds.top_gap([0.1, 0.9, 0.5, -0.2]) - 0.4) < 1e-12)

    # The equivalence test: a small gap is only indifference once it is SHOWN.
    check('a tight tie is indifferent',
          ds.indifferent([0.4, 0.4], [0.002, 0.002], 0.02))
    check('the same gap with wide beliefs is NOT',
          not ds.indifferent([0.4, 0.4], [0.3, 0.3], 0.02))
    check('a wide gap is never indifferent',
          not ds.indifferent([0.9, 0.0], [0.002, 0.002], 0.02))
    check('one action is trivially indifferent',
          ds.indifferent([0.4], [0.9], 0.02))
    check('it reads the top TWO, not the extremes',
          ds.indifferent([0.4, 0.4, -0.9], [0.002, 0.002, 0.002], 0.02))

    # This is what collapsed the search to its floor on an untrained net: every
    # action equal, nothing known, and a bare gap test firing on all of it.
    flat_q = np.full(8, 0.01)
    check('an untrained-looking root is not indifferent under the real test',
          not ds.indifferent(flat_q, np.full(8, 0.35), 0.02))
    check('…where a bare gap test would have fired',
          ds.top_gap(flat_q) < 0.02)



def test_drift_is_measured_across_a_doubling():
    """Drift must be read at geometric n, or it too becomes a visit counter."""
    print('\nDrift: compared across a doubling of the evidence')
    rule = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=0.05, patience=1)
    rule.update(probe(10, [0.5, 0.0], [0.4, 0.4], target=[1.0, 0.0]))
    check('the first probe only takes a snapshot',
          rule._snap_n == 10 and rule._drift == float('inf'))
    rule.update(probe(15, [0.5, 0.0], [0.4, 0.4], target=[0.0, 1.0]))
    check('a probe before the doubling does not re-measure',
          rule._snap_n == 10 and rule._drift == float('inf'))
    rule.update(probe(20, [0.5, 0.0], [0.4, 0.4], target=[0.0, 1.0]))
    check('the doubling does', rule._snap_n == 20 and rule._drift == 2.0,
          f'{rule._drift}')
    rule.update(probe(40, [0.5, 0.0], [0.4, 0.4], target=[0.0, 1.0]))
    check('a still target reads zero drift', rule._drift == 0.0)

    check('drift is L1', abs(ds.drift([0.5, 0.5], [0.25, 0.75]) - 0.5) < 1e-12)
    check('a changed length is maximal drift, not an exception',
          ds.drift([0.5, 0.5], [1.0]) == float('inf'))
    check('an empty target likewise', ds.drift([], []) == float('inf'))


def test_gates_are_anded():
    """Both purposes of search must be exhausted before it stops."""
    print('\nBoth gates must open')
    # Decided but still moving: the move is settled, the TARGET is not.
    r = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=0.05, patience=1)
    r.update(probe(10, [0.9, 0.0], [0.02, 0.02], target=[1.0, 0.0]))
    moving = r.update(probe(20, [0.9, 0.0], [0.02, 0.02], target=[0.0, 1.0]))
    check('a settled decision with a moving target keeps searching', not moving,
          f'{r.last}')
    # Still moving but undecided likewise.
    r2 = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=0.05, patience=1)
    r2.update(probe(10, [0.30, 0.24], [0.5, 0.5], target=[1.0, 0.0]))
    check('an unsettled decision with a still target keeps searching',
          not r2.update(probe(20, [0.30, 0.24], [0.5, 0.5], target=[1.0, 0.0])))
    # Both.
    r3 = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=0.05, patience=1)
    r3.update(probe(10, [0.9, 0.0], [0.02, 0.02], target=[1.0, 0.0]))
    check('both open together and it stops',
          r3.update(probe(20, [0.9, 0.0], [0.02, 0.02], target=[1.0, 0.0])))


def test_floor_ceiling_patience_and_solved():
    print('\nFloor, ceiling, patience, proof')
    r = ds.StopRule(min_sims=50, max_sims=200, delta=1e9, patience=1)
    r.update(probe(10, [0.9, 0.0], [0.01, 0.01]))
    check('nothing stops below the floor',
          not r.update(probe(49, [0.9, 0.0], [0.01, 0.01])))
    check('…and it does at the floor',
          r.update(probe(50, [0.9, 0.0], [0.01, 0.01])))

    r2 = ds.StopRule(min_sims=5, max_sims=100, p_stop=1.1, eps_indiff=-1,
                     delta=1e9, patience=1)   # gates that can never open
    check('an unresolvable position runs to the ceiling',
          not r2.update(probe(99, [0.0, 0.0], [1.0, 1.0]))
          and r2.update(probe(100, [0.0, 0.0], [1.0, 1.0])))
    check('and says so', r2.reason == 'ceiling')

    def good(n):
        return probe(n, [0.9, 0.0], [0.01, 0.01])
    r3 = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=1e9, patience=3)
    check('patience needs consecutive checks',
          [r3.update(good(n)) for n in (10, 20, 40, 80, 160)]
          == [False, False, False, True, True])
    r4 = ds.StopRule(min_sims=1, max_sims=10 ** 6, delta=1e9, patience=3)
    r4.update(good(10)); r4.update(good(20))
    check('a good check builds the streak', r4._streak == 1)
    # Genuinely undecided: a gap well outside the indifference band with
    # beliefs too wide to separate it.  (A dead TIE would open the gate by
    # indifference, which is correct and would not reset anything.)
    r4.update(probe(40, [0.30, 0.10], [1.0, 1.0]))
    check('and a genuinely undecided check resets it', r4._streak == 0,
          f'{r4.last}')

    r5 = ds.StopRule(min_sims=500, max_sims=10 ** 6, delta=1e9, patience=5)
    check('a proven root stops at once, floor and patience notwithstanding',
          r5.update(probe(3, [0.0], [1.0], solved=True)))
    check('and says so', r5.reason == 'solved')


def test_budget_pool_is_compute_matched():
    """The pool must hold the MEAN at the fixed budget, not below it."""
    print('\nBudget pool')
    p = ds.BudgetPool(100, floor_frac=0.25, ceil_mult=4.0, pool_mult=8.0)
    lo, hi, lent = p.reserve()
    check('the floor is a fraction of the base', lo == 25)
    check('with an empty pool the ceiling is the base', hi == 100 and lent == 0)
    p.settle(40, lent)
    check('an unspent position banks the difference', p.pool == 60)
    lo, hi, lent = p.reserve()
    check('and the next position may spend it', hi == 160 and lent == 60)
    p.settle(lo, lent)
    for _ in range(50):
        a, b, c = p.reserve()
        p.settle(a, c)
    check('the surplus is capped', p.pool <= p.cap, f'{p.pool}')
    check('the ceiling never exceeds ceil_mult x base', p.reserve()[1] == 400)

    # Reserving DEBITS.  Without that, concurrent positions each see the whole
    # surplus and the arm outspends the baseline it is compared against --
    # measured at +12% with four slots in flight before this was fixed.
    c = ds.BudgetPool(64)
    rng = np.random.default_rng(0)
    flight, tot, moves = [], 0, 0
    for _ in range(4000):
        if len(flight) < 4:
            flight.append(c.reserve())
        else:
            lo, hi, lent = flight.pop(0)
            used = int(rng.integers(lo, hi + 1))
            c.settle(used, lent)
            tot += used; moves += 1
    check('four concurrent positions cannot outspend the fixed budget',
          tot <= moves * c.base, f'{tot / moves:.2f} vs {c.base}')
    check('and the pool books stay balanced',
          c.outstanding == sum(f[2] for f in flight), f'{c.outstanding}')

    # Spending the whole allotment every time is exactly break-even.
    q = ds.BudgetPool(100, pool_mult=2.0)
    for _ in range(20):
        a, b, l = q.reserve()
        q.settle(b, l)
    check('spending the full allotment is break-even, never a deficit',
          q.pool == 0, f'{q.pool}')
    check('so the total never exceeds the fixed budget',
          q.spent <= q.base * q.moves, f'{q.spent} vs {q.base * q.moves}')
    r0 = ds.BudgetPool(100)
    for _ in range(30):
        a, b, l = r0.reserve()
        r0.settle(b + 10 ** 6, l)          # a caller ignoring its ceiling
    check('and a caller that ignores its ceiling cannot corrupt the pool',
          r0.pool == 0, f'{r0.pool}')

    # Over a long mixed run the realized mean must land on the base.
    rng = np.random.default_rng(3)
    r = ds.BudgetPool(100)
    for _ in range(4000):
        lo, hi, lent = r.reserve()
        r.settle(int(np.clip(rng.choice([lo, 100, hi]), lo, hi)), lent)
    check('the realized mean tracks the fixed budget',
          abs(r.realized_mean() - 100) < 5.0, f'{r.realized_mean():.1f}')

    # A mixed fast/full budget: the nominal is per POSITION, not per pool.
    # With one pool-wide base, surplus banked by a 150-simulation game was spent
    # by a 50-simulation one and the total stopped being bounded -- measured at
    # 82.8 simulations per move against a fixed arm's 75.
    m = ds.BudgetPool(150)
    rng2 = np.random.default_rng(11)
    tot = nom_tot = 0
    for _ in range(3000):
        nom = int(rng2.choice([50, 150], p=[0.75, 0.25]))
        lo, hi, lent = m.reserve(nom)
        used = int(rng2.integers(lo, hi + 1))
        m.settle(used, lent, nom)
        tot += used; nom_tot += nom
    check('a mixed fast/full budget stays bounded by the fixed arm',
          tot <= nom_tot, f'{tot} vs {nom_tot}')
    check('and the summary reports the nominal it was matched against',
          abs(m.nominal_mean() - nom_tot / m.moves) < 1e-9)
    check('the floor and ceiling scale with the POSITION, not the pool',
          ds.BudgetPool(150).reserve(50)[1] == 50
          and ds.BudgetPool(150).reserve(50)[0] == 12)
    try:
        ds.BudgetPool(100, floor_frac=0.0)
        check('a zero floor is rejected', False)
    except ValueError:
        check('a zero floor is rejected', True)
    try:
        ds.BudgetPool(100, ceil_mult=0.5)
        check('a ceiling below the base is rejected', False)
    except ValueError:
        check('a ceiling below the base is rejected', True)


def test_disabled_is_a_passthrough():
    """The fixed-budget arms must be bit-identical with the toggle off."""
    print('\nDisabled: the fixed-budget path is untouched')
    d = ds.DynamicSearch(300, enabled=False)
    check('begin returns the fixed budget', d.begin() == 300)
    check('should_stop never fires',
          not d.should_stop(probe(1, [0.9, 0.0], [0.001, 0.001], solved=True)))
    d.end(300)
    check('no pool is kept', d.pool is None and d.summary() == {})


def test_dynamic_search_end_to_end():
    print('\nDynamicSearch over a run of positions')
    d = ds.DynamicSearch(100, enabled=True, patience=1, delta=0.05)
    # An easy position: one move dominates and the target is not moving.
    cap = d.begin()
    check('the first position is capped at the base', cap == 100)
    used = 0
    for n in (25, 50, 100):
        used = n
        if d.should_stop(probe(n, [0.9, 0.0], [0.02, 0.02], target=[1.0, 0.0])):
            break
    d.end(used)
    check('an easy position stops early', used < 100, f'{used}')
    check('and banks what it saved', d.pool.pool == 100 - used)
    check('so the next position may search deeper', d.begin() > 100)
    check('the summary reports the realized mean, not the nominal one',
          abs(d.summary()['sims_mean'] - used) < 1e-9)
    check('and how the searches ended', d.summary()['stop_converged'] == 1.0)

    # fast_sims vs full_sims are drawn per GAME, so a position's nominal budget
    # changes under the SAME pool.  The pool no longer rebases -- each position
    # is accounted against its own nominal, which is what keeps a mixed budget
    # bounded by the fixed arm's spend.
    d2 = ds.DynamicSearch(150, enabled=True)
    d2.begin(50); d2.end(20)
    banked = d2.pool.pool
    check('a cheap position banks against ITS nominal, not the pool base',
          banked == 30, f'{banked}')
    cap = d2.begin(150)
    check('an expensive position may borrow that surplus', cap == 180, f'{cap}')
    check('and the floor scales with the position',
          d2.rule.min_sims == 38, f'{d2.rule.min_sims}')
    d2.end(180)
    check('the nominal the arm is matched against is what it drew',
          abs(d2.summary()['sims_base'] - 100.0) < 1e-9,
          f"{d2.summary()['sims_base']}")
    check('and the realized mean is what it spent',
          abs(d2.summary()['sims_mean'] - 100.0) < 1e-9)


def test_config_kwargs():
    print('\nConfig plumbing')
    class C(object):
        dynamic_search = True
        ds_p_stop = 0.9
    k = ds.config_kwargs(C(), 250)
    check('reads an object config', k['enabled'] and k['p_stop'] == 0.9)
    check('and defaults the rest', k['patience'] == 2 and k['delta'] == 0.05)
    check('base_sims comes from the caller', k['base_sims'] == 250)
    k2 = ds.config_kwargs({'dynamic_search': True, 'ds_delta': 0.2}, 100)
    check('reads a worker dict too', k2['enabled'] and k2['delta'] == 0.2)
    check('a config without the field is disabled',
          not ds.config_kwargs({}, 100)['enabled'])
    check('every kwarg is one DynamicSearch accepts',
          ds.DynamicSearch(**k2) is not None)



def test_multi_slot_form():
    """Per-root rules over one shared pool: what the self-play drivers need.

    The drivers interleave many games in one engine, so each slot must keep its
    own convergence state while all of them bank into and borrow from the same
    budget."""
    print('\nMulti-slot: per-root rules, one shared pool')
    d = ds.DynamicSearch(100, enabled=True, patience=1, delta=0.05)
    ra, capa = d.open_position()
    rb, capb = d.open_position()
    check('two positions get independent rules', ra is not rb)
    check('and the same ceiling from the same pool', capa == capb == 100)
    ra.update(probe(20, [0.9, 0.0], [0.02, 0.02], target=[1.0, 0.0]))
    check('advancing one leaves the other untouched',
          rb._snap is None and ra._snap is not None)
    d.close_position(ra, 40)
    check('the pool banks from whichever finishes first', d.pool.pool == 60)
    _, capc = d.open_position()
    check('and lends it to the next', capc == 160)
    d.close_position(rb, 100)
    check('both settle into the same pool', d.pool.moves == 2)

    off = ds.DynamicSearch(100, enabled=False)
    rule, cap = off.open_position()
    check('disabled hands back no rule and the fixed budget',
          rule is None and cap == 100)
    off.close_position(None, 100)
    check('and closing is a no-op', off.summary() == {})


def main():
    test_p_best_against_monte_carlo()
    test_the_pitfall()
    test_indifference()
    test_drift_is_measured_across_a_doubling()
    test_gates_are_anded()
    test_floor_ceiling_patience_and_solved()
    test_budget_pool_is_compute_matched()
    test_disabled_is_a_passthrough()
    test_dynamic_search_end_to_end()
    test_multi_slot_form()
    test_config_kwargs()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all dynamic-search tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
