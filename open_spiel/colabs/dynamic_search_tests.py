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
                       eps_indiff=0.0, delta=1e9, patience=2)
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
    rule.update(probe(8, [0.4, 0.4], [0.3, 0.3]))          # drift snapshot
    check('a dead tie stops on indifference, not on confidence',
          rule.update(probe(16, [0.4, 0.4], [0.3, 0.3])))
    check('and says so', rule.reason == 'converged')
    rule2 = ds.StopRule(min_sims=4, max_sims=10 ** 6, eps_indiff=0.02,
                        delta=1e9, patience=1)
    rule2.update(probe(8, [0.50, 0.40], [0.3, 0.3]))
    check('a gap well above the indifference band does not',
          not rule2.update(probe(16, [0.50, 0.40], [0.3, 0.3])))
    check('top_gap on one action is infinite (nothing to choose)',
          ds.top_gap([0.5]) == float('inf'))
    check('top_gap ignores the also-rans',
          abs(ds.top_gap([0.1, 0.9, 0.5, -0.2]) - 0.4) < 1e-12)


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
    check('the floor is a fraction of the base', p.allot()[0] == 25)
    check('with an empty pool the ceiling is the base', p.allot()[1] == 100)
    p.settle(40)
    check('an unspent position banks the difference', p.pool == 60)
    check('and the next position may spend it', p.allot()[1] == 160)
    for _ in range(50):
        p.settle(p.allot()[0])
    check('the surplus is capped', p.pool <= p.cap, f'{p.pool}')
    check('the ceiling never exceeds ceil_mult x base', p.allot()[1] == 400)

    # Spending the whole allotment every time is exactly break-even: a position
    # may spend at most base + pool, so the arm can never run a deficit against
    # the fixed-budget baseline.  That one-sidedness is the point -- dynamic
    # search cannot win by spending more.
    q = ds.BudgetPool(100, pool_mult=2.0)
    for _ in range(20):
        q.settle(q.allot()[1])
    check('spending the full allotment is break-even, never a deficit',
          q.pool == 0, f'{q.pool}')
    check('so the total never exceeds the fixed budget',
          q.spent <= q.base * q.moves, f'{q.spent} vs {q.base * q.moves}')
    r0 = ds.BudgetPool(100)
    for _ in range(30):
        r0.settle(r0.allot()[1] + 10 ** 6)      # a caller ignoring its ceiling
    check('and a caller that ignores its ceiling cannot corrupt the pool',
          r0.pool == 0, f'{r0.pool}')

    # Over a long mixed run the realized mean must land on the base.
    rng = np.random.default_rng(3)
    r = ds.BudgetPool(100)
    for _ in range(4000):
        lo, hi = r.allot()
        want = int(rng.choice([lo, 100, hi]))
        r.settle(int(np.clip(want, lo, hi)))
    check('the realized mean tracks the fixed budget',
          abs(r.realized_mean() - 100) < 5.0, f'{r.realized_mean():.1f}')

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

    # base_sims changes when a new game starts in the slot (fast vs full).
    d2 = ds.DynamicSearch(100, enabled=True)
    d2.begin(); d2.end(20)
    banked = d2.pool.pool
    d2.begin(300)
    check('a new base carries the pool over', d2.pool.pool == banked
          and d2.pool.base == 300, f'{d2.pool.pool}/{d2.pool.base}')
    check('and rescales the floor', d2.pool.floor == 75)


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
    test_config_kwargs()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all dynamic-search tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
