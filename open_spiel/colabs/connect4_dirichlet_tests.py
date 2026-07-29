"""Self-tests for connect4_dirichlet_utils.

Run:  python connect4_dirichlet_tests.py

The tree/target half needs only numpy (it runs without torch or pyspiel, against
a tiny built-in mock game).  The network/loss half is skipped automatically if
torch is missing, and the end-to-end self-play half if pyspiel is missing.
"""

import os
import sys
import numpy as np

import connect4_dirichlet_utils as c4

RNG = np.random.default_rng(0)
_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def close(a, b, tol):
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol))


# ══════════════════════════════════════════════════════════════════════════════
#  Dirichlet primitives + the three evidence-collapse rules
# ══════════════════════════════════════════════════════════════════════════════
def acc_of(alphas):
    """Build an accumulator by folding a list of Dirichlets through _acc_add."""
    acc = c4._new_acc()
    for a in alphas:
        c4._acc_add(acc, c4._payload(a)[0])
    return acc


def test_primitives():
    print('\nDirichlet primitives')
    a = np.array([3.0, 2.0, 5.0])
    check('dir_mean sums to 1', close(c4.dir_mean(a).sum(), 1.0, 1e-12))
    check('dir_value = m_w - m_l', close(c4.dir_value(a), 0.3 - 0.5, 1e-12))
    check('flip swaps win/loss', close(c4.flip_alpha(a), [5.0, 2.0, 3.0], 0))

    # Monte-Carlo the closed-form (E[v], Var[v]).
    ev, var = c4.dir_value_mean_var(a)
    x = RNG.dirichlet(a, size=400_000)
    v = x[:, 0] - x[:, 2]
    check('E[v] vs Monte Carlo', close(ev, v.mean(), 3e-3), f'{ev} vs {v.mean()}')
    check('Var[v] vs Monte Carlo', close(var, v.var(), 3e-3),
          f'{var} vs {v.var()}')


def test_single_component_roundtrip():
    print('\nSingle-observation round-trip')
    a = np.array([4.0, 1.0, 3.0])
    acc = acc_of([a])
    for mode in c4.AGGREGATIONS:
        got = c4.observed_alpha(acc, mode)
        if mode in (c4.AGG_ADDITIVE, c4.AGG_ADDITIVE_MLE):
            # 'additive' deliberately discards the leaf's concentration: one
            # observation is one unit of evidence, so it returns the MEAN.
            # 'additive_mle' falls back to it below 3 observations, where the
            # Dirichlet MLE is degenerate.
            check(f'{mode}: n=1 returns the mean, total 1',
                  close(got, c4.dir_mean(a), 1e-9) and abs(got.sum() - 1) < 1e-9,
                  f'{got}')
        else:
            check(f'{mode}: n=1 returns the observation itself',
                  close(got, a, 1e-9), f'{got}')


def test_identical_observations():
    print('\nn identical observations — how concentration scales')
    a = np.array([4.0, 1.0, 3.0])
    a0 = a.sum()
    for n in (1, 4, 16, 64):
        acc = acc_of([a] * n)
        mix = c4.observed_alpha(acc, c4.AGG_MIXTURE)
        mean = c4.observed_alpha(acc, c4.AGG_MEAN)
        summ = c4.observed_alpha(acc, c4.AGG_SUM)
        add = c4.observed_alpha(acc, c4.AGG_ADDITIVE)
        check(f'mixture stays at a0 (n={n})', close(mix.sum(), a0, 1e-6),
              f'{mix.sum()}')
        check(f'mean grows as n(a0+1)-1 (n={n})',
              close(mean.sum(), n * (a0 + 1) - 1, 1e-6), f'{mean.sum()}')
        check(f'sum grows as n*a0 (n={n})', close(summ.sum(), n * a0, 1e-6),
              f'{summ.sum()}')
        check(f'additive a0 == visit count (n={n})', close(add.sum(), n, 1e-9),
              f'{add.sum()}')
        for nm, got in (('mixture', mix), ('mean', mean), ('sum', summ),
                        ('additive', add)):
            check(f'{nm} preserves the mean (n={n})',
                  close(c4.dir_mean(got), c4.dir_mean(a), 1e-9))


def test_mixture_against_monte_carlo():
    print('\nMixture collapse vs Monte Carlo (disagreeing observations)')
    alphas = [np.array([8.0, 1.0, 1.0]), np.array([1.0, 1.0, 8.0]),
              np.array([2.0, 5.0, 3.0])]
    acc = acc_of(alphas)
    got = c4.observed_alpha(acc, c4.AGG_MIXTURE)
    # Sample the mixture: pick a component uniformly, then draw from it.
    n = 300_000
    pick = RNG.integers(3, size=n)
    x = np.empty((n, 3))
    for i, a in enumerate(alphas):
        m = pick == i
        x[m] = RNG.dirichlet(a, size=int(m.sum()))
    check('mixture mean matched', close(c4.dir_mean(got), x.mean(0), 4e-3),
          f'{c4.dir_mean(got)} vs {x.mean(0)}')
    # Total variance is what the collapse matches (a Dirichlet has one dof).
    tv_target = x.var(0).sum()
    m = c4.dir_mean(got)
    tv_got = (m * (1 - m)).sum() / (got.sum() + 1.0)
    check('mixture total variance matched', close(tv_got, tv_target, 5e-3),
          f'{tv_got} vs {tv_target}')
    check('agrees with moment_match_mixture',
          close(got, c4.moment_match_mixture(np.ones(3), alphas), 1e-6))


def test_mean_against_monte_carlo():
    print('\nSample-mean collapse vs Monte Carlo')
    alphas = [np.array([8.0, 1.0, 1.0]), np.array([1.0, 1.0, 8.0]),
              np.array([2.0, 5.0, 3.0])]
    acc = acc_of(alphas)
    got = c4.observed_alpha(acc, c4.AGG_MEAN)
    n = 300_000
    x = sum(RNG.dirichlet(a, size=n) for a in alphas) / 3.0
    check('mean matched', close(c4.dir_mean(got), x.mean(0), 4e-3))
    tv_target = x.var(0).sum()
    m = c4.dir_mean(got)
    tv_got = (m * (1 - m)).sum() / (got.sum() + 1.0)
    check('total variance of the AVERAGE matched',
          close(tv_got, tv_target, 5e-3), f'{tv_got} vs {tv_target}')


def test_additive_rule():
    print('\nAdditive backup: alpha0 IS the visit count')
    rng = np.random.default_rng(11)
    # alpha0 == n for arbitrary, differing leaf beliefs and concentrations.
    for n in (1, 3, 17, 240):
        alphas = [rng.dirichlet(np.ones(3)) * rng.uniform(0.05, 500)
                  for _ in range(n)]
        got = c4.observed_alpha(acc_of(alphas), c4.AGG_ADDITIVE)
        check(f'a0 == n regardless of leaf concentration (n={n})',
              abs(got.sum() - n) < 1e-9, f'{got.sum()} vs {n}')
        want = np.mean([a / a.sum() for a in alphas], axis=0)
        check(f'mean == average of leaf means (n={n})',
              close(c4.dir_mean(got), want, 1e-9))

    # The leaf's own confidence is discarded: same means, wildly different
    # concentrations, identical result.
    m = [np.array([0.7, 0.1, 0.2]), np.array([0.1, 0.1, 0.8])]
    weak = acc_of([x * 0.3 for x in m])
    strong = acc_of([x * 900.0 for x in m])
    check('leaf concentration does not affect the additive belief',
          close(c4.observed_alpha(weak, c4.AGG_ADDITIVE),
                c4.observed_alpha(strong, c4.AGG_ADDITIVE), 1e-9))

    # A proven-terminal spike contributes ONE unit, not TERMINAL_CONC units.
    spike = c4._SPIKE[c4._WIN]
    one = c4.observed_alpha(acc_of([spike]), c4.AGG_ADDITIVE)
    check('a terminal spike is worth one visit',
          abs(one.sum() - 1.0) < 1e-9 and one[c4._WIN] > 0.99, f'{one}')
    mixed = c4.observed_alpha(
        acc_of([spike] + [np.array([0.1, 0.1, 0.8]) * 40] * 3),
        c4.AGG_ADDITIVE)
    check('…so three ordinary losing visits outweigh one proven win',
          c4.dir_value(mixed) < 0, f'{c4.dir_value(mixed):.3f}')

    # Spread anneals as 1/sqrt(n+1) with agreeing leaves -- exactly, since
    # alpha0 == n and Var[v] carries the Dirichlet's 1/(alpha0+1).  (The
    # asymptotic 1/sqrt(n) only holds for large n: at n=4->16 the true ratio is
    # sqrt(17/5) = 1.844, not 2.)
    scaled = []
    for n in (4, 16, 64, 256):
        a = c4.observed_alpha(acc_of([np.array([0.8, 0.08, 0.12]) * 40] * n),
                              c4.AGG_ADDITIVE)
        _ev, var = c4.dir_value_mean_var(a)
        scaled.append(np.sqrt(var) * np.sqrt(n + 1))
    check('sd * sqrt(n+1) is constant across n',
          max(scaled) - min(scaled) < 1e-9,
          f'{[round(x, 6) for x in scaled]}')
    sd4, sd256 = scaled[0] / np.sqrt(5), scaled[-1] / np.sqrt(257)
    check('64x the visits shrinks sd by exactly sqrt(257/5) = 7.17x',
          abs(sd4 / sd256 - np.sqrt(257 / 5)) < 1e-9,
          f'{sd4 / sd256:.4f}')

    # Conflicting leaves stay less certain than agreeing ones at equal n.
    agree = c4.observed_alpha(
        acc_of([np.array([0.8, 0.08, 0.12]) * 40] * 8), c4.AGG_ADDITIVE)
    split = c4.observed_alpha(
        acc_of([np.array([0.9, 0.05, 0.05]) * 40] * 4
               + [np.array([0.05, 0.05, 0.9]) * 40] * 4), c4.AGG_ADDITIVE)
    check('disagreement keeps more spread at the same visit count',
          c4.dir_value_mean_var(split)[1] > c4.dir_value_mean_var(agree)[1],
          f'{c4.dir_value_mean_var(split)[1]:.4f} vs '
          f'{c4.dir_value_mean_var(agree)[1]:.4f}')


def test_additive_mle():
    print('\nadditive_mle: Dirichlet MLE of the backed-up leaf means')
    rng = np.random.default_rng(3)

    def mle_acc(xs):
        c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE,
                      target_agg=c4.AGG_ADDITIVE_MLE)
        a = c4._new_acc()
        for x in xs:
            c4._acc_add(a, c4._payload(np.asarray(x, float))[0])
        return a

    # Scalar special functions, against scipy where it is available.
    try:
        from scipy.special import digamma, polygamma
        xs = np.exp(np.linspace(np.log(1e-3), np.log(500), 500))
        e1 = max(abs(c4._digamma(float(x)) - digamma(x)) for x in xs)
        e2 = max(abs(c4._trigamma(float(x)) - polygamma(1, x)) / polygamma(1, x)
                 for x in xs)
        e3 = max(abs(c4._inv_digamma(digamma(x)) - x) / x for x in xs)
        check('digamma matches scipy', e1 < 1e-8, f'{e1:.2e}')
        check('trigamma matches scipy', e2 < 1e-7, f'{e2:.2e}')
        check('inverse_digamma inverts digamma (speed-tuned)', e3 < 2e-2,
              f'{e3:.2e}')
    except ImportError:
        print('  [skip] scipy not installed — special-function accuracy skipped')
    # Self-consistency needs no scipy: psi(psi^-1(y)) == y.
    err = max(abs(c4._digamma(c4._inv_digamma(y)) - y)
              for y in np.linspace(-8, 6, 400))
    check('psi(psi^-1(y)) round-trips', err < 5e-2, f'{err:.2e}')

    # Recovers a Dirichlet it was sampled from.
    for true, tol in (([2., 1., 1.], 0.35), ([30., 4., 6.], 0.35)):
        got = c4.observed_alpha(mle_acc(rng.dirichlet(true, size=400)),
                                c4.AGG_ADDITIVE_MLE)
        rel = abs(got.sum() - sum(true)) / sum(true)
        check(f'recovers true a0={sum(true):.0f} within {tol:.0%}', rel < tol,
              f'got {got.sum():.2f}')

    # alpha0 tracks min(n, dispersion): it grows while data are scarce and
    # saturates at the population value rather than at the visit count.
    wide = [c4.observed_alpha(mle_acc(rng.dirichlet([2.1, .3, .6], size=n)),
                              c4.AGG_ADDITIVE_MLE).sum() for n in (16, 64, 256)]
    check('saturates for a dispersed population (does not track n)',
          max(wide) < 8.0, f'{[round(v, 2) for v in wide]}')
    tight = [c4.observed_alpha(mle_acc(rng.dirichlet([70., 10., 20.], size=n)),
                               c4.AGG_ADDITIVE_MLE).sum() for n in (16, 256)]
    check('grows toward the population value when leaves agree',
          tight[1] > tight[0], f'{[round(v, 2) for v in tight]}')
    check('a concentrated population reads far above a dispersed one',
          tight[1] > 4 * max(wide), f'{tight[1]:.1f} vs {max(wide):.1f}')

    # Below 3 observations it falls back to 'additive'.
    for n in (1, 2):
        xs = [np.array([0.7, 0.1, 0.2]) * 40] * n
        got = c4.observed_alpha(mle_acc(xs), c4.AGG_ADDITIVE_MLE)
        check(f'n={n} falls back to additive (a0 == n)',
              abs(got.sum() - n) < 1e-9, f'{got.sum()}')

    # A zero component must not poison the node: log(0) would be -inf forever.
    got = c4.observed_alpha(mle_acc([np.array([1.0, 0.0, 0.0])] * 8
                                    + [np.array([0.3, 0.3, 0.4])] * 8),
                            c4.AGG_ADDITIVE_MLE)
    check('a zero component is clamped, not fatal',
          bool(np.all(np.isfinite(got))) and bool(np.all(got > 0)), f'{got}')

    # With no MLE rule selected the log-sums are not maintained; asking for the
    # rule anyway must degrade to 'additive' rather than return the [1,1,1] seed.
    c4.set_search(search_agg=c4.AGG_ADDITIVE, target_agg=c4.AGG_ADDITIVE)
    plain = c4._new_acc()
    for _ in range(8):
        c4._acc_add(plain, c4._payload(np.array([0.7, 0.1, 0.2]))[0])
    got = c4.observed_alpha(plain, c4.AGG_ADDITIVE_MLE)
    check('degrades to additive when the statistics were never maintained',
          abs(got.sum() - 8.0) < 1e-9, f'{got.sum()}')
    c4.set_search(search_agg=c4.AGG_MIXTURE, target_agg=c4.AGG_MIXTURE)


def test_incremental_equals_batch():
    print('\nIncremental accumulator == recompute from scratch')
    alphas = [RNG.gamma(2.0, size=3) + 0.05 for _ in range(37)]
    acc = c4._new_acc()
    for i, a in enumerate(alphas):
        c4._acc_add(acc, c4._payload(a)[0])
        if i < 2:
            continue
        for mode in c4.AGGREGATIONS:
            inc = c4.observed_alpha(acc, mode)
            fresh = c4.observed_alpha(acc_of(alphas[:i + 1]), mode)
            if not close(inc, fresh, 1e-9):
                check(f'{mode} incremental == batch at n={i + 1}', False,
                      f'{inc} vs {fresh}')
                return
    check('all three rules: incremental == batch at every n', True)


def test_payload_flip():
    print('\nPayload flipping')
    a = np.array([5.0, 2.0, 1.0])
    same, flip = c4._payload(a)
    same_of_flipped, _ = c4._payload(c4.flip_alpha(a))
    for j in range(4):
        check(f'flipped payload field {j} == payload of flipped alpha',
              close(flip[j], same_of_flipped[j], 1e-12))
    accs = acc_of([a])
    accf = c4._new_acc(); c4._acc_add(accf, flip)
    check('flipped observation collapses to the flipped belief',
          close(c4.observed_alpha(accf, c4.AGG_MIXTURE),
                c4.flip_alpha(c4.observed_alpha(accs, c4.AGG_MIXTURE)), 1e-9))


def test_floor_conc():
    print('\n_floor_conc preserves direction')
    a = np.array([[0.008, 0.001, 0.001]])
    out = c4._floor_conc(a, 0.15)
    check('total raised to the floor', close(out.sum(), 0.15, 1e-9))
    check('mean preserved', close(c4.dir_mean(out), c4.dir_mean(a), 1e-9))
    big = np.array([[10.0, 1.0, 1.0]])
    check('already-large belief untouched', close(c4._floor_conc(big, 0.15), big, 0))
    zero = np.array([[0.0, 0.0, 0.0]])
    check('all-zero row becomes uniform',
          close(c4._floor_conc(zero, 0.15), [[0.05, 0.05, 0.05]], 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
#  A tiny mock game so the tree can be tested without pyspiel
# ══════════════════════════════════════════════════════════════════════════════
class MockState:
    """Depth-limited binary tree, terminal at `depth` plies.  The result depends
    ONLY on the root player's first move — action 0 wins for player 0, action 1
    wins for player 1 — so the root is genuinely solvable and the solver's
    expected answer is unambiguous no matter how the rest is played.

    The observation is 3x2x2 rather than 3x1x1 on purpose: a GroupNorm over a
    single spatial element always outputs exactly 0, which would make the trunk
    constant and the "can it learn?" test vacuous."""

    def __init__(self, depth=3, hist=None):
        self.depth = depth
        self.hist = list(hist or [])

    def clone(self):
        return MockState(self.depth, self.hist)

    def current_player(self):
        return len(self.hist) % 2

    def legal_actions(self):
        return [0, 1]

    def apply_action(self, a):
        self.hist.append(int(a))

    def is_terminal(self):
        return len(self.hist) >= self.depth

    def returns(self):
        return [1.0, -1.0] if self.hist[0] == 0 else [-1.0, 1.0]

    def observation_tensor(self, player=0):
        v = [float(len(self.hist)), float(sum(self.hist)), float(player)]
        return [x for x in v for _ in range(4)]


class MockGame:
    def new_initial_state(self):
        return MockState()

    def observation_tensor_shape(self):
        return [3, 2, 2]

    def num_distinct_actions(self):
        return 2


MOCK_OBS_DIM = 12


def mock_node(state, pw=(0.34, 0.33, 0.33), conf=2.0):
    leg = state.legal_actions()
    k = len(leg)
    node = c4._CNode(state.current_player(), leg, np.array(pw), conf,
                     np.tile(np.array(pw), (k, 1)), np.full(k, conf),
                     obs=np.zeros(MOCK_OBS_DIM, np.float16))
    c4._seed_leaf(node)
    return node


def run_mock_search(root, root_state, sims, rng, aux=None):
    for _ in range(sims):
        if c4._node_solved_outcome(root) is not None:
            break
        path, st, payloads, edge = c4._select_leaf(root, root_state, rng)
        if st is None:
            c4._backup_terminal(path, payloads, aux)
            continue
        node, idx = edge
        child = mock_node(st)
        node.children[idx] = child
        c4._backup(path, c4._payload(child.v_alpha))


def test_search_bookkeeping():
    print('\nSearch bookkeeping on the mock game')
    rng = np.random.default_rng(1)
    state = MockState(depth=4)
    root = mock_node(state)
    run_mock_search(root, state, 200, rng)
    check('no virtual loss left in flight', int(root.vloss.sum()) == 0,
          f'{root.vloss}')
    # Every simulation contributes exactly one observation to the root node and
    # one to exactly one root edge; the root also seeded itself once.
    check('node evidence == 1 + edge evidence',
          abs(root.nacc[0] - (1.0 + root.visits().sum())) < 1e-9,
          f'{root.nacc[0]} vs {1 + root.visits().sum()}')
    for i in range(len(root.legal)):
        ch = root.children[i]
        if ch is None:
            continue
        check(f'child {i} evidence == its edge evidence',
              abs(ch.nacc[0] - root.visits()[i]) < 1e-9,
              f'{ch.nacc[0]} vs {root.visits()[i]}')

    # alpha_sel is maintained incrementally — it must equal a fresh recompute.
    for i in range(len(root.legal)):
        if root.term[i] >= 0:
            want = c4._SPIKE[root.term[i]]
        else:
            obs = c4.observed_alpha(root.eacc[i], c4._SEARCH_AGG)
            want = root.alpha_p[i] if obs is None else root.alpha_p[i] + obs
        check(f'alpha_sel[{i}] matches a fresh collapse',
              close(root.alpha_sel[i], want, 1e-9))


def test_backup_perspective():
    print('\nBackup perspective alternates correctly')
    state = MockState(depth=6)
    # A leaf that is a near-certain WIN for its own mover.
    root = mock_node(state)
    path = [(root, 0)]
    root.vloss[0] += 1
    child = mock_node(MockState(6, [0]), pw=(0.98, 0.01, 0.01), conf=100.0)
    root.children[0] = child
    c4._backup(path, c4._payload(child.v_alpha))
    v = c4.dir_value(root.alpha_sel[0])
    check('a winning child makes the parent edge LOSING',
          v < -0.5, f'edge value {v:.3f}')
    check('the root node itself also reads as losing',
          c4.dir_value(root.state_target()) < -0.4,
          f'{c4.dir_value(root.state_target()):.3f}')

    # Two plies down the sign comes back.
    root2 = mock_node(MockState(6))
    n1 = mock_node(MockState(6, [0]))
    root2.children[0] = n1
    leaf = mock_node(MockState(6, [0, 0]), pw=(0.98, 0.01, 0.01), conf=100.0)
    n1.children[0] = leaf
    root2.vloss[0] += 1; n1.vloss[0] += 1
    c4._backup([(root2, 0), (n1, 0)], c4._payload(leaf.v_alpha))
    check('two plies down, the sign is back to winning',
          c4.dir_value(root2.alpha_sel[0]) > 0.5,
          f'{c4.dir_value(root2.alpha_sel[0]):.3f}')
    check('the intermediate node reads as losing',
          c4.dir_value(n1.state_target()) < -0.4)


def test_solver():
    print('\nMCTS-Solver propagation')
    rng = np.random.default_rng(3)
    state = MockState(depth=2)
    root = mock_node(state)
    aux = []
    run_mock_search(root, state, 40, rng, aux)
    # Player 0 to move: action 0 fixes the result at a player-0 win, whatever
    # the opponent then does, so the root solves as a WIN via that edge.
    check('root solved as a WIN', c4._node_solved_outcome(root) == c4._WIN,
          f'{c4._node_solved_outcome(root)} term={root.term}')
    check('winning action proven a WIN', root.term[0] == c4._WIN, f'{root.term}')
    tgt = root.state_target()
    check('solved state target is the proof spike',
          c4.dir_value(tgt) > 0.9, f'{c4.dir_value(tgt):.3f}')
    check('root_pick(greedy) plays the proven win',
          c4.root_pick(root, rng, thompson=False) == 0)

    # Deeper: solving a subtree must prove the parent edge, flipped.
    state = MockState(depth=3)
    root = mock_node(state)
    aux = []
    run_mock_search(root, state, 400, rng, aux)
    check('deep root solved', c4._node_solved_outcome(root) is not None,
          f'term={root.term}')
    check('solver emitted exact aux samples', len(aux) > 0, f'{len(aux)}')
    if aux:
        s = aux[0]
        check('aux sample is flagged solved', bool(s['solved']))
        check('aux sample carries a proven z', abs(float(s['z'])) <= 1.0)


def test_targets():
    print('\nTraining targets')
    rng = np.random.default_rng(4)
    state = MockState(depth=5)
    root = mock_node(state)
    run_mock_search(root, state, 150, rng)
    t = c4.make_target(root)
    check('obs carried', t['obs'] is not None)
    check('every searched edge is evidence',
          set(t['ev_idx'].tolist())
          == set(np.nonzero((root.visits() > 0) | (root.term >= 0))[0].tolist()))
    check('evidence alphas are strictly positive', bool((t['ev_alpha'] > 0).all()))
    check('state target is strictly positive', bool((t['v_obs'] > 0).all()))
    check('state target total >= the smoothing floor',
          t['v_obs'].sum() >= 3 * c4.TARGET_EPS - 1e-6)
    check('played starts unset', t['played'] == -1)

    c4.finish_episode([t], [1.0, -1.0])
    check('finish_episode stamps z from the mover perspective',
          float(t['z']) == (1.0 if root.player == 0 else -1.0))
    c4.mark_unfinished([t])
    check('mark_unfinished drops the CE weight', float(t['z_w']) == 0.0)


def test_agg_toggles():
    print('\nsearch_agg / target_agg toggles are independent')
    try:
        c4.set_search(search_agg='nope')
        check('invalid aggregation rejected', False)
    except ValueError:
        check('invalid aggregation rejected', True)
    for sa in c4.AGGREGATIONS:
        for ta in c4.AGGREGATIONS:
            c4.set_search(search_agg=sa, target_agg=ta)
            rng = np.random.default_rng(5)
            state = MockState(depth=4)
            root = mock_node(state)
            run_mock_search(root, state, 120, rng)
            t = c4.make_target(root)
            ok = (np.isfinite(t['v_obs']).all() and np.isfinite(t['ev_alpha']).all()
                  and (t['v_obs'] > 0).all() and np.isfinite(root.alpha_sel).all()
                  and (root.alpha_sel > 0).all())
            check(f'search={sa} target={ta} produces finite positive beliefs', ok)
    c4.set_search(search_agg=c4.AGG_MIXTURE, target_agg=c4.AGG_MIXTURE)


def test_selection_modes():
    print('\nGaussian selection approximation')
    for mode in ('dirichlet', 'gaussian'):
        c4.set_search(selection=mode)
        rng = np.random.default_rng(6)
        state = MockState(depth=4)
        root = mock_node(state)
        run_mock_search(root, state, 120, rng)
        check(f'{mode}: search completes with finite beliefs',
              bool(np.isfinite(root.alpha_sel).all()))
        check(f'{mode}: every simulation was accounted for',
              abs(root.nacc[0] - (1.0 + root.visits().sum())) < 1e-9)
    c4.set_search(selection='dirichlet')


def test_degenerate_draws():
    print('\nDegenerate (α₀ → 0) sampling')
    rng = np.random.default_rng(7)
    a = np.full((64, 3), 1e-12)
    a[:, 0] = 9e-12
    g = rng.standard_gamma(a)
    v = c4._dir_v_from_gammas(g, a, rng)
    check('no NaNs from underflowed gamma draws', bool(np.isfinite(v).all()))
    check('values stay in [-1, 1]', bool((np.abs(v) <= 1.0 + 1e-9).all()))
    # In the α₀→0 limit the belief is a point mass on the largest component.
    check('limit picks the dominant corner most often', (v > 0).mean() > 0.7,
          f'{(v > 0).mean():.2f}')


# ══════════════════════════════════════════════════════════════════════════════
#  Torch: network shapes, loss finiteness, and that it can actually learn
# ══════════════════════════════════════════════════════════════════════════════
def test_torch():
    if not c4._HAS_TORCH:
        print('\n[skip] torch not installed — network/loss tests skipped')
        return
    import torch
    print('\nNetwork and losses')
    c4.set_game(MockGame())
    net = c4.C4DirichletNet(channels=8, num_blocks=1, head_ch=2)
    x = torch.zeros(4, *c4._OBS_SHAPE)
    v_logits, v_conf, a_logits, a_conf = net(x)
    check('state head shape', tuple(v_logits.shape) == (4, 3))
    check('state conc shape', tuple(v_conf.shape) == (4,))
    check('action head shape', tuple(a_logits.shape) == (4, 2, 3))
    check('action conc shape', tuple(a_conf.shape) == (4, 2))
    p = torch.softmax(v_logits, -1)
    check('untrained state belief is uniform', close(p.detach().numpy(),
                                                     np.full((4, 3), 1 / 3), 1e-6))
    check('untrained concentration is weak (softplus(1.4))',
          close(c4._conf(v_conf).detach().numpy(), np.full(4, 1.6204), 1e-3),
          f'{c4._conf(v_conf).detach().numpy()}')

    # Closed-form KL against a numerical check: KL(p‖p) == 0 and KL > 0 otherwise.
    a = torch.tensor([[3.0, 2.0, 5.0], [1.0, 1.0, 1.0]])
    b = torch.tensor([[3.0, 2.0, 5.0], [4.0, 1.0, 1.0]])
    kl = c4._dir_kl_rows(a, b)
    check('KL(p‖p) == 0', close(kl[0].item(), 0.0, 1e-6), f'{kl[0].item()}')
    check('KL(p‖q) > 0', kl[1].item() > 0, f'{kl[1].item()}')


def _dir_logpdf(x, a):
    from math import lgamma
    return (sum((ai - 1) * np.log(xi) for xi, ai in zip(x, a))
            + lgamma(sum(a)) - sum(lgamma(ai) for ai in a))


def test_torch_losses():
    if not c4._HAS_TORCH:
        return
    import torch
    print('\nDirichlet KL closed form vs Monte Carlo')
    a = np.array([3.0, 2.0, 5.0]); b = np.array([1.5, 4.0, 2.0])
    x = RNG.dirichlet(a, size=200_000)
    mc = np.mean([_dir_logpdf(xi, a) - _dir_logpdf(xi, b) for xi in x[:20_000]])
    cf = c4._dir_kl_rows(torch.tensor(a[None]), torch.tensor(b[None]))[0].item()
    check('closed-form KL matches Monte Carlo', close(cf, mc, 2e-2),
          f'{cf:.4f} vs {mc:.4f}')


def test_train_step_learns():
    if not c4._HAS_TORCH:
        return
    import torch
    print('\ntrain_step drives all five losses down')
    c4.set_game(MockGame())
    torch.manual_seed(0)
    net = c4.C4DirichletNet(channels=16, num_blocks=1, head_ch=4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)

    def sample(kind):
        """Two distinguishable positions with opposite, consistent labels."""
        obs = np.zeros(int(np.prod(c4._OBS_SHAPE)), np.float16)
        obs[0] = 1.0 if kind else 0.0
        nxt = obs.copy(); nxt[1] = 1.0
        win = kind
        v = np.array([8.0, 1.0, 1.0] if win else [1.0, 1.0, 8.0], np.float32)
        return {'obs': obs, 'legal': np.array([0, 1], np.int32),
                'v_obs': v, 'ev_idx': np.array([0, 1], np.int32),
                'ev_alpha': np.stack([v, v[::-1].copy()]),
                'played': 0, 'next_obs': nxt, 'next_term': -1,
                'z': np.float32(1.0 if win else -1.0), 'z_w': np.float32(1.0),
                'solved': False, 'player': 0}

    batch = [sample(i % 2) for i in range(64)]
    weights = (1.0, 2.0, 1.0, 1.0, 0.5)
    first = last = None
    for step in range(300):
        lv, parts = c4.train_step(net, opt, batch, 'cpu', weights, 1.0)
        if step == 0:
            first = parts
        last = parts
    for k in ('klv', 'kla', 'cev', 'cea'):
        check(f'{k} decreased', last[k] < first[k],
              f'{first[k]:.3f} -> {last[k]:.3f}')
    check('total loss decreased', last['loss'] < first['loss'],
          f'{first["loss"]:.3f} -> {last["loss"]:.3f}')
    check('all parts finite', all(np.isfinite(v) for v in last.values()),
          f'{last}')
    # The KL target Dir([8,1,1]) has mean 0.8 on the true corner, so the CE
    # cannot go below −log(0.8) = 0.223 while both terms are satisfied.
    check('CE learned the outcome (state)', last['cev'] < 0.30,
          f'{last["cev"]:.3f}')
    check('predicted concentration moved toward the target',
          last['cv_p'] > first['cv_p'], f'{first["cv_p"]:.2f} -> {last["cv_p"]:.2f}')

    # kl_normalize must not change the optimum, only the scale.
    _lv, pn = c4.train_step(net, opt, batch, 'cpu', weights, 1.0,
                            kl_normalize=True)
    check('kl_normalize path runs and is finite',
          all(np.isfinite(v) for v in pn.values()))
    _lv, pc = c4.train_step(net, opt, batch, 'cpu', weights, 1.0,
                            with_consistency=False)
    check('with_consistency=False zeroes the consistency term', pc['cons'] == 0.0)


def test_conf_no_overflow():
    """The `additive` rule drives target alpha0 to the visit count, so the
    concentration head is trained toward ~full_sims.  Past x = 88.7 a softplus
    written as log1p(exp(x)) overflows fp32 to +inf, which reaches lgamma as
    inf-inf = NaN and inf*0 = NaN in the masked action mean -- and one NaN
    backward writes NaN into every weight.  _conf must be finite for any input
    on either softplus path."""
    if not c4._HAS_TORCH:
        return
    import torch
    import torch.nn.functional as F
    print('\n_conf cannot overflow (the NaN-run regression)')
    c4.set_backend('cpu', device='cpu')
    raw = torch.tensor([-1e4, -50., 0., 1.4, 20., 88., 89., 200., 1e4])
    saved = c4._NATIVE['softplus']
    outs = {}
    for native in (True, False):
        c4._NATIVE['softplus'] = native
        x = raw.clone().requires_grad_(True)
        y = c4._conf(x)
        y.sum().backward()
        outs[native] = y.detach()
        check(f'_conf finite for every input (native softplus={native})',
              bool(torch.isfinite(y.detach()).all()), f'{y.detach()}')
        check(f'_conf gradient finite everywhere (native softplus={native})',
              bool(torch.isfinite(x.grad).all()), f'{x.grad}')
        check(f'_conf stays positive (native softplus={native})',
              bool((y.detach() > 0).all()))
    c4._NATIVE['softplus'] = saved
    check('both softplus paths agree',
          bool(torch.allclose(outs[True], outs[False], rtol=1e-5, atol=1e-6)))
    # Above the splice point _conf must be the identity to fp32 precision, and
    # below it must still be a real softplus.
    check('_conf(x) == x for large x',
          bool(torch.allclose(c4._conf(torch.tensor([89., 200., 1e4])),
                              torch.tensor([89., 200., 1e4]), rtol=1e-6)))
    check('_conf matches softplus below the splice',
          bool(torch.allclose(c4._conf(torch.tensor([-3., 0.5, 1.4, 19.])),
                              F.softplus(torch.tensor([-3., 0.5, 1.4, 19.])),
                              rtol=1e-6, atol=1e-9)))

    # The probe is what SELECTS the kernel, so it has to reject a bad one.
    check('_probe accepts a correct softplus', c4._probe(F.softplus, 'cpu'))
    check('_probe REJECTS a softplus that overflows at large x',
          not c4._probe(lambda t: torch.log1p(torch.exp(t)), 'cpu'))
    check('_probe accepts lgamma / digamma / group_norm',
          c4._probe(torch.lgamma, 'cpu') and c4._probe(torch.digamma, 'cpu')
          and c4._probe(lambda t: F.group_norm(t, 2), 'cpu', shape=(2, 4, 3, 3)))


def test_nonfinite_step_is_skipped():
    """A non-finite loss must never reach the weights: clip_grad_norm_ turns a
    NaN norm into a NaN scale, AdamW then writes NaN into every parameter and
    moment, and the run keeps going while being dead."""
    if not c4._HAS_TORCH:
        return
    import tempfile
    import torch
    print('\na non-finite step is skipped, not applied')
    c4.set_game(MockGame())
    c4.set_backend('cpu', device='cpu')
    net = c4.C4DirichletNet(8, 1, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    batch = [{'obs': np.zeros(MOCK_OBS_DIM, np.float16),
              'legal': np.array([0, 1], np.int32),
              'v_obs': np.array([2., 1., 1.], np.float32),
              'ev_idx': np.array([0], np.int32),
              'ev_alpha': np.array([[2., 1., 1.]], np.float32),
              'played': 0, 'next_obs': None, 'next_term': -1,
              'z': np.float32(0.0), 'z_w': np.float32(1.0), 'solved': False,
              'player': 0} for _ in range(8)]
    w = (1., 1., 1., 1., 0.)
    _lv, parts = c4.train_step(net, opt, batch, 'cpu', w, 1.0)
    check('a healthy step is flagged finite', parts['nonfinite'] == 0.0)

    with torch.no_grad():
        net.v_out.bias[3] = float('nan')
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    _lv, parts = c4.train_step(net, opt, batch, 'cpu', w, 1.0)
    check('a non-finite step is flagged', parts['nonfinite'] == 1.0)
    changed = [n for n, p in net.named_parameters()
               if n != 'v_out.bias' and not torch.equal(before[n], p.detach())]
    check('the skipped step changed NO weight', not changed, f'{changed}')
    check('the NaN did not spread beyond where it was injected',
          int(torch.isnan(net.v_out.bias.detach()).sum()) == 1)
    with torch.no_grad():
        net.v_out.bias[3] = 1.4
    _lv, parts = c4.train_step(net, opt, batch, 'cpu', w, 1.0)
    check('training resumes normally afterwards',
          parts['nonfinite'] == 0.0 and np.isfinite(parts['loss']))

    # Saving must refuse a poisoned net rather than clobber a good checkpoint.
    d = tempfile.mkdtemp()
    c4.save_benchmark_net(d, 'good', net)
    with torch.no_grad():
        net.v_out.bias[3] = float('nan')
    raised = False
    try:
        c4.save_benchmark_net(d, 'bad', net)
    except ValueError:
        raised = True
    check('saving a non-finite net raises instead of overwriting', raised)
    check('the healthy checkpoint is still readable',
          c4.load_benchmark_net(d, 'good', (8, 1, 2)) is not None)


def test_rewind_to_generation():
    """`latest.pt` is a run blob and `bench_N.pt` is a bare state_dict, so
    deleting latest.pt restarts from episode 1 rather than from the newest
    snapshot, and copying a bench file over it used to die with a bare
    KeyError: 'model'.  rewind_to_generation is the supported way back."""
    if not c4._HAS_TORCH:
        return
    import shutil
    import tempfile
    import torch
    print('\nrewind_to_generation')
    c4.set_backend('cpu', device='cpu')
    d = tempfile.mkdtemp()
    net = c4.C4DirichletNet(8, 1, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for g in ('1000', '4000', '12000'):
        c4.save_benchmark_net(d, g, net)
    pool = type('P', (), {
        'elo': {'random': 1000., '1000': 1200., '4000': 1400., '12000': 900.},
        'order': ['1000', '4000', '12000'],
        'pair_games': {('1000', '4000'): 10, ('4000', '12000'): 6}})()
    hist = dict(c4._new_hist(), ep=[1000, 4000, 12000], loss=[3., 2., 1.],
                quick_ep=[1000, 12000], q_w=[5, 5], q_d=[0, 0], q_l=[5, 5],
                elo=[{}, {}, {}])
    c4.save_checkpoint(d, 12000, net, opt, None, pool, hist)

    b = c4.rewind_to_generation(d, 4000, log=lambda *a: None)
    check('episode counter rewinds to the snapshot', b['ep'] == 4000)
    check('optimizer state is dropped', b['optim'] is None)
    check('later generations leave the Elo order',
          b['order'] == ['1000', '4000'], f"{b['order']}")
    check("'random' survives (every Elo update reads it)", 'random' in b['elo'])
    check('later generations leave the Elo table', '12000' not in b['elo'])
    check('pair games mentioning a dropped generation are removed',
          all('12000' not in k for k in b['pair_games']), f"{b['pair_games']}")
    check('history truncates at the snapshot', b['hist']['ep'] == [1000, 4000])
    check('the quick-eval series truncates independently',
          b['hist']['quick_ep'] == [1000], f"{b['hist']['quick_ep']}")
    check('every history column is present',
          set(b['hist']) == set(c4._HIST_KEYS))

    ck = c4.load_checkpoint(d)
    check('the rewound checkpoint reloads', ck is not None and ck['ep'] == 4000)
    n2 = c4.C4DirichletNet(8, 1, 2)
    n2.load_state_dict(ck['model'])
    check('the weights are the snapshot',
          all(torch.equal(a, b_) for a, b_
              in zip(net.parameters(), n2.parameters())))

    # A bench file copied over latest.pt must say what to do, not KeyError.
    shutil.copy(os.path.join(d, 'bench_4000.pt'), os.path.join(d, 'latest.pt'))
    raised = ''
    try:
        c4.load_checkpoint(d)
    except ValueError as e:
        raised = str(e)
    check('a bare state_dict at latest.pt raises a clear error',
          'rewind_to_generation' in raised, raised[:80])
    check('and rewinding still recovers from that state',
          c4.rewind_to_generation(d, 1000,
                                  log=lambda *a: None)['ep'] == 1000)
    check('rewinding to a generation with no snapshot fails loudly',
          _raises(FileNotFoundError, c4.rewind_to_generation, d, 7777,
                  log=lambda *a: None))


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


def test_batch_meta_masking():
    if not c4._HAS_TORCH:
        return
    print('\nbuild_batch_meta masking')
    c4.set_game(MockGame())
    base = {'obs': np.zeros(MOCK_OBS_DIM, np.float16),
            'legal': np.array([0, 1], np.int32),
            'v_obs': np.array([2., 1., 1.], np.float32),
            'ev_idx': np.array([0], np.int32),
            'ev_alpha': np.array([[2., 1., 1.]], np.float32),
            'played': 0, 'next_obs': None, 'next_term': -1,
            'z': np.float32(0.0), 'z_w': np.float32(1.0), 'solved': False,
            'player': 0}
    solved = dict(base, played=-1, z=np.float32(1.0), solved=True)
    term = dict(base, next_term=c4._LOSS)
    nn_ = dict(base, next_obs=np.zeros(MOCK_OBS_DIM, np.float16))
    meta, obs2 = c4.build_batch_meta([base, solved, term, nn_], 'cpu')
    pw = meta['played_w'].numpy(); cw = meta['cons_w'].numpy()
    check('solver sample has no played action', pw[1] == 0.0)
    check('sample with no successor has no consistency term', cw[0] == 0.0)
    check('terminal successor gets a consistency term from the spike',
          cw[2] == 1.0 and not bool(meta['cons_isnn'][2]))
    check('non-terminal successor is routed to the second forward block',
          cw[3] == 1.0 and bool(meta['cons_isnn'][3])
          and int(meta['cons_row'][3]) == 0)
    check('exactly one successor observation collected', len(obs2) == 1)
    check('unmasked evidence only where searched',
          meta['ev_mask'].numpy().tolist() == [[True, False]] * 4)


def test_consistency_direction():
    """The consistency target must be in the PLAYED node's perspective, both for
    a terminal successor (used as-is) and for a network successor (flipped one
    ply).  Getting either sign wrong teaches the action head the exact opposite
    of every decisive move, so pin both down explicitly."""
    if not c4._HAS_TORCH:
        return
    import torch
    print('\nConsistency-target perspective')
    c4.set_game(MockGame())
    cons_only = (0.0, 0.0, 0.0, 0.0, 1.0)

    def cons_of(sample, q_pred_corner):
        """L_cons when the action head predicts `q_pred_corner` for the played
        action, with a made-up successor state belief that is a clear WIN for
        the successor's own mover (so, flipped, a LOSS for the played node)."""
        meta, obs2 = c4.build_batch_meta([sample], 'cpu')
        big = 4.0
        q_log = torch.zeros(1, 2, 3)
        q_log[0, 0, q_pred_corner] = big
        out = (torch.zeros(1, 3), torch.full((1,), 2.0), q_log,
               torch.full((1, 2), 2.0))
        out2 = None
        if obs2:
            v2 = torch.zeros(1, 3); v2[0, c4._WIN] = big
            out2 = (v2, torch.full((1,), 2.0), torch.zeros(1, 2, 3),
                    torch.full((1, 2), 2.0))
        _t, parts = c4.full_loss(out, out2, meta, cons_only)
        return parts['cons']

    base = {'obs': np.zeros(MOCK_OBS_DIM, np.float16),
            'legal': np.array([0, 1], np.int32),
            'v_obs': np.array([2., 1., 1.], np.float32),
            'ev_idx': np.array([0], np.int32),
            'ev_alpha': np.array([[2., 1., 1.]], np.float32),
            'played': 0, 'next_obs': None, 'next_term': -1,
            'z': np.float32(0.0), 'z_w': np.float32(1.0), 'solved': False,
            'player': 0}

    # Terminal successor recorded as a WIN (this node's mover won by playing a*).
    term_win = dict(base, next_term=c4._WIN)
    check('terminal WIN successor: predicting a win costs less than a loss',
          cons_of(term_win, c4._WIN) < cons_of(term_win, c4._LOSS),
          f'{cons_of(term_win, c4._WIN):.3f} vs {cons_of(term_win, c4._LOSS):.3f}')

    # Network successor whose OWN mover (the opponent) is winning — so the
    # played action must look like a LOSS.
    nn_succ = dict(base, next_obs=np.zeros(MOCK_OBS_DIM, np.float16))
    check('network successor is flipped one ply',
          cons_of(nn_succ, c4._LOSS) < cons_of(nn_succ, c4._WIN),
          f'{cons_of(nn_succ, c4._LOSS):.3f} vs {cons_of(nn_succ, c4._WIN):.3f}')


# ══════════════════════════════════════════════════════════════════════════════
#  pyspiel: the real game — observation perspective and an end-to-end run
# ══════════════════════════════════════════════════════════════════════════════
def test_connect4():
    try:
        import pyspiel
    except Exception:
        print('\n[skip] pyspiel not installed — Connect 4 tests skipped')
        return
    print('\nConnect 4 wiring')
    game = pyspiel.load_game('connect_four')
    shape, A = c4.set_game(game)
    check('observation shape', shape == (3, 6, 7), f'{shape}')
    check('action count', A == 7)
    check("open_spiel's own tensor detected as absolute", c4._OBS_NEEDS_SWAP)

    st = game.new_initial_state()
    st.apply_action(3)                                   # player 0 plays column 3
    o = c4.make_obs(st).reshape(3, 6, 7)
    check('mover-relative: after p0 moves, p1 sees no discs of its own',
          o[0].sum() == 0.0 and o[1].sum() == 1.0,
          f'self={o[0].sum()} opp={o[1].sum()}')
    st.apply_action(3)                                   # player 1 replies
    o = c4.make_obs(st).reshape(3, 6, 7)
    check('…and p0 then sees one of each',
          o[0].sum() == 1.0 and o[1].sum() == 1.0)
    check('planes partition the board',
          float(o.sum()) == 42.0, f'{o.sum()}')

    # The same position reached with colours reversed must give the same tensor.
    a = game.new_initial_state()
    for m in (3, 4, 5):
        a.apply_action(m)
    b = game.new_initial_state()
    for m in (4, 3):
        b.apply_action(m)
    check('the tensor really is perspective-relative (not absolute)',
          not np.array_equal(c4.make_obs(a).reshape(3, 6, 7)[0],
                             np.asarray(a.observation_tensor(0),
                                        np.float16).reshape(3, 6, 7)[0])
          or a.current_player() == 0)


def test_connect4_selfplay():
    if not c4._HAS_TORCH:
        return
    try:
        import pyspiel
    except Exception:
        return
    print('\nConnect 4 end-to-end self-play (single process)')
    game = pyspiel.load_game('connect_four')
    c4.set_game(game)
    c4.set_search(c4.AGG_MIXTURE, c4.AGG_MIXTURE, 'dirichlet', 1.0)
    cfg = c4.Config(channels=8, num_blocks=1, head_ch=2, fast_sims=24,
                    full_sims=24, n_parallel_games=4, wave_per_game=4,
                    pool_prob=0.0, use_workers=False)
    net, sig = c4.build_network(cfg, 'cpu')
    sp = c4.ParallelSelfPlay(game, net, 'cpu', c4._worker_cfg(cfg, sig), seed=0)
    gen = sp.episodes()
    episodes = [next(gen) for _ in range(3)]
    check('episodes produced samples', all(len(e) > 0 for e in episodes))
    flat = [s for e in episodes for s in e]
    check('every sample has an observation',
          all(s['obs'] is not None and len(s['obs']) == 126 for s in flat))
    check('every non-solver sample recorded a played action',
          all(s['played'] >= 0 for s in flat if not s['solved']))
    check('every non-solver sample has a successor (obs or terminal)',
          all(s['next_obs'] is not None or s['next_term'] >= 0
              for s in flat if not s['solved']))
    check('outcomes are stamped', all(abs(float(s['z'])) <= 1.0 for s in flat))
    # A move that ENDED the game records its outcome in the same perspective as
    # z — both are "what this position's mover got".
    ends = [s for s in flat if s['next_term'] >= 0 and not s['solved']]
    corner = {1.0: c4._WIN, 0.0: c4._DRAW, -1.0: c4._LOSS}
    check('game-ending moves agree with z on perspective',
          bool(ends) and all(s['next_term'] == corner[float(s['z'])]
                             for s in ends),
          f'{[(s["next_term"], float(s["z"])) for s in ends[:5]]}')
    check('some game finished decisively',
          any(abs(float(s['z'])) == 1.0 for s in flat))
    check('game lengths are plausible', 0 < sp.stats['plies'] / max(sp.stats['games'], 1) <= 42,
          f"{sp.stats}")
    check('no cutoffs (Connect 4 always terminates)', sp.stats['cutoff'] == 0)

    # And that a training step consumes them.
    opt = c4.LerpFreeAdamW(net.parameters(), lr=1e-3)
    batch = flat[:64] if len(flat) >= 64 else flat
    lv, parts = c4.train_step(net, opt, batch, 'cpu', cfg.loss_weights, 1.0)
    check('train_step on real self-play data is finite',
          all(np.isfinite(v) for v in parts.values()), f'{parts}')

    # A searched bot must beat a random mover convincingly on Connect 4 —
    # even with an untrained net, because the solver finds immediate wins.
    print('\nSearch beats random (untrained net, MCTS-64)')
    rng = np.random.default_rng(0)
    bot = c4.C4MCTSBot(game, net, 'cpu', 64, batch_size=8, random_state=rng)
    wins = losses = draws = 0
    for g in range(12):
        state = game.new_initial_state()
        bot_side = g % 2
        while not state.is_terminal():
            if state.current_player() == bot_side:
                state.apply_action(c4.root_pick(bot.mcts_search(state), rng,
                                                thompson=False))
            else:
                leg = state.legal_actions()
                state.apply_action(int(leg[rng.integers(len(leg))]))
        r = state.returns()[bot_side]
        wins += r > 0; losses += r < 0; draws += r == 0
    check('MCTS-64 beats random', wins >= 9, f'W{wins} D{draws} L{losses}')


def main():
    test_primitives()
    test_single_component_roundtrip()
    test_identical_observations()
    test_additive_rule()
    test_additive_mle()
    test_mixture_against_monte_carlo()
    test_mean_against_monte_carlo()
    test_incremental_equals_batch()
    test_payload_flip()
    test_floor_conc()
    test_search_bookkeeping()
    test_backup_perspective()
    test_solver()
    test_targets()
    test_agg_toggles()
    test_selection_modes()
    test_degenerate_draws()
    test_torch()
    test_torch_losses()
    test_conf_no_overflow()
    test_nonfinite_step_is_skipped()
    test_rewind_to_generation()
    test_batch_meta_masking()
    test_consistency_direction()
    test_train_step_learns()
    test_connect4()
    test_connect4_selfplay()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
