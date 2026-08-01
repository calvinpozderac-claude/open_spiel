"""Self-tests for value_dist_utils.

Run:  python value_dist_tests.py

The primitives and tree need only numpy; the network and loss half is skipped
if torch is missing, and anything touching Othello if pyspiel is missing.
"""

import math
import sys

import numpy as np

import connect4_dirichlet_utils as c4
import value_dist_utils as vd

RNG = np.random.default_rng(0)
_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def game():
    import pyspiel
    g = pyspiel.load_game('othello')
    c4.set_game(g)
    return g


# ══════════════════════════════════════════════════════════════════════════════
def test_primitives():
    print('\nGaussian primitives')
    check('KL is zero for identical distributions',
          abs(float(vd.kl(0.3, 0.05, 0.3, 0.05))) < 1e-12)
    check('KL is positive otherwise', float(vd.kl(0.3, 0.05, -0.1, 0.2)) > 0)
    check('KL is asymmetric',
          abs(float(vd.kl(0.3, 0.05, -0.1, 0.2))
              - float(vd.kl(-0.1, 0.2, 0.3, 0.05))) > 1e-6)

    # Against Monte Carlo: KL(p||q) = E_p[log p - log q].
    m1, v1, m2, v2 = 0.2, 0.09, -0.1, 0.25
    x = RNG.normal(m1, math.sqrt(v1), 400_000)
    mc = float(np.mean(vd.nll(m2, v2, x) - vd.nll(m1, v1, x)))
    check('KL matches Monte Carlo', abs(float(vd.kl(m1, v1, m2, v2)) - mc) < 5e-3,
          f'{float(vd.kl(m1, v1, m2, v2)):.4f} vs {mc:.4f}')

    # NLL is a proper density: integrates to 1, minimised at the mean.
    xs = np.linspace(-6, 6, 200_001)
    dens = np.exp(-vd.nll(0.3, 0.16, xs))
    check('NLL is a normalised density',
          abs(np.trapezoid(dens, xs) - 1.0) < 1e-6
          if hasattr(np, 'trapezoid') else
          abs(np.trapz(dens, xs) - 1.0) < 1e-6)
    check('NLL is minimised at the mean',
          abs(xs[int(vd.nll(0.3, 0.16, xs).argmin())] - 0.3) < 1e-3)

    check('flip negates the mean and keeps the spread',
          vd.flip(0.4, 0.02) == (-0.4, 0.02))
    check('flip is an involution', vd.flip(*vd.flip(0.4, 0.02)) == (0.4, 0.02))

    # P(win) from the Gaussian.
    check('p_win is 0.5 at mean 0', abs(vd.p_win(0.0, 0.04) - 0.5) < 1e-12)
    check('p_win rises with the mean', vd.p_win(0.2, 0.04) > vd.p_win(0.1, 0.04))
    check('p_win is symmetric',
          abs(vd.p_win(0.2, 0.04) + vd.p_win(-0.2, 0.04) - 1.0) < 1e-12)
    check('a wider spread pulls p_win toward 0.5',
          vd.p_win(0.2, 1.0) < vd.p_win(0.2, 0.04))
    check('VAR_FLOOR keeps a zero-variance KL finite',
          np.isfinite(float(vd.kl(0.1, 0.0, 0.2, 0.0))))


def test_accumulator():
    print('\nBelief accumulator: prediction is observation 1, variance ~ 1/n')
    acc = vd.new_acc(0.2, 0.09)
    check('an unsearched edge IS the prediction',
          vd.belief(acc) == (0.2, 0.09))
    check('and so is its spread', vd.outcome_spread(acc) == (0.2, 0.09))

    # n identical observations N(m, v) -> belief N(m, v/n), the standard error.
    for n in (2, 4, 9, 25):
        a = vd.new_acc(0.2, 0.09)
        for _ in range(n - 1):
            vd.acc_add(a, 0.2, 0.09)
        m, v = vd.belief(a)
        check(f'belief sd = sd/sqrt(n) at n={n}',
              abs(v - 0.09 / n) < 1e-12 and abs(m - 0.2) < 1e-12,
              f'{math.sqrt(v):.5f} vs {math.sqrt(0.09 / n):.5f}')
        ms, vs = vd.outcome_spread(a)
        check(f'…while the OUTCOME spread does not narrow (n={n})',
              abs(vs - 0.09) < 1e-12, f'{vs}')

    # The mean is the plain average of the observed means.
    a = vd.new_acc(0.0, 0.04)
    for m in (0.4, -0.2, 0.1):
        vd.acc_add(a, m, 0.04)
    check('belief mean is the average of all observations, prediction included',
          abs(vd.belief(a)[0] - (0.0 + 0.4 - 0.2 + 0.1) / 4) < 1e-12)

    # A confident observation pulls the belief harder than a vague one only
    # through the mean; the variance term averages.
    a1 = vd.new_acc(0.0, 1.0); vd.acc_add(a1, 1.0, 1e-6)
    check('a near-certain observation collapses the belief spread',
          vd.belief(a1)[1] < 0.26, f'{vd.belief(a1)}')
    check('VAR_FLOOR applies to accumulated variance',
          vd.belief(vd.new_acc(0.0, 0.0))[1] >= vd.VAR_FLOOR)


def test_score():
    print('\nScore: normalised disc differential')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    rng = np.random.default_rng(1)
    ok_sign = ok_anti = 0
    n = 60
    for _ in range(n):
        st = g.new_initial_state()
        while not st.is_terminal():
            la = st.legal_actions()
            st.apply_action(int(la[rng.integers(len(la))]))
        s0, s1 = vd.othello_score(st, 0), vd.othello_score(st, 1)
        ok_anti += abs(s0 + s1) < 1e-12
        ok_sign += vd.sign_score(s0) == st.returns()[0]
    check(f'antisymmetric on all {n} games', ok_anti == n, f'{ok_anti}/{n}')
    check(f'sign equals the game result on all {n} games', ok_sign == n,
          f'{ok_sign}/{n}')
    check('normalised into [-1, 1]', abs(s0) <= 1.0)
    # The differential, not the count: a player with 30 discs can win or lose.
    check('it is a differential, so an even split is 0',
          abs(vd.othello_score(g.new_initial_state(), 0)) < 1e-12)


def test_tree():
    print('\nTree: selection, backup sign, terminal detection')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    rng = np.random.default_rng(2)
    st = g.new_initial_state()
    leg = st.legal_actions()
    k = len(leg)

    # Thompson selection must pick the argmax of the sampled values, and a
    # dominant mean must win almost always once beliefs are tight.
    mu = np.full(k, -0.5); mu[1] = 0.5
    node = vd.GNode(0, leg, mu, np.full(k, 1e-6), 0.0, 0.01)
    picks = [int(vd.sample_edges(node, rng).argmax()) for _ in range(200)]
    check('a clearly better action is selected with tight beliefs',
          picks.count(1) > 195, f'{picks.count(1)}/200')

    # With wide beliefs every action gets explored.
    node2 = vd.GNode(0, leg, np.zeros(k), np.full(k, 1.0), 0.0, 1.0)
    picks2 = {int(vd.sample_edges(node2, rng).argmax()) for _ in range(200)}
    check('wide beliefs explore every action', len(picks2) == k, f'{picks2}')

    # Proven edges are used at their exact value, whatever the belief says.
    node3 = vd.GNode(0, leg, np.full(k, 0.9), np.full(k, 1e-9), 0.0, 0.01)
    node3.term[2] = 0.95
    check('a proven edge overrides the sampled belief',
          int(vd.sample_edges(node3, rng).argmax()) == 2)

    # Backup flips the mean once per ply and leaves the variance alone.
    a = vd.GNode(0, leg, np.zeros(k), np.full(k, 0.04), 0.0, 0.04)
    b = vd.GNode(1, leg, np.zeros(k), np.full(k, 0.04), 0.0, 0.04)
    vd.backup([(a, 0), (b, 1)], 0.5, 0.01)
    check('the deepest node gets the value as given',
          abs(b.acc[1, 1] - 0.5) < 1e-12, f'{b.acc[1, 1]}')
    check('its parent gets the negation',
          abs(a.acc[1, 0] - (-0.5)) < 1e-12, f'{a.acc[1, 0]}')
    check('variance is carried unflipped',
          abs(b.acc[2, 1] - 0.05) < 1e-12 and abs(a.acc[2, 0] - 0.05) < 1e-12)
    check('visit counts exclude the seeded prediction',
          a.visits()[0] == 1 and a.visits()[1] == 0, f'{a.visits()}')

    # A terminal child is detected and recorded from the MOVER's perspective.
    st2 = g.new_initial_state()
    while not st2.is_terminal():
        la = st2.legal_actions()
        keep = st2.clone()
        st2.apply_action(int(la[rng.integers(len(la))]))
    root = vd.GNode(keep.current_player(), keep.legal_actions(),
                    np.zeros(len(keep.legal_actions())),
                    np.full(len(keep.legal_actions()), 0.04), 0.0, 0.04)
    path, leaf, val, edge = vd.select_leaf(root, keep, rng)
    check('a terminal child returns no leaf state', leaf is None)
    check('and returns the exact score with TERMINAL_VAR',
          val is not None and val[1] == vd.TERMINAL_VAR, f'{val}')
    check('the terminal value is from the moving node\'s perspective',
          abs(val[0] - vd.othello_score(
              _apply(keep, int(root.legal[path[0][1]])), keep.current_player()))
          < 1e-12)


def _apply(state, a):
    s = state.clone()
    s.apply_action(a)
    return s


def test_solver():
    print('\nSolver: a fully proven node proves its parent, negated')
    leg = [0, 1, 2]
    parent = vd.GNode(0, leg, np.zeros(3), np.full(3, 0.04), 0.0, 0.04)
    child = vd.GNode(1, leg, np.zeros(3), np.full(3, 0.04), 0.0, 0.04)
    child.obs = np.zeros(4, np.float16)
    parent.children[1] = child
    check('a partly proven node is not solved', vd.node_solved(child) is None)
    child.term[:] = [-0.2, 0.3, -0.5]
    check('a fully proven node takes the best action',
          vd.node_solved(child) == 0.3)
    aux = []
    vd.propagate_solved([(parent, 1), (child, 0)], aux)
    check('the parent edge is proven at the negated value',
          abs(parent.term[1] - (-0.3)) < 1e-12, f'{parent.term[1]}')
    check('a solver-labelled sample is emitted', len(aux) == 1)
    check('carrying the proven value and flagged solved',
          aux and abs(float(aux[0]['v_mu']) - 0.3) < 1e-6      # float32 target
          and aux[0]['solved'], f'{aux[:1]}')
    vd.propagate_solved([(parent, 1), (child, 0)], aux)
    check('an already-proven edge does not emit again', len(aux) == 1)


def test_targets():
    print('\nTargets')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    leg = [0, 1, 2]
    root = vd.GNode(0, leg, np.array([0.1, -0.2, 0.3]), np.full(3, 0.04),
                    0.05, 0.09)
    root.obs = np.zeros(4, np.float16)
    root.add(0, 0.4, 0.01)
    root.add(0, 0.2, 0.01)
    root.add(2, -0.1, 0.02)
    t = vd.make_target(root)
    check('only searched edges become targets',
          sorted(t['ev_idx'].tolist()) == [0, 2], f"{t['ev_idx']}")
    check('the edge target is the OUTCOME spread, not the belief',
          abs(float(t['ev_var'][0]) - float(root.spread()[1][0])) < 1e-9)
    check('the edge mean is the average of prediction and backups',
          abs(float(t['ev_mu'][0]) - (0.1 + 0.4 + 0.2) / 3) < 1e-6,
          f"{t['ev_mu'][0]}")
    check('the state target is the node value', np.isfinite(float(t['v_mu'])))
    check('z starts unweighted until the game finishes', float(t['z_w']) == 0.0)

    # finish_episode stamps the real differential per sample perspective.
    rng = np.random.default_rng(3)
    st = g.new_initial_state()
    while not st.is_terminal():
        la = st.legal_actions()
        st.apply_action(int(la[rng.integers(len(la))]))
    s0 = dict(t, player=0, solved=False)
    s1 = dict(t, player=1, solved=False)
    sv = dict(t, player=0, solved=True, z=np.float32(0.9))
    vd.finish_episode([s0, s1, sv], st)
    check('each sample gets its own mover\'s differential',
          abs(float(s0['z']) + float(s1['z'])) < 1e-12
          and abs(float(s0['z']) - vd.othello_score(st, 0)) < 1e-12)
    check('and is weighted in', float(s0['z_w']) == 1.0)
    check('a proven sample keeps its exact value',
          abs(float(sv['z']) - 0.9) < 1e-6, f"{sv['z']}")      # float32


def test_network_and_loss():
    if not vd._HAS_TORCH:
        print('\n(torch missing — network tests skipped)')
        return
    import torch
    print('\nNetwork and loss')
    try:
        game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    net = vd.GaussianNet(16, 1, 4)
    net.eval()
    x = torch.randn(5, *c4._OBS_SHAPE)
    vm, vlv, am, alv = net(x)
    check('state head shapes', vm.shape == (5,) and vlv.shape == (5,))
    check('action head shapes',
          am.shape == (5, c4._NUM_ACTIONS) and alv.shape == am.shape)
    mx = float(vm.detach().abs().max())
    check('an untrained net predicts an even game', mx < 1e-6, f'{mx}')
    check('with a deliberately wide spread',
          abs(float(vd._var(vlv).detach().mean()) - 0.25) < 1e-6)
    check('the action head is half the width of the Dirichlet one',
          net.a_out.out_features * 2 == c4.C4DirichletNet(16, 1, 4)
          .a_out.out_features)

    # torch KL must agree with the numpy one.
    a = torch.tensor([0.2, -0.4]); b = torch.tensor([0.09, 0.25])
    cm = torch.tensor([-0.1, 0.3]); cv = torch.tensor([0.16, 0.04])
    t = vd.kl_t(a, b, cm, cv).numpy()
    n_ = vd.kl(a.numpy(), b.numpy(), cm.numpy(), cv.numpy())
    check('torch KL matches the numpy KL', np.allclose(t, n_, atol=1e-6),
          f'{t} vs {n_}')
    check('torch NLL matches the numpy NLL',
          np.allclose(vd.nll_t(a, b, cm).numpy(),
                      vd.nll(a.numpy(), b.numpy(), cm.numpy()), atol=1e-6))
    check('_var clamps a huge log-variance',
          float(vd._var(torch.tensor([50.0]))) <= math.exp(vd.LOGVAR_MAX) + 1e-6)
    check('_var clamps a tiny one',
          float(vd._var(torch.tensor([-50.0]))) >= math.exp(vd.LOGVAR_MIN) - 1e-12)


def main():
    test_primitives()
    test_accumulator()
    test_score()
    test_tree()
    test_solver()
    test_targets()
    test_network_and_loss()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all value-dist tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
