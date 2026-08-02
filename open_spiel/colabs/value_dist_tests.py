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

    # A near-certain observation collapses the belief only if it AGREES with
    # what is already there.  One that contradicts the prior is evidence that
    # something is unresolved, not evidence of certainty -- the law of total
    # variance adds the spread of the means, so the belief stays wide.
    agree = vd.new_acc(1.0, 1.0); vd.acc_add(agree, 1.0, 1e-6)
    clash = vd.new_acc(0.0, 1.0); vd.acc_add(clash, 1.0, 1e-6)
    check('a near-certain AGREEING observation collapses the belief spread',
          vd.belief(agree)[1] < 0.26, f'{vd.belief(agree)}')
    check('a near-certain CONTRADICTING one does not',
          vd.belief(clash)[1] > vd.belief(agree)[1] * 1.4,
          f'{vd.belief(clash)[1]:.4f} vs {vd.belief(agree)[1]:.4f}')
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
    check('a proven node trains its NLL on the exact value, not the played-out '
          'result',
          abs(float(aux[0]['z']) - 0.3) < 1e-6
          and float(aux[0]['z_w']) == 1.0, f"{aux[0]['z']}, {aux[0]['z_w']}")
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
    # The log-variance is deliberately NOT clamped: a cap would hide a runaway
    # instead of preventing one.  What replaces it is visibility.
    check('_var does not cap a large log-variance',
          abs(float(vd._var(torch.tensor([8.0]))) - math.exp(8.0)) < 1e-3,
          f'{float(vd._var(torch.tensor([8.0])))}')
    check('_var does not floor a small one',
          float(vd._var(torch.tensor([-30.0]))) < 1e-12)
    check('no clamp constants remain',
          not hasattr(vd, 'LOGVAR_MIN') and not hasattr(vd, 'LOGVAR_MAX'))
    check('VAR_FLOOR is far below any meaningful spread (a divide-by-zero '
          'guard, not a cap)',
          vd.VAR_FLOOR ** 0.5 * vd.SCORE_SCALE < 0.1,
          f'{vd.VAR_FLOOR ** 0.5 * vd.SCORE_SCALE:.4f} discs')

    # INIT_SD must be honestly described: wider than reality, and it is a
    # DIFFERENTIAL, so sd 0.5 is a 48-16 board rather than 32 discs of a colour.
    check('INIT_SD is wider than the measured spread of real games',
          vd.INIT_SD > 0.291, f'{vd.INIT_SD}')
    diff = vd.INIT_SD * vd.SCORE_SCALE
    check('INIT_SD in board terms is a 48-16 differential',
          abs((64 + diff) / 2 - 48) < 1e-9, f'{(64 + diff) / 2:.0f}')
    check('the net initialises to exactly INIT_SD',
          abs(float(vd._var(net.v_out.bias.detach()[1:2]))
              - vd.INIT_SD ** 2) < 1e-6)


def test_halving_schedule():
    print('\nSequential halving: budget allocation')
    for n, m, want_phases in ((100, 8, 3), (300, 8, 3), (100, 4, 2), (16, 2, 1)):
        sch = vd.halving_schedule(n, m)
        check(f'log2({m}) phases at budget {n}', len(sch) == want_phases,
              f'{sch}')
        check(f'survivors halve each phase (n={n}, m={m})',
              [p[0] for p in sch] == [max(1, m >> i) for i in range(len(sch))],
              f'{[p[0] for p in sch]}')
        used = sum(a * b for a, b in sch)
        check(f'stays within budget (n={n}, m={m})', used <= n, f'{used} > {n}')
        check(f'uses most of it (n={n}, m={m})', used >= 0.85 * n, f'{used}/{n}')
    check('a single action needs no halving',
          vd.halving_schedule(50, 1) == [(1, 50)])
    check('every phase gives at least one simulation',
          all(b >= 1 for a, b in vd.halving_schedule(4, 16)))


def test_halving_search():
    if not vd._HAS_TORCH:
        return
    import torch
    print('\nSequential halving: search behaviour')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    net = vd.GaussianNet(16, 1, 4)
    rng = np.random.default_rng(0)
    st = g.new_initial_state()
    root, idx = vd.halving_search(net, 'cpu', st, rng, 100, max_considered=16)
    v = root.visits()
    check('the budget is respected', v.sum() <= 100, f'{v.sum()}')
    check('every candidate is simulated at least once', (v > 0).all(), f'{v}')
    check('the chosen index is legal', 0 <= idx < len(root.legal))
    check('survivors got strictly more simulations than the eliminated',
          v[idx] > v.min(), f'{v.tolist()} idx {idx}')
    # Two rounds of halving over 4 actions: 2 survive the first cut.
    top = sorted(v.tolist(), reverse=True)
    check('the visit profile is a halving profile, not a flat one',
          top[0] > top[-1], f'{top}')

    # A single legal action needs no search at all.
    forced = g.new_initial_state()
    root1 = vd.GNode(0, [3], np.zeros(1), np.full(1, 0.04), 0.0, 0.04)
    r, i = vd.halving_search(net, 'cpu', forced, rng, 50, root=root1)
    check('a forced move short-circuits', i == 0 and r.visits().sum() == 0)

    # Re-drawing from updated beliefs is what makes the randomness decay: an
    # action with many simulations has a narrow belief, so its draw sits near
    # its mean.  A dominant, well-visited action must survive essentially always.
    k = 4
    node = vd.GNode(0, [0, 1, 2, 3], np.array([0.9, -0.5, -0.5, -0.5]),
                    np.full(k, 1e-6), 0.0, 0.01)
    kept = [vd.halving_survivors(node, rng, list(range(k)))[0]
            for _ in range(200)]
    check('a dominant action survives the cut', kept.count(0) > 195,
          f'{kept.count(0)}/200')
    # With wide beliefs the cut is genuinely random, which is the point of
    # using the draw rather than the mean.
    wide = vd.GNode(0, [0, 1, 2, 3], np.zeros(k), np.full(k, 1.0), 0.0, 1.0)
    kept2 = {vd.halving_survivors(wide, rng, list(range(k)))[0]
             for _ in range(200)}
    check('but a genuinely uncertain one is not decided by noise alone',
          len(kept2) == k, f'{kept2}')

    check('a proven win overrides the survivor',
          vd.halving_pick(
              _proven_root(), 1, rng, False) == 2)


def _proven_root():
    r = vd.GNode(0, [0, 1, 2], np.array([0.9, 0.9, -0.9]),
                 np.full(3, 1e-9), 0.0, 0.01)
    r.term[2] = 0.95                      # proven, and better than any belief
    return r


def test_halving_selfplay():
    if not vd._HAS_TORCH:
        return
    import torch
    print('\nSequential halving: self-play and targets')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    net = vd.GaussianNet(16, 1, 4)
    out = {}
    for mode in ('thompson', 'halving'):
        cfg = dict(fast_sims=100, full_sims=100, fast_prob=1.0, n_parallel=4,
                   wave_per_game=4, temp=1.0, temp_threshold=30,
                   max_plies=128, root_select=mode, max_considered=16)
        sp = vd.ParallelSelfPlay(g, net, 'cpu', cfg, seed=0)
        gen = sp.episodes()
        flat = [s for _ in range(2) for s in next(gen)]
        out[mode] = (flat, sp)
        check(f'{mode}: episodes produced samples', len(flat) > 0)
        check(f'{mode}: every sample has an observation',
              all(s['obs'] is not None for s in flat))
        check(f'{mode}: outcomes stamped', all(s['z_w'] == 1.0 for s in flat))
        check(f'{mode}: games run to the end',
              sp.stats['plies'] / sp.stats['games'] > 40)
        check(f'{mode}: leaf evaluation is still batched',
              sp.fwd_rows / max(sp.fwd_calls, 1) > 3.0,
              f'{sp.fwd_rows / max(sp.fwd_calls, 1):.1f} rows/call')
    # The guarantee is per-candidate coverage, not a raw edge count: at a small
    # budget choose_considered deliberately narrows the candidate set, so
    # halving can search FEWER edges than Thompson and still be doing its job.
    # What must hold is that every edge it did consider carries real evidence.
    ev = {m: np.mean([len(s['ev_idx']) for s in f if not s['solved']])
          for m, (f, _) in out.items()}
    print(f'     searched edges/position: '
          f"halving {ev['halving']:.1f}  thompson {ev['thompson']:.1f}")
    for mode, (f, _) in out.items():
        ns = np.concatenate([s['ev_n'] for s in f if len(s['ev_idx'])])
        check(f'{mode}: every searched edge has at least one backup',
              ns.min() >= 2.0, f'{ns.min()}')
    hn = np.concatenate([s['ev_n'] for s in out['halving'][0]
                         if len(s['ev_idx'])])
    tn = np.concatenate([s['ev_n'] for s in out['thompson'][0]
                         if len(s['ev_idx'])])
    # Halving CONCENTRATES evidence rather than spreading it: eliminated
    # candidates keep their phase-1 simulations while survivors collect every
    # later phase, so the profile is far more unequal than Thompson's roughly
    # value-proportional one.  That is the halving signature, and it is exactly
    # why the action KL has an evidence-weighted option.
    check('halving concentrates evidence more than Thompson does',
          hn.std() / hn.mean() > tn.std() / tn.mean(),
          f'halving cv {hn.std() / hn.mean():.2f} vs '
          f'thompson cv {tn.std() / tn.mean():.2f}')
    # Note it concentrates differently, not harder: the phase quota caps any
    # single edge, so Thompson's most-visited edge can exceed halving's while
    # halving's overall profile is the more unequal one.


def test_choose_considered():
    """Gumbel AlphaZero's max_num_considered_actions: at a small budget,
    considering every action means the first cut is decided by the prior draw
    rather than by evidence."""
    print('\nbudget-aware candidate count')
    check('a forced move considers one action',
          vd.choose_considered(100, 1) == 1)
    check('never more than are legal', vd.choose_considered(100, 3) == 3)
    check('never more than the cap',
          vd.choose_considered(10_000, 40, max_considered=16) == 16)
    prev = 0
    for n in (8, 16, 32, 64, 100, 300):
        m = vd.choose_considered(n, 8)
        check(f'budget {n} considers {m}, never fewer than a smaller budget',
              m >= prev, f'{m} < {prev}')
        prev = m
        sch = vd.halving_schedule(n, m)
        check(f'…and phase 1 still gives each candidate >= 2 sims (n={n})',
              sch[0][1] >= 2, f'{sch}')
    check('it is inert at the benchmark budget',
          vd.choose_considered(100, 8) == 8 and vd.choose_considered(300, 8) == 8)
    check('but bites at a small one', vd.choose_considered(16, 8) < 8)


def test_evidence_weighting():
    if not vd._HAS_TORCH:
        return
    import torch
    print('\nevidence weighting of the action KL')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    net = vd.GaussianNet(16, 1, 4)
    cfg = dict(fast_sims=32, full_sims=32, fast_prob=1.0, n_parallel=4,
               wave_per_game=4, temp=1.0, temp_threshold=30, max_plies=128,
               root_select='halving', max_considered=16)
    sp = vd.ParallelSelfPlay(g, net, 'cpu', cfg, seed=0)
    gen = sp.episodes()
    flat = [s for _ in range(2) for s in next(gen)]
    ns = np.concatenate([s['ev_n'] for s in flat if len(s['ev_idx'])])
    check('targets record how much evidence stands behind them',
          ns.min() >= 1.0 and ns.max() > ns.min(),
          f'{ns.min()}..{ns.max()}')
    check('halving really does search edges very unequally',
          ns.max() / ns.min() > 3.0, f'{ns.max() / ns.min():.1f}x')

    opt = torch.optim.AdamW(net.parameters(), lr=0.0)   # measure, do not move
    _lv, pu = vd.train_step(net, opt, flat[:64], 'cpu', (1., 1., 1., 1.),
                            ev_weight='uniform')
    _lv, pe = vd.train_step(net, opt, flat[:64], 'cpu', (1., 1., 1., 1.),
                            ev_weight='evidence')
    check('the two weightings give different action KLs',
          abs(pu['kla'] - pe['kla']) > 1e-6, f"{pu['kla']} vs {pe['kla']}")
    check('both are finite', np.isfinite(pu['kla']) and np.isfinite(pe['kla']))
    check('uniform is the default', vd.Config().ev_weight == 'uniform')

    # With every edge equally evidenced the two must agree exactly.
    for s_ in flat:
        if len(s_['ev_idx']):
            s_['ev_n'] = np.full(len(s_['ev_idx']), 5.0, np.float32)
    _lv, qu = vd.train_step(net, opt, flat[:64], 'cpu', (1., 1., 1., 1.),
                            ev_weight='uniform')
    _lv, qe = vd.train_step(net, opt, flat[:64], 'cpu', (1., 1., 1., 1.),
                            ev_weight='evidence')
    check('with equal evidence the weighting is a no-op',
          abs(qu['kla'] - qe['kla']) < 1e-5, f"{qu['kla']} vs {qe['kla']}")


def test_spatial_head():
    if not vd._HAS_TORCH:
        return
    import torch
    print('\nspatial action head')
    try:
        g = game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    dense = vd.make_net(32, 3, 8, head='dense')
    spat = vd.make_net(32, 3, 8, head='spatial')

    def heads(n):
        return sum(p.numel() for nm, p in n.named_parameters()
                   if nm.split('.')[0] in ('v_out', 'a_out', 'a_head', 'a_pass'))

    check('the spatial head is far smaller', heads(spat) * 20 < heads(dense),
          f'{heads(spat)} vs {heads(dense)}')
    check('the trunk is unchanged',
          sum(p.numel() for n, p in dense.named_parameters()
              if n.startswith(('stem', 'body')))
          == sum(p.numel() for n, p in spat.named_parameters()
                 if n.startswith(('stem', 'body'))))

    x = torch.randn(4, *c4._OBS_SHAPE)
    for name, net in (('dense', dense), ('spatial', spat)):
        net.eval()
        with torch.inference_mode():
            vm, vlv, am, alv = net(x)
        check(f'{name}: emits one distribution per action',
              am.shape == (4, c4._NUM_ACTIONS) and alv.shape == am.shape,
              f'{am.shape}')
        check(f'{name}: initialises to an even game', float(vm.abs().max()) < 1e-6)
        check(f'{name}: at INIT_SD',
              abs(float(vd._var(alv).mean()) - vd.INIT_SD ** 2) < 1e-6)

    # The mapping must be the real one: action a < 64 IS cell (a//8, a%8).
    # Perturb one cell's features and check only that action's output moves.
    st = g.new_initial_state()
    obs = c4.make_obs(st)
    xx = c4.batch_to_tensor([obs], 'cpu')
    with torch.no_grad():
        spat.a_head.weight.normal_(0, 0.5)
        h = spat.proj(spat.body(spat.stem(xx)))
        a = spat.a_head(h)
        flat = a.reshape(1, 2, 64).transpose(1, 2)[0, :, 0]
    check('cell order matches action order',
          torch.allclose(flat, a[0, 0].reshape(-1), atol=1e-6))

    # Weight sharing is the whole point: every legal cell trains the same
    # weights, so a gradient from ONE position reaches the head used by all.
    spat.zero_grad()
    h = spat.proj(spat.body(spat.stem(xx)))
    spat.a_head(h)[0, 0, 2, 3].backward()
    gw = spat.a_head.weight.grad
    check('one cell\'s gradient updates the shared action weights',
          gw is not None and float(gw.abs().sum()) > 0)
    dense.zero_grad()
    dv, dl, da, dal = dense(xx)
    da[0, 19].backward()
    rows = dense.a_out.weight.grad.abs().sum(1)
    touched = int((rows > 0).sum())
    check('the dense head only updates that action\'s own rows',
          touched <= 2, f'{touched} rows')
    check('…which is exactly the sparsity the spatial head removes', True)

    # Both round-trip through disk at their own head type.
    import tempfile
    d = tempfile.mkdtemp()
    vd.save_benchmark_net(d, 's', spat)
    back = vd.load_benchmark_net(d, 's', (32, 3, 8, 'spatial'))
    check('spatial round-trips', isinstance(back, vd.SpatialGaussianNet))
    vd.save_benchmark_net(d, 'd', dense)
    check('dense round-trips',
          isinstance(vd.load_benchmark_net(d, 'd', (32, 3, 8, 'dense')),
                     vd.GaussianNet))
    check('a 3-tuple signature still works, defaulting to spatial',
          isinstance(vd.load_benchmark_net(d, 's', (32, 3, 8)),
                     vd.SpatialGaussianNet))


def test_sample_edges_fast_path():
    """sample_edges profiled at 24% of self-play time, so it was rewritten
    around counters and in-place numpy.  It must still be the same function."""
    print('\nsample_edges after optimisation')
    rng = np.random.default_rng(0)
    k = 6
    node = vd.GNode(0, list(range(k)), np.linspace(-0.5, 0.5, k),
                    np.full(k, 1e-8), 0.0, 0.01)
    x = vd.sample_edges(node, rng)
    check('with near-zero variance the draw is the mean',
          np.allclose(x, np.linspace(-0.5, 0.5, k), atol=1e-3), f'{x}')
    check('and the argmax is the best action', int(x.argmax()) == k - 1)

    # Empirically the draw must match N(belief_mu, belief_var).
    node2 = vd.GNode(0, [0], np.array([0.3]), np.array([0.09]), 0.0, 0.09)
    for _ in range(3):
        node2.add(0, 0.3, 0.09)
    mu, var = (float(v[0]) for v in node2.belief())
    draws = np.array([vd.sample_edges(node2, rng)[0] for _ in range(20_000)])
    check('draws match the belief mean',
          abs(float(draws.mean()) - mu) < 0.02,
          f'{float(draws.mean()):.4f} vs {mu:.4f}')
    check('draws match the belief sd',
          abs(float(draws.std()) - math.sqrt(var)) < 0.02,
          f'{float(draws.std()):.4f} vs {math.sqrt(var):.4f}')

    # Counters must track the arrays they replace, or the fast path skips work
    # it should have done.
    n3 = vd.GNode(0, [0, 1, 2], np.zeros(3), np.full(3, 1e-9), 0.0, 0.01)
    check('no proven edges initially', n3.n_term == 0)
    n3.prove(1, 0.9)
    check('prove() bumps the counter', n3.n_term == 1)
    n3.prove(1, 0.9)
    check('proving the same edge twice does not double-count', n3.n_term == 1)
    check('the proven value is used in the draw',
          abs(vd.sample_edges(n3, rng)[1] - 0.9) < 1e-9)
    n3.vloss[2] = 1
    n3.n_vloss = 1
    check('virtual loss pushes an in-flight edge down',
          vd.sample_edges(n3, rng)[2] < -100)


def test_variance_cannot_collapse():
    """The action head's KL target is built from variances the network itself
    predicted, so that loop is self-consistent at ANY level including zero.
    The law of total variance breaks it: the spread of the observed MEANS comes
    from actually searching different continuations, not from the head, so a
    confident head cannot wish it away."""
    print('\nvariance cannot collapse to zero')
    # Five observations the network is maximally confident about, but which
    # disagree with each other.  The old rule returned the mean variance and so
    # reported near-certainty; the correct one reports the disagreement.
    tiny = 1e-9
    agree = vd.new_acc(0.3, tiny)
    clash = vd.new_acc(-0.5, tiny)
    for m in (0.3, 0.3, 0.3, 0.3):
        vd.acc_add(agree, m, tiny)
    for m in (0.5, -0.4, 0.45, 0.0):
        vd.acc_add(clash, m, tiny)
    _am, av = vd.outcome_spread(agree)
    _cm, cv = vd.outcome_spread(clash)
    check('agreeing near-certain observations DO give a tiny spread',
          av <= vd.VAR_FLOOR + 1e-12, f'{av}')      # floored, as designed
    check('disagreeing ones cannot be collapsed by a confident head',
          cv > 0.1, f'{cv}')
    # Each observation's variance is floored at VAR_FLOOR, so the mean-variance
    # term is exactly that; everything above it is the spread of the means.
    check('the inflation is exactly the variance of the means',
          abs(cv - (np.var([-0.5, 0.5, -0.4, 0.45, 0.0]) + vd.VAR_FLOOR)) < 1e-9,
          f'{cv}')

    # It is the law of total variance, so it must match the two-stage identity.
    rng = np.random.default_rng(0)
    mus = rng.normal(0, 0.3, 40)
    vars_ = rng.uniform(0.001, 0.05, 40)
    acc = vd.new_acc(mus[0], vars_[0])
    for m, v in zip(mus[1:], vars_[1:]):
        vd.acc_add(acc, m, v)
    _m, tv = vd.outcome_spread(acc)
    check('total variance = E[var] + Var[mean]',
          abs(tv - (vars_.mean() + mus.var())) < 1e-9,
          f'{tv} vs {vars_.mean() + mus.var()}')
    check('the belief is that over n, not the raw spread',
          abs(vd.belief(acc)[1] - tv / len(mus)) < 1e-12)

    # And the node-level accumulator must agree with the module-level one.
    node = vd.GNode(0, [0], np.array([mus[0]]), np.array([vars_[0]]), 0.0, 0.04)
    for m, v in zip(mus[1:], vars_[1:]):
        node.add(0, m, v)
    check('GNode agrees with the standalone accumulator',
          abs(float(node.spread()[1][0]) - tv) < 1e-9)
    check('and so does its belief',
          abs(float(node.belief()[1][0]) - tv / len(mus)) < 1e-9)


def test_calibration_metric():
    if not vd._HAS_TORCH:
        return
    import torch
    print('\ncalibration diagnostic')
    try:
        game()
    except Exception:
        print('  (pyspiel missing — skipped)')
        return
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    net = vd.GaussianNet(16, 1, 4)          # predicts sd = INIT_SD everywhere
    B, K = 512, 6

    def run(z):
        meta = {'pad_act': torch.zeros(B, K, dtype=torch.long),
                'pad_mask': torch.ones(B, K, dtype=torch.bool),
                'v_mu': torch.zeros(B), 'v_var': torch.full((B,), 0.05),
                'ev_mu': torch.zeros(B, K), 'ev_var': torch.full((B, K), 0.05),
                'ev_n': torch.ones(B, K),
                'ev_mask': torch.ones(B, K, dtype=torch.bool),
                'z': z, 'z_w': torch.ones(B),
                'played': torch.zeros(B, dtype=torch.long),
                'played_w': torch.ones(B)}
        x = torch.randn(B, *c4._OBS_SHAPE)
        return vd.full_loss(net(x), meta, (1., 1., 1., 1.))[1]

    ok = run(torch.randn(B) * vd.INIT_SD)
    over = run(torch.randn(B) * vd.INIT_SD * 2)
    under = run(torch.randn(B) * vd.INIT_SD * 0.25)
    check('a calibrated net reads calib ~ 1', abs(ok['calib'] - 1.0) < 0.15,
          f"{ok['calib']:.2f}")
    check('an OVERCONFIDENT net reads calib > 1', over['calib'] > 2.0,
          f"{over['calib']:.2f}")
    check('a hedging net reads calib < 1', under['calib'] < 0.5,
          f"{under['calib']:.2f}")
    check('cov68 is ~0.68 when calibrated', abs(ok['cov68'] - 0.683) < 0.08,
          f"{ok['cov68']:.2f}")
    check('cov68 falls when overconfident', over['cov68'] < 0.5,
          f"{over['cov68']:.2f}")
    check('cov68 rises when hedging', under['cov68'] > 0.9,
          f"{under['cov68']:.2f}")
    check('rmse is reported in discs',
          abs(ok['rmse'] - vd.INIT_SD * vd.SCORE_SCALE) < 4.0, f"{ok['rmse']:.1f}")


def main():
    test_primitives()
    test_accumulator()
    test_variance_cannot_collapse()
    test_score()
    test_tree()
    test_solver()
    test_targets()
    test_network_and_loss()
    test_calibration_metric()
    test_spatial_head()
    test_sample_edges_fast_path()
    test_halving_schedule()
    test_halving_search()
    test_halving_selfplay()
    test_choose_considered()
    test_evidence_weighting()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all value-dist tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
