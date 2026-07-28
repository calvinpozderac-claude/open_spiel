"""Self-tests for connect4_alphazero_utils and connect4_benchmark.

Run:  python connect4_benchmark_tests.py

Covers the PUCT tree and its solver, the AlphaZero loss, the mixed-engine
tournament harness, and the Bradley-Terry rating fit.  Needs torch and pyspiel;
the tree half also runs against a tiny mock game.
"""

import sys
import numpy as np

import connect4_dirichlet_utils as c4
import connect4_alphazero_utils as az
import connect4_benchmark as B

_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def node(priors=(0.5, 0.3, 0.2), value=0.0, player=0, legal=(0, 1, 2)):
    return az._AZNode(player, list(legal), np.array(priors), value,
                      obs=np.zeros(126, np.float16))


# ══════════════════════════════════════════════════════════════════════════════
def test_puct():
    print('\nPUCT selection')
    n = node()
    check('priors normalised', abs(n.P.sum() - 1.0) < 1e-12)
    # With no visits, score order follows the prior.
    check('first visit follows the prior', int(n.scores().argmax()) == 0,
          f'{n.scores()}')
    # A losing result on the best-prior edge should push selection elsewhere.
    n.N[0] = 8; n.W[0] = -8.0
    check('a losing edge is abandoned', int(n.scores().argmax()) != 0,
          f'{n.scores()}')
    # Q is the running mean from THIS node's perspective.
    check('Q = W/N', abs(n.q()[0] + 1.0) < 1e-12, f'{n.q()}')
    # Virtual loss pushes an in-flight edge down.
    m = node(); before = int(m.scores().argmax())
    m.vloss[before] = 1
    check('virtual loss diversifies a wave',
          int(m.scores().argmax()) != before)
    # A proven win dominates regardless of statistics.
    p = node(); p.N[2] = 100; p.W[2] = 100.0; p.term[0] = c4._WIN
    check('a proven win is selected over a well-scoring edge',
          int(p.scores().argmax()) == 0, f'{p.scores()}')
    p2 = node(); p2.term[0] = c4._LOSS
    check('a proven loss is never selected', int(p2.scores().argmax()) != 0)


def test_backup_signs():
    print('\nBackup alternates perspective')
    root = node(player=0)
    n1 = node(player=1); root.children[0] = n1
    leaf = node(player=0); n1.children[1] = leaf
    root.vloss[0] += 1; n1.vloss[1] += 1
    az._backup([(root, 0), (n1, 1)], 1.0)     # leaf mover (player 0) wins
    check('deepest edge gets the value as given', abs(n1.W[1] - 1.0) < 1e-12)
    check('one ply up the sign flips', abs(root.W[0] + 1.0) < 1e-12,
          f'{root.W[0]}')
    check('virtual loss released', int(root.vloss.sum() + n1.vloss.sum()) == 0)
    check('visits counted', root.N[0] == 1 and n1.N[1] == 1)


def test_solver():
    print('\nMCTS-Solver overlay')
    root = node(player=0)
    child = node(player=1); root.children[0] = child
    for i in range(3):
        child.term[i] = c4._LOSS          # every reply loses for player 1
    check('child solved as a LOSS for its mover',
          az._node_solved_outcome(child) == c4._LOSS)
    az._propagate_solved([(root, 0), (child, 0)])
    check('parent edge proven a WIN', root.term[0] == c4._WIN, f'{root.term}')
    check('root now solved', az._node_solved_outcome(root) == c4._WIN)
    pi = az.visit_policy(root)
    check('a solved root targets only the proven-winning move',
          abs(pi[0] - 1.0) < 1e-12, f'{pi}')

    # A proven DRAW must not target the losing siblings.
    r2 = node(player=0)
    r2.term[:] = [c4._LOSS, c4._DRAW, c4._LOSS]
    r2.N[:] = [50, 1, 40]
    check('root solved as a DRAW', az._node_solved_outcome(r2) == c4._DRAW)
    pi = az.visit_policy(r2)
    check('the drawing move gets the whole target despite 1 visit',
          abs(pi[1] - 1.0) < 1e-12, f'{pi}')


def test_visit_policy():
    print('\nVisit-count policy target')
    n = node(); n.N[:] = [30, 10, 0]
    pi = az.visit_policy(n, 1.0)
    check('proportional to visits at temp 1',
          np.allclose(pi, [0.75, 0.25, 0.0]), f'{pi}')
    pi0 = az.visit_policy(n, 0.0)
    check('temp 0 is one-hot on the most-visited',
          np.allclose(pi0, [1, 0, 0]), f'{pi0}')
    fresh = node()
    check('an unvisited node falls back to uniform',
          np.allclose(az.visit_policy(fresh), 1 / 3), f'{az.visit_policy(fresh)}')
    rng = np.random.default_rng(0)
    picks = [az.root_pick(n, rng, sample=True) for _ in range(400)]
    frac = np.mean([p == 0 for p in picks])
    check('sampling follows the visit distribution', 0.68 < frac < 0.82,
          f'{frac:.2f}')
    check('greedy takes the argmax',
          az.root_pick(n, rng, sample=False) == 0)


def test_root_noise():
    print('\nRoot exploration noise')
    rng = np.random.default_rng(0)
    n = node(priors=(0.98, 0.01, 0.01))
    az.add_root_noise(n, rng, frac=0.25, alpha=1.0)
    check('priors stay normalised', abs(n.P.sum() - 1.0) < 1e-9)
    check('noise moves mass off the peak', n.P[0] < 0.98, f'{n.P}')
    m = node(priors=(0.98, 0.01, 0.01))
    az.add_root_noise(m, rng, frac=0.0)
    check('frac=0 is a no-op', np.allclose(m.P, [0.98, 0.01, 0.01]))


def test_loss():
    print('\nAlphaZero loss')
    import torch
    game = c4.load_game(); c4.set_game(game)
    torch.manual_seed(0)
    net = az.AlphaZeroNet(16, 1, 4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)

    def sample(kind):
        obs = np.zeros(126, np.float16); obs[kind] = 1.0
        pi = np.zeros(7, np.float32); pi[0 if kind else 6] = 1.0
        return {'obs': obs, 'legal': np.arange(7, dtype=np.int32), 'pi': pi,
                'z': np.float32(1.0 if kind else -1.0), 'player': 0}

    batch = [sample(i % 2) for i in range(64)]
    first = last = None
    for step in range(250):
        _lv, parts = az.train_step(net, opt, batch, 'cpu')
        if step == 0:
            first = parts
        last = parts
    check('policy loss decreased', last['pol'] < first['pol'],
          f'{first["pol"]:.3f} -> {last["pol"]:.3f}')
    check('value loss decreased', last['val'] < first['val'],
          f'{first["val"]:.3f} -> {last["val"]:.3f}')
    check('policy CE approaches the target entropy', last['pol'] < 0.15,
          f'{last["pol"]:.3f}')
    check('value regresses to +-1', last['absv'] > 0.8, f'{last["absv"]:.2f}')
    check('all parts finite', all(np.isfinite(v) for v in last.values()))
    # Illegal moves must get no probability mass.
    x = c4.batch_to_tensor([batch[0]['obs']], 'cpu')
    with torch.no_grad():
        lg, _v = net(x)
    meta = az.build_batch([dict(batch[0], legal=np.array([0, 1, 2], np.int32),
                                pi=np.array([1., 0., 0.], np.float32))], 'cpu')
    check('masking keeps the gathered layout',
          tuple(meta['act'].shape) == (1, 3) and bool(meta['mask'].all()))


def test_selfplay_and_strength():
    print('\nAlphaZero self-play end to end')
    import torch
    game = c4.load_game(); c4.set_game(game)
    torch.manual_seed(0)
    net = az.AlphaZeroNet(16, 1, 4)
    shared = B.default_shared(num_episodes=1, fast_sims=24, full_sims=24,
                              n_parallel_games=6, wave_per_game=4,
                              pool_prob=0.0, use_workers=False)
    cfg = B.alphazero_config(shared)
    sp = az.ParallelSelfPlay(game, net, 'cpu', az._worker_cfg(cfg, (16, 1, 4)),
                             seed=0)
    gen = sp.episodes()
    eps = [next(gen) for _ in range(3)]
    flat = [s for e in eps for s in e]
    check('episodes produced samples', all(len(e) > 0 for e in eps))
    check('every sample has an observation',
          all(s['obs'] is not None and len(s['obs']) == 126 for s in flat))
    check('policy targets are distributions',
          all(abs(s['pi'].sum() - 1) < 1e-5 for s in flat))
    check('policy target length matches legal count',
          all(len(s['pi']) == len(s['legal']) for s in flat))
    check('outcomes stamped in {-1,0,1}',
          all(float(s['z']) in (-1.0, 0.0, 1.0) for s in flat))
    check('some game was decisive',
          any(abs(float(s['z'])) == 1.0 for s in flat))
    check('no cutoffs', sp.stats['cutoff'] == 0, f"{sp.stats}")
    opt = az.LerpFreeAdamW(net.parameters(), lr=1e-3) if hasattr(az, 'LerpFreeAdamW') \
        else torch.optim.AdamW(net.parameters(), lr=1e-3)
    _lv, parts = az.train_step(net, opt, flat[:64], 'cpu')
    check('train_step on real self-play data is finite',
          all(np.isfinite(v) for v in parts.values()), f'{parts}')

    print('\nAlphaZero search beats random (untrained net, MCTS-64)')
    rng = np.random.default_rng(0)
    bot = az.AZMCTSBot(game, net, 'cpu', 64, batch_size=8, random_state=rng)
    w = l = d = 0
    for g in range(12):
        st = game.new_initial_state(); side = g % 2
        while not st.is_terminal():
            if st.current_player() == side:
                st.apply_action(az.root_pick(bot.mcts_search(st), rng,
                                             sample=False))
            else:
                leg = st.legal_actions()
                st.apply_action(int(leg[rng.integers(len(leg))]))
        r = st.returns()[side]
        w += r > 0; l += r < 0; d += r == 0
    check('MCTS-64 beats random', w >= 9, f'W{w} D{d} L{l}')


def test_tournament():
    print('\nTournament harness (mixed engines)')
    game = c4.load_game(); c4.set_game(game)
    B.GAME_REF[0] = game
    tz = c4.C4DirichletNet(16, 1, 4); tz.eval()
    a = az.AlphaZeroNet(16, 1, 4); a.eval()
    players = {'tz': ('thompson', tz), 'az': ('alphazero', a),
               'random': ('random', None)}
    names, W, elo = B.round_robin(players, sims=0, games_per_pair=12,
                                  log=lambda *a: None)
    check('every pair played', int(W.sum()) == 3 * 12, f'{W.sum()}')
    check('result matrix is zero-sum per pair',
          all(abs(W[i, j] + W[j, i] - 12) < 1e-9
              for i in range(3) for j in range(3) if i != j))
    check('random anchored at 0 Elo',
          abs(elo[names.index('random')]) < 1e-6, f'{elo}')
    check('ratings finite', bool(np.all(np.isfinite(elo))))
    names2, W2, elo2 = B.round_robin(players, sims=8, games_per_pair=4,
                                     log=lambda *a: None)
    check('MCTS mode also runs both engines', int(W2.sum()) == 3 * 4)

    print('\nBradley-Terry recovers known ratings')
    true = np.array([0., 120., 240., -180.])
    n, g = len(true), 600
    rng = np.random.default_rng(0)
    M = np.zeros((n, n))
    s = 400 / np.log(10)
    for i in range(n):
        for j in range(i + 1, n):
            p = 1 / (1 + np.exp(-(true[i] - true[j]) / s))
            wins = rng.binomial(g, p)
            M[i, j] += wins; M[j, i] += g - wins
    r = B.bradley_terry(M, anchor_idx=0)
    err = float(np.abs(r - true).max())
    check('recovers ratings within 25 Elo at 600 games/pair', err < 25,
          f'max err {err:.1f}')


def test_matched_settings():
    print('\nArms are matched where they should be')
    shared = B.default_shared()
    pc = B.param_counts(shared)
    check('trunks identical', pc['trunk matches'], f'{pc}')
    tz = B.thompson_config('AA', shared)
    a = B.alphazero_config(shared)
    for f in ('num_episodes', 'channels', 'num_blocks', 'head_ch', 'seed',
              'fast_sims', 'full_sims', 'fast_prob', 'temp_threshold',
              'games_per_worker', 'worker_wave', 'n_parallel_games',
              'wave_per_game', 'pool_prob', 'batch_size', 'train_steps_per_ep',
              'max_buffer', 'lr_peak', 'lr_decay_eps', 'weight_decay',
              'grad_clip', 'quick_eval_every', 'deep_eval_every', 'eval_sims'):
        if getattr(tz, f) != getattr(a, f):
            check(f'{f} matches across engines', False,
                  f'{getattr(tz, f)} vs {getattr(a, f)}')
            break
    else:
        check('every shared hyperparameter matches across engines', True)
    check('the four Thompson arms differ ONLY in the two rules',
          {(B.thompson_config(n, shared).search_agg,
            B.thompson_config(n, shared).target_agg)
           for n in ('AA', 'AM', 'MA', 'MM')} ==
          {('additive', 'additive'), ('additive', 'additive_mle'),
           ('additive_mle', 'additive'), ('additive_mle', 'additive_mle')})
    check('each arm gets its own checkpoint dir',
          len({B.arm_dir(shared['root'], n) for n in B.ARMS}) == len(B.ARMS))


def main():
    test_puct()
    test_backup_signs()
    test_solver()
    test_visit_policy()
    test_root_noise()
    test_loss()
    test_selfplay_and_strength()
    test_tournament()
    test_matched_settings()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all benchmark tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
