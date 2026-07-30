"""Self-tests for connect4_solved_eval.

Run:  python connect4_solved_eval_tests.py

The solver half needs nothing but the standard library.  The adapter half is
skipped automatically if torch or pyspiel is missing.
"""

import os
import random
import sys
import tempfile

import connect4_solved_eval as ev

_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def sample_positions(n_games=400, seed=0):
    """Random games that never take an immediate win, so play runs deep.
    Returns {moves_played: [Position]}."""
    rng = random.Random(seed)
    by_depth = {}
    for _ in range(n_games):
        p = ev.Position()
        while True:
            legal = [c for c in p.legal_columns() if not p.is_winning_move(c)]
            if not legal or p.is_draw():
                break
            by_depth.setdefault(p.moves, []).append(p)
            p = p.play(rng.choice(legal))
    return by_depth


def brute(p):
    """Naive full minimax — no pruning, no table, no bounds.  Deliberately
    shares nothing with Solver so agreement means something."""
    if p.winning_spots() & p.possible():
        return (ev.WIDTH * ev.HEIGHT + 1 - p.moves) // 2
    legal = p.legal_columns()
    if not legal or p.is_draw():
        return 0
    return max(-brute(p.play(c)) for c in legal)


# ══════════════════════════════════════════════════════════════════════════════
def test_bitboard():
    print('\nBitboard mechanics')
    p = ev.Position.from_sequence('4444444')
    check('a full column is rejected', p is None)
    check('a non-column digit is rejected',
          ev.Position.from_sequence('48') is None)
    check('an empty sequence is the empty board',
          ev.Position.from_sequence('').moves == 0)
    p = ev.Position.from_sequence('1212')
    check('moves counted', p.moves == 4)
    check('all seven columns playable early',
          p.legal_columns() == list(range(7)))

    # is_winning_move must require the LANDING square to be the winning one.
    # winning_spots() also marks cells higher up a column that are not playable
    # yet; without intersecting possible() those read as immediate wins.
    p = ev.Position.from_sequence('121212')
    check('a real immediate win is seen', p.is_winning_move(0))
    q = ev.Position.from_sequence('1212')
    hi = [c for c in range(7)
          if (q.winning_spots() & ev._column_mask(c))
          and not (q.winning_spots() & q.possible() & ev._column_mask(c))]
    check('a winning cell out of reach is NOT an immediate win',
          all(not q.is_winning_move(c) for c in hi), f'cols {hi}')


def test_half():
    print('\nInteger halving truncates toward zero (C semantics)')
    check('positive', ev._half(5) == 2 and ev._half(4) == 2)
    check('negative truncates toward zero, not down',
          ev._half(-5) == -2 and ev._half(-21) == -10,
          f'{ev._half(-5)}, {ev._half(-21)}')
    check('python // would have floored', -5 // 2 == -3)


def test_solver_against_brute_force():
    print('\nSolver vs an independent brute-force minimax')
    bd = sample_positions(seed=1)
    s = ev.Solver()
    n = bad = 0
    for depth in (34, 36, 38):
        for p in bd.get(depth, [])[:60]:
            n += 1
            if brute(p) != s.solve(p):
                bad += 1
    check(f'exact agreement on {n} endgame positions', bad == 0, f'{bad} bad')


def test_solver_consistency():
    print('\nSolver internal consistency')
    bd = sample_positions(seed=2)
    s = ev.Solver()
    n = b1 = b2 = b3 = 0
    for depth in (24, 28, 32):
        for p in bd.get(depth, [])[:25]:
            n += 1
            sc = s.solve(p)
            ms = s.move_scores(p)
            if max(ms.values()) != sc:
                b1 += 1
            for c, v in ms.items():
                if p.is_winning_move(c):
                    continue
                ch = p.play(c)
                if (0 if ch.is_draw() else -s.solve(ch)) != v:
                    b2 += 1
            # A won position has a winning move; a lost one has none; a drawn
            # one has a drawing move and no winning one.
            if sc > 0 and not any(v > 0 for v in ms.values()):
                b3 += 1
            if sc < 0 and any(v >= 0 for v in ms.values()):
                b3 += 1
            if sc == 0 and (any(v > 0 for v in ms.values())
                            or not any(v == 0 for v in ms.values())):
                b3 += 1
    check(f'solve(p) == max(move_scores(p)) over {n} positions', b1 == 0)
    check('move_scores(p)[c] == -solve(child)', b2 == 0)
    check('outcome agrees with the available moves', b3 == 0)


def test_score_convention():
    print("\nPons' score convention")
    s = ev.Solver()
    # Three in a column with the fourth playable: the mover wins with their 4th
    # stone, so 22 - 4 = 18.
    p = ev.Position.from_sequence('121212')
    check('win with the 4th stone scores 18', s.solve(p) == 18, f'{s.solve(p)}')
    check('formula matches the immediate-win shortcut',
          (ev.WIDTH * ev.HEIGHT + 1 - p.moves) // 2 == 18)
    # Deeper into the game the same win is worth less.
    p2 = ev.Position.from_sequence('1212123434343')
    if p2 is not None and (p2.winning_spots() & p2.possible()):
        check('a later win scores lower than an earlier one', s.solve(p2) < 18,
              f'{s.solve(p2)}')


def test_parse_and_roundtrip():
    print('\nFile format round-trip')
    check('parse_line', ev.parse_line('4453 -2') == ('4453', -2))
    check('parse_line rejects junk', ev.parse_line('4453') is None
          and ev.parse_line('a b') is None)
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'Test_L3_R1')
    ev.generate(path, n=40, min_moves=30, max_moves=36, seed=7, log=None)
    recs = ev.load_file(path)
    check('every generated line reloads', len(recs) == 40)
    check('sequences replay to legal non-terminal positions',
          all(p is not None and p.legal_columns() for _s, _sc, p in recs))
    s = ev.Solver()
    check('written scores are the solver scores',
          all(s.solve(p) == sc for _s, sc, p in recs))
    check('load_dir finds it', len(ev.load_dir(d)['Test_L3_R1']) == 40)
    try:
        ev.load_dir(tempfile.mkdtemp())
        check('an empty directory raises', False)
    except FileNotFoundError as e:
        check('an empty directory raises with the source URL',
              ev.SOURCE_URL in str(e))


def test_suite_and_oracle():
    print('\nSuite, cache, and the oracle upper bound')
    d = tempfile.mkdtemp()
    ev.generate(os.path.join(d, 'Test_L3_R1'), n=60, min_moves=30,
                max_moves=36, seed=8, log=None)
    suites = ev.Suite.build(d, log=lambda *a: None)
    s = suites['Test_L3_R1']
    check('bucket named', s.name == 'End-Easy')
    check('cache written', os.path.exists(os.path.join(d, 'solved_cache.json')))
    check('cached move scores reproduce the file score',
          not s.check_labels(log=lambda *a: None))

    # A second build must hit the cache, not re-solve.
    calls = []
    orig = ev.Solver.move_scores
    ev.Solver.move_scores = lambda self, p: (calls.append(1), orig(self, p))[1]
    try:
        ev.Suite.build(d, log=lambda *a: None)
    finally:
        ev.Solver.move_scores = orig
    check('a rebuild solves nothing (cache hit)', not calls, f'{len(calls)}')

    # Perfect play must score 100/100/0 -- this is what validates the metric.
    def oracle(states):
        return [int(max(s.truth[seq][1], key=lambda k: s.truth[seq][1][k]))
                for seq, _sc, _p in s.records]
    r = s.evaluate(oracle)
    check('oracle is 100% outcome-optimal', r['optimal'] == 1.0, f'{r}')
    check('oracle is 100% perfect', r['perfect'] == 1.0)
    check('oracle never blunders', r['blunder'] == 0.0)

    # The worst legal move must score strictly worse than the best.
    def worst(states):
        return [int(min(s.truth[seq][1], key=lambda k: s.truth[seq][1][k]))
                for seq, _sc, _p in s.records]
    rw = s.evaluate(worst)
    check('the worst move scores below the best', rw['optimal'] < r['optimal'])

    # An illegal move must be counted against the bot, not crash.
    ri = s.evaluate(lambda states: [99] * len(states))
    check('illegal moves score zero and are reported',
          ri['optimal'] == 0.0 and ri['legal'] == 0.0, f'{ri}')


def test_adapters():
    try:
        import torch                                            # noqa: F401
        import connect4_dirichlet_utils as c4
        import connect4_alphazero_utils as az
    except Exception:
        print('\n(torch/pyspiel missing — adapter tests skipped)')
        return
    if not c4._HAS_TORCH:
        return
    print('\nEngine adapters')
    d = tempfile.mkdtemp()
    ev.generate(os.path.join(d, 'Test_L3_R1'), n=50, min_moves=30,
                max_moves=36, seed=9, log=None)
    s = ev.Suite.build(d, log=lambda *a: None)['Test_L3_R1']
    game = c4.load_game()
    c4.set_game(game)
    c4.set_backend('cpu', device='cpu')
    c4.set_search(c4.AGG_ADDITIVE, c4.AGG_ADDITIVE, 'dirichlet', 1.0)
    import torch as _t
    _t.manual_seed(0)
    tz = c4.C4DirichletNet(16, 1, 4)
    tz.eval()
    azn = az.AlphaZeroNet(16, 1, 4)
    azn.eval()

    ch, va = ev.thompson_player(tz, 'cpu', sims=0)
    rt = s.evaluate(ch, va)
    check('thompson adapter returns legal moves', rt['legal'] == 1.0)
    check('value_acc is reported alongside its baseline',
          'value_acc' in rt and 'majority' in rt)

    ch, va = ev.alphazero_player(azn, 'cpu', sims=0)
    ra = s.evaluate(ch, va)
    check('alphazero adapter returns legal moves', ra['legal'] == 1.0)

    # THE fairness invariant.  Both engines' untrained nets are uninformative,
    # so a one-ply lookahead scores identically on both: whatever it gets comes
    # from seeing terminal children, not from the network.  If these two ever
    # diverge on untrained nets, the adapters are measuring different things and
    # cross-engine numbers are meaningless.
    ch_t, _ = ev.thompson_player(tz, 'cpu', sims=0, lookahead=True)
    ch_a, _ = ev.alphazero_player(azn, 'cpu', sims=0)
    lt, la = s.evaluate(ch_t), s.evaluate(ch_a)
    check('untrained one-ply lookahead scores the same on both engines',
          abs(lt['optimal'] - la['optimal']) < 1e-9,
          f"TZ {lt['optimal']:.3f} vs AZ {la['optimal']:.3f}")
    check('lookahead beats reading the action heads of an untrained net',
          lt['optimal'] > rt['optimal'],
          f"{lt['optimal']:.3f} vs {rt['optimal']:.3f}")

    ch, _ = ev.random_player(0)
    rr = s.evaluate(ch)
    check('random plays legally', rr['legal'] == 1.0)
    check('search beats no search on an untrained net',
          s.evaluate(ev.thompson_player(tz, 'cpu', sims=32)[0])['optimal']
          > rr['optimal'])


def test_value_probe():
    try:
        import numpy as np
        import connect4_dirichlet_utils as c4
    except Exception:
        print('\n(torch/pyspiel missing — value-probe tests skipped)')
        return
    if not c4._HAS_TORCH:
        return
    print('\nValue probe (per-category MSE)')
    d = tempfile.mkdtemp()
    ev.generate(os.path.join(d, 'Test_L3_R1'), n=60, min_moves=30,
                max_moves=36, seed=11, log=None)
    c4.set_game(c4.load_game())
    probe = ev.ValueProbe.build(d, log=lambda *a: None)
    check('probe has one row per position', len(probe) == 60)
    check('observations are the network input shape',
          probe.obs.shape == (60, int(np.prod(c4._OBS_SHAPE))),
          f'{probe.obs.shape}')
    check('true values are in {-1,0,1}',
          set(np.unique(probe.z)) <= {-1, 0, 1}, f'{np.unique(probe.z)}')
    check('cache file written',
          os.path.exists(os.path.join(d, 'value_probe.npz')))

    # A rebuild must reuse the cache rather than replaying sequences.
    calls = []
    orig = ev.states_for
    ev.states_for = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        p2 = ev.ValueProbe.build(d, log=lambda *a: None)
    finally:
        ev.states_for = orig
    check('a rebuild hits the cache', not calls, f'{len(calls)}')
    check('the cached probe is identical',
          np.array_equal(p2.z, probe.z) and np.array_equal(p2.obs, probe.obs))

    # Exact predictor -> zero MSE everywhere.  This is what pins the metric.
    r = probe.evaluate(lambda obs: probe.z.astype(float))
    check('a perfect predictor scores 0 in every category',
          r['mse'] == 0.0 and r['mse_macro'] == 0.0, f'{r}')

    # Constant 0: squared error is 1 on wins and losses, 0 on draws.
    r0 = probe.evaluate(lambda obs: np.zeros(len(probe)))
    check('constant 0 scores 1.0 on wins and losses',
          abs(r0['mse_win'] - 1.0) < 1e-12
          and abs(r0['mse_loss'] - 1.0) < 1e-12, f'{r0}')
    check('constant 0 scores 0 on draws',
          r0['n_draw'] == 0 or abs(r0['mse_draw']) < 1e-12)

    # Always-win: 0 on wins, 4 on losses.  The point of per-category reporting
    # is that this is obvious here and invisible in a pooled number on a
    # win-skewed set.
    r1 = probe.evaluate(lambda obs: np.ones(len(probe)))
    check('always-+1 scores 0 on wins but 4 on losses',
          abs(r1['mse_win']) < 1e-12 and abs(r1['mse_loss'] - 4.0) < 1e-12,
          f'{r1}')
    check('pooled MSE flatters always-+1 relative to its macro',
          r1['mse'] < r1['mse_macro'], f"{r1['mse']} vs {r1['mse_macro']}")
    check('mean prediction is reported per category',
          abs(r1['pred_win'] - 1.0) < 1e-9)
    check('categories partition the set',
          r1['n_win'] + r1['n_draw'] + r1['n_loss'] == len(probe))

    # A wrong-length prediction must fail loudly, not silently misalign.
    bad = False
    try:
        probe.evaluate(lambda obs: np.zeros(len(probe) - 1))
    except ValueError:
        bad = True
    check('a wrong-length prediction raises', bad)

    # Both engine adapters must run and stay in range.
    import torch
    import connect4_alphazero_utils as az
    c4.set_backend('cpu', device='cpu')
    torch.manual_seed(0)
    tz = c4.C4DirichletNet(16, 1, 4); tz.eval()
    azn = az.AlphaZeroNet(16, 1, 4); azn.eval()
    for name, fn in (('thompson', ev.thompson_value_fn(tz, 'cpu')),
                     ('alphazero', ev.alphazero_value_fn(azn, 'cpu'))):
        v = np.asarray(fn(probe.obs)).reshape(-1)
        check(f'{name} adapter returns one value per position',
              len(v) == len(probe))
        check(f'{name} values are within [-1, 1]',
              bool(np.all(np.abs(v) <= 1.0 + 1e-6)), f'{v.min()}..{v.max()}')
        check(f'{name} evaluate produces finite metrics',
              np.isfinite(probe.evaluate(fn)['mse_macro']))
    # An untrained net of either engine sits at value 0, so both must land on
    # the constant-0 reference.
    check('untrained nets match the constant-0 reference',
          abs(probe.evaluate(ev.alphazero_value_fn(azn, 'cpu'))['mse_macro']
              - r0['mse_macro']) < 1e-6)
    check('by_bucket splits the report',
          set(probe.by_bucket(ev.thompson_value_fn(tz, 'cpu'))) == {'End-Easy'})


def main():
    test_bitboard()
    test_half()
    test_score_convention()
    test_solver_against_brute_force()
    test_solver_consistency()
    test_parse_and_roundtrip()
    test_suite_and_oracle()
    test_adapters()
    test_value_probe()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all solved-eval tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
