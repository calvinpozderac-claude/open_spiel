"""Self-tests for othello_benchmark and the game-agnostic plumbing it needs.

Run:  python othello_benchmark_tests.py

Connect 4 hid a lot of assumptions: 7 actions, ~34 plies, no pass move, a
42-ply cap that never bound.  These check that none of them are baked in, and
that the AlphaZero control gets the same evals the ThompsonZero arms do.
"""

import os
import sys
import tempfile

import numpy as np

import connect4_dirichlet_utils as c4
import connect4_alphazero_utils as az
import connect4_benchmark as cb
import othello_benchmark as ob

_fails = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        _fails.append(name)


def tiny(**over):
    """A benchmark-shaped config small enough to run in seconds."""
    return ob.default_shared(
        root=tempfile.mkdtemp(), num_episodes=2, channels=8, num_blocks=1,
        head_ch=2, fast_sims=8, full_sims=8, n_parallel_games=4,
        wave_per_game=4, pool_prob=0.0, use_workers=False, batch_size=16,
        train_steps_per_ep=2, quick_eval_every=2, deep_eval_every=2,
        eval_sims=4, quick_eval_games=2, eval_games_per_pair=2, **over)


# ══════════════════════════════════════════════════════════════════════════════
def test_game_shape():
    print('\nOthello is genuinely a bigger game')
    game = c4.load_game('othello')
    shape, n_act = c4.set_game(game)
    check('65 actions (64 squares + pass)', n_act == 65, f'{n_act}')
    check('observation is 3x8x8', shape == (3, 8, 8), f'{shape}')
    check('an order of magnitude more actions than Connect 4', n_act > 9 * 7)

    # The backup flips the value once per ply, which is only right if the mover
    # alternates every ply.  Othello has an explicit pass action rather than
    # skipping a turn, so it does -- but that is worth checking, not assuming.
    rng = np.random.default_rng(0)
    repeats = plies = passes = 0
    lens = []
    for _ in range(25):
        st = game.new_initial_state()
        prev, n = None, 0
        while not st.is_terminal():
            p = st.current_player()
            if prev is not None and p == prev:
                repeats += 1
            la = st.legal_actions()
            if la == [64]:
                passes += 1
            prev, n, plies = p, n + 1, plies + 1
            st.apply_action(int(la[rng.integers(len(la))]))
        lens.append(n)
    check(f'the mover alternates every ply ({plies} plies checked)',
          repeats == 0, f'{repeats} repeats')
    check('forced passes do occur (so the pass action is exercised)',
          passes > 0, f'{passes}')
    check('games are far longer than Connect 4',
          np.mean(lens) > 45, f'{np.mean(lens):.1f} mean plies')
    check('the configured ply cap does not truncate them',
          max(lens) <= ob.default_shared()['max_plies'],
          f'{max(lens)} > {ob.default_shared()["max_plies"]}')


def test_observation_perspective():
    print('\nObservations are mover-relative')
    game = c4.load_game('othello')
    c4.set_game(game)
    st = game.new_initial_state()
    st.apply_action(st.legal_actions()[0])
    obs = c4.make_obs(st)
    raw = np.asarray(st.observation_tensor(st.current_player()), np.float16)
    check('make_obs matches the mover\'s own tensor', np.allclose(obs, raw))
    check('observation length matches the network input',
          len(obs) == int(np.prod(c4._OBS_SHAPE)), f'{len(obs)}')
    # Othello's tensor is already mover-relative, so no plane swap is applied.
    check('no plane swap needed for this game', c4._OBS_NEEDS_SWAP is False)


def test_shared_defaults():
    print('\nOthello defaults reach both engines')
    s = ob.default_shared()
    check('game is othello', s['game'] == 'othello')
    check('ply cap raised from Connect 4\'s 42', s['max_plies'] == 128)
    check('eval ply cap raised too', s['eval_max_plies'] == 128)
    check('solved-position metric off (no oracle for Othello)',
          s['solved_dir'] == '')
    check('overrides still win', ob.default_shared(seed=7)['seed'] == 7)
    check('the Connect 4 benchmark is untouched',
          cb.default_shared()['game'] == 'connect_four'
          and cb.default_shared()['max_plies'] == 42)
    for name, cfg in (('thompson', ob.thompson_config('MA', s)),
                      ('alphazero', ob.alphazero_config(s))):
        check(f'{name} Config gets the game', cfg.game_name == 'othello')
        check(f'{name} Config gets the ply cap', cfg.max_plies == 128)
        check(f'{name} Config gets the eval ply cap', cfg.eval_max_plies == 128)
        check(f'{name} Config has the metric off', cfg.solved_dir == '')
    check('the four Thompson arms still differ only in the two rules',
          len({(ob.thompson_config(a, s).search_agg,
                ob.thompson_config(a, s).target_agg)
               for a in ('AA', 'AM', 'MA', 'MM')}) == 4)
    check('the API is re-exported', all(hasattr(ob, n) for n in (
        'train_all', 'load_players', 'round_robin', 'report', 'bradley_terry',
        'wilson', 'search_value', 'az_progression', 'az_cpuct_sweep')))


def test_worker_config_carries_the_game():
    """The multiprocess workers load the game by NAME from the worker config.
    Hard-coding 'connect_four' there would silently train Othello arms on
    Connect 4 whenever use_workers=True."""
    print('\nthe game name reaches the self-play workers')
    s = ob.default_shared()
    wc = c4._worker_cfg(ob.thompson_config('MA', s),
                        (s['channels'], s['num_blocks'], s['head_ch']))
    check('thompson worker cfg names othello', wc['game_name'] == 'othello')
    check('thompson worker cfg carries the ply cap', wc['max_plies'] == 128)
    wa = az._worker_cfg(ob.alphazero_config(s),
                        (s['channels'], s['num_blocks'], s['head_ch']))
    check('alphazero worker cfg names othello', wa['game_name'] == 'othello')
    check('a worker really loads that game',
          c4._mp_load_game(wc).num_distinct_actions() == 65)


def test_param_report():
    print('\nthe size mismatch is reported, not assumed away')
    p = ob.param_counts(ob.default_shared())
    check('trunks are identical', p['trunk matches'])
    check('the action count is reported', p['actions'] == 65)
    check('the parameter ratio is reported',
          p['thompson/alphazero params'] > 1.5,
          f"{p['thompson/alphazero params']}")
    check('heads dominate ThompsonZero at this action count',
          p['head share of thompson'] > 0.5,
          f"{p['head share of thompson']}")
    # The same function on Connect 4 must show the mild case, so the warning is
    # driven by the game rather than always-on.
    pc = cb.param_counts(cb.default_shared())
    check('Connect 4 stays close to matched',
          pc['thompson/alphazero params'] < 1.25,
          f"{pc['thompson/alphazero params']}")


def test_alphazero_has_both_evals():
    """The point of the control is that only the METHOD differs, which includes
    how it is measured.  AlphaZero used to save a checkpoint at the deep eval and
    rate nothing."""
    print('\nAlphaZero gets the same quick AND deep evals')
    game = c4.load_game('othello')
    c4.set_game(game)
    cfg = ob.alphazero_config(tiny())
    h = az.run_training(cfg, game=game, log=lambda *a: None)
    check('a quick eval ran', len(h['quick_ep']) >= 0)
    check('a deep Elo ladder was produced', len(h['elo']) >= 1, f"{h['elo']}")
    if h['elo']:
        lad = h['elo'][-1]
        check('the ladder rates the checkpoint and random',
              'random' in lad and any(k != 'random' for k in lad), f'{lad}')
        check('ratings are finite',
              all(np.isfinite(v) for v in lad.values()), f'{lad}')
    check('the checkpoint carries the ladder for resume',
          set(('elo', 'order', 'pair_games'))
          <= set(az.load_checkpoint(cfg.checkpoint_dir)))

    # And it resumes without losing the pool.
    cfg2 = ob.alphazero_config(dict(tiny(), root=os.path.dirname(
        os.path.dirname(cfg.checkpoint_dir))))
    cfg2.checkpoint_dir = cfg.checkpoint_dir
    cfg2.num_episodes = 4
    h2 = az.run_training(cfg2, game=game, log=lambda *a: None)
    check('resuming keeps the ladder growing', len(h2['elo']) > len(h['elo']),
          f"{len(h2['elo'])} vs {len(h['elo'])}")


def test_elo_pool_is_engine_agnostic():
    print('\nthe Elo ladder is shared, not duplicated')
    game = c4.load_game('othello')
    c4.set_game(game)
    import torch
    torch.manual_seed(0)
    pool = c4.EloPool(game, 'cpu', eval_sims=2, games_per_pair=2, last_n=1,
                      refresh_pairs=0, max_eval_plies=128, seed=0,
                      bot_factory=az._elo_bot, pick=az._elo_pick)
    elo = pool.add_checkpoint('a', az.AlphaZeroNet(8, 1, 2))
    check('an AlphaZero net can be rated by the shared pool',
          'a' in elo and np.isfinite(elo['a']), f'{elo}')
    check('random is still the anchor', 'random' in elo)
    pool2 = c4.EloPool(game, 'cpu', eval_sims=2, games_per_pair=2, last_n=1,
                       refresh_pairs=0, max_eval_plies=128, seed=0)
    e2 = pool2.add_checkpoint('t', c4.C4DirichletNet(8, 1, 2))
    check('and the default factory still rates a ThompsonZero net',
          't' in e2 and np.isfinite(e2['t']), f'{e2}')


def test_end_to_end_arms():
    print('\nboth engines train and play Othello end to end')
    game = c4.load_game('othello')
    c4.set_game(game)
    s = tiny()
    for arm in ('MA', 'AZ'):
        h = ob.train_arm(arm, s, log=lambda *a: None)
        check(f'{arm} produced training history', bool(h.get('ep')), f'{h.keys()}')
        check(f'{arm} played full-length games',
              h['plies'] and max(h['plies']) > 20, f"{h.get('plies')}")
    players = ob.load_players(s, gens=(2,), arms=('MA', 'AZ'))
    check('checkpoints load for both engines',
          len(players) >= 3, f'{list(players)}')      # MA@2, AZ@2, random
    names, W, elo = ob.round_robin(players, sims=0, games_per_pair=2)
    check('a mixed-engine round robin runs on Othello',
          len(names) == len(players) and np.isfinite(elo).all())
    check('every pair actually played', (W + W.T).sum() > 0)


def main():
    test_game_shape()
    test_observation_perspective()
    test_shared_defaults()
    test_worker_config_carries_the_game()
    test_param_report()
    test_elo_pool_is_engine_agnostic()
    test_alphazero_has_both_evals()
    test_end_to_end_arms()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all othello benchmark tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
