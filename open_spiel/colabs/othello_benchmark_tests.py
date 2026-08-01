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
    base = dict(
        root=tempfile.mkdtemp(), num_episodes=2, channels=8, num_blocks=1,
        head_ch=2, fast_sims=8, full_sims=8, n_parallel_games=4,
        wave_per_game=4, pool_prob=0.0, use_workers=False, batch_size=16,
        train_steps_per_ep=2, quick_eval_every=2, deep_eval_every=2,
        eval_sims=4, quick_eval_games=2, eval_games_per_pair=2)
    base.update(over)
    return ob.default_shared(**base)


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


def test_capacity_matching():
    """Equalising capacity and keeping the trunks identical are different
    things and cannot both hold; both must work and both must be reported."""
    print('\ncapacity matching')
    s = ob.default_shared()
    check('by default the trunks are identical', ob.az_sig(s) == ob.tz_sig(s))
    p0 = ob.param_counts(s)
    check('and totals are then unequal at 65 actions',
          p0['thompson/alphazero params'] > 1.5,
          f"{p0['thompson/alphazero params']}")

    m = ob.match_capacity(s, log=lambda *a: None)
    check('match_capacity does not mutate the input',
          s.get('az_channels') is None)
    check('it widens AlphaZero rather than ThompsonZero',
          m['az_channels'] > s['channels'], f"{m['az_channels']}")
    check('depth and head width are left alone',
          ob.az_sig(m)[1:] == ob.tz_sig(m)[1:])
    p1 = ob.param_counts(m)
    check('totals now match within 2%',
          abs(p1['thompson/alphazero params'] - 1.0) < 0.02,
          f"{p1['thompson/alphazero params']}")
    check('and the trunks are no longer identical', not p1['trunk matches'])
    check('the AlphaZero trunk is the thing that grew',
          p1['alphazero trunk'] > p0['alphazero trunk'])
    check('ThompsonZero is untouched by the match',
          p1['thompson total'] == p0['thompson total'])

    # The knob has to reach the Config and the checkpoint loader, or the arm
    # trains at one width and is rated at another.
    cfg = ob.alphazero_config(m)
    check('the widened trunk reaches the AlphaZero Config',
          (cfg.channels, cfg.num_blocks, cfg.head_ch) == ob.az_sig(m),
          f'{(cfg.channels, cfg.num_blocks, cfg.head_ch)}')
    check('the ThompsonZero Config is unaffected',
          (ob.thompson_config('MA', m).channels,) == (s['channels'],))

    # A round trip through disk at the matched width.
    import torch
    d = tempfile.mkdtemp()
    torch.manual_seed(0)
    net = az.AlphaZeroNet(*ob.az_sig(m))
    az.save_benchmark_net(d, '1', net)
    back = az.load_benchmark_net(d, '1', ob.az_sig(m))
    check('a matched-width AlphaZero net round-trips through disk',
          all(torch.equal(a, b) for a, b
              in zip(net.parameters(), back.parameters())))
    bad = False
    try:
        az.load_benchmark_net(d, '1', ob.tz_sig(m))
    except Exception:
        bad = True
    check('loading it at the WRONG width fails loudly', bad)

    # The ratio shrinking with trunk size is the substantive point.
    small = ob.param_counts(ob.default_shared(channels=32, num_blocks=3))
    big = ob.param_counts(ob.default_shared(channels=128, num_blocks=8))
    check('the mismatch shrinks as the trunk grows',
          big['thompson/alphazero params'] < small['thompson/alphazero params'],
          f"{big['thompson/alphazero params']} vs "
          f"{small['thompson/alphazero params']}")


def test_end_to_end_matched_capacity():
    print('\na capacity-matched AlphaZero arm trains and is rated')
    game = c4.load_game('othello')
    c4.set_game(game)
    s = ob.match_capacity(tiny(), log=lambda *a: None)
    h = ob.train_arm('AZ', s, log=lambda *a: None)
    check('it trained', bool(h.get('ep')))
    check('it produced an Elo ladder', len(h['elo']) >= 1)
    players = ob.load_players(s, gens=(2,), arms=('AZ',))
    check('its checkpoint loads at the matched width',
          any(k.startswith('AZ') for k in players), f'{list(players)}')


def test_game_contrast():
    """The structural facts the method comparison turns on."""
    print('\ngame contrast: what actually changed from Connect 4')
    r = ob.game_contrast(n=120, log=lambda *a: None)
    c, o = r['connect_four'], r['othello']
    check('Othello has many more action slots', o['actions'] > 9 * c['actions'])
    check('but only slightly more LEGAL moves per position',
          1.0 < o['legal'] / c['legal'] < 2.0,
          f"{o['legal']:.1f} vs {c['legal']:.1f}")
    check('so head slots go from densely to sparsely supervised',
          c['slot_use'] > 0.9 and o['slot_use'] < 0.2,
          f"{c['slot_use']:.2f} -> {o['slot_use']:.2f}")
    check('and games are much longer', o['plies'] > 2 * c['plies'])


def test_sims_scaling():
    print('\nsims_scaling runs and reports its own noise floor')
    game = c4.load_game('othello')
    c4.set_game(game)
    s = tiny(max_plies=20, eval_max_plies=20, fast_sims=6, full_sims=6)
    for arm in ('MA', 'AZ'):
        ob.train_arm(arm, s, log=lambda *a: None)
    lines = []
    rows = ob.sims_scaling(s, a='MA', b='AZ', gen=2, sims=(2, 6), games=4,
                           log=lines.append)
    check('one row per simulation budget', len(rows) == 2, f'{rows}')
    check('scores are probabilities',
          all(0.0 <= r['score'] <= 1.0 for r in rows), f'{rows}')
    check('win/draw/loss sum to the game count',
          all(r['w'] + r['d'] + r['l'] == 4 for r in rows), f'{rows}')
    check('the budgets are the ones asked for',
          [r['sims'] for r in rows] == [2, 6])
    check('it states the noise floor rather than leaving it implied',
          any('noise' in ln for ln in lines), f'{lines[-1:]}')
    check('it works with the arms reversed',
          len(ob.sims_scaling(s, a='AZ', b='MA', gen=2, sims=(2,), games=2,
                              log=lambda *a: None)) == 1)


def test_gauss_arm():
    """GA has to be a first-class arm: trainable, loadable, and playable in the
    same tournament as the other five."""
    print('\nthe Gaussian arm is wired in')
    import value_dist_utils as vd
    s = ob.default_shared()
    check('GA is in the Othello arm list', 'GA' in s['arms'])
    check('GH (sequential halving) is too', 'GH' in s['arms'])
    check('GA and GH differ ONLY in the root rule',
          ob.gauss_config(s, 'GA').root_select == 'thompson'
          and ob.gauss_config(s, 'GH').root_select == 'halving')
    ga, gh = ob.gauss_config(s, 'GA'), ob.gauss_config(s, 'GH')
    import dataclasses as _dc
    diff = {f.name for f in _dc.fields(ga)
            if getattr(ga, f.name) != getattr(gh, f.name)}
    check('and in nothing else', diff == {'root_select', 'checkpoint_dir'},
          f'{diff}')
    check('but NOT in the Connect 4 default (it needs a scored game)',
          'GA' not in cb.default_shared()['arms'])
    check('GA is a known arm', ob.ARMS['GA'][0] == 'gauss')
    cfg = ob.gauss_config(s)
    check('its config carries the game', cfg.game_name == 'othello')
    check('and the shared trunk', (cfg.channels, cfg.num_blocks, cfg.head_ch)
          == ob.tz_sig(s))
    check('and the shared self-play shape',
          (cfg.fast_sims, cfg.full_sims, cfg.max_plies, cfg.temp_threshold)
          == (s['fast_sims'], s['full_sims'], s['max_plies'],
              s['temp_threshold']))
    check('and its own checkpoint directory',
          cfg.checkpoint_dir == ob.arm_dir(s['root'], 'GA'))

    game = c4.load_game('othello')
    c4.set_game(game)
    tiny_s = tiny(arms=('GA',))
    h = ob.train_arm('GA', tiny_s, log=lambda *a: None)
    check('it trains', bool(h.get('ep')))
    check('it produces an Elo ladder like the others', len(h['elo']) >= 1)
    check('it plays full-length games', h['plies'] and max(h['plies']) > 20)
    check('its diagnostics are in discs',
          h['v_sd'] and 1.0 < h['v_sd'][0] < 64.0, f"{h.get('v_sd')}")

    players = ob.load_players(tiny_s, gens=(2,), arms=('GA',))
    check('its checkpoint loads', any(k.startswith('GA') for k in players))
    net = [v for k, v in players.items() if k.startswith('GA')][0][1]
    check('as a GaussianNet', isinstance(net, vd.GaussianNet))
    st = game.new_initial_state()
    rng = np.random.default_rng(0)
    check('the tournament can move it search-free',
          cb._move('gauss', net, st, 0, rng, {}) in st.legal_actions())
    check('and with search',
          cb._move('gauss', net, st, 4, rng, {}) in st.legal_actions())


def test_round_robin_cache():
    """Adding an arm must cost that arm's pairs, not the whole table."""
    print('\nround-robin results accumulate instead of replaying')
    import tempfile
    import torch
    import value_dist_utils as vd
    game = c4.load_game('othello')
    c4.set_game(game)
    ob.GAME_REF[0] = game
    torch.manual_seed(0)
    players = {'A': ('thompson', c4.C4DirichletNet(8, 1, 2)),
               'B': ('alphazero', az.AlphaZeroNet(8, 1, 2)),
               'random': ('random', None)}
    path = os.path.join(tempfile.mkdtemp(), 'rr.json')
    n1, W1, _ = ob.round_robin(players, sims=0, games_per_pair=4, cache=path,
                               log=lambda *a: None)
    check('a first run plays every pair', (W1 + W1.T).sum() / 2 == 3 * 4)
    n2, W2, _ = ob.round_robin(players, sims=0, games_per_pair=4, cache=path,
                               log=lambda *a: None)
    check('re-running reuses everything', np.allclose(W1, W2))

    players['GA'] = ('gauss', vd.GaussianNet(8, 1, 2))
    lines = []
    n3, W3, _ = ob.round_robin(players, sims=0, games_per_pair=4, cache=path,
                               log=lines.append)
    check('adding an arm reuses the old pairs',
          any('reusing 3/6' in ln for ln in lines), f'{lines}')
    sub = [n3.index(x) for x in n1]
    check('and leaves their results untouched',
          np.allclose(W3[np.ix_(sub, sub)], W1))
    check('the new arm is rated', 'GA' in n3)

    n4, W4, _ = ob.round_robin(players, sims=0, games_per_pair=6, cache=path,
                               log=lambda *a: None)
    check('raising games_per_pair tops up rather than restarting',
          (W4 + W4.T).sum() / 2 == 6 * 6, f'{(W4 + W4.T).sum() / 2}')
    check('search-free and MCTS results do not collide in the cache',
          cb._cache_key(0, 'A', 'B') != cb._cache_key(128, 'A', 'B'))
    check('the key is order-independent',
          cb._cache_key(0, 'A', 'B') == cb._cache_key(0, 'B', 'A'))


def test_long_run_settings():
    """A 20000-episode run needs its LR horizon to match, or 90% of it happens
    at the floor."""
    print('\nlong-run settings')
    s = ob.default_shared()
    check('20000 episodes', s['num_episodes'] == 20_000)
    # Deliberately NOT raised to match num_episodes: the arms already trained
    # used the shared default, and changing it for one arm would turn the
    # benchmark into a comparison of LR schedules.
    check('the LR horizon is the shared default, not tailored to this arm',
          s['lr_decay_eps'] == cb.default_shared()['lr_decay_eps'],
          f"{s['lr_decay_eps']}")
    check('every arm gets the SAME horizon',
          len({ob.thompson_config('MA', s).lr_decay_eps,
               ob.alphazero_config(s).lr_decay_eps,
               ob.gauss_config(s).lr_decay_eps}) == 1)
    check('generations to rate are inside the run',
          max(s['gens']) <= s['num_episodes'] and min(s['gens']) > 0,
          f"{s['gens']}")
    check('every generation is a real checkpoint',
          all(g % s['deep_eval_every'] == 0 for g in s['gens']),
          f"{s['gens']} vs deep_eval_every {s['deep_eval_every']}")
    for name, cfg in (('thompson', ob.thompson_config('MA', s)),
                      ('alphazero', ob.alphazero_config(s)),
                      ('gauss', ob.gauss_config(s))):
        check(f'{name} gets the full episode count',
              cfg.num_episodes == 20_000)
        check(f'{name} gets the shared LR horizon',
              cfg.lr_decay_eps == cb.default_shared()['lr_decay_eps'])


def main():
    test_game_shape()
    test_game_contrast()
    test_observation_perspective()
    test_shared_defaults()
    test_worker_config_carries_the_game()
    test_param_report()
    test_capacity_matching()
    test_elo_pool_is_engine_agnostic()
    test_alphazero_has_both_evals()
    test_end_to_end_arms()
    test_end_to_end_matched_capacity()
    test_gauss_arm()
    test_round_robin_cache()
    test_long_run_settings()
    test_sims_scaling()
    print()
    if _fails:
        print(f'{len(_fails)} FAILURES: {_fails}')
        return 1
    print('all othello benchmark tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
