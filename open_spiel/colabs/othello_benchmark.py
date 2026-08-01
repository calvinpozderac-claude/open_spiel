"""Othello benchmark: the same five arms, on a bigger game.

Connect 4 has 7 actions and ~34-ply games.  Othello has 65 (64 squares plus an
explicit pass) and ~60-ply games, which is the point: the evidence rules were
compared on a game where a 50-simulation search visits every legal move several
times, and it is not obvious the ranking survives a branching factor an order of
magnitude larger.

Everything here is `connect4_benchmark` with Othello-shaped defaults — same five
arms, same matched-control discipline, same round robin and Bradley-Terry fit.
There is no solved-position metric: that one needs an exact oracle, and no
equivalent of Pons' Connect 4 test sets exists for Othello.

    import othello_benchmark as bench
    shared = bench.default_shared(num_episodes=4000)
    hists  = bench.train_all(shared)
    players = bench.load_players(shared, gens=(1000, 2000, 4000))
    names, W, elo = bench.round_robin(players, sims=0, games_per_pair=60)
    bench.report(names, W, elo, 'search-free')

What changes from Connect 4, and why
------------------------------------
`max_plies` / `eval_max_plies` 128
    Connect 4's 42 would truncate more than half of every Othello game and score
    it as a draw.  128 is the game's own max_game_length.
`temp_threshold` 30
    Sampling from the visit distribution for the first ~half of the game, as in
    Connect 4 (12 of ~34 plies).
`fast_sims` / `full_sims` 100 / 300
    Raised with the branching factor.  Both engines record a policy/value target
    at every move, and 75% of games run at `fast_sims` throughout, so this is
    the number of simulations most training targets are built from.  At 50 sims
    over ~10 legal moves a visit-count target is mostly noise — and that hurts
    the AlphaZero arm most, since visit counts are its ONLY policy signal, while
    the ThompsonZero arms also train per-action value heads and a cross-entropy
    on the move actually played.  Matched across all five arms either way.
`root_noise_alpha` 0.6
    AlphaZero scales Dirichlet noise inversely with the branching factor (0.3
    for chess at ~35 moves, 0.03 for Go at ~250).  Othello averages ~10 legal
    moves against Connect 4's ~6, so the Connect 4 value of 1.0 comes down.
`eval_sims` 64
    A deeper search per eval game, since a 32-simulation search over 10 moves
    separates two networks less cleanly than over 4.
"""

import connect4_benchmark as _b

# Re-export the whole benchmark API unchanged -- train_all, load_players,
# round_robin, report, bradley_terry, wilson, the AlphaZero diagnostics.  Only
# default_shared differs, and it is redefined below.
_REEXPORT = [n for n in dir(_b) if not n.startswith('_')]
globals().update({n: getattr(_b, n) for n in _REEXPORT})
__all__ = _REEXPORT + ['GAME', 'OTHELLO_DEFAULTS', 'describe']

GAME = 'othello'

#  Othello-shaped overrides; everything not listed keeps its Connect 4 value so
#  the two benchmarks stay comparable wherever the game does not force a change.
OTHELLO_DEFAULTS = dict(
    game=GAME,
    root='othello_benchmark',
    # The five original arms plus GA, the Gaussian value-distribution engine,
    # which needs a game with a natural score and so only appears here.
    arms=('AA', 'AM', 'MA', 'MM', 'AZ', 'GA'),
    num_episodes=20_000,
    # NOTE lr_decay_eps is deliberately left at the shared default (2000) rather
    # than raised to match num_episodes.  The cosine therefore completes at
    # episode 2000 and the remaining 90% of the run trains at the 10% floor.
    # That is a real cost, but the already-trained arms were run that way and
    # changing it for one arm would make the comparison a comparison of LR
    # schedules.  Raise it for ALL arms together, on a fresh set of runs.
    max_plies=128,          # the game's own max_game_length; 42 would truncate
    eval_max_plies=128,
    temp_threshold=30,      # ~half of a ~60-ply game, as 12 was for Connect 4
    fast_sims=100,          # raised with the branching factor (see module docs)
    full_sims=300,
    root_noise_alpha=0.6,   # ~10 legal moves vs Connect 4's ~6
    eval_sims=64,
    # Checkpoints land every deep_eval_every (1000), so these are the round-robin
    # generations worth rating on a 20000-episode run.
    gens=(5000, 10000, 15000, 20000),
    # No solved-position metric: there is no Othello equivalent of Pons' exact
    # test sets, so the absolute-strength eval is off and the ladder is the
    # relative one.
    solved_dir='',
)


def default_shared(**over):
    """Everything the five arms hold in common, with Othello defaults.

    Any keyword overrides, exactly as in the Connect 4 benchmark."""
    return _b.default_shared(**{**OTHELLO_DEFAULTS, **over})


def describe(log=print):
    """What this game looks like, and how the settings follow from it."""
    import connect4_dirichlet_utils as c4
    import numpy as np
    game = c4.load_game(GAME)
    shape, n_act = c4.set_game(game)
    rng = np.random.default_rng(0)
    lens, branch = [], []
    for _ in range(30):
        st = game.new_initial_state()
        n = 0
        while not st.is_terminal():
            la = st.legal_actions()
            branch.append(len(la))
            st.apply_action(int(la[rng.integers(len(la))]))
            n += 1
        lens.append(n)
    s = default_shared()
    log(f'{game.get_type().long_name}: {n_act} actions, observation {shape}, '
        f'max_game_length {game.max_game_length()}')
    log(f'  random games: {np.mean(lens):.0f} plies mean '
        f'({min(lens)}-{max(lens)}), {np.mean(branch):.1f} legal moves mean')
    log(f'  configured: max_plies={s["max_plies"]} '
        f'temp_threshold={s["temp_threshold"]} '
        f'sims={s["fast_sims"]}/{s["full_sims"]} '
        f'noise_alpha={s["root_noise_alpha"]} eval_sims={s["eval_sims"]}')
    return dict(actions=n_act, obs_shape=shape, mean_plies=float(np.mean(lens)),
                mean_branch=float(np.mean(branch)))


#  Candidate trunks, smallest first.  8x8 board games at this scale are usually
#  run at 64-128 channels and 5-10 residual blocks; the 32/3 default carried over
#  from Connect 4 is below that range, and Connect 4 is a solved 10^13-state game
#  where Othello is ~10^28.
SIZES = ((32, 3, 8), (48, 4, 8), (64, 5, 8), (64, 5, 16),
         (96, 6, 16), (128, 8, 16), (128, 10, 32))


def sizing_table(shared=None, sizes=SIZES, games_per_arm=4000, n_arms=5,
                 batch=128, log=print):
    """Parameters and measured forward cost per candidate trunk, plus what each
    would cost to actually run.

    The throughput is measured HERE, so on a different device only the ratios
    carry over — but the ratios are the part that decides the size.  The
    games/hour column is anchored to 0.65 games/s, the rate the Connect 4 runs
    achieved on an RX 5700 XT at 32/3/8, scaled by this game's evaluations per
    game and the measured relative cost."""
    import time
    import torch
    import connect4_alphazero_utils as _az
    import connect4_dirichlet_utils as _c4
    shared = shared or default_shared()
    _c4.set_game(_c4.load_game(shared['game']))

    def fwd_ms(net):
        net.eval()
        x = torch.randn(batch, *_c4._OBS_SHAPE)
        with torch.inference_mode():
            for _ in range(3):
                net(x)
            t0 = time.perf_counter()
            for _ in range(6):
                net(x)
        return (time.perf_counter() - t0) / 6 * 1000

    # Evaluations per self-play game, this game vs the Connect 4 reference.
    ref_evals = 34 * (0.75 * 50 + 0.25 * 150)
    evals = shared['max_plies'] and (
        60 * (0.75 * shared['fast_sims'] + 0.25 * shared['full_sims']))
    ref_rate = 0.65 * ref_evals            # NN evals/s observed at 32/3/8

    log(f'{"trunk":>12}{"TZ params":>11}{"AZ params":>11}{"TZ/AZ":>7}'
        f'{"rel cost":>10}{"games/s":>9}{"h/arm":>8}{"h x %d" % n_arms:>9}')
    base, rows = None, []
    for sig in sizes:
        tz = _c4.C4DirichletNet(*sig)
        a = _az.AlphaZeroNet(*sig)
        ms = fwd_ms(tz)
        base = base or ms
        rel = ms / base
        ntz = sum(p.numel() for p in tz.parameters())
        naz = sum(p.numel() for p in a.parameters())
        g = ref_rate / (evals * rel * (64 / 42))
        h = games_per_arm / g / 3600
        rows.append(dict(sig=sig, tz=ntz, az=naz, rel=rel, games_s=g, hours=h))
        log(f'{sig[0]}/{sig[1]}/{sig[2]}'.rjust(12)
            + f'{ntz:>11,}{naz:>11,}{ntz / naz:>7.2f}{rel:>9.1f}x'
              f'{g:>9.3f}{h:>8.1f}{h * n_arms:>9.0f}')
    log('  hours are indicative (tree overhead, batching and DirectML quirks '
        'are not modelled); the ratios are the reliable part.')
    log('  note TZ/AZ falls as the trunk grows: the head cost is fixed by the '
        'action count while the trunk scales, so a size suited to Othello is '
        'much closer to matched than 32/3/8 is.')
    return rows
