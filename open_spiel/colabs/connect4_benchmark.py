"""Benchmark harness: the four additive/additive_mle combinations, plus an
AlphaZero control, trained under matched settings and rated by a round-robin
tournament.

The notebook (`connect4_benchmark.ipynb`) is only a Config and two calls; all
the machinery lives here.

WHY THESE FIVE ARMS.  Search and training targets each independently take one of
two evidence rules, giving four cells:

                    target = additive        target = additive_mle
    search=additive       AA                        AM
    search=additive_mle   MA                        MM

'additive' makes a belief's concentration exactly the visit count; 'additive_mle'
fits a Dirichlet to the same observations by maximum likelihood, so the
concentration tracks how much the backed-up leaves actually disagree.  The two
roles want different things from that number — search wants it to grow so
exploration anneals, targets want it to mean something the network can predict —
so the off-diagonal cells are as interesting as the diagonal ones.

The AlphaZero arm is the control that says whether any of it beats the ordinary
algorithm.  It is matched on trunk, observation, self-play shape, solver, sim
counts, optimiser, batch size and steps per episode; it differs only in method
(policy + scalar value, PUCT, visit-count targets, CE + MSE).  See
`connect4_alphazero_utils` for the full list of what is held fixed.

Ratings come from a Bradley-Terry maximum-likelihood fit over the whole result
matrix rather than sequential Elo updates, which depend on match order and on a
K-factor schedule.  With a full round robin there is no reason to accept that
noise.
"""

import itertools
import json
import os
import time

import numpy as np

import connect4_dirichlet_utils as c4
import connect4_alphazero_utils as az
import value_dist_utils as vd


# ══════════════════════════════════════════════════════════════════════════════
#  Arm definitions
# ══════════════════════════════════════════════════════════════════════════════
# name -> (engine, search_agg, target_agg, human-readable description)
# (engine, search rule, target rule, description, backup rule).
#
# 'MS' is 'MM' with ONE thing changed: a simulation carries a DRAW from the leaf
# Dirichlet rather than its mean.  MM is the arm this matters most for, because
# 'additive_mle' fits a Dirichlet to the observations a node collects and reads
# its concentration off their dispersion -- and with mean backups the only
# dispersion available is disagreement BETWEEN leaves, so a node whose leaves
# all shrug in the same direction is fitted as certain.  Sampling puts each
# leaf's own uncertainty into that dispersion, where the fit can see it.
ARMS = {
    'AA':  ('thompson', c4.AGG_ADDITIVE,     c4.AGG_ADDITIVE,
            'search additive     · target additive', c4.BACKUP_MEAN),
    'AM':  ('thompson', c4.AGG_ADDITIVE,     c4.AGG_ADDITIVE_MLE,
            'search additive     · target additive_mle', c4.BACKUP_MEAN),
    'MA':  ('thompson', c4.AGG_ADDITIVE_MLE, c4.AGG_ADDITIVE,
            'search additive_mle · target additive', c4.BACKUP_MEAN),
    'MM':  ('thompson', c4.AGG_ADDITIVE_MLE, c4.AGG_ADDITIVE_MLE,
            'search additive_mle · target additive_mle', c4.BACKUP_MEAN),
    'MS':  ('thompson', c4.AGG_ADDITIVE_MLE, c4.AGG_ADDITIVE_MLE,
            'search additive_mle · target additive_mle · SAMPLED backup',
            c4.BACKUP_SAMPLE),
    'AZ':  ('alphazero', None, None,
            'AlphaZero control (policy + scalar value, PUCT)', None),
    'GA':  ('gauss', 'thompson', None,
            'Gaussian value dist · Thompson root', None),
    'GH':  ('gauss_halving', 'halving', None,
            'Gaussian value dist · sequential-halving root (Gumbel-AZ style)',
            None),
}


# Where the solved-position test files live by default: a sibling of the arms'
# checkpoint directories, so one benchmark root holds everything.
#     <root>/bench_AA/ ... <root>/bench_AZ/ ... <root>/solved_tests/Test_L3_R1
SOLVED_SUBDIR = 'solved_tests'


def tz_sig(shared):
    """(channels, num_blocks, head_ch) for the ThompsonZero arms."""
    return (shared['channels'], shared['num_blocks'], shared['head_ch'])


def arms_of(shared, arms=None):
    """Explicit `arms` wins, else the benchmark's own list, else everything."""
    return list(arms or shared.get('arms') or ARMS)


def gauss_config(shared, name='GA'):
    """A Gaussian value-distribution arm.  Matched to the others on trunk,
    self-play shape, optimiser, schedule, batch and evals; GA and GH differ only
    in how the ROOT spends its simulations."""
    _ch, _bl, _hd = tz_sig(shared)
    return vd.Config(
        game_name=shared['game'],
        root_select=ARMS[name][1] or 'thompson',
        max_considered=shared.get('max_considered', 16),
        head=shared.get('gauss_head', 'spatial'),
        ev_weight=shared.get('ev_weight', 'uniform'),
        checkpoint_dir=arm_dir(shared['root'], name),
        num_episodes=shared['num_episodes'],
        channels=_ch, num_blocks=_bl, head_ch=_hd, seed=shared['seed'],
        device_preference=shared['device'],
        fast_sims=shared['fast_sims'], full_sims=shared['full_sims'],
        fast_prob=shared['fast_prob'], temp_threshold=shared['temp_threshold'],
        max_plies=shared['max_plies'], eval_max_plies=shared['eval_max_plies'],
        n_parallel_games=shared['n_parallel_games'],
        wave_per_game=shared['wave_per_game'],
        batch_size=shared['batch_size'],
        train_steps_per_ep=shared['train_steps_per_ep'],
        max_buffer=shared['max_buffer'],
        lr_peak=shared['lr_peak'], lr_decay_eps=shared['lr_decay_eps'],
        weight_decay=shared['weight_decay'], grad_clip=shared['grad_clip'],
        quick_eval_every=shared['quick_eval_every'],
        quick_eval_games=shared['quick_eval_games'],
        deep_eval_every=shared['deep_eval_every'],
        eval_sims=shared['eval_sims'],
        eval_games_per_pair=shared.get('eval_games_per_pair', 4),
        resume=shared['resume'])


def az_sig(shared):
    """Same for AlphaZero.  Defaults to ThompsonZero's, so the trunks are
    identical unless the capacity knobs below are set."""
    return (shared.get('az_channels') or shared['channels'],
            shared.get('az_num_blocks') or shared['num_blocks'],
            shared.get('az_head_ch') or shared['head_ch'])


def _total_params(cls, sig):
    return sum(p.numel() for p in cls(*sig).parameters())


def match_capacity(shared, log=print, max_channels=1024):
    """Return a copy of `shared` with AlphaZero widened to ThompsonZero's total
    parameter count.

    ThompsonZero emits 4 numbers per action against AlphaZero's one, so at a
    large action count its output layer alone can outweigh the whole trunk: on
    Othello at 32 channels it carries 2.09x AlphaZero's parameters.  Matching by
    padding AlphaZero's HEAD would add parameters where they do nothing -- a
    wider output projection for the same 66 numbers -- so this widens its TRUNK
    instead, which is where capacity is actually used.  Depth and head width are
    left alone so the two networks stay the same shape.

    The trunks are then no longer identical, which was the previous definition of
    matched.  Both definitions are defensible and they cannot both hold; this
    picks equal capacity, and report_params prints whichever way it ends up.

    Note the mismatch is largely an artifact of a small trunk: the head cost is
    fixed while the trunk grows, so the ratio falls from 2.09 at 32 channels to
    1.08 at 128.  At a size actually suited to Othello there is little to match.
    """
    import connect4_alphazero_utils as _az
    import connect4_dirichlet_utils as _c4
    game = _c4.load_game(shared['game'])
    _c4.set_game(game)
    target = _total_params(_c4.C4DirichletNet, tz_sig(shared))
    blocks, head = shared['num_blocks'], shared['head_ch']
    best, best_err = shared['channels'], None
    ch = max(4, shared['channels'])
    while ch <= max_channels:
        err = abs(_total_params(_az.AlphaZeroNet, (ch, blocks, head)) - target)
        if best_err is None or err < best_err:
            best, best_err = ch, err
        if _total_params(_az.AlphaZeroNet, (ch, blocks, head)) > target:
            break
        ch += 1
    out = dict(shared, az_channels=best)
    got = _total_params(_az.AlphaZeroNet, (best, blocks, head))
    log(f'capacity match: AlphaZero {shared["channels"]} -> {best} channels '
        f'({got:,} params vs ThompsonZero {target:,}, '
        f'{100 * abs(got - target) / target:.1f}% apart)')
    return out


def arm_dir(root, name):
    return os.path.join(root, f'bench_{name}')


def solved_dir(root):
    return os.path.join(root, SOLVED_SUBDIR)


def thompson_config(name, shared):
    """A ThompsonZero Config from the shared settings plus this arm's rules."""
    _eng, sagg, tagg, _d, bk = ARMS[name]
    return c4.Config(
        game_name=shared['game'],
        checkpoint_dir=arm_dir(shared['root'], name),
        num_episodes=shared['num_episodes'],
        channels=shared['channels'], num_blocks=shared['num_blocks'],
        head_ch=shared['head_ch'], seed=shared['seed'],
        device_preference=shared['device'],
        search_agg=sagg, target_agg=tagg, backup=bk, kl_normalize=False,
        cons_frac=shared['cons_frac'],
        selection='dirichlet',
        fast_sims=shared['fast_sims'], full_sims=shared['full_sims'],
        fast_prob=shared['fast_prob'], temp_threshold=shared['temp_threshold'],
        max_plies=shared['max_plies'], eval_max_plies=shared['eval_max_plies'],
        use_workers=shared['use_workers'],
        selfplay_workers=shared['workers'],
        games_per_worker=shared['games_per_worker'],
        worker_wave=shared['worker_wave'],
        n_parallel_games=shared['n_parallel_games'],
        wave_per_game=shared['wave_per_game'],
        pool_prob=shared['pool_prob'],
        batch_size=shared['batch_size'],
        train_steps_per_ep=shared['train_steps_per_ep'],
        max_buffer=shared['max_buffer'],
        lr_peak=shared['lr_peak'], lr_decay_eps=shared['lr_decay_eps'],
        weight_decay=shared['weight_decay'], grad_clip=shared['grad_clip'],
        quick_eval_every=shared['quick_eval_every'],
        quick_eval_games=shared['quick_eval_games'],
        deep_eval_every=shared['deep_eval_every'],
        eval_sims=shared['eval_sims'],
        solved_dir=shared['solved_dir'],
        solved_every=shared['solved_every'],
        solved_n=shared['solved_n'],
        resume=shared['resume'])


def alphazero_config(shared):
    _ch, _bl, _hd = az_sig(shared)
    return az.Config(
        game_name=shared['game'],
        checkpoint_dir=arm_dir(shared['root'], 'AZ'),
        num_episodes=shared['num_episodes'],
        channels=_ch, num_blocks=_bl, head_ch=_hd, seed=shared['seed'],
        device_preference=shared['device'],
        c_puct=shared['c_puct'],
        root_noise_frac=shared['root_noise_frac'],
        root_noise_alpha=shared['root_noise_alpha'],
        fast_sims=shared['fast_sims'], full_sims=shared['full_sims'],
        fast_prob=shared['fast_prob'], temp_threshold=shared['temp_threshold'],
        max_plies=shared['max_plies'], eval_max_plies=shared['eval_max_plies'],
        use_workers=shared['use_workers'],
        selfplay_workers=shared['workers'],
        games_per_worker=shared['games_per_worker'],
        worker_wave=shared['worker_wave'],
        n_parallel_games=shared['n_parallel_games'],
        wave_per_game=shared['wave_per_game'],
        pool_prob=shared['pool_prob'],
        batch_size=shared['batch_size'],
        train_steps_per_ep=shared['train_steps_per_ep'],
        max_buffer=shared['max_buffer'],
        lr_peak=shared['lr_peak'], lr_decay_eps=shared['lr_decay_eps'],
        weight_decay=shared['weight_decay'], grad_clip=shared['grad_clip'],
        quick_eval_every=shared['quick_eval_every'],
        quick_eval_games=shared['quick_eval_games'],
        deep_eval_every=shared['deep_eval_every'],
        eval_sims=shared['eval_sims'],
        solved_dir=shared['solved_dir'],
        solved_every=shared['solved_every'],
        solved_n=shared['solved_n'],
        resume=shared['resume'])


def default_shared(**over):
    """Everything the five arms hold in common.  Override any of it by keyword."""
    s = dict(
        game='connect_four', root='c4_benchmark',
        # Which arms this benchmark runs.  GA (the Gaussian value-distribution
        # engine) needs a game with a natural score and is therefore not in the
        # Connect 4 default; othello_benchmark adds it.
        arms=('AA', 'AM', 'MA', 'MM', 'AZ'),
        max_considered=16,
        gauss_head='spatial',
        ev_weight='uniform',
        # Generations the round robins rate, and where the pairwise results are
        # cached so re-running tops up instead of replaying.
        gens=(1000, 2000),
        rr_cache='round_robin.json',
        # AlphaZero network size.  None = identical to the ThompsonZero trunk.
        # bench.match_capacity(shared) sets az_channels to equalise total
        # parameters instead; see its docstring for the trade-off.
        az_channels=None, az_num_blocks=None, az_head_ch=None,
        num_episodes=2000, seed=0, device='auto',
        max_plies=42, eval_max_plies=42,
        channels=32, num_blocks=3, head_ch=8,
        fast_sims=50, full_sims=150, fast_prob=0.75, temp_threshold=12,
        use_workers=True, workers=0, games_per_worker=16, worker_wave=4,
        n_parallel_games=16, wave_per_game=4, pool_prob=0.2,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_decay_eps=2000, weight_decay=1e-4, grad_clip=1.0,
        cons_frac=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=32, resume=True,
        c_puct=1.5, root_noise_frac=0.25, root_noise_alpha=1.0,
        # Absolute strength against exactly-solved positions, reported at every
        # deep eval for every arm.  Defaults to <root>/solved_tests so it
        # follows an overridden root; set to '' to switch the metric off.
        solved_dir=None, solved_every=0, solved_n=0)
    s.update(over)
    if s['solved_dir'] is None:
        s['solved_dir'] = os.path.join(s['root'], SOLVED_SUBDIR)
    return s


def param_counts(shared):
    """Trunk / head split for both networks, so "matched size" is checkable.

    The TRUNK is identical by construction — same stem, same residual blocks,
    same 1x1 head conv — and it is where essentially all the capacity lives.
    The output layers differ because the methods emit different numbers of
    numbers per position: ThompsonZero needs 4 for the state Dirichlet and 4 per
    action, AlphaZero needs one logit per action plus one value.  That is a real
    difference in what is being predicted, not a handicap, and padding AlphaZero
    with unused parameters to equalise the total would be worse than reporting
    it.

    HOW BIG that difference is depends on the action count, and it does not stay
    negligible.  On Connect 4 (7 actions) the heads are 15% of ThompsonZero and
    the totals sit 13% apart.  On Othello (65) the action head emits 260 numbers
    against AlphaZero's 66, the heads become the MAJORITY of ThompsonZero's
    parameters, and the totals differ by more than a factor of two.  The trunk is
    still identical, but "the trunk is where the capacity lives" stops being true
    -- so the ratio is reported and flagged rather than left to be assumed away.
    """
    game = c4.load_game(shared['game']); c4.set_game(game)
    tz = c4.C4DirichletNet(*tz_sig(shared))
    a = az.AlphaZeroNet(*az_sig(shared))
    def split(net, head_names):
        head = sum(p.numel() for n, p in net.named_parameters()
                   if n.split('.')[0] in head_names)
        return sum(p.numel() for p in net.parameters()) - head, head
    tz_trunk, tz_head = split(tz, {'v_out', 'a_out'})
    az_trunk, az_head = split(a, {'policy_out', 'value_out'})
    tz_tot, az_tot = tz_trunk + tz_head, az_trunk + az_head
    return {'thompson trunk': tz_trunk, 'alphazero trunk': az_trunk,
            'thompson heads': tz_head, 'alphazero heads': az_head,
            'thompson total': tz_tot, 'alphazero total': az_tot,
            'trunk matches': tz_trunk == az_trunk,
            'thompson trunk sig': tz_sig(shared),
            'alphazero trunk sig': az_sig(shared),
            'actions': c4._NUM_ACTIONS,
            'head share of thompson': round(tz_head / tz_tot, 3),
            'thompson/alphazero params': round(tz_tot / az_tot, 2)}


def report_params(shared, log=print):
    """param_counts, printed, with the caveat spelled out when it matters."""
    p = param_counts(shared)
    for k, v in p.items():
        log(f'  {k:26} {v}')
    # Warn in BOTH directions: the two definitions of "matched" cannot both
    # hold, and whichever one is in force is the caveat on the AlphaZero row.
    tr = p['alphazero trunk'] / p['thompson trunk']
    if p['thompson/alphazero params'] > 1.25:
        log(f'  NOTE at {p["actions"]} actions ThompsonZero carries '
            f'{p["thompson/alphazero params"]}x AlphaZero\'s parameters and '
            f'{100 * p["head share of thompson"]:.0f}% of them are in its heads. '
            f'The trunks are identical, so search and self-play are matched, but '
            f'the two networks are NOT the same size. bench.match_capacity('
            f'shared) equalises the totals by widening AlphaZero\'s trunk.')
    elif tr > 1.25:
        log(f'  NOTE totals are matched ({p["thompson/alphazero params"]}x) by '
            f'giving AlphaZero a {tr:.1f}x WIDER trunk. That is the other '
            f'horn: ThompsonZero\'s extra parameters are an output projection '
            f'it needs to emit 4 numbers per action, whereas AlphaZero\'s are '
            f'trunk capacity it can actually compute with. Matching totals may '
            f'now favour AlphaZero; matching trunks favours ThompsonZero.')
    return p


def train_arm(name, shared, log=print):
    """Train one arm to completion.  Resumable: re-running skips finished work
    when `resume` is on, because each driver reloads its own latest.pt."""
    game = c4.load_game(shared['game'])
    t0 = time.time()
    log(f'\n{"=" * 78}\n=== ARM {name}: {ARMS[name][3]}\n{"=" * 78}')
    if ARMS[name][0] in ('gauss', 'gauss_halving'):
        hist = vd.run_training(gauss_config(shared, name), game=game, log=log)
    elif ARMS[name][0] == 'alphazero':
        hist = az.run_training(alphazero_config(shared), game=game, log=log)
    else:
        hist = c4.run_training(thompson_config(name, shared), game=game, log=log)
    log(f'=== ARM {name} done in {(time.time() - t0) / 60:.1f} min')
    return hist


def train_all(shared, arms=None, log=print):
    """Train every arm, one after another.  Sequential on purpose: concurrent
    arms would contend for the GPU and distort the perf numbers, and each arm
    already saturates the device through its own batched inference server."""
    out = {}
    for name in arms_of(shared, arms):
        out[name] = train_arm(name, shared, log=log)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Round-robin tournament
# ══════════════════════════════════════════════════════════════════════════════
def rr_cache_path(shared):
    """Where pairwise tournament results accumulate, inside the benchmark root
    so it travels with the checkpoints."""
    name = shared.get('rr_cache') or 'round_robin.json'
    return name if os.path.isabs(name) else os.path.join(shared['root'], name)


def load_players(shared, gens=None, arms=None, include_random=True):
    """{label: (engine, net)} for every arm × generation that exists on disk.

    Also pins the tournament's game from `shared`, so round_robin and the
    diagnostics below inherit it."""
    GAME_REF[0] = c4.load_game(shared['game'])
    c4.set_game(GAME_REF[0])
    gens = gens or shared.get('gens') or (1000, 2000)
    tsig, asig = tz_sig(shared), az_sig(shared)
    players = {}
    for name in arms_of(shared, arms):
        engine = ARMS[name][0]
        d = arm_dir(shared['root'], name)
        for g in gens:
            if not os.path.exists(os.path.join(d, f'bench_{g}.pt')):
                continue
            gsig = tsig + (shared.get('gauss_head', 'spatial'),)
            net = (vd.load_benchmark_net(d, str(g), gsig)
                   if engine in ('gauss', 'gauss_halving')
                   else az.load_benchmark_net(d, str(g), asig)
                   if engine == 'alphazero'
                   else c4.load_benchmark_net(d, str(g), tsig))
            players[f'{name}@{g}'] = (engine, net)
    if include_random:
        players['random'] = ('random', None)
    return players


def _move(engine, net, state, sims, rng, bots, eval_temp=6.0):
    if engine == 'random' or net is None:
        leg = state.legal_actions()
        return int(leg[rng.integers(len(leg))])
    if engine in ('gauss', 'gauss_halving'):
        if sims <= 0:
            return vd.value_lookahead_move(net, state, 'cpu')
        b = bots.get(id(net))
        if b is None:
            b = bots[id(net)] = (
                vd.HalvingBot(GAME_REF[0], net, 'cpu', sims, batch_size=8,
                              random_state=rng)
                if engine == 'gauss_halving' else
                vd.GMCTSBot(GAME_REF[0], net, 'cpu', sims, batch_size=8,
                            random_state=rng))
        root = b.mcts_search(state)
        if engine == 'gauss_halving':
            return vd.halving_pick(root, b.last_choice, rng, False)
        return vd.root_pick(root, rng, thompson=False)
    if engine == 'alphazero':
        if sims <= 0:
            return az.value_greedy_move(net, state, 'cpu')
        b = bots.get(id(net))
        if b is None:
            b = bots[id(net)] = az.AZMCTSBot(GAME_REF[0], net, 'cpu', sims,
                                             batch_size=8, random_state=rng)
        return az.root_pick(b.mcts_search(state), rng, sample=False)
    if sims <= 0:
        # The SAME one ply of lookahead AlphaZero gets above.  Reading this
        # engine's action heads instead would pit a bare network against a
        # network plus a ply of search.
        return c4.value_lookahead_move(net, state, 'cpu')
    b = bots.get(id(net))
    if b is None:
        b = bots[id(net)] = c4.C4MCTSBot(GAME_REF[0], net, 'cpu', sims,
                                         batch_size=8, temp=eval_temp,
                                         random_state=rng)
    return c4.root_pick(b.mcts_search(state), rng, thompson=False)


GAME_REF = [None]


def play_game(pa, pb, sims, rng, opening_plies=2):
    """One game, `pa` moves first.  Returns pa's result in {1, 0.5, 0}."""
    game = GAME_REF[0]
    st = game.new_initial_state()
    for _ in range(opening_plies):
        if st.is_terminal():
            break
        leg = st.legal_actions()
        st.apply_action(int(leg[rng.integers(len(leg))]))
    bots = {}
    while not st.is_terminal():
        engine, net = pa if st.current_player() == 0 else pb
        st.apply_action(_move(engine, net, st, sims, rng, bots))
    r = st.returns()[0]
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def bradley_terry(W, iters=4000, lr=0.5, anchor_idx=None):
    """Ratings by maximum likelihood under
        P(i beats j) = sigmoid((r_i - r_j) · ln10 / 400).
    W[i][j] = points i scored against j.  Gauge-fixed to zero mean, then shifted
    so `anchor_idx` sits at 0 if given."""
    n = W.shape[0]
    r = np.zeros(n)
    N = W + W.T
    s = 400.0 / np.log(10.0)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :]) / s))
        grad = ((W - N * p) * (N > 0)).sum(1) / s
        r += lr * s * s * grad / np.maximum(N.sum(1), 1.0)
        r -= r.mean()
    if anchor_idx is not None:
        r = r - r[anchor_idx]
    return r


def _cache_key(sims, a, b):
    x, y = sorted((a, b))
    return f'{sims}|{x}|{y}'


def _load_rr_cache(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save_rr_cache(path, cache):
    if not path:
        return
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(cache, fh)
    os.replace(tmp, path)


def round_robin(players, sims, games_per_pair, seed=12345, log=print,
                opening_plies=2, game=None, cache=None):
    """Every pair plays `games_per_pair` games, colours alternating.  Returns
    (names, W, elo).

    The game comes from load_players (which reads it from `shared`); pass
    `game` only when calling this without load_players.  It is deliberately not
    defaulted to Connect 4 -- silently rating Othello networks on the wrong
    board is exactly the kind of thing that produces a confident wrong table.

    `cache` is a path to a JSON file of already-played pairs, keyed by
    (sims, the two labels).  Only the shortfall is played, so adding one arm to
    a finished tournament costs its own pairs rather than the whole table, and
    raising games_per_pair tops up rather than restarting.  Results accumulate,
    so the games already played are never thrown away."""
    if game is not None:
        GAME_REF[0] = c4.load_game(game) if isinstance(game, str) else game
        c4.set_game(GAME_REF[0])
    if GAME_REF[0] is None:
        raise RuntimeError('no game set — call load_players(shared, ...) first, '
                           'or pass game= explicitly')
    # Search is identical for every player, so the tournament compares NETWORKS
    # rather than two different search procedures.
    c4.set_search(search_agg=c4.AGG_ADDITIVE, target_agg=c4.AGG_ADDITIVE,
                  selection='dirichlet', backup=c4.BACKUP_MEAN)
    names = list(players)
    idx = {k: i for i, k in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    rng = np.random.default_rng(seed)
    store = _load_rr_cache(cache)
    pairs = list(itertools.combinations(names, 2))
    todo = []
    for a, b in pairs:
        rec = store.get(_cache_key(sims, a, b))
        played = int(rec[2]) if rec else 0
        if played < games_per_pair:
            todo.append((a, b, played))
    reused = len(pairs) - len(todo)
    if reused:
        log(f'  reusing {reused}/{len(pairs)} pairs from {cache}')
    t0 = time.time()
    for k, (a, b, played) in enumerate(todo, 1):
        key = _cache_key(sims, a, b)
        rec = store.get(key)
        x, y = sorted((a, b))
        sx = float(rec[0]) if rec else 0.0        # points for x
        for g in range(played, games_per_pair):
            first, second = (a, b) if g % 2 == 0 else (b, a)
            sc = play_game(players[first], players[second], sims, rng,
                           opening_plies)
            sx += sc if first == x else 1.0 - sc
        store[key] = [sx, games_per_pair - sx, games_per_pair]
        if k % 5 == 0 or k == len(todo):
            log(f'  … {k}/{len(todo)} new pairs  ({time.time() - t0:.0f}s)')
    _save_rr_cache(cache, store)
    for a, b in pairs:
        rec = store.get(_cache_key(sims, a, b))
        if not rec:
            continue
        x, y = sorted((a, b))
        W[idx[x], idx[y]] += float(rec[0])
        W[idx[y], idx[x]] += float(rec[1])
    anchor = idx.get('random')
    return names, W, bradley_terry(W, anchor_idx=anchor)


def wilson(score, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    c = (score + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def report(names, W, elo, title='', log=print):
    log(f'\n=== {title} ===')
    log(f'{"player":>10} {"Elo":>7} {"score":>17}  {"games":>6}   config')
    for i in np.argsort(-elo):
        gp = W[i].sum() + W[:, i].sum()
        sc = W[i].sum() / max(gp, 1)
        lo, hi = wilson(sc, gp)
        arm = names[i].split('@')[0]
        desc = ARMS[arm][3] if arm in ARMS else ''
        log(f'{names[i]:>10} {elo[i]:7.0f} {100*sc:6.1f}% '
            f'[{100*lo:4.1f},{100*hi:5.1f}] {gp:6.0f}   {desc}')


def head_to_head_table(names, W, log=print):
    """Raw pairwise scores — the tournament's evidence, not just its summary."""
    log('\npairwise score for ROW against COLUMN (%):')
    log('           ' + ' '.join(f'{n.split("@")[0][:4]:>6}' for n in names))
    for i, a in enumerate(names):
        row = []
        for j in range(len(names)):
            g = W[i, j] + W[j, i]
            row.append('     ·' if i == j or g == 0
                       else f'{100*W[i, j]/g:6.0f}')
        log(f'{a:>10} ' + ' '.join(row))
# ══════════════════════════════════════════════════════════════════════════════
#  Scrutinising the AlphaZero control
# ══════════════════════════════════════════════════════════════════════════════
# A large win over a control is only evidence if the control was configured
# competently.  These checks try to break that assumption before it is believed.
# They are cheap and run against checkpoints you already have.

def az_cpuct_sweep(shared, gen=4000, opponent='AA', sims=128, games=60,
                   values=(0.5, 1.0, 1.5, 2.5, 4.0), seed=77, log=print):
    """AlphaZero's strength is famously sensitive to c_puct, and the benchmark
    picked one value (1.5) without tuning.  Replay the same matchup at several
    values: if strength moves a lot, the control was mis-tuned at EVALUATION —
    and, worse, was also mis-tuned during self-play, so its training data was
    weaker than it needed to be."""
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game']); c4.set_game(GAME_REF[0])
    a = az.load_benchmark_net(arm_dir(shared['root'], 'AZ'), str(gen),
                              az_sig(shared))
    o = c4.load_benchmark_net(arm_dir(shared['root'], opponent), str(gen),
                              tz_sig(shared))
    old = az._C_PUCT
    log(f'\nAlphaZero c_puct sweep — AZ@{gen} vs {opponent}@{gen}, '
        f'MCTS-{sims}, {games} games each')
    out = {}
    for cp in values:
        az.set_search(c_puct=cp)
        rng = np.random.default_rng(seed)
        w = d = l = 0
        for g in range(games):
            first = ('alphazero', a) if g % 2 == 0 else ('thompson', o)
            second = ('thompson', o) if g % 2 == 0 else ('alphazero', a)
            s = play_game(first, second, sims, rng)
            s_az = s if g % 2 == 0 else 1.0 - s
            w += s_az == 1.0; d += s_az == 0.5; l += s_az == 0.0
        sc = (w + 0.5 * d) / games
        lo, hi = wilson(sc, games)
        out[cp] = sc
        log(f'  c_puct {cp:<5} AZ scores {100*sc:5.1f}% '
            f'[{100*lo:4.1f},{100*hi:5.1f}]   W{w} D{d} L{l}')
    az.set_search(c_puct=old)
    best = max(out, key=out.get)
    log(f'  best {best} at {100*out[best]:.1f}% vs configured 1.5 at '
        f'{100*out.get(1.5, float("nan")):.1f}%')
    return out


def az_progression(shared, gens=(1000, 2000, 4000), sims=128, games=40,
                   seed=88, log=print):
    """Did the control still improve late, or had it plateaued?  A plateau means
    the comparison is about a converged AlphaZero; continued improvement means it
    was simply cut off early and the gap partly measures budget, not method."""
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game']); c4.set_game(GAME_REF[0])
    sig = az_sig(shared)
    d = arm_dir(shared['root'], 'AZ')
    nets = {g: ('alphazero', az.load_benchmark_net(d, str(g), sig))
            for g in gens if os.path.exists(os.path.join(d, f'bench_{g}.pt'))}
    log(f'\nAlphaZero self-progression, MCTS-{sims}, {games} games/pair')
    ks = sorted(nets)
    for i in range(len(ks) - 1):
        a, b = ks[i + 1], ks[i]
        rng = np.random.default_rng(seed)
        sc = sum(play_game(nets[a], nets[b], sims, rng) if g % 2 == 0
                 else 1.0 - play_game(nets[b], nets[a], sims, rng)
                 for g in range(games)) / games
        lo, hi = wilson(sc, games)
        log(f'  AZ@{a} vs AZ@{b}: {100*sc:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}]'
            + ('   still improving' if lo > 0.5 else '   not resolved'))


def search_value(shared, arms=('AA', 'AZ'), gen=4000, sims=128, games=40,
                 seed=99, log=print):
    """How much does search add on top of each raw network?  If AlphaZero's
    search adds far less than ThompsonZero's, the deficit is in PUCT or the
    scalar value head rather than in the learned policy."""
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game']); c4.set_game(GAME_REF[0])
    log(f'\nWhat search buys — net@{gen} with MCTS-{sims} vs the SAME net '
        f'search-free, {games} games')
    for name in arms:
        engine = ARMS[name][0]
        d = arm_dir(shared['root'], name)
        net = (az.load_benchmark_net(d, str(gen), az_sig(shared))
               if engine == 'alphazero'
               else c4.load_benchmark_net(d, str(gen), tz_sig(shared)))
        rng = np.random.default_rng(seed)
        w = d_ = l = 0
        for g in range(games):
            st = GAME_REF[0].new_initial_state()
            for _ in range(2):
                leg = st.legal_actions()
                st.apply_action(int(leg[rng.integers(len(leg))]))
            searched_side = g % 2
            bots = {}
            while not st.is_terminal():
                use = sims if st.current_player() == searched_side else 0
                st.apply_action(_move(engine, net, st, use, rng, bots))
            r = st.returns()[searched_side]
            w += r > 0; l += r < 0; d_ += r == 0
        sc = (w + 0.5 * d_) / games
        lo, hi = wilson(sc, games)
        log(f'  {name}: searched side scores {100*sc:5.1f}% '
            f'[{100*lo:4.1f},{100*hi:5.1f}]   W{w} D{d_} L{l}')


# ══════════════════════════════════════════════════════════════════════════════
#  Absolute strength: every arm against exactly-solved positions
#
#  The round robin ranks the arms against EACH OTHER, which is silent about
#  whether any of them is actually good.  This scores each checkpoint against
#  ground truth on the same scale, so arms, generations and engines are all
#  directly comparable — and a collapsed run reads as collapsed instead of
#  merely losing.
# ══════════════════════════════════════════════════════════════════════════════
def solved_report(shared, test_dir=None, gens=(1000, 2000), arms=None,
                  sims=0, limit=200, log=print, include_random=True):
    """Score every arm x generation against the solved-position suites.

    `sims=0` is search-free.  Both engines then use the SAME one-ply value
    lookahead, because AlphaZero has no per-action value head to read and
    comparing a lookahead against a head-read measures the lookahead.  Set
    sims>0 to score the full search instead."""
    import connect4_solved_eval as sev
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game'])
        c4.set_game(GAME_REF[0])
    suites = sev.Suite.build(test_dir or shared['solved_dir'],
                             limit=limit, log=log)
    players = load_players(shared, gens=gens, arms=arms,
                           include_random=include_random)
    rows = {}
    for label, (engine, net) in players.items():
        if engine == 'random':
            chooser, values = sev.random_player(0)
        elif engine == 'alphazero':
            chooser, values = sev.alphazero_player(net, 'cpu', sims=sims)
        else:
            chooser, values = sev.thompson_player(net, 'cpu', sims=sims,
                                                  lookahead=(sims == 0))
        tot_n = 0
        acc = {'optimal': 0.0, 'perfect': 0.0, 'blunder': 0.0}
        va = mj = vs = 0.0
        for s in suites.values():
            r = s.evaluate(chooser, values)
            for k in acc:
                acc[k] += r[k] * r['n']
            if 'value_acc' in r:
                va += r['value_acc'] * r['n']
                mj += r['majority'] * r['n']
                vs += r['value_sign'] * r['n']
            tot_n += r['n']
        row = {k: v / tot_n for k, v in acc.items()}
        row['n'] = tot_n
        if values is not None:
            row.update(value_acc=va / tot_n, majority=mj / tot_n,
                       value_sign=vs / tot_n)
        rows[label] = row
    order = sorted(rows, key=lambda k: -rows[k]['optimal'])
    log(f'\n=== solved-position strength ('
        f'{"search-free" if sims == 0 else f"MCTS-{sims}"}, '
        f'{rows[order[0]]["n"]} positions)')
    log(f'{"player":<14}{"optimal":>10}{"perfect":>10}{"blunder":>10}'
        f'{"value":>9}{"base":>7}{"v-sign":>9}')
    for k in order:
        r = rows[k]
        log(f'{k:<14}{100 * r["optimal"]:>9.1f}%{100 * r["perfect"]:>9.1f}%'
            f'{100 * r["blunder"]:>9.1f}%'
            + (f'{100 * r["value_acc"]:>8.1f}%{100 * r["majority"]:>6.0f}%'
               f'{100 * r["value_sign"]:>8.1f}%' if 'value_acc' in r
               else f'{"-":>9}{"-":>7}{"-":>9}'))
    return rows


def value_report(shared, test_dir=None, gens=(1000, 2000), arms=None, limit=0,
                 log=print):
    """Value MSE per difficulty for every arm x generation, on one scale.

    Cheap by construction: the observations are built once and every model is
    then a single batched forward, so this is the metric to watch across a whole
    benchmark rather than the move-accuracy one that needs children solved.

    The prediction is the network's scalar value for the player to move --
    p_win - p_loss from ThompsonZero's state belief, the value head for
    AlphaZero -- against the exact outcome in {-1, 0, +1}."""
    import connect4_solved_eval as sev
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game'])
        c4.set_game(GAME_REF[0])
    probe = sev.ValueProbe.build(test_dir or shared['solved_dir'],
                                 limit=limit, log=log)
    players = load_players(shared, gens=gens, arms=arms, include_random=False)
    rows = {}
    for label, (engine, net) in players.items():
        fn = (sev.alphazero_value_fn(net, 'cpu') if engine == 'alphazero'
              else sev.thompson_value_fn(net, 'cpu'))
        rows[label] = probe.mse_by_bucket(fn)
    sev.value_report(rows, log=log,
                     title=f'value MSE vs exact outcomes ({len(probe)} '
                           f'positions, lower is better)')
    return rows

def game_contrast(games=('connect_four', 'othello'), n=400, seed=0, log=print):
    """Structural facts that decide which method's advantages transfer.

    Two of ThompsonZero's advantages are game-dependent rather than universal:
    an explicit draw category is only worth something in a game with draws, and
    a per-action belief head is only cheap to train when its slots are legal
    often enough to receive gradient."""
    import numpy as np
    import pyspiel
    rng = np.random.default_rng(seed)
    log(f'{"game":<14}{"draw %":>9}{"legal/pos":>11}{"plies":>8}'
        f'{"slots":>7}{"slot use":>10}')
    out = {}
    for name in games:
        g = pyspiel.load_game(name)
        A = g.num_distinct_actions()
        draws, legal, lens = 0, [], []
        for _ in range(n):
            st = g.new_initial_state()
            L = 0
            while not st.is_terminal():
                la = st.legal_actions()
                legal.append(len(la))
                L += 1
                st.apply_action(int(la[rng.integers(len(la))]))
            draws += st.returns()[0] == 0
            lens.append(L)
        r = dict(draw_pct=100 * draws / n, legal=float(np.mean(legal)),
                 plies=float(np.mean(lens)), actions=A,
                 slot_use=float(np.mean(legal)) / A)
        out[name] = r
        log(f'{name:<14}{r["draw_pct"]:>8.1f}%{r["legal"]:>11.1f}'
            f'{r["plies"]:>8.1f}{A:>7}{r["slot_use"]:>9.1%}')
    log('  draw % is under RANDOM play — read it next to the draw rate your own '
        'trained runs report, which is the one that matters.')
    log('  slot use is the fraction of action-head outputs that are legal, and '
        'so receive gradient, at a typical position.')
    return out


def sims_scaling(shared, a='MA', b='AZ', gen=4000, sims=(32, 64, 128, 256),
                 games=40, seed=1234, log=print):
    """Head-to-head at increasing simulation budgets, both sides equal.

    Answers one specific question: does `a`'s advantage over `b` GROW with
    search?  ThompsonZero's per-action posteriors tighten as an edge collects
    visits (under the additive rule alpha0 IS the visit count), so if its edge
    is search-limited the score should climb with sims.  A flat curve says the
    difference is in the trained networks, not in how much search they are
    given, and more simulations will not recover it.

    This is an EVALUATION sweep on existing checkpoints — cheap, and it says
    nothing directly about training with more simulations, which would need a
    retrain.  It bounds the question rather than settling it."""
    import numpy as np
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game(shared['game'])
        c4.set_game(GAME_REF[0])
    c4.set_search(search_agg=c4.AGG_ADDITIVE, target_agg=c4.AGG_ADDITIVE,
                  selection='dirichlet', backup=c4.BACKUP_MEAN)
    pa, pb = None, None
    for name, slot in ((a, 'a'), (b, 'b')):
        engine = ARMS[name][0]
        d = arm_dir(shared['root'], name)
        sig = az_sig(shared) if engine == 'alphazero' else tz_sig(shared)
        net = (az.load_benchmark_net(d, str(gen), sig) if engine == 'alphazero'
               else c4.load_benchmark_net(d, str(gen), sig))
        if slot == 'a':
            pa = (engine, net)
        else:
            pb = (engine, net)
    log(f'\n{a}@{gen} vs {b}@{gen} across simulation budgets, {games} games each')
    log(f'{"sims":>7}{"score":>9}{"W-D-L":>12}{"Elo":>8}{"95% CI":>20}')
    rows = []
    for s in sims:
        rng = np.random.default_rng(seed)
        w = dr = l = 0
        for g in range(games):
            first, second = (pa, pb) if g % 2 == 0 else (pb, pa)
            r = play_game(first, second, s, rng)
            r = r if g % 2 == 0 else 1.0 - r
            w += r == 1.0
            dr += r == 0.5
            l += r == 0.0
        score = (w + 0.5 * dr) / games
        lo, hi = wilson(score, games)
        elo = (400 * np.log10(score / (1 - score))
               if 0 < score < 1 else float('nan'))
        rows.append(dict(sims=s, score=score, w=w, d=dr, l=l, elo=elo))
        log(f'{s:>7}{score:>8.1%}{f"{w}-{dr}-{l}":>12}{elo:>8.0f}'
            f'   [{lo:.1%}, {hi:.1%}]')
    trend = rows[-1]['score'] - rows[0]['score']
    log(f'  {a} gains {trend:+.1%} going from {sims[0]} to {sims[-1]} sims. '
        f'With {games} games a swing under about '
        f'{2 * (0.5 / games ** 0.5):.0%} is noise.')
    return rows
