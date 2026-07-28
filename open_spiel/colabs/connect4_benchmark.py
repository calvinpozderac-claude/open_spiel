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
import os
import time

import numpy as np

import connect4_dirichlet_utils as c4
import connect4_alphazero_utils as az


# ══════════════════════════════════════════════════════════════════════════════
#  Arm definitions
# ══════════════════════════════════════════════════════════════════════════════
# name -> (engine, search_agg, target_agg, human-readable description)
ARMS = {
    'AA':  ('thompson', c4.AGG_ADDITIVE,     c4.AGG_ADDITIVE,
            'search additive     · target additive'),
    'AM':  ('thompson', c4.AGG_ADDITIVE,     c4.AGG_ADDITIVE_MLE,
            'search additive     · target additive_mle'),
    'MA':  ('thompson', c4.AGG_ADDITIVE_MLE, c4.AGG_ADDITIVE,
            'search additive_mle · target additive'),
    'MM':  ('thompson', c4.AGG_ADDITIVE_MLE, c4.AGG_ADDITIVE_MLE,
            'search additive_mle · target additive_mle'),
    'AZ':  ('alphazero', None, None,
            'AlphaZero control (policy + scalar value, PUCT)'),
}


def arm_dir(root, name):
    return os.path.join(root, f'bench_{name}')


def thompson_config(name, shared):
    """A ThompsonZero Config from the shared settings plus this arm's rules."""
    _eng, sagg, tagg, _d = ARMS[name]
    return c4.Config(
        checkpoint_dir=arm_dir(shared['root'], name),
        num_episodes=shared['num_episodes'],
        channels=shared['channels'], num_blocks=shared['num_blocks'],
        head_ch=shared['head_ch'], seed=shared['seed'],
        device_preference=shared['device'],
        search_agg=sagg, target_agg=tagg, kl_normalize=False,
        cons_frac=shared['cons_frac'],
        selection='dirichlet',
        fast_sims=shared['fast_sims'], full_sims=shared['full_sims'],
        fast_prob=shared['fast_prob'], temp_threshold=shared['temp_threshold'],
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
        resume=shared['resume'])


def alphazero_config(shared):
    return az.Config(
        checkpoint_dir=arm_dir(shared['root'], 'AZ'),
        num_episodes=shared['num_episodes'],
        channels=shared['channels'], num_blocks=shared['num_blocks'],
        head_ch=shared['head_ch'], seed=shared['seed'],
        device_preference=shared['device'],
        c_puct=shared['c_puct'],
        root_noise_frac=shared['root_noise_frac'],
        root_noise_alpha=shared['root_noise_alpha'],
        fast_sims=shared['fast_sims'], full_sims=shared['full_sims'],
        fast_prob=shared['fast_prob'], temp_threshold=shared['temp_threshold'],
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
        resume=shared['resume'])


def default_shared(**over):
    """Everything the five arms hold in common.  Override any of it by keyword."""
    s = dict(
        root='c4_benchmark', num_episodes=2000, seed=0, device='auto',
        channels=32, num_blocks=3, head_ch=8,
        fast_sims=50, full_sims=150, fast_prob=0.75, temp_threshold=12,
        use_workers=True, workers=0, games_per_worker=16, worker_wave=4,
        n_parallel_games=16, wave_per_game=4, pool_prob=0.2,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_decay_eps=2000, weight_decay=1e-4, grad_clip=1.0,
        cons_frac=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=32, resume=True,
        c_puct=1.5, root_noise_frac=0.25, root_noise_alpha=1.0)
    s.update(over)
    return s


def param_counts(shared):
    """Trunk / head split for both networks, so "matched size" is checkable.

    The TRUNK is identical by construction — same stem, same residual blocks,
    same 1x1 head conv — and it is where essentially all the capacity lives.
    The output layers differ because the methods emit different numbers of
    numbers per position: ThompsonZero needs 4 for the state Dirichlet and 4 per
    action (32 total for 7 columns), AlphaZero needs 7 policy logits and 1 value
    (8 total).  That is a real difference in what is being predicted, not a
    handicap, and padding AlphaZero with unused parameters to equalise the total
    would be worse than reporting it.
    """
    game = c4.load_game(); c4.set_game(game)
    sig = (shared['channels'], shared['num_blocks'], shared['head_ch'])
    tz, a = c4.C4DirichletNet(*sig), az.AlphaZeroNet(*sig)
    def split(net, head_names):
        head = sum(p.numel() for n, p in net.named_parameters()
                   if n.split('.')[0] in head_names)
        return sum(p.numel() for p in net.parameters()) - head, head
    tz_trunk, tz_head = split(tz, {'v_out', 'a_out'})
    az_trunk, az_head = split(a, {'policy_out', 'value_out'})
    return {'trunk (identical)': tz_trunk,
            'thompson heads': tz_head, 'alphazero heads': az_head,
            'thompson total': tz_trunk + tz_head,
            'alphazero total': az_trunk + az_head,
            'trunk matches': tz_trunk == az_trunk}


def train_arm(name, shared, log=print):
    """Train one arm to completion.  Resumable: re-running skips finished work
    when `resume` is on, because each driver reloads its own latest.pt."""
    game = c4.load_game()
    t0 = time.time()
    log(f'\n{"=" * 78}\n=== ARM {name}: {ARMS[name][3]}\n{"=" * 78}')
    if ARMS[name][0] == 'alphazero':
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
    for name in (arms or list(ARMS)):
        out[name] = train_arm(name, shared, log=log)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Round-robin tournament
# ══════════════════════════════════════════════════════════════════════════════
def load_players(shared, gens=(1000, 2000), arms=None, include_random=True):
    """{label: (engine, net)} for every arm × generation that exists on disk."""
    sig = (shared['channels'], shared['num_blocks'], shared['head_ch'])
    players = {}
    for name in (arms or list(ARMS)):
        engine = ARMS[name][0]
        d = arm_dir(shared['root'], name)
        for g in gens:
            if not os.path.exists(os.path.join(d, f'bench_{g}.pt')):
                continue
            net = (az.load_benchmark_net(d, str(g), sig) if engine == 'alphazero'
                   else c4.load_benchmark_net(d, str(g), sig))
            players[f'{name}@{g}'] = (engine, net)
    if include_random:
        players['random'] = ('random', None)
    return players


def _move(engine, net, state, sims, rng, bots, eval_temp=6.0):
    if engine == 'random' or net is None:
        leg = state.legal_actions()
        return int(leg[rng.integers(len(leg))])
    if engine == 'alphazero':
        if sims <= 0:
            return az.policy_move(net, state, 'cpu')
        b = bots.get(id(net))
        if b is None:
            b = bots[id(net)] = az.AZMCTSBot(GAME_REF[0], net, 'cpu', sims,
                                             batch_size=8, random_state=rng)
        return az.root_pick(b.mcts_search(state), rng, sample=False)
    if sims <= 0:
        return c4.value_greedy_move(net, state, 'cpu')
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


def round_robin(players, sims, games_per_pair, seed=12345, log=print,
                opening_plies=2):
    """Every pair plays `games_per_pair` games, colours alternating.  Returns
    (names, W, elo)."""
    if GAME_REF[0] is None:
        GAME_REF[0] = c4.load_game()
        c4.set_game(GAME_REF[0])
    # Search is identical for every player, so the tournament compares NETWORKS
    # rather than two different search procedures.
    c4.set_search(search_agg=c4.AGG_ADDITIVE, target_agg=c4.AGG_ADDITIVE,
                  selection='dirichlet')
    names = list(players)
    idx = {k: i for i, k in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(names, 2))
    t0 = time.time()
    for k, (a, b) in enumerate(pairs, 1):
        for g in range(games_per_pair):
            first, second = (a, b) if g % 2 == 0 else (b, a)
            s = play_game(players[first], players[second], sims, rng,
                          opening_plies)
            W[idx[first], idx[second]] += s
            W[idx[second], idx[first]] += 1.0 - s
        if k % 5 == 0 or k == len(pairs):
            log(f'  … {k}/{len(pairs)} pairs  ({time.time() - t0:.0f}s)')
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
