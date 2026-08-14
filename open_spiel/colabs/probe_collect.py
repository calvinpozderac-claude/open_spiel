"""Collect oracle "how much search did this position ACTUALLY need" labels,
together with everything a model might predict them from.

This is the data-collection half of the probe that gates the learned
sims-needed head. The question it exists to answer: the deployed 'predictor'
rule is a 4-parameter linear fit on 3 scalars and reaches r2=0.53 -- is the
remaining 47% signal that a bigger model could reach, or is it label noise?

WHAT GETS STORED, AND WHY IT IS THE RAW BELIEF RATHER THAN A DISTANCE

Nothing here is reduced to a label. Two label definitions were tried and
discarded during development, both of which looked reasonable and were not:

  KL(belief_i || belief_final) on the root's value Dirichlet.  Under additive
  aggregation alpha0 grows with visits, so this is dominated by the
  CONCENTRATION gap and traces out nearly the same curve for every position --
  it measures the budget schedule, not the position.

  Total variation on that same belief's MEAN.  Scale-free, but the root's
  value mean is a running average that settles inside the first checkpoint
  (TV < 0.007 over an entire 1000-sim search), so every position labels at the
  floor.

Both failed for the same underlying reason: the root's own value belief p(v)
is not what search is FOR. The move played is
`argmax dir_value(root.alpha_sel)` over edges search actually touched, and the
training target is the per-edge beliefs. So the per-action values are recorded
at every checkpoint and every label is derived from those offline.

  feats        [1, margin, log conf, n_legal]  -- exactly what an upfront rule
               is allowed to see today (c4._root_features on the raw prior)
  obs          the (3,8,8) board -- what a network HEAD would see
  action_vals  (ckpt, K) dir_value(alpha_sel) per legal action
  touched      (ckpt, K) which edges had evidence -- root_pick ignores the rest
  best_idx     (ckpt,) the move that WOULD be played at this checkpoint
  state_alpha  (ckpt, 3) the root value belief, for comparison with the
               existing rules, which all watch this and not the actions
  consec_kl    KL(previous checkpoint || this one) on state_alpha -- the ONLINE
               curve oracle_sims_needed's patience rule reads.  Unaffected by
               the concentration artifact above: consecutive checkpoints differ
               by one block, so their concentrations are comparable.
  game_id      which self-play game the position came from, so the analysis can
               split train/test by GAME -- positions sampled every 4th ply of
               one game are correlated, and a position-level split would leak
               across it and inflate held-out r2.

    python probe_collect.py            # both nets, writes probe_data_*.npz
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np

CAP = 1000
CHECK_EVERY = 50
SAMPLE_SIMS = 150
N_POSITIONS = 5000
N_WORKERS = 5
K_MAX = 65                 # othello num_distinct_actions; rows padded to this

# (tag, benchmark dir, checkpoint label).  bench_MM/2000 is the net the
# deployed predictor was actually fit against, so it reproduces the r2=0.53
# baseline; bench_R2conf/5000 is the strongest net we have, and asks whether
# predictability holds up as the net improves.
NETS = [
    ('mm2000', 'othello_benchmark/bench_MM', '2000'),
    ('conf5000', 'othello_benchmark/bench_R2conf', '5000'),
]


def _init_process():
    """Every worker is one process of a pool; keep each single-threaded so 5 of
    them do not oversubscribe 6 cores fighting over torch's own thread pool."""
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return game


def sample_positions_with_games(c4, ad, net, game, n_positions, seed):
    """ad.sample_positions, but returning which game each position came from.

    Same sampling (every 4th ply of the net's own play, so opening/midgame/
    endgame are all represented); the game index is the only addition, and it
    exists purely so the analysis can hold out whole games."""
    rng = np.random.default_rng(seed)
    positions, game_ids = [], []
    g = 0
    while len(positions) < n_positions:
        state = game.new_initial_state()
        for _ in range(2):
            if state.is_terminal():
                break
            leg = state.legal_actions()
            state.apply_action(int(leg[rng.integers(len(leg))]))
        bot = c4.C4MCTSBot(game, net, 'cpu', SAMPLE_SIMS, batch_size=16,
                           random_state=rng)
        ply = 0
        while not state.is_terminal() and len(positions) < n_positions:
            root = bot.mcts_search(state)
            if ply % 4 == 0:
                positions.append(state.clone())
                game_ids.append(g)
            thompson = ply < 30
            a = c4.root_pick(root, rng, thompson=thompson,
                             temp=(1.0 if thompson else 8.0))
            state.apply_action(a)
            ply += 1
        g += 1
    return positions, np.array(game_ids)


def _snapshot(c4, root):
    """What the search would hand downstream if it stopped right now: the
    per-action values root_pick reads, which edges are eligible, and the move
    that would therefore be played."""
    vals = c4.dir_value(root.alpha_sel)
    touched = (root.visits() > 0) | (root.term >= 0)
    if touched.any():
        best = int(np.argmax(np.where(touched, vals, -np.inf)))
    else:
        best = int(np.argmax(vals))
    return vals, touched, best


def trace_position(c4, game, net, state, rng):
    """One capped search, snapshotting the root at every checkpoint.

    Mirrors C4MCTSBot.mcts_search_checkpointed (same wave loop, same solved
    check, same reference belief before any search) but records the state of
    the root instead of collapsing it to one scalar, so labels stay a
    post-processing decision."""
    bot = c4.C4MCTSBot(game, net, 'cpu', CAP, batch_size=16, random_state=rng)
    root = c4.expand_node(net, 'cpu', state)
    feats = c4._root_features(root)
    prior = c4.observed_alpha(root.nacc, c4._SEARCH_AGG)

    n_ckpt = CAP // CHECK_EVERY
    av = np.zeros((n_ckpt, K_MAX), dtype=np.float32)
    tch = np.zeros((n_ckpt, K_MAX), dtype=bool)
    best = np.full(n_ckpt, -1, dtype=np.int16)
    salpha = np.zeros((n_ckpt, 3))
    consec = np.zeros(n_ckpt)

    sims, i, solved = 0, 0, False
    prev = prior
    while sims < CAP and i < n_ckpt:
        if c4._node_solved_outcome(root) is not None:
            solved = True
            break
        sims = bot._run_until(root, state, sims, min(sims + CHECK_EVERY, CAP))
        vals, touched, b = _snapshot(c4, root)
        k = len(vals)
        av[i, :k], tch[i, :k], best[i] = vals, touched, b
        cur = c4.observed_alpha(root.nacc, c4._SEARCH_AGG)
        salpha[i] = cur
        consec[i] = c4._dir_kl(prev, cur)
        prev = cur
        i += 1

    # A root that SOLVED stops early; its answer is exact, so hold the last
    # snapshot flat over the remaining checkpoints rather than leaving zeros
    # that a distance-to-final label would read as a wild late swing.
    if i == 0:
        vals, touched, b = _snapshot(c4, root)
        k = len(vals)
        av[0, :k], tch[0, :k], best[0] = vals, touched, b
        salpha[0] = prior
        i = 1
    av[i:], tch[i:], best[i:], salpha[i:] = av[i-1], tch[i-1], best[i-1], salpha[i-1]

    return dict(feats=feats, action_vals=av, touched=tch, best_idx=best,
                state_alpha=salpha, consec_kl=consec, n_valid=i, solved=solved)


def _worker(args):
    tag, bench_dir, label, seed, n_pos = args
    game = _init_process()
    import connect4_dirichlet_utils as c4
    import adaptive_stop_utils as ad

    net = c4.load_benchmark_net(bench_dir, label, ad.MM_SIG)
    positions, game_ids = sample_positions_with_games(
        c4, ad, net, game, n_pos, seed)

    rng = np.random.default_rng(seed + 7717)
    rows = []
    for st in positions:
        r = trace_position(c4, game, net, st, rng)
        margin, conf, n_legal = r['feats']
        r['feats'] = [1.0, margin, np.log(max(conf, 1e-6)), float(n_legal)]
        r['obs'] = np.asarray(c4.make_obs(st), dtype=np.float32)
        rows.append(r)

    keys = ('feats', 'obs', 'action_vals', 'touched', 'best_idx',
            'state_alpha', 'consec_kl', 'n_valid', 'solved')
    out = {k: np.asarray([r[k] for r in rows]) for k in keys}
    out['game_id'] = game_ids
    return out


def collect(tag, bench_dir, label):
    per = N_POSITIONS // N_WORKERS
    jobs = [(tag, bench_dir, label, 1000 * (w + 1), per) for w in range(N_WORKERS)]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)

    merged = {k: np.concatenate([p[k] for p in parts])
              for k in parts[0] if k != 'game_id'}
    # Game ids restart per worker; offset them so they stay globally unique
    # (the analysis splits on these).
    gids, off = [], 0
    for p in parts:
        gids.append(p['game_id'] + off)
        off += p['game_id'].max() + 1
    merged['game_id'] = np.concatenate(gids)
    merged['cap'] = CAP
    merged['check_every'] = CHECK_EVERY

    out = f'probe_data_{tag}.npz'
    np.savez_compressed(out, **merged)
    print(f'[{tag}] {len(merged["feats"])} positions from '
          f'{merged["game_id"].max()+1} games in {time.time()-t0:.0f}s -> {out}',
          flush=True)


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, d, lab in NETS:
        if only and tag != only:
            continue
        if not os.path.exists(os.path.join(d, f'bench_{lab}.pt')):
            print(f'[{tag}] missing checkpoint, skipping')
            continue
        collect(tag, d, lab)
