"""Find which Edax levels bracket our model, before wiring them into the ladder.

A rung is only useful if it is beatable-but-not-trivially so; rungs pinned at
0% or 100% measure nothing, which is exactly how the MCTS ladder failed above
episode 4000. So the levels get chosen from data rather than guessed.

Our side plays the fixed inference configuration the ladder uses everywhere
(64 sims, deterministic descent), so these numbers are directly comparable to
the existing MCTS rungs.

    python edax_calibrate.py [--ckpt DIR] [--label 20000] [--games 20]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

# (level, epsilon).  Level is unusable as a dial -- level 1 scores 50% against
# bench_OPT@20000 and level 2 scores 0% -- so strength is dialled by how often
# Edax is overridden with a random legal move instead.
RUNGS = [(2, 0.60), (2, 0.45), (2, 0.30), (2, 0.20), (2, 0.10), (2, 0.05),
         (2, 0.0), (1, 0.0)]
GAMES = 20
OUR_SIMS = 64
OUR_POLICY = dict(mode='value', k=0.0, sel_temp=1e4)
OPENING_PLIES = 2
N_WORKERS = 5
DEFAULT_CKPT = ('othello_benchmark/bench_OPT', '20000')


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _worker(args):
    ckpt_dir, label, (level, eps), n_games, seed = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    import edax_bot as eb

    net = c4.load_benchmark_net(ckpt_dir, label, ad.MM_SIG)
    bot = eb.EdaxBot(game, level=level, epsilon=eps, seed=seed)
    rng = np.random.default_rng(seed)

    pts, played, t0 = 0.0, 0, time.time()
    for _pair in range(max(1, n_games // 2)):
        st0 = game.new_initial_state()
        opening = []
        for _ in range(OPENING_PLIES):
            leg = st0.legal_actions()
            a = int(leg[rng.integers(len(leg))])
            st0.apply_action(a)
            opening.append(a)
        for seat in (0, 1):
            st = game.new_initial_state()
            for a in opening:
                st.apply_action(int(a))
            while not st.is_terminal():
                if st.current_player() == seat:
                    b = c4.C4MCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=16,
                                     temp=OUR_POLICY['sel_temp'],
                                     random_state=rng)
                    a = ps.pick(c4, b.mcts_search(st), OUR_POLICY)
                else:
                    a = bot.step(st)
                st.apply_action(a)
            r = st.returns()[seat]
            pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
            played += 1
    bot.close()
    return (level, eps), pts, played, time.time() - t0


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main(ckpt_dir, label, games, rungs=RUNGS):
    print(f'{ckpt_dir}@{label}   our side: {OUR_SIMS} sims, {OUR_POLICY}')
    jobs = []
    per = max(2, games // N_WORKERS)
    for rung in rungs:
        for w in range(N_WORKERS):
            jobs.append((ckpt_dir, label, rung, per, 4000 + 91 * w + rung[0]))
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    agg = {}
    for rung, pts, n, el in parts:
        p, m, t = agg.get(rung, (0.0, 0, 0.0))
        agg[rung] = (p + pts, m + n, max(t, el))
    for rung in rungs:
        pts, n, el = agg[rung]
        frac = pts / n
        lo, hi = wilson(frac, n)
        note = ''
        if lo > 0.9:
            note = '  <- too weak, saturated'
        elif hi < 0.1:
            note = '  <- too strong, saturated'
        elif 0.2 < frac < 0.8:
            note = '  <- USEFUL RUNG'
        print(f'  vs Edax L{rung[0]} eps={rung[1]:<5} {frac:6.1%}  '
              f'[{lo:.1%}, {hi:.1%}]  n={n}  {el:.0f}s{note}')


if __name__ == '__main__':
    ck, lab = DEFAULT_CKPT
    if '--ckpt' in sys.argv:
        ck = sys.argv[sys.argv.index('--ckpt') + 1]
    if '--label' in sys.argv:
        lab = sys.argv[sys.argv.index('--label') + 1]
    g = GAMES
    if '--games' in sys.argv:
        g = int(sys.argv[sys.argv.index('--games') + 1])
    main(ck, lab, g)
