"""Generic head-to-head duel between two trained nets.

    python duel_nets.py <specA> <specB> [sims] [games]
    spec = 'othello_benchmark/bench_XXX:<label>'

Both deploy the same search (pathsum, Thompson), greedy value pick at eval,
colors alternated, fixed sims/side.  Reports A's score with a Wilson CI.
"""
import multiprocessing as mp
import sys
import time

import numpy as np

A_SPEC = sys.argv[1]
B_SPEC = sys.argv[2]
SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 121
OUR_TEMP = 64.0
N_WORKERS = 4
SIG = (32, 10, 8)
A_DIR, A_LABEL = A_SPEC.rsplit(':', 1)
B_DIR, B_LABEL = B_SPEC.rsplit(':', 1)
A_NAME = A_DIR.split('/')[-1].replace('bench_', '')
B_NAME = B_DIR.split('/')[-1].replace('bench_', '')
LOG = f'duel_{A_NAME}_{A_LABEL}_vs_{B_NAME}_{B_LABEL}.log'


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def _worker(args):
    n_games, seed, a_first = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg='pathsum', target_agg='avgconc',
                  selection='dirichlet', virtual_loss=1.0)
    net_a = c4.load_benchmark_net(A_DIR, A_LABEL, SIG)
    net_b = c4.load_benchmark_net(B_DIR, B_LABEL, SIG)
    rng = np.random.default_rng(seed)
    a_pts = 0.0
    for gi in range(n_games):
        a_seat = (0 if a_first else 1)
        if gi % 2 == 1:
            a_seat = 1 - a_seat
        st = game.new_initial_state()
        while not st.is_terminal():
            net = net_a if (st.current_player() == a_seat) else net_b
            bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                               temp=OUR_TEMP, random_state=rng)
            root = bot.mcts_search(st)
            a = c4.root_pick(root, rng, thompson=False, temp=OUR_TEMP)
            st.apply_action(int(a))
        r = st.returns()[a_seat]
        a_pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return a_pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main():
    _log(f'=== {A_NAME}@{A_LABEL} [A] vs {B_NAME}@{B_LABEL} [B] -- {SIMS} '
         f'sims/side, {GAMES} games (colors alternated) ===')
    per = GAMES // N_WORKERS
    rem = GAMES - per * N_WORKERS
    jobs = [(per + (1 if w < rem else 0), 4200 + 97 * w, w % 2 == 0)
            for w in range(N_WORKERS)]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    a_pts = sum(p for p, _ in parts)
    n = sum(g for _, g in parts)
    frac = a_pts / n
    lo, hi = wilson(frac, n)
    _log(f'{A_NAME} [A] score: {a_pts:.1f}/{n} = {frac:.1%}  [{lo:.1%}, {hi:.1%}]'
         f'  ({time.time()-t0:.0f}s)')
    _log(f'{B_NAME} [B] score: {n-a_pts:.1f}/{n} = {1-frac:.1%}  '
         f'[{1-hi:.1%}, {1-lo:.1%}]')
    verdict = (f'{A_NAME} wins' if lo > 0.5 else
               f'{B_NAME} wins' if hi < 0.5 else 'statistically even')
    _log(f'=== {verdict} ===')


if __name__ == '__main__':
    main()
