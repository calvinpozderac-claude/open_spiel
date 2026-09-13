"""Head-to-head: DISTILL vs ZGROUND, both @ep10000, playing each other directly.

Both trained identically except the loss (DISTILL klv+kla+cons; ZGROUND
+cev+cea) and both deploy the same search (pathsum, Thompson selection).  So
one set_search(pathsum) serves both; only the weights differ.  Greedy value
pick at eval (root_pick thompson=False), colors alternated, fixed sims/side.

    python duel_distill_zground.py [sims] [games]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 121
OUR_TEMP = 64.0
N_WORKERS = 4
LOG = 'duel_distill_zground.log'
SIG = (32, 10, 8)
A_DIR, A_LABEL = 'othello_benchmark/bench_DISTILL', '10000'
B_DIR, B_LABEL = 'othello_benchmark/bench_ZGROUND', '10000'


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
            is_a = (st.current_player() == a_seat)
            net = net_a if is_a else net_b
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
    _log(f'=== DISTILL@{A_LABEL} vs ZGROUND@{B_LABEL} -- {SIMS} sims/side, '
         f'temp {OUR_TEMP}, {GAMES} games (colors alternated) ===')
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
    _log(f'DISTILL score: {a_pts:.1f}/{n} = {frac:.1%}  [{lo:.1%}, {hi:.1%}]  '
         f'({time.time()-t0:.0f}s)')
    _log(f'ZGROUND score: {n-a_pts:.1f}/{n} = {1-frac:.1%}  '
         f'[{1-hi:.1%}, {1-lo:.1%}]')
    verdict = ('DISTILL wins' if lo > 0.5 else
               'ZGROUND wins' if hi < 0.5 else 'statistically even')
    _log(f'=== {verdict} ===')


if __name__ == '__main__':
    main()
