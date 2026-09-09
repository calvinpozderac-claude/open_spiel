"""Direct head-to-head: MM_long's final network vs MIXTURE's final network.

Same protocol as mm_long_bench.py's Edax matches -- 1000 sims/side (the
budget both were trained and deployed at), OUR_TEMP=64 for both in-tree
descent and the final move pick (C4MCTSBot's `temp` argument IS the eval-time
descent temperature; it is unrelated to the SELF-PLAY descent_temp each run
trained under, which only the training-time _SlotEngine reads) -- but the
opponent is the other trained net instead of Edax, and each side's own
search_agg is set via c4.set_search() immediately before ITS move, so MM_long
reads its own additive_mle accumulator statistics and MIXTURE reads its
mixture-collapsed ones off the identical tree bookkeeping.

This is the test the whole thread has been building to: MM_long has a ~0.04
lower MSE and a ~4x update-count advantage (batch 256 x 4 steps/ep vs
mixture's 1024 x 1) but a confidence profile inverted 300x from correct;
MIXTURE has honest calibration and a clean, monotone Elo pool but has never
played a game against anything but Edax or its own earlier checkpoints. This
answers whether the calibration difference shows up on the board.

    python head_to_head.py
"""
import multiprocessing as mp
import time

import numpy as np

SIMS = 1000
GAMES = 101             # odd: an outright draw in total games is impossible
OUR_TEMP = 64.0
N_WORKERS = 3
LOG = 'head_to_head_avg.log'

MM_DIR, MM_LABEL, MM_AGG = 'othello_benchmark/bench_MM_long_snaps', '50000', 'additive_mle'
MX_DIR, MX_LABEL, MX_AGG = 'othello_benchmark/bench_AVGCONC', '50000', 'avgconc'


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    # selection='dirichlet' (Thompson) and virtual_loss are shared machinery;
    # only search_agg differs per side, toggled per move below.
    c4.set_search(selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _worker(args):
    n_games, seed, mm_first = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps

    net_mm = c4.load_benchmark_net(MM_DIR, MM_LABEL, ad.MM_SIG)
    import re
    import torch
    sd = torch.load(f'{MX_DIR}/bench_{MX_LABEL}.pt', map_location='cpu', weights_only=True)
    mx_sig = (int(sd['stem.0.weight'].shape[0]),
              len({int(m.group(1)) for k in sd
                   if (m := re.match(r'body\.(\d+)\.', k))}),
              int(sd['head.0.weight'].shape[0]))
    net_mx = c4.load_benchmark_net(MX_DIR, MX_LABEL, mx_sig)

    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    mm_pts = 0.0
    for gi in range(n_games):
        mm_seat = (0 if mm_first else 1)
        if gi % 2 == 1:
            mm_seat = 1 - mm_seat          # alternate colors within the batch
        st = game.new_initial_state()
        while not st.is_terminal():
            is_mm = (st.current_player() == mm_seat)
            net, agg = (net_mm, MM_AGG) if is_mm else (net_mx, MX_AGG)
            c4.set_search(search_agg=agg, target_agg=agg)
            bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                               temp=OUR_TEMP, random_state=rng)
            a = ps.pick(c4, bot.mcts_search(st), pol)
            st.apply_action(a)
        r = st.returns()[mm_seat]
        mm_pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return mm_pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main():
    _log(f'=== head-to-head: MM_long@{MM_LABEL} (additive_mle) vs '
         f'MIXTURE@{MX_LABEL} (MLE-search/mix-target) -- {SIMS} sims/side, temp {OUR_TEMP}, '
         f'{GAMES} games ===')
    per = GAMES // N_WORKERS
    rem = GAMES - per * N_WORKERS
    jobs = []
    for w in range(N_WORKERS):
        n = per + (1 if w < rem else 0)
        jobs.append((n, 4200 + 97 * w, w % 2 == 0))    # alternate who opens
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    mm_pts = sum(p for p, _ in parts)
    n = sum(g for _, g in parts)
    frac = mm_pts / n
    lo, hi = wilson(frac, n)
    _log(f'MM_long score: {mm_pts:.1f}/{n} = {frac:.1%}  '
         f'[{lo:.1%}, {hi:.1%}]  ({time.time()-t0:.0f}s)')
    _log(f'MIXTURE score: {n-mm_pts:.1f}/{n} = {1-frac:.1%}  '
         f'[{1-hi:.1%}, {1-lo:.1%}]')
    verdict = ('MM_long wins' if lo > 0.5 else
              'MIXTURE wins' if hi < 0.5 else 'statistically even')
    _log(f'=== {verdict} ===')


if __name__ == '__main__':
    main()
