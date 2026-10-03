"""Tighter version of the search-benefit test: does MORE search beat LESS?

    python az_selfsearch_big.py [report]

az_selfsearch.py gave 54.2% for the 1000-sim side against the same net at 100
sims, over only 12 games (95% CI roughly 27-79%). A healthy AlphaZero should
show ~75-85% for a 10x search advantage, so that number points at a problem --
but 12 games cannot distinguish "search is inert" from bad luck.

This runs the CHEAP budgets with many more games, and adds intermediate rungs,
so the result is a curve rather than a point:

    25 vs 100, 100 vs 400, 100 vs 1000, 400 vs 1600

If each rung sits near 50%, search is contributing almost nothing and the AZ
search path is buggy -- which would also mean arm A trained under a degraded
search and a lot more than this benchmark needs revisiting. If the rungs climb
to 70-85%, search works and the flat Edax result was saturation below L4.

All rungs are cheap (<=1600 sims) so this stays a quick diagnostic even at 40
games per rung.
"""
import json
import os
import sys
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
EP = 50000
SIG = (64, 10, 8)
C_PUCT = 1.5
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926
N_OPENINGS = 20                  # x2 seats = 40 games per rung
N_WORKERS = 2
OUT = 'az_selfsearch_big.json'
LOG = 'az_selfsearch_big.log'

PLAN = [(25, 100), (100, 400), (100, 1000), (400, 1600)]


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def openings():
    with open(XOT_FILE) as fh:
        seqs = [ln.strip() for ln in fh if ln.strip()]
    rng = np.random.default_rng(XOT_SEED)
    pick = rng.choice(len(seqs), size=N_OPENINGS, replace=False)
    return [seqs[i] for i in sorted(pick)]


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return c - h, c + h


def _worker(args):
    ep, lo, hi, ops, seed = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import connect4_alphazero_utils as az
    import edax_bot as eb
    game = c4.load_game('othello')
    c4.set_game(game)
    az.set_search(c_puct=C_PUCT, virtual_loss=1.0)
    amap = eb._action_map(game)
    net = az.load_benchmark_net(CKPT_DIR, str(ep), SIG)
    rng = np.random.default_rng(seed)
    hw = hd = hl = 0
    for seq in ops:
        for hi_seat in (0, 1):
            st = game.new_initial_state()
            for k in range(0, 16, 2):
                st.apply_action(amap[seq[k:k + 2]])
            while not st.is_terminal():
                sims = hi if st.current_player() == hi_seat else lo
                bot = az.AZMCTSBot(game, net, 'cpu', sims, batch_size=16,
                                   random_state=rng, noise_frac=0.0)
                a = az.root_pick(bot.mcts_search(st), rng, sample=False)
                st.apply_action(int(a))
            r = st.returns()[hi_seat]
            if r > 0:
                hw += 1
            elif r < 0:
                hl += 1
            else:
                hd += 1
    return hw, hd, hl


def main():
    import multiprocessing as mp
    ops = openings()
    shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    _log(f'=== az_selfsearch_big: AZ{EP//1000}k vs ITSELF, {PLAN}, '
         f'{N_OPENINGS*2} games/rung ===')
    t0 = time.time()
    for lo, hi in PLAN:
        key = f'{lo}v{hi}'
        if key in done:
            continue
        jobs = [(EP, lo, hi, sh, 7700 + 53 * k)
                for k, sh in enumerate(shards) if sh]
        tc = time.time()
        with mp.Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        hw = sum(p[0] for p in parts)
        hd = sum(p[1] for p in parts)
        hl = sum(p[2] for p in parts)
        n = hw + hd + hl
        sc = (hw + 0.5 * hd) / n
        blo, bhi = wilson(sc, n)
        done[key] = dict(hw=hw, hd=hd, hl=hl, n=n, score=sc, lo=blo, hi=bhi)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{lo} vs {hi}] HIGH: {hw}W-{hd}D-{hl}L = {sc:.1%} '
             f'[{blo:.0%},{bhi:.0%}] of {n} ({time.time()-tc:.0f}s)')
    _log('=== AZ_SELFSEARCH_BIG DONE ===')
    report()


def report():
    if not os.path.exists(OUT):
        print('no results yet')
        return
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'\n  AZ{EP//1000}k vs ITSELF -- same network, only sims differ')
    print(f'  {"matchup":>14}{"high W-D-L":>14}{"n":>5}{"high score":>12}'
          f'{"95% CI":>16}')
    for lo, hi in PLAN:
        c = done.get(f'{lo}v{hi}')
        if c:
            ci = f'[{c["lo"]:.0%},{c["hi"]:.0%}]'
            print(f'  {f"{lo} vs {hi}":>14}'
                  f'{f"{c["hw"]}-{c["hd"]}-{c["hl"]}":>14}{c["n"]:>5}'
                  f'{c["score"]:>12.1%}{ci:>16}')
    print('\n  Healthy AlphaZero: a 4x search edge is worth ~+150 Elo (~70%),')
    print('  a 10x edge ~+250 Elo (~80%). Rungs near 50% mean search is inert.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
