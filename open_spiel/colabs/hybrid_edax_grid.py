"""HYBRID vs Edax: full grid over training checkpoint x Edax level x MCTS sims.

    python hybrid_edax_grid.py [report]

Saves incrementally (one cell at a time) to hybrid_edax_grid.json so an
external kill loses at most one in-flight cell, not the whole run --
tournament.py's all-at-the-end save has bitten this project twice already.
"""
import json
import os
import re
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_HYBRID'
CHECKPOINTS = [10000, 30000, 50000, 70000, 90000, 100000]
EDAX_LEVELS = [1, 3, 5, 7]
SIMS_LIST = [100, 400, 1000]
GAMES_PER_CELL = 50
OUR_TEMP = 64.0
N_WORKERS = 4
OUT = 'hybrid_edax_grid.json'
LOG = 'hybrid_edax_grid.log'


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def _sig():
    import torch
    sd = torch.load(f'{CKPT_DIR}/bench_{CHECKPOINTS[0]}.pt', map_location='cpu',
                    weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def _worker(args):
    ep, level, sims, n_games, seed, seat0 = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import edax_bot as eb
    import policy_screen as ps
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                  selection='dirichlet', virtual_loss=1.0)
    sig = _sig()
    net = c4.load_benchmark_net(CKPT_DIR, str(ep), sig)
    opp = eb.EdaxBot(game, level=level)
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    pts = 0.0
    for gi in range(n_games):
        seat = (seat0 + gi) % 2
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.current_player() == seat:
                bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                                   temp=OUR_TEMP, random_state=rng)
                a = ps.pick(c4, bot.mcts_search(st), pol)
            else:
                a = opp.step(st)
            st.apply_action(a)
        r = st.returns()[seat]
        pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    opp.close()
    return pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main():
    import multiprocessing as mp
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    cells = [(ep, lv, sm) for ep in CHECKPOINTS
             for lv in EDAX_LEVELS for sm in SIMS_LIST]
    todo = [c for c in cells if f'{c[0]}|L{c[1]}|s{c[2]}' not in done]
    _log(f'=== hybrid_edax_grid: {len(CHECKPOINTS)} checkpoints x '
         f'{len(EDAX_LEVELS)} levels x {len(SIMS_LIST)} sim counts = '
         f'{len(cells)} cells, {len(done)} already done, '
         f'{len(todo)} to run ({GAMES_PER_CELL} games/cell) ===')
    t0 = time.time()
    for i, (ep, level, sims) in enumerate(todo, 1):
        per = max(2, GAMES_PER_CELL // N_WORKERS)
        jobs = [(ep, level, sims, per, 5100 + 61 * w, w % 2)
                for w in range(N_WORKERS)]
        tc = time.time()
        with mp.Pool(N_WORKERS) as pool:
            parts = pool.map(_worker, jobs)
        pts = sum(p for p, _ in parts)
        n = sum(g for _, g in parts)
        frac = pts / n
        lo, hi = wilson(frac, n)
        key = f'{ep}|L{level}|s{sims}'
        done[key] = dict(score=frac, n=n, lo=lo, hi=hi)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{i}/{len(todo)}] ep{ep} vs EdaxL{level} @{sims}sims: '
             f'{frac:6.1%} [{lo:.1%},{hi:.1%}] n={n} '
             f'({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)')
    _log('=== HYBRID_EDAX_GRID DONE ===')


def report():
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'{"ep":>8}' + ''.join(f'{f"L{lv}":>24}' for lv in EDAX_LEVELS))
    for ep in CHECKPOINTS:
        for sm in SIMS_LIST:
            row = f'{ep:>7}@{sm:<4}'
            for lv in EDAX_LEVELS:
                key = f'{ep}|L{lv}|s{sm}'
                if key in done:
                    d = done[key]
                    row += f'{d["score"]:>10.1%} [{d["lo"]:.0%},{d["hi"]:.0%}]'
                else:
                    row += f'{"--":>24}'
            print(row)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
