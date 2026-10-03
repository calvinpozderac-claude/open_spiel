"""OLIVAW-TZ vs Edax, same protocol as az_edax_bench.py -- does the Thompson
arm's ~415 Elo lead over AZ (see olivaw_live_ladder) translate into any real
absolute strength, or does it share AZ's capacity floor?

    python tz_edax_bench.py [report]

AZ's result at this exact protocol (1000 sims, XOT, greedy): 1 draw in 216
games against L4/L6/L8 -- a hard floor, confirmed NOT a search problem (see
az_selfsearch_big: 10x search is worth a normal +203 Elo on a fixed net). If
TZ30k also reads ~0%, the two arms share one capacity ceiling and their BT gap
is a within-floor effect. If TZ30k scores meaningfully above zero, the Thompson
recipe buys real absolute strength, not just relative standing against a weak
opponent pool.

Starts with TZ30k alone (the latest completed, currently-best checkpoint) to
keep this cheap and out of the way of training + the live cross-arm ladder,
both already running. Add checkpoints to CHECKPOINTS to extend later.
"""
import json
import os
import sys
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_OLIVAW_TZ'
CHECKPOINTS = [30000]
EDAX_LEVELS = [4, 6, 8]
SIMS = 1000
SIG = (64, 10, 8)
N_OPENINGS = 12
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926          # the SAME openings az_edax_bench.py used
N_WORKERS = 2                 # low: training + the live ladder share this box
OUT = 'tz_edax_bench.json'
LOG = 'tz_edax_bench.log'


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


def _worker(args):
    ep, level, ops, seed = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import edax_bot as eb
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                  selection='dirichlet', virtual_loss=1.0)
    amap = eb._action_map(game)
    net = c4.load_benchmark_net(CKPT_DIR, str(ep), SIG)
    opp = eb.EdaxBot(game, level=level)
    rng = np.random.default_rng(seed)
    w = d = l = 0
    for seq in ops:
        for seat in (0, 1):
            st = game.new_initial_state()
            for k in range(0, 16, 2):
                st.apply_action(amap[seq[k:k + 2]])
            while not st.is_terminal():
                if st.current_player() == seat:
                    bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                                       temp=64.0, random_state=rng)
                    root = bot.mcts_search(st)
                    a = c4.root_pick(root, rng, thompson=False, temp=64.0)
                else:
                    a = opp.step(st)
                st.apply_action(int(a))
            r = st.returns()[seat]
            if r > 0:
                w += 1
            elif r < 0:
                l += 1
            else:
                d += 1
    opp.close()
    return w, d, l


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return c - h, c + h


def main():
    import multiprocessing as mp
    ops = openings()
    shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    cells = [(ep, lv) for ep in CHECKPOINTS for lv in EDAX_LEVELS]
    todo = [c for c in cells if f'{c[0]}|L{c[1]}' not in done]
    _log(f'=== tz_edax_bench: {len(cells)} cells, {len(done)} done, '
         f'{len(todo)} to run | {SIMS} sims vs Edax L{EDAX_LEVELS}, '
         f'{N_OPENINGS} XOT openings x 2 seats = {N_OPENINGS*2} games/cell ===')
    t0 = time.time()
    for i, (ep, level) in enumerate(todo, 1):
        jobs = [(ep, level, sh, 3300 + 61 * k)
                for k, sh in enumerate(shards) if sh]
        tc = time.time()
        with mp.Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        w = sum(p[0] for p in parts)
        d = sum(p[1] for p in parts)
        l = sum(p[2] for p in parts)
        n = w + d + l
        score = (w + 0.5 * d) / n
        lo, hi = wilson(score, n)
        done[f'{ep}|L{level}'] = dict(w=w, d=d, l=l, n=n, score=score,
                                      lo=lo, hi=hi)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{i}/{len(todo)}] TZ{ep//1000}k vs EdaxL{level} @{SIMS}sims: '
             f'{w}W-{d}D-{l}L = {score:.1%} [{lo:.0%},{hi:.0%}] '
             f'({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)')
    _log('=== TZ_EDAX_BENCH DONE ===')
    report()


def report():
    if not os.path.exists(OUT):
        print(f'no results yet ({OUT})')
        return
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'\n  OLIVAW-TZ vs Edax @{SIMS} sims, XOT openings, '
          f'{N_OPENINGS*2} games/cell')
    print(f'  {"ckpt":<8}' + ''.join(f'{f"EdaxL{lv}":>22}' for lv in EDAX_LEVELS))
    for ep in CHECKPOINTS:
        row = f'  TZ{ep//1000}k'.ljust(10)
        for lv in EDAX_LEVELS:
            c = done.get(f'{ep}|L{lv}')
            row += (f'{c["w"]}-{c["d"]}-{c["l"]} {c["score"]:>6.1%}'.rjust(22)
                    if c else f'{"--":>22}')
        print(row)
    print('\n  AZ reference, same protocol: 1 draw in 216 games (~0%) -- '
          'the capacity floor.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
