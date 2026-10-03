"""OLIVAW-AZ vs Edax, at OLIVAW's own levels and node budget.

    python az_edax_bench.py [report]

THE QUESTION. Our AlphaZero arm ran what is meant to be OLIVAW's procedure, yet
the paper reports O-1000 (1000 nodes/move) beating E4 and E6 and losing only to
E8 -- i.e. roughly Edax depth 6-7. Is our reproduction really far short of that,
and if so by how much? Everything else is speculation until this is measured.

PROTOCOL, matched to the paper as closely as we can:
  * Edax levels 4, 6, 8 -- literally their E4/E6/E8 (level = alpha-beta depth).
  * 1000 sims for our net = their O-1000's 1.0e3 max nodes/move.
  * XOT random openings, which is what they used for these games.
  * Edax with -book-usage off, as they also specify.
  * Greedy play from our side; each opening played from BOTH seats.

Deliberately runs with FEW workers: the TZ arm and the live ladder are both
using this 6-core box, and this benchmark is the lowest-priority of the three.
"""
import json
import os
import sys
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
CHECKPOINTS = [25000, 45000, 50000]
EDAX_LEVELS = [4, 6, 8]
SIMS = 1000
SIG = (64, 10, 8)
C_PUCT = 1.5
N_OPENINGS = 12
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926          # the SAME openings the BT ladder uses
N_WORKERS = 2                # low, deliberately: two other jobs share the box
OUT = 'az_edax_bench.json'
LOG = 'az_edax_bench.log'


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
    import connect4_alphazero_utils as az
    import edax_bot as eb
    game = c4.load_game('othello')
    c4.set_game(game)
    az.set_search(c_puct=C_PUCT, virtual_loss=1.0)
    amap = eb._action_map(game)
    net = az.load_benchmark_net(CKPT_DIR, str(ep), SIG)
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
                    bot = az.AZMCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                                       random_state=rng, noise_frac=0.0)
                    a = az.root_pick(bot.mcts_search(st), rng, sample=False)
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
    _log(f'=== az_edax_bench: {len(cells)} cells, {len(done)} done, '
         f'{len(todo)} to run | {SIMS} sims vs Edax L{EDAX_LEVELS}, '
         f'{N_OPENINGS} XOT openings x 2 seats = {N_OPENINGS*2} games/cell ===')
    t0 = time.time()
    for i, (ep, level) in enumerate(todo, 1):
        jobs = [(ep, level, sh, 4400 + 97 * k)
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
        _log(f'[{i}/{len(todo)}] AZ{ep//1000}k vs EdaxL{level} @{SIMS}sims: '
             f'{w}W-{d}D-{l}L = {score:.1%} [{lo:.0%},{hi:.0%}] '
             f'({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)')
    _log('=== AZ_EDAX_BENCH DONE ===')
    report()


def report():
    if not os.path.exists(OUT):
        print(f'no results yet ({OUT})')
        return
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'\n  OLIVAW-AZ vs Edax @{SIMS} sims, XOT openings, '
          f'{N_OPENINGS*2} games/cell')
    print(f'  {"ckpt":<8}' + ''.join(f'{f"EdaxL{lv}":>22}' for lv in EDAX_LEVELS))
    for ep in CHECKPOINTS:
        row = f'  AZ{ep//1000}k'.ljust(10)
        for lv in EDAX_LEVELS:
            c = done.get(f'{ep}|L{lv}')
            row += (f'{c["w"]}-{c["d"]}-{c["l"]} {c["score"]:>6.1%}'.rjust(22)
                    if c else f'{"--":>22}')
        print(row)
    print('\n  OLIVAW paper, same budget: O-1000 beats E4 and E6, '
          'loses to E8 (~depth 6-7).')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
