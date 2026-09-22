"""HYBRID vs Edax, XOT-opening rerun of hybrid_edax_grid.py.

WHY THIS EXISTS. The fixed-start grid (hybrid_edax_grid.json) is not measuring
what its n=48 claims. Our agent plays near-greedy (temp 64) and Edax at a fixed
level is fully deterministic, so from the ONE standard start position those 48
"games" are a handful of lines replayed, not 48 independent samples. The tell is
in the output: eleven cells read exactly 0.0% [0%,7%] -- an agent with genuine
~6% strength essentially never goes 0/48, but an agent that keeps walking into
the same refutation does. Those Wilson intervals are therefore too narrow and
the L5/L7 columns most likely UNDERSTATE strength.

OLIVAW (arXiv 2103.17228) hit the same problem and solved it the standard way:
random XOT openings -- the first 8 plies taken from a curated list of 10,784
sequences whose resulting position Edax and Ntest both score within +/-2 (i.e.
near-even, so the opening does not hand either side the game). This script is
hybrid_edax_grid.py with that one change, so the two JSONs are directly
comparable cell-for-cell.

DESIGN. 24 openings x 2 seats = 48 games/cell, and it is the SAME 24 openings
(fixed shuffle seed) in every cell. That makes the whole grid a PAIRED
comparison: checkpoint-vs-checkpoint and level-vs-level differences are no
longer confounded by which openings happened to come up, and playing each
opening from both seats cancels any residual black/white imbalance in it.

Incremental per-cell save, as in the original -- an external kill loses at most
the in-flight cell.

    python hybrid_edax_grid_xot.py [report|compare]
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
N_OPENINGS = 24          # x2 seats = 48 games/cell, matching the original n
OUR_TEMP = 64.0
N_WORKERS = 4
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260922
OUT = 'hybrid_edax_grid_xot.json'
LOG = 'hybrid_edax_grid_xot.log'
BASE = 'hybrid_edax_grid.json'    # the fixed-start run, for `compare`


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def load_openings():
    """The fixed set of XOT openings used by EVERY cell (paired design)."""
    with open(XOT_FILE) as fh:
        seqs = [ln.strip() for ln in fh if ln.strip()]
    rng = np.random.default_rng(XOT_SEED)
    pick = rng.choice(len(seqs), size=N_OPENINGS, replace=False)
    return [seqs[i] for i in sorted(pick)]


def _sig():
    import torch
    sd = torch.load(f'{CKPT_DIR}/bench_{CHECKPOINTS[0]}.pt', map_location='cpu',
                    weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def _apply_opening(state, seq, amap):
    """Play the 8-ply XOT sequence. Validated legal for all 10,784 entries;
    a pass is impossible this early, so no pass handling is needed."""
    for i in range(0, 16, 2):
        a = amap[seq[i:i + 2]]
        if a not in state.legal_actions():
            raise RuntimeError(f'XOT {seq} illegal at ply {i//2}')
        state.apply_action(a)


def _worker(args):
    ep, level, sims, seqs, seed = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import edax_bot as eb
    import policy_screen as ps
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                  selection='dirichlet', virtual_loss=1.0)
    amap = eb._action_map(game)
    net = c4.load_benchmark_net(CKPT_DIR, str(ep), _sig())
    opp = eb.EdaxBot(game, level=level)
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    pts = 0.0
    n = 0
    for seq in seqs:
        for seat in (0, 1):           # same opening from both sides
            st = game.new_initial_state()
            _apply_opening(st, seq, amap)
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
            n += 1
    opp.close()
    return pts, n


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main():
    import multiprocessing as mp
    openings = load_openings()
    shards = [openings[i::N_WORKERS] for i in range(N_WORKERS)]
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    cells = [(ep, lv, sm) for ep in CHECKPOINTS
             for lv in EDAX_LEVELS for sm in SIMS_LIST]
    todo = [c for c in cells if f'{c[0]}|L{c[1]}|s{c[2]}' not in done]
    _log(f'=== hybrid_edax_grid_xot: {len(cells)} cells, {len(done)} done, '
         f'{len(todo)} to run | {N_OPENINGS} XOT openings x 2 seats = '
         f'{N_OPENINGS*2} games/cell, same openings every cell (paired) ===')
    t0 = time.time()
    for i, (ep, level, sims) in enumerate(todo, 1):
        jobs = [(ep, level, sims, shards[w], 8100 + 61 * w)
                for w in range(N_WORKERS)]
        tc = time.time()
        with mp.Pool(N_WORKERS) as pool:
            parts = pool.map(_worker, jobs)
        pts = sum(p for p, _ in parts)
        n = sum(g for _, g in parts)
        frac = pts / n
        lo, hi = wilson(frac, n)
        done[f'{ep}|L{level}|s{sims}'] = dict(score=frac, n=n, lo=lo, hi=hi)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{i}/{len(todo)}] ep{ep} vs EdaxL{level} @{sims}sims: '
             f'{frac:6.1%} [{lo:.1%},{hi:.1%}] n={n} '
             f'({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)')
    _log('=== HYBRID_EDAX_GRID_XOT DONE ===')


def report(path=OUT, title='XOT openings'):
    with open(path) as fh:
        done = json.load(fh)
    print(f'\n--- {title} ---')
    print(f'{"ep":>12}' + ''.join(f'{f"L{lv}":>24}' for lv in EDAX_LEVELS))
    for ep in CHECKPOINTS:
        for sm in SIMS_LIST:
            row = f'{ep:>7}@{sm:<4}'
            for lv in EDAX_LEVELS:
                d = done.get(f'{ep}|L{lv}|s{sm}')
                row += (f'{d["score"]:>10.1%} [{d["lo"]:.0%},{d["hi"]:.0%}]'
                        if d else f'{"--":>24}')
            print(row)


def compare():
    """XOT vs fixed-start, cell by cell -- the point of the rerun."""
    with open(OUT) as fh:
        xot = json.load(fh)
    with open(BASE) as fh:
        base = json.load(fh)
    print(f'{"cell":>22}{"fixed":>9}{"XOT":>9}{"delta":>9}')
    deltas = []
    for ep in CHECKPOINTS:
        for sm in SIMS_LIST:
            for lv in EDAX_LEVELS:
                k = f'{ep}|L{lv}|s{sm}'
                if k not in xot or k not in base:
                    continue
                b, x = base[k]['score'], xot[k]['score']
                deltas.append(x - b)
                print(f'{f"ep{ep} L{lv} @{sm}":>22}'
                      f'{b:>9.1%}{x:>9.1%}{x-b:>+9.1%}')
    if deltas:
        d = np.array(deltas)
        print(f'\nmean delta {d.mean():+.1%}  median {np.median(d):+.1%}  '
              f'n={len(d)} cells, XOT higher in {(d > 0).sum()}')
        zeros_base = sum(1 for v in base.values() if v['score'] == 0.0)
        zeros_xot = sum(1 for v in xot.values() if v['score'] == 0.0)
        print(f'exact-0% cells: fixed-start {zeros_base}, XOT {zeros_xot}')


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'report':
        report()
    elif cmd == 'compare':
        report(BASE, 'fixed start (original)')
        report()
        print()
        compare()
    else:
        main()
