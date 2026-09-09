"""How much of a benchmark cell is the network, and how much is the seed?

The ep16000 recheck reproduced 32.4%/2.0% to the digit, so a cell is a pure
function of (network, level, seeds) -- the Wilson interval is measuring
sampling noise that this protocol does not have. What it cannot see is
matchup variance: Edax at fixed -l is deterministic, so 51 games explore only
as many distinct lines as OUR sampling happens to produce, and every
checkpoint faces the identical seeds (3300 + 71*w, hardcoded in _play_level).

This replays one cell under several seed sets. If the spread across seeds is
large, the per-checkpoint numbers are matchup draws, not strength estimates.

    python seed_sensitivity.py <label> <rung> [n_seed_sets]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

import mm_long_bench as B

if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else '17000'
    rung = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    n_sets = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    per = max(2, B.GAMES_PER_LEVEL // B.N_WORKERS)
    print(f'[seedsens] ep {label} vs Edax L{rung:g}, {n_sets} seed sets of '
          f'{per * B.N_WORKERS} games (baseline seeds are 3300+71w)',
          flush=True)

    scores = []
    for s, base in enumerate([3300, 9100, 5500, 7700, 1200][:n_sets]):
        jobs = [(label, rung, per, base + 71 * w, w % 2)
                for w in range(B.N_WORKERS)]
        t0 = time.time()
        with mp.Pool(B.N_WORKERS) as pool:
            parts = pool.map(B._worker, jobs)
        pts = sum(p for p, _ in parts)
        n = sum(g for _, g in parts)
        frac = pts / n
        scores.append(frac)
        lo, hi = B.wilson(frac, n)
        tag = ' <- the seeds the benchmark actually uses' if base == 3300 else ''
        print(f'[seedsens]   base seed {base}: {frac:6.1%} '
              f'[{lo:.1%}, {hi:.1%}]  n={n}  ({time.time()-t0:.0f}s){tag}',
              flush=True)

    a = np.array(scores)
    print(f'[seedsens] spread across seeds: min {a.min():.1%} '
          f'max {a.max():.1%} mean {a.mean():.1%} sd {a.std(ddof=1):.1%}',
          flush=True)
    print('[seedsens] SEEDSENS DONE', flush=True)
