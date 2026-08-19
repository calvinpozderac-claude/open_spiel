"""Score annealed checkpoints against Edax L1 while training continues.

The internal DEEP Elo cannot answer this -- it is the metric that reported
+215 Elo across the span Edax says is flat. Runs on 2 workers so the training
job keeps its cores.
"""
import multiprocessing as mp, sys, time
import numpy as np
import edax_eval as ee

WORKERS = 3


def run(models, games):
    jobs = []
    per = max(2, games // WORKERS)
    for name, d, lab in models:
        for w in range(WORKERS):
            jobs.append((name, d, lab, 'thompson', ('L1',), per,
                         6100 + 53 * w, w % 2))
    with mp.Pool(WORKERS) as pool:
        parts = pool.map(ee._worker, jobs)
    agg = {}
    for name, pts, n in parts:
        p, m = agg.get(name, (0.0, 0))
        agg[name] = (p + pts, m + n)
    for name, _d, _l in models:
        pts, n = agg[name]
        f = pts / n
        lo, hi = ee.wilson(f, n)
        print(f'  {name:<16} {f:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}', flush=True)


if __name__ == '__main__':
    labs = sys.argv[1:] or ['21000', '22000']
    t0 = time.time()
    print('annealed checkpoints vs Edax L1 (baseline: OPT@20000 = 37.0%)')
    run([(f'anneal@{l}', 'othello_benchmark/bench_OPTanneal', l) for l in labs],
        200)
    print(f'  ({time.time()-t0:.0f}s)')
