"""Estimate the CEILING on the probe's r2 by re-labelling the same positions
with a different search RNG.

This is the number that decides whether a learned head is worth building at
all, and neither the deployed r2=0.53 nor any model comparison means much
without it. MCTS is stochastic: Thompson selection and the wave order differ
run to run, so "sims needed" is a noisy measurement of the position, not a
deterministic property of it. Whatever variance that noise contributes is
unpredictable BY CONSTRUCTION -- no model, however large, can fit it.

  Var(y1 - y2) = 2 * Var(noise)          two independent labels, same position
  ceiling r2   = 1 - Var(noise) / Var(y)

If the ceiling is ~0.55 the deployed linear model is already at it and the
head is dead on arrival. If it is ~0.9 the 47% unexplained is real signal.

Positions are regenerated rather than stored: sample_positions_with_games is
seeded, so asking the same worker seed for the first N_REPLICATE positions
reproduces exactly the states the main collection traced.

    python probe_replicate.py           # writes probe_replicate_*.npz
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np

import probe_collect as pc

N_REPLICATE = 400          # per worker, out of the 1000 the main run traced

# One reseed per independent replicate. Two runs bound the ceiling loosely
# (allocating off a single noisy label understates a model that predicts the
# EXPECTED label); a third lets the allocator average two labels and still be
# scored on a run it has never seen, which tightens the bound a lot.
RESEEDS = {2: 99991, 3: 277019}


def _worker(args):
    tag, bench_dir, label, seed, n_pos, reseed = args
    game = pc._init_process()
    import connect4_dirichlet_utils as c4
    import adaptive_stop_utils as ad

    net = c4.load_benchmark_net(bench_dir, label, ad.MM_SIG)
    positions, _ = pc.sample_positions_with_games(c4, ad, net, game, n_pos, seed)

    rng = np.random.default_rng(seed + reseed)
    rows = [pc.trace_position(c4, game, net, st, rng) for st in positions]
    keys = ('action_vals', 'touched', 'best_idx', 'state_alpha', 'consec_kl',
            'n_valid', 'solved')
    return {k: np.asarray([r[k] for r in rows]) for k in keys}


def collect(tag, bench_dir, label, run):
    jobs = [(tag, bench_dir, label, 1000 * (w + 1), N_REPLICATE, RESEEDS[run])
            for w in range(pc.N_WORKERS)]
    t0 = time.time()
    with mp.Pool(pc.N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)

    # Index back into the main run's arrays: worker w owns the contiguous block
    # [w*per, (w+1)*per) there, and its first N_REPLICATE entries are these.
    per = pc.N_POSITIONS // pc.N_WORKERS
    idx = np.concatenate([np.arange(w * per, w * per + N_REPLICATE)
                          for w in range(pc.N_WORKERS)])

    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    merged.update(main_index=idx, cap=pc.CAP, check_every=pc.CHECK_EVERY)
    suffix = '' if run == 2 else str(run)
    out = f'probe_replicate{suffix}_{tag}.npz'
    np.savez_compressed(out, **merged)
    print(f'[{tag}] run {run}: {len(idx)} replicate labels in '
          f'{time.time()-t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    only = sys.argv[2] if len(sys.argv) > 2 else None
    for tag, d, lab in pc.NETS:
        if only and tag != only:
            continue
        if not os.path.exists(f'probe_data_{tag}.npz'):
            print(f'[{tag}] main collection missing, skipping')
            continue
        collect(tag, d, lab, run)
