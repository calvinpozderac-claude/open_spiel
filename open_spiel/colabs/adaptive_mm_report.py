"""Compares the three adaptive-stopping MM checkpoints (bench_MMadapt005/010/030,
trained by run_adaptive_mm_sweep.py) against the reference bench_MM checkpoints
at the same episode counts (1000, 2000): sims/move used, head-to-head strength
under MM's own search rule, and reports the wall-clock/episode already
recorded in adaptive_sweep.log.

    python adaptive_mm_report.py
"""
import json
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

THRESHOLDS = {'005': 0.005, '010': 0.010, '030': 0.030}
GENS = ('1000', '2000')
ROOT = 'othello_benchmark'
BENCH_MM = f'{ROOT}/bench_MM'
GAMES_PER_PAIR = 60
EVAL_SIMS = 64          # matches the trained cfg's own eval_sims


def sims_used_stats(tag, eps, gen):
    net = c4.load_benchmark_net(f'{ROOT}/bench_MMadapt{tag}', gen, ad.MM_SIG)
    positions = ad.sample_positions(net, c4.load_game('othello'), 50, sims=100)
    rng = np.random.default_rng(0)
    used = []
    for st in positions:
        bot = c4.C4MCTSBot(c4.load_game('othello'), net, 'cpu', 300,
                           batch_size=16, random_state=rng)
        _root, n = bot.mcts_search_checkpointed(st, check_every=100, kl_eps=eps)
        used.append(n)
    return np.array(used)


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)

    results = {}
    for tag, eps in THRESHOLDS.items():
        for gen in GENS:
            key = f'{tag}@{gen}'
            print(f'\n=== {key} (kl_eps={eps}) ===')
            t0 = time.time()
            used = sims_used_stats(tag, eps, gen)
            print(f'  sims/move: mean={used.mean():.1f} median={np.median(used):.0f} '
                 f'min={used.min()} max={used.max()}  '
                 f'(n={len(used)}, {time.time()-t0:.0f}s)')

            t0 = time.time()
            frac, lo, hi = ad.compare_to_bench_mm(
                f'{ROOT}/bench_MMadapt{tag}', gen, BENCH_MM, gen,
                sims=EVAL_SIMS, games_per_pair=GAMES_PER_PAIR)
            print(f'  vs bench_MM@{gen}: {frac:.1%} [{lo:.1%},{hi:.1%}] '
                 f'({time.time()-t0:.0f}s)')

            results[key] = dict(
                eps=eps, gen=gen,
                sims_mean=float(used.mean()), sims_median=float(np.median(used)),
                sims_min=int(used.min()), sims_max=int(used.max()),
                h2h_score=frac, h2h_lo=lo, h2h_hi=hi)

    with open('adaptive_mm_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('\nWrote adaptive_mm_results.json')


if __name__ == '__main__':
    main()
