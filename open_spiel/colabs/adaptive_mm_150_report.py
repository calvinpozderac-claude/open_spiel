"""Compares the compute-matched adaptive-MM run (bench_MMadapt150match,
check_every=75, kl_eps=0.1, calibrated to bench_MM's own ~150 sims/move
training average) against bench_MM at 1000/2000 episodes -- this is the
comparison that isolates the adaptive-stopping effect from the "just spent
more compute" confound found in the first sweep.

    python adaptive_mm_150_report.py
"""
import json
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

TAG = '150match'
CHECK_EVERY = 75
KL_EPS = 0.1
GENS = ('1000', '2000')
ROOT = 'othello_benchmark'
BENCH_MM = f'{ROOT}/bench_MM'
GAMES_PER_PAIR = 60
EVAL_SIMS = 64


def sims_used_stats(gen):
    net = c4.load_benchmark_net(f'{ROOT}/bench_MMadapt{TAG}', gen, ad.MM_SIG)
    positions = ad.sample_positions(net, c4.load_game('othello'), 50, sims=100)
    rng = np.random.default_rng(0)
    used = []
    for st in positions:
        bot = c4.C4MCTSBot(c4.load_game('othello'), net, 'cpu', 300,
                           batch_size=16, random_state=rng)
        _root, n = bot.mcts_search_checkpointed(
            st, check_every=CHECK_EVERY, kl_eps=KL_EPS)
        used.append(n)
    return np.array(used)


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)

    results = {}
    for gen in GENS:
        key = f'{TAG}@{gen}'
        print(f'\n=== {key} ===')
        t0 = time.time()
        used = sims_used_stats(gen)
        print(f'  sims/move: mean={used.mean():.1f} median={np.median(used):.0f} '
             f'min={used.min()} max={used.max()}  '
             f'(n={len(used)}, {time.time()-t0:.0f}s)')

        t0 = time.time()
        frac, lo, hi = ad.compare_to_bench_mm(
            f'{ROOT}/bench_MMadapt{TAG}', gen, BENCH_MM, gen,
            sims=EVAL_SIMS, games_per_pair=GAMES_PER_PAIR)
        print(f'  vs bench_MM@{gen}: {frac:.1%} [{lo:.1%},{hi:.1%}] '
             f'({time.time()-t0:.0f}s)')

        results[key] = dict(
            sims_mean=float(used.mean()), sims_median=float(np.median(used)),
            sims_min=int(used.min()), sims_max=int(used.max()),
            h2h_score=frac, h2h_lo=lo, h2h_hi=hi)

    with open('adaptive_mm_150_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('\nWrote adaptive_mm_150_results.json')


if __name__ == '__main__':
    main()
