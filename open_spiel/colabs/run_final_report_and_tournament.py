"""Final phase of the adaptive-stopping experiment:

  1. Per-arm report: for each of the 9 experimental arms, at each of its
     1000/2000/3000/4000/5000 checkpoints, sims/move distribution + head-to-
     head vs bench_MM at the matching checkpoint (fixed eval_sims=64 both
     sides, same methodology as the earlier partial reports, for continuity).

  2. Round-robin tournament among the 9 arms' FINAL (5000-episode)
     checkpoints, plus bench_MM@5000 and a random mover as anchors -- each
     arm plays under its OWN native search recipe (not a shared neutral
     sims value), since the question here is which deployed policy
     (network + search regime) is actually strongest. Bradley-Terry scores
     computed from the result matrix.

     k (games per pair) is chosen adaptively to target ~4 hours total: play
     1 game per pair first (that game counts, it isn't thrown away), time
     it, then use the measured per-game cost to compute how many more games
     per pair the remaining budget affords.

    python run_final_report_and_tournament.py
"""
import json
import os
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

ROOT = 'othello_benchmark'
BENCH_MM = f'{ROOT}/bench_MM'
CHECKPOINTS = ('1000', '2000', '3000', '4000', '5000')
GAMES_PER_PAIR_REPORT = 60
EVAL_SIMS = 64
TOURNAMENT_BUDGET_S = 4 * 3600
TOURNAMENT_CACHE = 'arms_tournament_cache.json'

# label -> (checkpoint dir suffix, recipe)
ARMS = {
    '005':        ('bench_MMadapt005',      ('adaptive', 100, 0.005, 300)),
    '010':        ('bench_MMadapt010',      ('adaptive', 100, 0.010, 300)),
    '030':        ('bench_MMadapt030',      ('adaptive', 100, 0.030, 300)),
    '150match':   ('bench_MMadapt150match', ('adaptive', 75,  0.1,   300)),
    '100avg':     ('bench_MMadapt100avg',   ('adaptive', 50,  0.08,  300)),
    'fixed500':   ('bench_MMfixed500',      ('fixed', 500)),
    'mix300_500': ('bench_MMmix300_500',    ('mix', 300, 500, 0.75)),
    'adapt350':   ('bench_MMadapt350',      ('adaptive', 100, 0.005, 1000)),
    'adapt500':   ('bench_MMadapt500',      ('adaptive', 100, 0.001, 1000)),
}
BENCH_MM_RECIPE = ('mix', 100, 300, 0.75)   # bench_MM's own training recipe

LOG_PATH = os.path.join(os.path.dirname(__file__), 'final_report_tournament.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def phase1_per_arm_report():
    results = {}
    out_path = 'final_per_arm_results.json'
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    for tag, (ckpt_dir, recipe) in ARMS.items():
        for gen in CHECKPOINTS:
            key = f'{tag}@{gen}'
            if key in results:
                _log(f'{key}: already done, skipping')
                continue
            ckpt_path = os.path.join(ROOT, ckpt_dir)
            if not os.path.exists(os.path.join(ckpt_path, f'bench_{gen}.pt')):
                _log(f'{key}: checkpoint missing, skipping')
                continue
            _log(f'=== {key} ===')
            c4.set_game(c4.load_game('othello'))
            net = c4.load_benchmark_net(ckpt_path, gen, ad.MM_SIG)
            game = c4.load_game('othello')
            positions = ad.sample_positions(net, game, 50, sims=100)
            t0 = time.time()
            used = ad.recipe_sims_used(game, net, recipe, positions)
            _log(f'  sims/move: mean={used.mean():.1f} median={np.median(used):.0f} '
                f'min={used.min()} max={used.max()}  '
                f'hist(<=100/200/300/400/500/600+)='
                f'{[int(((used>lo)&(used<=hi)).sum()) for lo,hi in [(-1,100),(100,200),(200,300),(300,400),(400,500),(500,10000)]]}'
                f'  ({time.time()-t0:.0f}s)')

            t0 = time.time()
            frac, lo, hi = ad.compare_to_bench_mm(
                ckpt_path, gen, BENCH_MM, gen,
                sims=EVAL_SIMS, games_per_pair=GAMES_PER_PAIR_REPORT)
            _log(f'  vs bench_MM@{gen}: {frac:.1%} [{lo:.1%},{hi:.1%}] '
                f'({time.time()-t0:.0f}s)')

            results[key] = dict(
                sims_mean=float(used.mean()), sims_median=float(np.median(used)),
                sims_min=int(used.min()), sims_max=int(used.max()),
                sims_hist=[int(x) for x in
                          [((used>lo)&(used<=hi)).sum() for lo,hi in
                           [(-1,100),(100,200),(200,300),(300,400),(400,500),(500,10000)]]],
                h2h_score=frac, h2h_lo=lo, h2h_hi=hi)
            with open(out_path, 'w') as fh:
                json.dump(results, fh, indent=2)
    _log('phase 1 (per-arm report) complete')
    return results


def phase2_tournament():
    c4.set_game(c4.load_game('othello'))
    entrants = {}
    for tag, (ckpt_dir, recipe) in ARMS.items():
        net = c4.load_benchmark_net(os.path.join(ROOT, ckpt_dir), '5000', ad.MM_SIG)
        entrants[tag] = (net, recipe)
    entrants['bench_MM'] = (c4.load_benchmark_net(BENCH_MM, '5000', ad.MM_SIG),
                            BENCH_MM_RECIPE)
    entrants['random'] = (None, ('random', None))
    _log(f'tournament entrants: {list(entrants)}')

    n_pairs = len(entrants) * (len(entrants) - 1) // 2
    _log(f'{n_pairs} pairs -- playing 1 game/pair first to measure real cost')
    t0 = time.time()
    _names, _W, _elo, n_new = ad.arms_round_robin(
        entrants, games_per_pair=1, cache=TOURNAMENT_CACHE, log=_log)
    elapsed = time.time() - t0
    if n_new == 0:
        # Every pair already had >=1 game cached (e.g. a prior interrupted
        # run got this far already) -- fall back to a fixed, conservative
        # per-game estimate rather than dividing by zero.
        sec_per_game = 20.0
        _log(f'  (all pairs already had a game cached; assuming '
            f'{sec_per_game:.0f}s/game for the k calculation)')
    else:
        sec_per_game = elapsed / n_new
        _log(f'  {n_new} new games took {elapsed:.0f}s '
            f'({sec_per_game:.1f}s/game avg)')
    remaining = max(TOURNAMENT_BUDGET_S - elapsed, 0)
    extra_games = int(remaining // (n_pairs * sec_per_game))
    k = 1 + extra_games
    _log(f'budget {TOURNAMENT_BUDGET_S}s, remaining {remaining:.0f}s -> '
        f'k={k} games/pair total')

    names, W, elo, _n_new = ad.arms_round_robin(entrants, games_per_pair=k,
                                                cache=TOURNAMENT_CACHE, log=_log)
    _log('\n=== Bradley-Terry ratings (anchored to random=0) ===')
    order = np.argsort(-elo)
    for i in order:
        _log(f'  {names[i]:<12} {elo[i]:+7.1f}')
    with open('arms_tournament_results.json', 'w') as fh:
        json.dump({'names': names, 'W': W.tolist(), 'bt': elo.tolist(),
                  'games_per_pair': k}, fh, indent=2)
    _log('phase 2 (tournament) complete')


if __name__ == '__main__':
    _log('=== phase 1: per-arm report ===')
    phase1_per_arm_report()
    _log('=== phase 2: round-robin tournament ===')
    phase2_tournament()
    _log('ALL DONE')
