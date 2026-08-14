"""Round robin with ONE fixed simulation budget for every entrant.

The round-2/3 tournaments let each arm evaluate under its own search rule --
play_game_arms says so explicitly: "Each side searches under its OWN recipe --
this compares full deployed policies (network + search regime)". That answers
"which deployed system is best" but confounds the two effects we actually want
separated:

  1. training effect    did adaptive sims produce a BETTER NETWORK?
  2. inference effect    is that arm's search rule better AT PLAY TIME?

Giving every entrant an identical ('fixed', N) budget removes (2) entirely, so
the ranking that comes out is about the weights alone. Run at more than one N
to check whether the ordering is budget-dependent, which is itself a finding:
a net trained under a ~150-sim regime having its ranking collapse at 400 would
say the arms are tuned to their budget rather than generally stronger.

Two implementation notes that matter for correctness:

  cache namespace   bench._cache_key is f'{sims}|{a}|{b}'. arms_round_robin
                    passes the literal string 'native' for every tournament it
                    has ever run, so those cached results are all under one
                    key regardless of the recipes used to produce them.
                    Passing the real budget here keeps fixed-sims results in
                    their own namespace and stops them colliding with -- or
                    silently reusing -- own-recipe games.
  parallel by pair  arms_round_robin plays every game in one process, which is
                    why the round-3 tournament took three hours. Pairs are
                    independent, so they distribute cleanly.

    python fixed_sims_tournament.py [--sims N] [--games K]
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SIMS = 150               # the budget every arm was calibrated to in training
GAMES_PER_PAIR = 80      # matches the round-3 tournament's k
OPENING_PLIES = 2
N_WORKERS = 5
ROOT = 'othello_benchmark'
LABEL = '5000'

# label -> checkpoint directory, or None for the uniform-random anchor.
ENTRANTS = {
    'conf':      'bench_R2conf',
    'predictor': 'bench_R2predictor',
    'conf_nl':   'bench_R3conf_nl',
    'confkl':    'bench_R3confkl',
    '150match':  'bench_MMadapt150match',
    '100avg':    'bench_MMadapt100avg',
    'bench_MM':  'bench_MM',
    'random':    None,
}


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _play_pair(args):
    a, b, n_games, sims, seed = args
    c4, game = _init()
    import adaptive_stop_utils as ad

    def load(label):
        d = ENTRANTS[label]
        if d is None:
            return None, ('random', None)
        net = c4.load_benchmark_net(os.path.join(ROOT, d), LABEL, ad.MM_SIG)
        return net, ('fixed', sims)

    net_a, rec_a = load(a)
    net_b, rec_b = load(b)
    rng = np.random.default_rng(seed)
    score_a = 0.0
    for g in range(n_games):
        a_first = (g % 2 == 0)
        if a_first:
            sc = ad.play_game_arms(game, net_a, rec_a, net_b, rec_b, rng,
                                   OPENING_PLIES)
        else:
            sc = ad.play_game_arms(game, net_b, rec_b, net_a, rec_a, rng,
                                   OPENING_PLIES)
        score_a += sc if a_first else (1.0 - sc)
    return a, b, score_a, n_games


def run(sims, games_per_pair, cache_path):
    import connect4_benchmark as bench
    names = [k for k in ENTRANTS
             if ENTRANTS[k] is None
             or os.path.exists(os.path.join(ROOT, ENTRANTS[k],
                                            f'bench_{LABEL}.pt'))]
    missing = set(ENTRANTS) - set(names)
    if missing:
        print(f'  skipping (no {LABEL} checkpoint): {sorted(missing)}')
    idx = {k: i for i, k in enumerate(names)}
    store = bench._load_rr_cache(cache_path)

    pairs = list(itertools.combinations(names, 2))
    todo = [(a, b) for a, b in pairs
            if int((store.get(bench._cache_key(sims, a, b)) or [0, 0, 0])[2])
            < games_per_pair]
    print(f'  {len(pairs)-len(todo)}/{len(pairs)} pairs cached at {sims} sims; '
          f'playing {len(todo)}')

    if todo:
        jobs = [(a, b, games_per_pair, sims, 4242 + 7 * i)
                for i, (a, b) in enumerate(todo)]
        t0 = time.time()
        done = 0
        with mp.Pool(N_WORKERS) as pool:
            for a, b, sc, n in pool.imap_unordered(_play_pair, jobs):
                store[bench._cache_key(sims, a, b)] = [sc, n - sc, n]
                bench._save_rr_cache(cache_path, store)
                done += 1
                print(f'    {done}/{len(todo)} pairs  ({time.time()-t0:.0f}s)  '
                      f'{a} {sc:.1f}-{n-sc:.1f} {b}', flush=True)

    W = np.zeros((len(names), len(names)))
    for a, b in pairs:
        rec = store.get(bench._cache_key(sims, a, b))
        if not rec:
            continue
        W[idx[a], idx[b]] += float(rec[0])
        W[idx[b], idx[a]] += float(rec[1])
    elo = bench.bradley_terry(W, anchor_idx=idx.get('random'))

    print(f'\n=== Bradley-Terry @ {LABEL} eps, FIXED {sims} sims '
          f'(anchored random=0) ===')
    for i in np.argsort(-elo):
        wins, losses = W[i].sum(), W[:, i].sum()
        print(f'  {names[i]:<12} {elo[i]:+8.1f}   W{wins:6.1f} L{losses:6.1f}')
    out = f'fixed_sims_tournament_{sims}.json'
    with open(out, 'w') as fh:
        json.dump({'names': names, 'W': W.tolist(), 'bt': elo.tolist(),
                   'sims': sims, 'games_per_pair': games_per_pair}, fh, indent=2)
    print(f'  wrote {out}')
    return names, W, elo


if __name__ == '__main__':
    sims = SIMS
    games = GAMES_PER_PAIR
    if '--sims' in sys.argv:
        sims = int(sys.argv[sys.argv.index('--sims') + 1])
    if '--games' in sys.argv:
        games = int(sys.argv[sys.argv.index('--games') + 1])
    run(sims, games, 'fixed_sims_tournament_cache.json')
