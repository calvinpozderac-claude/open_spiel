"""Round-robin among frozen checkpoints of one run, to see whether it kept
improving after the external ladder went blind.

The ladder tops out: every MCTS rung sits at 90-100% from about episode 4000,
and the only rung with any resolution left is the frozen champion, whose 30
games carry a ~9% standard error -- wide enough that 40% and 80% are not
distinguishable from a flat line. So the second half of the 20k run is
unmeasured rather than known to be flat.

Playing the checkpoints against EACH OTHER fixes the resolution problem. It is
a relative comparison, but every entrant is frozen, so unlike the run's own
Elo pool (4 games per pair, saturated to the point of printing byte-identical
ladders for different networks) it cannot drift, and Bradley-Terry pools every
pairing rather than leaning on one noisy opponent.

Every entrant plays at the same fixed budget under the same inference policy,
so this measures the WEIGHTS.

    python ckpt_tournament.py [--games 60] [--sims 64]
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SIMS = 64
GAMES_PER_PAIR = 60
OPENING_PLIES = 2
N_WORKERS = 5
INF_TEMP = 1e4          # the inference optimum, held fixed for every entrant

RUN_DIR = 'othello_benchmark/bench_OPT'
LABELS = ['2000', '4000', '6000', '8000', '12000', '16000', '20000']
# The round-3 winner, as a fixed outside reference point.
EXTRA = [('champion', 'othello_benchmark/bench_R2conf', '5000')]


def _entrants():
    out = [(f'ep{l}', RUN_DIR, l) for l in LABELS]
    out += [(n, d, l) for n, d, l in EXTRA]
    return [(n, d, l) for n, d, l in out
            if os.path.exists(os.path.join(d, f'bench_{l}.pt'))]


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
    (na, da, la), (nb, db, lb), n_games, sims, seed = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    net_a = c4.load_benchmark_net(da, la, ad.MM_SIG)
    net_b = c4.load_benchmark_net(db, lb, ad.MM_SIG)
    pol = dict(mode='value', k=0.0, sel_temp=INF_TEMP)
    rng = np.random.default_rng(seed)

    def move(net, st):
        bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                           temp=INF_TEMP, random_state=rng)
        return ps.pick(c4, bot.mcts_search(st), pol)

    score_a = 0.0
    for _pair in range(n_games // 2):
        st0 = game.new_initial_state()
        opening = []
        for _ in range(OPENING_PLIES):
            leg = st0.legal_actions()
            act = int(leg[rng.integers(len(leg))])
            st0.apply_action(act)
            opening.append(act)
        for a_seat in (0, 1):
            st = game.new_initial_state()
            for act in opening:
                st.apply_action(int(act))
            while not st.is_terminal():
                st.apply_action(move(net_a if st.current_player() == a_seat
                                     else net_b, st))
            r = st.returns()[a_seat]
            score_a += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return na, nb, score_a, 2 * (n_games // 2)


def main(games, sims):
    import connect4_benchmark as bench
    ents = _entrants()
    names = [e[0] for e in ents]
    idx = {n: i for i, n in enumerate(names)}
    pairs = list(itertools.combinations(ents, 2))
    print(f'{len(names)} entrants, {len(pairs)} pairs x {games} games '
          f'@ {sims} sims, inference temp {INF_TEMP:g}')

    cache_path = 'ckpt_tournament_cache.json'
    store = bench._load_rr_cache(cache_path)
    W = np.zeros((len(names), len(names)))
    todo = []
    for a, b in pairs:
        rec = store.get(bench._cache_key(f'{sims}t{INF_TEMP:g}', a[0], b[0]))
        if rec and int(rec[2]) >= games:
            W[idx[a[0]], idx[b[0]]] += float(rec[0])
            W[idx[b[0]], idx[a[0]]] += float(rec[1])
        else:
            todo.append((a, b))
    print(f'  {len(pairs)-len(todo)} pairs cached, playing {len(todo)}')

    jobs = [(a, b, games, sims, 8100 + 17 * i) for i, (a, b) in enumerate(todo)]
    t0, done = time.time(), 0
    if jobs:
        with mp.Pool(N_WORKERS) as pool:
            for na, nb, sc, n in pool.imap_unordered(_play_pair, jobs):
                W[idx[na], idx[nb]] += sc
                W[idx[nb], idx[na]] += n - sc
                store[bench._cache_key(f'{sims}t{INF_TEMP:g}', na, nb)] = \
                    [sc, n - sc, n]
                bench._save_rr_cache(cache_path, store)
                done += 1
                print(f'    {done}/{len(todo)}  {na} {sc:.1f}-{n-sc:.1f} {nb}'
                      f'  ({time.time()-t0:.0f}s)', flush=True)

    elo = bench.bradley_terry(W)
    order = np.argsort(-elo)
    print(f'\n=== Bradley-Terry, fixed {sims} sims, inference temp '
          f'{INF_TEMP:g} (zero-mean) ===')
    for i in order:
        print(f'  {names[i]:<10} {elo[i]:+8.1f}   '
              f'W{W[i].sum():6.1f} L{W[:, i].sum():6.1f}')
    with open('ckpt_tournament.json', 'w') as fh:
        json.dump({'names': names, 'W': W.tolist(), 'bt': elo.tolist(),
                   'sims': sims, 'games_per_pair': games}, fh, indent=2)
    print('wrote ckpt_tournament.json')


if __name__ == '__main__':
    g = GAMES_PER_PAIR
    s = SIMS
    if '--games' in sys.argv:
        g = int(sys.argv[sys.argv.index('--games') + 1])
    if '--sims' in sys.argv:
        s = int(sys.argv[sys.argv.index('--sims') + 1])
    main(g, s)
