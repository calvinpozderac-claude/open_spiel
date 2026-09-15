"""Round-robin tournament to rank models, with Elo fitted from pairwise results.

Each participant plays with ITS OWN trained search rule (set per move by the
mover, like head_to_head.py), greedy value pick at eval, colors alternated.
Every unordered pair plays GAMES games; a Bradley-Terry/Elo MLE is fit to the
full score matrix and printed as a ranked table.

Participants (name, dir, label, search_agg):
  the allmse 2x2 factorial (decoupled-distillation loss) + current references,
  all at ep5000 for a matched ranking.

    python tournament.py [games] [sims]
    python tournament.py report            # re-print from tournament.json
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SIG = (32, 10, 8)
OUR_TEMP = 64.0
N_WORKERS = 4
OUT = 'tournament.json'

# (name, dir, label, search_agg) -- each net deploys the search it trained with.
PARTICIPANTS = [
    ('allmse',        'othello_benchmark/bench_ALLMSE',                '5000', 'pathsum'),
    ('allmse_admle',  'othello_benchmark/bench_ALLMSE_ADMLE_AVGCONC',  '5000', 'additive_mle'),
    ('allmse_pathav', 'othello_benchmark/bench_ALLMSE_PATHSUM_PATHAVG','5000', 'pathsum'),
    ('allmse_ad_pav', 'othello_benchmark/bench_ALLMSE_ADMLE_PATHAVG',  '5000', 'additive_mle'),
    ('ZGROUND',       'othello_benchmark/bench_ZGROUND',               '5000', 'pathsum'),
    ('HYBRID',        'othello_benchmark/bench_HYBRID',                '5000', 'additive_mle'),
    ('DISTILL',       'othello_benchmark/bench_DISTILL',               '5000', 'pathsum'),
]
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != 'report' else 100
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 400


def _worker(args):
    (an, ad, al, aagg), (bn, bd, bl, bagg), n_games, seed, a_first = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(selection='dirichlet', virtual_loss=1.0)
    net_a = c4.load_benchmark_net(ad, al, SIG)
    net_b = c4.load_benchmark_net(bd, bl, SIG)
    rng = np.random.default_rng(seed)
    a_pts = 0.0
    for gi in range(n_games):
        a_seat = (0 if a_first else 1)
        if gi % 2 == 1:
            a_seat = 1 - a_seat
        st = game.new_initial_state()
        while not st.is_terminal():
            is_a = (st.current_player() == a_seat)
            net, agg = (net_a, aagg) if is_a else (net_b, bagg)
            c4.set_search(search_agg=agg, target_agg=agg)
            bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                               temp=OUR_TEMP, random_state=rng)
            root = bot.mcts_search(st)
            mv = c4.root_pick(root, rng, thompson=False, temp=OUR_TEMP)
            st.apply_action(int(mv))
        r = st.returns()[a_seat]
        a_pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return (an, bn), a_pts, n_games


def fit_elo(names, wins, games, anchor=1500.0, iters=10000, lr=1.0):
    """Bradley-Terry MLE -> Elo. wins[i][j] = points i scored vs j (of games)."""
    n = len(names)
    r = np.full(n, anchor, float)
    S = 400.0 / np.log(10.0)
    for _ in range(iters):
        grad = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j or games[i][j] == 0:
                    continue
                p = 1.0 / (1.0 + np.exp((r[j] - r[i]) / S))
                grad[i] += (wins[i][j] - games[i][j] * p)
        r += lr * grad / max(games.sum(), 1)
        r -= r.mean() - anchor
    return r


def main():
    parts = PARTICIPANTS
    idx = {p[0]: k for k, p in enumerate(parts)}
    n = len(parts)
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh).get('pairs', {})

    jobs = []
    for a, b in itertools.combinations(parts, 2):
        key = f'{a[0]}|{b[0]}'
        if key in done:
            continue
        per = GAMES // N_WORKERS
        rem = GAMES - per * N_WORKERS
        for w in range(N_WORKERS):
            g = per + (1 if w < rem else 0)
            if g:
                jobs.append((a, b, g, 7000 + 131 * w, w % 2 == 0))
    print(f'[tourn] {n} nets, {len(list(itertools.combinations(parts,2)))} pairs, '
          f'{GAMES} games/pair @ {SIMS} sims, {len(jobs)} shard-jobs', flush=True)

    agg = {}   # (a,b) -> [a_pts, games]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for (an, bn), a_pts, g in pool.imap_unordered(_worker, jobs):
            e = agg.setdefault((an, bn), [0.0, 0])
            e[0] += a_pts
            e[1] += g
            print(f'[tourn] {an} vs {bn}: +{a_pts:.1f}/{g} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    for (an, bn), (a_pts, g) in agg.items():
        done[f'{an}|{bn}'] = [a_pts, g]
    with open(OUT, 'w') as fh:
        json.dump({'pairs': done, 'sims': SIMS}, fh)

    for key, (a_pts, g) in done.items():
        an, bn = key.split('|')
        i, j = idx[an], idx[bn]
        wins[i][j] += a_pts
        games[i][j] += g
        wins[j][i] += g - a_pts
        games[j][i] += g
    report(parts, wins, games)


def report(parts=None, wins=None, games=None):
    if parts is None:
        parts = PARTICIPANTS
        idx = {p[0]: k for k, p in enumerate(parts)}
        n = len(parts)
        wins = np.zeros((n, n)); games = np.zeros((n, n))
        with open(OUT) as fh:
            done = json.load(fh)['pairs']
        for key, (a_pts, g) in done.items():
            an, bn = key.split('|')
            i, j = idx[an], idx[bn]
            wins[i][j] += a_pts; games[i][j] += g
            wins[j][i] += g - a_pts; games[j][i] += g
    names = [p[0] for p in parts]
    elo = fit_elo(names, wins, games)
    order = np.argsort(-elo)
    tot_pts = wins.sum(1)
    tot_g = games.sum(1)
    print(f'\n{"rank":<5}{"model":<15}{"Elo":>6}{"score":>9}{"win%":>7}   search')
    for rank, i in enumerate(order, 1):
        wp = tot_pts[i] / max(tot_g[i], 1)
        print(f'{rank:<5}{names[i]:<15}{elo[i]:>6.0f}'
              f'{tot_pts[i]:>6.0f}/{tot_g[i]:<3.0f}{wp:>6.1%}   {parts[i][3]}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
