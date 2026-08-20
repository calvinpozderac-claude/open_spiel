"""OPT@20000 vs MM@20000 across opponent strength AND our own search budget.

These two are the matched-episode comparison: same 20000 episodes, differing
only in conf allocation + descent_temp=16. At 64 sims and 200 games they
looked like 37.0/13.5/7.8 against 27.0/11.2/6.0 -- OPT ahead on every rung but
no single gap significant.

Two axes are swept because both can invert a ranking, as this project has
repeatedly found:

  opponent   L1, 0.5L2 (half L1 half L2), L2, 0.5L3, L3.  Model rankings here
             have already been shown to flip with opponent strength -- dt16
             leads at L1 and is last at L2 -- so one rung is not a verdict.
  our sims   250 / 500 / 1000, versus the 64 everything else used.  A model
             that is better only at low budget is a different claim from one
             that is better in general, and search budget is the axis most
             likely to interact with a training change that altered how the
             tree is built.

Our side plays at temp=64 throughout: the highest temperature that still
yields fully distinct games, so Edax being deterministic does not collapse the
sample.

    python opt_vs_mm_grid.py [--games 100]
"""
import itertools
import json
import multiprocessing as mp
import sys
import time

import numpy as np

MODELS = [
    ('OPT@20000', 'othello_benchmark/bench_OPT',  '20000'),
    ('MM@20000',  'othello_benchmark/bench_MM',   '20000'),
]
# (name, lo, hi, p_lo) -- p_lo=1.0 with lo==hi is a plain fixed level.
OPPONENTS = [
    ('L1',    1, 1, 1.0),
    ('0.5L2', 1, 2, 0.5),
    ('L2',    2, 2, 1.0),
    ('0.5L3', 2, 3, 0.5),
    ('L3',    3, 3, 1.0),
]
SIMS = [250, 500, 1000]
GAMES = 100
OUR_TEMP = 64.0
N_WORKERS = 5
RESULTS = 'opt_vs_mm_grid.json'


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _worker(args):
    mname, ckpt, label, oname, lo, hi, p_lo, sims, n_games, seed, seat0 = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    import edax_bot as eb

    net = c4.load_benchmark_net(ckpt, label, ad.MM_SIG)
    opp = (eb.EdaxBot(game, level=lo) if lo == hi
           else eb.EdaxMixBot(game, p_lo=p_lo, lo=lo, hi=hi, seed=seed))
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    pts = 0.0
    for gi in range(n_games):
        seat = (seat0 + gi) % 2
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.current_player() == seat:
                b = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                                 temp=OUR_TEMP, random_state=rng)
                a = ps.pick(c4, b.mcts_search(st), pol)
            else:
                a = opp.step(st)
            st.apply_action(a)
        r = st.returns()[seat]
        pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    opp.close()
    return (mname, oname, sims), pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main(games):
    out = {}
    per = max(2, games // N_WORKERS)
    for sims in SIMS:
        print(f'\n===== our search budget: {sims} sims '
              f'({games} games/cell) =====', flush=True)
        print(f'{"opponent":<9}' + ''.join(f'{m[0]:>22}' for m in MODELS),
              flush=True)
        for oname, lo, hi, p_lo in OPPONENTS:
            jobs = []
            for mname, ck, lab in MODELS:
                for w in range(N_WORKERS):
                    jobs.append((mname, ck, lab, oname, lo, hi, p_lo, sims,
                                 per, 5200 + 97 * w, w % 2))
            t0 = time.time()
            with mp.Pool(N_WORKERS) as pool:
                parts = pool.map(_worker, jobs)
            agg = {}
            for key, pts, n in parts:
                p, m = agg.get(key, (0.0, 0))
                agg[key] = (p + pts, m + n)
            cells = []
            for mname, _c, _l in MODELS:
                pts, n = agg[(mname, oname, sims)]
                f = pts / n
                loci, hici = wilson(f, n)
                out[f'{mname}|{oname}|{sims}'] = dict(score=f, n=n,
                                                      lo=loci, hi=hici)
                cells.append(f'{f:>10.1%} [{loci:.0%},{hici:.0%}]')
            print(f'{oname:<9}' + ''.join(cells)
                  + f'   ({time.time()-t0:.0f}s)', flush=True)
            with open(RESULTS, 'w') as fh:
                json.dump(out, fh, indent=2, sort_keys=True)
    print('\nGRID DONE', flush=True)


if __name__ == '__main__':
    g = GAMES
    if '--games' in sys.argv:
        g = int(sys.argv[sys.argv.index('--games') + 1])
    main(g)
