"""Same AVGCONC net vs MM_long, varying ONLY the AVGCONC side's SEARCH rule.

The question: is play strength set by how the net was TRAINED, or by how it
SEARCHES at deploy time?  The AVGCONC net was trained (and first tested) with
avgconc search and lost to MM_long 22-78.  DECOUPLED, whose head is likewise
mixture-family but which SEARCHES with additive_mle, drew MM_long 47-53.  If
the difference is the search rule, then searching THIS net with additive_mle
should recover strength -- with no retraining.

MM_long is unchanged (additive_mle) in every match.  Only the aggregation the
AVGCONC net uses to collapse its search tree varies.  100 games per variation,
1000 sims/side, temp 64 -- identical to the earlier head-to-heads.

    python ht_avg_searchsweep.py
"""
import multiprocessing as mp
import re
import time

import numpy as np

SIMS = 1000
GAMES = 100
OUR_TEMP = 64.0
N_WORKERS = 3
LOG = 'ht_avg_searchsweep.log'

MM_DIR, MM_LABEL = 'othello_benchmark/bench_MM_long_snaps', '50000'
AV_DIR, AV_LABEL = 'othello_benchmark/bench_AVGCONC', '50000'
# The AVGCONC side's search rule, swept.  MM stays additive_mle throughout.
VARIATIONS = ['additive_mle', 'additive', 'sum', 'mixture', 'avgconc']


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _sig(c4, d, label):
    import torch
    sd = torch.load(f'{d}/bench_{label}.pt', map_location='cpu', weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def _worker(args):
    av_agg, n_games, seed, mm_first = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    net_mm = c4.load_benchmark_net(MM_DIR, MM_LABEL, ad.MM_SIG)
    net_av = c4.load_benchmark_net(AV_DIR, AV_LABEL, _sig(c4, AV_DIR, AV_LABEL))
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    mm_pts = 0.0
    for gi in range(n_games):
        mm_seat = (0 if mm_first else 1)
        if gi % 2 == 1:
            mm_seat = 1 - mm_seat
        st = game.new_initial_state()
        while not st.is_terminal():
            is_mm = (st.current_player() == mm_seat)
            net, agg = (net_mm, 'additive_mle') if is_mm else (net_av, av_agg)
            c4.set_search(search_agg=agg, target_agg=agg)
            bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                               temp=OUR_TEMP, random_state=rng)
            a = ps.pick(c4, bot.mcts_search(st), pol)
            st.apply_action(a)
        r = st.returns()[mm_seat]
        mm_pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return mm_pts, n_games


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def main():
    _log(f'=== AVGCONC net vs MM_long, sweeping AVGCONC search rule '
         f'({GAMES} games, {SIMS} sims/side) ===')
    _log(f'{"AVGCONC search":>16}{"AVGCONC score":>16}{"95% CI":>18}')
    results = {}
    for agg in VARIATIONS:
        per = GAMES // N_WORKERS
        rem = GAMES - per * N_WORKERS
        jobs = [(agg, per + (1 if w < rem else 0), 5100 + 83 * w, w % 2 == 0)
                for w in range(N_WORKERS)]
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            parts = pool.map(_worker, jobs)
        mm = sum(p for p, _ in parts)
        n = sum(g for _, g in parts)
        av = n - mm
        frac = av / n
        lo, hi = wilson(frac, n)
        results[agg] = frac
        _log(f'{agg:>16}{f"{av:.1f}/{n} = {frac:.1%}":>16}'
             f'{f"[{lo:.1%}, {hi:.1%}]":>18}   ({time.time()-t0:.0f}s)')
    best = max(results, key=results.get)
    _log(f'=== best AVGCONC search vs MM_long: {best} at {results[best]:.1%} ===')


if __name__ == '__main__':
    main()
