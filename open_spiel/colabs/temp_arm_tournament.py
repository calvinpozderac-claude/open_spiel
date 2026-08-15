"""Rank the descent-temp arms under a FIXED inference configuration.

The in-training DEEP Elo cannot do this job. It is measured against the run's
own earlier checkpoints with eval_games_per_pair=4 over a 3-4 player pool, so
the matches sweep 4-0 and the rating collapses to a deterministic function of
a saturated record: the `coupled` and `dt4` arms printed byte-identical
ladders (1220/1069/939/915) despite a relative weight difference of 1.21, i.e.
entirely different networks. Anything read off that number is noise.

So the arms are played off directly, every entrant on the same fixed budget,
which is what separates the two effects:

  1. training effect     did sharpening self-play descent produce better
                         WEIGHTS?
  2. inference effect     is sharpened descent better AT PLAY TIME?  Already
                         answered by policy_screen: +204 Elo at 150 sims.

THE TOURNAMENT IS RUN TWICE, at inference descent temp 1.0 and 1e4, because
the interaction is the real question. If dtinf only wins when it is also
evaluated with sharp descent, then training and inference temperature have to
match and the weights are not independently better. If it wins under both, the
training change stands on its own. Reporting one setting could not tell those
apart.

    python temp_arm_tournament.py [--sims 64] [--games 40]
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SIMS = 64
GAMES_PER_PAIR = 40
OPENING_PLIES = 2
N_WORKERS = 5
ROOT = 'othello_benchmark'
LABEL = '2000'

# All arms are at 2000 episodes; bench_MM is the pre-existing line at the same
# episode count, so it is a fixed outside reference rather than a sibling.
ENTRANTS = {
    'coupled':  'bench_Tcoupled',
    'dt4':      'bench_Tdt4',
    'dt16':     'bench_Tdt16',
    'dtinf':    'bench_Tdtinf',
    'bench_MM': 'bench_MM',
    'random':   None,
}
INFERENCE_TEMPS = (1.0, 1e4)


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
    a, b, n_games, sims, inf_temp, seed = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps

    def load(label):
        d = ENTRANTS[label]
        if d is None:
            return None
        return c4.load_benchmark_net(os.path.join(ROOT, d), LABEL, ad.MM_SIG)

    net_a, net_b = load(a), load(b)
    pol = dict(mode='value', k=0.0, sel_temp=inf_temp)
    rng = np.random.default_rng(seed)

    def move(net, st):
        if net is None:                       # the random anchor
            leg = st.legal_actions()
            return int(leg[rng.integers(len(leg))])
        bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                           temp=inf_temp, random_state=rng)
        return ps.pick(c4, bot.mcts_search(st), pol)

    score_a = 0.0
    for pair in range(n_games // 2):
        st0 = game.new_initial_state()
        opening = []
        for _ in range(OPENING_PLIES):
            leg = st0.legal_actions()
            act = int(leg[rng.integers(len(leg))])
            st0.apply_action(act)
            opening.append(act)
        for a_seat in (0, 1):                 # same opening, both colours
            st = game.new_initial_state()
            for act in opening:
                st.apply_action(int(act))
            while not st.is_terminal():
                st.apply_action(move(net_a if st.current_player() == a_seat
                                     else net_b, st))
            r = st.returns()[a_seat]
            score_a += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return a, b, score_a, 2 * (n_games // 2)


def run(sims, games, inf_temp):
    import connect4_benchmark as bench
    names = [k for k in ENTRANTS
             if ENTRANTS[k] is None
             or os.path.exists(os.path.join(ROOT, ENTRANTS[k],
                                            f'bench_{LABEL}.pt'))]
    missing = set(ENTRANTS) - set(names)
    if missing:
        print(f'  skipping (no bench_{LABEL}.pt): {sorted(missing)}')
    idx = {k: i for i, k in enumerate(names)}
    pairs = list(itertools.combinations(names, 2))
    jobs = [(a, b, games, sims, inf_temp, 700 + 13 * i)
            for i, (a, b) in enumerate(pairs)]

    t0 = time.time()
    W = np.zeros((len(names), len(names)))
    done = 0
    with mp.Pool(N_WORKERS) as pool:
        for a, b, sc, n in pool.imap_unordered(_play_pair, jobs):
            W[idx[a], idx[b]] += sc
            W[idx[b], idx[a]] += n - sc
            done += 1
            print(f'    {done}/{len(pairs)}  {a} {sc:.1f}-{n-sc:.1f} {b}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)

    elo = bench.bradley_terry(W, anchor_idx=idx.get('random'))
    print(f'\n=== Bradley-Terry @ {LABEL} eps, fixed {sims} sims, '
          f'inference descent temp {inf_temp:g} (random=0) ===')
    for i in np.argsort(-elo):
        print(f'  {names[i]:<10} {elo[i]:+8.1f}   '
              f'W{W[i].sum():6.1f} L{W[:, i].sum():6.1f}')
    return names, W.tolist(), elo.tolist()


if __name__ == '__main__':
    sims = SIMS
    games = GAMES_PER_PAIR
    if '--sims' in sys.argv:
        sims = int(sys.argv[sys.argv.index('--sims') + 1])
    if '--games' in sys.argv:
        games = int(sys.argv[sys.argv.index('--games') + 1])
    out = {}
    for t in INFERENCE_TEMPS:
        names, W, elo = run(sims, games, t)
        out[f'temp_{t:g}'] = {'names': names, 'W': W, 'bt': elo}
    with open('temp_arm_tournament.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nwrote temp_arm_tournament.json')
