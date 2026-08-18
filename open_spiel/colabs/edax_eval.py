"""Score models against Edax: 50 games, alternating colours, standard start.

No imposed openings and no random moves by either side. Both engines just play
the game.

WHY OUR SIDE PLAYS AT temp=64

Edax is deterministic, so all game-to-game variety has to come from our model,
and at the deterministic-descent setting (temp=1e4) it has almost none:
measured against Edax L1 from the standard start, 6 games as black produced 1
distinct move sequence, and over 20 alternating-colour games temp=1e4 gave
only 13 distinct games (6 of 10 as black). Fifty games at that setting is a
handful of games replayed, and the score can only land on a few values.

Variety by temperature, 20 alternating-colour games vs Edax L1:

    temp    16    64   256  1024  4096   1e4
  distinct  20    20    18    16     9    13

temp=64 is the HIGHEST setting that still produces fully distinct games, so it
buys the stochasticity at the smallest strength cost: in the policy_screen
sweep temp=64 scored 68.2% against the temp=1 baseline versus 74.5% for
deterministic, about 53 Elo. That is a real handicap and these numbers are
NOT comparable to the deterministic-inference figures elsewhere in the
project -- but it is the same handicap for every model scored here, so the
ranking between them is fair.

Two phases:

  vs L1     every model against Edax level 1, the rung it is closest to
  mixture   for ranking: the opponent plays level `lo` with probability p_lo
            and level `hi` otherwise, so strength is dialled between two
            competent engines rather than by injecting blunders

    python edax_eval.py vs-l1
    python edax_eval.py mixture --p 0.75 0.5 0.25 0.0
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np

GAMES = 50
OUR_SIMS = 64
OUR_TEMP = 64.0            # see module docstring
# TUNED, from az_policy_screen (400 games/cell, root noise on both sides so
# every cell is a real sample rather than one deterministic game replayed).
# c_puct is monotone increasing over 0.5/1.0/1.5/2.5/6.0 -> 41.4/46.6/50/61.6/
# 68.5%, so the trained 1.5 was far too low for PLAY.  Visit sampling, which an
# earlier version of this eval used to manufacture variety, costs ~90% against
# plain argmax and is dropped; root Dirichlet noise supplies variety instead at
# a fraction of the cost.
AZ_CPUCT = 6.0
AZ_NOISE = 0.15
N_WORKERS = 5
RESULTS = 'edax_eval.json'

# The best of each line, plus the reference points worth placing on the scale.
# (name, dir, label, kind).  kind 'az' drives the AlphaZero engine instead of
# the Dirichlet/Thompson one: PUCT descent, move by visit count.  AlphaZero is
# deterministic (no root noise at eval, argmax visits), so like our side it
# needs a stochastic setting to produce distinct games -- its dial is the
# visit-sampling temperature, and AZ_TEMP is the smallest value giving full
# variety (10/10 distinct games, versus 2/10 at argmax).
MODELS = [
    ('AZ@40000',    'othello_benchmark/bench_AZ',          '40000', 'az'),
    ('AZ@20000',    'othello_benchmark/bench_AZ',          '20000', 'az'),
    ('AZ@10000',    'othello_benchmark/bench_AZ',          '10000', 'az'),
    ('OPT@20000',   'othello_benchmark/bench_OPT',         '20000'),
    ('OPT@16000',   'othello_benchmark/bench_OPT',         '16000'),
    ('OPT@12000',   'othello_benchmark/bench_OPT',         '12000'),
    ('OPT@8000',    'othello_benchmark/bench_OPT',          '8000'),
    ('OPT@4000',    'othello_benchmark/bench_OPT',          '4000'),
    ('OPT@2000',    'othello_benchmark/bench_OPT',          '2000'),
    ('R2conf@5000', 'othello_benchmark/bench_R2conf',       '5000'),
    ('R2pred@5000', 'othello_benchmark/bench_R2predictor',  '5000'),
    ('dt16@2000',   'othello_benchmark/bench_Tdt16',        '2000'),
    ('coupled@2000', 'othello_benchmark/bench_Tcoupled',    '2000'),
    ('benchMM@2000', 'othello_benchmark/bench_MM',          '2000'),
]


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
    name, ckpt_dir, label, kind, opp_spec, n_games, seed, first_seat = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    import edax_bot as eb

    if kind == 'az':
        import connect4_alphazero_utils as az
        az.set_search(c_puct=AZ_CPUCT, virtual_loss=1.0)
        net = az.load_benchmark_net(ckpt_dir, label, ad.MM_SIG)
    else:
        net = c4.load_benchmark_net(ckpt_dir, label, ad.MM_SIG)
    if opp_spec[0] == 'L1':
        opp = eb.EdaxBot(game, level=1)
    else:
        opp = eb.EdaxMixBot(game, p_lo=opp_spec[1], lo=1, hi=2, seed=seed)
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    pts = 0.0
    for gi in range(n_games):
        seat = (first_seat + gi) % 2          # colours alternate
        st = game.new_initial_state()         # standard start, no opening
        while not st.is_terminal():
            if st.current_player() == seat and kind == 'az':
                b = az.AZMCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=8,
                                 random_state=rng, noise_frac=AZ_NOISE)
                a = az.root_pick(b.mcts_search(st), rng, sample=False)
            elif st.current_player() == seat:
                b = c4.C4MCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=16,
                                 temp=OUR_TEMP, random_state=rng)
                a = ps.pick(c4, b.mcts_search(st), pol)
            else:
                a = opp.step(st)
            st.apply_action(a)
        r = st.returns()[seat]
        pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    opp.close()
    return name, pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def run(opp_spec, games, models=MODELS, log=print):
    jobs = []
    per = max(2, games // N_WORKERS)
    for m in models:
        name, d, lab = m[0], m[1], m[2]
        kind = m[3] if len(m) > 3 else 'thompson'
        for w in range(N_WORKERS):
            # first_seat alternates per worker so the colour split stays even
            jobs.append((name, d, lab, kind, opp_spec, per,
                         6100 + 53 * w, w % 2))
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    agg = {}
    for name, pts, n in parts:
        p, m = agg.get(name, (0.0, 0))
        agg[name] = (p + pts, m + n)
    out = {}
    for name in [m[0] for m in models]:
        pts, n = agg[name]
        frac = pts / n
        lo, hi = wilson(frac, n)
        out[name] = dict(score=frac, n=n, lo=lo, hi=hi)
        log(f'  {name:<14} {frac:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}')
    log(f'  ({time.time()-t0:.0f}s)')
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'vs-l1'
    games = GAMES
    if '--games' in sys.argv:
        games = int(sys.argv[sys.argv.index('--games') + 1])
    models = MODELS
    if '--top' in sys.argv:
        # The candidates worth resolving properly, plus a weak control so the
        # scale stays anchored.
        keep = {'AZ@40000', 'AZ@20000', 'AZ@10000', 'dt16@2000',
                'OPT@20000', 'R2conf@5000'}
        models = [m for m in MODELS if m[0] in keep]
    all_out = {}
    if mode == 'vs-l1':
        print(f'vs Edax L1 -- {games} games, alternating colours, standard '
              f'start, our side {OUR_SIMS} sims @ temp {OUR_TEMP:g}')
        all_out['L1'] = run(('L1',), games, models)
    else:
        ps_ = [float(x) for x in
               sys.argv[sys.argv.index('--p') + 1:]] if '--p' in sys.argv \
            else [0.75, 0.5, 0.25, 0.0]
        for p in ps_:
            print(f'\nvs Edax mixture p(L1)={p:g}, else L2 -- {games} games')
            all_out[f'mix_p{p:g}'] = run(('mix', p), games, models)
    prev = {}
    try:
        with open(RESULTS) as fh:
            prev = json.load(fh)
    except FileNotFoundError:
        pass
    prev.update(all_out)
    with open(RESULTS, 'w') as fh:
        json.dump(prev, fh, indent=2, sort_keys=True)
    print(f'\nwrote {RESULTS}')
