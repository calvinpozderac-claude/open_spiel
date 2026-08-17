"""An ABSOLUTE strength yardstick: our net against a fixed opponent ladder.

Every other eval in this project is RELATIVE -- Elo against the run's own
earlier checkpoints, or a round robin among sibling arms.
connect4_dirichlet_utils says why that is dangerous: it "cannot tell an
improving run from a collapsing one whose opponents collapsed with it". None
of the opponents here move, so a rising curve means the weights really improved.

THE LADDER, AND WHY IT LOOKS LIKE THIS

External rungs are open_spiel.python.algorithms.mcts.MCTSBot with a
RandomRolloutEvaluator -- plain UCT over random playouts, no neural network
involved, already vendored here so nothing is downloaded or executed from
outside. Skill is dialled purely by max_simulations.

Two things were measured before settling on this, both worth recording because
both are easy to assume otherwise:

  A positional evaluator is WEAKER than random rollouts here, per simulation.
  Swapping UCT's random playout for the classical Othello square-weight table
  plus mobility (and an endgame disc-count switch) scored 12.5% / 6.2% / 28.1%
  against random rollouts at equal budget across three scalings. It is ~10x
  cheaper per simulation, but not nearly enough to make up the strength gap.

  The external ladder cannot bracket us from above. bench_R2conf@5000 already
  beats MCTS-1600 93.3% of the time, and simulations get linearly more
  expensive (MCTS-6400 is ~215s/game), so no affordable rung is a ceiling.

So the ladder is deliberately mixed: the MCTS rungs are genuinely external and
resolve the early and middle of a from-scratch run, while a FROZEN checkpoint
of the current champion anchors the top. The champion is ours rather than
external, but it does not move, which is the property that matters for reading
a training curve.

Our side always plays under ONE fixed inference configuration, held constant
across every checkpoint, so a moving curve is the WEIGHTS moving and not the
search rule. Defaults come from the policy_screen sweep: deterministic descent
(sampling in descent measured ~-186 Elo) and the plain posterior-mean pick.

    python othello_ladder.py --calibrate
    python othello_ladder.py --ckpt DIR --label 3000
"""
import multiprocessing as mp
import sys
import time

import numpy as np

# Our side's inference config -- FIXED for every checkpoint ever scored here.
OUR_SIMS = 64
OUR_POLICY = dict(mode='value', k=0.0, sel_temp=1e4)

# (name, spec, games).  Games shrink as rungs get expensive: MCTS cost is
# linear in simulations, and MCTS-1600 alone is ~56s/game.
CHAMPION = ('othello_benchmark/bench_R2conf', '5000')

# Edax rungs are dialled by EPSILON (fraction of moves replaced by a uniform
# random legal move), not by level. Level is unusably coarse here: against
# bench_OPT@20000 at 64 sims, Edax level 1 scores 50% and level 2 scores 0%,
# with level 0 losing every game to level 1. Its pattern evaluation is simply
# far stronger than one 64-simulation search of ours, so depth cannot resolve
# the range we live in. Epsilon is continuous, monotone, reproducible, and
# ~11s per 30 games -- far cheaper than MCTS-1600 and, unlike it, still has
# headroom above the current model.
RUNGS = [
    ('MCTS-100',    ('mcts', 100),          30),   # kept for continuity
    ('edax-e45',    ('edax', 2, 0.45),      30),
    ('edax-e30',    ('edax', 2, 0.30),      30),
    ('edax-e20',    ('edax', 2, 0.20),      30),
    ('edax-e10',    ('edax', 2, 0.10),      30),
    ('edax-e05',    ('edax', 2, 0.05),      30),
    ('edax-full',   ('edax', 2, 0.0),       20),   # the ceiling; 0% for now
    ('champion',    ('net',) + CHAMPION,    30),
]
UCT_C = 2.0
OPENING_PLIES = 2
N_WORKERS = 2                      # leave cores for training when co-resident


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _our_move(c4, ps, game, net, st, rng):
    bot = c4.C4MCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=16,
                       temp=OUR_POLICY.get('sel_temp', 1.0), random_state=rng)
    return ps.pick(c4, bot.mcts_search(st), OUR_POLICY)


def _worker(args):
    ckpt_dir, label, name, spec, n_games, seed = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    net = c4.load_benchmark_net(ckpt_dir, label, ad.MM_SIG)

    opp_net = edax = None
    if spec[0] == 'net':
        opp_net = c4.load_benchmark_net(spec[1], spec[2], ad.MM_SIG)
    elif spec[0] == 'edax':
        # One persistent process for the whole job: loading eval.dat costs
        # about a second, which would dominate if respawned per game.
        import edax_bot as eb
        edax = eb.EdaxBot(game, level=spec[1], epsilon=spec[2], seed=seed)

    rng = np.random.default_rng(seed)
    pts = 0.0
    for pair in range(n_games // 2):
        st0 = game.new_initial_state()
        opening = []
        for _ in range(OPENING_PLIES):
            leg = st0.legal_actions()
            a = int(leg[rng.integers(len(leg))])
            st0.apply_action(a)
            opening.append(a)
        for seat in (0, 1):
            if spec[0] == 'mcts':
                from open_spiel.python.algorithms import mcts
                rs = np.random.RandomState(seed + 31 * pair + seat)
                opp = mcts.MCTSBot(game, UCT_C, spec[1],
                                   mcts.RandomRolloutEvaluator(
                                       n_rollouts=1, random_state=rs),
                                   random_state=rs)
            st = game.new_initial_state()
            for a in opening:
                st.apply_action(int(a))
            while not st.is_terminal():
                if st.current_player() == seat:
                    a = _our_move(c4, ps, game, net, st, rng)
                elif spec[0] == 'mcts':
                    a = opp.step(st)
                elif spec[0] == 'edax':
                    a = edax.step(st)
                else:
                    a = _our_move(c4, ps, game, opp_net, st, rng)
                st.apply_action(a)
            r = st.returns()[seat]
            pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    if edax is not None:
        edax.close()
    return name, pts, 2 * (n_games // 2)


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def evaluate(ckpt_dir, label, rungs=RUNGS, workers=N_WORKERS, log=print):
    """Score one checkpoint against every rung. Returns {rung name: frac}."""
    jobs = []
    for name, spec, games in rungs:
        per = max(2, games // workers)
        for w in range(workers):
            jobs.append((ckpt_dir, label, name, spec, per,
                         9000 + 137 * w + abs(hash(name)) % 1000))
    t0 = time.time()
    with mp.Pool(workers) as pool:
        parts = pool.map(_worker, jobs)

    agg = {}
    for name, pts, n in parts:
        p, m = agg.get(name, (0.0, 0))
        agg[name] = (p + pts, m + n)
    out = {}
    for name, _spec, _g in rungs:
        pts, n = agg[name]
        frac = pts / n
        lo, hi = wilson(frac, n)
        out[name] = frac
        log(f'    vs {name:<11} {frac:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}')
    log(f'    ({time.time()-t0:.0f}s)')
    return out


if __name__ == '__main__':
    if '--calibrate' in sys.argv:
        d, lab = CHAMPION
        print(f'calibrating ladder against {d}@{lab} (it IS the champion rung, '
              f'so that row should read ~50%)')
        print(f'our side: {OUR_SIMS} sims, policy={OUR_POLICY}')
        evaluate(d, lab, workers=5)
    else:
        ck = sys.argv[sys.argv.index('--ckpt') + 1]
        lab = sys.argv[sys.argv.index('--label') + 1]
        print(f'{ck}@{lab}')
        evaluate(ck, lab)
