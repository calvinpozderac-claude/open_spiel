"""A/B inference-time move policies by playing one network against ITSELF.

A training arm costs hours; a change to how a searched tree is CONSUMED costs
minutes to evaluate, because the same weights can play both sides. Same net,
same budget, same opening, paired colours -- so the only difference is the
policy and any score away from 50% is that policy's doing.

Every tournament here uses a single FIXED simulation budget for both sides, so
what is measured is purely the inference rule, with no training or
search-budget effect mixed in.

A policy is a dict:
  mode      'value'        argmax of the per-action posterior mean (incumbent)
            'visits'       argmax visit count, as AlphaZero picks
            'visits_value' argmax visits, ties broken by value
            'evidence'     value from backed-up evidence ONLY, dropping the
                           network prior alpha_p that alpha_sel folds in
  k         risk coefficient on sd[v]: argmax (E[v] + k*sd[v]).  k<0 prefers
            sure moves, k>0 prefers uncertain ones, k=0 is the incumbent.
            Only meaningful for mode='value'.
  sel_temp  sharpens the in-tree Thompson draw (temp>1 concentrates toward the
            mean, so descent is less random)

Presets record what was actually run, so a batch stays reproducible:
    python policy_screen.py batch3 --games 600
"""
import multiprocessing as mp
import sys
import time

import numpy as np

SIMS = 64
OPENING_PLIES = 2
MAX_PLIES = 128
N_WORKERS = 5
NET = ('othello_benchmark/bench_R2conf', '5000')
BASELINE = dict(mode='value', k=0.0, sel_temp=1.0)

PRESETS = {
    # Batch 1: how the value is scored, and how random descent is.
    'batch1': [('risk k=+1.0', dict(k=+1.0)), ('risk k=+0.5', dict(k=+0.5)),
               ('risk k=-0.5', dict(k=-0.5)), ('risk k=-1.0', dict(k=-1.0)),
               ('seltemp 2', dict(sel_temp=2.0)),
               ('seltemp 4', dict(sel_temp=4.0)),
               ('seltemp 0.5', dict(sel_temp=0.5)),
               ('null', dict())],
    # Batch 2: which statistic the move is read off at all.
    'batch2': [('visits', dict(mode='visits')),
               ('visits+value', dict(mode='visits_value')),
               ('evidence only', dict(mode='evidence')),
               ('null', dict())],
    # Batch 3: batch 1 found both axes monotone and still improving at the
    # edge of the sweep, so push both further and test whether they compose.
    'batch3': [('risk k=-1.5', dict(k=-1.5)), ('risk k=-2.0', dict(k=-2.0)),
               ('risk k=-3.0', dict(k=-3.0)),
               ('seltemp 8', dict(sel_temp=8.0)),
               ('seltemp 16', dict(sel_temp=16.0)),
               ('k=-1 temp=4', dict(k=-1.0, sel_temp=4.0)),
               ('k=-2 temp=8', dict(k=-2.0, sel_temp=8.0)),
               ('null', dict())],
    # Batch 3 was interrupted after k=-1.5 (56.7%) and k=-2.0 (58.5%); this
    # picks up the rest.  k is pushed until it turns over -- it has to
    # eventually, since k -> -inf degenerates into picking the most CERTAIN
    # edge irrespective of its value.
    'batch3b': [('risk k=-3.0', dict(k=-3.0)), ('risk k=-4.0', dict(k=-4.0)),
                ('risk k=-6.0', dict(k=-6.0)),
                ('seltemp 8', dict(sel_temp=8.0)),
                ('seltemp 16', dict(sel_temp=16.0)),
                ('k=-2 temp=8', dict(k=-2.0, sel_temp=8.0)),
                ('null', dict())],
    # Batch 4: k turned over at -2.0 (58.5%, then -3: 54.8, -4: 53.8, -6:
    # 49.1), so that axis is solved.  seltemp did NOT turn -- 8: 62.9%, 16:
    # 65.2% -- and the two compose (k=-2 temp=8: 66.8%).  Push temp to the
    # greedy limit: as temp -> inf the Thompson draw collapses to the mean, so
    # descent becomes deterministic best-first and only virtual loss still
    # diversifies a wave.  That either keeps paying or falls off a cliff when
    # exploration disappears, and which one matters a great deal.
    'batch4': [('seltemp 32', dict(sel_temp=32.0)),
               ('seltemp 64', dict(sel_temp=64.0)),
               ('seltemp 1e4', dict(sel_temp=1e4)),
               ('k=-2 temp=16', dict(k=-2.0, sel_temp=16.0)),
               ('k=-2 temp=32', dict(k=-2.0, sel_temp=32.0)),
               ('k=-1 temp=16', dict(k=-1.0, sel_temp=16.0)),
               ('null', dict())],
    # Budget check.  Everything above was measured at 64 sims, but the arms
    # TRAIN at ~150, and the two reasons to expect a budget interaction pull in
    # opposite directions: sd[v] shrinks as sims grow, so a fixed k is a
    # smaller correction at 150; and a sharper tree plausibly helps most when
    # the budget is too small to explore broadly anyway.  Run this preset at
    # --sims 150 and --sims 400 before treating any of it as a recipe.
    'budget': [('seltemp 1e4', dict(sel_temp=1e4)),
               ('k=-2 temp=1e4', dict(k=-2.0, sel_temp=1e4)),
               ('seltemp 8', dict(sel_temp=8.0)),
               ('null', dict())],
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


def pick(c4, root, pol):
    """The move this policy plays from an already-searched root."""
    mode = pol.get('mode', 'value')
    touched = (root.visits() > 0) | (root.term >= 0)
    if mode == 'value':
        ev, var = c4.dir_value_mean_var(root.alpha_sel)
        k = pol.get('k', 0.0)
        score = ev if k == 0.0 else ev + k * np.sqrt(np.maximum(var, 0.0))
    elif mode == 'visits':
        score = root.visits().astype(np.float64)
    elif mode == 'visits_value':
        # Value only ever breaks ties: scaled below one visit so it can never
        # outrank a genuinely more-visited edge.
        score = (root.visits().astype(np.float64)
                 + 0.49 * np.tanh(c4.dir_value(root.alpha_sel)))
    elif mode == 'evidence':
        rows = []
        for i in range(len(root.legal)):
            if root.term[i] >= 0:
                rows.append(c4._SPIKE[root.term[i]])
                continue
            obs = c4.observed_alpha(root.eacc[i], c4._SEARCH_AGG)
            rows.append(root.alpha_p[i] if obs is None else obs)
        score = c4.dir_value(np.asarray(rows))
    else:
        raise ValueError(mode)
    if touched.any():
        score = np.where(touched, score, -np.inf)
    return int(root.legal[int(np.argmax(score))])


def play(c4, game, net, pol0, pol1, sims, opening, rng):
    state = game.new_initial_state()
    for a in opening:
        state.apply_action(int(a))
    ply = 0
    while not state.is_terminal() and ply < MAX_PLIES:
        pol = pol0 if state.current_player() == 0 else pol1
        bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                           temp=pol.get('sel_temp', 1.0), random_state=rng)
        state.apply_action(pick(c4, bot.mcts_search(state), pol))
        ply += 1
    return state.returns()[0]


def _worker(args):
    cand, n_games, seed, sims = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    net = c4.load_benchmark_net(NET[0], NET[1], ad.MM_SIG)
    rng = np.random.default_rng(seed)

    pts, played = 0.0, 0
    for _pair in range(n_games // 2):
        # One opening played with both colour assignments, so an opening that
        # simply favours the mover cancels instead of adding noise.
        st = game.new_initial_state()
        opening = []
        for _ in range(OPENING_PLIES):
            leg = st.legal_actions()
            a = int(leg[rng.integers(len(leg))])
            st.apply_action(a)
            opening.append(a)
        r0 = play(c4, game, net, cand, BASELINE, sims, opening, rng)
        r1 = play(c4, game, net, BASELINE, cand, sims, opening, rng)
        # returns()[0] is player 0's result; the candidate is player 1 in the
        # second game, so its result is the negation.
        pts += 0.5 * (r0 + 1.0) + 0.5 * ((-r1) + 1.0)
        played += 2
    return pts, played


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def run(name, cand, games, sims):
    per = max(2, games // N_WORKERS)
    jobs = [(cand, per, 500 + 977 * w, sims) for w in range(N_WORKERS)]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    pts = sum(p for p, _ in parts)
    n = sum(g for _, g in parts)
    frac = pts / n
    lo, hi = wilson(frac, n)
    flag = '' if lo < 0.5 < hi else '   <-- separated from 50%'
    print(f'  {name:<16} {frac:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}  '
          f'{time.time()-t0:.0f}s{flag}', flush=True)
    return frac, lo, hi, n


if __name__ == '__main__':
    preset = sys.argv[1] if len(sys.argv) > 1 else 'batch1'
    games = 600
    sims = SIMS
    if '--games' in sys.argv:
        games = int(sys.argv[sys.argv.index('--games') + 1])
    if '--sims' in sys.argv:
        sims = int(sys.argv[sys.argv.index('--sims') + 1])
    print(f'preset={preset}  net={NET[0]}@{NET[1]}  FIXED sims={sims}  '
          f'games={games}/candidate')
    print("score is the CANDIDATE's, vs the incumbent (value argmax, k=0, "
          "temp=1)")
    for nm, c in PRESETS[preset]:
        run(nm, {**BASELINE, **c}, games, sims)
