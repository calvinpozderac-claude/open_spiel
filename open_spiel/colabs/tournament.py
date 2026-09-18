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

# (name, dir, label, engine, param) -- each net deploys the search it trained
# with.  engine 'thompson' -> param is the Dirichlet search_agg (c4 module);
# engine 'az' -> param is c_puct (connect4_alphazero_utils, PUCT + policy/value).
PARTICIPANTS = [
    ('allmse',        'othello_benchmark/bench_ALLMSE',                '5000', 'thompson', 'pathsum'),
    ('allmse_admle',  'othello_benchmark/bench_ALLMSE_ADMLE_AVGCONC',  '5000', 'thompson', 'additive_mle'),
    ('allmse_pathav', 'othello_benchmark/bench_ALLMSE_PATHSUM_PATHAVG','5000', 'thompson', 'pathsum'),
    ('allmse_ad_pav', 'othello_benchmark/bench_ALLMSE_ADMLE_PATHAVG',  '5000', 'thompson', 'additive_mle'),
    ('ZGROUND',       'othello_benchmark/bench_ZGROUND',               '5000', 'thompson', 'pathsum'),
    ('HYBRID',        'othello_benchmark/bench_HYBRID',                '5000', 'thompson', 'additive_mle'),
    ('DISTILL',       'othello_benchmark/bench_DISTILL',               '5000', 'thompson', 'pathsum'),
    ('AZ',            'othello_benchmark/bench_AZ_MATCH',              '5000', 'az',       1.5),
]
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != 'report' else 100
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 400


def _load(engine, d, label, c4, az):
    mod = az if engine == 'az' else c4
    return mod.load_benchmark_net(d, label, SIG)


def _move(engine, param, net, game, state, rng, c4, az):
    """One eval move from `net` using its own engine.

    AZ: PUCT is otherwise fully deterministic (no root noise, no stochastic
    tie-break) -- greedy argmax collapses many "games" to the same handful of
    trajectories (confirmed: az_ladder.json pairwise scores were all exact
    multiples of 1/4). Sample from the visit distribution at temp=0.3, this
    repo's own established standard-AlphaZero eval protocol (az_screen_pick.
    log baseline), for real per-game variance while still close to greedy."""
    if engine == 'az':
        az.set_search(c_puct=float(param))
        bot = az.AZMCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                           random_state=rng, noise_frac=0.0)
        root = bot.mcts_search(state)
        return az.root_pick(root, rng, sample=True, temp=0.3)
    c4.set_search(search_agg=param, target_agg=param)
    bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                       temp=OUR_TEMP, random_state=rng)
    root = bot.mcts_search(state)
    return c4.root_pick(root, rng, thompson=False, temp=OUR_TEMP)


def _worker(args):
    ((an, ad, al, aeng, aparam), (bn, bd, bl, beng, bparam),
     n_games, seed, a_first) = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import connect4_alphazero_utils as az
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(selection='dirichlet', virtual_loss=1.0)
    net_a = _load(aeng, ad, al, c4, az)
    net_b = _load(beng, bd, bl, c4, az)
    rng = np.random.default_rng(seed)
    a_pts = 0.0
    for gi in range(n_games):
        a_seat = (0 if a_first else 1)
        if gi % 2 == 1:
            a_seat = 1 - a_seat
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.current_player() == a_seat:
                mv = _move(aeng, aparam, net_a, game, st, rng, c4, az)
            else:
                mv = _move(beng, bparam, net_b, game, st, rng, c4, az)
            st.apply_action(int(mv))
        r = st.returns()[a_seat]
        a_pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return (an, bn), a_pts, n_games


def fit_elo(names, wins, games, anchor=1500.0, iters=None, lr=None, reg=1.0):
    """Bradley-Terry MAP -> Elo: minimise the ridge-penalised negative
    log-likelihood with L-BFGS (exact, scale-free -- no hand-tuned iters/lr).

    reg is a Gaussian prior VARIANCE in Elo^2 on each rating around `anchor`
    (bigger reg = weaker prior = closer to raw MLE); default 1.0 is
    deliberately tiny (stdev ~1 Elo) -- see why below.

    Plain MLE (reg=inf i.e. no penalty) is only finite when every pairwise
    score is strictly between 0 and games. If one group sweeps another (e.g.
    a near-0/80 cross record -- exactly what happened AZ vs HYBRID here), the
    true MLE gap is INFINITE: verified empirically, unregularised gradient
    ascent drifted the fitted gap from ~190 to ~370 to ~560 Elo as iterations
    went 10k->50k->200k, never converging. A ridge prior (equivalent to
    Firth-type penalisation for separable logistic regression) makes the
    posterior mode finite and well-defined, but the specific reg value then
    directly sets how far "near-total domination" gets reported (see
    az-vs-thompson-plateau memory for the reg-sweep this default was picked
    from) -- report it alongside every fitted gap, and prefer citing the raw
    empirical score (e.g. "3/2000, 99.85% [.15-2.4%] Wilson CI") for any
    near-perfect-separation matchup rather than leaning on the point Elo."""
    from scipy.optimize import minimize
    n = len(names)
    idx_i, idx_j, w_ij, g_ij = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i != j and games[i][j] > 0:
                idx_i.append(i); idx_j.append(j)
                w_ij.append(wins[i][j]); g_ij.append(games[i][j])
    idx_i, idx_j = np.array(idx_i), np.array(idx_j)
    w_ij, g_ij = np.array(w_ij), np.array(g_ij)
    S = 400.0 / np.log(10.0)

    def nll_grad(r):
        d = (r[idx_j] - r[idx_i]) / S
        logp = -np.logaddexp(0.0, d)              # log sigmoid(-d) = log p_i_beats_j
        logq = -np.logaddexp(0.0, -d)
        nll = -(w_ij * logp + (g_ij - w_ij) * logq).sum()
        p = 1.0 / (1.0 + np.exp(d))
        dL = (w_ij - g_ij * p)                     # d(loglik)/d r_i, per row
        grad = np.zeros(n)
        np.add.at(grad, idx_i, dL)
        np.add.at(grad, idx_j, -dL)
        pen = ((r - anchor) ** 2).sum() / (2 * reg)
        pen_grad = (r - anchor) / reg
        return nll + pen, -grad + pen_grad          # minimise -loglik + penalty

    r0 = np.full(n, anchor, float)
    res = minimize(nll_grad, r0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=2000, ftol=1e-12, gtol=1e-10))
    r = res.x
    return r - r.mean() + anchor


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
    print(f'\n{"rank":<5}{"model":<15}{"Elo":>6}{"score":>9}{"win%":>7}   engine/search')
    for rank, i in enumerate(order, 1):
        wp = tot_pts[i] / max(tot_g[i], 1)
        eng = f'{parts[i][3]}:{parts[i][4]}'
        print(f'{rank:<5}{names[i]:<15}{elo[i]:>6.0f}'
              f'{tot_pts[i]:>6.0f}/{tot_g[i]:<3.0f}{wp:>6.1%}   {eng}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
