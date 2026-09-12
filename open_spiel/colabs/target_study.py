"""STUDY 2 — which TARGET family should training regress on?

Three families, scored per labelled position against Edax ground truth, all
from the same search tree (search_agg fixed to the incumbent additive_mle,
400 sims) so the comparison isolates the TARGET rule:

  (a) backed-up leaf predictions -- observed_alpha(root.nacc, agg) for agg in
      {additive_mle, avgconc, mixture, pathavg}.  Dense (every position gets
      one), but self-referential: quality is bounded by the net's own leaves.
  (b) final game results -- K greedy-ish 64-sim playouts from the position;
      each z is what a training sample would carry.  Unbiased w.r.t. the
      playing policy but ONE noisy sample; scored as bias (mean_z - truth) and
      variance separately, since E[(z-truth)^2] = bias^2 + var.
  (c) child-consistency -- flip(V_net(s')) after the Edax-best move: the
      cons-term target.  Dense and cheap, but bounded by the net at s'.

  baseline: the net's own prior V_net(s) with no search at all.

Edge-level too: for the Edax-best edge, family (a) collapses eacc[best] per
agg; (b)'s z and (c)'s flip both target exactly that edge; truth for
Q(s, best) under optimal play is value(s) itself.

    python target_study.py [n_per_phase]      # run (default 40)
    python target_study.py report
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
OUT = 'target_study.json'
SIG = (32, 10, 8)
NETS = [
    ('HYBRID@50k', 'othello_benchmark/bench_HYBRID', '50000'),
    ('EDL@50k',    'othello_benchmark/bench_EDL',    '50000'),
]
SEARCH_SIMS = 400
PLAYOUT_SIMS = 64
K_PLAYOUTS = 3
TEMP_THRESHOLD = 20            # match self-play: thompson until ply 20, then greedy
AGGS = ['additive_mle', 'avgconc', 'mixture', 'pathavg']
PHASES = [(45, 61, 'opening'), (33, 45, 'early-mid'), (25, 33, 'late-mid'),
          (15, 25, 'early-end'), (8, 15, 'endgame'), (0, 8, 'final')]
N_WORKERS = 4


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    return c4, game


def _alpha_val(c4, a):
    a0 = max(float(np.sum(a)), 1e-9)
    return float(a[c4._WIN] - a[c4._LOSS]) / a0, a0


def _ell(c4, a, truth):
    from scipy.special import digamma
    ti = c4._WIN if truth > 0 else (c4._LOSS if truth < 0 else c4._DRAW)
    a0 = max(float(np.sum(a)), 1e-9)
    return float(digamma(max(float(a[ti]), 1e-9)) - digamma(a0))


def _playout(c4, game, net, state, rng):
    """Self-play-like continuation: thompson temp-1 until ply TEMP_THRESHOLD,
    near-greedy after (matches the training runs' temp schedule, at a cheaper
    64-sim budget).  Returns the outcome for the mover at `state`."""
    st = state.clone()
    me = st.current_player()
    bot = c4.C4MCTSBot(game, net, 'cpu', PLAYOUT_SIMS, batch_size=16,
                       temp=1.0, random_state=rng)
    root = None
    while not st.is_terminal():
        ply = st.move_number()
        root = bot.mcts_search(st, root=root)
        thom = ply < TEMP_THRESHOLD
        mv = c4.root_pick(root, rng, thompson=thom,
                          temp=1.0 if thom else 64.0)
        root = c4._descend(root, mv)
        st.apply_action(mv)
    r = st.returns()[me]
    return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)


def _worker(args):
    net_name, ckpt_dir, label, recs = args
    c4, game = _init()
    net = c4.load_benchmark_net(ckpt_dir, label, SIG)
    c4.set_search(search_agg='additive_mle', target_agg='pathavg',
                  selection='dirichlet', virtual_loss=1.0)
    rng = np.random.default_rng(7)

    out = []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        truth = float(r['value'])
        row = dict(empties=r['empties'], tier=r['tier'], truth=truth)

        # net prior baseline (no search)
        v3, vc, p3, cf, _ = c4.nn_eval_states(net, 'cpu', [st])
        pa = np.maximum(float(vc[0]) * np.asarray(v3[0], float), 1e-9)
        v, a0 = _alpha_val(c4, pa)
        row['prior'] = dict(v=v, a0=a0, ell=_ell(c4, pa, truth))

        # (a) one search, collapsed under each agg -- state and best-edge
        bot = c4.C4MCTSBot(game, net, 'cpu', SEARCH_SIMS, batch_size=16,
                           temp=64.0, random_state=rng)
        root = bot.mcts_search(st.clone())
        best_pos = np.nonzero(root.legal == r['best'])[0]
        bi = int(best_pos[0]) if len(best_pos) else -1
        for agg in AGGS:
            ra = c4.observed_alpha(root.nacc, agg)
            v, a0 = _alpha_val(c4, ra)
            row[f'a_{agg}'] = dict(v=v, a0=a0, ell=_ell(c4, ra, truth))
            if bi >= 0 and (root.eacc[bi][0] > 0 or root.term[bi] >= 0):
                ea = (np.asarray(c4._SPIKE[root.term[bi]], float)
                      if root.term[bi] >= 0
                      else c4.observed_alpha(root.eacc[bi], agg))
                v, a0 = _alpha_val(c4, ea)
                row[f'ae_{agg}'] = dict(v=v, a0=a0, ell=_ell(c4, ea, truth))

        # (c) child-consistency target for the Edax-best edge
        st2 = st.clone()
        st2.apply_action(int(r['best']))
        if st2.is_terminal():
            ret = st2.returns()[st.current_player()]
            ti = c4._WIN if ret > 0 else (c4._LOSS if ret < 0 else c4._DRAW)
            ca = np.asarray(c4._SPIKE[ti], float)
        else:
            v3b, vcb, _, _, _ = c4.nn_eval_states(net, 'cpu', [st2])
            cb = np.maximum(float(vcb[0]) * np.asarray(v3b[0], float), 1e-9)
            ca = c4.flip_alpha(cb)          # one ply up, mover flips
        v, a0 = _alpha_val(c4, ca)
        row['cons'] = dict(v=v, a0=a0, ell=_ell(c4, ca, truth))

        # (b) game-result samples
        zs = [_playout(c4, game, net, st, rng) for _ in range(K_PLAYOUTS)]
        row['z'] = zs
        out.append(row)
    return (net_name,), out


def run(npp):
    with open(CACHE) as fh:
        allrecs = json.load(fh)['positions']
    recs = []
    for lo, hi, _ in PHASES:
        pool = [r for r in allrecs if lo <= r['empties'] < hi]
        recs.extend(pool[:npp])
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    # shard positions across workers per net, merge on completion
    jobs = []
    for (nm, d, lab) in NETS:
        if nm in done:
            continue
        shards = [recs[i::N_WORKERS] for i in range(N_WORKERS)]
        jobs.extend((nm, d, lab, sh) for sh in shards if sh)
    print(f'[s2] {len(recs)} positions x {len(NETS)} nets '
          f'({len(jobs)} shards)', flush=True)
    parts = {}
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for (key,), rows in pool.imap_unordered(_worker, jobs):
            parts.setdefault(key, []).extend(rows)
            print(f'[s2] shard for {key} done ({time.time()-t0:.0f}s)',
                  flush=True)
    for k, v in parts.items():
        done[k] = v
    with open(OUT, 'w') as fh:
        json.dump(done, fh)
    print('[s2] TARGET STUDY DONE', flush=True)


def report():
    from scipy.stats import spearmanr
    with open(OUT) as fh:
        done = json.load(fh)
    fams = (['prior'] + [f'a_{g}' for g in AGGS]
            + [f'ae_{g}' for g in AGGS] + ['cons'])
    for nm, rows in done.items():
        print(f'=== {nm}  ({len(rows)} positions) ===')
        print(f'{"family":<18}{"n":>5}{"MSE":>8}{"sign%":>7}{"a0(med)":>9}'
              f'{"ell":>8}{"rho(a0,-err)":>13}')
        for f in fams:
            sub = [r for r in rows if f in r]
            if not sub:
                continue
            errs = [(r[f]['v'] - r['truth']) ** 2 for r in sub]
            dec = [r for r in sub if r['truth'] != 0]
            sg = (np.mean([(r[f]['v'] > 0) == (r['truth'] > 0) for r in dec])
                  if dec else float('nan'))
            a0s = [r[f]['a0'] for r in sub]
            ell = np.mean([r[f]['ell'] for r in sub])
            rho = spearmanr(a0s, [-e for e in errs]).statistic
            print(f'{f:<18}{len(sub):>5}{np.mean(errs):>8.3f}{sg:>7.1%}'
                  f'{np.median(a0s):>9.2f}{ell:>8.3f}{rho:>13.3f}')
        zs = np.array([r['z'] for r in rows], float)      # (N, K)
        tr = np.array([r['truth'] for r in rows], float)
        bias = float(np.mean(zs.mean(1) - tr))
        mse1 = float(np.mean((zs[:, 0] - tr) ** 2))       # one-sample target
        var = float(np.mean(zs.var(1)))
        sg = float(np.mean((zs.mean(1) > 0) == (tr > 0))) if len(tr) else 0
        print(f'{"z (1 sample)":<18}{len(rows):>5}{mse1:>8.3f}{"":>7}'
              f'  bias {bias:+.3f}  var {var:.3f}  sign(mean_z) {sg:.1%}')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
