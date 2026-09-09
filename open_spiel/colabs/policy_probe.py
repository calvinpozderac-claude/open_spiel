"""A per-position metric, so a checkpoint can be scored without playing games.

The Edax match benchmark has a measurement problem that no amount of care in
mm_long_bench.py fixes. A cell is 51 games against a deterministic opponent
from one seed set; seed sweeps put its sd near 5 points per cell, which
propagates to ~3.2 on the four-rung average against an observed
checkpoint-to-checkpoint sd of 5.2. So roughly 40% of the swing between
neighbouring checkpoints is the measurement rather than the network, and the
two are not separable from match results alone.

That ambiguity is the whole problem: "the eval is too noisy and the run is
learning fine" and "the network is genuinely unstable" produce identical
match-play curves. This scores the NETWORK on a fixed, pre-labelled set of
positions instead:

  * n positions, so the standard error falls as 1/sqrt(n) -- at n=3000 top-1
    agreement has se ~0.9% against the match metric's effective ~3.2%.
  * the position set never changes, so checkpoints are exactly comparable and
    there is no seed dependence at all.
  * scoring is forward passes only -- seconds, versus 36 minutes for a full
    L1-L4 benchmark.

Whatever fluctuation survives THIS metric is the network, not the yardstick.

What is measured, and why not "policy CE" as such: this architecture has no
softmax policy head. `nn_eval_states` returns p3 with shape (B, A, 3) -- a
per-action Dirichlet over {loss, draw, win} -- so a move distribution has to be
derived. The honest primitives are:

  top-1 agreement   does argmax_a (win_a - loss_a) match Edax's move?  A pure
                    Bernoulli trial, the highest-power thing available per
                    position, and it measures the DECISION rather than the
                    calibration.
  policy CE         cross-entropy of softmax(q_legal) against Edax's move.
                    Reported for continuity with the usual metric, but the
                    softmax temperature is arbitrary -- treat movement in it as
                    meaningful only relative to other checkpoints.
  value MSE/sign    E[win]-E[loss] against exact solved truth, on the subset
                    with few enough empties for Edax to solve outright.

Three caveats worth keeping in view:

  Head quality is not playing strength. fa76db55 measured OPT as a strong
  evaluator its own search could not improve, and AZ as a weak evaluator search
  substantially rescued -- a net can gain agreement here and still lose games.
  This replaces the TRACKING signal, not the ground truth; keep running Edax
  matches, just less often.

  Endgames flatter a value head, so the value numbers are optimistic by
  construction. Top-1 agreement over midgame positions is the harder half.

  Rising agreement means "more Edax-like", not "stronger". It is an external
  check rather than a circular one -- the training targets come from search,
  not from Edax -- but it is still imitation of a fixed engine.

    python policy_probe.py --build                 # once: label the positions
    python policy_probe.py --score <dir> <label>   # per checkpoint, seconds
    python policy_probe.py --sweep <dir> <labels…> # several at once
"""
import json
import os
import sys
import time

import numpy as np

CACHE = 'policy_probe_positions.json'
N_POSITIONS = 3000
TRUTH_LEVEL = 18          # Edax level defining the "best move" target
PLAY_LEVEL = 10           # cheaper level used to walk games forward
SOLVE_EMPTIES = 14        # at or below this, TRUTH_LEVEL solves outright
OPEN_MIN, OPEN_MAX = 4, 10    # random opening plies, for position diversity
SNAP_MIN_PLY = 8          # skip the random opening itself
SNAP_STRIDE = 6           # plies between snapshots within one game
CHUNK = 256               # forward-pass batch


def _init():
    import torch
    torch.set_num_threads(4)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _empties(state):
    t = np.asarray(state.observation_tensor(0)).reshape(3, 8, 8)
    return int(t[0].sum())


def _replay(game, actions):
    st = game.new_initial_state()
    for a in actions:
        st.apply_action(int(a))
    return st


def _solved_value(game, state, bot):
    """Game-theoretic value in the mover's frame, by playing the position out
    with a level that solves it.  Only called when empties <= SOLVE_EMPTIES."""
    mover = state.current_player()
    st = state.clone()
    while not st.is_terminal():
        st.apply_action(bot.step(st))
    r = st.returns()[mover]
    return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)


def build(n_positions, seed=17):
    """Sample positions from Edax-vs-Edax play and label them once.

    Positions come from real engine games with randomized openings, NOT from
    random play as head_diagnostic.py samples them -- a value head is being
    asked about positions it would actually face, and random-play positions are
    off-distribution enough to make the metric measure the wrong thing.
    """
    c4, game = _init()
    import edax_bot as eb
    rng = np.random.default_rng(seed)
    truth = eb.EdaxBot(game, level=TRUTH_LEVEL)
    walker = eb.EdaxBot(game, level=PLAY_LEVEL)

    recs, t0, games = [], time.time(), 0
    while len(recs) < n_positions:
        st, actions = game.new_initial_state(), []
        for _ in range(int(rng.integers(OPEN_MIN, OPEN_MAX + 1))):
            if st.is_terminal():
                break
            la = st.legal_actions()
            a = int(la[rng.integers(len(la))])
            st.apply_action(a)
            actions.append(a)

        while not st.is_terminal() and len(recs) < n_positions:
            ply = len(actions)
            if ply >= SNAP_MIN_PLY and (ply - SNAP_MIN_PLY) % SNAP_STRIDE == 0:
                emp = _empties(st)
                best = int(truth.step(st))
                val = (_solved_value(game, st, truth)
                       if emp <= SOLVE_EMPTIES else None)
                recs.append(dict(actions=[int(x) for x in actions],
                                 best=best, value=val, ply=ply, empties=emp))
                if len(recs) % 250 == 0:
                    print(f'  {len(recs)}/{n_positions} positions '
                          f'({time.time()-t0:.0f}s)', flush=True)
            mv = int(walker.step(st))
            st.apply_action(mv)
            actions.append(mv)
        games += 1

    truth.close()
    walker.close()
    solved = sum(1 for r in recs if r['value'] is not None)
    with open(CACHE, 'w') as fh:
        json.dump(dict(truth_level=TRUTH_LEVEL, play_level=PLAY_LEVEL,
                       solve_empties=SOLVE_EMPTIES, seed=seed,
                       positions=recs), fh)
    print(f'[build] {len(recs)} positions from {games} games, '
          f'{solved} with exact solved values -> {CACHE} '
          f'({time.time()-t0:.0f}s)')


def _load():
    with open(CACHE) as fh:
        blob = json.load(fh)
    return blob['positions']


def score(ckpt_dir, label, recs=None, c4=None, game=None, quiet=False):
    """Top-1 agreement, policy CE and value MSE for one checkpoint."""
    import adaptive_stop_utils as ad
    if c4 is None:
        c4, game = _init()
    recs = recs if recs is not None else _load()
    net = c4.load_benchmark_net(ckpt_dir, label, ad.MM_SIG)

    states = [_replay(game, r['actions']) for r in recs]
    hits, ces, verrs, vsign = [], [], [], []
    t0 = time.time()
    for i0 in range(0, len(states), CHUNK):
        chunk = states[i0:i0 + CHUNK]
        v3, _vc, p3, _cf, _o = c4.nn_eval_states(net, 'cpu', chunk)
        for j, st in enumerate(chunk):
            r = recs[i0 + j]
            legal = [int(x) for x in st.legal_actions()]
            if r['best'] not in legal:
                # Should not happen -- the target came from this position --
                # but a silent mismatch would corrupt the metric, so be loud.
                raise RuntimeError(
                    f'target move {r["best"]} not legal at ply {r["ply"]}; '
                    f'the cached position set does not match this game build')
            q = p3[j][legal][:, c4._WIN] - p3[j][legal][:, c4._LOSS]
            hits.append(legal[int(np.argmax(q))] == r['best'])
            e = np.exp(q - q.max())
            ces.append(-float(np.log(e[legal.index(r['best'])] / e.sum())))
            if r['value'] is not None:
                vp = float(v3[j][c4._WIN] - v3[j][c4._LOSS])
                verrs.append((vp - r['value']) ** 2)
                vsign.append(np.sign(vp) == np.sign(r['value']))

    n = len(hits)
    top1 = float(np.mean(hits))
    se = float(np.sqrt(top1 * (1 - top1) / n))
    out = dict(label=label, n=n, top1=top1, top1_se=se,
               ce=float(np.mean(ces)), ce_se=float(np.std(ces, ddof=1)/np.sqrt(n)),
               n_solved=len(verrs),
               value_mse=float(np.mean(verrs)) if verrs else float('nan'),
               value_sign=float(np.mean(vsign)) if vsign else float('nan'),
               secs=time.time() - t0)
    if not quiet:
        print(f'  ep {label:>6}  top1 {top1:6.1%} +/-{se:.1%}   '
              f'CE {out["ce"]:.3f}   '
              f'vMSE {out["value_mse"]:.3f} (n={out["n_solved"]})   '
              f'vsign {out["value_sign"]:.1%}   [{out["secs"]:.0f}s]',
              flush=True)
    return out


def main():
    a = sys.argv[1:]
    if '--build' in a:
        i = a.index('--build')
        n = int(a[i + 1]) if len(a) > i + 1 and a[i + 1].isdigit() else N_POSITIONS
        build(n)
        return
    if '--score' in a or '--sweep' in a:
        flag = '--score' if '--score' in a else '--sweep'
        i = a.index(flag)
        ckpt_dir, labels = a[i + 1], a[i + 2:]
        if not os.path.exists(CACHE):
            sys.exit(f'no {CACHE}; run --build first')
        recs = _load()
        c4, game = _init()
        print(f'[probe] {len(recs)} positions, Edax L{TRUTH_LEVEL} targets')
        for lab in labels:
            score(ckpt_dir, lab, recs, c4, game)
        return
    print(__doc__)


if __name__ == '__main__':
    main()
