"""Escalating Edax benchmark for MM_long: how high up the ladder can it climb?

Every new checkpoint plays 50 games against Edax level 1. If it scores above
50% it moves up a level and plays 50 more, repeating until Edax takes the
majority. The reported number is the highest level the model beat -- a single
interpretable figure that does not saturate, unlike a fixed rung.

This design fixes the failure that dominated the project: a rung the model
beats easily carries no information, and the ranking of two models can invert
between rungs. Escalating means the measurement always lands where the model
actually is.

Both sides use 1000 simulations, matching training, since the grid sweep
showed rankings invert with search budget and the only honest budget to
evaluate at is the one you deploy at.

Our side plays at temp=64. Edax is deterministic, so at the deterministic
descent setting a 50-game match would replay a couple of games; temp=64 is the
highest temperature that still yields fully distinct games. It costs roughly
53 Elo relative to deterministic descent but applies identically to every
checkpoint, so the curve is comparable across the run.

    python mm_long_bench.py [--once]
"""
import json
import os
import re
import sys
import time

import numpy as np

# Configured by environment so a second run can reuse this protocol unchanged.
# It MUST be the environment rather than module globals a wrapper overwrites:
# _play_level spawns a multiprocessing.Pool, Windows starts those with 'spawn',
# and each child re-imports this module fresh -- an override assigned in the
# parent would be silently lost in the workers, which would then score the
# WRONG run's checkpoints. Defaults reproduce the MM_long run exactly.
CKPT_DIR = os.environ.get('BENCH_CKPT_DIR', 'othello_benchmark/bench_MM_long')
# latest.pt is rewritten every quick_eval_every (1000) episodes, whereas
# bench_<ep>.pt only lands every deep_eval_every (5000). Snapshotting latest.pt
# gives 5x the benchmark resolution with no restart and no lost episodes.
#
# The snapshots go in a SEPARATE directory on purpose: _pool_labels() scans the
# training checkpoint dir to pick frozen self-play opponents, so dropping extra
# bench_<ep>.pt files in there would silently change the training distribution.
SNAP_DIR = os.environ.get('BENCH_SNAP_DIR',
                          'othello_benchmark/bench_MM_long_snaps')
BENCH_EVERY = int(os.environ.get('BENCH_EVERY', '1000'))
RESULTS = os.environ.get('BENCH_RESULTS', 'mm_long_bench.json')
LOG = os.environ.get('BENCH_LOG', 'mm_long_bench.log')
SIMS = 1000
GAMES_PER_LEVEL = 51        # 17 x 3 workers, so the split is exact
OUR_TEMP = 64.0
MAX_LEVEL = 20
# ALWAYS play these, whatever happens on the lower ones. A pure climb-until-
# you-lose protocol stops at the first rung the model drops below 50%, which
# makes it blind to the failure mode this project keeps hitting: a model can
# get slightly WORSE against a weak opponent while getting better against a
# strong one. MM@5000 beat MM@20000 by 17.8 points against L1 and lost to it
# by 5 against L2. Stopping at L1 would have recorded that as pure regression.
FIXED_RUNGS = [1.0, 2.0, 3.0, 4.0]
RUNG_STEP = 0.5          # step size when extending above or bracketing below
N_WORKERS = 3          # leave cores for the trainer running alongside
POLL_S = 300


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
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def snapshot_latest():
    """Copy the model out of latest.pt into SNAP_DIR as a plain state dict.
    Returns the episode number, or None if there is nothing new."""
    import torch
    src = os.path.join(CKPT_DIR, 'latest.pt')
    if not os.path.exists(src):
        return None
    try:
        blob = torch.load(src, map_location='cpu', weights_only=False)
    except Exception:
        return None                      # mid-write; try again next poll
    ep = int(blob.get('ep', 0))
    if ep <= 0 or ep % BENCH_EVERY:
        return None
    os.makedirs(SNAP_DIR, exist_ok=True)
    dst = os.path.join(SNAP_DIR, f'bench_{ep}.pt')
    if not os.path.exists(dst):
        torch.save(blob['model'], dst)
    return ep


def _rung_opponent(eb, game, rung, seed):
    """Integer rung -> that Edax level; half rung -> 50/50 of its neighbours."""
    lo = int(np.floor(rung))
    if float(rung).is_integer():
        return eb.EdaxBot(game, level=lo)
    return eb.EdaxMixBot(game, p_lo=0.5, lo=lo, hi=lo + 1, seed=seed)


def _worker(args):
    label, rung, n_games, seed, seat0 = args
    c4, game = _init()
    import adaptive_stop_utils as ad
    import policy_screen as ps
    import edax_bot as eb

    # Read the architecture off the checkpoint. ad.MM_SIG is (32, 3, 8), which
    # is bench_MM's shape; the OLIVAW run is 10 blocks and would raise here.
    import re as _re
    import torch as _torch
    _sd = _torch.load(os.path.join(SNAP_DIR, f'bench_{label}.pt'),
                      map_location='cpu', weights_only=True)
    _sig = (int(_sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in _sd
                 if (m := _re.match(r'body\.(\d+)\.', k))}),
            int(_sd['head.0.weight'].shape[0]))
    del _sd
    net = c4.load_benchmark_net(SNAP_DIR, label, _sig)
    opp = _rung_opponent(eb, game, rung, seed)
    pol = dict(mode='value', k=0.0, sel_temp=OUR_TEMP)
    rng = np.random.default_rng(seed)

    pts = 0.0
    for gi in range(n_games):
        seat = (seat0 + gi) % 2
        st = game.new_initial_state()
        while not st.is_terminal():
            if st.current_player() == seat:
                b = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16,
                                 temp=OUR_TEMP, random_state=rng)
                a = ps.pick(c4, b.mcts_search(st), pol)
            else:
                a = opp.step(st)
            st.apply_action(a)
        r = st.returns()[seat]
        pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    opp.close()
    return pts, n_games


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def _play_level(label, rung, rungs):
    """One rung, cached in `rungs`."""
    import multiprocessing as mp
    key = f'L{rung:g}'
    if key in rungs:
        return rungs[key]['score']
    per = max(2, GAMES_PER_LEVEL // N_WORKERS)
    jobs = [(label, rung, per, 3300 + 71 * w, w % 2)
            for w in range(N_WORKERS)]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    pts = sum(p for p, _ in parts)
    n = sum(g for _, g in parts)
    frac = pts / n
    lo, hi = wilson(frac, n)
    rungs[key] = dict(score=frac, n=n, lo=lo, hi=hi)
    _log(f'    ep {label} vs Edax L{rung:g}: {frac:6.1%} '
         f'[{lo:.1%}, {hi:.1%}]  n={n}  ({time.time()-t0:.0f}s)')
    return frac


def climb(label):
    """Score the full FIXED_RUNGS profile, extend above it if still winning,
    bracket below it if losing even L1, then report a continuous rating.

    The profile is the point. A single "highest level beaten" is both a hard
    threshold on a noisy estimate (51 games is ~7% standard error, which is why
    ep3000 and ep4000 read L1 and L0 off 62.5% and 47.9%) and blind to a model
    trading weak-opponent skill for strong-opponent skill. Recording every rung
    every time makes that trade visible instead of invisible.
    """
    rungs = {}
    for rung in FIXED_RUNGS:
        _play_level(label, rung, rungs)

    # Still winning at the top of the fixed set -> keep going.
    rung = max(FIXED_RUNGS) + RUNG_STEP
    while (rungs[f'L{rung - RUNG_STEP:g}']['score'] > 0.5
           and rung <= MAX_LEVEL):
        if _play_level(label, rung, rungs) <= 0.5:
            break
        rung += RUNG_STEP

    # Losing even the bottom of the fixed set -> bracket downward.
    probe = min(FIXED_RUNGS) - RUNG_STEP
    while rungs[f'L{min(FIXED_RUNGS):g}']['score'] <= 0.5 and probe >= 0.0:
        if _play_level(label, probe, rungs) > 0.5:
            break
        probe -= RUNG_STEP

    # Rating: where the win rate crosses 50%, over whatever rungs were played.
    played = sorted((float(k[1:]), v['score']) for k, v in rungs.items())
    highest_beaten = max([r for r, sc in played if sc > 0.5], default=0.0)
    rating = highest_beaten
    for (r_lo, s_lo), (r_hi, s_hi) in zip(played, played[1:]):
        if s_lo > 0.5 >= s_hi and s_lo > s_hi:
            rating = r_lo + (r_hi - r_lo) * (s_lo - 0.5) / (s_lo - s_hi)
    return highest_beaten, rating, rungs


def available(d):
    if not os.path.isdir(d):
        return []
    out = [int(m.group(1)) for f in os.listdir(d)
           if (m := re.fullmatch(r'bench_(\d+)\.pt', f))]
    return sorted(out)


def main():
    once = '--once' in sys.argv
    _log(f'watching {CKPT_DIR}: {SIMS} sims both sides, '
         f'{GAMES_PER_LEVEL} games/level, every {BENCH_EVERY} eps, '
         f'climbing until Edax wins')
    while True:
        done = {}
        if os.path.exists(RESULTS):
            with open(RESULTS) as fh:
                done = json.load(fh)
        snapshot_latest()
        for ep in available(SNAP_DIR):
            if str(ep) in done or ep == 0:
                continue
            _log(f'  === benchmarking ep {ep} ===')
            best, rating, rungs = climb(str(ep))
            done[str(ep)] = dict(highest_level_beaten=best,
                                 edax_rating=rating, rungs=rungs)
            with open(RESULTS, 'w') as fh:
                json.dump(done, fh, indent=2, sort_keys=True)
            ordered = sorted(rungs.items(), key=lambda kv: float(kv[0][1:]))
            prof = '  '.join(f'{k} {v["score"]:.0%}' for k, v in ordered)
            # The profile leads, not the rating. `rating` interpolates a 50%
            # crossing, which assumes win rate falls as level rises -- and on
            # ep17000 that assumption broke badly enough to matter: L3 11.8%
            # (3 seed sets) against L4 38.6% made it report 2.14 off the L2->L3
            # crossing while discarding a 38% L4 entirely. It is still written
            # to the JSON for continuity with the earlier rows, but a scalar
            # that silently drops the top rung should not be the headline.
            scores = [v['score'] for _, v in ordered]
            inverted = [f'{ordered[i][0]}<{ordered[i+1][0]}'
                        for i in range(len(scores) - 1)
                        if scores[i] + 0.10 < scores[i + 1]]
            warn = (f'  ** NON-MONOTONE ({", ".join(inverted)}): rating '
                    f'{rating:.2f} is not meaningful **' if inverted else
                    f'  (rating {rating:.2f})')
            _log(f'  BENCHMARK ep {ep}: {prof}{warn}')
        if once:
            return
        time.sleep(POLL_S)


if __name__ == '__main__':
    main()
