"""Offline Bradley-Terry ladder over the OLIVAW checkpoints.

    python olivaw_ladder.py az            # arm A alone
    python olivaw_ladder.py tz            # arm B alone
    python olivaw_ladder.py both          # one fit containing BOTH arms
    python olivaw_ladder.py report [az|tz|both]

WHY THIS IS NEEDED AT ALL. The in-training ladder these runs print is an
incremental K-factor Elo update, which is order- and path-dependent: replaying
byte-identical results in a different order moved it 40.2 Elo in a controlled
test, and it is the entire explanation for OLIVAW-AZ's apparent "regressions"
(ep25k dropping below ep20k, ep35k landing third). Worse, arm A was started
before per-pair W/D/L tracking existed, so its ladder cannot be refitted --
`pair_games` recorded how often a pair met, never who won. The only way to
give those checkpoints an honest ranking is to replay the head-to-heads, which
is what this does.

`both` is the interesting one: putting every checkpoint of both arms into a
single round-robin means the AlphaZero-vs-ThompsonZero comparison comes out of
one BT fit over one connected win matrix, rather than from comparing two
separately-anchored ladders (which is not a valid comparison at all -- each
ladder's zero is arbitrary).

Saves incrementally, one pair at a time. tournament.py's all-at-the-end save
has lost a full ladder to an external kill twice in this project.
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import tournament as T

EPS = list(range(5000, 50001, 5000))
# 6 games/pair. Few per pair, but every checkpoint still plays 19 opponents
# = 114 games, and BT pools the whole matrix -- a sparse-but-connected design
# beats a dense one on few opponents. Simulated on a 20-player ladder with a
# realistic 450-Elo spread: mean |error| 28 Elo, p95 67, rank rho 0.976. Good
# enough to rank arms separated by >~50 Elo; NOT enough to call a photo finish,
# in which case top the pairs up (results accumulate per pair in the JSON).
GAMES = 6
SIMS = 400          # override on the CLI; the in-training ladder's 64 is a
                    # SEPARATE protocol and must never be pooled with this one
N_WORKERS = 4
SIG = (64, 10, 8)          # both OLIVAW arms; NOT the 32ch legacy library


def participants(arm):
    out = []
    if arm in ('az', 'both'):
        out += [(f'AZ{ep//1000}k', 'othello_benchmark/bench_OLIVAW_AZ',
                 str(ep), 'az', 1.5) for ep in EPS]
    if arm in ('tz', 'both'):
        out += [(f'TZ{ep//1000}k', 'othello_benchmark/bench_OLIVAW_TZ',
                 str(ep), 'thompson', 'additive_mle') for ep in EPS]
    return [p for p in out
            if os.path.exists(os.path.join(p[1], f'bench_{p[2]}.pt'))]


def out_path(arm):
    return f'olivaw_ladder_{arm}_s{SIMS}.json'


def main(arm):
    T.SIG = SIG
    T.SIMS = SIMS
    parts = participants(arm)
    if len(parts) < 2:
        print(f'need >=2 checkpoints for arm {arm!r}, found {len(parts)}')
        return
    out = out_path(arm)
    done = {}
    if os.path.exists(out):
        with open(out) as fh:
            done = json.load(fh).get('pairs', {})
    pairs = [(a, b) for a, b in itertools.combinations(parts, 2)
             if f'{a[0]}|{b[0]}' not in done]
    print(f'[ladder:{arm}] {len(parts)} checkpoints, '
          f'{len(list(itertools.combinations(parts, 2)))} pairs, '
          f'{len(done)} done, {len(pairs)} to play, '
          f'{GAMES} games/pair @ {SIMS} sims', flush=True)
    t0 = time.time()
    for n, (a, b) in enumerate(pairs, 1):
        per = GAMES // N_WORKERS
        rem = GAMES - per * N_WORKERS
        # sig/sims go IN the tuple: spawn workers re-import tournament and
        # would otherwise use its 32ch module default on our 64ch nets.
        jobs = [(a, b, per + (1 if w < rem else 0), 7000 + 131 * w, w % 2 == 0,
                 SIG, SIMS)
                for w in range(N_WORKERS) if per + (1 if w < rem else 0)]
        w = d = l = 0
        with mp.Pool(N_WORKERS) as pool:
            for _key, (pw, pd, pl) in pool.imap_unordered(T._worker, jobs):
                w += pw; d += pd; l += pl
        done[f'{a[0]}|{b[0]}'] = [w, d, l]          # W/D/L, never a points total
        with open(out, 'w') as fh:
            json.dump({'pairs': done, 'sims': SIMS, 'games': GAMES}, fh)
        print(f'[ladder:{arm}] [{n}/{len(pairs)}] {a[0]} vs {b[0]}: '
              f'{w}W-{d}D-{l}L ({time.time()-t0:.0f}s)', flush=True)
    print(f'=== OLIVAW_LADDER_{arm.upper()} DONE ===', flush=True)
    report(arm)


def report(arm):
    path = out_path(arm)
    if not os.path.exists(path):
        print(f'no results yet for arm {arm!r} @ {SIMS} sims ({path}); '
              f'run `python olivaw_ladder.py {arm} {SIMS}` first')
        return
    parts = participants(arm)
    names = [p[0] for p in parts]
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n)); games = np.zeros((n, n))
    wdl = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
    with open(out_path(arm)) as fh:
        done = json.load(fh)['pairs']
    for key, entry in done.items():
        an, bn = key.split('|')
        if an not in idx or bn not in idx:
            continue
        i, j = idx[an], idx[bn]
        w, d, l, g = T._wdl_of(entry)
        wins[i][j] += w + 0.5 * d; games[i][j] += g
        wins[j][i] += l + 0.5 * d; games[j][i] += g
        wdl[i][j] = [wdl[i][j][k] + v for k, v in enumerate((w, d, l))]
        wdl[j][i] = [wdl[j][i][k] + v for k, v in enumerate((l, d, w))]
    T.PARTICIPANTS = parts
    T.report(parts, wins, games, wdl)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    # trailing integer = sims, e.g. `olivaw_ladder.py both 1000`
    if args and args[-1].isdigit():
        SIMS = int(args.pop())
    if args and args[0] == 'report':
        report(args[1] if len(args) > 1 else 'az')
    else:
        main(args[0] if args else 'az')
