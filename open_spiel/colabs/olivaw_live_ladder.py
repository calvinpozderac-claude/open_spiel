"""Live cross-arm Bradley-Terry ladder for the OLIVAW runs.

    python olivaw_live_ladder.py seed        # AZ-AZ round robin (run once, after
                                             # the AZ arm finishes)
    python olivaw_live_ladder.py watch       # poll for new TZ benchmarks, play
                                             # each vs ALL AZ + earlier TZ, refit
    python olivaw_live_ladder.py report      # print the current BT table

WHAT THIS REPLACES. Both arms used to rate themselves with an in-training
incremental Elo ladder at 64 sims. That was wrong twice over: the incremental
update is order- and path-dependent (replaying identical results in a different
order moved it 40.2 Elo in a controlled test, which is the entire explanation
for OLIVAW-AZ's apparent ep35k/ep45k "regressions"), and a 64-sim rating cannot
be pooled with a 400-sim one because BT assumes ONE strength per player -- and
this project has measured that search budget can invert rankings outright.

So the TZ arm runs with deep_eval_elo=False and this process does the rating
instead, on exactly the protocol the final tournament uses.

THE DESIGN. Every new TZ checkpoint plays all 10 AZ checkpoints plus every
earlier TZ checkpoint. That keeps the comparison graph CONNECTED from the very
first update -- TZ5k is anchored to the whole AZ block immediately, so its
rating is on the same scale as AZ's rather than floating in a separate ladder
with an arbitrary zero.

The work also lands exactly once:
    45  AZ-AZ pairs      (seed)
  + 145  TZ pairs        (10 AZ + k-1 earlier TZ, summed over k = 1..10)
  = 190  = the complete 20-player round robin, nothing replayed.

Measured 97s per pair (6 games @400 sims, 4 workers, under training load), so
the seed is ~73 min and the live phase ~3.9h spread across a ~60h TZ run.

Results accumulate per pair as [W, D, L] in one JSON, so a kill costs at most
the in-flight pair, and the pairs can be topped up later if a result is close.
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import tournament as T

AZ_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
TZ_DIR = 'othello_benchmark/bench_OLIVAW_TZ'
EPS = list(range(5000, 50001, 5000))
GAMES = 6
SIMS = 400
SIG = (64, 10, 8)
N_WORKERS = 4
OUT = f'olivaw_live_ladder_s{SIMS}.json'
LOG = 'olivaw_live_ladder.log'
POLL_SECONDS = 300
IDLE_GIVE_UP_HOURS = 12.0


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def az_part(ep):
    return (f'AZ{ep//1000}k', AZ_DIR, str(ep), 'az', 1.5)


def tz_part(ep):
    return (f'TZ{ep//1000}k', TZ_DIR, str(ep), 'thompson', 'additive_mle')


def _exists(p):
    return os.path.exists(os.path.join(p[1], f'bench_{p[2]}.pt'))


def available(which):
    mk = az_part if which == 'az' else tz_part
    return [p for p in (mk(ep) for ep in EPS) if _exists(p)]


def load_done():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh).get('pairs', {})
    return {}


def save_done(done):
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'pairs': done, 'sims': SIMS, 'games': GAMES}, fh)
    os.replace(tmp, OUT)


def pair_key(a, b):
    """Order-independent key, so a pair is never played twice under two names."""
    return '|'.join(sorted((a[0], b[0])))


def play_pair(a, b):
    per = GAMES // N_WORKERS
    rem = GAMES - per * N_WORKERS
    # sig/sims ride IN the tuple: spawn workers re-import tournament and would
    # otherwise use its 32ch module default against our 64ch nets.
    jobs = [(a, b, per + (1 if w < rem else 0), 7000 + 131 * w, w % 2 == 0,
             SIG, SIMS)
            for w in range(N_WORKERS) if per + (1 if w < rem else 0)]
    w = d = l = 0
    with mp.Pool(N_WORKERS) as pool:
        for (an, _bn), (pw, pd, pl) in pool.imap_unordered(T._worker, jobs):
            # _worker reports from ITS first player's view; normalise to `a`.
            if an == a[0]:
                w += pw; d += pd; l += pl
            else:
                w += pl; d += pd; l += pw
    return w, d, l


def run_pairs(pairs, done, tag):
    t0 = time.time()
    for i, (a, b) in enumerate(pairs, 1):
        k = pair_key(a, b)
        if k in done:
            continue
        w, d, l = play_pair(a, b)
        # stored under the sorted key, from the FIRST name's perspective
        first = sorted((a[0], b[0]))[0]
        done[k] = [w, d, l] if first == a[0] else [l, d, w]
        save_done(done)
        _log(f'[{tag} {i}/{len(pairs)}] {a[0]} vs {b[0]}: {w}W-{d}D-{l}L '
             f'({time.time()-t0:.0f}s)')
    return done


def fit(done, parts):
    names = [p[0] for p in parts]
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n)); games = np.zeros((n, n))
    wdl = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
    for key, entry in done.items():
        an, bn = key.split('|')
        if an not in idx or bn not in idx:
            continue
        i, j = idx[an], idx[bn]
        w, d, l, g = T._wdl_of(entry)
        wins[i][j] += w + 0.5 * d; games[i][j] += g
        wins[j][i] += l + 0.5 * d; games[j][i] += g
        wdl[i][j] = [wdl[i][j][m] + v for m, v in enumerate((w, d, l))]
        wdl[j][i] = [wdl[j][i][m] + v for m, v in enumerate((l, d, w))]
    return names, wins, games, wdl


def report():
    done = load_done()
    parts = available('az') + available('tz')
    if len(parts) < 2 or not done:
        print(f'nothing to report yet ({len(parts)} checkpoints, '
              f'{len(done)} pairs in {OUT})')
        return
    names, wins, games, wdl = fit(done, parts)
    bt = T.fit_elo(names, wins, games, reg=1.6e5)
    order = np.argsort(-np.asarray(bt))
    n = len(names)
    tot = wins.sum(1); tg = games.sum(1)
    tw = [sum(wdl[i][j][0] for j in range(n)) for i in range(n)]
    td = [sum(wdl[i][j][1] for j in range(n)) for i in range(n)]
    tl = [sum(wdl[i][j][2] for j in range(n)) for i in range(n)]
    print(f'\n  BT ladder @{SIMS} sims, {GAMES} games/pair, '
          f'{len(done)} pairs played')
    print(f'  {"rank":<5}{"ckpt":<9}{"arm":<5}{"BT":>6}{"W-D-L":>14}{"score%":>9}')
    for rank, i in enumerate(order, 1):
        arm = 'AZ' if names[i].startswith('AZ') else 'TZ'
        rec = f'{tw[i]:.0f}-{td[i]:.0f}-{tl[i]:.0f}'
        sc = tot[i] / max(tg[i], 1)
        print(f'  {rank:<5}{names[i]:<9}{arm:<5}{bt[i]:>6.0f}{rec:>14}'
              f'{sc:>9.1%}')
    # arm-level summary: the actual question these runs are asking
    for arm in ('AZ', 'TZ'):
        vals = [bt[i] for i, nm in enumerate(names) if nm.startswith(arm)]
        if vals:
            print(f'  {arm}: n={len(vals)} best={max(vals):.0f} '
                  f'median={np.median(vals):.0f}')


def seed():
    az = available('az')
    if len(az) < 2:
        _log(f'AZ arm not ready ({len(az)} checkpoints found); nothing to seed')
        return
    done = load_done()
    pairs = list(itertools.combinations(az, 2))
    _log(f'=== seed: {len(az)} AZ checkpoints, {len(pairs)} pairs, '
         f'{GAMES} games/pair @ {SIMS} sims ===')
    run_pairs(pairs, done, 'seed')
    _log('=== SEED DONE ===')
    report()


def watch():
    az = available('az')
    if len(az) < 2:
        _log('AZ arm not ready; run `seed` after it finishes')
        return
    done = load_done()
    _log(f'=== watch: {len(az)} AZ anchors, polling {TZ_DIR} every '
         f'{POLL_SECONDS}s ({len(done)} pairs already played) ===')
    rated = set()
    for key in done:
        rated.update(k for k in key.split('|') if k.startswith('TZ'))
    idle = 0.0
    while True:
        tz = available('tz')
        new = [p for p in tz if p[0] not in rated]
        if new:
            idle = 0.0
            for p in new:
                earlier = [q for q in tz if q[0] in rated]
                pairs = [(p, o) for o in az + earlier]
                _log(f'--- rating {p[0]}: {len(pairs)} pairs '
                     f'({len(az)} AZ + {len(earlier)} earlier TZ) ---')
                run_pairs(pairs, done, p[0])
                rated.add(p[0])
                report()
                if len(tz) >= len(EPS) and p[0] == tz[-1][0]:
                    _log('=== all TZ checkpoints rated; WATCH DONE ===')
                    return
        else:
            time.sleep(POLL_SECONDS)
            idle += POLL_SECONDS / 3600.0
            if idle >= IDLE_GIVE_UP_HOURS:
                _log(f'=== no new checkpoint in {IDLE_GIVE_UP_HOURS}h; '
                     f'WATCH STOPPING ===')
                return


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'seed':
        seed()
    elif cmd == 'watch':
        watch()
    else:
        report()
