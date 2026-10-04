"""Live BT ladder for the CORR arm, anchored on the 10 OLIVAW-TZ checkpoints.

    python corr_live_ladder.py watch     # poll for new CORR benchmarks, rate each
    python corr_live_ladder.py report

Same protocol as olivaw_live_ladder.py -- 12 fixed XOT openings x both seats = 24
games/pair, 400 sims, greedy, one Bradley-Terry fit, W/D/L stored per pair -- so
the numbers are directly comparable to that ladder, and the openings are the same
12 positions.

Every new CORR checkpoint plays all 10 TZ checkpoints plus every earlier CORR one.
Anchoring on the full TZ ladder keeps the fit connected from the first rating, so
CORR ratings sit on the TZ scale rather than in a separate ladder with an
arbitrary zero.  Compare GAPS within one table, never absolute values across
tables: the fit is mean-anchored, so adding checkpoints moves everything.
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import tournament as T

TZ_DIR = 'othello_benchmark/bench_OLIVAW_TZ'
CORR_DIR = 'othello_benchmark/bench_CORR_TZ'
EPS = list(range(5000, 50001, 5000))
N_OPENINGS = 12
GAMES = N_OPENINGS * 2
SIMS = 400
SIG = (64, 10, 8)
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926          # the SAME openings every other ladder used
N_WORKERS = 4
OUT = f'corr_live_ladder_s{SIMS}_g{GAMES}_xot_greedy.json'
LOG = 'corr_live_ladder.log'
HEARTBEAT = 'corr_watch.heartbeat'
POLL_SECONDS = 300
IDLE_GIVE_UP_HOURS = 96.0    # must exceed the gap between checkpoints (~13h)

_OPENINGS = None


def openings():
    global _OPENINGS
    if _OPENINGS is None:
        with open(XOT_FILE) as fh:
            seqs = [ln.strip() for ln in fh if ln.strip()]
        rng = np.random.default_rng(XOT_SEED)
        pick = rng.choice(len(seqs), size=N_OPENINGS, replace=False)
        _OPENINGS = tuple(seqs[i] for i in sorted(pick))
    return _OPENINGS


def beat():
    with open(HEARTBEAT, 'w') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S'))


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def tz_part(ep):
    return (f'TZ{ep//1000}k', TZ_DIR, str(ep), 'thompson', 'additive_mle')


def corr_part(ep):
    # deploys the search it trained with, like every other participant
    return (f'CO{ep//1000}k', CORR_DIR, str(ep), 'thompson', 'corr')


def _exists(p):
    return os.path.exists(os.path.join(p[1], f'bench_{p[2]}.pt'))


def available(which):
    mk = tz_part if which == 'tz' else corr_part
    return [p for p in (mk(ep) for ep in EPS) if _exists(p)]


# The TZ-vs-TZ pairs already exist in the OLIVAW ladder, played on the identical
# protocol with the same 12 XOT openings, so they are directly poolable.  Without
# them this fit has no TZ-TZ edges at all and the anchor ordering is an artifact
# of each anchor's record against the one CORR checkpoint -- observed: three TZ
# checkpoints tied at 1857 on 24-0-0 records.
SEED_FROM = 'olivaw_live_ladder_s400_g24_xot_greedy.json'


def _seed_tz_pairs():
    if not os.path.exists(SEED_FROM):
        return {}
    with open(SEED_FROM) as fh:
        src = json.load(fh).get('pairs', {})
    out = {}
    for k, v in src.items():
        a, b = k.split('|')
        if a.startswith('TZ') and b.startswith('TZ'):
            out[k] = v
    return out


def load_done():
    done = dict(_seed_tz_pairs())
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done.update(json.load(fh).get('pairs', {}))
    return done


def save_done(done):
    # persist only pairs involving a CORR checkpoint; the TZ-TZ edges are seeded
    # from the OLIVAW ladder each time and must not be duplicated here
    mine = {k: v for k, v in done.items() if 'CO' in k}
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'pairs': mine, 'sims': SIMS, 'games': GAMES}, fh)
    os.replace(tmp, OUT)


def pair_key(a, b):
    return '|'.join(sorted((a[0], b[0])))


def play_pair(a, b):
    ops = openings()
    shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
    jobs = [(a, b, 0, 7000 + 131 * w, True, SIG, SIMS, sh)
            for w, sh in enumerate(shards) if sh]
    w = d = l = 0
    with mp.Pool(len(jobs)) as pool:
        for (an, _bn), (pw, pd, pl) in pool.imap_unordered(T._worker, jobs):
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
        beat()
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
    parts = available('tz') + available('corr')
    if len(parts) < 2 or not done:
        print(f'nothing to report yet ({len(parts)} checkpoints, '
              f'{len(done)} pairs in {OUT})')
        return
    names, wins, games, wdl = fit(done, parts)
    bt = T.fit_elo(names, wins, games, reg=1.6e5)
    order = np.argsort(-np.asarray(bt))
    n = len(names)
    tot, tg = wins.sum(1), games.sum(1)
    tw = [sum(wdl[i][j][0] for j in range(n)) for i in range(n)]
    td = [sum(wdl[i][j][1] for j in range(n)) for i in range(n)]
    tl = [sum(wdl[i][j][2] for j in range(n)) for i in range(n)]
    print(f'\n  BT ladder @{SIMS} sims, {GAMES} games/pair, {len(done)} pairs')
    print(f'  {"rank":<5}{"ckpt":<8}{"arm":<6}{"BT":>6}{"W-D-L":>14}{"score%":>9}')
    for rank, i in enumerate(order, 1):
        arm = 'CORR' if names[i].startswith('CO') else 'TZ'
        rec = f'{tw[i]:.0f}-{td[i]:.0f}-{tl[i]:.0f}'
        print(f'  {rank:<5}{names[i]:<8}{arm:<6}{bt[i]:>6.0f}{rec:>14}'
              f'{tot[i] / max(tg[i], 1):>9.1%}')
    for arm, pref in (('TZ', 'TZ'), ('CORR', 'CO')):
        v = [bt[i] for i, nm in enumerate(names) if nm.startswith(pref)]
        if v:
            print(f'  {arm}: n={len(v)} best={max(v):.0f} median={np.median(v):.0f}')


def watch():
    tz = available('tz')
    if len(tz) < 2:
        _log(f'TZ anchors not ready ({len(tz)} found in {TZ_DIR})')
        return
    done = load_done()
    _log(f'=== watch: {len(tz)} TZ anchors, polling {CORR_DIR} every '
         f'{POLL_SECONDS}s ({len(done)} pairs already played) ===')
    rated = {k for key in done for k in key.split('|') if k.startswith('CO')}
    idle = 0.0
    while True:
        beat()
        co = available('corr')
        new = [p for p in co if p[0] not in rated]
        if new:
            idle = 0.0
            for p in new:
                earlier = [q for q in co if q[0] in rated]
                pairs = [(p, o) for o in tz + earlier]
                _log(f'--- rating {p[0]}: {len(pairs)} pairs '
                     f'({len(tz)} TZ + {len(earlier)} earlier CORR) ---')
                run_pairs(pairs, done, p[0])
                rated.add(p[0])
                report()
                if len(co) >= len(EPS) and p[0] == co[-1][0]:
                    _log('=== all CORR checkpoints rated; WATCH DONE ===')
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
    if cmd == 'watch':
        watch()
    else:
        report()
