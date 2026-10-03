"""Where does either arm's real strength actually start? Auto-escalating probe.

    python edax_floor_probe.py [report]

Both arms are ~0% against Edax L4/L6/L8 at 1000 sims (az_edax_bench.py,
tz_edax_bench.py) -- a floor with no visible bottom from above. This probes
DOWNWARD from the weak end to find where either net first shows real skill:
Edax L2 first; if BOTH best checkpoints sweep it 24-0, automatically continue
to L3 (Edax has no intermediate rung between depth-limited levels, so L2/L3
are the finest resolution available). Stops as soon as a cell is not a clean
sweep for both -- that is the informative result and further escalation would
just be re-measuring 0%/100% at the wrong end.

Uses each arm's CURRENT BEST checkpoint by BT rating (AZ45k, TZ30k as of this
run) rather than the final one -- the point is "can this recipe produce any
real strength at all", which the best checkpoint answers most charitably.

Same protocol as az_edax_bench.py / tz_edax_bench.py: 1000 sims, 12 XOT
openings x 2 seats = 24 games/cell, greedy play, same opening seed so every
cell in this whole investigation is directly comparable.
"""
import json
import os
import sys
import time

import numpy as np

AZ_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
TZ_DIR = 'othello_benchmark/bench_OLIVAW_TZ'
AZ_BEST = 45000          # highest BT as of this run; see olivaw_live_ladder
TZ_BEST = 30000          # latest completed checkpoint (still climbing)
SIG = (64, 10, 8)
N_OPENINGS = 12
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926       # same openings as every other Edax cell this session
N_WORKERS = 3
OUT = 'edax_floor_probe.json'
LOG = 'edax_floor_probe.log'
LEVEL_SEQUENCE = [2, 3]   # escalate only while every arm sweeps 24-0


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def openings():
    with open(XOT_FILE) as fh:
        seqs = [ln.strip() for ln in fh if ln.strip()]
    rng = np.random.default_rng(XOT_SEED)
    pick = rng.choice(len(seqs), size=N_OPENINGS, replace=False)
    return [seqs[i] for i in sorted(pick)]


def _worker(args):
    arm, ckpt_dir, ep, level, ops, seed = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import edax_bot as eb
    game = c4.load_game('othello')
    c4.set_game(game)
    amap = eb._action_map(game)
    if arm == 'tz':
        c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                      selection='dirichlet', virtual_loss=1.0)
        net = c4.load_benchmark_net(ckpt_dir, str(ep), SIG)
    else:
        import connect4_alphazero_utils as az
        az.set_search(c_puct=1.5, virtual_loss=1.0)
        net = az.load_benchmark_net(ckpt_dir, str(ep), SIG)
    opp = eb.EdaxBot(game, level=level)
    rng = np.random.default_rng(seed)
    w = d = l = 0
    for seq in ops:
        for seat in (0, 1):
            st = game.new_initial_state()
            for k in range(0, 16, 2):
                st.apply_action(amap[seq[k:k + 2]])
            while not st.is_terminal():
                if st.current_player() == seat:
                    if arm == 'tz':
                        bot = c4.C4MCTSBot(game, net, 'cpu', 1000,
                                           batch_size=16, temp=64.0,
                                           random_state=rng)
                        a = c4.root_pick(bot.mcts_search(st), rng,
                                         thompson=False, temp=64.0)
                    else:
                        bot = az.AZMCTSBot(game, net, 'cpu', 1000,
                                           batch_size=16, random_state=rng,
                                           noise_frac=0.0)
                        a = az.root_pick(bot.mcts_search(st), rng,
                                         sample=False)
                else:
                    a = opp.step(st)
                st.apply_action(int(a))
            r = st.returns()[seat]
            if r > 0:
                w += 1
            elif r < 0:
                l += 1
            else:
                d += 1
    opp.close()
    return w, d, l


def run_cell(arm, ckpt_dir, ep, level, ops, out_workers):
    shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
    jobs = [(arm, ckpt_dir, ep, level, sh, 2200 + 43 * k)
            for k, sh in enumerate(shards) if sh]
    import multiprocessing as mp
    with mp.Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    w = sum(p[0] for p in parts)
    d = sum(p[1] for p in parts)
    l = sum(p[2] for p in parts)
    return w, d, l


def main():
    ops = openings()
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    arms = [('az', AZ_DIR, AZ_BEST), ('tz', TZ_DIR, TZ_BEST)]
    _log(f'=== edax_floor_probe: AZ{AZ_BEST//1000}k + TZ{TZ_BEST//1000}k, '
         f'auto-escalating levels {LEVEL_SEQUENCE} ===')
    t0 = time.time()
    for level in LEVEL_SEQUENCE:
        cell_results = {}
        for arm, ckpt_dir, ep in arms:
            key = f'{arm}{ep}|L{level}'
            if key in done:
                cell_results[arm] = done[key]
                continue
            tc = time.time()
            w, d, l = run_cell(arm, ckpt_dir, ep, level, ops, N_WORKERS)
            n = w + d + l
            score = (w + 0.5 * d) / n
            done[key] = dict(w=w, d=d, l=l, n=n, score=score)
            with open(OUT, 'w') as fh:
                json.dump(done, fh)
            cell_results[arm] = done[key]
            _log(f'[L{level}] {arm.upper()}{ep//1000}k vs EdaxL{level}: '
                 f'{w}W-{d}D-{l}L = {score:.1%} ({time.time()-tc:.0f}s, '
                 f'total {time.time()-t0:.0f}s)')
        both_swept = all(c['w'] == c['n'] for c in cell_results.values())
        if not both_swept:
            _log(f'=== L{level}: not a clean sweep for both -- stopping '
                 f'escalation here ===')
            break
        _log(f'=== L{level}: BOTH arms swept 24-0 -- escalating ===')
    else:
        _log('=== reached the top of LEVEL_SEQUENCE while still sweeping '
             '-- extend LEVEL_SEQUENCE to keep probing ===')
    _log('=== EDAX_FLOOR_PROBE DONE ===')
    report()


def report():
    if not os.path.exists(OUT):
        print('no results yet')
        return
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'\n  Edax floor probe -- best checkpoint per arm, 1000 sims, '
          f'XOT, {N_OPENINGS*2} games/cell')
    print(f'  {"arm/ckpt":<12}' + ''.join(f'{f"EdaxL{lv}":>18}'
                                          for lv in LEVEL_SEQUENCE))
    for arm, _, ep in [('az', AZ_DIR, AZ_BEST), ('tz', TZ_DIR, TZ_BEST)]:
        row = f'  {arm.upper()}{ep//1000}k'.ljust(14)
        for lv in LEVEL_SEQUENCE:
            c = done.get(f'{arm}{ep}|L{lv}')
            row += (f'{c["w"]}-{c["d"]}-{c["l"]} {c["score"]:>5.0%}'.rjust(18)
                    if c else f'{"--":>18}')
        print(row)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
