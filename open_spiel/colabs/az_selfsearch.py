"""Does MORE SEARCH beat LESS SEARCH, with the network held identical?

    python az_selfsearch.py [report]

THE CONCERN. AZ50k scored 2.1% / 2.1% / 0.0% against Edax L4 at 1000 / 2500 /
10000 sims. A weak evaluator explains a LOW score, but it does not obviously
explain a FLAT one: deeper search should still help, and the MCTS-Solver means
enough nodes gives perfect endgame play regardless of the network. Flat-at-zero
is consistent with "search is doing nothing at all", which would be a bug.

THE TEST. Same checkpoint on both sides, differing ONLY in simulation count.
Any asymmetry is attributable to search alone -- no network difference, no
opponent-strength confound, and the Edax matchup (where we may simply be far
past the point of competitiveness, so every budget loses) is out of the picture.

  more sims ~ 50%  -> search contributes nothing; investigate the search path
  more sims >> 50% -> search works, and the flat Edax result means our whole
                      checkpoint series is simply below Edax L4's floor, where
                      any budget loses ~everything

Deliberately small: this is a diagnostic, not a measurement.
"""
import json
import os
import sys
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
EP = 50000
SIG = (64, 10, 8)
C_PUCT = 1.5
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926
N_WORKERS = 2
OUT = 'az_selfsearch.json'
LOG = 'az_selfsearch.log'

# (low_sims, high_sims, n_openings); each opening played from BOTH seats
PLAN = [(100, 1000, 6), (1000, 10000, 4)]


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def openings(n):
    with open(XOT_FILE) as fh:
        seqs = [ln.strip() for ln in fh if ln.strip()]
    rng = np.random.default_rng(XOT_SEED)
    pick = rng.choice(len(seqs), size=12, replace=False)
    return [seqs[i] for i in sorted(pick)][:n]


def _worker(args):
    ep, lo, hi, ops, seed = args
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import connect4_alphazero_utils as az
    import edax_bot as eb
    game = c4.load_game('othello')
    c4.set_game(game)
    az.set_search(c_puct=C_PUCT, virtual_loss=1.0)
    amap = eb._action_map(game)
    net = az.load_benchmark_net(CKPT_DIR, str(ep), SIG)
    rng = np.random.default_rng(seed)
    hw = hd = hl = 0
    for seq in ops:
        for hi_seat in (0, 1):          # the HIGH-sims side takes each colour
            st = game.new_initial_state()
            for k in range(0, 16, 2):
                st.apply_action(amap[seq[k:k + 2]])
            while not st.is_terminal():
                sims = hi if st.current_player() == hi_seat else lo
                bot = az.AZMCTSBot(game, net, 'cpu', sims, batch_size=16,
                                   random_state=rng, noise_frac=0.0)
                a = az.root_pick(bot.mcts_search(st), rng, sample=False)
                st.apply_action(int(a))
            r = st.returns()[hi_seat]
            if r > 0:
                hw += 1
            elif r < 0:
                hl += 1
            else:
                hd += 1
    return hw, hd, hl


def main():
    import multiprocessing as mp
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    _log(f'=== az_selfsearch: AZ{EP//1000}k vs ITSELF, '
         f'{[(a, b) for a, b, _ in PLAN]} sims ===')
    t0 = time.time()
    for lo, hi, n_ops in PLAN:
        key = f'{lo}v{hi}'
        if key in done:
            continue
        ops = openings(n_ops)
        shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
        jobs = [(EP, lo, hi, sh, 6600 + 71 * k)
                for k, sh in enumerate(shards) if sh]
        tc = time.time()
        with mp.Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        hw = sum(p[0] for p in parts)
        hd = sum(p[1] for p in parts)
        hl = sum(p[2] for p in parts)
        n = hw + hd + hl
        sc = (hw + 0.5 * hd) / n
        done[key] = dict(hw=hw, hd=hd, hl=hl, n=n, score=sc)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{lo} vs {hi} sims] HIGH side: {hw}W-{hd}D-{hl}L = {sc:.1%} '
             f'of {n} games ({time.time()-tc:.0f}s)')
    _log('=== AZ_SELFSEARCH DONE ===')
    report()


def report():
    if not os.path.exists(OUT):
        print('no results yet')
        return
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'\n  AZ{EP//1000}k vs ITSELF -- identical network, sims differ only')
    print(f'  {"matchup":>16}{"high-sims W-D-L":>18}{"games":>7}'
          f'{"high score":>12}')
    for lo, hi, _ in PLAN:
        c = done.get(f'{lo}v{hi}')
        if c:
            print(f'  {f"{lo} vs {hi}":>16}'
                  f'{f"{c["hw"]}-{c["hd"]}-{c["hl"]}":>18}{c["n"]:>7}'
                  f'{c["score"]:>12.1%}')
    print('\n  ~50% => search contributes nothing (suspect the search path).')
    print('  >>50% => search works; the flat Edax result means the whole')
    print('           checkpoint series sits below Edax L4\'s floor.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
