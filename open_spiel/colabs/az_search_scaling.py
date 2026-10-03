"""Does OLIVAW-AZ's weakness come from its NETWORK or from its SEARCH?

    python az_search_scaling.py [report]

THE DISCRIMINATOR. AZ50k scores 1 draw in 216 games against Edax L4/L6/L8 at
1000 sims, where OLIVAW reports O-1000 beating E4 and E6 at the same node
budget. Two very different explanations survive:

  (a) the NETWORK is too weak (we use 64ch/808k params; if OLIVAW really used
      AlphaGo Zero's block width of 256 filters that is 12.3M, ~15x ours), or
  (b) our SEARCH is worth far less per "sim" than their per "node" -- we
      evaluate leaves in batches of 16 under virtual loss, which explores a
      shallower, more redundant tree than 1000 sequential simulations would.

Scaling search on a FIXED network separates them. If 10x the sims barely moves
the score, the evaluation function is the ceiling and (a) is the answer. If it
climbs steeply, each of our sims is cheap relative to theirs and (b) is live.

The 10000-sim cell uses half the openings: it costs ~10x per game and the
question there is "does it move at all", which 12 games answers.
"""
import json
import os
import sys
import time

import numpy as np

CKPT_DIR = 'othello_benchmark/bench_OLIVAW_AZ'
EP = 50000
LEVEL = 4
SIG = (64, 10, 8)
C_PUCT = 1.5
XOT_FILE = 'xot_large.txt'
XOT_SEED = 20260926
N_WORKERS = 2
OUT = 'az_search_scaling.json'
LOG = 'az_search_scaling.log'

# (sims, n_openings) -- each opening is played from BOTH seats
PLAN = [(2500, 12), (10000, 6)]


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
    ops = [seqs[i] for i in sorted(pick)]
    return ops[:n]          # a PREFIX of the same set, so cells stay comparable


def _worker(args):
    ep, level, sims, ops, seed = args
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
                    bot = az.AZMCTSBot(game, net, 'cpu', sims, batch_size=16,
                                       random_state=rng, noise_frac=0.0)
                    a = az.root_pick(bot.mcts_search(st), rng, sample=False)
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


def main():
    import multiprocessing as mp
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    _log(f'=== az_search_scaling: AZ{EP//1000}k vs EdaxL{LEVEL}, '
         f'sims {[s for s, _ in PLAN]} ===')
    t0 = time.time()
    for sims, n_ops in PLAN:
        key = str(sims)
        if key in done:
            continue
        ops = openings(n_ops)
        shards = [ops[i::N_WORKERS] for i in range(N_WORKERS)]
        jobs = [(EP, LEVEL, sims, sh, 5500 + 89 * k)
                for k, sh in enumerate(shards) if sh]
        tc = time.time()
        with mp.Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        w = sum(p[0] for p in parts)
        d = sum(p[1] for p in parts)
        l = sum(p[2] for p in parts)
        n = w + d + l
        done[key] = dict(w=w, d=d, l=l, n=n, score=(w + 0.5 * d) / n)
        with open(OUT, 'w') as fh:
            json.dump(done, fh)
        _log(f'[{sims} sims] AZ{EP//1000}k vs EdaxL{LEVEL}: {w}W-{d}D-{l}L '
             f'= {(w + 0.5*d)/n:.1%} of {n} games '
             f'({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s)')
    _log('=== AZ_SEARCH_SCALING DONE ===')
    report()


def report():
    print(f'\n  AZ{EP//1000}k vs Edax L{LEVEL}, search scaling on a FIXED net')
    print(f'  {"sims":>8}{"W-D-L":>12}{"games":>7}{"score":>9}')
    print(f'  {1000:>8}{"0-1-23":>12}{24:>7}{0.021:>9.1%}   (from az_edax_bench)')
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
        for sims, _ in PLAN:
            c = done.get(str(sims))
            if c:
                print(f'  {sims:>8}{f"{c["w"]}-{c["d"]}-{c["l"]}":>12}'
                      f'{c["n"]:>7}{c["score"]:>9.1%}')
    print('\n  Flat => the NETWORK is the ceiling (capacity).')
    print('  Steep => our sims are worth less than OLIVAW\'s nodes (search).')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
