"""STUDY 1 — which evidence rule should drive SEARCH?

For net x budget x scheme, one MCTS per labelled position (edax_eval_set),
recording play-strength proxies and root-belief calibration:

    move_match   argmax collapsed edge value == Edax best move
    sign_ok      sign(root v) == sign(truth), decisive positions only
    val_err      |root v - truth|
    a0           root state concentration under the scheme
    ell          E_{p~Dir(root alpha)}[log p_true]  (proper score: confident-
                 wrong punished, confident-right rewarded)

Schemes: the incumbent ('additive_mle'), the backup-study target winner
('avgconc'), the original ('mixture'), and the NEW path-attenuated rule
('pathsum': concentration = sum over backups of min(leaf a0, selection-belief
a0 of every edge below the node) -- a confident leaf reached through
exploratory moves backs up as low-confidence evidence).

Nets span loss regimes AND maturity: HYBRID@50k (KL-trained, the incumbent
lineage), HYBRID@10k (young), EDL@50k (evidential-trained, whose confidences
are grounded -- the interaction that matters for a confidence-driven backup).

    python search_study.py [n_per_phase]      # run (default 60)
    python search_study.py report             # print ranked tables
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
OUT = 'search_study.json'
SIG = (32, 10, 8)
NETS = [
    ('HYBRID@50k', 'othello_benchmark/bench_HYBRID', '50000'),
    ('HYBRID@10k', 'othello_benchmark/bench_HYBRID', '10000'),
    ('EDL@50k',    'othello_benchmark/bench_EDL',    '50000'),
]
BUDGETS = [64, 256, 1000]
SCHEMES = ['additive_mle', 'avgconc', 'mixture', 'pathsum']
PHASES = [(45, 61, 'opening'), (33, 45, 'early-mid'), (25, 33, 'late-mid'),
          (15, 25, 'early-end'), (8, 15, 'endgame'), (0, 8, 'final')]
OUR_TEMP = 64.0
N_WORKERS = 4


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    return c4, game


def _worker(args):
    net_name, ckpt_dir, label, budget, scheme, recs = args
    from scipy.special import digamma
    c4, game = _init()
    net = c4.load_benchmark_net(ckpt_dir, label, SIG)
    # target_agg pathavg alongside pathsum keeps slot-8 maintenance on for the
    # path pass; for the others it matches the scheme like backup_study did.
    tgt = 'pathavg' if scheme == 'pathsum' else scheme
    c4.set_search(search_agg=scheme, target_agg=tgt,
                  selection='dirichlet', virtual_loss=1.0)
    rng = np.random.default_rng(11)

    out = []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        bot = c4.C4MCTSBot(game, net, 'cpu', budget, batch_size=16,
                           temp=OUR_TEMP, random_state=rng)
        root = bot.mcts_search(st)
        ra = c4.observed_alpha(root.nacc, scheme)
        rv = float(ra[c4._WIN] - ra[c4._LOSS]) / max(float(np.sum(ra)), 1e-9)
        a0 = float(np.sum(ra))
        ev = (root.visits() > 0) | (root.term >= 0)
        idx = np.nonzero(ev)[0]
        best_mv = -1
        if len(idx):
            vals = []
            for i in idx:
                ea = c4.observed_alpha(root.eacc[int(i)], scheme)
                vals.append(float(c4.dir_value(ea)) if ea is not None else -9.9)
            best_mv = int(root.legal[int(idx[int(np.argmax(vals))])])
        truth = r['value']
        ti = c4._WIN if truth > 0 else (c4._LOSS if truth < 0 else c4._DRAW)
        ell = float(digamma(max(ra[ti], 1e-9)) - digamma(max(a0, 1e-9)))
        out.append(dict(empties=r['empties'], tier=r['tier'],
                        move_match=int(best_mv == r['best']),
                        decisive=int(truth != 0),
                        sign_ok=int((rv > 0) == (truth > 0)) if truth != 0 else -1,
                        val_err=abs(rv - truth), a0=a0, ell=ell))
    return (net_name, budget, scheme), out


def run(npp):
    with open(CACHE) as fh:
        allrecs = json.load(fh)['positions']
    recs = []
    for lo, hi, _ in PHASES:
        pool = [r for r in allrecs if lo <= r['empties'] < hi]
        recs.extend(pool[:npp])
    jobs = [(nm, d, lab, b, s, recs)
            for (nm, d, lab) in NETS for b in BUDGETS for s in SCHEMES]
    print(f'[s1] {len(recs)} positions x {len(jobs)} passes', flush=True)
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    jobs = [j for j in jobs if '|'.join(map(str, j[:1] + j[3:5])) not in done]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for (key, rows) in pool.imap_unordered(_worker, jobs):
            done['|'.join(map(str, key))] = rows
            with open(OUT, 'w') as fh:
                json.dump(done, fh)
            print(f'[s1] {key} done ({time.time()-t0:.0f}s, '
                  f'{len(done)}/{len(jobs)})', flush=True)
    print('[s1] SEARCH STUDY DONE', flush=True)


def report():
    from scipy.stats import spearmanr
    with open(OUT) as fh:
        done = json.load(fh)
    print(f'{"net":<12}{"sims":>6}{"scheme":<15}{"move%":>7}{"sign%":>7}'
          f'{"val_err":>9}{"a0(med)":>9}{"ell":>8}{"rho(a0,-err)":>13}')
    for nm, _, _ in NETS:
        for b in BUDGETS:
            for s in SCHEMES:
                rows = done.get(f'{nm}|{b}|{s}')
                if not rows:
                    continue
                mm = np.mean([r['move_match'] for r in rows])
                dec = [r for r in rows if r['decisive']]
                sg = np.mean([r['sign_ok'] for r in dec]) if dec else float('nan')
                ve = np.mean([r['val_err'] for r in rows])
                a0 = np.median([r['a0'] for r in rows])
                ell = np.mean([r['ell'] for r in rows])
                rho = spearmanr([r['a0'] for r in rows],
                                [-r['val_err'] for r in rows]).statistic
                print(f'{nm:<12}{b:>6}{s:<15}{mm:>7.1%}{sg:>7.1%}'
                      f'{ve:>9.3f}{a0:>9.2f}{ell:>8.3f}{rho:>13.3f}')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
