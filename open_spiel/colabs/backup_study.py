"""Which backup scheme is best for SEARCH (finds Edax's move) and which is best
for a TARGET (its concentration tracks closeness to truth)?

For each net (MM_long, AVGCONC) x sim budget x backup scheme, run one MCTS
search per position using that scheme as search_agg, then record against the
Edax-verified ground truth in edax_eval_set.json:

  SEARCH quality
    move_match   argmax edge value == Edax's best move
    sign_ok      sign(root E[win]-E[loss]) == sign(true outcome)  (decisive only)
    val_err      |root value - true value|

  TARGET quality (calibration)
    root_a0      the scheme's collapsed root concentration
    -- aggregated as: does HIGH a0 predict being RIGHT?  A good target scheme
       has high Spearman(a0, sign_ok) and high Spearman(a0, -val_err): confident
       when correct, unsure when not.  A scheme whose a0 is uncorrelated with
       correctness (or just tracks the budget) is a poor target.

Ground truth is exact (Edax solved) for the low-empties tier and strong-play
outcome for the midgame tier -- the two are flagged so the report can separate
them.

    python backup_study.py [n_positions_per_phase]
"""
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
OUT = 'backup_study2.json'
# (label_name, dir, ckpt, is_mm_arch) across TRAINING MATURITY: untrained,
# mid (10k), full (50k), for both the additive_mle-trained (MM_long) and the
# avgconc-trained (AVGCONC) lineages.
NETS = [
    ('MM_long@0',     'othello_benchmark/bench_MM_long', '0',     True),
    ('MM_long@10k',   'othello_benchmark/bench_MM_long', '10000', True),
    ('MM_long@50k',   'othello_benchmark/bench_MM_long', '50000', True),
    ('AVGCONC@0',     'othello_benchmark/bench_AVGCONC', '0',     False),
    ('AVGCONC@10k',   'othello_benchmark/bench_AVGCONC', '10000', False),
    ('AVGCONC@50k',   'othello_benchmark/bench_AVGCONC', '50000', False),
]
BUDGETS = [64, 256, 1000]
SCHEMES = ['additive_mle', 'sum', 'mixture', 'avgconc']
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
    c4.set_search(selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _sig(c4, d, label):
    import torch
    sd = torch.load(f'{d}/bench_{label}.pt', map_location='cpu', weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def _worker(args):
    net_name, ckpt_dir, label, is_mm, budget, scheme, recs = args
    from scipy.special import digamma
    c4, game = _init()
    import adaptive_stop_utils as ad
    sig = ad.MM_SIG if is_mm else _sig(c4, ckpt_dir, label)
    net = c4.load_benchmark_net(ckpt_dir, label, sig)
    c4.set_search(search_agg=scheme, target_agg=scheme,
                  selection='dirichlet', virtual_loss=1.0)
    rng = np.random.default_rng(11)

    out = []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        legal = [int(x) for x in st.legal_actions()]
        bot = c4.C4MCTSBot(game, net, 'cpu', budget, batch_size=16,
                           temp=OUR_TEMP, random_state=rng)
        root = bot.mcts_search(st)
        # root value in the mover's frame
        ra = c4.observed_alpha(root.nacc, scheme)
        rv = float(ra[c4._WIN] - ra[c4._LOSS])
        a0 = float(np.sum(ra))
        # best edge by collapsed value over searched edges.  eacc is indexed by
        # LEGAL-MOVE POSITION; root.legal[pos] is the actual action id, which is
        # what Edax's r['best'] is expressed in.
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
        # Expected log-likelihood of the TRUE outcome under Dir(alpha):
        #   E_{p~Dir}[log p_true] = digamma(alpha_true) - digamma(alpha0).
        # Proper, concentration-sensitive (unlike rank correlation): confident-
        # wrong is punished, confident-right rewarded, and it is NOT invariant
        # to shifting/scaling alpha0.
        ell = float(digamma(max(ra[ti], 1e-9)) - digamma(max(a0, 1e-9)))
        out.append(dict(empties=r['empties'], tier=r['tier'],
                        move_match=int(best_mv == r['best']),
                        decisive=int(truth != 0),
                        sign_ok=int((rv > 0) == (truth > 0)) if truth != 0 else -1,
                        val_err=abs(rv - truth), a0=a0, ell=ell))
    return (net_name, budget, scheme), out


def main():
    npp = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    with open(CACHE) as fh:
        allrecs = json.load(fh)['positions']
    # stratified sample: up to npp positions per phase
    recs = []
    for lo, hi, _ in PHASES:
        pool = [r for r in allrecs if lo <= r['empties'] < hi]
        recs.extend(pool[:npp])
    print(f'[study] {len(recs)} positions, {len(NETS)} nets x {len(BUDGETS)} '
          f'budgets x {len(SCHEMES)} schemes = '
          f'{len(NETS)*len(BUDGETS)*len(SCHEMES)} search passes', flush=True)

    jobs = [(nm, d, lab, mm, b, s, recs)
            for (nm, d, lab, mm) in NETS for b in BUDGETS for s in SCHEMES]
    results = {}
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = json.load(fh)
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for (key, rows) in pool.imap_unordered(_worker, jobs):
            done['|'.join(map(str, key))] = rows
            with open(OUT, 'w') as fh:
                json.dump(done, fh)
            print(f'[study] {key} done ({time.time()-t0:.0f}s, '
                  f'{len(done)}/{len(jobs)})', flush=True)
    print('[study] BACKUP STUDY DONE', flush=True)


if __name__ == '__main__':
    main()
