"""Is the value head over-confident early in the game?

The training log prints one batch-average concentration -- "conc(pred/tgt) v
46.4/97.4" -- which cannot distinguish a net that is sensibly confident in
solved endgames from one that is loudly confident about move 10. Confidence is
only a fault when it outruns accuracy, so this buckets BOTH by game phase, on
the labelled eval set where the outcome under strong play is known.

alpha0 is the total Dirichlet concentration from _conf(): unbounded above, and
under the 'additive' rule a target alpha0 is roughly the visit count, so ~100
is the scale the targets live at.

Read it as: within a bucket, high alpha0 with accuracy near 50% is
over-confidence. High alpha0 with high accuracy is just knowing the answer,
which is expected once the board is nearly full.

    python conc_by_phase.py <ckpt_dir> <label> [<ckpt_dir> <label> ...]
"""
import json
import os
import re
import sys

import numpy as np

CACHE = 'edax_eval_set.json'
CHUNK = 256
# Fewer empties = later in the game. The eval set's exact tier is <=20.
BUCKETS = [(45, 61, 'opening'), (35, 45, 'early mid'), (25, 35, 'late mid'),
           (15, 25, 'early end'), (8, 15, 'endgame'), (0, 8, 'final')]


def _init():
    import torch
    torch.set_num_threads(4)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _sig(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def run(c4, game, ckpt_dir, label, recs):
    path = os.path.join(ckpt_dir, f'bench_{label}.pt')
    net = c4.load_benchmark_net(ckpt_dir, label, _sig(path))
    states, emp, truth = [], [], []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        states.append(st)
        emp.append(r['empties'])
        truth.append(r['value'])
    emp, truth = np.array(emp), np.array(truth)

    a0, dec = [], []
    for i0 in range(0, len(states), CHUNK):
        v3, vc, _p3, _cf, _o = c4.nn_eval_states(net, 'cpu', states[i0:i0 + CHUNK])
        for j in range(len(vc)):
            a0.append(float(vc[j]))
            dec.append(float(v3[j][c4._WIN] - v3[j][c4._LOSS]))
    a0, dec = np.array(a0), np.array(dec)

    print(f'\n=== {os.path.basename(ckpt_dir)} @ {label} '
          f'({len(states)} positions) ===')
    print(f'{"phase":<11}{"empties":>9}{"n":>6}{"alpha0":>9}'
          f'{"|E[w]-E[l]|":>13}{"accuracy":>10}')
    for lo, hi, name in BUCKETS:
        m = (emp >= lo) & (emp < hi)
        if m.sum() < 10:
            continue
        d = m & (truth != 0)
        acc = float(((dec[d] > 0) == (truth[d] > 0)).mean()) if d.any() else float('nan')
        print(f'{name:<11}{f"{lo}-{hi-1}":>9}{int(m.sum()):>6}'
              f'{a0[m].mean():>9.1f}{np.abs(dec[m]).mean():>13.3f}{acc:>9.1%}')
    d = truth != 0
    print(f'{"ALL":<11}{"":>9}{len(a0):>6}{a0.mean():>9.1f}'
          f'{np.abs(dec).mean():>13.3f}'
          f'{float(((dec[d] > 0) == (truth[d] > 0)).mean()):>9.1%}')


def main():
    if not os.path.exists(CACHE):
        sys.exit(f'no {CACHE}')
    with open(CACHE) as fh:
        recs = json.load(fh)['positions']
    c4, game = _init()
    args = sys.argv[1:]
    for i in range(0, len(args) - 1, 2):
        run(c4, game, args[i], args[i + 1], recs)


if __name__ == '__main__':
    main()
