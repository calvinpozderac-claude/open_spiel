"""Does in-tree descent temperature inflate the TARGET concentration?

MM_long searches the opening at descent temp 1.0 and its value head reads
alpha0 ~57-67 there; the OLIVAW run holds 16.0 everywhere and reads 106.6. That
comparison is confounded by depth, sims and learning rate, so it cannot say
whether the temperature is the cause.

This isolates it: ONE fixed network, the SAME positions, only the descent
temperature varies. C4MCTSBot's `temp` is the in-tree descent temperature --
root move choice is a separate step -- so the search targets it produces are
directly comparable across temperatures.

What is measured is the TARGET alpha0, i.e. observed_alpha(root.nacc) collapsed
with the same rule training uses. If a sharp descent inflates it in the opening
and not in the endgame, that is the mechanism, and it is a property of the
search rather than of the trained weights.

    python dtemp_vs_conc.py <ckpt_dir> <label> [sims]
"""
import json
import os
import re
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
TEMPS = [1.0, 4.0, 16.0, 64.0]
PHASES = [(45, 61, 'opening'), (15, 25, 'early end')]
N_PER_PHASE = 40


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


def main():
    ckpt_dir, label = sys.argv[1], sys.argv[2]
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 220
    c4, game = _init()
    net = c4.load_benchmark_net(ckpt_dir, label,
                                _sig(os.path.join(ckpt_dir, f'bench_{label}.pt')))
    with open(CACHE) as fh:
        recs = json.load(fh)['positions']

    sets = {}
    for lo, hi, name in PHASES:
        pool = [r for r in recs if lo <= r['empties'] < hi][:N_PER_PHASE]
        states = []
        for r in pool:
            st = game.new_initial_state()
            for a in r['actions']:
                st.apply_action(int(a))
            states.append(st)
        sets[name] = states

    print(f'{os.path.basename(ckpt_dir)} @ {label}, {sims} sims, '
          f'{N_PER_PHASE} positions/phase')
    print(f'\n{"descent temp":>13}' + ''.join(f'{n:>14}' for _, _, n in PHASES))
    for t in TEMPS:
        row, t0 = [], time.time()
        for _, _, name in PHASES:
            a0s = []
            for st in sets[name]:
                rng = np.random.default_rng(0)
                bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                                   temp=t, random_state=rng)
                root = bot.mcts_search(st)
                a = c4.observed_alpha(root.nacc, c4._SEARCH_AGG)
                a0s.append(float(np.sum(a)) if a is not None else float('nan'))
            row.append(np.nanmean(a0s))
        print(f'{t:>13.0f}' + ''.join(f'{v:>14.1f}' for v in row)
              + f'   [{time.time()-t0:.0f}s]', flush=True)
    print('\nA sharp descent visits fewer distinct lines, so the backed-up '
          'means agree\nmore and additive_mle fits a larger alpha0. If the '
          'opening column rises with\ntemperature and the endgame column does '
          'not, the inflation is the search.')


if __name__ == '__main__':
    main()
