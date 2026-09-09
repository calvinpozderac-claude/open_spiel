"""Does backing up SAMPLES instead of means fix the inverted confidence profile?

The value head is most concentrated in the opening (alpha0 ~106 at 45-60
empties) and least in the early endgame (~12 at 15-24), which is backwards.
The descent-temperature sweep showed sharpening the in-tree descent inflates
the opening figure by 51% and leaves the endgame untouched -- but even at
temp 1 the profile is still inverted ~4:1, so the temperature is a contributing
factor and not the cause.

The remaining suspect is the aggregation. `_payload(alpha)` reduces each
backed-up Dirichlet to its MEAN m = alpha/alpha0, and `observed_alpha` under
additive_mle fits a concentration to the SPREAD of those means. In the opening
every line evaluates to roughly the same balanced vector, the means agree, and
the fit returns a large alpha0 -- so alpha0 measures agreement among estimates
rather than certainty about the outcome.

This patches `_payload` and nothing else, so the search, the tree and the
collapse rule are identical across modes:

  mean       m = alpha/alpha0                 (what the runs do today)
  dirichlet  m ~ Dir(alpha)                   posterior sample: spread is set
                                              by the leaf's OWN alpha0, which
                                              is the quantity under suspicion
  onehot     m ~ Categorical(alpha/alpha0)    outcome sample: maximal variance
                                              regardless of leaf confidence,
                                              included as a bound rather than
                                              a proposal

`mean` must reproduce the descent-temp sweep's 113.1 at temp 16; if it does
not, the patch is not reaching the backup and nothing else here is valid.

Terminal backups are left alone -- a solved position's outcome is not an
estimate and should not be resampled.

    python sample_backup_probe.py <ckpt_dir> <label> [sims] [descent_temp]
"""
import json
import os
import re
import sys
import time

import numpy as np

import connect4_dirichlet_utils as c4

CACHE = 'edax_eval_set.json'
PHASES = [(45, 61, 'opening'), (15, 25, 'early end')]
N_PER_PHASE = 40
MODES = ['mean', 'dirichlet', 'onehot']

_ORIG_PAYLOAD = c4._payload


def make_payload(mode, rng):
    if mode == 'mean':
        return _ORIG_PAYLOAD

    def _p(alpha):
        aw, ad, al = float(alpha[0]), float(alpha[1]), float(alpha[2])
        a0 = aw + ad + al
        if mode == 'dirichlet':
            s = rng.dirichlet((max(aw, 1e-9), max(ad, 1e-9), max(al, 1e-9)))
            mw, md, ml = float(s[0]), float(s[1]), float(s[2])
        else:
            k = int(rng.choice(3, p=(aw / a0, ad / a0, al / a0)))
            mw, md, ml = (1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0,
                          1.0 if k == 2 else 0.0)
        inv = 1.0 / (a0 + 1.0)
        m = (mw, md, ml)
        q = (mw * mw, md * md, ml * ml)
        v = (mw * (1.0 - mw) * inv, md * (1.0 - md) * inv, ml * (1.0 - ml) * inv)
        a = (aw, ad, al)
        return ((m, q, v, a),
                ((ml, md, mw), (q[2], q[1], q[0]), (v[2], v[1], v[0]),
                 (al, ad, aw)))
    return _p


def _sig(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    return (int(sd['stem.0.weight'].shape[0]),
            len({int(m.group(1)) for k in sd
                 if (m := re.match(r'body\.(\d+)\.', k))}),
            int(sd['head.0.weight'].shape[0]))


def main():
    import torch
    torch.set_num_threads(4)
    ckpt_dir, label = sys.argv[1], sys.argv[2]
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 220
    dtemp = float(sys.argv[4]) if len(sys.argv) > 4 else 16.0

    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE,
                  target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net(
        ckpt_dir, label, _sig(os.path.join(ckpt_dir, f'bench_{label}.pt')))
    with open(CACHE) as fh:
        recs = json.load(fh)['positions']

    sets = {}
    for lo, hi, name in PHASES:
        states = []
        for r in [x for x in recs if lo <= x['empties'] < hi][:N_PER_PHASE]:
            st = game.new_initial_state()
            for a in r['actions']:
                st.apply_action(int(a))
            states.append(st)
        sets[name] = states

    print(f'{os.path.basename(ckpt_dir)} @ {label}, {sims} sims, '
          f'descent temp {dtemp:g}, {N_PER_PHASE} positions/phase')
    print(f'\n{"backup":>11}' + ''.join(f'{n:>14}' for _, _, n in PHASES)
          + f'{"ratio":>9}')
    try:
        for mode in MODES:
            rng_p = np.random.default_rng(7)
            c4._payload = make_payload(mode, rng_p)
            row, t0 = [], time.time()
            for _, _, name in PHASES:
                a0s = []
                for st in sets[name]:
                    bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                                       temp=dtemp,
                                       random_state=np.random.default_rng(0))
                    a = c4.observed_alpha(bot.mcts_search(st).nacc,
                                          c4._SEARCH_AGG)
                    a0s.append(float(np.sum(a)) if a is not None
                               else float('nan'))
                row.append(float(np.nanmean(a0s)))
            print(f'{mode:>11}' + ''.join(f'{v:>14.1f}' for v in row)
                  + f'{row[0]/row[1]:>9.2f}   [{time.time()-t0:.0f}s]',
                  flush=True)
    finally:
        c4._payload = _ORIG_PAYLOAD

    print('\nratio = opening alpha0 / early-end alpha0. Today it is ~4-5x the '
          'wrong way;\nbelow 1.0 would mean the endgame is finally the '
          'confident phase.')


if __name__ == '__main__':
    main()
