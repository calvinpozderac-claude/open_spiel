"""The honest headroom: how much of the oracle's advantage SURVIVES the search
being stochastic.

probe_allocation.py's oracle reads the label from the very run whose final move
it is then scored against, so it knows that run's dice. No predictor can know
them. Its ~20-point margin over a fixed budget is therefore an over-estimate of
what any model -- learned head included -- could ever reach.

This splits that margin in two using the replicate traces:

  same-run oracle   label and evaluation from one run.  What probe_allocation
                    reports, and an upper bound that is not attainable.
  cross-run oracle  label from run 1, move agreement evaluated on run 2.  A
                    LOWER bound on what a head could reach, because it
                    allocates off a single noisy draw of the label, while a
                    model would be predicting the label's expectation.
  denoised oracle   labels from runs 1 and 2 averaged, scored on run 3.  A
                    less noisy stand-in for that expectation, still evaluated
                    on a run neither label saw.  With the single-draw row
                    beside it, the two show which direction and how fast the
                    bound moves as the label is denoised -- if averaging two
                    labels barely helps, averaging infinitely many will not
                    either, and the ceiling really is near the lower bound.

The gap between same-run and the rest is pure search noise: budget that looked
well spent only because the labeller and the scorer rolled the same dice.

    python probe_transfer.py [tag]
"""
import json
import sys

import numpy as np

import probe_analyze as pa
import probe_allocation as al


def analyze(tag, cal2):
    d = np.load(f'probe_data_{tag}.npz')
    r = np.load(f'probe_replicate_{tag}.npz')
    cap, ce = int(d['cap']), int(d['check_every'])
    idx_main = r['main_index']

    y1 = pa.all_labels(d, cap, ce)['decision'][idx_main]
    y2 = pa.all_labels(r, cap, ce)['decision']
    F = d['feats'][idx_main]
    b1, n1 = d['best_idx'][idx_main], d['n_valid'][idx_main]
    b2, n2 = r['best_idx'], r['n_valid']

    print(f'\n{"="*78}\n{tag}: transferable headroom ({len(y1)} positions, '
          f'two independent searches)\n{"="*78}')
    ly1, ly2 = np.log(y1), np.log(y2)
    print(f'  label reliability corr(run1, run2) = '
          f'{np.corrcoef(ly1, ly2)[0,1]:+.3f}   '
          f'exact agreement {np.mean(y1 == y2):.0%}')
    print(f'  run-to-run move agreement at full {cap} sims: '
          f'{np.mean(b1[np.arange(len(b1)), n1-1] == b2[np.arange(len(b2)), n2-1]):.1%}'
          f'   <- even the "answer" is not deterministic')

    # Third independent run, if collected: lets the allocator average two
    # labels and still be scored on a trajectory neither of them came from.
    try:
        r3 = np.load(f'probe_replicate3_{tag}.npz')
        y3 = pa.all_labels(r3, cap, ce)['decision']
        b3, n3 = r3['best_idx'], r3['n_valid']
        y12 = np.exp(0.5 * (np.log(y1) + np.log(y2)))
        print(f'  reliability of the 2-label average = '
              f'{np.corrcoef(np.log(y12), np.log(y3))[0,1]:+.3f} '
              f'(vs {np.corrcoef(np.log(y1), np.log(y3))[0,1]:+.3f} for one)')
    except FileNotFoundError:
        r3 = None

    coef = np.array(cal2['rules']['predictor']['cfg']['predictor_coef'])
    rows = ['fixed', 'conf', 'predictor', 'oracle same-run', 'oracle cross-run']
    if r3 is not None:
        rows.append('oracle denoised')
    print(f'\n{"":22}' + ''.join(f'{f"mean {int(t)}":>12}' for t in al.TARGETS))
    for name in rows:
        cells = []
        for target in al.TARGETS:
            # Everything is scored on run 2, except the denoised oracle, whose
            # allocation already used run 2 and so must be scored on run 3.
            ev_b, ev_n = b2, n2
            if name == 'fixed':
                alloc = np.full(len(F), target)
            elif name == 'oracle same-run':
                alloc = al.oracle_alloc(y2, target, cap)
            elif name == 'oracle cross-run':
                alloc = al.oracle_alloc(y1, target, cap)
            elif name == 'oracle denoised':
                alloc = al.oracle_alloc(y12, target, cap)
                ev_b, ev_n = b3, n3
            else:
                s = al.calibrate(lambda v: al.rule_sims(name, F, v, coef),
                                 target)
                alloc = al.rule_sims(name, F, s, coef)
            cells.append(al.agreement(ev_b, ev_n, al._snap(alloc, cap, ce)))
        print(f'  {name:<20}' + ''.join(f'{c:>11.1%} ' for c in cells))
    if r3 is not None:
        # The denoised row is scored on a different run, so it needs its own
        # baseline to be read against rather than the run-2 fixed row.
        base3 = [al.agreement(b3, n3, al._snap(np.full(len(F), t), cap, ce))
                 for t in al.TARGETS]
        print(f'  {"(fixed, on run 3)":<20}'
              + ''.join(f'{c:>11.1%} ' for c in base3))


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    tags = [sys.argv[1]] if len(sys.argv) > 1 else ['mm2000', 'conf5000']
    for t in tags:
        try:
            analyze(t, cal2)
        except FileNotFoundError as e:
            print(f'{t}: missing data ({e})')
