"""Score budget rules on what they are actually for.

r2 against an oracle label is a proxy. The thing an allocation rule exists to
do is spend its simulations where they change the MOVE, so this scores rules
directly: give each one a per-position budget, look up the move it would have
played at that budget, and ask how often that is the move a 1000-simulation
search settles on.

Every rule is bisected to the SAME mean sims/move before scoring, so the
comparison is at matched compute -- the same discipline calibrate_rule imposes
on the training arms, which is what makes their tournament fair.

Two reference points bracket everything else:
  fixed     every position gets the same budget.  An adaptive rule that cannot
            beat this is not earning its complexity.
  oracle    perfect foreknowledge of the label under the same total budget.
            Note this is a knapsack, not a rescaling: a position needs its
            whole label to come out right and anything less is simply wrong,
            so scaling every label down to hit a mean would fail EVERY
            position that needs more than the floor.  The optimal play is to
            fund the cheapest positions in full and abandon the expensive
            ones, which is also the interesting behaviour -- the value of
            knowing "sims needed" is mostly knowing what to give up on.

Budgets are snapped up to the 50-sim checkpoint grid the traces were recorded
on, so all numbers carry that quantisation; it applies equally to every rule.

    python probe_allocation.py [tag]
"""
import json
import sys

import numpy as np

import probe_analyze as pa

ALLOC_MIN, ALLOC_MAX = 16, 600
TARGETS = (100.0, 150.0, 300.0)


def _snap(sims, cap, check_every):
    """Budget -> index of the checkpoint that budget actually reaches."""
    idx = np.ceil(np.asarray(sims) / check_every).astype(int) - 1
    return np.clip(idx, 0, cap // check_every - 1)


def rule_sims(kind, F, scale, coef=None):
    """The deployed budget rules, reproduced from c4._alloc_sims."""
    margin, conf, n_legal = F[:, 1], np.exp(F[:, 2]), F[:, 3]
    lo, hi = ALLOC_MIN, ALLOC_MAX
    if kind == 'conf':
        score = 1.0 / (1.0 + conf / max(scale, 1e-9))
        return lo + (hi - lo) * score
    if kind == 'margin':
        score = np.exp(-margin / max(scale, 1e-9))
        return lo + (hi - lo) * score
    if kind == 'predictor':
        c = coef
        z = c[0] + c[1] * margin + c[2] * F[:, 2] + c[3] * n_legal
        return np.clip(np.exp(z) * scale, lo, hi)
    raise ValueError(kind)


def calibrate(fn, target, lo=1e-6, hi=1e8, iters=60):
    """Bisect the rule's single knob to a target mean sims. Direction is read
    off the bracket rather than special-cased, exactly as calibrate_rule does,
    since some rules increase in their knob and some decrease."""
    f_lo = fn(lo).mean()
    f_hi = fn(hi).mean()
    inc = f_hi > f_lo
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        m = fn(mid).mean()
        if (m < target) == inc:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def oracle_alloc(y, target, cap):
    """Greedy knapsack against perfect knowledge of `y`: buy the cheapest
    positions first until the total budget runs out, floor the rest.

    Cheapest-first is optimal here because every position is worth the same
    (one correct move) and costs exactly its own label, so the best use of a
    fixed total is to maximise how many labels fit inside it."""
    total = target * len(y)
    order = np.argsort(y, kind='stable')
    spend = np.full(len(y), float(ALLOC_MIN))
    running = spend.sum()
    for i in order:
        extra = min(y[i], cap) - spend[i]
        if extra <= 0 or running + extra > total:
            continue
        spend[i] += extra
        running += extra
    return spend


def agreement(best_idx, n_valid, idx):
    """Fraction of positions whose move at the allocated budget equals the move
    the full search settles on."""
    fin = best_idx[np.arange(len(best_idx)), np.maximum(n_valid - 1, 0)]
    got = best_idx[np.arange(len(best_idx)), idx]
    return float((got == fin).mean())


def analyze(tag, cal2):
    d = np.load(f'probe_data_{tag}.npz')
    F = d['feats']
    best, nval = d['best_idx'], d['n_valid']
    cap, ce = int(d['cap']), int(d['check_every'])
    labels = pa.all_labels(d, cap, ce)
    y = labels['decision']

    print(f'\n{"="*78}\n{tag}: move-agreement with the {cap}-sim answer'
          f'\n{"="*78}')
    print(f'{"":22}' + ''.join(f'{f"mean {int(t)}":>12}' for t in TARGETS))

    coef = np.array(cal2['rules']['predictor']['cfg']['predictor_coef'])
    rows = []

    for name in ('fixed', 'conf', 'predictor', 'margin', 'oracle'):
        cells = []
        for target in TARGETS:
            if name == 'fixed':
                idx = _snap(np.full(len(F), target), cap, ce)
            elif name == 'oracle':
                idx = _snap(oracle_alloc(y, target, cap), cap, ce)
            else:
                s = calibrate(lambda v: rule_sims(name, F, v, coef), target)
                idx = _snap(rule_sims(name, F, s, coef), cap, ce)
            cells.append(agreement(best, nval, idx))
        rows.append((name, cells))
        print(f'  {name:<20}' + ''.join(f'{c:>11.1%} ' for c in cells))

    base = dict(rows)['fixed']
    orc = dict(rows)['oracle']
    print('\n  headroom over a fixed budget (oracle - fixed):')
    print('  ' + ''.join(f'{o-b:>11.1%} ' for o, b in zip(orc, base)))
    print('  captured by conf: ' + ''.join(
        f'{(c-b)/(o-b) if o>b else float("nan"):>11.1%} '
        for c, b, o in zip(dict(rows)['conf'], base, orc)))


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    tags = [sys.argv[1]] if len(sys.argv) > 1 else ['mm2000', 'conf5000']
    for t in tags:
        try:
            analyze(t, cal2)
        except FileNotFoundError as e:
            print(f'{t}: missing data ({e})')
