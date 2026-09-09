"""Same quick metrics as quick_report.py, but for saved bench_<ep>.pt files, so
arms can be lined up at matched episodes.

Reports per checkpoint:
  MSE / top1 / sign   against the 1200 labelled positions in edax_eval_set.json
  a0 open / final     value-head concentration in the 45-60 and 0-7 empties
                      buckets, with each bucket's accuracy
  ratio               final / opening.  Above 1 means the net is more certain
                      where it is actually right.  Every additive_mle arm runs
                      this INVERTED at roughly 0.15.

    python compare_runs.py <dir>:<ep> [<dir>:<ep> ...]
"""
import json
import os
import re
import sys

import numpy as np

CACHE = 'edax_eval_set.json'
CHUNK = 256


def load_net(c4, d, label):
    import torch
    p = os.path.join(d, f'bench_{label}.pt')
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location='cpu', weights_only=True)
    sig = (int(sd['stem.0.weight'].shape[0]),
           len({int(m.group(1)) for k in sd
                if (m := re.match(r'body\.(\d+)\.', k))}),
           int(sd['head.0.weight'].shape[0]))
    net = c4.C4DirichletNet(*sig)
    net.load_state_dict(sd)
    net.eval()
    return net


def score(c4, game, net, states, truth, emp, recs):
    pred, a0, top1 = [], [], []
    for i0 in range(0, len(states), CHUNK):
        chunk = states[i0:i0 + CHUNK]
        v3, vc, p3, cf, _ = c4.nn_eval_states(net, 'cpu', chunk)
        for j, st in enumerate(chunk):
            pred.append(float(v3[j][c4._WIN] - v3[j][c4._LOSS]))
            a0.append(float(vc[j]))
            legal = [int(x) for x in st.legal_actions()]
            q = p3[j][legal][:, c4._WIN] - p3[j][legal][:, c4._LOSS]
            top1.append(legal[int(np.argmax(q))] == recs[i0 + j]['best'])
    pred, a0 = np.array(pred), np.array(a0)
    dec = truth != 0

    def buck(lo, hi):
        m = (emp >= lo) & (emp < hi)
        d = m & dec
        return float(a0[m].mean()), float(((pred[d] > 0) == (truth[d] > 0)).mean())

    oa, oacc = buck(45, 61)
    fa, facc = buck(0, 8)
    return dict(mse=float(((pred - truth) ** 2).mean()),
                top1=float(np.mean(top1)),
                sign=float(((pred[dec] > 0) == (truth[dec] > 0)).mean()),
                oa=oa, oacc=oacc, fa=fa, facc=facc, ratio=fa / max(oa, 1e-9))


def main():
    import torch
    torch.set_num_threads(3)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    with open(CACHE) as fh:
        recs = json.load(fh)['positions']
    states, truth, emp = [], [], []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        states.append(st)
        truth.append(r['value'])
        emp.append(r['empties'])
    truth, emp = np.array(truth), np.array(emp)

    print(f'{"run":<12}{"ep":>7}{"MSE":>8}{"top1":>8}{"sign":>8}'
          f'{"a0 open":>9}{"(acc)":>7}{"a0 final":>10}{"(acc)":>7}{"ratio":>8}')
    for spec in sys.argv[1:]:
        d, label = spec.rsplit(':', 1)
        net = load_net(c4, d, label)
        name = os.path.basename(d).replace('bench_', '').replace('_snaps', '')
        if net is None:
            print(f'{name:<12}{label:>7}   (no checkpoint)')
            continue
        r = score(c4, game, net, states, truth, emp, recs)
        print(f'{name:<12}{label:>7}{r["mse"]:>8.3f}{r["top1"]:>7.1%}'
              f'{r["sign"]:>8.1%}{r["oa"]:>9.2f}{r["oacc"]:>7.0%}'
              f'{r["fa"]:>10.2f}{r["facc"]:>7.0%}{r["ratio"]:>8.2f}')


if __name__ == '__main__':
    main()
