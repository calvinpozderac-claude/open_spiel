"""Edax-probe for AlphaZero checkpoints -- the plateau curve, comparable to the
Thompson runs' compare_runs.py.

Per checkpoint, on the 1200 Edax-labelled positions:
  MSE    (tanh value - truth)^2            value head vs ground truth
  sign   sign(value)==sign(truth)          decisive positions only
  top1   argmax policy over legal == best  policy head vs Edax best move
  |v|    mean |value|                       confidence proxy (AZ has no alpha0)
  by-phase MSE (opening 45-60 empties / endgame 0-8) to see WHERE it plateaus.

    python az_probe.py <dir>:<ep> [<dir>:<ep> ...]
    python az_probe.py curve othello_benchmark/bench_AZ_MATCH   # all bench_*.pt
"""
import json
import os
import re
import sys

import numpy as np

CACHE = 'edax_eval_set.json'
CHUNK = 256


def load_net(az, d, label):
    import torch
    p = os.path.join(d, f'bench_{label}.pt')
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location='cpu', weights_only=True)
    blocks = len({int(m.group(1)) for k in sd
                  if (m := re.match(r'body\.(\d+)\.', k))})
    ch = int(sd['stem.0.weight'].shape[0])
    hc = int(sd['head.0.weight'].shape[0])
    net = az.AlphaZeroNet(ch, blocks, hc)
    net.load_state_dict(sd)
    net.eval()
    return net


def score(az, net, states, truth, emp, recs):
    pred, absv, top1 = [], [], []
    for i0 in range(0, len(states), CHUNK):
        chunk = states[i0:i0 + CHUNK]
        lg, v, _ob = az.nn_eval_states(net, 'cpu', chunk)
        for j, st in enumerate(chunk):
            pred.append(float(v[j]))
            absv.append(abs(float(v[j])))
            legal = [int(x) for x in st.legal_actions()]
            row = np.asarray(lg[j])[legal]
            top1.append(legal[int(row.argmax())] == recs[i0 + j]['best'])
    pred = np.array(pred)
    dec = truth != 0

    def bmse(lo, hi):
        m = (emp >= lo) & (emp < hi)
        return float(((pred[m] - truth[m]) ** 2).mean()) if m.any() else float('nan')

    return dict(mse=float(((pred - truth) ** 2).mean()),
                sign=float(((pred[dec] > 0) == (truth[dec] > 0)).mean()),
                top1=float(np.mean(top1)), absv=float(np.mean(absv)),
                mse_open=bmse(45, 61), mse_end=bmse(0, 8))


def main():
    import torch
    torch.set_num_threads(3)
    import connect4_alphazero_utils as az
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
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
    truth, emp = np.array(truth, float), np.array(emp)

    if sys.argv[1] == 'curve':
        d = sys.argv[2]
        labels = sorted((int(m.group(1)) for f in os.listdir(d)
                         if (m := re.match(r'bench_(\d+)\.pt$', f))))
        specs = [f'{d}:{lab}' for lab in labels]
    else:
        specs = sys.argv[1:]

    print(f'{"run":<16}{"ep":>7}{"MSE":>8}{"sign":>8}{"top1":>8}{"|v|":>7}'
          f'{"MSE_open":>10}{"MSE_end":>9}')
    for spec in specs:
        d, label = spec.rsplit(':', 1)
        net = load_net(az, d, label)
        name = os.path.basename(d).replace('bench_', '')
        if net is None:
            print(f'{name:<16}{label:>7}   (missing)')
            continue
        r = score(az, net, states, truth, emp, recs)
        print(f'{name:<16}{label:>7}{r["mse"]:>8.3f}{r["sign"]:>8.1%}'
              f'{r["top1"]:>8.1%}{r["absv"]:>7.2f}'
              f'{r["mse_open"]:>10.3f}{r["mse_end"]:>9.3f}')


if __name__ == '__main__':
    main()
