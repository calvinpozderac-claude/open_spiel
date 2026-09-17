"""compare_runs.py's probe, extended with phase-split MSE (opening/endgame) so
it is directly comparable to az_probe.py's mse_open/mse_end columns.  No
search -- pure net forward pass, like az_probe, so this is fast.

    python thompson_probe_phased.py <dir>:<ep> [...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
import compare_runs as CR

CACHE = 'edax_eval_set.json'


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
          f'{"MSE_open":>10}{"MSE_end":>9}{"a0open":>8}{"a0end":>8}')
    for spec in sys.argv[1:]:
        d, label = spec.rsplit(':', 1)
        net = CR.load_net(c4, d, label)
        name = d.split('/')[-1].replace('bench_', '')
        if net is None:
            print(f'{name:<12}{label:>7}   (missing)')
            continue
        pred, a0 = [], []
        for i0 in range(0, len(states), CR.CHUNK):
            chunk = states[i0:i0 + CR.CHUNK]
            v3, vc, _p3, _cf, _ = c4.nn_eval_states(net, 'cpu', chunk)
            for j in range(len(chunk)):
                pred.append(float(v3[j][c4._WIN] - v3[j][c4._LOSS]))
                a0.append(float(vc[j]))
        pred, a0 = np.array(pred), np.array(a0)
        dec = truth != 0
        r = CR.score(c4, game, net, states, truth, emp, recs)

        def bmse(lo, hi):
            m = (emp >= lo) & (emp < hi)
            return float(((pred[m] - truth[m]) ** 2).mean()) if m.any() else float('nan')

        def ba0(lo, hi):
            m = (emp >= lo) & (emp < hi)
            return float(a0[m].mean()) if m.any() else float('nan')

        print(f'{name:<12}{label:>7}{r["mse"]:>8.3f}{r["top1"]:>8.1%}'
              f'{r["sign"]:>8.1%}{bmse(45,61):>10.3f}{bmse(0,8):>9.3f}'
              f'{ba0(45,61):>8.2f}{ba0(0,8):>8.2f}')


if __name__ == '__main__':
    main()
