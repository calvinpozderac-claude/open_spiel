"""One-line health report for a live run, cheap enough to run every 1000 eps.

The probe and the phase profile both need a network, and bench_<ep>.pt only
lands every deep_eval_every (5000).  latest.pt is rewritten every
quick_eval_every (1000), so it is read directly here -- it is a run blob, so
the model comes out of blob['model'] rather than being a bare state dict.

Reports, all from forward passes only (~2s, no games played):

  probe    MSE / top1 / sign against the 1200 labelled positions in
           edax_eval_set.json, the metric that ranks a trajectory at
           Spearman ~0.95 where Edax match play manages ~0.73
  alpha0   value-head concentration in the OPENING vs the FINAL bucket, with
           the accuracy in each.  Confidence should be LOW where the net is
           near chance and HIGH where it is reliable; every mean-backup net
           measured so far has this exactly inverted.

    python quick_report.py <ckpt_dir>
"""
import json
import os
import re
import sys

import numpy as np

CACHE = 'edax_eval_set.json'
CHUNK = 256
OPEN_LO, OPEN_HI = 45, 61
FINAL_LO, FINAL_HI = 0, 8


def main():
    import torch
    torch.set_num_threads(2)
    import connect4_dirichlet_utils as c4
    d = sys.argv[1] if len(sys.argv) > 1 else 'othello_benchmark/bench_SAMPLE'
    path = os.path.join(d, 'latest.pt')
    try:
        blob = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:                      # mid-write; caller can retry
        print(f'[quick] {path} unreadable ({type(e).__name__})')
        return
    ep, sd = int(blob.get('ep', 0)), blob['model']

    game = c4.load_game('othello')
    c4.set_game(game)
    # A run trained with conf_offset must be SCORED with it too -- _conf is
    # used at inference, so scoring a floored net at offset 0 would report an
    # alpha0 three lower than the network actually emits.
    offset = 0.0
    if '--conf-offset' in sys.argv:
        offset = float(sys.argv[sys.argv.index('--conf-offset') + 1])
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0, conf_offset=offset)
    sig = (int(sd['stem.0.weight'].shape[0]),
           len({int(m.group(1)) for k in sd
                if (m := re.match(r'body\.(\d+)\.', k))}),
           int(sd['head.0.weight'].shape[0]))
    net = c4.C4DirichletNet(*sig)
    net.load_state_dict(sd)
    net.eval()

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
    mse = float(((pred - truth) ** 2).mean())
    sign = float(((pred[dec] > 0) == (truth[dec] > 0)).mean())

    def bucket(lo, hi):
        m = (emp >= lo) & (emp < hi)
        d2 = m & dec
        return (float(a0[m].mean()),
                float(((pred[d2] > 0) == (truth[d2] > 0)).mean()))

    o_a, o_acc = bucket(OPEN_LO, OPEN_HI)
    f_a, f_acc = bucket(FINAL_LO, FINAL_HI)
    print(f'[quick] ep {ep:>6}  MSE {mse:.3f}  top1 {np.mean(top1):5.1%}  '
          f'sign {sign:5.1%}  |  a0 open {o_a:5.2f} (acc {o_acc:4.0%})  '
          f'final {f_a:5.2f} (acc {f_acc:4.0%})  ratio {f_a/max(o_a,1e-9):4.2f}')


if __name__ == '__main__':
    main()
