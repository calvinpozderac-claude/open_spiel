"""Value-head concentration as a continuous function of pieces played, for
every 5000-episode MIXTURE checkpoint (plus ep0).

conc_by_phase.py bucketed into 6 wide phases; this reports one point per
ply (pieces played = 64 - empties) so the curve's shape is visible directly,
not just its endpoints. Raw network only (sims=0) -- the object here is what
the TRAINED HEAD believes unprompted, not what search does to it.

    python conc_vs_ply.py > conc_vs_ply.json
"""
import json
import os
import re

import numpy as np

CACHE = 'edax_eval_set.json'
CKPT_DIR = 'othello_benchmark/bench_MIXTURE'
EPS = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
CHUNK = 256


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
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_MIXTURE, target_agg=c4.AGG_MIXTURE,
                  selection='dirichlet', virtual_loss=1.0)

    with open(CACHE) as fh:
        recs = json.load(fh)['positions']
    states, pieces, truth = [], [], []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        states.append(st)
        pieces.append(64 - r['empties'])
        truth.append(r['value'])
    pieces, truth = np.array(pieces), np.array(truth)
    plies = sorted(set(pieces.tolist()))

    out = {'plies': plies, 'n_per_ply': [int((pieces == p).sum()) for p in plies],
           'checkpoints': {}}
    for ep in EPS:
        path = os.path.join(CKPT_DIR, f'bench_{ep}.pt')
        if not os.path.exists(path):
            continue
        net = c4.load_benchmark_net(CKPT_DIR, str(ep), _sig(path))
        a0, dec_ok = [], []
        for i0 in range(0, len(states), CHUNK):
            chunk = states[i0:i0 + CHUNK]
            v3, vc, _p3, _cf, _o = c4.nn_eval_states(net, 'cpu', chunk)
            for j in range(len(chunk)):
                a0.append(float(vc[j]))
        a0 = np.array(a0)
        pred = None  # not needed; accuracy handled via conc_by_phase already
        by_ply = []
        for p in plies:
            m = pieces == p
            by_ply.append(float(a0[m].mean()) if m.any() else None)
        out['checkpoints'][str(ep)] = by_ply
        print(f'[conc_vs_ply] ep {ep:>6} done', flush=True)

    with open('conc_vs_ply.json', 'w') as fh:
        json.dump(out, fh)
    print('[conc_vs_ply] wrote conc_vs_ply.json')


if __name__ == '__main__':
    main()
