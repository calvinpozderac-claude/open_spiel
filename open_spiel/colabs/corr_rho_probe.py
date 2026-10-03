"""Measure rho offline from a FROZEN checkpoint.  The only uncontaminated way.

    python corr_rho_probe.py [checkpoint_dir] [label ...]

Learning rho from drift inside a live run does not work: self-play reads the
weights live and training updates them 6x per episode, so a search's early
simulations are evaluated by a different network from its late ones.  That weight
movement is drift the model blames on idiosyncratic leaf error, which pushes rho
down.  Demonstrated by perturbing a frozen net and re-measuring:

    frozen TZ50k         rho 0.764
    +0.02 sigma          rho 0.720
    +0.05 sigma          rho 0.553
    +0.10 sigma          rho 0.001

A live attempt (w_drift=1.0, rho_mode='global') duly collapsed 0.80 -> 0.477 by
ep2000 with the term at 83% of the loss, and was abandoned.

With the weights FROZEN the confound is gone, and this is cheap: a few hundred
positions suffice (+-0.03 at 400).  The practical route to an adaptive rho is to
run this between generations and set rho_search from it, rather than to
backpropagate through a live, non-stationary estimate.
"""
import sys

import numpy as np

import connect4_dirichlet_utils as c4
import corr_posterior as cp

FRACS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
SIMS = 400
SIG = (64, 10, 8)
N_POS = 400


def probe(ckpt_dir, label, n_pos=N_POS, sims=SIMS, seed=5):
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg='corr', target_agg='corr', selection='dirichlet',
                  virtual_loss=1.0, snapshot_fracs=FRACS,
                  rho_search=0.8, rho_target=0.999)
    net = (c4.C4DirichletNet(*SIG) if label == 'untrained'
           else c4.load_benchmark_net(ckpt_dir, label, SIG))
    cfg = c4._worker_cfg(c4.Config(
        game_name='othello', channels=SIG[0], num_blocks=SIG[1], head_ch=SIG[2],
        search_agg='corr', target_agg='corr', snapshot_fracs=FRACS,
        fast_sims=sims, full_sims=sims, fast_prob=1.0, worker_wave=8,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0,
        use_workers=False, n_parallel_games=4), SIG)
    sp = c4.ParallelSelfPlay(game, net, 'cpu', cfg, checkpoint_dir=None, seed=seed)
    it = sp.episodes()
    rows, solved = [], 0
    while len(rows) < n_pos:
        for s_ in next(it):
            if len(s_.get('snaps', ())) == 5:
                rows.append(np.array(s_['snaps']))
                solved += bool(s_['solved'])
    rows = np.array(rows[:n_pos])
    rho, npair = cp.rho_from_drift(rows)
    d, delta = cp.drift_terms(rows)
    r = d / np.maximum(delta, 1e-30)
    return dict(rho=rho, pairs=npair, n=len(rows), solved=solved,
                med=float(np.median(r)), p90=float(np.percentile(r, 90)),
                frac_gt1=float((r > 1).mean()))


def main(argv):
    ckpt = argv[1] if len(argv) > 1 else 'othello_benchmark/bench_OLIVAW_TZ'
    labels = argv[2:] or ['untrained', '10000', '30000', '50000']
    print(f'rho from snapshot drift, FROZEN weights, {SIMS} sims, {N_POS} positions')
    print(f'  {"label":<10}{"rho":>7}{"median d/delta":>16}{"p90":>8}'
          f'{"frac>1":>9}{"solved":>8}')
    for lab in labels:
        try:
            r = probe(ckpt, lab)
        except Exception as e:                                    # noqa: BLE001
            print(f'  {lab:<10}  failed: {type(e).__name__}: {e}')
            continue
        print(f'  {lab:<10}{r["rho"]:>7.3f}{r["med"]:>16.3f}{r["p90"]:>8.2f}'
              f'{r["frac_gt1"]:>9.1%}{r["solved"]:>8d}')
    print('\n  frac>1 is the share of snapshot pairs drifting more than ANY rho can')
    print('  explain; a large value means the i.i.d.-leaf model is a poor fit there.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
