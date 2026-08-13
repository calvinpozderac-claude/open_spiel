"""Round 2 setup: fit the 'predictor' rule's coefficients, then calibrate
every candidate budget rule to the SAME mean of 150 sims/move, so the arms
that follow differ only in HOW they spend a fixed compute budget.

Writes round2_calibration.json for run_round2_sweep.py to consume.

Rules (see the _ALLOC_MODES / _REACTIVE_MODES block in
connect4_dirichlet_utils for what each does):
  margin     upfront, from the top1-top2 prior action-value gap
  conf       upfront, from the state head's concentration
  predictor  upfront, from a linear fit on prior features vs oracle labels
  decision   reactive, stop when the MOVE stops changing (not the value)
  hybrid     reactive KL, but eps relaxed in proportion to the margin
"""
import json

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

TARGET = 150.0
N_POSITIONS = 80
N_FIT_POSITIONS = 250


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '2000', ad.MM_SIG)

    print('=== fitting the predictor (is sims-needed learnable at all?) ===')
    fit_pos = ad.sample_positions(net, game, N_FIT_POSITIONS, sims=150, seed=21)
    X = ad.position_features(net, game, fit_pos)
    y = ad.oracle_sims_needed(game, net, fit_pos)
    print(f'  oracle labels: mean={y.mean():.0f} median={np.median(y):.0f} '
         f'min={y.min():.0f} max={y.max():.0f}  '
         f'(hit cap: {(y >= 1000).mean():.1%})')
    coef, r2 = ad.fit_sims_predictor(X, y)

    print('\n=== calibrating every rule to mean 150 sims/move ===')
    pos = ad.sample_positions(net, game, N_POSITIONS, sims=150, seed=22)
    base = dict(adaptive_check_every=50, alloc_min=16, alloc_max=600)
    out = {'predictor_coef': coef.tolist(), 'predictor_r2': r2, 'rules': {}}

    specs = [
        ('margin',    dict(base, adaptive_mode='margin'), 'alloc_scale', 1e-5, 1e3),
        ('conf',      dict(base, adaptive_mode='conf'), 'alloc_scale', 1e-3, 1e6),
        ('predictor', dict(base, adaptive_mode='predictor',
                          predictor_coef=tuple(coef.tolist())),
                      'alloc_scale', 1e-4, 1e4),
        ('decision',  dict(base, adaptive_mode='decision'),
                      'decision_margin', 1e-6, 2.0),
        ('hybrid',    dict(base, adaptive_mode='hybrid',
                          hybrid_margin_scale=0.05),
                      'adaptive_kl_eps', 1e-6, 10.0),
    ]
    for name, cfg, knob, lo, hi in specs:
        print(f'{name}:')
        v, mean = ad.calibrate_rule(game, net, pos, cfg, knob, target=TARGET,
                                    lo=lo, hi=hi)
        cfg[knob] = v
        out['rules'][name] = {'cfg': {k: (list(x) if isinstance(x, tuple) else x)
                                     for k, x in cfg.items()},
                             'knob': knob, 'value': v, 'mean_sims': mean}

    with open('round2_calibration.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nwrote round2_calibration.json')


if __name__ == '__main__':
    main()
