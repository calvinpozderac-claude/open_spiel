"""Round 3 setup, driven by what round 2 measured:

  conf_nl   the predictor REFIT WITHOUT the margin feature. Round 2 found
            margin's univariate correlation with sims-needed points the wrong
            way (+0.208) and the pure 'margin' arm finished near the bottom,
            so it is dropped and only confidence + legal-move count are kept.
            Prints both fits so the r2 cost of dropping margin is explicit.

  confkl    confidence sets this position's budget, and a reactive KL test may
            still end it early -- pairing the round-2 winner ('conf') with the
            only mechanism that can notice a position settled sooner than its
            budget assumed.

Both calibrated to the same 150 sims/move as round 2, so every arm across
both rounds is directly comparable. Writes round3_calibration.json.
"""
import json

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

TARGET = 150.0
N_POSITIONS = 80
N_FIT_POSITIONS = 250
# eps for confkl's early-exit test: round 2's calibrated 'hybrid'/'kl' value,
# held fixed so the single free knob stays alloc_scale and one bisection is
# enough to hit the target mean.
CONFKL_EPS = 0.00552896


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '2000', ad.MM_SIG)

    print('=== refitting the sims predictor without margin ===')
    fit_pos = ad.sample_positions(net, game, N_FIT_POSITIONS, sims=150, seed=21)
    X = ad.position_features(net, game, fit_pos)      # [1, margin, logconf, n_legal]
    y = ad.oracle_sims_needed(game, net, fit_pos)
    print('  full fit (with margin):')
    coef_full, r2_full = ad.fit_sims_predictor(X, y)
    print('  reduced fit (margin zeroed out):')
    X_nl = X.copy(); X_nl[:, 1] = 0.0
    coef_nl, r2_nl = ad.fit_sims_predictor(X_nl, y)
    coef_nl[1] = 0.0
    print(f'  r2 with margin {r2_full:.4f} vs without {r2_nl:.4f} '
         f'(delta {r2_full - r2_nl:+.4f})')

    print('\n=== calibrating to mean 150 sims/move ===')
    pos = ad.sample_positions(net, game, N_POSITIONS, sims=150, seed=22)
    base = dict(adaptive_check_every=50, alloc_min=16, alloc_max=600)
    out = {'predictor_coef_full': coef_full.tolist(), 'r2_full': r2_full,
           'predictor_coef_nl': coef_nl.tolist(), 'r2_nl': r2_nl, 'rules': {}}

    specs = [
        ('conf_nl', dict(base, adaptive_mode='predictor',
                        predictor_coef=tuple(coef_nl.tolist())),
                    'alloc_scale', 1e-4, 1e4),
        ('confkl',  dict(base, adaptive_mode='confkl',
                        adaptive_kl_eps=CONFKL_EPS),
                    'alloc_scale', 1e-3, 1e6),
    ]
    for name, cfg, knob, lo, hi in specs:
        print(f'{name}:')
        v, mean = ad.calibrate_rule(game, net, pos, cfg, knob, target=TARGET,
                                    lo=lo, hi=hi)
        cfg[knob] = v
        out['rules'][name] = {
            'cfg': {k: (list(x) if isinstance(x, tuple) else x)
                   for k, x in cfg.items()},
            'knob': knob, 'value': v, 'mean_sims': mean}

    with open('round3_calibration.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nwrote round3_calibration.json')


if __name__ == '__main__':
    main()
