"""End-to-end smoke test of the 'corr' rule: real worker processes, real training.

Not a strength test -- 40 episodes of a tiny net cannot say anything about play.
It checks the plumbing the unit tests cannot: that the rule survives spawn workers
(set_search reaches them), that targets are finite and sane through the training
loss, and that nothing NaNs.  Runs three configurations in sequence:

    legacy   search additive_mle / target avgconc      (the HYBRID rules)
    P1       search additive_mle / target corr          rho_target 0.999
    P2       search corr         / target corr          rho_search 0.1
"""
import math
import os
import re
import shutil
import sys
import tempfile
import time

import connect4_dirichlet_utils as c4

ARMS = [
    ('legacy', dict(search_agg='additive_mle', target_agg='avgconc')),
    ('P1',     dict(search_agg='additive_mle', target_agg='corr', rho_target=0.999)),
    ('P2',     dict(search_agg='corr', target_agg='corr', rho_search=0.1,
                    rho_target=0.999)),
]


def make_cfg(root, name, **over):
    return c4.Config(
        game_name='othello', checkpoint_dir=os.path.join(root, name),
        num_episodes=40, channels=16, num_blocks=2, head_ch=4, seed=0,
        device_preference='cpu', selection='dirichlet', kl_normalize=False,
        backup='mean', fast_sims=16, full_sims=32, fast_prob=0.5,
        adaptive_mode=None, adaptive_kl_eps=None,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0, max_plies=128,
        eval_max_plies=128, use_workers=True, selfplay_workers=2,
        games_per_worker=2, worker_wave=4, n_parallel_games=4, wave_per_game=4,
        pool_prob=0.0, random_pool_frac=0.0, batch_size=64, train_steps_per_ep=2,
        max_buffer=20000,
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0, 'cons': 0.5},
        lr_peak=2e-3, lr_warmup_eps=5, lr_decay_eps=40, lr_min_factor=0.1,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=20, quick_eval_games=4, deep_eval_every=10 ** 9,
        eval_sims=8, eval_games_per_pair=2, eval_last_n=2, eval_refresh_pairs=0,
        resume=False, solved_dir='', **over)


def run_arm(root, name, over):
    lines = []
    t0 = time.time()
    c4.run_training(make_cfg(root, name, **over), log=lambda m: lines.append(str(m)))
    return lines, time.time() - t0


def summarize(name, lines, secs):
    ep = [l for l in lines if re.search(r'ep\s+\d+ \|', l)]
    last = ep[-1] if ep else ''
    m = re.search(r'loss ([\d.]+|nan|inf)', last)
    conc = re.search(r'conc\(pred/tgt\) v ([\d.]+)/([\d.]+) a ([\d.]+)/([\d.]+)', last)
    bad = [l for l in lines if re.search(r'nan|non-?finite|skipped|Traceback', l, re.I)]
    buf = re.search(r'buf (\d+)k', last)
    print(f'  {name:7s} {secs:5.0f}s  loss {m.group(1) if m else "?":>6s}  '
          f'conc v {conc.group(1)}/{conc.group(2)} a {conc.group(3)}/{conc.group(4)}'
          if conc else f'  {name:7s} {secs:5.0f}s  (no episode line)')
    return m, conc, bad


def main():
    root = tempfile.mkdtemp(prefix='smoke_corr_')
    ok = True
    try:
        print('arm      time    final loss   concentration pred/target (state, action)')
        for name, over in ARMS:
            lines, secs = run_arm(root, name, over)
            m, conc, bad = summarize(name, lines, secs)
            if not m or not math.isfinite(float(m.group(1))):
                ok = False; print(f'  !! {name}: loss not finite')
            if bad:
                ok = False; print(f'  !! {name}: suspicious log lines: {bad[:3]}')
            if conc:
                for g in conc.groups():
                    if not math.isfinite(float(g)):
                        ok = False; print(f'  !! {name}: non-finite concentration')
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print('SMOKE OK' if ok else 'SMOKE FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
