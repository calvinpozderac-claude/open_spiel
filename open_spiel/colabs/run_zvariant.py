"""Concentration-fix loss variants (follow-ups to the zkl arms).

Each arm is a from-scratch 10k run, pathsum search, seed 0, lr_decay 50k --
identical to run_zkl.py except the loss.  Baseline for all is bench_ZGROUND
(plain-CE z-grounding) at ep10000.

  zkl2          KL z-grounding with c_z = 2 * predicted alpha0 (relpred): the
                z-KL nudges confidence UP relative to the net's own scale, so
                it can't fight klv's distilled concentration the way a fixed
                absolute c_z did.
  cemse_double  CE z-mean (cev/cea) + MSE concentration growth to 2*alpha0.
  cemse_shift   CE z-mean + MSE concentration growth to alpha0 + 16.
  allmse        replace the distillation KL (klv/kla) with CE-on-mean
                (dcev/dcea) + MSE-on-concentration to the search belief
                (mkcv/mkca); keep plain-CE z-grounding.

    python run_zvariant.py <arm>
"""
import os
import sys
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 10000
LR_DECAY_EPS = 50000
ROOT = 'othello_benchmark'

ARMS = {
    'zkl2': dict(
        loss={'klv': 1.0, 'kla': 2.0, 'zklv': 0.1, 'zkla': 0.1, 'cons': 0.5},
        extra=dict(z_kl_conc_mode='relpred', z_kl_rel_mult=2.0)),
    'cemse_double': dict(
        loss={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
              'mzcv': 1.0, 'mzca': 1.0, 'cons': 0.5},
        extra=dict(z_conc_grow_mode='double')),
    'cemse_shift': dict(
        loss={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
              'mzcv': 1.0, 'mzca': 1.0, 'cons': 0.5},
        extra=dict(z_conc_grow_mode='shift', z_conc_grow_shift=16.0)),
    'allmse': dict(
        loss={'dcev': 1.0, 'dcea': 2.0, 'mkcv': 0.3, 'mkca': 0.3,
              'cev': 1.0, 'cea': 1.0, 'cons': 0.5},
        extra=dict()),
}

ARM = sys.argv[1]
spec = ARMS[ARM]
CKPT = f'bench_{ARM.upper()}'
TAG = ARM.upper()
LOG_PATH = os.path.join(os.path.dirname(__file__), f'zvar_{ARM}.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(ROOT, CKPT),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=10, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_PATHSUM, target_agg=c4.AGG_AVGCONC,
        selection='dirichlet', kl_normalize=False,
        backup='mean',
        fast_sims=100, full_sims=400, fast_prob=0.6,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.0, random_pool_frac=0.0,
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        loss_weights=spec['loss'],
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=LR_DECAY_EPS, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='', **spec['extra'])


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== {TAG} from scratch: {NUM_EPISODES} eps, loss={cfg.loss_weights} '
         f'extra={spec["extra"]} -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log(f'{TAG} DONE')
