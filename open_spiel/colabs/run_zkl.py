"""ZKL arms: KL-form z-grounding with a reliability-scaled concentration.

Same as run_zground.py (klv+kla+cons + z-grounding, pathsum search, from
scratch, 10k eps, seed 0, lr_decay 50k) EXCEPT the z terms are the KL form
(zklv+zkla) instead of plain CE (cev+cea) -- so the game outcome trains
CONCENTRATION too, scaled by how reliable z is at each position (_z_conc).

    python run_zkl.py <sched>      # sched in {const, plies, ply}
Compared against bench_ZGROUND (plain-CE z-grounding, already at ep10000).
"""
import os
import sys
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 10000
LR_DECAY_EPS = 50000       # match DISTILL/ZGROUND schedule exactly
ROOT = 'othello_benchmark'

SCHED = sys.argv[1] if len(sys.argv) > 1 else 'ply'
assert SCHED in ('const', 'plies', 'ply'), SCHED
CKPT = f'bench_ZKL_{SCHED}'
TAG = f'ZKL_{SCHED.upper()}'
LOG_PATH = os.path.join(os.path.dirname(__file__), f'zkl_{SCHED}.log')


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
        loss_weights={'klv': 1.0, 'kla': 2.0, 'zklv': 1.0, 'zkla': 1.0,
                      'cons': 0.5},
        z_kl_sched=SCHED, z_kl_cmax=16.0, z_kl_cmin=1.0, z_kl_tau=12.0,
        z_kl_ply_sat=None,             # -> temp_threshold (20)
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=LR_DECAY_EPS, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== {TAG} from scratch: {NUM_EPISODES} eps, loss={cfg.loss_weights} '
         f'z_sched={cfg.z_kl_sched} cmax={cfg.z_kl_cmax} cmin={cfg.z_kl_cmin} '
         f'tau={cfg.z_kl_tau} -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log(f'{TAG} DONE')
