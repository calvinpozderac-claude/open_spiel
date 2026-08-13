"""Trains ONE adaptive-stopping MM arm calibrated to match bench_MM's own
150-sims/move training average (check_every=75, kl_eps=0.1 -- see
adaptive_calibrate_150.py), so the head-to-head against bench_MM is
compute-matched rather than confounded by a higher average search budget
(the issue found after the first 3-threshold sweep, which averaged
196-274 sims/move against bench_MM's 150).

    python run_adaptive_mm_150.py
"""
import os
import time

import connect4_dirichlet_utils as c4

CHECK_EVERY = 75
KL_EPS = 0.1
NUM_EPISODES = 2000
TAG = '150match'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'adaptive_150_sweep.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join('othello_benchmark', f'bench_MMadapt{TAG}'),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        fast_sims=100, full_sims=300, fast_prob=0.75,
        adaptive_kl_eps=KL_EPS, adaptive_check_every=CHECK_EVERY,
        temp_threshold=30, late_temp=8.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=0, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_warmup_eps=100, lr_decay_eps=30_000,
        lr_min_factor=0.1, weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    _log(f'start: check_every={CHECK_EVERY} kl_eps={KL_EPS} '
        f'episodes={NUM_EPISODES} (target ~150 sims/move, matching bench_MM)')
    t0 = time.time()
    cfg = make_cfg()
    try:
        c4.run_training(cfg, log=_log)
    except Exception as e:
        _log(f'FAILED: {type(e).__name__}: {e}')
        raise
    dt = time.time() - t0
    _log(f'done: {dt:.0f}s total, {dt / NUM_EPISODES:.2f} s/episode')
