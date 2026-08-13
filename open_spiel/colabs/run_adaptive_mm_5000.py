"""Extends the four existing adaptive-MM arms from 2000 to 5000 episodes, and
trains a fifth new arm (check_every=50, kl_eps=0.08, calibrated to mean sims
~100) from scratch to 5000. Sequential -- concurrent runs would contend for
the one GPU/CPU and distort per-episode timing.

Existing arms resume automatically: their checkpoint_dir already has
latest.pt at ep 2000 (resume=True), so raising num_episodes to 5000 just
continues past where run_adaptive_mm_sweep.py / run_adaptive_mm_150.py left
off. Checkpoints land every 1000 episodes (deep_eval_every), so bench_MM's
own checkpoints at 3000/4000/5000 (already on disk) stay directly comparable.

    python run_adaptive_mm_5000.py

This script is itself resumable if killed mid-arm: re-running it re-enters
run_training with resume=True for whichever arm was in progress and picks up
from its last saved episode.
"""
import os
import time

import connect4_dirichlet_utils as c4

# tag -> (checkpoint_dir_suffix, adaptive_check_every, adaptive_kl_eps)
ARMS = [
    ('005',      'bench_MMadapt005',      100, 0.005),
    ('010',      'bench_MMadapt010',      100, 0.010),
    ('030',      'bench_MMadapt030',      100, 0.030),
    ('150match', 'bench_MMadapt150match', 75,  0.1),
    ('100avg',   'bench_MMadapt100avg',   50,  0.08),
]
NUM_EPISODES = 5000
LOG_PATH = os.path.join(os.path.dirname(__file__), 'adaptive_5000_sweep.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(ckpt_dir, check_every, eps):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join('othello_benchmark', ckpt_dir),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        fast_sims=100, full_sims=300, fast_prob=0.75,
        adaptive_kl_eps=eps, adaptive_check_every=check_every,
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
    _log(f'sweep-to-5000 start: arms={[a[0] for a in ARMS]}')
    for tag, ckpt_dir, check_every, eps in ARMS:
        _log(f'--- arm {tag} (check_every={check_every} eps={eps}) starting ---')
        t0 = time.time()
        cfg = make_cfg(ckpt_dir, check_every, eps)
        try:
            c4.run_training(cfg, log=_log)
        except Exception as e:
            _log(f'--- arm {tag} FAILED: {type(e).__name__}: {e} ---')
            raise
        dt = time.time() - t0
        _log(f'--- arm {tag} segment done: {dt:.0f}s this segment ---')
    _log('sweep-to-5000 complete')
