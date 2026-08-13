"""Four new arms, all trained fresh to 5000 episodes, checkpointing every
1000 (matching bench_MM's own cadence):

  fixed500    -- 500 sims every move (fixed, non-adaptive)
  mix300_500  -- fixed fast/full split: 300 sims 75% of moves, 500 sims 25%
                 (average 350, same STRUCTURE as bench_MM's own 100/300 mix,
                 just shifted up -- a fixed-budget point of comparison for
                 the adaptive arm at the same average)
  adapt350    -- adaptive stopping calibrated to ~350 avg sims/move
                 (eps=0.005, check_every=100; see
                 adaptive_calibrate_350_500.py -- measured mean 352.5)
  adapt500    -- adaptive stopping calibrated to ~500 avg sims/move
                 (eps=0.001, check_every=100; measured mean 507.2)

Both adaptive arms use a raised cap (max_simulations=1000) since 500 avg
sims/move needs headroom above bench_MM's original 300 ceiling.

    python run_adaptive_mm_500family.py

Run via Scheduled Task (see the wrapper pattern in
run_adaptive_mm_5000_forever.ps1) rather than a plain background shell
command -- that's what actually survives unattended for this long.
"""
import os
import time

import connect4_dirichlet_utils as c4

# tag -> (checkpoint_dir_suffix, fast_sims, full_sims, fast_prob,
#         adaptive_check_every, adaptive_kl_eps)
ARMS = [
    ('fixed500',   'bench_MMfixed500',   500, 500, 0.0,  None, None),
    ('mix300_500', 'bench_MMmix300_500', 300, 500, 0.75, None, None),
    ('adapt350',   'bench_MMadapt350',   100, 1000, 0.75, 100, 0.005),
    ('adapt500',   'bench_MMadapt500',   100, 1000, 0.75, 100, 0.001),
]
NUM_EPISODES = 5000
LOG_PATH = os.path.join(os.path.dirname(__file__), 'adaptive_500family_sweep.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(ckpt_dir, fast_sims, full_sims, fast_prob, check_every, eps):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join('othello_benchmark', ckpt_dir),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        fast_sims=fast_sims, full_sims=full_sims, fast_prob=fast_prob,
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
    _log(f'500family sweep start: arms={[a[0] for a in ARMS]}')
    for tag, ckpt_dir, fast_sims, full_sims, fast_prob, check_every, eps in ARMS:
        _log(f'--- arm {tag} (fast={fast_sims} full={full_sims} '
            f'fast_prob={fast_prob} check_every={check_every} eps={eps}) '
            f'starting ---')
        t0 = time.time()
        cfg = make_cfg(ckpt_dir, fast_sims, full_sims, fast_prob, check_every, eps)
        try:
            c4.run_training(cfg, log=_log)
        except Exception as e:
            _log(f'--- arm {tag} FAILED: {type(e).__name__}: {e} ---')
            raise
        dt = time.time() - t0
        _log(f'--- arm {tag} segment done: {dt:.0f}s this segment ---')
    _log('500family sweep complete')
