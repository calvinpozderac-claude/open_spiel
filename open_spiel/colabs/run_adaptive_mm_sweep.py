"""Trains the three calibrated adaptive-stopping thresholds on the MM arm
(Othello), 2000 episodes each, sequentially -- concurrent runs would contend
for the one GPU/CPU and distort the per-episode timing this experiment cares
about (same reasoning as connect4_benchmark.train_all).

Matches othello_benchmark's bench_MM run as closely as possible (same trunk
size, search/target aggregation, sim cap, LR schedule -- read back from that
run's own saved `latest.pt['cfg']` rather than assumed) so the comparison
against the existing bench_MM/bench_{1000,2000}.pt checkpoints is apples to
apples; the only things that differ are adaptive_kl_eps and the checkpoint
directory.

    python run_adaptive_mm_sweep.py

Run from open_spiel/colabs (relative checkpoint paths match the rest of the
benchmark tooling). Logs go to adaptive_sweep.log next to this file.
"""
import os
import time

import connect4_dirichlet_utils as c4

THRESHOLDS = {'005': 0.005, '010': 0.010, '030': 0.030}
NUM_EPISODES = 2000
LOG_PATH = os.path.join(os.path.dirname(__file__), 'adaptive_sweep.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(tag, eps):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join('othello_benchmark', f'bench_MMadapt{tag}'),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        fast_sims=100, full_sims=300, fast_prob=0.75,
        adaptive_kl_eps=eps, adaptive_check_every=100,
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
    _log(f'sweep start: thresholds={THRESHOLDS} episodes={NUM_EPISODES}')
    for tag, eps in THRESHOLDS.items():
        _log(f'--- arm kl_eps={eps} (tag={tag}) starting ---')
        t0 = time.time()
        cfg = make_cfg(tag, eps)
        try:
            c4.run_training(cfg, log=_log)
        except Exception as e:
            _log(f'--- arm {tag} FAILED: {type(e).__name__}: {e} ---')
            raise
        dt = time.time() - t0
        _log(f'--- arm {tag} done: {dt:.0f}s total, '
            f'{dt / NUM_EPISODES:.2f} s/episode ---')
    _log('sweep complete')
