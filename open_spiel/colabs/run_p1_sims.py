"""Plateau study Phase 1 -- SIMS arm (targets H4: MCTS target-label noise floor).

Same resume point as run_p1_control.py (bench_HYBRID @ ep40000), same 8000
episodes, ONE field changed: full_sims 400 -> 1000 (mean self-play sim budget
220 -> 460). If the plateau is a target-label noise ceiling -- the net has
already learned everything a 100/400-sim search can teach it -- sharper
targets should reopen the probe-MSE decline relative to the control.
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 48000
ROOT = 'othello_benchmark'
CKPT = 'bench_HYBRID_p1_sims'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'p1_sims.log')


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
        channels=32, num_blocks=10, head_ch=8, seed=1,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_AVGCONC,
        selection='dirichlet', kl_normalize=False,
        backup='mean',
        fast_sims=100, full_sims=1000, fast_prob=0.6,     # <-- changed: 400->1000
        adaptive_mode=None, adaptive_kl_eps=None,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.0, random_pool_frac=0.0,
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=50000, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== P1 SIMS: resume ep40000 -> {NUM_EPISODES}, full_sims 400->1000 '
         f'(mean sims 220->460) -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('P1SIMS DONE')
