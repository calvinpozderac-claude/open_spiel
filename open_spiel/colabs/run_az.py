"""Standard AlphaZero baseline for the tournament -- the ordinary algorithm.

Uses connect4_alphazero_utils (policy logits + tanh scalar value, PUCT search,
visit-count policy CE + value MSE loss, root Dirichlet noise), held as close as
possible to the current runs: same trunk (32ch x 10blk x 8head), same othello
game and observation, same self-play shape (4 workers x 16 games, 100/400 sims,
fast_prob 0.6, temp_threshold 20, no frozen pool), same optimiser/LR
(2e-3 cosine over 50k, wd 1e-4), same batch 1024 x 1 step/ep, buffer 300k,
5000 episodes, seed 0.  Faithful-AlphaZero specifics kept at standard values:
c_puct 1.5, root_noise_frac 0.25, root_noise_alpha 1.0, value_weight 1.0.

    python run_az.py
"""
import os
import time

import connect4_alphazero_utils as az

NUM_EPISODES = 50000       # extended from 5000; resumes from the ep5000 ckpt.
ROOT = 'othello_benchmark'  # lr_decay_eps is already 50000, so the cosine is
CKPT = 'bench_AZ_MATCH'     # continuous -- this is one 50k run, not a restart.
LOG_PATH = os.path.join(os.path.dirname(__file__), 'az_match.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return az.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(ROOT, CKPT),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=10, head_ch=8, seed=0,
        device_preference='auto',
        c_puct=1.5, virtual_loss=1.0,
        root_noise_frac=0.25, root_noise_alpha=1.0,   # standard AlphaZero
        fast_sims=100, full_sims=400, fast_prob=0.6,
        temp_threshold=20,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        max_plies=128, eval_max_plies=128,
        pool_prob=0.0, random_pool_frac=0.0,           # match current runs
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=50000, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0, value_weight=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='')


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== AZ_MATCH (standard AlphaZero): {NUM_EPISODES} eps, '
         f'{cfg.num_blocks}x{cfg.channels}, c_puct={cfg.c_puct}, '
         f'{cfg.fast_sims}/{cfg.full_sims} sims -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    az.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('AZ_MATCH DONE')
