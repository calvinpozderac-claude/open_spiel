"""MM_long: the original MM configuration, 50000 episodes at 1000 sims/move.

MM is the configuration that survived this project's measurement problems. On
the grid sweep MM@20000 beat OPT@20000 in 15 of 15 cells above 64 sims, and
MM@30000 is the strongest network we have against strong opposition. So the
long run is built on MM rather than on any of the arms whose advantages turned
out to be artifacts of a low search budget, a weak opponent, or sibling
tournaments.

Two changes from the original MM run:

  1000 sims/move      MM trained on a 100/300 mix, mean ~150.  A fixed 1000
                      removes the budget mismatch that appears to have hurt
                      OPT: it trained under a ~150-sim regime and its
                      advantage evaporated the moment it was given more
                      search.  Cost is real -- search is ~6.7x the original,
                      throughput drops from 0.46 to roughly 0.07 games/s, so
                      50000 episodes is on the order of 8-9 days.

  annealed schedule   lr_decay_eps = num_episodes so the cosine actually
                      completes, and a 2% floor rather than 10%.  The original
                      MM used a 30000 horizon over 40000 episodes, which
                      annealed at ep30000 and then trained 10000 more episodes
                      at the floor -- and MM@40000 scored worse than MM@30000.
                      Ending AT the floor avoids that tail.

Checkpoints every 5000 episodes for mm_long_bench.py to score.

    python run_mm_long.py       # under a restart wrapper
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
SIMS = 1000
ROOT = 'othello_benchmark'
CKPT = 'bench_MM_long'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'mm_long.log')


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
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        # Fixed 1000 sims: both arms of the coin flip are the same value, so
        # fast_prob is irrelevant.
        fast_sims=SIMS, full_sims=SIMS, fast_prob=1.0,
        # MM's defining choice: no adaptive allocation, no descent_temp.
        adaptive_mode=None, adaptive_kl_eps=None,
        temp_threshold=30, late_temp=8.0, descent_temp=None,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=NUM_EPISODES,        # cosine completes at the end
        lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000,             # one benchmarkable checkpoint / 5000
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    _log(f'=== MM_long: {NUM_EPISODES} eps @ {SIMS} sims/move -> '
         f'{ROOT}/{CKPT} ===')
    _log(f'    lr 2e-3 -> cosine over {NUM_EPISODES} -> floor 4e-05; '
         f'no adaptive allocation, no descent_temp (stock MM)')
    t0 = time.time()
    c4.run_training(make_cfg(), log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('MM_LONG DONE')
