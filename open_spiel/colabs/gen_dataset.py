"""Generate a frozen self-play dataset for the offline loss lab.

Forks bench_HYBRID at a chosen generation into a scratch checkpoint dir,
then runs the normal self-play + training loop with lr ~ 0 (the net stays
fixed; the machinery is untouched) and dumps the replay buffer at the end.
Search uses the study-1 winner (pathsum) and targets collapse under avgconc
(study 2's best-ordered concentration), so the dataset is what the proposed
scheme would actually train on.

Two datasets, two regimes:
    python gen_dataset.py strong    # HYBRID@50000, 2000 eps (~240k samples)
    python gen_dataset.py weak      # HYBRID@10000, 1500 eps (~180k samples)
The weak one matters most: early-training data quality is where the
loss-function differences (mean-learning speed) showed up in the EDL runs.
"""
import os
import shutil
import sys
import time

import connect4_dirichlet_utils as c4

ROOT = 'othello_benchmark'
SRC = f'{ROOT}/bench_HYBRID'
SETS = {
    'strong': dict(gen=50000, episodes=2000),
    'weak':   dict(gen=10000, episodes=1500),
}
LOG_PATH = os.path.join(os.path.dirname(__file__), 'gen_dataset.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def main(which):
    spec = SETS[which]
    gen = spec['gen']
    d = f'{ROOT}/dataset_{which}_src'
    os.makedirs(d, exist_ok=True)
    shutil.copy(f'{SRC}/bench_{gen}.pt', f'{d}/bench_{gen}.pt')
    c4.rewind_to_generation(d, gen, log=_log)
    out = f'{ROOT}/dataset_{which}.pkl'

    cfg = c4.Config(
        game_name='othello',
        checkpoint_dir=d,
        num_episodes=gen + spec['episodes'],
        channels=32, num_blocks=10, head_ch=8, seed=3,
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
        batch_size=1024, train_steps_per_ep=1,
        max_buffer=400_000,
        loss_weights=(1.0, 1.0, 0.5),
        edl_lambda=1.0, edl_anneal_eps=10_000,
        lr_peak=1e-12, lr_warmup_eps=1,          # net effectively frozen
        lr_decay_eps=50000, lr_min_factor=1.0,
        weight_decay=0.0, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=2,
        deep_eval_every=10 ** 9, eval_sims=64,
        eval_games_per_pair=2, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='',
        dump_buffer_path=out,
    )
    _log(f'=== DATASET {which}: HYBRID@{gen} frozen, '
         f'{spec["episodes"]} eps, search=pathsum target=avgconc -> {out} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== dataset {which} done in {time.time()-t0:.0f}s ===')
    _log(f'DATASET {which.upper()} DONE')


if __name__ == '__main__':
    main(sys.argv[1])
