"""2x2 factorial over the two open levers, all with the allmse loss (decoupled
distillation: CE direction + MSE concentration-to-search-belief, + CE z + cons).

  lever 1  SEARCH rule:  pathsum (path-attenuated Thompson) | additive_mle
  lever 2  CONC target:  avgconc | pathavg (path-attenuated concentration)

Cells (bench_ALLMSE = pathsum x avgconc is already trained, this covers the
other three):
  admle_avgconc     additive_mle search, avgconc target
  pathsum_pathavg   pathsum search,      pathavg target
  admle_pathavg     additive_mle search, pathavg target

From scratch to 5000 eps, seed 0, lr_decay 50k -- matched to the existing
5000-ep checkpoints of allmse/ZGROUND/HYBRID/DISTILL for a fair tournament.

    python run_factorial.py <cell>
"""
import os
import sys
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 5000
LR_DECAY_EPS = 50000
ROOT = 'othello_benchmark'
ALLMSE_LOSS = {'dcev': 1.0, 'dcea': 2.0, 'mkcv': 0.3, 'mkca': 0.3,
               'cev': 1.0, 'cea': 1.0, 'cons': 0.5}

CELLS = {
    'admle_avgconc':   dict(search=c4.AGG_ADDITIVE_MLE, target=c4.AGG_AVGCONC),
    'pathsum_pathavg': dict(search=c4.AGG_PATHSUM,       target=c4.AGG_PATHAVG),
    'admle_pathavg':   dict(search=c4.AGG_ADDITIVE_MLE, target=c4.AGG_PATHAVG),
}

CELL = sys.argv[1]
spec = CELLS[CELL]
CKPT = f'bench_ALLMSE_{CELL.upper()}'
TAG = f'ALLMSE_{CELL.upper()}'
LOG_PATH = os.path.join(os.path.dirname(__file__), f'fact_{CELL}.log')


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
        search_agg=spec['search'], target_agg=spec['target'],
        selection='dirichlet', kl_normalize=False,
        backup='mean',
        fast_sims=100, full_sims=400, fast_prob=0.6,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.0, random_pool_frac=0.0,
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        loss_weights=dict(ALLMSE_LOSS),
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=LR_DECAY_EPS, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='')


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== {TAG}: {NUM_EPISODES} eps, search={cfg.search_agg} '
         f'target={cfg.target_agg} loss=allmse -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log(f'{TAG} DONE')
