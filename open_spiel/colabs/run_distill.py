"""DISTILL arm: pure search-belief distillation, the loss-lab winner.

Loss = klv + kla + cons (NO grounded-outcome cev/cea/evidential terms).
Search = pathsum (study-1 winner), targets collapsed under avgconc.
From scratch, 50000 eps, otherwise run_hybrid.py's config.

Paired with run_zground.py (identical except it adds cev+cea) to answer the
question the OFFLINE loss lab could not: in a real training loop, where the
distillation targets come from search on the CURRENT net, does dropping the
z-grounding cause the self-referential drift AlphaZero used z to avoid, or is
the ~50% solver-labelled buffer + terminal backups grounding enough?  Compare
the two on the Edax probe (compare_runs.py) and head-to-head Elo.

    python run_distill.py             # under run_distill_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 10000       # capped at 10k for a fast head-to-head vs ZGROUND
LR_DECAY_EPS = 50000       # keep the ORIGINAL 50k cosine so the schedule the
                           # ep1-7000 checkpoints already saw stays continuous
ROOT = 'othello_benchmark'
CKPT = 'bench_DISTILL'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'distill.log')


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
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cons': 0.5},
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
    _log(f'=== DISTILL from scratch: {NUM_EPISODES} eps, '
         f'loss={cfg.loss_weights} search={cfg.search_agg} '
         f'target={cfg.target_agg} -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('DISTILL DONE')
