"""ZGROUND arm: distillation PLUS z-grounding — the control for run_distill.py.

Loss = klv + kla + cev + cea + cons (the original AlphaZero-style pairing:
distill the search beliefs AND ground the mean on the real game outcome z).
Identical to run_distill.py in every other respect (search=pathsum,
target=avgconc, from scratch, 50000 eps, same seed).

The single variable between the two runs is the presence of cev+cea.  If
DISTILL matches or beats this, the z-grounding is unnecessary here (the
solver-labelled buffer + terminal backups anchor enough); if ZGROUND pulls
ahead over the long run, the offline loss lab missed the self-referential
drift that grounding prevents.

    python run_zground.py             # under run_zground_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 10000       # capped at 10k for a fast head-to-head vs DISTILL
LR_DECAY_EPS = 50000       # match DISTILL's schedule exactly (same 50k cosine)
ROOT = 'othello_benchmark'
CKPT = 'bench_ZGROUND'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'zground.log')


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
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
                      'cons': 0.5},
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
    _log(f'=== ZGROUND from scratch: {NUM_EPISODES} eps, '
         f'loss={cfg.loss_weights} search={cfg.search_agg} '
         f'target={cfg.target_agg} -> {ROOT}/{CKPT} ===')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('ZGROUND DONE')
