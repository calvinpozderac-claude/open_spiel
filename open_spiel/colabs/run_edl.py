"""EDL: evidential deep learning losses, trained FROM SCRATCH.

Same config as run_hybrid.py (10 blocks x 32 ch, additive_mle search +
avgconc target, 100/400 sims, batch 1024 x 1 step/ep, lr 2e-3 cosine over
50000, buffer 300k, no pool, descent 1.0, seed 0) with ONE change: the loss.
klv/kla/cev/cea are replaced by the evidential pair per head (risk +
evidence regulariser) -- see connect4_dirichlet_utils.py's module header and
full_loss.

Unlike run_p1_edl.py (which fine-tuned bench_HYBRID from ep40000, so its mean
head was already trained and edl_lambda saturated at 1.0 from step one), this
run starts untrained, so `edl_anneal_eps` actually does its job: lambda ramps
0 -> 1 over the first 10000 episodes, letting the mean prediction settle
before the evidence regulariser starts taxing wrong-class concentration.
This is the unconfounded read on whether the concentration calibration holds
when the whole network is shaped by this loss from the start.

    python run_edl.py             # under run_edl_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_EDL'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'edl.log')


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
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_AVGCONC,
        selection='dirichlet', kl_normalize=False,
        backup='mean',
        fast_sims=100, full_sims=400, fast_prob=0.6,
        adaptive_mode=None, adaptive_kl_eps=None,
        temp_threshold=20, late_temp=1e4, descent_temp=1.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.0, random_pool_frac=0.0,
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        loss_weights=(1.0, 1.0, 0.5),   # evidential (evv, eva, cons)
        edl_lambda=1.0, edl_anneal_eps=10_000,
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=NUM_EPISODES, lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000, eval_sims=64,
        eval_games_per_pair=16, eval_last_n=999, eval_refresh_pairs=0,
        resume=True, solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== EDL from scratch: {NUM_EPISODES} eps, {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch, {cfg.fast_sims}/{cfg.full_sims} sims -> '
         f'{ROOT}/{CKPT} ===')
    _log(f'    loss=evidential weights={cfg.loss_weights} '
         f'edl_lambda={cfg.edl_lambda} anneal_eps={cfg.edl_anneal_eps} | '
         f'SEARCH={cfg.search_agg} TARGET={cfg.target_agg} | '
         f'lr {cfg.lr_peak:.0e} -> cosine over {cfg.lr_decay_eps} | '
         f'batch {cfg.batch_size} x {cfg.train_steps_per_ep} step/ep | '
         f'quick {cfg.quick_eval_every} deep {cfg.deep_eval_every}')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('EDL DONE')
