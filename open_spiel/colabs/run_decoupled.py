"""Decoupled aggregation: additive_mle for SEARCH, mixture for TARGETS.

The best-of-both-worlds arm.  The two aggregation rules do different jobs and
want different things:

  SEARCH (additive_mle)   Thompson selection samples the per-action Dirichlets,
                          so it wants SHARP concentration -- additive_mle's
                          alpha0 tracks the visit count, which keeps selection
                          decisive and is why MM_long's search played strongly.
                          Mixture-for-search gave lower alpha0 (~14 vs ~35 on a
                          trained net), pushing selection toward the flat/corner
                          regime; the pure-mixture run lost the head-to-head
                          96.5% to 3.5%, largely on near-random opening play.

  TARGETS (mixture)       The TRAINED network learns from these, and mixture is
                          the only rule whose alpha0 means "how sure am I",
                          because its variance is within-leaf plus between-leaf
                          (law of total variance) rather than additive_mle's
                          agreement-among-means.  It is what produced the
                          correctly-ordered confidence profile: low where the
                          net is near chance, high where it is reliable.

So: search stays sharp (play strength), targets stay honest (calibration).
This is the split the earlier experiments kept pointing at -- the pure-mixture
runs proved mixture targets calibrate well but mixture SEARCH weakens play, and
MM_long proved additive_mle search plays well but its targets never calibrate.
This run takes the good half of each.

NO artificial floor.  The target_conc_floor stays at its historical 0.15.  The
tfloor3 experiment showed a floor of 3 works and costs nothing measurable, but
it is a hack; mixture targets under additive_mle SEARCH do not collapse the way
pure-mixture did, because the search that generates the tree statistics is the
sharp rule, so the between-leaf spread feeding the mixture target is wider.
Whether that is enough to keep opening targets off the 0.15 floor on its own is
one of the things this run measures -- watch quick_report's `a0 open`.

Everything else matches run_mixture.py: 10 blocks x 32 ch, 100/400 sims,
batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool, descent_temp 1.0.

    python run_decoupled.py          # under run_decoupled_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_DECOUPLED'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'decoupled.log')


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
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_MIXTURE,
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
    _log(f'=== MIXTURE agg: {NUM_EPISODES} eps, {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch, {cfg.fast_sims}/{cfg.full_sims} sims '
         f'(mean {cfg.fast_prob*cfg.fast_sims + (1-cfg.fast_prob)*cfg.full_sims:.0f})'
         f' -> {ROOT}/{CKPT} ===')
    _log(f'    SEARCH={cfg.search_agg} TARGET={cfg.target_agg} '
         f'backup={cfg.backup} target_floor={cfg.target_conc_floor:g} '
         f'descent_temp={cfg.descent_temp:g} | '
         f'lr {cfg.lr_peak:.0e} -> cosine over {cfg.lr_decay_eps} -> floor '
         f'{cfg.lr_peak*cfg.lr_min_factor:.1e} | batch {cfg.batch_size} x '
         f'{cfg.train_steps_per_ep} step/ep | buffer {cfg.max_buffer//1000}k | '
         f'temp_threshold {cfg.temp_threshold}, greedy after | '
         f'pool {cfg.pool_prob} | elo {cfg.eval_games_per_pair} games/pair '
         f'vs all previous')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('MIXTURE DONE')
