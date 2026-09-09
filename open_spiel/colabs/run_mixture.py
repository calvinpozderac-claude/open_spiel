"""Mixture aggregation for BOTH search and targets, at OLIVAW scale.

Every run in this repo uses additive_mle for both.  Mixture has never been run
past 2000 episodes at 70k params (commit 132c6506), and never at 10 blocks.
It was dropped as the default in f4f760b0 for one specific reason -- "about 54%
of unsolved state targets were landing on the _floor_conc floor, carrying a
direction but no scale" -- and that defect no longer reproduces: measured on
40 unsolved positions at 200 sims, 0% floor both untrained and on OLIVAW
ep20000.  It was never beaten head to head; the only A/B on record had mixture
beating 'mean' targets by +143 Elo search-free.

WHY MIXTURE.  additive_mle fits a concentration to the SPREAD of the backed-up
means, so alpha0 measures agreement among estimates rather than certainty about
the outcome.  Uninformed leaves agree with each other, so an untrained net gets
alpha0 ~ 110 in the opening for a position it has never seen.  Mixture instead
matches the moments of the mixture distribution -- ex2 = w @ (var_c + m*m), so
the variance is WITHIN-leaf plus BETWEEN-leaf, the law of total variance -- and
therefore carries each leaf's own uncertainty into the total.

Measured discrimination, solved / unsolved target alpha0 at 200 sims:

                       untrained net        trained net (alpha0 103)
    additive_mle            1.0x                    1.7x
    mixture                57.5x                    3.7x

and its target MEAN is the cleanest of any rule tried: sd 0.00081 across
repeat searches, against 0.00109 for additive_mle and 0.01250 for sampled
backup, which is what made sampling collapse the search.

CONFIGURATION.  Identical to run_olivaw.py except the aggregation, so the
comparison isolates it: 10 blocks x 32 ch, 100/400 sims, batch 1024, MM_long's
2e-3 cosine, buffer 300k, no frozen-opponent pool, 16 games/pair against every
previous checkpoint.  descent_temp is 1.0: an unsharpened descent is the
principled default, and sharpening it is precisely what inflates opening
confidence -- holding the net fixed and varying only the descent temperature
moved opening target alpha0 from 79.6 at temp 1 to 120.4 at temp 64, while the
endgame did not move at all.  The dt16 result in 328901cb is 2000-episode arms
on a single seed scored on Elo alone; it is not a reason to bias the search.

kl_normalize stays False: the Config's own note says to leave it off for
mixture, whose concentration is bounded, and turn it on only for 'mean'/'sum'.

Also included, and new since every earlier run: the output heads initialise
with noise rather than zeros (zero weights made every position bit-identical,
which alone inflated the untrained mean-backup target from 110 to 205), and
_conf no longer clamps concentration from below (the clamp was an absorbing
state with exactly zero gradient).

WHAT TO WATCH.  Mixture's unsolved alpha0 is lower than additive_mle's (13.7 vs
34.7 on a trained net), and Thompson selection samples those distributions --
the same direction that broke the sampled run, though at 13.7 rather than the
0.2 that made selection random.  If the Elo pool and the self-play gate stall
while the probe keeps improving, that is the signature.

    python run_mixture.py       # under run_mixture_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_MIXTURE'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'mixture.log')


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
        search_agg=c4.AGG_MIXTURE, target_agg=c4.AGG_MIXTURE,
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
    _log(f'    search_agg={cfg.search_agg} target_agg={cfg.target_agg} '
         f'backup={cfg.backup} descent_temp={cfg.descent_temp:g} | '
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
