"""OLIVAW scale, but sampled backup and a flat in-tree descent.

Two changes from run_olivaw.py, both aimed at the inverted confidence profile:
the value head is most concentrated in the OPENING (alpha0 ~110 at 45-60
empties) and least in the early endgame (~12 at 15-24), which is backwards.

  descent_temp 16 -> 1   Sweeping descent temperature on a fixed net moved
                         opening target alpha0 79.6 -> 120.4 from temp 1 to 64
                         while the endgame stayed flat at ~21. A sharp descent
                         visits fewer distinct lines, so the backed-up means
                         agree more and additive_mle fits a larger alpha0.
                         Cost: 328901cb measured training strength peaking at
                         dt16, with dt4 already -23 Elo against the coupled
                         baseline, so this is expected to cost playing strength.

  backup mean -> sample  A leaf now contributes a draw from Dir(alpha) instead
                         of alpha's mean, so agreement between lines costs
                         something. Under mean backup alpha0 measures agreement
                         among estimates rather than certainty about the
                         outcome, which is why the opening -- where every line
                         looks like a draw -- scores highest.

Measured on the mean-trained ep15000 net, the two together take target alpha0
from 113/26 (opening/early-end) to 37/6. That net's profile does NOT predict
this run's: sampling from a posterior only reproduces the confidence already
baked into it, and this run never passes through that state. What it does show
is the absolute collapse, which is the point of interest -- TERMINAL_CONC is
100, so solved endgames now dominate the concentration budget instead of
competing with an opening inflated to the same level.

WHAT COULD GO WRONG, and what to watch. Sampled backup is positive feedback in
both directions: wide draws -> low target alpha0 -> the net learns low alpha0
-> wider draws. Selection here is Thompson sampling over these very
distributions, so a run that settles at uniformly low concentration searches
almost at random. That failure shows up as weak play long before it shows up as
bad calibration, so the ep5000 Edax benchmark is the early warning: run_olivaw
scored L1 95% there, and a sampled run far below that is the attractor, not
slow learning. Terminal spikes are the only force pulling the other way.

Everything else matches run_olivaw.py: 10 blocks x 32 ch, 100/400 sims,
batch 1024, MM_long's 2e-3 cosine, buffer 300k, no frozen-opponent pool,
16 games/pair against every previous checkpoint.

    python run_sample.py        # under run_sample_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_SAMPLE'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'sample.log')


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
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        backup='sample',                  # leaves contribute Dir(alpha) draws
        fast_sims=100, full_sims=400, fast_prob=0.6,
        adaptive_mode=None, adaptive_kl_eps=None,
        temp_threshold=20, late_temp=1e4,
        descent_temp=1.0,                 # flat in-tree descent
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
    _log(f'=== SAMPLE backup: {NUM_EPISODES} eps, {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch, {cfg.fast_sims}/{cfg.full_sims} sims '
         f'(mean {cfg.fast_prob*cfg.fast_sims + (1-cfg.fast_prob)*cfg.full_sims:.0f})'
         f' -> {ROOT}/{CKPT} ===')
    _log(f'    backup={cfg.backup} descent_temp={cfg.descent_temp:g} | '
         f'lr {cfg.lr_peak:.0e} -> cosine over {cfg.lr_decay_eps} -> floor '
         f'{cfg.lr_peak*cfg.lr_min_factor:.1e} | batch {cfg.batch_size} x '
         f'{cfg.train_steps_per_ep} step/ep | buffer {cfg.max_buffer//1000}k | '
         f'temp_threshold {cfg.temp_threshold}, greedy after | '
         f'pool {cfg.pool_prob} | elo {cfg.eval_games_per_pair} games/pair '
         f'vs all previous')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('SAMPLE DONE')
