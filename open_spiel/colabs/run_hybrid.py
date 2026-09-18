"""HYBRID: additive_mle SEARCH + avgconc TARGET -- the synthesis run.

Every result in this investigation points here. Search rule sets play strength
(sharp wins), target rule sets calibration, and they are separable. Decoupled
(additive_mle search + MIXTURE target) matched MM_long's evaluator and drew it
47-53 in a head-to-head -- but its target was mixture, which the backup study
found is the WORST rule (confidently wrong, worsens with training), and its
opening alpha0 duly collapsed to 0.15. avgconc was the study's best target: the
only rule good on BOTH proper likelihood AND discrimination, with a usable,
non-collapsing alpha0 (~7-23 vs mixture's floored 3).

So this is decoupled with the one field changed: target_agg mixture -> avgconc.
Prediction: decoupled's play strength (sharp additive_mle search, unchanged)
PLUS avgconc's honest, phase-tracking, non-collapsing calibration -- the two
things no single prior run had at once.

THE OPEN UNCERTAINTY. avgconc's calibration was measured collapsing trees built
WITH avgconc search. Here the tree is explored by additive_mle search, so the
avgconc collapse sees a differently-shaped tree. Its non-collapse comes from
being a plain AVERAGE of leaf concentrations (no variance amplification), which
should hold regardless of the search rule -- but that is the thing this run
tests. WATCH quick_report's `a0 open`: if it holds ~1-7 (as avgconc did) the
recipe works; if it collapses toward 0.15 (as mixture did under this same
sharp search in decoupled), the target rule is search-rule-dependent.

Everything else matches decoupled/avgconc: 10 blocks x 32 ch, 100/400 sims,
batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool, descent 1.0.

    python run_hybrid.py             # under run_hybrid_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 100000      # extended past 50k; resumes from the ep50000 ckpt.
LR_DECAY_EPS = 50000       # keep the ORIGINAL 50k cosine shape for 0-50k; the
                           # 50k-100k extension trains at the (raised) floor LR.
ROOT = 'othello_benchmark'
CKPT = 'bench_HYBRID'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'hybrid.log')


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
        lr_peak=2e-3, lr_warmup_eps=100,
        # 50k-100k extension floor: back to the FAITHFUL original, 0.02 (4e-5).
        # Both 2e-4 and 1e-4 degraded the opening value head at ep50k-60k on
        # nearly IDENTICAL trajectories (MSE_open ~0.77->~0.86-0.87 either way)
        # -- halving the LR barely changed it, so this isn't an LR-magnitude
        # problem. Reverted to ep50000 twice; this run isolates whether ANY
        # active LR does this (faithful floor also degrades) or whether 4e-5
        # is gentle enough to hold/improve (LR was the answer after all).
        lr_decay_eps=LR_DECAY_EPS, lr_min_factor=0.02,
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
    _log('HYBRID DONE')
