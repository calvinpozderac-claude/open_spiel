"""Naive average-concentration aggregation for BOTH search and targets.

The target Dirichlet is built from the accumulator as:
  direction     = mean of the leaf mean-predictions   (SM / n, equal-weighted)
  concentration = mean of the leaf concentrations      (SA.sum() / n)
  alpha         = direction * concentration

Why this might capture what the other rules could not.  It is an AVERAGE of
leaf concentrations, so it cannot grow with visit count the way additive_mle
does (that reads alpha0 ~ the sim budget, ~400 at every phase).  And it is a
plain mean, so disagreement between leaf MEANS moves only the direction, never
the averaged concentration -- so it cannot suffer mixture's variance-driven
collapse.  Measured on the (already-compressed) decoupled net it rises cleanly
opening->endgame: 0.15 -> 57 as proven-terminal leaves (alpha0=100) enter the
average, while additive_mle stays flat at 400 throughout.

THE OPEN QUESTION, and what to watch.  On a FRESH net the opening has no
terminal leaves, so the opening target is just the mean of the net's own
opening leaves -- a neutral fixed point.  There is no restoring force there
(terminals do not reach the opening) and no amplification either, so the
opening alpha0 should random-walk around its init (~1.6) rather than actively
collapse.  WATCH quick_report's `a0 open`: if it holds ~1.5-3, the rule works
and the opening is honestly-uncertain-but-not-corner-peaked; if it drifts to
0.15, the martingale wandered into the floor and the rule needs an anchor.

Everything else matches the mixture/decoupled runs: 10 blocks x 32 ch,
100/400 sims, batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool,
descent_temp 1.0.

    python run_avgconc.py            # under run_avgconc_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_AVGCONC'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'avgconc.log')


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
        search_agg=c4.AGG_AVGCONC, target_agg=c4.AGG_AVGCONC,
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
    _log('AVGCONC DONE')
