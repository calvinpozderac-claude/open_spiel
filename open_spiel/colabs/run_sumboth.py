"""SUM aggregation for BOTH search and targets -- expected overconfident, run to see.

sum returns alpha = elementwise SUM of the backed-up leaf alpha vectors (SA), the
Dirichlet-conjugate accumulation of the leaves as soft pseudo-counts.  Its
concentration GROWS with visit count (10 leaves -> alpha0 60, 200 -> 1200), so
as a TARGET it is expected to be overconfident everywhere -- the same
calibration failure as additive_mle, whose alpha0 is also ~ the visit count.

But sum was the STRONGEST deploy-time search rule in the sweep: the avgconc-
trained net searched with sum beat MM_long 63-37, where avgconc search scored
27.5%.  Sharp, fast-growing concentration makes selection decisive.  This run
uses sum for BOTH, so self-play gets that sharp search AND the head trains on
sum targets.  The open question is whether the overconfident-but-sharp target
still produces a strong player -- maybe the calibration cost does not matter for
strength, which is the whole point of trying it.

WATCH: target alpha0 could reach the thousands in resolved endgames (no upper
cap), so the value-KL may run large and spiky; _assert_finite_sd guards against
an actual NaN.  And quick_report's `a0 open` is expected to be HIGH (overconfident
opening), the opposite of avgconc -- that is the predicted outcome, not a failure.

Everything else matches the avgconc/decoupled runs: 10 blocks x 32 ch, 100/400
sims, batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool, descent 1.0.

    python run_sumboth.py            # under run_sumboth_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_SUMBOTH'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'sumboth.log')


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
        search_agg=c4.AGG_SUM, target_agg=c4.AGG_SUM,
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
    _log('SUMBOTH DONE')
