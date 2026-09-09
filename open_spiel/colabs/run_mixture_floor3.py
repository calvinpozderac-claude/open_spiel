"""Original MIXTURE config, but with a hard FLOOR of 3 under the concentration.

The first mixture run collapsed in the opening: 95% of unsolved opening targets
landed on the _floor_conc floor (0.15) and the network learned to sit there.
At alpha0 = 0.15 over three classes each alpha_i is ~0.05, deep in the
corner-peaked regime, so Thompson selection in the opening was drawing from a
density that piles mass at the simplex corners -- effectively random opening
play, which is the likeliest reason it lost the head-to-head 96.5% to 3.5%.

Two changes, both aimed at that:

  conf_offset = 3.0     _conf now returns 3 + softplus(raw), so alpha0 can
                        never fall below 3.  This is not an arbitrary number:
                        alpha0 = 3 over three classes means every alpha_i = 1,
                        which is EXACTLY the uniform density on the simplex --
                        the critical point between corner-peaked (alpha_i < 1)
                        and interior-mode (alpha_i > 1).  The floor guarantees
                        the search never samples from a corner-peaked belief.
                        It is added, not clamped, so the gradient stays alive
                        everywhere (verified: d/draw = sigmoid(raw) throughout).

  head_conf_bias = -0.4328   softplus(-0.4328) = 0.5, so the net STARTS at
                        alpha0 = 3.5 -- just above the floor rather than on top
                        of it, which leaves room to move down toward 3 without
                        the gradient dying and up freely.

WHAT THIS DOES NOT FIX.  The feedback loop is still present: mixture variance
is within-leaf plus between-leaf, within-leaf is m(1-m)/(alpha0+1), so a
falling alpha0 raises the variance and lowers the fitted beta0.  The floor
bounds where that loop can end up; it does not remove it.  If the opening
alpha0 pins at exactly 3.00 the loop has simply found the new floor, and the
question becomes whether a floored-but-uniform belief searches better than a
floored-but-corner-peaked one.  That is the actual experiment.

Everything else matches run_mixture.py exactly: 10 blocks x 32 ch, 100/400
sims, batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool,
descent_temp 1.0, mixture for both search and target.

    python run_mixture_floor3.py     # under run_mixture_floor3_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_MIXTURE_floor3'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'mixture_floor3.log')


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
        conf_offset=3.0,          # alpha0 floored at 3 -> every alpha_i >= 1
        head_conf_bias=-0.4328,   # softplus(-0.4328)=0.5 -> starts at 3.5
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
         f'backup={cfg.backup} | ALPHA0 FLOOR {cfg.conf_offset:g} '
         f'(starts at {cfg.conf_offset + 0.5:.2f}, bias {cfg.head_conf_bias:g}) | '
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
