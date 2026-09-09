"""Original MIXTURE config with the concentration floor on the TARGET, not the output.

The previous attempt (run_mixture_floor3.py, conf_offset=3) welded the head to
3.00 within 1000 episodes and never moved again: ratio 1.00, no phase
discrimination, targets at 57.7 with the prediction stuck at 3.0.

The mechanism, measured rather than guessed.  Early in training the network's
means are wrong, so the KL correctly pushes concentration DOWN -- at init with a
sharp target and a uniform prediction the gradient is +0.18 (downward), and with
an outright wrong mean it is +0.57.  That dip is normal and every run has it.
What differs is the escape.  Under pure softplus the Dirichlet KL DIVERGES like
1/conf as concentration falls, and that divergence cancels the vanishing
sigmoid(raw) to give a constant restoring gradient of about -2.0 from
arbitrarily deep, so the head always comes back once the means improve.  Adding
a constant floor to the OUTPUT bounds the KL, kills the divergence, and leaves
sigmoid(raw) -> 0 with nothing opposing it: a one-way door.

So the floor moves to the target.  _conf stays pure softplus (conf_offset back
to 0, head_conf_bias back to 1.4) and _floor_conc's minimum total rises from
0.15 to 3.0.  _floor_conc SCALES the vector rather than clamping components, so
a weak target keeps its direction and only gains scale: [0.30, 0.12, 0.18]
(alpha0 0.60) becomes [1.50, 0.60, 0.90] (alpha0 3.0).  Targets already above
the floor are untouched.

Why 3: alpha0 = 3 over three classes means every alpha_i = 1, exactly the
uniform density on the simplex -- the critical point between corner-peaked
(alpha_i < 1) and interior-mode.  The first mixture run's opening targets
collapsed onto the old 0.15 floor, i.e. alpha_i ~ 0.05, so Thompson selection
was drawing from a density piling mass in the simplex corners.  Nothing now asks
the network for a belief that degenerate.

The network is still FREE to predict below 3 -- and if it does, the 1/conf
divergence pulls it back.  It should settle at or above 3 because no target ever
asks for less, not because it is mechanically prevented.

Everything else matches run_mixture.py: 10 blocks x 32 ch, 100/400 sims,
batch 1024 x 1 step/ep, lr 2e-3 cosine, buffer 300k, no pool, descent_temp 1.0,
mixture for both search and target.

    python run_mixture_tfloor3.py    # under run_mixture_tfloor3_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_MIXTURE_tfloor3'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'mixture_tfloor3.log')


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
        conf_offset=0.0,          # pure softplus: keeps the 1/conf restoring force
        head_conf_bias=1.4,       # historical init, alpha0 starts at 1.62
        target_conc_floor=3.0,    # no TARGET may ask for alpha0 < 3
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
    _log(f'=== MIXTURE target-floor 3: {NUM_EPISODES} eps, {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch, {cfg.fast_sims}/{cfg.full_sims} sims '
         f'(mean {cfg.fast_prob*cfg.fast_sims + (1-cfg.fast_prob)*cfg.full_sims:.0f})'
         f' -> {ROOT}/{CKPT} ===')
    _log(f'    search_agg={cfg.search_agg} target_agg={cfg.target_agg} '
         f'backup={cfg.backup} | TARGET conc floor {cfg.target_conc_floor:g} '
         f'(conf_offset {cfg.conf_offset:g}, pure softplus; init alpha0 1.62) | '
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
