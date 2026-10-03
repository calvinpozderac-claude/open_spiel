"""CORR arm: the correlated-error posterior, matched to run_olivaw_tz.py.

Every knob is taken from olivaw_common, exactly as the OLIVAW-TZ arm took them,
so this is a one-variable change against that run: the evidence-collapse rule.

    OLIVAW-TZ   search additive_mle   target avgconc
    CORR        search corr           target corr     (rho fixed at 0.8)

WHY rho IS FIXED HERE, with the drift loss still on.  Learned rho has no
transport to the self-play workers yet: they receive rho through set_search at
spawn, from the config, and the learned value lives in the network and changes
during training.  So:

  * SEARCH and TARGETS use rho_search / rho_target below -- specified, not learned.
  * The rho head and the 'drift' loss term are ON anyway, which costs ~0.7% of
    self-play (measured) and makes the run report what rho the data actually
    implies, generation by generation.  If the learned rho_bar stays near 0.8,
    the fixed value was a fair stand-in and a later learned-rho run should behave
    much like this one; if it drifts, this run tells us the fixed choice was
    wrong and by how much.

rho_search = 0.8 is not a guess: snapshot drift on real Othello trees measured
0.797 (untrained), 0.804 (TZ10k) and 0.834 (TZ50k).  The spec's rho_init of 0.3
would have been far off.

rho_target stays at 0.999 -- the near-avgconc end -- so the TARGET side changes
as little as possible from the OLIVAW-TZ run and the comparison isolates the
search rule plus the terminal-handling change.

    python run_corr_tz.py
"""
import os
import time

import connect4_dirichlet_utils as c4
import olivaw_common as OC

CKPT = 'bench_CORR_TZ'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'corr_tz.log')

RHO_SEARCH = 0.8        # measured on real trees; see the module docstring
RHO_TARGET = 0.999      # near-avgconc, to keep the target side close to OLIVAW-TZ
SNAPSHOT_FRACS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
W_DRIFT = 1.0           # trains rho_bar for logging; search still uses RHO_SEARCH


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(OC.ROOT, CKPT),
        num_episodes=OC.NUM_EPISODES,
        channels=OC.CHANNELS, num_blocks=OC.NUM_BLOCKS, head_ch=OC.HEAD_CH,
        seed=OC.SEED, device_preference='auto',

        # ── THE ONE CHANGE against run_olivaw_tz.py ──────────────────────────
        search_agg=c4.AGG_CORR, target_agg=c4.AGG_CORR,
        rho_search=RHO_SEARCH, rho_target=RHO_TARGET,
        corr_target_cap=c4.TERMINAL_CONC,
        snapshot_fracs=SNAPSHOT_FRACS,
        rho_head=True, rho_init=0.8, rho_mode='global',

        # ── identical to the OLIVAW-TZ arm from here down ────────────────────
        selection='dirichlet', kl_normalize=False, backup='mean',
        virtual_loss=1.0,
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
                      'cons': 0.5, 'drift': W_DRIFT},
        late_temp=1e4, descent_temp=1.0,
        adaptive_mode=None, adaptive_kl_eps=None,
        sims_ramp=OC.SIMS_RAMP,
        double_aux=OC.DOUBLE_AUX, double_min_visits=OC.DOUBLE_MIN_VISITS,
        batch_size=OC.BATCH_SIZE, train_steps_per_ep=OC.TRAIN_STEPS_PER_EP,
        max_buffer=OC.MAX_BUFFER,
        lr_peak=OC.LR_PEAK, lr_warmup_eps=OC.LR_WARMUP_EPS,
        lr_steps=OC.LR_STEPS,
        weight_decay=OC.WEIGHT_DECAY, grad_clip=OC.GRAD_CLIP,
        temp_threshold=OC.TEMP_THRESHOLD,
        max_plies=OC.MAX_PLIES, eval_max_plies=OC.MAX_PLIES,
        pool_prob=OC.POOL_PROB, random_pool_frac=OC.RANDOM_POOL_FRAC,
        use_workers=True, selfplay_workers=OC.SELFPLAY_WORKERS,
        games_per_worker=OC.GAMES_PER_WORKER, worker_wave=OC.WORKER_WAVE,
        n_parallel_games=OC.N_PARALLEL_GAMES, wave_per_game=OC.WAVE_PER_GAME,
        quick_eval_every=OC.QUICK_EVAL_EVERY,
        quick_eval_games=OC.QUICK_EVAL_GAMES,
        deep_eval_every=OC.DEEP_EVAL_EVERY, eval_sims=OC.EVAL_SIMS,
        eval_games_per_pair=OC.EVAL_GAMES_PER_PAIR,
        eval_last_n=OC.EVAL_LAST_N, eval_refresh_pairs=OC.EVAL_REFRESH_PAIRS,
        deep_eval_elo=False,            # rated externally by corr_live_ladder.py
        resume=True, solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(OC.banner(cfg, 'CORR-TZ',
                   f'engine=thompson SEARCH={cfg.search_agg} '
                   f'TARGET={cfg.target_agg} rho_search={cfg.rho_search} '
                   f'rho_target={cfg.rho_target} rho_mode={cfg.rho_mode} '
                   f'w_drift={W_DRIFT}'))
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('CORR_TZ DONE')
