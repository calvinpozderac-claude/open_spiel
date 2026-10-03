"""CORR arm: the correlated-error posterior, matched to run_olivaw_tz.py.

Every knob is taken from olivaw_common, exactly as the OLIVAW-TZ arm took them,
so this is a one-variable change against that run: the evidence-collapse rule.

    OLIVAW-TZ   search additive_mle   target avgconc
    CORR        search corr           target corr     (rho fixed at 0.8)

WHY rho IS FIXED HERE AND THE DRIFT LOSS IS OFF.  Two separate reasons.

1. Learned rho has no transport to the self-play workers: they get rho through
   set_search at spawn, from config, while the learned value lives in the network
   and changes during training.
2. More importantly, learning rho from drift DURING a live run does not measure
   what it is supposed to.  Self-play reads the weights live and training updates
   them 6 times per episode, so a search's early simulations are evaluated by a
   different network from its late ones.  That weight movement is drift the model
   attributes to idiosyncratic leaf error, which pushes rho down.  Measured by
   perturbing a frozen net's weights and re-measuring:

       frozen TZ50k                  rho = 0.764
       weights +0.02 sigma           rho = 0.720
       weights +0.05 sigma           rho = 0.553
       weights +0.10 sigma           rho = 0.001

   A first attempt at this run had w_drift = 1.0 and rho_mode = 'global'.  By
   ep2000 the term was 83% of the total loss and rho_bar had fallen 0.80 -> 0.477,
   tracking weight non-stationarity rather than leaf-error correlation.  It was
   stopped and is not used.

So rho is SPECIFIED at 0.8, from measuring frozen checkpoints (0.797 untrained,
0.804 TZ10k, 0.834 TZ50k, 0.764 re-measured), and the rho head is OFF -- which
also makes the parameter count identical to the OLIVAW-TZ arm (909,816), so the
only difference between the two runs is the evidence-collapse rule.

corr_rho_probe.py measures rho offline from a frozen checkpoint; that is the
uncontaminated way to get it, and the viable route to a learned rho is a
generation-level update from such a measurement rather than a per-step gradient.

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

RHO_SEARCH = 0.8        # measured on FROZEN nets; see the module docstring
RHO_TARGET = 0.999      # near-avgconc, to keep the target side close to OLIVAW-TZ
SNAPSHOT_FRACS = ()     # off: the drift loss is off, so nothing consumes them
W_DRIFT = 0.0           # see the docstring -- confounded by live weight updates


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
        rho_head=False, rho_mode='fixed',

        # ── identical to the OLIVAW-TZ arm from here down ────────────────────
        selection='dirichlet', kl_normalize=False, backup='mean',
        virtual_loss=1.0,
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
                      'cons': 0.5},          # identical to the OLIVAW-TZ arm
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
                   f'rho_target={cfg.rho_target} rho_mode={cfg.rho_mode}'))
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('CORR_TZ DONE')
