"""OLIVAW arm B: ThompsonZero (the HYBRID recipe), OLIVAW recipe, 50k episodes.

The same four OLIVAW training modifications as arm A, bolted onto the
Dirichlet-belief engine instead of AlphaZero. The question it answers: do
OLIVAW's gains transfer to the Bayesian framework, or are they specific to
policy/value AlphaZero?

Search/target/loss are the HYBRID recipe -- additive_mle search + avgconc
target, the best configuration this project has found, and the one whose 100k
checkpoints the XOT Edax grid just measured. Holding that fixed is what makes
this a test of the RECIPE rather than another search-rule sweep.

THE DOUBLING TRICK HERE. OLIVAW's extra pairs are {s, (pi_s, q_s)} -- the
SEARCH value q_s, not the game outcome z, because an unplayed line has no
outcome. In this engine that is a sample with z_w = 0: it keeps klv/kla (the
distillation half, which is the pi/q target) and drops cev/cea (the
z-grounding half). See _emit_doubles in connect4_dirichlet_utils.

Everything not listed here comes from olivaw_common and is character-for-
character identical to the AlphaZero arm.

    python run_olivaw_tz.py
"""
import os
import time

import connect4_dirichlet_utils as c4
import olivaw_common as OC

CKPT = 'bench_OLIVAW_TZ'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'olivaw_tz.log')


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

        # ── engine-specific: the HYBRID recipe ───────────────────────────────
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_AVGCONC,
        selection='dirichlet', kl_normalize=False, backup='mean',
        virtual_loss=1.0,
        # Set EXPLICITLY. This script's ancestor relied on the Config default,
        # which was silently changed to distillation-only during a refactor and
        # dropped z-grounding on resume -- the "can't learn from a wiped
        # buffer" bug. Never rely on the default for a run of record.
        loss_weights={'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0,
                      'cons': 0.5},
        late_temp=1e4, descent_temp=1.0,
        adaptive_mode=None, adaptive_kl_eps=None,

        # ── OLIVAW recipe (shared) ───────────────────────────────────────────
        sims_ramp=OC.SIMS_RAMP,
        double_aux=OC.DOUBLE_AUX, double_min_visits=OC.DOUBLE_MIN_VISITS,
        batch_size=OC.BATCH_SIZE, train_steps_per_ep=OC.TRAIN_STEPS_PER_EP,
        max_buffer=OC.MAX_BUFFER,
        lr_peak=OC.LR_PEAK, lr_warmup_eps=OC.LR_WARMUP_EPS,
        lr_steps=OC.LR_STEPS,
        weight_decay=OC.WEIGHT_DECAY, grad_clip=OC.GRAD_CLIP,

        # ── self-play protocol (shared) ──────────────────────────────────────
        temp_threshold=OC.TEMP_THRESHOLD,
        max_plies=OC.MAX_PLIES, eval_max_plies=OC.MAX_PLIES,
        pool_prob=OC.POOL_PROB, random_pool_frac=OC.RANDOM_POOL_FRAC,
        use_workers=True, selfplay_workers=OC.SELFPLAY_WORKERS,
        games_per_worker=OC.GAMES_PER_WORKER, worker_wave=OC.WORKER_WAVE,
        n_parallel_games=OC.N_PARALLEL_GAMES, wave_per_game=OC.WAVE_PER_GAME,

        # ── evaluation (shared) ──────────────────────────────────────────────
        quick_eval_every=OC.QUICK_EVAL_EVERY,
        quick_eval_games=OC.QUICK_EVAL_GAMES,
        deep_eval_every=OC.DEEP_EVAL_EVERY, eval_sims=OC.EVAL_SIMS,
        eval_games_per_pair=OC.EVAL_GAMES_PER_PAIR,
        eval_last_n=OC.EVAL_LAST_N, eval_refresh_pairs=OC.EVAL_REFRESH_PAIRS,
        resume=True, solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(OC.banner(cfg, 'OLIVAW-TZ',
                   f'engine=thompson SEARCH={cfg.search_agg} '
                   f'TARGET={cfg.target_agg}'))
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('OLIVAW_TZ DONE')
