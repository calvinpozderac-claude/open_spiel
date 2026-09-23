"""OLIVAW arm A: standard AlphaZero, OLIVAW recipe, 50k episodes.

This is the reproduction attempt -- policy+value heads, PUCT, CE-policy /
MSE-value, exactly the algorithm OLIVAW ran, with their four training
modifications (see olivaw_common for the full rationale). The question it
answers: does the AlphaZero framework's plateau in this project survive the
OLIVAW recipe, or was the plateau ours rather than the algorithm's?

Everything not listed here comes from olivaw_common and is character-for-
character identical to the ThompsonZero arm.

    python run_olivaw_az.py
"""
import os
import time

import connect4_alphazero_utils as az
import olivaw_common as OC

CKPT = 'bench_OLIVAW_AZ'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'olivaw_az.log')


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return az.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(OC.ROOT, CKPT),
        num_episodes=OC.NUM_EPISODES,
        channels=OC.CHANNELS, num_blocks=OC.NUM_BLOCKS, head_ch=OC.HEAD_CH,
        seed=OC.SEED, device_preference='auto',

        # ── engine-specific: PUCT ────────────────────────────────────────────
        c_puct=1.5, virtual_loss=1.0,
        root_noise_frac=OC.ROOT_NOISE_FRAC,
        root_noise_alpha=OC.ROOT_NOISE_ALPHA,
        # their loss is (w - v)^2 - pi^T log p, i.e. value weighted 1:1
        value_weight=1.0,

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
    _log(OC.banner(cfg, 'OLIVAW-AZ', f'engine=alphazero PUCT c_puct={cfg.c_puct} '
                                    f'value_weight={cfg.value_weight}'))
    t0 = time.time()
    az.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('OLIVAW_AZ DONE')
