"""The OLIVAW recipe, in one place, shared by both arms.

WHY A SHARED MODULE. The whole point of these two runs is a controlled
comparison -- AlphaZero-with-OLIVAW vs ThompsonZero-with-OLIVAW, on the same
recipe -- and the fastest way to ruin that is to let two run scripts drift
apart by one knob nobody notices. Everything the two arms MUST agree on lives
here; each run script sets only what is genuinely engine-specific (the search
rule and the loss).

THE RECIPE (OLIVAW, arXiv 2103.17228, "Mastering Othello with neither Humans
nor a Penny"). Our own HYBRID run plateaus after ~30k episodes and, measured
against Edax with XOT openings, plays at roughly Edax depth 2-3 at 1000 nodes
a move. OLIVAW's O-1000 -- same node budget -- beats E6 and loses only to E8.
Four differences in how they train, ported here:

  1. DOUBLING (their one advertised modification). Positions the tree explored
     heavily but never played are added to the training set, threshold chosen
     dynamically to exactly double it. Free data: the search already paid for
     it, and those are the best-resolved targets in the tree. `double_aux=1`.

  2. GRADIENT WORK. 16,384,000 positions per generation at minibatch 1024 over
     ~2500 games = 6.4 optimizer steps per self-play game. Ours was ONE. On
     comparable unique-position counts (~5.8M) they do 3.2x the total gradient
     work: each generated position is trained on ~56 times against our ~17.

  3. LEARNING RATE. A step schedule -- 3e-3, 1e-3 from gen 4/20, 1e-4 from gen
     11/20 -- so the last ~45% of the run sits at a flat 1e-4. Our cosine
     annealed to 4e-5, 2.5x lower, and did it right where the curve flattens.
     This project has already caught that exact mechanism once before.

  4. SIMS RAMP. 100 -> 200 -> 400 per move as generations progress, rather
     than a fixed mix. Cheap games early, expensive games once the net is
     worth sharpening.

NOT PORTED, deliberately: gating (they promote a new net only if it beats the
old over 40 games). It needs a generation loop both engines lack, AlphaZero
itself dropped it, and it mostly buys curve monotonicity rather than final
strength. Both arms are ungated, so they stay comparable to each other; this
is the one place they knowingly diverge from the paper.

CAPACITY. The paper says "10 residual blocks" and never states a filter count.
We use 64 channels for both arms -- up from the 32 every previous run in this
project used, which was a plausible hard ceiling. It is held IDENTICAL across
the two arms, so it cannot explain a difference between them; it does mean
neither is directly comparable to the existing 32ch checkpoint library.
"""

NUM_EPISODES = 50_000

# --- capacity (identical both arms) ---------------------------------------
CHANNELS, NUM_BLOCKS, HEAD_CH = 64, 10, 8

# --- OLIVAW recipe --------------------------------------------------------
# 6 steps/episode at batch 1024 ~ their 6.4 (16,384,000 positions / 2500 games).
TRAIN_STEPS_PER_EP = 6
BATCH_SIZE = 1024

# 3e-3 -> 1e-3 at gen 4/20 (20%) -> 1e-4 at gen 11/20 (55%).
LR_PEAK = 3e-3
LR_WARMUP_EPS = 100
LR_STEPS = ((0.0, 3e-3), (0.20, 1e-3), (0.55, 1e-4))

# Their window is the last 2-5 generations = 5,000-12,500 games. With doubling
# our episodes yield ~150 samples each, so 600k ~ 4,000 games: the low end of
# their window, and ~0.9 GB of RAM on the ThompsonZero side (measured).
MAX_BUFFER = 600_000

# 100 -> 200 -> 400 in even thirds of the run.
SIMS_RAMP = ((0.0, 100), (1.0 / 3.0, 200), (2.0 / 3.0, 400))

# The doubling trick: 1 highest-visit UNPLAYED child per move ~= 2x the set.
DOUBLE_AUX = 1
DOUBLE_MIN_VISITS = 8

# --- self-play protocol (identical both arms, and already matching OLIVAW) --
# They sample from pi for the first 20 plies and play argmax after; both
# engines already express exactly that, so this is a match, not a port.
TEMP_THRESHOLD = 20
# Dirichlet root noise eps=0.25, alpha=min(1, 10/B); Othello's branching is
# ~10, so alpha=1.0.
ROOT_NOISE_FRAC = 0.25
ROOT_NOISE_ALPHA = 1.0
WEIGHT_DECAY = 1e-4          # their loss carries c = 1e-4
GRAD_CLIP = 1.0
# OLIVAW's self-play is always latest-vs-latest: no opponent pool, no random
# opponents. Both arms therefore switch the pool off.
POOL_PROB = 0.0
RANDOM_POOL_FRAC = 0.0
MAX_PLIES = 128              # Othello
SEED = 0

# --- infrastructure (identical both arms) ---------------------------------
# 6 logical cores on this box, and self-play is ~83% of wall time and CPU
# bound, so the runs go SEQUENTIALLY at 4 workers rather than concurrently at
# 2 -- concurrency cannot beat total core throughput, it just delays both
# results and risks thrashing.
SELFPLAY_WORKERS = 4
GAMES_PER_WORKER = 16
WORKER_WAVE = 4
N_PARALLEL_GAMES = 16
WAVE_PER_GAME = 4

# --- evaluation (identical both arms, so the two Elo ladders mean the same) -
QUICK_EVAL_EVERY = 1000
QUICK_EVAL_GAMES = 30
DEEP_EVAL_EVERY = 5000
EVAL_SIMS = 64
EVAL_GAMES_PER_PAIR = 16
EVAL_LAST_N = 999
EVAL_REFRESH_PAIRS = 0

ROOT = 'othello_benchmark'


def banner(cfg, name, extra=''):
    """One log line both arms print identically, so two logs can be diffed."""
    return (f'=== {name}: {cfg.num_episodes} eps, {cfg.num_blocks} blocks x '
            f'{cfg.channels} ch | OLIVAW recipe: double_aux={cfg.double_aux} '
            f'sims_ramp={[s for _, s in cfg.sims_ramp]} '
            f'steps/ep={cfg.train_steps_per_ep} batch={cfg.batch_size} '
            f'buf={cfg.max_buffer // 1000}k '
            f'lr={[l for _, l in cfg.lr_steps]} '
            f'temp_thr={cfg.temp_threshold} | {extra} ===')
