"""OLIVAW-analogous run: the paper's scale, this repo's objective.

MM_long finished at Edax depth ~2.6 -- L1 96%, L2 73%, L3 36%, L4 16%, and
0-4% at L5 and above. OLIVAW (arXiv 2103.17228) reports its generation-20
agent at 1000 nodes/move scoring 35.5/70 in an 8-way round robin, between E6
(13) and E8 (36.5), with even its 400-node version at 28 against E4's 4. Both
ladders are the same knob: Edax's own help calls `-l|level <n>` "search using
limited depth", which is OLIVAW's definition of E_k.

So at an equal ~1000-position search budget OLIVAW plays around depth 8 and
MM_long plays around depth 2.6, on roughly the SAME number of self-play games
(~50,000) and with MM_long using 2.5-10x MORE search per training move. The
obvious suspect is capacity: 10 residual blocks against 3.

This run matches OLIVAW's scale where the hardware allows, to test that.

  MATCHED
    ~50,000 self-play games            their ~2500/generation x 20
    10 residual blocks                 the one architecture number they give
    100/200/400 sims per move          their escalating budget, as a mixture
    batch 1024                         their stated batch size
    weight decay 1e-4                  their L2 c=1e-4
    sample first 20 plies, then greedy their "first 20 moves ... according to
                                       pi", argmax after
    no frozen-opponent mixing          they self-play current-best only

  DELIBERATELY NOT MATCHED, and why
    32 channels          The paper never states its filter count. At 64 the
                         forward pass is 4x wider and this box needs ~18 days
                         for 50000 games; at 32 it needs ~3. Depth is the
                         number they actually publish, so depth is what moves.
    lr 2e-3, MM_long's   They use SGD+momentum at 0.003 with a step schedule.
                         Transplanting 0.003 onto AdamW destabilised a 10-block
                         net -- klv 0.50 -> 2.07 in one checkpoint interval with
                         kla flat -- so this uses MM_long's proven AdamW
                         schedule instead (2e-3, cosine over the run, 2% floor).
                         Matching the paper's NUMBER while running a different
                         optimizer matches nothing.
    buffer 300k          Their window grows to the last five generations,
                         ~750k positions after their position-doubling. This
                         box has 15.9 GB.
    no promotion gate    They accept a generation only if it wins a 40-game
                         match. This trainer updates continuously.
    the objective        Still this repo's Dirichlet value-distribution loss,
                         not AlphaZero's scalar value + policy cross-entropy.
                         Changing scale AND method at once would confound the
                         very question being asked. Note their "doubling the
                         positions" trick -- training on explored-but-unplayed
                         positions labelled with the search value q -- is close
                         to what the kla term already does here, since it is a
                         KL over all SEARCHED edges rather than the played one.

If this reaches E4 and beyond, capacity was the ceiling and fa76db55's "the
network was never the ceiling" was a claim about Edax L1, not about depth 8.
If it lands at ~2.6 again, the gap is the method or the search, not the size.

    python run_olivaw.py        # under run_olivaw_forever.ps1
"""
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 50000
ROOT = 'othello_benchmark'
CKPT = 'bench_OLIVAW'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'olivaw.log')


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
        # 10 residual blocks is OLIVAW's stated depth.  Width stays at 32.
        channels=32, num_blocks=10, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        # OLIVAW escalates 100 -> 200 -> 400 across generations.  A per-game
        # coin flip with the same mean (0.6*100 + 0.4*400 = 220) is the closest
        # this config can express without splitting the run into three phases.
        fast_sims=100, full_sims=400, fast_prob=0.6,
        adaptive_mode=None, adaptive_kl_eps=None,
        # "The first 20 moves of each game are selected extracting at random
        # according to pi", argmax after.  late_temp 1e4 is effectively greedy;
        # descent_temp 16 keeps the in-tree descent sharp, which is the nearest
        # analogue to PUCT's deterministic selection.
        temp_threshold=20, late_temp=1e4, descent_temp=16.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        # OLIVAW self-plays the current best against itself.
        pool_prob=0.0, random_pool_frac=0.0,
        batch_size=1024, train_steps_per_ep=1, max_buffer=300_000,
        # MM_long's schedule exactly.  The first attempt transplanted OLIVAW's
        # 0.003 onto AdamW, but that number is an SGD-with-momentum rate and
        # the two are not interchangeable -- klv blew up from 0.50 to 2.07
        # between ep5000 and ep6000 on a 10-block net while kla sat unchanged,
        # which is what an over-large adaptive step looks like.  Matching the
        # optimizer that is actually in use beats matching the paper's number.
        lr_peak=2e-3, lr_warmup_eps=100,
        lr_decay_eps=NUM_EPISODES,
        lr_min_factor=0.02,
        weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=1000, quick_eval_games=30,
        deep_eval_every=5000,
        eval_sims=64,
        # 4 games/pair made the Elo pool too coarse to separate anything: at
        # ep5000 this run and MM_long produced byte-identical ratings
        # (5000=1118 0=972 random=936) from different architectures, because
        # the fit is decided by the win/loss pattern alone.  16 games, and the
        # new checkpoint plays EVERY previous one rather than the last 3.
        eval_games_per_pair=16,
        eval_last_n=999,          # new vs all previous
        eval_refresh_pairs=0,     # no old-vs-old replays; only the new model
        resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    cfg = make_cfg()
    _log(f'=== OLIVAW-analogous: {NUM_EPISODES} eps, {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch, {cfg.fast_sims}/{cfg.full_sims} sims '
         f'(mean {cfg.fast_prob*cfg.fast_sims + (1-cfg.fast_prob)*cfg.full_sims:.0f})'
         f' -> {ROOT}/{CKPT} ===')
    # Derived from cfg, not retyped: the first attempt's banner claimed 3e-3
    # after the config had been changed to 2e-3.
    _log(f'    lr {cfg.lr_peak:.0e} -> cosine over {cfg.lr_decay_eps} -> floor '
         f'{cfg.lr_peak*cfg.lr_min_factor:.1e} | batch {cfg.batch_size} x '
         f'{cfg.train_steps_per_ep} step/ep | buffer {cfg.max_buffer//1000}k | '
         f'temp_threshold {cfg.temp_threshold}, greedy after | '
         f'pool {cfg.pool_prob} | elo {cfg.eval_games_per_pair} games/pair '
         f'vs all previous')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('OLIVAW DONE')
