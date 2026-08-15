"""Long run under the best configuration found so far.

Previous best (the round-3 tournament winner, `conf`) plus the one change the
policy screens turned up, so anything that moves is attributable to that
change:

  descent_temp = 1e4    Take the sampling out of in-tree descent.  _temp() used
                        to drive BOTH the root Thompson pick and every
                        _select_leaf below it, so self-play exploration and
                        descent noise were the same knob.  They should not be:
                        the root pick is where self-play diversity has to come
                        from, while noise in the DESCENT only adds variance to
                        the estimate search returns.  Splitting them and making
                        descent deterministic scored 74.5% against the coupled
                        default in self-play at a fixed 64 sims (about +186
                        Elo).  Root exploration is untouched -- temp_threshold
                        30 / late_temp 8.0 as before.

Everything else is round 3's configuration verbatim, including the `conf`
allocation rule at its round-2 calibrated scale.  Allocation is kept not
because it earns much -- probe_transfer put the whole upfront-allocation family
within ~2 points of a fixed budget -- but because changing two things at once
would make the result unreadable.

Checkpoints land every 1000 episodes for othello_ladder.py to score against an
external opponent that cannot drift with the run.

    python run_optimal_long.py          # under run_optimal_long_forever.ps1
"""
import json
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 20000
ROOT = 'othello_benchmark'
CKPT = 'bench_OPT'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'optimal_long.log')
DESCENT_TEMP = 1e4


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(rule_cfg):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(ROOT, CKPT),
        num_episodes=NUM_EPISODES,
        channels=32, num_blocks=3, head_ch=8, seed=0,
        device_preference='auto',
        search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
        selection='dirichlet', kl_normalize=False,
        fast_sims=100, full_sims=300, fast_prob=0.75,
        adaptive_mode=rule_cfg.get('adaptive_mode'),
        adaptive_kl_eps=rule_cfg.get('adaptive_kl_eps'),
        adaptive_check_every=rule_cfg.get('adaptive_check_every', 50),
        alloc_scale=rule_cfg.get('alloc_scale'),
        alloc_min=rule_cfg.get('alloc_min', 16),
        alloc_max=rule_cfg.get('alloc_max', 600),
        predictor_coef=(tuple(rule_cfg['predictor_coef'])
                        if rule_cfg.get('predictor_coef') else None),
        decision_margin=rule_cfg.get('decision_margin'),
        hybrid_margin_scale=rule_cfg.get('hybrid_margin_scale'),
        # THE change under test: deterministic descent, root exploration as before.
        temp_threshold=30, late_temp=8.0, descent_temp=DESCENT_TEMP,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_warmup_eps=100, lr_decay_eps=30_000,
        lr_min_factor=0.1, weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=250, quick_eval_games=30,
        deep_eval_every=1000,          # one ladder-scorable checkpoint per 1000
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    rule = cal2['rules']['conf']['cfg']
    _log(f'=== optimal long run -> {NUM_EPISODES} eps ({ROOT}/{CKPT}) ===')
    _log(f'    conf allocation @ alloc_scale={rule.get("alloc_scale"):.4g}, '
         f'descent_temp={DESCENT_TEMP:g} (root exploration unchanged)')
    t0 = time.time()
    c4.run_training(make_cfg(rule), log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('OPTIMAL LONG DONE')
