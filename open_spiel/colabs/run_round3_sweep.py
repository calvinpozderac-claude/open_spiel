"""Round 3: take the promising matched-150-sim rules to 5000 episodes and
rank them against the other ~150-sim arms at the same episode count.

Two kinds of arm here:
  RESUMED  round-2 arms that already have 2000 episodes in
           othello_benchmark/bench_R2<name> -- resume=True plus a raised
           num_episodes continues them to 5000 rather than retraining.
  NEW      round-3 rules (conf_nl, confkl), trained from scratch to 5000.

Which round-2 arms get promoted is PROMOTE below; it is set from the round-2
tournament result rather than guessed.

    python calibrate_round3.py     # writes round3_calibration.json
    python run_round3_sweep.py     # under run_round3_forever.ps1
"""
import json
import os
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

NUM_EPISODES = 5000
ROOT = 'othello_benchmark'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'round3_sweep.log')
TOURNAMENT_CACHE = 'round3_tournament_cache.json'
TOURNAMENT_BUDGET_S = 3 * 3600

# Round-2 arms carried on to 5000 (their calibrated cfgs come from
# round2_calibration.json, so they keep searching exactly as they trained).
# Set from the round-2 tournament: conf won it and predictor was the other
# clearly-above-bench_MM allocation rule. 'hybrid' is dropped -- at 57.9% it
# is barely separated from bench_MM's 55.0% and is a reactive variant of the
# family that already lost round 1. 'margin' and 'decision' both finished
# BELOW bench_MM and are dead ends.
PROMOTE = ('conf', 'predictor')

# Already-trained ~150-sim reference points, at 5000 episodes. 150match is
# the one that matters: it tied conf in round 2 (49.2%, CI straddling 50%),
# so it is a real contender rather than just a baseline.
COMPARATORS = {
    '150match': ('bench_MMadapt150match', ('adaptive', 75, 0.1, 300)),
    '100avg':   ('bench_MMadapt100avg',   ('adaptive', 50, 0.08, 300)),
    'bench_MM': ('bench_MM',              ('mix', 100, 300, 0.75)),
}


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(ckpt_dir, rule_cfg):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(ROOT, ckpt_dir),
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
        temp_threshold=30, late_temp=8.0,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=0, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_warmup_eps=100, lr_decay_eps=30_000,
        lr_min_factor=0.1, weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


def arm_specs():
    """[(label, checkpoint_dir, rule_cfg)] for every arm round 3 trains."""
    cal2 = json.load(open('round2_calibration.json'))
    cal3 = json.load(open('round3_calibration.json'))
    out = []
    for name in PROMOTE:
        out.append((name, f'bench_R2{name}', cal2['rules'][name]['cfg']))
    for name, spec in cal3['rules'].items():
        out.append((name, f'bench_R3{name}', spec['cfg']))
    return out


def train_all(specs):
    for label, ckpt_dir, rule_cfg in specs:
        _log(f'--- R3 arm {label} -> {NUM_EPISODES} eps ({ckpt_dir}) ---')
        t0 = time.time()
        try:
            c4.run_training(make_cfg(ckpt_dir, rule_cfg), log=_log)
        except Exception as e:
            _log(f'--- R3 arm {label} FAILED: {type(e).__name__}: {e} ---')
            raise
        _log(f'--- R3 arm {label} segment done: {time.time()-t0:.0f}s ---')


def tournament(specs):
    c4.set_game(c4.load_game('othello'))
    entrants = {}
    for label, ckpt_dir, rule_cfg in specs:
        p = os.path.join(ROOT, ckpt_dir, 'bench_5000.pt')
        if not os.path.exists(p):
            _log(f'  {label}: no 5000-ep checkpoint, skipping')
            continue
        entrants[label] = (
            c4.load_benchmark_net(os.path.join(ROOT, ckpt_dir), '5000',
                                  ad.MM_SIG),
            ('rule', rule_cfg))
    for label, (d, recipe) in COMPARATORS.items():
        p = os.path.join(ROOT, d)
        if os.path.exists(os.path.join(p, 'bench_5000.pt')):
            entrants[label] = (c4.load_benchmark_net(p, '5000', ad.MM_SIG),
                               recipe)
    entrants['random'] = (None, ('random', None))
    _log(f'tournament entrants: {list(entrants)}')

    n_pairs = len(entrants) * (len(entrants) - 1) // 2
    t0 = time.time()
    *_, n_new = ad.arms_round_robin(entrants, games_per_pair=1,
                                    cache=TOURNAMENT_CACHE, log=_log)
    elapsed = time.time() - t0
    spg = elapsed / n_new if n_new else 20.0
    k = 1 + int(max(TOURNAMENT_BUDGET_S - elapsed, 0) // (n_pairs * spg))
    _log(f'{n_new} games in {elapsed:.0f}s ({spg:.1f}s/game) -> k={k} games/pair')

    names, W, elo, _ = ad.arms_round_robin(entrants, games_per_pair=k,
                                           cache=TOURNAMENT_CACHE, log=_log)
    _log('\n=== Round 3 Bradley-Terry @5000 eps (anchored random=0) ===')
    for i in np.argsort(-elo):
        _log(f'  {names[i]:<12} {elo[i]:+8.1f}')
    with open('round3_tournament_results.json', 'w') as fh:
        json.dump({'names': names, 'W': W.tolist(), 'bt': elo.tolist(),
                  'games_per_pair': k}, fh, indent=2)


if __name__ == '__main__':
    specs = arm_specs()
    _log(f'round 3 start: arms={[s[0] for s in specs]}')
    train_all(specs)
    _log('=== round 3 training complete; tournament ===')
    tournament(specs)
    _log('ROUND3 DONE')
