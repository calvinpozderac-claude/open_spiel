"""Round 2: train each calibrated budget rule to 2000 episodes, all at a
matched ~150 sims/move average, then rank them by Bradley-Terry against the
existing ~150-sims arms at the same episode count.

Everything here is matched-compute by construction (calibrate_round2.py fixes
the mean sims/move), so any strength difference is attributable to HOW the
budget is spent per position rather than how much is spent overall -- the
confound that made the first round's comparisons uninterpretable.

    python calibrate_round2.py      # first -- writes round2_calibration.json
    python run_round2_sweep.py

Run under a Scheduled Task wrapper (see run_round2_forever.ps1); resume=True
plus checkpoints every 250 episodes make interruption free.
"""
import json
import os
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

NUM_EPISODES = 2000
ROOT = 'othello_benchmark'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'round2_sweep.log')
CAL_PATH = 'round2_calibration.json'
TOURNAMENT_CACHE = 'round2_tournament_cache.json'
TOURNAMENT_BUDGET_S = 3 * 3600

# The already-trained arms whose training averaged ~150 sims/move, plus
# bench_MM itself (100/300 mix == 150 mean). 100avg (~100) is included as the
# nearest lower-budget reference point.
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


def make_cfg(name, rule_cfg):
    return c4.Config(
        game_name='othello',
        checkpoint_dir=os.path.join(ROOT, f'bench_R2{name}'),
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


def train_all(cal):
    for name, spec in cal['rules'].items():
        _log(f'--- R2 arm {name} starting (knob {spec["knob"]}={spec["value"]:.6g}, '
            f'calibrated mean {spec["mean_sims"]:.1f} sims/move) ---')
        t0 = time.time()
        try:
            c4.run_training(make_cfg(name, spec['cfg']), log=_log)
        except Exception as e:
            _log(f'--- R2 arm {name} FAILED: {type(e).__name__}: {e} ---')
            raise
        _log(f'--- R2 arm {name} segment done: {time.time()-t0:.0f}s ---')


def tournament(cal):
    c4.set_game(c4.load_game('othello'))
    entrants = {}
    for name, spec in cal['rules'].items():
        d = os.path.join(ROOT, f'bench_R2{name}')
        if not os.path.exists(os.path.join(d, 'bench_2000.pt')):
            _log(f'  {name}: no 2000-ep checkpoint, skipping')
            continue
        entrants[name] = (c4.load_benchmark_net(d, '2000', ad.MM_SIG),
                          ('rule', spec['cfg']))
    for label, (d, recipe) in COMPARATORS.items():
        p = os.path.join(ROOT, d)
        if os.path.exists(os.path.join(p, 'bench_2000.pt')):
            entrants[label] = (c4.load_benchmark_net(p, '2000', ad.MM_SIG),
                               recipe)
    entrants['random'] = (None, ('random', None))
    _log(f'tournament entrants: {list(entrants)}')

    n_pairs = len(entrants) * (len(entrants) - 1) // 2
    t0 = time.time()
    *_ , n_new = ad.arms_round_robin(entrants, games_per_pair=1,
                                     cache=TOURNAMENT_CACHE, log=_log)
    elapsed = time.time() - t0
    sec_per_game = elapsed / n_new if n_new else 20.0
    k = 1 + int(max(TOURNAMENT_BUDGET_S - elapsed, 0)
                // (n_pairs * sec_per_game))
    _log(f'{n_new} games in {elapsed:.0f}s ({sec_per_game:.1f}s/game) -> '
        f'k={k} games/pair')

    names, W, elo, _ = ad.arms_round_robin(entrants, games_per_pair=k,
                                           cache=TOURNAMENT_CACHE, log=_log)
    _log('\n=== Round 2 Bradley-Terry (anchored random=0) ===')
    for i in np.argsort(-elo):
        _log(f'  {names[i]:<12} {elo[i]:+8.1f}')
    with open('round2_tournament_results.json', 'w') as fh:
        json.dump({'names': names, 'W': W.tolist(), 'bt': elo.tolist(),
                  'games_per_pair': k}, fh, indent=2)


if __name__ == '__main__':
    cal = json.load(open(CAL_PATH))
    _log(f'round 2 start: rules={list(cal["rules"])} '
        f'(predictor r2={cal["predictor_r2"]:.4f})')
    train_all(cal)
    _log('=== round 2 training complete; tournament ===')
    tournament(cal)
    _log('ROUND2 DONE')
