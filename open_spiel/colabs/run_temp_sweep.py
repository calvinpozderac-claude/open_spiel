"""Does sharpened descent help TRAINING, or only inference?

policy_screen established the inference effect and it is large: deterministic
descent scores 74.5% at 64 sims and 76.4% at 150 against the coupled default,
roughly +200 Elo, with the incumbent policy on both sides otherwise.

Training is a separate question and the honest answer is that nobody knows
yet. The worry is exploration: a sharper descent could narrow the tree, and
make_target can only label edges search actually TOUCHED, so a narrow tree
would mean sparse action-head targets. Measured beforehand, that does not
appear to happen -- at 150 sims every legal root edge gets evidence at every
temperature from 1 to 1e4, and the top edge keeps ~20% of visits, because
virtual_loss=1.0 on a [-1,1] scale swamps any value gap and forces a wave to
fan out. Breadth here comes from virtual loss, not from Thompson noise, and
self-play diversity comes from the ROOT pick, which is untouched.

That is an argument, not a result, so this sweeps it.

  coupled   descent_temp=None -- the incumbent, where one _temp() drives both
            the root pick and descent (1.0 early, late_temp=8.0 after ply 30)
  dt4/dt16  progressively sharper descent, root exploration unchanged
  dtinf     deterministic descent

Every arm is otherwise identical, including the `conf` allocation rule and the
seed, so the only difference is descent temperature. Arms are then compared
under ONE fixed inference configuration, so what comes out is the TRAINING
effect with the inference effect held constant.

    python run_temp_sweep.py        # under run_temp_sweep_forever.ps1
"""
import json
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 2000
ROOT = 'othello_benchmark'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'temp_sweep.log')

# (label, checkpoint dir, descent_temp).  None reproduces the coupled default.
ARMS = [
    ('coupled', 'bench_Tcoupled', None),
    ('dt4',     'bench_Tdt4',     4.0),
    ('dt16',    'bench_Tdt16',    16.0),
    ('dtinf',   'bench_Tdtinf',   1e4),
]


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg(ckpt_dir, rule_cfg, descent_temp):
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
        # The swept knob.  Root exploration is identical across arms.
        temp_threshold=30, late_temp=8.0, descent_temp=descent_temp,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        lr_peak=2e-3, lr_warmup_eps=100, lr_decay_eps=30_000,
        lr_min_factor=0.1, weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    rule = cal2['rules']['conf']['cfg']
    _log(f'=== descent-temp sweep, {NUM_EPISODES} eps/arm, conf allocation ===')
    for label, ckpt, dt in ARMS:
        done = os.path.exists(os.path.join(ROOT, ckpt,
                                           f'bench_{NUM_EPISODES}.pt'))
        if done:
            _log(f'--- {label}: already at {NUM_EPISODES}, skipping ---')
            continue
        _log(f'--- arm {label}: descent_temp={dt} -> {ckpt} ---')
        t0 = time.time()
        try:
            c4.run_training(make_cfg(ckpt, rule, dt), log=_log)
        except Exception as e:
            _log(f'--- arm {label} FAILED: {type(e).__name__}: {e} ---')
            raise
        _log(f'--- arm {label} done: {time.time()-t0:.0f}s ---')
    _log('TEMP SWEEP DONE')
