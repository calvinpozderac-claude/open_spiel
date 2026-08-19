"""Does the 20k run plateau because the learning rate never annealed?

The plateau is real and external: against Edax L1, bench_OPT scores 35.0% at
ep4000 and 37.0% at ep20000, statistically indistinguishable, while the
sibling round-robin over the same span claimed +215 Elo.

The schedule is a genuine suspect. lr_decay_eps was 30000 on a 20000-episode
run, so the cosine only reached f=0.67 and training ended at 6.53e-04 -- 33%
of the 2e-3 peak, never approaching the 2e-4 floor, let alone the 1e-5 region
where AlphaZero-style runs do their final polish.

The obvious objection is that LR fell 3x across the plateau (1.9e-3 at ep4000
to 6.5e-4 at ep20000) and bought nothing. But that whole range is the high-LR
regime; "a 3x decay did not help" is not evidence that a 30x decay will not.

So this resumes from ep20000 and anneals properly: horizon set to the actual
end of the run and the floor dropped to 2% of peak, giving

    ep 20000  1.72e-04     (an immediate 4x drop from where the run stopped)
    ep 22000  7.4e-05
    ep 24000  4.0e-05

If annealing is the missing piece, 4000 episodes of polish should move the
Edax score off 37%. If it does not, the plateau is something else -- most
likely the narrow 150k replay window, or the network's capacity.

Runs on a COPY of the checkpoint directory so bench_OPT stays intact.

    python run_anneal.py
"""
import json
import os
import time

import connect4_dirichlet_utils as c4

NUM_EPISODES = 24000          # resumes at 20000, so 4000 episodes of polish
ROOT = 'othello_benchmark'
CKPT = 'bench_OPTanneal'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'anneal.log')
DESCENT_TEMP = 16.0
DECAY_EPS = 24000             # horizon = end of run, so the cosine completes
MIN_FACTOR = 0.02             # floor 4e-05 rather than 2e-04


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
        temp_threshold=30, late_temp=8.0, descent_temp=DESCENT_TEMP,
        max_plies=128, eval_max_plies=128,
        use_workers=True, selfplay_workers=4, games_per_worker=16,
        worker_wave=4, n_parallel_games=16, wave_per_game=4,
        pool_prob=0.2, random_pool_frac=0.5,
        batch_size=256, train_steps_per_ep=4, max_buffer=150_000,
        # THE change under test.
        lr_peak=2e-3, lr_warmup_eps=100, lr_decay_eps=DECAY_EPS,
        lr_min_factor=MIN_FACTOR, weight_decay=1e-4, grad_clip=1.0,
        quick_eval_every=250, quick_eval_games=30, deep_eval_every=1000,
        eval_sims=64, eval_games_per_pair=4, resume=True,
        solved_dir='',
    )


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    rule = cal2['rules']['conf']['cfg']
    _log(f'=== anneal probe: resume {ROOT}/{CKPT} -> {NUM_EPISODES} eps ===')
    _log(f'    lr_decay_eps={DECAY_EPS} lr_min_factor={MIN_FACTOR} '
         f'(was 30000 / 0.1, which left the run at 6.53e-04)')
    t0 = time.time()
    c4.run_training(make_cfg(rule), log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('ANNEAL DONE')
