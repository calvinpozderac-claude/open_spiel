"""MM_long's exact configuration, with mixture aggregation instead of additive_mle.

The first mixture run (run_mixture.py) was built at OLIVAW scale and differed
from MM_long in SEVENTEEN fields -- 1000 sims/move vs a 220 average, 3 blocks
vs 10, 4 gradient steps of 256 vs 1 of 1024, a 20% frozen-opponent pool vs pure
self-play, and more.  It lost the head-to-head 96.5% to 3.5%, which says almost
nothing about the aggregation rule because so much else moved with it.

This run changes exactly one thing.  The config is built with
dataclasses.replace() on run_mm_long.make_cfg(), so every field except
checkpoint_dir and the two aggregation knobs is identical BY CONSTRUCTION
rather than by retyping -- see the assertion in __main__, which refuses to
start if anything else differs.

WHY IT MIGHT STILL FAIL, and what to watch.  The OLIVAW-scale mixture run
collapsed: 95% of its opening targets landed on the _floor_conc floor (0.15)
and the network learned to sit there, which is exactly the defect that got
mixture dropped as the default in f4f760b0 ("about 54% of unsolved state
targets were landing on the floor, carrying a direction but no scale").  The
mechanism is a feedback loop -- mixture variance is within-leaf plus
between-leaf, within-leaf is m(1-m)/(alpha0+1), so a falling alpha0 RAISES the
variance, which lowers the fitted beta0, which teaches a lower alpha0.  At
alpha0 = 0.15 across three classes each component is ~0.05, deep in the
corner-peaked regime, so Thompson selection in the opening becomes nearly
random.  That is the likeliest explanation for the head-to-head loss.

This run is not immune to that loop.  What differs is the evidence feeding it:
1000 sims/move instead of 220 (more observations per node, so the between-leaf
term carries more weight relative to within-leaf), and a 20% frozen/random
opponent pool (more genuinely disagreeing positions).  Both push toward
between-leaf variance dominating, which is the term that does not feed back.
Whether that is enough is the open question.

WATCH: the fraction of unsolved opening targets at the 0.15 floor.  If it
climbs past ~50% the loop is winning again; quick_report's `a0 open` pinning at
0.15 is the same signal from the network side.

ONE UNAVOIDABLE DEVIATION from MM_long as it actually ran.  Two bugs have been
fixed since: _conf no longer clamps concentration from below (it was an
absorbing state with exactly zero gradient), and the output heads initialise
with noise rather than zeros (zero weights made every position bit-identical,
which alone inflated the untrained target alpha0 from 110 to 205).  Both are in
the shared module and apply here.  The _conf fix is behaviourally inert for
MM_long's regime (its alpha0 lived at 50-100, nowhere near the dead zone), but
the init change is real and affects training from step one.  Reverting to a
known-bad init for strict parity would be worse science than accepting this,
but it means the comparison is "mixture on fixed code" vs "additive_mle on the
code as it was", not a perfect A/B.

    python run_mixture_long.py       # under run_mixture_long_forever.ps1
"""
import os
import time
from dataclasses import replace

import connect4_dirichlet_utils as c4
import run_mm_long as MM

ROOT = 'othello_benchmark'
CKPT = 'bench_MIXTURE_long'
LOG_PATH = os.path.join(os.path.dirname(__file__), 'mixture_long.log')

# Everything except these is inherited from MM_long verbatim.
OVERRIDES = dict(
    checkpoint_dir=os.path.join(ROOT, CKPT),
    search_agg=c4.AGG_MIXTURE,
    target_agg=c4.AGG_MIXTURE,
)


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def make_cfg():
    return replace(MM.make_cfg(), **OVERRIDES)


def assert_parity():
    """Refuse to start if anything but the intended knobs differs."""
    from dataclasses import asdict
    a, b = asdict(MM.make_cfg()), asdict(make_cfg())
    diffs = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    expected = set(OVERRIDES)
    if diffs != expected:
        raise SystemExit(
            f'PARITY FAILURE -- unexpected differences vs MM_long: '
            f'{sorted(diffs - expected)}; missing: {sorted(expected - diffs)}')
    return sorted(diffs)


if __name__ == '__main__':
    changed = assert_parity()
    cfg = make_cfg()
    _log(f'=== MIXTURE_long: MM_long config, mixture aggregation -> '
         f'{ROOT}/{CKPT} ===')
    _log(f'    parity verified: the ONLY fields differing from run_mm_long '
         f'are {changed}')
    _log(f'    {cfg.num_episodes} eps | {cfg.num_blocks} blocks x '
         f'{cfg.channels} ch | sims {cfg.fast_sims}/{cfg.full_sims}'
         f'@{cfg.fast_prob} | batch {cfg.batch_size} x '
         f'{cfg.train_steps_per_ep} steps/ep | buffer {cfg.max_buffer//1000}k '
         f'| pool {cfg.pool_prob}/{cfg.random_pool_frac} | '
         f'temp_thr {cfg.temp_threshold} late {cfg.late_temp} '
         f'descent {cfg.descent_temp} | lr {cfg.lr_peak:.0e} -> '
         f'{cfg.lr_peak*cfg.lr_min_factor:.1e} | backup={cfg.backup}')
    t0 = time.time()
    c4.run_training(cfg, log=_log)
    _log(f'=== done in {time.time()-t0:.0f}s ===')
    _log('MIXTURE_LONG DONE')
