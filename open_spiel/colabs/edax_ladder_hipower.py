"""High-power Edax scoring on a subset of checkpoints.

The 30-game pass in othello_ladder is enough to CHOOSE rungs but not to track a
curve: 30 games is a ~9% standard error, so "48.3%" really means [31.7%,
65.4%]. Underpowered evaluation has already produced three wrong readings in
this project -- the in-training DEEP Elo printing identical ladders for
different networks, the MCTS rungs pinned at 100%, and three consecutive
insignificant checkpoint steps that summed to a decisive +215 Elo -- so the
curve that gets believed should be measured properly.

Edax makes that affordable in a way the MCTS rungs never were: ~11s per 30
games against a rung, against ~56s per single game for MCTS-1600. So games
are spent where they resolve something.

  discriminating rungs (e30..e05)   200 games -> ~3.5% standard error
  bracket rungs (e45, full)         100 games; these are near 0% or 100% for
                                    most checkpoints and only need to confirm
                                    which side of the ladder a checkpoint is on
  fewer checkpoints                 9 instead of 21; the coarse pass already
                                    gives the shape, this gives the error bars

    python edax_ladder_hipower.py
"""
import json
import os
import sys

import othello_ladder as ol

CKPT_DIR = 'othello_benchmark/bench_OPT'
LABELS = ['0', '1000', '2000', '4000', '6000', '8000', '12000', '16000',
          '20000']
RESULTS = 'ladder_results_edax_hipower.json'
WORKERS = 5

RUNGS = [
    ('edax-e45',  ('edax', 2, 0.45), 100),
    ('edax-e30',  ('edax', 2, 0.30), 200),
    ('edax-e20',  ('edax', 2, 0.20), 200),
    ('edax-e10',  ('edax', 2, 0.10), 200),
    ('edax-e05',  ('edax', 2, 0.05), 200),
    ('edax-full', ('edax', 2, 0.0),  100),
]


def _log(msg):
    print(msg, flush=True)
    with open('edax_hipower.log', 'a') as fh:
        fh.write(msg + '\n')


def main():
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            done = json.load(fh)
    for lab in LABELS:
        if lab in done:
            continue
        if not os.path.exists(os.path.join(CKPT_DIR, f'bench_{lab}.pt')):
            _log(f'  {lab}: no checkpoint, skipping')
            continue
        _log(f'--- ep {lab} ---')
        frac = ol.evaluate(CKPT_DIR, lab, rungs=RUNGS, workers=WORKERS,
                           log=_log)
        done[lab] = frac
        with open(RESULTS, 'w') as fh:
            json.dump(done, fh, indent=2, sort_keys=True)
    _log('HIPOWER DONE')


if __name__ == '__main__':
    main()
