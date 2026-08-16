"""Score every new training checkpoint against the external MCTS ladder.

Runs alongside training rather than inside it: the training loop is not slowed
by evaluation, a crashed or restarted watcher costs nothing, and the whole
history can be recomputed later by deleting the results file.

Each bench_<ep>.pt that appears in the checkpoint directory is played against
every ladder level under ONE fixed inference configuration (othello_ladder's
OUR_SIMS / OUR_POLICY), so the curve reflects the weights changing and not the
search rule.

    python ladder_watch.py [--ckpt DIR] [--once]
"""
import json
import os
import re
import sys
import time

import othello_ladder as ol

CKPT_DIR = 'othello_benchmark/bench_OPT'
RESULTS = 'ladder_results.json'
POLL_S = 120
# othello_ladder defaults to 2 so it can sit alongside a running trainer; when
# scoring after the fact there is nothing to share with.
WORKERS = 5


def _log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open('ladder_watch.log', 'a') as fh:
        fh.write(line + '\n')


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            return json.load(fh)
    return {}


def available(ckpt_dir):
    out = []
    if not os.path.isdir(ckpt_dir):
        return out
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r'bench_(\d+)\.pt', f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def score(ckpt_dir, ep, results):
    names = [r[0] for r in ol.RUNGS]
    _log(f'  scoring {ckpt_dir}@{ep} vs {names}')
    frac = ol.evaluate(ckpt_dir, str(ep), workers=WORKERS, log=_log)
    results[str(ep)] = frac
    with open(RESULTS, 'w') as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    _log(f'  ep {ep}: '
         + '  '.join(f'{n} {frac[n]:.1%}' for n in names))


def main():
    ckpt_dir = CKPT_DIR
    if '--ckpt' in sys.argv:
        ckpt_dir = sys.argv[sys.argv.index('--ckpt') + 1]
    once = '--once' in sys.argv
    _log(f'watching {ckpt_dir} (our side: {ol.OUR_SIMS} sims, {ol.OUR_POLICY})')
    while True:
        results = load_results()
        todo = [e for e in available(ckpt_dir) if str(e) not in results]
        for ep in todo:
            score(ckpt_dir, ep, results)
        if once:
            return
        time.sleep(POLL_S)


if __name__ == '__main__':
    main()
