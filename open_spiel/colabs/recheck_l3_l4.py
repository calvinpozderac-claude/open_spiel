"""Re-score a already-benchmarked checkpoint at L3/L4 to test whether the
ep17000 L3/L4 inversion is the model or the measurement.

ep16000 was scored on Aug 21 as L3 32.4%, L4 2.0%. The network file is frozen,
so any change here is the harness, not the model. Writes nothing to
mm_long_bench.json -- climb() is bypassed and the rungs dict is a throwaway.

    python recheck_l3_l4.py [label]
"""
import sys
import time

import mm_long_bench as B

LABEL = sys.argv[1] if len(sys.argv) > 1 else '16000'
PRIOR = {'16000': {3.0: 0.324, 4.0: 0.020}}

if __name__ == '__main__':
    print(f'[recheck] re-scoring ep {LABEL} at L3 and L4 '
          f'({B.GAMES_PER_LEVEL} games, {B.SIMS} sims, temp {B.OUR_TEMP})',
          flush=True)
    rungs = {}
    for rung in (3.0, 4.0):
        t0 = time.time()
        frac = B._play_level(LABEL, rung, rungs)
        was = PRIOR.get(LABEL, {}).get(rung)
        tag = '' if was is None else f'  (was {was:.1%}, delta {frac-was:+.1%})'
        print(f'[recheck] ep {LABEL} L{rung:g}: {frac:.1%}{tag} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    print('[recheck] RECHECK DONE', flush=True)
