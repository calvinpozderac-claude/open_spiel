"""Joint AZ+HYBRID ladder: one calibrated Bradley-Terry scale across both
model families.

Two disjoint round-robins (az_ladder.json, hybrid_ladder.json) each fit their
own Elo anchored to a mean of 1500 -- the offset BETWEEN the two graphs is
unidentified without games connecting them.  This seeds a combined pool with
BOTH existing within-population results (reused, not recomputed) and plays
every AZ-vs-HYBRID cross pair (25 = 5x5) to link the two scales, then fits
ONE joint Bradley-Terry/Elo model over all 10 checkpoints.

    python combined_ladder.py
"""
import json
import os

import tournament as T

EPS = [10000, 20000, 30000, 40000, 50000]
AZ = [(f'AZ{ep//1000}k', 'othello_benchmark/bench_AZ_MATCH', str(ep), 'az', 1.5)
      for ep in EPS]
HY = [(f'HYBRID{ep//1000}k', 'othello_benchmark/bench_HYBRID', str(ep),
      'thompson', 'additive_mle') for ep in EPS]
T.PARTICIPANTS = AZ + HY
T.GAMES = 80
T.SIMS = 400
T.OUT = 'combined_ladder.json'

if __name__ == '__main__':
    # Seed with the existing within-population results so main() only plays
    # the new cross pairs.
    seed = {}
    for src in ('az_ladder.json', 'hybrid_ladder.json'):
        with open(src) as fh:
            seed.update(json.load(fh)['pairs'])
    if not os.path.exists(T.OUT):
        with open(T.OUT, 'w') as fh:
            json.dump({'pairs': seed, 'sims': T.SIMS}, fh)
        print(f'[combined] seeded {len(seed)} within-population pairs from '
              f'az_ladder.json + hybrid_ladder.json', flush=True)
    T.main()
