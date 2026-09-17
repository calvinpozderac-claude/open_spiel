"""Internal AZ strength ladder: round-robin among AZ's own checkpoints to
measure searched strength vs training episode, cutting through the noisy
self-play Elo pool.  Reuses tournament.py's engine + Elo fit.
"""
import tournament as T

EPS = [10000, 20000, 30000, 40000, 50000]
T.PARTICIPANTS = [
    (f'AZ{ep//1000}k', 'othello_benchmark/bench_AZ_MATCH', str(ep), 'az', 1.5)
    for ep in EPS
]
T.GAMES = 80
T.SIMS = 400
T.OUT = 'az_ladder.json'
if __name__ == '__main__':
    T.main()
