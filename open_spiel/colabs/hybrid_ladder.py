"""Internal HYBRID strength ladder -- the Thompson-side analogue of
az_ladder.py, same protocol (80 games/pair, 400 sims, same 5 episode points),
so the searched-strength-vs-episode curves are directly comparable.

Thompson's own search (Thompson sampling draws every simulation) is
stochastic even under greedy root_pick, so this doesn't have the AZ-vs-AZ
determinism bug az_ladder.py hit -- no eval-protocol fix needed here.
"""
import tournament as T

EPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
T.PARTICIPANTS = [
    (f'HYBRID{ep//1000}k', 'othello_benchmark/bench_HYBRID', str(ep),
     'thompson', 'additive_mle')
    for ep in EPS
]
T.GAMES = 80
T.SIMS = 400
T.OUT = 'hybrid_ladder.json'
if __name__ == '__main__':
    T.main()
