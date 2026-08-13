"""Calibrates two more adaptive arms for the next round of experiments:
mean sims/move ~350 and ~500 (both well above bench_MM's own 300 cap, so
max_simulations is raised accordingly). The ~350 target is already close to
what eps=0.005/check_every=100 gave under an uncapped test (mean=347.3, see
adaptive_uncapped_and_100calibrate.py); this refines around that and searches
lower eps values for the ~500 target.
"""
import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

CAP = 1000
CHECK_EVERY = 100
CANDIDATES = [0.005, 0.004, 0.003, 0.002, 0.0015, 0.001, 0.0007, 0.0005]


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '2000', ad.MM_SIG)
    positions = ad.sample_positions(net, game, 60, sims=100, seed=3)

    print('targets: ~350 and ~500 sims/move\n')
    for eps in CANDIDATES:
        rng = np.random.default_rng(0)
        used = []
        for st in positions:
            bot = c4.C4MCTSBot(game, net, 'cpu', CAP, batch_size=16,
                               random_state=rng)
            _root, n = bot.mcts_search_checkpointed(st, check_every=CHECK_EVERY,
                                                     kl_eps=eps)
            used.append(n)
        used = np.array(used)
        print(f'eps={eps:<8} mean={used.mean():7.1f} median={np.median(used):6.0f} '
             f'min={used.min():4d} max={used.max():4d}')


if __name__ == '__main__':
    main()
