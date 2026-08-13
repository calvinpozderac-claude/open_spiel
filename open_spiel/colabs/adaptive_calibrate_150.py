"""Finds a kl_eps that pushes mean sims/move down to ~150 (bench_MM's own
fast_sims/full_sims blended average), so a follow-up adaptive-MM run is
compute-MATCHED to bench_MM rather than compared against it at a higher
average budget (the confound flagged after the first sweep).

Sweeps a handful of eps candidates against bench_MM/bench_2000.pt (a
representative trained net) and reports mean sims/move for each.
"""
import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

TARGET = 150
# The first checkpoint's KL is against the raw (or lightly-visited, under tree
# reuse) prior, which is a big belief shift almost regardless of eps -- the
# sweep in adaptive_mm_report.py showed eps>=0.08 already never converges at
# checkpoint 1, so mean sims floors at ~2*check_every regardless of how large
# eps gets. To reach ~150 the block size itself has to shrink, not just eps.
GRID = [
    # (check_every, [eps candidates])
    (100, [0.6]),           # confirms the floor found in adaptive_mm_report.py
    (75,  [0.1, 0.3, 0.6]),
    (50,  [0.1, 0.3, 0.6]),
    (40,  [0.1, 0.3, 0.6]),
]


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '2000', ad.MM_SIG)
    positions = ad.sample_positions(net, game, 50, sims=100, seed=1)

    print(f'target: <= {TARGET} sims/move (bench_MM\'s own fast/full average)\n')
    for check_every, epss in GRID:
        for eps in epss:
            rng = np.random.default_rng(0)
            used = []
            for st in positions:
                bot = c4.C4MCTSBot(game, net, 'cpu', 300, batch_size=16,
                                   random_state=rng)
                _root, n = bot.mcts_search_checkpointed(
                    st, check_every=check_every, kl_eps=eps)
                used.append(n)
            used = np.array(used)
            print(f'check_every={check_every:<4} eps={eps:<5} '
                 f'mean={used.mean():6.1f} median={np.median(used):5.0f} '
                 f'min={used.min():4d} max={used.max():4d}')


if __name__ == '__main__':
    main()
