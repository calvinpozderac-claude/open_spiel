"""Two quick investigations before extending the sweep:

1. Does the 300-sim cap actually bind? Re-run the checkpointed search with a
   MUCH higher cap (1000) for a few of our thresholds and see whether the
   distribution of natural stopping points has real mass beyond 300 (some
   positions "still learning") or whether 300 already comfortably contains
   convergence.

2. Refine check_every=50 to land closer to a mean of 100 sims/move (prior
   sweep found eps=0.1 -> 97.5, but that was with only 3 candidates >= 0.1;
   this tries smaller eps too, both for precision and to keep the search
   genuinely adaptive rather than saturated at "always 2 blocks").
"""
import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4


def hist_str(used, edges):
    counts, _ = np.histogram(used, bins=edges)
    return '  '.join(f'{edges[i]}-{edges[i+1]}:{counts[i]}'
                     for i in range(len(counts)))


def run(game, net, positions, check_every, eps, cap, seed=0):
    rng = np.random.default_rng(seed)
    used = []
    for st in positions:
        bot = c4.C4MCTSBot(game, net, 'cpu', cap, batch_size=16,
                           random_state=rng)
        _root, n = bot.mcts_search_checkpointed(st, check_every=check_every,
                                                 kl_eps=eps)
        used.append(n)
    return np.array(used)


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '2000', ad.MM_SIG)
    positions = ad.sample_positions(net, game, 60, sims=100, seed=2)

    print('=== 1. Does the 300 cap bind? (cap raised to 1000) ===')
    for check_every, eps, tag in [(100, 0.005, '005'), (100, 0.030, '030'),
                                   (75, 0.1, '150match')]:
        used = run(game, net, positions, check_every, eps, cap=1000)
        n_over_300 = int((used > 300).sum())
        n_over_500 = int((used >= 500).sum())
        print(f'{tag:10} check_every={check_every:3} eps={eps:<6} '
             f'mean={used.mean():6.1f} max={used.max():4d}  '
             f'>300: {n_over_300}/{len(used)}  >=500: {n_over_500}/{len(used)}')
        print('   ' + hist_str(used, [0, 100, 200, 300, 400, 500, 700, 1000, 1001]))

    print('\n=== 2. Refining check_every=50 toward mean=100 ===')
    for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        used = run(game, net, positions, 50, eps, cap=300)
        print(f'eps={eps:<6} mean={used.mean():6.1f} median={np.median(used):5.0f} '
             f'min={used.min():4d} max={used.max():4d}')


if __name__ == '__main__':
    main()
