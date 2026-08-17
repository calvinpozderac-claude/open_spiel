"""Is OPT@20000 > R2conf an artifact of the inference temperature?

The internal checkpoint round-robin (deterministic temp=1e4) put OPT@20000
about 190 Elo above R2conf@5000. Against Edax at temp=64, R2conf matches or
beats it at every mixture level. Exactly two things differ: the OPPONENT
(siblings vs Edax) and the inference TEMP.

This pins the temp down by replaying the same head-to-head at both settings.
If OPT still wins at temp=64, temperature is not the explanation and the
disagreement is really about who the opponent is.
"""
import itertools, multiprocessing as mp, sys, time
import numpy as np

A = ('OPT@20000', 'othello_benchmark/bench_OPT', '20000')
B = ('R2conf@5000', 'othello_benchmark/bench_R2conf', '5000')
GAMES, SIMS, N_WORKERS = 200, 64, 5


def _init():
    import torch; torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    g = c4.load_game('othello'); c4.set_game(g)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, g


def _worker(args):
    temp, n_games, seed, first_seat = args
    c4, game = _init()
    import adaptive_stop_utils as ad, policy_screen as ps
    na = c4.load_benchmark_net(A[1], A[2], ad.MM_SIG)
    nb = c4.load_benchmark_net(B[1], B[2], ad.MM_SIG)
    pol = dict(mode='value', k=0.0, sel_temp=temp)
    rng = np.random.default_rng(seed)

    def mv(net, st):
        b = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16, temp=temp,
                         random_state=rng)
        return ps.pick(c4, b.mcts_search(st), pol)

    pts = 0.0
    for gi in range(n_games):
        seat = (first_seat + gi) % 2
        st = game.new_initial_state()
        while not st.is_terminal():
            st.apply_action(mv(na if st.current_player() == seat else nb, st))
        r = st.returns()[seat]
        pts += 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
    return temp, pts, n_games


if __name__ == '__main__':
    print(f'{A[0]} vs {B[0]}, {GAMES} games, {SIMS} sims, standard start')
    for temp in (64.0, 1e4):
        per = GAMES // N_WORKERS
        jobs = [(temp, per, 7700 + 31 * w, w % 2) for w in range(N_WORKERS)]
        t0 = time.time()
        with mp.Pool(N_WORKERS) as pool:
            parts = pool.map(_worker, jobs)
        pts = sum(p for _, p, _ in parts); n = sum(k for _, _, k in parts)
        se = (pts / n * (1 - pts / n) / n) ** 0.5
        print(f'  inference temp {temp:>7g}: {A[0]} scores {pts/n:6.1%} '
              f'+/- {se:.1%}   ({time.time()-t0:.0f}s)')
