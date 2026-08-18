"""Give AlphaZero a fair shake: tune its INFERENCE parameters before judging it.

The Thompson line has had three rounds of arm sweeps plus a descent-temperature
study; the AlphaZero arm has only ever been run at the settings it trained
under. Comparing a tuned method against an untuned one measures the tuning, not
the method, so this screens AZ's inference knobs the same way policy_screen did
for ours: one network plays ITSELF under two policies, same weights, same
simulation budget, so any score away from 50% is the policy.

Knobs, all inference-only (no retraining):

  c_puct    exploration constant in PUCT.  Trained at 1.5, but the value that
            is right for generating training data is not necessarily right for
            playing strength -- the same split that turned out to matter for
            our own descent temperature.
  pick      how the move is read off the searched tree:
              visits  argmax visit count -- AlphaZero's standard
              q       argmax mean action value over VISITED edges
              sample  sample the visit distribution at `temp`
            Worth testing because on the Thompson engine the equivalent choice
            was worth ~186 Elo, though the reasoning does not carry over
            directly: PUCT concentrates visits, so unlike Thompson selection
            its visit distribution is a real policy.
  temp      visit-sampling temperature, when pick='sample'

Baseline is the standard AlphaZero eval configuration: c_puct as trained,
argmax visits, no root noise.

    python az_policy_screen.py [preset] [--games 400]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

SIMS = 64
MAX_PLIES = 128
N_WORKERS = 5
NET = ('othello_benchmark/bench_AZ', '40000')
# Root Dirichlet noise on BOTH sides, always. Without it AZ is fully
# deterministic -- PUCT descent, argmax read-out, no sampling -- so a self-play
# A/B from the standard start replays ONE game and the only reachable scores
# are 0%, 50% and 100%. An earlier version of this screen ran without it and
# produced exactly that: q at c_puct 1.5 and 0.5 both "100%" while c_puct 1.0
# read "0%", which is a single game dressed up as 400. Noise perturbs the
# PRIOR rather than the final choice, so it costs far less than visit sampling
# (which measured ~9% against plain argmax) while making every game distinct.
NOISE = 0.15
BASELINE = dict(c_puct=1.5, pick='visits', temp=0.3, noise=NOISE)

PRESETS = {
    # Is the trained c_puct the right one to PLAY at?
    'cpuct': [('c_puct 0.5', dict(c_puct=0.5)),
              ('c_puct 1.0', dict(c_puct=1.0)),
              ('c_puct 2.5', dict(c_puct=2.5)),
              ('c_puct 4.0', dict(c_puct=4.0)),
              ('null', dict())],
    # How should the move be read off the tree?
    # Combining the two winners, plus root Dirichlet noise as a variety source.
    # Variety is needed because q-argmax is deterministic, and the obvious
    # alternative -- visit sampling -- costs ~90% of games (sample t=0.3 scored
    # 9.1% against plain argmax visits). Noise perturbs the PRIOR instead of
    # the final choice, which should be far cheaper.
    # Redone with noise on both sides so every cell is a real 400 games.
    'combo': [('q-argmax', dict(pick='q')),
              ('cpuct 0.5', dict(c_puct=0.5)),
              ('cpuct 1.0', dict(c_puct=1.0)),
              ('cpuct 2.5', dict(c_puct=2.5)),
              ('q + cpuct1.0', dict(pick='q', c_puct=1.0)),
              ('q + cpuct0.5', dict(pick='q', c_puct=0.5)),
              ('null', dict())],
    # c_puct is monotone INCREASING up to 2.5 (0.5: 41.4%, 1.0: 46.6%,
    # 1.5: baseline, 2.5: 61.6%) and q-argmax is worth 64.2% on its own, so
    # push exploration further and test the combination.
    'combo2': [('cpuct 4.0', dict(c_puct=4.0)),
               ('cpuct 6.0', dict(c_puct=6.0)),
               ('q + cpuct2.5', dict(pick='q', c_puct=2.5)),
               ('q + cpuct4.0', dict(pick='q', c_puct=4.0)),
               ('q + cpuct6.0', dict(pick='q', c_puct=6.0)),
               ('null', dict())],
    'pick': [('q-argmax', dict(pick='q')),
             ('sample t=0.3', dict(pick='sample', temp=0.3)),
             ('sample t=1.0', dict(pick='sample', temp=1.0)),
             ('null', dict())],
}


def _init():
    import torch
    torch.set_num_threads(1)
    import connect4_dirichlet_utils as c4
    import connect4_alphazero_utils as az
    game = c4.load_game('othello')
    c4.set_game(game)
    az.set_search(c_puct=1.5, virtual_loss=1.0)
    return c4, az, game


def pick_move(az, root, rng, pol):
    """Read a move off an already-searched AZ root."""
    mode = pol.get('pick', 'visits')
    if mode == 'visits':
        return az.root_pick(root, rng, sample=False)
    if mode == 'sample':
        return az.root_pick(root, rng, sample=True, temp=pol.get('temp', 1.0))
    if mode == 'q':
        # Mean action value, restricted to edges search actually visited so an
        # untouched edge cannot win on a zero default.
        q = root.q()
        seen = (root.N > 0) | (root.term >= 0)
        if seen.any():
            q = np.where(seen, q, -np.inf)
        return int(root.legal[int(np.argmax(q))])
    raise ValueError(mode)


def play(c4, az, game, net, pol0, pol1, rng):
    """One game from the standard start; pol0 moves first."""
    st = game.new_initial_state()
    ply = 0
    while not st.is_terminal() and ply < MAX_PLIES:
        pol = pol0 if st.current_player() == 0 else pol1
        az.set_search(c_puct=pol.get('c_puct', 1.5))
        bot = az.AZMCTSBot(game, net, 'cpu', SIMS, batch_size=8,
                           random_state=rng, noise_frac=pol.get('noise', 0.0))
        st.apply_action(pick_move(az, bot.mcts_search(st), rng, pol))
        ply += 1
    return st.returns()[0]


def _worker(args):
    cand, n_games, seed = args
    c4, az, game = _init()
    net = az.load_benchmark_net(NET[0], NET[1], (32, 3, 8))
    rng = np.random.default_rng(seed)
    pts, played = 0.0, 0
    for gi in range(n_games // 2):
        # Colours swapped between the two games of each pair.
        r0 = play(c4, az, game, net, cand, BASELINE, rng)
        r1 = play(c4, az, game, net, BASELINE, cand, rng)
        pts += 0.5 * (r0 + 1.0) + 0.5 * ((-r1) + 1.0)
        played += 2
    return pts, played


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return c - h, c + h


def run(name, cand, games):
    per = max(2, games // N_WORKERS)
    jobs = [(cand, per, 300 + 811 * w) for w in range(N_WORKERS)]
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        parts = pool.map(_worker, jobs)
    pts = sum(p for p, _ in parts)
    n = sum(g for _, g in parts)
    frac = pts / n
    lo, hi = wilson(frac, n)
    flag = '' if lo < 0.5 < hi else '   <-- separated from 50%'
    print(f'  {name:<16} {frac:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}  '
          f'{time.time()-t0:.0f}s{flag}', flush=True)


if __name__ == '__main__':
    preset = sys.argv[1] if len(sys.argv) > 1 else 'cpuct'
    games = 400
    if '--games' in sys.argv:
        games = int(sys.argv[sys.argv.index('--games') + 1])
    print(f'AZ self-play A/B: {NET[0]}@{NET[1]}, {SIMS} sims, {games} games')
    print(f'baseline = {BASELINE} (standard AlphaZero eval)')
    for nm, c in PRESETS[preset]:
        run(nm, {**BASELINE, **c}, games)
