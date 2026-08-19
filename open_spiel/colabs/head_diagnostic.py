"""Is AlphaZero's value head the broken part -- and does its own search make it
worse?

Motivation: AZ's best inference configuration suppresses its own value
estimates. c_puct 6.0 scores 68.5% against the trained 1.5, and at 64 sims
with c_puct that high the exploration term swamps Q, so visits follow the
PRIOR and the move approaches the raw policy output. An agent that improves
when you stop it using its search values is pointing at the evaluator.

That is a hypothesis, not a measurement. Four things get measured here, on the
same positions, for BOTH engines:

  raw value       the value head with no search at all
  searched value  the root value after a 64-simulation search
  sign accuracy   does it get win/loss right
  search delta    searched error minus raw error.  If this is POSITIVE the
                  value head is poisoning its own tree, which is the whole
                  hypothesis.

GROUND TRUTH is exact, not a proxy. Positions are sampled with few empties and
played out Edax-vs-Edax at a level that solves them outright, so the result is
the game-theoretic value rather than one engine's opinion. Endgames are the
easy case for a value head, so a net can pass here and still be poor in the
midgame -- midgame positions are therefore sampled too, with the caveat that
their "truth" is only strong-play outcome.

The two engines report differently: ours is a Dirichlet over {win, draw, loss}
and AZ is a scalar. Both are reduced to E[win] - E[loss] in the mover's
frame so the comparison is like for like.

    python head_diagnostic.py [--positions 200]
"""
import sys
import time

import numpy as np

OUR_SIMS = 64
N_POSITIONS = 200
ENDGAME_EMPTIES = 12          # few enough that Edax solves exactly
MIDGAME_EMPTIES = 30
EDAX_SOLVE_LEVEL = 18         # comfortably exact at 12 empties
# Midgame truth is a strong-play OUTCOME, not a solve, so a cheaper level costs
# little fidelity and a lot of wall clock: at level 18 a 30-ply playout ran
# ~15s, which is 50 minutes of pure ground truth for 200 positions.
EDAX_MID_LEVEL = 10
AZ_CPUCT = 6.0                # AZ's tuned setting
OUR_TEMP = 64.0

NETS = [
    ('OPT@20000',  'othello_benchmark/bench_OPT',   '20000', 'thompson'),
    ('dt16@2000',  'othello_benchmark/bench_Tdt16',  '2000', 'thompson'),
    ('AZ@20000',   'othello_benchmark/bench_AZ',    '20000', 'az'),
    ('AZ@40000',   'othello_benchmark/bench_AZ',    '40000', 'az'),
]


def _init():
    import torch
    torch.set_num_threads(4)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def sample_positions(c4, game, n, empties, rng):
    """Random-play positions with roughly `empties` squares free."""
    out = []
    target_plies = 60 - empties
    while len(out) < n:
        st = game.new_initial_state()
        for _ in range(target_plies):
            if st.is_terminal():
                break
            la = st.legal_actions()
            st.apply_action(int(la[rng.integers(len(la))]))
        if not st.is_terminal():
            out.append(st.clone())
    return out


def ground_truth(game, positions, eb, level=EDAX_SOLVE_LEVEL):
    """Outcome under Edax-vs-Edax play, in each position's mover's frame."""
    bot = eb.EdaxBot(game, level=level)
    out = []
    for st0 in positions:
        mover = st0.current_player()
        st = st0.clone()
        while not st.is_terminal():
            st.apply_action(bot.step(st))
        r = st.returns()[mover]
        out.append(1.0 if r > 0 else (-1.0 if r < 0 else 0.0))
    bot.close()
    return np.array(out)


def evaluate_net(c4, game, name, ckpt, label, kind, positions):
    """(raw value, searched value) per position, both in the mover's frame."""
    import adaptive_stop_utils as ad
    raw, searched = [], []
    if kind == 'az':
        import connect4_alphazero_utils as az
        az.set_search(c_puct=AZ_CPUCT, virtual_loss=1.0)
        net = az.load_benchmark_net(ckpt, label, ad.MM_SIG)
        rng = np.random.default_rng(0)
        for st in positions:
            node = az.expand_node(net, 'cpu', st)
            raw.append(float(node.value))          # scalar head, mover's frame
            root = az.AZMCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=8,
                                random_state=rng).mcts_search(st)
            searched.append(float(az.root_value(root)))
    else:
        net = c4.load_benchmark_net(ckpt, label, ad.MM_SIG)
        rng = np.random.default_rng(0)
        for st in positions:
            v3, _vc, _p3, _cf, _o = c4.nn_eval_states(net, 'cpu', [st])
            raw.append(float(v3[0][c4._WIN] - v3[0][c4._LOSS]))
            root = c4.C4MCTSBot(game, net, 'cpu', OUR_SIMS, batch_size=16,
                                temp=OUR_TEMP, random_state=rng
                                ).mcts_search(st)
            a = c4.observed_alpha(root.nacc, c4._SEARCH_AGG)
            searched.append(float(c4.dir_value(a)))
    return np.array(raw), np.array(searched)


def report(tag, truth, results):
    print(f'\n=== {tag}  (n={len(truth)}, ground truth = Edax-solved) ===')
    print(f'{"net":<12}{"raw MSE":>9}{"srch MSE":>10}{"delta":>8}'
          f'{"raw sign":>10}{"srch sign":>11}')
    for name, (raw, srch) in results.items():
        mr = float(((raw - truth) ** 2).mean())
        ms = float(((srch - truth) ** 2).mean())
        sr = float((np.sign(raw) == np.sign(truth)).mean())
        ss = float((np.sign(srch) == np.sign(truth)).mean())
        flag = '   <-- search HURTS' if ms > mr else ''
        print(f'{name:<12}{mr:>9.3f}{ms:>10.3f}{ms-mr:>+8.3f}'
              f'{sr:>9.1%}{ss:>10.1%}{flag}')


def main(n):
    c4, game = _init()
    import edax_bot as eb
    rng = np.random.default_rng(17)
    for tag, empties in (('ENDGAME', ENDGAME_EMPTIES),
                         ('MIDGAME', MIDGAME_EMPTIES)):
        t0 = time.time()
        pos = sample_positions(c4, game, n, empties, rng)
        lvl = EDAX_SOLVE_LEVEL if tag == 'ENDGAME' else EDAX_MID_LEVEL
        truth = ground_truth(game, pos, eb, level=lvl)
        results = {}
        for name, ck, lab, kind in NETS:
            results[name] = evaluate_net(c4, game, name, ck, lab, kind, pos)
        report(f'{tag} (~{empties} empties, Edax L{lvl})', truth, results)
        print(f'  truth mix: win {np.mean(truth>0):.0%} '
              f'draw {np.mean(truth==0):.0%} loss {np.mean(truth<0):.0%}'
              f'   ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    n = N_POSITIONS
    if '--positions' in sys.argv:
        n = int(sys.argv[sys.argv.index('--positions') + 1])
    main(n)
