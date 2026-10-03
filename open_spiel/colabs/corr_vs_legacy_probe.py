"""How does the 'corr' belief compare with the rules TZ actually used, on real
searched positions?  Measurement only -- no training.

One search tree carries every accumulator, so the SAME tree is read through each
rule: avgconc (the HYBRID target), additive_mle (the HYBRID search rule), and the
corr posterior at several rho.  Positions come from the trained OLIVAW-TZ net,
reached from XOT openings by search-free play, bucketed by empties left.

Answers two concrete questions the spec leaves implicit:
  * is corr at rho = 0.999 really "close to avgconc" (its P1 step)?
  * how strongly does rho move concentration, i.e. what does the P2 sweep span?

    python corr_vs_legacy_probe.py
"""
import sys

import numpy as np

import connect4_dirichlet_utils as c4

CKPT = ('othello_benchmark/bench_OLIVAW_TZ', '50000', (64, 10, 8))
RHOS = (0.05, 0.1, 0.25, 0.5, 0.9, 0.999)
PLIES = (8, 22, 36, 50)
SIMS = 400


def positions(game, net, rng, n_open=10):
    ops = [l.strip() for l in open('xot_large.txt')][:n_open]
    import edax_bot as eb
    amap = eb._action_map(game)
    out = []
    for seq in ops:
        for ply in PLIES:
            st = game.new_initial_state()
            for k in range(0, 16, 2):
                st.apply_action(amap[seq[k:k + 2]])
            while not st.is_terminal() and len(st.history()) < ply:
                st.apply_action(c4.thompson_value_move(net, st, 'cpu', rng, 8.0))
            if not st.is_terminal():
                out.append((ply, st.clone()))
    return out


def mean_vec(a):
    a = np.asarray(a, dtype=np.float64)
    return a / a.sum()


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    # HYBRID search, corr bookkeeping on (accumulators then hold every statistic)
    c4.set_search(search_agg='additive_mle', target_agg='corr', selection='dirichlet',
                  virtual_loss=1.0, rho_target=0.999)
    net = c4.load_benchmark_net(*CKPT)
    rng = np.random.default_rng(0)
    bot = c4.C4MCTSBot(game, net, 'cpu', SIMS, batch_size=16, temp=64.0,
                       random_state=rng)
    rows = []
    for ply, st in positions(game, net, rng):
        root = bot.mcts_search(st)
        if c4._node_solved_outcome(root) is not None:
            continue
        acc, prior = root.nacc, root.v_alpha
        rec = {'ply': ply, 'n': acc[0], 'prior': prior,
               'avgconc': c4.observed_alpha(acc, c4.AGG_AVGCONC),
               'mle': c4.observed_alpha(acc, c4.AGG_ADDITIVE_MLE)}
        for r in RHOS:
            rec[r] = c4._corr_belief(prior, acc, r)
        rows.append(rec)
    print(f'{len(rows)} unsolved searched roots, {SIMS} sims, TZ50k\n')

    def med(f, sel):
        v = [f(r) for r in rows if sel(r)]
        return float(np.median(v)) if v else float('nan')

    print('median state alpha0 of the ROOT belief, by empties-removed bucket (ply):')
    print('  ply   |  prior  avgconc  add_mle | ' + ' '.join(f'rho{r:<5g}' for r in RHOS))
    for p in PLIES:
        sel = lambda r, p=p: r['ply'] == p
        cells = [med(lambda r: r['prior'].sum(), sel),
                 med(lambda r: r['avgconc'].sum(), sel),
                 med(lambda r: r['mle'].sum(), sel)]
        corr = [med(lambda r, k=k: r[k].sum(), sel) for k in RHOS]
        print(f'  {p:3d}   | {cells[0]:6.1f}  {cells[1]:6.1f}  {cells[2]:6.1f}  | '
              + ' '.join(f'{c:7.1f}' for c in corr))
    print('\nrho = 0.999 vs avgconc (the spec\'s "close to avgconc" claim):')
    ratio = [r[0.999].sum() / r['avgconc'].sum() for r in rows]
    l1 = [np.abs(mean_vec(r[0.999]) - mean_vec(r['avgconc'])).sum() for r in rows]
    print(f'  alpha0 ratio corr/avgconc: median {np.median(ratio):.2f}  '
          f'p10 {np.percentile(ratio, 10):.2f}  p90 {np.percentile(ratio, 90):.2f}')
    print(f'  L1 distance between the two MEANS: median {np.median(l1):.3f}  '
          f'p90 {np.percentile(l1, 90):.3f}')
    print('\nrho = 0.1 vs additive_mle (a P2 search candidate vs the HYBRID rule):')
    ratio = [r[0.1].sum() / r['mle'].sum() for r in rows]
    print(f'  alpha0 ratio corr/mle: median {np.median(ratio):.2f}  '
          f'p10 {np.percentile(ratio, 10):.2f}  p90 {np.percentile(ratio, 90):.2f}')
    spikes = np.mean([r[0.1].sum() >= 0.99 * c4._CORR_CFG.a0_max for r in rows])
    print(f'  fraction of roots resolved to a spike at rho 0.1: {spikes:.3f}')


if __name__ == '__main__':
    sys.exit(main())
