"""Tier 0 diagnostic: does the "plateau then jump" failure mode (an early
checkpoint looks converged, a later one isn't) actually happen, how often,
how much belief-change it hides, and does anything computable BEFORE search
(the network's own raw prior) predict it? No training involved -- this is
pure data collection + analysis on an existing checkpoint, to decide whether
building a learned head is actually worth it before we build one.

Method: for each of n_positions, record pre-search features from the raw
network prior (top1-vs-top2 action-value margin, state confidence), then run
a full checkpointed search with kl_eps=None (never stop early) so we get the
COMPLETE block-by-block KL trace out to a generous cap. Then check: would our
tightest calibrated threshold (eps=0.005) have stopped early, and if so, how
much KL showed up in LATER blocks that it would have missed.
"""
import json
import time

import numpy as np

import adaptive_stop_utils as ad
import connect4_dirichlet_utils as c4

CHECK_EVERY = 50
CAP = 1000
N_POSITIONS = 150
STOP_EPS = 0.005          # our tightest calibrated threshold (the '005' arm)
JUMP_ABS = 0.02            # a later block KL bigger than this is a "real" jump
JUMP_MULT = 5.0            # ...or 5x the block that would have triggered stop


def pre_search_features(net, game, state):
    """Features computable before any search: from the network's raw prior
    only (root.alpha_p / root.v_alpha at expansion, before any backup)."""
    root = c4.expand_node(net, 'cpu', state)
    vals = c4.dir_value(root.alpha_p)          # per-legal-action v = p_win-p_loss
    order = np.argsort(-vals)
    top1 = float(vals[order[0]])
    top2 = float(vals[order[1]]) if len(vals) > 1 else top1
    margin = top1 - top2
    state_conf = float(root.v_alpha.sum())
    action_conf_top1 = float(root.alpha_p[order[0]].sum())
    return dict(margin=margin, state_conf=state_conf,
               action_conf_top1=action_conf_top1, n_legal=len(vals))


def main():
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    net = c4.load_benchmark_net('othello_benchmark/bench_MM', '5000', ad.MM_SIG)
    positions = ad.sample_positions(net, game, N_POSITIONS, sims=150, seed=7)
    print(f'{len(positions)} positions sampled, running full traces '
         f'(check_every={CHECK_EVERY}, cap={CAP})...')

    rows = []
    t0 = time.time()
    rng = np.random.default_rng(0)
    for i, st in enumerate(positions):
        feats = pre_search_features(net, game, st)
        bot = c4.C4MCTSBot(game, net, 'cpu', CAP, batch_size=16,
                           random_state=rng)
        trace = []
        bot.mcts_search_checkpointed(st, check_every=CHECK_EVERY, kl_eps=None,
                                     trace=trace)
        rows.append(dict(feats=feats, trace=trace))
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(positions)}  ({time.time()-t0:.0f}s)')

    with open('tier0_diagnostic_raw.json', 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'raw data written to tier0_diagnostic_raw.json ({time.time()-t0:.0f}s)')

    # ── analysis ────────────────────────────────────────────────────────────
    plateau_jump = []     # 1/0 per position: would eps=0.005 have stopped
                          # early AND missed a later real jump?
    missed_kl = []         # sum of KL in blocks after the would-have-stopped
                          # point, for positions that show the pattern (0 else)
    never_converges = 0
    for row in rows:
        trace = row['trace']       # [(sims, kl), ...], kl is None-free here
                                   # since kl_eps=None still logs every block
        kls = [kl for _sims, kl in trace]
        if not kls:
            continue
        k_star = next((i for i, kl in enumerate(kls) if kl < STOP_EPS), None)
        if k_star is None:
            never_converges += 1
            plateau_jump.append(0)
            missed_kl.append(0.0)
            continue
        later = kls[k_star + 1:]
        jump = any(kl > max(JUMP_ABS, JUMP_MULT * kls[k_star]) for kl in later)
        plateau_jump.append(1 if jump else 0)
        missed_kl.append(sum(later) if jump else 0.0)

    plateau_jump = np.array(plateau_jump)
    missed_kl = np.array(missed_kl)
    margins = np.array([r['feats']['margin'] for r in rows])
    state_conf = np.array([r['feats']['state_conf'] for r in rows])

    print(f'\n=== results (n={len(rows)}) ===')
    print(f'never converged under eps={STOP_EPS} even by cap={CAP}: '
         f'{never_converges}/{len(rows)} ({never_converges/len(rows):.1%})')
    print(f'"plateau then jump" (would stop early AND miss a later real '
         f'jump): {plateau_jump.sum()}/{len(rows)} ({plateau_jump.mean():.1%})')
    if plateau_jump.sum() > 0:
        print(f'  among those, mean missed KL = {missed_kl[plateau_jump==1].mean():.4g}')

    print('\n--- does |top1-top2 margin| predict it? ---')
    abs_margin = np.abs(margins)
    q = np.percentile(abs_margin, [25, 50, 75])
    buckets = np.digitize(abs_margin, q)
    for b in range(4):
        mask = buckets == b
        lo = 0 if b == 0 else q[b-1]
        hi = q[b] if b < 3 else abs_margin.max()
        print(f'  |margin| in [{lo:.3f},{hi:.3f}]: n={mask.sum():3d}  '
             f'plateau-jump rate={plateau_jump[mask].mean():.1%}')
    corr = np.corrcoef(abs_margin, plateau_jump)[0, 1]
    print(f'  correlation(|margin|, plateau-jump): {corr:+.3f}')

    print('\n--- does state confidence predict it? ---')
    q = np.percentile(state_conf, [25, 50, 75])
    buckets = np.digitize(state_conf, q)
    for b in range(4):
        mask = buckets == b
        lo = 0 if b == 0 else q[b-1]
        hi = q[b] if b < 3 else state_conf.max()
        print(f'  conf in [{lo:.2f},{hi:.2f}]: n={mask.sum():3d}  '
             f'plateau-jump rate={plateau_jump[mask].mean():.1%}')
    corr = np.corrcoef(state_conf, plateau_jump)[0, 1]
    print(f'  correlation(state_conf, plateau-jump): {corr:+.3f}')


if __name__ == '__main__':
    main()
