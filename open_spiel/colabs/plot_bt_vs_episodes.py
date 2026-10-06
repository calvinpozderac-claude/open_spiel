"""Plot Bradley-Terry rating against training episodes for the TZ and CORR arms.

    python plot_bt_vs_episodes.py [out.png]

ONE joint BT fit over every participant, pooling the TZ-TZ pairs from the OLIVAW
ladder with the CORR pairs from the CORR ladder.  Both ladders used the identical
protocol -- the same 12 XOT openings x both seats, 400 sims, greedy -- so the
pairs are poolable and the two arms land on a single scale.  A per-arm fit would
NOT be comparable: the fit is mean-anchored, so each arm's zero would be its own.

Uncertainty is a nonparametric bootstrap over GAMES WITHIN EACH PAIR: every pair's
(W,D,L) is resampled multinomially at its own game count and the whole ladder is
refit.  That propagates exactly the sampling noise that 24-game pairs carry, which
is the thing that has repeatedly made small gaps look real.  It does NOT cover the
other error source here -- each checkpoint is a single training seed, so a gap
inside the band is not evidence of a difference, and a gap outside it is evidence
about these two runs, not about the two methods in general.
"""
import json
import sys

import numpy as np

import tournament as T

TZ_JSON = 'olivaw_live_ladder_s400_g24_xot_greedy.json'
CO_JSON = 'corr_live_ladder_s400_g24_xot_greedy.json'
REG = 1.6e5
N_BOOT = 400
SEED = 11


def load_pairs():
    pairs = {}
    for path in (TZ_JSON, CO_JSON):
        with open(path) as fh:
            for k, v in json.load(fh)['pairs'].items():
                w, d, l, _g = T._wdl_of(v)
                pairs[k] = (w, d, l)          # CO_JSON holds only CORR pairs
    return pairs


def matrices(names, pairs, draw=None):
    """draw=None uses the observed W/D/L; otherwise draw(w,d,l) resamples it."""
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    for key, (w, d, l) in pairs.items():
        a, b = key.split('|')
        if a not in idx or b not in idx:
            continue
        if draw is not None:
            w, d, l = draw(w, d, l)
        i, j = idx[a], idx[b]
        g = w + d + l
        wins[i][j] += w + 0.5 * d
        wins[j][i] += l + 0.5 * d
        games[i][j] += g
        games[j][i] += g
    return wins, games


def main(argv):
    out = argv[1] if len(argv) > 1 else 'bt_vs_episodes.png'
    pairs = load_pairs()
    present = {nm for k in pairs for nm in k.split('|')}
    arms = {'TZ': 'ThompsonZero (additive_mle / avgconc)',
            'CO': 'CORR (corr / corr, rho=0.8)'}
    eps = {a: sorted(int(nm[len(a):-1]) * 1000 for nm in present
                     if nm.startswith(a)) for a in arms}
    names = [f'{a}{e // 1000}k' for a in arms for e in eps[a]]

    wins, games = matrices(names, pairs)
    bt = np.asarray(T.fit_elo(names, wins, games, reg=REG))

    rng = np.random.default_rng(SEED)

    def draw(w, d, l):
        g = int(w + d + l)
        if g == 0:
            return w, d, l
        return rng.multinomial(g, [w / g, d / g, l / g])

    boot = np.empty((N_BOOT, len(names)))
    for b in range(N_BOOT):
        bw, bg = matrices(names, pairs, draw=draw)
        boot[b] = T.fit_elo(names, bw, bg, reg=REG)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    idx = {nm: i for i, nm in enumerate(names)}
    print(f'joint BT fit, {len(pairs)} pairs, {N_BOOT} bootstraps')
    print(f'  {"ep":>7}{"TZ":>8}{"95% CI":>16}{"CORR":>8}{"95% CI":>16}{"gap":>8}')
    for e in sorted(set(eps['TZ']) | set(eps['CO'])):
        row = [f'  {e:>7}']
        vals = {}
        for a in ('TZ', 'CO'):
            nm = f'{a}{e // 1000}k'
            if nm in idx:
                i = idx[nm]
                vals[a] = bt[i]
                row.append(f'{bt[i]:>8.0f}  [{lo[i]:.0f},{hi[i]:.0f}]'.ljust(16))
            else:
                row.append(' ' * 24)
        row.append(f'{vals["CO"] - vals["TZ"]:>+8.0f}'
                   if len(vals) == 2 else ' ' * 8)
        print(''.join(row))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = '#1b1d21', '#6b7076', '#dcdee1'
    COLOR = {'TZ': '#2f6f9f', 'CO': '#c2622d'}
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=160)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for a, label in arms.items():
        x = np.array(eps[a])
        i = [idx[f'{a}{e // 1000}k'] for e in x]
        ax.fill_between(x, lo[i], hi[i], color=COLOR[a], alpha=0.13, lw=0)
        ax.plot(x, bt[i], '-o', color=COLOR[a], ms=5.5, lw=2.0, label=label,
                markerfacecolor='white', markeredgewidth=1.8, zorder=3)

    # the crossover: last episode where TZ leads, first where CORR leads
    common = sorted(set(eps['TZ']) & set(eps['CO']))
    gaps = [bt[idx[f'CO{e // 1000}k']] - bt[idx[f'TZ{e // 1000}k']]
            for e in common]
    cross = next((e for e, g in zip(common, gaps) if g > 0), None)
    if cross is not None:
        ax.axvline(cross, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate('crossover', xy=(cross, ax.get_ylim()[0]),
                    xytext=(cross + 700, 1030), color=MUTED, fontsize=8.5)

    ax.axvspan(30000, max(eps['TZ']), color='#000000', alpha=0.035, lw=0,
               zorder=0)
    ax.annotate('CORR paused here\n(TZ ran to 50k)', xy=(31000, 1760),
                color=MUTED, fontsize=8.5, va='top')

    ax.set_xlabel('training episodes', color=INK, fontsize=10)
    ax.set_ylabel('Bradley-Terry rating (joint fit, mean-anchored)',
                  color=INK, fontsize=10)
    title = ('CORR starts behind and overtakes ThompsonZero'
             + (f' at ep{cross}' if cross is not None else ''))
    ax.set_title(title, color=INK, fontsize=12.5, pad=12, loc='left')
    ax.annotate(f'{len(pairs)} pairs, 24 games each, 400 sims, 12 XOT openings, '
                f'greedy; bands are 95% bootstrap over games',
                xy=(0, 1.013), xycoords='axes fraction', color=MUTED,
                fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(range(0, 50001, 10000))
    ax.set_xticklabels([f'{v // 1000}k' for v in range(0, 50001, 10000)])
    leg = ax.legend(loc='lower right', frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
