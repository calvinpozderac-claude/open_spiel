"""A held-out position set with strong-Edax values, and every checkpoint scored
against it with and without search.

The match benchmark cannot answer the question this run actually raised. Its
cells carry ~5 points of seed noise each, ~3.2 on the four-rung average, against
an observed checkpoint-to-checkpoint sd of 5.2 -- so "the eval is too noisy and
the run is learning fine" and "the network is genuinely unstable" produce the
same curve. Scoring a fixed labelled position set removes the measurement half
and leaves only the network.

BUILD.  Positions come from Edax-vs-Edax games at DELIBERATELY MISMATCHED
levels (2v8, 4v12, 6v10, 8v16, 10v14, 12v18) over randomized openings. Equal
strong players produce a narrow band of balanced positions; mismatched ones
produce won, lost and unbalanced positions too, which is the range a value head
actually has to score. Note the three wEdax binaries are the same engine built
for different instruction sets, so they play identically -- LEVEL is the only
axis that varies play, which is why the matchups vary level rather than exe.

Every position is labelled by playing it out with the strongest Edax, giving
the outcome in the mover's frame. Two tiers, kept separate because they are not
the same claim:

  EXACT    <= EXACT_EMPTIES empties, played out at SOLVE_LEVEL, which solves
           outright at that depth.  This is the game-theoretic value.
  STRONG   midgame, played out at STRONG_LEVEL.  This is one strong engine's
           outcome, NOT truth, and a net can be right where Edax is wrong.

Endgames flatter a value head, so the EXACT tier is the trustworthy one and
also the easy one. Report both; believe neither alone.

SCORE.  Each checkpoint is evaluated at several search budgets -- raw network
first, then MCTS at increasing simulation counts. That is the comparison
fa76db55 could only make for two nets: it found OPT was a strong evaluator its
own search could not improve, while AZ was a weak evaluator search
substantially rescued. Running it across a whole training trajectory shows
whether MM_long's evaluator improved, or whether its search was carrying it.

RANK.  Checkpoints are ranked at each budget and the Spearman correlation
between episode number and score is reported, alongside the same statistic
computed from the match benchmark. That is the actual question: does this
metric show more consistent improvement than win rates did?

    python edax_eval_set.py --build [n_exact] [n_strong]
    python edax_eval_set.py --score [--sims 0,16,64,256,1000]
    python edax_eval_set.py --rank
"""
import json
import os
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
RESULTS = 'edax_eval_scores.json'
BENCH_JSON = 'mm_long_bench.json'
SNAP_DIR = 'othello_benchmark/bench_MM_long_snaps'
CKPT_DIR = 'othello_benchmark/bench_MM_long'

N_EXACT, N_STRONG = 800, 400
MATCHUPS = [(2, 8), (4, 12), (6, 10), (8, 16), (10, 14), (12, 18)]
SOLVE_LEVEL = 24          # exact at EXACT_EMPTIES
STRONG_LEVEL = 14         # midgame playouts; level 18+ costs ~15s per playout
EXACT_EMPTIES = 20
OPEN_MIN, OPEN_MAX = 4, 10
SNAP_MIN_PLY, SNAP_STRIDE = 8, 5
SIM_LEVELS = [0, 16, 64, 256, 1000]      # 0 = raw network, no search
EVAL_EPS = [0, 5000, 10000, 15000, 20000, 25000,
            30000, 35000, 40000, 45000, 50000]
OUR_TEMP = 64.0
CHUNK = 256


def _init():
    import torch
    torch.set_num_threads(4)
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                  selection='dirichlet', virtual_loss=1.0)
    return c4, game


def _empties(state):
    t = np.asarray(state.observation_tensor(0)).reshape(3, 8, 8)
    return int(t[0].sum())


def _replay(game, actions):
    st = game.new_initial_state()
    for a in actions:
        st.apply_action(int(a))
    return st


def _playout_value(game, state, bot):
    """Outcome in the mover's frame after `bot` plays both sides out."""
    mover = state.current_player()
    st = state.clone()
    while not st.is_terminal():
        st.apply_action(bot.step(st))
    r = st.returns()[mover]
    return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)


def build(n_exact, n_strong, seed=23):
    c4, game = _init()
    import edax_bot as eb
    rng = np.random.default_rng(seed)
    solver = eb.EdaxBot(game, level=SOLVE_LEVEL)
    strong = eb.EdaxBot(game, level=STRONG_LEVEL)
    players = {lv: eb.EdaxBot(game, level=lv)
               for lv in sorted({l for m in MATCHUPS for l in m})}

    exact, mid, t0, games = [], [], time.time(), 0
    while len(exact) < n_exact or len(mid) < n_strong:
        lo, hi = MATCHUPS[int(rng.integers(len(MATCHUPS)))]
        seat_lo = int(rng.integers(2))
        st, actions = game.new_initial_state(), []
        for _ in range(int(rng.integers(OPEN_MIN, OPEN_MAX + 1))):
            if st.is_terminal():
                break
            la = st.legal_actions()
            a = int(la[rng.integers(len(la))])
            st.apply_action(a)
            actions.append(a)

        while not st.is_terminal():
            ply = len(actions)
            if ply >= SNAP_MIN_PLY and (ply - SNAP_MIN_PLY) % SNAP_STRIDE == 0:
                emp = _empties(st)
                tier = 'exact' if emp <= EXACT_EMPTIES else 'strong'
                bucket = exact if tier == 'exact' else mid
                want = n_exact if tier == 'exact' else n_strong
                if len(bucket) < want:
                    bot = solver if tier == 'exact' else strong
                    rec = dict(actions=[int(x) for x in actions],
                               best=int(solver.step(st)),
                               value=_playout_value(game, st, bot),
                               ply=ply, empties=emp, tier=tier,
                               matchup=f'{lo}v{hi}')
                    bucket.append(rec)
                    n = len(exact) + len(mid)
                    if n % 100 == 0:
                        print(f'  {len(exact)}/{n_exact} exact, '
                              f'{len(mid)}/{n_strong} strong '
                              f'({time.time()-t0:.0f}s)', flush=True)
            mover_lv = lo if st.current_player() == seat_lo else hi
            st.apply_action(players[mover_lv].step(st))
            actions.append(int(st.history()[-1]))
        games += 1

    for b in list(players.values()) + [solver, strong]:
        b.close()
    recs = exact + mid
    with open(CACHE, 'w') as fh:
        json.dump(dict(solve_level=SOLVE_LEVEL, strong_level=STRONG_LEVEL,
                       exact_empties=EXACT_EMPTIES, matchups=MATCHUPS,
                       seed=seed, positions=recs), fh)
    v = np.array([r['value'] for r in recs])
    print(f'[build] {len(recs)} positions from {games} games '
          f'({len(exact)} exact, {len(mid)} strong) -> {CACHE}')
    print(f'        label mix: win {np.mean(v>0):.0%} draw {np.mean(v==0):.0%} '
          f'loss {np.mean(v<0):.0%}   ({time.time()-t0:.0f}s)')


def _load():
    with open(CACHE) as fh:
        return json.load(fh)['positions']


SEARCH_DIRS = [SNAP_DIR, CKPT_DIR]


def _ckpt_for(ep):
    """ep 0 and the 5000-multiples live in the training dir; the snaps dir has
    everything.  Prefer snaps, fall back to the training checkpoints."""
    for d in SEARCH_DIRS:
        if os.path.exists(os.path.join(d, f'bench_{ep}.pt')):
            return d
    return None


def _infer_sig(path):
    """(channels, num_blocks, head_ch) read off the checkpoint itself.

    adaptive_stop_utils.MM_SIG is (32, 3, 8), hardcoded for bench_MM's shape.
    Passing it for a 10-block net raises on the unexpected body.3..body.9 keys,
    so read the shape from the weights instead of naming it -- the probe should
    work on any run's checkpoints without being told the architecture."""
    import re
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    blocks = len({int(m.group(1)) for k in sd
                  if (m := re.match(r'body\.(\d+)\.', k))})
    channels = sd['stem.0.weight'].shape[0]
    head_ch = sd['head.0.weight'].shape[0]
    return (int(channels), int(blocks), int(head_ch))


def score_checkpoint(c4, game, ep, recs, sims_list):
    d = _ckpt_for(ep)
    if d is None:
        return None
    path = os.path.join(d, f'bench_{ep}.pt')
    net = c4.load_benchmark_net(d, str(ep), _infer_sig(path))
    states = [_replay(game, r['actions']) for r in recs]
    truth = np.array([r['value'] for r in recs])
    is_exact = np.array([r['tier'] == 'exact' for r in recs])
    out = {}

    for sims in sims_list:
        t0 = time.time()
        pred, top1 = [], []
        if sims == 0:
            for i0 in range(0, len(states), CHUNK):
                chunk = states[i0:i0 + CHUNK]
                v3, _vc, p3, _cf, _o = c4.nn_eval_states(net, 'cpu', chunk)
                for j, st in enumerate(chunk):
                    pred.append(float(v3[j][c4._WIN] - v3[j][c4._LOSS]))
                    legal = [int(x) for x in st.legal_actions()]
                    q = (p3[j][legal][:, c4._WIN]
                         - p3[j][legal][:, c4._LOSS])
                    top1.append(legal[int(np.argmax(q))]
                                == recs[i0 + j]['best'])
        else:
            rng = np.random.default_rng(0)
            for i, st in enumerate(states):
                root = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                                    temp=OUR_TEMP, random_state=rng
                                    ).mcts_search(st)
                a = c4.observed_alpha(root.nacc, c4._SEARCH_AGG)
                pred.append(float(c4.dir_value(a)))
                try:
                    import policy_screen as ps
                    mv = ps.pick(c4, root,
                                 dict(mode='value', k=0.0, sel_temp=OUR_TEMP))
                    top1.append(int(mv) == recs[i]['best'])
                except Exception:
                    pass
        pred = np.array(pred)
        rec = {}
        for tag, m in (('all', np.ones(len(truth), bool)),
                       ('exact', is_exact), ('strong', ~is_exact)):
            if not m.any():
                continue
            # Sign accuracy is scored only where the truth is decisive, and on
            # pred > 0 rather than np.sign(pred): an untrained net predicts
            # ~0 everywhere, and sign(0) matches neither label, which would
            # report a uniform net as 0% correct rather than the ~50% it is.
            dec = m & (truth != 0)
            rec[tag] = dict(
                mse=float(((pred[m] - truth[m]) ** 2).mean()),
                sign=(float(((pred[dec] > 0) == (truth[dec] > 0)).mean())
                      if dec.any() else float('nan')),
                n=int(m.sum()), n_decisive=int(dec.sum()))
        if top1:
            rec['top1'] = float(np.mean(top1))
        rec['secs'] = time.time() - t0
        out[str(sims)] = rec
        lbl = 'raw NN' if sims == 0 else f'{sims} sims'
        t1 = f"  top1 {rec['top1']:5.1%}" if 'top1' in rec else ''
        print(f'  ep {ep:>6} {lbl:>9}  MSE {rec["all"]["mse"]:.3f}  '
              f'sign {rec["all"]["sign"]:5.1%}  '
              f'(exact {rec["exact"]["sign"]:5.1%} / '
              f'strong {rec["strong"]["sign"]:5.1%}){t1}  '
              f'[{rec["secs"]:.0f}s]', flush=True)
    return out


def score(sims_list, limit=None):
    if not os.path.exists(CACHE):
        sys.exit(f'no {CACHE}; run --build first')
    recs = _load()
    if limit and limit < len(recs):
        # A deterministic stride keeps both tiers and the whole ply range
        # represented; taking the first N would be all-exact, since build()
        # writes the exact tier first.
        step = len(recs) / float(limit)
        recs = [recs[int(i * step)] for i in range(limit)]
    c4, game = _init()
    print(f'[score] {len(recs)} positions, budgets {sims_list}')
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            done = json.load(fh)
    for ep in EVAL_EPS:
        r = score_checkpoint(c4, game, ep, recs, sims_list)
        if r is None:
            print(f'  ep {ep:>6}  (no checkpoint)')
            continue
        # MERGE, don't replace: budgets are run in separate passes (the cheap
        # ones finish in minutes, 1000 sims takes hours), and overwriting the
        # entry would discard the earlier pass's results for this checkpoint.
        done.setdefault(str(ep), {}).update(r)
        with open(RESULTS, 'w') as fh:
            json.dump(done, fh, indent=2, sort_keys=True)


def _spearman(x, y):
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, float)))
        return o.astype(float)
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float('nan')


def rank():
    with open(RESULTS) as fh:
        res = json.load(fh)
    eps = sorted(int(e) for e in res)
    print(f'\n=== ranking {len(eps)} checkpoints ===')
    print(f'{"budget":>9}  {"rho(ep,sign)":>13}  {"rho(ep,-MSE)":>13}  '
          f'{"best ep":>8}  {"worst ep":>9}')
    for sims in SIM_LEVELS:
        k = str(sims)
        have = [e for e in eps if k in res[str(e)]]
        if len(have) < 3:
            continue
        sign = [res[str(e)][k]['all']['sign'] for e in have]
        mse = [res[str(e)][k]['all']['mse'] for e in have]
        lbl = 'raw NN' if sims == 0 else f'{sims} sims'
        print(f'{lbl:>9}  {_spearman(have, sign):>13.3f}  '
              f'{_spearman(have, [-m for m in mse]):>13.3f}  '
              f'{have[int(np.argmax(sign))]:>8}  '
              f'{have[int(np.argmin(sign))]:>9}')

    if os.path.exists(BENCH_JSON):
        with open(BENCH_JSON) as fh:
            b = json.load(fh)
        me, mv = [], []
        for e in eps:
            r = b.get(str(e))
            if not r:
                continue
            cells = [v['score'] for k, v in r['rungs'].items()
                     if k in ('L1', 'L2', 'L3', 'L4')]
            if len(cells) == 4:
                me.append(e)
                mv.append(float(np.mean(cells)))
        if len(me) >= 3:
            print(f'\n  match play (L1-L4 mean), same {len(me)} checkpoints: '
                  f'rho = {_spearman(me, mv):+.3f}')
            print('  A metric with higher |rho| orders the trajectory more '
                  'consistently with training progress.')


def main():
    global SEARCH_DIRS, RESULTS
    a = sys.argv[1:]
    # --dir points the probe at another run's checkpoints; results go to their
    # own file so one run's curve never overwrites another's.
    if '--dir' in a:
        d = a[a.index('--dir') + 1]
        SEARCH_DIRS = [d]
        RESULTS = f'edax_eval_scores_{os.path.basename(d.rstrip("/\\"))}.json'
        print(f'[probe] checkpoints from {d} -> {RESULTS}')
    if '--eps' in a:
        globals()['EVAL_EPS'] = [int(x) for x in a[a.index('--eps') + 1].split(',')]
    if '--build' in a:
        i = a.index('--build')
        nums = [int(x) for x in a[i + 1:i + 3] if x.isdigit()]
        build(nums[0] if nums else N_EXACT,
              nums[1] if len(nums) > 1 else N_STRONG)
    elif '--score' in a:
        sims = SIM_LEVELS
        if '--sims' in a:
            sims = [int(x) for x in a[a.index('--sims') + 1].split(',')]
        lim = int(a[a.index('--limit') + 1]) if '--limit' in a else None
        score(sims, lim)
    elif '--rank' in a:
        rank()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
