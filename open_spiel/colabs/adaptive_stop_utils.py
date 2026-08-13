"""Adaptive-stopping search experiment: instead of always running a fixed
simulation budget, re-check the root's OWN Dirichlet belief (p(v) --
`observed_alpha(root.nacc, search_agg)`, NOT any per-action q_a(v)) every
`check_every` simulations against its value at the previous checkpoint, and
stop as soon as the forward KL between the two drops below a threshold. Tested
on the MM arm (search=additive_mle, target=additive_mle), since it's already
the most volatile of the four ThompsonZero cells to a hand-picked stopping
rule -- if this holds up there it should hold up on the additive cells too.

`connect4_dirichlet_utils.C4MCTSBot.mcts_search_checkpointed` (the primitive
this module drives) lives in that file, next to the tree it searches. This
module holds everything specific to the EXPERIMENT: sampling calibration
positions, running the calibration pass, and (once thresholds are picked)
comparing trained adaptive checkpoints back to the fixed-budget `bench_MM`
ones this was calibrated against.
"""

import os

import numpy as np

import connect4_dirichlet_utils as c4
import connect4_benchmark as bench

MM_SIG = (32, 3, 8)                 # channels, num_blocks, head_ch (bench_MM)


def sample_positions(net, game, n_positions, sims=100, opening_plies=2,
                     temp_threshold=30, late_temp=8.0, sample_every=4,
                     seed=0):
    """Realistic self-play positions from `net`'s own play: several games,
    sampling every `sample_every`-th ply so openings, midgame and endgame are
    all represented rather than just one phase."""
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < n_positions:
        state = game.new_initial_state()
        for _ in range(opening_plies):
            if state.is_terminal():
                break
            leg = state.legal_actions()
            state.apply_action(int(leg[rng.integers(len(leg))]))
        bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                           random_state=rng)
        ply = 0
        while not state.is_terminal() and len(positions) < n_positions:
            root = bot.mcts_search(state)
            if ply % sample_every == 0:
                positions.append(state.clone())
            thompson = ply < temp_threshold
            a = c4.root_pick(root, rng, thompson=thompson,
                             temp=(1.0 if thompson else late_temp))
            state.apply_action(a)
            ply += 1
    return positions


def calibrate(bench_dir, labels=('1000', '2000'), n_positions=50,
             full_sims=300, check_every=100, sig=MM_SIG,
             game_name='othello', seed=0, log=print):
    """For each of `labels` (bench_{label}.pt checkpoints in bench_dir),
    sample `n_positions` positions from that checkpoint's own play and run a
    full checkpointed search (no early stop) on each, recording the root-KL
    trace. Returns {label: [[(sims, kl), ...], ...]} -- one trace per
    position."""
    game = c4.load_game(game_name)
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    out = {}
    for label in labels:
        net = c4.load_benchmark_net(bench_dir, label, sig)
        positions = sample_positions(net, game, n_positions, seed=seed)
        rng = np.random.default_rng(seed + 1)
        traces = []
        for state in positions:
            bot = c4.C4MCTSBot(game, net, 'cpu', full_sims, batch_size=16,
                               random_state=rng)
            trace = []
            bot.mcts_search_checkpointed(state, check_every=check_every,
                                         kl_eps=None, trace=trace)
            traces.append(trace)
        out[label] = traces
        log(f'  {label}: {len(traces)} positions traced, '
            f'{len(traces[0])} checkpoints each')
    return out


def summarize(traces_by_label, log=print):
    """Percentiles of the STEADY-STATE checkpoint KLs -- every checkpoint
    after the first, which is always a large jump off the network's raw prior
    (nothing has been searched yet) and says nothing about convergence. These
    steady-state values are what a stopping threshold should be sized
    against."""
    all_steady = []
    for label, traces in traces_by_label.items():
        steady = [kl for trace in traces for _, kl in trace[1:]]
        all_steady += steady
        arr = np.array(steady)
        log(f'{label}: n={len(arr)} '
            f'p10={np.percentile(arr, 10):.4g} p25={np.percentile(arr, 25):.4g} '
            f'p50={np.percentile(arr, 50):.4g} p75={np.percentile(arr, 75):.4g} '
            f'p90={np.percentile(arr, 90):.4g}')
    arr = np.array(all_steady)
    log(f'combined: p10={np.percentile(arr, 10):.4g} '
        f'p25={np.percentile(arr, 25):.4g} p50={np.percentile(arr, 50):.4g} '
        f'p75={np.percentile(arr, 75):.4g} p90={np.percentile(arr, 90):.4g}')
    return arr


def head_to_head(net_a, net_b, game_name='othello', sims=64,
                 games_per_pair=60, opening_plies=2, seed=12345,
                 search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 log=print):
    """net_a vs net_b, colours alternating, under a FIXED search rule applied
    to BOTH sides (default: MM's own, additive_mle/additive_mle).

    `connect4_benchmark.round_robin` is the wrong tool for this: it forces
    every arm to plain 'additive' search so the tournament compares networks
    independently of how each was trained to be searched. Here the question
    IS whether a network trained under adaptive stopping still performs under
    MM's own search rule, so neutralising it would test something else.
    Reuses `connect4_benchmark.play_game`/`wilson` rather than reimplementing
    game-playing -- only the search-rule override is skipped.

    Returns (net_a's score fraction, wilson_lo, wilson_hi)."""
    game = c4.load_game(game_name)
    bench.GAME_REF[0] = game
    c4.set_game(game)
    c4.set_search(search_agg=search_agg, target_agg=target_agg,
                 selection='dirichlet', virtual_loss=1.0)
    rng = np.random.default_rng(seed)
    pts_a = 0.0
    for g in range(games_per_pair):
        a_first = (g % 2 == 0)
        pa, pb = (('thompson', net_a), ('thompson', net_b)) if a_first else \
                 (('thompson', net_b), ('thompson', net_a))
        sc = bench.play_game(pa, pb, sims, rng, opening_plies)
        pts_a += sc if a_first else (1.0 - sc)
    frac = pts_a / games_per_pair
    lo, hi = bench.wilson(frac, games_per_pair)
    log(f'  score: {frac:.1%}  [{lo:.1%}, {hi:.1%}]  (n={games_per_pair})')
    return frac, lo, hi


def compare_to_bench_mm(new_dir, new_label, bench_dir, bench_label,
                        sig=MM_SIG, game_name='othello', **kw):
    """head_to_head between a checkpoint from a new (adaptive) run and the
    reference bench_MM checkpoint at a matching episode count."""
    c4.set_game(c4.load_game(game_name))    # load_benchmark_net needs _OBS_SHAPE set
    net_a = c4.load_benchmark_net(new_dir, new_label, sig)
    net_b = c4.load_benchmark_net(bench_dir, bench_label, sig)
    return head_to_head(net_a, net_b, game_name=game_name, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  Arm "recipes" -- each arm IS a (network, search policy) pair, not just a
#  network. A recipe is a tuple describing how that arm actually searches:
#    ('fixed', sims)                          -- always `sims` simulations
#    ('mix', fast_sims, full_sims, fast_prob) -- fast_sims w.p. fast_prob,
#                                                 else full_sims (bench_MM's
#                                                 own training recipe, and
#                                                 mix300_500's)
#    ('adaptive', check_every, kl_eps, cap)   -- checkpointed KL-based
#                                                 early stopping
#    ('random', None)                         -- uniform random legal move
# ══════════════════════════════════════════════════════════════════════════════

def search_with_rule(game, net, rule_cfg, state, rng, root=None):
    """Search `state` under a budget rule expressed as a worker-cfg dict --
    the SAME dict shape self-play uses, run through the SAME
    `_adaptive_mode` / `_alloc_sims` / `_adaptive_converged` functions.

    Sharing those instead of reimplementing the rules at eval time is the
    whole point: a tournament arm must search exactly the way its training
    did, and two parallel implementations would silently drift.

    Returns (searched root, sims used)."""
    mode = c4._adaptive_mode(rule_cfg)
    cap = rule_cfg.get('alloc_max') or 300
    step = rule_cfg.get('adaptive_check_every', 100)
    if root is None:
        root = c4.expand_node(net, 'cpu', state)
    s = {'root': root, 'n': 0, 'sims': cap, 'next_check': cap,
         'prev_alpha': None}
    c4._prime_adaptive(rule_cfg, s)          # may set s['sims'] (upfront modes)
    bot = c4.C4MCTSBot(game, net, 'cpu', s['sims'], batch_size=16,
                       random_state=rng)
    if mode in c4._ALLOC_MODES:
        n = bot._run_until(root, state, 0, s['sims'])
        return root, n
    s['next_check'] = min(step, s['sims'])
    n = 0
    while n < s['sims']:
        if c4._node_solved_outcome(root) is not None:
            break
        n = bot._run_until(root, state, n, s['next_check'])
        s['n'] = n
        prev_check = s['next_check']
        if c4._adaptive_converged(rule_cfg, s):
            break
        if s['next_check'] <= prev_check:     # checkpoint hit the cap
            break
    return root, n


def recipe_move(game, net, recipe, state, rng):
    """One move under `recipe`'s own search policy (thompson=False: posterior
    mean argmax over searched edges, the normal eval-time pick)."""
    kind = recipe[0]
    if kind == 'random':
        leg = state.legal_actions()
        return int(leg[rng.integers(len(leg))])
    if kind == 'rule':
        root, _n = search_with_rule(game, net, recipe[1], state, rng)
        return c4.root_pick(root, rng, thompson=False)
    if kind == 'fixed':
        sims = recipe[1]
    elif kind == 'mix':
        _, fast, full, fast_prob = recipe
        sims = fast if rng.random() < fast_prob else full
    elif kind == 'adaptive':
        _, check_every, eps, cap = recipe
        bot = c4.C4MCTSBot(game, net, 'cpu', cap, batch_size=16,
                           random_state=rng)
        root, _n = bot.mcts_search_checkpointed(state, check_every=check_every,
                                                kl_eps=eps)
        return c4.root_pick(root, rng, thompson=False)
    else:
        raise ValueError(f'unknown recipe kind {kind!r}')
    bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16, random_state=rng)
    root = bot.mcts_search(state)
    return c4.root_pick(root, rng, thompson=False)


def recipe_sims_used(game, net, recipe, positions, seed=0):
    """Sims actually spent per move in `positions`, under `recipe`. Uses
    root.nacc[0]-1 (nacc is seeded with 1 observation at expansion, so this
    is exactly the simulation count) since mcts_search doesn't return it."""
    rng = np.random.default_rng(seed)
    used = []
    for st in positions:
        if recipe[0] == 'random':
            used.append(0)
            continue
        if recipe[0] == 'rule':
            _root, n = search_with_rule(game, net, recipe[1], st, rng)
            used.append(n)
            continue
        if recipe[0] == 'adaptive':
            _, check_every, eps, cap = recipe
            bot = c4.C4MCTSBot(game, net, 'cpu', cap, batch_size=16,
                               random_state=rng)
            _root, n = bot.mcts_search_checkpointed(st, check_every=check_every,
                                                    kl_eps=eps)
            used.append(n)
            continue
        if recipe[0] == 'fixed':
            sims = recipe[1]
        else:  # 'mix'
            _, fast, full, fast_prob = recipe
            sims = fast if rng.random() < fast_prob else full
        bot = c4.C4MCTSBot(game, net, 'cpu', sims, batch_size=16,
                           random_state=rng)
        root = bot.mcts_search(st)
        used.append(int(root.nacc[0]) - 1)
    return np.array(used)


def oracle_sims_needed(game, net, positions, check_every=50, eps=0.005,
                       patience=3, cap=1000, seed=11):
    """"How much search did this position ACTUALLY need", measured carefully
    offline: run in small blocks and only declare convergence after `patience`
    CONSECUTIVE quiet blocks, out to a generous cap.

    This is the label the cheap upfront rules are trying to predict. Patience
    is what makes it a trustworthy target rather than just a slower copy of
    the reactive rule: a single quiet block is exactly the signal that misses
    plateau-then-jump positions, so a label built from one would bake the
    failure mode into the thing meant to fix it."""
    rng = np.random.default_rng(seed)
    out = []
    for st in positions:
        bot = c4.C4MCTSBot(game, net, 'cpu', cap, batch_size=16,
                           random_state=rng)
        trace = []
        bot.mcts_search_checkpointed(st, check_every=check_every, kl_eps=None,
                                     trace=trace)
        quiet = 0
        needed = cap
        for sims, kl in trace:
            quiet = quiet + 1 if kl < eps else 0
            if quiet >= patience:
                needed = sims
                break
        out.append(needed)
    return np.array(out, dtype=float)


def position_features(net, game, positions):
    """(N, 4) design matrix [1, margin, log conf, n_legal] from the raw prior
    -- the inputs an upfront rule is allowed to use."""
    rows = []
    for st in positions:
        root = c4.expand_node(net, 'cpu', st)
        margin, conf, n_legal = c4._root_features(root)
        rows.append([1.0, margin, np.log(max(conf, 1e-6)), float(n_legal)])
    return np.asarray(rows)


def fit_sims_predictor(X, y, log=print):
    """Least-squares fit of log(sims needed) on the prior features. Returns
    (coef, r2). r2 IS the headline number: it says whether "how much search
    does this position need" is predictable from the prior at all, which is
    the premise the whole upfront-allocation family rests on."""
    ly = np.log(np.maximum(y, 1.0))
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    log(f'  predictor fit: r2={r2:.4f}  coef={np.round(coef, 4).tolist()}')
    log(f'    (per-feature corr with log-sims: '
        f'margin={np.corrcoef(X[:,1], ly)[0,1]:+.3f} '
        f'logconf={np.corrcoef(X[:,2], ly)[0,1]:+.3f} '
        f'n_legal={np.corrcoef(X[:,3], ly)[0,1]:+.3f})')
    return coef, r2


def calibrate_rule(game, net, positions, rule_cfg, knob, target=150.0,
                   lo=1e-6, hi=1e3, iters=14, seed=0, log=print):
    """Bisect `rule_cfg[knob]` until mean sims/move hits `target`.

    Every rule is monotone increasing in its knob (bigger alloc_scale -> more
    search; bigger decision_margin -> harder to satisfy -> more search), and
    'kl'/'hybrid' are monotone DECREASING in eps, which is handled by
    detecting the direction from the bracket ends rather than special-casing
    each mode."""
    def mean_sims(v):
        cfg = dict(rule_cfg); cfg[knob] = v
        rng = np.random.default_rng(seed)
        used = [search_with_rule(game, net, cfg, st, rng)[1] for st in positions]
        return float(np.mean(used))

    f_lo, f_hi = mean_sims(lo), mean_sims(hi)
    increasing = f_hi >= f_lo
    if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
        best = lo if abs(f_lo - target) < abs(f_hi - target) else hi
        log(f'  {knob}: target {target} outside reachable [{min(f_lo,f_hi):.0f},'
            f'{max(f_lo,f_hi):.0f}] -- clamping to {best:g}')
        return best, (f_lo if best == lo else f_hi)
    for _ in range(iters):
        mid = (lo + hi) / 2 if hi < 100 * lo else (lo * hi) ** 0.5
        f_mid = mean_sims(mid)
        if (f_mid < target) == increasing:
            lo = mid
        else:
            hi = mid
    v = (lo + hi) / 2
    f = mean_sims(v)
    log(f'  {knob}={v:.6g} -> mean sims {f:.1f} (target {target})')
    return v, f


def play_game_arms(game, net_a, recipe_a, net_b, recipe_b, rng, opening_plies=2):
    """One game, `net_a`/`recipe_a` as player 0. Each side searches under its
    OWN recipe -- this compares full deployed policies (network + search
    regime), not just networks under a shared neutral search rule the way
    connect4_benchmark.round_robin does. Returns player 0's result in
    {1, 0.5, 0}."""
    st = game.new_initial_state()
    for _ in range(opening_plies):
        if st.is_terminal():
            break
        leg = st.legal_actions()
        st.apply_action(int(leg[rng.integers(len(leg))]))
    while not st.is_terminal():
        if st.current_player() == 0:
            a = recipe_move(game, net_a, recipe_a, st, rng)
        else:
            a = recipe_move(game, net_b, recipe_b, st, rng)
        st.apply_action(a)
    r = st.returns()[0]
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def arms_round_robin(entrants, games_per_pair, game_name='othello', seed=54321,
                     opening_plies=2, cache=None, log=print):
    """Round robin among `entrants` = {label: (net, recipe)}, each playing
    under its OWN search recipe. Reuses connect4_benchmark's Wilson/
    Bradley-Terry math and JSON pair-cache format (so it tops up rather than
    restarting if interrupted) -- only the game-playing loop is arm-aware
    instead of assuming one shared `sims` value for every player, which is
    what round_robin() itself assumes and why it can't be reused directly
    here.

    Returns (names, W, elo, n_games_played_this_call) -- the last element is
    how many NEW games this call actually played (not from cache), so callers
    doing their own timing (e.g. cost-per-game calibration) get an accurate
    denominator even if a previous, interrupted call already seeded part of
    the cache."""
    import itertools
    import time as _time

    game = c4.load_game(game_name)
    c4.set_game(game)
    c4.set_search(search_agg=c4.AGG_ADDITIVE_MLE, target_agg=c4.AGG_ADDITIVE_MLE,
                 selection='dirichlet', virtual_loss=1.0)
    names = list(entrants)
    idx = {k: i for i, k in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    rng = np.random.default_rng(seed)
    store = bench._load_rr_cache(cache)
    pairs = list(itertools.combinations(names, 2))
    todo = []
    for a, b in pairs:
        rec = store.get(bench._cache_key('native', a, b))
        played = int(rec[2]) if rec else 0
        if played < games_per_pair:
            todo.append((a, b, played))
    reused = len(pairs) - len(todo)
    if reused:
        log(f'  reusing {reused}/{len(pairs)} pairs from {cache}')
    n_new_games = sum(games_per_pair - played for _a, _b, played in todo)
    t0 = _time.time()
    for k, (a, b, played) in enumerate(todo, 1):
        key = bench._cache_key('native', a, b)
        rec = store.get(key)
        x, y = sorted((a, b))
        sx = float(rec[0]) if rec else 0.0
        net_x, recipe_x = entrants[x]
        net_y, recipe_y = entrants[y]
        for g in range(played, games_per_pair):
            first_is_x = (g % 2 == 0)
            if first_is_x:
                sc = play_game_arms(game, net_x, recipe_x, net_y, recipe_y,
                                    rng, opening_plies)
            else:
                sc = play_game_arms(game, net_y, recipe_y, net_x, recipe_x,
                                    rng, opening_plies)
            sx += sc if first_is_x else (1.0 - sc)
        store[key] = [sx, games_per_pair - sx, games_per_pair]
        bench._save_rr_cache(cache, store)   # persist after EVERY pair -- long
                                              # tournament, cheap to save often
        if k % 3 == 0 or k == len(todo):
            log(f'  … {k}/{len(todo)} new pairs  ({_time.time() - t0:.0f}s)')
    for a, b in pairs:
        rec = store.get(bench._cache_key('native', a, b))
        if not rec:
            continue
        x, y = sorted((a, b))
        W[idx[x], idx[y]] += float(rec[0])
        W[idx[y], idx[x]] += float(rec[1])
    anchor = idx.get('random')
    return names, W, bench.bradley_terry(W, anchor_idx=anchor), n_new_games
