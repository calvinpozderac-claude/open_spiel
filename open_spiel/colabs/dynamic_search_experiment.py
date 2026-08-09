"""A pilot A/B of dynamic-search stop rules.

    python dynamic_search_experiment.py [episodes] [games_per_pair]

Trains one short run per rule, all from the same seed and the same trunk, then
plays the resulting networks against each other at a FIXED simulation count so
the tournament compares networks rather than two different search procedures.
Records, per arm: wall clock, simulations actually spent per move, why searches
ended, and the Elo that came out.

Run on CONNECT 4, not Othello.  The mechanism under test — whether a stop rule
reads something that survives more simulations — is game-agnostic, and Connect 4
is roughly 4x cheaper per episode here, which buys enough episodes for the
differences to mean anything.  A pilot at this size separates large effects and
nothing else; treat a 30-Elo gap as noise.

The two naive rules are included deliberately.  They are what the design
argues against, and an argument that is never run is not evidence.
"""

import json
import os
import sys
import time

import numpy as np

import connect4_benchmark as B
import connect4_dirichlet_utils as c4
import dynamic_search_utils as ds

ROOT = os.environ.get('DS_EXP_ROOT', 'ds_experiment')

# name -> Config overrides.  Everything not named here is identical across arms.
# Four arms, several seeds each.  The pilot ran eleven arms at one seed and
# produced a 126-Elo spread that was entirely noise -- with one run per arm
# there is no way to tell an arm apart from the seed it drew.  Fewer arms and
# repeated seeds buys the power to say something.
ARMS = {
    # The control: a fixed budget, which is what every trained arm used.
    'fixed':      dict(dynamic_search=False),
    # The rule the design argues for.
    'full':       dict(ds_rule='full'),
    # The strict best-arm condition.
    'lucb':       dict(ds_rule='lucb'),
    # The only rule whose gate actually opens.  It reads MOVEMENT of the target
    # rather than the disagreement spread, which the diagnostic showed does not
    # narrow with search at all (median 0.506 after 500 simulations).
    'block':      dict(ds_rule='block', ds_delta=0.05),
}

SEEDS = (7, 11, 23)
# One budget for every position, so `sims/mv` is directly comparable to it and
# the mean is not a blend of a fast and a full setting.
SIMS = int(os.environ.get('DS_EXP_SIMS', 500))
ORDER = list(ARMS)


def runs():
    return [(a, sd) for a in ORDER for sd in SEEDS]


def run_name(arm, seed):
    return f'{arm}#{seed}'


def shared(episodes, sims=None):
    """One small, cheap, identical setting for every arm."""
    s = B.default_shared(
        root=ROOT,
        num_episodes=episodes,
        channels=32, num_blocks=3, head_ch=8,
        use_workers=False,          # in-process: the nets are too small for the
        n_parallel_games=8,         # worker handshake to pay for itself
        wave_per_game=4,
        fast_sims=(sims or SIMS), full_sims=(sims or SIMS), fast_prob=1.0,
        batch_size=128, train_steps_per_ep=2,
        quick_eval_every=10 ** 9,   # the tournament below is the measurement
        deep_eval_every=10 ** 9,
        solved_every=0,
        device='cpu',
        seed=7,
    )
    return s


def arm_config(arm, seed, episodes):
    s = shared(episodes)
    cfg = B.thompson_config('MM', s)          # every arm is MM apart from the rule
    over = dict(ARMS[arm])
    over.setdefault('dynamic_search', True)
    over['seed'] = seed
    over['checkpoint_dir'] = os.path.join(ROOT, f'arm_{arm}_s{seed}')
    import dataclasses
    return dataclasses.replace(cfg, **over)


def train_one(arm, seed, episodes, log=print):
    """One run.  Returns the row of measurements for this arm/seed."""
    import dataclasses
    name = run_name(arm, seed)
    cfg = arm_config(arm, seed, episodes)
    c4.set_game(c4.load_game(cfg.game_name))
    t0 = time.perf_counter()
    import contextlib
    with open(os.devnull, 'w') as devnull, \
            contextlib.redirect_stderr(devnull):
        hist = c4.run_training(cfg, log=lambda *a, **k: None)
    wall = time.perf_counter() - t0
    row = {'arm': arm, 'seed': seed, 'run': name, 'wall_s': wall,
           'episodes': episodes, 'rule': ARMS[arm].get('ds_rule', '-'),
           'sims_mean': float('nan'), 'stop': {}}
    stats = (hist or {}).get('ds') or {}
    row.update({k: v for k, v in stats.items() if k != 'stop'})
    row['stop'] = dict(stats.get('stop') or {})
    if not stats:
        # The fixed arm keeps no pool, so report what it was always going to
        # spend rather than leaving the column blank.
        row['sims_mean'] = float(SIMS)
        row['nominal'] = True
    log(f'  {name:<14} {wall:7.1f}s  sims/move '
        f'{row.get("sims_mean", float("nan")):6.1f}')
    return row


def final_net(arm, seed, s):
    """The trained network of one run, from its own checkpoint."""
    import torch
    d = os.path.join(ROOT, f'arm_{arm}_s{seed}')
    blob = torch.load(os.path.join(d, 'latest.pt'), map_location='cpu',
                      weights_only=False)
    net = c4.C4DirichletNet(s['channels'], s['num_blocks'], s['head_ch'])
    net.load_state_dict(blob['model'])
    net.eval()
    return net


def tournament(rows, episodes, games_per_pair, sims, log=print):
    s = shared(episodes)
    players = {r['run']: ('thompson', final_net(r['arm'], r['seed'], s))
               for r in rows}
    c4.set_game(c4.load_game(s['game']))
    B.GAME_REF[0] = c4.load_game(s['game'])
    names, _W, elo = B.round_robin(
        players, sims=sims, games_per_pair=games_per_pair,
        game=s['game'], log=lambda *a, **k: None)
    return dict(zip(names, [float(x) for x in np.asarray(elo).reshape(-1)]))


def report(rows, elo, sims, log=print):
    """Per run, then pooled by arm with the seed spread.

    The pooled line is the one to read.  A single run's Elo carries the seed it
    drew as much as the rule it used, which is what made the eleven-arm pilot
    unreadable."""
    log('')
    log(f'{"run":<15}{"rule":<16}{"Elo":>6}{"wall s":>9}{"sims/mv":>9}'
        f'{"nominal":>9}{"stop conv/ceil":>16}')
    for r in sorted(rows, key=lambda r: -elo.get(r['run'], 0)):
        st = r.get('stop', {}) or {}
        mix = '/'.join(f'{st.get(k, 0.0):.2f}' for k in ('converged', 'ceiling'))
        log(f'{r["run"]:<15}{r["rule"]:<16}{elo.get(r["run"], 0):>6.0f}'
            f'{r["wall_s"]:>9.1f}{r.get("sims_mean", float("nan")):>9.1f}'
            f'{r.get("sims_base", float("nan")):>9.1f}{mix:>16}')

    log('')
    log(f'{"arm":<12}{"rule":<16}{"Elo (mean+-sd)":>18}{"wall s":>9}'
        f'{"sims/mv":>9}{"vs fixed time":>15}{"vs fixed sims":>15}')
    agg = {}
    for a in ORDER:
        rs = [r for r in rows if r['arm'] == a]
        if not rs:
            continue
        e = np.array([elo.get(r['run'], 0.0) for r in rs])
        agg[a] = dict(
            elo=float(e.mean()), elo_sd=float(e.std(ddof=1)) if len(e) > 1 else 0.0,
            wall=float(np.mean([r['wall_s'] for r in rs])),
            sims=float(np.nanmean([r.get('sims_mean', np.nan) for r in rs])),
            n=len(rs), rule=rs[0]['rule'])
    bw = agg.get('fixed', {}).get('wall')
    bs = agg.get('fixed', {}).get('sims')
    for a, v in sorted(agg.items(), key=lambda kv: -kv[1]['elo']):
        log(f'{a:<12}{v["rule"]:<16}'
            f'{v["elo"]:>11.0f} +-{v["elo_sd"]:>4.0f}'
            f'{v["wall"]:>9.1f}{v["sims"]:>9.1f}'
            f'{(v["wall"] / bw if bw else float("nan")):>14.2f}x'
            f'{(v["sims"] / bs if bs else float("nan")):>14.2f}x')
    log('')
    log(f'(tournament: fixed {sims} simulations for every player, so this rates '
        f'the NETWORKS each rule produced, not the rules at play time.)')
    log(f'(+-sd is over {len(SEEDS)} seeds; a gap smaller than the sd is not a '
        f'result.)')
    return agg


def main(argv):
    episodes = int(argv[1]) if len(argv) > 1 else 400
    gpp = int(argv[2]) if len(argv) > 2 else 24
    sims = int(argv[3]) if len(argv) > 3 else 64
    os.makedirs(ROOT, exist_ok=True)
    todo = runs()
    print(f'training {len(todo)} runs ({len(ORDER)} arms x {len(SEEDS)} seeds) '
          f'x {episodes} episodes (Connect 4)')
    # Warm-up: the first run in a process pays for imports, the pyspiel game
    # load and torch's first kernels.  Timing it as if it were the arm's own
    # cost put 15.3s against 4s for identical work in the pilot.
    print('warm-up...')
    train_one(ORDER[0], SEEDS[0], 2, log=lambda *a, **k: None)
    rows = []
    for arm, sd in todo:
        rows.append(train_one(arm, sd, episodes))
        with open(os.path.join(ROOT, 'rows.json'), 'w') as f:
            json.dump(rows, f, indent=1)
    print(f'\ntournament: {gpp} games per pair at {sims} simulations')
    elo = tournament(rows, episodes, gpp, sims)
    with open(os.path.join(ROOT, 'elo.json'), 'w') as f:
        json.dump({k: float(v) for k, v in elo.items()}, f, indent=1)
    agg = report(rows, elo, sims)
    with open(os.path.join(ROOT, 'agg.json'), 'w') as f:
        json.dump(agg, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
