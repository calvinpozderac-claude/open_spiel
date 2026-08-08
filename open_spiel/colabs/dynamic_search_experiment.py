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
ARMS = {
    # The control: a fixed budget, which is what every trained arm used.
    'fixed':      dict(dynamic_search=False),

    # The real rule at its default thresholds, then one knob at a time.
    'full':       dict(ds_rule='full'),
    'full_p99':   dict(ds_rule='full', ds_p_stop=0.99),
    'full_p90':   dict(ds_rule='full', ds_p_stop=0.90),
    'full_d02':   dict(ds_rule='full', ds_delta=0.02),
    'full_d15':   dict(ds_rule='full', ds_delta=0.15),
    'full_wide':  dict(ds_rule='full', ds_ceil_mult=8.0, ds_floor_frac=0.125),

    # Is each gate earning its place?
    'decision':   dict(ds_rule='decision'),
    'drift':      dict(ds_rule='drift'),

    # The traps, run rather than argued.
    'naive_post': dict(ds_rule='naive_posterior'),
    'naive_gap':  dict(ds_rule='naive_gap'),
}

ORDER = list(ARMS)


def shared(episodes):
    """One small, cheap, identical setting for every arm."""
    s = B.default_shared(
        root=ROOT,
        num_episodes=episodes,
        channels=32, num_blocks=3, head_ch=8,
        use_workers=False,          # in-process: the nets are too small for the
        n_parallel_games=8,         # worker handshake to pay for itself
        wave_per_game=4,
        fast_sims=50, full_sims=150, fast_prob=0.75,
        batch_size=128, train_steps_per_ep=2,
        quick_eval_every=10 ** 9,   # the tournament below is the measurement
        deep_eval_every=10 ** 9,
        solved_every=0,
        device='cpu',
        seed=7,
    )
    return s


def arm_config(name, episodes):
    s = shared(episodes)
    cfg = B.thompson_config('MM', s)          # every arm is MM apart from the rule
    over = dict(ARMS[name])
    over.setdefault('dynamic_search', True)
    over['checkpoint_dir'] = os.path.join(ROOT, f'arm_{name}')
    import dataclasses
    return dataclasses.replace(cfg, **over)


def train_one(name, episodes, log=print):
    """One short run.  Returns the row of measurements for this arm."""
    import dataclasses
    cfg = arm_config(name, episodes)
    c4.set_game(c4.load_game(cfg.game_name))
    t0 = time.perf_counter()
    import contextlib
    with open(os.devnull, 'w') as devnull, \
            contextlib.redirect_stderr(devnull):
        hist = c4.run_training(cfg, log=lambda *a, **k: None)
    wall = time.perf_counter() - t0
    row = {'arm': name, 'wall_s': wall, 'episodes': episodes,
           'rule': ARMS[name].get('ds_rule', '-'),
           'sims_mean': float('nan'), 'stop': {}}
    stats = (hist or {}).get('ds') or {}
    row.update({k: v for k, v in stats.items() if k != 'stop'})
    row['stop'] = dict(stats.get('stop') or {})
    if not stats:
        # The fixed arm keeps no pool, so report what it was always going to
        # spend rather than leaving the column blank.
        sh = shared(episodes)
        row['sims_mean'] = (sh['fast_prob'] * sh['fast_sims']
                            + (1 - sh['fast_prob']) * sh['full_sims'])
        row['nominal'] = True
    log(f'  {name:<11} {wall:7.1f}s  sims/move '
        f'{row.get("sims_mean", float("nan")):6.1f}')
    return row


def final_net(name, s):
    """The trained network of one arm, from its own checkpoint."""
    import torch
    d = os.path.join(ROOT, f'arm_{name}')
    blob = torch.load(os.path.join(d, 'latest.pt'), map_location='cpu',
                      weights_only=False)
    net = c4.C4DirichletNet(s['channels'], s['num_blocks'], s['head_ch'])
    net.load_state_dict(blob['model'])
    net.eval()
    return net


def tournament(rows, episodes, games_per_pair, sims, log=print):
    s = shared(episodes)
    players = {r['arm']: ('thompson', final_net(r['arm'], s)) for r in rows}
    c4.set_game(c4.load_game(s['game']))
    B.GAME_REF[0] = c4.load_game(s['game'])
    names, _W, elo = B.round_robin(
        players, sims=sims, games_per_pair=games_per_pair,
        game=s['game'], log=lambda *a, **k: None)
    return dict(zip(names, [float(x) for x in np.asarray(elo).reshape(-1)]))


def report(rows, elo, sims, log=print):
    log('')
    log(f'{"arm":<12}{"rule":<16}{"Elo":>6}{"wall s":>9}{"sims/mv":>9}'
        f'{"vs fixed":>10}{"stop: conv/ceil/solved":>26}')
    base_wall = next((r['wall_s'] for r in rows if r['arm'] == 'fixed'), None)
    for r in sorted(rows, key=lambda r: -elo.get(r['arm'], 0)):
        st = r.get('stop', {}) or {}
        mix = '/'.join(f'{st.get(k, 0.0):.2f}'
                       for k in ('converged', 'ceiling', 'solved'))
        rel = (r['wall_s'] / base_wall) if base_wall else float('nan')
        log(f'{r["arm"]:<12}{r["rule"]:<16}{elo.get(r["arm"], 0):>6.0f}'
            f'{r["wall_s"]:>9.1f}{r.get("sims_mean", float("nan")):>9.1f}'
            f'{rel:>9.2f}x{mix:>26}')
    log('')
    log(f'(tournament: fixed {sims} simulations for every player, so this rates '
        f'the NETWORKS each rule produced, not the rules at play time.)')


def main(argv):
    episodes = int(argv[1]) if len(argv) > 1 else 400
    gpp = int(argv[2]) if len(argv) > 2 else 24
    sims = int(argv[3]) if len(argv) > 3 else 64
    os.makedirs(ROOT, exist_ok=True)
    print(f'training {len(ORDER)} arms x {episodes} episodes (Connect 4)')
    # Warm-up: the first run in a process pays for imports, the pyspiel game
    # load and torch's first kernels.  Timing it as if it were the arm's own
    # cost put 15.3s against 4s for identical work in the pilot.
    print('warm-up...')
    train_one(ORDER[0], 2, log=lambda *a, **k: None)
    rows = []
    for name in ORDER:
        rows.append(train_one(name, episodes))
        with open(os.path.join(ROOT, 'rows.json'), 'w') as f:
            json.dump(rows, f, indent=1)
    print(f'\ntournament: {gpp} games per pair at {sims} simulations')
    elo = tournament(rows, episodes, gpp, sims)
    with open(os.path.join(ROOT, 'elo.json'), 'w') as f:
        json.dump({k: float(v) for k, v in elo.items()}, f, indent=1)
    report(rows, elo, sims)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
