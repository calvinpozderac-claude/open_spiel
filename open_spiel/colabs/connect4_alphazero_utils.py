"""AlphaZero on Connect 4 — the CONTROL for the ThompsonZero-C4 benchmark.

This is deliberately the ordinary algorithm: a policy head and a SCALAR value
head, PUCT selection, visit-count policy targets, and value regression to the
game outcome.  Everything distributional is gone — no per-action Dirichlets, no
evidence accumulators, no win/draw/loss triples.  That is the point: it isolates
what the Dirichlet machinery buys by holding everything else fixed.

Held fixed against `connect4_dirichlet_utils`, deliberately:
  * the same trunk (channels / blocks / head width) and the same head plumbing,
    so parameter counts are within a few percent;
  * the same observation tensor, built mover-relative by the same code;
  * the same self-play shape — parallel games, batched leaf evaluation, virtual
    loss, subtree reuse, fast/full simulation caps, temperature threshold,
    frozen-benchmark pool opponents;
  * the same MCTS-Solver overlay, so neither side gets free exact endgames;
  * the same optimiser, LR schedule, batch size, steps per episode, buffer size;
  * the same multiprocess worker + central batched inference server design.

What differs is only the method:
  network   policy logits (A,) + tanh value scalar   [vs two Dirichlet heads]
  search    PUCT: Q + c_puct·P·√ΣN/(1+N)             [vs Thompson sampling]
  backup    running mean of scalar values             [vs evidence accumulators]
  targets   visit distribution + game outcome         [vs KL to searched beliefs]
  loss      policy cross-entropy + value MSE          [vs 5 distributional terms]

Shared plumbing (game wiring, observations, progress bar, device selection,
residual blocks, the DirectML-safe optimiser) is imported rather than copied, so
there is one implementation of each and no chance of the two arms drifting apart
on something that is supposed to be identical.
"""

import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np

import connect4_dirichlet_utils as c4

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

_WIN, _DRAW, _LOSS = c4._WIN, c4._DRAW, c4._LOSS
_FLIP_TERM = c4._FLIP_TERM


# ══════════════════════════════════════════════════════════════════════════════
#  PUCT tree (pure numpy, so it imports and unit-tests without torch)
# ══════════════════════════════════════════════════════════════════════════════
_C_PUCT = 1.5
_VIRTUAL_LOSS = 1.0


def set_search(c_puct=None, virtual_loss=None):
    global _C_PUCT, _VIRTUAL_LOSS
    if c_puct is not None:
        _C_PUCT = float(c_puct)
    if virtual_loss is not None:
        _VIRTUAL_LOSS = float(virtual_loss)


class _AZNode:
    """One expanded state.  Per legal action a:
        P[a]   prior probability from the policy head
        N[a]   visit count
        W[a]   summed value, from THIS node's mover's perspective
        term[a] proven outcome (_WIN/_DRAW/_LOSS) or −1
    """
    __slots__ = ('player', 'legal', 'P', 'N', 'W', 'term', 'vloss', 'children',
                 'obs', 'value')

    def __init__(self, player, legal, priors, value, obs=None):
        self.player = player
        self.legal = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        p = np.asarray(priors, dtype=np.float64).reshape(k)
        s = p.sum()
        self.P = p / s if s > 0 else np.full(k, 1.0 / k)
        self.N = np.zeros(k)
        self.W = np.zeros(k)
        self.term = np.full(k, -1, dtype=np.int8)
        self.vloss = np.zeros(k, dtype=np.int32)
        self.children = [None] * k
        self.obs = obs
        self.value = float(value)          # leaf evaluation, mover's view

    def q(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.N > 0, self.W / np.maximum(self.N, 1e-12), 0.0)

    def scores(self):
        """PUCT: Q + c·P·√(ΣN)/(1+N), minus virtual loss on in-flight edges."""
        tot = self.N.sum()
        u = _C_PUCT * self.P * math.sqrt(max(tot, 1e-8)) / (1.0 + self.N)
        s = self.q() + u
        # A proven edge is worth exactly its proof, whatever the running mean is.
        pr = self.term >= 0
        if pr.any():
            s = np.where(pr, np.where(self.term == _WIN, 1e6,
                                      np.where(self.term == _DRAW, 0.0, -1e6)), s)
        if self.vloss.any():
            s = s - _VIRTUAL_LOSS * self.vloss
        return s


def add_root_noise(node, rng, frac=0.25, alpha=1.0):
    """AlphaZero's root exploration noise: mix Dirichlet noise into the priors."""
    if frac <= 0 or len(node.legal) < 2:
        return
    noise = rng.dirichlet(np.full(len(node.legal), alpha))
    node.P = (1.0 - frac) * node.P + frac * noise


def _set_term(node, idx, outcome):
    node.term[idx] = outcome


def _select_leaf(root, root_state):
    """Descend by PUCT to an unexpanded or terminal edge, applying virtual loss.
    Returns (path, leaf_state_or_None, value_or_None, edge_or_None); `value` is
    the terminal value from the DEEPEST node's mover's perspective."""
    node, state, path = root, root_state.clone(), []
    while True:
        idx = int(node.scores().argmax())
        node.vloss[idx] += 1
        path.append((node, idx))
        if node.term[idx] >= 0:
            t = node.term[idx]
            return path, None, (1.0 if t == _WIN else
                                (-1.0 if t == _LOSS else 0.0)), None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            r = state.returns()[node.player]
            _set_term(node, idx, _WIN if r > 0 else (_LOSS if r < 0 else _DRAW))
            return path, None, float(r), None
        child = node.children[idx]
        if child is None:
            return path, state, None, (node, idx)
        node = child


def _backup(path, value):
    """`value` is from the perspective of the mover at the DEEPEST node on the
    path; flip once per ply on the way up."""
    v = value
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        node.N[idx] += 1.0
        node.W[idx] += v
        v = -v


def _node_solved_outcome(node):
    t = node.term
    if (t == _WIN).any():
        return _WIN
    if (t >= 0).all():
        return _DRAW if (t == _DRAW).any() else _LOSS
    return None


def _propagate_solved(path, aux=None):
    """Prove parent edges bottom-up, and emit an EXACT solver-labelled training
    sample per newly solved node.

    The ThompsonZero arms have always done this, and for a long time this
    function did not — which quietly gave them roughly three times the training
    signal per game (their buffers measured ~70% solver-labelled) while the
    control trained on self-play moves alone.  The solver is part of both
    engines' search, so it has to be part of both engines' data."""
    for k in range(len(path) - 1, 0, -1):
        node = path[k][0]
        out = _node_solved_outcome(node)
        if out is None:
            break
        parent, pidx = path[k - 1]
        if parent.term[pidx] >= 0:
            break
        _set_term(parent, pidx, int(_FLIP_TERM[out]))
        if aux is not None and node.obs is not None:
            aux.append({'obs': node.obs, 'legal': node.legal.copy(),
                        'pi': visit_policy(node, 1.0).astype(np.float32),
                        'z': np.float32(1.0 if out == _WIN else
                                        (-1.0 if out == _LOSS else 0.0)),
                        'player': int(node.player), 'solved': True})


def _backup_terminal(path, value, aux=None):
    _backup(path, value)
    _propagate_solved(path, aux)


def _descend(root, action):
    if root is None:
        return None
    hit = np.nonzero(root.legal == action)[0]
    return root.children[int(hit[0])] if len(hit) else None


def visit_policy(root, temp=1.0):
    """Normalised visit counts — AlphaZero's policy target and move sampler.
    A solved node reports its proven-best moves instead, so the target never
    teaches a losing move the search merely happened to visit."""
    solved = _node_solved_outcome(root)
    if solved == _WIN:
        pi = (root.term == _WIN).astype(np.float64)
        return pi / pi.sum()
    n = root.N.astype(np.float64)
    if solved is not None:                       # proven draw or loss
        keep = root.term == (_DRAW if solved == _DRAW else _LOSS)
        if keep.any():
            n = n * keep
    if n.sum() <= 0:
        return np.full(len(root.legal), 1.0 / len(root.legal))
    if temp <= 0:
        pi = (n == n.max()).astype(np.float64)
        return pi / pi.sum()
    n = n ** (1.0 / temp)
    return n / n.sum()


def root_pick(root, rng, sample, temp=1.0):
    pi = visit_policy(root, temp if sample else 0.0)
    if sample:
        return int(root.legal[rng.choice(len(pi), p=pi)])
    return int(root.legal[int(pi.argmax())])


def root_value(root):
    n = root.N.sum()
    return float((root.N * root.q()).sum() / n) if n > 0 else root.value


# ══════════════════════════════════════════════════════════════════════════════
#  Targets
# ══════════════════════════════════════════════════════════════════════════════
# obs (D,) fp16 | legal (k,) int32 | pi (k,) fp32 visit target | z fp32 outcome

def make_target(root):
    return {'obs': root.obs, 'legal': root.legal.copy(),
            'pi': visit_policy(root, 1.0).astype(np.float32),
            'z': np.float32(0.0), 'player': int(root.player), 'solved': False}


def finish_episode(samples, returns):
    for s in samples:
        if s.get('solved'):
            continue          # already carries an exact proven outcome
        s['z'] = np.float32(returns[s['player']])
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Config — mirrors c4.Config field for field wherever the two share a knob
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    num_episodes: int = 2000
    channels: int = 32
    num_blocks: int = 3
    head_ch: int = 8
    device_preference: str = 'auto'
    checkpoint_dir: str = 'c4_alphazero_ckpt'
    resume: bool = True
    seed: int = 0

    # search
    c_puct: float = 1.5
    virtual_loss: float = 1.0
    root_noise_frac: float = 0.25     # AlphaZero root exploration noise
    root_noise_alpha: float = 1.0     # ~10/branching for 7 columns
    fast_sims: int = 50
    full_sims: int = 150
    fast_prob: float = 0.75
    temp_threshold: int = 12          # sample from visits for N plies, then greedy

    # self-play
    use_workers: bool = True
    selfplay_workers: int = 0
    games_per_worker: int = 16
    worker_wave: int = 4
    n_parallel_games: int = 16
    wave_per_game: int = 4
    max_plies: int = 42
    pool_prob: float = 0.2
    random_pool_frac: float = 0.5

    # training
    batch_size: int = 256
    train_steps_per_ep: int = 4
    max_buffer: int = 150_000
    lr_peak: float = 2e-3
    lr_warmup_eps: int = 100
    lr_decay_eps: int = 2000
    lr_min_factor: float = 0.10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_weight: float = 1.0         # loss = policy CE + value_weight * MSE
    # Per-category value MSE against exactly-solved positions; see the
    # ThompsonZero Config for why a relative Elo ladder is not enough.
    solved_dir: str = ''
    solved_every: int = 0
    solved_n: int = 0

    # eval
    quick_eval_every: int = 250
    quick_eval_games: int = 30
    deep_eval_every: int = 1000
    eval_sims: int = 32
    eval_max_plies: int = 42
    eval_device: str = 'cpu'

    def resolved_workers(self):
        if self.selfplay_workers > 0:
            return self.selfplay_workers
        return min(8, max(2, (os.cpu_count() or 8) - 2))


# ══════════════════════════════════════════════════════════════════════════════
#  Multiprocess self-play worker (top level so `spawn` can import it)
# ══════════════════════════════════════════════════════════════════════════════
# Wire format:  request  (worker_id, net_id, obs (n,D) fp16, [legals int32, …])
#               response [(priors (k,) f32, value float), …]

def noise_root(slot, rng, cfg):
    """AlphaZero mixes Dirichlet noise into the priors at the root of EVERY
    move's search.  Subtree reuse means the next root is a node that was
    expanded as a leaf, so noise has to be applied when the root ADVANCES, not
    only when one is built from scratch — otherwise a game gets noise on move 1
    and none afterwards, and the control loses most of its exploration."""
    if slot['root'] is not None and not slot['noised']:
        add_root_noise(slot['root'], rng, cfg['root_noise_frac'],
                       cfg['root_noise_alpha'])
        slot['noised'] = True


def _slot_new(game, cfg, rng, checkpoint_dir):
    sims = cfg['fast_sims'] if rng.random() < cfg['fast_prob'] else cfg['full_sims']
    slot = {'state': game.new_initial_state(), 'hist': [], 'aux': [],
            'actions': [], 'move': 0, 'sims': sims, 'root': None, 'n': 0,
            'pool': None, 'noised': False}
    if cfg['pool_prob'] > 0 and rng.random() < cfg['pool_prob']:
        try:
            labels = [f[6:-3] for f in os.listdir(checkpoint_dir)
                      if f.startswith('bench_') and f.endswith('.pt')] \
                if checkpoint_dir else []
        except OSError:
            labels = []
        label = ('random' if not labels or rng.random() < cfg['random_pool_frac']
                 else labels[rng.integers(len(labels))])
        slot['pool'] = {'label': label, 'side': int(rng.integers(2))}
    return slot


def mp_worker(worker_id, req_q, resp_q, pool_resp_q, episode_q, cfg):
    import pyspiel
    game = pyspiel.load_game(cfg.get('game_name', 'connect_four'))
    c4.set_game(game)
    set_search(cfg['c_puct'], cfg['virtual_loss'])
    rng = np.random.default_rng(cfg['seed'] + worker_id * 7919)
    ckdir = cfg.get('checkpoint_dir')

    def finish_and_reset(i):
        s = slots[i]; st = s['state']
        if st.is_terminal():
            ret = st.returns()
            finish_episode(s['hist'], ret)
            result = 'draw' if ret[0] == 0.0 else 'decisive'
        else:
            result = 'cutoff'
        episode_q.put((s['hist'] + s['aux'], len(s['aux']), result,
                       int(s['move'])))
        slots[i] = _slot_new(game, cfg, rng, ckdir)

    slots = [_slot_new(game, cfg, rng, ckdir)
             for _ in range(cfg['games_per_worker'])]
    mid = max(1, cfg['games_per_worker'] // 2)
    halves = [list(range(mid)), list(range(mid, cfg['games_per_worker']))]

    def collect(idxs):
        evals, pending, obs, legals = [], [], [], []
        seen = set()
        for i in idxs:
            s = slots[i]; st0 = s['state']; pool = s['pool']
            if pool is not None and st0.current_player() == pool['side']:
                continue
            if s['root'] is None:
                leg = st0.legal_actions(); o = c4.make_obs(st0)
                evals.append(('root', i, None, st0.current_player(), leg, o))
                obs.append(o); legals.append(np.asarray(leg, dtype=np.int32))
                continue
            if _node_solved_outcome(s['root']) is not None:
                continue
            noise_root(s, rng, cfg)
            wave = min(cfg['wave'], s['sims'] - s['n'])
            for _ in range(max(wave, 0)):
                path, st, val, edge = _select_leaf(s['root'], st0)
                if st is None:
                    _backup_terminal(path, val, s['aux']); s['n'] += 1; continue
                node, idx = edge
                pending.append((i, path, node, idx))
                if (id(node), idx) not in seen:
                    seen.add((id(node), idx))
                    leg = st.legal_actions(); o = c4.make_obs(st)
                    evals.append(('leaf', node, idx, st.current_player(), leg, o))
                    obs.append(o); legals.append(np.asarray(leg, dtype=np.int32))
        return evals, pending, obs, legals

    def apply_and_advance(idxs, evals, pending, resp):
        if evals:
            for e, (pri, val) in zip(evals, resp):
                kind, a, b, player, leg, o = e
                nd = _AZNode(player, leg, pri, val, obs=o)
                if kind == 'root':
                    slots[a]['root'] = nd
                    slots[a]['noised'] = False
                    noise_root(slots[a], rng, cfg)
                else:
                    a.children[b] = nd
        for i, path, node, idx in pending:
            child = node.children[idx]
            _backup(path, -child.value)   # child's value is the CHILD mover's
            slots[i]['n'] += 1
        for i in idxs:
            s = slots[i]
            if s['root'] is None:
                continue
            if s['n'] < s['sims'] and _node_solved_outcome(s['root']) is None:
                continue
            root = s['root']
            s['hist'].append(make_target(root))
            a = root_pick(root, rng, sample=(s['move'] < cfg['temp_threshold']))
            pidx = int(np.nonzero(root.legal == a)[0][0])
            s['actions'].append(int(a))
            s['root'] = root.children[pidx]
            s['noised'] = False          # the next search needs its own noise
            s['state'].apply_action(a); s['move'] += 1; s['n'] = 0
            if s['state'].is_terminal() or s['move'] >= cfg['max_plies']:
                finish_and_reset(i)

    def resolve_pool_moves(idxs):
        for i in idxs:
            s = slots[i]; pool = s['pool']
            if pool is None:
                continue
            state = s['state']
            if state.current_player() != pool['side']:
                continue
            legal = state.legal_actions()
            if pool['label'] == 'random':
                a = int(legal[rng.integers(len(legal))])
            else:
                req_q.put((worker_id, pool['label'], c4.make_obs(state)[None],
                           [np.asarray(legal, dtype=np.int32)]))
                (pri, _v), = pool_resp_q.get()
                a = int(legal[int(np.asarray(pri).argmax())])
            s['root'] = _descend(s['root'], a)
            s['noised'] = False
            state.apply_action(a); s['actions'].append(a); s['move'] += 1
            if state.is_terminal() or s['move'] >= cfg['max_plies']:
                finish_and_reset(i)

    inflight = [None, None]
    while True:
        for h in (0, 1):
            if inflight[h] is not None:
                evals, pending, sent = inflight[h]
                resp = resp_q.get() if sent else None
                apply_and_advance(halves[h], evals, pending, resp)
                inflight[h] = None
            resolve_pool_moves(halves[h])
            evals, pending, obs, legals = collect(halves[h])
            sent = False
            if evals:
                req_q.put((worker_id, 'live', np.stack(obs), legals)); sent = True
            inflight[h] = (evals, pending, sent)


# ══════════════════════════════════════════════════════════════════════════════
#  Torch: network, loss, self-play drivers, training
# ══════════════════════════════════════════════════════════════════════════════
if _HAS_TORCH:

    class AlphaZeroNet(nn.Module):
        """Trunk (identical to C4DirichletNet's) → policy logits + tanh value."""

        def __init__(self, channels=32, num_blocks=3, head_ch=8):
            super().__init__()
            in_ch = c4._OBS_SHAPE[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                c4._norm(channels), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[c4.ResBlock(channels)
                                        for _ in range(num_blocks)])
            self.head = nn.Sequential(
                nn.Conv2d(channels, head_ch, 1, bias=False),
                c4._norm(head_ch), nn.ReLU(inplace=True), nn.Flatten())
            flat = head_ch * c4._OBS_SHAPE[1] * c4._OBS_SHAPE[2]
            self.policy_out = nn.Linear(flat, c4._NUM_ACTIONS)
            self.value_out = nn.Linear(flat, 1)
            nn.init.zeros_(self.policy_out.weight); nn.init.zeros_(self.policy_out.bias)
            nn.init.zeros_(self.value_out.weight); nn.init.zeros_(self.value_out.bias)

        def forward(self, x):
            h = self.head(self.body(self.stem(x)))
            return self.policy_out(h), torch.tanh(self.value_out(h)).squeeze(-1)

    def nn_eval_states(network, device, states):
        obs16 = np.asarray([c4.make_obs(s) for s in states], dtype=np.float16)
        x = c4.batch_to_tensor(obs16, device)
        with torch.inference_mode():
            logits, value = network(x)
        return logits.cpu().numpy(), value.cpu().numpy(), obs16

    def expand_node(network, device, state, rng=None, noise_frac=0.0,
                    noise_alpha=1.0):
        lg, v, ob = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        row = lg[0][leg]
        row = np.exp(row - row.max())
        node = _AZNode(state.current_player(), leg, row / row.sum(), float(v[0]),
                       obs=ob[0])
        if rng is not None and noise_frac > 0:
            add_root_noise(node, rng, noise_frac, noise_alpha)
        return node

    def build_batch(batch, device):
        B = len(batch)
        K = max(len(s['legal']) for s in batch)
        act = np.zeros((B, K), np.int64)
        mask = np.zeros((B, K), bool)
        pi = np.zeros((B, K), np.float32)
        z = np.zeros(B, np.float32)
        for i, s in enumerate(batch):
            k = len(s['legal'])
            act[i, :k] = s['legal']; mask[i, :k] = True
            pi[i, :k] = s['pi']; z[i] = s['z']
        # One transfer per dtype (see build_batch_meta in the ThompsonZero
        # module for why: DirectML fragments its D3D12 heap under a long run of
        # many small host->device copies).
        ti = torch.from_numpy(act.reshape(-1)).to(device)
        tf = torch.from_numpy(np.concatenate([pi.reshape(-1), z])).to(device)
        tb = torch.from_numpy(mask.reshape(-1)).to(device)
        BK = B * K
        return {'act': ti.view(B, K), 'mask': tb.view(B, K),
                'pi': tf[:BK].view(B, K), 'z': tf[BK:BK + B]}

    def az_loss(logits, value, meta, value_weight=1.0):
        """AlphaZero's loss: cross-entropy to the visit distribution over LEGAL
        moves, plus mean-squared error of the scalar value against the outcome."""
        lg = logits.gather(1, meta['act']).masked_fill(~meta['mask'], -1e9)
        logp = F.log_softmax(lg, dim=1)
        L_pol = -(meta['pi'] * logp).sum(1).mean()
        L_val = ((meta['z'] - value) ** 2).mean()
        total = L_pol + value_weight * L_val
        with torch.no_grad():
            ent = -(meta['pi'].clamp_min(1e-9).log() * meta['pi']).sum(1).mean()
            diag = torch.stack([total, L_pol, L_val, ent,
                                value.abs().mean()]).to('cpu', copy=False).tolist()
        return total, dict(zip(('loss', 'pol', 'val', 'tgt_ent', 'absv'), diag))

    def train_step(network, optimizer, batch, device, value_weight=1.0,
                   grad_clip=1.0, model_lock=None):
        import contextlib
        lock = model_lock or contextlib.nullcontext()

        def _once():
            x = c4.batch_to_tensor([s['obs'] for s in batch], device)
            meta = build_batch(batch, device)
            with lock:
                logits, value = network(x)
                optimizer.zero_grad()
                loss, parts = az_loss(logits, value, meta, value_weight)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(network.parameters(),
                                                       grad_clip)
                # Same guard as the ThompsonZero engine: a NaN norm becomes a
                # NaN scale factor, AdamW writes NaN into every weight, and the
                # run continues while being dead.  Skipping costs one batch.
                ok = (math.isfinite(parts['loss'])
                      and bool(torch.isfinite(gnorm)))
                if ok:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                parts['nonfinite'] = 0.0 if ok else 1.0
            return parts['loss'], parts

        return c4.device_retry(_once)

    # ── Bots ──────────────────────────────────────────────────────────────────
    class AZMCTSBot:
        def __init__(self, game, network, device, max_simulations, batch_size=8,
                     random_state=None, noise_frac=0.0):
            self.game, self.network, self.device = game, network, device
            self.max_simulations = max_simulations
            self.batch_size = batch_size
            self.noise_frac = noise_frac
            self._rng = random_state or np.random.default_rng()

        def mcts_search(self, state, root=None):
            if root is None:
                root = expand_node(self.network, self.device, state, self._rng,
                                   self.noise_frac)
            sims = 0
            while sims < self.max_simulations:
                if _node_solved_outcome(root) is not None:
                    break
                wave = min(self.batch_size, self.max_simulations - sims)
                pending, uniq = [], {}
                for _ in range(wave):
                    path, st, val, edge = _select_leaf(root, state)
                    if st is None:
                        _backup_terminal(path, val); sims += 1
                    else:
                        node, idx = edge
                        pending.append((path, node, idx))
                        uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    items = list(uniq.values())
                    lg, v, ob = nn_eval_states(self.network, self.device,
                                               [e[2] for e in items])
                    for (node, idx, st), row, val, o in zip(items, lg, v, ob):
                        leg = st.legal_actions()
                        r = row[leg]; r = np.exp(r - r.max())
                        node.children[idx] = _AZNode(st.current_player(), leg,
                                                     r / r.sum(), float(val),
                                                     obs=o)
                for path, node, idx in pending:
                    _backup(path, -node.children[idx].value)
                    sims += 1
            return root

    def policy_move(network, state, device):
        """Search-free move: argmax of the policy head over legal actions."""
        lg, _v, _o = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        return int(leg[int(np.asarray(lg[0][leg]).argmax())])

    def value_greedy_move(network, state, device):
        """Search-free move: one-ply lookahead with the value head.

        AlphaZero has no per-action value estimate to read off directly (unlike
        ThompsonZero's action heads) — the policy head is only a search prior,
        trained to match visit counts, not to be argmax-accurate on its own. The
        value head IS a genuine position evaluation, so the comparable
        search-free move is to actually apply each legal action and take the
        one whose child position it likes least for the opponent, exactly
        mirroring how ThompsonZero's value_greedy_move reads its own per-action
        heads. Terminal children use the true game outcome instead of the net,
        matching how the network is (never) asked to evaluate terminal states
        during search."""
        leg = state.legal_actions()
        pending_idx, pending_states, vals = [], [], [None] * len(leg)
        for i, a in enumerate(leg):
            cs = state.clone()
            cs.apply_action(int(a))
            if cs.is_terminal():
                r = cs.returns()[state.current_player()]
                vals[i] = float(r)
            else:
                pending_idx.append(i)
                pending_states.append(cs)
        if pending_states:
            _lg, v, _o = nn_eval_states(network, device, pending_states)
            for i, val in zip(pending_idx, v):
                vals[i] = -float(val)
        return int(leg[int(np.argmax(vals))])

    def quick_match(net_a, net_b, game, n_games, device, rng=None,
                    opening_plies=2, max_plies=42):
        rng = rng or np.random.default_rng()
        wa = d = wb = 0
        for i in range(n_games):
            a_side = i % 2
            st = game.new_initial_state()
            for _ in range(opening_plies):
                if st.is_terminal():
                    break
                leg = st.legal_actions()
                st.apply_action(int(leg[rng.integers(len(leg))]))
            ply = 0
            while not st.is_terminal() and ply < max_plies:
                net = net_a if st.current_player() == a_side else net_b
                if net is None:
                    leg = st.legal_actions()
                    mv = int(leg[rng.integers(len(leg))])
                else:
                    mv = policy_move(net, st, device)
                st.apply_action(mv); ply += 1
            if not st.is_terminal():
                d += 1; continue
            r = st.returns()[a_side]
            wa += r > 0; wb += r < 0; d += r == 0
        return int(wa), int(d), int(wb)

    # ── Self-play drivers ─────────────────────────────────────────────────────
    class ParallelSelfPlay:
        """Single-process fallback: n games advanced in lockstep, one batched
        forward per leaf wave."""

        def __init__(self, game, network, device, cfg, checkpoint_dir=None,
                     seed=None):
            self.game, self.network, self.device = game, network, device
            self.cfg = cfg
            self.rng = np.random.default_rng(seed)
            self.checkpoint_dir = checkpoint_dir
            self.n_parallel = cfg['n_parallel']
            self.wave = cfg['wave_per_game']
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0}
            self.fwd_calls = self.fwd_rows = 0
            self._pool_nets = {}
            self.slots = [_slot_new(game, cfg, self.rng, checkpoint_dir)
                          for _ in range(self.n_parallel)]

        def _finish(self, i):
            s = self.slots[i]; st = s['state']
            if st.is_terminal():
                ret = st.returns()
                finish_episode(s['hist'], ret)
                result = 'draw' if ret[0] == 0.0 else 'decisive'
            else:
                result = 'cutoff'
            self.stats['games'] += 1; self.stats['plies'] += int(s['move'])
            if result == 'draw':   self.stats['draw'] += 1
            if result == 'cutoff': self.stats['cutoff'] += 1
            self.last_aux = len(s['aux'])
            data = s['hist'] + s['aux']
            self.slots[i] = _slot_new(self.game, self.cfg, self.rng,
                                      self.checkpoint_dir)
            return data

        def _pool_net(self, label):
            net = self._pool_nets.get(label)
            if net is None:
                net = load_benchmark_net(self.checkpoint_dir, label,
                                         self.cfg['net_sig'])
                self._pool_nets[label] = net
            return net

        def _step(self):
            done = []
            for i, s in enumerate(self.slots):
                pool, state = s['pool'], s['state']
                if pool is None or state.current_player() != pool['side']:
                    continue
                if pool['label'] == 'random':
                    leg = state.legal_actions()
                    a = int(leg[self.rng.integers(len(leg))])
                else:
                    a = policy_move(self._pool_net(pool['label']), state, 'cpu')
                s['root'] = _descend(s['root'], a)
                s['noised'] = False
                state.apply_action(a); s['move'] += 1
                if state.is_terminal() or s['move'] >= self.cfg['max_plies']:
                    done.append(self._finish(i))
            pending, evals, seen = [], [], set()
            for i, s in enumerate(self.slots):
                pool = s['pool']
                if pool is not None and s['state'].current_player() == pool['side']:
                    continue
                if s['root'] is None:
                    evals.append(('root', i, None, s['state'])); continue
                if _node_solved_outcome(s['root']) is not None:
                    continue
                noise_root(s, self.rng, self.cfg)
                wave = min(self.wave, s['sims'] - s['n'])
                for _ in range(max(wave, 0)):
                    path, st, val, edge = _select_leaf(s['root'], s['state'])
                    if st is None:
                        _backup_terminal(path, val, s['aux']); s['n'] += 1
                    else:
                        node, idx = edge
                        pending.append((i, path, node, idx))
                        if (id(node), idx) not in seen:
                            seen.add((id(node), idx))
                            evals.append(('leaf', node, idx, st))
            if evals:
                self.fwd_calls += 1; self.fwd_rows += len(evals)
                lg, v, ob = nn_eval_states(self.network, self.device,
                                           [e[3] for e in evals])
                for (kind, a, b, st), row, val, o in zip(evals, lg, v, ob):
                    leg = st.legal_actions()
                    r = row[leg]; r = np.exp(r - r.max())
                    nd = _AZNode(st.current_player(), leg, r / r.sum(),
                                 float(val), obs=o)
                    if kind == 'root':
                        self.slots[a]['root'] = nd
                        self.slots[a]['noised'] = False
                        noise_root(self.slots[a], self.rng, self.cfg)
                    else:
                        a.children[b] = nd
            for i, path, node, idx in pending:
                _backup(path, -node.children[idx].value)
                self.slots[i]['n'] += 1
            for i, s in enumerate(self.slots):
                if s['root'] is None:
                    continue
                if (s['n'] < s['sims']
                        and _node_solved_outcome(s['root']) is None):
                    continue
                root = s['root']
                s['hist'].append(make_target(root))
                a = root_pick(root, self.rng,
                              sample=(s['move'] < self.cfg['temp_threshold']))
                pidx = int(np.nonzero(root.legal == a)[0][0])
                s['root'] = root.children[pidx]
                s['noised'] = False
                s['state'].apply_action(a); s['move'] += 1; s['n'] = 0
                if s['state'].is_terminal() or s['move'] >= self.cfg['max_plies']:
                    done.append(self._finish(i))
            return done

        def episodes(self):
            while True:
                for d in self._step():
                    yield d

        def shutdown(self):
            pass

    import threading as _threading
    import queue as _queue
    import multiprocessing as _mp

    class MPSelfPlayPool:
        """Worker processes run the trees; a server thread here batches all of
        their NN requests into one forward pass.  Same design as the
        ThompsonZero pool, with AlphaZero's (priors, value) wire format."""

        def __init__(self, network, device, n_workers, cfg, batch_window_s=0.002,
                     checkpoint_dir=None, max_batch_rows=1024):
            self.network, self.device = network, device
            self.checkpoint_dir = checkpoint_dir
            self.net_sig = cfg['net_sig']
            self._pool_nets = {}
            self.lock = _threading.Lock()
            self._stop = _threading.Event()
            self.window, self.max_batch_rows = batch_window_s, max_batch_rows
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0}
            self.fwd_calls = self.fwd_rows = 0
            ctx = _mp.get_context('spawn')
            self.req_q = ctx.Queue()
            self.episode_q = ctx.Queue(maxsize=64)
            self.resp_qs = [ctx.Queue() for _ in range(n_workers)]
            self.pool_resp_qs = [ctx.Queue() for _ in range(n_workers)]
            if str(device) != 'cpu':
                _t = torch.zeros(4, device=device, requires_grad=True)
                (_t * 2.0).sum().backward()
            self.procs = [ctx.Process(target=mp_worker,
                                      args=(i, self.req_q, self.resp_qs[i],
                                            self.pool_resp_qs[i],
                                            self.episode_q, cfg),
                                      daemon=True) for i in range(n_workers)]
            self.server = None
            try:
                for p in self.procs:
                    p.start()
            except Exception:
                self.procs = [p for p in self.procs if p.is_alive()]
                self.shutdown()
                raise
            self.server = _threading.Thread(target=self._serve, daemon=True)
            self.server.start()

        def _get_net(self, net_id):
            if net_id == 'live':
                return self.network, self.device, True
            net = self._pool_nets.get(net_id)
            if net is None:
                try:
                    net = load_benchmark_net(self.checkpoint_dir, net_id,
                                             self.net_sig)
                except Exception as e:
                    print(f'pool net {net_id} unavailable ({e}) — using live net')
                    return self.network, self.device, True
                self._pool_nets[net_id] = net
            return net, 'cpu', False

        def _serve(self):
            import contextlib
            while not self._stop.is_set():
                try:
                    reqs = [self.req_q.get(timeout=0.1)]
                except _queue.Empty:
                    continue
                rows = reqs[0][2].shape[0]
                deadline = time.monotonic() + self.window
                while time.monotonic() < deadline and rows < self.max_batch_rows:
                    try:
                        r = self.req_q.get_nowait(); reqs.append(r)
                        rows += r[2].shape[0]
                    except _queue.Empty:
                        time.sleep(0.0003)
                groups = {}
                for wid, net_id, obs, legals in reqs:
                    groups.setdefault(net_id, []).append((wid, obs, legals))
                for net_id, group in groups.items():
                    net, dev, needs_lock = self._get_net(net_id)
                    obs = np.concatenate([o for _, o, _ in group], axis=0)
                    xin = obs.reshape(-1, *c4._OBS_SHAPE).astype(np.float32)
                    if net_id == 'live':
                        self.fwd_calls += 1; self.fwd_rows += xin.shape[0]
                    ctxm = self.lock if needs_lock else contextlib.nullcontext()
                    with ctxm, torch.no_grad():
                        lg, v = net(torch.from_numpy(xin).to(dev))
                        lg = lg.cpu().numpy(); v = v.cpu().numpy()
                    tqs = self.resp_qs if net_id == 'live' else self.pool_resp_qs
                    ri = 0
                    for wid, o, ls in group:
                        out = []
                        for l in ls:
                            row = lg[ri][l]
                            row = np.exp(row - row.max())
                            out.append((row / row.sum(), float(v[ri])))
                            ri += 1
                        tqs[wid].put(out)

        def episodes(self):
            while True:
                samples, n_aux, result, plies = self.episode_q.get()
                self.stats['games'] += 1; self.stats['plies'] += plies
                if result == 'draw':   self.stats['draw'] += 1
                if result == 'cutoff': self.stats['cutoff'] += 1
                yield samples

        def shutdown(self):
            self._stop.set()
            try:
                if self.server is not None:
                    self.server.join(timeout=2.0)
            except Exception:
                pass
            for p in self.procs:
                p.terminate()
            for p in self.procs:
                p.join(timeout=2.0)
            for q in ([self.req_q, self.episode_q] + self.resp_qs
                      + self.pool_resp_qs):
                try:
                    q.close(); q.cancel_join_thread()
                except Exception:
                    pass

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def _cpu_sd(net):
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    def cpu_clone(net, sig):
        c = AlphaZeroNet(*sig); c.load_state_dict(_cpu_sd(net)); c.eval()
        return c

    def save_benchmark_net(checkpoint_dir, label, net):
        os.makedirs(checkpoint_dir, exist_ok=True)
        sd = _cpu_sd(net)
        c4._assert_finite_sd(sd, f'bench_{label}.pt')
        torch.save(sd, os.path.join(checkpoint_dir, f'bench_{label}.pt'))

    def load_benchmark_net(checkpoint_dir, label, sig):
        net = AlphaZeroNet(*sig)
        net.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f'bench_{label}.pt'),
            map_location='cpu', weights_only=True))
        net.eval()
        return net

    def save_checkpoint(checkpoint_dir, ep, net, optimizer, scheduler, hist,
                        cfg=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        c4._assert_finite_sd(_cpu_sd(net), f'latest.pt at ep {ep}')
        blob = {'ep': ep, 'model': _cpu_sd(net), 'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict() if scheduler else None,
                'hist': hist, 'cfg': asdict(cfg) if cfg is not None else None}
        tmp = os.path.join(checkpoint_dir, 'latest.pt.tmp')
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(checkpoint_dir, 'latest.pt'))

    def load_checkpoint(checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'latest.pt')
        if not os.path.exists(path):
            return None
        blob = torch.load(path, map_location='cpu', weights_only=False)
        # Same trap as the ThompsonZero engine: bench_N.pt is a bare state_dict,
        # latest.pt is a run blob, and copying one over the other to rewind a
        # run otherwise fails with a bare KeyError: 'model'.
        if isinstance(blob, dict) and 'model' not in blob:
            if all(torch.is_tensor(v) for v in blob.values()):
                raise ValueError(
                    f'{path} is a bare network state_dict (a bench_N.pt), not a '
                    f'resumable checkpoint. Use '
                    f'rewind_to_generation(checkpoint_dir, N) instead.')
            raise ValueError(f'{path} has no "model" key; not a checkpoint.')
        return blob

    def rewind_to_generation(checkpoint_dir, gen, log=print):
        """Rebuild latest.pt from bench_{gen}.pt.  See the ThompsonZero engine's
        docstring — this engine keeps no Elo pool, so only weights, episode
        counter and history carry over."""
        bench = os.path.join(checkpoint_dir, f'bench_{gen}.pt')
        if not os.path.exists(bench):
            raise FileNotFoundError(bench)
        sd = torch.load(bench, map_location='cpu', weights_only=True)
        c4._assert_finite_sd(sd, f'bench_{gen}.pt')
        path = os.path.join(checkpoint_dir, 'latest.pt')
        old = {}
        if os.path.exists(path):
            try:
                old = torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                old = {}
            if not isinstance(old, dict) or 'model' not in old:
                old = {}
        prev = old.get('hist') or {}
        keep = sum(1 for e in prev.get('ep', []) if e <= gen)
        qkeep = sum(1 for e in prev.get('quick_ep', []) if e <= gen)
        hist = {}
        for k, v in prev.items():
            hist[k] = (v[:qkeep] if k in ('quick_ep', 'q_w', 'q_d', 'q_l')
                       else v[:keep]) if isinstance(v, list) else v
        blob = {'ep': int(gen), 'model': sd, 'optim': None, 'sched': None,
                'hist': hist, 'cfg': old.get('cfg')}
        tmp = path + '.tmp'
        torch.save(blob, tmp)
        os.replace(tmp, path)
        log(f'rewound {checkpoint_dir} to generation {gen}')
        return blob

    # ── Training driver ───────────────────────────────────────────────────────
    def _worker_cfg(cfg, sig):
        return dict(seed=cfg.seed, game_name='connect_four', net_sig=sig,
                    games_per_worker=cfg.games_per_worker, wave=cfg.worker_wave,
                    n_parallel=cfg.n_parallel_games,
                    wave_per_game=cfg.wave_per_game,
                    fast_sims=cfg.fast_sims, full_sims=cfg.full_sims,
                    fast_prob=cfg.fast_prob, temp_threshold=cfg.temp_threshold,
                    max_plies=cfg.max_plies, pool_prob=cfg.pool_prob,
                    random_pool_frac=cfg.random_pool_frac,
                    checkpoint_dir=cfg.checkpoint_dir,
                    c_puct=cfg.c_puct, virtual_loss=cfg.virtual_loss,
                    root_noise_frac=cfg.root_noise_frac,
                    root_noise_alpha=cfg.root_noise_alpha)

    def _load_solved_probe(cfg, log):
        if not cfg.solved_dir:
            return None
        try:
            import connect4_solved_eval as sev
            probe = sev.ValueProbe.build(cfg.solved_dir, limit=cfg.solved_n,
                                         log=log)
            return probe
        except Exception as e:
            log(f'solved-position eval DISABLED ({type(e).__name__}: {e})')
            return None

    def _solved_eval_line(probe, net, cfg):
        """Value MSE per difficulty for one checkpoint: a single batched forward
        over the precomputed observations, so this is cheap enough to run at
        every deep eval.  Returns (printable line, pooled MSE)."""
        import connect4_solved_eval as sev
        d = probe.mse_by_bucket(sev.alphazero_value_fn(net, cfg.eval_device))
        return sev.line(d), d['all']

    def run_training(cfg, game=None, log=print):
        """Self-play + training, logging in the same shape as the ThompsonZero
        driver so the two runs read alike."""
        import threading
        from collections import defaultdict
        game = game or c4.load_game()
        device, backend = c4.pick_device(cfg.device_preference)
        c4.set_game(game)
        set_search(cfg.c_puct, cfg.virtual_loss)
        c4.set_backend(backend, device)     # probe fused kernels on this device
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        sig = (cfg.channels, cfg.num_blocks, cfg.head_ch)
        base_network = AlphaZeroNet(*sig).to(device)
        network = base_network
        optimizer = (c4.LerpFreeAdamW if backend == 'directml'
                     else torch.optim.AdamW)(
            network.parameters(), lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

        def _lr(ep):
            if ep < cfg.lr_warmup_eps:
                return ep / max(cfg.lr_warmup_eps, 1)
            f = min((ep - cfg.lr_warmup_eps)
                    / max(cfg.lr_decay_eps - cfg.lr_warmup_eps, 1), 1.0)
            return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                1 + np.cos(np.pi * f))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)

        wcfg = _worker_cfg(cfg, sig)
        if cfg.use_workers:
            n_w = cfg.resolved_workers()
            self_play = MPSelfPlayPool(network, device, n_w, wcfg,
                                       checkpoint_dir=cfg.checkpoint_dir)
            torch.set_num_threads(max(2, (os.cpu_count() or 8) - n_w))
            log(f'Self-play: {n_w} WORKER PROCESSES x {cfg.games_per_worker} games')
        else:
            self_play = ParallelSelfPlay(game, network, device, wcfg,
                                         checkpoint_dir=cfg.checkpoint_dir,
                                         seed=cfg.seed)
            log(f'Self-play: SINGLE PROCESS, {cfg.n_parallel_games} games')
        episode_stream = self_play.episodes()
        model_lock = getattr(self_play, 'lock', None) or threading.Lock()

        hist = {'ep': [], 'loss': [], 'pol': [], 'val': [], 'tgt_ent': [],
                'absv': [], 'draw_pct': [], 'plies': [], 'buf': [],
                'quick_ep': [], 'q_w': [], 'q_d': [], 'q_l': [],
                'solved': [], 'solved_ep': []}
        replay_buffer, start_ep = [], 1
        ckpt = load_checkpoint(cfg.checkpoint_dir) if cfg.resume else None
        if ckpt is not None:
            base_network.load_state_dict(ckpt['model'])
            if ckpt.get('optim'):
                optimizer.load_state_dict(ckpt['optim'])
            else:
                log('  (no optimizer state — starting Adam moments from zero)')
            if ckpt.get('sched'):
                scheduler.load_state_dict(ckpt['sched'])
            old = ckpt.get('hist') or {}; n = len(old.get('ep', []))
            for k in hist:
                if k not in old:
                    old[k] = [float('nan')] * n
            hist = old; start_ep = ckpt['ep'] + 1
            log(f'resumed at ep {ckpt["ep"]}')

        solved_probe = _load_solved_probe(cfg, log)

        n_params = sum(p.numel() for p in base_network.parameters())
        log(f'device={device} backend={backend} | params={n_params:,} '
            f'| ALPHAZERO | start_ep={start_ep}')
        ref = None
        if start_ep == 1:
            init = cpu_clone(base_network, sig)
            save_benchmark_net(cfg.checkpoint_dir, '0', init)
            ref = init
        else:
            try:
                labels = sorted((int(f[6:-3]) for f in
                                 os.listdir(cfg.checkpoint_dir)
                                 if f.startswith('bench_')), reverse=True)
                ref = load_benchmark_net(cfg.checkpoint_dir, str(labels[0]), sig)
            except Exception:
                ref = cpu_clone(base_network, sig)

        prev = dict(self_play.stats)
        t_sp = t_tr = 0.0
        win_t0 = time.perf_counter(); win_games = self_play.stats['games']
        bar = c4.Progress(cfg.quick_eval_every, label='ep ')
        bar_mark = (time.perf_counter(), self_play.stats['games']); inst = 0.0
        prev_fwd = (self_play.fwd_calls, self_play.fwd_rows)
        step_fails = 0
        try:
            for ep in range(start_ep, cfg.num_episodes + 1):
                network.eval()
                a = time.perf_counter()
                raw = next(episode_stream)
                t_sp += time.perf_counter() - a
                replay_buffer.extend(raw)
                if len(replay_buffer) > cfg.max_buffer:
                    del replay_buffer[:-cfg.max_buffer]
                network.train()
                accs = defaultdict(list)
                b = time.perf_counter()
                if len(replay_buffer) >= cfg.batch_size:
                    for _ in range(cfg.train_steps_per_ep):
                        batch = random.sample(replay_buffer, cfg.batch_size)
                        try:
                            _lv, parts = train_step(
                                network, optimizer, batch, device,
                                cfg.value_weight, cfg.grad_clip,
                                model_lock=model_lock)
                        except RuntimeError as e:
                            step_fails += 1
                            log(f'  ! train step failed, skipped '
                                f'({type(e).__name__}: {e}) '
                                f'[{step_fails} consecutive]')
                            if step_fails >= 20:
                                raise
                            continue
                        step_fails = 0
                        for k, v in parts.items():
                            accs[k].append(v)
                    scheduler.step()
                t_tr += time.perf_counter() - b

                done = ep - (ep - 1) // cfg.quick_eval_every * cfg.quick_eval_every
                if ep % cfg.quick_eval_every != 0:
                    tn, gn = time.perf_counter(), self_play.stats['games']
                    if tn - bar_mark[0] >= 3.0:
                        inst = (gn - bar_mark[1]) / (tn - bar_mark[0])
                        bar_mark = (tn, gn)
                    bar.update(done, extra=f'{inst:.2f} g/s')
                    continue
                bar.close()
                st = self_play.stats
                dg = max(st['games'] - prev['games'], 1)
                draw_pct = 100 * (st['draw'] - prev['draw']) / dg
                plies = (st['plies'] - prev['plies']) / dg
                prev = dict(st)
                ml = lambda k: float(np.mean(accs[k])) if accs[k] else float('nan')
                hist['ep'].append(ep); hist['loss'].append(ml('loss'))
                for k in ('pol', 'val', 'tgt_ent', 'absv'):
                    hist[k].append(ml(k))
                hist['draw_pct'].append(draw_pct); hist['plies'].append(plies)
                hist['buf'].append(len(replay_buffer))
                diag = (f'loss {ml("loss"):.3f} (pol {ml("pol"):.3f} '
                        f'val {ml("val"):.3f}) | tgt_ent {ml("tgt_ent"):.2f} '
                        f'|v| {ml("absv"):.2f} | dr {draw_pct:.0f}% '
                        f'ply {plies:.0f} buf {len(replay_buffer)//1000}k '
                        f'| lr {optimizer.param_groups[0]["lr"]:.2e}')
                wall = max(time.perf_counter() - win_t0, 1e-9)
                dg2 = self_play.stats['games'] - win_games
                dfc = self_play.fwd_calls - prev_fwd[0]
                dfr = self_play.fwd_rows - prev_fwd[1]
                perf = (f'{dg2/wall:.2f} games/s | wait(sp) {100*t_sp/wall:.0f}% '
                        f'train {100*t_tr/wall:.0f}% '
                        f'| NNbatch {dfr/max(dfc,1):.0f} ({dfc/wall:.0f} fwd/s)')
                t_sp = t_tr = 0.0; win_t0 = time.perf_counter()
                win_games = self_play.stats['games']
                prev_fwd = (self_play.fwd_calls, self_play.fwd_rows)

                if ep % cfg.deep_eval_every == 0:
                    snap = cpu_clone(base_network, sig)
                    save_benchmark_net(cfg.checkpoint_dir, str(ep), snap)
                    ref = snap
                    log(f'ep {ep:6d} | {diag}   [checkpoint saved]')
                    if solved_probe and (not cfg.solved_every
                                          or ep % cfg.solved_every == 0):
                        line, agg = _solved_eval_line(solved_probe, snap, cfg)
                        hist['solved'].append(agg)
                        hist['solved_ep'].append(ep)
                        log(f'         SOLVED {line}')
                else:
                    eval_net = cpu_clone(base_network, sig)
                    w, d, l = quick_match(eval_net, ref, game,
                                          cfg.quick_eval_games, cfg.eval_device,
                                          max_plies=cfg.eval_max_plies)
                    hist['quick_ep'].append(ep); hist['q_w'].append(w)
                    hist['q_d'].append(d); hist['q_l'].append(l)
                    log(f'ep {ep:6d} | {diag} | vs last ckpt (no-MCTS) '
                        f'W{w} D{d} L{l}')
                log(f'         perf: {perf}')
                save_checkpoint(cfg.checkpoint_dir, ep, base_network, optimizer,
                                scheduler, hist, cfg)
                bar.reset()
        finally:
            bar.close()
            self_play.shutdown()
        return hist
