#!/usr/bin/env python3
"""Chess live-play web server — play against a checkpointed model, or watch two
models play each other, in a browser (works through ngrok).

Handles BOTH engines from this repo, auto-detected per checkpoint:
  • ThompsonHybrid — state-value Dirichlet head + policy-prior ordering head,
    PUCT-style Thompson selection, mixture propagation, MCTS-Solver.
  • AlphaZero      — policy + scalar-value heads, standard PUCT MCTS +
    MCTS-Solver (the control notebook's engine).
Each --model is inspected (its head weights identify the engine) and driven by
the matching search — identical semantics to that notebook's training worker.
You can even watch one against the other (--model hybrid.pt --model2 az.pt).

Usage:
    python chess_play_server.py --model chess_checkpoints_thompson_hybrid/bench_2000.pt
    python chess_play_server.py --model chess_checkpoints_alphazero/bench_2000.pt
    python chess_play_server.py --model A.pt --model2 B.pt      # watch A vs B
    ngrok http 8765                                             # then share URL

Flags:
    --model  PATH   checkpoint for the engine (bench_*.pt or latest.pt)
    --model2 PATH   second engine for watch mode (default: same as --model)
    --port   N      HTTP port (default 8765)
    --device D      inference device (default cpu — batch-1..8 is CPU's regime)
    --snapshot-secs S   seconds between analysis snapshots while thinking (2)

Network architecture (channels/blocks) and engine type are inferred from the
checkpoint. Net + tree-ops are inlined from the two training notebooks.
"""
import argparse
import json as _json
import math
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

# Chess is OpenSpiel's native C++ game — no inlined engine needed (unlike Boop).
GAME = pyspiel.load_game('chess')
_NUM_ACTIONS = GAME.num_distinct_actions()
_OBS_SHAPE   = tuple(GAME.observation_tensor_shape())          # (20, 8, 8)
# Which player id is White (the side to move in the initial position). Used for
# board orientation + mapping the human's colour choice to a player id.
WHITE_PLAYER = GAME.new_initial_state().current_player()

# ── Outcome / confidence constants (shared by the net and the tree) ───────────
_WIN, _DRAW, _LOSS = 0, 1, 2
_FLIP_TERM = np.array([_LOSS, _DRAW, _WIN], dtype=np.int8)
ALPHA_FLOOR  = 0.05
CONF_MIN     = 0.5
CONF_MAX     = 100.0
POL_CONF_MIN = 1.0
POL_CONF_MAX = 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Network  (ThompsonHybridChessNet — value head + gathered flat policy head)
# ═══════════════════════════════════════════════════════════════════════════════

def state_to_tensor(state, device):
    obs = np.array(state.observation_tensor(state.current_player()), dtype=np.float32)
    x   = obs.reshape(1, *_OBS_SHAPE)
    return torch.from_numpy(x).to(device)


def batch_to_tensor(obs_list, device):
    obs = np.asarray(obs_list, dtype=np.float32)
    x   = obs.reshape(-1, *_OBS_SHAPE)
    return torch.from_numpy(x).to(device)


class _GroupNorm(nn.Module):
    """GroupNorm from elementwise ops (DirectML-safe; no running stats)."""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        n, c = x.shape[0], x.shape[1]
        xg = x.reshape(n, self.num_groups, -1)
        mean = xg.mean(dim=2, keepdim=True)
        var = (xg - mean).pow(2).mean(dim=2, keepdim=True)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        x = xg.reshape(x.shape)
        return x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)


def _norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return _GroupNorm(groups, channels)


def _softplus(x):
    return torch.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 2),
        )

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        scale, bias = self.fc(s).chunk(2, dim=1)
        scale = torch.sigmoid(scale)
        return x * scale[:, :, None, None] + bias[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _norm(channels),
        )
        self.se  = SEBlock(channels) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.net(x)) + x)


class ThompsonHybridChessNet(nn.Module):
    """Value head: (p_win, p_draw, p_loss) + confidence for the current position.
    Policy head: logits over the 4674 actions + a scalar concentration β."""
    _HEAD_CH = 8

    def __init__(self, channels=64, num_blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(_OBS_SHAPE[0], channels, 3, padding=1, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, self._HEAD_CH, 1, bias=False),
            _norm(self._HEAD_CH),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        flat = self._HEAD_CH * _OBS_SHAPE[1] * _OBS_SHAPE[2]
        self.v_dist = nn.Linear(flat, 3)
        self.v_conf = nn.Linear(flat, 1)
        self.p_log  = nn.Linear(flat, _NUM_ACTIONS)
        self.p_conf = nn.Linear(flat, 1)

    def forward(self, x):
        h       = self.head(self.body(self.stem(x)))
        v_probs = F.softmax(self.v_dist(h), dim=-1)
        v_conf  = CONF_MIN + (CONF_MAX - CONF_MIN) * torch.sigmoid(
            self.v_conf(h).squeeze(-1))
        p_logits = self.p_log(h)
        p_conf  = POL_CONF_MIN + (POL_CONF_MAX - POL_CONF_MIN) * torch.sigmoid(
            self.p_conf(h).squeeze(-1))
        return v_probs, v_conf, p_logits, p_conf


class AlphaZeroChessNet(nn.Module):
    """Standard AlphaZero net: policy logits over all actions + a scalar value
    in [-1,1] (mover's perspective). Same shared body as the hybrid; only the
    heads differ. Inlined from chess_alphazero_training.ipynb."""
    _HEAD_CH = 8

    def __init__(self, channels=64, num_blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(_OBS_SHAPE[0], channels, 3, padding=1, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, self._HEAD_CH, 1, bias=False),
            _norm(self._HEAD_CH),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        flat = self._HEAD_CH * _OBS_SHAPE[1] * _OBS_SHAPE[2]
        self.policy_out = nn.Linear(flat, _NUM_ACTIONS)
        self.value_out  = nn.Sequential(
            nn.Linear(flat, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        h      = self.head(self.body(self.stem(x)))
        logits = self.policy_out(h)
        value  = torch.tanh(self.value_out(h)).squeeze(-1)
        return logits, value


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Bayesian-MCTS tree ops (torch-free numpy — identical to the worker)
# ═══════════════════════════════════════════════════════════════════════════════

_GRID_G  = 33
_GRID_X  = (np.arange(_GRID_G) + 0.5) / _GRID_G
_GRID_V  = 2.0 * _GRID_X - 1.0
_GRID_V2 = _GRID_V ** 2
_GRID_LX  = np.log(_GRID_X)
_GRID_L1X = np.log1p(-_GRID_X)
_SPIKE   = np.eye(_GRID_G)
_SPIKE_WIN, _SPIKE_LOSS, _SPIKE_DRAW = (_SPIKE[_GRID_G - 1], _SPIKE[0],
                                        _SPIKE[_GRID_G // 2])
_VLOSS_PEN = 1.0
_MAX_FRAC  = 0.0
_C_PUCT = 1.5
_FPU_REDUCTION = 0.25
_FORCING_MAXLEGAL = 12

_TDOF    = 4.0
_VAR_RES = (2.0 / _GRID_G) ** 2 / 12.0

def _psi(z2):
    return (_TDOF + 1.0) * z2 / (_TDOF + z2)

_zg = np.linspace(-10.0, 10.0, 8001)
_PSI_NORM = float(np.sum(_psi(_zg ** 2)
                         * np.exp(-0.5 * _zg ** 2) / np.sqrt(2.0 * np.pi))
                  * (_zg[1] - _zg[0]))
del _zg


class _Calib:
    __slots__ = ('s', 'n', 'd')

    def __init__(self, prior_n=50.0, halflife=2000.0):
        self.s = prior_n
        self.n = prior_n
        self.d = 0.5 ** (1.0 / halflife)

    def observe(self, e2, var_sum):
        z2 = e2 / (var_sum + 2.0 * _VAR_RES)
        self.s = self.s * self.d + _psi(z2) / _PSI_NORM
        self.n = self.n * self.d + 1.0

    def lam(self):
        return self.s / self.n


def _beta_pmf_rows(alpha, beta):
    logw = ((alpha[:, None] - 1.0) * _GRID_LX[None, :]
            + (beta[:, None] - 1.0) * _GRID_L1X[None, :])
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)


def _flip_pmf(pmf):
    return pmf[::-1].copy()


def _dirichlet_leaf_belief(alpha_w, alpha_d, alpha_l, lam=1.0):
    C = alpha_w + alpha_d + alpha_l
    Ed = alpha_d / C
    Vd = alpha_d * (alpha_w + alpha_l) / (C**2 * (C + 1.0))
    Eq = alpha_w / (alpha_w + alpha_l)
    Vq = alpha_w * alpha_l / ((alpha_w + alpha_l)**2 * (alpha_w + alpha_l + 1.0))
    EX, VX = 1.0 - Ed, Vd
    EY, VY = 2.0 * Eq - 1.0, 4.0 * Vq
    Ev = EX * EY
    Vv = EX**2 * VY + EY**2 * VX + VX * VY
    mu01  = (Ev + 1.0) / 2.0
    var01 = np.clip(Vv / 4.0 * lam, 1e-9, None)
    var01 = np.minimum(var01, mu01 * (1.0 - mu01) * 0.999)
    conc  = np.maximum(mu01 * (1.0 - mu01) / var01 - 1.0, 2.0 * ALPHA_FLOOR)
    a_beta = np.maximum(mu01 * conc, ALPHA_FLOOR)
    b_beta = np.maximum((1.0 - mu01) * conc, ALPHA_FLOOR)
    return _beta_pmf_rows(a_beta, b_beta), Ed


def _prob_best(E, C):
    logG = np.log(np.clip(C - 0.5 * E, 1e-12, 1.0))
    loo  = np.exp(logG.sum(axis=0)[None, :] - logG)
    w = (E * loo).sum(axis=1)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


class _TNode:
    __slots__ = ('player', 'legal', 'vself', 'dself', 'edge', 'draw', 'pol',
                 'beta', 'visits', 'vloss', 'term', 'children', 'obs')

    def __init__(self, player, legal, v_probs, v_conf, pol, beta, lam=1.0):
        self.player = player
        self.legal  = np.asarray(legal, dtype=np.int32)
        self.obs    = None
        k = len(self.legal)
        vp = np.asarray(v_probs, dtype=np.float64)
        cf = float(v_conf)
        aw = max(cf * vp[0], ALPHA_FLOOR)
        ad = max(cf * vp[1], ALPHA_FLOOR)
        al = max(cf * max(vp[2], 0.0), ALPHA_FLOOR)
        vpmf, dd = _dirichlet_leaf_belief(np.array([aw]), np.array([ad]),
                                          np.array([al]), lam)
        self.vself = vpmf[0]
        self.dself = float(dd[0])
        self.edge  = np.tile(self.vself, (k, 1))
        self.draw  = np.full(k, self.dself)
        self.pol   = np.asarray(pol, dtype=np.float64)
        self.beta  = float(beta)
        self.visits   = np.zeros(k, dtype=np.int64)
        self.vloss    = np.zeros(k, dtype=np.int32)
        self.term     = np.full(k, -1, dtype=np.int8)
        self.children = [None] * k


_TERM_PMF  = np.stack([_SPIKE_WIN, _SPIKE_DRAW, _SPIKE_LOSS])
_TERM_DRAW_VAL = np.array([0.0, 1.0, 0.0])


def _set_term(node, idx, outcome):
    node.term[idx] = outcome
    node.edge[idx] = _TERM_PMF[outcome]
    node.draw[idx] = _TERM_DRAW_VAL[outcome]


def _opened_mask(node):
    m = (node.term >= 0)
    for i, c in enumerate(node.children):
        if c is not None:
            m[i] = True
    return m


def _node_beliefs(node):
    mask = _opened_mask(node)
    if not mask.any():
        return node.vself.copy(), node.dself
    E = node.edge[mask]
    D = node.draw[mask]
    if E.shape[0] == 1:
        return E[0].copy(), float(D[0])
    C = np.cumsum(E, axis=1)
    np.clip(C, 0.0, 1.0, out=C)
    w = _prob_best(E, C)
    mix = w @ E
    mean_draw = float(w @ D)
    s = mix.sum()
    return (mix / s if s > 0 else np.full(_GRID_G, 1.0 / _GRID_G)), mean_draw


def _sample_pmf(pmf, u):
    c = np.cumsum(pmf)
    idx = min(int((c < u).sum()), _GRID_G - 1)
    return _GRID_V[idx]


def _choose_edge(node, rng, exclude=None):
    C = np.cumsum(node.edge, axis=1)
    u = rng.random_sample(len(node.legal))
    gi = np.minimum((C < u[:, None]).sum(axis=1), _GRID_G - 1)
    v = _GRID_V[gi].copy()
    opened = _opened_mask(node)
    if not opened.all():
        v[~opened] -= _FPU_REDUCTION
    N = node.visits.astype(np.float64)
    U = _C_PUCT * node.pol * np.sqrt(N.sum() + 1.0) / (1.0 + N)
    score = v + U - _VLOSS_PEN * node.vloss
    if exclude:
        masked = score.copy()
        for i in exclude:
            if 0 <= i < len(masked):
                masked[i] = -np.inf
        if np.isfinite(masked).any():
            score = masked
    return int(score.argmax())


def _select_leaf(root, root_state, rng, calib=None, exclude=None):
    node  = root
    state = root_state.clone()
    path  = []
    while True:
        ex = None
        if exclude:
            ex = {i for (nid, i) in exclude if nid == id(node)}
        idx = _choose_edge(node, rng, ex)
        node.vloss[idx] += 1
        node.visits[idx] += 1
        path.append((node, idx))
        if node.term[idx] >= 0:
            return path, None, None
        child = node.children[idx]
        if child is None:
            state.apply_action(int(node.legal[idx]))
            if state.is_terminal():
                r = state.returns()[node.player]
                if calib is not None:
                    m0 = float(node.edge[idx] @ _GRID_V)
                    v0 = max(float(node.edge[idx] @ _GRID_V2) - m0 * m0, 0.0)
                    calib.observe((m0 - r) ** 2, v0)
                _set_term(node, idx,
                          _WIN if r > 0 else (_LOSS if r < 0 else _DRAW))
                return path, None, None
            return path, state, (node, idx)
        state.apply_action(int(node.legal[idx]))
        node = child


def _backup(path):
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        if node.term[idx] < 0 and node.children[idx] is not None:
            vpmf, mdraw = _node_beliefs(node.children[idx])
            node.edge[idx] = _flip_pmf(vpmf)
            node.draw[idx] = mdraw


def _edge_innovation(node, idx, m0, v0, calib):
    p = node.edge[idx]
    m1 = float(p @ _GRID_V)
    v1 = max(float(p @ _GRID_V2) - m1 * m1, 0.0)
    calib.observe((m0 - m1) ** 2, v0 + v1)


def _node_solved_outcome(node):
    t = node.term
    if (t == _WIN).any():
        return _WIN
    if (t >= 0).all():
        return _DRAW if (t == _DRAW).any() else _LOSS
    return None


def _solver_sweep(path):
    for k in range(len(path) - 1, -1, -1):
        parent, idx = path[k]
        child = parent.children[idx]
        if child is None:
            break
        out = _node_solved_outcome(child)
        if out is None or parent.term[idx] >= 0:
            break
        _set_term(parent, idx, int(_FLIP_TERM[out]))


def _scan_terminal_children(node, state, calib=None):
    me = node.player
    for i, a in enumerate(node.legal):
        ch = state.clone()
        ch.apply_action(int(a))
        if ch.is_terminal():
            r = ch.returns()[me]
            if calib is not None:
                m0 = float(node.edge[i] @ _GRID_V)
                v0 = max(float(node.edge[i] @ _GRID_V2) - m0 * m0, 0.0)
                calib.observe((m0 - r) ** 2, v0)
            _set_term(node, i, _WIN if r > 0 else (_LOSS if r < 0 else _DRAW))


def _np_softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _node_from_eval(state, v_probs, v_conf, p_logits_legal, p_conf, lam,
                    calib=None):
    legal = state.legal_actions()
    pol   = _np_softmax(np.asarray(p_logits_legal, dtype=np.float64))
    node  = _TNode(state.current_player(), legal, v_probs, float(v_conf),
                   pol, float(p_conf), lam)
    node.obs = np.asarray(state.observation_tensor(state.current_player()),
                          dtype=np.float16)
    if len(legal) <= _FORCING_MAXLEGAL:
        _scan_terminal_children(node, state, calib)
    return node


def _root_value(node):
    vpmf, _ = _node_beliefs(node)
    return float(vpmf @ _GRID_V)


def root_pick(root, rng, thompson):
    """Commit a root move from the search statistics. VISIT-BASED
    (AlphaZero-standard), NOT value-mean-based: the visit count integrates the
    value posterior over every simulation, so it is robust to a single lucky
    Thompson draw on a barely-visited edge — the failure mode that made a
    weak/uncertain value head commit rim/rook moves. The Thompson posterior
    still drives the SEARCH itself; this is only the final commit.
      thompson=True  (temperature): sample a move ∝ its visit count.
      thompson=False (greedy): most-visited; a proven win is played outright;
                     ties break by posterior mean value, then at random."""
    opened = _opened_mask(root)
    if not opened.any():
        return int(root.legal[int(root.pol.argmax())])
    if thompson:
        w = root.visits.astype(np.float64)
        s = w.sum()
        if s <= 0:
            return int(root.legal[int(root.pol.argmax())])
        i = int((np.cumsum(w / s) < rng.random_sample()).sum())
        return int(root.legal[min(i, len(root.legal) - 1)])
    wins = np.nonzero(root.term == _WIN)[0]
    pool = wins if len(wins) else np.nonzero(opened)[0]
    vis = root.visits[pool]
    cand = pool[vis == vis.max()]
    if len(cand) > 1:
        vmean = root.edge[cand] @ _GRID_V
        cand = cand[vmean >= vmean.max() - 1e-9]
        i = int(cand[rng.randint(len(cand))])
    else:
        i = int(cand[0])
    return int(root.legal[i])


def _expand_state_node(net, device, state, lam, calib):
    x = state_to_tensor(state, device)
    with torch.no_grad():
        vp, vc, pl, pc = net(x)
    legal = state.legal_actions()
    return _node_from_eval(state, vp[0].cpu().numpy(), float(vc[0]),
                           pl[0].cpu().numpy()[legal], float(pc[0]), lam, calib)


# ── Action ↔ UCI (from/to squares) via board diffing ──────────────────────────
# OpenSpiel chess has no UCI accessor, so infer each legal move's from/to (and
# promotion) by applying it to a clone and diffing the FEN piece placement.
_FILES = 'abcdefgh'
_ALL_SQ = [f + r for r in '12345678' for f in _FILES]


def _piece_map(fen):
    board = fen.split(' ')[0]
    m = {}
    for ri, row in enumerate(board.split('/')):        # rank 8 … rank 1
        rank = 8 - ri
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                m[_FILES[file] + str(rank)] = ch
                file += 1
    return m


def _action_uci(state, action):
    fen = str(state)
    before = _piece_map(fen)
    stm = fen.split(' ')[1]                              # 'w' | 'b'
    mine = str.isupper if stm == 'w' else str.islower
    ch = state.clone()
    ch.apply_action(int(action))
    after = _piece_map(str(ch))
    frm, to = [], []
    for sq in _ALL_SQ:
        b = before.get(sq, '.')
        a = after.get(sq, '.')
        if b == a:
            continue
        if b != '.' and mine(b) and a == '.':
            frm.append((sq, b))
        if a != '.' and mine(a) and (b == '.' or not mine(b) or a != b):
            to.append((sq, a))
    if len(frm) >= 2 and len(to) >= 2:                  # castling → king move
        kf = [s for s, p in frm if p in 'Kk']
        kt = [s for s, p in to if p in 'Kk']
        if kf and kt:
            return kf[0] + kt[0]
    if not frm or not to:
        return None
    f, fp = frm[0]
    t, tp = to[0]
    uci = f + t
    if fp in 'Pp' and tp not in 'Pp':                   # promotion
        uci += tp.lower()
    return uci


def _legal_uci_map(state):
    out = {}
    for a in state.legal_actions():
        u = _action_uci(state, int(a))
        if u is not None:
            out[u] = int(a)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive searcher (incremental waves, snapshots, stop event)
# ═══════════════════════════════════════════════════════════════════════════════

class Searcher:
    """Incremental hybrid MCTS for one move. run() can be bounded by a sim
    budget or run until a stop event; a callback records analysis snapshots on
    a wall-clock cadence. Same tree machinery as the training worker."""

    def __init__(self, net, state, device, wave=8):
        self.net = net
        self.state = state.clone()
        self.device = device
        self.wave = wave
        self.calib = _Calib()
        self.rng = np.random.RandomState()
        self.root = _expand_state_node(net, device, self.state,
                                       self.calib.lam(), self.calib)
        self.uci_map = {int(a): _action_uci(state, int(a))
                        for a in state.legal_actions()}

    def _wave(self):
        root = self.root
        pending, opening, evals = [], set(), {}
        for _ in range(self.wave):
            if _node_solved_outcome(root) is not None:
                break
            path, st, edge = _select_leaf(root, self.state, self.rng,
                                          self.calib, opening)
            if st is None:
                _backup(path); _solver_sweep(path)
            else:
                node, idx = edge
                opening.add((id(node), idx))
                pending.append((path, edge))
                evals.setdefault((id(node), idx), (node, idx, st))
        pre = {}
        if evals:
            items = list(evals.values())
            x = batch_to_tensor([st.observation_tensor(st.current_player())
                                 for _, _, st in items], self.device)
            with torch.no_grad():
                vp, vc, pl, pc = self.net(x)
            vp = vp.cpu().numpy(); vc = vc.cpu().numpy()
            pl = pl.cpu().numpy();  pc = pc.cpu().numpy()
            lam = self.calib.lam()
            for j, (node, idx, st) in enumerate(items):
                legal = st.legal_actions()
                m0 = float(node.edge[idx] @ _GRID_V)
                v0 = max(float(node.edge[idx] @ _GRID_V2) - m0 * m0, 0.0)
                pre[(id(node), idx)] = (node, idx, m0, v0)
                node.children[idx] = _node_from_eval(
                    st, vp[j], float(vc[j]), pl[j][legal], float(pc[j]),
                    lam, self.calib)
        for path, (node, idx) in pending:
            _backup(path)
        for path, (node, idx) in pending:
            _solver_sweep(path)
        for node, idx, m0, v0 in pre.values():
            _edge_innovation(node, idx, m0, v0, self.calib)

    def run(self, max_sims=None, stop_evt=None, snap_cb=None, snap_secs=2.0):
        last = time.time()
        while True:
            if stop_evt is not None and stop_evt.is_set():
                break
            if _node_solved_outcome(self.root) is not None:
                break
            n = int(self.root.visits.sum())
            if max_sims is not None and n >= max_sims:
                break
            if n >= 2_000_000:
                break
            self._wave()
            if snap_cb is not None and time.time() - last >= snap_secs:
                snap_cb(self)
                last = time.time()

    def snapshot(self):
        root = self.root
        probs = {}
        solved = _node_solved_outcome(root)
        if solved == _WIN:
            wins = np.nonzero(root.term == _WIN)[0]
            share = round(1.0 / len(wins), 4)
            for i in wins:
                u = self.uci_map.get(int(root.legal[i]))
                if u:
                    probs[u] = share
        else:
            vis = root.visits.astype(float)
            tot = vis.sum()
            if tot > 0:
                for i, a in enumerate(root.legal):
                    if vis[i] > 0:
                        u = self.uci_map.get(int(a))
                        if u:
                            probs[u] = round(float(vis[i] / tot), 4)
        return {'sims': int(root.visits.sum()),
                'value': round(_root_value(root), 3), 'probs': probs}

    def n_sims(self):
        return int(self.root.visits.sum())

    def best(self):
        # Most-visited root move (see root_pick): robust to value-mean noise,
        # and exactly what the UI shows as preferences — the engine plays what
        # it displays.
        return root_pick(self.root, self.rng, thompson=False)


# ═══════════════════════════════════════════════════════════════════════════════
# AlphaZero engine (standard PUCT MCTS + MCTS-Solver) — inlined from the control
# notebook. Names are `_az_`-prefixed so both engines coexist in one process.
# ═══════════════════════════════════════════════════════════════════════════════

_AZ_C_PUCT   = 1.5
_AZ_FPU      = 0.0
_TERM_VALUE  = np.array([1.0, 0.0, -1.0])          # value of a proven outcome


def _az_softmax_legal(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class _AZNode:
    __slots__ = ('player', 'legal', 'P', 'N', 'W', 'vloss', 'term', 'children',
                 'obs')

    def __init__(self, player, legal, priors):
        self.player   = player
        self.legal    = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        self.P        = np.asarray(priors, dtype=np.float64)
        self.N        = np.zeros(k, dtype=np.int64)
        self.W        = np.zeros(k, dtype=np.float64)
        self.vloss    = np.zeros(k, dtype=np.int64)
        self.term     = np.full(k, -1, dtype=np.int8)
        self.children = [None] * k
        self.obs      = None


def _az_edge_scores(node):
    N, W, vl, t = node.N, node.W, node.vloss, node.term
    ne = N + vl
    Q = np.where(ne > 0, W / np.maximum(ne, 1), _AZ_FPU)
    Q = Q - vl * 1.0 / np.maximum(ne, 1)
    if (t >= 0).any():
        Q = Q.copy()
        Q[t >= 0] = _TERM_VALUE[t[t >= 0]]
    sqrt_sum = math.sqrt(max(1, int(ne.sum())))
    U = _AZ_C_PUCT * node.P * sqrt_sum / (1.0 + ne)
    return Q + U


def _az_select_leaf(root, root_state):
    node  = root
    state = root_state.clone()
    path  = []
    while True:
        idx = int(_az_edge_scores(node).argmax())
        node.vloss[idx] += 1
        path.append((node, idx))
        if node.term[idx] >= 0:
            return path, None, None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            r = state.returns()[node.player]
            node.term[idx] = _WIN if r > 0 else (_LOSS if r < 0 else _DRAW)
            return path, None, None
        child = node.children[idx]
        if child is None:
            return path, state, (node, idx)
        node = child


def _az_backup(path, leaf_value, leaf_player):
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        node.N[idx] += 1
        node.W[idx] += leaf_value if node.player == leaf_player else -leaf_value


def _az_node_solved(node):
    t = node.term
    if (t == _WIN).any():
        return _WIN
    if (t >= 0).all():
        return _DRAW if (t == _DRAW).any() else _LOSS
    return None


def _az_propagate_solved(path):
    for k in range(len(path) - 1, 0, -1):
        node = path[k][0]
        out = _az_node_solved(node)
        if out is None:
            break
        parent, pidx = path[k - 1]
        if parent.term[pidx] >= 0:
            break
        parent.term[pidx] = _FLIP_TERM[out]


def _az_backup_terminal(path, leaf_value, leaf_player):
    _az_backup(path, leaf_value, leaf_player)
    _az_propagate_solved(path)


def _az_solved_adjust_counts(node, counts):
    t = node.term
    if (t == _WIN).any():
        out = np.zeros_like(counts)
        out[t == _WIN] = 1.0
        return out
    if (t == _LOSS).any() and not (t == _LOSS).all():
        out = counts.copy()
        out[t == _LOSS] = 0.0
        if out.sum() > 0:
            return out
    return counts


def _az_root_value(node):
    ne = node.N
    tot = ne.sum()
    return float(node.W[ne > 0].sum() / tot) if tot > 0 else 0.0


def _az_policy_move(net, device, state):
    """Search-free move: argmax of the legal-action policy logits."""
    with torch.no_grad():
        logits, _ = net(state_to_tensor(state, device))
    legal = state.legal_actions()
    p = _az_softmax_legal(logits[0].cpu().numpy()[legal])
    return int(legal[int(p.argmax())])


class AZSearcher:
    """Incremental AlphaZero PUCT MCTS with the same run()/snapshot()/best()
    interface as the hybrid Searcher, so the Session drives either engine
    identically. Batched waves (virtual loss diversifies a wave into one NN
    forward pass); snapshots on a wall-clock cadence."""

    def __init__(self, net, state, device, wave=8):
        self.net = net
        self.state = state.clone()
        self.device = device
        self.wave = wave
        logits, _ = self._eval([self.state])
        legal = state.legal_actions()
        self.root = _AZNode(state.current_player(), legal,
                            _az_softmax_legal(logits[0][legal]))
        self.uci_map = {int(a): _action_uci(state, int(a))
                        for a in legal}

    def _eval(self, states):
        obs = [s.observation_tensor(s.current_player()) for s in states]
        x = batch_to_tensor(obs, self.device)
        with torch.no_grad():
            logits, values = self.net(x)
        return logits.cpu().numpy(), values.cpu().numpy()

    def _wave(self):
        root = self.root
        pending = []
        for _ in range(self.wave):
            if _az_node_solved(root) is not None:
                break
            path, st, edge = _az_select_leaf(root, self.state)
            if st is None:
                node, idx = path[-1]
                _az_backup_terminal(path, float(_TERM_VALUE[node.term[idx]]),
                                    node.player)
            else:
                pending.append((path, st, edge))
        unique = {}
        for path, st, (node, idx) in pending:
            unique.setdefault((id(node), idx), (node, idx, st))
        if unique:
            entries = list(unique.values())
            lg, vl = self._eval([st for _, _, st in entries])
            made = {}
            for (node, idx, st), l_row, v_row in zip(entries, lg, vl):
                leg = st.legal_actions()
                node.children[idx] = _AZNode(st.current_player(), leg,
                                             _az_softmax_legal(l_row[leg]))
                made[(id(node), idx)] = float(v_row)
            for path, st, (node, idx) in pending:
                _az_backup(path, made[(id(node), idx)],
                           node.children[idx].player)

    def run(self, max_sims=None, stop_evt=None, snap_cb=None, snap_secs=2.0):
        last = time.time()
        while True:
            if stop_evt is not None and stop_evt.is_set():
                break
            if _az_node_solved(self.root) is not None:
                break
            n = int(self.root.N.sum())
            if max_sims is not None and n >= max_sims:
                break
            if n >= 2_000_000:
                break
            self._wave()
            if snap_cb is not None and time.time() - last >= snap_secs:
                snap_cb(self)
                last = time.time()

    def snapshot(self):
        root = self.root
        probs = {}
        if _az_node_solved(root) == _WIN:
            wins = np.nonzero(root.term == _WIN)[0]
            share = round(1.0 / len(wins), 4)
            for i in wins:
                u = self.uci_map.get(int(root.legal[i]))
                if u:
                    probs[u] = share
        else:
            vis = root.N.astype(float)
            tot = vis.sum()
            if tot > 0:
                for i, a in enumerate(root.legal):
                    if vis[i] > 0:
                        u = self.uci_map.get(int(a))
                        if u:
                            probs[u] = round(float(vis[i] / tot), 4)
        return {'sims': int(root.N.sum()),
                'value': round(_az_root_value(root), 3), 'probs': probs}

    def n_sims(self):
        return int(self.root.N.sum())

    def best(self):
        root = self.root
        counts = _az_solved_adjust_counts(root, root.N.astype(np.float64))
        idx = int(np.argmax(counts + 1e-6 * root.P))
        return int(root.legal[idx])


# Engine dispatch: (net, engine-tag) → the matching incremental searcher.
_SEARCHERS = {'hybrid': Searcher, 'alphazero': AZSearcher}


def make_searcher(net, engine, state, device):
    return _SEARCHERS[engine](net, state, device)


# ═══════════════════════════════════════════════════════════════════════════════
# Front-end
# ═══════════════════════════════════════════════════════════════════════════════

PAGE = r'''<!doctype html>
<meta charset="utf-8">
<title>Chess — play the model</title>
<style>
 body{font-family:system-ui,sans-serif;background:#1c1e22;color:#eee;margin:0;
      display:flex;flex-wrap:wrap;gap:24px;padding:24px;justify-content:center}
 h1{font-size:20px;margin:0 0 12px}
 #setup{background:#26292f;padding:24px;border-radius:12px;max-width:420px}
 #setup label{display:block;margin:10px 0 4px;color:#aab}
 #setup .row{margin-bottom:6px}
 select,input[type=number]{background:#15171a;color:#eee;border:1px solid #444;
      border-radius:6px;padding:6px 8px}
 button{background:#3a6df0;color:#fff;border:0;border-radius:8px;
      padding:8px 14px;cursor:pointer;font-size:14px}
 button:disabled{background:#444;cursor:default}
 #game{display:none;gap:24px;flex-wrap:wrap;justify-content:center}
 #board{display:grid;grid-template-columns:repeat(8,64px);
      grid-template-rows:repeat(8,64px);border:3px solid #15171a;border-radius:6px}
 .sq{width:64px;height:64px;position:relative;display:flex;align-items:center;
      justify-content:center;font-size:46px;line-height:1;cursor:pointer;
      user-select:none}
 .light{background:#b7c0cc}.dark{background:#6d7787}
 .sq.sel{outline:4px solid #ffd479;outline-offset:-4px}
 .sq.hint{outline:3px dashed #46d17a;outline-offset:-3px}
 .sq.tgt::after{content:"";position:absolute;width:20px;height:20px;
      border-radius:50%;background:rgba(58,109,240,.55)}
 .wp{color:#f7f9fc;text-shadow:0 1px 2px #000,0 0 2px #000}
 .bp{color:#1c1e22;text-shadow:0 1px 1px rgba(255,255,255,.25)}
 .ov{position:absolute;inset:0;pointer-events:none;display:flex;
      align-items:flex-start;justify-content:flex-end}
 .ov span{font-size:11px;color:#fff;background:rgba(0,0,0,.6);
      border-radius:4px;padding:0 3px;margin:2px}
 .coord{position:absolute;font-size:10px;color:#2b2f36;opacity:.8}
 .coord.f{bottom:1px;right:3px}.coord.r{top:1px;left:3px}
 #side{max-width:360px;background:#26292f;padding:18px;border-radius:12px}
 #side .stat{margin:6px 0;color:#ccd}
 #status{font-weight:600;color:#ffd479;margin:8px 0}
 #prefsbox{margin-top:12px;border-top:1px solid #3a3d44;padding-top:10px}
 #simslider{width:100%}
 .pill{display:inline-block;background:#15171a;border-radius:6px;
      padding:2px 8px;margin:2px;color:#9fb}
 #movelog{max-height:180px;overflow-y:auto;font-size:12px;color:#889;margin-top:8px}
 .kbtn{background:#2e3138;border:1px solid #555;margin-right:6px}
 .mv{cursor:pointer}.mv:hover{color:#fff}.mv.cur{color:#ffd479;font-weight:600}
 #navlabel{color:#9ab;font-size:12px;margin-left:6px}
</style>

<div id="setup">
  <h1>♟️ Chess — play the model</h1>
  <div class="row"><label>Mode</label>
    <select id="mode">
      <option value="play">Play against the model</option>
      <option value="watch">Watch model vs model</option>
    </select></div>
  <div class="row" id="siderow"><label>Your side</label>
    <select id="human">
      <option value="white">White</option>
      <option value="black">Black</option>
    </select></div>
  <div class="row"><label>AI thinking per move</label>
    <select id="thinkmode">
      <option value="fixed">Fixed number of MCTS simulations</option>
      <option value="manual">Think until I click &ldquo;Make AI move&rdquo;</option>
    </select></div>
  <div class="row" id="simsrow"><label>Simulations per move</label>
    <input type="number" id="sims" value="400" min="1" max="1000000"></div>
  <div class="row" style="margin-top:14px">
    <button onclick="newGame()">Start game</button></div>
</div>

<div id="game">
  <div><div id="board"></div></div>
  <div id="side">
    <div id="status">…</div>
    <div class="stat" id="thinkinfo"></div>
    <button id="commitbtn" style="display:none" onclick="commitAI()">Make AI move now</button>
    <div id="actionbtns" style="margin:6px 0">
      <button class="kbtn" id="takebackbtn" style="display:none" onclick="takeback()">↩ Take back</button>
      <button class="kbtn" id="analyzebtn" style="display:none" onclick="analyze()">🔍 What would the AI do?</button>
      <button class="kbtn" id="stopanalyzebtn" style="display:none" onclick="stopAnalyze()">⏹ Stop analysis</button>
      <button class="kbtn" id="playhintbtn" style="display:none" onclick="playHint()">▶ Play the AI&rsquo;s move</button>
    </div>
    <div id="prefsbox">
      <label><input type="checkbox" id="prefs" checked> Show AI move preferences</label>
      <div id="sliderbox" style="display:none">
        <input type="range" id="simslider" min="0" max="0" value="0">
        <div class="stat" id="sliderlabel"></div>
      </div>
      <div class="stat" id="evalline"></div>
    </div>
    <div class="stat" style="margin-top:10px">
      <button class="kbtn" onclick="nav(-1e9)">⏮</button>
      <button class="kbtn" onclick="nav(-1)">◀</button>
      <button class="kbtn" onclick="nav(1)">▶</button>
      <button class="kbtn" onclick="nav(1e9)">live ⏭</button>
      <span id="navlabel"></span>
    </div>
    <div id="movelog"></div>
    <div style="margin-top:12px"><button class="kbtn" onclick="location.reload()">New game</button></div>
  </div>
</div>

<script>
const GLYPH={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',
             k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
let SID=null, S=null, sel=null, sliderStick=true, viewMove=null, flip=false;

function pieceMap(fen){
  const m={}, rows=fen.split(' ')[0].split('/');
  for(let ri=0;ri<8;ri++){ let rank=8-ri, file=0;
    for(const ch of rows[ri]){
      if(ch>='1'&&ch<='8') file+=(+ch);
      else { m['abcdefgh'[file]+rank]=ch; file++; } } }
  return m;
}
function sqName(r,c){                     // r,c are on-screen row/col (0 top-left)
  const file = flip ? 7-c : c;
  const rank = flip ? r+1 : 8-r;
  return 'abcdefgh'[file]+rank;
}

function nav(d){
  if(!S || !S.fen_hist) return;
  const last=S.fen_hist.length-1;
  let cur=viewMove===null?last:viewMove;
  cur=Math.max(0,Math.min(last,cur+d));
  viewMove=(cur>=last)?null:cur;
  render();
}
function viewAt(i){
  if(!S||!S.fen_hist) return;
  viewMove=(i>=S.fen_hist.length-1)?null:i;
  render();
}

document.getElementById('mode').onchange=e=>{
  document.getElementById('siderow').style.display=e.target.value==='play'?'block':'none';
};
document.getElementById('thinkmode').onchange=e=>{
  document.getElementById('simsrow').style.display=e.target.value==='fixed'?'block':'none';
};
document.getElementById('simslider').oninput=e=>{
  sliderStick=(+e.target.value===+e.target.max); render();
};

async function api(path,body){
  const r=await fetch(path,body?{method:'POST',body:JSON.stringify(body)}:{});
  return r.json();
}
async function newGame(){
  const mode=document.getElementById('mode').value;
  const sims=document.getElementById('thinkmode').value==='fixed'
           ? +document.getElementById('sims').value : 0;
  const r=await api('/new',{mode,sims,color:document.getElementById('human').value});
  SID=r.sid;
  document.getElementById('setup').style.display='none';
  document.getElementById('game').style.display='flex';
  poll(); setInterval(poll,1000);
}
async function poll(){ if(!SID)return; S=await api('/state?sid='+SID); render(); }
async function commitAI(){ await api('/ai_commit',{sid:SID}); }
async function takeback(){ viewMove=null; sel=null; const r=await api('/takeback',{sid:SID}); if(!r.ok&&r.error) alert(r.error); poll(); }
async function analyze(){ viewMove=null; await api('/analyze',{sid:SID}); poll(); }
async function stopAnalyze(){ await api('/stop_analyze',{sid:SID}); poll(); }
async function playHint(){ const r=await api('/play_hint',{sid:SID}); if(!r.ok&&r.error) alert(r.error); poll(); }

async function clickSq(r,c){
  if(!S||S.status!=='human_turn'||viewMove!==null) return;
  const sq=sqName(r,c);
  const froms=new Set(Object.keys(S.legal).map(u=>u.slice(0,2)));
  if(sel===null){ if(froms.has(sq)) sel=sq; render(); return; }
  if(sq===sel){ sel=null; render(); return; }
  // try to complete a move sel -> sq
  const keys=Object.keys(S.legal).filter(u=>u.slice(0,4)===sel+sq);
  if(keys.length){
    let uci=keys[0];
    if(keys.length>1){                       // promotion choice
      let p=(prompt('Promote to? q / r / b / n','q')||'q').toLowerCase();
      const k=keys.find(u=>u.endsWith(p)); uci=k||keys[0];
    }
    sel=null;
    const res=await api('/move',{sid:SID,uci}); if(res.ok) poll(); else render();
    return;
  }
  if(froms.has(sq)) sel=sq; else sel=null;   // reselect or clear
  render();
}

function snapNow(){
  if(!S||!S.snapshots.length) return null;
  const sl=document.getElementById('simslider');
  const idx=sliderStick?S.snapshots.length-1:Math.min(+sl.value,S.snapshots.length-1);
  return {snap:S.snapshots[idx], idx};
}

function render(){
  if(!S) return;
  if(flip!== (S.mode==='play' && S.human!==S.white_player)){
    flip = (S.mode==='play' && S.human!==S.white_player);
  }
  const hist=S.fen_hist||[];
  const last=Math.max(0,hist.length-1);
  const showIdx=viewMove===null?last:Math.min(viewMove,last);
  const live=viewMove===null;
  const fen=hist.length?hist[showIdx]:S.fen;
  const pm=pieceMap(fen);
  const showPrefs=document.getElementById('prefs').checked;
  const liveThink=live && (S.status==='thinking'||S.status==='analyzing');
  // Show preferences when the checkbox is on, OR whenever an analysis is
  // running / a fresh AI suggestion is available for the live position.
  const prefsOn=showPrefs || S.status==='analyzing' || (live && !!S.hint);

  // preference source (live growing snapshots, the AI suggestion, or the saved
  // analysis of the move that was played from the viewed position)
  let sn=null, snInfo='';
  if(prefsOn){
    if(liveThink){ const c=snapNow(); if(c) sn=c.snap; }
    else if(live && S.hint && showIdx===last){
      sn=S.hint; snInfo=`AI suggestion (${S.hint_uci||'?'}) — ${S.hint.sims} sims`;
    } else{
      const a=(S.analysis_hist||[])[showIdx];
      if(a){ sn=a; snInfo=`saved analysis of move ${showIdx} — ${a.sims} sims`; }
      else if(showIdx>0 && showIdx<=(S.move_log||[]).length) snInfo='(that move was yours)';
    }
  }
  const ov={};
  if(sn){ for(const [u,p] of Object.entries(sn.probs)){
      const t=u.slice(2,4); if(!(t in ov)||p>ov[t]) ov[t]=p; } }
  // Highlight the AI's suggested move (from+to) once analysis is done.
  const hintSq=(live && !liveThink && S.hint_uci)
      ? [S.hint_uci.slice(0,2), S.hint_uci.slice(2,4)] : [];

  const board=document.getElementById('board'); board.innerHTML='';
  for(let r=0;r<8;r++) for(let c=0;c<8;c++){
    const sq=sqName(r,c);
    const d=document.createElement('div');
    d.className='sq '+(((r+c)&1)?'dark':'light');
    if(sel===sq) d.classList.add('sel');
    if(hintSq.includes(sq)) d.classList.add('hint');
    if(sel && Object.keys(S.legal).some(u=>u.slice(0,4)===sel+sq)) d.classList.add('tgt');
    const pc=pm[sq];
    if(pc){ const sp=document.createElement('span');
      sp.className=(pc===pc.toUpperCase())?'wp':'bp';
      sp.textContent=GLYPH[pc]; d.appendChild(sp); }
    if(ov[sq]!==undefined){ const o=document.createElement('div'); o.className='ov';
      o.innerHTML=`<span>${(ov[sq]*100).toFixed(0)}%</span>`;
      o.style.background=`rgba(58,109,240,${Math.min(.5,ov[sq]*1.2)})`;
      d.appendChild(o); }
    if(c===(flip?7:0)){ const rr=document.createElement('div');
      rr.className='coord r'; rr.textContent=sq[1]; d.appendChild(rr); }
    if(r===(flip?0:7)){ const ff=document.createElement('div');
      ff.className='coord f'; ff.textContent=sq[0]; d.appendChild(ff); }
    d.onclick=()=>clickSq(r,c);
    board.appendChild(d);
  }

  const st=document.getElementById('status');
  if(S.terminal){
    const w=S.returns[S.white_player];
    st.textContent = w>0?'🏆 White wins (checkmate)':w<0?'🏆 Black wins (checkmate)':'½–½ Draw';
  } else if(S.status==='human_turn'){
    st.textContent = S.hint ? 'Your move — AI suggestion shown' : 'Your move';
  } else if(S.status==='analyzing'){
    const eng=(S.current_player===S.white_player?S.white_engine:S.black_engine)||'';
    st.textContent=`Analyzing your position… (${eng})`;
  } else if(S.status==='thinking'){
    const eng=(S.current_player===S.white_player?S.white_engine:S.black_engine)||'';
    const who=S.current_player===S.white_player?'White':'Black';
    st.textContent=(S.mode==='watch'?`${who} (${eng}) thinking…`
                                    :`AI thinking… (${eng})`);
  } else st.textContent=S.status;

  document.getElementById('thinkinfo').textContent=
    (S.status==='thinking'||S.status==='analyzing')?`simulations so far: ${S.thinking_sims}`:'';
  document.getElementById('commitbtn').style.display=
    (S.status==='thinking' && S.manual)?'inline-block':'none';
  const showBtn=(id,on)=>document.getElementById(id).style.display=on?'inline-block':'none';
  showBtn('takebackbtn', live && !S.terminal && S.can_takeback);
  showBtn('analyzebtn', live && !S.terminal && S.status==='human_turn' && !S.analyzing);
  showBtn('stopanalyzebtn', S.status==='analyzing');
  showBtn('playhintbtn', live && !S.terminal && S.status==='human_turn' && !!S.hint_uci);

  const box=document.getElementById('sliderbox'), ev=document.getElementById('evalline');
  if(liveThink && prefsOn && S.snapshots.length>0){
    box.style.display='block';
    const sl=document.getElementById('simslider'); sl.max=S.snapshots.length-1;
    if(sliderStick) sl.value=sl.max;
    const cur=snapNow();
    document.getElementById('sliderlabel').textContent=
      `preferences after ${cur.snap.sims} simulations (snapshot ${(+cur.idx)+1}/${S.snapshots.length})`;
    ev.textContent=`AI eval (side to move): ${cur.snap.value>0?'+':''}${cur.snap.value}`;
  } else {
    box.style.display='none';
    ev.textContent = sn && !liveThink
      ? `${snInfo} · eval (side to move): ${sn.value>0?'+':''}${sn.value}` : snInfo;
  }

  document.getElementById('navlabel').textContent=live
    ? `live (after move ${last})`
    : `viewing after move ${showIdx} of ${last} — “live ⏭” to return`;
  document.getElementById('movelog').innerHTML=(S.move_log||[]).map((m,i)=>
    `<span class="mv ${(!live && showIdx===i+1)?'cur':''}" onclick="viewAt(${i+1})">`+
    `${(i%2===0)?(i/2+1)+'.':''} ${m}</span>`).join(' ');
}
</script>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Sessions + HTTP
# ═══════════════════════════════════════════════════════════════════════════════

class Session:
    def __init__(self, mode, human, sims, nets, engines, device, snap_secs):
        self.mode = mode                    # 'play' | 'watch'
        self.human = human                  # player id in play mode, None in watch
        self.sims = sims                    # >0 fixed budget, 0 = manual/indefinite
        self.nets = nets
        self.engines = engines              # player id -> 'hybrid' | 'alphazero'
        self.device = device
        self.snap_secs = snap_secs
        self.state = GAME.new_initial_state()
        self.snapshots = []
        self.move_log = []
        self.analysis_hist = [None]         # per position: AI snapshot or None
        self.fen_hist = []
        self.actions = []                   # applied action ints (for take-back)
        self.movers = []                    # player who made each ply
        self._snap_board()
        self.status = 'init'
        self.searcher = None
        self.stop_evt = threading.Event()   # interrupt search -> AI commits best
        self.abort_evt = threading.Event()  # interrupt search -> DON'T commit
        self.analyze_stop = threading.Event()
        self.analyzing = False
        self.hint = None                    # analysis of the current human position
        self.hint_action = None
        self.lock = threading.RLock()
        self.thread = None
        self.kick()

    def _snap_board(self):
        self.fen_hist.append(str(self.state))

    def engine_to_move(self):
        return (not self.state.is_terminal() and
                (self.human is None or self.state.current_player() != self.human))

    def kick(self):
        with self.lock:
            if self.state.is_terminal():
                self.status = 'over'
                return
            if not self.engine_to_move():
                self.status = 'human_turn'
                return
            if self.thread is not None and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._think_loop, daemon=True)
            self.thread.start()

    def _think_loop(self):
        while self.engine_to_move():
            cur = self.state.current_player()
            searcher = make_searcher(self.nets[cur], self.engines[cur],
                                     self.state, self.device)
            with self.lock:
                self.snapshots = []
                self.searcher = searcher
                self.status = 'thinking'
            searcher.run(max_sims=(self.sims or None), stop_evt=self.stop_evt,
                         snap_cb=self._snap, snap_secs=self.snap_secs)
            if self.abort_evt.is_set():          # take-back aborted the search
                with self.lock:
                    self.searcher = None
                return
            self._snap(searcher)
            with self.lock:
                action = searcher.best()
                self.analysis_hist.append(searcher.snapshot())
                self.move_log.append(self.state.action_to_string(cur, action))
                self.state.apply_action(action)
                self.actions.append(int(action)); self.movers.append(cur)
                self._snap_board()
                self.stop_evt.clear()
                self.searcher = None
            if self.mode == 'watch' and self.sims:
                time.sleep(0.3)
        with self.lock:
            self.status = 'over' if self.state.is_terminal() else 'human_turn'

    def _snap(self, searcher):
        with self.lock:
            snap = searcher.snapshot()
            if not self.snapshots or snap['sims'] > self.snapshots[-1]['sims']:
                self.snapshots.append(snap)

    def human_move(self, uci, analysis=None):
        with self.lock:
            if self.status != 'human_turn':
                return False, 'not your turn'
            amap = _legal_uci_map(self.state)
            if uci not in amap:
                return False, 'illegal move'
            action = amap[uci]
            mover = self.state.current_player()
            self.analysis_hist.append(analysis)   # None, or the played hint
            self.move_log.append(self.state.action_to_string(mover, action))
            self.state.apply_action(action)
            self.actions.append(int(action)); self.movers.append(mover)
            self._snap_board()
            self.hint = None; self.hint_action = None
            self.snapshots = []
        self.kick()
        return True, ''

    def commit_ai(self):
        self.stop_evt.set()

    # ── Take-back ────────────────────────────────────────────────────────────
    def take_back(self):
        """Undo back to the human's previous decision point (their last move +
        any AI reply). Works whether it's the human's turn or the AI is still
        thinking (the search is aborted WITHOUT committing)."""
        if self.human is None:
            return False, 'take-back is only available in play mode'
        with self.lock:
            idxs = [i for i, m in enumerate(self.movers) if m == self.human]
            if not idxs:
                return False, 'no moves to take back'
        self.abort_evt.set(); self.stop_evt.set(); self.analyze_stop.set()
        th = self.thread
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
        with self.lock:
            i = [j for j, m in enumerate(self.movers) if m == self.human][-1]
            del self.actions[i:]; del self.movers[i:]
            del self.move_log[i:]; del self.analysis_hist[i + 1:]  # [0]=start
            del self.fen_hist[i + 1:]
            st = GAME.new_initial_state()
            for a in self.actions:
                st.apply_action(int(a))
            self.state = st
            self.snapshots = []; self.searcher = None
            self.hint = None; self.hint_action = None; self.analyzing = False
            self.abort_evt.clear(); self.stop_evt.clear(); self.analyze_stop.clear()
            self.status = 'human_turn'
        return True, ''

    # ── Analyze the current (human) position without committing a move ────────
    def analyze(self):
        with self.lock:
            if self.status != 'human_turn' or self.state.is_terminal():
                return False, 'can only analyze on your turn'
            if self.analyzing:
                return True, ''
            self.analyzing = True
            self.analyze_stop.clear()
            self.snapshots = []
            self.hint = None; self.hint_action = None
            self.status = 'analyzing'
            cur = self.state.current_player()
            net, eng, st = self.nets[cur], self.engines[cur], self.state.clone()
        self.thread = threading.Thread(target=self._analyze_loop,
                                       args=(net, eng, st), daemon=True)
        self.thread.start()
        return True, ''

    def _analyze_loop(self, net, eng, st):
        searcher = make_searcher(net, eng, st, self.device)
        with self.lock:
            self.searcher = searcher
        # Manual (sims==0) analysis runs until the human stops it; else a fixed
        # budget (a bit deeper than a normal move so the hint is worth showing).
        budget = self.sims if self.sims else None
        searcher.run(max_sims=budget, stop_evt=self.analyze_stop,
                     snap_cb=self._snap, snap_secs=self.snap_secs)
        self._snap(searcher)
        with self.lock:
            snap = searcher.snapshot()
            self.hint = snap
            self.hint_action = searcher.best()
            self.searcher = None
            self.analyzing = False
            if self.status == 'analyzing':
                self.status = 'human_turn'

    def stop_analyze(self):
        self.analyze_stop.set()
        return True, ''

    def play_hint(self):
        """Commit the AI's suggested move (from the last analysis) as your own."""
        with self.lock:
            if self.hint_action is None or self.status != 'human_turn':
                return False, 'no suggestion to play'
            uci = self.uci_of(self.hint_action)
            analysis = self.hint
        if uci is None:
            return False, 'suggestion unavailable'
        return self.human_move(uci, analysis=analysis)

    def uci_of(self, action):
        return _action_uci(self.state, int(action))

    def to_json(self):
        with self.lock:
            st = self.state
            human_turn = self.status == 'human_turn'
            legal = _legal_uci_map(st) if (human_turn and not st.is_terminal()) else {}
            return {
                'fen': str(st),
                'legal': legal,
                'status': self.status,
                'mode': self.mode,
                'human': self.human,
                'white_player': WHITE_PLAYER,
                'current_player': int(st.current_player()) if not st.is_terminal() else -1,
                'white_engine': self.engines.get(WHITE_PLAYER, ''),
                'black_engine': self.engines.get(1 - WHITE_PLAYER, ''),
                'manual': self.sims == 0,
                'thinking_sims': self.searcher.n_sims() if self.searcher else 0,
                'snapshots': self.snapshots,
                'move_log': self.move_log,
                'fen_hist': self.fen_hist,
                'analysis_hist': self.analysis_hist,
                'analyzing': self.analyzing,
                'can_takeback': self.human is not None and self.human in self.movers,
                'hint': self.hint,
                'hint_uci': (_action_uci(st, int(self.hint_action))
                             if self.hint_action is not None and not st.is_terminal()
                             else None),
                'terminal': st.is_terminal(),
                'returns': st.returns() if st.is_terminal() else None,
            }


SESSIONS = {}
NETS = {}
ENGINES = {}            # player id -> 'hybrid' | 'alphazero'
DEVICE_ARG = 'cpu'
SNAP_SECS = 2.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = _json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == '/state':
            sid = parse_qs(u.query).get('sid', [''])[0]
            s = SESSIONS.get(sid)
            self._json(s.to_json() if s else {'error': 'no such session'},
                       200 if s else 404)
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = _json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            self._json({'error': 'bad json'}, 400)
            return
        if self.path == '/new':
            mode = req.get('mode', 'play')
            if mode == 'watch':
                human = None
            else:
                color = req.get('color', 'white')
                human = WHITE_PLAYER if color == 'white' else 1 - WHITE_PLAYER
            sims = max(0, int(req.get('sims', 400)))
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = Session(mode, human, sims, NETS, ENGINES,
                                    DEVICE_ARG, SNAP_SECS)
            self._json({'sid': sid})
        elif self.path == '/move':
            s = SESSIONS.get(req.get('sid', ''))
            if s is None:
                self._json({'error': 'no such session'}, 404)
                return
            ok, msg = s.human_move(str(req.get('uci', '')))
            self._json({'ok': ok, 'error': msg})
        elif self.path in ('/ai_commit', '/takeback', '/analyze',
                           '/stop_analyze', '/play_hint'):
            s = SESSIONS.get(req.get('sid', ''))
            if s is None:
                self._json({'error': 'no such session'}, 404)
                return
            if self.path == '/ai_commit':
                s.commit_ai(); self._json({'ok': True})
            elif self.path == '/takeback':
                ok, msg = s.take_back(); self._json({'ok': ok, 'error': msg})
            elif self.path == '/analyze':
                ok, msg = s.analyze(); self._json({'ok': ok, 'error': msg})
            elif self.path == '/stop_analyze':
                ok, msg = s.stop_analyze(); self._json({'ok': ok, 'error': msg})
            else:  # /play_hint
                ok, msg = s.play_hint(); self._json({'ok': ok, 'error': msg})
        else:
            self._json({'error': 'not found'}, 404)


def load_model(path, device):
    """Load a checkpoint and AUTO-DETECT the engine from its head weights:
    `policy_out.*` ⇒ AlphaZero, `v_dist.*` ⇒ ThompsonHybrid. Returns
    (net, engine-tag). channels/blocks are inferred from the trunk."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']                      # latest.pt full checkpoint
    channels = sd['stem.0.weight'].shape[0]
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    if 'policy_out.weight' in sd:
        engine = 'alphazero'
        net = AlphaZeroChessNet(channels, blocks).to(device)
    elif 'v_dist.weight' in sd:
        engine = 'hybrid'
        net = ThompsonHybridChessNet(channels, blocks).to(device)
    else:
        raise ValueError(f'{path}: unrecognized checkpoint (no policy_out/v_dist '
                         f'head — not an AlphaZero or ThompsonHybrid chess net)')
    net.load_state_dict(sd)
    net.eval()
    print(f'Loaded {path}: {engine} engine, {channels} channels x {blocks} '
          f'blocks ({sum(p.numel() for p in net.parameters()):,} params)')
    return net, engine


def main():
    global DEVICE_ARG, SNAP_SECS
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--model2', default=None)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--snapshot-secs', type=float, default=2.0)
    args = ap.parse_args()
    DEVICE_ARG = args.device
    SNAP_SECS = args.snapshot_secs
    NETS[WHITE_PLAYER], ENGINES[WHITE_PLAYER] = load_model(args.model, args.device)
    if args.model2:
        NETS[1 - WHITE_PLAYER], ENGINES[1 - WHITE_PLAYER] = load_model(
            args.model2, args.device)
    else:
        NETS[1 - WHITE_PLAYER] = NETS[WHITE_PLAYER]
        ENGINES[1 - WHITE_PLAYER] = ENGINES[WHITE_PLAYER]
    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'Serving on http://localhost:{args.port}')
    print(f'To share:  ngrok http {args.port}')
    srv.serve_forever()


if __name__ == '__main__':
    main()
