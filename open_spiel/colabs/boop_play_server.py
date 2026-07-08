#!/usr/bin/env python3
"""Boop live-play web server — play against a checkpointed model, or watch two
models play each other, in a browser (works through ngrok).

Usage:
    python boop_play_server.py --model boop_checkpoints_gpu/bench_7000.pt
    python boop_play_server.py --model A.pt --model2 B.pt      # watch A vs B
    ngrok http 8765                                             # then share URL

Flags:
    --model  PATH   checkpoint for the engine (bench_*.pt or latest.pt)
    --model2 PATH   second engine for watch mode (default: same as --model)
    --port   N      HTTP port (default 8765)
    --device D      inference device (default cpu — batch-1..8 is CPU's regime)
    --snapshot-secs S   seconds between analysis snapshots while thinking (5)

Network architecture (channels/blocks) is inferred from the checkpoint.
AUTO-GENERATED from boop_kataboop_training.ipynb (engine + network cells).
"""
import argparse
import json as _json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Boop board game -- inlined so Colab works without the custom package.
# Faithful rules: graduation conserves pieces (kittens BECOME cats), and a
# player whose pool is empty must graduate a kitten of their choice rather than
# being stuck (which previously caused spurious draws).
# Perf: win checks, promotion, legal moves and the observer are vectorized over
# a precomputed 3-in-a-row line table, and clone() is a hand-rolled field copy
# instead of deepcopy — MCTS clones a state once per simulation, so these are
# the hottest paths in the whole pipeline.

import numpy as np
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_ROWS = 6
_COLS = 6
_NUM_CELLS = _ROWS * _COLS
_NUM_PIECE_TYPES = 2
_NUM_PIECES = 8                              # each player has exactly 8 pieces
_GRADUATE_OFFSET = _NUM_PIECE_TYPES * _NUM_CELLS  # 72: graduation actions start here
_NUM_ACTIONS = _GRADUATE_OFFSET + _NUM_CELLS      # 108 = 72 placement + 36 graduate
_MAX_KITTENS = 8
_MAX_CATS = 8                               # all 8 pieces can become cats
_MAX_GAME_LENGTH = 500

_EMPTY = 0
_P0_KITTEN = 1
_P0_CAT = 2
_P1_KITTEN = 3
_P1_CAT = 4

_KITTEN_VAL = [_P0_KITTEN, _P1_KITTEN]
_CAT_VAL = [_P0_CAT, _P1_CAT]
_PIECE_VALS = [[_P0_KITTEN, _P0_CAT], [_P1_KITTEN, _P1_CAT]]

# Every 3-in-a-row line on the board (horizontal, vertical, both diagonals) as
# indices into the flattened 36-cell board: shape (80, 3). Win detection and
# promotion become single vectorized gathers instead of triple Python loops.
_LINE_IDX = np.array(
    [[(r + k * dr) * _COLS + (c + k * dc) for k in range(3)]
     for r in range(_ROWS) for c in range(_COLS)
     for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
     if 0 <= r + 2 * dr < _ROWS and 0 <= c + 2 * dc < _COLS],
    dtype=np.int64)

_GAME_TYPE = pyspiel.GameType(
    short_name='python_boop',
    long_name='Python Boop',
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_ACTIONS,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH)


class BoopGame(pyspiel.Game):
  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    return BoopState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoopObserver(params)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)


class BoopState(pyspiel.State):
  def __init__(self, game):
    super().__init__(game)
    self._game_ref = game
    self._cur_player = 0
    self._is_terminal = False
    self._winner = None
    self._move_count = 0
    self.board = np.zeros((_ROWS, _COLS), dtype=np.int8)
    self._hand = [[_MAX_KITTENS, 0], [_MAX_KITTENS, 0]]

  def clone(self):
    # Fast clone: MCTS calls this once per simulation. Copying the handful of
    # fields directly is ~10x cheaper than the default deepcopy-based clone.
    cp = BoopState(self._game_ref)
    cp._cur_player  = self._cur_player
    cp._is_terminal = self._is_terminal
    cp._winner      = self._winner
    cp._move_count  = self._move_count
    cp.board        = self.board.copy()
    cp._hand        = [self._hand[0][:], self._hand[1][:]]
    return cp

  def current_player(self):
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    # Forced-graduation rule: if the pool is empty, the player must graduate one
    # of their kittens on the board (returning it to the pool as a cat) instead
    # of placing. Graduation actions are _GRADUATE_OFFSET + cell.
    flat = self.board.reshape(-1)
    hk, hc = self._hand[player]
    if hk == 0 and hc == 0:
      kittens = np.flatnonzero(flat == _KITTEN_VAL[player])
      return (_GRADUATE_OFFSET + kittens).tolist()
    empty = np.flatnonzero(flat == _EMPTY)
    if hk > 0 and hc > 0:
      return np.concatenate([empty, _NUM_CELLS + empty]).tolist()
    if hk > 0:
      return empty.tolist()
    return (_NUM_CELLS + empty).tolist()

  def _apply_action(self, action):
    p = self._cur_player
    if action >= _GRADUATE_OFFSET:
      # Forced graduation: a kitten on the board becomes a cat in the pool.
      cell = action - _GRADUATE_OFFSET
      r, c = cell // _COLS, cell % _COLS
      self.board[r, c] = _EMPTY
      self._hand[p][1] += 1
      self._move_count += 1
      self._post_move(p)        # no piece placed → no boop
      return
    piece_type = action // _NUM_CELLS
    cell = action % _NUM_CELLS
    r, c = cell // _COLS, cell % _COLS
    self._hand[p][piece_type] -= 1
    self.board[r, c] = _PIECE_VALS[p][piece_type]
    self._boop(r, c, is_cat=(piece_type == 1))
    self._move_count += 1
    self._post_move(p)

  def _post_move(self, p):
    """Shared post-move resolution: win checks, promotion, turn handoff."""
    if self._move_count >= _MAX_GAME_LENGTH:
      self._is_terminal = True
      return
    for player in (p, 1 - p):
      if self._check_win(player):
        self._is_terminal = True
        self._winner = player
        return
    self._promote_kittens(p)
    self._promote_kittens(1 - p)
    for player in (p, 1 - p):
      if self._check_win(player):
        self._is_terminal = True
        self._winner = player
        return
    self._cur_player = 1 - p
    # With forced graduation a player is never permanently stuck: an empty pool
    # forces a graduation, and a board of eight cats is already a win. The guard
    # below is defensive only and should not trigger in normal play.
    if not self._legal_actions(self._cur_player):
      self._is_terminal = True

  def _action_to_string(self, player, action):
    if action >= _GRADUATE_OFFSET:
      cell = action - _GRADUATE_OFFSET
      r, c = cell // _COLS, cell % _COLS
      return f'p{player}:graduate@({r},{c})'
    pt = action // _NUM_CELLS
    cell = action % _NUM_CELLS
    r, c = cell // _COLS, cell % _COLS
    piece = 'cat' if pt else 'kitten'
    return f'p{player}:{piece}@({r},{c})'

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    if self._winner == 0:
      return [1.0, -1.0]
    if self._winner == 1:
      return [-1.0, 1.0]
    return [0.0, 0.0]

  def __str__(self):
    syms = {
        _EMPTY: '.', _P0_KITTEN: 'k', _P0_CAT: 'K',
        _P1_KITTEN: 'o', _P1_CAT: 'O',
    }
    rows = [
        ''.join(syms[self.board[r, c]] for c in range(_COLS))
        for r in range(_ROWS)
    ]
    rows.append(
        f'P0: {self._hand[0][0]}k {self._hand[0][1]}K  '
        f'P1: {self._hand[1][0]}k {self._hand[1][1]}K  '
        f'move={self._move_count}')
    return '\n'.join(rows)

  def _boop(self, r, c, is_cat):
    board = self.board
    for dr in (-1, 0, 1):
      for dc in (-1, 0, 1):
        if dr == 0 and dc == 0:
          continue
        nr, nc = r + dr, c + dc
        if not (0 <= nr < _ROWS and 0 <= nc < _COLS):
          continue
        neighbor = board[nr, nc]
        if neighbor == _EMPTY:
          continue
        neighbor_is_cat = neighbor == _P0_CAT or neighbor == _P1_CAT
        if not is_cat and neighbor_is_cat:
          continue
        dest_r, dest_c = nr + dr, nc + dc
        owner = 0 if (neighbor == _P0_KITTEN or neighbor == _P0_CAT) else 1
        n_type = 1 if neighbor_is_cat else 0
        if not (0 <= dest_r < _ROWS and 0 <= dest_c < _COLS):
          board[nr, nc] = _EMPTY
          self._hand[owner][n_type] += 1
        elif board[dest_r, dest_c] == _EMPTY:
          board[dest_r, dest_c] = neighbor
          board[nr, nc] = _EMPTY

  def _promote_kittens(self, player):
    # Faithful rule (rulebook p.4): a line of 3 of the player's own pieces —
    # kittens AND/OR cats mixed — graduates. Every kitten in the line becomes
    # a cat; every cat in the line simply returns to the pool; either way all
    # three board cells clear and the pool gains 3 cats (pieces conserved).
    # A pure 3-cats-in-a-row is a WIN, checked before this runs, so it never
    # reaches here as a live case.
    kv, cv = _KITTEN_VAL[player], _CAT_VAL[player]
    flat = self.board.reshape(-1)
    while True:
      mine = (flat == kv) | (flat == cv)
      if int(mine.sum()) < 3:
        return
      full = mine[_LINE_IDX].all(axis=1)
      if not full.any():
        return
      # Resolve ONE qualifying line per pass, chosen UNIFORMLY AT RANDOM
      # among all lines that currently qualify (not the player's choice —
      # that would need a new action type, which isn't worth the added
      # action-space complexity). Clearing the chosen line's cells
      # invalidates any OVERLAPPING line, so it is not picked again this
      # call — matching "choose one, leave the rest" for a connected run of
      # 4+ (fig.4). An independent (non-overlapping) line elsewhere still
      # qualifies and is resolved on the loop's next pass.
      candidates = _LINE_IDX[full]
      line = candidates[np.random.randint(len(candidates))]
      flat[line] = _EMPTY
      self._hand[player][1] += 3

  def _check_win(self, player):
    flat = self.board.reshape(-1)
    cats = flat == _CAT_VAL[player]
    n = int(cats.sum())
    if n >= _NUM_PIECES:        # win condition 1: all eight pieces are cats
      return True
    if n < 3:
      return False
    # Win condition 2: three cats in a row (orthogonal or diagonal).
    return bool(cats[_LINE_IDX].all(axis=1).any())


class BoopObserver:
  def __init__(self, params):
    if params:
      raise ValueError(f'Observation parameters not supported; passed {params}')
    board_size = 5 * _ROWS * _COLS
    self.tensor = np.zeros(board_size + 4, np.float32)
    self.dict = {
        'observation': np.reshape(self.tensor[:board_size], (5, _ROWS, _COLS)),
        'hand': self.tensor[board_size:],
    }

  def set_from(self, state, player):
    self.tensor.fill(0)
    obs = self.dict['observation']
    hand = self.dict['hand']
    b = state.board
    opp = 1 - player
    obs[0][b == _EMPTY] = 1.0
    obs[1][b == _KITTEN_VAL[player]] = 1.0
    obs[2][b == _CAT_VAL[player]] = 1.0
    obs[3][b == _KITTEN_VAL[opp]] = 1.0
    obs[4][b == _CAT_VAL[opp]] = 1.0
    hand[0] = state._hand[player][0] / _MAX_KITTENS
    hand[1] = state._hand[player][1] / _MAX_CATS
    hand[2] = state._hand[opp][0] / _MAX_KITTENS
    hand[3] = state._hand[opp][1] / _MAX_CATS

  def string_from(self, state, player):
    del player
    return str(state)


try:
    pyspiel.register_game(_GAME_TYPE, BoopGame)
except Exception:
    pass


# ── Input helpers ──────────────────────────────────────────────────────────────────────────────

def _obs_to_9ch(obs_np):
    """Flat 184-float obs → (9, 6, 6): 5 board planes + 4 hand scalars broadcast."""
    board = obs_np[..., :180].reshape(*obs_np.shape[:-1], 5, 6, 6)
    hand  = obs_np[..., 180:]
    # broadcast hand scalars spatially so the CNN sees them at every cell
    hand_planes = np.broadcast_to(
        hand[..., None, None], hand.shape + (6, 6)).copy()
    return np.concatenate([board, hand_planes], axis=-3)   # (..., 9, 6, 6)


def state_to_tensor(state, device):
    """Single game state → (1, 9, 6, 6) float tensor."""
    obs = np.array(state.observation_tensor(state.current_player()), dtype=np.float32)
    x   = _obs_to_9ch(obs)[None]        # (1, 9, 6, 6)
    return torch.from_numpy(x).to(device)


def batch_to_tensor(obs_list, device):
    """List of flat 184-float observations → (B, 9, 6, 6) float tensor."""
    obs = np.stack(obs_list).astype(np.float32)   # (B, 184)
    x   = _obs_to_9ch(obs)                         # (B, 9, 6, 6)
    return torch.from_numpy(x).to(device)


# ── Network modules ────────────────────────────────────────────────────────────────────────────

class _GroupNorm(nn.Module):
    """GroupNorm built from elementwise ops (reshape/mean/var/affine) rather than
    torch's fused native_group_norm. Mathematically identical to nn.GroupNorm and
    keeps NO running stats (train == eval), but avoids the fused kernel whose
    backward is broken on DirectML (it raised "NativeBatchNormBackward0 returned
    an invalid gradient" for the 1-channel value-head norm)."""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        n, c = x.shape[0], x.shape[1]
        xg = x.reshape(n, self.num_groups, -1)          # group channels+spatial
        mean = xg.mean(dim=2, keepdim=True)
        var = (xg - mean).pow(2).mean(dim=2, keepdim=True)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        x = xg.reshape(x.shape)
        return x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)


def _norm(channels):
    """Normalizer with no running stats (train == eval) so the value head can't be
    miscalibrated between self-play (eval) and training the way BatchNorm was.
    Uses the hand-rolled GroupNorm above to stay DirectML-safe. `groups` divides
    `channels`."""
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return _GroupNorm(groups, channels)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (KataGo-style)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 2),
        )

    def forward(self, x):
        s = x.mean(dim=(2, 3))             # global avg pool → (B, C)
        scale, bias = self.fc(s).chunk(2, dim=1)
        scale = torch.sigmoid(scale)
        return (x * scale[:, :, None, None]
                  + bias[:, :, None, None])


class ResBlock(nn.Module):
    """Residual block: Conv-GN-ReLU-Conv-GN + SE attention + skip."""
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


class BoopNet(nn.Module):
    """
    KataGo-style network for Boop.

    Input  : (B, 9, 6, 6) — 5 board planes + 4 hand scalars broadcast
    Body   : Conv stem → N × ResBlock(channels, SE)
    Policy : 1×1 conv (2 ch) → flatten → Linear(_NUM_ACTIONS=108)
    Value  : 1×1 conv (1 ch) → flatten → Linear(128) → ReLU → Linear(1) → Tanh
    """
    def __init__(self, channels=128, num_blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(9, channels, 3, padding=1, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            _norm(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 6 * 6, _NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            _norm(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(36, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.policy_head(x), self.value_head(x)



PAGE = '<!doctype html>\n<meta charset="utf-8">\n<title>Boop — play the model</title>\n<style>\n body{font-family:system-ui,sans-serif;background:#1c1e22;color:#eee;margin:0;\n      display:flex;flex-wrap:wrap;gap:24px;padding:24px;justify-content:center}\n h1{font-size:20px;margin:0 0 12px}\n #setup{background:#26292f;padding:24px;border-radius:12px;max-width:420px}\n #setup label{display:block;margin:10px 0 4px;color:#aab}\n #setup .row{margin-bottom:6px}\n select,input[type=number]{background:#15171a;color:#eee;border:1px solid #444;\n      border-radius:6px;padding:6px 8px}\n button{background:#3a6df0;color:#fff;border:0;border-radius:8px;\n      padding:8px 14px;cursor:pointer;font-size:14px}\n button:disabled{background:#444;cursor:default}\n #game{display:none;gap:24px;flex-wrap:wrap;justify-content:center}\n #board{display:grid;grid-template-columns:repeat(6,64px);gap:4px}\n .cell{width:64px;height:64px;background:#2e3138;border-radius:8px;position:relative;\n      display:flex;align-items:center;justify-content:center;font-size:30px;\n      cursor:pointer;user-select:none}\n .cell:hover{outline:2px solid #3a6df0}\n .pc{display:flex;align-items:center;justify-content:center;\n     filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))}\n .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;\n     vertical-align:middle}\n .d0{background:#ff9d2e}.d1{background:#9aa2ad}\n .ov{position:absolute;inset:0;border-radius:8px;pointer-events:none;\n     display:flex;align-items:flex-end;justify-content:flex-end}\n .ov span{font-size:11px;color:#fff;background:rgba(0,0,0,.55);\n     border-radius:4px;padding:0 3px;margin:2px}\n #side{max-width:360px;background:#26292f;padding:18px;border-radius:12px}\n #side .stat{margin:6px 0;color:#ccd}\n #status{font-weight:600;color:#ffd479;margin:8px 0}\n #prefsbox{margin-top:12px;border-top:1px solid #3a3d44;padding-top:10px}\n #simslider{width:100%}\n .pill{display:inline-block;background:#15171a;border-radius:6px;\n      padding:2px 8px;margin:2px;color:#9fb}\n #movelog{max-height:140px;overflow-y:auto;font-size:12px;color:#889;margin-top:8px}\n .kbtn{background:#2e3138;border:1px solid #555;margin-right:6px}\n .kbtn.sel{background:#3a6df0;border-color:#3a6df0}\n .mv{cursor:pointer}\n .mv:hover{color:#fff}\n .mv.cur{color:#ffd479;font-weight:600}\n #navlabel{color:#9ab;font-size:12px;margin-left:6px}\n</style>\n\n<div id="setup">\n  <h1>🐱 Boop — play the model</h1>\n  <div class="row"><label>Mode</label>\n    <select id="mode">\n      <option value="play">Play against the model</option>\n      <option value="watch">Watch model vs model</option>\n    </select></div>\n  <div class="row" id="siderow"><label>Your side</label>\n    <select id="human">\n      <option value="0">First player (orange)</option>\n      <option value="1">Second player (grey)</option>\n    </select></div>\n  <div class="row"><label>AI thinking per move</label>\n    <select id="thinkmode">\n      <option value="fixed">Fixed number of MCTS simulations</option>\n      <option value="manual">Think until I click “Make AI move”</option>\n    </select></div>\n  <div class="row" id="simsrow"><label>Simulations per move</label>\n    <input type="number" id="sims" value="200" min="1" max="100000"></div>\n  <div class="row" style="margin-top:14px">\n    <button onclick="newGame()">Start game</button></div>\n</div>\n\n<div id="game">\n  <div>\n    <div id="board"></div>\n    <div style="margin-top:10px" id="piecepick">\n      <button class="kbtn sel" id="pickk" onclick="pick(\'k\')">Place kitten</button>\n      <button class="kbtn" id="pickc" onclick="pick(\'c\')">Place cat</button>\n    </div>\n  </div>\n  <div id="side">\n    <div id="status">…</div>\n    <div class="stat" id="hands"></div>\n    <div class="stat" id="thinkinfo"></div>\n    <button id="commitbtn" style="display:none" onclick="commitAI()">Make AI move now</button>\n    <div id="prefsbox">\n      <label><input type="checkbox" id="prefs"> Show AI move preferences</label>\n      <div id="sliderbox" style="display:none">\n        <input type="range" id="simslider" min="0" max="0" value="0">\n        <div class="stat" id="sliderlabel"></div>\n      </div>\n      <div class="stat" id="evalline"></div>\n    </div>\n    <div class="stat" style="margin-top:10px">\n      <button class="kbtn" onclick="nav(-1e9)">⏮</button>\n      <button class="kbtn" onclick="nav(-1)">◀</button>\n      <button class="kbtn" onclick="nav(1)">▶</button>\n      <button class="kbtn" onclick="nav(1e9)">live ⏭</button>\n      <span id="navlabel"></span>\n    </div>\n    <div id="movelog"></div>\n    <div style="margin-top:12px"><button class="kbtn" onclick="location.reload()">New game</button></div>\n  </div>\n</div>\n\n<script>\nlet SID=null, S=null, piece=\'k\', sliderStick=true, viewMove=null;\n\nfunction nav(d){\n  if(!S || !S.board_hist) return;\n  const last = S.board_hist.length-1;\n  let cur = viewMove===null ? last : viewMove;\n  cur = Math.max(0, Math.min(last, cur+d));\n  viewMove = (cur >= last) ? null : cur;   // at the end = follow live\n  render();\n}\nfunction viewAt(i){\n  if(!S || !S.board_hist) return;\n  viewMove = (i >= S.board_hist.length-1) ? null : i;\n  render();\n}\n\nfunction pieceSVG(isP0, isCat){\n  // kitten: small head, rounded ears; cat: bigger head, tall ears + whiskers\n  const col = isP0 ? \'#ff9d2e\' : \'#9aa2ad\';\n  const sz  = isCat ? 54 : 38;\n  const earL = isCat ? \'22,40 10,2 46,24\' : \'25,44 19,16 47,29\';\n  const earR = isCat ? \'78,40 90,2 54,24\' : \'75,44 81,16 53,29\';\n  const whisk = isCat ? `\n    <g stroke="#fff" stroke-width="3" opacity=".8">\n      <line x1="7"  y1="60" x2="33" y2="63"/><line x1="7"  y1="73" x2="33" y2="69"/>\n      <line x1="93" y1="60" x2="67" y2="63"/><line x1="93" y1="73" x2="67" y2="69"/>\n    </g>` : \'\';\n  return `<svg viewBox="0 0 100 100" width="${sz}" height="${sz}">\n    <polygon points="${earL}" fill="${col}"/>\n    <polygon points="${earR}" fill="${col}"/>\n    <circle cx="50" cy="62" r="34" fill="${col}"/>\n    <circle cx="38" cy="56" r="4.5" fill="#1c1e22"/>\n    <circle cx="62" cy="56" r="4.5" fill="#1c1e22"/>\n    <polygon points="50,67 45,73 55,73" fill="#1c1e22"/>\n    ${whisk}\n  </svg>`;\n}\n\nfunction pick(p){piece=p; render();}\ndocument.getElementById(\'mode\').onchange = e => {\n  document.getElementById(\'siderow\').style.display =\n    e.target.value===\'play\' ? \'block\':\'none\';\n};\ndocument.getElementById(\'thinkmode\').onchange = e => {\n  document.getElementById(\'simsrow\').style.display =\n    e.target.value===\'fixed\' ? \'block\':\'none\';\n};\ndocument.getElementById(\'simslider\').oninput = e => {\n  sliderStick = (+e.target.value === +e.target.max); render();\n};\n\nasync function api(path, body){\n  const r = await fetch(path, body?{method:\'POST\',body:JSON.stringify(body)}:{});\n  return r.json();\n}\n\nasync function newGame(){\n  const mode = document.getElementById(\'mode\').value;\n  const sims = document.getElementById(\'thinkmode\').value===\'fixed\'\n             ? +document.getElementById(\'sims\').value : 0;\n  const r = await api(\'/new\', {mode, sims,\n      human:+document.getElementById(\'human\').value});\n  SID = r.sid;\n  document.getElementById(\'setup\').style.display=\'none\';\n  document.getElementById(\'game\').style.display=\'flex\';\n  poll();\n  setInterval(poll, 1000);\n}\n\nasync function poll(){\n  if(!SID) return;\n  S = await api(\'/state?sid=\'+SID);\n  render();\n}\n\nasync function commitAI(){ await api(\'/ai_commit\',{sid:SID}); }\n\nasync function clickCell(idx){\n  if(!S || S.status!==\'human_turn\' || viewMove!==null) return;\n  const grads = S.legal.filter(a=>a>=72);\n  let action;\n  if(grads.length){ action = 72+idx; }\n  else{ action = (piece===\'c\'?36:0)+idx; }\n  if(!S.legal.includes(action)){\n    // fall back to whichever piece IS legal on this cell\n    const alt = (piece===\'c\'?0:36)+idx;\n    if(S.legal.includes(alt)) action = alt; else return;\n  }\n  const r = await api(\'/move\',{sid:SID, action});\n  if(r.ok) poll();\n}\n\nfunction snapNow(){\n  if(!S || !S.snapshots.length) return null;\n  const sl = document.getElementById(\'simslider\');\n  const idx = sliderStick ? S.snapshots.length-1\n                          : Math.min(+sl.value, S.snapshots.length-1);\n  return {snap:S.snapshots[idx], idx};\n}\n\nfunction render(){\n  if(!S) return;\n  const board  = document.getElementById(\'board\');\n  const hist = S.board_hist || [];\n  const lastIdx = Math.max(0, hist.length-1);\n  const showIdx = viewMove===null ? lastIdx : Math.min(viewMove, lastIdx);\n  const live = viewMove===null;\n  const shown = hist.length ? hist[showIdx] : {board:S.board, hands:S.hands};\n  const showPrefs = document.getElementById(\'prefs\').checked;\n  const liveThink = live && S.status===\'thinking\';\n  // Preference source: while the AI thinks on the live position, use the\n  // growing snapshot list (slider-scrubbable); on any other position show\n  // the SAVED analysis of the move that was played from that position.\n  let sn=null, snInfo=\'\';\n  if(showPrefs){\n    if(liveThink){\n      const c=snapNow(); if(c) sn=c.snap;\n    } else {\n      const a=(S.analysis_hist||[])[showIdx];\n      if(a){ sn=a; snInfo=`saved analysis of move ${showIdx+1} — ${a.sims} sims`; }\n      else if(showIdx < (S.move_log||[]).length)\n        snInfo=\'(that move was yours — no AI analysis)\';\n    }\n  }\n\n  // per-cell best preference from the selected snapshot\n  const ov = {};\n  if(sn){\n    for(const [a,p] of Object.entries(sn.probs)){\n      const ai=+a, cell = ai>=72 ? ai-72 : ai%36;\n      const kind = ai>=72 ? \'G\' : (ai>=36 ? \'C\' : \'k\');\n      if(!(cell in ov) || p>ov[cell].p) ov[cell]={p, kind};\n    }\n  }\n  board.innerHTML=\'\';\n  for(let i=0;i<36;i++){\n    const v=shown.board[i];\n    const d=document.createElement(\'div\');\n    d.className=\'cell\';\n    if(v){\n      const pc=document.createElement(\'div\');\n      pc.className=\'pc\';\n      pc.innerHTML=pieceSVG(v===1||v===2, v===2||v===4);\n      d.appendChild(pc);\n    }\n    d.onclick=()=>clickCell(i);\n    if(ov[i]){\n      const o=document.createElement(\'div\'); o.className=\'ov\';\n      o.style.background=`rgba(58,109,240,${Math.min(.65,ov[i].p*1.3)})`;\n      o.innerHTML=`<span>${ov[i].kind} ${(ov[i].p*100).toFixed(0)}%</span>`;\n      d.appendChild(o);\n    }\n    board.appendChild(d);\n  }\n\n  const st=document.getElementById(\'status\');\n  if(S.terminal){\n    const r=S.returns;\n    st.textContent = r[0]>0?\'🏆 First player (orange) wins!\'\n                   : r[1]>0?\'🏆 Second player (grey) wins!\':\'Draw\';\n  } else if(S.status===\'human_turn\'){\n    const grads = S.legal.filter(a=>a>=72);\n    st.textContent = grads.length\n      ? \'Your turn — pool empty: click one of your kittens to graduate it\'\n      : \'Your turn\';\n  } else if(S.status===\'thinking\'){\n    st.textContent = (S.mode===\'watch\'?`Model ${S.current_player===0?\'A\':\'B\'} thinking…`\n                                       :\'AI thinking…\');\n  } else st.textContent = S.status;\n\n  document.getElementById(\'hands\').innerHTML =\n    `<span class="pill"><span class="dot d0"></span>pool: ${shown.hands[0][0]} kittens, ${shown.hands[0][1]} cats</span>`+\n    `<span class="pill"><span class="dot d1"></span>pool: ${shown.hands[1][0]} kittens, ${shown.hands[1][1]} cats</span>`;\n  document.getElementById(\'thinkinfo\').textContent =\n    S.status===\'thinking\' ? `simulations so far: ${S.thinking_sims}` : \'\';\n  document.getElementById(\'commitbtn\').style.display =\n    (S.status===\'thinking\' && S.manual) ? \'inline-block\':\'none\';\n\n  // kitten/cat picker enable state on human turns\n  const canK = S.legal.some(a=>a<36), canC = S.legal.some(a=>a>=36&&a<72);\n  document.getElementById(\'pickk\').disabled=!canK;\n  document.getElementById(\'pickc\').disabled=!canC;\n  if(piece===\'k\'&&!canK&&canC) piece=\'c\';\n  if(piece===\'c\'&&!canC&&canK) piece=\'k\';\n  document.getElementById(\'pickk\').classList.toggle(\'sel\',piece===\'k\');\n  document.getElementById(\'pickc\').classList.toggle(\'sel\',piece===\'c\');\n\n  // snapshot slider (live thinking) / saved-analysis line (history)\n  const box=document.getElementById(\'sliderbox\');\n  const ev=document.getElementById(\'evalline\');\n  if(liveThink && showPrefs && S.snapshots.length>0){\n    box.style.display=\'block\';\n    const sl=document.getElementById(\'simslider\');\n    sl.max = S.snapshots.length-1;\n    if(sliderStick) sl.value = sl.max;\n    const cur=snapNow();\n    document.getElementById(\'sliderlabel\').textContent =\n      `preferences after ${cur.snap.sims} simulations `+\n      `(snapshot ${(+cur.idx)+1}/${S.snapshots.length})`;\n    ev.textContent =\n      `AI eval (side to move): ${cur.snap.value>0?\'+\':\'\'}${cur.snap.value}`;\n  } else {\n    box.style.display=\'none\';\n    ev.textContent = sn && !liveThink\n      ? `${snInfo} · eval (side to move): ${sn.value>0?\'+\':\'\'}${sn.value}`\n      : snInfo;\n  }\n\n  document.getElementById(\'navlabel\').textContent = live\n    ? `live (after move ${lastIdx})`\n    : `viewing after move ${showIdx} of ${lastIdx} — “live ⏭” to return`;\n  document.getElementById(\'movelog\').innerHTML =\n    S.move_log.map((m,i)=>\n      `<span class="mv ${(!live && showIdx===i+1)?\'cur\':\'\'}" `+\n      `onclick="viewAt(${i+1})">${i+1}. ${m}</span>`).join(\'<br>\');\n}\n</script>\n'


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS (no Dirichlet noise, no forced playouts — full-strength deterministic play)
# ═══════════════════════════════════════════════════════════════════════════════
import random


class _SNode:
    __slots__ = ('action', 'player', 'prior', 'n', 'w', 'vloss', 'outcome',
                 'children')

    def __init__(self, action, player, prior):
        self.action = action
        self.player = player
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.vloss = 0
        self.outcome = None
        self.children = []


def _puct(ch, parent_n_eff, uct_c):
    if ch.outcome is not None:
        return ch.outcome[ch.player]
    ec = ch.n + ch.vloss
    q = (ch.w - ch.vloss) / ec if ec > 0 else 0.0
    return q + uct_c * ch.prior * math.sqrt(parent_n_eff) / (ec + 1)


def _select_leaf(root, root_state, uct_c):
    path = [root]
    root.vloss += 1
    state = root_state.clone()
    node = root
    while node.children and not state.is_terminal():
        pne = node.n + node.vloss
        best = max(node.children, key=lambda c: _puct(c, pne, uct_c))
        state.apply_action(best.action)
        best.vloss += 1
        path.append(best)
        node = best
    return path, state, node


def _expand(node, cur_player, legal, logits_row):
    lg = logits_row[legal] - logits_row[legal].max()
    pr = np.exp(lg)
    pr /= pr.sum()
    node.children = [_SNode(a, cur_player, float(p)) for a, p in zip(legal, pr)]


def _backup_value(path, leaf_cur, value):
    for node in reversed(path):
        node.vloss -= 1
        node.n += 1
        node.w += value if node.player == leaf_cur else -value


def _backup_terminal(path, returns, max_utility=1.0):
    path[-1].outcome = returns
    solved = True
    for node in reversed(path):
        node.vloss -= 1
        node.n += 1
        node.w += returns[node.player]
        if solved and node.children:
            player = node.children[0].player
            best, all_solved = None, True
            for ch in node.children:
                if ch.outcome is None:
                    all_solved = False
                elif best is None or ch.outcome[player] > best.outcome[player]:
                    best = ch
            if best is not None and (all_solved or
                                     best.outcome[player] == max_utility):
                node.outcome = best.outcome
            else:
                solved = False


class Searcher:
    """Incremental batched-leaf MCTS for one move. run() can be bounded by a
    sim budget or run until a stop event; a callback records analysis
    snapshots on a wall-clock cadence."""

    def __init__(self, net, state, device, uct_c=1.4, wave=8):
        self.net = net
        self.state = state.clone()
        self.device = device
        self.uct_c = uct_c
        self.wave = wave
        self.root = _SNode(None, self.state.current_player(), 1.0)

    def _eval(self, states):
        obs = [s.observation_tensor(s.current_player()) for s in states]
        x = batch_to_tensor(obs, self.device)
        with torch.no_grad():
            logits, values = self.net(x)
        return logits.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def run(self, max_sims=None, stop_evt=None, snap_cb=None, snap_secs=5.0):
        last_snap = time.time()
        while True:
            if stop_evt is not None and stop_evt.is_set():
                break
            if self.root.outcome is not None:
                break
            if max_sims is not None and self.root.n >= max_sims:
                break
            if self.root.n >= 2_000_000:
                break
            wave = 1 if not self.root.children else self.wave
            if max_sims is not None:
                wave = min(wave, max(1, max_sims - self.root.n))
            pending = []
            for _ in range(wave):
                if self.root.outcome is not None:
                    break
                path, st, leaf = _select_leaf(self.root, self.state, self.uct_c)
                pending.append((path, leaf, st))
            to_eval = {}
            for path, leaf, st in pending:
                if not st.is_terminal() and id(leaf) not in to_eval:
                    to_eval[id(leaf)] = (leaf, st)
            results = {}
            if to_eval:
                entries = list(to_eval.values())
                logits, values = self._eval([s for _, s in entries])
                for (leaf, st), lg, val in zip(entries, logits, values):
                    _expand(leaf, st.current_player(), st.legal_actions(), lg)
                    results[id(leaf)] = (st.current_player(), float(val))
            for path, leaf, st in pending:
                if st.is_terminal():
                    _backup_terminal(path, st.returns())
                else:
                    cur, val = results[id(leaf)]
                    _backup_value(path, cur, val)
            if snap_cb is not None and time.time() - last_snap >= snap_secs:
                snap_cb(self)
                last_snap = time.time()

    def snapshot(self):
        kids = [c for c in self.root.children if c.n > 0]
        total = sum(c.n for c in kids)
        probs = {int(c.action): round(c.n / total, 4) for c in kids} if total else {}
        q = self.root.w / self.root.n if self.root.n else 0.0
        return {'sims': int(self.root.n), 'value': round(float(q), 3),
                'probs': probs}

    def best(self):
        return max(self.root.children, key=lambda c: c.n).action


# ═══════════════════════════════════════════════════════════════════════════════
# Sessions
# ═══════════════════════════════════════════════════════════════════════════════

GAME = pyspiel.load_game('python_boop')


class Session:
    def __init__(self, mode, human, sims, nets, device, snap_secs):
        self.mode = mode                    # 'play' | 'watch'
        self.human = human                  # 0/1 in play mode, None in watch
        self.sims = sims                    # >0 fixed budget, 0 = manual/indefinite
        self.nets = nets
        self.device = device
        self.snap_secs = snap_secs
        self.state = GAME.new_initial_state()
        self.snapshots = []
        self.move_log = []
        self.analysis_hist = []            # per move: final AI snapshot or None
        self.board_hist = []               # board after every move (incl. start)
        self._snap_board()
        self.status = 'init'
        self.searcher = None
        self.stop_evt = threading.Event()
        self.lock = threading.RLock()
        self.thread = None
        self.kick()

    def _snap_board(self):
        st = self.state
        self.board_hist.append(
            {'board': [int(v) for v in st.board.reshape(-1)],
             'hands': [list(st._hand[0]), list(st._hand[1])]})

    def engine_to_move(self):
        return (not self.state.is_terminal() and
                (self.human is None or
                 self.state.current_player() != self.human))

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
            searcher = Searcher(self.nets[cur], self.state, self.device)
            with self.lock:
                self.snapshots = []
                self.searcher = searcher
                self.status = 'thinking'
            searcher.run(max_sims=(self.sims or None), stop_evt=self.stop_evt,
                         snap_cb=self._snap, snap_secs=self.snap_secs)
            self._snap(searcher)                      # final snapshot
            with self.lock:
                if searcher.root.children:
                    action = searcher.best()
                    self.analysis_hist.append(searcher.snapshot())
                    self.move_log.append(
                        self.state.action_to_string(cur, action))
                    self.state.apply_action(action)
                    self._snap_board()
                self.stop_evt.clear()
                self.searcher = None
            if self.mode == 'watch' and self.sims:
                time.sleep(0.3)                        # watchable pace
        with self.lock:
            self.status = 'over' if self.state.is_terminal() else 'human_turn'

    def _snap(self, searcher):
        with self.lock:
            snap = searcher.snapshot()
            if not self.snapshots or snap['sims'] > self.snapshots[-1]['sims']:
                self.snapshots.append(snap)

    def human_move(self, action):
        with self.lock:
            if self.status != 'human_turn':
                return False, 'not your turn'
            if action not in self.state.legal_actions():
                return False, 'illegal move'
            self.analysis_hist.append(None)          # human move: no analysis
            self.move_log.append(
                self.state.action_to_string(self.state.current_player(), action))
            self.state.apply_action(action)
            self._snap_board()
        self.kick()
        return True, ''

    def commit_ai(self):
        self.stop_evt.set()

    def to_json(self):
        with self.lock:
            st = self.state
            out = {
                'board': [int(v) for v in st.board.reshape(-1)],
                'hands': st._hand,
                'current_player': int(st.current_player()) if not st.is_terminal() else -1,
                'legal': st.legal_actions() if not st.is_terminal() else [],
                'status': self.status,
                'mode': self.mode,
                'human': self.human,
                'manual': self.sims == 0,
                'thinking_sims': int(self.searcher.root.n) if self.searcher else 0,
                'snapshots': self.snapshots,
                'move_log': self.move_log,
                'board_hist': self.board_hist,
                'analysis_hist': self.analysis_hist,
                'terminal': st.is_terminal(),
                'returns': st.returns() if st.is_terminal() else None,
            }
        return out


SESSIONS = {}
NETS = {}
DEVICE_ARG = 'cpu'
SNAP_SECS = 5.0


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
            if s is None:
                self._json({'error': 'no such session'}, 404)
            else:
                self._json(s.to_json())
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
            human = None if mode == 'watch' else int(req.get('human', 0))
            sims = max(0, int(req.get('sims', 200)))
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = Session(mode, human, sims, NETS, DEVICE_ARG,
                                    SNAP_SECS)
            self._json({'sid': sid})
        elif self.path == '/move':
            s = SESSIONS.get(req.get('sid', ''))
            if s is None:
                self._json({'error': 'no such session'}, 404)
                return
            ok, msg = s.human_move(int(req.get('action', -1)))
            self._json({'ok': ok, 'error': msg})
        elif self.path == '/ai_commit':
            s = SESSIONS.get(req.get('sid', ''))
            if s is None:
                self._json({'error': 'no such session'}, 404)
                return
            s.commit_ai()
            self._json({'ok': True})
        else:
            self._json({'error': 'not found'}, 404)


def load_model(path, device):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']                      # latest.pt full checkpoint
    channels = sd['stem.0.weight'].shape[0]
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    net = BoopNet(channels, blocks).to(device)
    net.load_state_dict(sd)
    net.eval()
    print(f'Loaded {path}: {channels} channels x {blocks} blocks '
          f'({sum(p.numel() for p in net.parameters()):,} params)')
    return net


def main():
    global DEVICE_ARG, SNAP_SECS
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--model2', default=None)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--snapshot-secs', type=float, default=5.0)
    args = ap.parse_args()
    DEVICE_ARG = args.device
    SNAP_SECS = args.snapshot_secs
    NETS[0] = load_model(args.model, args.device)
    NETS[1] = load_model(args.model2, args.device) if args.model2 else NETS[0]
    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'Serving on http://localhost:{args.port}')
    print(f'To share:  ngrok http {args.port}')
    srv.serve_forever()


if __name__ == '__main__':
    main()
