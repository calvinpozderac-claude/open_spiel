# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boss Monster: The Dungeon-Building Card Game, implemented in Python.

A 2-player implementation of Brotherwise Games' *Boss Monster* base set
(https://boardgamegeek.com/boardgame/131835/boss-monster-the-dungeon-building-card-game),
modeling the game's central loop: each player secretly builds a row of
Rooms leading to their Boss, lures the shared pool of Heroes into their
dungeon with the right kind of treasure, and tries to kill 10 Heroes
("Souls") before taking 5 Wounds from Heroes who survive to reach the Boss.

See `open_spiel/python/games/boss_monster_data.py` for the card data and a
detailed note on which rules are verified against reachable primary sources
versus approximated (this environment's network access blocked
boardgamegeek.com, the fan wiki, and every rulebook PDF host it tried).

Simplifications made to keep this a clean, terminating, RL-trainable game:
  * Two players only (the physical game supports up to 4).
  * Room cards do not have an "evolve" back side; a built Room's stats are
    fixed for the rest of the game.
  * Only one Room may be built and one Spell cast per player per round (this
    matches the real "one build per turn" rule; spells in the physical game
    have varied timing windows, simplified here to "once, after building").
  * A hero that is tied between the two dungeons' treasure normally stays
    put (matching the real rule) unless a boss/spell effect breaks ties.
  * Games are capped at `_MAX_ROUNDS` rounds; if nobody has won or lost by
    then, the player with the better Souls-minus-Wounds margin wins (an
    exact tie is a draw). This guarantees termination for tree search /
    self-play, which the physical game (no round limit) does not need.

This is a Python-only game (registered as "python_boss_monster"). It is a
sequential game with chance nodes for all card draws/reveals (so game trees
are well-defined for search), imperfect information (each player's hand is
private), terminal rewards in {-1, 0, 1}, and a full `observation_tensor`
suitable for AlphaZero-style training (see
`open_spiel/python/examples/boss_monster_alpha_zero.py`).
"""

import collections

import numpy as np

from open_spiel.python.games import boss_monster_data as data
import pyspiel

_NUM_PLAYERS = 2
_NUM_CLASSES = data.NUM_CLASSES
_BASE_MAX_ROOMS = 5
_HAND_LIMIT = 5
_HAND_MAX = 10  # Defensive hard cap on hand size (never reached in practice).
_WOUND_LIMIT = 5
_SOUL_TARGET = 10
_EPIC_THRESHOLD_SOULS = 2
_MAX_ROUNDS = 40
_STARTING_HAND = _HAND_LIMIT

_ACTION_PASS = 0
_NUM_DISTINCT_ACTIONS = _HAND_MAX + 1  # PASS, plus one action per hand slot.

_GAME_TYPE = pyspiel.GameType(
    short_name="python_boss_monster",
    long_name="Python Boss Monster",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification={"boss0": 0, "boss1": 1})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=data.NUM_BUILD_CARDS,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=1200)

# Card entries carried around in hands / discard / dungeons are simple
# ("room" or "spell" or "hero", template_id) tuples; see boss_monster_data.


class BossMonsterGame(pyspiel.Game):
  """A Python implementation of (2-player) Boss Monster."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
    params = params or {}
    self.boss_ids = [int(params.get("boss0", 0)) % data.NUM_BOSSES,
                      int(params.get("boss1", 1)) % data.NUM_BOSSES]

  def new_initial_state(self):
    return BossMonsterState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return BossMonsterObserver(params)


def _boss_ability(boss_id):
  return data.BOSSES[boss_id].ability


class BossMonsterState(pyspiel.State):
  """A python version of the Boss Monster state."""

  def __init__(self, game):
    super().__init__(game)
    self._boss = list(game.boss_ids)
    self._max_rooms = [
        _BASE_MAX_ROOMS + (1 if _boss_ability(b) == "extra_room_slot" else 0)
        for b in self._boss
    ]

    self._deck = list(data.BUILD_DECK)  # list of ("room"|"spell", tmpl_id)
    self._build_discard = []  # list of ("room"|"spell", template_id)

    self._hero_pool = list(data.ORDINARY_HERO_DECK)
    self._epic_pool = list(data.EPIC_HERO_DECK)
    self._epic_merged = False
    self._hero_discard = []  # heroes that dealt a wound; recyclable.
    self._waiting = []  # hero template ids revealed but not yet sent.

    self._hands = [[], []]
    self._rooms = [[], []]  # list of room template_ids, build order.
    self._wounds = [0, 0]
    self._souls = [0, 0]
    self._round_effects = [set(), set()]  # this-round spell effects.
    self._last_spell_caster = None

    self._round = 1
    self._is_terminal = False
    self._returns = [0.0, 0.0]
    self._history_strings = []

    # Queue of upcoming decision points. Each entry is a tuple whose first
    # element names the kind of step: "draw", "reveal", "build", "spell",
    # "resolve", "discard", "new_round". "resolve"/"new_round" are automatic
    # (no chance/player action); everything else requires one _apply_action.
    self._queue = collections.deque()
    for p in range(_NUM_PLAYERS):
      count = _STARTING_HAND + (
          1 if _boss_ability(self._boss[p]) == "extra_start_hand" else 0)
      for _ in range(count):
        self._queue.append(("draw", p))
    self._queue.append(("new_round",))

  # -- Small helpers ---------------------------------------------------

  def _hand_limit(self, p):
    limit = _HAND_LIMIT
    if _boss_ability(self._boss[p]) == "hand_limit_plus1":
      limit += 1
    if "panic_limited" in self._round_effects[p]:
      limit -= 1
    return max(1, limit)

  def _wound_limit(self, p):
    return _WOUND_LIMIT + (
        1 if _boss_ability(self._boss[p]) == "wound_limit_plus1" else 0)

  def _card_of(self, kind, template_id):
    if kind == "room":
      return data.ROOM_TEMPLATES[template_id]
    else:
      return data.SPELL_TEMPLATES[template_id]

  # -- Queue-driven turn structure --------------------------------------

  def _push_front(self, step):
    self._queue.appendleft(step)

  def _advance(self):
    """Auto-processes any non-decision steps at the front of the queue."""
    while not self._is_terminal:
      if not self._queue:
        self._start_or_end_round()
        continue
      step = self._queue[0]
      if step[0] == "resolve":
        self._queue.popleft()
        self._resolve_round()
        continue
      if step[0] == "new_round":
        self._queue.popleft()
        self._start_or_end_round()
        continue
      # A real decision (draw/reveal/build/spell/discard) sits at the front.
      return

  def _start_or_end_round(self):
    if self._round > _MAX_ROUNDS:
      self._finish_by_round_limit()
      return
    self._round_effects = [set(), set()]
    for p in range(_NUM_PLAYERS):
      self._queue.append(("draw", p))
    for _ in range(_NUM_PLAYERS):
      self._queue.append(("reveal",))
    for p in range(_NUM_PLAYERS):
      self._queue.append(("build", p))
    for p in range(_NUM_PLAYERS):
      self._queue.append(("spell", p))
    self._queue.append(("resolve",))

  def _finish_by_round_limit(self):
    self._is_terminal = True
    margin0 = (self._souls[0] - self._wounds[0]) - (
        self._souls[1] - self._wounds[1])
    if margin0 > 0:
      self._returns = [1.0, -1.0]
    elif margin0 < 0:
      self._returns = [-1.0, 1.0]
    else:
      self._returns = [0.0, 0.0]

  # -- pyspiel.State API --------------------------------------------------

  def current_player(self):
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    self._advance()
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    step = self._queue[0]
    if step[0] in ("draw", "reveal"):
      return pyspiel.PlayerId.CHANCE
    return step[1]

  def _legal_actions(self, player):
    self._advance()
    step = self._queue[0]
    kind = step[0]
    p = step[1]
    assert player == p
    hand = self._hands[p]
    if kind == "build":
      actions = [_ACTION_PASS]
      if len(self._rooms[p]) < self._max_rooms[p]:
        for i, card in enumerate(hand):
          if card[0] == "room":
            actions.append(1 + i)
      return sorted(actions)
    elif kind == "spell":
      actions = [_ACTION_PASS]
      for i, card in enumerate(hand):
        if card[0] == "spell":
          actions.append(1 + i)
      return sorted(actions)
    elif kind == "discard":
      return sorted(1 + i for i in range(len(hand)))
    raise ValueError(f"No legal player actions at step {step}")

  def chance_outcomes(self):
    self._advance()
    step = self._queue[0]
    if step[0] == "draw":
      n = len(self._deck)
      if n == 0:
        return [(0, 1.0)]  # No-op outcome; `_apply_action` checks deck size.
      p = 1.0 / n
      return [(i, p) for i in range(n)]
    elif step[0] == "reveal":
      n = len(self._hero_pool)
      if n == 0:
        return [(0, 1.0)]
      p = 1.0 / n
      return [(i, p) for i in range(n)]
    raise ValueError(f"Not a chance node: {step}")

  def _apply_action(self, action):
    self._advance()
    step = self._queue[0]
    kind = step[0]
    if kind == "draw":
      self._queue.popleft()
      p = step[1]
      if self._deck:
        idx = action if action < len(self._deck) else 0
        card = self._deck.pop(idx)
        if len(self._hands[p]) < _HAND_MAX:
          self._hands[p].append(card)
        self._history_strings.append(f"draw p{p}")
      else:
        self._recycle_build_discard()
        self._history_strings.append(f"draw p{p} (deck empty)")
    elif kind == "reveal":
      self._queue.popleft()
      if self._hero_pool:
        idx = action if action < len(self._hero_pool) else 0
        hero_id = self._hero_pool.pop(idx)
        self._waiting.append(hero_id)
        self._history_strings.append(
            f"reveal {data.HERO_TEMPLATES[hero_id].name}")
      else:
        self._recycle_hero_discard()
    elif kind == "build":
      self._queue.popleft()
      p = step[1]
      if action != _ACTION_PASS:
        i = action - 1
        card = self._hands[p].pop(i)
        assert card[0] == "room"
        self._rooms[p].append(card[1])
        self._history_strings.append(
            f"p{p} builds {data.ROOM_TEMPLATES[card[1]].name}")
      else:
        self._history_strings.append(f"p{p} builds nothing")
    elif kind == "spell":
      self._queue.popleft()
      p = step[1]
      if action != _ACTION_PASS:
        i = action - 1
        card = self._hands[p].pop(i)
        assert card[0] == "spell"
        self._build_discard.append(card)
        self._cast_spell(p, card[1])
      else:
        self._history_strings.append(f"p{p} casts nothing")
    elif kind == "discard":
      p = step[1]
      i = action - 1
      card = self._hands[p].pop(i)
      self._build_discard.append(card)
      self._history_strings.append(f"p{p} discards {self._card_of(*card).name}")
      if len(self._hands[p]) <= self._hand_limit(p):
        self._queue.popleft()
      # else: leave the "discard" step at the front for another discard.
    else:
      raise ValueError(f"Unexpected step kind {kind}")
    self._advance()

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"Chance:{action}"
    if action == _ACTION_PASS:
      return "Pass"
    return f"Hand slot {action - 1}"

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    return list(self._returns)

  def debug_log(self, n=15):
    """Returns the last `n` human-readable event strings (for UIs/servers)."""
    return list(self._history_strings[-n:])

  def __str__(self):
    lines = [f"Round {self._round}"]
    for p in range(_NUM_PLAYERS):
      boss_name = data.BOSSES[self._boss[p]].name
      rooms = ", ".join(data.ROOM_TEMPLATES[r].name for r in self._rooms[p])
      lines.append(
          f"P{p} [{boss_name}] wounds={self._wounds[p]} "
          f"souls={self._souls[p]} hand={len(self._hands[p])} "
          f"rooms=[{rooms}]")
    waiting = ", ".join(data.HERO_TEMPLATES[h].name for h in self._waiting)
    lines.append(f"Waiting heroes: [{waiting}]")
    if self._queue:
      lines.append(f"Next: {self._queue[0]}")
    return "\n".join(lines)

  # -- Game logic -----------------------------------------------------

  def _recycle_build_discard(self):
    # Shuffling isn't needed here: draws are already chance nodes that pick
    # a uniformly random remaining card, so simply refilling the deck from
    # the discard pile (in any fixed order) is equivalent to a fresh
    # shuffle. If both are empty, the draw/reveal is simply skipped.
    if self._build_discard:
      self._deck.extend(self._build_discard)
      self._build_discard = []

  def _recycle_hero_discard(self):
    if self._hero_discard:
      self._hero_pool.extend(self._hero_discard)
      self._hero_discard = []

  def _cast_spell(self, p, spell_template_id):
    effect = data.SPELL_TEMPLATES[spell_template_id].effect
    opp = 1 - p
    self._history_strings.append(
        f"p{p} casts {data.SPELL_TEMPLATES[spell_template_id].name}")
    if effect == "draw1":
      self._push_front(("draw", p))
    elif effect == "draw2":
      self._push_front(("draw", p))
      self._push_front(("draw", p))
    elif effect == "scry":
      self._push_front(("discard", p))
      self._push_front(("draw", p))
    elif effect == "treasure_cache":
      self._round_effects[p].add("treasure_cache")
    elif effect == "reckless_strike":
      self._round_effects[p].add("reckless_strike")
    elif effect == "ambush":
      self._round_effects[p].add("ambush")
    elif effect == "fortify":
      self._round_effects[p].add("fortify")
    elif effect == "panic":
      self._round_effects[opp].add("panic_limited")
    elif effect == "bone_pile":
      if self._build_discard:
        card = self._build_discard.pop()
        if len(self._hands[p]) < _HAND_MAX:
          self._hands[p].append(card)

  def _tiebreak_favor(self):
    for p in range(_NUM_PLAYERS):
      if "ambush" in self._round_effects[p]:
        return p
      if _boss_ability(self._boss[p]) == "tiebreak_favor":
        return p
    return None

  def _resolve_round(self):
    # --- Bait phase ---
    treasure = [[0] * _NUM_CLASSES for _ in range(_NUM_PLAYERS)]
    for p in range(_NUM_PLAYERS):
      for rid in self._rooms[p]:
        room = data.ROOM_TEMPLATES[rid]
        treasure[p][room.cls] += room.treasure
      if "treasure_cache" in self._round_effects[p] and self._rooms[p]:
        last_room = data.ROOM_TEMPLATES[self._rooms[p][-1]]
        treasure[p][last_room.cls] += 1

    favor = self._tiebreak_favor()
    enroute = {p: [] for p in range(_NUM_PLAYERS)}
    still_waiting = []
    for hero_id in self._waiting:
      hero = data.HERO_TEMPLATES[hero_id]
      t = [treasure[p][hero.cls] for p in range(_NUM_PLAYERS)]
      if all(v == 0 for v in t):
        still_waiting.append(hero_id)
        continue
      best = max(t)
      leaders = [p for p in range(_NUM_PLAYERS) if t[p] == best]
      if len(leaders) == 1:
        enroute[leaders[0]].append(hero_id)
      elif favor is not None and favor in leaders:
        enroute[favor].append(hero_id)
      else:
        still_waiting.append(hero_id)
    self._waiting = still_waiting

    # --- Adventure phase ---
    for p in range(_NUM_PLAYERS):
      rooms = self._rooms[p]
      bonus_entrance = 1 if _boss_ability(
          self._boss[p]) == "entrance_bonus_dmg" else 0
      bonus_entrance += 1 if "reckless_strike" in self._round_effects[p] else 0
      bonus_boss_room = 1 if _boss_ability(
          self._boss[p]) == "boss_room_bonus_dmg" else 0
      wound_reduction = 1 if "fortify" in self._round_effects[p] else 0

      for hero_id in enroute[p]:
        hero = data.HERO_TEMPLATES[hero_id]
        hp = hero.health
        died = False
        for idx, rid in enumerate(rooms):
          dmg = data.ROOM_TEMPLATES[rid].damage
          if idx == 0:
            dmg += bonus_entrance
          if idx == len(rooms) - 1:
            dmg += bonus_boss_room
          hp -= dmg
          if hp <= 0:
            died = True
            break
        if died:
          self._souls[p] += 1
          if (not self._epic_merged and
              self._souls[p] >= _EPIC_THRESHOLD_SOULS):
            self._epic_merged = True
            self._hero_pool.extend(self._epic_pool)
            self._epic_pool = []
        else:
          self._wounds[p] += max(0, 1 - wound_reduction)
          self._hero_discard.append(hero_id)

    self._check_terminal()
    if not self._is_terminal:
      self._round += 1
      # `_hand_limit` still sees this round's spell effects (e.g. Panic) --
      # they are only cleared once discards are done, at the start of the
      # next round (see `_start_or_end_round`).
      for p in range(_NUM_PLAYERS):
        if len(self._hands[p]) > self._hand_limit(p):
          self._queue.append(("discard", p))
      self._queue.append(("new_round",))

  def _check_terminal(self):
    lost = [self._wounds[p] >= self._wound_limit(p)
            for p in range(_NUM_PLAYERS)]
    won = [self._souls[p] >= _SOUL_TARGET and not lost[p]
           for p in range(_NUM_PLAYERS)]
    if not (any(lost) or any(won)):
      return
    self._is_terminal = True
    if lost[0] and lost[1]:
      self._returns = [0.0, 0.0]
    elif lost[0]:
      self._returns = [-1.0, 1.0]
    elif lost[1]:
      self._returns = [1.0, -1.0]
    elif won[0] and won[1]:
      self._returns = [0.0, 0.0]
    elif won[0]:
      self._returns = [1.0, -1.0]
    else:
      self._returns = [-1.0, 1.0]


class BossMonsterObserver:
  """Observer, conforming to the PyObserver interface (see observation.py).

  Public: both players' built rooms, wound/soul counts, waiting heroes,
  round number, and own-hand size of the opponent.
  Private: the observing player's own hand, described by card stats.
  """

  def __init__(self, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    max_hand = _HAND_MAX
    max_rooms = _BASE_MAX_ROOMS + 1
    max_waiting = 16

    pieces = [
        ("own_hand", max_hand * 7, (max_hand, 7)),
        ("opp_hand_count", 1, (1,)),
        ("rooms", _NUM_PLAYERS * max_rooms * (_NUM_CLASSES + 2),
         (_NUM_PLAYERS, max_rooms, _NUM_CLASSES + 2)),
        ("waiting", max_waiting * (_NUM_CLASSES + 2),
         (max_waiting, _NUM_CLASSES + 2)),
        ("scalars", 8, (8,)),
    ]
    self._max_hand = max_hand
    self._max_rooms = max_rooms
    self._max_waiting = max_waiting

    total_size = sum(size for _, size, _ in pieces)
    self.tensor = np.zeros(total_size, np.float32)
    self.dict = {}
    idx = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[idx:idx + size].reshape(shape)
      idx += size

  def set_from(self, state, player):
    self.tensor.fill(0)
    own_hand = self.dict["own_hand"]
    for i, card in enumerate(state._hands[player][:self._max_hand]):
      kind, template_id = card
      own_hand[i, 0] = 1.0
      if kind == "room":
        room = data.ROOM_TEMPLATES[template_id]
        own_hand[i, 1] = 1.0
        own_hand[i, 2 + room.cls] = 1.0
        own_hand[i, 6] = room.damage / 5.0
      else:
        own_hand[i, 1] = 0.0
    self.dict["opp_hand_count"][0] = len(state._hands[1 - player]) / float(
        _HAND_MAX)

    rooms = self.dict["rooms"]
    for p in range(_NUM_PLAYERS):
      for i, rid in enumerate(state._rooms[p][:self._max_rooms]):
        room = data.ROOM_TEMPLATES[rid]
        rooms[p, i, 0] = 1.0
        rooms[p, i, 1 + room.cls] = 1.0
        rooms[p, i, 1 + _NUM_CLASSES] = room.damage / 5.0

    waiting = self.dict["waiting"]
    for i, hid in enumerate(state._waiting[:self._max_waiting]):
      hero = data.HERO_TEMPLATES[hid]
      waiting[i, 0] = 1.0
      waiting[i, 1 + hero.cls] = 1.0
      waiting[i, 1 + _NUM_CLASSES] = hero.health / 9.0

    scalars = self.dict["scalars"]
    scalars[0] = state._wounds[player] / float(_WOUND_LIMIT)
    scalars[1] = state._wounds[1 - player] / float(_WOUND_LIMIT)
    scalars[2] = state._souls[player] / float(_SOUL_TARGET)
    scalars[3] = state._souls[1 - player] / float(_SOUL_TARGET)
    scalars[4] = state._round / float(_MAX_ROUNDS)
    scalars[5] = len(state._deck) / float(data.NUM_BUILD_CARDS)
    scalars[6] = 1.0 if state.current_player() == player else 0.0
    scalars[7] = 1.0

  def string_from(self, state, player):
    hand = ", ".join(
        state._card_of(*c).name for c in state._hands[player])
    return (f"p{player}: hand=[{hand}] wounds={state._wounds} "
            f"souls={state._souls} round={state._round}")


pyspiel.register_game(_GAME_TYPE, BossMonsterGame)
