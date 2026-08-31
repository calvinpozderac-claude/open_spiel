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

"""Card and rules data for `boss_monster.py`.

This module encodes the core, verified rules of Brotherwise Games' *Boss
Monster: The Dungeon-Building Card Game* (base set, BoardGameGeek #131835:
https://boardgamegeek.com/boardgame/131835/boss-monster-the-dungeon-building-card-game),
together with a card set that is *mechanically* faithful to the base game.

IMPORTANT SOURCING NOTE
------------------------
This module was written inside a sandboxed environment whose outbound
network access blocked essentially every primary source for this game
(boardgamegeek.com, the official rulebook PDFs, and the Boss Monster fan
wiki all returned "egress blocked"). What follows was assembled from the
handful of search-result snippets that *were* reachable plus general
knowledge of the game, and should be treated accordingly:

Confirmed directly from reachable sources (high confidence):
  * Turn phases: Beginning of Turn (reveal Heroes, draw cards), Build Phase,
    Bait Phase, Adventure Phase, End of Turn.
  * Bait Phase: each Hero moves to the dungeon entrance with the *most*
    treasure of its type; ties mean the Hero does not move.
  * Adventure Phase: a Hero that dies before reaching the Boss Chamber
    becomes a Soul for the dungeon's owner; a Hero that reaches the Boss
    Chamber deals a Wound to the dungeon's owner.
  * Win/Loss: any player with >=5 Wounds loses (regardless of Soul count);
    any player with >=10 Souls (and <5 Wounds) wins.
  * The four Hero classes are Fighter, Cleric, Mage and Thief, each with a
    matching treasure-type icon (weapon / relic / tome / coin) that a
    Dungeon must offer the most of to lure that class of Hero.
  * Card composition of the base set: 8 Boss cards, 75 Room cards, 31 Spell
    cards, 25 ordinary Hero cards, 16 Epic Hero cards (155 cards total).
  * The 8 base-set Bosses are: King Croak, Gorgona, Cleopatra, Seducia,
    Cerebellus (the Father Brain), Draculord, Xyzax and Robobo.
  * Epic Heroes are shuffled into the Hero deck once a player collects
    their second Soul.

NOT independently re-verified against a primary source (best effort /
approximated, and clearly called out as such):
  * The exact name, Class, treasure value, damage, and Health of every
    individual one of the 75 Room / 31 Spell / 41 Hero cards.
  * The exact wording and mechanical effect of each Boss's unique ability.
  * Whether Room cards have a "front"/"back" (Basic/Advanced) evolving
    side. This implementation deliberately omits room evolution to keep
    the action space tractable for RL training; see the module docstring
    in `boss_monster.py` for the full list of simplifications.

The card *counts* per type (75/31/25/16/8) match the confirmed base-set
composition above, and every card's stats were hand-authored to follow the
game's well-documented cost/power curve (cheap early rooms deal 1-2 damage
and offer 1 treasure; late-game rooms deal up to 4-5 damage and offer up to
3 treasure). If you have access to the real card list, the templates below
are a natural place to paste in exact names/stats -- nothing elsewhere in
the engine depends on the specific flavor text.
"""

import collections

CLASSES = ("fighter", "cleric", "mage", "thief")
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

RoomTemplate = collections.namedtuple(
    "RoomTemplate", ["name", "cls", "treasure", "damage"])
SpellTemplate = collections.namedtuple(
    "SpellTemplate", ["name", "effect", "text"])
HeroTemplate = collections.namedtuple(
    "HeroTemplate", ["name", "cls", "health", "epic"])
BossTemplate = collections.namedtuple(
    "BossTemplate", ["name", "ability", "text"])

# ---------------------------------------------------------------------------
# Bosses (8). Names are the real base-set bosses. Abilities are a compact,
# mechanically-simple approximation of each boss's flavor (see sourcing note
# above) chosen so every boss has a distinct, easy-to-verify effect.
# ---------------------------------------------------------------------------

BOSSES = [
    BossTemplate("King Croak", "hand_limit_plus1",
                  "Hand size limit is 6 instead of 5."),
    BossTemplate("Gorgona", "entrance_bonus_dmg",
                  "Your entrance room deals +1 damage."),
    BossTemplate("Cleopatra", "extra_start_hand",
                  "Starts the game with 1 extra card."),
    BossTemplate("Seducia", "tiebreak_favor",
                  "Heroes tied between dungeons always come to you."),
    BossTemplate("Cerebellus", "first_turn_extra_draw",
                  "Draws 1 extra card on your first turn."),
    BossTemplate("Draculord", "wound_limit_plus1",
                  "You lose at 6 Wounds instead of 5."),
    BossTemplate("Xyzax", "boss_room_bonus_dmg",
                  "Your room next to the Boss deals +1 damage."),
    BossTemplate("Robobo", "extra_room_slot",
                  "Your dungeon may hold 6 rooms instead of 5."),
]
NUM_BOSSES = len(BOSSES)

# ---------------------------------------------------------------------------
# Room cards. Each (class, treasure, damage, count) tier is instantiated as
# `count` physically distinct (but mechanically identical) copies, matching
# how a physical Boss Monster deck has multiple copies of some rooms.
# ---------------------------------------------------------------------------

_ROOM_TIERS = {
    "fighter": [
        ("Goblin Ambush", 1, 1, 5),
        ("Skeleton Squad", 1, 2, 4),
        ("Orc Barracks", 2, 2, 4),
        ("Minotaur Maze", 2, 3, 3),
        ("Dragon's Lair", 3, 4, 3),
    ],
    "cleric": [
        ("Cursed Chapel", 1, 1, 5),
        ("Ghoul Crypt", 1, 2, 4),
        ("Wraith Sanctum", 2, 2, 4),
        ("Banshee Choir", 2, 3, 3),
        ("Lich's Altar", 3, 4, 3),
    ],
    "mage": [
        ("Arcane Ward", 1, 1, 5),
        ("Mimic Vault", 1, 2, 4),
        ("Elemental Rift", 2, 2, 4),
        ("Golem Workshop", 2, 3, 3),
        ("Sorcerer's Study", 3, 4, 3),
    ],
    "thief": [
        ("Rat Warren", 1, 1, 5),
        ("Spiked Pit", 1, 2, 4),
        ("Poison Dart Hall", 2, 2, 4),
        ("Gilded Vault", 2, 3, 3),
        ("Serpent Nest", 3, 4, 2),
    ],
}

ROOM_TEMPLATES = []
_ROOM_COUNTS = []
for cls_name in CLASSES:
  for name, treasure, damage, count in _ROOM_TIERS[cls_name]:
    ROOM_TEMPLATES.append(
        RoomTemplate(name, CLASS_INDEX[cls_name], treasure, damage))
    _ROOM_COUNTS.append(count)
NUM_ROOM_TEMPLATES = len(ROOM_TEMPLATES)

# ---------------------------------------------------------------------------
# Spell cards (31 total). Each has a simple, automatically-resolved effect
# (no extra target selection) so the action space stays small.
# ---------------------------------------------------------------------------

# effect ids: "draw1", "scry", "treasure_cache", "reckless_strike", "ambush",
#             "fortify", "bone_pile", "panic", "draw2"
_SPELL_DEFS = [
    ("Reinforcements", "draw1", "Draw 1 card.", 4),
    ("Scrying Orb", "scry",
     "Draw 1 card, then discard 1 card.", 4),
    ("Treasure Cache", "treasure_cache",
     "This round, your room closest to the Boss counts as having "
     "+1 treasure of its type for the Bait Phase.", 4),
    ("Reckless Strike", "reckless_strike",
     "This round, your entrance room deals +1 damage.", 4),
    ("Ambush", "ambush",
     "This round, Heroes tied between dungeons come to you instead of "
     "staying put.", 3),
    ("Fortify", "fortify",
     "This round, reduce Wounds you would take by 1 (minimum 0).", 4),
    ("Bone Pile", "bone_pile",
     "Return the top card of the shared discard pile to your hand.", 3),
    ("Panic", "panic",
     "This round, your opponent's hand size limit is reduced by 1.", 3),
    ("Second Wind", "draw2", "Draw 2 cards.", 2),
]

SPELL_TEMPLATES = []
_SPELL_COUNTS = []
for name, effect, text, count in _SPELL_DEFS:
  SPELL_TEMPLATES.append(SpellTemplate(name, effect, text))
  _SPELL_COUNTS.append(count)
NUM_SPELL_TEMPLATES = len(SPELL_TEMPLATES)
SPELL_EFFECT_IDS = [effect for _, effect, _, _ in _SPELL_DEFS]

# ---------------------------------------------------------------------------
# Hero cards: 25 ordinary + 16 Epic = 41 total.
# ---------------------------------------------------------------------------

_ORDINARY_HEROES = {
    "fighter": [("Squire", 2), ("Man-at-Arms", 3), ("Knight", 4),
                ("Berserker", 4), ("Paladin", 5), ("Barbarian", 3)],
    "cleric": [("Acolyte", 2), ("Priest", 3), ("Monk", 3), ("Bishop", 4),
               ("Crusader", 4), ("High Priestess", 5)],
    "mage": [("Apprentice", 2), ("Conjurer", 3), ("Wizard", 4),
             ("Warlock", 3), ("Archmage", 5), ("Illusionist", 4)],
    "thief": [("Pickpocket", 2), ("Rogue", 3), ("Burglar", 3),
              ("Assassin", 4), ("Smuggler", 3), ("Master Thief", 5),
              ("Bandit", 2)],
}

_EPIC_HEROES = {
    "fighter": [("Champion", 6), ("Warlord", 7), ("Dragon Slayer", 8),
                ("Titan", 9)],
    "cleric": [("Inquisitor", 6), ("Archbishop", 7), ("Saint", 8),
               ("Prophet", 9)],
    "mage": [("Runemaster", 6), ("Elementalist", 7), ("Grand Sorcerer", 8),
             ("Time Weaver", 9)],
    "thief": [("Duelist", 6), ("Shadow Master", 7), ("Guild Leader", 8),
              ("Phantom", 9)],
}

HERO_TEMPLATES = []
_ORDINARY_HERO_IDS = []
_EPIC_HERO_IDS = []
for cls_name in CLASSES:
  for name, health in _ORDINARY_HEROES[cls_name]:
    _ORDINARY_HERO_IDS.append(len(HERO_TEMPLATES))
    HERO_TEMPLATES.append(
        HeroTemplate(name, CLASS_INDEX[cls_name], health, False))
for cls_name in CLASSES:
  for name, health in _EPIC_HEROES[cls_name]:
    _EPIC_HERO_IDS.append(len(HERO_TEMPLATES))
    HERO_TEMPLATES.append(
        HeroTemplate(name, CLASS_INDEX[cls_name], health, True))
NUM_HERO_TEMPLATES = len(HERO_TEMPLATES)

assert len(_ORDINARY_HERO_IDS) == 25
assert len(_EPIC_HERO_IDS) == 16

# ---------------------------------------------------------------------------
# Flattened decks (physical multi-copy card pools).
# ---------------------------------------------------------------------------

# Each entry is a ("room" or "spell", template_id) pair.
BUILD_DECK = []
for template_id, count in enumerate(_ROOM_COUNTS):
  BUILD_DECK.extend([("room", template_id)] * count)
for template_id, count in enumerate(_SPELL_COUNTS):
  BUILD_DECK.extend([("spell", template_id)] * count)
NUM_BUILD_CARDS = len(BUILD_DECK)

assert sum(_ROOM_COUNTS) == 75, sum(_ROOM_COUNTS)
assert sum(_SPELL_COUNTS) == 31, sum(_SPELL_COUNTS)
assert NUM_BUILD_CARDS == 106

ORDINARY_HERO_DECK = list(_ORDINARY_HERO_IDS)
EPIC_HERO_DECK = list(_EPIC_HERO_IDS)

assert len(ORDINARY_HERO_DECK) == 25
assert len(EPIC_HERO_DECK) == 16
