// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/boss_monster/boss_monster.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace boss_monster {
namespace {

const GameType kGameType{
    /*short_name=*/"boss_monster",
    /*long_name=*/"Boss Monster",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"boss0", GameParameter(0)}, {"boss1", GameParameter(1)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BossMonsterGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// ---------------------------------------------------------------------------
// Card data. This mirrors open_spiel/python/games/boss_monster_data.py exactly,
// including ordering, so that template ids agree between the two
// implementations (see boss_monster_equivalence_test.py). That file also
// carries the note on which rules/cards are verified against primary sources
// versus approximated.
// ---------------------------------------------------------------------------

// Classes: 0 = fighter, 1 = cleric, 2 = mage, 3 = thief.
struct RoomTier {
  const char* name;
  int cls;
  int treasure;
  int damage;
  int count;
};

constexpr RoomTier kRoomTiers[] = {
    // Fighter (weapon treasure).
    {"Goblin Ambush", 0, 1, 1, 5},
    {"Skeleton Squad", 0, 1, 2, 4},
    {"Orc Barracks", 0, 2, 2, 4},
    {"Minotaur Maze", 0, 2, 3, 3},
    {"Dragon's Lair", 0, 3, 4, 3},
    // Cleric (relic treasure).
    {"Cursed Chapel", 1, 1, 1, 5},
    {"Ghoul Crypt", 1, 1, 2, 4},
    {"Wraith Sanctum", 1, 2, 2, 4},
    {"Banshee Choir", 1, 2, 3, 3},
    {"Lich's Altar", 1, 3, 4, 3},
    // Mage (tome treasure).
    {"Arcane Ward", 2, 1, 1, 5},
    {"Mimic Vault", 2, 1, 2, 4},
    {"Elemental Rift", 2, 2, 2, 4},
    {"Golem Workshop", 2, 2, 3, 3},
    {"Sorcerer's Study", 2, 3, 4, 3},
    // Thief (coin treasure).
    {"Rat Warren", 3, 1, 1, 5},
    {"Spiked Pit", 3, 1, 2, 4},
    {"Poison Dart Hall", 3, 2, 2, 4},
    {"Gilded Vault", 3, 2, 3, 3},
    {"Serpent Nest", 3, 3, 4, 2},
};

struct SpellDef {
  const char* name;
  SpellEffect effect;
  const char* text;
  int count;
};

constexpr SpellDef kSpellDefs[] = {
    {"Reinforcements", SpellEffect::kDraw1, "Draw 1 card.", 4},
    {"Scrying Orb", SpellEffect::kScry, "Draw 1 card, then discard 1 card.", 4},
    {"Treasure Cache", SpellEffect::kTreasureCache,
     "This round, your room closest to the Boss counts as having +1 treasure "
     "of its type for the Bait Phase.",
     4},
    {"Reckless Strike", SpellEffect::kRecklessStrike,
     "This round, your entrance room deals +1 damage.", 4},
    {"Ambush", SpellEffect::kAmbush,
     "This round, Heroes tied between dungeons come to you instead of staying "
     "put.",
     3},
    {"Fortify", SpellEffect::kFortify,
     "This round, reduce Wounds you would take by 1 (minimum 0).", 4},
    {"Bone Pile", SpellEffect::kBonePile,
     "Return the top card of the shared discard pile to your hand.", 3},
    {"Panic", SpellEffect::kPanic,
     "This round, your opponent's hand size limit is reduced by 1.", 3},
    {"Second Wind", SpellEffect::kDraw2, "Draw 2 cards.", 2},
};

struct HeroDef {
  const char* name;
  int cls;
  int health;
  bool epic;
};

// Ordinary Heroes first (ids 0..24), then Epic Heroes (ids 25..40), each
// group ordered fighter, cleric, mage, thief.
constexpr HeroDef kHeroDefs[] = {
    {"Squire", 0, 2, false},          {"Man-at-Arms", 0, 3, false},
    {"Knight", 0, 4, false},          {"Berserker", 0, 4, false},
    {"Paladin", 0, 5, false},         {"Barbarian", 0, 3, false},
    {"Acolyte", 1, 2, false},         {"Priest", 1, 3, false},
    {"Monk", 1, 3, false},            {"Bishop", 1, 4, false},
    {"Crusader", 1, 4, false},        {"High Priestess", 1, 5, false},
    {"Apprentice", 2, 2, false},      {"Conjurer", 2, 3, false},
    {"Wizard", 2, 4, false},          {"Warlock", 2, 3, false},
    {"Archmage", 2, 5, false},        {"Illusionist", 2, 4, false},
    {"Pickpocket", 3, 2, false},      {"Rogue", 3, 3, false},
    {"Burglar", 3, 3, false},         {"Assassin", 3, 4, false},
    {"Smuggler", 3, 3, false},        {"Master Thief", 3, 5, false},
    {"Bandit", 3, 2, false},
    // Epic Heroes.
    {"Champion", 0, 6, true},         {"Warlord", 0, 7, true},
    {"Dragon Slayer", 0, 8, true},    {"Titan", 0, 9, true},
    {"Inquisitor", 1, 6, true},       {"Archbishop", 1, 7, true},
    {"Saint", 1, 8, true},            {"Prophet", 1, 9, true},
    {"Runemaster", 2, 6, true},       {"Elementalist", 2, 7, true},
    {"Grand Sorcerer", 2, 8, true},   {"Time Weaver", 2, 9, true},
    {"Duelist", 3, 6, true},          {"Shadow Master", 3, 7, true},
    {"Guild Leader", 3, 8, true},     {"Phantom", 3, 9, true},
};

constexpr BossTemplate kBosses[] = {
    {"King Croak", BossAbility::kHandLimitPlus1,
     "Hand size limit is 6 instead of 5."},
    {"Gorgona", BossAbility::kEntranceBonusDmg,
     "Your entrance room deals +1 damage."},
    {"Cleopatra", BossAbility::kExtraStartHand,
     "Starts the game with 1 extra card."},
    {"Seducia", BossAbility::kTiebreakFavor,
     "Heroes tied between dungeons always come to you."},
    {"Cerebellus", BossAbility::kFirstTurnExtraDraw,
     "Draws 1 extra card on your first turn."},
    {"Draculord", BossAbility::kWoundLimitPlus1,
     "You lose at 6 Wounds instead of 5."},
    {"Xyzax", BossAbility::kBossRoomBonusDmg,
     "Your room next to the Boss deals +1 damage."},
    {"Robobo", BossAbility::kExtraRoomSlot,
     "Your dungeon may hold 6 rooms instead of 5."},
};

}  // namespace

const std::vector<std::string>& ClassNames() {
  static const std::vector<std::string>* names =
      new std::vector<std::string>({"fighter", "cleric", "mage", "thief"});
  return *names;
}

const std::vector<RoomTemplate>& RoomTemplates() {
  static const std::vector<RoomTemplate>* templates = []() {
    auto* v = new std::vector<RoomTemplate>();
    for (const RoomTier& tier : kRoomTiers) {
      v->push_back({tier.name, tier.cls, tier.treasure, tier.damage});
    }
    return v;
  }();
  return *templates;
}

const std::vector<SpellTemplate>& SpellTemplates() {
  static const std::vector<SpellTemplate>* templates = []() {
    auto* v = new std::vector<SpellTemplate>();
    for (const SpellDef& def : kSpellDefs) {
      v->push_back({def.name, def.effect, def.text});
    }
    return v;
  }();
  return *templates;
}

const std::vector<HeroTemplate>& HeroTemplates() {
  static const std::vector<HeroTemplate>* templates = []() {
    auto* v = new std::vector<HeroTemplate>();
    for (const HeroDef& def : kHeroDefs) {
      v->push_back({def.name, def.cls, def.health, def.epic});
    }
    return v;
  }();
  return *templates;
}

const std::vector<BossTemplate>& Bosses() {
  static const std::vector<BossTemplate>* bosses = []() {
    auto* v = new std::vector<BossTemplate>();
    for (const BossTemplate& boss : kBosses) v->push_back(boss);
    return v;
  }();
  return *bosses;
}

const std::vector<Card>& BuildDeck() {
  static const std::vector<Card>* deck = []() {
    auto* v = new std::vector<Card>();
    for (int i = 0; i < static_cast<int>(std::size(kRoomTiers)); ++i) {
      for (int c = 0; c < kRoomTiers[i].count; ++c) {
        v->push_back({CardKind::kRoom, static_cast<int16_t>(i)});
      }
    }
    for (int i = 0; i < static_cast<int>(std::size(kSpellDefs)); ++i) {
      for (int c = 0; c < kSpellDefs[i].count; ++c) {
        v->push_back({CardKind::kSpell, static_cast<int16_t>(i)});
      }
    }
    SPIEL_CHECK_EQ(v->size(), kNumBuildCards);
    return v;
  }();
  return *deck;
}

const std::vector<int>& OrdinaryHeroDeck() {
  static const std::vector<int>* heroes = []() {
    auto* v = new std::vector<int>();
    for (int i = 0; i < static_cast<int>(std::size(kHeroDefs)); ++i) {
      if (!kHeroDefs[i].epic) v->push_back(i);
    }
    SPIEL_CHECK_EQ(v->size(), kNumOrdinaryHeroes);
    return v;
  }();
  return *heroes;
}

const std::vector<int>& EpicHeroDeck() {
  static const std::vector<int>* heroes = []() {
    auto* v = new std::vector<int>();
    for (int i = 0; i < static_cast<int>(std::size(kHeroDefs)); ++i) {
      if (kHeroDefs[i].epic) v->push_back(i);
    }
    SPIEL_CHECK_EQ(v->size(), kNumEpicHeroes);
    return v;
  }();
  return *heroes;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

BossMonsterState::BossMonsterState(std::shared_ptr<const Game> game, int boss0,
                                   int boss1)
    : State(std::move(game)),
      deck_(BuildDeck()),
      hero_pool_(OrdinaryHeroDeck()),
      epic_pool_(EpicHeroDeck()),
      returns_(kNumPlayers, 0.0) {
  boss_[0] = boss0;
  boss_[1] = boss1;
  wounds_.fill(0);
  souls_.fill(0);
  round_effects_.fill(kEffectNone);
  for (Player p = 0; p < kNumPlayers; ++p) {
    max_rooms_[p] =
        kBaseMaxRooms + (Ability(p) == BossAbility::kExtraRoomSlot ? 1 : 0);
  }
  for (Player p = 0; p < kNumPlayers; ++p) {
    const int count =
        kStartingHand + (Ability(p) == BossAbility::kExtraStartHand ? 1 : 0);
    for (int i = 0; i < count; ++i) {
      queue_.push_back({StepKind::kDraw, static_cast<int8_t>(p)});
    }
  }
  queue_.push_back({StepKind::kNewRound, -1});
  Advance();
}

BossAbility BossMonsterState::Ability(Player player) const {
  return Bosses()[boss_[player]].ability;
}

int BossMonsterState::HandLimit(Player player) const {
  int limit = kHandLimit;
  if (Ability(player) == BossAbility::kHandLimitPlus1) limit += 1;
  if (round_effects_[player] & kEffectPanicLimited) limit -= 1;
  return std::max(1, limit);
}

int BossMonsterState::WoundLimit(Player player) const {
  return kWoundLimit +
         (Ability(player) == BossAbility::kWoundLimitPlus1 ? 1 : 0);
}

// Auto-processes the steps that need no action, leaving the front of the queue
// on a real decision (chance, build, spell or discard) or the game terminal.
// Called from the constructor and at the end of every DoApplyAction, so the
// state is always "settled" and CurrentPlayer() can stay const.
void BossMonsterState::Advance() {
  while (!is_terminal_) {
    if (queue_.empty()) {
      StartOrEndRound();
      continue;
    }
    const StepKind kind = queue_.front().kind;
    if (kind == StepKind::kResolve) {
      queue_.pop_front();
      ResolveRound();
      continue;
    }
    if (kind == StepKind::kNewRound) {
      queue_.pop_front();
      StartOrEndRound();
      continue;
    }
    return;
  }
}

void BossMonsterState::StartOrEndRound() {
  if (round_ > kMaxRounds) {
    FinishByRoundLimit();
    return;
  }
  round_effects_.fill(kEffectNone);
  for (Player p = 0; p < kNumPlayers; ++p) {
    queue_.push_back({StepKind::kDraw, static_cast<int8_t>(p)});
    if (round_ == 1 && Ability(p) == BossAbility::kFirstTurnExtraDraw) {
      queue_.push_back({StepKind::kDraw, static_cast<int8_t>(p)});
    }
  }
  for (int i = 0; i < kNumPlayers; ++i) {
    queue_.push_back({StepKind::kReveal, -1});
  }
  for (Player p = 0; p < kNumPlayers; ++p) {
    queue_.push_back({StepKind::kBuild, static_cast<int8_t>(p)});
  }
  for (Player p = 0; p < kNumPlayers; ++p) {
    queue_.push_back({StepKind::kSpell, static_cast<int8_t>(p)});
  }
  queue_.push_back({StepKind::kResolve, -1});
}

void BossMonsterState::FinishByRoundLimit() {
  is_terminal_ = true;
  const int margin0 =
      (souls_[0] - wounds_[0]) - (souls_[1] - wounds_[1]);
  if (margin0 > 0) {
    returns_ = {1.0, -1.0};
  } else if (margin0 < 0) {
    returns_ = {-1.0, 1.0};
  } else {
    returns_ = {0.0, 0.0};
  }
}

void BossMonsterState::RecycleBuildDiscard() {
  // Shuffling isn't needed: draws are chance nodes that pick a uniformly
  // random remaining card, so refilling the deck from the discard pile in any
  // fixed order is equivalent to a fresh shuffle. If both are empty, the
  // draw is simply skipped.
  if (!build_discard_.empty()) {
    deck_.insert(deck_.end(), build_discard_.begin(), build_discard_.end());
    build_discard_.clear();
  }
}

void BossMonsterState::RecycleHeroDiscard() {
  if (!hero_discard_.empty()) {
    hero_pool_.insert(hero_pool_.end(), hero_discard_.begin(),
                      hero_discard_.end());
    hero_discard_.clear();
  }
}

void BossMonsterState::CastSpell(Player player, int spell_template) {
  const SpellEffect effect = SpellTemplates()[spell_template].effect;
  const Player opp = 1 - player;
  const int8_t p8 = static_cast<int8_t>(player);
  switch (effect) {
    case SpellEffect::kDraw1:
      queue_.push_front({StepKind::kDraw, p8});
      break;
    case SpellEffect::kDraw2:
      queue_.push_front({StepKind::kDraw, p8});
      queue_.push_front({StepKind::kDraw, p8});
      break;
    case SpellEffect::kScry:
      queue_.push_front({StepKind::kDiscard, p8});
      queue_.push_front({StepKind::kDraw, p8});
      break;
    case SpellEffect::kTreasureCache:
      round_effects_[player] |= kEffectTreasureCache;
      break;
    case SpellEffect::kRecklessStrike:
      round_effects_[player] |= kEffectRecklessStrike;
      break;
    case SpellEffect::kAmbush:
      round_effects_[player] |= kEffectAmbush;
      break;
    case SpellEffect::kFortify:
      round_effects_[player] |= kEffectFortify;
      break;
    case SpellEffect::kPanic:
      round_effects_[opp] |= kEffectPanicLimited;
      break;
    case SpellEffect::kBonePile:
      if (!build_discard_.empty()) {
        const Card card = build_discard_.back();
        build_discard_.pop_back();
        if (static_cast<int>(hands_[player].size()) < kHandMax) {
          hands_[player].push_back(card);
        }
      }
      break;
  }
}

Player BossMonsterState::TiebreakFavor() const {
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (round_effects_[p] & kEffectAmbush) return p;
    if (Ability(p) == BossAbility::kTiebreakFavor) return p;
  }
  return kInvalidPlayer;
}

void BossMonsterState::ResolveRound() {
  // --- Bait phase: each Hero walks into the dungeon offering the most
  // treasure of its own class; ties leave it waiting (unless broken). ---
  std::array<std::array<int, kNumClasses>, kNumPlayers> treasure;
  for (Player p = 0; p < kNumPlayers; ++p) {
    treasure[p].fill(0);
    for (int room_id : rooms_[p]) {
      const RoomTemplate& room = RoomTemplates()[room_id];
      treasure[p][room.cls] += room.treasure;
    }
    if ((round_effects_[p] & kEffectTreasureCache) && !rooms_[p].empty()) {
      const RoomTemplate& last_room = RoomTemplates()[rooms_[p].back()];
      treasure[p][last_room.cls] += 1;
    }
  }

  const Player favor = TiebreakFavor();
  std::array<std::vector<int>, kNumPlayers> enroute;
  std::vector<int> still_waiting;
  for (int hero_id : waiting_) {
    const HeroTemplate& hero = HeroTemplates()[hero_id];
    std::array<int, kNumPlayers> t;
    bool all_zero = true;
    for (Player p = 0; p < kNumPlayers; ++p) {
      t[p] = treasure[p][hero.cls];
      if (t[p] != 0) all_zero = false;
    }
    if (all_zero) {
      still_waiting.push_back(hero_id);
      continue;
    }
    const int best = *std::max_element(t.begin(), t.end());
    std::vector<Player> leaders;
    for (Player p = 0; p < kNumPlayers; ++p) {
      if (t[p] == best) leaders.push_back(p);
    }
    if (leaders.size() == 1) {
      enroute[leaders[0]].push_back(hero_id);
    } else if (favor != kInvalidPlayer &&
               std::find(leaders.begin(), leaders.end(), favor) !=
                   leaders.end()) {
      enroute[favor].push_back(hero_id);
    } else {
      still_waiting.push_back(hero_id);
    }
  }
  waiting_ = std::move(still_waiting);

  // --- Adventure phase: Heroes take each Room's damage in build order. ---
  for (Player p = 0; p < kNumPlayers; ++p) {
    const std::vector<int>& rooms = rooms_[p];
    int bonus_entrance =
        (Ability(p) == BossAbility::kEntranceBonusDmg ? 1 : 0) +
        ((round_effects_[p] & kEffectRecklessStrike) ? 1 : 0);
    const int bonus_boss_room =
        (Ability(p) == BossAbility::kBossRoomBonusDmg ? 1 : 0);
    const int wound_reduction =
        ((round_effects_[p] & kEffectFortify) ? 1 : 0);

    for (int hero_id : enroute[p]) {
      const HeroTemplate& hero = HeroTemplates()[hero_id];
      int hp = hero.health;
      bool died = false;
      for (int idx = 0; idx < static_cast<int>(rooms.size()); ++idx) {
        int dmg = RoomTemplates()[rooms[idx]].damage;
        if (idx == 0) dmg += bonus_entrance;
        if (idx == static_cast<int>(rooms.size()) - 1) dmg += bonus_boss_room;
        hp -= dmg;
        if (hp <= 0) {
          died = true;
          break;
        }
      }
      if (died) {
        souls_[p] += 1;
        if (!epic_merged_ && souls_[p] >= kEpicThresholdSouls) {
          epic_merged_ = true;
          hero_pool_.insert(hero_pool_.end(), epic_pool_.begin(),
                            epic_pool_.end());
          epic_pool_.clear();
        }
      } else {
        wounds_[p] += std::max(0, 1 - wound_reduction);
        hero_discard_.push_back(hero_id);
      }
    }
  }

  CheckTerminal();
  if (!is_terminal_) {
    round_ += 1;
    // HandLimit() still sees this round's spell effects (e.g. Panic); they
    // are cleared once discards are done, in StartOrEndRound().
    for (Player p = 0; p < kNumPlayers; ++p) {
      if (static_cast<int>(hands_[p].size()) > HandLimit(p)) {
        queue_.push_back({StepKind::kDiscard, static_cast<int8_t>(p)});
      }
    }
    queue_.push_back({StepKind::kNewRound, -1});
  }
}

void BossMonsterState::CheckTerminal() {
  std::array<bool, kNumPlayers> lost;
  std::array<bool, kNumPlayers> won;
  bool any_lost = false;
  bool any_won = false;
  for (Player p = 0; p < kNumPlayers; ++p) {
    lost[p] = wounds_[p] >= WoundLimit(p);
    won[p] = souls_[p] >= kSoulTarget && !lost[p];
    any_lost |= lost[p];
    any_won |= won[p];
  }
  if (!any_lost && !any_won) return;
  is_terminal_ = true;
  if (lost[0] && lost[1]) {
    returns_ = {0.0, 0.0};
  } else if (lost[0]) {
    returns_ = {-1.0, 1.0};
  } else if (lost[1]) {
    returns_ = {1.0, -1.0};
  } else if (won[0] && won[1]) {
    returns_ = {0.0, 0.0};
  } else if (won[0]) {
    returns_ = {1.0, -1.0};
  } else {
    returns_ = {-1.0, 1.0};
  }
}

Player BossMonsterState::CurrentPlayer() const {
  if (is_terminal_) return kTerminalPlayerId;
  SPIEL_CHECK_FALSE(queue_.empty());
  const Step& step = queue_.front();
  if (step.kind == StepKind::kDraw || step.kind == StepKind::kReveal) {
    return kChancePlayerId;
  }
  return step.player;
}

std::vector<Action> BossMonsterState::LegalActions() const {
  if (is_terminal_) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  const Step& step = queue_.front();
  const Player p = step.player;
  const std::vector<Card>& hand = hands_[p];
  std::vector<Action> actions;
  switch (step.kind) {
    case StepKind::kBuild:
      actions.push_back(kActionPass);
      if (static_cast<int>(rooms_[p].size()) < max_rooms_[p]) {
        for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
          if (hand[i].kind == CardKind::kRoom) actions.push_back(1 + i);
        }
      }
      break;
    case StepKind::kSpell:
      actions.push_back(kActionPass);
      for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
        if (hand[i].kind == CardKind::kSpell) actions.push_back(1 + i);
      }
      break;
    case StepKind::kDiscard:
      // A forced discard with nothing to discard (only reachable if the deck
      // ran dry mid-Scrying Orb) just passes; a decision node must never have
      // an empty action set.
      if (hand.empty()) {
        actions.push_back(kActionPass);
      } else {
        for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
          actions.push_back(1 + i);
        }
      }
      break;
    default:
      SpielFatalError("LegalActions called on a non-decision step.");
  }
  return actions;  // Already in ascending order.
}

std::vector<std::pair<Action, double>> BossMonsterState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  const Step& step = queue_.front();
  const int n = (step.kind == StepKind::kDraw)
                    ? static_cast<int>(deck_.size())
                    : static_cast<int>(hero_pool_.size());
  // An exhausted deck still needs one (no-op) outcome; DoApplyAction then
  // recycles the discard pile instead of drawing.
  if (n == 0) return {{0, 1.0}};
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(n);
  const double prob = 1.0 / n;
  for (int i = 0; i < n; ++i) outcomes.push_back({i, prob});
  return outcomes;
}

void BossMonsterState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(is_terminal_);
  SPIEL_CHECK_FALSE(queue_.empty());
  const Step step = queue_.front();
  switch (step.kind) {
    case StepKind::kDraw: {
      queue_.pop_front();
      const Player p = step.player;
      if (!deck_.empty()) {
        const int idx =
            action < static_cast<Action>(deck_.size()) ? action : 0;
        const Card card = deck_[idx];
        deck_.erase(deck_.begin() + idx);
        if (static_cast<int>(hands_[p].size()) < kHandMax) {
          hands_[p].push_back(card);
        }
      } else {
        RecycleBuildDiscard();
      }
      break;
    }
    case StepKind::kReveal: {
      queue_.pop_front();
      if (!hero_pool_.empty()) {
        const int idx =
            action < static_cast<Action>(hero_pool_.size()) ? action : 0;
        waiting_.push_back(hero_pool_[idx]);
        hero_pool_.erase(hero_pool_.begin() + idx);
      } else {
        RecycleHeroDiscard();
      }
      break;
    }
    case StepKind::kBuild: {
      queue_.pop_front();
      const Player p = step.player;
      if (action != kActionPass) {
        const int i = action - 1;
        SPIEL_CHECK_LT(i, static_cast<int>(hands_[p].size()));
        const Card card = hands_[p][i];
        SPIEL_CHECK_TRUE(card.kind == CardKind::kRoom);
        hands_[p].erase(hands_[p].begin() + i);
        rooms_[p].push_back(card.tmpl);
      }
      break;
    }
    case StepKind::kSpell: {
      queue_.pop_front();
      const Player p = step.player;
      if (action != kActionPass) {
        const int i = action - 1;
        SPIEL_CHECK_LT(i, static_cast<int>(hands_[p].size()));
        const Card card = hands_[p][i];
        SPIEL_CHECK_TRUE(card.kind == CardKind::kSpell);
        hands_[p].erase(hands_[p].begin() + i);
        build_discard_.push_back(card);
        CastSpell(p, card.tmpl);
      }
      break;
    }
    case StepKind::kDiscard: {
      const Player p = step.player;
      if (action == kActionPass || hands_[p].empty()) {
        queue_.pop_front();
        break;
      }
      const int i = action - 1;
      SPIEL_CHECK_LT(i, static_cast<int>(hands_[p].size()));
      build_discard_.push_back(hands_[p][i]);
      hands_[p].erase(hands_[p].begin() + i);
      if (static_cast<int>(hands_[p].size()) <= HandLimit(p)) {
        queue_.pop_front();
      }
      // Otherwise the discard step stays at the front for another discard.
      break;
    }
    default:
      SpielFatalError("Unexpected step kind in DoApplyAction.");
  }
  Advance();
}

std::string BossMonsterState::ActionToString(Player player,
                                             Action action) const {
  if (player == kChancePlayerId) return absl::StrCat("Chance:", action);
  if (action == kActionPass) return "Pass";
  return absl::StrCat("Hand slot ", action - 1);
}

bool BossMonsterState::IsTerminal() const { return is_terminal_; }

std::vector<double> BossMonsterState::Returns() const { return returns_; }

std::string BossMonsterState::PhaseString() const {
  if (is_terminal_) return "game_over";
  if (queue_.empty()) return "none";
  switch (queue_.front().kind) {
    case StepKind::kDraw:
      return "draw";
    case StepKind::kReveal:
      return "reveal";
    case StepKind::kBuild:
      return "build";
    case StepKind::kSpell:
      return "spell";
    case StepKind::kDiscard:
      return "discard";
    default:
      return "none";
  }
}

std::string BossMonsterState::ToString() const {
  std::string out = absl::StrCat("Round ", round_, "\n");
  for (Player p = 0; p < kNumPlayers; ++p) {
    std::vector<std::string> room_names;
    room_names.reserve(rooms_[p].size());
    for (int room_id : rooms_[p]) {
      room_names.push_back(RoomTemplates()[room_id].name);
    }
    absl::StrAppend(&out, "P", p, " [", Bosses()[boss_[p]].name,
                    "] wounds=", wounds_[p], " souls=", souls_[p],
                    " hand=", hands_[p].size(), " rooms=[",
                    absl::StrJoin(room_names, ", "), "]\n");
  }
  std::vector<std::string> hero_names;
  hero_names.reserve(waiting_.size());
  for (int hero_id : waiting_) {
    hero_names.push_back(HeroTemplates()[hero_id].name);
  }
  absl::StrAppend(&out, "Waiting heroes: [", absl::StrJoin(hero_names, ", "),
                  "]\n");
  absl::StrAppend(&out, "Next: ", PhaseString());
  if (!is_terminal_ && !queue_.empty() && queue_.front().player >= 0) {
    absl::StrAppend(&out, " p", static_cast<int>(queue_.front().player));
  }
  return out;
}

std::string BossMonsterState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::vector<std::string> card_names;
  card_names.reserve(hands_[player].size());
  for (const Card& card : hands_[player]) {
    card_names.push_back(card.kind == CardKind::kRoom
                             ? RoomTemplates()[card.tmpl].name
                             : SpellTemplates()[card.tmpl].name);
  }
  return absl::StrCat("p", player, ": hand=[", absl::StrJoin(card_names, ", "),
                      "] wounds=[", wounds_[0], ", ", wounds_[1], "] souls=[",
                      souls_[0], ", ", souls_[1], "] round=", round_);
}

void BossMonsterState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(static_cast<int>(values.size()), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);
  int offset = 0;

  // The observing player's own hand (private information).
  const std::vector<Card>& hand = hands_[player];
  const int num_hand = std::min<int>(hand.size(), kHandMax);
  for (int i = 0; i < num_hand; ++i) {
    float* row = &values[offset + i * kHandFeatures];
    row[0] = 1.0f;
    if (hand[i].kind == CardKind::kRoom) {
      const RoomTemplate& room = RoomTemplates()[hand[i].tmpl];
      row[1] = 1.0f;
      row[2 + room.cls] = 1.0f;
      row[6] = room.damage / 5.0f;
    }
  }
  offset += kHandMax * kHandFeatures;

  // The opponent's hand is hidden; only its size is public.
  values[offset] =
      hands_[1 - player].size() / static_cast<float>(kHandMax);
  offset += 1;

  // Both dungeons, from the observing player's perspective: index 0 is always
  // the observer's own dungeon, index 1 the opponent's.
  for (int rel = 0; rel < kNumPlayers; ++rel) {
    const Player p = (rel == 0) ? player : 1 - player;
    const int num_rooms = std::min<int>(rooms_[p].size(), kMaxRoomSlots);
    for (int i = 0; i < num_rooms; ++i) {
      float* row = &values[offset + (rel * kMaxRoomSlots + i) * kRoomFeatures];
      const RoomTemplate& room = RoomTemplates()[rooms_[p][i]];
      row[0] = 1.0f;
      row[1 + room.cls] = 1.0f;
      row[1 + kNumClasses] = room.damage / 5.0f;
    }
  }
  offset += kNumPlayers * kMaxRoomSlots * kRoomFeatures;

  // Heroes waiting to be lured (public).
  const int num_waiting = std::min<int>(waiting_.size(), kMaxWaitingHeroes);
  for (int i = 0; i < num_waiting; ++i) {
    const HeroTemplate& hero = HeroTemplates()[waiting_[i]];
    float* row = &values[offset + i * kHeroFeatures];
    row[0] = 1.0f;
    row[1 + hero.cls] = 1.0f;
    row[1 + kNumClasses] = hero.health / 9.0f;
  }
  offset += kMaxWaitingHeroes * kHeroFeatures;

  values[offset + 0] = wounds_[player] / static_cast<float>(kWoundLimit);
  values[offset + 1] = wounds_[1 - player] / static_cast<float>(kWoundLimit);
  values[offset + 2] = souls_[player] / static_cast<float>(kSoulTarget);
  values[offset + 3] = souls_[1 - player] / static_cast<float>(kSoulTarget);
  values[offset + 4] = round_ / static_cast<float>(kMaxRounds);
  values[offset + 5] = deck_.size() / static_cast<float>(kNumBuildCards);
  values[offset + 6] = (CurrentPlayer() == player) ? 1.0f : 0.0f;
  values[offset + 7] = 1.0f;
}

std::unique_ptr<State> BossMonsterState::Clone() const {
  return std::unique_ptr<State>(new BossMonsterState(*this));
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

namespace {
int NormalizeBoss(int value) {
  return ((value % kNumBosses) + kNumBosses) % kNumBosses;
}
}  // namespace

BossMonsterGame::BossMonsterGame(const GameParameters& params)
    : Game(kGameType, params) {
  boss_[0] = NormalizeBoss(ParameterValue<int>("boss0"));
  boss_[1] = NormalizeBoss(ParameterValue<int>("boss1"));
}

std::unique_ptr<State> BossMonsterGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new BossMonsterState(shared_from_this(), boss_[0], boss_[1]));
}

}  // namespace boss_monster
}  // namespace open_spiel
