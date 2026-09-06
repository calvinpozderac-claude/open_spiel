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

#ifndef OPEN_SPIEL_GAMES_BOSS_MONSTER_BOSS_MONSTER_H_
#define OPEN_SPIEL_GAMES_BOSS_MONSTER_BOSS_MONSTER_H_

// Boss Monster: The Dungeon-Building Card Game (base set), 2 players.
// https://boardgamegeek.com/boardgame/131835/boss-monster-the-dungeon-building-card-game
//
// Each player builds a row of Rooms leading to their Boss, lures Heroes out of
// a shared pool with the kind of treasure those Heroes want, and tries to kill
// 10 of them (Souls) before 5 of them survive the dungeon and reach the Boss
// (Wounds).
//
// Round structure (all card draws / Hero reveals are explicit chance nodes):
//   draw p0, draw p1, reveal x2, build p0, build p1, spell p0, spell p1,
//   [bait + adventure resolve automatically], discard down to hand size.
//
// Bait phase: each waiting Hero walks into whichever dungeon offers the most
// treasure of its own class; a tie means it waits another round (unless a
// Boss/Spell effect breaks ties). Adventure phase: the Hero takes each Room's
// damage in build order (oldest Room is nearest the entrance); if it dies the
// dungeon's owner gains a Soul, otherwise the owner takes a Wound.
//
// Terminal: >=5 Wounds loses (even with 10+ Souls); >=10 Souls with <5 Wounds
// wins; games are capped at kMaxRounds rounds, after which the better
// Souls-minus-Wounds margin wins (exact tie = draw).
//
// This is a C++ port of open_spiel/python/games/boss_monster.py, kept
// action-for-action identical to it so the two can be cross-validated (see
// boss_monster_test.cc and boss_monster_equivalence_test.py). See
// open_spiel/python/games/boss_monster_data.py for the card data and a
// detailed note on which rules are verified against primary sources versus
// approximated.
//
// Parameters:
//     "boss0"   int   Boss card index (0-7) for player 0   (default = 0)
//     "boss1"   int   Boss card index (0-7) for player 1   (default = 1)

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace boss_monster {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumClasses = 4;  // Fighter, Cleric, Mage, Thief.
inline constexpr int kBaseMaxRooms = 5;
inline constexpr int kHandLimit = 5;
// Defensive hard cap on hand size; also the number of "act on hand slot i"
// actions. Never reached in practice.
inline constexpr int kHandMax = 10;
inline constexpr int kWoundLimit = 5;
inline constexpr int kSoulTarget = 10;
inline constexpr int kEpicThresholdSouls = 2;
inline constexpr int kMaxRounds = 40;
inline constexpr int kStartingHand = kHandLimit;

// Action 0 is "do nothing this phase"; action (1 + i) acts on hand slot i,
// with the meaning (build / cast / discard) determined by the current phase.
inline constexpr Action kActionPass = 0;
inline constexpr int kNumDistinctActions = kHandMax + 1;

// Sizes of the shared decks, as in the physical base set.
inline constexpr int kNumRoomCards = 75;
inline constexpr int kNumSpellCards = 31;
inline constexpr int kNumBuildCards = kNumRoomCards + kNumSpellCards;  // 106
inline constexpr int kNumOrdinaryHeroes = 25;
inline constexpr int kNumEpicHeroes = 16;
inline constexpr int kNumBosses = 8;

// Observation tensor layout (see BossMonsterState::ObservationTensor).
inline constexpr int kMaxRoomSlots = kBaseMaxRooms + 1;  // Robobo gets 6.
inline constexpr int kMaxWaitingHeroes = 16;
inline constexpr int kHandFeatures = 7;
inline constexpr int kRoomFeatures = kNumClasses + 2;
inline constexpr int kHeroFeatures = kNumClasses + 2;
inline constexpr int kNumScalars = 8;
inline constexpr int kObservationTensorSize =
    kHandMax * kHandFeatures + 1 + kNumPlayers * kMaxRoomSlots * kRoomFeatures +
    kMaxWaitingHeroes * kHeroFeatures + kNumScalars;  // 247

enum class SpellEffect : int8_t {
  kDraw1,
  kScry,
  kTreasureCache,
  kRecklessStrike,
  kAmbush,
  kFortify,
  kBonePile,
  kPanic,
  kDraw2,
};

enum class BossAbility : int8_t {
  kHandLimitPlus1,      // King Croak
  kEntranceBonusDmg,    // Gorgona
  kExtraStartHand,      // Cleopatra
  kTiebreakFavor,       // Seducia
  kFirstTurnExtraDraw,  // Cerebellus
  kWoundLimitPlus1,     // Draculord
  kBossRoomBonusDmg,    // Xyzax
  kExtraRoomSlot,       // Robobo
};

// Spell effects that last until the end of the round in which they are cast,
// stored as a bitmask per player.
enum RoundEffect : uint8_t {
  kEffectNone = 0,
  kEffectTreasureCache = 1 << 0,
  kEffectRecklessStrike = 1 << 1,
  kEffectAmbush = 1 << 2,
  kEffectFortify = 1 << 3,
  kEffectPanicLimited = 1 << 4,
};

enum class CardKind : int8_t { kRoom = 0, kSpell = 1 };

// What the state is waiting for next. kResolve and kNewRound are resolved
// automatically (no action is applied for them); the rest are chance nodes
// (kDraw, kReveal) or player decisions (kBuild, kSpell, kDiscard).
enum class StepKind : int8_t {
  kDraw,
  kReveal,
  kBuild,
  kSpell,
  kDiscard,
  kResolve,
  kNewRound,
};

struct RoomTemplate {
  const char* name;
  int cls;
  int treasure;
  int damage;
};

struct SpellTemplate {
  const char* name;
  SpellEffect effect;
  const char* text;
};

struct HeroTemplate {
  const char* name;
  int cls;
  int health;
  bool epic;
};

struct BossTemplate {
  const char* name;
  BossAbility ability;
  const char* text;
};

// A physical card in the shared build deck.
struct Card {
  CardKind kind;
  int16_t tmpl;  // Index into RoomTemplates() or SpellTemplates().

  bool operator==(const Card& o) const {
    return kind == o.kind && tmpl == o.tmpl;
  }
};

struct Step {
  StepKind kind;
  int8_t player;  // Unused (-1) for kReveal / kResolve / kNewRound.
};

// The static card data, mirroring open_spiel/python/games/boss_monster_data.py
// exactly (same order, so template ids match between the two implementations).
const std::vector<RoomTemplate>& RoomTemplates();
const std::vector<SpellTemplate>& SpellTemplates();
const std::vector<HeroTemplate>& HeroTemplates();
const std::vector<BossTemplate>& Bosses();
const std::vector<Card>& BuildDeck();
const std::vector<int>& OrdinaryHeroDeck();
const std::vector<int>& EpicHeroDeck();
const std::vector<std::string>& ClassNames();

class BossMonsterGame;

class BossMonsterState : public State {
 public:
  BossMonsterState(std::shared_ptr<const Game> game, int boss0, int boss1);
  BossMonsterState(const BossMonsterState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Accessors, mainly for tests and for UIs driving the game.
  int round() const { return round_; }
  int wounds(Player player) const { return wounds_[player]; }
  int souls(Player player) const { return souls_[player]; }
  int boss(Player player) const { return boss_[player]; }
  int max_rooms(Player player) const { return max_rooms_[player]; }
  int deck_size() const { return deck_.size(); }
  const std::vector<int>& rooms(Player player) const { return rooms_[player]; }
  const std::vector<Card>& hand(Player player) const { return hands_[player]; }
  const std::vector<int>& waiting_heroes() const { return waiting_; }
  // The phase the game is waiting on, e.g. "build" or "discard".
  std::string PhaseString() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  BossAbility Ability(Player player) const;
  int HandLimit(Player player) const;
  int WoundLimit(Player player) const;

  // Turn structure.
  void Advance();
  void StartOrEndRound();
  void ResolveRound();
  void FinishByRoundLimit();
  void CheckTerminal();

  // Card handling.
  void RecycleBuildDiscard();
  void RecycleHeroDiscard();
  void CastSpell(Player player, int spell_template);
  // Returns the player heroes tied between dungeons should favor, or
  // kInvalidPlayer if ties leave the Hero waiting.
  Player TiebreakFavor() const;

  std::array<int, kNumPlayers> boss_;
  std::array<int, kNumPlayers> max_rooms_;

  std::vector<Card> deck_;
  std::vector<Card> build_discard_;

  std::vector<int> hero_pool_;
  std::vector<int> epic_pool_;
  bool epic_merged_ = false;
  std::vector<int> hero_discard_;  // Heroes that dealt a Wound; recyclable.
  std::vector<int> waiting_;       // Revealed, not yet lured into a dungeon.

  std::array<std::vector<Card>, kNumPlayers> hands_;
  std::array<std::vector<int>, kNumPlayers> rooms_;  // Room ids, build order.
  std::array<int, kNumPlayers> wounds_;
  std::array<int, kNumPlayers> souls_;
  std::array<uint8_t, kNumPlayers> round_effects_;

  int round_ = 1;
  bool is_terminal_ = false;
  std::vector<double> returns_;
  std::deque<Step> queue_;
};

class BossMonsterGame : public Game {
 public:
  explicit BossMonsterGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kNumBuildCards; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  // Generous upper bound: kMaxRounds rounds of at most ~20 actions each,
  // plus the opening deal.
  int MaxGameLength() const override { return 1200; }

  int boss(Player player) const { return boss_[player]; }

 private:
  std::array<int, kNumPlayers> boss_;
};

}  // namespace boss_monster
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BOSS_MONSTER_BOSS_MONSTER_H_
