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
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace boss_monster {
namespace {

namespace testing = open_spiel::testing;

void BasicBossMonsterTests() {
  testing::LoadGameTest("boss_monster");
  testing::ChanceOutcomesTest(*LoadGame("boss_monster"));
  testing::RandomSimTest(*LoadGame("boss_monster"), 25);
  // A few different boss pairings, to exercise every boss ability.
  for (int boss0 = 0; boss0 < kNumBosses; ++boss0) {
    const int boss1 = (boss0 + 1) % kNumBosses;
    testing::RandomSimTest(
        *LoadGame(absl::StrCat("boss_monster(boss0=", boss0, ",boss1=", boss1,
                               ")")),
        3);
  }
}

// The deck composition should match the physical base set.
void CardDataTests() {
  SPIEL_CHECK_EQ(BuildDeck().size(), kNumBuildCards);
  SPIEL_CHECK_EQ(OrdinaryHeroDeck().size(), kNumOrdinaryHeroes);
  SPIEL_CHECK_EQ(EpicHeroDeck().size(), kNumEpicHeroes);
  SPIEL_CHECK_EQ(Bosses().size(), kNumBosses);

  int num_rooms = 0;
  int num_spells = 0;
  for (const Card& card : BuildDeck()) {
    if (card.kind == CardKind::kRoom) {
      ++num_rooms;
    } else {
      ++num_spells;
    }
  }
  SPIEL_CHECK_EQ(num_rooms, kNumRoomCards);
  SPIEL_CHECK_EQ(num_spells, kNumSpellCards);

  // Every class must be represented among rooms and heroes, or the Bait
  // phase could never lure some Hero class.
  std::vector<int> room_classes(kNumClasses, 0);
  for (const RoomTemplate& room : RoomTemplates()) ++room_classes[room.cls];
  std::vector<int> hero_classes(kNumClasses, 0);
  for (const HeroTemplate& hero : HeroTemplates()) ++hero_classes[hero.cls];
  for (int c = 0; c < kNumClasses; ++c) {
    SPIEL_CHECK_GT(room_classes[c], 0);
    SPIEL_CHECK_GT(hero_classes[c], 0);
  }
}

// Bosses grant their advertised passives.
void BossAbilityTests() {
  // Robobo (7) may build a 6th room; Draculord (5) survives to 6 Wounds.
  std::shared_ptr<const Game> game =
      LoadGame("boss_monster(boss0=7,boss1=5)");
  std::unique_ptr<State> state = game->NewInitialState();
  const auto* bm_state = static_cast<const BossMonsterState*>(state.get());
  SPIEL_CHECK_EQ(bm_state->max_rooms(0), kBaseMaxRooms + 1);
  SPIEL_CHECK_EQ(bm_state->max_rooms(1), kBaseMaxRooms);

  // Cleopatra (2) opens with an extra card, so player 0's opening deal is one
  // draw longer than player 1's.
  std::shared_ptr<const Game> game2 =
      LoadGame("boss_monster(boss0=2,boss1=0)");
  std::unique_ptr<State> state2 = game2->NewInitialState();
  const auto* bm2 = static_cast<const BossMonsterState*>(state2.get());
  std::mt19937 rng(12345);
  // Deal out the opening hands (all chance nodes at the start of the game).
  while (state2->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state2->ChanceOutcomes();
    state2->ApplyAction(outcomes[rng() % outcomes.size()].first);
  }
  SPIEL_CHECK_EQ(bm2->hand(0).size(), kStartingHand + 1 + 1);  // +1 opening,
  SPIEL_CHECK_EQ(bm2->hand(1).size(), kStartingHand + 1);      // +1 round-1.
}

// Play out full games and check the terminal conditions the rules promise.
void TerminalConditionTests() {
  std::shared_ptr<const Game> game = LoadGame("boss_monster");
  std::mt19937 rng(9876);
  for (int i = 0; i < 30; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    int num_actions = 0;
    while (!state->IsTerminal()) {
      std::vector<Action> actions;
      if (state->IsChanceNode()) {
        for (const auto& [action, prob] : state->ChanceOutcomes()) {
          actions.push_back(action);
        }
      } else {
        actions = state->LegalActions();
        SPIEL_CHECK_FALSE(actions.empty());
      }
      state->ApplyAction(actions[rng() % actions.size()]);
      ++num_actions;
      SPIEL_CHECK_LE(num_actions, game->MaxGameLength());
    }
    const auto* bm = static_cast<const BossMonsterState*>(state.get());
    const std::vector<double> returns = state->Returns();
    SPIEL_CHECK_EQ(returns.size(), kNumPlayers);
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
    for (double r : returns) {
      SPIEL_CHECK_TRUE(r == -1.0 || r == 0.0 || r == 1.0);
    }
    // A game ends on a Wound-out, a Soul win, or the round cap -- never
    // silently mid-round.
    const bool wound_out =
        bm->wounds(0) >= kWoundLimit || bm->wounds(1) >= kWoundLimit;
    const bool soul_win =
        bm->souls(0) >= kSoulTarget || bm->souls(1) >= kSoulTarget;
    SPIEL_CHECK_TRUE(wound_out || soul_win || bm->round() > kMaxRounds);
    // Neither dungeon may exceed its room limit.
    for (Player p = 0; p < kNumPlayers; ++p) {
      SPIEL_CHECK_LE(static_cast<int>(bm->rooms(p).size()), bm->max_rooms(p));
    }
  }
}

// Observation tensors must be the advertised size and perspective-correct.
void ObservationTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("boss_monster");
  SPIEL_CHECK_EQ(game->ObservationTensorSize(), kObservationTensorSize);
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(4242);
  for (int step = 0; step < 60 && !state->IsTerminal(); ++step) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes[rng() % outcomes.size()].first);
    } else {
      std::vector<Action> actions = state->LegalActions();
      state->ApplyAction(actions[rng() % actions.size()]);
    }
    for (Player p = 0; p < kNumPlayers; ++p) {
      std::vector<float> values = state->ObservationTensor(p);
      SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
      for (float v : values) {
        SPIEL_CHECK_FALSE(std::isnan(v));
        SPIEL_CHECK_GE(v, 0.0f);
        SPIEL_CHECK_LE(v, 2.0f);
      }
    }
  }
}

// Cloning must be a deep copy: playing on the clone must not disturb the
// original (AlphaZero's tree search depends on this).
void CloneTests() {
  std::shared_ptr<const Game> game = LoadGame("boss_monster");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(77);
  for (int i = 0; i < 40 && !state->IsTerminal(); ++i) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes[rng() % outcomes.size()].first);
    } else {
      state->ApplyAction(state->LegalActions()[0]);
    }
  }
  std::unique_ptr<State> clone = state->Clone();
  const std::string before = state->ToString();
  SPIEL_CHECK_EQ(clone->ToString(), before);
  while (!clone->IsTerminal()) {
    if (clone->IsChanceNode()) {
      clone->ApplyAction(clone->ChanceOutcomes()[0].first);
    } else {
      clone->ApplyAction(clone->LegalActions()[0]);
    }
  }
  SPIEL_CHECK_EQ(state->ToString(), before);
  SPIEL_CHECK_TRUE(clone->IsTerminal());
}

}  // namespace
}  // namespace boss_monster
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::boss_monster::CardDataTests();
  open_spiel::boss_monster::BasicBossMonsterTests();
  open_spiel::boss_monster::BossAbilityTests();
  open_spiel::boss_monster::TerminalConditionTests();
  open_spiel::boss_monster::ObservationTensorTests();
  open_spiel::boss_monster::CloneTests();
}
