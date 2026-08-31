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

"""Tests for Python Boss Monster."""

import numpy as np
from absl.testing import absltest

from open_spiel.python.games import boss_monster
from open_spiel.python.observation import make_observation
import pyspiel


class BossMonsterTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    game = boss_monster.BossMonsterGame()
    state = game.new_initial_state()
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)

  def test_random_games_terminate_with_valid_returns(self):
    game = pyspiel.load_game("python_boss_monster")
    rng = np.random.RandomState(1234)
    for _ in range(20):
      state = game.new_initial_state()
      num_actions = 0
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes, probs = zip(*state.chance_outcomes())
          action = rng.choice(outcomes, p=probs)
        else:
          legal = state.legal_actions()
          action = rng.choice(legal)
        state.apply_action(action)
        num_actions += 1
        self.assertLess(num_actions, 5000, msg=str(state))
      returns = state.returns()
      self.assertEqual(len(returns), 2)
      self.assertAlmostEqual(sum(returns), 0.0)
      for r in returns:
        self.assertIn(r, (-1.0, 0.0, 1.0))

  def test_game_from_cc_random_sim(self):
    """Runs OpenSpiel's standard API-conformance random-sim test."""
    game = pyspiel.load_game("python_boss_monster")
    pyspiel.random_sim_test(
        game, num_sims=5, serialize=False, verbose=False)

  def test_observation_tensor_shapes_and_range(self):
    game = pyspiel.load_game("python_boss_monster")
    state = game.new_initial_state()
    rng = np.random.RandomState(7)
    while state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(rng.choice(outcomes, p=probs))
    for player in range(2):
      obs = np.asarray(state.observation_tensor(player))
      self.assertTrue(np.all(np.isfinite(obs)))

  def test_bosses_confer_distinct_abilities(self):
    game = boss_monster.BossMonsterGame({"boss0": 7, "boss1": 5})
    state = game.new_initial_state()
    # Robobo (index 7): extra room slot.
    self.assertEqual(state._max_rooms[0], 6)
    # Draculord (index 5): +1 wound limit.
    self.assertEqual(state._wound_limit(1), 6)

  def test_win_loss_thresholds(self):
    game = boss_monster.BossMonsterGame()
    state = game.new_initial_state()
    state._wounds[0] = 5
    state._check_terminal()
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [-1.0, 1.0])

  def test_soul_win(self):
    game = boss_monster.BossMonsterGame()
    state = game.new_initial_state()
    state._souls[1] = 10
    state._check_terminal()
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [-1.0, 1.0])

  def test_make_observation(self):
    game = pyspiel.load_game("python_boss_monster")
    state = game.new_initial_state()
    rng = np.random.RandomState(3)
    while state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(rng.choice(outcomes, p=probs))
    observation = make_observation(game)
    observation.set_from(state, player=0)
    self.assertTrue(np.all(np.isfinite(observation.tensor)))


if __name__ == "__main__":
  absltest.main()
