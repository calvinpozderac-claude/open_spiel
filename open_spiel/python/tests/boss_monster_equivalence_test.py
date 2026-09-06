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

"""Checks that C++ `boss_monster` and Python `python_boss_monster` agree.

The C++ game (open_spiel/games/boss_monster/) is a performance port of the
Python one (open_spiel/python/games/boss_monster.py). They are written to be
action-for-action identical, so feeding the same action sequence to both must
produce the same legal actions, chance outcomes, observations, and returns at
every step. This test is what keeps the two from drifting apart.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

_NUM_PLAYERS = 2


def _chance_outcomes_match(state_cc, state_py):
  outcomes_cc = state_cc.chance_outcomes()
  outcomes_py = state_py.chance_outcomes()
  if len(outcomes_cc) != len(outcomes_py):
    return False, f"{len(outcomes_cc)} vs {len(outcomes_py)} outcomes"
  for (a_cc, p_cc), (a_py, p_py) in zip(outcomes_cc, outcomes_py):
    if a_cc != a_py or abs(p_cc - p_py) > 1e-9:
      return False, f"({a_cc},{p_cc}) vs ({a_py},{p_py})"
  return True, ""


class BossMonsterEquivalenceTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 1), (2, 3), (4, 5), (6, 7), (7, 0), (5, 2), (3, 4), (1, 6),
  )
  def test_cc_and_python_implementations_agree(self, boss0, boss1):
    params = f"(boss0={boss0},boss1={boss1})"
    game_cc = pyspiel.load_game("boss_monster" + params)
    game_py = pyspiel.load_game("python_boss_monster" + params)

    self.assertEqual(game_cc.num_distinct_actions(),
                     game_py.num_distinct_actions())
    self.assertEqual(game_cc.observation_tensor_shape(),
                     game_py.observation_tensor_shape())
    self.assertEqual(game_cc.max_chance_outcomes(),
                     game_py.max_chance_outcomes())

    rng = np.random.RandomState(boss0 * 8 + boss1)
    for _ in range(3):
      state_cc = game_cc.new_initial_state()
      state_py = game_py.new_initial_state()
      num_actions = 0
      while not state_cc.is_terminal():
        self.assertFalse(state_py.is_terminal(), msg=str(state_cc))
        self.assertEqual(state_cc.current_player(), state_py.current_player(),
                         msg=str(state_cc))

        # Observations must agree for both players, at every step.
        for player in range(_NUM_PLAYERS):
          np.testing.assert_allclose(
              np.asarray(state_cc.observation_tensor(player)),
              np.asarray(state_py.observation_tensor(player)),
              rtol=1e-6, atol=1e-6,
              err_msg=f"observation_tensor({player}) at action {num_actions}")
          self.assertEqual(state_cc.observation_string(player),
                           state_py.observation_string(player))

        if state_cc.is_chance_node():
          self.assertTrue(state_py.is_chance_node())
          match, detail = _chance_outcomes_match(state_cc, state_py)
          self.assertTrue(match, msg=f"chance outcomes differ: {detail}")
          outcomes, probs = zip(*state_cc.chance_outcomes())
          action = rng.choice(outcomes, p=probs)
        else:
          legal_cc = state_cc.legal_actions()
          legal_py = state_py.legal_actions()
          self.assertEqual(legal_cc, legal_py, msg=str(state_cc))
          self.assertNotEmpty(legal_cc)
          action = rng.choice(legal_cc)
          self.assertEqual(
              state_cc.action_to_string(state_cc.current_player(), action),
              state_py.action_to_string(state_py.current_player(), action))

        state_cc.apply_action(action)
        state_py.apply_action(action)
        num_actions += 1
        self.assertLess(num_actions, game_cc.max_game_length())

      self.assertTrue(state_py.is_terminal(), msg=str(state_py))
      self.assertEqual(state_cc.returns(), state_py.returns())

  def test_cc_game_registered_and_playable(self):
    game = pyspiel.load_game("boss_monster")
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)


if __name__ == "__main__":
  absltest.main()
