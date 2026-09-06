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

"""Benchmarks the C++ Boss Monster game against the Python one.

Three measurements, from narrowest to most representative:
  * random playouts  -- dominated by the Python driver loop in both cases,
    so it understates the engine difference;
  * clone()          -- the raw state-copy cost, which MCTS pays once per
    simulation, and where the port wins by ~2 orders of magnitude;
  * MCTS search      -- what an AlphaZero actor actually does. Note the MCTS
    driver itself (open_spiel/python/algorithms/mcts.py) is Python, so this
    is the number that matters and it is much smaller than the clone()
    ratio: the remaining time is Python tree bookkeeping and pybind
    crossings, not the game.

Usage:
  python3 open_spiel/python/examples/boss_monster_benchmark.py --games=200
"""

import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.algorithms import mcts
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_integer("games", 200, "Number of random playouts per game impl.")
flags.DEFINE_integer("clone_reps", 20000, "Number of Clone() calls to time.")
flags.DEFINE_integer("mcts_moves", 40, "Number of MCTS-searched moves to time.")
flags.DEFINE_integer("mcts_simulations", 50, "Simulations per MCTS move.")
flags.DEFINE_integer("seed", 0, "RNG seed.")


def random_playouts(game_name, num_games, seed):
  """Plays `num_games` uniformly-random games; returns (seconds, num_states)."""
  game = pyspiel.load_game(game_name)
  rng = np.random.RandomState(seed)
  num_states = 0
  start = time.time()
  for _ in range(num_games):
    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = rng.choice(outcomes, p=probs)
      else:
        action = rng.choice(state.legal_actions())
      state.apply_action(action)
      num_states += 1
  return time.time() - start, num_states


def clone_benchmark(game_name, reps, seed):
  """Times State.clone(), the inner loop of MCTS. Returns seconds."""
  game = pyspiel.load_game(game_name)
  rng = np.random.RandomState(seed)
  state = game.new_initial_state()
  # Walk to a mid-game state, which is what the search actually copies.
  for _ in range(60):
    if state.is_terminal():
      break
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(rng.choice(outcomes, p=probs))
    else:
      state.apply_action(rng.choice(state.legal_actions()))
  start = time.time()
  for _ in range(reps):
    state.clone()
  return time.time() - start


def mcts_benchmark(game_name, num_moves, simulations, seed):
  """Times MCTS moves, which is what an AlphaZero actor actually does.

  Uses a random-rollout evaluator rather than a neural network so the timing
  reflects game-engine cost (clone / apply_action / legal_actions) rather
  than JAX. Returns seconds for `num_moves` searched moves.
  """
  game = pyspiel.load_game(game_name)
  evaluator = mcts.RandomRolloutEvaluator(1, np.random.RandomState(seed))
  bot = mcts.MCTSBot(game, 2.0, simulations, evaluator,
                     random_state=np.random.RandomState(seed))
  rng = np.random.RandomState(seed)
  state = game.new_initial_state()
  moves = 0
  start = time.time()
  while moves < num_moves:
    if state.is_terminal():
      state = game.new_initial_state()
      continue
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(rng.choice(outcomes, p=probs))
    else:
      state.apply_action(bot.step(state))
      moves += 1
  return time.time() - start


def main(_):
  results = {}
  for name in ["boss_monster", "python_boss_monster"]:
    seconds, num_states = random_playouts(name, FLAGS.games, FLAGS.seed)
    clone_seconds = clone_benchmark(name, FLAGS.clone_reps, FLAGS.seed)
    mcts_seconds = mcts_benchmark(name, FLAGS.mcts_moves,
                                  FLAGS.mcts_simulations, FLAGS.seed)
    results[name] = (seconds, num_states, clone_seconds, mcts_seconds)
    print(f"{name}:")
    print(f"  {FLAGS.games} random games: {seconds:.2f}s "
          f"({FLAGS.games / seconds:,.0f} games/s, "
          f"{num_states / seconds:,.0f} states/s)")
    print(f"  {FLAGS.clone_reps} clones: {clone_seconds:.2f}s "
          f"({FLAGS.clone_reps / clone_seconds:,.0f} clones/s)")
    print(f"  {FLAGS.mcts_moves} MCTS moves @ {FLAGS.mcts_simulations} sims: "
          f"{mcts_seconds:.2f}s "
          f"({FLAGS.mcts_moves / mcts_seconds:,.1f} moves/s)")

  cc, py = results["boss_monster"], results["python_boss_monster"]
  print(f"\nSpeedup (C++ vs Python): "
        f"{py[0] / cc[0]:.1f}x on playouts, "
        f"{py[2] / cc[2]:.1f}x on clone(), "
        f"{py[3] / cc[3]:.1f}x on MCTS search")


if __name__ == "__main__":
  app.run(main)
