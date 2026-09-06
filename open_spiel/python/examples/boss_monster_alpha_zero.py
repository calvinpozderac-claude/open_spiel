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

"""Train an AlphaZero agent on Boss Monster.

This mirrors `open_spiel/python/examples/alpha_zero.py` but is pre-wired for
Boss Monster. It defaults to the C++ implementation
(`open_spiel/games/boss_monster/`, game name "boss_monster"), which is the
one to train on. Measured with
`open_spiel/python/examples/boss_monster_benchmark.py` on this repo: the C++
state is ~280x faster to `clone()` (the per-simulation cost MCTS pays), which
works out to ~2.7x more MCTS moves/second end to end. The gap between those
two numbers is Python: OpenSpiel's MCTS driver and the pybind boundary are
still interpreted, and are now the bottleneck. Pass
`--game=python_boss_monster` to train against the Python implementation
instead (the two are kept action-for-action identical; see
`open_spiel/python/tests/boss_monster_equivalence_test.py`).

Notes for training this particular game:
  * Boss Monster has hidden information (each player's hand). AlphaZero's
    MCTS clones the full game state, including the opponent's hand, so this
    is "cheating" (perfect-information-Monte-Carlo-style) self-play, a
    common and pragmatic simplification for card games -- not a faithful
    imperfect-information solver. Treat the resulting policy as a strong
    heuristic bot, not a game-theoretically sound one.
  * Boss Monster games are long (tens of decisions per player plus a chance
    node for every card drawn or Hero revealed), so `temperature_drop` is
    set much higher than the Tic-Tac-Toe default.
  * Run the tests first if you've modified the game:
      ./build/games/boss_monster_test                       # C++
      python3 open_spiel/python/games/boss_monster_test.py  # Python
      python3 open_spiel/python/tests/boss_monster_equivalence_test.py

Example:
  python3 open_spiel/python/examples/boss_monster_alpha_zero.py \
      --path=/tmp/boss_monster_az --actors=3 --evaluators=1 \
      --max_simulations=50 --max_steps=200
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import utils
# Importing this registers "python_boss_monster" with pyspiel.
from open_spiel.python.games import boss_monster  # pylint: disable=unused-import
from open_spiel.python.utils import spawn

flags.DEFINE_enum(
    "nn_api",
    "linen",
    ["linen", "nnx"],
    "What type of flax api should be used for training?",
)

flags.DEFINE_string(
    "game", "boss_monster",
    "Name of the game: 'boss_monster' (C++, fast) or 'python_boss_monster'.")
flags.DEFINE_float("uct_c", 1.41, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 50, "How many simulations to run.")
flags.DEFINE_integer("train_batch_size", 2**6, "Batch size for learning.")
flags.DEFINE_integer(
    "replay_buffer_size",
    2**13,
    "How many states to store in the replay buffer.",
)
flags.DEFINE_integer(
    "replay_buffer_reuse", 4, "How many times to learn from each state."
)
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-4, "L2 regularization strength.")
flags.DEFINE_bool(
    "decouple_weight_decay",
    False,
    "Whether to use explicit regulariser or decouple the weights when update",
)
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1.0, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1, "Temperature for final move selection.")
flags.DEFINE_integer(
    "temperature_drop",
    40,  # Boss Monster games run for many more actions than Tic-Tac-Toe.
    "Drop the temperature to 0 after this many moves.",
)
flags.DEFINE_enum(
    "nn_model",
    "resnet",
    utils.api_selector(utils.AVIALABLE_APIS[0]).Model.valid_model_types,
    "What type of model should be used?",
)
flags.DEFINE_integer("nn_width", 2**7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 2, "How deep should the network be.")
flags.DEFINE_string("path", None, "Where to save checkpoints.")
flags.DEFINE_integer("checkpoint_freq", 25, "Save a checkpoint every N steps.")
flags.DEFINE_integer("actors", 3, "How many actors to run.")
flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_integer(
    "evaluation_window", 30, "How many games to average results over."
)
flags.DEFINE_integer(
    "eval_levels",
    5,
    (
        "Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
        " simulations for n in range(eval_levels)."
    ),
)
flags.DEFINE_integer("max_steps", 200, "How many learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def main(unused_argv):
  config = alpha_zero.Config(
      game=FLAGS.game,
      path=FLAGS.path,
      learning_rate=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay,
      decouple_weight_decay=FLAGS.decouple_weight_decay,
      train_batch_size=FLAGS.train_batch_size,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_buffer_reuse=FLAGS.replay_buffer_reuse,
      max_steps=FLAGS.max_steps,
      checkpoint_freq=FLAGS.checkpoint_freq,
      actors=FLAGS.actors,
      evaluators=FLAGS.evaluators,
      uct_c=FLAGS.uct_c,
      max_simulations=FLAGS.max_simulations,
      policy_alpha=FLAGS.policy_alpha,
      policy_epsilon=FLAGS.policy_epsilon,
      temperature=FLAGS.temperature,
      temperature_drop=FLAGS.temperature_drop,
      evaluation_window=FLAGS.evaluation_window,
      eval_levels=FLAGS.eval_levels,
      nn_model=FLAGS.nn_model,
      nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth,
      observation_shape=None,
      output_size=None,
      quiet=FLAGS.quiet,
      verbose=FLAGS.verbose,
      nn_api_version=FLAGS.nn_api,
  )

  alpha_zero.alpha_zero(config)


if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)
