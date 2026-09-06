# Boss Monster AlphaZero training runs

Saved artifacts from AlphaZero self-play on the C++ `boss_monster` game. See
`open_spiel/python/examples/boss_monster_README.md` for the game itself and
`open_spiel/python/examples/boss_monster_alpha_zero.py` for the trainer.

## `run-2026-09-06`

First real run, on a 4-core CPU container. Config (also in `config.json`):

| | |
|---|---|
| game | `boss_monster` (C++) |
| model | `mlp(128, 2)` — 82,828 variables |
| MCTS simulations / move | 25 |
| actors / evaluators | 3 / 1 |
| replay buffer / reuse | 4096 / 4 |
| batch size | 64 |
| temperature drop | move 40 |

Pace: **~1.85 minutes per learn step** (~1030 new states, ~24 self-play
games), i.e. about 9.3 states/s of self-play across 3 actors.

### Result after 20 steps: the loop works, the agent does not improve

This is the honest headline, and it matters more than the loss curve.

| | steps 1-5 | steps 16-20 |
|---|---|---|
| policy loss | 1.061 | 0.972 |
| avg return vs MCTS-25 | -0.553 | -0.493 |
| avg return vs MCTS-250 | -0.400 | -0.567 |

Loss fell quickly over the first ~6 steps (1.575 → 1.21) and then **flat-lined
for the next 14**. Strength against the MCTS baselines never went anywhere:
it is roughly where it started, and against the stronger 250-simulation
opponent it is slightly *worse*. Notably the step-1 evaluation (-0.27 /
-0.17), taken with an essentially untrained network, is better than
everything from steps 3-15. That is not a learning curve.

Mechanically everything is fine — win split stays balanced (no first-player
degeneracy), game lengths hold at ~40-46 moves, the value head is accurate
late-game, and no policy mass leaks onto illegal actions. So this is not a
broken pipeline; it is a pipeline that needs far more than 492 self-play
games, and quite possibly a different approach. Candidate reasons, roughly
in order of how much I'd bet on them:

1. **Nowhere near enough data.** 20 steps is ~21K states / 492 games.
   AlphaZero results are quoted in millions of games.
2. **Hidden information is fought, not modeled.** MCTS here clones the full
   state, so it searches *while seeing the opponent's hand*, and then trains
   the network toward those policy targets — but the network's observation
   deliberately hides that hand. It is being asked to regress targets that
   depend on information it cannot see, which puts a hard floor under the
   policy loss no matter how long it trains. A ~0.97 policy loss that will
   not move is consistent with exactly this.
3. **25 simulations is shallow** for 40+ move games, so the targets
   themselves are weak and noisy.
4. **Heavy stochasticity.** Every card draw and Hero reveal is a chance
   node, so outcomes are high-variance and the value signal is noisy.

If you want a genuinely strong Boss Monster agent, (2) is the one to take
seriously: an information-set method (e.g. Deep CFR, R-NaD, or a
determinized/ISMCTS variant) fits this game better than vanilla AlphaZero.
AlphaZero here is best understood as a solid engineering baseline on top of
a correct, fast game implementation.

### What the numbers looked like at step 10

| metric | value |
|---|---|
| loss (total / policy / value) | 1.187 / 0.963 / 0.188 (from 1.575 / 1.132 / 0.392 at step 1) |
| game length | avg 45.6 moves (min 29, max 73) |
| win split (P1 / P2 / draw) | 14 / 9 / 0 |
| value-head MSE, early game | 1.004 |
| value-head MSE, late game | 0.0034 |
| avg return vs MCTS (25 / 250 sims) | -0.47 / -0.57 |
| total self-play states | 10,445 |

Read that as: the loop is healthy and learning (loss falling, no degenerate
first-player bias, sane game lengths, and a value head that is already
near-perfect late in a game where the outcome is largely determined) but the
agent is still **losing to a plain MCTS baseline** — which is exactly what
10 learn steps and ~250 self-play games should look like. This is a
correctly-wired pipeline caught early, not a trained agent.

### Using the checkpoint

The saved checkpoint is for **inference**, not for resuming training:
upstream `alpha_zero.py` always initializes a fresh model in the learner, so
continuing this run would need code changes.

Two upstream gotchas make loading less obvious than it should be, so this
snippet is one that has actually been run against the committed checkpoint:

* `Model.from_checkpoint()` is **broken** — it calls the instance method
  `load_checkpoint(path)` as though it were a classmethod, when the real
  signature takes a *step number*. Build the model, then load by step.
* The checkpoint path must be **absolute**, or orbax raises
  `Checkpoint path should be absolute`.

```python
import json, os
import numpy as np, pyspiel
from open_spiel.python.algorithms.alpha_zero import utils

run = os.path.abspath("boss_monster_training/run-2026-09-06")
cfg = json.load(open(f"{run}/config.json"))
game = pyspiel.load_game(cfg["game"])

model_lib = utils.api_selector("linen")
model = model_lib.Model.build_model(
    cfg["nn_model"], game.observation_tensor_shape(),
    game.num_distinct_actions(), nn_width=cfg["nn_width"],
    nn_depth=cfg["nn_depth"], weight_decay=cfg["weight_decay"],
    learning_rate=cfg["learning_rate"], path=run)
model.load_checkpoint(10)          # by step number, not path

state = game.new_initial_state()
mask = np.zeros(game.num_distinct_actions(), np.bool_)
mask[state.legal_actions()] = True
value, policy = model.inference(
    [np.asarray(state.observation_tensor(0), np.float32)], [mask])
```

Pair it with `open_spiel.python.algorithms.alpha_zero.evaluator
.AlphaZeroEvaluator` and `mcts.MCTSBot` to actually play it.

### Why it stopped where it did

The run was executing in an ephemeral container; these artifacts were
committed so the work survives the container, not because the run reached a
natural end. To take it further, use a GPU box (the ~14ms per batch-1
inference that dominates this run is mostly CPU dispatch overhead) or the
C++ AlphaZero in `open_spiel/algorithms/alpha_zero_torch`, which keeps the
whole search loop out of Python.
