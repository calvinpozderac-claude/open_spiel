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

Pace: **~1.8 minutes per learn step** (~1030 new states, ~24 self-play
games), i.e. about 9.3 states/s of self-play across 3 actors.

### Result after 72 steps: better fit, no better play

The run ended at step 72 / 1,785 self-play games when the container was
restarted (not a crash -- see "Why it stopped" below). That is enough data
to answer the question cleanly, using the correlation of each metric with
step number across all 72 points:

| metric | correlation with step | reading |
|---|---|---|
| policy loss | **-0.788** | strong, consistent improvement |
| return vs MCTS-25 | +0.273 | weak, within noise |
| return vs MCTS-250 | +0.028 | none |

Window means tell the same story:

| | steps 1-5 | steps 26-30 | steps 46-50 | steps 68-72 |
|---|---|---|---|---|
| policy loss | 1.061 | 0.927 | 0.932 | **0.880** |
| value loss | 0.297 | 0.187 | 0.168 | 0.181 |
| avg return vs MCTS-25 | -0.553 | -0.487 | **-0.273** | -0.467 |
| avg return vs MCTS-250 | -0.400 | -0.600 | -0.573 | -0.467 |

**The apparent improvement at step 50 did not hold.** Return against the
equal-budget opponent bounced to -0.273 around steps 46-50 and was back to
-0.467 by steps 68-72 -- exactly the one-standard-error wobble that the
step-50 note warned it might be. Against deeper search there is no trend
whatsoever across the whole run (r = +0.03).

So the conclusion is the one the step-30 data suggested, now on 72 steps
rather than 30 and with the noise properly accounted for: **the network
gets steadily and measurably better at fitting its training targets
(r = -0.79) while getting no better at winning.** That dissociation, not
the raw loss curve, is the result.

Mechanically everything is fine — win split stays balanced (no first-player
degeneracy), game lengths hold at ~40-46 moves, the value head is accurate
late-game, and no policy mass leaks onto illegal actions. So this is not a
broken pipeline; it is a pipeline that needs far more than 1,785 self-play
games, and quite possibly a different approach. Candidate reasons, roughly
in order of how much I'd bet on them:

1. **Nowhere near enough data.** 72 steps is ~76K states / 1,785 games.
   AlphaZero results are quoted in millions of games.
2. **Hidden information is fought, not modeled.** MCTS here clones the full
   state, so it searches *while seeing the opponent's hand*, and then trains
   the network toward those policy targets — but the network's observation
   deliberately hides that hand. It is being asked to regress targets that
   depend on information it cannot see, which puts a hard floor under the
   policy loss no matter how long it trains. A policy loss that stalls at
   ~0.93 while the value loss keeps improving is consistent with exactly
   this: the value of a position is largely predictable from public state,
   the opponent's best reply often is not.
3. **25 simulations is shallow** for 40+ move games, so the targets
   themselves are weak and noisy.
4. **Heavy stochasticity.** Every card draw and Hero reveal is a chance
   node, so outcomes are high-variance and the value signal is noisy.

If you want a genuinely strong Boss Monster agent, (2) is the one to take
seriously: an information-set method (e.g. Deep CFR, R-NaD, or a
determinized/ISMCTS variant) fits this game better than vanilla AlphaZero.
AlphaZero here is best understood as a solid engineering baseline on top of
a correct, fast game implementation.

### Replication (run 2, independent seed)

After the container restart the run was relaunched from scratch with the
same config and a fresh seed, which turns "re-derive the same curve" into a
useful control. Comparing the two runs over their first 21 steps:

| window | run 1 policy | run 2 policy | run 1 vs25 | run 2 vs25 | run 1 vs250 | run 2 vs250 |
|---|---|---|---|---|---|---|
| 1-5 | 1.061 | 1.053 | -0.553 | -0.387 | -0.400 | -0.307 |
| 6-10 | 0.975 | 1.042 | -0.553 | -0.600 | -0.653 | -0.600 |
| 11-15 | 0.978 | 0.969 | -0.633 | -0.493 | -0.627 | -0.447 |
| 16-20 | 0.972 | 0.967 | -0.493 | -0.587 | -0.567 | -0.373 |

Average game length: 43.0 (run 1) vs 43.1 (run 2).

**The learning side replicates; the winning side does not exist to
replicate.** The policy-loss curve tracks closely across independent seeds
and game length matches to a tenth of a move, while the evaluation numbers
scatter differently in each run inside the same -0.4 to -0.65 band, trending
in neither. That is the cleanest available evidence that the dissociation
above is systematic rather than a quirk of one seed -- and that run 1's
step-50 bump was seed noise, since run 2 produces its own, differently
placed bumps.

### Mid-run snapshot (step 10), kept for reference

At step 10 the picture looked encouraging in isolation -- loss 1.187 (from
1.575), games averaging 45.6 moves, a balanced 14/9/0 win split, and a value
head already near-perfect late-game (MSE 0.0034) though weak early (1.004).
Read on its own it looked like "healthy and learning". The 72-step analysis
above is why that reading was premature: the loss really was falling, but
it was not turning into wins.

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
model.load_checkpoint(70)          # by step number, not path

state = game.new_initial_state()
mask = np.zeros(game.num_distinct_actions(), np.bool_)
mask[state.legal_actions()] = True
value, policy = model.inference(
    [np.asarray(state.observation_tensor(0), np.float32)], [mask])
```

Pair it with `open_spiel.python.algorithms.alpha_zero.evaluator
.AlphaZeroEvaluator` and `mcts.MCTSBot` to actually play it.

### Why it stopped where it did

It did not crash. The run was executing in an ephemeral container, which was
restarted at 03:37 while the learner sat in "Collecting trajectories" after
step 72; the log simply ends there, with no error and no OOM. The disk
survived, so these artifacts (final checkpoint at step 70, config, full
per-step metrics, learner log) are committed here. To take it further, use a GPU box (the ~14ms per batch-1
inference that dominates this run is mostly CPU dispatch overhead) or the
C++ AlphaZero in `open_spiel/algorithms/alpha_zero_torch`, which keeps the
whole search loop out of Python.
