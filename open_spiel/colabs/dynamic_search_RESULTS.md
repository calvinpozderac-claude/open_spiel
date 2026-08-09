# Dynamic search: does it work?

Connect 4, MM (`additive_mle` search and target) throughout, 4 arms x 3 seeds x
1000 episodes, 32/3/8 trunk, then a 12-network round robin at a fixed 64
simulations (20 games per pair, ~220 games per network).

Reproduce with `python dynamic_search_experiment.py 1000 20 64`.

## Pooled by arm

| arm | rule | Elo (mean ± sd over 3 seeds) | wall s | sims/mv | vs fixed time | vs fixed sims |
|---|---|---|---|---|---|---|
| fixed | — | 14 ± 24 | 1169 | 75.0 | 1.00x | 1.00x |
| full | full | 4 ± 14 | 1161 | 68.4 | 0.99x | 0.91x |
| naive_post | naive_posterior | −7 ± 33 | 1220 | 67.2 | 1.04x | 0.90x |
| drift | drift | −12 ± 39 | 1158 | 67.5 | 0.99x | 0.90x |

## Verdict: no measurable benefit, and no measurable harm

The whole field spans 26 Elo and every seed standard deviation is comparable
to or larger than that.  Within an arm the seeds spread further than the arms
do from each other — `fixed` runs scored 35, 19 and −12; `drift` runs scored
18, 3 and −56.  Nothing here separates.

The design's own control makes the same point from the other side.
`naive_posterior` is the rule this project argues is broken, and it lands
between `full` and `drift`.  The statistic it reads really is inflated —
P(best) 0.92 where the honest rule says 0.51, pinned in the unit tests — but
that difference does not reach the trained network at this scale.  "The naive
rules are bad" is established for what the number MEANS, not for training
outcomes.

## Cost: a real 10% saving that mostly evaporates

Every adaptive arm spent about 0.90x the simulations, consistently, across all
three seeds.  End-to-end wall clock moved by 1%.

Self-play is not the whole run.  Measured in isolation the same rules cut
self-play wall clock to 0.81x (`full`) and 0.72x (`drift`), but training steps,
the optimiser and evaluation do not shrink with the simulation count, so a 10%
self-play saving is ~1% of a training run.  An adaptive budget has to save a
lot more than 10% before it is worth anything end to end.

## The diagnosis — CORRECTED, and it is the opposite of what this file first said

The original write-up claimed "no search on any seed in any arm ever reached
its ceiling", from `stop_ceiling` reading 0.00 across all nine adaptive runs.
**That measurement was broken.**  The stop reason was recorded only when the
RULE ended a search; the driver's own `n < cap` check ends a search without
calling `update()` at all, leaving `reason` empty and the position uncounted.
Every ceiling stop was therefore invisible, and the mix looked like pure
convergence because convergence stops were the only ones being counted.

With every position attributed, the picture inverts: **93-94% of searches run
to their ceiling.**  The gates almost never fire.  The rules are not stopping
early and reallocating nothing -- they are barely stopping at all, and the
~10% simulation saving comes from the pool's own floor/ceiling arithmetic
rather than from any gate opening.

So the reason the arms are indistinguishable is not "they all do the same
small thing because the ceiling is unreachable".  It is that the gates are
essentially inert at these thresholds, and every arm therefore runs close to
its allotted budget.  Which of those two stories is true changes what to do
next entirely, and the first one was an artefact of the instrumentation.

## Where to look next, if anywhere

1. **Re-run the whole comparison with the fixed accounting.** The Elo result
   stands -- it never depended on the stop mix -- but every statement in this
   file about WHY was derived from a broken counter, and the numbers above are
   from a short diagnostic rather than the 12-run experiment.
2. **The gates are too slack, not too eager.** With 93% of searches hitting
   the ceiling, the thing to measure is the distribution of `p_best` at the
   moment a search ends, and how far the disagreement spread actually falls
   over a search.  If it barely falls, no threshold on it will ever
   discriminate and the approach is dead on that ground alone.
3. **A 10% simulation saving is not worth pursuing** on its own, and it is not
   currently coming from the mechanism it was supposed to come from.
3. **Scale and game are untested.** This is Connect 4 at 1000 episodes.
   Othello has ~8.5 legal moves against 7, near-zero draws, and 60-ply games;
   the disagreement structure the gates read could behave differently.  But
   given the mechanism is inert here for a structural reason, re-running it
   elsewhere unchanged is unlikely to be informative.

## What this pilot can and cannot resolve

With ~220 games per network and 3 seeds it can separate arm effects of roughly
50 Elo.  A genuine ±20 Elo effect would be invisible.  The claim is therefore
"no large effect", not "no effect".
