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


# Follow-up: why the gates are inert (dynamic_search_sweep.py)

Measured at 500 simulations per position, reading the statistic at the moment a
full search ENDS:

    p_best        median 0.209    >= 0.95 on 12% of positions
    top gap       median 0.052
    mean spread   median 0.506
    LUCB separated on 0% of positions (c = 2)

**After 500 simulations the disagreement spread is still ~0.5 on a value scale
of [-1, 1], and the leader is favoured 21% of the time.**  The spread does not
narrow with search.

That is not a threshold problem.  The disagreement spread was chosen *because*
it is independent of the visit count -- that is what makes it immune to the
1/sqrt(n) trap this whole module is built around.  The same property means it
carries no information about how much search has been done, so no threshold on
it can ever say "enough".  A decision gate reading it is inert by construction,
which is why 93% of searches ran to their ceiling and why every rule keyed to it
lands at 0.93-0.95x of nominal no matter how the knobs move:

    fixed                     500.0   1.00x   ceiling 0.00  converged 0.00
    lucb c=1.0                469.1   0.94x   ceiling 0.86  converged 0.09
    lucb c=2.0                466.5   0.93x   ceiling 0.93  converged 0.01
    lucb c=3.0                468.9   0.94x   ceiling 0.94  converged 0.00
    full p=0.95               468.9   0.94x   ceiling 0.94  converged 0.00
    full p=0.99               468.9   0.94x   ceiling 0.94  converged 0.00
    block_lucb c=2 d=0.05     468.9   0.94x   ceiling 0.94  converged 0.00
    block d=0.12              474.2   0.95x   ceiling 0.16  converged 0.79
    block d=0.05              457.7   0.92x   ceiling 0.32  converged 0.59
    block d=0.02              429.0   0.86x   ceiling 0.35  converged 0.50

The one rule whose gate actually opens is **block**, and it is the only one that
does not read the spread.  It measures how far a fresh block of simulations
MOVES the belief, and movement does fall as a tree stabilises even though
dispersion does not.  Its converged fraction responds to `delta` in the right
direction (0.79 -> 0.59 -> 0.50 as delta tightens) where every spread-based rule
is pinned near zero.

There is no threshold that puts the average near 500 from below, because none of
these rules stops early enough to matter: the whole family sits within 14% of
the fixed budget.  Tuning further would be tuning a gate that does not open.

## Consequence

The p_best / LUCB / indifference family should be considered dead.  It rests on
a statistic that cannot, even in principle, signal that a search is finished.
If dynamic search is worth another run it is `block` alone, and the first thing
to establish is whether its per-position simulation counts actually SPREAD --
a mean of 0.86x is equally consistent with "every position trimmed 14%" (worth
nothing) and with "half the positions halved, the rest doubled" (the point of
the exercise).  The summary only reports the mean, so that is not yet known.

The wall-clock column in the sweep is unreliable; this box was contended and the
same configuration varied by more than 2x between runs.


# Final: the block rule does not allocate adaptively either

Per-position spend as a ratio of what a fixed budget would have given that
position, nominal 500 simulations:

    rule                   p05    p25    p50    p75    p95   <0.8x  >1.25x   mean
    block d=0.02          0.00   0.99   0.99   1.00   1.02    0.15    0.00    429
    block d=0.05          0.08   0.99   0.99   1.00   1.02    0.07    0.00    466
    block d=0.12          0.78   0.99   0.99   0.99   1.03    0.05    0.00    474
    block d=0.12 blk50    0.23   0.50   0.50   0.63   0.79    0.95    0.03    283
    block d=0.20 blk50    0.50   0.50   0.50   0.50   0.63    0.99    0.00    258

This settles the question the mean could not.  It is "every position trimmed",
not "half halved and the rest doubled" -- and it is more degenerate than that.

At `block_sims` = 100 the whole interquartile range sits at 0.99-1.00: nearly
every position spends its exact nominal budget, and the sub-0.8x tail is
terminal and solved positions rather than the rule deciding anything.  At
`block_sims` = 50 the distribution collapses onto a different spike, 0.50, with
95-99% of positions below 0.8x.

**`>1.25x` is 0.00 in every configuration.**  Not one position in any setting
received materially more search than the fixed budget would have given it.  The
reallocation half of the design produces nothing, and this time it is not an
accounting artefact -- the counter was fixed first and the distribution is
recorded per position.

The rule is behaving as a near-uniform multiplier on the simulation count, set
by `block_sims` rather than by anything about the position.  A multiplier is
available for free by lowering `full_sims`, without a stop rule, a budget pool,
a probe or a gate.

## Conclusion

Close this line.  Three independent measurements now agree:

  * strength: 4 arms x 3 seeds x 1000 episodes, whole field within 26 Elo and
    inside the seed spread;
  * the decision statistic: the disagreement spread does not narrow with search
    (median 0.506 after 500 simulations), so no threshold on it can signal
    completion;
  * the allocation: per-position spend is a spike, and no position ever gets
    more than 1.25x nominal.

What survives is worth keeping on its own terms: `p_best` by stratified
quadrature, the equivalence test for indifference, the block fixed point as a
CONVERGENCE diagnostic, and the per-position accounting.  The stopping rule
built on them does not work.
