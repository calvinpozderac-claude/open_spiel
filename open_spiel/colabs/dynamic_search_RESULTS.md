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

## The diagnosis: half the mechanism never fires

**No search in any arm, on any seed, ever hit its ceiling** — `stop_ceiling`
is 0.00 for all nine adaptive runs.  Every search ended because the gates
opened.

That means the "spend more on hard positions" half of the design is inert.
The budget pool banks surplus that nothing ever draws on, and the ceiling of
4x the nominal budget is unreachable in practice.  What is actually running is
a rule that trims ~10% off easy positions and reallocates none of it.

That, and not the choice of rule, is why the arms are indistinguishable: they
are all doing roughly the same small thing.

## Where to look next, if anywhere

1. **Make the reallocation half work.** The gates open on essentially every
   position, so either `p_stop` = 0.95 is met too easily on a searched root, or
   the disagreement spread collapses faster than expected once a few
   simulations land.  Instrument `p_best` at the moment of stopping before
   changing thresholds — the useful question is whether there EXIST positions
   the rule would give 4x to, not what happens if the knobs move.
2. **A 10% simulation saving is not worth pursuing** on its own.  If the
   ceiling cannot be made to fire, the honest conclusion is that this is not a
   productive direction and the fixed budget is fine.
3. **Scale and game are untested.** This is Connect 4 at 1000 episodes.
   Othello has ~8.5 legal moves against 7, near-zero draws, and 60-ply games;
   the disagreement structure the gates read could behave differently.  But
   given the mechanism is inert here for a structural reason, re-running it
   elsewhere unchanged is unlikely to be informative.

## What this pilot can and cannot resolve

With ~220 games per network and 3 seeds it can separate arm effects of roughly
50 Elo.  A genuine ±20 Elo effect would be invisible.  The claim is therefore
"no large effect", not "no effect".
