# Resuming the CORR run — read this first

Paused 2026-10-05 22:05 for a PC shutdown, at ep30000. Branch `claude/correlated`.

## The two commands

Run both from `open_spiel/colabs`, with the venv at
`../../.venv312/Scripts/python.exe`. Order does not matter, but start training
first — the watcher just polls for files.

```
# 1. training  (resumes from latest.pt automatically; resume=True in the cfg)
python run_corr_tz.py

# 2. the rating watcher  (polls for new bench_*.pt, rates each, appends to the JSON)
python corr_live_ladder.py watch
```

Both are long-running. Training has 20,000 of 50,000 episodes left at roughly
105 min per 1,000 episodes, so about **35 hours**, plus ~70 min of rating per
5,000-episode checkpoint (the rating runs concurrently and steals CPU, so blocks
that overlap a rating take ~125 min).

To read the ladder at any time without disturbing anything:

```
python corr_live_ladder.py report
```

## What state is on disk

| What | Where | Status |
|---|---|---|
| Resumable training blob | `othello_benchmark/bench_CORR_TZ/latest.pt` | **ep30000**, 14 MB — weights + Adam moments + LR scheduler + Elo pool + history |
| Rated checkpoints | `othello_benchmark/bench_CORR_TZ/bench_{0,5000..30000}.pt` | 7 files, bare state_dicts |
| Ladder results | `corr_live_ladder_s400_g24_xot_greedy.json` | 120 pairs, W/D/L per pair, CORR-involving only |
| TZ anchor pairs | `olivaw_live_ladder_s400_g24_xot_greedy.json` | 45 TZ-TZ pairs, seeded into every fit — **do not delete** |
| Training log | `corr_tz.log` | appended, not truncated, on restart |
| Ladder log | `corr_live_ladder.log` | same |

**`othello_benchmark/` is gitignored** (`.gitignore:9`), so the checkpoints are
not in version control — this machine's disk is the only copy. The logs and the
ladder JSON *are* committed, so the results survive even if the checkpoints
don't, but losing `latest.pt` means losing the ability to resume.

`latest.pt` is written atomically (`.tmp` then `os.replace`) every 1,000
episodes, inside the `quick_eval_every` block at
`connect4_dirichlet_utils.py:4419`. The last write was ep30000 at 19:41, and the
run was stopped at ~ep30850 without waiting for the ep31000 write, so **about 850
episodes (~1.5h) are discarded** and training resumes at ep30001. Verified
intact: ep=30000, 110 optimizer state entries, scheduler last_epoch=29993, no
stray `latest.pt.tmp`.

## The one real cost of pausing: the replay buffer

**The 600k-position replay buffer is not saved and is not reloaded.**
`run_training` sets `replay_buffer, start_ep, aux_total = [], 1, 0`
(`connect4_dirichlet_utils.py:4137`) and only `start_ep` is restored from the
checkpoint. There is a `dump_buffer_path` config field, but it only *writes* a
pickle on shutdown for `loss_lab.py` — nothing reads it back, and it was not set
for this run.

So training resumes at ep30001 with an **empty buffer** that refills from
self-play. The log's own counters put this at ~158 positions per episode (61
played plies plus 97.4 OLIVAW doubling-aux positions, from aux 2,921,458 over
30,000 episodes), so the buffer takes about **3,800 episodes** to return to 600k.
During that stretch each gradient step samples from a smaller, more recent, more
correlated pool than before the pause.

Expect this to show up as a transient at ep35000: the
training loss will likely drop (easier to fit a small recent buffer) while play
strength stalls or dips. **Do not read a CO35k dip as the corr posterior
failing.** It is the same artifact MM_long had when it was paused at ep17000.
CO45k and CO50k are the checkpoints that carry the comparison.

If a clean unpaused comparison matters more than the 52 hours already banked,
the alternative is to restart from ep0 with `dump_buffer_path` set — but that
throws away 30,000 episodes and 3 days, and the TZ arm it is being compared
against was itself never paused, so the asymmetry is in CORR's disfavour and any
surviving CORR lead is therefore conservative.

## Where the result stands at the pause

ρ is **fixed at 0.8** for search and 0.999 for targets, `rho_mode='fixed'`,
`rho_head=False` — no learned ρ anywhere in this run (see the `run_corr_tz.py`
docstring for why). Parameter count 909,816, identical to the TZ arm, so the
single variable between the arms is the evidence-collapse rule.

CORR started behind and crossed over at ep15000:

| episode | TZ BT | CORR BT | gap | matched-opponent (CORR / TZ) |
|---|---|---|---|---|
| 5000 | 1153 | 1002 | −151 | 6.9% / 8.3% |
| 10000 | 1360 | 1269 | −91 | 18.3% / 24.8% |
| 15000 | 1475 | 1486 | +11 | 38.5% / 35.0% |
| 20000 | 1550 | 1580 | +30 | 48.3% / 47.2% |
| 25000 | 1600 | 1691 | +91 | 62.7% / 52.5% |
| 30000 | 1637 | 1723 | +86 | 66.0% / 63.4% |

BT values shift between tables because the fit is mean-anchored (TZ25k has moved
1613 → 1587 as checkpoints were added), so compare gaps within one table, never
absolute values across tables. The two metrics also disagree about the size of
the lead at ep30000 — 86 Elo vs 2.6 points — so the defensible claim is "CORR is
ahead at matched episodes", not a specific margin.

CO30k is 3rd of 16 overall, statistically tied with TZ45k, and drew TZ40k 12-0-12
and TZ45k 12-0-12 at 30k episodes.

## Open threads

1. **ρ was never re-measured on the CORR nets.** They have trained 30k episodes
   under a ρ=0.8 search and nobody has checked whether their actual leaf-error
   correlation is still 0.8. `python corr_rho_probe.py othello_benchmark/bench_CORR_TZ 5000 15000 30000`
   measures it offline on frozen weights in a few minutes per checkpoint. Calvin
   asked for these numbers on 2026-10-05 and the probe was never run because it
   competes with training for the 6 cores — it is cheap to run *while the box is
   otherwise idle*, i.e. before restarting training.
2. **The loss trace stopped tracking strength after the ep28000 LR drop.** CO30k
   gained +50 Elo and beat CO25k 15-1-8 while its training loss rose 1.20 → 1.62
   (klv 0.112 → 0.416). Target concentration is climbing (state 32 → 45) while
   predicted sits at 26, so the targets are sharpening faster than the net
   tracks them. Judge this run by ratings only.
3. **Deferred by Calvin:** the 256-channel AlphaZero capacity experiment; the
   unstaged deletion of `duel_ALLMSE_10000_vs_ZGROUND_10000.log`.
4. **Not built:** per-node ρ transport to self-play workers (needs ρ in the
   inference-server response); generation-level ρ updates driven by
   `corr_rho_probe.py`.

## Standing instructions that apply to this run

- Report each rating as soon as it finishes and send a push notification; never
  batch them or hold one for the next milestone.
- Never stop a run unprompted — report and keep it going.
- Always Bradley-Terry from stored per-pair W/D/L, never incremental Elo.
- Eval protocol is 12 XOT openings × both seats = 24 games, 400 sims, greedy.
