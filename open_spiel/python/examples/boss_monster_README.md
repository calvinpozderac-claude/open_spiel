# Boss Monster on OpenSpiel

A 2-player Python implementation of Brotherwise Games' [*Boss Monster: The
Dungeon-Building Card Game*](https://boardgamegeek.com/boardgame/131835/boss-monster-the-dungeon-building-card-game)
(base set), registered with OpenSpiel as `python_boss_monster`, plus a
browser-playable server and an AlphaZero training entry point.

## Files

| File | Purpose |
|---|---|
| `open_spiel/python/games/boss_monster_data.py` | Card/boss data + a detailed sourcing note. |
| `open_spiel/python/games/boss_monster.py` | The game engine (`pyspiel.Game`/`pyspiel.State`). |
| `open_spiel/python/games/boss_monster_test.py` | Unit tests, incl. OpenSpiel's `pyspiel.random_sim_test`. |
| `open_spiel/python/examples/boss_monster_server.py` | Flask server + browser UI for 2 humans. |
| `open_spiel/python/examples/boss_monster_ngrok.py` | Runs the server behind an ngrok tunnel. |
| `open_spiel/python/examples/boss_monster_colab.ipynb` | Colab notebook wrapping the above. |
| `open_spiel/python/examples/boss_monster_alpha_zero.py` | AlphaZero training entry point. |

## IMPORTANT: how faithful is this to the real game?

This was built in a sandboxed environment whose network egress blocked
essentially every primary source for this game (boardgamegeek.com, the
official rulebook PDFs, and the Boss Monster fan wiki all refused
connections). The **rules** below were confirmed from the search-result
snippets that did get through, and the **card counts** (75 rooms / 31
spells / 25 ordinary heroes / 16 Epic heroes / 8 bosses = 155) are the
documented base-set composition. The **individual card names/stats** and
the **boss abilities**, however, are a mechanically-faithful approximation,
not a verified transcription of every card. See the top of
`boss_monster_data.py` for the full breakdown of what's confirmed vs.
approximated, and for where to paste in exact card data if you have access
to it — nothing else in the engine depends on the specific flavor text.

## Rules implemented

- Each player secretly builds a row of Rooms leading to their Boss. A newly
  built Room always goes next to the Boss, pushing older Rooms toward the
  dungeon entrance (so build order == the order a Hero walks through them).
- Each round: draw, reveal Heroes, Build phase (0 or 1 Room), Spell phase
  (0 or 1 Spell), Bait phase, Adventure phase, then discard down to hand
  size if needed.
- **Bait phase**: each waiting Hero goes to whichever dungeon has the most
  treasure of its Class (Fighter/Cleric/Mage/Thief); ties mean it stays put.
- **Adventure phase**: a Hero walks the Rooms in order taking their damage;
  if its HP drops to 0 it dies (its owner's dungeon gets a Soul), otherwise
  it reaches the Boss (that dungeon's owner takes a Wound).
- **Win/loss**: 5+ Wounds = you lose (even with more Souls); 10+ Souls with
  <5 Wounds = you win.
- Once a player gets their 2nd Soul, Epic Heroes are shuffled into the
  Hero deck.
- Each of the 8 real base-set Bosses (King Croak, Gorgona, Cleopatra,
  Seducia, Cerebellus, Draculord, Xyzax, Robobo) has a small distinct
  passive ability — see `boss_monster_data.BOSSES`.

**Deliberate simplifications** (see the docstring at the top of
`boss_monster.py` for the full list): 2 players only, no Room-evolving back
sides, one Spell effect resolves automatically (no extra targeting
sub-decisions), and games are capped at 40 rounds (ties broken by
Souls-minus-Wounds) so self-play always terminates.

## Running the tests

```bash
python3 open_spiel/python/games/boss_monster_test.py
```

This runs a random-rollout smoke test plus OpenSpiel's own
`pyspiel.random_sim_test` API-conformance checker (also exercised with
`serialize=True`, and with an MCTS bot playing full self-play games, to
confirm the game works correctly under tree search).

## Playing it yourself (2 players)

Locally:

```bash
pip install flask
python3 open_spiel/python/examples/boss_monster_server.py --port=8080
```

Open `http://localhost:8080`, click **New Game**, and open the two
`/play/<token>` links it gives you in two separate browser tabs/windows —
one per player. Each link is that player's private seat; don't open both
yourself if you want each player's hand to stay hidden from the other.

Over the internet, via ngrok:

```bash
pip install flask pyngrok
python3 open_spiel/python/examples/boss_monster_ngrok.py \
    --ngrok_authtoken=<your token from https://dashboard.ngrok.com>
```

This prints a public URL; use it the same way as above, then send the two
`/play/<token>` links to your two players.

From Google Colab: open `boss_monster_colab.ipynb`, edit the repo URL/branch
in cell 2 if needed, run all cells, and paste your ngrok authtoken when
prompted.

## Training an AlphaZero agent

```bash
pip install flax jax  # AlphaZero's neural-net dependencies
python3 open_spiel/python/examples/boss_monster_alpha_zero.py \
    --path=/tmp/boss_monster_az --actors=2 --evaluators=1 \
    --max_simulations=20 --max_steps=50
```

The game satisfies what OpenSpiel's `alpha_zero.py` needs
(`Dynamics.SEQUENTIAL`, `RewardModel.TERMINAL`, a full `observation_tensor`
per player) and was smoke-tested end-to-end with an `MCTSBot` self-play
loop. Two caveats worth knowing before you invest serious compute:

1. **Hidden information.** Boss Monster has private hands; AlphaZero's MCTS
   clones the *entire* state, including the opponent's hand, during tree
   search. That's a standard "determinized"/perfect-information-Monte-Carlo
   simplification for card games, not a game-theoretically sound solver —
   treat the trained policy as a strong heuristic bot.
2. **Speed.** This is a pure-Python game (see the performance note at the
   top of `boss_monster.py`), so self-play actors will be much slower
   per-game than OpenSpiel's C++ games. Start with small
   `max_simulations`/`actors` to validate the pipeline before scaling up,
   and consider porting the hot path to C++ if you want to train seriously.
