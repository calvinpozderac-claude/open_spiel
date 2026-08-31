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

"""A small Flask server for playing `python_boss_monster` head-to-head.

Two people play from two browser tabs (on the same machine, on a LAN, or
over the internet via an ngrok tunnel -- see `boss_monster_ngrok.py` and
`boss_monster_colab.ipynb`). There is no login system: instead, when a game
is created the server mints one private URL per seat
(`/play/<token>`); whoever holds a URL controls that seat. Don't share both
links with the same person if you want the hidden-hand information to stay
hidden.

Chance events (drawing cards, revealing Heroes) have no real decision to
make, so the server resolves them automatically and instantly with a
uniform random draw -- players only ever see Build / Spell / Discard
choices.

Run directly for a local game:
    python3 open_spiel/python/examples/boss_monster_server.py --port=8080

Requires: `pip install flask`.
"""

import random
import secrets
import threading

from absl import app as absl_app
from absl import flags
from flask import Flask
from flask import jsonify
from flask import request

# Importing this registers "python_boss_monster" with pyspiel.
from open_spiel.python.games import boss_monster  # pylint: disable=unused-import
from open_spiel.python.games import boss_monster_data as card_data
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8080, "Port to serve on.")
flags.DEFINE_string("host", "0.0.0.0", "Host to bind to.")


_LOCK = threading.Lock()


class GameSession:
  """Holds one in-progress (or finished) 2-player Boss Monster game."""

  def __init__(self, boss0=0, boss1=1):
    self.game = pyspiel.load_game(
        f"python_boss_monster(boss0={boss0},boss1={boss1})")
    self.state = self.game.new_initial_state()
    self.tokens = [secrets.token_urlsafe(12), secrets.token_urlsafe(12)]
    self._resolve_chance()

  def _resolve_chance(self):
    """Auto-resolves chance nodes (card draws / Hero reveals)."""
    while not self.state.is_terminal() and self.state.is_chance_node():
      outcomes, probs = zip(*self.state.chance_outcomes())
      action = random.choices(outcomes, weights=probs, k=1)[0]
      self.state.apply_action(action)

  def player_for_token(self, token):
    for p, t in enumerate(self.tokens):
      if secrets.compare_digest(t, token):
        return p
    return None

  def apply(self, player, action):
    with _LOCK:
      if self.state.is_terminal():
        raise ValueError("Game is already over.")
      if self.state.is_chance_node():
        raise ValueError("Waiting on a chance event; try again shortly.")
      if self.state.current_player() != player:
        raise ValueError("It is not your turn.")
      legal = self.state.legal_actions()
      if action not in legal:
        raise ValueError(f"Illegal action {action}; legal={legal}")
      self.state.apply_action(action)
      self._resolve_chance()

  def public_view(self):
    st = self.state
    players = []
    for p in range(2):
      rooms = [{
          "name": card_data.ROOM_TEMPLATES[r].name,
          "cls": card_data.CLASSES[card_data.ROOM_TEMPLATES[r].cls],
          "treasure": card_data.ROOM_TEMPLATES[r].treasure,
          "damage": card_data.ROOM_TEMPLATES[r].damage,
      } for r in st._rooms[p]]  # pylint: disable=protected-access
      players.append({
          "boss": card_data.BOSSES[st._boss[p]].name,  # pylint: disable=protected-access
          "boss_ability": card_data.BOSSES[st._boss[p]].text,  # pylint: disable=protected-access
          "wounds": st._wounds[p],  # pylint: disable=protected-access
          "souls": st._souls[p],  # pylint: disable=protected-access
          "hand_count": len(st._hands[p]),  # pylint: disable=protected-access
          "rooms": rooms,
          "max_rooms": st._max_rooms[p],  # pylint: disable=protected-access
      })
    waiting = [{
        "name": card_data.HERO_TEMPLATES[h].name,
        "cls": card_data.CLASSES[card_data.HERO_TEMPLATES[h].cls],
        "health": card_data.HERO_TEMPLATES[h].health,
        "epic": card_data.HERO_TEMPLATES[h].epic,
    } for h in st._waiting]  # pylint: disable=protected-access

    is_terminal = st.is_terminal()
    acting_player = None if is_terminal or st.is_chance_node() else (
        st.current_player())
    return {
        "round": st._round,  # pylint: disable=protected-access
        "acting_player": acting_player,
        "players": players,
        "waiting_heroes": waiting,
        "log": st.debug_log(12),
        "game_over": is_terminal,
        "returns": st.returns() if is_terminal else None,
    }

  def hand_for(self, player):
    st = self.state
    out = []
    for i, (kind, tid) in enumerate(st._hands[player]):  # pylint: disable=protected-access
      if kind == "room":
        c = card_data.ROOM_TEMPLATES[tid]
        out.append({"idx": i, "kind": "room", "name": c.name,
                    "cls": card_data.CLASSES[c.cls], "treasure": c.treasure,
                    "damage": c.damage, "text": ""})
      else:
        c = card_data.SPELL_TEMPLATES[tid]
        out.append({"idx": i, "kind": "spell", "name": c.name, "cls": None,
                    "treasure": None, "damage": None, "text": c.text})
    return out

  def legal_actions_for(self, player):
    st = self.state
    if st.is_terminal() or st.is_chance_node():
      return []
    if st.current_player() != player:
      return []
    actions = []
    for a in st.legal_actions():
      actions.append({"id": a, "label": st.action_to_string(player, a)})
    return actions

  def phase_for_ui(self):
    st = self.state
    if st.is_terminal():
      return "game_over"
    if st.is_chance_node():
      return "chance"
    step = st._queue[0]  # pylint: disable=protected-access
    return step[0]


flask_app = Flask(__name__)
_SESSION = {"game": None}


def _require_session():
  session = _SESSION["game"]
  if session is None:
    raise ValueError("No game in progress. POST /api/new first.")
  return session


@flask_app.route("/")
def index():
  return _INDEX_HTML


@flask_app.route("/api/new", methods=["POST"])
def api_new():
  body = request.get_json(silent=True) or {}
  boss0 = int(body.get("boss0", 0))
  boss1 = int(body.get("boss1", 1))
  with _LOCK:
    _SESSION["game"] = GameSession(boss0, boss1)
    session = _SESSION["game"]
  return jsonify({
      "player0_url": f"/play/{session.tokens[0]}",
      "player1_url": f"/play/{session.tokens[1]}",
  })


@flask_app.route("/play/<token>")
def play(token):
  del token  # The client JS reads the token from the URL itself.
  return _PLAY_HTML


@flask_app.route("/api/state")
def api_state():
  try:
    session = _require_session()
  except ValueError as e:
    return jsonify({"error": str(e)}), 400
  token = request.args.get("token", "")
  player = session.player_for_token(token)
  if player is None:
    return jsonify({"error": "Unknown or expired player link."}), 404
  view = session.public_view()
  view["you_are_player"] = player
  view["your_turn"] = (view["acting_player"] == player)
  view["your_hand"] = session.hand_for(player)
  view["legal_actions"] = (
      session.legal_actions_for(player) if view["your_turn"] else [])
  view["phase"] = session.phase_for_ui()
  return jsonify(view)


@flask_app.route("/api/act", methods=["POST"])
def api_act():
  try:
    session = _require_session()
  except ValueError as e:
    return jsonify({"error": str(e)}), 400
  body = request.get_json(silent=True) or {}
  token = body.get("token", "")
  action = body.get("action")
  player = session.player_for_token(token)
  if player is None:
    return jsonify({"error": "Unknown or expired player link."}), 404
  try:
    session.apply(player, int(action))
  except ValueError as e:
    return jsonify({"error": str(e)}), 400
  return jsonify({"ok": True})


_INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Boss Monster Server</title>
<style>body{font-family:sans-serif;max-width:640px;margin:40px auto;
line-height:1.5} button{font-size:1rem;padding:8px 16px}
a{word-break:break-all}</style></head><body>
<h1>Boss Monster (2-player) server</h1>
<p>Click "New Game" to deal a fresh game, then send each player their own
private link. <b>Do not open both links yourself if you want each player's
hand to stay hidden from the other.</b></p>
<label>Player 0 boss (0-7): <input id="b0" type="number" value="0" min="0"
max="7"></label><br>
<label>Player 1 boss (0-7): <input id="b1" type="number" value="1" min="0"
max="7"></label><br><br>
<button onclick="newGame()">New Game</button>
<div id="out"></div>
<script>
async function newGame(){
  const b0 = document.getElementById('b0').value;
  const b1 = document.getElementById('b1').value;
  const r = await fetch('/api/new', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({boss0:b0, boss1:b1})});
  const j = await r.json();
  const base = window.location.origin;
  document.getElementById('out').innerHTML =
    '<p>Player 0 link: <a href="' + base + j.player0_url + '">' +
    base + j.player0_url + '</a></p>' +
    '<p>Player 1 link: <a href="' + base + j.player1_url + '">' +
    base + j.player1_url + '</a></p>';
}
</script>
</body></html>
"""

_PLAY_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Boss Monster</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:20px auto;padding:0 12px}
.board{display:flex;gap:16px;flex-wrap:wrap}
.dungeon{flex:1;min-width:260px;border:2px solid #333;border-radius:8px;
padding:10px}
.dungeon.me{border-color:#2a6}
.room{display:inline-block;border:1px solid #999;border-radius:6px;
padding:4px 8px;margin:2px;font-size:0.85rem;background:#f6f6f6}
.card{display:inline-block;border:1px solid #666;border-radius:6px;
padding:6px 10px;margin:4px;cursor:pointer;background:#fff}
.card:hover{background:#eef}
.card.disabled{opacity:0.4;cursor:default}
.hero{display:inline-block;border:1px dashed #a55;border-radius:6px;
padding:3px 6px;margin:2px;font-size:0.85rem}
#log{background:#111;color:#0f0;font-family:monospace;font-size:0.8rem;
padding:8px;height:140px;overflow-y:auto;border-radius:6px}
h2{margin-bottom:4px}
.badge{display:inline-block;background:#333;color:#fff;border-radius:10px;
padding:2px 8px;margin-right:6px;font-size:0.8rem}
</style></head><body>
<h1>Boss Monster</h1>
<div id="status"></div>
<div class="board" id="board"></div>
<h3>Waiting Heroes</h3>
<div id="waiting"></div>
<h3>Your Hand / Actions</h3>
<div id="hand"></div>
<h3>Log</h3>
<div id="log"></div>
<script>
const token = window.location.pathname.split('/').pop();

function classIcon(c){
  return {fighter:'⚔', cleric:'✝', mage:'📖',
          thief:'💰'}[c] || c;
}

async function poll(){
  const r = await fetch('/api/state?token=' + encodeURIComponent(token));
  if(!r.ok){ document.getElementById('status').innerText =
    'Error loading game (bad or expired link).'; return; }
  const s = await r.json();
  render(s);
}

function render(s){
  const you = s.you_are_player;
  let status = 'You are Player ' + you + '. Round ' + s.round + '. ';
  if(s.game_over){
    const r = s.returns;
    status += r[you] > 0 ? 'You WIN!' : (r[you] < 0 ? 'You lose.' : 'Draw.');
  } else if (s.phase === 'chance') {
    status += 'Dealing cards...';
  } else {
    status += s.your_turn ? ('Your move (' + s.phase + ' phase).')
                           : ("Waiting on Player " + s.acting_player + '.');
  }
  document.getElementById('status').innerHTML = '<b>' + status + '</b>';

  let board = '';
  for(const p of [0,1]){
    const pl = s.players[p];
    const mine = p === you;
    board += '<div class="dungeon' + (mine?' me':'') + '">' +
      '<h2>' + (mine?'You':'Opponent') + ' &mdash; ' + pl.boss + '</h2>' +
      '<div><i>' + pl.boss_ability + '</i></div>' +
      '<div><span class="badge">Wounds ' + pl.wounds + '/5</span>' +
      '<span class="badge">Souls ' + pl.souls + '/10</span>' +
      '<span class="badge">Hand ' + pl.hand_count + '</span></div>' +
      '<p>Entrance &rarr; Boss:</p>';
    for(const rm of pl.rooms){
      board += '<span class="room">' + classIcon(rm.cls) + ' ' + rm.name +
        ' (T' + rm.treasure + '/D' + rm.damage + ')</span>';
    }
    if(pl.rooms.length === 0) board += '<i>(no rooms built yet)</i>';
    board += '</div>';
  }
  document.getElementById('board').innerHTML = board;

  let waiting = '';
  for(const h of s.waiting_heroes){
    waiting += '<span class="hero">' + classIcon(h.cls) + ' ' + h.name +
      ' (HP ' + h.health + ')' + (h.epic?' ★':'') + '</span>';
  }
  document.getElementById('waiting').innerHTML = waiting || '<i>none</i>';

  let hand = '<p>';
  for(const c of s.your_hand){
    hand += '<span class="card">' +
      (c.kind === 'room'
        ? (classIcon(c.cls) + ' ' + c.name + ' (T' + c.treasure + '/D' +
           c.damage + ')')
        : ('✨ ' + c.name + ' -- ' + c.text)) +
      '</span>';
  }
  hand += '</p><p>';
  if(s.your_turn){
    for(const a of s.legal_actions){
      hand += '<button onclick="act(' + a.id + ')">' + a.label + '</button> ';
    }
  } else {
    hand += '<i>Not your turn.</i>';
  }
  hand += '</p>';
  document.getElementById('hand').innerHTML = hand;

  document.getElementById('log').innerHTML =
    s.log.map(x => '&gt; ' + x).join('<br>');
}

async function act(actionId){
  const r = await fetch('/api/act', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({token: token, action: actionId})});
  if(!r.ok){ const j = await r.json(); alert(j.error || 'Error'); }
  poll();
}

poll();
setInterval(poll, 1500);
</script>
</body></html>
"""


def main(unused_argv):
  flask_app.run(host=FLAGS.host, port=FLAGS.port, threaded=True)


if __name__ == "__main__":
  absl_app.run(main)
