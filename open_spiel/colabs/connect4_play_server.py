#!/usr/bin/env python3
"""Connect 4 live-play web server — play against a checkpointed model, or watch
two models play each other, in a browser (works through ngrok).

Drives the ThompsonZero-C4 engine from `connect4_dirichlet_utils.py`: a network
with a STATE Dirichlet head and an ACTION Dirichlet head, searched by Thompson
sampling over the per-action beliefs with an MCTS-Solver overlay.  The search is
IMPORTED, not re-implemented — this server builds its trees out of the very same
`_select_leaf` / `_backup` / `root_pick` the training loop uses, so what you play
against is exactly what was trained (the sibling chess/boop servers inline their
engines because those live inside notebooks; this one does not have to).

Usage:
    python connect4_play_server.py --model c4_dirichlet_ckpt/bench_5000.pt
    python connect4_play_server.py --model A.pt --model2 B.pt   # watch A vs B
    ngrok http 8765                                             # then share URL

Flags:
    --model  PATH        checkpoint (bench_*.pt or latest.pt)
    --model2 PATH        second model for watch mode (default: same as --model)
    --port   N           HTTP port (default 8765)
    --device D           inference device (default cpu — batch-1..8 is CPU's regime)
    --search-agg A       evidence rule for selection: mixture | mean | sum.
                         Default: whatever --model's checkpoint was trained with
                         (latest.pt records it), else 'mixture'.
    --wave   N           leaves evaluated per batched search wave (default 8)
    --snapshot-secs S    seconds between analysis snapshots while thinking (2)

Network shape (channels / blocks / head width) is inferred from the checkpoint.

What the UI gives you, per the sibling servers: play as either colour or watch
two models, a fixed simulation budget or "think until I say stop", live move
preferences with a slider back through the search's own history, the engine's
win/draw/loss belief, take-back, "what would the AI do?" with one-click play,
and a move log you can scrub through.
"""
import argparse
import json as _json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import pyspiel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import connect4_dirichlet_utils as c4          # noqa: E402  (needs the path first)

GAME = pyspiel.load_game('connect_four')
c4.set_game(GAME)
ROWS, COLS = c4._OBS_SHAPE[1], c4._OBS_SHAPE[2]         # 6 x 7
FIRST_PLAYER = GAME.new_initial_state().current_player()
DISC = {0: 'x', 1: 'o'}                                  # matches str(state)


def move_label(player, col):
    """Move-log entry. open_spiel's own action_to_string is 0-indexed ('o2'),
    which reads off-by-one against the 1-indexed column labels under the board,
    so name the colour and use the column number the player actually clicked."""
    return f'{"R" if player == FIRST_PLAYER else "Y"}{int(col) + 1}'


# ═══════════════════════════════════════════════════════════════════════════════
# Search driver — one move's tree, run incrementally so the UI can watch it
# ═══════════════════════════════════════════════════════════════════════════════
class Searcher:
    """Batched-wave Thompson MCTS for a single position, built on the training
    module's tree primitives.  `run()` can be bounded by a simulation budget or
    left to run until a stop event fires, and a callback records analysis
    snapshots on a wall-clock cadence."""

    def __init__(self, net, state, device, wave=8, pick='best'):
        self.net = net
        self.device = device
        self.state = state.clone()
        self.wave = int(wave)
        self.pick = pick                      # 'best' | 'thompson'
        self.rng = np.random.default_rng()
        self.root = c4.expand_node(net, device, self.state)

    # One batched wave: select `wave` leaves (virtual loss keeps them apart),
    # evaluate the unique ones in a single forward pass, then back them all up.
    def _run_wave(self):
        root = self.root
        pending, evals, seen = [], [], set()
        for _ in range(self.wave):
            if c4._node_solved_outcome(root) is not None:
                break
            path, st, payloads, edge = c4._select_leaf(root, self.state, self.rng)
            if st is None:                    # terminal or already-proven edge
                c4._backup_terminal(path, payloads, None)
                continue
            node, idx = edge
            pending.append((path, node, idx))
            if (id(node), idx) not in seen:
                seen.add((id(node), idx))
                evals.append((node, idx, st))
        if evals:
            v3, vc, p3, cf, ob = c4.nn_eval_states(
                self.net, self.device, [e[2] for e in evals])
            for (node, idx, st), a, b, p, c, o in zip(evals, v3, vc, p3, cf, ob):
                leg = st.legal_actions()
                child = c4._CNode(st.current_player(), leg, a, float(b),
                                  p[leg], c[leg], obs=o)
                c4._seed_leaf(child)
                node.children[idx] = child
        for path, node, idx in pending:
            c4._backup(path, c4._payload(node.children[idx].v_alpha))

    def run(self, max_sims=None, stop_evt=None, snap_cb=None, snap_secs=2.0):
        last = time.time()
        while True:
            if stop_evt is not None and stop_evt.is_set():
                break
            if c4._node_solved_outcome(self.root) is not None:
                break                          # proven — more search is wasted
            n = self.n_sims()
            if max_sims is not None and n >= max_sims:
                break
            if n >= 5_000_000:
                break
            self._run_wave()
            if snap_cb is not None and time.time() - last >= snap_secs:
                snap_cb(self)
                last = time.time()

    def n_sims(self):
        # Every simulation deposits exactly one observation on exactly one root
        # edge, so the edge evidence counts ARE the simulation count.
        return int(self.root.visits().sum())

    def snapshot(self):
        """What the UI draws: per-column preference, the engine's win/draw/loss
        belief for the side to move, and how concentrated that belief is."""
        root = self.root
        probs = {}
        solved = c4._node_solved_outcome(root)
        if solved == c4._WIN:
            # Proven win: the preference IS the set of winning moves, whatever
            # the visit counts happened to be when the proof landed.
            wins = np.nonzero(root.term == c4._WIN)[0]
            share = round(1.0 / len(wins), 4)
            for i in wins:
                probs[int(root.legal[i])] = share
        else:
            vis = root.visits().astype(float)
            tot = vis.sum()
            if tot > 0:
                for i, a in enumerate(root.legal):
                    if vis[i] > 0:
                        probs[int(a)] = round(float(vis[i] / tot), 4)
        tgt = root.state_target()              # the searched state belief
        m = c4.dir_mean(tgt)
        return {'sims': self.n_sims(),
                'value': round(float(m[c4._WIN] - m[c4._LOSS]), 3),
                'wdl': [round(float(x), 3) for x in m],
                'conc': round(float(tgt.sum()), 1),
                'solved': (None if solved is None else
                           ['win', 'draw', 'loss'][solved]),
                'probs': probs}

    def best(self):
        """The move to play.  'best' = posterior value-mean argmax over searched
        edges (what evaluation and the Elo pool use); 'thompson' = one more
        Thompson draw, exactly how self-play picks its move."""
        return int(c4.root_pick(self.root, self.rng,
                                thompson=(self.pick == 'thompson')))


# ═══════════════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════════════
PAGE = r'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect 4 — play the model</title>
<!-- Inline icon: without one the browser requests /favicon.ico on every load
     and logs a 404 in the console. -->
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%232f5fbf'/%3E%3Ccircle cx='16' cy='16' r='9' fill='%23d63b28'/%3E%3C/svg%3E">
<style>
 body{font-family:system-ui,sans-serif;background:#1c1e22;color:#eee;margin:0;
      display:flex;flex-wrap:wrap;gap:24px;padding:24px;justify-content:center}
 h1{font-size:20px;margin:0 0 12px}
 #setup{background:#26292f;padding:24px;border-radius:12px;max-width:420px}
 #setup label{display:block;margin:10px 0 4px;color:#aab}
 #setup .row{margin-bottom:6px}
 select,input[type=number]{background:#15171a;color:#eee;border:1px solid #444;
      border-radius:6px;padding:6px 8px}
 button{background:#3a6df0;color:#fff;border:0;border-radius:8px;
      padding:8px 14px;cursor:pointer;font-size:14px}
 #game{display:none;gap:24px;flex-wrap:wrap;justify-content:center}
 #boardwrap{display:flex;flex-direction:column;align-items:center}
 #prefrow{display:grid;grid-template-columns:repeat(7,64px);gap:4px;
      margin-bottom:4px}
 .pref{height:26px;position:relative;border-radius:4px;background:#22252b;
      display:flex;align-items:center;justify-content:center;font-size:11px;
      color:#cfe;overflow:hidden}
 .pref .bar{position:absolute;left:0;bottom:0;top:0;background:#3a6df0;opacity:.55}
 .pref span{position:relative}
 #board{display:grid;grid-template-columns:repeat(7,64px);
      grid-template-rows:repeat(6,64px);gap:4px;background:#2f5fbf;padding:8px;
      border-radius:10px}
 .cell{width:64px;height:64px;border-radius:50%;background:#15171a;
      cursor:pointer;position:relative;display:flex;align-items:center;
      justify-content:center}
 .cell.p0{background:radial-gradient(circle at 34% 30%,#ff8a7a,#d63b28 70%)}
 .cell.p1{background:radial-gradient(circle at 34% 30%,#ffe07a,#e0a800 70%)}
 .cell.drop{box-shadow:inset 0 0 0 3px rgba(255,255,255,.35)}
 .cell.last{box-shadow:0 0 0 3px #ffd479}
 .cell.win{box-shadow:0 0 0 4px #46d17a}
 .cell.hintcol{outline:2px dashed #46d17a;outline-offset:-6px}
 .cell.dead{cursor:default}
 #collabels{display:grid;grid-template-columns:repeat(7,64px);gap:4px;
      margin-top:6px;color:#7d8694;font-size:12px;text-align:center}
 #side{max-width:360px;background:#26292f;padding:18px;border-radius:12px}
 #side .stat{margin:6px 0;color:#ccd;font-size:13px}
 #status{font-weight:600;color:#ffd479;margin:8px 0}
 #prefsbox{margin-top:12px;border-top:1px solid #3a3d44;padding-top:10px}
 #simslider{width:100%}
 #wdl{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:6px 0}
 #wdl div{height:100%}
 .wseg{background:#46d17a}.dseg{background:#7d8694}.lseg{background:#e0574a}
 #movelog{max-height:180px;overflow-y:auto;font-size:12px;color:#889;margin-top:8px}
 .kbtn{background:#2e3138;border:1px solid #555;margin-right:6px}
 .mv{cursor:pointer}.mv:hover{color:#fff}.mv.cur{color:#ffd479;font-weight:600}
 #navlabel{color:#9ab;font-size:12px;margin-left:6px}
</style>

<div id="setup">
  <h1>🔴 Connect 4 — play the model</h1>
  <div class="row"><label>Mode</label>
    <select id="mode">
      <option value="play">Play against the model</option>
      <option value="watch">Watch model vs model</option>
    </select></div>
  <div class="row" id="siderow"><label>Your side</label>
    <select id="human">
      <option value="first">Red (moves first)</option>
      <option value="second">Yellow (moves second)</option>
    </select></div>
  <div class="row"><label>AI thinking per move</label>
    <select id="thinkmode">
      <option value="fixed">Fixed number of MCTS simulations</option>
      <option value="manual">Think until I click &ldquo;Make AI move&rdquo;</option>
    </select></div>
  <div class="row" id="simsrow"><label>Simulations per move</label>
    <input type="number" id="sims" value="400" min="1" max="5000000"></div>
  <div class="row"><label>How the AI commits its move</label>
    <select id="pick">
      <option value="best">Best — posterior value argmax (evaluation style)</option>
      <option value="thompson">Thompson sample (self-play style)</option>
    </select></div>
  <div class="row" style="margin-top:14px">
    <button onclick="newGame()">Start game</button></div>
</div>

<div id="game">
  <div id="boardwrap">
    <div id="prefrow"></div>
    <div id="board"></div>
    <div id="collabels"></div>
  </div>
  <div id="side">
    <div id="status">…</div>
    <div class="stat" id="thinkinfo"></div>
    <button id="commitbtn" style="display:none" onclick="commitAI()">Make AI move now</button>
    <div id="actionbtns" style="margin:6px 0">
      <button class="kbtn" id="takebackbtn" style="display:none" onclick="takeback()">↩ Take back</button>
      <button class="kbtn" id="analyzebtn" style="display:none" onclick="analyze()">🔍 What would the AI do?</button>
      <button class="kbtn" id="stopanalyzebtn" style="display:none" onclick="stopAnalyze()">⏹ Stop analysis</button>
      <button class="kbtn" id="playhintbtn" style="display:none" onclick="playHint()">▶ Play the AI&rsquo;s move</button>
    </div>
    <div id="prefsbox">
      <label><input type="checkbox" id="prefs" checked> Show AI move preferences</label>
      <div id="sliderbox" style="display:none">
        <input type="range" id="simslider" min="0" max="0" value="0">
        <div class="stat" id="sliderlabel"></div>
      </div>
      <div id="wdl" style="display:none"><div class="wseg"></div><div class="dseg"></div><div class="lseg"></div></div>
      <div class="stat" id="evalline"></div>
    </div>
    <div class="stat" style="margin-top:10px">
      <button class="kbtn" onclick="nav(-1e9)">⏮</button>
      <button class="kbtn" onclick="nav(-1)">◀</button>
      <button class="kbtn" onclick="nav(1)">▶</button>
      <button class="kbtn" onclick="nav(1e9)">live ⏭</button>
      <span id="navlabel"></span>
    </div>
    <div id="movelog"></div>
    <div style="margin-top:12px"><button class="kbtn" onclick="location.reload()">New game</button></div>
  </div>
</div>

<script>
let SID=null, S=null, sliderStick=true, viewMove=null, hoverCol=null;
const ROWS=6, COLS=7;

function grid(b){                       // board string -> rows[r][c]
  return (b||'').split('\n').filter(x=>x.length).map(r=>r.split(''));
}
function dropRow(g,c){                  // lowest empty row in column c, or -1
  for(let r=ROWS-1;r>=0;r--) if(g[r][c]==='.') return r;
  return -1;
}
function winLine(g){                    // [[r,c],…] of a 4-in-a-row, or []
  const dirs=[[0,1],[1,0],[1,1],[1,-1]];
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    const p=g[r][c]; if(p==='.') continue;
    for(const [dr,dc] of dirs){
      const cells=[];
      for(let k=0;k<4;k++){
        const rr=r+dr*k, cc=c+dc*k;
        if(rr<0||rr>=ROWS||cc<0||cc>=COLS||g[rr][cc]!==p){ cells.length=0; break; }
        cells.push([rr,cc]);
      }
      if(cells.length===4) return cells;
    }
  }
  return [];
}
function diffCell(a,b){                 // the one cell that b has and a does not
  if(!a||!b) return null;
  const ga=grid(a), gb=grid(b);
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++)
    if(ga[r][c]==='.'&&gb[r][c]!=='.') return [r,c];
  return null;
}

function nav(d){
  if(!S||!S.board_hist) return;
  const last=S.board_hist.length-1;
  let cur=viewMove===null?last:viewMove;
  cur=Math.max(0,Math.min(last,cur+d));
  viewMove=(cur>=last)?null:cur;
  render();
}
function viewAt(i){
  if(!S||!S.board_hist) return;
  viewMove=(i>=S.board_hist.length-1)?null:i;
  render();
}

document.getElementById('mode').onchange=e=>{
  document.getElementById('siderow').style.display=e.target.value==='play'?'block':'none';
};
document.getElementById('thinkmode').onchange=e=>{
  document.getElementById('simsrow').style.display=e.target.value==='fixed'?'block':'none';
};
document.getElementById('simslider').oninput=e=>{
  sliderStick=(+e.target.value===+e.target.max); render();
};

async function api(path,body){
  const r=await fetch(path,body?{method:'POST',body:JSON.stringify(body)}:{});
  return r.json();
}
async function newGame(){
  const mode=document.getElementById('mode').value;
  const sims=document.getElementById('thinkmode').value==='fixed'
           ? +document.getElementById('sims').value : 0;
  const r=await api('/new',{mode,sims,side:document.getElementById('human').value,
                           pick:document.getElementById('pick').value});
  SID=r.sid;
  document.getElementById('setup').style.display='none';
  document.getElementById('game').style.display='flex';
  poll(); setInterval(poll,1000);
}
async function poll(){ if(!SID)return; S=await api('/state?sid='+SID); render(); }
async function commitAI(){ await api('/ai_commit',{sid:SID}); }
async function takeback(){ viewMove=null; const r=await api('/takeback',{sid:SID});
  if(!r.ok&&r.error) alert(r.error); poll(); }
async function analyze(){ viewMove=null; await api('/analyze',{sid:SID}); poll(); }
async function stopAnalyze(){ await api('/stop_analyze',{sid:SID}); poll(); }
async function playHint(){ const r=await api('/play_hint',{sid:SID});
  if(!r.ok&&r.error) alert(r.error); poll(); }

async function drop(c){
  if(!S||S.status!=='human_turn'||viewMove!==null) return;
  if(!(S.legal||[]).includes(c)) return;
  const res=await api('/move',{sid:SID,col:c});
  if(res.ok) poll(); else render();
}

function snapNow(){
  if(!S||!S.snapshots.length) return null;
  const sl=document.getElementById('simslider');
  const idx=sliderStick?S.snapshots.length-1
                       :Math.min(+sl.value,S.snapshots.length-1);
  return {snap:S.snapshots[idx], idx};
}

function render(){
  if(!S) return;
  const hist=S.board_hist||[];
  const last=Math.max(0,hist.length-1);
  const showIdx=viewMove===null?last:Math.min(viewMove,last);
  const live=viewMove===null;
  const g=grid(hist.length?hist[showIdx]:S.board);
  const showPrefs=document.getElementById('prefs').checked;
  const liveThink=live&&(S.status==='thinking'||S.status==='analyzing');
  const prefsOn=showPrefs||S.status==='analyzing'||(live&&!!S.hint);

  // Preference source: the live growing search, the standing AI suggestion, or
  // the saved analysis of the move that was played FROM the position on screen
  // (analysis_hist[i] is the analysis for the move out of position i-1, so the
  // one belonging to this board is at showIdx+1).
  let sn=null, snInfo='';
  if(prefsOn){
    if(liveThink){ const cur=snapNow(); if(cur) sn=cur.snap; }
    else if(live&&S.hint&&showIdx===last){
      sn=S.hint; snInfo=`AI suggestion (column ${S.hint_col+1}) — ${S.hint.sims} sims`;
    } else {
      const a=(S.analysis_hist||[])[showIdx+1];
      if(a){ sn=a; snInfo=`saved analysis of move ${showIdx+1} — ${a.sims} sims`; }
      else if(showIdx<(S.move_log||[]).length) snInfo='(that move was yours)';
    }
  }

  const wl=(showIdx===last&&S.terminal)?winLine(g):[];
  const lastCell=showIdx>0?diffCell(hist[showIdx-1],hist[showIdx]):null;
  const hintCol=(live&&!liveThink&&S.hint_col!==null&&S.hint_col!==undefined)
      ?S.hint_col:-1;

  // Column preference bars
  const pr=document.getElementById('prefrow'); pr.innerHTML='';
  // Bars are scaled so the FAVOURITE column fills its slot: with 7 columns a
  // raw-percentage bar tops out near 20% of the width and they all look alike.
  const pmax=(sn&&sn.probs)?Math.max(...Object.values(sn.probs),1e-9):1;
  for(let c=0;c<COLS;c++){
    const d=document.createElement('div'); d.className='pref';
    const p=(sn&&sn.probs)?sn.probs[c]:undefined;
    if(p!==undefined){
      const bar=document.createElement('div'); bar.className='bar';
      bar.style.width=Math.max(3,p/pmax*100)+'%'; d.appendChild(bar);
      const s=document.createElement('span'); s.textContent=(p*100).toFixed(0)+'%';
      d.appendChild(s);
    }
    pr.appendChild(d);
  }

  const playable=live&&S.status==='human_turn';
  const board=document.getElementById('board'); board.innerHTML='';
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    const d=document.createElement('div');
    d.className='cell';
    if(g[r][c]==='x') d.classList.add('p0');
    else if(g[r][c]==='o') d.classList.add('p1');
    if(!playable) d.classList.add('dead');
    if(lastCell&&lastCell[0]===r&&lastCell[1]===c) d.classList.add('last');
    if(wl.some(([wr,wc])=>wr===r&&wc===c)) d.classList.add('win');
    if(c===hintCol&&r===dropRow(g,c)) d.classList.add('hintcol');
    if(playable&&hoverCol===c&&r===dropRow(g,c)) d.classList.add('drop');
    d.onmouseenter=()=>{ if(playable&&hoverCol!==c){ hoverCol=c; render(); } };
    d.onclick=()=>drop(c);
    board.appendChild(d);
  }
  const cl=document.getElementById('collabels');
  cl.innerHTML=Array.from({length:COLS},(_,c)=>`<div>${c+1}</div>`).join('');

  const st=document.getElementById('status');
  const NAME={0:'Red',1:'Yellow'};
  if(S.terminal){
    const r0=S.returns[0];
    st.textContent=r0>0?'🏆 Red wins':r0<0?'🏆 Yellow wins':'½–½ Draw (board full)';
  } else if(S.status==='human_turn'){
    st.textContent=S.hint?'Your move — AI suggestion shown':'Your move';
  } else if(S.status==='analyzing'){
    st.textContent='Analyzing your position…';
  } else if(S.status==='thinking'){
    const who=NAME[S.current_player];
    const eng=S.engines[S.current_player]||'';
    st.textContent=(S.mode==='watch'?`${who} (${eng}) thinking…`:'AI thinking…');
  } else st.textContent=S.status;

  document.getElementById('thinkinfo').textContent=
    (S.status==='thinking'||S.status==='analyzing')
      ?`simulations so far: ${S.thinking_sims}`:'';
  document.getElementById('commitbtn').style.display=
    (S.status==='thinking'&&S.manual)?'inline-block':'none';
  const showBtn=(id,on)=>document.getElementById(id).style.display=on?'inline-block':'none';
  showBtn('takebackbtn', live&&!S.terminal&&S.can_takeback);
  showBtn('analyzebtn', live&&!S.terminal&&S.status==='human_turn'&&!S.analyzing);
  showBtn('stopanalyzebtn', S.status==='analyzing');
  showBtn('playhintbtn', live&&!S.terminal&&S.status==='human_turn'&&S.hint_col!==null);

  const box=document.getElementById('sliderbox');
  if(liveThink&&prefsOn&&S.snapshots.length>0){
    box.style.display='block';
    const sl=document.getElementById('simslider'); sl.max=S.snapshots.length-1;
    if(sliderStick) sl.value=sl.max;
    const cur=snapNow();
    document.getElementById('sliderlabel').textContent=
      `preferences after ${cur.snap.sims} simulations (snapshot ${(+cur.idx)+1}/${S.snapshots.length})`;
  } else box.style.display='none';

  // Win/draw/loss belief for whoever is to move in the shown analysis.
  const wdl=document.getElementById('wdl'), ev=document.getElementById('evalline');
  if(sn&&sn.wdl){
    wdl.style.display='flex';
    const segs=wdl.children;
    for(let i=0;i<3;i++) segs[i].style.width=(sn.wdl[i]*100).toFixed(1)+'%';
    const solved=sn.solved?` · PROVEN ${sn.solved.toUpperCase()}`:'';
    ev.textContent=`side to move — W ${(sn.wdl[0]*100).toFixed(0)}% `+
      `D ${(sn.wdl[1]*100).toFixed(0)}% L ${(sn.wdl[2]*100).toFixed(0)}% · `+
      `eval ${sn.value>0?'+':''}${sn.value} · confidence α₀ ${sn.conc}${solved}`+
      (snInfo?` · ${snInfo}`:'');
  } else { wdl.style.display='none'; ev.textContent=snInfo; }

  document.getElementById('navlabel').textContent=live
    ?`live (after move ${last})`
    :`viewing after move ${showIdx} of ${last} — “live ⏭” to return`;
  document.getElementById('movelog').innerHTML=(S.move_log||[]).map((m,i)=>
    `<span class="mv ${(!live&&showIdx===i+1)?'cur':''}" onclick="viewAt(${i+1})">`+
    `${(i%2===0)?(i/2+1)+'.':''} ${m}</span>`).join(' ');
}
</script>
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Sessions + HTTP
# ═══════════════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, mode, human, sims, pick, nets, names, device, wave,
                 snap_secs):
        self.mode = mode                    # 'play' | 'watch'
        self.human = human                  # player id in play mode, None in watch
        self.sims = sims                    # >0 fixed budget, 0 = manual/indefinite
        self.pick = pick                    # 'best' | 'thompson'
        self.nets = nets
        self.names = names                  # player id -> model label
        self.device = device
        self.wave = wave
        self.snap_secs = snap_secs
        self.state = GAME.new_initial_state()
        self.snapshots = []
        self.move_log = []
        self.analysis_hist = [None]         # [i] = analysis for the move out of i-1
        self.board_hist = []
        self.actions = []                   # applied columns (for take-back)
        self.movers = []                    # player who made each ply
        self._snap_board()
        self.status = 'init'
        self.searcher = None
        self.stop_evt = threading.Event()   # interrupt search -> AI commits best
        self.abort_evt = threading.Event()  # interrupt search -> DON'T commit
        self.analyze_stop = threading.Event()
        self.analyzing = False
        self.hint = None                    # analysis of the current human position
        self.hint_action = None
        self.lock = threading.RLock()
        self.thread = None
        self.kick()

    def _snap_board(self):
        self.board_hist.append(str(self.state))

    def engine_to_move(self):
        return (not self.state.is_terminal() and
                (self.human is None or self.state.current_player() != self.human))

    def kick(self):
        with self.lock:
            if self.state.is_terminal():
                self.status = 'over'
                return
            if not self.engine_to_move():
                self.status = 'human_turn'
                return
            if self.thread is not None and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._think_loop, daemon=True)
            self.thread.start()

    def _think_loop(self):
        while self.engine_to_move():
            cur = self.state.current_player()
            searcher = Searcher(self.nets[cur], self.state, self.device,
                                self.wave, self.pick)
            with self.lock:
                self.snapshots = []
                self.searcher = searcher
                self.status = 'thinking'
            searcher.run(max_sims=(self.sims or None), stop_evt=self.stop_evt,
                         snap_cb=self._snap, snap_secs=self.snap_secs)
            if self.abort_evt.is_set():          # take-back aborted the search
                with self.lock:
                    self.searcher = None
                return
            self._snap(searcher)
            with self.lock:
                action = searcher.best()
                self.analysis_hist.append(searcher.snapshot())
                self.move_log.append(move_label(cur, action))
                self.state.apply_action(action)
                self.actions.append(int(action)); self.movers.append(cur)
                self._snap_board()
                self.stop_evt.clear()
                self.searcher = None
            if self.mode == 'watch' and self.sims:
                time.sleep(0.3)                  # let watchers see each move land
        with self.lock:
            self.status = 'over' if self.state.is_terminal() else 'human_turn'

    def _snap(self, searcher):
        with self.lock:
            snap = searcher.snapshot()
            if not self.snapshots or snap['sims'] > self.snapshots[-1]['sims']:
                self.snapshots.append(snap)

    def human_move(self, col, analysis=None):
        with self.lock:
            if self.status != 'human_turn':
                return False, 'not your turn'
            if col not in self.state.legal_actions():
                return False, 'that column is full'
            mover = self.state.current_player()
            self.analysis_hist.append(analysis)   # None, or the played hint
            self.move_log.append(move_label(mover, col))
            self.state.apply_action(col)
            self.actions.append(int(col)); self.movers.append(mover)
            self._snap_board()
            self.hint = None; self.hint_action = None
            self.snapshots = []
        self.kick()
        return True, ''

    def commit_ai(self):
        self.stop_evt.set()
        return True, ''

    # ── Take-back ────────────────────────────────────────────────────────────
    def take_back(self):
        """Undo back to your previous decision point (your last move plus any AI
        reply). Works whether it is your turn or the AI is still thinking — in
        that case the search is aborted WITHOUT committing its move."""
        if self.human is None:
            return False, 'take-back is only available in play mode'
        with self.lock:
            if self.human not in self.movers:
                return False, 'no moves to take back'
        self.abort_evt.set(); self.stop_evt.set(); self.analyze_stop.set()
        th = self.thread
        if th is not None and th.is_alive():
            th.join(timeout=5.0)
        with self.lock:
            i = [j for j, m in enumerate(self.movers) if m == self.human][-1]
            del self.actions[i:]; del self.movers[i:]
            del self.move_log[i:]; del self.analysis_hist[i + 1:]  # [0] = start
            del self.board_hist[i + 1:]
            st = GAME.new_initial_state()
            for a in self.actions:
                st.apply_action(int(a))
            self.state = st
            self.snapshots = []; self.searcher = None
            self.hint = None; self.hint_action = None; self.analyzing = False
            self.abort_evt.clear(); self.stop_evt.clear(); self.analyze_stop.clear()
            self.status = 'human_turn'
        return True, ''

    # ── Analyze the current (human) position without committing a move ────────
    def analyze(self):
        with self.lock:
            if self.status != 'human_turn' or self.state.is_terminal():
                return False, 'can only analyze on your turn'
            if self.analyzing:
                return True, ''
            self.analyzing = True
            self.analyze_stop.clear()
            self.snapshots = []
            self.hint = None; self.hint_action = None
            self.status = 'analyzing'
            net, st = self.nets[self.state.current_player()], self.state.clone()
        self.thread = threading.Thread(target=self._analyze_loop,
                                       args=(net, st), daemon=True)
        self.thread.start()
        return True, ''

    def _analyze_loop(self, net, st):
        searcher = Searcher(net, st, self.device, self.wave, self.pick)
        with self.lock:
            self.searcher = searcher
        # Manual mode analyses until you stop it; otherwise the normal budget.
        searcher.run(max_sims=(self.sims or None), stop_evt=self.analyze_stop,
                     snap_cb=self._snap, snap_secs=self.snap_secs)
        self._snap(searcher)
        with self.lock:
            self.hint = searcher.snapshot()
            self.hint_action = searcher.best()
            self.searcher = None
            self.analyzing = False
            if self.status == 'analyzing':
                self.status = 'human_turn'

    def stop_analyze(self):
        self.analyze_stop.set()
        return True, ''

    def play_hint(self):
        """Commit the AI's suggested move (from the last analysis) as your own."""
        with self.lock:
            if self.hint_action is None or self.status != 'human_turn':
                return False, 'no suggestion to play'
            col, analysis = int(self.hint_action), self.hint
        return self.human_move(col, analysis=analysis)

    def to_json(self):
        with self.lock:
            st = self.state
            human_turn = self.status == 'human_turn'
            legal = ([int(a) for a in st.legal_actions()]
                     if (human_turn and not st.is_terminal()) else [])
            return {
                'board': str(st),
                'legal': legal,
                'status': self.status,
                'mode': self.mode,
                'human': self.human,
                'first_player': FIRST_PLAYER,
                'current_player': (int(st.current_player())
                                   if not st.is_terminal() else -1),
                'engines': {str(p): n for p, n in self.names.items()},
                'manual': self.sims == 0,
                'pick': self.pick,
                'thinking_sims': self.searcher.n_sims() if self.searcher else 0,
                'snapshots': self.snapshots,
                'move_log': self.move_log,
                'board_hist': self.board_hist,
                'analysis_hist': self.analysis_hist,
                'analyzing': self.analyzing,
                'can_takeback': self.human is not None and self.human in self.movers,
                'hint': self.hint,
                'hint_col': (int(self.hint_action)
                             if self.hint_action is not None else None),
                'terminal': st.is_terminal(),
                'returns': st.returns() if st.is_terminal() else None,
            }


SESSIONS = {}
NETS = {}
NAMES = {}
DEVICE_ARG = 'cpu'
SNAP_SECS = 2.0
WAVE = 8


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = _json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == '/state':
            sid = parse_qs(u.query).get('sid', [''])[0]
            s = SESSIONS.get(sid)
            self._json(s.to_json() if s else {'error': 'no such session'},
                       200 if s else 404)
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = _json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            self._json({'error': 'bad json'}, 400)
            return
        if self.path == '/new':
            mode = req.get('mode', 'play')
            if mode == 'watch':
                human = None
            else:
                human = (FIRST_PLAYER if req.get('side', 'first') == 'first'
                         else 1 - FIRST_PLAYER)
            sims = max(0, int(req.get('sims', 400)))
            pick = 'thompson' if req.get('pick') == 'thompson' else 'best'
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = Session(mode, human, sims, pick, NETS, NAMES,
                                    DEVICE_ARG, WAVE, SNAP_SECS)
            self._json({'sid': sid})
            return
        s = SESSIONS.get(req.get('sid', ''))
        if s is None:
            self._json({'error': 'no such session'}, 404)
            return
        if self.path == '/move':
            try:
                col = int(req.get('col', -1))
            except (TypeError, ValueError):
                col = -1
            ok, msg = s.human_move(col)
        elif self.path == '/ai_commit':
            ok, msg = s.commit_ai()
        elif self.path == '/takeback':
            ok, msg = s.take_back()
        elif self.path == '/analyze':
            ok, msg = s.analyze()
        elif self.path == '/stop_analyze':
            ok, msg = s.stop_analyze()
        elif self.path == '/play_hint':
            ok, msg = s.play_hint()
        else:
            self._json({'error': 'not found'}, 404)
            return
        self._json({'ok': ok, 'error': msg})


def load_model(path, device):
    """Load a checkpoint and infer the network shape from its weights.  Accepts
    both a bare `bench_*.pt` state dict and a full `latest.pt` (which also
    records the Config the run used).  Returns (net, label, cfg_or_None)."""
    blob = torch.load(path, map_location='cpu', weights_only=False)
    cfg = None
    if isinstance(blob, dict) and isinstance(blob.get('model'), dict):
        cfg = blob.get('cfg')
        sd = blob['model']
    else:
        sd = blob
    missing = [k for k in ('stem.0.weight', 'v_out.weight', 'a_out.weight')
               if k not in sd]
    if missing:
        raise ValueError(
            f'{path}: not a ThompsonZero-C4 checkpoint (missing {missing}). '
            f'This server drives the two-Dirichlet-head net from '
            f'connect4_dirichlet_utils.py; the chess/boop engines have their '
            f'own servers.')
    channels = sd['stem.0.weight'].shape[0]
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    head_ch = sd['head.0.weight'].shape[0]
    actions = sd['a_out.weight'].shape[0] // 4
    if actions != c4._NUM_ACTIONS:
        raise ValueError(f'{path}: action head covers {actions} actions, but '
                         f'Connect 4 has {c4._NUM_ACTIONS}')
    net = c4.C4DirichletNet(channels, blocks, head_ch).to(device)
    net.load_state_dict(sd)
    net.eval()
    print(f'Loaded {path}: {channels} channels x {blocks} blocks, head {head_ch} '
          f'({sum(p.numel() for p in net.parameters()):,} params)')
    return net, os.path.basename(path), cfg


def main():
    global DEVICE_ARG, SNAP_SECS, WAVE
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--model2', default=None)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--search-agg', default=None, choices=c4.AGGREGATIONS)
    ap.add_argument('--wave', type=int, default=8)
    ap.add_argument('--snapshot-secs', type=float, default=2.0)
    args = ap.parse_args()
    DEVICE_ARG = args.device
    SNAP_SECS = args.snapshot_secs
    WAVE = max(1, args.wave)

    net, name, cfg = load_model(args.model, args.device)
    NETS[FIRST_PLAYER], NAMES[FIRST_PLAYER] = net, name
    if args.model2:
        net2, name2, _ = load_model(args.model2, args.device)
        NETS[1 - FIRST_PLAYER], NAMES[1 - FIRST_PLAYER] = net2, name2
    else:
        NETS[1 - FIRST_PLAYER] = net
        NAMES[1 - FIRST_PLAYER] = name

    # The evidence rule is a module-wide setting, so one choice covers the whole
    # process: the flag if given, else what --model was trained with (only a
    # full latest.pt records that), else the module default.
    agg = args.search_agg or (cfg or {}).get('search_agg') or c4.AGG_MIXTURE
    c4.set_search(search_agg=agg, target_agg=agg, selection='dirichlet')
    print(f'Search: Thompson sampling, evidence rule {agg!r}, wave {WAVE}')

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'Serving on http://localhost:{args.port}')
    print(f'To share:  ngrok http {args.port}')
    srv.serve_forever()


if __name__ == '__main__':
    main()
