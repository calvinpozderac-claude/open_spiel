"""Edax as an opponent, for an absolute strength yardstick that does not
saturate.

The MCTS ladder in othello_ladder tops out well below our model -- every rung
sits at 90-100% from about episode 4000, so anything past that point is
unmeasured. Edax is the standard open-source Othello engine and its `level`
setting (search depth) reaches far beyond anything we can beat, which puts a
real ceiling back on the ladder.

Protocol notes, all verified against wEdax-x86-64-v3 (v4.6) rather than
assumed:

  setboard <64 chars> <X|O>   64 squares in row-major order from a1, using
                              'X' black, 'O' white, '-' empty; then the side
                              to move.  This is stateless -- the position is
                              pushed every move -- so the adapter can never
                              drift out of sync with the real game, which a
                              replay-the-moves protocol can.
  go                          engine moves; replies "Edax plays D3"

  -l <n>       search depth: THE skill dial
  -n 1         single-threaded, so parallelism happens across games instead
  -book-usage off   an opening book would make it play canned openings and
                    distort the ladder
  -eval-file   absolute path, so cwd does not matter

Coordinates are mapped through OpenSpiel's own action_to_string rather than
recomputed, so the row/col convention cannot silently disagree.
"""
import os
import re
import subprocess

EDAX_DIR = r'C:\Users\Calvin Pozderac\Documents\edax'
EDAX_EXE = os.path.join(EDAX_DIR, 'wEdax-x86-64-v3.exe')
EDAX_EVAL = os.path.join(EDAX_DIR, 'data', 'eval.dat')

_PLAYS = re.compile(r'Edax plays (\w+)', re.I)
_BLACK, _WHITE = 0, 1


def _action_map(game):
    """{'d3': 19, ..., 'pass': 64} straight from OpenSpiel."""
    st = game.new_initial_state()
    n = game.num_distinct_actions()
    return {st.action_to_string(0, a).strip().lower(): a for a in range(n)}


def board_string(state):
    """(64-char board, side-to-move char) in Edax's format.

    Planes are [empty, own, opponent] relative to the player asked, so asking
    as player 0 gives black in plane 1 and white in plane 2."""
    import numpy as np
    t = np.asarray(state.observation_tensor(_BLACK)).reshape(3, 8, 8)
    black, white = t[1], t[2]
    cells = []
    for r in range(8):
        for c in range(8):
            cells.append('X' if black[r][c] else ('O' if white[r][c] else '-'))
    side = 'X' if state.current_player() == _BLACK else 'O'
    return ''.join(cells), side


class EdaxBot:
    """One persistent Edax process. Loading eval.dat costs ~a second, so the
    process is reused across moves and games rather than respawned.

    A plain fixed-level opponent. For a tunable one, see EdaxMixBot."""

    def __init__(self, game, level, seed=0,
                 exe=EDAX_EXE, eval_file=EDAX_EVAL):
        self.game = game
        self.level = level
        self.amap = _action_map(game)
        self.proc = subprocess.Popen(
            [exe, '-q', '-l', str(level), '-n', '1',
             '-book-usage', 'off', '-eval-file', eval_file],
            cwd=EDAX_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._drain_to_prompt()

    def _drain_to_prompt(self):
        """Consume the banner/book chatter emitted at startup."""
        self.proc.stdin.write('\n')
        self.proc.stdin.flush()
        for _ in range(200):
            line = self.proc.stdout.readline()
            if line.strip().endswith('>') or line.strip() == '>':
                return
            if not line:
                return

    def step(self, state):
        """Edax's move for `state`, as an OpenSpiel action id."""
        legal = state.legal_actions()
        if len(legal) == 1 and self.amap.get('pass') == legal[0]:
            return legal[0]
        board, side = board_string(state)
        self.proc.stdin.write(f'setboard {board} {side}\ngo\n')
        self.proc.stdin.flush()
        for _ in range(400):
            line = self.proc.stdout.readline()
            if not line:
                break
            m = _PLAYS.search(line)
            if not m:
                continue
            tok = m.group(1).strip().lower()
            if tok in ('ps', 'pass'):
                return self.amap['pass']
            a = self.amap.get(tok)
            if a is None or a not in legal:
                raise RuntimeError(
                    f'Edax returned {tok!r} -> {a}, not legal here '
                    f'(legal={[state.action_to_string(0, x) for x in legal]})')
            return a
        raise RuntimeError('Edax produced no move')

    def close(self):
        try:
            self.proc.stdin.write('quit\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EdaxMixBot:
    """Opponent that plays level `lo` with probability `p_lo` and level `hi`
    otherwise -- a strength dial built from two COMPETENT players.

    This replaces an earlier design that handicapped Edax by substituting
    uniformly random legal moves. Mixing levels is better on two counts: every
    move played is one a real engine chose, so the opponent never throws a
    game away on a blunder, and the two endpoints are meaningful reference
    points rather than "Edax" and "noise".

    A dial is needed because Edax's own `level` has no resolution in our
    range: measured against bench_OPT@20000 at 64 sims, level 1 scores 50%,
    level 2 scores 0%, and level 0 loses every game to level 1. p_lo
    interpolates continuously between those two rungs.

    Note the mixture is also the only source of game-to-game variety in a
    match between two deterministic engines -- at p_lo of exactly 0 or 1 every
    game from a given starting position is identical.
    """

    def __init__(self, game, p_lo, lo=1, hi=2, seed=0,
                 exe=EDAX_EXE, eval_file=EDAX_EVAL):
        import numpy as np
        self.p_lo = p_lo
        self._rng = np.random.default_rng(seed)
        self.lo = EdaxBot(game, level=lo, exe=exe, eval_file=eval_file)
        self.hi = EdaxBot(game, level=hi, exe=exe, eval_file=eval_file)

    def step(self, state):
        pick = self.lo if self._rng.random() < self.p_lo else self.hi
        return pick.step(state)

    def close(self):
        self.lo.close()
        self.hi.close()
