"""Objective evaluation against exactly-solved Connect 4 positions.

Every strength number this project has produced so far is RELATIVE: Elo against
the run's own earlier checkpoints, or a head-to-head against another arm.  That
cannot tell you whether a run is improving in absolute terms, and it cannot tell
you when a run has died — the NaN collapse kept reporting a healthy-looking Elo
ladder for 8000 episodes while the network was scoring near random, because its
opponents were equally dead.

This module scores a bot against ground truth instead: positions whose
game-theoretic value is known exactly.

Test data
---------
Pascal Pons' test protocol, http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
Six sets of 1000 positions each, bucketed by how far into the game they are and
how deep the win is:

    Test_L1_R1  Begin-Easy     Test_L2_R1  Middle-Easy    Test_L3_R1  End-Easy
    Test_L1_R2  Begin-Medium   Test_L2_R2  Middle-Medium
    Test_L1_R3  Begin-Hard

Each line is `<moves> <score>`: a move sequence in 1-indexed columns
("4453..."), then the exact score for the player to move.  The score is
`+(22 - k)` if that player wins using their k-th stone, 0 for a draw, and
negative by the same count if they lose — so +1 is a win with the very last
stone and +18 is a win four moves from now.  `load_dir` reads that format;
nothing here assumes which bucket a file came from, the bucket boundaries are
derived from the data.

What is measured
----------------
`move_accuracy`  — the headline.  Fraction of positions where the bot plays a
    move that PRESERVES the theoretical outcome (a won position stays won, a
    drawn one stays drawn).  Engine-agnostic: it only needs a move, so
    ThompsonZero, AlphaZero, a random mover and anything added later are scored
    on exactly the same scale.  Also reports `perfect`, the stricter fraction
    that additionally wins as fast as possible.
`value_accuracy` — does the network's own value head get the win/draw/loss
    right, with no search at all?  This is the one that catches a dead network
    immediately.

The test files give the value of each POSITION but not of each MOVE, so
move accuracy needs the children solved.  That is what the solver below is for,
and the result is cached to disk (`solved_cache.json`) so the cost is paid once
per test set rather than once per evaluation.
"""

import json
import os

# ══════════════════════════════════════════════════════════════════════════════
#  Bitboard solver
#
#  Standard 7x6 layout: one 7-bit column per file (6 playable rows + a sentinel
#  bit that stops vertical alignments and carries from wrapping into the next
#  column).  `position` is the bitmap of the CURRENT player's stones and `mask`
#  of all stones, so `position ^ mask` is the opponent's — which makes switching
#  sides a single xor.  Score convention is Pons': a win with your k-th stone is
#  worth 22-k, which is exactly `(42 + 1 - moves) // 2` for a win available
#  right now, so the solver's numbers are directly comparable to the file's.
# ══════════════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 7, 6
_MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3
_MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3
_BOTTOM = 0
for _c in range(WIDTH):
    _BOTTOM |= 1 << (_c * (HEIGHT + 1))
_BOARD_MASK = _BOTTOM * ((1 << HEIGHT) - 1)
# Centre-out, the ordering that makes alpha-beta cut early.
_ORDER = sorted(range(WIDTH), key=lambda c: abs(c - WIDTH // 2))


def _top_mask_col(c):
    return 1 << ((HEIGHT - 1) + c * (HEIGHT + 1))


def _bottom_mask_col(c):
    return 1 << (c * (HEIGHT + 1))


def _column_mask(c):
    return ((1 << HEIGHT) - 1) << (c * (HEIGHT + 1))


def _alignment(pos):
    m = pos & (pos >> (HEIGHT + 1))          # horizontal
    if m & (m >> (2 * (HEIGHT + 1))):
        return True
    m = pos & (pos >> HEIGHT)                # diagonal /
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> (HEIGHT + 2))          # diagonal \
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    m = pos & (pos >> 1)                     # vertical
    return bool(m & (m >> 2))


def _compute_winning(position, mask):
    """Every empty cell that would complete a line for `position`'s owner."""
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)
    # horizontal
    p = (position << (HEIGHT + 1)) & (position << (2 * (HEIGHT + 1)))
    r |= p & (position << (3 * (HEIGHT + 1)))
    r |= p & (position >> (HEIGHT + 1))
    p = (position >> (HEIGHT + 1)) & (position >> (2 * (HEIGHT + 1)))
    r |= p & (position << (HEIGHT + 1))
    r |= p & (position >> (3 * (HEIGHT + 1)))
    # diagonal /
    p = (position << HEIGHT) & (position << (2 * HEIGHT))
    r |= p & (position << (3 * HEIGHT))
    r |= p & (position >> HEIGHT)
    p = (position >> HEIGHT) & (position >> (2 * HEIGHT))
    r |= p & (position << HEIGHT)
    r |= p & (position >> (3 * HEIGHT))
    # diagonal \
    p = (position << (HEIGHT + 2)) & (position << (2 * (HEIGHT + 2)))
    r |= p & (position << (3 * (HEIGHT + 2)))
    r |= p & (position >> (HEIGHT + 2))
    p = (position >> (HEIGHT + 2)) & (position >> (2 * (HEIGHT + 2)))
    r |= p & (position << (HEIGHT + 2))
    r |= p & (position >> (3 * (HEIGHT + 2)))
    return r & (_BOARD_MASK ^ mask)


class Position:
    """A Connect 4 position as (position, mask, moves).  Immutable-ish: `play`
    returns a new Position rather than mutating."""

    __slots__ = ('position', 'mask', 'moves')

    def __init__(self, position=0, mask=0, moves=0):
        self.position, self.mask, self.moves = position, mask, moves

    @classmethod
    def from_sequence(cls, seq):
        """`seq` is 1-indexed columns, e.g. '4453'.  Returns None if the
        sequence is illegal or ends the game early (Pons' files never do)."""
        p = cls()
        for ch in str(seq).strip():
            if not ch.isdigit():
                return None
            c = int(ch) - 1
            if not 0 <= c < WIDTH or not p.can_play(c) or p.is_winning_move(c):
                return None
            p = p.play(c)
        return p

    def can_play(self, c):
        return (self.mask & _top_mask_col(c)) == 0

    def play(self, c):
        mask = self.mask | (self.mask + _bottom_mask_col(c))
        return Position(self.position ^ self.mask, mask, self.moves + 1)

    def is_winning_move(self, c):
        # Must intersect possible(): winning_spots() marks every cell that would
        # complete a line, including ones higher up the column that are not
        # playable yet.  Without the intersection this reports a win for a
        # column whose landing square is harmless.
        return bool(self.winning_spots() & self.possible() & _column_mask(c))

    def winning_spots(self):
        return _compute_winning(self.position, self.mask)

    def opponent_winning_spots(self):
        return _compute_winning(self.position ^ self.mask, self.mask)

    def possible(self):
        return (self.mask + _BOTTOM) & _BOARD_MASK

    def possible_non_losing(self):
        """Playable cells that do not hand the opponent an immediate win.
        Returns 0 when every move loses."""
        possible = self.possible()
        opp = self.opponent_winning_spots()
        forced = possible & opp
        if forced:
            if forced & (forced - 1):
                return 0              # two separate threats: lost
            possible = forced
        return possible & ~(opp >> 1)  # never play directly under a threat

    def key(self):
        return self.position + self.mask

    def legal_columns(self):
        return [c for c in range(WIDTH) if self.can_play(c)]

    def is_draw(self):
        return self.moves == WIDTH * HEIGHT


def _half(x):
    """Integer division by two TRUNCATING TOWARD ZERO.

    Every score bound below comes from Pons' C++, where `/` truncates.  Python's
    `//` floors instead, so -5//2 is -3 where C++ gives -2 — an off-by-one in
    the search window that makes the null-window driver fail to converge."""
    return -((-x) // 2) if x < 0 else x // 2


class Solver:
    """Alpha-beta with a transposition table and a null-window driver.

    Strong-solves (exact score, not just win/draw/loss).  Endgame positions are
    microseconds; the Begin-* buckets are genuinely expensive, which is why
    results are cached rather than recomputed."""

    def __init__(self):
        self.table = {}
        self.nodes = 0

    def _negamax(self, p, alpha, beta):
        self.nodes += 1
        possible = p.possible_non_losing()
        if possible == 0:                       # every move loses next turn
            return -_half(WIDTH * HEIGHT - p.moves)
        if p.moves >= WIDTH * HEIGHT - 2:
            return 0                            # draw, no room left to win
        min_s = -_half(WIDTH * HEIGHT - 2 - p.moves)
        if alpha < min_s:
            alpha = min_s
            if alpha >= beta:
                return alpha
        max_s = _half(WIDTH * HEIGHT - 1 - p.moves)
        if beta > max_s:
            beta = max_s
            if alpha >= beta:
                return beta

        key = p.key()
        hit = self.table.get(key)
        if hit is not None:
            lo, hi = hit
            if hi <= alpha:
                return hi
            if lo >= beta:
                return lo
            if lo > alpha:
                alpha = lo
            if hi < beta:
                beta = hi
            if alpha >= beta:
                return alpha

        orig_alpha, orig_beta = alpha, beta
        # Order by how many winning cells the move creates, centre as tie-break.
        moves = []
        for c in _ORDER:
            bit = possible & _column_mask(c)
            if not bit:
                continue
            child = p.play(c)
            n = bin(child.opponent_winning_spots()).count('1')
            moves.append((-n, c, child))
        moves.sort(key=lambda t: t[0])

        best = -1 << 30
        for _n, _c, child in moves:
            s = -self._negamax(child, -beta, -alpha)
            if s > best:
                best = s
            if s > alpha:
                alpha = s
            if alpha >= beta:
                break

        lo = _MIN_SCORE - 1 if best <= orig_alpha else best
        hi = _MAX_SCORE + 1 if best >= orig_beta else best
        prev = self.table.get(key)
        if prev is not None:
            lo, hi = max(lo, prev[0]), min(hi, prev[1])
        self.table[key] = (lo, hi)
        return best

    def solve(self, p):
        """Exact score for the player to move."""
        if p.is_draw():
            return 0
        if p.winning_spots() & p.possible():
            return _half(WIDTH * HEIGHT + 1 - p.moves)
        lo = -_half(WIDTH * HEIGHT - p.moves)
        hi = _half(WIDTH * HEIGHT + 1 - p.moves)
        while lo < hi:                        # null-window binary search
            med = lo + _half(hi - lo)
            if med <= 0 and _half(lo) < med:
                med = _half(lo)
            elif med >= 0 and _half(hi) > med:
                med = _half(hi)
            if self._negamax(p, med, med + 1) <= med:
                hi = med
            else:
                lo = med + 1
        return lo

    def move_scores(self, p):
        """{column: score for the player to move} — the value of each legal move
        to the player making it.  `max(move_scores) == solve(p)`."""
        out = {}
        for c in p.legal_columns():
            if p.is_winning_move(c):
                out[c] = (WIDTH * HEIGHT + 1 - p.moves) // 2
            else:
                child = p.play(c)
                out[c] = 0 if child.is_draw() else -self.solve(child)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Test-set loading
# ══════════════════════════════════════════════════════════════════════════════
BUCKETS = ('Test_L1_R1', 'Test_L1_R2', 'Test_L1_R3',
           'Test_L2_R1', 'Test_L2_R2', 'Test_L3_R1')
BUCKET_NAMES = {'Test_L1_R1': 'Begin-Easy', 'Test_L1_R2': 'Begin-Medium',
                'Test_L1_R3': 'Begin-Hard', 'Test_L2_R1': 'Middle-Easy',
                'Test_L2_R2': 'Middle-Medium', 'Test_L3_R1': 'End-Easy'}
SOURCE_URL = 'http://blog.gamesolver.org/solving-connect-four/02-test-protocol/'


def parse_line(line):
    """'2252576253462244111 -1' → ('2252576253462244111', -1), or None."""
    parts = line.split()
    if len(parts) != 2:
        return None
    seq, score = parts
    try:
        return seq, int(score)
    except ValueError:
        return None


def load_file(path, limit=None):
    """[(sequence, score, Position)] for one test file."""
    out = []
    with open(path) as fh:
        for line in fh:
            rec = parse_line(line)
            if rec is None:
                continue
            seq, score = rec
            p = Position.from_sequence(seq)
            if p is None:
                raise ValueError(f'{path}: illegal move sequence {seq!r}')
            out.append((seq, score, p))
            if limit and len(out) >= limit:
                break
    return out


def load_dir(directory, buckets=None, limit=None):
    """{bucket: [(seq, score, Position)]} for whichever test files are present.

    The files are not redistributed with this repo; download them from
    SOURCE_URL and drop them in `directory`."""
    found = {}
    for b in (buckets or BUCKETS):
        path = os.path.join(directory, b)
        if os.path.exists(path):
            found[b] = load_file(path, limit)
    if not found:
        raise FileNotFoundError(
            f'no test files in {directory!r}. Expected any of {list(BUCKETS)} — '
            f'download them from {SOURCE_URL} and unpack them there.')
    return found


def verify(records, solver=None, log=print):
    """Re-solve every position and check the file's score.  A full pass on a
    bucket is strong mutual validation: an independent solver agreeing with
    Pons' labels on 1000 positions means both are right."""
    solver = solver or Solver()
    bad = []
    for seq, score, p in records:
        got = solver.solve(p)
        if got != score:
            bad.append((seq, score, got))
    log(f'verified {len(records) - len(bad)}/{len(records)} positions'
        + (f' — MISMATCHES: {bad[:5]}' if bad else ''))
    return bad


# ══════════════════════════════════════════════════════════════════════════════
#  Ground truth: position score + per-move scores, cached to disk
# ══════════════════════════════════════════════════════════════════════════════
def _sign(x):
    return (x > 0) - (x < 0)


def solve_records(records, cache=None, solver=None, log=None, every=100):
    """Attach per-move ground truth to `records`.

    Returns {sequence: (score, {column: score})}.  The test files give the value
    of the POSITION; move accuracy needs the value of each MOVE, which means
    solving every child.  That is the expensive part, so results are cached by
    move sequence and reused across evaluations and across runs."""
    solver = solver or Solver()
    cache = {} if cache is None else cache
    for i, (seq, score, p) in enumerate(records):
        if seq in cache:
            continue
        ms = solver.move_scores(p)
        cache[seq] = (score, {str(k): v for k, v in ms.items()})
        if log and every and (i + 1) % every == 0:
            log(f'  solved {i + 1}/{len(records)}')
    return cache


def load_cache(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_cache(path, cache):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(cache, fh)
    os.replace(tmp, path)


class Suite:
    """A set of solved positions plus the ground truth needed to score a bot.

    `records` is [(sequence, score, Position)] and `truth[sequence]` is
    (position_score, {column: move_score})."""

    def __init__(self, records, truth, name=''):
        self.records, self.truth, self.name = records, truth, name
        self._states = None

    def states(self, game=None):
        """pyspiel states for every record, built once and reused.  The bots
        need a real game state; the solver half of this module does not, which
        is why it is only built on demand."""
        if self._states is None:
            self._states = states_for(self.records, game)
        return self._states

    def __len__(self):
        return len(self.records)

    @classmethod
    def build(cls, directory, buckets=None, limit=None, cache_path=None,
              log=print):
        """Load the test files and make sure per-move ground truth exists.

        `cache_path` defaults to `solved_cache.json` beside the test files.  The
        first build on the deep buckets is slow (that is the point of those
        buckets); every build after it is a file read."""
        by_bucket = load_dir(directory, buckets, limit)
        cache_path = cache_path or os.path.join(directory, 'solved_cache.json')
        cache = load_cache(cache_path)
        n_before = len(cache)
        solver = Solver()
        out = {}
        for b, recs in by_bucket.items():
            missing = sum(1 for s, _, _ in recs if s not in cache)
            if missing:
                log(f'{BUCKET_NAMES.get(b, b)}: solving {missing} positions '
                    f'(one-off, cached to {os.path.basename(cache_path)})')
            solve_records(recs, cache, solver,
                          log=log if missing > 200 else None)
            out[b] = cls(recs, cache, BUCKET_NAMES.get(b, b))
        if len(cache) != n_before:
            save_cache(cache_path, cache)
        return out

    def check_labels(self, log=print):
        """The cached per-move scores must reproduce the file's position score.
        Catches a corrupt cache and cross-validates the solver against Pons."""
        bad = []
        for seq, score, _p in self.records:
            _s, ms = self.truth[seq]
            if not ms:
                continue
            if max(ms.values()) != score:
                bad.append((seq, score, max(ms.values())))
        log(f'{self.name}: {len(self.records) - len(bad)}/{len(self.records)} '
            f'positions agree with the file'
            + (f' — MISMATCHES {bad[:3]}' if bad else ''))
        return bad

    def evaluate(self, chooser, values=None):
        """Score one bot.

        `chooser(states) -> [column]` picks a move for every position.
        `values(states) -> [outcome in {-1,0,1}]` optionally predicts the
        win/draw/loss of the position itself, with no search.

        Returns a dict of rates in [0, 1]."""
        states = self.states()
        picks = list(chooser(states))
        n = len(self.records)
        opt = perfect = blunder = legal = 0
        for (seq, score, p), a in zip(self.records, picks):
            ms = self.truth[seq][1]
            v = ms.get(str(int(a)))
            if v is None:                       # illegal move: worst case
                continue
            legal += 1
            if _sign(v) == _sign(score):
                opt += 1
            if v == score:
                perfect += 1
            if _sign(v) < _sign(score):
                blunder += 1
        res = {'n': n, 'legal': legal / n if n else 0.0,
               'optimal': opt / n if n else 0.0,
               'perfect': perfect / n if n else 0.0,
               'blunder': blunder / n if n else 0.0}
        if values is not None and n:
            pred = list(values(states))
            dec = [(pv, sc) for pv, (_s, sc, _p) in zip(pred, self.records)
                   if sc != 0]
            # Always-predict-the-commonest-outcome.  Without it `value_acc` is
            # unreadable: an untrained net that answers "win" to everything
            # scores exactly this, which on a win-heavy bucket looks excellent.
            res['majority'] = max(
                sum(1 for _s, sc, _p in self.records if _sign(sc) == k)
                for k in (-1, 0, 1)) / n
            res['value_acc'] = sum(
                pv == _sign(sc)
                for pv, (_s, sc, _p) in zip(pred, self.records)) / n
            # Decisive positions only, so it does not depend on where a scalar
            # value head's draw band is drawn.  A net with no opinion (0)
            # matches neither sign and scores 0, which is the honest answer.
            res['value_sign'] = ((sum(_sign(pv) == _sign(sc) for pv, sc in dec)
                                  / len(dec)) if dec else float('nan'))
        return res


def report(results, log=print, title=''):
    """`results` is {bucket_name: metrics-dict} from Suite.evaluate."""
    if title:
        log(f'\n=== {title}')
    log(f'{"bucket":<16}{"n":>6}{"optimal":>10}{"perfect":>10}'
        f'{"blunder":>10}{"value":>9}{"base":>7}{"v-sign":>9}')
    for name, r in results.items():
        va = r.get('value_acc')
        vs = r.get('value_sign')
        mj = r.get('majority')
        log(f'{name:<16}{r["n"]:>6}{100 * r["optimal"]:>9.1f}%'
            f'{100 * r["perfect"]:>9.1f}%{100 * r["blunder"]:>9.1f}%'
            + (f'{100 * va:>8.1f}%' if va is not None else f'{"-":>9}')
            + (f'{100 * mj:>6.0f}%' if mj is not None else f'{"-":>7}')
            + (f'{100 * vs:>8.1f}%' if vs is not None else f'{"-":>9}'))
    tot = sum(r['n'] for r in results.values())
    if tot:
        for k in ('optimal', 'perfect', 'blunder'):
            w = sum(r[k] * r['n'] for r in results.values()) / tot
            log(f'{"ALL " + k:<16}{tot:>6}{100 * w:>9.1f}%')
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Engine adapters
#
#  Everything above is pure Python and needs neither torch nor pyspiel.  These
#  turn a network into the (chooser, values) pair Suite.evaluate wants, batching
#  the whole test set into as few forward passes as possible so the metric is
#  cheap enough to run at every eval interval.
# ══════════════════════════════════════════════════════════════════════════════
def states_for(records, game=None):
    """[pyspiel state] for a Suite's records, replaying each move sequence."""
    import connect4_dirichlet_utils as c4
    game = game or c4.load_game()
    out = []
    for seq, _score, _p in records:
        st = game.new_initial_state()
        for ch in str(seq).strip():
            st.apply_action(int(ch) - 1)
        out.append(st)
    return out


def random_player(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)

    def chooser(states):
        return [int(s.legal_actions()[rng.integers(len(s.legal_actions()))])
                for s in states]
    return chooser, None


def thompson_player(net, device='cpu', sims=0, batch=256, eval_temp=6.0,
                    game=None, lookahead=False):
    """ThompsonZero.  sims>0 runs the real MCTS per position.

    At sims=0 there are two different search-free moves, and they are NOT
    interchangeable:

    `lookahead=False` reads the ACTION heads directly — one batched forward for
        the whole suite, and the natural way to track this engine's own
        progress.  It expands no nodes at all.
    `lookahead=True` applies each legal move and scores the child with the STATE
        head, using the true outcome for terminal children.  This is what
        alphazero_player does at sims=0, because AlphaZero has no per-action
        value to read.  Use it on BOTH engines for a cross-engine comparison:
        one-ply lookahead sees every immediate win and loss, which is worth
        tens of points on the endgame buckets, so comparing it against a
        no-lookahead reading measures the lookahead, not the network."""
    import numpy as np
    import connect4_dirichlet_utils as c4

    def _lookahead_move(s):
        leg = s.legal_actions()
        vals = [None] * len(leg)
        pend_i, pend_s = [], []
        for i, a in enumerate(leg):
            cs = s.clone()
            cs.apply_action(int(a))
            if cs.is_terminal():
                vals[i] = float(cs.returns()[s.current_player()])
            else:
                pend_i.append(i)
                pend_s.append(cs)
        if pend_s:
            v3, vc, _p3, _cf, _o = c4.nn_eval_states(net, device, pend_s)
            a3 = np.maximum(np.asarray(vc)[:, None] * np.asarray(v3),
                            c4.ALPHA_FLOOR)
            for i, val in zip(pend_i, c4.dir_value(a3)):
                vals[i] = -float(val)      # child is the opponent's turn
        return int(leg[int(np.argmax(vals))])

    def chooser(states):
        if sims == 0 and lookahead:
            return [_lookahead_move(s) for s in states]
        if sims > 0:
            g = game or c4.load_game()
            rng = np.random.default_rng(0)
            bot = c4.C4MCTSBot(g, net, device, sims, batch_size=8,
                               temp=eval_temp, random_state=rng)
            return [c4.root_pick(bot.mcts_search(s), rng, thompson=False)
                    for s in states]
        out = []
        for i in range(0, len(states), batch):
            chunk = states[i:i + batch]
            _v3, _vc, p3, cf, _o = c4.nn_eval_states(net, device, chunk)
            for j, s in enumerate(chunk):
                leg = s.legal_actions()
                a = np.maximum(cf[j][leg, None] * p3[j][leg], c4.ALPHA_FLOOR)
                out.append(int(leg[int(c4.dir_value(a).argmax())]))
        return out

    def values(states):
        out = []
        for i in range(0, len(states), batch):
            chunk = states[i:i + batch]
            v3, _vc, _p3, _cf, _o = c4.nn_eval_states(net, device, chunk)
            # v3 is (win, draw, loss) for the player to move.
            idx = np.asarray(v3).argmax(-1)
            out.extend(int({c4._WIN: 1, c4._DRAW: 0, c4._LOSS: -1}[int(k)])
                       for k in idx)
        return out
    return chooser, values


def alphazero_player(net, device='cpu', sims=0, batch=256, draw_band=0.25,
                     game=None):
    """AlphaZero.  sims=0 uses the one-ply value lookahead (the policy head is a
    search prior, not a per-move value, so reading its argmax would not be the
    same measurement the ThompsonZero adapter makes).  `draw_band` is the |v|
    below which the scalar head is read as a draw — only `value_acc` depends on
    it; `value_sign` is measured on decisive positions and does not."""
    import numpy as np
    import connect4_alphazero_utils as az

    def chooser(states):
        if sims > 0:
            import connect4_dirichlet_utils as c4
            g = game or c4.load_game()
            rng = np.random.default_rng(0)
            bot = az.AZMCTSBot(g, net, device, sims, batch_size=8,
                               random_state=rng)
            return [az.root_pick(bot.mcts_search(s), rng, sample=False)
                    for s in states]
        return [az.value_greedy_move(net, s, device) for s in states]

    def values(states):
        out = []
        for i in range(0, len(states), batch):
            chunk = states[i:i + batch]
            _lg, v, _o = az.nn_eval_states(net, device, chunk)
            for x in np.asarray(v).reshape(-1):
                out.append(0 if abs(x) < draw_band else int(_sign(x)))
        return out
    return chooser, values


# ══════════════════════════════════════════════════════════════════════════════
#  Generating a test set locally
#
#  The Pons files are not redistributed here, and this sandbox cannot reach the
#  blog.  `generate` writes the SAME format from positions solved by the solver
#  above, so the harness is testable and usable before the download — and the
#  two can be compared directly, since both carry exact scores.
# ══════════════════════════════════════════════════════════════════════════════
def generate(path, n=200, min_moves=24, max_moves=34, seed=0, log=print):
    """Write `n` solved positions in Pons' `<moves> <score>` format."""
    import random as _random
    rng = _random.Random(seed)
    solver = Solver()
    seen, lines = set(), []
    while len(lines) < n:
        seq, p = '', Position()
        target = rng.randint(min_moves, max_moves)
        ok = True
        while p.moves < target:
            legal = [c for c in p.legal_columns() if not p.is_winning_move(c)]
            if not legal:
                ok = False
                break
            c = rng.choice(legal)
            seq += str(c + 1)
            p = p.play(c)
        if not ok or p.is_draw() or not p.legal_columns() or seq in seen:
            continue
        seen.add(seq)
        lines.append(f'{seq} {solver.solve(p)}')
        if log and len(lines) % 50 == 0:
            log(f'  generated {len(lines)}/{n}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def fetch(directory, url=None, log=print):
    """Put the Pons test files in `directory`.

    `url` may point at the zip or at a single test file; a zip is unpacked.
    No URL is hard-coded on purpose — the download location is not something
    this module can verify, and a stale guess fails more confusingly than an
    instruction.  Get the archive from SOURCE_URL and either pass its URL here
    or just unzip it into `directory` yourself; the file names are in BUCKETS.
    """
    import shutil
    import urllib.request
    import zipfile
    os.makedirs(directory, exist_ok=True)
    if url is None:
        have = [b for b in BUCKETS if os.path.exists(os.path.join(directory, b))]
        log(f'{directory}: {len(have)}/{len(BUCKETS)} test files present'
            + (f' ({", ".join(have)})' if have else ''))
        if len(have) < len(BUCKETS):
            log(f'Download the rest from {SOURCE_URL} and unpack them here, '
                f'or call fetch(directory, url=...). Expected names: '
                f'{", ".join(BUCKETS)}')
        return have
    dest = os.path.join(directory, os.path.basename(url) or 'download')
    log(f'fetching {url}')
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as fh:
        shutil.copyfileobj(r, fh)
    if zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as z:
            for member in z.namelist():
                name = os.path.basename(member)
                if name in BUCKETS:
                    with z.open(member) as src, \
                            open(os.path.join(directory, name), 'wb') as out:
                        shutil.copyfileobj(src, out)
        os.remove(dest)
    return [b for b in BUCKETS if os.path.exists(os.path.join(directory, b))]
