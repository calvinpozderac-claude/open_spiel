"""How many positions does our Edax actually search at each level?

The OLIVAW comparison rests on one assumption: that their E4 (alpha-beta depth
limited to 4) is the same opponent as our Edax L1-L4, since both are the `-l`
flag on the same engine. If our level-k searches a very different number of
positions than their reported counts, the two ladders are not the same ladder
and the comparison is worthless.

OLIVAW's reported max positions searched (Table I):
    E4  1.7 +/- 0.2 x 10^4      E8  1.1 +/- 0.3 x 10^6
    E6  1.0 +/- 0.2 x 10^5      E10 8.3 +/- 3.2 x 10^6

Runs our own binary WITHOUT -q so the search line is visible, on midgame
positions, and reports nodes per move per level.

    python edax_nodes.py
"""
import os
import re
import subprocess

import numpy as np

import edax_bot as eb

LEVELS = [1, 2, 3, 4, 6, 8, 10]
N_POS = 6
MID_PLIES = 20


def probe(level, boards):
    """Run one Edax at `level`, verbose, and pull node counts from its output."""
    proc = subprocess.Popen(
        [eb.EDAX_EXE, '-l', str(level), '-n', '1',
         '-book-usage', 'off', '-eval-file', eb.EDAX_EVAL],
        cwd=eb.EDAX_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    for board, side in boards:
        proc.stdin.write(f'setboard {board} {side}\ngo\n')
    proc.stdin.write('quit\n')
    proc.stdin.flush()
    text = proc.stdout.read()
    proc.wait(timeout=30)
    # Edax prints a search table; node counts appear as bare integers or with
    # M/k suffixes depending on build. Grab every plausible count on lines that
    # also carry a depth and a score.
    for line in text.splitlines():
        m = re.search(r'\b(\d[\d\s]{3,})\s*(?:nodes|N)\b', line, re.I)
        if m:
            out.append(int(re.sub(r'\s', '', m.group(1))))
    return out, text


def main():
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    rng = np.random.default_rng(5)
    boards = []
    while len(boards) < N_POS:
        st = game.new_initial_state()
        for _ in range(MID_PLIES):
            if st.is_terminal():
                break
            la = st.legal_actions()
            st.apply_action(int(la[rng.integers(len(la))]))
        if not st.is_terminal():
            boards.append(eb.board_string(st))

    print(f'{N_POS} midgame positions ({64-MID_PLIES-4} empties), '
          f'{os.path.basename(eb.EDAX_EXE)}\n')
    first_raw = None
    for lv in LEVELS:
        nodes, raw = probe(lv, boards)
        if first_raw is None:
            first_raw = raw
        if nodes:
            print(f'  level {lv:>2}: median {int(np.median(nodes)):>12,} '
                  f'positions   (n={len(nodes)})')
        else:
            print(f'  level {lv:>2}: no node counts parsed')
    if not any(True for _ in []):
        pass
    print('\n--- raw output, level 1, first lines (to fix the parser if empty) ---')
    print('\n'.join(first_raw.splitlines()[:25]))


if __name__ == '__main__':
    main()
