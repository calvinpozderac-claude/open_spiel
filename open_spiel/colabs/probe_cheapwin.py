"""Does the "cheap win" actually win?

probe_analyze showed the deployed 'predictor' rule is held back by its linear
form and a missing game-phase feature: an MLP on 3 scalars + phase lifts
held-out r2 on the convergence labels from 0.15 to 0.63. This asks the only
question that matters -- whether that better PREDICTION turns into better
ALLOCATION -- and it is a real question rather than a formality, because
probe_transfer showed the convergence labels are weakly coupled to which move
actually gets played.

Scored the same way as probe_allocation/probe_transfer: every rule is bisected
to the same mean sims/move, then judged on how often the move it would play
matches the move a full search settles on.

Two guards make the comparison honest, and without them a learned allocator
looks far better than it is:

  out-of-fold predictions   The model is never asked about a game it trained
                            on.  K folds by GAME, each fitted on the other
                            folds, so all 2000 evaluation positions get a
                            prediction from a model that has not seen them.
  cross-run scoring         Labels come from run 1, move agreement is scored
                            on run 2.  A model fitted against run-1 labels and
                            scored on run 1 would be rewarded for memorising
                            that run's search noise, which no deployed rule
                            could ever reproduce.

Rules compared, at each budget:
  fixed / conf / predictor  the incumbents (predictor uses its shipped coefs)
  phase only                is a stones-on-board count the whole cheap win?
  linear / MLP (3+phase)    the cheap win, fitted against each candidate label

    python probe_cheapwin.py [tag]
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

import probe_analyze as pa
import probe_allocation as al

N_FOLDS = 5
SEED = 0


def _fit_predict(Xtr, ytr, Xva, yva, Xall, kind):
    """Fit on the training fold, return predictions for every row of Xall."""
    if kind == 'linear':
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return np.column_stack([np.ones(len(Xall)), Xall]) @ coef

    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state, bad = -1e9, None, 0
    for _ep in range(300):
        model.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 256):
            i = perm[k:k + 256]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xt[i]).squeeze(-1), yt[i]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).squeeze(-1).numpy() * sd + mu
        v = pa.r2(pv, yva)
        if v > best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xall, dtype=torch.float32)
                     ).squeeze(-1).numpy() * sd + mu


def out_of_fold(X, y, gid, eval_rows, kind):
    """Out-of-fold log-sims predictions for `eval_rows`, folded by game."""
    rng = np.random.default_rng(SEED)
    games = np.unique(gid)
    rng.shuffle(games)
    folds = np.array_split(games, N_FOLDS)
    pred = np.full(len(eval_rows), np.nan)
    for f in folds:
        held = set(f.tolist())
        te_sel = np.array([gid[r] in held for r in eval_rows])
        if not te_sel.any():
            continue
        tr_pool = np.array([g not in held for g in gid])
        tr_games = np.array([g for g in games if g not in held])
        va_games = set(tr_games[:max(1, len(tr_games) // 6)].tolist())
        va = tr_pool & np.array([g in va_games for g in gid])
        tr = tr_pool & ~va
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        pred[te_sel] = _fit_predict((X[tr] - mu) / sd, y[tr],
                                    (X[va] - mu) / sd, y[va],
                                    (X[eval_rows[te_sel]] - mu) / sd, kind)
    return pred


def analyze(tag, cal2):
    d = np.load(f'probe_data_{tag}.npz')
    r = np.load(f'probe_replicate_{tag}.npz')
    cap, ce = int(d['cap']), int(d['check_every'])
    gid, obs, F = d['game_id'], d['obs'], d['feats']
    n = len(F)
    stones = obs.reshape(n, 3, 64)[:, :2].sum((1, 2))

    labels_main = pa.all_labels(d, cap, ce)
    rows = r['main_index']
    b2, n2 = r['best_idx'], r['n_valid']
    Fe = F[rows]

    print(f'\n{"="*80}\n{tag}: does the cheap win beat the incumbents on MOVE '
          f'AGREEMENT?\n  {len(rows)} positions, out-of-fold predictions, '
          f'scored on an independent run\n{"="*80}')

    feat_sets = {
        'phase only': np.column_stack([stones]),
        '3+phase': np.column_stack([F[:, 1], F[:, 2], F[:, 3], stones]),
    }

    coef = np.array(cal2['rules']['predictor']['cfg']['predictor_coef'])
    print(f'{"":34}' + ''.join(f'{f"mean {int(t)}":>11}' for t in al.TARGETS))

    fin = b2[np.arange(len(b2)), np.maximum(n2 - 1, 0)]
    hits = {}

    def show(name, alloc_fn):
        cells, per_pos = [], []
        for t in al.TARGETS:
            idx = al._snap(alloc_fn(t), cap, ce)
            ok = b2[np.arange(len(b2)), idx] == fin
            per_pos.append(ok)
            cells.append(float(ok.mean()))
        hits[name] = per_pos
        print(f'  {name:<32}' + ''.join(f'{c:>10.1%} ' for c in cells))
        return cells

    base = show('fixed', lambda t: np.full(len(rows), t))
    for nm in ('conf', 'predictor'):
        show(nm, lambda t, nm=nm: al.rule_sims(
            nm, Fe, al.calibrate(
                lambda v: al.rule_sims(nm, Fe, v, coef), t), coef))

    for label_nm in ('patience', 'action_eps', 'decision'):
        y = np.log(np.maximum(labels_main[label_nm], 1.0))
        print(f'  -- fitted against the {label_nm} label --')
        for fs_nm, X in feat_sets.items():
            for kind in ('linear', 'mlp'):
                if fs_nm == 'phase only' and kind == 'linear':
                    continue
                p = out_of_fold(X, y, gid, rows, kind)
                oof_r2 = pa.r2(p, y[rows])
                cells = show(f'{kind}({fs_nm}) [oof r2={oof_r2:+.2f}]',
                             lambda t, p=p: np.clip(
                                 np.exp(p) * al.calibrate(
                                     lambda v: np.clip(np.exp(p) * v,
                                                       al.ALLOC_MIN,
                                                       al.ALLOC_MAX), t),
                                 al.ALLOC_MIN, al.ALLOC_MAX))
                del cells

    y1 = labels_main['decision'][rows]
    show('oracle cross-run (ceiling)',
         lambda t: al.oracle_alloc(y1, t, cap))
    print(f'  {"(fixed baseline again)":<32}'
          + ''.join(f'{c:>10.1%} ' for c in base))

    # Every rule is scored on the SAME positions, so the paired difference is
    # far tighter than each rule's own standard error and is the right test
    # for "is this actually an improvement".
    print(f'\n  paired difference vs predictor (same positions; '
          f'+/- 1 SE, n={len(rows)}):')
    ref = hits['predictor']
    for name, per_pos in hits.items():
        if name == 'predictor':
            continue
        parts = []
        for a, b in zip(per_pos, ref):
            dd = a.astype(float) - b.astype(float)
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            sig = '*' if abs(dd.mean()) > 2 * se else ' '
            parts.append(f'{dd.mean():+7.2%}+-{se:5.2%}{sig}')
        print(f'    {name:<32}' + '  '.join(parts))
    print('    (* = differs from predictor by more than 2 SE)')


if __name__ == '__main__':
    cal2 = json.load(open('round2_calibration.json'))
    tags = [sys.argv[1]] if len(sys.argv) > 1 else ['mm2000', 'conf5000']
    for t in tags:
        try:
            analyze(t, cal2)
        except FileNotFoundError as e:
            print(f'{t}: missing data ({e})')
