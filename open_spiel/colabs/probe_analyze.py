"""Analysis half of the sims-needed probe: is "how much search does this
position need" predictable beyond the 3 scalars the deployed rule uses?

Three labels are scored, because they disagree and the choice matters:

  patience    what oracle_sims_needed produces today: N consecutive quiet
              blocks of the root's VALUE-belief KL, judged online.  This is the
              target the deployed predictor coefficients were fit against.
  action_eps  hindsight, decision-relevant: the earliest checkpoint after which
              the per-action values root_pick reads never again move more than
              eps from where they finish.
  decision    hindsight, strictest reading of "the search was done": the
              earliest checkpoint after which the MOVE that would be played
              never changes again.

And four models, on each label. The decomposition is the point -- each answers
a different question about whether a learned head is worth building:

  linear(3 scalars)  reproduces the deployed 'predictor' -- the baseline
  MLP(3 scalars)     same information, nonlinear -- is the LINEAR FORM the
                     binding constraint?
  conv(board)        the board itself -- is there information the 3 scalars
                     never carried?  This is the head analogue: a real head
                     hangs off a trunk that has already seen the board.
  conv + scalars     the union

Everything is scored on games HELD OUT from training. Positions come from every
4th ply of the same self-play games, so a position-level split would put
near-duplicate boards on both sides and inflate every held-out number.

None of it means anything without probe_replicate.py's ceiling: MCTS is
stochastic, so these labels are noisy MEASUREMENTS of a position, not
properties of it, and the noise share is unfittable by construction.

    python probe_analyze.py [tag]
"""
import sys

import numpy as np
import torch
import torch.nn as nn

EPS_KL = 0.005     # Dirichlet-KL threshold, as oracle_sims_needed uses
EPS_ACT = 0.01     # action-value threshold for the hindsight label
PATIENCE = 3
SEED = 0


# ── labels ────────────────────────────────────────────────────────────────────

def patience_label(consec_kl, n_valid, solved, cap, check_every):
    """oracle_sims_needed's rule, replayed off the stored trace."""
    n_ckpt = consec_kl.shape[1]
    grid = (np.arange(n_ckpt) + 1) * check_every
    out = np.full(len(consec_kl), float(cap))
    for i in range(len(consec_kl)):
        quiet, needed = 0, float(cap)
        nv = int(n_valid[i])
        for j in range(nv):
            quiet = quiet + 1 if consec_kl[i, j] < EPS_KL else 0
            if quiet >= PATIENCE:
                needed = float(grid[j])
                break
        else:
            # Trace ended early because the root SOLVED: the belief is exact,
            # so the position needed exactly the search it got.
            if solved[i] and nv > 0:
                needed = float(grid[nv - 1])
        out[i] = needed
    return out


def action_eps_label(action_vals, touched, n_valid, cap, check_every,
                     eps=EPS_ACT):
    """Earliest checkpoint after which the per-action values never again move
    more than `eps` from their final values, over edges search actually
    touched.  These are the numbers root_pick reads, so this is "the search
    output stopped changing" rather than "some internal belief stopped
    changing"."""
    n_ckpt = action_vals.shape[1]
    grid = (np.arange(n_ckpt) + 1) * check_every
    out = np.full(len(action_vals), float(cap))
    for i in range(len(action_vals)):
        nv = int(n_valid[i])
        if nv == 0:
            continue
        m = touched[i, nv - 1]
        if not m.any():
            out[i] = float(grid[0])
            continue
        fin = action_vals[i, nv - 1][m]
        dev = np.abs(action_vals[i, :nv][:, m] - fin).max(1)
        far = np.where(dev >= eps)[0]
        idx = (far[-1] + 1) if len(far) else 0
        out[i] = float(grid[min(idx, nv - 1)])
    return out


def decision_label(best_idx, n_valid, cap, check_every):
    """Earliest checkpoint after which the played MOVE never changes again."""
    n_ckpt = best_idx.shape[1]
    grid = (np.arange(n_ckpt) + 1) * check_every
    out = np.full(len(best_idx), float(cap))
    for i in range(len(best_idx)):
        nv = int(n_valid[i])
        if nv == 0:
            continue
        seq = best_idx[i, :nv]
        far = np.where(seq != seq[-1])[0]
        idx = (far[-1] + 1) if len(far) else 0
        out[i] = float(grid[min(idx, nv - 1)])
    return out


def all_labels(d, cap, ce):
    return {
        'patience': patience_label(d['consec_kl'], d['n_valid'], d['solved'],
                                   cap, ce),
        'action_eps': action_eps_label(d['action_vals'], d['touched'],
                                       d['n_valid'], cap, ce),
        'decision': decision_label(d['best_idx'], d['n_valid'], cap, ce),
    }


# ── models ────────────────────────────────────────────────────────────────────

def r2(pred, true):
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def fit_linear(Xtr, ytr, Xte, yte):
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return r2(Xtr @ coef, ytr), r2(Xte @ coef, yte), coef


def _train_torch(model, Xtr, ytr, Xva, yva, Xte, yte, epochs=300, lr=1e-3,
                 wd=1e-4, bs=256):
    """Adam + early stopping on a validation split. The val split is carved out
    of the held-out-by-game pool but never scored, so the reported test r2 is
    not the number the stopping rule optimised."""
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    ttr = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    best_va, best_state, bad = -1e9, None, 0
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(Xtr), bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[idx]).squeeze(-1), ttr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).squeeze(-1).numpy() * sd + mu
        v = r2(pv, yva)
        if v > best_va:
            best_va = v
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 40:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ptr = model(Xtr).squeeze(-1).numpy() * sd + mu
        pte = model(Xte).squeeze(-1).numpy() * sd + mu
    return r2(ptr, ytr), r2(pte, yte), best_va


def fit_mlp(F, y, tr, va, te, hidden=64):
    """Same 3 scalars the linear model gets, nonlinear."""
    Z = F[:, 1:].copy()
    Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-9)
    X = torch.tensor(Z, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
    return _train_torch(model, X[tr], y[tr], X[va], y[va], X[te], y[te])


class ConvNet(nn.Module):
    """Deliberately small -- this stands in for a HEAD on an existing trunk,
    not for a bigger network. If something this size already closes the gap, a
    real head (which gets the trunk's features for free) will too."""

    def __init__(self, ch=32, extra=0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(ch * 64 + extra, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 1))
        self.extra = extra

    def forward(self, x):
        h = self.body(x[:, :192].view(-1, 3, 8, 8)).flatten(1)
        if self.extra:
            h = torch.cat([h, x[:, 192:]], 1)
        return self.head(h)


def fit_conv(obs, F, y, tr, va, te, with_feats=False):
    parts, extra = [obs], 0
    if with_feats:
        Z = F[:, 1:].copy()
        Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-9)
        parts.append(Z)
        extra = 3
    X = torch.tensor(np.concatenate(parts, 1), dtype=torch.float32)
    return _train_torch(ConvNet(extra=extra), X[tr], y[tr], X[va], y[va],
                        X[te], y[te])


# ── ceiling ───────────────────────────────────────────────────────────────────

def ceiling(tag, labels, cap, ce):
    """Upper bound on any model's r2, from two independent labels of the same
    positions.  Reliability corr(y1, y2) = Var(signal)/Var(total) IS that
    bound; the variance-of-difference route is an independent estimate of the
    same quantity and should agree."""
    try:
        d = np.load(f'probe_replicate_{tag}.npz')
    except FileNotFoundError:
        print('  (no replicate data -- run probe_replicate.py for the ceiling)')
        return {}
    idx = d['main_index']
    rep = all_labels(d, cap, ce)
    print(f'\n--- label noise ceiling ({len(idx)} positions re-searched with '
          f'different search RNG) ---')
    out = {}
    for nm in labels:
        a = np.log(np.maximum(labels[nm][idx], 1.0))
        b = np.log(np.maximum(rep[nm], 1.0))
        if a.std() < 1e-9 or b.std() < 1e-9:
            print(f'  {nm:>11}: label is constant -- nothing to predict')
            out[nm] = 0.0
            continue
        rel = float(np.corrcoef(a, b)[0, 1])
        v_noise = float(np.var(a - b) / 2.0)
        v_tot = float((np.var(a) + np.var(b)) / 2.0)
        out[nm] = rel
        print(f'  {nm:>11}: reliability corr={rel:+.3f}  '
              f'1-Var(noise)/Var(y)={1 - v_noise/max(v_tot,1e-12):+.3f}  '
              f'=> max attainable r2 ~ {max(rel, 0.0):.2f}   '
              f'(exact agreement {np.mean(labels[nm][idx] == rep[nm]):.0%})')
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def analyze(tag):
    d = np.load(f'probe_data_{tag}.npz')
    F, obs = d['feats'], d['obs']
    gid, solved = d['game_id'], d['solved']
    cap, ce = int(d['cap']), int(d['check_every'])
    n = len(F)
    print(f'\n{"="*78}\n{tag}: {n} positions from {gid.max()+1} games\n{"="*78}')
    print(f'solved-early positions: {solved.mean():.1%}')

    labels = all_labels(d, cap, ce)

    print('\n--- label distributions ---')
    for nm, y in labels.items():
        print(f'  {nm:>11}: mean={y.mean():6.0f} median={np.median(y):5.0f} '
              f'p10={np.percentile(y,10):5.0f} p90={np.percentile(y,90):5.0f} '
              f'at cap {(y>=cap).mean():5.1%}  sd(log)={np.log(y).std():.3f}')
    print('  eps sweep for action_eps:')
    for e in (0.002, 0.005, 0.01, 0.02, 0.05):
        ys = action_eps_label(d['action_vals'], d['touched'], d['n_valid'],
                              cap, ce, eps=e)
        print(f'    eps={e:<6} mean={ys.mean():6.0f} '
              f'median={np.median(ys):5.0f} at cap {(ys>=cap).mean():5.1%}')
    yp, ya = labels['patience'], labels['action_eps']
    print(f'  patience vs action_eps: under {100*(yp<ya).mean():.0f}% / '
          f'over {100*(yp>ya).mean():.0f}% / equal {100*(yp==ya).mean():.0f}%'
          f'   mean gap {np.mean(ya-yp):+.0f} sims')

    ceil = ceiling(tag, labels, cap, ce)

    rng = np.random.default_rng(SEED)
    games = np.unique(gid)
    rng.shuffle(games)
    n_tr, n_va = int(0.7 * len(games)), int(0.15 * len(games))
    g_tr, g_va = set(games[:n_tr].tolist()), set(games[n_tr:n_tr+n_va].tolist())
    tr = np.array([g in g_tr for g in gid])
    va = np.array([g in g_va for g in gid])
    te = ~(tr | va)
    print(f'\nsplit by game: train {tr.sum()} / val {va.sum()} / test {te.sum()}')

    stones = obs.reshape(n, 3, 64)[:, :2].sum((1, 2))

    for nm, y_raw in labels.items():
        y = np.log(np.maximum(y_raw, 1.0))
        if y.std() < 1e-9:
            print(f'\n--- target: {nm} -- constant, skipping ---')
            continue
        cap_txt = (f'   [ceiling r2 ~ {ceil[nm]:.2f}]' if nm in ceil else '')
        print(f'\n--- target: {nm}, log sims{cap_txt} ---')
        print(f'{"model":<24}{"train r2":>10}{"TEST r2":>10}')
        rtr, rte, coef = fit_linear(F[tr], y[tr], F[te], y[te])
        print(f'{"linear(3 scalars)":<24}{rtr:>10.3f}{rte:>10.3f}'
              f'   coef={np.round(coef,3).tolist()}')
        for lbl, fn in (('MLP(3 scalars)', lambda: fit_mlp(F, y, tr, va, te)),
                        ('conv(board)',
                         lambda: fit_conv(obs, F, y, tr, va, te, False)),
                        ('conv(board)+scalars',
                         lambda: fit_conv(obs, F, y, tr, va, te, True))):
            a, b, _ = fn()
            print(f'{lbl:<24}{a:>10.3f}{b:>10.3f}')

        print('  by game phase (stones on board):')
        for a, b in zip([0, 20, 30, 40, 50], [20, 30, 40, 50, 65]):
            m = te & (stones >= a) & (stones < b)
            if m.sum() < 30:
                continue
            print(f'    {a:>2}-{b:<2} n={m.sum():<5} mean={y_raw[m].mean():6.0f}'
                  f'  sd(log)={y[m].std():.3f}'
                  f'  linear r2={r2(F[m] @ coef, y[m]):+.3f}')


if __name__ == '__main__':
    tags = [sys.argv[1]] if len(sys.argv) > 1 else ['mm2000', 'conf5000']
    for t in tags:
        try:
            analyze(t)
        except FileNotFoundError as e:
            print(f'{t}: missing data ({e})')
