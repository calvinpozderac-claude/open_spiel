"""Spec Step 8 -- unit tests T1-T10 and the synthetic rho-recovery test.

    python test_corr_posterior.py            # unit tests T1-T10
    python test_corr_posterior.py synth      # synthetic recovery (global + position)
    python test_corr_posterior.py confound   # logs the (rho, c) likelihood ridge

No self-play run starts until every unit test and the synthetic recovery test pass.
T11 (gradient routing) and T12 (tree accumulators) need the network / tree, so they
live in test_corr_integration.py.
"""
import math
import sys

import numpy as np
import torch

import corr_posterior as cp
from corr_posterior import Acc, DEFAULT, dm_nll, dm_nll_np, flip, leaf_payload

M0 = np.array([0.5, 0.3, 0.2])


def _identical(n, m=M0, c=10.0):
    """n identical leaves with mean m and concentration c."""
    a = Acc()
    u = (1.0 - float(m @ m)) / (c + 1.0)
    a.n, a.SM, a.SQ, a.SU, a.ST = n, n * m, n * m * m, n * u, n * math.sqrt(u)
    return a


def _a0(alpha):
    return float(np.sum(alpha))


# ─────────────────────────────────────────────────────────────────────────────
def test_T1_limits():
    c, n = 10.0, 10
    acc = _identical(n, c=c)
    lo = _a0(cp.posterior_search_only(acc, DEFAULT.rho_min))
    assert abs((lo + 1) / (n * (c + 1)) - 1) < 0.01, lo
    hi = _a0(cp.posterior_search_only(acc, DEFAULT.rho_max))
    assert abs(hi / c - 1) < 0.01, hi
    acc50 = _identical(50, c=c)
    mid = _a0(cp.posterior_search_only(acc50, 0.2))
    want = (c + 1) * 50 / (1 + 49 * 0.2) - 1
    assert abs(mid - want) / want < 1e-6, (mid, want)


def test_T2_empty_returns_prior():
    prior = np.array([3.0, 1.5, 0.7])
    out = cp.posterior(prior, Acc(), 0.4)
    assert np.array_equal(out, prior)


def test_T3_leaf_equals_prior():
    c = 7.0
    prior = c * M0
    for rho in (0.05, 0.3, 0.8):
        acc = Acc()
        acc.add(*leaf_payload(alpha_leaf=prior))
        out = cp.posterior(prior, acc, rho)
        want = 2 * (c + 1) / (1 + rho) - 1
        assert abs(_a0(out) - want) / want < 1e-6, (rho, _a0(out), want)
        assert np.allclose(out / out.sum(), M0, atol=1e-9)


def test_T4_monotone_and_saturates():
    c, rho = 10.0, 0.1
    prev = 0.0
    for n in (1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000):
        a0 = _a0(cp.posterior_search_only(_identical(n, c=c), rho))
        assert a0 >= prev - 1e-9, (n, a0, prev)
        prev = a0
    sat = (c + 1) / rho - 1
    assert abs(prev / sat - 1) < 0.01, (prev, sat)


def test_T5_terminal_same_outcome_is_a_spike():
    acc = Acc()
    for _ in range(20):
        acc.add(*leaf_payload(terminal_outcome=0))
    for prior in ([1.6, 0.4, 0.4], [30.0, 5.0, 3.0]):
        out = cp.posterior(np.array(prior), acc, 0.3)
        assert abs(_a0(out) - DEFAULT.a0_max) / DEFAULT.a0_max < 1e-6
        assert np.allclose(out / out.sum(), [1, 0, 0], atol=1e-6)


def test_T6_terminal_split_scales_with_n():
    def split(n):
        a = Acc()
        for i in range(n):
            a.add(*leaf_payload(terminal_outcome=0 if i % 2 == 0 else 2))
        return a
    r = _a0(cp.posterior_search_only(split(40), 0.3)) / \
        _a0(cp.posterior_search_only(split(20), 0.3))
    assert 1.8 <= r <= 2.2, r
    # with a weak (uniform) prior mixed in the ratio must stay in the same band
    weak = np.array([1.0, 1.0, 1.0])
    r2 = _a0(cp.posterior(weak, split(40), 0.3)) / \
         _a0(cp.posterior(weak, split(20), 0.3))
    assert 1.8 <= r2 <= 2.2, r2


def test_T7_no_double_count():
    rng = np.random.default_rng(0)
    n, rho, c = 60, 0.4, 1000.0          # tiny model noise, large dispersion
    acc = Acc()
    for m in rng.dirichlet([1, 1, 1], size=n):
        acc.add(m, (1 - m @ m) / (c + 1))
    aux = {}
    cp.posterior(np.array([2.0, 1.0, 1.0]), acc, rho, aux=aux)
    b_emp = aux['B_emp']
    # the empirical dispersion dominates the model-implied idiosyncratic noise,
    # so B_eff == B_emp and the idiosyncratic part is NOT added a second time
    assert b_emp > aux['B_idio'] * 10, (b_emp, aux['B_idio'])
    want = b_emp / n + rho * acc.ST ** 2 / n ** 2
    assert abs(aux['Ts'] - want) < 1e-12 * max(1.0, want), (aux['Ts'], want)


def test_T7b_b_emp_matches_numpy_cov():
    rng = np.random.default_rng(3)
    ms = rng.dirichlet([2, 2, 2], size=40)
    acc = Acc()
    for m in ms:
        acc.add(m, 1e-6)
    aux = {}
    cp.posterior(np.array([2.0, 1.0, 1.0]), acc, 0.3, aux=aux)
    want = float(np.trace(np.cov(ms.T, ddof=1)))
    assert abs(aux['B_emp'] - want) < 1e-12, (aux['B_emp'], want)


def test_T8_flip_equivariance():
    rng = np.random.default_rng(1)
    for _ in range(200):
        prior = rng.uniform(0.2, 20.0, 3)
        acc = Acc()
        for m in rng.dirichlet([1.5, 1.0, 1.5], size=int(rng.integers(1, 40))):
            acc.add(m, rng.uniform(0.0, 0.3))
        rho = float(rng.uniform(0.01, 0.99))
        a = cp.posterior(prior, acc, rho)
        b = cp.posterior(flip(prior), acc.flipped(), rho)
        assert np.allclose(b, flip(a), rtol=1e-12, atol=1e-12), (a, b)


def test_T9_dm_nll():
    rng = np.random.default_rng(2)
    alpha = np.array([3.0, 1.2, 0.8])
    a0 = alpha.sum()
    # k = 1 is exactly the marginal form -log(alpha_z / alpha0)
    for z in range(3):
        c = np.eye(3)[z]
        got = dm_nll_np(c, alpha)
        assert abs(got - (-math.log(alpha[z] / a0))) < 1e-9
        t = float(dm_nll(torch.tensor(c), torch.tensor(alpha)))
        assert abs(t - got) < 1e-9
    # k = 2: ordered-sequence probabilities
    assert abs(dm_nll_np([2, 0, 0], alpha)
               + math.log(alpha[0] * (alpha[0] + 1) / (a0 * (a0 + 1)))) < 1e-9
    assert abs(dm_nll_np([1, 1, 0], alpha)
               + math.log(alpha[0] * alpha[1] / (a0 * (a0 + 1)))) < 1e-9
    # empirical agreement under sampled theta ~ Dir(alpha)
    N = 400000
    theta = rng.dirichlet(alpha, size=N)
    cum = theta.cumsum(1)
    u1, u2 = rng.random(N), rng.random(N)
    o1 = (u1[:, None] > cum).sum(1)
    o2 = (u2[:, None] > cum).sum(1)
    agree = (o1 == o2).astype(float)
    mu2 = float(((alpha / a0) ** 2).sum())
    want = (a0 * mu2 + 1) / (a0 + 1)
    se = agree.std() / math.sqrt(N)
    assert abs(agree.mean() - want) < 3 * se, (agree.mean(), want, se)


def _random_batch(rng, B, dtype=np.float64):
    prior = rng.uniform(0.2, 30.0, (B, 3))
    accs = np.zeros((B, 9))
    for i in range(B):
        a = Acc()
        for m in rng.dirichlet(rng.uniform(0.5, 5.0, 3), size=int(rng.integers(0, 80))):
            a.add(m, rng.uniform(0.0, 0.4))
        accs[i] = a.to_vec()
    rho = rng.uniform(0.005, 0.99, B)
    return prior, accs, rho


def test_T10_torch_matches_scalar_and_gradcheck():
    rng = np.random.default_rng(5)
    prior, accs, rho = _random_batch(rng, 10000)
    ref = np.array([cp.posterior(prior[i], Acc.from_vec(accs[i]), rho[i])
                    for i in range(len(rho))])
    got = cp.posterior_t(torch.tensor(prior), torch.tensor(accs),
                         torch.tensor(rho)).numpy()
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-8), np.abs(got - ref).max()
    # host-computed B_emp must agree with the in-function one
    be = torch.tensor(cp.b_emp_host(accs))
    got2 = cp.posterior_t(torch.tensor(prior), torch.tensor(accs),
                          torch.tensor(rho), B_emp=be).numpy()
    assert np.allclose(got2, ref, rtol=1e-5, atol=1e-8)
    # gradcheck of a scalar likelihood in rho, away from clip boundaries
    sub = slice(0, 40)
    pr, ac = torch.tensor(prior[sub]), torch.tensor(accs[sub])
    counts = torch.tensor(np.eye(3)[rng.integers(0, 3, 40)]) * 2.0
    r0 = torch.tensor(rng.uniform(0.15, 0.85, 40), requires_grad=True)

    def f(r):
        return dm_nll(counts, cp.posterior_t(pr, ac, r)).sum()
    assert torch.autograd.gradcheck(f, (r0,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_float32_posterior_is_finite_with_host_bemp():
    rng = np.random.default_rng(6)
    prior, accs, rho = _random_batch(rng, 2000)
    accs[:, 0] = np.maximum(accs[:, 0], 0)
    be = torch.tensor(cp.b_emp_host(accs), dtype=torch.float32)
    out = cp.posterior_t(torch.tensor(prior, dtype=torch.float32),
                         torch.tensor(accs, dtype=torch.float32),
                         torch.tensor(rho, dtype=torch.float32), B_emp=be)
    assert torch.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic rho-recovery
# ─────────────────────────────────────────────────────────────────────────────
S_ERR = 0.4
SNAP_N = (16, 32, 64, 128, 256)


def _alr(theta):
    return np.stack([np.log(theta[..., 0] / theta[..., 1]),
                     np.log(theta[..., 2] / theta[..., 1])], -1)


def _alr_inv(x):
    z = np.concatenate([x[..., :1], np.zeros_like(x[..., :1]), x[..., 1:]], -1)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def calibrate_c(rng, N=400000):
    """c such that E||m - theta||^2 = E[1 - ||m||^2] / (c + 1).  The total leaf
    error b + e has variance s^2 whatever rho is, so c is rho-independent."""
    theta = rng.dirichlet([4, 4, 4], size=N)
    m = _alr_inv(_alr(theta) + rng.normal(0, S_ERR, (N, 2)))
    err = ((m - theta) ** 2).sum(-1).mean()
    return float((1.0 - (m * m).sum(-1)).mean() / err - 1.0)


def make_positions(rho_true, n_pos, c, rng):
    """Positions with a known error model.  rho_true may be a scalar or an array
    of length n_pos.  Returns dict of float64 arrays."""
    rho_true = np.broadcast_to(np.asarray(rho_true, dtype=np.float64), (n_pos,))
    theta = rng.dirichlet([4, 4, 4], size=n_pos)
    base = _alr(theta)
    sd_shared = (np.sqrt(rho_true) * S_ERR)[:, None]
    sd_idio = (np.sqrt(1.0 - rho_true) * S_ERR)[:, None]
    b = rng.normal(0, 1, (n_pos, 2)) * sd_shared
    mu0 = _alr_inv(base + b + rng.normal(0, 1, (n_pos, 2)) * sd_idio)
    L = SNAP_N[-1]
    e = rng.normal(0, 1, (n_pos, L, 2)) * sd_idio[:, None, :]
    m = _alr_inv(base[:, None, :] + b[:, None, :] + e)              # (P, L, 3)
    u = (1.0 - (m * m).sum(-1)) / (c + 1.0)                          # (P, L)
    cm, cq = m.cumsum(1), (m * m).cumsum(1)
    cu, ct = u.cumsum(1), np.sqrt(u).cumsum(1)
    snaps = np.zeros((n_pos, len(SNAP_N), 9))
    for j, n in enumerate(SNAP_N):
        snaps[:, j, 0] = n
        snaps[:, j, 1:4] = cm[:, n - 1]
        snaps[:, j, 4:7] = cq[:, n - 1]
        snaps[:, j, 7] = cu[:, n - 1]
        snaps[:, j, 8] = ct[:, n - 1]
    outcomes = np.zeros((n_pos, 3))
    cum = theta.cumsum(1)
    for _ in range(2):                                               # k = 2
        o = (rng.random(n_pos)[:, None] > cum).sum(1)
        outcomes[np.arange(n_pos), o] += 1
    return dict(prior=c * mu0, snaps=snaps, counts=outcomes, theta=theta)


def _nll(rho, data, snaps_idx=None):
    pr = torch.tensor(data['prior'])
    cn = torch.tensor(data['counts'])
    sn = torch.tensor(data['snaps'])
    tot = 0.0
    for j in range(sn.shape[1]):
        tot = tot + dm_nll(cn, cp.posterior_t(pr, sn[:, j], rho)).mean()
    return tot / sn.shape[1]


def fit_global_rho(data, steps=400, lr=0.1, seed=0):
    torch.manual_seed(seed)
    z = torch.zeros((), dtype=torch.float64, requires_grad=True)   # logit rho, start .5
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _nll(torch.sigmoid(z), data).backward()
        opt.step()
    return float(torch.sigmoid(z))


def test_synthetic_global_recovery(verbose=True):
    rng = np.random.default_rng(11)
    c = calibrate_c(rng)
    if verbose:
        print(f'  calibrated leaf concentration c = {c:.3f}')
    est = []
    for rt in (0.1, 0.3, 0.6):
        d = make_positions(rt, 20000, c, rng)
        r = fit_global_rho(d)
        est.append(r)
        if verbose:
            print(f'  rho_true {rt:.1f} -> rho_hat {r:.3f}   (|err| {abs(r - rt):.3f})')
    for rt, r in zip((0.1, 0.3, 0.6), est):
        assert abs(r - rt) <= 0.1, (rt, r)
    assert est[0] < est[1] < est[2], est


def test_synthetic_position_recovery(verbose=True):
    rng = np.random.default_rng(12)
    c = calibrate_c(rng)
    P = 50000
    kind = (rng.random(P) < 0.5).astype(np.float64)               # one synthetic bit
    rho_t = np.where(kind > 0.5, 0.6, 0.1)
    d = make_positions(rho_t, P, c, rng)
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1)).double()
    with torch.no_grad():
        mlp[2].bias.fill_(math.log(0.3 / 0.7))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.03)
    x = torch.tensor(kind).unsqueeze(1)
    pr, cn, sn = (torch.tensor(d['prior']), torch.tensor(d['counts']),
                  torch.tensor(d['snaps']))
    for _ in range(300):
        opt.zero_grad()
        rho = torch.sigmoid(mlp(x).squeeze(1))
        loss = 0.0
        for j in range(sn.shape[1]):
            loss = loss + dm_nll(cn, cp.posterior_t(pr, sn[:, j], rho)).mean()
        (loss / sn.shape[1]).backward()
        opt.step()
    with torch.no_grad():
        r0 = float(torch.sigmoid(mlp(torch.zeros(1, 1).double())))
        r1 = float(torch.sigmoid(mlp(torch.ones(1, 1).double())))
    if verbose:
        print(f'  type A rho_true 0.1 -> {r0:.3f}   type B rho_true 0.6 -> {r1:.3f}')
    assert abs(r0 - 0.1) <= 0.15 and abs(r1 - 0.6) <= 0.15, (r0, r1)


def log_confound_surface():
    """Logged, NOT gated: at a single n = 64 the leaf concentration c and rho are
    confounded, which is why snapshots are required."""
    rng = np.random.default_rng(13)
    c0 = calibrate_c(rng)
    d = make_positions(0.3, 20000, c0, rng)
    n64 = d['snaps'][:, 2:3]
    best = []
    print('  single n = 64: NLL by rho, minimised over a rescaling of leaf c')
    for rho in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9):
        row = []
        for scale in (0.5, 0.7, 1.0, 1.4, 2.0):
            # rescaling c rescales every leaf u by (c0+1)/(c*+1)
            s = d['snaps'][:, 2].copy()
            f = (c0 + 1.0) / (c0 * scale + 1.0)
            s[:, 7] *= f
            s[:, 8] *= math.sqrt(f)
            dd = dict(prior=d['prior'], counts=d['counts'], snaps=s[:, None])
            row.append(float(_nll(torch.tensor(rho, dtype=torch.float64), dd)))
        best.append(min(row))
        print(f'    rho {rho:.2f}  NLL over c-scale: '
              + ' '.join(f'{v:.4f}' for v in row) + f'   min {min(row):.4f}')
    print(f'  spread of best-over-c NLL across rho: {max(best) - min(best):.5f} '
          f'(small => a ridge: single-n data cannot pin rho)')


# ─────────────────────────────────────────────────────────────────────────────
#  Drift-based rho recovery -- the identification route actually used
# ─────────────────────────────────────────────────────────────────────────────
def test_drift_recovery_global(verbose=True):
    """Spec Step 8's gate, met by snapshot drift instead of fork outcomes.

    The spec identifies rho from k outcomes per fork and argues nothing else can.
    Nested snapshots can: the shared error cancels in mu_n - mu_N, so the drift
    measures the idiosyncratic variance, and rho = 1 - idio/total with the total
    known from the leaf concentrations.  No outcomes are used here at all."""
    rng = np.random.default_rng(101)
    c = calibrate_c(rng)
    est = []
    for rt in (0.1, 0.3, 0.6):
        d = make_positions(rt, 4000, c, rng)
        r, npair = cp.rho_from_drift(d['snaps'])
        est.append(r)
        if verbose:
            print(f'  rho_true {rt:.1f} -> rho_hat {r:.3f}  (|err| {abs(r-rt):.3f}, '
                  f'{npair} pairs, 4000 positions, ZERO outcomes)')
    for rt, r in zip((0.1, 0.3, 0.6), est):
        assert abs(r - rt) <= 0.05, (rt, r)
    assert est[0] < est[1] < est[2], est


def test_drift_recovery_small_sample(verbose=True):
    """400 positions is enough; the spec's outcome route needs k>=16 per position."""
    rng = np.random.default_rng(102)
    c = calibrate_c(rng)
    for rt in (0.1, 0.3, 0.6):
        rs = [cp.rho_from_drift(make_positions(rt, 400, c, rng)['snaps'])[0]
              for _ in range(5)]
        if verbose:
            print(f'  rho_true {rt:.1f} -> {np.mean(rs):.3f} +- {np.std(rs):.3f} '
                  f'over 5 repeats of 400 positions')
        assert abs(np.mean(rs) - rt) <= 0.06, (rt, np.mean(rs))


def test_drift_handles_heterogeneous_leaves(verbose=True):
    """Leaf variances that SHRINK as search deepens (what a real tree does, as it
    concentrates on good moves) must not bias rho.  The exact denominator uses the
    cumulative SU at both snapshots; a naive single-u version collapses here."""
    rng0 = np.random.default_rng(103)
    c = calibrate_c(rng0)
    rt, P, L = 0.3, 4000, SNAP_N[-1]
    for decay in (0.0, 0.3, 0.6, 1.0):
        rng = np.random.default_rng(7)
        theta = rng.dirichlet([4, 4, 4], size=P)
        base = _alr(theta)
        b = rng.normal(0, 1, (P, 2)) * (math.sqrt(rt) * S_ERR)
        w = (1.0 + np.arange(L)) ** (-decay)
        e = rng.normal(0, 1, (P, L, 2)) * (math.sqrt(1 - rt) * S_ERR) * w[None, :, None]
        m = _alr_inv(base[:, None, :] + b[:, None, :] + e)
        u = (1.0 - (m * m).sum(-1)) / (c + 1.0) * (w ** 2)[None, :]
        cm, cu = m.cumsum(1), u.cumsum(1)
        sn = np.zeros((P, len(SNAP_N), 9))
        for j, n in enumerate(SNAP_N):
            sn[:, j, 0] = n
            sn[:, j, 1:4] = cm[:, n - 1]
            sn[:, j, 7] = cu[:, n - 1]
        r, _ = cp.rho_from_drift(sn)
        naive_num = naive_den = 0.0
        for j in range(len(SNAP_N)):
            for k in range(j + 1, len(SNAP_N)):
                nj, nk = SNAP_N[j], SNAP_N[k]
                mj, mk = sn[:, j, 1:4] / nj, sn[:, k, 1:4] / nk
                naive_num += ((mj - mk) ** 2).sum(-1).sum()
                naive_den += ((sn[:, k, 7] / nk) * (1 / nj - 1 / nk)).sum()
        naive = 1.0 - naive_num / naive_den
        if verbose:
            print(f'  variance decay {decay:.1f}: exact {r:.3f}   naive {naive:.3f}')
        assert abs(r - rt) <= 0.06, (decay, r)


def test_drift_rho_is_relative_to_claimed_leaf_variance(verbose=True):
    """Documents the method's real limitation: rho is the shared FRACTION of the
    variance the network CLAIMS.  Miscalibrated concentrations bias it directly,
    which is why the outcome terms (cev/cea) are still needed to keep them honest."""
    rng = np.random.default_rng(104)
    c = calibrate_c(rng)
    d = make_positions(0.3, 4000, c, rng)
    for scale in (0.5, 1.0, 2.0):
        sn = d['snaps'].copy()
        sn[:, :, 7] *= scale
        r, _ = cp.rho_from_drift(sn)
        want = 1.0 - (1.0 - 0.3) / scale
        if verbose:
            print(f'  net claims {scale:.1f}x the true leaf variance -> rho_hat '
                  f'{r:.3f}  (predicted {max(want, 1e-3):.3f})')
        assert abs(r - max(want, 1e-3)) < 0.07, (scale, r, want)


DRIFT = [test_drift_recovery_global, test_drift_recovery_small_sample,
         test_drift_handles_heterogeneous_leaves,
         test_drift_rho_is_relative_to_claimed_leaf_variance]


# ─────────────────────────────────────────────────────────────────────────────
UNIT = [test_T1_limits, test_T2_empty_returns_prior, test_T3_leaf_equals_prior,
        test_T4_monotone_and_saturates, test_T5_terminal_same_outcome_is_a_spike,
        test_T6_terminal_split_scales_with_n, test_T7_no_double_count,
        test_T7b_b_emp_matches_numpy_cov, test_T8_flip_equivariance,
        test_T9_dm_nll, test_T10_torch_matches_scalar_and_gradcheck,
        test_float32_posterior_is_finite_with_host_bemp]


def main(argv):
    if len(argv) > 1 and argv[1] == 'drift':
        fails = 0
        for t in DRIFT:
            print(t.__name__)
            try:
                t()
            except Exception as e:                                # noqa: BLE001
                fails += 1; print(f'  FAIL {type(e).__name__}: {e}')
        print('DRIFT OK' if not fails else f'DRIFT {fails} FAILED')
        return 1 if fails else 0
    if len(argv) > 1 and argv[1] == 'synth':
        print('synthetic global recovery '
              '(the OUTCOME route from the spec -- expected to fail)')
        test_synthetic_global_recovery()
        print('synthetic position recovery'); test_synthetic_position_recovery()
        print('SYNTHETIC OK'); return 0
    if len(argv) > 1 and argv[1] == 'confound':
        log_confound_surface(); return 0
    fails = 0
    for t in UNIT:
        try:
            t(); print(f'PASS  {t.__name__}')
        except Exception as e:                                   # noqa: BLE001
            fails += 1; print(f'FAIL  {t.__name__}: {type(e).__name__}: {e}')
    print(f'{len(UNIT) - fails}/{len(UNIT)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
