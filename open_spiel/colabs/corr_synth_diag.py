"""Why does the synthetic rho-recovery test fail?  Diagnostics, not a test.

Separates three hypotheses:
  A. a bug in the posterior   -> posterior variance is MISCALIBRATED at the true rho
  B. too little statistical power at the spec's sizes (20k positions, k = 2)
                              -> the NLL curve is flat/noisy, min moves with the seed
  C. model misspecification   -> the NLL minimum sits away from rho_true even with
                              abundant data (large N, large k)
"""
import sys

import numpy as np
import torch

import corr_posterior as cp
from corr_posterior import dm_nll
import test_corr_posterior as T

GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95)


def posterior_np(d, j, rho):
    pr, sn = torch.tensor(d['prior']), torch.tensor(d['snaps'][:, j])
    return cp.posterior_t(pr, sn, torch.tensor(float(rho), dtype=torch.float64),
                          aux=True)


def calibration(d, rho):
    """E||mu - theta||^2 versus E[T] at each snapshot.  1.0 = calibrated."""
    out = []
    for j, n in enumerate(T.SNAP_N):
        alpha, aux = posterior_np(d, j, rho)
        mu = (alpha / alpha.sum(-1, keepdim=True)).numpy()
        err = ((mu - d['theta']) ** 2).sum(-1).mean()
        claimed = float(aux['T'].mean())
        out.append((n, err, claimed, err / claimed, float(aux['w'].mean())))
    return out


def nll_curve(d, k_scale=1):
    pr, cn = torch.tensor(d['prior']), torch.tensor(d['counts'])
    sn = torch.tensor(d['snaps'])
    res = []
    for rho in GRID:
        tot = 0.0
        for j in range(sn.shape[1]):
            a = cp.posterior_t(pr, sn[:, j], torch.tensor(rho, dtype=torch.float64))
            tot += float(dm_nll(cn, a).mean())
        res.append(tot / sn.shape[1])
    return np.array(res)


def draw_counts(theta, k, rng):
    cum = theta.cumsum(1)
    cnt = np.zeros_like(theta)
    for _ in range(k):
        o = (rng.random(len(theta))[:, None] > cum).sum(1)
        cnt[np.arange(len(theta)), o] += 1
    return cnt


def main(seed=21):
    rng = np.random.default_rng(seed)
    c = T.calibrate_c(rng)
    print(f'calibrated c = {c:.3f}\n')
    for rt in (0.1, 0.3, 0.6):
        d = T.make_positions(rt, 20000, c, rng)
        print(f'=== rho_true {rt} ===')
        print('  calibration of the posterior AT rho_true (err / claimed T; 1.0 = ok):')
        for n, err, claimed, ratio, w in calibration(d, rt):
            print(f'    n={n:3d}  err {err:.5f}  claimed T {claimed:.5f}  '
                  f'ratio {ratio:.3f}  mean prior weight {w:.3f}')
        print('  NLL by rho  ' + '  '.join(f'{g:.2f}' for g in GRID))
        for label, N, k in (('spec  N=20k  k=2 ', 20000, 2),
                            ('N=200k k=2         ', 200000, 2),
                            ('N=20k  k=40        ', 20000, 40)):
            dd = d if N == 20000 else T.make_positions(rt, N, c, rng)
            dd = dict(dd)
            if k != 2:
                dd['counts'] = draw_counts(dd['theta'], k, rng)
            nll = nll_curve(dd)
            # per-outcome NLL; shift so the min is 0 to expose the curve shape
            rel = (nll - nll.min()) / k
            print(f'  {label} argmin rho = {GRID[int(nll.argmin())]:.2f}   '
                  + ' '.join(f'{v:.5f}' for v in rel))
        print()


if __name__ == '__main__':
    if not (len(sys.argv) > 1 and sys.argv[1] == "scan"):
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 21)


FINE = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def fit_fine(d, counts):
    pr, cn, sn = torch.tensor(d['prior']), torch.tensor(counts), torch.tensor(d['snaps'])
    best, bv = None, 1e18
    for rho in FINE:
        tot = 0.0
        for j in range(sn.shape[1]):
            a = cp.posterior_t(pr, sn[:, j], torch.tensor(float(rho), dtype=torch.float64))
            tot += float(dm_nll(cn, a).mean())
        if tot < bv:
            best, bv = float(rho), tot
    return best


def scan_k(seed=31):
    """How many outcomes per position (k) does it take to recover rho?"""
    rng = np.random.default_rng(seed)
    c = T.calibrate_c(rng)
    print('argmin rho on a 0.05 grid; N = positions, k = outcomes per position')
    print('  rho_true |' + ''.join(f'  k={k:<3d}' for k in (2, 4, 8, 16, 40, 100)))
    for N in (20000, 100000):
        print(f'  N = {N}')
        for rt in (0.1, 0.3, 0.6):
            d = T.make_positions(rt, N, c, rng)
            row = []
            for k in (2, 4, 8, 16, 40, 100):
                row.append(fit_fine(d, draw_counts(d['theta'], k, rng)))
            print(f'   {rt:.1f}     |' + ''.join(f'  {v:5.2f}' for v in row))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'scan':
    scan_k()
