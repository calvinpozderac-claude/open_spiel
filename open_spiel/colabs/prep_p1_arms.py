"""Fork three Phase-1 plateau-study arms from bench_HYBRID:40000.

Each arm gets its own checkpoint dir seeded with a rewound `latest.pt` at
ep=40000: weights from bench_40000.pt, Elo/hist truncated by
rewind_to_generation, optimizer state dropped (Adam moments restart at zero --
symmetric across all three arms, not a differential confound), and a
correctly-repositioned LR scheduler state so the cosine continues from ep40000
instead of restarting from ep0 (LambdaLR has no memory of `ep` across a
rewind -- left unpatched every arm would silently rewarm the LR, which is
exactly the confound this study must not introduce by accident).
"""
import os
import shutil

import numpy as np
import torch

import connect4_dirichlet_utils as c4

SRC = 'othello_benchmark/bench_HYBRID'
GEN = 40000
ARMS = ['bench_HYBRID_p1_control', 'bench_HYBRID_p1_sims', 'bench_HYBRID_p1_explore']

# Must match run_hybrid.py's make_cfg() exactly.
LR_WARMUP, LR_DECAY, LR_MINF = 100, 50000, 0.02


def _lr(ep):
    if ep < LR_WARMUP:
        return ep / max(LR_WARMUP, 1)
    f = min((ep - LR_WARMUP) / max(LR_DECAY - LR_WARMUP, 1), 1.0)
    return LR_MINF + (1 - LR_MINF) * 0.5 * (1 + np.cos(np.pi * f))


def sched_state_at(ep):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=2e-3)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    for _ in range(ep):
        sch.step()
    assert sch.last_epoch == ep
    return sch.state_dict(), opt.param_groups[0]['lr']


def main():
    for arm in ARMS:
        d = f'othello_benchmark/{arm}'
        os.makedirs(d, exist_ok=True)
        shutil.copy(f'{SRC}/bench_{GEN}.pt', f'{d}/bench_{GEN}.pt')
        blob = c4.rewind_to_generation(d, GEN, log=print)
        sched_sd, lr_now = sched_state_at(GEN)
        blob['sched'] = sched_sd
        torch.save(blob, f'{d}/latest.pt.tmp')
        os.replace(f'{d}/latest.pt.tmp', f'{d}/latest.pt')
        print(f'{arm}: ep={blob["ep"]} lr@resume={lr_now:.4e} '
              f'elo_entries={len(blob["order"])}')


if __name__ == '__main__':
    main()
