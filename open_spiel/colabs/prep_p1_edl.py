"""Fork the Phase-1 EDL arm from bench_HYBRID:40000, same base as the other
four Phase-1 arms (control/sims/explore/cev), so `compare_runs.py` lines them
all up at matched episodes.

Does NOT touch the existing arm directories -- prep_p1_arms.py's main() would
re-copy bench_40000.pt over their already-progressed latest.pt (destroying
finished training), so this is a standalone copy of just its per-arm logic,
scoped to the one new directory.
"""
import os
import shutil

import numpy as np
import torch

import connect4_dirichlet_utils as c4

SRC = 'othello_benchmark/bench_HYBRID'
GEN = 40000
ARM = 'bench_HYBRID_p1_edl'

# Must match run_hybrid.py's make_cfg() exactly (see prep_p1_arms.py).
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
    d = f'othello_benchmark/{ARM}'
    os.makedirs(d, exist_ok=True)
    shutil.copy(f'{SRC}/bench_{GEN}.pt', f'{d}/bench_{GEN}.pt')
    blob = c4.rewind_to_generation(d, GEN, log=print)
    sched_sd, lr_now = sched_state_at(GEN)
    blob['sched'] = sched_sd
    torch.save(blob, f'{d}/latest.pt.tmp')
    os.replace(f'{d}/latest.pt.tmp', f'{d}/latest.pt')
    print(f'{ARM}: ep={blob["ep"]} lr@resume={lr_now:.4e} '
          f'elo_entries={len(blob["order"])}')


if __name__ == '__main__':
    main()
