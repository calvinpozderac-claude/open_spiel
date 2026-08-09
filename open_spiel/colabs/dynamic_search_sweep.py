"""What the gates see at the moment a search ends, and what threshold puts the
average position at ~500 simulations."""
import dataclasses
import time

import numpy as np
import torch

import connect4_dirichlet_utils as c4
import connect4_benchmark as B
import dynamic_search_utils as ds

g = c4.load_game('connect_four')
c4.set_game(g)
c4.set_backend('cpu', device='cpu')
SH = B.default_shared(channels=32, num_blocks=3, head_ch=8)
BASE = B.thompson_config('MM', SH)
torch.manual_seed(0)
_ref = c4.C4DirichletNet(32, 3, 8)
SD0 = {k: v.clone() for k, v in _ref.state_dict().items()}

SIMS = 500          # the budget the experiment will run at


def run(over, samples=90, sims=SIMS):
    cfg = dataclasses.replace(
        BASE, fast_sims=sims, full_sims=sims, fast_prob=1.0,
        n_parallel_games=8, worker_wave=8, **over)
    c4.set_search(cfg.search_agg, cfg.target_agg, cfg.selection,
                  cfg.virtual_loss, cfg.backup)
    net = c4.C4DirichletNet(32, 3, 8)
    net.load_state_dict(SD0)
    net.eval()
    sp = c4.ParallelSelfPlay(g, net, 'cpu', c4._worker_cfg(cfg, (32, 3, 8)),
                             seed=1)
    it = sp.episodes()
    next(it)
    t0 = time.perf_counter()
    n = 0
    while n < samples:
        n += len(next(it))
    dt = time.perf_counter() - t0
    d = sp.ds.summary()
    return dt, d


# ── 1. what do the gates actually see when a search ends? ────────────────────
print(f'DIAGNOSTIC at {SIMS} simulations: the statistic at the end of a search')
cfg = dataclasses.replace(BASE, fast_sims=SIMS, full_sims=SIMS, fast_prob=1.0,
                          n_parallel_games=4, worker_wave=8,
                          dynamic_search=False)
c4.set_search(cfg.search_agg, cfg.target_agg, cfg.selection, cfg.virtual_loss,
              cfg.backup)
net = c4.C4DirichletNet(32, 3, 8)
net.load_state_dict(SD0)
net.eval()
bot = c4.C4MCTSBot(g, net, 'cpu', SIMS, batch_size=16,
                   random_state=np.random.default_rng(0))
rows = []
st = g.new_initial_state()
rng = np.random.default_rng(2)
for _ in range(24):
    if st.is_terminal():
        st = g.new_initial_state()
    root = bot.mcts_search(st)
    p = root.probe(SIMS)
    rows.append((ds.p_best(p.q, p.sd), ds.top_gap(p.q), float(np.mean(p.sd)),
                 ds.lucb_separated(p.q, p.sd)))
    la = st.legal_actions()
    st.apply_action(int(la[rng.integers(len(la))]))
a = np.array([r[:3] for r in rows])
sep = np.array([r[3] for r in rows])
print(f'  p_best      median {np.median(a[:, 0]):.3f}   '
      f'>=0.95 on {(a[:, 0] >= 0.95).mean():.0%} of positions')
print(f'  top gap     median {np.median(a[:, 1]):.3f}')
print(f'  mean spread median {np.median(a[:, 2]):.3f}')
print(f'  LUCB separated on {sep.mean():.0%} of positions (c=2)')
for c in (0.5, 1.0, 1.5, 2.0, 3.0):
    s = np.mean([ds.lucb_separated(r[0], r[1], c)
                 for r in [(np.array([0.0]), np.array([1.0]))]]) if False else None
print()

# ── 2. threshold sweep: what lands the average position near 500? ────────────
print(f'SWEEP: mean simulations per position, nominal {SIMS}')
print(f'{"rule":<28}{"sims/mv":>9}{"of nominal":>12}{"wall s":>9}'
      f'{"ceiling":>9}{"converged":>11}')
grid = [('fixed', dict(dynamic_search=False))]
for c in (1.0, 1.5, 2.0, 3.0):
    grid.append((f'lucb c={c}', dict(dynamic_search=True, ds_rule='lucb',
                                     ds_lucb_c=c)))
for d_ in (0.02, 0.05, 0.12):
    grid.append((f'block d={d_}', dict(dynamic_search=True, ds_rule='block',
                                       ds_delta=d_, ds_block_sims=100)))
for p_ in (0.95, 0.99):
    grid.append((f'full p={p_}', dict(dynamic_search=True, ds_rule='full',
                                      ds_p_stop=p_)))
grid.append(('block_lucb c=2 d=0.05',
             dict(dynamic_search=True, ds_rule='block_lucb')))
for name, over in grid:
    dt, d = run(over)
    sm = d.get('sims_mean', float(SIMS))
    nom = d.get('sims_base', float(SIMS))
    print(f'{name:<28}{sm:>9.1f}{sm / nom:>11.2f}x{dt:>9.1f}'
          f'{d.get("stop_ceiling", 0.0):>9.2f}{d.get("stop_converged", 0.0):>11.2f}')
