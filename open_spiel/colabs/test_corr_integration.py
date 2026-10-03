"""Integration tests for the 'corr' aggregation rule inside the real tree code.

    python test_corr_integration.py          # fast tests (T12, regressions, guards)
    python test_corr_integration.py smoke    # tiny end-to-end self-play + training

T12 is the spec's "hand-built 2-ply tree, 4 simulations".  The rest guard the
thing most likely to go wrong when a new rule is bolted on: silently changing what
every EXISTING rule does.  test_legacy_unchanged runs the old rules through the new
code and through the previous commit's code and demands identical beliefs.
"""
import importlib.util
import math
import subprocess
import sys
import types

import numpy as np

import connect4_dirichlet_utils as c4
import corr_posterior as cp
from corr_posterior import Acc

LEGACY_REV = 'ff53a670'        # last commit before the 'corr' rule touched this file


def _legacy_module():
    try:
        src = subprocess.run(
            ['git', 'show', f'{LEGACY_REV}:open_spiel/colabs/connect4_dirichlet_utils.py'],
            capture_output=True, check=True, cwd='.').stdout.decode('utf-8')
    except Exception:
        return None
    mod = types.ModuleType('c4_legacy')
    mod.__file__ = 'c4_legacy.py'
    sys.modules['c4_legacy'] = mod
    exec(compile(src, 'c4_legacy.py', 'exec'), mod.__dict__)
    return mod


def _tree(mod):
    """The fixed two-ply toy tree.  Returns (R, C)."""
    R = mod._CNode(0, [0, 1], [.5, .3, .2], 4.0,
                   [[.5, .3, .2], [.2, .3, .5]], [3.0, 5.0])
    C = mod._CNode(1, [0, 1], [.2, .3, .5], 3.0,
                   [[.3, .3, .4], [.6, .2, .2]], [2.0, 6.0])
    R.children[0] = C
    mod._seed_leaf(R)
    mod._seed_leaf(C)
    return R, C


def _four_sims(mod, R, C):
    """Three evaluated leaves under C, then a proven WIN on R's second edge."""
    for path, leaf in (([(R, 0), (C, 1)], (4.0, 2.0, 2.0)),
                       ([(R, 0), (C, 0)], (1.0, 1.0, 6.0)),
                       ([(R, 0), (C, 1)], (3.0, 3.0, 2.0))):
        for node, idx in path:
            node.vloss[idx] += 1                       # as _select_leaf would
        mod._backup(path, mod._payload(np.array(leaf)))
    mod._set_term(R, 1, mod._WIN)
    R.vloss[1] += 1
    mod._backup([(R, 1)], mod._payload(mod.flip_alpha(mod._SPIKE[mod._WIN]),
                                       **({'exact': True} if hasattr(mod, 'AGG_CORR')
                                          and mod._CORR_ACTIVE else {})))


def _ref(*items):
    """Independent reference accumulator.  items: (m, u, flip?)"""
    a = Acc()
    for m, u, fl in items:
        a.add(cp.flip(m) if fl else m, u)
    return a


def _leaf(alpha):
    return cp.leaf_payload(alpha_leaf=np.array(alpha))


def _same(list_acc, ref, tol=1e-12):
    assert abs(list_acc[0] - ref.n) < tol, (list_acc[0], ref.n)
    assert np.allclose(list_acc[1], ref.SM, atol=tol), (list_acc[1], ref.SM)
    assert np.allclose(list_acc[2], ref.SQ, atol=tol)
    assert abs(sum(list_acc[3]) - ref.SU) < tol, (sum(list_acc[3]), ref.SU)
    assert abs(list_acc[9] - ref.ST) < tol, (list_acc[9], ref.ST)


def test_T12_hand_built_two_ply_tree():
    c4.set_search(search_agg='corr', target_agg='corr', rho_search=0.2,
                  rho_target=0.7, corr_target_cap=100.0, selection='dirichlet')
    R, C = _tree(c4)
    _four_sims(c4, R, C)

    m_self_R, u_self_R = cp.leaf_payload(alpha_leaf=R.v_alpha)
    m_self_C, u_self_C = cp.leaf_payload(alpha_leaf=C.v_alpha)
    L = [_leaf((4, 2, 2)), _leaf((1, 1, 6)), _leaf((3, 3, 2))]
    win = cp.leaf_payload(terminal_outcome=0)

    # C's edges/node are in C's mover's perspective: the leaf mover is the OTHER
    # player, so the leaf mean is FLIPPED on the way in.  R's are as-is.
    refs = {
        'C.eacc[1]': (C.eacc[1], _ref((*L[0], True), (*L[2], True))),
        'C.eacc[0]': (C.eacc[0], _ref((*L[1], True))),
        'C.nacc':    (C.nacc, _ref((m_self_C, u_self_C, False),
                                   (*L[0], True), (*L[1], True), (*L[2], True))),
        'R.eacc[0]': (R.eacc[0], _ref((*L[0], False), (*L[1], False),
                                      (*L[2], False))),
        'R.eacc[1]': (R.eacc[1], _ref((*win, False))),
        'R.nacc':    (R.nacc, _ref((m_self_R, u_self_R, False), (*L[0], False),
                                   (*L[1], False), (*L[2], False),
                                   (*win, False))),
    }
    for name, (got, ref) in refs.items():
        try:
            _same(got, ref)
        except AssertionError as e:
            raise AssertionError(f'{name}: {e}')

    # beliefs: selection uses rho_search, targets use rho_target
    want = cp.posterior(R.alpha_p[0], refs['R.eacc[0]'][1], 0.2)
    assert np.allclose(R.alpha_sel[0], want, rtol=1e-12), (R.alpha_sel[0], want)
    assert np.allclose(R.alpha_sel[1], c4._SPIKE[c4._WIN]), R.alpha_sel[1]   # proven
    tgt = R.edge_target(0)
    want_t = c4._corr_target(cp.posterior(R.alpha_p[0], refs['R.eacc[0]'][1], 0.7))
    assert np.allclose(tgt, want_t, rtol=1e-12), (tgt, want_t)
    assert np.allclose(R.edge_target(1), c4._SPIKE[c4._WIN])
    assert np.allclose(R.state_target(), c4._SPIKE[c4._WIN])      # R is solved
    # C is unsolved: its state target is the corr posterior of (prior, nacc)
    want_c = c4._corr_target(cp.posterior(C.v_alpha, refs['C.nacc'][1], 0.7))
    assert np.allclose(C.state_target(), want_c, rtol=1e-12)
    # the legacy rules must not have been consulted: no MLE bookkeeping
    assert R.nacc[5] == [0.0, 0.0, 0.0]


def test_exact_terminal_payload():
    c4.set_search(search_agg='corr', target_agg='corr')
    for out in (c4._WIN, c4._DRAW, c4._LOSS):
        (m, q, v, a), (fm, fq, fv, fa) = c4._payload(
            c4.flip_alpha(c4._SPIKE[out]), exact=True)
        assert sum(m) == 1.0 and max(m) == 1.0 and v == (0.0, 0.0, 0.0)
        assert tuple(fm) == tuple(cp.flip(np.array(m))) and fv == (0.0, 0.0, 0.0)
    # default path is untouched
    (m, q, v, a), _ = c4._payload(np.array([4.0, 2.0, 2.0]))
    assert abs(sum(m) - 1.0) < 1e-12 and sum(v) > 0


def test_legacy_unchanged():
    old = _legacy_module()
    if old is None:
        print('  (git unavailable; skipped)')
        return
    for agg in ('additive', 'additive_mle', 'avgconc', 'mixture', 'pathsum'):
        c4.set_search(search_agg=agg, target_agg=agg, selection='dirichlet')
        old.set_search(search_agg=agg, target_agg=agg, selection='dirichlet')
        Rn, Cn = _tree(c4)
        Ro, Co = _tree(old)
        _four_sims(c4, Rn, Cn)
        _four_sims(old, Ro, Co)
        for a, b in ((Rn, Ro), (Cn, Co)):
            assert np.allclose(a.alpha_sel, b.alpha_sel, rtol=1e-12, atol=0), agg
            for i in range(2):
                ta, tb = a.edge_target(i), b.edge_target(i)
                assert (ta is None and tb is None) or np.allclose(ta, tb, rtol=1e-12), agg
            assert np.allclose(a.state_target(), b.state_target(), rtol=1e-12), agg
        assert Rn.nacc[9] == 0.0 and Cn.eacc[1][9] == 0.0, 'slot 9 touched in legacy'
    c4.set_search(search_agg='additive_mle', target_agg='avgconc')


def test_target_guards():
    c4.set_search(search_agg='corr', target_agg='corr', corr_target_cap=100.0)
    spike = np.array([1e6, 1e-4, 1e-4])
    t = c4._corr_target(spike)
    # capped at 100, floor adds at most 3 * TARGET_EPS on top
    assert t.sum() <= 100.0 + 3 * c4.TARGET_EPS + 1e-9, t
    assert t.min() >= c4.TARGET_EPS - 1e-12, t
    assert t.argmax() == 0
    soft = np.array([3.0, 2.0, 1.0])                 # untouched when already safe
    assert np.allclose(c4._corr_target(soft), soft)


def test_observed_alpha_corr_fallback():
    """Peripheral callers with no prior get the search estimate alone."""
    c4.set_search(search_agg='corr', target_agg='corr', rho_search=0.3)
    R, C = _tree(c4)
    _four_sims(c4, R, C)
    got = c4.observed_alpha(C.nacc, 'corr')
    sv = C.nacc[3]
    want = np.array(cp.posterior_scalar((1, 1, 1), C.nacc[0], C.nacc[1], C.nacc[2],
                                        sum(sv), C.nacc[9], 0.3, search_only=True))
    assert np.allclose(got, want, rtol=1e-12)


FAST = [test_T12_hand_built_two_ply_tree, test_exact_terminal_payload,
        test_legacy_unchanged, test_target_guards, test_observed_alpha_corr_fallback]


def main(argv):
    if len(argv) > 1 and argv[1] == 'smoke':
        import smoke_corr
        return smoke_corr.main()
    fails = 0
    for t in FAST:
        try:
            t(); print(f'PASS  {t.__name__}')
        except Exception as e:                                    # noqa: BLE001
            fails += 1
            import traceback; traceback.print_exc()
            print(f'FAIL  {t.__name__}: {type(e).__name__}: {e}')
    print(f'{len(FAST) - fails}/{len(FAST)} passed')
    return 1 if fails else 0




# ─────────────────────────────────────────────────────────────────────────────
#  rho head + drift loss
# ─────────────────────────────────────────────────────────────────────────────
def _batch_with_snaps(n=48, sims=64, seed=0, rho_init=0.5):
    import torch
    g = c4.load_game('othello'); c4.set_game(g)
    FR = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    cfg = c4._worker_cfg(c4.Config(
        game_name='othello', channels=16, num_blocks=2, head_ch=4,
        search_agg='corr', target_agg='corr', snapshot_fracs=FR,
        fast_sims=sims, full_sims=sims, fast_prob=1.0, worker_wave=4,
        use_workers=False, n_parallel_games=4), (16, 2, 4))
    c4.set_search(search_agg='corr', target_agg='corr', selection='dirichlet',
                  snapshot_fracs=FR)
    net = c4.C4DirichletNet(16, 2, 4, rho_head=True, rho_init=rho_init)
    sp = c4.ParallelSelfPlay(g, net, 'cpu', cfg, checkpoint_dir=None, seed=seed)
    it = sp.episodes()
    batch = []
    while len(batch) < n:
        batch += list(next(it))
    return net, batch[:n]


def test_rho_head_optional_and_backcompat():
    import torch
    g = c4.load_game('othello'); c4.set_game(g)
    plain = c4.C4DirichletNet(8, 1, 2)
    assert len(plain(torch.zeros(1, *c4._OBS_SHAPE))) == 4
    assert plain(torch.zeros(1, *c4._OBS_SHAPE), return_rho=True)[4] is None
    assert not any('rho' in k for k in plain.state_dict())
    withrho = c4.C4DirichletNet(8, 1, 2, rho_head=True)
    assert any('rho' in k for k in withrho.state_dict())
    # a checkpoint saved without the head still loads into the default net
    import io
    buf = io.BytesIO(); torch.save(plain.state_dict(), buf); buf.seek(0)
    c4.C4DirichletNet(8, 1, 2).load_state_dict(torch.load(buf, weights_only=True))


def test_T11_gradient_routing():
    """Spec Step 7 routing table: the drift term must reach rho and the trunk and
    NOT the alpha heads; the distillation/outcome terms must not reach rho."""
    import torch
    net, batch = _batch_with_snaps()
    # cev/cea are weighted by z_w, which only finish_episode sets -- without this
    # those two terms have zero gradient for a trivial reason and the routing
    # assertions below would pass vacuously.
    c4.finish_episode(batch, [1.0, -1.0])
    meta, obs2 = c4.build_batch_meta(batch, 'cpu')
    assert bool((meta['snap_dl'] > 0).any()), 'no drift data in the batch'
    assert float(meta['z_w'].sum()) > 0, 'z_w not stamped'
    assert float(meta['played_w'].sum()) > 0, 'played_w not set'
    obs = [b['obs'] for b in batch]
    x = c4.batch_to_tensor(obs, 'cpu')

    groups = {
        'alpha_heads': [p for n, p in net.named_parameters()
                        if n.startswith(('v_out', 'a_out'))],
        'rho_head':    [p for n, p in net.named_parameters()
                        if n.startswith('rho_mlp')],
        'rho_bar':     [net.rho_bar_logit],
        'trunk':       [p for n, p in net.named_parameters()
                        if n.startswith(('stem', 'body', 'head'))],
    }

    def grads(weights, rho_mode):
        net.zero_grad(set_to_none=True)
        v, vc, a, ac, rr = net(x, return_rho=True)
        loss, _ = c4.full_loss((v, vc, a, ac), None, meta, weights,
                               rho_raw=rr, rho_bar_logit=net.rho_bar_logit,
                               rho_mode=rho_mode)
        loss.backward()
        return {k: any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in ps) for k, ps in groups.items()}

    g = grads({'drift': 1.0}, 'position')
    assert g == {'alpha_heads': False, 'rho_head': True, 'rho_bar': False,
                 'trunk': True}, g
    g = grads({'drift': 1.0, 'rhoreg': 1.0}, 'position')
    assert g['rho_bar'] and not g['alpha_heads'], g
    g = grads({'drift': 1.0}, 'global')
    assert g == {'alpha_heads': False, 'rho_head': False, 'rho_bar': True,
                 'trunk': False}, g           # global mode: no trunk gradient
    for term in ('klv', 'kla', 'cev', 'cea'):
        g = grads({term: 1.0}, 'position')
        assert g['alpha_heads'] and not g['rho_head'] and not g['rho_bar'], (term, g)


def test_drift_loss_learns_rho():
    """The drift loss must move rho from its init to the value the data implies.

    It is compared against the closed-form moment estimator with a LOOSE tolerance
    on purpose: the two are different estimators of the same quantity.  The moment
    version pools a ratio of sums over all pairs; drift_nll averages a per-pair
    exponential NLL, which weights positions differently.  Measured difference on
    real self-play batches is ~0.01-0.03, and float32 packing of the drift terms
    contributes nothing (verified exactly 0.0000).  What is asserted is therefore
    direction and rough agreement, not identity."""
    import torch
    net, batch = _batch_with_snaps(n=160, sims=128, rho_init=0.5)
    sn = np.array([np.array(b['snaps']) for b in batch
                   if len(b.get('snaps', ())) == 5])
    want, _ = cp.rho_from_drift(sn)
    assert want > 0.6, f'test batch has unexpectedly low rho ({want})'
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(300):
        c4.train_step(net, opt, batch, 'cpu', {'drift': 1.0}, rho_mode='global')
    got = float(torch.sigmoid(net.rho_bar_logit))
    assert got > 0.5 + 0.1, f'rho did not move up from its 0.5 init: {got}'
    assert abs(got - want) < 0.06, (got, want)


def test_snapshots_off_by_default():
    c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                  snapshot_fracs=())
    assert c4._SNAP_FRACS == ()
    t = c4.make_target(_tree(c4)[0])
    assert t['snaps'] == ()
    net, batch = _batch_with_snaps(n=8, sims=32)       # restores corr + fracs
    c4.set_search(search_agg='additive_mle', target_agg='avgconc',
                  snapshot_fracs=())


FAST += [test_rho_head_optional_and_backcompat, test_T11_gradient_routing,
         test_drift_loss_learns_rho, test_snapshots_off_by_default]


if __name__ == '__main__':
    sys.exit(main(sys.argv))
