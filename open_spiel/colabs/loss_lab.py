"""STUDY 3 — the loss lab: train loss COMBINATIONS on identical frozen data.

Self-play is ~83% of wall time, so isolating the loss makes each arm ~30 min
instead of ~36 h: every arm trains a fresh net from scratch on the SAME
pickled replay buffer (gen_dataset.py) with the SAME batch sequence, and is
probed on the Edax-labelled positions as it trains.  This removes the data
confound entirely and adds what studies 1-2 could not see: gradient
interference between loss components under actual descent.

Components (weights per arm):
    klv/kla   Dirichlet KL to backed-up beliefs (OLD; concentration-coupled)
    dcev/dcea direction-only CE to backed-up MEANS (dense mean signal,
              no concentration coupling)
    evv/eva   evidential risk + annealed regularizer on grounded z
              (the concentration owner)
    cev/cea   plain CE of the mean against z
    cons      child-consistency KL (unchanged)

Arms:
    OLD        klv1 kla2 cev1 cea1 cons.5      -- the original loss
    EDL        evv1 eva1 cons.5                -- the claude/nll loss
    HYB        dcev1 dcea2 evv1 eva1 cons.5    -- the proposed synthesis
    HYB_noVd   dcea2 evv1 eva1 cons.5          -- state mean from evv alone
    DCE_only   dcev1 dcea2 cons.5              -- ablation: no grounded term,
                                                  conf head starves on purpose
    HYB_heavy  dcev2 dcea4 evv1 eva1 cons.5    -- weight sensitivity

    python loss_lab.py <dataset.pkl> [steps]   # run  (default 4000)
    python loss_lab.py report <out.json>
"""
import json
import os
import pickle
import sys
import time

import numpy as np

CACHE = 'edax_eval_set.json'
ARMS = {
    'OLD':       dict(klv=1.0, kla=2.0, cev=1.0, cea=1.0, cons=0.5),
    'EDL':       dict(evv=1.0, eva=1.0, cons=0.5),
    'HYB':       dict(dcev=1.0, dcea=2.0, evv=1.0, eva=1.0, cons=0.5),
    'HYB_noVd':  dict(dcea=2.0, evv=1.0, eva=1.0, cons=0.5),
    'DCE_only':  dict(dcev=1.0, dcea=2.0, cons=0.5),
    'HYB_heavy': dict(dcev=2.0, dcea=4.0, evv=1.0, eva=1.0, cons=0.5),
}
BATCH = 1024
LR_PEAK, LR_WARM, LR_MINF = 2e-3, 100, 0.1
LAM_ANNEAL_FRAC = 0.375          # lambda ramps over the first 3/8 of steps
EVAL_EVERY = 400
SEED = 5


def lab_loss(c4, torch, F, out, out2, meta, w, lam):
    """Generalized loss: any weighted combination of the components."""
    v_logits, v_conf_raw, a_logits, a_conf_raw = out
    act, mask = meta['pad_act'], meta['pad_mask']
    idx3 = act.unsqueeze(-1).expand(-1, -1, 3)
    q_logp = F.log_softmax(a_logits.gather(1, idx3), dim=-1)
    q_cf = c4._conf(a_conf_raw.gather(1, act))
    q_alpha = (q_cf.unsqueeze(-1) * q_logp.exp()).clamp_min(c4.ALPHA_FLOOR)
    q_alpha = torch.where(mask.unsqueeze(-1), q_alpha,
                          torch.full_like(q_alpha, c4.ALPHA_FLOOR))
    v_logp = F.log_softmax(v_logits, dim=-1)
    v_alpha = (c4._conf(v_conf_raw).unsqueeze(-1)
               * v_logp.exp()).clamp_min(c4.ALPHA_FLOOR)

    zw = meta['z_w']
    zwn = zw.sum().clamp_min(1.0)
    z1 = meta['z_idx'].unsqueeze(1)
    y = (torch.arange(3, device=v_alpha.device).unsqueeze(0) == z1).float()
    pw = meta['played_w'] * zw
    pwn = pw.sum().clamp_min(1.0)
    pl_idx = meta['played'].clamp_min(0)
    q_played = q_alpha.gather(
        1, pl_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    q_played_logp = q_logp.gather(
        1, pl_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    evm = meta['ev_mask'].reshape(-1).float()
    evn = evm.sum().clamp_min(1.0)

    total = v_logits.new_zeros(())
    parts = {}

    if w.get('klv'):
        L = c4._kl_terms(meta['v_obs'], v_alpha, False).mean()
        total = total + w['klv'] * L; parts['klv'] = L
    if w.get('kla'):
        kl = c4._kl_terms(meta['ev_alpha'].reshape(-1, 3),
                          q_alpha.reshape(-1, 3), False)
        L = (kl * evm).sum() / evn
        total = total + w['kla'] * L; parts['kla'] = L
    if w.get('dcev'):
        m = meta['v_obs'] / meta['v_obs'].sum(-1, keepdim=True).clamp_min(1e-9)
        L = -(m * v_logp).sum(-1).mean()
        total = total + w['dcev'] * L; parts['dcev'] = L
    if w.get('dcea'):
        ev = meta['ev_alpha'].reshape(-1, 3)
        m = ev / ev.sum(-1, keepdim=True).clamp_min(1e-9)
        ce = -(m * q_logp.reshape(-1, 3)).sum(-1)
        L = (ce * evm).sum() / evn
        total = total + w['dcea'] * L; parts['dcea'] = L
    if w.get('evv'):
        a0v = v_alpha.sum(-1)
        azv = v_alpha.gather(1, z1).squeeze(1)
        L = (zw * (c4._dg(a0v) - c4._dg(azv))).sum() / zwn
        tld = 1.0 + (1.0 - y) * F.relu(v_alpha - 1.0)
        R = (zw * c4._dir_kl_rows(tld, torch.ones_like(tld))).sum() / zwn
        total = total + w['evv'] * (L + lam * R)
        parts['evv'] = L; parts['regv'] = R
    if w.get('eva'):
        a0a = q_played.sum(-1)
        aza = q_played.gather(1, z1).squeeze(1)
        L = (pw * (c4._dg(a0a) - c4._dg(aza))).sum() / pwn
        tld = 1.0 + (1.0 - y) * F.relu(q_played - 1.0)
        R = (pw * c4._dir_kl_rows(tld, torch.ones_like(tld))).sum() / pwn
        total = total + w['eva'] * (L + lam * R)
        parts['eva'] = L; parts['rega'] = R
    if w.get('cev'):
        ce = -v_logp.gather(1, z1).squeeze(1)
        L = (zw * ce).sum() / zwn
        total = total + w['cev'] * L; parts['cev'] = L
    if w.get('cea'):
        ce = -q_played_logp.gather(1, z1).squeeze(1)
        L = (pw * ce).sum() / pwn
        total = total + w['cea'] * L; parts['cea'] = L
    if w.get('cons'):
        cw = meta['cons_w']
        tgt = meta['cons_term']
        if out2 is not None:
            v2_logits, v2_conf_raw, _a, _b = out2
            v2 = (c4._conf(v2_conf_raw).unsqueeze(-1)
                  * F.softmax(v2_logits, dim=-1)).clamp_min(c4.ALPHA_FLOOR)
            v2f = v2.detach()[:, [c4._LOSS, c4._DRAW, c4._WIN]]
            rows = v2f.index_select(0, meta['cons_row'].clamp_min(0))
            tgt = torch.where(meta['cons_isnn'].unsqueeze(-1), rows, tgt)
        kl = c4._kl_terms(tgt.detach(), q_played, False)
        L = (cw * kl).sum() / cw.sum().clamp_min(1.0)
        total = total + w['cons'] * L; parts['cons'] = L
    return total, parts


def probe_eval(c4, np_, net, device, states, truth, emp, recs):
    # NOT nn_eval_states: that runs under torch.inference_mode(), which
    # errors on DirectML once the optimizer has touched the params
    # ("Cannot set version_counter for inference tensor") — same quirk the
    # self-play server hit; use no_grad like it does.
    import torch
    import torch.nn.functional as F

    def _eval(chunk):
        obs16 = np_.asarray([c4.make_obs(s) for s in chunk],
                            dtype=np_.float16)
        x = c4.batch_to_tensor(obs16, device)
        with torch.no_grad():
            v_logits, v_conf_raw, a_logits, a_conf_raw = net(x)
            v3 = F.softmax(v_logits, dim=-1)
            vc = c4._conf(v_conf_raw)
            p3 = F.softmax(a_logits, dim=-1)
        return v3.cpu().numpy(), vc.cpu().numpy(), p3.cpu().numpy()

    pred, a0s, top1 = [], [], []
    for i0 in range(0, len(states), 256):
        chunk = states[i0:i0 + 256]
        v3, vc, p3 = _eval(chunk)
        for j, st in enumerate(chunk):
            pred.append(float(v3[j][c4._WIN] - v3[j][c4._LOSS]))
            a0s.append(float(vc[j]))
            legal = [int(x) for x in st.legal_actions()]
            q = p3[j][legal][:, c4._WIN] - p3[j][legal][:, c4._LOSS]
            top1.append(legal[int(np_.argmax(q))] == recs[i0 + j]['best'])
    pred, a0s = np_.array(pred), np_.array(a0s)
    err = (pred - truth) ** 2
    dec = truth != 0
    from scipy.stats import spearmanr
    rho = float(spearmanr(a0s, -err).statistic)

    def buck(lo, hi):
        m = (emp >= lo) & (emp < hi)
        d = m & dec
        acc = float(((pred[d] > 0) == (truth[d] > 0)).mean()) if d.any() else 0
        return float(a0s[m].mean()), acc

    oa, oacc = buck(45, 61)
    fa, facc = buck(0, 8)
    return dict(mse=float(err.mean()), top1=float(np_.mean(top1)),
                sign=float(((pred[dec] > 0) == (truth[dec] > 0)).mean()),
                oa=oa, oacc=oacc, fa=fa, facc=facc,
                ratio=fa / max(oa, 1e-9), rho=rho)


def run(data_path, steps):
    import torch
    import torch.nn.functional as F
    import connect4_dirichlet_utils as c4
    game = c4.load_game('othello')
    c4.set_game(game)
    device, backend = c4.pick_device('auto')
    c4.set_backend(backend, device)

    with open(data_path, 'rb') as fh:
        buf = pickle.load(fh)
    print(f'[lab] {len(buf)} samples from {data_path}', flush=True)
    with open(CACHE) as fh:
        recs = json.load(fh)['positions']
    states, truth, emp = [], [], []
    for r in recs:
        st = game.new_initial_state()
        for a in r['actions']:
            st.apply_action(int(a))
        states.append(st)
        truth.append(r['value'])
        emp.append(r['empties'])
    truth, emp = np.array(truth, float), np.array(emp)

    out_path = os.path.splitext(data_path)[0] + '_lab.json'
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            results = json.load(fh)

    cfg = c4.Config(game_name='othello', channels=32, num_blocks=10, head_ch=8)
    for arm, w in ARMS.items():
        if arm in results:
            print(f'[lab] {arm} already done, skipping', flush=True)
            continue
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        rng = np.random.default_rng(SEED)
        srng = np.random.default_rng(SEED + 1)
        net, _sig = c4.build_network(cfg, device)
        opt = c4.LerpFreeAdamW(net.parameters(), lr=LR_PEAK,
                               weight_decay=1e-4)

        def lr_at(s):
            if s < LR_WARM:
                return s / LR_WARM
            f = min((s - LR_WARM) / max(steps - LR_WARM, 1), 1.0)
            return LR_MINF + (1 - LR_MINF) * 0.5 * (1 + np.cos(np.pi * f))

        curve = []
        t0 = time.time()
        skipped = 0
        for s in range(1, steps + 1):
            for g in opt.param_groups:
                g['lr'] = LR_PEAK * lr_at(s)
            lam = min(1.0, s / max(int(steps * LAM_ANNEAL_FRAC), 1))
            batch = [buf[i] for i in rng.integers(0, len(buf), BATCH)]
            meta, obs2 = c4.build_batch_meta(batch, device, 1.0, srng)
            obs = [smp['obs'] for smp in batch]
            B = len(obs)
            x = c4.batch_to_tensor(obs + obs2, device)
            net.train()
            v_logits, v_conf, a_logits, a_conf = net(x)
            o1 = (v_logits[:B], v_conf[:B], a_logits[:B], a_conf[:B])
            o2 = ((v_logits[B:], v_conf[B:], a_logits[B:], a_conf[B:])
                  if obs2 else None)
            opt.zero_grad()
            loss, _parts = lab_loss(c4, torch, F, o1, o2, meta, w, lam)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if bool(torch.isfinite(gnorm)) and np.isfinite(float(loss)):
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                skipped += 1
            if s % EVAL_EVERY == 0 or s == steps:
                net.eval()
                m = probe_eval(c4, np, net, device, states, truth, emp, recs)
                m['step'] = s
                m['loss'] = float(loss)
                curve.append(m)
                print(f'[lab] {arm} s{s}: mse {m["mse"]:.3f} '
                      f'top1 {m["top1"]:.1%} sign {m["sign"]:.1%} '
                      f'a0 {m["oa"]:.2f}/{m["fa"]:.2f} r {m["ratio"]:.2f} '
                      f'rho {m["rho"]:.2f} ({time.time()-t0:.0f}s)', flush=True)
        results[arm] = dict(weights=w, curve=curve, skipped=skipped)
        with open(out_path, 'w') as fh:
            json.dump(results, fh)
        print(f'[lab] {arm} DONE in {time.time()-t0:.0f}s', flush=True)
    print('[lab] LOSS LAB DONE', flush=True)


def report(out_path):
    with open(out_path) as fh:
        results = json.load(fh)
    print(f'{"arm":<11}{"step":>6}{"MSE":>7}{"top1":>7}{"sign":>7}'
          f'{"a0 open":>9}{"a0 fin":>8}{"ratio":>7}{"rho":>7}')
    for arm, r in results.items():
        best = min(r['curve'], key=lambda m: m['mse'])
        last = r['curve'][-1]
        for tag, m in (('best', best), ('last', last)):
            print(f'{arm:<11}{m["step"]:>6}{m["mse"]:>7.3f}{m["top1"]:>7.1%}'
                  f'{m["sign"]:>7.1%}{m["oa"]:>9.2f}{m["fa"]:>8.2f}'
                  f'{m["ratio"]:>7.2f}{m["rho"]:>7.2f}  ({tag})')
        print()


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report(sys.argv[2])
    else:
        run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
