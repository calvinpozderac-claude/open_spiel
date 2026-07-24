"""ThompsonZero-FULL — shared helpers for chess_thompson_full_training.ipynb.

This module is the entire implementation of the "full" ThompsonZero
methodology; the notebook holds only hyperparameters + the training driver +
plots + arena.  See the notebook's first markdown cell (or the PR description)
for the full method write-up.  In one paragraph:

The network has a common ResNet trunk and two learned heads —
  (1) a POLICY-PRIOR head: per-action logits + a scalar concentration beta,
      giving a Dirichlet-categorical prior pi_a = softmax(logits) over "which
      move is optimal", with prior strength beta;
  (2) a per-action VALUE head: for every action a, a 3-outcome mean
      (p_win, p_draw, p_loss) and a concentration c_a, i.e. a Dirichlet belief
      pv_a = Dir(c_a * (p_win, p_draw, p_loss)_a) over that action's outcome.
Two DERIVED (non-parametric) quantities fall out of those heads:
  (3) scalar value  v  = sum_a pi_a * (p_win_a - p_loss_a);
  (4) position value-distribution  qv = moment-match( sum_a pi_a * pv_a )  — a
      pi-weighted mixture of the per-action Dirichlets, collapsed back to a
      single Dirichlet by matching its (exact, closed-form) mean+variance.

Search is Thompson sampling over the pv_a (draw one outcome per edge, play the
argmax of v = win-loss); pi_a is NOT used to steer selection.  Backup, bottom
up along the path: (a) credit the selected edge with one policy-prior
observation (pcount += 1) so pi_a converges to the visit distribution; (b)
recompute the node's qv as the pi-weighted mixture; (c) replace the parent's
pv_a with the child's qv, flipped one ply (win<->loss).  MCTS-Solver overlays
exact proofs.  Four training losses are summed: policy KL, per-action value KL,
position value-dist KL (all closed-form Dirichlet-Dirichlet KLs), and value MSE
to the game outcome z.

Only the torch-dependent pieces (network, losses, optimizer, checkpointing) need
torch; the tree/search/target/self-play/eval logic is pure numpy, so it imports
and unit-tests without torch or a GPU.
"""

import copy
import math
import os
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                   # numpy-only (tree unit tests)
    torch = None
    _HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  Outcome convention + Dirichlet primitives (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════
# A per-action / per-node belief is a Dirichlet over the 3-way outcome
# (win, draw, loss) FROM THE MOVER'S PERSPECTIVE.  One ply up the tree the mover
# changes, so the belief FLIPS (win<->loss; draw unchanged).  The scalar value
# used for Thompson selection is v = p_win - p_loss in [-1, 1] — exactly chess's
# own utility scale (WinUtility=+1, DrawUtility=0, LossUtility=-1).
_WIN, _DRAW, _LOSS = 0, 1, 2
_FLIP = np.array([_LOSS, _DRAW, _WIN])              # win<->loss swap, draw fixed
_FLIP_TERM = np.array([_LOSS, _DRAW, _WIN], dtype=np.int8)

ALPHA_FLOOR = 0.05        # per-component Dirichlet floor (keeps lgamma/KL finite)
CONF_MIN    = 0.5         # smallest predictable / matched prior strength
CONF_MAX    = 100.0       # cap on any single concentration (anti-runaway)

# Near-degenerate Dirichlets used as terminal/proven spikes (a real point mass
# would make KL infinite; CONF_MAX mass on the proven corner is the proxy).
_SPIKE = np.full((3, 3), ALPHA_FLOOR)
_SPIKE[_WIN, _WIN] = _SPIKE[_DRAW, _DRAW] = _SPIKE[_LOSS, _LOSS] = CONF_MAX


def flip_alpha(alpha):
    """Reflect a (…,3) Dirichlet belief one ply up: swap win<->loss columns."""
    return np.ascontiguousarray(alpha[..., _FLIP])


def dir_mean(alpha):
    """Mean of a Dirichlet (…,3)."""
    return alpha / alpha.sum(-1, keepdims=True)


def dir_value(alpha):
    """Scalar value v = E[p_win] - E[p_loss] of a Dirichlet belief (…,3)."""
    m = dir_mean(alpha)
    return m[..., _WIN] - m[..., _LOSS]


def dir_value_mean_var(alpha):
    """(E[v], Var[v]) for v = p_win - p_loss under Dir(alpha).  Closed form:
    for a Dirichlet, Cov(x_i,x_j) = -m_i m_j/(a0+1), Var(x_i)=m_i(1-m_i)/(a0+1),
    so Var(x_w - x_l) = (Var_w + Var_l + 2 m_w m_l)/(...) collapses to
    (m_w(1-m_w)+m_l(1-m_l)+2 m_w m_l)/(a0+1)."""
    a0 = alpha.sum(-1)
    m = alpha / a0[..., None]
    mw, ml = m[..., _WIN], m[..., _LOSS]
    ev = mw - ml
    var = (mw * (1 - mw) + ml * (1 - ml) + 2 * mw * ml) / (a0 + 1.0)
    return ev, var


def moment_match_mixture(weights, alphas, conf_max=CONF_MAX):
    """Collapse a mixture  sum_k weights[k] * Dir(alphas[k])  to a single
    Dirichlet by matching mean + total variance (exact, closed form).

    weights: (k,) non-negative (renormalised here); alphas: (k, 3).
    Returns beta: (3,) concentration of the matched Dirichlet.

    Mean is matched EXACTLY; concentration comes from the aggregated
    precision  beta0 = sum_i M_i(1-M_i) / sum_i Var_mix(x_i) - 1  (Minka).  A
    single component round-trips to its own a0 exactly; validated against
    Monte-Carlo (mean <4e-4, total var <1e-4).  Clamped to [2*floor, conf_max].
    """
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
    a = np.asarray(alphas, dtype=np.float64)
    a0 = a.sum(1)                                   # (k,)
    m = a / a0[:, None]                             # (k,3) component means
    M = w @ m                                       # (3,)  mixture mean
    var_c = m * (1.0 - m) / (a0[:, None] + 1.0)     # within-component variance
    ex2 = w @ (var_c + m * m)                       # E[x^2] of the mixture
    var_mix = np.maximum(ex2 - M * M, 1e-12)        # (3,)
    beta0 = (M * (1.0 - M)).sum() / var_mix.sum() - 1.0
    beta0 = float(np.clip(beta0, 2.0 * ALPHA_FLOOR, conf_max))
    return np.maximum(M * beta0, ALPHA_FLOOR)


# ══════════════════════════════════════════════════════════════════════════════
#  Tree: closed-form Dirichlet beliefs, Thompson selection, policy-weighted backup
# ══════════════════════════════════════════════════════════════════════════════

class _TNode:
    """One expanded state.  Per legal action a it holds:
        alpha[a]  (3,)  the value belief pv_a = Dir(win,draw,loss), mover's view
        pprior[a]       policy-prior pseudocount  beta * softmax(logits)[a]
        pcount[a]       observed "a is optimal" counts (Thompson-argmax visits)
        term[a]         proven outcome (_WIN/_DRAW/_LOSS) or -1
    Constructed from ALREADY-GATHERED per-legal-action network rows.

    INCREMENTAL MIXTURE ACCUMULATORS (W, Mu, Su): node_qv needs the
    policy-weighted mixture  sum_a u_a * (mean_a, E[x^2]_a) / W,  u_a=pprior+pcount.
    Recomputing that sum over all k edges every backup was ~half of self-play
    CPU.  Instead we keep it UN-normalised so the moving denominator W is trivial,
    and every edge mutation applies an O(1) delta (see _acc): identical result,
    O(1) instead of O(k).  Stored as Python scalars/lists — numpy's per-op
    dispatch overhead on 3-vectors would erase the win.
        W  : float          = sum_a u_a
        Mu : [mw, md, ml]   = sum_a u_a * mean_a
        Su : [sw, sd, sl]   = sum_a u_a * E[x^2]_a   (E[x^2]=var_within+mean^2)
    """
    __slots__ = ('player', 'legal', 'alpha', 'pprior', 'pcount', 'term',
                 'vloss', 'children', 'obs', 'W', 'Mu', 'Su', 'ev', 'sd')

    def __init__(self, player, legal, p3, conf, plogits, beta):
        self.player = player
        self.legal  = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        p = np.asarray(p3, dtype=np.float64)
        c = np.asarray(conf, dtype=np.float64)
        self.alpha  = np.maximum(c[:, None] * p, ALPHA_FLOOR)          # (k,3)
        lg = np.asarray(plogits, dtype=np.float64)
        lg = lg - lg.max()
        pi = np.exp(lg); pi /= pi.sum()
        self.pprior = np.maximum(float(beta) * pi, 1e-6)               # (k,)
        self.pcount = np.zeros(k)
        self.term   = np.full(k, -1, dtype=np.int8)
        self.vloss  = np.zeros(k, dtype=np.int32)
        self.children = [None] * k
        self.obs = None
        # Initialise the mixture accumulators from all edges (the one O(k) pass).
        a0 = self.alpha.sum(1)                                         # (k,)
        m  = self.alpha / a0[:, None]                                  # (k,3)
        s  = m * (1.0 - m) / (a0[:, None] + 1.0) + m * m               # E[x^2]
        u  = self.pprior                                               # pcount=0
        self.W  = float(u.sum())
        self.Mu = (u[:, None] * m).sum(0).tolist()
        self.Su = (u[:, None] * s).sum(0).tolist()
        # Gaussian selection needs per-edge (E[v], sd[v]); cache only when on.
        if _GAUSSIAN_SELECT:
            mw, ml = m[:, _WIN], m[:, _LOSS]
            self.ev = mw - ml
            self.sd = np.sqrt((mw * (1 - mw) + ml * (1 - ml) + 2 * mw * ml)
                              / (a0 + 1.0))
        else:
            self.ev = self.sd = None


def _edge_ms(a):
    """(mean, E[x^2]) of one edge's Dirichlet, as two 3-tuples of Python floats.
    Pure-scalar (no numpy dispatch) — this is on the per-node backup hot path."""
    aw, ad, al = a.tolist()
    a0 = aw + ad + al
    inv = 1.0 / (a0 + 1.0)
    mw, md, ml = aw / a0, ad / a0, al / a0
    return ((mw, md, ml),
            (mw * (1.0 - mw) * inv + mw * mw,
             md * (1.0 - md) * inv + md * md,
             ml * (1.0 - ml) * inv + ml * ml))


def _acc(node, idx, sign):
    """Add (sign=+1) or remove (sign=-1) edge idx's contribution to the node's
    mixture accumulators, using the edge's CURRENT alpha/pcount.  Wrap every edge
    mutation as: _acc(node, idx, -1); <mutate>; _acc(node, idx, +1)."""
    u = float(node.pprior[idx] + node.pcount[idx]) * sign
    (mw, md, ml), (sw, sd, sl) = _edge_ms(node.alpha[idx])
    node.W += u
    Mu, Su = node.Mu, node.Su
    Mu[0] += u * mw; Mu[1] += u * md; Mu[2] += u * ml
    Su[0] += u * sw; Su[1] += u * sd; Su[2] += u * sl


def _set_edge_v(node, idx):
    """Refresh the cached (E[v], sd[v]) for one edge after its alpha changed —
    only called in Gaussian-selection mode (pure scalar, backup hot path)."""
    aw, ad, al = node.alpha[idx].tolist()
    a0 = aw + ad + al
    mw, ml = aw / a0, al / a0
    node.ev[idx] = mw - ml
    node.sd[idx] = ((mw * (1 - mw) + ml * (1 - ml) + 2 * mw * ml)
                    / (a0 + 1.0)) ** 0.5


def _pi_post(node):
    """Posterior policy weights pi_a = (pprior + pcount) / total — the belief
    over which move is optimal (NN prior updated by search observations).  Also
    the mixture weights inside node_qv (there via the incremental accumulator)."""
    w = node.pprior + node.pcount
    return w / w.sum()


def node_qv(node, conf_max=CONF_MAX):
    """Position value-distribution qv = moment-match( sum_a pi_a * pv_a ), the
    policy-weighted mixture of the edge beliefs (mover's perspective).  Read
    straight off the incremental accumulators in O(1) — algebraically identical
    to moment_match_mixture(pi_post, alpha)."""
    W = node.W
    Mw, Md, Ml = node.Mu[0] / W, node.Mu[1] / W, node.Mu[2] / W
    Vw = node.Su[0] / W - Mw * Mw
    Vd = node.Su[1] / W - Md * Md
    Vl = node.Su[2] / W - Ml * Ml
    if Vw < 1e-12: Vw = 1e-12
    if Vd < 1e-12: Vd = 1e-12
    if Vl < 1e-12: Vl = 1e-12
    beta0 = (Mw * (1 - Mw) + Md * (1 - Md) + Ml * (1 - Ml)) / (Vw + Vd + Vl) - 1.0
    if beta0 < 2.0 * ALPHA_FLOOR: beta0 = 2.0 * ALPHA_FLOOR
    elif beta0 > conf_max:        beta0 = conf_max
    return np.array([max(Mw * beta0, ALPHA_FLOOR),
                     max(Md * beta0, ALPHA_FLOOR),
                     max(Ml * beta0, ALPHA_FLOOR)])


def _set_term(node, idx, outcome, proven_win_bonus=0.0):
    """Mark edge idx proven; bake in the spike belief.  A proven WIN also gets
    extra policy-prior pseudocounts so pi_a snaps onto the mating move (teaches
    the prior the max at the end of the game fast — the MCTS-Solver accelerates
    the *value*; this accelerates the *policy*)."""
    _acc(node, idx, -1.0)                        # remove old contribution
    node.term[idx]  = outcome
    node.alpha[idx] = _SPIKE[outcome]
    if outcome == _WIN and proven_win_bonus:
        node.pcount[idx] += proven_win_bonus
    if _GAUSSIAN_SELECT and node.ev is not None:
        _set_edge_v(node, idx)
    _acc(node, idx, +1.0)                         # add spike contribution


def _sample_edge_values(node, rng, temp=1.0):
    """One Thompson draw of each edge's value v=p_win-p_loss from Dir(temp*alpha)
    (temp>1 sharpens toward the mean → argmax≈max; temp=1 samples as-is), minus
    a virtual-loss penalty on in-flight edges for within-wave diversification.

    Exact Dirichlet sampling via 3 gamma draws/edge (optimal per the standard
    method); the only shaved cost is dropping the no-op temp copy and forming
    v directly from two gamma columns + a (k,) sum instead of normalising the
    full (k,3).  With `rng` a numpy Generator (PCG64) the gammas are ~15% faster
    than legacy RandomState; the draw is identical in distribution either way.

    In 'gaussian' selection mode (set_selection), instead sample v ~ N(E[v],
    Var[v]) from the cached per-edge moments — ~6-7x cheaper, an approximation
    (temperature narrows the sd; unbounded draws are fine for an argmax)."""
    if _GAUSSIAN_SELECT and node.ev is not None:
        sd = node.sd if temp == 1.0 else node.sd * (temp ** -0.5)
        v = node.ev + sd * rng.standard_normal(len(node.legal))
    else:
        g = rng.standard_gamma(node.alpha if temp == 1.0 else node.alpha * temp)
        s = g.sum(1)                                # (k,)  Dir normaliser
        v = (g[:, _WIN] - g[:, _LOSS]) / s          # p_win - p_loss, exact
    if node.vloss.any():
        v = v - 2.0 * node.vloss                    # push in-flight edges down
    return v


def _select_leaf(root, root_state, rng, temp=1.0, proven_win_bonus=0.0):
    """Thompson-descend to an unexpanded/terminal edge, applying virtual loss.
    Returns (path, leaf_state_or_None, (node, idx)_or_None):
      terminal/proven edge -> (path, None, None)   (outcome already in node.term)
      unexpanded edge      -> (path, state, (node, idx))  (needs a NN eval).
    """
    node, state, path = root, root_state.clone(), []
    while True:
        idx = int(_sample_edge_values(node, rng, temp).argmax())
        node.vloss[idx] += 1
        path.append((node, idx))
        if node.term[idx] >= 0:
            return path, None, None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            r = state.returns()[node.player]                    # +1 / 0 / -1
            _set_term(node, idx, _WIN if r > 0 else (_LOSS if r < 0 else _DRAW),
                      proven_win_bonus)
            return path, None, None
        child = node.children[idx]
        if child is None:
            return path, state, (node, idx)
        node = child


def _backup(path, conf_max=CONF_MAX):
    """Bottom-up backup.  For each (node, idx) leaf->root:
      - remove virtual loss;
      - credit the selected edge with a policy-prior observation (pcount += 1);
      - replace the edge belief pv_a with the child's qv, flipped one ply.
    Nothing is accumulated for the value — the belief is RECOMPUTED from the
    (now-updated) child, so a fresh leaf estimate propagates to the root."""
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        _acc(node, idx, -1.0)                     # remove edge's old contribution
        node.pcount[idx] += 1.0                   # policy observation
        if node.term[idx] < 0 and node.children[idx] is not None:
            node.alpha[idx] = flip_alpha(node_qv(node.children[idx], conf_max))
            if _GAUSSIAN_SELECT:
                _set_edge_v(node, idx)
        _acc(node, idx, +1.0)                      # add updated contribution


def _node_solved_outcome(node):
    """Proven outcome for node.player if the node is solved, else None.
    WIN if any edge is a proven win; else (all edges proven) DRAW if any proven
    draw else LOSS (the mover forces at least a draw over a loss)."""
    t = node.term
    if (t == _WIN).any():
        return _WIN
    if (t >= 0).all():
        return _DRAW if (t == _DRAW).any() else _LOSS
    return None


def _propagate_solved(path, aux=None, conf_cap=CONF_MAX, proven_win_bonus=0.0):
    """Walk leaf->root; when a node becomes fully solved, prove the parent edge
    entering it (flipped).  Emits exact solver-labelled training samples into
    `aux` (once per node — a solved node is never re-descended)."""
    for k in range(len(path) - 1, 0, -1):
        node = path[k][0]
        out = _node_solved_outcome(node)
        if out is None:
            break
        parent, pidx = path[k - 1]
        if parent.term[pidx] >= 0:
            break
        _set_term(parent, pidx, int(_FLIP_TERM[out]), proven_win_bonus)
        if aux is not None and node.obs is not None:
            t = make_target(node, conf_cap)
            # Exact solver label: its value belief IS ground truth, so skip the
            # game-outcome value-MSE (the played-out z can differ from the proven
            # value) and mark it as an exact sample.
            t['solved'] = True
            t['no_z'] = True
            aux.append(t)


def _backup_terminal(path, aux=None, conf_cap=CONF_MAX, conf_max=CONF_MAX,
                     proven_win_bonus=0.0):
    _backup(path, conf_max)
    _propagate_solved(path, aux, conf_cap, proven_win_bonus)


def _descend(root, action):
    """Tree reuse: the subtree under `action` becomes the next search's root.
    Returns None if that edge was never expanded."""
    if root is None:
        return None
    hit = np.nonzero(root.legal == action)[0]
    return root.children[int(hit[0])] if len(hit) else None


def root_pick(root, rng, thompson, temp=1.0):
    """Final move: Thompson-sample the edge beliefs (exploratory phase) or take
    the posterior value-mean argmax (endgame / evaluation)."""
    if thompson:
        v = _sample_edge_values(root, rng, temp)
    else:
        v = dir_value(root.alpha)
    return int(root.legal[int(v.argmax())])


# ══════════════════════════════════════════════════════════════════════════════
#  Training targets  (extracted from a searched root)
# ══════════════════════════════════════════════════════════════════════════════
# A per-move sample is a dict of numpy arrays (all tiny; obs is fp16):
#   obs      (D,) fp16            network input for this position
#   legal    (k,) int32          legal action ids (the policy support)
#   pcount   (k,) fp32           policy visit/observation counts  (L_policyprior)
#   ev_idx   (m,) int32          legal-index of each EVIDENCE edge (expanded/proven)
#   ev_alpha (m,3) fp32          backed-up per-action value Dirichlet (L_actionvaluedist)
#   qv       (3,) fp32           backed-up position value Dirichlet   (L_valuedist)
#   z        fp32                game outcome for root.player          (L_value)
#   solved   bool               solver-labelled (exact) sample?

def make_target(root, conf_cap=CONF_MAX):
    """Root search state -> training-target dict (z filled later, at game end).
    Evidence edges are those actually searched (expanded child or proven); the
    per-action value target is scored only there so the loss never grades the
    net against its own untouched prior.  Concentrations are capped at conf_cap
    (mean-preserving) so no single generation teaches unbounded certainty."""
    k = len(root.legal)
    ev = (root.term >= 0) | np.array([c is not None for c in root.children])
    ev_idx = np.nonzero(ev)[0].astype(np.int32)
    ev_alpha = _cap_conc(root.alpha[ev_idx], conf_cap).astype(np.float32)
    pcount = np.minimum(root.pcount.copy(), conf_cap).astype(np.float32)
    qv = _cap_conc(node_qv(root)[None, :], conf_cap)[0].astype(np.float32)
    return {'obs': root.obs, 'legal': root.legal.copy(),
            'pcount': pcount, 'ev_idx': ev_idx, 'ev_alpha': ev_alpha,
            'qv': qv, 'z': np.float32(0.0), 'solved': False,
            'player': int(root.player)}


def _cap_conc(alpha, conf_cap):
    """Rescale any (…,3) Dirichlet whose total exceeds conf_cap down to conf_cap,
    preserving the mean (direction).  Floors kept."""
    a = np.asarray(alpha, dtype=np.float64)
    tot = a.sum(-1, keepdims=True)
    scale = np.minimum(1.0, conf_cap / np.maximum(tot, 1e-9))
    return np.maximum(a * scale, ALPHA_FLOOR)


def z_mix_episode(samples, returns, z_mix, z_gamma=1.0):
    """Blend the observed game outcome z into each sample's value TARGETS.

    Sets sample['z'] to the outcome from that position's mover's perspective
    (used by L_value/MSE).  Additionally nudges the position value-dist target
    `qv` toward the observed corner by weight  w = z_mix * z_gamma**(moves-to-end)
    — the same dense-outcome grounding AlphaZero/KataGo use, ramped so late
    positions (closer to ground truth) are pulled harder.  Proven (solved)
    samples are exact and left untouched."""
    n = len(samples)
    for i, s in enumerate(samples):
        z = float(returns[s['player']])                     # +1 / 0 / -1
        s['z'] = np.float32(z)
        if s['solved']:
            continue
        w = z_mix * (z_gamma ** (n - 1 - i))
        if w <= 0:
            continue
        corner = _WIN if z > 0 else (_LOSS if z < 0 else _DRAW)
        obs_alpha = np.full(3, ALPHA_FLOOR)
        obs_alpha[corner] = 1.0
        m = (1 - w) * dir_mean(s['qv']) + w * dir_mean(obs_alpha)
        s['qv'] = np.maximum(m * s['qv'].sum(), ALPHA_FLOOR).astype(np.float32)
    return samples


def strip_episode_meta(samples):
    """Ply-capped game: the outcome was never observed, so leave z=0 and mark
    these samples value-MSE-free (z_weight handled in the loss via 'no_z')."""
    for s in samples:
        s['no_z'] = True
    return samples


def restart_prefix(seq, rng, k_min, k_max):
    """A backward-restart curriculum prefix: drop a random tail of `seq`."""
    if len(seq) <= k_min:
        return []
    k = rng.integers(k_min, min(k_max, len(seq)) + 1)
    return list(seq[:len(seq) - k])


def opening_decision(p3, conf, plog, rng, conf_thresh):
    """One cheap (search-free) opening step from a gathered NN eval at a position.
    p3 (k,3), conf (k,), plog (k,) are the LEGAL-action rows.  Returns
    (local_action_idx, confident):
      - move = Thompson sample of the per-action value beliefs, argmax of v=w−l
        (same rule the tree uses, applied to the raw net — no search);
      - confident = the moment-matched STATE value-distribution concentration
        (α0 of qv = Σ_a π_a·pv_a) ≥ conf_thresh, i.e. the net is sure of this
        position's outcome.  As the net learns endgames this fires earlier, so
        MCTS self-play takes over earlier and earlier — no global schedule."""
    pv = np.maximum(conf[:, None] * np.asarray(p3, float), ALPHA_FLOOR)   # (k,3)
    g = rng.standard_gamma(pv); x = g / g.sum(1, keepdims=True)
    a = int((x[:, _WIN] - x[:, _LOSS]).argmax())
    lg = np.asarray(plog, float); lg = lg - lg.max()
    pi = np.exp(lg); pi /= pi.sum()
    conc = moment_match_mixture(pi, pv).sum()
    return a, bool(conc >= conf_thresh)


def random_backbone(game, rng, depth, max_plies):
    """Adaptive backward curriculum.  Play a uniform-RANDOM game to its terminal
    (capped at max_plies), then resume `depth` plies before the end: the random
    prefix is the (untrained) opening, and MCTS self-play takes over only for the
    last ~`depth` plies.  Early in training `depth` is small, so MCTS + training
    focus on near-terminal ENDGAME positions (high signal: short horizon, dense
    solver proofs); as the net masters them, the training loop grows `depth`,
    pushing the MCTS frontier back toward the opening.  When `depth` >= the game
    length, resume=0 and it's an ordinary full self-play game.

    Returns (resume_state, replayed_actions, resume_ply).  A cheap all-random
    rollout (no NN) is used to find a realistic terminal to back up from."""
    st = game.new_initial_state()
    seq = []
    while not st.is_terminal() and len(seq) < max_plies:
        legal = st.legal_actions()
        a = int(legal[rng.integers(len(legal))])
        st.apply_action(a); seq.append(a)
    resume = max(0, len(seq) - int(round(depth)))
    state = game.new_initial_state()
    for a in seq[:resume]:
        state.apply_action(a)
    return state, seq[:resume], resume


# ══════════════════════════════════════════════════════════════════════════════
#  Multiprocess self-play worker (top-level so `spawn` can import it)
# ══════════════════════════════════════════════════════════════════════════════
# Same design as the sibling ThompsonZero notebooks: N CPU worker processes run
# the trees; a central inference-server THREAD in the parent batches all their
# NN requests into one forward pass (see MPSelfPlayPool).  This is the dominant
# self-play speedup on a multi-core box — the tree work (numpy, per node) is
# CPU-bound and serial within a process, so parallelising it across cores while
# the GPU serves big fused batches is what makes the sibling runs fast.
#
# Wire format (crosses pickling queues, so kept tiny):
#   request : (worker_id, net_id, obs (n, D) fp16, [legals int32, ...])
#   response: [(p3 (k,3) f32, conf (k,) f32, plog (k,) f32, beta float), ...]
#             — only the gathered legal entries, never the dense 4674-wide rows.

def _mp_load_game(cfg):
    loader = cfg.get('game_loader')                 # dotted 'module:function'
    if loader:
        mod, fn = loader.split(':')
        import importlib
        return getattr(importlib.import_module(mod), fn)()
    import pyspiel
    return pyspiel.load_game(cfg.get('game_name', 'chess'))


def mp_worker(worker_id, req_q, resp_q, pool_resp_q, episode_q, cfg):
    game = _mp_load_game(cfg)
    try:
        set_game(game)
    except Exception:
        pass
    set_selection(cfg.get('selection', 'dirichlet'))     # honour it in-process
    rng = np.random.default_rng(cfg['seed'] + worker_id * 7919)
    conf_cap = cfg['conf_cap']; pwb = cfg.get('proven_win_bonus', 0.0)
    max_plies = cfg.get('max_plies', 400)
    z_mix = cfg.get('z_mix', 0.5); z_gamma = cfg.get('z_gamma', 0.97)
    temp_thr = cfg.get('temp_threshold', 20); late_temp = cfg.get('late_temp', 8.0)
    pool_prob = cfg.get('pool_prob', 0.0); pool_dir = cfg.get('checkpoint_dir')
    rand_pool_frac = cfg.get('random_pool_frac', 0.5)
    restart_prob = cfg.get('restart_prob', 0.0)
    restart_kmin = cfg.get('restart_k_min', 2); restart_kmax = cfg.get('restart_k_max', 30)
    restart_cap = cfg.get('restart_pool_cap', 128)
    restart_pool = []
    curriculum = cfg.get('curriculum', False)
    curr_conf_thresh = cfg.get('curr_conf_thresh', 8.0)
    curr_backup = cfg.get('curr_backup', 3)

    def _temp(move):
        return 1.0 if move < temp_thr else late_temp

    def _push_seed(seq):
        if len(seq) >= 2:
            restart_pool.append(list(seq))
            if len(restart_pool) > restart_cap:
                del restart_pool[0]

    def handoff(i):
        s = slots[i]
        resume = max(0, len(s['open_actions']) - curr_backup)
        st = game.new_initial_state()
        for a in s['open_actions'][:resume]:
            st.apply_action(a)
        s['state'] = st; s['actions'] = list(s['open_actions'][:resume])
        s['resume'] = resume; s['move'] = resume
        s['phase'] = 'mcts'; s['root'] = None; s['n'] = 0

    def new_game():
        sims = cfg['fast_sims'] if rng.random() < cfg['fast_prob'] else cfg['full_sims']
        state, actions, resume, phase = game.new_initial_state(), [], 0, 'mcts'
        if curriculum:
            phase = 'open'
        elif restart_pool and rng.random() < restart_prob:
            seq = restart_pool[rng.integers(len(restart_pool))]
            pref = restart_prefix(seq, rng, restart_kmin, restart_kmax)
            st = game.new_initial_state(); ok = True
            for a in pref:
                if st.is_terminal() or a not in st.legal_actions():
                    ok = False; break
                st.apply_action(int(a))
            if ok and pref and not st.is_terminal():
                state, actions = st, list(pref)
        slot = {'state': state, 'hist': [], 'aux': [], 'actions': actions,
                'move': resume, 'resume': resume, 'sims': sims, 'phase': phase,
                'open_actions': [], 'root': None, 'n': 0, 'pool': None}
        if pool_prob > 0 and rng.random() < pool_prob:
            try:
                labels = [f[6:-3] for f in os.listdir(pool_dir)
                          if f.startswith('bench_') and f.endswith('.pt')] \
                    if pool_dir else []
            except OSError:
                labels = []
            label = ('random' if not labels or rng.random() < rand_pool_frac
                     else labels[rng.integers(len(labels))])
            slot['pool'] = {'label': label, 'side': int(rng.integers(2))}
        return slot

    def finish_and_reset(i):
        s = slots[i]; st = s['state']
        if st.is_terminal():
            ret = st.returns()
            z_mix_episode(s['hist'], ret, z_mix, z_gamma)
            result = 'draw' if ret[0] == 0.0 else 'decisive'
            if restart_prob > 0 and result == 'decisive':
                _push_seed(s['actions'])
        else:
            strip_episode_meta(s['hist']); result = 'cutoff'
        episode_q.put((s['hist'] + s['aux'], len(s['aux']), result,
                       int(s['move']), int(s['resume'])))
        slots[i] = new_game()

    slots = [new_game() for _ in range(cfg['games_per_worker'])]
    mid = max(1, cfg['games_per_worker'] // 2)
    halves = [list(range(mid)), list(range(mid, cfg['games_per_worker']))]

    def collect(idxs):
        evals, paths, obs, legals = [], [], [], []
        seen = set()
        for i in idxs:
            s = slots[i]; st0 = s['state']; pool = s['pool']
            if pool is not None and st0.current_player() == pool['side']:
                continue
            if s['phase'] == 'open':               # cheap Thompson opening
                if st0.is_terminal():
                    handoff(i)
                else:
                    leg = st0.legal_actions()
                    o = np.asarray(st0.observation_tensor(st0.current_player()),
                                   dtype=np.float16)
                    evals.append(('open', i, leg)); obs.append(o)
                    legals.append(np.asarray(leg, dtype=np.int32))
                continue
            if s['root'] is None:
                leg = st0.legal_actions()
                o = np.asarray(st0.observation_tensor(st0.current_player()),
                               dtype=np.float16)
                evals.append(('root', i, leg, o)); obs.append(o)
                legals.append(np.asarray(leg, dtype=np.int32)); continue
            if _node_solved_outcome(s['root']) is not None:
                continue
            wave = min(cfg['wave'], s['sims'] - s['n'])
            for _ in range(max(wave, 0)):
                path, st, edge = _select_leaf(s['root'], st0, rng,
                                              _temp(s['move'] - s['resume']), pwb)
                if st is None:
                    _backup_terminal(path, s['aux'], conf_cap, CONF_MAX, pwb)
                    s['n'] += 1; continue
                node, idx = edge
                paths.append((i, path, node, idx))
                if (id(node), idx) not in seen:
                    seen.add((id(node), idx))
                    leg = st.legal_actions()
                    o = np.asarray(st.observation_tensor(st.current_player()),
                                   dtype=np.float16)
                    evals.append(('leaf', node, idx, st.current_player(), leg, o))
                    obs.append(o); legals.append(np.asarray(leg, dtype=np.int32))
        return evals, paths, obs, legals

    def apply_and_advance(idxs, evals, paths, resp):
        if evals:
            for e, (p3, c, pl, bt) in zip(evals, resp):
                if e[0] == 'open':
                    _, i, leg = e
                    loc, confident = opening_decision(p3, c, pl, rng,
                                                      curr_conf_thresh)
                    s = slots[i]
                    if confident:
                        handoff(i)
                    else:
                        a = int(leg[loc])
                        s['open_actions'].append(a); s['state'].apply_action(a)
                        s['move'] += 1
                        if s['state'].is_terminal() or s['move'] >= max_plies:
                            handoff(i)
                    continue
                if e[0] == 'root':
                    _, i, leg, o = e; st = slots[i]['state']
                    nd = _TNode(st.current_player(), leg, p3, c, pl, bt)
                    nd.obs = o; slots[i]['root'] = nd
                else:
                    _, node, idx, player, leg, o = e
                    nd = _TNode(player, leg, p3, c, pl, bt)
                    nd.obs = o; node.children[idx] = nd
        for i, path, node, idx in paths:
            _backup(path); slots[i]['n'] += 1
        for i in idxs:
            s = slots[i]
            if s['root'] is None:
                continue
            if s['n'] < s['sims'] and _node_solved_outcome(s['root']) is None:
                continue
            root = s['root']
            s['hist'].append(make_target(root, conf_cap))
            mm = s['move'] - s['resume']         # plies since MCTS took over
            a = root_pick(root, rng, thompson=(mm < temp_thr), temp=_temp(mm))
            pidx = int(np.nonzero(root.legal == a)[0][0])
            s['actions'].append(int(a))
            s['root'] = root.children[pidx]
            s['state'].apply_action(a); s['move'] += 1; s['n'] = 0
            if s['state'].is_terminal() or s['move'] >= max_plies:
                finish_and_reset(i)

    def resolve_pool_moves(idxs):
        for i in idxs:
            s = slots[i]; pool = s['pool']
            if pool is None:
                continue
            state = s['state']
            if state.current_player() != pool['side']:
                continue
            legal = state.legal_actions()
            if pool['label'] == 'random':
                a = int(legal[rng.integers(len(legal))])
            else:
                o = np.asarray(state.observation_tensor(state.current_player()),
                               dtype=np.float16)
                req_q.put((worker_id, pool['label'], o[None],
                           [np.asarray(legal, dtype=np.int32)]))
                (p3, _c, pl, _b), = pool_resp_q.get()
                lg = pl - pl.max(); pi = np.exp(lg); pi /= pi.sum()
                a = int(legal[int(pi.argmax())])
            s['root'] = _descend(s['root'], a)
            state.apply_action(a); s['actions'].append(a); s['move'] += 1
            if state.is_terminal() or s['move'] >= max_plies:
                finish_and_reset(i)

    inflight = [None, None]
    while True:
        for h in (0, 1):
            if inflight[h] is not None:
                evals, paths, sent = inflight[h]
                resp = resp_q.get() if sent else None
                apply_and_advance(halves[h], evals, paths, resp)
                inflight[h] = None
            resolve_pool_moves(halves[h])
            evals, paths, obs, legals = collect(halves[h])
            sent = False
            if evals:
                req_q.put((worker_id, 'live', np.stack(obs), legals)); sent = True
            inflight[h] = (evals, paths, sent)


# ══════════════════════════════════════════════════════════════════════════════
#  Game wiring (obs shape + action count), set once from the notebook
# ══════════════════════════════════════════════════════════════════════════════
_OBS_SHAPE = None
_NUM_ACTIONS = None
_HEAD_CH_DEFAULT = 8        # width of the 1x1 conv feeding the flat action heads;
                           # THE dominant lever on parameter count (see the net)


def set_head_ch(n):
    """Set the default head width for every ThompsonFullNet built afterwards.
    head_ch controls the flat penultimate width (head_ch*8*8) that feeds the
    three action-wide heads, so it (not channels/num_blocks) sets the param
    count: 8→~12M, 4→~6.5M, 2→~3.5M, 1→~2.0M for chess's 4674 actions."""
    global _HEAD_CH_DEFAULT
    _HEAD_CH_DEFAULT = int(n)


# In-tree Thompson selection sampler. 'dirichlet' (default) is the EXACT draw
# (3 gammas/edge → v=p_win-p_loss).  'gaussian' is a ~6-7x-cheaper APPROXIMATION:
# sample v ~ N(E[v], Var[v]) from the belief's closed-form moments (cached per
# edge; only maintained when this is on, so 'dirichlet' pays nothing).  It
# matches mean+variance exactly and is a good fit (KS~0.01 vs the true v), but
# drops skewness and is unbounded — harmless for an argmax selection.  Backup,
# targets, and losses are UNCHANGED either way; this only affects which action a
# rollout explores.  A/B it against an exact run before trusting it.
_GAUSSIAN_SELECT = False


def set_selection(mode):
    """mode: 'dirichlet' (exact, default) | 'gaussian' (fast approximation).
    Set once before building the self-play/eval bots; workers receive it via
    cfg['selection'] and call this themselves."""
    global _GAUSSIAN_SELECT
    if mode not in ('dirichlet', 'gaussian'):
        raise ValueError(f"selection must be 'dirichlet' or 'gaussian', got {mode!r}")
    _GAUSSIAN_SELECT = (mode == 'gaussian')


def set_game(game):
    """Record chess's observation-plane shape + action count for the network and
    the loss's flat action indexing.  Call once before building the network."""
    global _OBS_SHAPE, _NUM_ACTIONS
    _OBS_SHAPE = tuple(game.observation_tensor_shape())      # (20, 8, 8)
    _NUM_ACTIONS = game.num_distinct_actions()               # 4674
    return _OBS_SHAPE, _NUM_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
#  Everything below needs torch (network, losses, optimizer, bots, eval, ckpt)
# ══════════════════════════════════════════════════════════════════════════════
if _HAS_TORCH:

    # ── Device selection ──────────────────────────────────────────────────────
    def pick_device(pref='auto'):
        if pref in ('directml', 'auto'):
            try:
                import torch_directml
                try:    name = torch_directml.device_name(0)
                except Exception: name = 'DirectML GPU'
                print(f'Using DirectML: {name}')
                return torch_directml.device(), 'directml'
            except Exception:
                if pref == 'directml':
                    print('DirectML requested but unavailable — falling back.')
        if pref in ('cuda', 'auto') and torch.cuda.is_available():
            return torch.device('cuda'), 'cuda'
        return torch.device('cpu'), 'cpu'

    # ── Input tensors ─────────────────────────────────────────────────────────
    def state_to_tensor(state, device):
        obs = np.array(state.observation_tensor(state.current_player()),
                       dtype=np.float32)
        return torch.from_numpy(obs.reshape(1, *_OBS_SHAPE)).to(device)

    def batch_to_tensor(obs_list, device):
        obs = np.asarray(obs_list, dtype=np.float32)
        return torch.from_numpy(obs.reshape(-1, *_OBS_SHAPE)).to(device)

    # ── DirectML-safe primitives (fused kernels lack DML backward) ────────────
    class _GroupNorm(nn.Module):
        def __init__(self, num_groups, num_channels, eps=1e-5):
            super().__init__()
            self.num_groups, self.eps = num_groups, eps
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        def forward(self, x):
            n, c = x.shape[0], x.shape[1]
            xg = x.reshape(n, self.num_groups, -1)
            mean = xg.mean(dim=2, keepdim=True)
            var = (xg - mean).pow(2).mean(dim=2, keepdim=True)
            xg = (xg - mean) / torch.sqrt(var + self.eps)
            return (xg.reshape(x.shape) * self.weight.view(1, c, 1, 1)
                    + self.bias.view(1, c, 1, 1))

    def _norm(channels):
        g = min(8, channels)
        while channels % g != 0:
            g -= 1
        return _GroupNorm(g, channels)

    def _softplus(x):
        return torch.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            mid = max(channels // reduction, 4)
            self.fc = nn.Sequential(nn.Linear(channels, mid),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(mid, channels * 2))

        def forward(self, x):
            s = x.mean(dim=(2, 3))
            scale, bias = self.fc(s).chunk(2, dim=1)
            return (x * torch.sigmoid(scale)[:, :, None, None]
                    + bias[:, :, None, None])

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                _norm(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                _norm(channels))
            self.se = SEBlock(channels)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.se(self.net(x)) + x)

    # ── The network: trunk + policy-prior head + per-action value head ────────
    POLICY_CONC_MAX = CONF_MAX          # cap on the policy Dirichlet's beta

    class ThompsonFullNet(nn.Module):
        """Trunk → two learned heads.
        forward(x) → (probs (B,A,3) softmax, conf (B,A) in [CONF_MIN,CONF_MAX],
                      plogits (B,A), beta (B,) in [CONF_MIN,POLICY_CONC_MAX]).
        The per-action value belief is Dir(conf[a]*probs[a]); the policy-prior is
        Dir(beta*softmax(plogits)); v and qv are DERIVED (see full_loss / the
        tree), never separate parameters.

        NOTE ON SIZE: with 4674 chess actions the three action-wide heads
        (dist_out 3*A, conf_out A, plog_out A) dominate the parameter count and
        scale with `head_ch` (which sets the flat width = head_ch*8*8), NOT with
        `channels`/`num_blocks` (the trunk is <0.5M).  head_ch=8→~12M, 4→~6.5M,
        2→~3.5M, 1→~2.0M.  Defaults to the module global set by set_head_ch()
        so every construction site (self-play, eval, resume) stays consistent."""

        def __init__(self, channels=128, num_blocks=10, head_ch=None):
            super().__init__()
            hc = int(head_ch if head_ch is not None else _HEAD_CH_DEFAULT)
            self._head_ch = hc
            in_ch = _OBS_SHAPE[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                _norm(channels), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[ResBlock(channels)
                                        for _ in range(num_blocks)])
            self.head = nn.Sequential(
                nn.Conv2d(channels, hc, 1, bias=False),
                _norm(hc), nn.ReLU(inplace=True), nn.Flatten())
            flat = hc * _OBS_SHAPE[1] * _OBS_SHAPE[2]
            self.dist_out = nn.Linear(flat, _NUM_ACTIONS * 3)   # value: W/D/L
            self.conf_out = nn.Linear(flat, _NUM_ACTIONS)       # value: conf
            self.plog_out = nn.Linear(flat, _NUM_ACTIONS)       # policy: logits
            self.beta_out = nn.Linear(flat, 1)                  # policy: conc
            # Untrained net: near-uniform value, WEAK confidence (α0≈2.4) so
            # search dominates from generation 0; policy prior weak too.
            nn.init.constant_(self.conf_out.bias, 1.4)
            nn.init.constant_(self.beta_out.bias, 1.4)

        def forward(self, x):
            # Returns RAW dist logits + RAW conf (softmax-over-3 and softplus are
            # per-action/elementwise, so they're deferred and applied only to the
            # ~35 LEGAL entries by the consumer — never the full 4674 actions).
            h = self.head(self.body(self.stem(x)))
            dist_logits = self.dist_out(h).view(-1, _NUM_ACTIONS, 3)
            conf_raw = self.conf_out(h)
            plog = self.plog_out(h)
            beta = torch.clamp(CONF_MIN + _softplus(self.beta_out(h)).squeeze(-1),
                               max=POLICY_CONC_MAX)
            return dist_logits, conf_raw, plog, beta

    def _act_p3(logits):
        """RAW dist logits → (p_win, p_draw, p_loss) probabilities."""
        return F.softmax(logits, dim=-1)

    def _act_conf(raw):
        """RAW conf output → concentration α0 ∈ [CONF_MIN, CONF_MAX]."""
        return torch.clamp(CONF_MIN + _softplus(raw), max=CONF_MAX)

    # ── lgamma / digamma with a DirectML-friendly fallback ────────────────────
    # The closed-form Dirichlet KL is already minimal; its ONLY expensive pieces
    # are lgamma (log-normaliser) and digamma (E[log x]).  Neither has a DirectML
    # kernel, which is why the loss was offloaded to the CPU (dense download +
    # CPU math under the model lock — the dominant training cost).  These
    # elementary-op approximations (8-step upward recurrence + Stirling/asymptotic
    # series) run entirely on the GPU: float32 error ~2e-4, and autograd through
    # _lgamma_dml reproduces digamma to ~1e-9 — negligible for a loss.  Used only
    # on DirectML; CUDA/CPU keep the exact native ops.
    _LG_SHIFT = 8

    def _lgamma_dml(x):
        g = torch.zeros_like(x); xx = x
        for _ in range(_LG_SHIFT):
            g = g - torch.log(xx); xx = xx + 1.0
        inv = 1.0 / xx; inv2 = inv * inv
        return g + ((xx - 0.5) * torch.log(xx) - xx + 0.5 * math.log(2 * math.pi)
                    + inv * (1.0/12 - inv2 * (1.0/360 - inv2 * (1.0/1260))))

    def _digamma_dml(x):
        g = torch.zeros_like(x); xx = x
        for _ in range(_LG_SHIFT):
            g = g - 1.0 / xx; xx = xx + 1.0
        inv = 1.0 / xx; inv2 = inv * inv
        return g + (torch.log(xx) - 0.5 * inv
                    - inv2 * (1.0/12 - inv2 * (1.0/120 - inv2 / 252.0)))

    _LG_APPROX = False           # set True on DirectML (see set_backend)
    _DML_GPU_LOSS = True         # run the whole loss on the DML GPU; auto-falls
                                 # back to the CPU-split path if a device op fails

    def set_backend(backend):
        """Pick exact (CUDA/CPU) vs GPU-approx (DirectML) lgamma/digamma so the
        whole loss can run on-device.  Called from the training setup."""
        global _LG_APPROX
        _LG_APPROX = (backend == 'directml')

    def _lg(x):
        return _lgamma_dml(x) if _LG_APPROX else torch.lgamma(x)

    def _dg(x):
        return _digamma_dml(x) if _LG_APPROX else torch.digamma(x)

    # ── Closed-form Dirichlet KL (batched, differentiable) ────────────────────
    def _dir_kl_rows(a, b):
        """forward KL( Dir(a) ‖ Dir(b) ) per row, a/b: (N,3)."""
        a0, b0 = a.sum(-1), b.sum(-1)
        return (_lg(a0) - _lg(a).sum(-1) - _lg(b0) + _lg(b).sum(-1)
                + ((a - b) * (_dg(a) - _dg(a0).unsqueeze(-1))).sum(-1))

    def full_loss(dist_logits, conf_raw, plog, beta, meta, weights):
        """All four losses on a batch.  `dist_logits` (B,A,3) and `conf_raw` (B,A)
        are the network's RAW head outputs; softmax/softplus are applied only to
        the gathered LEGAL entries (never the full 4674 actions).  Padded legal
        layout (B, K): entries beyond a sample's legal count are masked out.
        Returns (total, parts-dict-of-floats)."""
        FLOOR = ALPHA_FLOOR
        act = meta['pad_act']                       # (B,K) long
        mask = meta['pad_mask']                     # (B,K) bool
        maskf = mask.float()
        idx3 = act.unsqueeze(-1).expand(-1, -1, 3)
        p3 = _act_p3(dist_logits.gather(1, idx3))   # (B,K,3) softmax on legal only
        cf = _act_conf(conf_raw.gather(1, act))     # (B,K)   softplus on legal only
        pl = plog.gather(1, act)                    # (B,K)
        # Per-sample policy π = softmax over legal (masked).
        pl = pl.masked_fill(~mask, -1e9)
        pi = torch.softmax(pl, dim=1) * maskf       # (B,K)
        # Predicted per-action value Dirichlet (floored, pad kept finite).
        pv = (cf.unsqueeze(-1) * p3).clamp_min(FLOOR)
        pv = torch.where(mask.unsqueeze(-1), pv,
                         torch.full_like(pv, FLOOR))
        # (1) derived scalar value  v = Σ π_a (p_win − p_loss)   → MSE to z
        v_pred = (pi * (p3[..., _WIN] - p3[..., _LOSS])).sum(1)     # (B,)
        zw = meta['z_w']
        L_value = (zw * (meta['z'] - v_pred) ** 2).sum() / zw.sum().clamp_min(1.0)
        # (2) derived qv = moment-match(Σ π_a pv_a)  → KL to backed-up qv target
        a0 = pv.sum(-1)                                            # (B,K)
        meanc = pv / a0.unsqueeze(-1)                              # (B,K,3)
        piw = pi.unsqueeze(-1)                                     # (B,K,1)
        Mmean = (piw * meanc).sum(1)                               # (B,3)
        varc = meanc * (1 - meanc) / (a0.unsqueeze(-1) + 1.0)
        ex2 = (piw * (varc + meanc ** 2)).sum(1)                   # (B,3)
        varmix = (ex2 - Mmean ** 2).clamp_min(1e-9)
        beta0 = ((Mmean * (1 - Mmean)).sum(1)
                 / varmix.sum(1)).sub(1.0).clamp(2 * FLOOR, CONF_MAX)
        qv_pred = (Mmean * beta0.unsqueeze(-1)).clamp_min(FLOOR)   # (B,3)
        L_vd = _dir_kl_rows(meta['qv_t'], qv_pred).mean()
        # (3) per-action value KL over EVIDENCE edges only
        klav = _dir_kl_rows(meta['ev_alpha'].reshape(-1, 3), pv.reshape(-1, 3))
        evm = meta['ev_mask'].reshape(-1).float()
        L_av = (klav * evm).sum() / evm.sum().clamp_min(1.0)
        # (4) policy-prior KL:  KL( Dir(pcount+floor) ‖ Dir(beta·π) )  per sample
        t = (meta['pcount'] + FLOOR) * maskf                      # (B,K)
        b = (beta.unsqueeze(1) * pi) * maskf                      # (B,K), Σ_k=beta
        Tsum = t.sum(1); Bsum = b.sum(1)
        zeros = torch.zeros_like(t)
        lgt = torch.where(mask, _lg(t.clamp_min(FLOOR)), zeros).sum(1)
        lgb = torch.where(mask, _lg(b.clamp_min(FLOOR)), zeros).sum(1)
        dig = torch.where(mask, (t - b) * (_dg(t.clamp_min(FLOOR))
                          - _dg(Tsum).unsqueeze(1)), zeros).sum(1)
        L_pol = (_lg(Tsum) - lgt - _lg(Bsum) + lgb + dig).mean()
        wv, wvd, wav, wpol = weights
        total = wpol * L_pol + wav * L_av + wvd * L_vd + wv * L_value
        # Loss values + concentration (α0) diagnostics (predicted vs target total
        # pseudo-counts per Dirichlet head).  Stack every reported scalar and pull
        # it back in ONE .cpu() transfer — each separate float()/.item() is a full
        # pipeline sync on DirectML, so 11 of them per step (×TRAIN_STEPS_PER_EP)
        # was pure stall.  parts['loss'] is returned so the caller needn't sync
        # the loss tensor again.
        with torch.no_grad():
            evc = evm.sum().clamp_min(1.0)
            av_p = (pv.reshape(-1, 3).sum(-1) * evm).sum() / evc      # ≈ conf
            av_t = (meta['ev_alpha'].reshape(-1, 3).sum(-1) * evm).sum() / evc
            diag = torch.stack([total, L_pol, L_av, L_vd, L_value, av_p, av_t,
                                qv_pred.sum(1).mean(), meta['qv_t'].sum(1).mean(),
                                beta.mean(), Tsum.mean()]).to('cpu', copy=False).tolist()
        parts = dict(zip(('loss', 'pol', 'av', 'vd', 'val', 'cav_p', 'cav_t',
                          'cqv_p', 'cqv_t', 'cpol_p', 'cpol_t'), diag))
        return total, parts

    def build_batch_meta(batch, device):
        """Pad a list of sample dicts into fixed-(B,K) target tensors on device.
        K = max legal count in the batch."""
        B = len(batch)
        K = max(len(s['legal']) for s in batch)
        pad_act = np.zeros((B, K), np.int64)
        pad_mask = np.zeros((B, K), bool)
        pcount = np.zeros((B, K), np.float32)
        ev_mask = np.zeros((B, K), bool)
        ev_alpha = np.full((B, K, 3), ALPHA_FLOOR, np.float32)
        qv_t = np.empty((B, 3), np.float32)
        z = np.zeros(B, np.float32)
        z_w = np.zeros(B, np.float32)
        for i, s in enumerate(batch):
            k = len(s['legal'])
            pad_act[i, :k] = s['legal']
            pad_mask[i, :k] = True
            pcount[i, :k] = s['pcount']
            ev_mask[i, s['ev_idx']] = True
            ev_alpha[i, s['ev_idx']] = s['ev_alpha']
            qv_t[i] = s['qv']
            z[i] = s['z']
            z_w[i] = 0.0 if s.get('no_z') else 1.0
        t = lambda a: torch.from_numpy(a).to(device)
        return {'pad_act': t(pad_act), 'pad_mask': t(pad_mask),
                'pcount': t(pcount), 'ev_mask': t(ev_mask),
                'ev_alpha': t(ev_alpha), 'qv_t': t(qv_t),
                'z': t(z), 'z_w': t(z_w)}

    def _cpu_split_loss(probs, conf, plog, beta, batch, device, weights):
        """Fallback: gather + KL on CPU, seed dense grads back to the device graph
        (used on CUDA/CPU never; on DirectML only if the on-device loss can't run
        a device op like gather-backward).  Downloads the dense head outputs — the
        cost the on-device path exists to avoid."""
        meta = build_batch_meta(batch, 'cpu')
        pc = probs.detach().cpu().requires_grad_(True)
        cc = conf.detach().cpu().requires_grad_(True)
        lc = plog.detach().cpu().requires_grad_(True)
        bc = beta.detach().cpu().requires_grad_(True)
        loss_cpu, parts = full_loss(pc, cc, lc, bc, meta, weights)
        loss_cpu.backward()
        torch.autograd.backward([probs, conf, plog, beta],
                                [pc.grad.to(device), cc.grad.to(device),
                                 lc.grad.to(device), bc.grad.to(device)])
        return parts['loss'], parts

    def train_step(network, optimizer, batch, device, backend, weights,
                   grad_clip=1.0, model_lock=None):
        """One optimiser step, loss computed ON-DEVICE for every backend — on
        DirectML the lgamma/digamma use GPU-friendly approximations (set_backend),
        so nothing is offloaded to the CPU.  If a DML device op (e.g. gather
        backward) is unsupported, it falls back ONCE to the CPU-split path and
        stays there.  Returns (loss_float, parts)."""
        import contextlib
        global _DML_GPU_LOSS
        lock = model_lock or contextlib.nullcontext()
        x = batch_to_tensor([s['obs'] for s in batch], device)
        with lock:
            probs, conf, plog, beta = network(x)
            optimizer.zero_grad()
            if backend == 'directml' and not _DML_GPU_LOSS:
                lv, parts = _cpu_split_loss(probs, conf, plog, beta, batch,
                                            device, weights)
            else:
                try:
                    meta = build_batch_meta(batch, device)
                    loss, parts = full_loss(probs, conf, plog, beta, meta, weights)
                    loss.backward()
                    lv = parts['loss']
                except Exception as e:
                    if backend != 'directml':
                        raise
                    _DML_GPU_LOSS = False       # permanent fallback for this run
                    print(f'on-device loss unavailable ({type(e).__name__}: {e}) '
                          f'— using CPU-split loss')
                    optimizer.zero_grad()
                    lv, parts = _cpu_split_loss(probs, conf, plog, beta, batch,
                                                device, weights)
            torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()
        return lv, parts

    # ── DirectML-safe AdamW (aten::lerp has no DML kernel) ────────────────────
    class LerpFreeAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=1e-2):
            super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                          weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for g in self.param_groups:
                lr, (b1, b2) = g['lr'], g['betas']
                eps, wd = g['eps'], g['weight_decay']
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st['step'] = 0
                        st['m'] = torch.zeros_like(p)
                        st['v'] = torch.zeros_like(p)
                    m, v = st['m'], st['v']
                    st['step'] += 1
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st['step']
                    bc2 = 1 - b2 ** st['step']
                    denom = (v / bc2).sqrt_().add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)
            return loss

    # ── Evaluation bot: batched-leaf Thompson MCTS on a frozen net ────────────
    def nn_eval_states(network, device, states):
        """States → gathered-nothing dense outputs as numpy + fp16 obs.
        Returns (probs (B,A,3), conf (B,A), plog (B,A), beta (B,), obs16)."""
        obs16 = np.asarray([s.observation_tensor(s.current_player())
                            for s in states], dtype=np.float16)
        x = batch_to_tensor(obs16, device)
        with torch.inference_mode():
            dl, cr, plog, beta = network(x)
            probs = _act_p3(dl); conf = _act_conf(cr)      # activate for the tree
        return (probs.cpu().numpy(), conf.cpu().numpy(),
                plog.cpu().numpy(), beta.cpu().numpy(), obs16)

    class ThompsonMCTSBot:
        """mcts_search(state) → searched root _TNode.  Batched-leaf waves;
        Thompson sampling's own randomness (plus virtual loss) diversifies a
        wave.  Reads weights live from `network`."""

        def __init__(self, game, network, device, max_simulations,
                     batch_size=16, temp=1.0, proven_win_bonus=4.0,
                     conf_cap=CONF_MAX, random_state=None):
            self.game, self.network, self.device = game, network, device
            self.max_simulations = max_simulations
            self.batch_size = batch_size
            self.temp = temp
            self.pwb = proven_win_bonus
            self.conf_cap = conf_cap
            self._rng = random_state or np.random.default_rng()

        def _expand(self, state):
            pr, cf, pl, bt, ob = nn_eval_states(self.network, self.device,
                                                [state])
            leg = state.legal_actions()
            node = _TNode(state.current_player(), leg, pr[0][leg], cf[0][leg],
                          pl[0][leg], bt[0])
            node.obs = ob[0]
            return node

        def mcts_search(self, state):
            root = self._expand(state)
            sims = 0
            while sims < self.max_simulations:
                if _node_solved_outcome(root) is not None:
                    break
                wave = min(self.batch_size, self.max_simulations - sims)
                pending = []
                for _ in range(wave):
                    path, st, edge = _select_leaf(root, state, self._rng,
                                                  self.temp, self.pwb)
                    if st is None:
                        _backup_terminal(path, None, self.conf_cap, CONF_MAX,
                                         self.pwb)
                        sims += 1
                    else:
                        pending.append((path, st, edge))
                uniq = {}
                for path, st, (node, idx) in pending:
                    uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    entries = list(uniq.values())
                    pr, cf, pl, bt, ob = nn_eval_states(
                        self.network, self.device, [e[2] for e in entries])
                    for (node, idx, st), pr_i, cf_i, pl_i, bt_i, ob_i in zip(
                            entries, pr, cf, pl, bt, ob):
                        leg = st.legal_actions()
                        child = _TNode(st.current_player(), leg, pr_i[leg],
                                       cf_i[leg], pl_i[leg], bt_i)
                        child.obs = ob_i
                        node.children[idx] = child
                for path, st, edge in pending:
                    _backup(path)
                    sims += 1
            return root

    def mcts_move(bot, state, temp_greedy=True):
        root = bot.mcts_search(state)
        return root_pick(root, bot._rng, thompson=not temp_greedy)

    def policy_move(network, state, device, rng=None, sample=False):
        """Search-free move: argmax (or sample) of the policy-prior mean π."""
        pr, cf, pl, bt, _ = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        lg = pl[0][leg] - pl[0][leg].max()
        pi = np.exp(lg); pi /= pi.sum()
        if sample and rng is not None:
            return int(leg[rng.choice(len(leg), p=pi)])
        return int(leg[int(pi.argmax())])

    def thompson_value_move(network, state, device, rng, temp=1.0):
        """Search-free move via ONE Thompson sample of the NN's per-action value
        beliefs: draw v=p_win−p_loss from each legal action's Dir(conf·probs) and
        play the argmax.  No tree — this is exactly the in-tree selection rule
        applied directly to the raw network output (the cheap quick-eval mover)."""
        pr, cf, pl, bt, _ = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        a = np.maximum(cf[0][leg, None] * pr[0][leg], ALPHA_FLOOR) * temp
        g = rng.standard_gamma(a)
        x = g / g.sum(1, keepdims=True)
        v = x[:, _WIN] - x[:, _LOSS]
        return int(leg[int(v.argmax())])

    def quick_match(net_a, net_b, game, n_games, device, rng=None,
                    opening_plies=2, max_plies=200, temp=1.0):
        """Search-free Thompson-value match, net_a vs net_b, alternating colours.
        net == None → uniform-random mover.  Returns (wins_a, draws, wins_b).
        A lightweight progress pulse (no MCTS) — used by the quick eval."""
        rng = rng or np.random.default_rng()
        wa = d = wb = 0
        for i in range(n_games):
            a_side = i % 2
            state = game.new_initial_state()
            for _ in range(opening_plies):
                if state.is_terminal():
                    break
                leg = state.legal_actions()
                state.apply_action(int(leg[rng.integers(len(leg))]))
            ply = 0
            while not state.is_terminal() and ply < max_plies:
                net = net_a if state.current_player() == a_side else net_b
                if net is None:
                    leg = state.legal_actions()
                    mv = int(leg[rng.integers(len(leg))])
                else:
                    mv = thompson_value_move(net, state, device, rng, temp)
                state.apply_action(mv); ply += 1
            if not state.is_terminal():
                d += 1; continue
            r = state.returns()[a_side]
            wa += r > 0; wb += r < 0; d += r == 0
        return int(wa), int(d), int(wb)

    # ══════════════════════════════════════════════════════════════════════════
    #  Parallel self-play (single process, batched across games)
    # ══════════════════════════════════════════════════════════════════════════
    class ThompsonParallelSelfPlay:
        """Runs `n_parallel` games at once, sharing ONE NN forward pass per leaf
        wave across all games (the dominant self-play speedup).  Weights are read
        live, so in-flight games pick up training updates.  Emits finished
        episodes as lists of target dicts (per-move root targets + solver-labelled
        aux samples)."""

        def __init__(self, game, network, device, n_parallel=8, wave_per_game=8,
                     fast_sims=250, full_sims=1000, fast_prob=0.75,
                     temp_threshold=20, late_temp=8.0, conf_cap=CONF_MAX,
                     proven_win_bonus=4.0, max_plies=400, z_mix=0.5, z_gamma=0.97,
                     pool_prob=0.0, checkpoint_dir=None, channels=None,
                     num_blocks=None, restart_prob=0.0, restart_k_min=2,
                     restart_k_max=30, restart_pool_cap=128, random_pool_frac=0.5,
                     curriculum=False, curr_conf_thresh=8.0, curr_backup=3,
                     seed=None):
            self.game, self.network, self.device = game, network, device
            self.curriculum = curriculum
            self.curr_conf_thresh = float(curr_conf_thresh)   # α0 handoff gate
            self.curr_backup = int(curr_backup)               # plies to back up
            self.n_parallel, self.wave = n_parallel, wave_per_game
            self.fast_sims, self.full_sims = fast_sims, full_sims
            self.fast_prob = fast_prob
            self.temp_threshold, self.late_temp = temp_threshold, late_temp
            self.conf_cap, self.pwb = conf_cap, proven_win_bonus
            self.max_plies = max_plies
            self.z_mix, self.z_gamma = z_mix, z_gamma
            self.pool_prob = pool_prob
            self.checkpoint_dir = checkpoint_dir
            self.channels, self.num_blocks = channels, num_blocks
            self.restart_prob = restart_prob
            self.restart_k_min, self.restart_k_max = restart_k_min, restart_k_max
            self.restart_pool_cap = restart_pool_cap
            self.random_pool_frac = random_pool_frac
            self._rng = np.random.default_rng(seed)
            self._restart_pool = []
            self._pool_nets = {}
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0, 'open': 0}
            self.fwd_calls = 0     # NN forward passes + rows served — the loop
            self.fwd_rows = 0      # diffs these to report avg GPU batch size
            self.slots = [self._new_game() for _ in range(n_parallel)]

        def _push_seed(self, seq):
            if len(seq) >= 2:
                self._restart_pool.append(list(seq))
                if len(self._restart_pool) > self.restart_pool_cap:
                    del self._restart_pool[0]

        def _load_pool_net(self, label):
            net = self._pool_nets.get(label)
            if net is None:
                path = os.path.join(self.checkpoint_dir, f'bench_{label}.pt')
                net = ThompsonFullNet(self.channels, self.num_blocks)
                net.load_state_dict(torch.load(path, map_location='cpu',
                                               weights_only=True))
                net.eval()
                self._pool_nets[label] = net
            return net

        def _handoff(self, i):
            """Curriculum: the cheap Thompson opening reached a confident/terminal
            node — back up curr_backup plies and start full MCTS self-play there.
            The opening moves are the (untrained) prefix; MCTS covers the rest."""
            s = self.slots[i]
            resume = max(0, len(s['open_actions']) - self.curr_backup)
            state = self.game.new_initial_state()
            for a in s['open_actions'][:resume]:
                state.apply_action(a)
            s['state'] = state; s['actions'] = list(s['open_actions'][:resume])
            s['resume'] = resume; s['move'] = resume
            s['phase'] = 'mcts'; s['root'] = None; s['n'] = 0

        def _new_game(self):
            rng = self._rng
            sims = (self.fast_sims if rng.random() < self.fast_prob
                    else self.full_sims)
            state, actions, resume, phase = self.game.new_initial_state(), [], 0, 'mcts'
            if self.curriculum:
                phase = 'open'                     # cheap Thompson opening first
            elif self._restart_pool and rng.random() < self.restart_prob:
                seq = self._restart_pool[rng.integers(len(self._restart_pool))]
                pref = restart_prefix(seq, rng, self.restart_k_min,
                                      self.restart_k_max)
                st = self.game.new_initial_state(); ok = True
                for a in pref:
                    if st.is_terminal() or a not in st.legal_actions():
                        ok = False; break
                    st.apply_action(int(a))
                if ok and not st.is_terminal():
                    state, actions = st, list(pref)
            slot = {'state': state, 'hist': [], 'aux': [], 'actions': actions,
                    'move': resume, 'resume': resume, 'sims': sims, 'phase': phase,
                    'open_actions': [], 'root': None, 'n': 0, 'pool': None}
            if (self.pool_prob > 0 and self.checkpoint_dir
                    and rng.random() < self.pool_prob):
                try:
                    labels = [f[6:-3] for f in os.listdir(self.checkpoint_dir)
                              if f.startswith('bench_') and f.endswith('.pt')]
                except OSError:
                    labels = []
                if not labels or rng.random() < self.random_pool_frac:
                    slot['pool'] = {'label': 'random',
                                    'side': int(rng.integers(2)), 'net': None}
                else:
                    lb = labels[rng.integers(len(labels))]
                    slot['pool'] = {'label': lb, 'side': int(rng.integers(2)),
                                    'net': self._load_pool_net(lb)}
            return slot

        def _temp(self, move):
            return 1.0 if move < self.temp_threshold else self.late_temp

        def _finish(self, i):
            s = self.slots[i]; st = s['state']
            if st.is_terminal():
                ret = st.returns()
                z_mix_episode(s['hist'], ret, self.z_mix, self.z_gamma)
                result = 'draw' if ret[0] == 0.0 else 'decisive'
                if self.restart_prob > 0 and result == 'decisive':
                    self._push_seed(s['actions'])
            else:
                strip_episode_meta(s['hist'])
                result = 'cutoff'
            self.last_aux = len(s['aux'])
            self.stats['games'] += 1
            self.stats['plies'] += int(s['move'])
            self.stats['open'] += int(s['resume'])   # opening length (frontier)
            if result == 'draw':   self.stats['draw'] += 1
            if result == 'cutoff': self.stats['cutoff'] += 1
            data = s['hist'] + s['aux']
            self.slots[i] = self._new_game()
            return data

        def _resolve_pool_moves(self):
            done = []
            for i, s in enumerate(self.slots):
                pool, state = s['pool'], s['state']
                if pool is None or state.current_player() != pool['side']:
                    continue
                if pool['label'] == 'random':
                    leg = state.legal_actions()
                    a = int(leg[self._rng.integers(len(leg))])
                else:
                    a = policy_move(pool['net'], state, 'cpu')
                s['root'] = _descend(s['root'], a)
                state.apply_action(a); s['actions'].append(a); s['move'] += 1
                if state.is_terminal() or s['move'] >= self.max_plies:
                    done.append(self._finish(i))
            return done

        def _play_move(self, i):
            s = self.slots[i]; root = s['root']; mm = s['move'] - s['resume']
            s['hist'].append(make_target(root, self.conf_cap))
            a = root_pick(root, self._rng, thompson=(mm < self.temp_threshold),
                          temp=self._temp(mm))
            pidx = int(np.nonzero(root.legal == a)[0][0])
            s['actions'].append(int(a))
            s['root'] = root.children[pidx]
            s['state'].apply_action(a)
            s['move'] += 1; s['n'] = 0

        def _step(self):
            rng = self._rng
            done = self._resolve_pool_moves()
            pending, evals, seen = [], [], set()
            for i, s in enumerate(self.slots):
                pool = s['pool']
                if pool is not None and s['state'].current_player() == pool['side']:
                    continue
                if s['phase'] == 'open':               # cheap Thompson opening
                    if s['state'].is_terminal():
                        self._handoff(i)
                    else:
                        evals.append(('open', i, None, s['state']))
                    continue
                if s['root'] is None:
                    evals.append(('root', i, None, s['state'])); continue
                if _node_solved_outcome(s['root']) is not None:
                    continue
                wave = min(self.wave, s['sims'] - s['n'])
                for _ in range(max(wave, 0)):
                    path, st, edge = _select_leaf(s['root'], s['state'], rng,
                                                  self._temp(s['move'] - s['resume']),
                                                  self.pwb)
                    if st is None:
                        _backup_terminal(path, s['aux'], self.conf_cap, CONF_MAX,
                                         self.pwb)
                        s['n'] += 1
                    else:
                        node, idx = edge
                        pending.append((i, path, edge))
                        if (id(node), idx) not in seen:
                            seen.add((id(node), idx))
                            evals.append(('leaf', node, idx, st))
            if evals:
                self.fwd_calls += 1; self.fwd_rows += len(evals)
                pr, cf, pl, bt, ob = nn_eval_states(self.network, self.device,
                                                    [e[3] for e in evals])
                for (kind, a, b, st), pr_i, cf_i, pl_i, bt_i, ob_i in zip(
                        evals, pr, cf, pl, bt, ob):
                    leg = st.legal_actions()
                    if kind == 'open':             # opening move + confidence gate
                        loc, confident = opening_decision(
                            pr_i[leg], cf_i[leg], pl_i[leg], rng,
                            self.curr_conf_thresh)
                        s = self.slots[a]
                        if confident:
                            self._handoff(a)
                        else:
                            s['open_actions'].append(int(leg[loc]))
                            s['state'].apply_action(int(leg[loc])); s['move'] += 1
                            if (s['state'].is_terminal()
                                    or s['move'] >= self.max_plies):
                                self._handoff(a)
                        continue
                    node = _TNode(st.current_player(), leg, pr_i[leg], cf_i[leg],
                                  pl_i[leg], bt_i)
                    node.obs = ob_i
                    if kind == 'root':
                        self.slots[a]['root'] = node
                    else:
                        a.children[b] = node
            for i, path, _edge in pending:
                _backup(path); self.slots[i]['n'] += 1
            for i, s in enumerate(self.slots):
                if s['root'] is None:
                    continue
                if (s['n'] < s['sims']
                        and _node_solved_outcome(s['root']) is None):
                    continue
                self._play_move(i)
                if s['state'].is_terminal() or s['move'] >= self.max_plies:
                    done.append(self._finish(i))
            return done

        def episodes(self):
            while True:
                for data in self._step():
                    yield data

    # ══════════════════════════════════════════════════════════════════════════
    #  Multiprocess self-play pool: worker processes + central GPU inference server
    # ══════════════════════════════════════════════════════════════════════════
    import threading as _threading, queue as _queue, time as _time
    import multiprocessing as _mp

    def _probe_gather(device):
        """Does index_select run on this device (forward)?  Lets the server
        gather the ~1% legal entries ON-device so only those cross the bus."""
        if str(device) == 'cpu':
            return True
        try:
            x = torch.arange(12.0, device=device).reshape(4, 3)
            idx = torch.tensor([2, 0, 3], device=device)
            y = x.index_select(0, idx).cpu()
            return bool(torch.equal(y, torch.tensor([[6., 7., 8.], [0., 1., 2.],
                                                     [9., 10., 11.]])))
        except Exception:
            return False

    class MPSelfPlayPool:
        """n_workers CPU processes run the trees; an inference-server thread here
        batches ALL their NN requests into one forward pass on `device`.  Exposes
        the same .episodes()/.stats/.last_aux interface as ThompsonParallelSelfPlay,
        so the training loop is identical.  `lock` serialises model access between
        the server thread and training (DirectML is not thread-safe)."""

        def __init__(self, network, device, n_workers, cfg, batch_window_s=0.002,
                     checkpoint_dir=None, channels=None, num_blocks=None,
                     max_batch_rows=1024):
            self.network, self.device = network, device
            self.checkpoint_dir = checkpoint_dir
            self.channels, self.num_blocks = channels, num_blocks
            self._pool_nets = {}
            self.lock = _threading.Lock()
            self._stop = _threading.Event()
            self.window, self.max_batch_rows = batch_window_s, max_batch_rows
            self._gather_ok = _probe_gather(device)
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0, 'open': 0}
            self.fwd_calls = 0     # NN forward passes + rows served — the loop
            self.fwd_rows = 0      # diffs these to report avg GPU batch size
            ctx = _mp.get_context('spawn')
            self.req_q = ctx.Queue()
            self.episode_q = ctx.Queue(maxsize=64)
            self.resp_qs = [ctx.Queue() for _ in range(n_workers)]
            self.pool_resp_qs = [ctx.Queue() for _ in range(n_workers)]
            # Initialise the autograd engine's device state from the MAIN thread
            # before any other thread touches the device (DirectML assert).
            if str(device) != 'cpu':
                _t = torch.zeros(4, device=device, requires_grad=True)
                (_t * 2.0).sum().backward()
            self.procs = [ctx.Process(target=mp_worker,
                          args=(i, self.req_q, self.resp_qs[i],
                                self.pool_resp_qs[i], self.episode_q, cfg),
                          daemon=True) for i in range(n_workers)]
            for p in self.procs:
                p.start()
            self.server = _threading.Thread(target=self._serve, daemon=True)
            self.server.start()

        def _get_net(self, net_id):
            if net_id == 'live':
                return self.network, self.device, True
            net = self._pool_nets.get(net_id)
            if net is None:
                net = ThompsonFullNet(self.channels, self.num_blocks)
                net.load_state_dict(torch.load(
                    os.path.join(self.checkpoint_dir, f'bench_{net_id}.pt'),
                    map_location='cpu', weights_only=True))
                net.eval(); self._pool_nets[net_id] = net
            return net, 'cpu', False

        def _forward_gathered(self, net, dev, xin, flat, rowsel):
            # net returns RAW dist logits + RAW conf; gather the ~1% legal entries
            # first, then softmax/softplus only those (never the dense 4674).
            x = torch.from_numpy(xin).to(dev)
            dl, cr, plog, beta = net(x)
            if self._gather_ok and str(dev) != 'cpu':
                ft = torch.from_numpy(flat).to(dev)
                rs = torch.from_numpy(rowsel).to(dev)
                p = _act_p3(dl.reshape(-1, 3).index_select(0, ft)).cpu().numpy()
                c = _act_conf(cr.reshape(-1).index_select(0, ft)).cpu().numpy()
                pl = plog.reshape(-1).index_select(0, ft).cpu().numpy()
                bt = beta.index_select(0, rs).cpu().numpy()
            else:
                p = _act_p3(dl.reshape(-1, 3)).cpu().numpy()[flat]
                c = _act_conf(cr.reshape(-1)).cpu().numpy()[flat]
                pl = plog.reshape(-1).cpu().numpy()[flat]
                bt = beta.cpu().numpy()[rowsel]
            return p, c, pl, bt

        def _serve(self):
            A = _NUM_ACTIONS
            while not self._stop.is_set():
                try:
                    reqs = [self.req_q.get(timeout=0.1)]
                except _queue.Empty:
                    continue
                rows = reqs[0][2].shape[0]
                deadline = _time.monotonic() + self.window
                while _time.monotonic() < deadline and rows < self.max_batch_rows:
                    try:
                        r = self.req_q.get_nowait(); reqs.append(r)
                        rows += r[2].shape[0]
                    except _queue.Empty:
                        _time.sleep(0.0003)
                groups = {}
                for wid, net_id, obs, legals in reqs:
                    groups.setdefault(net_id, []).append((wid, obs, legals))
                for net_id, group in groups.items():
                    net, dev, needs_lock = self._get_net(net_id)
                    obs = np.concatenate([o for _, o, _ in group], axis=0)
                    xin = obs.reshape(-1, *_OBS_SHAPE).astype(np.float32)
                    if net_id == 'live':
                        self.fwd_calls += 1; self.fwd_rows += xin.shape[0]
                    row_legals = [l for _, _, ls in group for l in ls]
                    flat = np.concatenate([l.astype(np.int64) + r * A
                                           for r, l in enumerate(row_legals)])
                    rowsel = np.concatenate([np.full(len(l), r, dtype=np.int64)
                                             for r, l in enumerate(row_legals)])
                    # rowsel maps each gathered entry to its row; beta needs the
                    # per-ROW index, so take unique row ids in order.
                    beta_rows = np.arange(len(row_legals), dtype=np.int64)
                    offs = np.zeros(len(row_legals) + 1, dtype=np.int64)
                    np.cumsum([len(l) for l in row_legals], out=offs[1:])
                    if needs_lock:
                        with self.lock, torch.no_grad():
                            p, c, pl, bt = self._forward_gathered(
                                net, dev, xin, flat, beta_rows)
                    else:
                        with torch.no_grad():
                            p, c, pl, bt = self._forward_gathered(
                                net, dev, xin, flat, beta_rows)
                    tqs = self.resp_qs if net_id == 'live' else self.pool_resp_qs
                    ri = 0
                    for wid, o, ls in group:
                        out = []
                        for _ in ls:
                            a, b = offs[ri], offs[ri + 1]
                            out.append((p[a:b], c[a:b], pl[a:b], float(bt[ri])))
                            ri += 1
                        tqs[wid].put(out)

        def episodes(self):
            while True:
                samples, n_aux, result, plies, resume = self.episode_q.get()
                self.last_aux = n_aux
                self.stats['games'] += 1
                self.stats['plies'] += plies
                self.stats['open'] += resume        # opening length (frontier)
                if result == 'draw':   self.stats['draw'] += 1
                if result == 'cutoff': self.stats['cutoff'] += 1
                yield samples

        def shutdown(self):
            self._stop.set()
            try:
                self.server.join(timeout=2.0)
            except Exception:
                pass
            for p in self.procs:
                p.terminate()
            for p in self.procs:
                p.join(timeout=2.0)
            for q in ([self.req_q, self.episode_q] + self.resp_qs
                      + self.pool_resp_qs):
                try:
                    q.close(); q.cancel_join_thread()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Sparse deep eval — running-Elo pool of checkpoints@MCTS-128 + random
    # ══════════════════════════════════════════════════════════════════════════
    # Every checkpoint enters a SINGLE Elo table, each rated at MCTS=`eval_sims`
    # (default 128), alongside a `random` mover.  Per new checkpoint the cost is
    # FIXED (independent of pool size): it plays `games_per_pair` games against
    # each of the last-3 checkpoints + random, then `refresh_pairs` random pairs
    # from the whole pool play too (keeps old ratings mixing).  Elo K decays with
    # the number of games a pair has already played, so ratings settle.
    class EloPool:
        def __init__(self, game, device, eval_sims=128, k_base=32.0,
                     k_halflife=30.0, games_per_pair=2, last_n=3,
                     refresh_pairs=10, opening_plies=4, batch_size=16,
                     start_elo=1000.0, eval_temp=6.0, max_eval_plies=200,
                     seed=0):
            self.game, self.device = game, device
            self.eval_sims = eval_sims
            self.eval_temp = eval_temp
            self.k_base, self.k_hl = k_base, k_halflife
            self.games_per_pair = games_per_pair
            self.last_n, self.refresh_pairs = last_n, refresh_pairs
            self.opening_plies = opening_plies
            self.max_eval_plies = max_eval_plies      # cap: uncapped games vs a
                                                      # weak/random net run forever
            self.batch_size = batch_size
            self.start_elo = start_elo
            self.rng = np.random.default_rng(seed)
            self.players = ['random']
            self.nets = {'random': None}
            self.elo = {'random': start_elo}
            self.order = []                      # checkpoint labels, in add order
            self.pair_games = {}                 # frozenset({a,b}) -> games count

        def _bot(self, label):
            return ThompsonMCTSBot(self.game, self.nets[label], self.device,
                                   self.eval_sims, batch_size=self.batch_size,
                                   temp=self.eval_temp, random_state=self.rng)

        def _move(self, label, bot_cache, state):
            if self.nets[label] is None:
                leg = state.legal_actions()
                return int(leg[self.rng.integers(len(leg))])
            bot = bot_cache.setdefault(label, self._bot(label))
            root = bot.mcts_search(state)
            return root_pick(root, self.rng, thompson=False)

        def _play(self, a, b, bot_cache):
            """One game: player `a` is White, `b` is Black.  A few random opening
            plies add variety.  Returns White's result in {1,0.5,0}."""
            state = self.game.new_initial_state()
            for _ in range(self.opening_plies):
                if state.is_terminal():
                    break
                leg = state.legal_actions()
                state.apply_action(int(leg[self.rng.integers(len(leg))]))
            ply = 0
            while not state.is_terminal() and ply < self.max_eval_plies:
                lab = a if state.current_player() == 0 else b
                state.apply_action(self._move(lab, bot_cache, state))
                ply += 1
            if state.is_terminal():
                r = state.returns()[0]
                return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
            return 0.5                             # adjudicate a capped game a draw

        def _update(self, a, b, sa):
            key = frozenset((a, b))
            n = self.pair_games.get(key, 0)
            k = self.k_base * self.k_hl / (self.k_hl + n)     # decayed K
            ea = 1.0 / (1.0 + 10 ** ((self.elo[b] - self.elo[a]) / 400.0))
            self.elo[a] += k * (sa - ea)
            self.elo[b] += k * ((1.0 - sa) - (1.0 - ea))
            self.pair_games[key] = n + 1

        def _match(self, a, b, bot_cache):
            for g in range(self.games_per_pair):
                w, x = (a, b) if g % 2 == 0 else (b, a)       # alternate colours
                s_white = self._play(w, x, bot_cache)
                sa = s_white if w == a else 1.0 - s_white
                self._update(a, b, sa)

        def add_checkpoint(self, label, net):
            """Register a new checkpoint (rated at MCTS-`eval_sims`), warm-start
            its Elo from the previous checkpoint, then run its fixed-cost eval."""
            self.nets[label] = net
            self.elo.setdefault(label, self.elo[self.order[-1]] if self.order
                                else self.start_elo)
            self.players.append(label)
            bot_cache = {}
            opponents = self.order[-self.last_n:] + ['random']
            for opp in opponents:
                self._match(label, opp, bot_cache)
            # Refresh random pairs across the whole pool — only once the pool is
            # big enough that these are genuinely NEW pairings (not the same two
            # players replayed).  With <= last_n+2 players every pair was already
            # played above, so refresh would just burn eval time (the iter-0
            # blocker: 10 refresh matches of the ONLY pair, '0' vs random).
            existing = self.order + ['random']       # players before this add
            n_new_pairs = len(existing) * (len(existing) - 1) // 2 - \
                len(opponents)
            if n_new_pairs > 0:
                seen = set()
                for _ in range(min(self.refresh_pairs, n_new_pairs)):
                    for _try in range(20):
                        a, b = self.rng.choice(len(existing), 2, replace=False)
                        key = frozenset((existing[a], existing[b]))
                        if key not in seen and label not in key:
                            seen.add(key); break
                    else:
                        break
                    self._match(existing[a], existing[b], bot_cache)
            self.order.append(label)
            return dict(self.elo)

    # ══════════════════════════════════════════════════════════════════════════
    #  Checkpointing
    # ══════════════════════════════════════════════════════════════════════════
    def _cpu_sd(net):
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    def cpu_clone(net, channels, num_blocks):
        c = ThompsonFullNet(channels, num_blocks)
        c.load_state_dict(_cpu_sd(net)); c.eval()
        return c

    def save_benchmark_net(checkpoint_dir, label, net):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(_cpu_sd(net), os.path.join(checkpoint_dir, f'bench_{label}.pt'))

    def save_checkpoint(checkpoint_dir, ep, base_network, optimizer, scheduler,
                        elo_pool, hist, replay_buffer=None, keep_buffer=0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        blob = {'ep': ep, 'model': _cpu_sd(base_network),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict() if scheduler else None,
                'elo': elo_pool.elo, 'order': elo_pool.order,
                'pair_games': {tuple(sorted(k)): v
                               for k, v in elo_pool.pair_games.items()},
                'hist': hist}
        if keep_buffer and replay_buffer:
            blob['buffer'] = replay_buffer[-keep_buffer:]
        tmp = os.path.join(checkpoint_dir, 'latest.pt.tmp')
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(checkpoint_dir, 'latest.pt'))

    def load_checkpoint(checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'latest.pt')
        if not os.path.exists(path):
            return None
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            return torch.load(path, map_location='cpu')
