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
    """
    __slots__ = ('player', 'legal', 'alpha', 'pprior', 'pcount', 'term',
                 'vloss', 'children', 'obs')

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


def _pi_post(node):
    """Posterior policy weights pi_a = (pprior + pcount) / total  — the current
    belief over which move is optimal (NN prior updated by search observations).
    """
    w = node.pprior + node.pcount
    return w / w.sum()


def node_qv(node, conf_max=CONF_MAX):
    """Position value-distribution: qv = moment-match( sum_a pi_a * pv_a ),
    the policy-weighted mixture of the edge beliefs (mover's perspective)."""
    if len(node.legal) == 1:
        return node.alpha[0].copy()
    return moment_match_mixture(_pi_post(node), node.alpha, conf_max)


def _set_term(node, idx, outcome, proven_win_bonus=0.0):
    """Mark edge idx proven; bake in the spike belief.  A proven WIN also gets
    extra policy-prior pseudocounts so pi_a snaps onto the mating move (teaches
    the prior the max at the end of the game fast — the MCTS-Solver accelerates
    the *value*; this accelerates the *policy*)."""
    node.term[idx]  = outcome
    node.alpha[idx] = _SPIKE[outcome]
    if outcome == _WIN and proven_win_bonus:
        node.pcount[idx] += proven_win_bonus


def _sample_edge_values(node, rng, temp=1.0):
    """One Thompson draw of each edge's value v=p_win-p_loss from Dir(temp*alpha)
    (temp>1 sharpens toward the mean → argmax≈max; temp=1 samples as-is), minus
    a virtual-loss penalty on in-flight edges for within-wave diversification."""
    a = node.alpha * temp
    g = rng.standard_gamma(a)                       # (k,3) Gamma(a_i,1)
    x = g / g.sum(1, keepdims=True)                 # ~ Dir(temp*alpha)
    v = x[:, _WIN] - x[:, _LOSS]
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
        node.pcount[idx] += 1.0
        if node.term[idx] < 0 and node.children[idx] is not None:
            node.alpha[idx] = flip_alpha(node_qv(node.children[idx], conf_max))


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
    k = rng.randint(k_min, min(k_max, len(seq)) + 1)
    return list(seq[:len(seq) - k])


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
            h = self.head(self.body(self.stem(x)))
            probs = F.softmax(self.dist_out(h).view(-1, _NUM_ACTIONS, 3), dim=-1)
            conf = torch.clamp(CONF_MIN + _softplus(self.conf_out(h)),
                               max=CONF_MAX)
            plog = self.plog_out(h)
            beta = torch.clamp(CONF_MIN + _softplus(self.beta_out(h)).squeeze(-1),
                               max=POLICY_CONC_MAX)
            return probs, conf, plog, beta

    # ── Closed-form Dirichlet KL (batched, differentiable) ────────────────────
    def _dir_kl_rows(a, b):
        """forward KL( Dir(a) ‖ Dir(b) ) per row, a/b: (N,3)."""
        a0, b0 = a.sum(-1), b.sum(-1)
        return (torch.lgamma(a0) - torch.lgamma(a).sum(-1)
                - torch.lgamma(b0) + torch.lgamma(b).sum(-1)
                + ((a - b) * (torch.digamma(a)
                              - torch.digamma(a0).unsqueeze(-1))).sum(-1))

    def full_loss(probs, conf, plog, beta, meta, weights):
        """All four losses on a batch, from the network's dense head outputs and
        the padded per-sample targets in `meta` (a dict of tensors on the same
        device).  Padded legal layout (B, K): entries beyond a sample's legal
        count are masked out.  Returns (total, parts-dict-of-floats)."""
        FLOOR = ALPHA_FLOOR
        act = meta['pad_act']                       # (B,K) long
        mask = meta['pad_mask']                     # (B,K) bool
        maskf = mask.float()
        idx3 = act.unsqueeze(-1).expand(-1, -1, 3)
        p3 = probs.gather(1, idx3)                  # (B,K,3)
        cf = conf.gather(1, act)                    # (B,K)
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
        lgt = torch.where(mask, torch.lgamma(t.clamp_min(FLOOR)), zeros).sum(1)
        lgb = torch.where(mask, torch.lgamma(b.clamp_min(FLOOR)), zeros).sum(1)
        dig = torch.where(mask, (t - b) * (torch.digamma(t.clamp_min(FLOOR))
                          - torch.digamma(Tsum).unsqueeze(1)), zeros).sum(1)
        L_pol = (torch.lgamma(Tsum) - lgt - torch.lgamma(Bsum) + lgb + dig).mean()
        wv, wvd, wav, wpol = weights
        total = wpol * L_pol + wav * L_av + wvd * L_vd + wv * L_value
        parts = {'pol': float(L_pol.detach()), 'av': float(L_av.detach()),
                 'vd': float(L_vd.detach()), 'val': float(L_value.detach())}
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

    def train_step(network, optimizer, batch, device, backend, weights,
                   grad_clip=1.0, model_lock=None):
        """One optimiser step.  On DirectML the lgamma/digamma-bearing loss is
        computed on CPU and its gradients are seeded back into the device graph
        (no exotic op ever runs on the DML device); elsewhere it runs on-device.
        Returns (loss_float, parts)."""
        import contextlib
        lock = model_lock or contextlib.nullcontext()
        x = batch_to_tensor([s['obs'] for s in batch], device)
        with lock:
            probs, conf, plog, beta = network(x)
            optimizer.zero_grad()
            if backend == 'directml':
                meta = build_batch_meta(batch, 'cpu')
                pc = probs.detach().cpu().requires_grad_(True)
                cc = conf.detach().cpu().requires_grad_(True)
                lc = plog.detach().cpu().requires_grad_(True)
                bc = beta.detach().cpu().requires_grad_(True)
                loss_cpu, parts = full_loss(pc, cc, lc, bc, meta, weights)
                loss_cpu.backward()
                torch.autograd.backward(
                    [probs, conf, plog, beta],
                    [pc.grad.to(device), cc.grad.to(device),
                     lc.grad.to(device), bc.grad.to(device)])
                lv = float(loss_cpu)
            else:
                meta = build_batch_meta(batch, device)
                loss, parts = full_loss(probs, conf, plog, beta, meta, weights)
                loss.backward()
                lv = float(loss)
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
        with torch.no_grad():
            probs, conf, plog, beta = network(x)
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
            self._rng = random_state or np.random.RandomState()

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
                     seed=None):
            self.game, self.network, self.device = game, network, device
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
            self._rng = np.random.RandomState(seed)
            self._restart_pool = []
            self._pool_nets = {}
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'cutoff': 0, 'plies': 0}
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

        def _new_game(self):
            rng = self._rng
            sims = (self.fast_sims if rng.rand() < self.fast_prob
                    else self.full_sims)
            state, actions = self.game.new_initial_state(), []
            if self._restart_pool and rng.rand() < self.restart_prob:
                seq = self._restart_pool[rng.randint(len(self._restart_pool))]
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
                    'move': 0, 'sims': sims, 'root': None, 'n': 0, 'pool': None}
            if (self.pool_prob > 0 and self.checkpoint_dir
                    and rng.rand() < self.pool_prob):
                try:
                    labels = [f[6:-3] for f in os.listdir(self.checkpoint_dir)
                              if f.startswith('bench_') and f.endswith('.pt')]
                except OSError:
                    labels = []
                if not labels or rng.rand() < self.random_pool_frac:
                    slot['pool'] = {'label': 'random',
                                    'side': int(rng.randint(2)), 'net': None}
                else:
                    lb = labels[rng.randint(len(labels))]
                    slot['pool'] = {'label': lb, 'side': int(rng.randint(2)),
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
                    a = int(leg[self._rng.randint(len(leg))])
                else:
                    a = policy_move(pool['net'], state, 'cpu')
                s['root'] = _descend(s['root'], a)
                state.apply_action(a); s['actions'].append(a); s['move'] += 1
                if state.is_terminal() or s['move'] >= self.max_plies:
                    done.append(self._finish(i))
            return done

        def _play_move(self, i):
            s = self.slots[i]; root = s['root']
            s['hist'].append(make_target(root, self.conf_cap))
            a = root_pick(root, self._rng, thompson=(s['move'] < self.temp_threshold),
                          temp=self._temp(s['move']))
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
                if s['root'] is None:
                    evals.append(('root', i, None, s['state'])); continue
                if _node_solved_outcome(s['root']) is not None:
                    continue
                wave = min(self.wave, s['sims'] - s['n'])
                for _ in range(max(wave, 0)):
                    path, st, edge = _select_leaf(s['root'], s['state'], rng,
                                                  self._temp(s['move']), self.pwb)
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
                pr, cf, pl, bt, ob = nn_eval_states(self.network, self.device,
                                                    [e[3] for e in evals])
                for (kind, a, b, st), pr_i, cf_i, pl_i, bt_i, ob_i in zip(
                        evals, pr, cf, pl, bt, ob):
                    leg = st.legal_actions()
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
                     k_halflife=30.0, games_per_pair=6, last_n=3,
                     refresh_pairs=10, opening_plies=4, batch_size=16,
                     start_elo=1000.0, eval_temp=6.0, seed=0):
            self.game, self.device = game, device
            self.eval_sims = eval_sims
            self.eval_temp = eval_temp
            self.k_base, self.k_hl = k_base, k_halflife
            self.games_per_pair = games_per_pair
            self.last_n, self.refresh_pairs = last_n, refresh_pairs
            self.opening_plies = opening_plies
            self.batch_size = batch_size
            self.start_elo = start_elo
            self.rng = np.random.RandomState(seed)
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
                return int(leg[self.rng.randint(len(leg))])
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
                state.apply_action(int(leg[self.rng.randint(len(leg))]))
            while not state.is_terminal():
                lab = a if state.current_player() == 0 else b
                state.apply_action(self._move(lab, bot_cache, state))
            r = state.returns()[0]
            return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)

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
            # Refresh random pairs across the whole pool.
            if len(self.players) >= 2:
                for _ in range(self.refresh_pairs):
                    a, b = self.rng.choice(len(self.players), 2, replace=False)
                    self._match(self.players[a], self.players[b], bot_cache)
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
