"""Gaussian value distributions over the final score.

ThompsonZero represents each action as a Dirichlet over (win, draw, loss).  This
engine represents it as a Gaussian over the FINAL SCORE DIFFERENTIAL instead —
how many more discs than the opponent you finish with, normalised to [-1, 1].

Why the differential and not your own disc count: your count alone does not
determine the winner when the board does not fill (measured: 1% of random
Othello games, and games as short as 17 discs happen), and more importantly the
change-of-perspective flip is only `64 - count` when the board DOES fill.  The
differential is antisymmetric, so flipping perspective is exactly `mu -> -mu`
with the variance untouched, the same one-line operation `flip_alpha` is for the
Dirichlet.

    score(state, player) = (discs_player - discs_opponent) / 64   in [-1, 1]
    sign(score) is the game result, so P(win) = Phi(mu / sigma) is still
    available and the win/draw/loss diagnostics keep working.

Two distributions, deliberately kept apart
------------------------------------------
The network predicts the ALEATORIC spread: given this position, how widely does
the final differential actually land?  That spread does not shrink with search —
it is a property of the position.

Selection needs the EPISTEMIC one: how sure are we of the MEAN?  That does
shrink with evidence, and it is what makes Thompson sampling anneal instead of
exploring forever.  So each edge accumulates observations and the belief is the
standard error of their mean:

    n observations (mu_i, var_i)
    belief_mu  = (1/n) * sum(mu_i)
    belief_var = (1/n) * sum(var_i) / n          <- mean variance, over n

The network's own prediction is seeded as observation 1, so an unvisited edge is
sampled from the prediction itself and every backup narrows the belief.  With n
identical observations N(m, v) the belief is N(m, v/n) — the textbook standard
error, and the direct analogue of the `additive` Dirichlet rule where alpha0 is
the visit count.

The Dirichlet engine has to overload alpha0 as a visit count to get that
annealing, conflating "how spread out are outcomes" with "how sure am I".  Here
they are two separate numbers, which is the main representational argument for
this design.

Losses
------
    KL(target || prediction) for the state belief and for every searched edge,
    closed form and elementary:
        log(s2/s1) + (v1 + (m1-m2)^2) / (2*v2) - 1/2
    No lgamma, no digamma — the whole DirectML fallback problem the Dirichlet
    loss needed simply does not arise here.

    NLL of the actual final differential under the state and played-action
    distributions, which is what ties the predicted spread to reality.

Scope
-----
Draws are measure-zero under a continuous density.  At Othello's ~2% that is
ignorable; in a game like Connect 4 with 36-44% draws it would not be, so this
representation is for games with a natural score, not a replacement everywhere.
"""

import math

import numpy as np

import connect4_dirichlet_utils as c4

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════
SCORE_SCALE = 64.0        # discs on an Othello board; normalises to [-1, 1]
VAR_FLOOR = 1e-6          # keeps 1/(2v) and log(v) finite
# A proven outcome has no spread, but a zero-variance TARGET sends
# KL = log(sigma_pred/sigma_target) to +inf, so proofs get a small floor rather
# than an exact spike.  1e-4 is sd 0.01, i.e. 0.64 of a disc.
TERMINAL_VAR = 1e-4
# Prior spread of an untrained net.  Measured over 2000 random Othello games the
# final differential has sd 0.291 normalised (18.6 discs, range [-52, +54]), so
# 0.5 is 1.7x the real spread -- deliberately wide, and in board terms sd 0.5 is
# a differential of 32, i.e. a 48-16 result, NOT 32 discs of one colour.
INIT_SD = 0.5

# The network's log-variance is NOT clamped.  A cap would hide a runaway rather
# than prevent one, and there is no mechanism here that should produce one; if
# it happens it should be visible and fixed at the source.  What that costs:
# exp() overflows fp32 above logvar ~88.7 and underflows to 0 below ~-87, so the
# loss diagnostics report the observed range every eval (`lv_lo`/`lv_hi`) and the
# training step refuses to apply a non-finite update.  See the drift before the
# explosion, rather than after.

_WIN, _DRAW, _LOSS = 0, 1, 2            # only for reporting; matches c4


# ══════════════════════════════════════════════════════════════════════════════
#  Score
# ══════════════════════════════════════════════════════════════════════════════
def othello_score(state, player):
    """Final differential for `player`, normalised to [-1, 1].

    Read off the observation tensor rather than the string board: planes are
    (empty, player-to-move's discs, opponent's discs) and are mover-relative, so
    plane 1 is always the tensor's own perspective."""
    t = np.asarray(state.observation_tensor(player)).reshape(
        c4._OBS_SHAPE if c4._OBS_SHAPE else (3, 8, 8))
    return float(t[1].sum() - t[2].sum()) / SCORE_SCALE


def sign_score(x):
    return (x > 0) - (x < 0)


# ══════════════════════════════════════════════════════════════════════════════
#  Gaussian primitives
# ══════════════════════════════════════════════════════════════════════════════
def flip(mu, var):
    """Change of perspective.  The differential is antisymmetric, so only the
    mean turns over; the spread of outcomes is the same fact either way."""
    return -mu, var


def kl(m1, v1, m2, v2):
    """KL( N(m1,v1) || N(m2,v2) ), elementwise.  `1` is the target."""
    v1 = np.maximum(v1, VAR_FLOOR)
    v2 = np.maximum(v2, VAR_FLOOR)
    return (0.5 * np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / (2.0 * v2) - 0.5)


def nll(mu, var, x):
    """Negative log likelihood of observing `x` under N(mu, var)."""
    var = np.maximum(var, VAR_FLOOR)
    return 0.5 * (math.log(2 * math.pi) + np.log(var) + (x - mu) ** 2 / var)


def p_win(mu, var):
    """P(differential > 0) — recovers an outcome probability from the Gaussian,
    so the existing win/draw/loss diagnostics still have something to read."""
    sd = np.sqrt(np.maximum(var, VAR_FLOOR))
    return 0.5 * (1.0 + _erf(mu / (sd * math.sqrt(2.0))))


def _erf(x):
    x = np.asarray(x, dtype=np.float64)
    return np.vectorize(math.erf)(x) if x.ndim else math.erf(float(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Belief accumulator
#
#  [n, sum_mu, sum_var].  The network prediction is seeded as observation 1, so
#  an edge that has never been searched is sampled from the prediction and each
#  backup narrows the belief.  belief_var divides the MEAN variance by n, which
#  is the standard error of the mean: n copies of N(m, v) give N(m, v/n).
# ══════════════════════════════════════════════════════════════════════════════
def new_acc(mu_pred, var_pred):
    return [1.0, float(mu_pred), float(max(var_pred, VAR_FLOOR))]


def acc_add(acc, mu, var):
    acc[0] += 1.0
    acc[1] += float(mu)
    acc[2] += float(max(var, VAR_FLOOR))


def belief(acc):
    """(mu, var) of the belief about the MEAN final differential."""
    n = acc[0]
    return acc[1] / n, max(acc[2] / (n * n), VAR_FLOOR)


def outcome_spread(acc):
    """(mu, var) of the predicted OUTCOME, i.e. the aleatoric distribution: the
    same mean, but the average spread rather than the standard error.  This is
    what the network is trained toward; `belief` is what search samples."""
    n = acc[0]
    return acc[1] / n, max(acc[2] / n, VAR_FLOOR)


# ══════════════════════════════════════════════════════════════════════════════
#  Tree
# ══════════════════════════════════════════════════════════════════════════════
_TERM_NONE = -2.0     # sentinel in `term`: not proven


class GNode:
    """One expanded state.  Per legal action a:
        acc[a]   belief accumulator, seeded with the network's prediction
        term[a]  proven final differential from THIS node's mover's view, or
                 _TERM_NONE
    """

    __slots__ = ('player', 'legal', 'acc', 'term', 'vloss', 'children', 'obs',
                 'mu', 'var', 'n_pred')

    def __init__(self, player, legal, a_mu, a_var, mu, var, obs=None):
        self.player = player
        self.legal = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        a_mu = np.asarray(a_mu, dtype=np.float64).reshape(k)
        a_var = np.maximum(np.asarray(a_var, dtype=np.float64).reshape(k),
                           VAR_FLOOR)
        # n, sum_mu, sum_var per action, vectorised.
        self.acc = np.stack([np.ones(k), a_mu, a_var])       # (3, k)
        self.term = np.full(k, _TERM_NONE)
        self.vloss = np.zeros(k, dtype=np.int32)
        self.children = [None] * k
        self.obs = obs
        self.mu = float(mu)          # the STATE's predicted distribution
        self.var = float(max(var, VAR_FLOOR))
        self.n_pred = 1.0

    def belief(self):
        """(mu, var) per action — what Thompson sampling draws from."""
        n = self.acc[0]
        return self.acc[1] / n, np.maximum(self.acc[2] / (n * n), VAR_FLOOR)

    def spread(self):
        """(mu, var) per action — the aleatoric target for the action head."""
        n = self.acc[0]
        return self.acc[1] / n, np.maximum(self.acc[2] / n, VAR_FLOOR)

    def add(self, idx, mu, var):
        self.acc[0, idx] += 1.0
        self.acc[1, idx] += mu
        self.acc[2, idx] += max(var, VAR_FLOOR)

    def visits(self):
        """Observations minus the seeded prediction — the true visit count."""
        return self.acc[0] - 1.0


def sample_edges(node, rng, temp=1.0):
    """Thompson sample: one draw per action from its belief, proven edges
    replaced by their exact value.  `temp` scales the sampled spread."""
    mu, var = node.belief()
    x = rng.normal(mu, np.sqrt(var) * temp)
    pr = node.term > _TERM_NONE
    if pr.any():
        x = np.where(pr, node.term, x)
    if node.vloss.any():
        x = x - 1e3 * node.vloss        # keep parallel leaves off the same edge
    return x


def select_leaf(root, root_state, rng, temp=1.0):
    """Descend by Thompson sampling to an unexpanded or terminal edge.

    Returns (path, leaf_state_or_None, (mu,var)_or_None, edge_or_None); the
    returned distribution is from the perspective of the mover at the DEEPEST
    node on the path."""
    node, state, path = root, root_state.clone(), []
    while True:
        idx = int(sample_edges(node, rng, temp).argmax())
        node.vloss[idx] += 1
        path.append((node, idx))
        if node.term[idx] > _TERM_NONE:
            return path, None, (float(node.term[idx]), TERMINAL_VAR), None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            s = othello_score(state, node.player)
            node.term[idx] = s
            return path, None, (s, TERMINAL_VAR), None
        child = node.children[idx]
        if child is None:
            return path, state, None, (node, idx)
        node = child


def backup(path, mu, var):
    """`(mu, var)` is from the perspective of the mover at the DEEPEST node;
    flip once per ply on the way up.  The variance is carried unchanged — it is
    the spread of the same outcome seen from the other side."""
    for node, idx in reversed(path):
        node.vloss[idx] -= 1
        node.add(idx, mu, var)
        mu = -mu


def node_value(node):
    """The node's own backed-up distribution: the visit-weighted belief over its
    actions, from this node's mover's view."""
    mu, var = node.belief()
    n = node.acc[0]
    w = n / n.sum()
    m = float((w * mu).sum())
    # Variance of a weighted mixture: mean variance plus spread of the means.
    v = float((w * var).sum() + (w * (mu - m) ** 2).sum())
    return m, max(v, VAR_FLOOR)


def node_solved(node):
    """If every action is proven, so is the node: the mover takes the best."""
    if (node.term > _TERM_NONE).all():
        return float(node.term.max())
    return None


def propagate_solved(path, aux=None):
    """Prove parent edges bottom-up.  A proven node's value is exact, and from
    the parent's view it is negated."""
    for k in range(len(path) - 1, 0, -1):
        node = path[k][0]
        out = node_solved(node)
        if out is None:
            break
        parent, pidx = path[k - 1]
        if parent.term[pidx] > _TERM_NONE:
            break
        parent.term[pidx] = -out
        if aux is not None and node.obs is not None:
            aux.append(make_target(node, solved_value=out))


def backup_terminal(path, mu, var, aux=None):
    backup(path, mu, var)
    propagate_solved(path, aux)


def descend(root, action):
    if root is None:
        return None
    hit = np.nonzero(root.legal == action)[0]
    return root.children[int(hit[0])] if len(hit) else None


def root_pick(root, rng, thompson, temp=1.0):
    """Move choice.  `thompson=True` samples the beliefs (self-play); otherwise
    take the action with the highest believed mean (evaluation)."""
    if thompson:
        return int(root.legal[int(sample_edges(root, rng, temp).argmax())])
    mu, _var = root.belief()
    pr = root.term > _TERM_NONE
    if pr.any():
        mu = np.where(pr, root.term, mu)
    return int(root.legal[int(mu.argmax())])


# ══════════════════════════════════════════════════════════════════════════════
#  Targets
# ══════════════════════════════════════════════════════════════════════════════
def make_target(root, solved_value=None):
    """One training sample from a searched node.

    `v_mu/v_var`  the state's backed-up distribution (KL target)
    `ev_idx`      which legal edges were actually searched
    `ev_mu/ev_var` their backed-up OUTCOME spreads (KL targets)
    `z`           filled in later with the real final differential (NLL target)
    """
    mu, var = root.spread()
    searched = np.nonzero(root.visits() > 0)[0]
    if solved_value is None:
        v_mu, v_var = node_value(root)
    else:
        v_mu, v_var = float(solved_value), TERMINAL_VAR
    return {
        'obs': root.obs,
        'legal': root.legal.copy(),
        'v_mu': np.float32(v_mu), 'v_var': np.float32(v_var),
        'ev_idx': searched.astype(np.int32),
        'ev_mu': mu[searched].astype(np.float32),
        'ev_var': var[searched].astype(np.float32),
        'played': -1,
        'z': np.float32(0.0), 'z_w': np.float32(0.0),
        'solved': solved_value is not None,
        'player': int(root.player),
    }


def finish_episode(samples, final_state):
    """Stamp the real final differential on every sample, from that sample's own
    mover's perspective.  Proven samples keep their exact value."""
    for s in samples:
        if s.get('solved'):
            continue
        s['z'] = np.float32(othello_score(final_state, s['player']))
        s['z_w'] = np.float32(1.0)
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Network and losses
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


if _HAS_TORCH:

    class GaussianNet(nn.Module):
        """Trunk → two Gaussian heads.

        forward(x) → (v_mu (B,), v_logvar (B,), a_mu (B,A), a_logvar (B,A))

        The trunk is c4's, unchanged, so this engine is matched to the Dirichlet
        arms and the AlphaZero control on everything except what it predicts.
        Each head emits 2 numbers where the Dirichlet emits 4, so the action head
        is half the size at the same action count."""

        def __init__(self, channels=64, num_blocks=5, head_ch=16):
            super().__init__()
            self._sig = (channels, num_blocks, head_ch)
            in_ch = c4._OBS_SHAPE[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                c4._norm(channels), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[c4.ResBlock(channels)
                                        for _ in range(num_blocks)])
            self.head = nn.Sequential(
                nn.Conv2d(channels, head_ch, 1, bias=False),
                c4._norm(head_ch), nn.ReLU(inplace=True), nn.Flatten())
            flat = head_ch * c4._OBS_SHAPE[1] * c4._OBS_SHAPE[2]
            self.v_out = nn.Linear(flat, 2)                    # mu, logvar
            self.a_out = nn.Linear(flat, c4._NUM_ACTIONS * 2)  # …per action
            # Untrained net: mean 0 (an even game) with a WIDE spread, so search
            # dominates the prior from generation 0.  INIT_SD = 0.5 is 1.7x the
            # measured spread of real games — see the note on INIT_SD for what
            # that means on a board.
            nn.init.zeros_(self.v_out.weight); nn.init.zeros_(self.a_out.weight)
            with torch.no_grad():
                lv0 = math.log(INIT_SD ** 2)
                self.v_out.bias.zero_(); self.v_out.bias[1] = lv0
                self.a_out.bias.view(c4._NUM_ACTIONS, 2).zero_()
                self.a_out.bias.view(c4._NUM_ACTIONS, 2)[:, 1] = lv0

        def forward(self, x):
            h = self.head(self.body(self.stem(x)))
            v = self.v_out(h)
            a = self.a_out(h).view(-1, c4._NUM_ACTIONS, 2)
            return v[:, 0], v[:, 1], a[..., 0], a[..., 1]

    def _var(logvar):
        """log-variance → variance.  Deliberately unclamped: see the note on
        INIT_SD.  VAR_FLOOR still applies inside kl_t/nll_t, but that is a
        divide-by-zero guard six orders below any meaningful spread (sd 0.001 is
        0.064 of a disc), not a cap on what the network may predict."""
        return torch.exp(logvar)

    def kl_t(m1, v1, m2, v2):
        """KL( N(m1,v1) || N(m2,v2) ) in torch.  `1` is the target.

        Elementary — no lgamma and no digamma, so none of the DirectML kernel
        problems the Dirichlet loss ran into apply here."""
        v1 = v1.clamp_min(VAR_FLOOR)
        v2 = v2.clamp_min(VAR_FLOOR)
        return 0.5 * (torch.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    def nll_t(mu, var, x):
        var = var.clamp_min(VAR_FLOOR)
        return 0.5 * (math.log(2 * math.pi) + torch.log(var)
                      + (x - mu) ** 2 / var)

    def nn_eval_states(network, device, states):
        """States → (v_mu, v_var, a_mu, a_var, obs16), all numpy."""
        obs16 = np.asarray([c4.make_obs(s) for s in states], dtype=np.float16)
        x = c4.batch_to_tensor(obs16, device)
        with torch.inference_mode():
            vm, vlv, am, alv = network(x)
            vv, av = _var(vlv), _var(alv)
        return (vm.cpu().numpy(), vv.cpu().numpy(),
                am.cpu().numpy(), av.cpu().numpy(), obs16)

    def expand_node(network, device, state):
        vm, vv, am, av, ob = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        return GNode(state.current_player(), leg, am[0][leg], av[0][leg],
                     vm[0], vv[0], obs=ob[0])

    def full_loss(out, meta, weights):
        """The four terms the design calls for.

        (1) KL of the state's backed-up distribution against the state head
        (2) KL of each searched edge's backed-up spread against the action head
        (3) NLL of the real final differential under the state head
        (4) NLL of the same under the played action's distribution

        (1) and (2) teach the shape of the search; (3) and (4) tie that shape to
        what actually happened, which is the only thing anchoring the predicted
        spread to reality."""
        v_mu, v_lv, a_mu, a_lv = out
        act = meta['pad_act']
        mask = meta['pad_mask']
        v_var = _var(v_lv)
        q_mu = a_mu.gather(1, act)
        q_var = _var(a_lv.gather(1, act))

        L_klv = kl_t(meta['v_mu'], meta['v_var'], v_mu, v_var).mean()

        evm = meta['ev_mask'].float()
        kla = kl_t(meta['ev_mu'], meta['ev_var'], q_mu, q_var)
        L_kla = (kla * evm).sum() / evm.sum().clamp_min(1.0)

        zw = meta['z_w']
        L_nllv = (nll_t(v_mu, v_var, meta['z']) * zw).sum() / zw.sum().clamp_min(1.0)

        pw = meta['played_w'] * zw
        pl = meta['played'].clamp_min(0)
        pm = q_mu.gather(1, pl.view(-1, 1)).squeeze(1)
        pv = q_var.gather(1, pl.view(-1, 1)).squeeze(1)
        L_nlla = (nll_t(pm, pv, meta['z']) * pw).sum() / pw.sum().clamp_min(1.0)

        w_klv, w_kla, w_nllv, w_nlla = weights
        total = (w_klv * L_klv + w_kla * L_kla
                 + w_nllv * L_nllv + w_nlla * L_nlla)
        with torch.no_grad():
            evc = evm.sum().clamp_min(1.0)
            lv = torch.cat([v_lv.reshape(-1),
                            a_lv.gather(1, act)[mask].reshape(-1)])
            diag = torch.stack([
                total, L_klv, L_kla, L_nllv, L_nlla,
                v_mu.mean(), v_var.sqrt().mean(),
                (q_var.sqrt() * evm).sum() / evc,
                (meta['ev_var'].sqrt() * evm).sum() / evc,
                lv.min(), lv.max(),
            ]).to('cpu', copy=False).tolist()
        parts = dict(zip(('loss', 'klv', 'kla', 'nllv', 'nlla',
                          'v_mu', 'v_sd', 'a_sd_pred', 'a_sd_tgt',
                          'lv_lo', 'lv_hi'), diag))
        return total, parts
