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
    mu = float(mu_pred)
    return [1.0, mu, float(max(var_pred, VAR_FLOOR)), mu * mu]


def acc_add(acc, mu, var):
    mu = float(mu)
    acc[0] += 1.0
    acc[1] += mu
    acc[2] += float(max(var, VAR_FLOOR))
    acc[3] += mu * mu


def total_var(acc):
    """Variance of ONE outcome through this edge, by the law of total variance:

        Var(X) = E[Var(X | observation)] + Var(E[X | observation])
               = mean of the observed variances + spread of their means

    The second term is what stops the variance collapsing.  Every observation's
    variance is a number the NETWORK predicted, so training the head toward the
    mean of those is self-referential -- predicted variance becomes the target
    becomes the predicted variance, and the loop is consistent at any level
    including zero, with only the NLL against real outcomes anchoring it.  The
    disagreement between the observed MEANS is not self-referential: it comes
    from actually searching different continuations.  If the search finds the
    children disagree, the target variance goes up and the head cannot shrink it
    however confident it would like to be."""
    n = acc[0]
    mu = acc[1] / n
    between = acc[3] / n - mu * mu
    if hasattr(between, 'clip'):
        between = between.clip(0.0)             # float error can go slightly <0
    else:
        between = max(between, 0.0)
    return acc[2] / n + between


def belief(acc):
    """(mu, var) of the belief about the MEAN final differential."""
    n = acc[0]
    return acc[1] / n, max(total_var(acc) / n, VAR_FLOOR)


def outcome_spread(acc):
    """(mu, var) of the predicted OUTCOME -- the aleatoric distribution the
    network is trained toward.  `belief` is the standard error of its mean, which
    is what search samples."""
    return acc[1] / acc[0], max(total_var(acc), VAR_FLOOR)


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

    # `n_term` / `n_vloss` are counts of proven and in-flight edges.  They exist
    # so the hot path can skip two whole-array scans: sample_edges runs once per
    # simulation per node and profiled at 24% of self-play time, where numpy's
    # per-call overhead on 8-element arrays dominates the arithmetic.
    __slots__ = ('player', 'legal', 'acc', 'term', 'vloss', 'children', 'obs',
                 'mu', 'var', 'n_term', 'n_vloss')

    def __init__(self, player, legal, a_mu, a_var, mu, var, obs=None):
        self.player = player
        self.legal = np.asarray(legal, dtype=np.int32)
        k = len(self.legal)
        # One allocation for the accumulator instead of three plus a stack.
        acc = np.empty((4, k))
        acc[0] = 1.0
        acc[1] = a_mu
        np.maximum(a_var, VAR_FLOOR, out=acc[2])
        acc[3] = np.square(a_mu)                # for the law of total variance
        self.acc = acc
        self.term = np.full(k, _TERM_NONE)
        self.vloss = np.zeros(k, dtype=np.int32)
        self.children = [None] * k
        self.obs = obs
        self.mu = float(mu)          # the STATE's predicted distribution
        self.var = float(max(var, VAR_FLOOR))
        self.n_term = 0
        self.n_vloss = 0

    def total_var(self):
        """Per action: mean observed variance PLUS the spread of the observed
        means.  See the module-level total_var for why the second term is what
        keeps the head honest."""
        a = self.acc
        n = a[0]
        mu = a[1] / n
        return a[2] / n + (a[3] / n - mu * mu).clip(0.0)

    def belief(self):
        """(mu, var) per action — what Thompson sampling draws from."""
        a = self.acc
        n = a[0]
        return a[1] / n, np.maximum(self.total_var() / n, VAR_FLOOR)

    def spread(self):
        """(mu, var) per action — the aleatoric target for the action head."""
        return self.acc[1] / self.acc[0], np.maximum(self.total_var(), VAR_FLOOR)

    def prove(self, idx, value):
        if self.term[idx] <= _TERM_NONE:
            self.n_term += 1
        self.term[idx] = value

    def add(self, idx, mu, var):
        self.acc[0, idx] += 1.0
        self.acc[1, idx] += mu
        self.acc[2, idx] += max(var, VAR_FLOOR)
        self.acc[3, idx] += mu * mu

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
            if node.term[idx] <= _TERM_NONE:
                node.n_term += 1
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
        node.n_vloss -= 1
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
        parent.n_term += 1
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
        # z is filled in by finish_episode once the real result is known.
        z, z_w = 0.0, 0.0
    else:
        v_mu, v_var = float(solved_value), TERMINAL_VAR
        # A proven node's final differential is KNOWN exactly, and it is better
        # information than the played-out result: it is what optimal play from
        # here yields, whereas the game may have deviated afterwards.  So the
        # NLL trains on it directly rather than waiting for the outcome.
        z, z_w = float(solved_value), 1.0
    return {
        'obs': root.obs,
        'legal': root.legal.copy(),
        'v_mu': np.float32(v_mu), 'v_var': np.float32(v_var),
        'ev_idx': searched.astype(np.int32),
        'ev_mu': mu[searched].astype(np.float32),
        'ev_var': var[searched].astype(np.float32),
        # Observations behind each edge target (prediction + backups).  Under
        # sequential halving these differ by an order of magnitude across edges,
        # so the loss can weight a 25-simulation target above a 1-simulation one.
        'ev_n': root.acc[0][searched].astype(np.float32),
        'played': -1,
        'z': np.float32(z), 'z_w': np.float32(z_w),
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

    class SpatialGaussianNet(nn.Module):
        """Trunk → state head + a SPATIAL action head.

        The dense action head is a Linear(flat, A*2).  At Othello's 65 actions
        that is ~67k parameters, and each action's row is only updated when that
        square happens to be legal -- measured at 12.9% of positions, against
        96.8% on Connect 4.  So the widest part of the network learns from the
        thinnest signal.

        Othello's actions ARE board cells: action a < 64 is exactly cell
        (a//8, a%8), verified against the engine.  So emit them with a 1x1
        convolution instead.  The same weights apply at all 64 cells, so every
        legal move at every position trains them -- the sparsity disappears, not
        by inventing targets for dead slots but by not having per-slot weights
        at all.  It is also the right prior: Othello value is strongly
        translation-structured (corners, edges, X-squares), and this is how
        AlphaGo Zero emits its policy over board points.

        Action 64 (pass) is not a cell, and is always the ONLY legal action when
        it appears, so it never competes for selection.  It gets its own tiny
        head off the pooled trunk purely so its target has somewhere to go."""

        def __init__(self, channels=64, num_blocks=5, head_ch=16):
            super().__init__()
            self._sig = (channels, num_blocks, head_ch)
            in_ch = c4._OBS_SHAPE[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                c4._norm(channels), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[c4.ResBlock(channels)
                                        for _ in range(num_blocks)])
            # ONE 1x1 projection shared by both heads.  Giving the action head
            # its own cost ~11% of self-play wall clock for no benefit: the two
            # heads want the same per-cell features, one flattened and one kept
            # spatial.
            self.proj = nn.Sequential(
                nn.Conv2d(channels, head_ch, 1, bias=False),
                c4._norm(head_ch), nn.ReLU(inplace=True))
            flat = head_ch * c4._OBS_SHAPE[1] * c4._OBS_SHAPE[2]
            self.v_out = nn.Linear(flat, 2)
            self.a_head = nn.Conv2d(head_ch, 2, 1)
            self.a_pass = nn.Linear(head_ch, 2)
            self._cells = c4._OBS_SHAPE[1] * c4._OBS_SHAPE[2]
            self._extra = c4._NUM_ACTIONS - self._cells
            lv0 = math.log(INIT_SD ** 2)
            nn.init.zeros_(self.v_out.weight)
            nn.init.zeros_(self.a_head.weight)
            nn.init.zeros_(self.a_pass.weight)
            with torch.no_grad():
                self.v_out.bias.zero_(); self.v_out.bias[1] = lv0
                self.a_head.bias.zero_(); self.a_head.bias[1] = lv0
                self.a_pass.bias.zero_(); self.a_pass.bias[1] = lv0

        def forward(self, x):
            h = self.proj(self.body(self.stem(x)))   # (B, head_ch, H, W)
            v = self.v_out(h.flatten(1))
            a = self.a_head(h)                       # (B, 2, H, W)
            B = a.shape[0]
            a = a.reshape(B, 2, self._cells).transpose(1, 2)      # (B, cells, 2)
            if self._extra > 0:                      # pass, and anything like it
                p = self.a_pass(h.mean(dim=(2, 3)))  # (B, 2)
                a = torch.cat([a, p.unsqueeze(1).expand(B, self._extra, 2)], 1)
            return v[:, 0], v[:, 1], a[..., 0], a[..., 1]

    def make_net(channels, num_blocks, head_ch, head='spatial'):
        """`head='spatial'` shares action weights across board cells;
        `'dense'` is the original Linear(flat, A*2)."""
        cls = SpatialGaussianNet if head == 'spatial' else GaussianNet
        return cls(channels, num_blocks, head_ch)

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

    def full_loss(out, meta, weights, ev_weight='uniform'):
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

        # Edge targets differ enormously in how much evidence stands behind
        # them -- under sequential halving by design, since eliminated actions
        # keep only a couple of simulations.  'evidence' weights each edge by
        # its observation count so a 25-simulation target is not averaged in
        # alongside a 1-simulation one as though they were equally trustworthy.
        # Default stays 'uniform' so this is an explicit experiment rather than
        # a silent change to what the existing arm optimises.
        evm = meta['ev_mask'].float()
        if ev_weight == 'evidence':
            evm = evm * meta['ev_n']
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
            # CALIBRATION.  The only direct check that the network is not
            # quietly confident about everything.  Under a correct Gaussian the
            # squared error equals the predicted variance in expectation, so
            #     calib = E[(z-mu)^2] / E[sigma^2]
            # sits at 1.0; above 1 the spread is too NARROW for the errors it is
            # actually making, which is overconfidence, and below 1 it is hedging.
            # cov68 is the same question without the squaring: the share of
            # outcomes landing inside one predicted standard deviation, which a
            # calibrated Gaussian puts at 0.683.
            zw_ = meta['z_w']
            wsum = zw_.sum().clamp_min(1.0)
            err2 = ((meta['z'] - v_mu) ** 2 * zw_).sum() / wsum
            pvar = (v_var * zw_).sum() / wsum
            calib = err2 / pvar.clamp_min(VAR_FLOOR)
            inside = (((meta['z'] - v_mu).abs() <= v_var.sqrt()).float()
                      * zw_).sum() / wsum
            lv = torch.cat([v_lv.reshape(-1),
                            a_lv.gather(1, act)[mask].reshape(-1)])
            # Means and spreads are reported in DISCS, not in the internal
            # normalised units.  The scale is X = differential/64, so a
            # normalised sd of 0.29 is 19 discs of differential — the log line
            # is what gets read, and it should read in board terms.
            S = SCORE_SCALE
            diag = torch.stack([
                total, L_klv, L_kla, L_nllv, L_nlla,
                v_mu.mean() * S, v_var.sqrt().mean() * S,
                (q_var.sqrt() * evm).sum() / evc * S,
                (meta['ev_var'].sqrt() * evm).sum() / evc * S,
                lv.min(), lv.max(), calib, inside,
                err2.sqrt() * S,
            ]).to('cpu', copy=False).tolist()
        parts = dict(zip(('loss', 'klv', 'kla', 'nllv', 'nlla',
                          'v_mu', 'v_sd', 'a_sd_pred', 'a_sd_tgt',
                          'lv_lo', 'lv_hi', 'calib', 'cov68', 'rmse'), diag))
        return total, parts

    # ── Batching ──────────────────────────────────────────────────────────────
    def build_batch_meta(batch, device):
        """Pad the sample dicts into fixed-(B,K) tensors on `device`.

        Packed into one transfer per dtype rather than one per array: DirectML
        fragments its D3D12 heap under a long run of many small host-to-device
        copies, which is what killed a multi-hour Dirichlet run."""
        B = len(batch)
        K = max(len(s['legal']) for s in batch)
        act = np.zeros((B, K), np.int64)
        mask = np.zeros((B, K), bool)
        ev_mask = np.zeros((B, K), bool)
        ev_mu = np.zeros((B, K), np.float32)
        ev_var = np.ones((B, K), np.float32)
        ev_n = np.ones((B, K), np.float32)
        v_mu = np.zeros(B, np.float32)
        v_var = np.ones(B, np.float32)
        z = np.zeros(B, np.float32)
        z_w = np.zeros(B, np.float32)
        played = np.zeros(B, np.int64)
        played_w = np.zeros(B, np.float32)
        for i, s in enumerate(batch):
            leg = np.asarray(s['legal'])
            k = len(leg)
            act[i, :k] = leg
            mask[i, :k] = True
            idx = np.asarray(s['ev_idx'], np.int64)
            if len(idx):
                ev_mask[i, idx] = True
                ev_mu[i, idx] = s['ev_mu']
                ev_var[i, idx] = s['ev_var']
                ev_n[i, idx] = s.get('ev_n', np.ones(len(idx), np.float32))
            v_mu[i] = s['v_mu']
            v_var[i] = s['v_var']
            z[i] = s['z']
            z_w[i] = s['z_w']
            p = int(s.get('played', -1))
            if p >= 0:
                played[i] = p
                played_w[i] = 1.0
        ti = torch.from_numpy(np.concatenate([act.reshape(-1), played])).to(device)
        tf = torch.from_numpy(np.concatenate([
            ev_mu.reshape(-1), ev_var.reshape(-1), ev_n.reshape(-1),
            v_mu, v_var, z, z_w, played_w])).to(device)
        tb = torch.from_numpy(np.concatenate([
            mask.reshape(-1), ev_mask.reshape(-1)])).to(device)
        BK = B * K
        o = 0

        def nxt(n, shape=None):
            nonlocal o
            t = tf[o:o + n]
            o += n
            return t.view(*shape) if shape else t

        return {
            'pad_act': ti[:BK].view(B, K), 'played': ti[BK:BK + B],
            'pad_mask': tb[:BK].view(B, K),
            'ev_mask': tb[BK:BK + BK].view(B, K),
            'ev_mu': nxt(BK, (B, K)), 'ev_var': nxt(BK, (B, K)),
            'ev_n': nxt(BK, (B, K)),
            'v_mu': nxt(B), 'v_var': nxt(B), 'z': nxt(B), 'z_w': nxt(B),
            'played_w': nxt(B),
        }

    def train_step(network, optimizer, batch, device, weights, grad_clip=1.0,
                   model_lock=None, ev_weight='uniform'):
        """One optimiser step.  A non-finite loss or gradient is SKIPPED, never
        applied: clip_grad_norm_ turns a NaN norm into a NaN scale and the
        optimiser then poisons every weight and moment.  With the log-variance
        deliberately unclamped this is the guard that makes a runaway visible
        and survivable instead of silently fatal."""
        import contextlib
        lock = model_lock or contextlib.nullcontext()

        def _once():
            meta = build_batch_meta(batch, device)
            x = c4.batch_to_tensor([s['obs'] for s in batch], device)
            with lock:
                out = network(x)
                optimizer.zero_grad()
                loss, parts = full_loss(out, meta, weights, ev_weight)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(network.parameters(),
                                                       grad_clip)
                ok = (math.isfinite(parts['loss'])
                      and bool(torch.isfinite(gnorm)))
                if ok:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                parts['nonfinite'] = 0.0 if ok else 1.0
            return parts['loss'], parts

        return c4.device_retry(_once)

    # ── Bots ──────────────────────────────────────────────────────────────────
    class GMCTSBot:
        """Batched Thompson-sampling search over Gaussian beliefs."""

        def __init__(self, game, network, device, max_simulations, batch_size=8,
                     temp=1.0, random_state=None):
            self.game, self.network, self.device = game, network, device
            self.max_simulations = max_simulations
            self.batch_size = batch_size
            self.temp = temp
            self._rng = random_state or np.random.default_rng()

        def mcts_search(self, state, root=None):
            if root is None:
                root = expand_node(self.network, self.device, state)
            sims = 0
            while sims < self.max_simulations:
                if node_solved(root) is not None:
                    break
                wave = min(self.batch_size, self.max_simulations - sims)
                pending, uniq = [], {}
                for _ in range(wave):
                    path, st, val, edge = select_leaf(root, state, self._rng,
                                                      self.temp)
                    if st is None:
                        backup_terminal(path, val[0], val[1])
                        sims += 1
                    else:
                        node, idx = edge
                        pending.append((path, node, idx))
                        uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    items = list(uniq.values())
                    vm, vv, am, av, ob = nn_eval_states(
                        self.network, self.device, [e[2] for e in items])
                    for (node, idx, st), m, v, a_m, a_v, o in zip(
                            items, vm, vv, am, av, ob):
                        leg = st.legal_actions()
                        node.children[idx] = GNode(st.current_player(), leg,
                                                   a_m[leg], a_v[leg], m, v,
                                                   obs=o)
                for path, node, idx in pending:
                    ch = node.children[idx]
                    backup(path, -ch.mu, ch.var)     # child's view -> parent's
                    sims += 1
            return root

    def value_greedy_move(network, state, device):
        """Search-free: the action head's own believed mean, nothing expanded."""
        _vm, _vv, am, _av, _o = nn_eval_states(network, device, [state])
        leg = state.legal_actions()
        return int(leg[int(np.asarray(am[0][leg]).argmax())])

    def value_lookahead_move(network, state, device):
        """Search-free with ONE ply: apply each move, score the child with the
        STATE head, negate, true outcome for terminal children.  This is the
        like-for-like read across engines — see c4.value_lookahead_move."""
        leg = state.legal_actions()
        vals = [None] * len(leg)
        pend_i, pend_s = [], []
        for i, a in enumerate(leg):
            cs = state.clone()
            cs.apply_action(int(a))
            if cs.is_terminal():
                vals[i] = othello_score(cs, state.current_player())
            else:
                pend_i.append(i)
                pend_s.append(cs)
        if pend_s:
            vm, _vv, _am, _av, _o = nn_eval_states(network, device, pend_s)
            for i, m in zip(pend_i, vm):
                vals[i] = -float(m)
        return int(leg[int(np.argmax(vals))])

    # ── Self-play ─────────────────────────────────────────────────────────────
    def _slot_new(game, cfg, rng):
        return {'state': game.new_initial_state(), 'hist': [], 'aux': [],
                'move': 0, 'root': None, 'n': 0,
                'sims': (cfg['fast_sims'] if rng.random() < cfg['fast_prob']
                         else cfg['full_sims']),
                # sequential-halving state, unused when root_select='thompson'
                'cand': None, 'sched': None, 'phase': 0, 'done_in_phase': 0}

    def _halving_begin(slot, rng, cfg):
        """Draw the candidate set and lay out the budget for one move."""
        root = slot['root']
        k = len(root.legal)
        m = choose_considered(slot['sims'], k, cfg['max_considered'])
        draw = sample_edges(root, rng, cfg['temp'])
        slot['cand'] = sorted(range(k), key=lambda i: -draw[i])[:m]
        slot['sched'] = halving_schedule(slot['sims'], m)
        slot['phase'] = 0
        slot['done_in_phase'] = 0
        slot['cand'] = slot['cand'][:slot['sched'][0][0]]

    def _halving_step(slot, rng, cfg):
        """Called once per wave: close out the phase if its quota is met.
        Returns True when the move's search is finished."""
        sched = slot['sched']
        slot['done_in_phase'] += 1
        if slot['done_in_phase'] < sched[slot['phase']][1]:
            return False
        slot['phase'] += 1
        slot['done_in_phase'] = 0
        if slot['phase'] >= len(sched) or len(slot['cand']) <= 1:
            return True
        slot['cand'] = halving_survivors(slot['root'], rng, slot['cand'],
                                         cfg['temp'])[:sched[slot['phase']][0]]
        return False

    class ParallelSelfPlay:
        """`n_parallel` games advanced in lockstep so every leaf wave is one
        batched forward.  Mirrors the AlphaZero driver's shape, so the arms are
        matched on self-play as well as on the trunk."""

        def __init__(self, game, network, device, cfg, seed=None):
            self.game, self.network, self.device = game, network, device
            self.cfg = cfg
            self.rng = np.random.default_rng(seed)
            self.last_aux = 0
            self.stats = {'games': 0, 'draw': 0, 'plies': 0, 'cutoff': 0}
            self.fwd_calls = self.fwd_rows = 0
            self.slots = [_slot_new(game, cfg, self.rng)
                          for _ in range(cfg['n_parallel'])]
            self._done = []

        def _finish(self, i):
            s = self.slots[i]
            st = s['state']
            if st.is_terminal():
                finish_episode(s['hist'], st)
                sc = othello_score(st, 0)
                self.stats['draw'] += sc == 0
            else:
                self.stats['cutoff'] += 1
                s['hist'] = []
            self.stats['games'] += 1
            self.stats['plies'] += s['move']
            self.last_aux = len(s['aux'])
            out = s['hist'] + s['aux']
            self.slots[i] = _slot_new(self.game, self.cfg, self.rng)
            return out

        def episodes(self):
            cfg = self.cfg
            while True:
                idxs = list(range(len(self.slots)))
                # 1. make sure every slot has a root
                need = [i for i in idxs if self.slots[i]['root'] is None]
                if need:
                    states = [self.slots[i]['state'] for i in need]
                    vm, vv, am, av, ob = nn_eval_states(self.network,
                                                        self.device, states)
                    self.fwd_calls += 1; self.fwd_rows += len(states)
                    for j, i in enumerate(need):
                        st = states[j]
                        leg = st.legal_actions()
                        self.slots[i]['root'] = GNode(
                            st.current_player(), leg, am[j][leg], av[j][leg],
                            vm[j], vv[j], obs=ob[j])
                # 2. one wave of leaves across all slots
                halving = cfg.get('root_select') == 'halving'
                pending, uniq = [], {}
                for i in idxs:
                    s = self.slots[i]
                    if node_solved(s['root']) is not None:
                        continue
                    if halving and s['cand'] is None:
                        _halving_begin(s, self.rng, cfg)
                    # One leaf per surviving candidate under halving, or
                    # wave_per_game free Thompson descents otherwise.
                    starts = (list(s['cand']) if halving
                              else [None] * cfg['wave_per_game'])
                    for first in starts:
                        if first is None:
                            path, st, val, edge = select_leaf(
                                s['root'], s['state'], self.rng, cfg['temp'])
                        else:
                            path, st, val, edge = select_leaf_forced(
                                s['root'], s['state'], first, self.rng,
                                cfg['temp'])
                        if st is None:
                            backup_terminal(path, val[0], val[1], s['aux'])
                            s['n'] += 1
                        else:
                            node, idx = edge
                            pending.append((i, path, node, idx))
                            uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    items = list(uniq.values())
                    vm, vv, am, av, ob = nn_eval_states(
                        self.network, self.device, [e[2] for e in items])
                    self.fwd_calls += 1; self.fwd_rows += len(items)
                    for (node, idx, st), m, v, a_m, a_v, o in zip(
                            items, vm, vv, am, av, ob):
                        leg = st.legal_actions()
                        node.children[idx] = GNode(st.current_player(), leg,
                                                   a_m[leg], a_v[leg], m, v,
                                                   obs=o)
                for i, path, node, idx in pending:
                    ch = node.children[idx]
                    backup(path, -ch.mu, ch.var)
                    self.slots[i]['n'] += 1
                # 3. advance any slot whose search budget is spent
                for i in idxs:
                    s = self.slots[i]
                    root = s['root']
                    if root is None:
                        continue
                    solved = node_solved(root) is not None
                    if halving and not solved:
                        if not _halving_step(s, self.rng, cfg):
                            continue
                    elif s['n'] < s['sims'] and not solved:
                        continue
                    t = make_target(root)
                    if halving and not solved and s['cand']:
                        a = halving_pick(root, s['cand'][0], self.rng, True)
                    else:
                        a = root_pick(root, self.rng,
                                      thompson=s['move'] < cfg['temp_threshold'],
                                      temp=cfg['temp'])
                    pidx = int(np.nonzero(root.legal == a)[0][0])
                    t['played'] = pidx
                    s['hist'].append(t)
                    s['state'].apply_action(a)
                    s['move'] += 1
                    s['n'] = 0
                    s['cand'] = None            # next move draws a fresh set
                    s['root'] = root.children[pidx]
                    if (s['state'].is_terminal()
                            or s['move'] >= cfg['max_plies']):
                        out = self._finish(i)
                        if out:
                            yield out

        def shutdown(self):
            pass

    def quick_match(net_a, net_b, game, n_games, device, rng=None,
                    opening_plies=2, max_plies=128):
        """Search-free match, alternating colours.  Uses the one-ply lookahead so
        it measures the same thing the benchmark's sims=0 round robin does."""
        rng = rng or np.random.default_rng()
        wa = d = wb = 0
        for i in range(n_games):
            a_side = i % 2
            st = game.new_initial_state()
            for _ in range(opening_plies):
                if st.is_terminal():
                    break
                leg = st.legal_actions()
                st.apply_action(int(leg[rng.integers(len(leg))]))
            ply = 0
            while not st.is_terminal() and ply < max_plies:
                net = net_a if st.current_player() == a_side else net_b
                if net is None:
                    leg = st.legal_actions()
                    mv = int(leg[rng.integers(len(leg))])
                else:
                    mv = value_lookahead_move(net, st, device)
                st.apply_action(mv); ply += 1
            if not st.is_terminal():
                d += 1
                continue
            r = st.returns()[a_side]
            wa += r > 0; wb += r < 0; d += r == 0
        return int(wa), int(d), int(wb)

    # ── Sequential-halving root search ────────────────────────────────────────
    def halving_search(network, device, state, rng, n_sims, root=None,
                       max_considered=16, temp=1.0, batch_size=8):
        """Sequential halving over the root's actions, Thompson below.

        Returns (root, chosen_action_index).  The survivor of the last round IS
        the move, which is the property Gumbel AlphaZero's halving buys: the
        choice is a sample from an improved policy even at a small budget,
        rather than an argmax over noisy visit counts."""
        if root is None:
            root = expand_node(network, device, state)
        k = len(root.legal)
        if k == 1:
            return root, 0
        # Candidates: the top of ONE Thompson draw, in place of Gumbel top-k.
        m = choose_considered(n_sims, k, max_considered)
        draw = sample_edges(root, rng, temp)
        cand = sorted(range(k), key=lambda i: -draw[i])[:m]

        for survivors, each in halving_schedule(n_sims, m):
            cand = cand[:survivors]
            if node_solved(root) is not None:
                break
            for _ in range(each):
                pend, uniq = [], {}
                for i in cand:                      # one wave across survivors
                    path, st, val, edge = select_leaf_forced(root, state, i,
                                                             rng, temp)
                    if st is None:
                        backup_terminal(path, val[0], val[1])
                    else:
                        node, idx = edge
                        pend.append((path, node, idx))
                        uniq.setdefault((id(node), idx), (node, idx, st))
                if uniq:
                    items = list(uniq.values())
                    vm, vv, am, av, ob = nn_eval_states(
                        network, device, [e[2] for e in items])
                    for (node, idx, st), mu, v, a_m, a_v, o in zip(
                            items, vm, vv, am, av, ob):
                        leg = st.legal_actions()
                        node.children[idx] = GNode(st.current_player(), leg,
                                                   a_m[leg], a_v[leg], mu, v,
                                                   obs=o)
                for path, node, idx in pend:
                    ch = node.children[idx]
                    backup(path, -ch.mu, ch.var)
            if len(cand) > 1:
                cand = halving_survivors(root, rng, cand, temp)
        return root, cand[0]

    class HalvingBot:
        """Same interface as GMCTSBot, so the tournament and the Elo ladder can
        hold the two side by side."""

        def __init__(self, game, network, device, max_simulations, batch_size=8,
                     temp=1.0, random_state=None, max_considered=16):
            self.game, self.network, self.device = game, network, device
            self.max_simulations = max_simulations
            self.batch_size = batch_size
            self.temp = temp
            self.max_considered = max_considered
            self._rng = random_state or np.random.default_rng()
            self.last_choice = None

        def mcts_search(self, state, root=None):
            root, idx = halving_search(
                self.network, self.device, state, self._rng,
                self.max_simulations, root=root,
                max_considered=self.max_considered, temp=self.temp,
                batch_size=self.batch_size)
            self.last_choice = idx
            return root

    def halving_pick(root, idx, rng, thompson):
        """The survivor is the move.  A proven win still overrides it — the
        solver knows more than any amount of sampling."""
        pr = root.term > _TERM_NONE
        if pr.any():
            mu, _v = root.belief()
            mu = np.where(pr, root.term, mu)
            if float(np.max(root.term[pr])) >= float(mu[idx]):
                return int(root.legal[int(mu.argmax())])
        return int(root.legal[int(idx)])

    # ── Config ────────────────────────────────────────────────────────────────
    from dataclasses import dataclass, asdict

    @dataclass
    class Config:
        """Matched to the ThompsonZero / AlphaZero arms wherever the method does
        not force a difference, so a benchmark compares representations rather
        than budgets."""
        game_name: str = 'othello'
        checkpoint_dir: str = 'othello_gauss_ckpt'
        num_episodes: int = 4000
        seed: int = 0
        device_preference: str = 'auto'
        resume: bool = True
        channels: int = 32
        num_blocks: int = 3
        head_ch: int = 8
        # 'spatial' shares the action head's weights across board cells, which
        # is exact for Othello (action a<64 IS cell (a//8, a%8)) and removes the
        # 12.9%-supervision problem the dense head has.  Same speed, 64x fewer
        # head parameters.  'dense' is the original Linear(flat, A*2).
        head: str = 'spatial'
        # search
        fast_sims: int = 100
        full_sims: int = 300
        fast_prob: float = 0.75
        temp: float = 1.0
        temp_threshold: int = 30
        # 'thompson' samples the root like every other node; 'halving' runs
        # Gumbel-AlphaZero-style sequential halving over the root's actions,
        # using Thompson draws in place of Gumbel noise.
        root_select: str = 'thompson'
        max_considered: int = 16
        # 'uniform' or 'evidence' -- how the action KL weights edge targets that
        # were searched very unequally.  See full_loss.
        ev_weight: str = 'uniform'
        n_parallel_games: int = 16
        wave_per_game: int = 4
        max_plies: int = 128
        # training
        batch_size: int = 256
        train_steps_per_ep: int = 4
        max_buffer: int = 150_000
        lr_peak: float = 2e-3
        lr_warmup_eps: int = 50
        lr_decay_eps: int = 4000
        lr_min_factor: float = 0.1
        weight_decay: float = 1e-4
        grad_clip: float = 1.0
        # (state KL, action KL, state NLL, played-action NLL)
        loss_weights: tuple = (1.0, 1.0, 1.0, 1.0)
        # evals
        quick_eval_every: int = 250
        quick_eval_games: int = 30
        deep_eval_every: int = 1000
        eval_sims: int = 64
        eval_games_per_pair: int = 4
        eval_last_n: int = 3
        eval_refresh_pairs: int = 10
        eval_opening_plies: int = 2
        eval_max_plies: int = 128
        eval_device: str = 'cpu'
        start_elo: float = 1000.0

    def _worker_cfg(cfg):
        return dict(fast_sims=cfg.fast_sims, full_sims=cfg.full_sims,
                    fast_prob=cfg.fast_prob, n_parallel=cfg.n_parallel_games,
                    wave_per_game=cfg.wave_per_game, temp=cfg.temp,
                    temp_threshold=cfg.temp_threshold, max_plies=cfg.max_plies,
                    root_select=cfg.root_select,
                    max_considered=cfg.max_considered)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    def build_network(cfg, device):
        sig = (cfg.channels, cfg.num_blocks, cfg.head_ch, cfg.head)
        return make_net(*sig).to(device), sig

    def cpu_clone(net, sig):
        c = make_net(*_sig4(sig))
        c.load_state_dict(c4._cpu_sd(net))
        c.eval()
        return c

    def _sig4(sig):
        """Accept a 3-tuple (channels, blocks, head_ch) from callers that
        predate the head option, defaulting to the spatial head."""
        return tuple(sig) if len(sig) == 4 else (tuple(sig) + ('spatial',))

    def save_benchmark_net(d, label, net):
        import os
        os.makedirs(d, exist_ok=True)
        sd = c4._cpu_sd(net)
        c4._assert_finite_sd(sd, f'bench_{label}.pt')
        torch.save(sd, os.path.join(d, f'bench_{label}.pt'))

    def load_benchmark_net(d, label, sig):
        import os
        net = make_net(*_sig4(sig))
        net.load_state_dict(torch.load(os.path.join(d, f'bench_{label}.pt'),
                                       map_location='cpu', weights_only=True))
        net.eval()
        return net

    def save_checkpoint(d, ep, net, optimizer, scheduler, hist, cfg=None,
                        elo_pool=None):
        import os
        os.makedirs(d, exist_ok=True)
        c4._assert_finite_sd(c4._cpu_sd(net), f'latest.pt at ep {ep}')
        blob = {'ep': ep, 'model': c4._cpu_sd(net),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict() if scheduler else None,
                'hist': hist, 'cfg': asdict(cfg) if cfg is not None else None,
                'elo': elo_pool.elo if elo_pool else {},
                'order': elo_pool.order if elo_pool else [],
                'pair_games': ({tuple(sorted(k)): v for k, v
                                in elo_pool.pair_games.items()}
                               if elo_pool else {})}
        tmp = os.path.join(d, 'latest.pt.tmp')
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(d, 'latest.pt'))

    def load_checkpoint(d):
        import os
        p = os.path.join(d, 'latest.pt')
        if not os.path.exists(p):
            return None
        blob = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(blob, dict) and 'model' not in blob:
            raise ValueError(f'{p} is a bare state_dict (a bench_N.pt), not a '
                             f'resumable checkpoint.')
        return blob

    def _elo_bot(game, net, device, sims, batch_size, rng, temp):
        return GMCTSBot(game, net, device, sims, batch_size=batch_size,
                        temp=temp, random_state=rng)

    def _elo_pick(root, rng):
        return root_pick(root, rng, thompson=False)

    # ── Training driver ───────────────────────────────────────────────────────
    def run_training(cfg, game=None, log=print):
        """Self-play + training, logging in the same shape as the ThompsonZero
        and AlphaZero drivers so the three runs read alike."""
        import random
        from collections import defaultdict
        game = game or c4.load_game(cfg.game_name)
        c4.set_game(game)
        device, backend = c4.pick_device(cfg.device_preference)
        c4.set_backend(backend, device)
        base_network, sig = build_network(cfg, device)
        optimizer = c4.LerpFreeAdamW(base_network.parameters(),
                                     lr=cfg.lr_peak,
                                     weight_decay=cfg.weight_decay)

        def lr_at(ep):
            if ep <= cfg.lr_warmup_eps:
                return ep / max(cfg.lr_warmup_eps, 1)
            t = min(1.0, (ep - cfg.lr_warmup_eps)
                    / max(cfg.lr_decay_eps - cfg.lr_warmup_eps, 1))
            return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        hist = {k: [] for k in (
            'ep', 'loss', 'klv', 'kla', 'nllv', 'nlla', 'v_mu', 'v_sd',
            'a_sd_pred', 'a_sd_tgt', 'lv_lo', 'lv_hi', 'calib', 'cov68',
            'rmse', 'draw_pct', 'plies',
            'buf', 'aux', 'elo', 'quick_ep', 'q_w', 'q_d', 'q_l')}
        elo_pool = c4.EloPool(
            game, cfg.eval_device, eval_sims=cfg.eval_sims,
            games_per_pair=cfg.eval_games_per_pair, last_n=cfg.eval_last_n,
            refresh_pairs=cfg.eval_refresh_pairs,
            opening_plies=cfg.eval_opening_plies,
            max_eval_plies=cfg.eval_max_plies, start_elo=cfg.start_elo,
            seed=cfg.seed, bot_factory=_elo_bot, pick=_elo_pick)
        replay, start_ep, aux_total = [], 1, 0

        ckpt = load_checkpoint(cfg.checkpoint_dir) if cfg.resume else None
        if ckpt is not None:
            base_network.load_state_dict(ckpt['model'])
            if ckpt.get('optim'):
                optimizer.load_state_dict(ckpt['optim'])
            if ckpt.get('sched'):
                scheduler.load_state_dict(ckpt['sched'])
            elo_pool.elo = dict(ckpt.get('elo') or {})
            elo_pool.elo.setdefault('random', cfg.start_elo)
            elo_pool.order = list(ckpt.get('order') or [])
            elo_pool.pair_games = {frozenset(k): v for k, v
                                   in (ckpt.get('pair_games') or {}).items()}
            kept = []
            for lb in list(elo_pool.order):
                try:
                    elo_pool.nets[lb] = load_benchmark_net(cfg.checkpoint_dir,
                                                           lb, sig)
                    kept.append(lb)
                except FileNotFoundError:
                    elo_pool.elo.pop(lb, None)
            elo_pool.order = kept
            elo_pool.players = ['random'] + list(kept)
            old = ckpt.get('hist') or {}
            for k in hist:
                if k in old:
                    hist[k] = old[k]
            start_ep = ckpt['ep'] + 1
            log(f'resumed at ep {ckpt["ep"]}')

        n_params = sum(p.numel() for p in base_network.parameters())
        log(f'device={device} backend={backend} | params={n_params:,} | '
            f'GAUSSIAN value-dist | start_ep={start_ep}')
        if not elo_pool.order:
            init = cpu_clone(base_network, sig)
            save_benchmark_net(cfg.checkpoint_dir, '0', init)
            elo_pool.add_checkpoint('0', init)

        self_play = ParallelSelfPlay(game, base_network, device,
                                     _worker_cfg(cfg), seed=cfg.seed)
        gen = self_play.episodes()
        prev = dict(self_play.stats)
        bar = c4.Progress(cfg.quick_eval_every, label='ep ')
        step_rng = random.Random(cfg.seed)
        ref = cpu_clone(base_network, sig)
        try:
            for ep in range(start_ep, cfg.num_episodes + 1):
                samples = next(gen)
                aux_total += self_play.last_aux
                replay.extend(samples)
                if len(replay) > cfg.max_buffer:
                    del replay[:-cfg.max_buffer]

                base_network.train()
                accs = defaultdict(list)
                if len(replay) >= cfg.batch_size:
                    for _ in range(cfg.train_steps_per_ep):
                        b = step_rng.sample(replay, cfg.batch_size)
                        try:
                            _lv, parts = train_step(
                                base_network, optimizer, b, device,
                                cfg.loss_weights, cfg.grad_clip,
                                ev_weight=cfg.ev_weight)
                        except RuntimeError as e:
                            log(f'  ! train step failed, skipped ({e})')
                            continue
                        for k, v in parts.items():
                            accs[k].append(v)
                    scheduler.step()
                base_network.eval()

                done = ep - (ep - 1) // cfg.quick_eval_every * cfg.quick_eval_every
                if ep % cfg.quick_eval_every != 0:
                    bar.update(done)
                    continue
                bar.close()

                st = self_play.stats
                dg = max(st['games'] - prev['games'], 1)
                draw_pct = 100 * (st['draw'] - prev['draw']) / dg
                plies = (st['plies'] - prev['plies']) / dg
                prev = dict(st)
                ml = lambda k: (float(np.mean(accs[k])) if accs[k]
                                else float('nan'))
                hist['ep'].append(ep)
                for k in ('loss', 'klv', 'kla', 'nllv', 'nlla', 'v_mu', 'v_sd',
                          'a_sd_pred', 'a_sd_tgt', 'lv_lo', 'lv_hi',
                          'calib', 'cov68', 'rmse'):
                    hist[k].append(ml(k))
                hist['draw_pct'].append(draw_pct); hist['plies'].append(plies)
                hist['buf'].append(len(replay)); hist['aux'].append(aux_total)
                nf = float(np.sum(accs['nonfinite'])) if accs['nonfinite'] else 0.0
                if nf:
                    log(f'  ! {nf:.0f} non-finite train steps SKIPPED — weights '
                        f'intact, but the log-variance is deliberately uncapped '
                        f'so check lv_hi below.')
                diag = (f'loss {ml("loss"):.3f} (klv {ml("klv"):.3f} '
                        f'kla {ml("kla"):.3f} nllv {ml("nllv"):.3f} '
                        f'nlla {ml("nlla"):.3f}) | discs: mu {ml("v_mu"):+.1f} '
                        f'sd {ml("v_sd"):.1f} a_sd {ml("a_sd_pred"):.1f}/'
                        f'{ml("a_sd_tgt"):.1f} rmse {ml("rmse"):.1f} | '
                        f'calib {ml("calib"):.2f} cov68 {ml("cov68"):.2f} | '
                        f'logvar [{ml("lv_lo"):.1f}, '
                        f'{ml("lv_hi"):.1f}] | dr {draw_pct:.0f}% ply {plies:.0f} '
                        f'buf {len(replay)//1000}k aux {aux_total}')

                if ep % cfg.deep_eval_every == 0:
                    snap = cpu_clone(base_network, sig)
                    save_benchmark_net(cfg.checkpoint_dir, str(ep), snap)
                    ref = snap
                    elo = elo_pool.add_checkpoint(str(ep), snap)
                    hist['elo'].append(dict(elo))
                    ladder = '  '.join(f'{k}={v:.0f}' for k, v in
                                       sorted(elo.items(),
                                              key=lambda kv: -kv[1])[:6])
                    log(f'ep {ep:6d} | {diag}')
                    log(f'         DEEP Elo@{cfg.eval_sims}: {ladder}')
                else:
                    w, d, l = quick_match(cpu_clone(base_network, sig), ref,
                                          game, cfg.quick_eval_games,
                                          cfg.eval_device,
                                          opening_plies=cfg.eval_opening_plies,
                                          max_plies=cfg.eval_max_plies)
                    hist['quick_ep'].append(ep); hist['q_w'].append(w)
                    hist['q_d'].append(d); hist['q_l'].append(l)
                    log(f'ep {ep:6d} | {diag} | vs last ckpt (no-MCTS) '
                        f'W{w} D{d} L{l}')
                save_checkpoint(cfg.checkpoint_dir, ep, base_network, optimizer,
                                scheduler, hist, cfg, elo_pool)
                bar.reset()
        finally:
            bar.close()
            self_play.shutdown()
        return hist


# ══════════════════════════════════════════════════════════════════════════════
#  Sequential halving at the root (Gumbel AlphaZero, without the Gumbel)
#
#  Gumbel AlphaZero (Danihelka et al., ICLR 2022) does two things at the root:
#  pick k candidate actions by adding Gumbel noise to the POLICY LOGITS and
#  taking the top k, then spend the simulation budget on those k by sequential
#  halving instead of by UCT.  The halving is what buys the policy-improvement
#  guarantee at small budgets; the Gumbel is only how you draw k actions without
#  replacement from a categorical policy.
#
#  This engine has no policy head to add Gumbel to — it has a posterior per
#  action.  A Thompson draw from those posteriors is already a randomised
#  ranking, and a better-founded one: its spread is the actual uncertainty about
#  each action's value rather than a fixed-scale noise, so it shrinks on its own
#  as evidence arrives.  So the top-m of one Thompson draw replaces Gumbel top-k,
#  and each halving round re-draws from the UPDATED beliefs.  Gumbel AlphaZero
#  has to bolt a growing sigma(Q) term onto a fixed g(a) to get that decay; here
#  it falls out of the representation.
#
#  Root only.  Below the root the tree keeps plain Thompson sampling, exactly as
#  Gumbel AlphaZero keeps its own interior rule.
#
#  One thing this does NOT inherit: Gumbel AlphaZero needs a completed-Q policy
#  target because visit counts stop being a valid improved policy once the budget
#  is allocated by a schedule rather than by value.  Our targets are the
#  backed-up distributions per action, which do not care how the visits were
#  allocated — only how precise they ended up.  So the training side is unchanged.
# ══════════════════════════════════════════════════════════════════════════════
def halving_schedule(n_sims, m, phases=None):
    """[(survivors, sims_each), …] for a budget of `n_sims` over `m` actions.

    Every phase gives each surviving action the same number of simulations and
    then keeps the better half, so the budget is spread over log2(m) rounds
    rather than concentrated by value the way Thompson sampling would."""
    m = max(int(m), 1)
    if m == 1:
        return [(1, max(int(n_sims), 0))]
    phases = phases or max(1, int(math.ceil(math.log2(m))))
    out, cur = [], m
    for _ in range(phases):
        each = max(1, int(n_sims // (phases * cur)))
        out.append((cur, each))
        if cur == 1:
            break
        cur = max(1, cur // 2)
    return out


def choose_considered(n_sims, k, max_considered=16, min_visits=2):
    """How many actions the halving should consider, given the budget.

    Gumbel AlphaZero's max_num_considered_actions exists because considering
    every action at a small budget is self-defeating: with 16 simulations over 8
    candidates each gets ONE before half are cut, so the first cut is decided by
    the prior draw rather than by evidence.  Pick the largest candidate set whose
    FIRST phase still gives each action `min_visits` simulations.

    At the benchmark's budgets this is inert (100 sims over 8 actions already
    gives 4 each); it only bites in the low-simulation regime the method is
    actually designed for."""
    k = int(k)
    if k <= 1:
        return k
    cap = min(k, int(max_considered))
    best = 2
    for m in range(2, cap + 1):
        phases = max(1, int(math.ceil(math.log2(m))))
        if n_sims // (phases * m) >= min_visits:
            best = m
    return min(best, cap)


def select_leaf_forced(root, root_state, first_idx, rng, temp=1.0):
    """One descent that takes `first_idx` at the root and Thompson-samples
    below it.  Sequential halving decides the root action; the rest of the tree
    is unchanged."""
    node, state, path = root, root_state.clone(), []
    idx = first_idx
    while True:
        node.vloss[idx] += 1
        node.n_vloss += 1
        path.append((node, idx))
        if node.term[idx] > _TERM_NONE:
            return path, None, (float(node.term[idx]), TERMINAL_VAR), None
        state.apply_action(int(node.legal[idx]))
        if state.is_terminal():
            s = othello_score(state, node.player)
            node.prove(idx, s)
            return path, None, (s, TERMINAL_VAR), None
        child = node.children[idx]
        if child is None:
            return path, state, None, (node, idx)
        node = child
        idx = int(sample_edges(node, rng, temp).argmax())


def halving_survivors(root, rng, cand, temp=1.0):
    """Keep the better half of `cand` by a FRESH Thompson draw from the updated
    beliefs.  Re-drawing rather than carrying a fixed noise offset is what makes
    the randomness decay automatically: an action that has collected simulations
    has a narrow belief, so its draw sits close to its mean."""
    if len(cand) <= 1:
        return list(cand)
    x = sample_edges(root, rng, temp)
    keep = max(1, len(cand) // 2)
    order = sorted(cand, key=lambda i: -x[i])
    return order[:keep]
