"""Maximum Occupancy Principle in the belief POMDP -- their Sec. 3.1, Eq. 6.

    V*(s, b) = ln sum_a exp[ beta H(s', b' | s, b, a)
                             + gamma sum_{s', b'} P(s', b' | s, b, a) V*(s', b') ]

with beta = 0 and gamma = 0.99 as in their simulations, V = 0 at death (E = 0),
and the softmax policy pi(a | s, b) = exp(Q(s, b, a) - V(s, b)) (their Eq. 8).

Beliefs live on their 0.1 grid; a propagated belief that falls between grid
points is handled by bilinear interpolation of V (their Appendix A.3).  At a
food source the observation is exact, so the observed component lands on 0 or
1 and only the other component needs interpolating.

With beta = 0 the only rewarded quantity is action entropy.  Every action is
available in every live state (bumping a wall is a legal action that leaves
the agent in place), so MOP sees no difference between a corner and the open
grid; what it values is staying alive with its options open.
"""

import numpy as np
import scipy.sparse as sp

from .mylonas import EGAIN, EMAX, FOOD, MOVE, NA, NPOS, THEIRS, gate_g

NB = 11                                   # belief grid 0, 0.1, ..., 1
BG = np.linspace(0.0, 1.0, NB)
NE = EMAX + 1
NS = NPOS * NE * NB * NB


def sidx(x, E, i1, i2):
    return ((x * NE + E) * NB + i1) * NB + i2


def _corners(b):
    """Bilinear corners on the belief grid: (lo, hi, weight on hi)."""
    f = np.clip(b, 0.0, 1.0) * (NB - 1)
    lo = np.minimum(np.floor(f + 1e-12).astype(int), NB - 2)
    return lo, lo + 1, f - lo


def build_transitions(world=THEIRS, gated=False, return_entropy=False):
    """One sparse (NS x NS) matrix per action: P(s', b' | s, b, a).

    Rows for dead states are empty; their value is pinned to zero.

    With return_entropy, also returns H (NA, NS): the entropy (nats) of the
    agent's next state, H(s', b' | s, b, a), over the TRUE outcomes -- the
    landing cell and, on a source, whether it was full.  The belief-grid
    interpolation weights are a numerical device, not outcomes, so they do not
    enter H.  This is the beta term of their Eq. 6.
    """
    E = np.arange(1, NE)[:, None, None]                     # alive energies
    I1, I2 = np.meshgrid(np.arange(NB), np.arange(NB), indexing='ij')
    I1 = np.broadcast_to(I1[None], (NE - 1, NB, NB))
    I2 = np.broadcast_to(I2[None], (NE - 1, NB, NB))
    Eb = np.broadcast_to(E, (NE - 1, NB, NB))
    B1, B2 = BG[I1], BG[I2]
    g = (np.array([gate_g(e) for e in range(1, NE)])[:, None, None]
         if gated else np.zeros((NE - 1, 1, 1)))

    mats = []
    H = np.zeros((NA, NPOS, NE, NB, NB))

    def ent(p):
        p = np.broadcast_to(p, (NE - 1, NB, NB))
        return np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)

    for a in range(NA):
        rows, cols, vals = [], [], []

        def add(src, x2, E2, j1, j2, w):
            rows.append(src.ravel())
            cols.append(sidx(x2, np.broadcast_to(E2, src.shape),
                             np.broadcast_to(j1, src.shape),
                             np.broadcast_to(j2, src.shape)).ravel())
            vals.append(np.broadcast_to(w, src.shape).ravel())

        for x in range(NPOS):
            src = sidx(x, Eb, I1, I2)
            p1 = world.mu + (world.stay(x, 0) - world.mu) * B1     # predicted beliefs
            p2 = world.mu + (world.stay(x, 1) - world.mu) * B2

            dest = {}
            for a2 in range(NA):
                w = (1.0 - g) * (a2 == a) + g / NA
                x2 = int(MOVE[a2, x])
                dest[x2] = dest.get(x2, 0.0) + w

            for x2, wpos in dest.items():
                E_plain = Eb - 1
                if x2 in FOOD:
                    i = FOOD.index(x2)
                    p_full = p1 if i == 0 else p2
                    H[a, x, 1:] += ent(wpos * p_full) + ent(wpos * (1.0 - p_full))
                    other = p2 if i == 0 else p1
                    lo, hi, wh = _corners(other)
                    E_fed = np.minimum(Eb - 1 + EGAIN, EMAX)
                    for E2, pout, obs in ((E_fed, p_full, NB - 1),
                                          (E_plain, 1.0 - p_full, 0)):
                        for jo, wo in ((lo, 1.0 - wh), (hi, wh)):
                            w = wpos * pout * wo
                            if i == 0:
                                add(src, x2, E2, obs, jo, w)
                            else:
                                add(src, x2, E2, jo, obs, w)
                else:
                    H[a, x, 1:] += ent(wpos)
                    l1, h1, w1 = _corners(p1)
                    l2, h2, w2 = _corners(p2)
                    for j1, u1 in ((l1, 1.0 - w1), (h1, w1)):
                        for j2, u2 in ((l2, 1.0 - w2), (h2, w2)):
                            add(src, x2, E_plain, j1, j2, wpos * u1 * u2)

        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        keep = v > 0
        T = sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(NS, NS))
        mats.append(T)
    if return_entropy:
        return mats, H.reshape(NA, NS)
    return mats


def alive_mask():
    m = np.ones((NPOS, NE, NB, NB), dtype=bool)
    m[:, 0] = False
    return m.ravel()


def action_mask(walls='legal'):
    """(NA, NS) availability.  Their text allows two readings at a wall:

    'legal'      bumping a wall is an action that leaves the agent in place
                 (their x'(s, a) definition), so all five actions count
    'redundant'  only actions that change the position, plus STAY, are
                 available, as their dead state ('the only available action
                 is to stay') suggests action sets can depend on the state
    """
    m = np.ones((NA, NPOS, NE, NB, NB), dtype=bool)
    if walls == 'redundant':
        for x in range(NPOS):
            for a in range(NA - 1):
                if MOVE[a, x] == x:
                    m[a, x] = False
    return m.reshape(NA, NS)


def value_iteration(mats, gamma=0.99, tol=1e-2, max_iter=3000, verbose=False,
                    walls='legal', beta=0.0, H=None, reward=None, alpha=1.0):
    """Their Eq. 6.  Returns V (NS,) and Q (NA, NS).

    Q includes the beta * H term, so the policy is softmax(Q) for any beta.
    (Their Eq. 7 writes Q without it, which is the same thing at beta = 0, the
    only value they simulate.)  beta < 0 penalises unpredictable transitions,
    which they note approximates empowerment; beta > 0 rewards them.
    """
    alive = alive_mask()
    avail = action_mask(walls)
    bH = np.zeros((len(mats), NS)) if (beta == 0.0 or H is None) else beta * H
    if reward is not None:                       # per-state reward, e.g. empowerment
        bH = bH + reward[None, :]
    V = np.zeros(NS)
    for it in range(max_iter):
        Q = bH + np.stack([gamma * (T @ V) for T in mats])
        Q = np.where(avail, Q, -np.inf)
        m = Q.max(axis=0)
        if alpha > 0:
            Vn = m + alpha * np.log(np.exp((Q - m) / alpha).sum(axis=0))
        else:                                    # alpha = 0: hard Bellman max
            Vn = m
        Vn[~alive] = 0.0
        delta = float(np.abs(Vn - V).max())
        V = Vn
        if verbose and it % 50 == 0:
            print('    iter %4d  max change %.4f' % (it, delta), flush=True)
        if delta < tol:
            break
    Q = bH + np.stack([gamma * (T @ V) for T in mats])
    Q = np.where(avail, Q, -np.inf)
    return V, Q, it + 1


class MOPAgent:
    """Samples from the MOP softmax policy at the current (x, E, b).

    greedy=True takes the argmax instead.  That is not MOP -- MOP's optimal
    policy is the softmax -- it is a diagnostic that keeps MOP's values and
    removes its behavioural variability.
    """

    belief_step = 0.0

    def __init__(self, Q, greedy=False, alpha=1.0):
        self.greedy = greedy
        self.alpha = alpha
        # unavailable actions carry -inf; a finite floor keeps the bilinear
        # interpolation free of 0 * inf while still giving them zero probability
        self.Q = np.where(np.isfinite(Q), Q, -1e9).reshape(NA, NPOS, NE, NB, NB)

    def q(self, x, E, b1, b2):
        t = self.Q[:, x, int(np.clip(E, 0, EMAX))]
        f1, f2 = np.clip(b1, 0, 1) * (NB - 1), np.clip(b2, 0, 1) * (NB - 1)
        i, j = min(int(f1), NB - 2), min(int(f2), NB - 2)
        u, v = f1 - i, f2 - j
        return ((1 - u) * (1 - v) * t[:, i, j] + u * (1 - v) * t[:, i + 1, j]
                + (1 - u) * v * t[:, i, j + 1] + u * v * t[:, i + 1, j + 1])

    def policy(self, x, E, b1, b2):
        q = self.q(x, E, b1, b2)
        p = np.exp((q - q.max()) / self.alpha)
        return p / p.sum()

    def act(self, x, E, b1, b2, rng, gated=False):
        if self.greedy:
            q = self.q(x, E, b1, b2)
            return int(rng.choice(np.flatnonzero(q >= q.max() - 1e-9)))
        return int(rng.choice(NA, p=self.policy(x, E, b1, b2)))
