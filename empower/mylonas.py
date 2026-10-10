"""Replication of the Mylonas & Moreno-Bote two-food-source POMDP
(arXiv:2609.39342v1, IWAI 2026), with empowerment run as four arms.

Their paper reports that empowerment "reaches a food source and performs
limited local exploration", qualitatively like Active Inference, while MOP
alternates between the two sources with an energy-dependent regime switch.
Their empowerment arm is (a) the empowerment LEVEL, greedily ascended, and
(b) computed in a world where energy gates nothing except death.  Those are
exactly the two ingredients Part I identifies as necessary for alternation.

This module runs the 2x2:

    {level, gain} x {no actuator gate, graded actuator gate g(E)}

Arms with no gate run in THEIR environment, unmodified, so the numbers are
directly comparable to their Table/Fig. 2-3 values.  Arms with the gate add
one ingredient to the environment (the body throttles actuators, as in
Part I); nothing is added to the objective, and no mixing weight exists in
any arm.

Environment (their Sec. 2 and Appendix A.2):
  5x5 grid, food at (1,1) and (5,5) -> flat indices 0 and 24
  E in 0..30, Egain = 15, E = 0 is death (absorbing, no actions)
  hidden h_i in {0,1} per source, p(h'=1|h=1) = lam, p(h'=1|h=0) = mu
  observations exact at a source, uniform noise away from one
"""

import itertools
import numpy as np

from .arena import ACTIONS, NA, g_noise
from .empowerment import blahut_arimoto

GRID = 5
NPOS = GRID * GRID
FOOD = (0, NPOS - 1)              # their (1,1) and (5,5), 1-indexed
EMAX = 30
EGAIN = 15
LAM = 0.8                          # p(h'=1 | h=1), away from the source
MU = 0.2                           # p(h'=1 | h=0)
RHO = 0.8                          # p(h'=1 | h=1), standing on the source


class World:
    """Food dynamics, in their parameterisation (their Sec. 2.2).

    lam  p(h'=1 | h=1) with the agent away from the source
    rho  p(h'=1 | h=1) with the agent standing on it   (their 'depletion at visit')
    mu   p(h'=1 | h=0) anywhere                         (refill)

    Their location-independent and depletion conditions are both rho = lam = 0.8,
    i.e. the same world.  The h-transition reads the agent's CURRENT position
    (their Eq. 1), so standing on a source is what depletes it.
    """

    def __init__(self, lam=LAM, mu=MU, rho=RHO, name='theirs'):
        self.lam, self.mu, self.rho = float(lam), float(mu), float(rho)
        self.name = name

    def __repr__(self):
        return 'World(%s: lam=%g mu=%g rho=%g)' % (self.name, self.lam, self.mu, self.rho)

    def stay(self, x, i):
        return self.rho if x == FOOD[i] else self.lam

    def predict(self, b, x, i):
        """One-step belief prediction from position x (no observation)."""
        return self.mu + (self.stay(x, i) - self.mu) * b

    def stationary_away(self):
        d = 1.0 - self.lam + self.mu
        return self.mu / d if d > 0 else 1.0

    def propagate(self, b, x, i, tau):
        """Belief that source i is full after tau steps starting at x.

        The first step reads x (which may be the source itself); the rest are
        taken away from it, so they use lam.  Closed form for the away chain.
        """
        if tau <= 0:
            return b
        b = self.predict(b, x, i)
        pi = self.stationary_away()
        return pi + (b - pi) * (self.lam - self.mu) ** (tau - 1)


THEIRS = World()


def rc(p):
    return divmod(p, GRID)


def pidx(r, c):
    return r * GRID + c


def manhattan(p, q):
    (r1, c1), (r2, c2) = rc(p), rc(q)
    return abs(r1 - r2) + abs(c1 - c2)


DIST = np.array([[manhattan(p, q) for q in range(NPOS)] for p in range(NPOS)])
DIST_FOOD = np.minimum(DIST[:, FOOD[0]], DIST[:, FOOD[1]])


def gate_g(E, b0=0.40, k=14.0):
    """Actuator corruption probability as a function of energy.

    Identical functional form to Part I's body gate, with body integrity read
    off as E/EMAX.  g ~ 0 on a high-energy plateau, steep collapse near
    0.4*EMAX = 12, g -> 1 as the agent empties.  This is the ONLY difference
    between the gated and ungated environments.
    """
    return float(g_noise(np.asarray(E, float) / EMAX, b0, k))


# ---------------------------------------------------------------------------
# environment tensors
# ---------------------------------------------------------------------------

def move_table():
    """D[a, p] -> next position under action a, ignoring the energy freeze."""
    D = np.zeros((NA, NPOS), dtype=int)
    for a, (dr, dc) in enumerate(ACTIONS):
        for p in range(NPOS):
            r, c = rc(p)
            nr, nc = r + dr, c + dc
            D[a, p] = pidx(nr, nc) if (0 <= nr < GRID and 0 <= nc < GRID) else p
    return D


MOVE = move_table()


def h_kernels(world=THEIRS):
    """Mh[p] : 4x4 kernel on h = (h1, h2) flattened as h1*2 + h2, given position p."""
    Mh = np.zeros((NPOS, 4, 4))
    for p in range(NPOS):
        per = []
        for i in (0, 1):
            stay = world.stay(p, i)
            m = np.array([[1.0 - world.mu, world.mu],      # from h=0
                          [1.0 - stay, stay]])     # from h=1
            per.append(m)
        Mh[p] = np.kron(per[0], per[1])
    return Mh


def obs_kernel():
    """Q[p, h, w] = q(omega | x = p, h); w and h flattened as i1*2 + i2."""
    Q = np.zeros((NPOS, 4, 4))
    for p in range(NPOS):
        for h in range(4):
            hs = (h >> 1, h & 1)
            for w in range(4):
                ws = (w >> 1, w & 1)
                pr = 1.0
                for i in (0, 1):
                    if p == FOOD[i]:
                        pr *= 1.0 if ws[i] == hs[i] else 0.0
                    else:
                        pr *= 0.5
                Q[p, h, w] = pr
    return Q


OBSQ = obs_kernel()


def energy_next(E, p, h):
    """E'(E, x', h') -- their Sec. 2.2.  Death is absorbing."""
    if E <= 0:
        return 0
    base = E - 1
    for i in (0, 1):
        if p == FOOD[i] and ((h >> (1 - i)) & 1) == 1:
            return min(base + EGAIN, EMAX)
    return base


def reachable_energies(E0, n):
    """Energies reachable within n steps.

    Energies that first appear on step n are never sources of a further
    remap, so the grid is closed over everything the propagation touches;
    _step_tensors self-maps the handful of leaf-only values.
    """
    cur, seen = {int(E0)}, {int(E0)}
    for _ in range(n):
        nxt = set()
        for e in cur:
            if e <= 0:
                nxt.add(0)
            else:
                nxt.add(e - 1)
                nxt.add(min(e - 1 + EGAIN, EMAX))
        cur = nxt
        seen |= nxt
    return sorted(seen)


# ---------------------------------------------------------------------------
# n-step empowerment C(x, E, b) in the belief-POMDP
# ---------------------------------------------------------------------------
# Their Eq. 15: C = max_{p(a^n)} I(A^n ; o_{t+n}) with o = (s_{t+n}, omega_{t+n}),
# beliefs propagated open-loop (their Eq. 18, no conditioning on intermediate
# observations).  We propagate the exact joint over (x, E, h) rather than their
# factorised P(s|a^n) P(omega|s, b) approximation, which if anything gives
# empowerment MORE signal than they allow it.
#
# The propagation is linear in the initial h-distribution, so we carry the four
# basis initial h-states through the action-sequence prefix tree once and then
# read off every belief as a linear combination at the leaves.  Belief costs
# nothing but a Blahut-Arimoto run.


def _step_ops(E_grid, gated, world):
    """Operators for one environment step, in the layout A[pos, h, energy, batch].

    Every factor is either a tiny batched matmul (the 4x4 hidden-state kernel)
    or a scatter (positions and energies both move deterministically, so the
    expensive PxP and nExnE contractions are never formed).
    """
    nE = len(E_grid)
    idx = {E: i for i, E in enumerate(E_grid)}
    Mh = h_kernels(world).transpose(0, 2, 1).copy()      # (P, H, H), pre-transposed

    g = np.array([gate_g(E) if (gated and E > 0) else 0.0 for E in E_grid])
    frozen = np.array([E <= 0 for E in E_grid])              # dead: no movement

    # energy scatters: leaf-only energies self-map (never sources, see above)
    shift = np.array([idx.get(max(E - 1, 0), i) if E > 0 else i
                      for i, E in enumerate(E_grid)])
    eat = np.array([idx.get(min(E - 1 + EGAIN, EMAX), i) if E > 0 else i
                    for i, E in enumerate(E_grid)])
    eat_mask = np.zeros((NPOS, 4), dtype=bool)
    for i in (0, 1):
        for h in range(4):
            if (h >> (1 - i)) & 1:
                eat_mask[FOOD[i], h] = True
    return dict(Mh=Mh, g=g, frozen=frozen, shift=shift, eat=eat,
                eat_mask=eat_mask, nE=nE)


def _scatter_pos(A, mapping, out=None):
    """out[mapping[p]] += A[p] along the leading position axis."""
    out = np.zeros_like(A) if out is None else out
    out[...] = 0.0
    for p in range(NPOS):
        out[mapping[p]] += A[p]
    return out


def _scatter_pos_uniform(A, out=None):
    """The action-averaged position kernel, formed once per step."""
    out = np.zeros_like(A) if out is None else out
    out[...] = 0.0
    for a in range(NA):
        for p in range(NPOS):
            out[MOVE[a, p]] += A[p]
    out /= NA
    return out


def _scatter_energy(A, mapping, out=None):
    """out[..., mapping[e], :] += A[..., e, :] along the energy axis."""
    out = np.zeros_like(A) if out is None else out
    out[...] = 0.0
    for e in range(A.shape[2]):
        out[:, :, mapping[e]] += A[:, :, e]
    return out


def _advance_all(A, ops):
    """h-step once, then the five position branches, then the energy step.

    Branches are written straight into the output slab rather than held as a
    list: the prefix tree is wide at depth n and five live copies of it is the
    difference between fitting in memory and not.
    """
    P, H, nE, KB = A.shape
    A = np.matmul(ops['Mh'], A.reshape(P, H, nE * KB)).reshape(P, H, nE, KB)

    gm = ops['g'][None, None, :, None]
    fz = ops['frozen'][None, None, :, None]
    mask = ops['eat_mask'][:, :, None, None]
    mix = _scatter_pos_uniform(A)

    out = np.empty((P, H, nE, NA * KB))
    br = np.empty_like(A)
    sh = np.empty_like(A)
    et = np.empty_like(A)
    for a in range(NA):
        _scatter_pos(A, MOVE[a], out=br)
        np.copyto(br, np.where(fz, A, (1.0 - gm) * br + gm * mix))
        _scatter_energy(br, ops['shift'], out=sh)
        _scatter_energy(br, ops['eat'], out=et)
        out[:, :, :, a * KB:(a + 1) * KB] = np.where(mask, et, sh)
    return out


def outcome_basis(x0, E0, n=5, gated=False, world=THEIRS):
    """P(o_{t+n} | a^n, x0, E0, h0) for each of the four basis initial h states.

    Linear in the initial hidden-state distribution, so one propagation serves
    every belief: a belief is read off at the leaves as a linear combination.
    Returns B with shape (4, NA**n, NPOS * nE * 4) and the energy grid.
    """
    E_grid = reachable_energies(E0, n)
    ops = _step_ops(E_grid, gated, world)
    nE = ops['nE']

    A = np.zeros((NPOS, 4, nE, 4))                 # batch axis = the 4 basis states
    A[x0, np.arange(4), E_grid.index(int(E0)), np.arange(4)] = 1.0

    for _ in range(n):
        A = _advance_all(A, ops)                   # batch grows x NA per step

    # batch axis is ordered (a_1, ..., a_n) outermost-first, basis innermost
    A = A.reshape(NPOS, 4, nE, NA ** n, 4)
    out = np.einsum('phebk,phw->kbpew', A, OBSQ, optimize=True)
    return out.reshape(4, NA ** n, NPOS * nE * 4), E_grid


def empowerment_at(x0, E0, beliefs, n=5, gated=False, world=THEIRS,
                   max_iter=80, tol=1e-7):
    """C(x0, E0, b) in bits for a list of beliefs b = (b1, b2)."""
    B, _ = outcome_basis(x0, E0, n=n, gated=gated, world=world)

    keep = B.reshape(-1, B.shape[-1]).sum(axis=0) > 1e-12       # live outcomes only
    B = np.ascontiguousarray(B[:, :, keep])

    # Rows identical across all four basis components are identical for every
    # belief, so dedupe once and reuse.  Merging duplicate input rows leaves
    # the channel capacity unchanged.
    flat = np.ascontiguousarray(B.transpose(1, 0, 2)).reshape(B.shape[1], -1)
    _, uniq = np.unique(np.round(flat, 12), axis=0, return_index=True)
    Bu = np.ascontiguousarray(B[:, uniq, :])

    out = []
    for b1, b2 in beliefs:
        w = np.array([(1 - b1) * (1 - b2), (1 - b1) * b2,
                      b1 * (1 - b2), b1 * b2])
        P = np.tensordot(w, Bu, axes=(0, 0))
        cap, _ = blahut_arimoto(P, max_iter=max_iter, tol=tol)
        out.append(cap)
    return np.array(out)


# ---------------------------------------------------------------------------
# empowerment table and lookup
# ---------------------------------------------------------------------------
# C is computed exactly on (position, energy) and on a 5x5 belief grid, then
# interpolated bilinearly in belief -- the same device their Appendix A.3 uses
# for the value function.  Checked against the exact 0.1 belief grid at the
# hardest cell (a food source, where belief matters most): max error 0.004 bits.

BELIEF_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
BELIEF_PAIRS = [(float(a), float(b)) for a in BELIEF_GRID for b in BELIEF_GRID]


def _table_job(job):
    x0, E0, n, gated, world = job
    if E0 <= 0:
        return x0, E0, np.zeros(len(BELIEF_PAIRS))      # dead: no actions, no control
    return x0, E0, empowerment_at(x0, E0, BELIEF_PAIRS, n=n,
                                  gated=gated, world=world)


def build_table(n=5, gated=False, world=THEIRS, workers=None, progress=None):
    """C[position, energy, b1, b2] in bits over the full state space."""
    from concurrent.futures import ProcessPoolExecutor
    nb = len(BELIEF_GRID)
    table = np.zeros((NPOS, EMAX + 1, nb, nb))
    jobs = [(x, E, n, gated, world)
            for x in range(NPOS) for E in range(EMAX + 1)]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for x0, E0, vals in ex.map(_table_job, jobs, chunksize=4):
            table[x0, E0] = vals.reshape(nb, nb)
            done += 1
            if progress and done % progress == 0:
                print('  %d/%d' % (done, len(jobs)), flush=True)
    return table


class MylonasLUT:
    """C(x, E, b1, b2), exact in (x, E) and bilinear in belief."""

    def __init__(self, table):
        self.table = table
        self.gr = BELIEF_GRID
        self.step = float(BELIEF_GRID[1] - BELIEF_GRID[0])

    @classmethod
    def load(cls, path):
        return cls(np.load(path)['table'])

    def C(self, x, E, b1, b2):
        E = int(np.clip(E, 0, EMAX))
        t = self.table[x, E]
        f1 = np.clip(b1, 0.0, 1.0) / self.step
        f2 = np.clip(b2, 0.0, 1.0) / self.step
        i = min(int(f1), len(self.gr) - 2)
        j = min(int(f2), len(self.gr) - 2)
        u, v = f1 - i, f2 - j
        return float((1 - u) * (1 - v) * t[i, j] + u * (1 - v) * t[i + 1, j]
                     + (1 - u) * v * t[i, j + 1] + u * v * t[i + 1, j + 1])


# ---------------------------------------------------------------------------
# the four arms
# ---------------------------------------------------------------------------
# Every arm uses the SAME planner, the same discount, the same empowerment
# functional, and has no mixing weight.  Only two things vary:
#   drive  'level' -- value the empowerment you would HAVE at the target
#          'gain'  -- value the CHANGE in empowerment the trip causes
#   gated  whether the body throttles the actuators (Part I's g), in both the
#          agent's model and the world
# gamma is shared by all four arms, so no arm difference can be attributed to it.

class MylonasAgent:
    """One planner, four arms.

    drive='level'      their arm: one-step greedy ascent on C over successors,
                       the standard empowerment policy (no discount -- a level
                       drive values states, so there is nothing to discount).
    drive='level_plan' the same drive run through the gain arm's target planner,
                       reported only to show the two readings agree.
    drive='gain'       value a trip by the change in C it causes, discounted by
                       travel time: gamma^tau * (C_arrival - C_now).  Staying is
                       worth exactly zero, so the agent moves iff some target
                       improves its control by more than the trip costs.

    gamma is shared, and no arm has a mixing weight.
    """

    def __init__(self, lut: MylonasLUT, drive='gain', gamma=0.9, belief_step=0.0,
                 world=THEIRS, deliberate_entry=True):
        """deliberate_entry: a food cell may be entered only when it is the
        planner's chosen target, never in passing.  The two cells beside a
        source are often tied, and half the shortest paths between them cross
        the corner the food sits in, so without this an agent can eat purely
        through tie-breaking.  Used to separate chosen meals from accidental
        ones; affects the planner arms only.
        """
        assert drive in ('level', 'level_plan', 'gain')
        self.lut = lut
        self.drive = drive
        self.gamma = gamma
        self.belief_step = belief_step
        self.world = world
        self.deliberate_entry = deliberate_entry
        self.last_target = None

    # --- one-step lookahead, used by the greedy level arm ------------------
    def successor_value(self, x, E, b1, b2, a, gated):
        """E[ C(successor) ], averaging over the food state and the gate."""
        if E <= 0:
            return 0.0
        if gated:
            g = gate_g(E)
            dest = {int(MOVE[a, x]): 1.0 - g}
            for aa in range(NA):
                q = int(MOVE[aa, x])
                dest[q] = dest.get(q, 0.0) + g / NA
        else:
            dest = {int(MOVE[a, x]): 1.0}

        p1 = self.world.predict(b1, x, 0)
        p2 = self.world.predict(b2, x, 1)
        tot = 0.0
        for x2, w in dest.items():
            if x2 == FOOD[0]:
                tot += w * (p1 * self.lut.C(x2, min(E - 1 + EGAIN, EMAX), 1.0, p2)
                            + (1 - p1) * self.lut.C(x2, E - 1, 0.0, p2))
            elif x2 == FOOD[1]:
                tot += w * (p2 * self.lut.C(x2, min(E - 1 + EGAIN, EMAX), p1, 1.0)
                            + (1 - p2) * self.lut.C(x2, E - 1, p1, 0.0))
            else:
                tot += w * self.lut.C(x2, E - 1, p1, p2)
        return tot

    # --- target planner, used by the gain arm ------------------------------
    def values(self, x, E, b1, b2):
        C_now = self.lut.C(x, E, b1, b2)
        out = np.empty(NPOS)
        for c in range(NPOS):
            # STAY is a real one-step action: it costs a step of energy and the
            # food state moves on, exactly as for a one-step move.
            tau = max(int(DIST[x, c]), 1)
            p1 = self.world.propagate(b1, x, 0, tau)
            p2 = self.world.propagate(b2, x, 1, tau)
            E_arr = max(E - tau, 0)

            if E < tau:
                # starves before arriving: no meal, no control
                out[c] = (self.gamma ** tau) * (0.0 if self.drive == 'level_plan'
                                                else -C_now)
                continue
            if c == FOOD[0]:
                arr = (p1 * self.lut.C(c, min(E_arr + EGAIN, EMAX), 1.0, p2)
                       + (1 - p1) * self.lut.C(c, E_arr, 0.0, p2))
            elif c == FOOD[1]:
                arr = (p2 * self.lut.C(c, min(E_arr + EGAIN, EMAX), p1, 1.0)
                       + (1 - p2) * self.lut.C(c, E_arr, p1, 0.0))
            else:
                arr = self.lut.C(c, E_arr, p1, p2)

            val = arr if self.drive == 'level_plan' else arr - C_now
            out[c] = (self.gamma ** tau) * val
        return out

    def act(self, x, E, b1, b2, rng, gated=False):
        if self.drive == 'level':
            q = [self.successor_value(x, E, b1, b2, a, gated) for a in range(NA)]
            q = np.asarray(q)
            return int(rng.choice(np.flatnonzero(q >= q.max() - 1e-12)))

        v = self.values(x, E, b1, b2)
        c = int(rng.choice(np.flatnonzero(v >= v.max() - 1e-12)))
        self.last_target = c
        if c == x:
            return NA - 1                                   # STAY
        steps = [a for a in range(NA - 1)
                 if DIST[MOVE[a, x], c] == DIST[x, c] - 1]
        if self.deliberate_entry:
            clean = [a for a in steps if not (MOVE[a, x] in FOOD and MOVE[a, x] != c)]
            steps = clean or steps
        return int(rng.choice(steps)) if steps else NA - 1


# ---------------------------------------------------------------------------
# episodes and the metrics their paper reports
# ---------------------------------------------------------------------------

def simulate(agent, gated=False, world=THEIRS, seed=0,
             max_steps=3000, x0=NPOS // 2, E0=EMAX):
    """One episode in their environment (plus the gate, if gated).

    Initial conditions follow their Table 1: both sources full, both beliefs
    certain.  The start cell is the grid centre, equidistant from the two
    sources, so no arm is handed a head start toward either.

    Beliefs are kept continuous (belief_step = 0).  Rounding them to a 0.1 grid
    each step, as an earlier version did, creates spurious fixed points
    wherever the per-step change is below 0.05 (a source last seen empty
    sticks at 0.4 in their world instead of relaxing to 0.5).  Their Appendix
    A.3 discretises the value function and interpolates off-grid beliefs; the
    lookup table here does the same.
    """
    rng = np.random.default_rng(seed)
    x, E = x0, E0
    h = [1, 1]
    b = [1.0, 1.0]
    rec = dict(x=[], E=[], b1=[], b2=[], act=[])

    for _ in range(max_steps):
        a = agent.act(x, E, b[0], b[1], rng, gated=gated)
        rec['x'].append(x)
        rec['E'].append(E)
        rec['b1'].append(b[0])
        rec['b2'].append(b[1])
        rec['act'].append(a)

        # hidden food state, using the CURRENT position (their Eq. 1)
        stay = [world.stay(x, i) for i in (0, 1)]
        h = [1 if rng.random() < (stay[i] if h[i] else world.mu) else 0 for i in (0, 1)]

        # position: frozen at death, otherwise gated by the body
        if E <= 0:
            x2 = x
        else:
            a_eff = a
            if gated and rng.random() < gate_g(E):
                a_eff = int(rng.integers(NA))
            x2 = MOVE[a_eff, x]

        E = energy_next(E, x2, h[0] * 2 + h[1])

        # observe, then update the belief (predict, then condition)
        for i in (0, 1):
            pred = world.mu + (stay[i] - world.mu) * b[i]
            if x2 == FOOD[i]:
                w = h[i]                        # exact at the source
                b[i] = float(w)
            else:
                b[i] = pred
            if agent.belief_step:
                b[i] = round(b[i] / agent.belief_step) * agent.belief_step
        x = x2
        if E <= 0:
            break

    for k in rec:
        rec[k] = np.asarray(rec[k])
    rec['died'] = bool(E <= 0)
    return rec


def episode_metrics(rec):
    x = rec['x']
    T = len(x)
    on_food = np.isin(x, FOOD)
    d = DIST_FOOD[x]

    trips, last = 0, None
    for p in x:
        if p in FOOD:
            if last is not None and p != last:
                trips += 1
            last = p

    counts = np.bincount(x, minlength=NPOS).astype(float)
    q = counts / counts.sum()
    nz = q[q > 0]
    return dict(
        steps=T,
        died=rec['died'],
        trips_per_1k=1000.0 * trips / T,
        frac_on_food=float(on_food.mean()),
        frac_far=float((d >= 2).mean()),
        both_sources=bool(np.isin(FOOD[0], x) and np.isin(FOOD[1], x)),
        cells_visited=int((counts > 0).sum()),
        state_entropy=float(-(nz * np.log(nz)).sum()),
        mean_E=float(rec['E'].mean()),
    )


def run_arm(lut, drive, gated, n_episodes=30, seed0=0, gamma=0.9,
            world=THEIRS, max_steps=3000):
    agent = MylonasAgent(lut, drive=drive, gamma=gamma, world=world)
    recs, mets = [], []
    for k in range(n_episodes):
        r = simulate(agent, gated=gated, world=world,
                     seed=seed0 + k, max_steps=max_steps)
        recs.append(r)
        mets.append(episode_metrics(r))
    agg = {}
    for key in mets[0]:
        vals = np.array([m[key] for m in mets], dtype=float)
        agg[key] = float(vals.mean())
        agg[key + '_sem'] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    agg['n_episodes'] = n_episodes
    return agg, recs


def empowerment_factorized(x0, E0, b, n=5, gated=False, world=THEIRS,
                           max_iter=400, tol=1e-9):
    """C computed with THEIR factorisation of the predictive model (Eq. 17).

    They write P(o | a^n, s, b) = P(s_{t+n} | a^n, s, b) P(omega | s_{t+n}, b_{t+n}),
    which drops the correlation between the future energy and the hidden food
    state -- if you ate, the food was there, and the observation is then no
    longer independent of the energy you arrive with.  Keeping the exact joint
    (the default everywhere else here) gives empowerment strictly more signal;
    this function exists to check whether the approximation is what separates
    our replication from their reported result.

    The product is quadratic in the belief, so unlike the exact path this
    cannot reuse one propagation across beliefs.
    """
    b1, b2 = b
    E_grid = reachable_energies(E0, n)
    ops = _step_ops(E_grid, gated, world)
    nE = ops['nE']

    A = np.zeros((NPOS, 4, nE, 1))
    for h in range(4):
        w = (b1 if (h >> 1) else 1 - b1) * (b2 if (h & 1) else 1 - b2)
        A[x0, h, E_grid.index(int(E0)), 0] = w
    for _ in range(n):
        A = _advance_all(A, ops)

    A = A.reshape(NPOS, 4, nE, NA ** n)
    Pxe = A.sum(axis=1)                                  # (P, nE, B)

    # belief propagated n steps, open loop
    q1, q2 = world.propagate(b1, x0, 0, n), world.propagate(b2, x0, 1, n)
    bw = np.array([(1 - q1) * (1 - q2), (1 - q1) * q2, q1 * (1 - q2), q1 * q2])
    Pw = np.einsum('h,phw->pw', bw, OBSQ)                # (P, 4)

    P = np.einsum('peb,pw->bpew', Pxe, Pw).reshape(NA ** n, -1)
    keep = P.sum(axis=0) > 1e-12
    P = np.ascontiguousarray(P[:, keep])
    P /= P.sum(axis=1, keepdims=True)
    return blahut_arimoto(P, max_iter=max_iter, tol=tol)[0]


# ---------------------------------------------------------------------------
# worlds and objective-free reference policies
# ---------------------------------------------------------------------------
# PATCH was chosen by select_patch_world.py using only the two reference
# policies below (data/patch_world_selection.json): same refill as theirs, but a
# source left alone stays full (lam = 1) and standing on a full one empties it
# with probability 0.8 per step (rho = 0.2).  There, switching pays.

PATCH = World(lam=1.0, mu=0.2, rho=0.2, name='patch')
WORLDS = {'theirs': THEIRS, 'patch': PATCH}


def _step_toward(x, c, rng):
    if x == c:
        return NA - 1
    steps = [a for a in range(NA - 1) if DIST[MOVE[a, x], c] == DIST[x, c] - 1]
    return int(rng.choice(steps))


class Parker:
    """Walk to the nearest source, then stand on it.  No objective."""

    belief_step = 0.0

    def act(self, x, E, b1, b2, rng, gated=False):
        if x in FOOD:
            return NA - 1
        c = FOOD[0] if DIST[x, FOOD[0]] <= DIST[x, FOOD[1]] else FOOD[1]
        return _step_toward(x, c, rng)


class Switcher:
    """Go to (or stay on) the source believed more likely full.

    So it leaves a source as soon as it has seen it empty.  No objective.
    """

    belief_step = 0.0

    def act(self, x, E, b1, b2, rng, gated=False):
        if b1 > b2 + 1e-9:
            c = FOOD[0]
        elif b2 > b1 + 1e-9:
            c = FOOD[1]
        else:
            c = x if x in FOOD else (FOOD[0] if DIST[x, FOOD[0]] <= DIST[x, FOOD[1]]
                                     else FOOD[1])
        return _step_toward(x, c, rng)


def run_policy(policy, gated, world=THEIRS, n_episodes=30, seed0=0, max_steps=3000):
    """run_arm for any object with .act and .belief_step."""
    recs = [simulate(policy, gated=gated, world=world, seed=seed0 + k,
                     max_steps=max_steps) for k in range(n_episodes)]
    mets = [episode_metrics(r) for r in recs]
    agg = {}
    for key in mets[0]:
        vals = np.array([m[key] for m in mets], dtype=float)
        agg[key] = float(vals.mean())
        agg[key + '_sem'] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    agg['n_episodes'] = n_episodes
    return agg, recs
