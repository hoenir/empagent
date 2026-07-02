"""The agent: empowerment lookup, uncertainty model, the single gain drive, planner.

Everything the agent values is a face of one functional, empowerment E. There is
NO mixing coefficient between exploration and exploitation: the gain drive adds a
somatic face and an epistemic face, each the increment of E from a different
slow variable (body integrity b vs model reliability u), with coefficient 1.
"""

import numpy as np

from .arena import Arena
from .empowerment import empowerment_table


class EmpowermentLUT:
    """Precomputed E(cell, b) on the TRUE dynamics; b looked up by binning."""

    def __init__(self, arena: Arena, n=3, n_bins=21, sensor=False):
        self.arena = arena
        self.b_bins = np.linspace(0.0, 1.0, n_bins)
        self.table = empowerment_table(arena, self.b_bins, n=n, sensor=sensor)  # (bins, S)
        self.Emax = self.table[-1]            # E at b=1: the positional ceiling

    def E(self, cell, b):
        i = int(round(np.clip(b, 0, 1) * (len(self.b_bins) - 1)))
        return self.table[i, cell]

    def E_percell(self, b_vec):
        """E(c, b_vec[c]) for every cell at its own body level (vectorized)."""
        idx = np.rint(np.clip(b_vec, 0, 1) * (len(self.b_bins) - 1)).astype(int)
        return self.table[idx, np.arange(self.arena.S)]


class UncertaintyModel:
    """Tabular model uncertainty u(cell) in [0,1] from decaying visit counts.

    N(cell) <- lam * N(cell) every step (forgetting, applied EVERYWHERE incl.
    the home/resource region) and N(current) += 1 on visitation. u = 1/(1+N).
    Forgetting regenerates epistemic gain during absence -- the second slow
    variable -- so consolidation is genuine re-grooving, not one-shot refueling.
    """

    def __init__(self, arena: Arena, lam=0.97):
        self.arena = arena
        self.lam = lam
        self.N = np.zeros(arena.S)

    def u(self):
        return 1.0 / (1.0 + self.N)

    def step(self, current_cell, learn=1.0):
        self.N *= self.lam
        self.N[current_cell] += learn


class Planner:
    """Receding-horizon goal planner on shortest paths (the model's distances).

    Each step: score every reachable cell c by gamma^tau * R(c, predicted b),
    where tau is travel time (so period scales with arena distances and with the
    metabolic/forgetting timescales, not a tuned weight). Step toward the best.
    """

    def __init__(self, arena: Arena, lut: EmpowermentLUT, gamma=0.85,
                 delta=0.02, intake_to_full=True):
        self.arena = arena
        self.lut = lut
        self.gamma = gamma
        self.delta = delta

    def _predict_b(self, b, tau):
        return float(np.clip(b - self.delta * tau, 0.0, 1.0))

    # --- drive faces (all increments of the SAME E functional) ----------
    def somatic_gain(self, cell, b_arr):
        """Restorable empowerment if the agent refuels here. Nonzero only at the
        resource. Large when hungry (steep dE/db), ~0 when well-fed."""
        if cell != self.arena.resource_s:
            return 0.0
        return max(0.0, self.lut.E(cell, 1.0) - self.lut.E(cell, b_arr))

    def epistemic_gain(self, cell, u, b_arr):
        """Unrealized empowerment locked behind model uncertainty, GATED by the
        body: u(c) * E(c, b_arr).  The gate is the data-processing inequality --
        epistemic control cannot be realized on a depleted body, so this face
        vanishes when hungry and only the somatic (refuel) face survives. That
        is "to explore effectively, first consolidate", derived not imposed."""
        return u[cell] * self.lut.E(cell, b_arr)

    # --- value of a candidate goal under each drive (vectorized) --------
    def value_level(self, s, b):
        tau = self.arena.dist[s]
        barr = np.clip(b - self.delta * tau, 0.0, 1.0)
        return (self.gamma ** tau) * self.lut.E_percell(barr)

    def value_gain(self, s, b, u, use_somatic=True, use_epistemic=True):
        S = self.arena.S
        res = self.arena.resource_s
        tau = self.arena.dist[s]
        d_home = self.arena.dist[:, res]            # cell -> resource
        barr = np.clip(b - self.delta * tau, 0.0, 1.0)

        som = np.zeros(S)
        if use_somatic:                              # nonzero only at the resource
            som[res] = max(0.0, self.lut.Emax[res] - self.lut.E(res, barr[res]))

        epi = np.zeros(S)
        if use_epistemic:
            # Body-gate epistemic by integrity left AFTER budgeting the return to
            # refuel: a cell you cannot afford to leave yields no realizable
            # empowerment (viability throttles control -- the DPI).
            b_viab = np.clip(b - self.delta * (tau + d_home), 0.0, 1.0)
            epi = u * self.lut.E_percell(b_viab)

        disc = self.gamma ** tau
        return disc * (som + epi), disc * som, disc * epi

    def value_hybrid(self, s, b, u, beta, refuel=0.4):
        """The canonical two-term baseline (the shared structure of de Abril &
        Kanai's empowerment+curiosity and the active-inference EFE sum):

            R(c) = (1-beta) * empowerment_level(c)  +  beta * novelty(c)

        Both terms are active at EVERY step, blended by the fixed weight beta.
        The exploit term is body-aware (it sees the resource's refuel value, so
        the baseline can forage); the explore term is plain potential-weighted
        novelty -- crucially NOT viability-gated, unlike our gain drive. Returns
        (total, exploit_contribution, explore_contribution)."""
        res = self.arena.resource_s
        tau = self.arena.dist[s]
        barr = np.clip(b - self.delta * tau, 0.0, 1.0)
        b_after = barr.copy()
        b_after[res] = min(1.0, barr[res] + refuel)      # refuel value at resource
        exploit = self.lut.E_percell(b_after)
        explore = u * self.lut.Emax                       # ungated novelty
        disc = self.gamma ** tau
        ex = disc * (1.0 - beta) * exploit
        xp = disc * beta * explore
        return ex + xp, ex, xp

    # --- pick an action --------------------------------------------------
    def act(self, s, value):
        c = int(np.argmax(value))
        if c == s:
            return 4  # STAY
        a = self.arena.first_step[s, c]
        return int(a) if a >= 0 else 4


class RolloutPlanner:
    """A3: viability WITHOUT a hand-coded return-budget term.

    Instead of gating epistemic gain by `b - delta*(tau + d_home)` (the explicit
    distance-to-home that a critic calls a smuggled survival objective), this
    planner estimates each first action's value by SIMULATING the body forward in
    (s, b) under the agent's own greedy continuation policy. Stranding becomes
    worthless on its own: depleting b collapses empowerment through the bottleneck
    (E(.,b)->0), so any rollout that runs the body down accumulates ~no gain in its
    tail. "You cannot afford to leave" emerges from forward planning, not a term.
    The continuation policy contains NO distance-to-home anywhere.
    """

    def __init__(self, arena: Arena, lut: EmpowermentLUT, gamma=0.94,
                 gamma_roll=0.9, delta=0.006, intake=0.04, horizon=16):
        self.arena = arena
        self.lut = lut
        self.gamma = gamma
        self.gamma_roll = gamma_roll
        self.delta = delta
        self.intake = intake
        self.H = horizon
        self.res = arena.resource_s

    def _greedy_value(self, s, b, u):
        """Continuation policy value over goals: somatic + epistemic gated only by
        the body AT ARRIVAL (no d_home). Returns the value vector over cells."""
        tau = self.arena.dist[s]
        barr = np.clip(b - self.delta * tau, 0.0, 1.0)
        som = np.zeros(self.arena.S)
        som[self.res] = max(0.0, self.lut.Emax[self.res] - self.lut.E(self.res, barr[self.res]))
        epi = u * self.lut.E_percell(barr)
        return (self.gamma ** tau) * (som + epi)

    def _greedy_action(self, s, b, u):
        val = self._greedy_value(s, b, u)
        c = int(np.argmax(val))
        if c == s:
            return 4
        a = self.arena.first_step[s, c]
        return int(a) if a >= 0 else 4

    def _apply(self, s, b, u, a):
        """Forward-simulate one step in (s, b, u). Returns next state and the
        realized empowerment gain (somatic refuel + epistemic learning)."""
        s2 = self.arena.cell_index[self.arena.step_cell(self.arena.cells[s], a)]
        b_dep = np.clip(b - self.delta, 0.0, 1.0)
        if s2 == self.res:
            b2 = min(1.0, b_dep + self.intake + self.delta)
            som = max(0.0, self.lut.E(self.res, b2) - self.lut.E(self.res, b_dep))
        else:
            b2, som = b_dep, 0.0
        epi = u[s2] * self.lut.E(s2, b2)               # bottleneck-gated by current body
        u2 = u.copy()
        u2[s2] *= 0.3                                   # imagined learning at the visited cell
        return s2, b2, u2, som + epi, som, epi

    def _rollout(self, s, b, u, first_a):
        s, b, u, g, som0, epi0 = self._apply(s, b, u, first_a)
        total = g
        disc = 1.0
        for _ in range(self.H - 1):
            a = self._greedy_action(s, b, u)
            s, b, u, g, _, _ = self._apply(s, b, u, a)
            disc *= self.gamma_roll
            total += disc * g
        return total, som0, epi0

    def act(self, s, b, u):
        Q = np.empty(5); som = np.empty(5); epi = np.empty(5)
        for a in range(5):
            Q[a], som[a], epi[a] = self._rollout(s, b, u, a)
        a_best = int(np.argmax(Q))
        return a_best, float(som[a_best]), float(epi[a_best])
