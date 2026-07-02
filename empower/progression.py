"""Developmental-progression sibling study: empowerment learning-progress.

The drive is the recent rate of realized empowerment gain on the learned model --
empowerment LEARNING PROGRESS (Oudeyer's region-based LP, but with empowerment
increment in place of prediction-error decrement). It forages the competence
frontier, abandons mastered regions (LP -> 0), and -- the headline -- ignores the
"noisy TV": a region with high intrinsic (aleatoric) action noise never sharpens,
so its empowerment never rises, so its learning progress stays ~0. Empowerment is
control-grounded, so unlearnable stochasticity is simply uncontrollable.
"""

import numpy as np

from .arena import Arena, g_noise, NA
from .empowerment import empowerment_field, body_gate
from .model import LearnedDynamics


class ProgArena(Arena):
    """Arena with a per-cell ENVIRONMENTAL action-noise field (region-tied,
    independent of the body). High env noise -> low empowerment ceiling even when
    fully learned: the noisy TV."""

    def __init__(self, grid_map, env_noise=None, region=None, g_b0=0.40, g_k=14.0):
        super().__init__(grid_map, g_b0=g_b0, g_k=g_k)
        self.env = np.zeros(self.S)
        if env_noise:
            for rc, val in env_noise.items():
                if rc in self.cell_index:
                    self.env[self.cell_index[rc]] = val
        self.region = {}                       # cell index -> label
        if region:
            for rc, lab in region.items():
                if rc in self.cell_index:
                    self.region[self.cell_index[rc]] = lab

    def transition_matrices(self, b):
        g = g_noise(b, self.g_b0, self.g_k)
        U = self.D.mean(axis=0)                # (S,S) uniform-over-actions
        env = self.env[None, :, None]          # broadcast over (a, s')
        Denv = (1 - env) * self.D + env * U[None]
        return (1 - g) * Denv + g * U[None]

    def region_cells(self, label):
        return [s for s, lab in self.region.items() if lab == label]


def build_arena_A(noise=0.75):
    """Equidistant hub-and-spoke: four 3x3 chambers off a central hub by equal
    corridors. FRONTIER and NOISE are structurally identical (so any difference
    is the noise); EASY is a small constrained pocket (mastered fast); the fourth
    arm replicates FRONTIER. Equidistant -> preference can't be about distance."""
    R, C = 15, 15
    g = [["#"] * C for _ in range(R)]

    def carve(r0, r1, c0, c1, ch="."):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = ch

    carve(6, 8, 6, 8)                          # hub
    g[7][7] = "R"
    # North = FRONTIER, South = FRONTIER2 (replicate), East = NOISE, West = EASY
    carve(1, 3, 6, 8); carve(4, 5, 7, 7)       # N chamber + corridor
    carve(11, 13, 6, 8); carve(9, 10, 7, 7)    # S chamber + corridor
    carve(6, 8, 11, 13); carve(7, 7, 9, 10)    # E chamber + corridor
    carve(7, 7, 1, 3); carve(7, 7, 4, 5)       # W = EASY (1-wide pocket) + corridor

    grid_map = ["".join(row) for row in g]

    region, env = {}, {}
    for r in range(1, 4):
        for c in range(6, 9):
            region[(r, c)] = "frontier"
    for r in range(11, 14):
        for c in range(6, 9):
            region[(r, c)] = "frontier2"
    for r in range(6, 9):
        for c in range(11, 14):
            region[(r, c)] = "noise"; env[(r, c)] = noise
    for c in range(1, 4):
        region[(7, c)] = "easy"

    return ProgArena(grid_map, env_noise=env, region=region)


def build_arena_B():
    """A competence ladder: four 3x3 chambers R1..R4 hung off a horizontal
    corridor at increasing distance from the resource. The agent must traverse
    each to reach the next; the test is whether the competence frontier MARCHES
    (per-region learning-progress peaking in sequence) as nearer chambers master."""
    R, C = 7, 21
    g = [["#"] * C for _ in range(R)]

    def carve(r0, r1, c0, c1):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = "."

    carve(4, 4, 1, 19)                         # corridor
    g[4][1] = "R"
    region = {}
    centers = [4, 9, 14, 19] if False else [4, 8, 12, 16]
    for k, cc in enumerate(centers, start=1):
        carve(1, 3, cc - 1, cc + 1)            # 3x3 chamber above
        carve(3, 3, cc, cc)                    # link to corridor
        for r in range(1, 4):
            for c in range(cc - 1, cc + 2):
                region[(r, c)] = "R%d" % k
    grid_map = ["".join(row) for row in g]
    return ProgArena(grid_map, env_noise={}, region=region)


def simulate_progression(arena, steps=2500, delta=0.004, intake=0.05, lam_model=0.999,
                         gamma=0.94, n=3, K=20, n_bins=5, alpha=0.35, lp_init=0.9,
                         eps=0.3, drive="lp", b0=1.0, seed=0):
    """Curriculum agent. drive='lp' uses empowerment learning-progress (recent
    realized dE on the learned model); drive='novelty' uses raw model uncertainty
    (the comparison that gets trapped by the noisy TV)."""
    rng = np.random.default_rng(seed)
    model = LearnedDynamics(arena, alpha=0.25, lam=lam_model)
    res = arena.resource_s; S = arena.S
    dist = arena.dist; d_home = dist[:, res]
    bins = np.linspace(0, 1, n_bins)
    s = res; b = b0
    LP = np.full(S, lp_init)                   # optimistic: unvisited looks promising
    visited_ever = np.zeros(S, bool)

    def recompute():
        W = model.world_matrices()
        return np.array([empowerment_field(arena, bb, n=n,
                         T=body_gate(W, bb, arena.g_b0, arena.g_k), max_iter=40) for bb in bins])

    def Eidx(bv):
        return int(round(np.clip(bv, 0, 1) * (n_bins - 1)))

    Ehat = recompute(); Emax_prev = Ehat[-1].copy()
    window = set()
    labels = sorted(set(arena.region.values()))
    log = {"cell": [], "region": [], "target": [], "b": [], "som": [], "epi": [],
           "lp_region": {lab: [] for lab in labels},
           "dwell_region": {lab: [] for lab in labels}}

    for t in range(steps):
        if t > 0 and t % K == 0:
            Ehat = recompute()
            dE = np.maximum(0.0, Ehat[-1] - Emax_prev)     # realized empowerment gain
            for c in window:
                LP[c] = (1 - alpha) * LP[c] + alpha * dE[c]  # learning-progress EMA
            window = set(); Emax_prev = Ehat[-1].copy()

        u = model.uncertainty()
        tau = dist[s]
        barr = np.clip(b - delta * tau, 0, 1)
        b_viab = np.clip(b - delta * (tau + d_home), 0, 1)
        som = np.zeros(S)
        som[res] = max(0.0, Ehat[-1, res] - Ehat[Eidx(barr[res]), res])
        gate = np.clip(b_viab / 0.3, 0, 1) if delta > 0 else 1.0
        if drive == "lp":
            epi = LP * gate                               # empowerment learning-progress
        else:                                             # surprise/prediction-error curiosity:
            epi = u * gate                                # stays high in the noisy TV -> trapped
        disc = gamma ** tau
        val = disc * (som + epi)

        c = int(np.argmax(val))
        target_lab = arena.region.get(c, "corridor")
        som_c = float(disc[c] * som[c]); epi_c = float(disc[c] * epi[c])
        a = 4 if c == s else int(arena.first_step[s, c])
        if a < 0: a = 4
        if rng.random() < eps:                  # motor babbling: try actions to LEARN local control
            a = int(rng.integers(NA))
        a_cmd = a
        p_corrupt = 1 - (1 - g_noise(b, arena.g_b0, arena.g_k)) * (1 - arena.env[s])
        if rng.random() < p_corrupt:
            a = int(rng.integers(NA))
        s2 = arena.cell_index[arena.step_cell(arena.cells[s], a)]

        b = b - delta
        if s2 == res:
            b = min(1.0, b + intake + delta)
        b = float(np.clip(b, 0, 1))
        model.update(s, a_cmd, s2); model.decay()   # learn from the COMMANDED action
        if not visited_ever[s2]:                # first real visit: drop optimism toward measured LP
            LP[s2] = min(LP[s2], 0.3); visited_ever[s2] = True
        window.add(s2)

        lab_now = arena.region.get(s2, "corridor")
        log["cell"].append(s2); log["region"].append(lab_now); log["target"].append(target_lab)
        log["b"].append(b); log["som"].append(som_c); log["epi"].append(epi_c)
        for lab in labels:
            cells = arena.region_cells(lab)
            log["lp_region"][lab].append(float(np.mean([LP[x] for x in cells])))
            log["dwell_region"][lab].append(1.0 if lab_now == lab else 0.0)
        s = s2

    log["labels"] = labels
    log["final_u"] = model.uncertainty(); log["final_LP"] = LP.copy()
    log["b"] = np.array(log["b"]); log["region"] = np.array(log["region"])
    return log


if __name__ == "__main__":
    from .empowerment import empowerment_field, field_to_grid
    A = build_arena_A()
    print("arena %dx%d  free cells %d  resource %s" % (A.rows, A.cols, A.S, A.resource_rc))
    E = empowerment_field(A, 1.0, n=3)
    grid = field_to_grid(A, E)
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    for r in range(A.rows):
        row = ""
        for c in range(A.cols):
            row += " ## " if np.isnan(grid[r, c]) else "%4.1f" % grid[r, c]
        print(row)
    for lab in ("frontier", "frontier2", "noise", "easy"):
        cells = A.region_cells(lab)
        d = np.mean([A.dist[A.resource_s, s] for s in cells])
        print("%-10s  meanE %.2f   meanDist %.1f   cells %d" %
              (lab, np.mean([E[s] for s in cells]), d, len(cells)))
