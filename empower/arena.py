"""Arena, body-gated dynamics, and sensors for the empowerment-alternation experiments.

A minimal grid agent. State for empowerment purposes is the cell (position).
Five actions: N, E, S, W, STAY. Body integrity b in [0,1] gates ACTUATOR
reliability: with probability g(b) the commanded action is replaced by a
uniform-random primitive (the "body" channel of the serial bottleneck
A -> A_tilde -> S_{t+n}). g(b) is steep near b=0 so control collapses as the
body fails. The world stage is deterministic, so empowerment of the
*composition* tracks min(C_body(b), E_max(cell)) by the data-processing
inequality -- exactly the bottleneck-handover story in the brief.
"""

import numpy as np

# Default arena: two large open basins (high E) flanking a small central
# chamber that holds the resource (moderate E -- the existence condition for a
# completable satiation phase), all joined by a constricted corridor (low E).
DEFAULT_MAP = [
    "#################",
    "#.....#...#.....#",
    "#.....#.R.#.....#",
    "#.....#...#.....#",
    "#...............#",
    "#.....#...#.....#",
    "#.....#...#.....#",
    "#.....#...#.....#",
    "#################",
]

# Actions: dr, dc.  Order matters (index 4 = STAY).
ACTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
ACTION_NAMES = ["N", "E", "S", "W", "STAY"]
NA = len(ACTIONS)


def g_noise(b, b0=0.40, k=14.0):
    """Actuator corruption probability as a function of body integrity b.

    Sigmoid gate: g(b) = 1 / (1 + exp(k (b - b0))).  Near-zero over a high-b
    PLATEAU (so feeding when well-fed yields ~0 empowerment gain -> the
    satiation flip), a steep COLLAPSE around b0 (so feeding when hungry yields
    large gain), and ~1 as b -> 0 (control fails). g(1)~0, g(0)~1.
    """
    b = np.clip(b, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(k * (b - b0)))


class Arena:
    def __init__(self, grid_map=DEFAULT_MAP, g_b0=0.40, g_k=14.0):
        self.rows = len(grid_map)
        self.cols = len(grid_map[0])
        self.wall = np.zeros((self.rows, self.cols), dtype=bool)
        self.resource_rc = None
        for r, line in enumerate(grid_map):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.wall[r, c] = True
                elif ch == "R":
                    self.resource_rc = (r, c)
        self.g_b0 = g_b0
        self.g_k = g_k

        # Enumerate free cells and index them.
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if not self.wall[r, c]]
        self.S = len(self.cells)
        self.cell_index = {rc: i for i, rc in enumerate(self.cells)}
        self.resource_s = self.cell_index[self.resource_rc]

        self._build_deterministic_transitions()
        self._build_sensors()
        self._build_distances()

    # --- dynamics -------------------------------------------------------
    def step_cell(self, rc, a):
        """Deterministic move for action index a from cell rc; walls block."""
        dr, dc = ACTIONS[a]
        nr, nc = rc[0] + dr, rc[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.wall[nr, nc]:
            return (nr, nc)
        return rc

    def _build_deterministic_transitions(self):
        # D[a] is an S x S 0/1 matrix: D[a][s, s'] = 1 if action a maps s -> s'.
        self.D = np.zeros((NA, self.S, self.S))
        for s, rc in enumerate(self.cells):
            for a in range(NA):
                s2 = self.cell_index[self.step_cell(rc, a)]
                self.D[a, s, s2] = 1.0

    def transition_matrices(self, b):
        """Body-gated per-step transition matrices T[a] (S x S) at integrity b.

        Executed-action distribution: (1-g) on the command, g spread uniformly.
        """
        g = g_noise(b, self.g_b0, self.g_k)
        uniform = self.D.mean(axis=0)  # (1/NA) sum_k D[k]
        T = (1.0 - g) * self.D + g * uniform[None, :, :]
        return T  # (NA, S, S)

    # --- sensors --------------------------------------------------------
    def _build_sensors(self):
        """Egocentric sensor = 4-bit wall-adjacency pattern (N,E,S,W).

        Multiple cells can share a reading (perceptual aliasing), which is what
        makes sensor-state empowerment differ from full-state empowerment.
        """
        obs_of_cell = np.zeros(self.S, dtype=int)
        for s, (r, c) in enumerate(self.cells):
            bits = 0
            for k, (dr, dc) in enumerate(ACTIONS[:4]):
                nr, nc = r + dr, c + dc
                blocked = not (0 <= nr < self.rows and 0 <= nc < self.cols
                               and not self.wall[nr, nc])
                if blocked:
                    bits |= (1 << k)
            obs_of_cell[s] = bits
        self.obs_of_cell = obs_of_cell
        uniq = sorted(set(obs_of_cell.tolist()))
        self.obs_index = {o: i for i, o in enumerate(uniq)}
        self.n_obs = len(uniq)
        # Projection S x n_obs: deterministic cell -> observation class.
        self.proj = np.zeros((self.S, self.n_obs))
        for s in range(self.S):
            self.proj[s, self.obs_index[obs_of_cell[s]]] = 1.0

    # --- distances (planner uses BFS shortest paths) --------------------
    def _build_distances(self):
        from collections import deque
        S = self.S
        self.dist = np.full((S, S), 1e9)
        self.first_step = np.full((S, S), -1, dtype=int)  # action from i toward j
        for src in range(S):
            self.dist[src, src] = 0
            dq = deque([src])
            # parent action that led INTO a node, to recover the first step
            came_action = {src: -1}
            order = [src]
            while dq:
                u = dq.popleft()
                urc = self.cells[u]
                for a in range(4):  # moves only for path edges
                    v = self.cell_index[self.step_cell(urc, a)]
                    if v != u and self.dist[src, v] > self.dist[src, u] + 1:
                        self.dist[src, v] = self.dist[src, u] + 1
                        came_action[v] = a if u == src else came_action[u]
                        dq.append(v)
                        order.append(v)
            for v in order:
                self.first_step[src, v] = came_action[v]
