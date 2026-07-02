"""Switch-gate dependency environment with AUGMENTED state (cell, gate-bits).

A gate is passable only when its switch has been operated. Empowerment is computed
over the augmented state, so a downstream room's reachable futures -- hence its
empowerment -- are UNLOCKED only once the model has learned that the switch opens
the gate. This is the prerequisite dependency the independent-chamber ladder
lacked: mastering the "tool" (switch dynamics) raises downstream empowerment.
"""

import numpy as np
from .arena import ACTIONS, NA, g_noise


class GatedArena:
    def __init__(self, grid_map, switches, gates, g_b0=0.40, g_k=14.0):
        """switches: {(r,c): gate_id}; gates: {(r,c): gate_id}. Stepping onto a
        switch cell sets its gate bit (latching); a gate cell is passable iff its
        bit is set."""
        self.rows = len(grid_map); self.cols = len(grid_map[0])
        self.wall = np.zeros((self.rows, self.cols), bool)
        self.resource_rc = None
        for r, line in enumerate(grid_map):
            for c, ch in enumerate(line):
                if ch == "#": self.wall[r, c] = True
                elif ch == "R": self.resource_rc = (r, c)
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if not self.wall[r, c]]
        self.cidx = {rc: i for i, rc in enumerate(self.cells)}
        self.switches = switches            # cell -> gate_id
        self.gates = gates                  # cell -> gate_id
        self.gate_ids = sorted(set(gates.values()))
        self.NG = len(self.gate_ids)
        self.g_b0 = g_b0; self.g_k = g_k

        # augmented states = (cell_index, gatebits)
        self.states = [(ci, gb) for ci in range(len(self.cells)) for gb in range(2 ** self.NG)]
        self.sidx = {st: i for i, st in enumerate(self.states)}
        self.S = len(self.states)
        self.region = {}                    # augmented-state index -> room label (set by builder)
        self._build(true=True)
        self.D_ign = self._build(true=False, store=False)
        self._build_distances()

    def _passable(self, rc, gatebits):
        if rc not in self.cidx: return False
        if rc in self.gates:
            gid = self.gates[rc]
            return bool(gatebits >> self.gate_ids.index(gid) & 1)
        return True

    def _step(self, rc, a, gatebits, true):
        dr, dc = ACTIONS[a]
        nrc = (rc[0] + dr, rc[1] + dc)
        if not self._passable(nrc, gatebits):
            nrc = rc
        ngb = gatebits
        if true and nrc in self.switches:    # operate the switch (latching)
            ngb = gatebits | (1 << self.gate_ids.index(self.switches[nrc]))
        return nrc, ngb

    def _build(self, true=True, store=True):
        D = np.zeros((NA, self.S, self.S))
        for si, (ci, gb) in enumerate(self.states):
            rc = self.cells[ci]
            for a in range(NA):
                nrc, ngb = self._step(rc, a, gb, true)
                sj = self.sidx[(self.cidx[nrc], ngb)]
                D[a, si, sj] = 1.0
        if store: self.D = D
        return D

    def _build_distances(self):
        from collections import deque
        nxt = self.D.argmax(axis=2)           # nxt[a,s] = next state under action a
        self.dist = np.full((self.S, self.S), 1e9)
        self.first = np.full((self.S, self.S), -1, dtype=int)
        for src in range(self.S):
            self.dist[src, src] = 0
            dq = deque([src]); came = {src: -1}
            while dq:
                u = dq.popleft()
                for a in range(NA):
                    v = int(nxt[a, u])
                    if self.dist[src, v] > self.dist[src, u] + 1:
                        self.dist[src, v] = self.dist[src, u] + 1
                        came[v] = a if u == src else came[u]
                        self.first[src, v] = came[v]
                        dq.append(v)

    def partial_D(self, known_gate_ids):
        """Deterministic operator where only the listed switches function (their
        gates open); unknown switches do nothing. Models an agent that has
        mastered a SUBSET of the prerequisite chain."""
        known = set(known_gate_ids)
        D = np.zeros((NA, self.S, self.S))
        for si, (ci, gb) in enumerate(self.states):
            rc = self.cells[ci]
            for a in range(NA):
                dr, dc = ACTIONS[a]
                nrc = (rc[0] + dr, rc[1] + dc)
                if not self._passable(nrc, gb):
                    nrc = rc
                ngb = gb
                if nrc in self.switches and self.switches[nrc] in known:
                    ngb = gb | (1 << self.gate_ids.index(self.switches[nrc]))
                D[a, si, self.sidx[(self.cidx[nrc], ngb)]] = 1.0
        return D

    def transition_matrices(self, b, ignorant=False):
        g = g_noise(b, self.g_b0, self.g_k)
        D = self.D_ign if ignorant else self.D
        U = D.mean(axis=0)
        return (1 - g) * D + g * U[None]

    def state_cell(self, si):
        return self.cells[self.states[si][0]]

    def state_gatebits(self, si):
        return self.states[si][1]


class GatedModel:
    """Dirichlet model over augmented states; candidate outcomes from the true
    one-step support (the agent knows what states are locally possible, but must
    learn which action -- including operating a switch -- leads where)."""
    def __init__(self, arena, alpha=0.3, lam=1.0):
        self.A = arena; self.S = arena.S; self.lam = lam; self.alpha = alpha
        self.N = np.zeros((self.S, NA, self.S))
        self.cand = np.zeros((self.S, self.S))
        nxt = arena.D.argmax(2)
        for s in range(self.S):
            for a in range(NA):
                self.cand[s, nxt[a, s]] = 1.0
    def world(self):
        post = self.N + self.alpha * self.cand[:, None, :]
        post /= post.sum(2, keepdims=True)
        return np.transpose(post, (1, 0, 2))
    def uncertainty(self):
        post = self.N + self.alpha * self.cand[:, None, :]
        post /= post.sum(2, keepdims=True)
        return 1 - post.max(2).mean(1)
    def update(self, s, a, s2): self.N[s, a, s2] += 1
    def decay(self):
        if self.lam < 1: self.N *= self.lam


def simulate_gated(arena, steps=4000, n=4, K=25, alpha=0.35, eps=0.35, seed=0):
    """Empowerment-learning-progress agent in the gated chain. Tracks per-room
    learned empowerment and occupancy over developmental time."""
    from .empowerment import empowerment_field
    rng = np.random.default_rng(seed)
    m = GatedModel(arena, alpha=0.3, lam=1.0)
    S = arena.S; dist = arena.dist
    rooms = sorted(set(arena.region.values()))
    s = next(i for i, (ci, gb) in enumerate(arena.states)
             if gb == 0 and arena.region[i] == rooms[0] and arena.cells[ci] == (1, 2))
    LP = np.full(S, 0.8)
    visited = np.zeros(S, bool)

    def Ehat():
        return empowerment_field(arena, 1.0, n=n, T=m.world(), max_iter=40)

    start0 = s
    E = Ehat(); prev = E.copy()
    log = {"room_E": {r: [] for r in rooms}, "room_occ": {r: [] for r in rooms},
           "t": [], "cur_room": [], "E_start": [], "switches_known": []}
    win = set()
    for t in range(steps):
        if t > 0 and t % K == 0:
            E = Ehat(); dE = np.maximum(0, E - prev)
            for c in win:
                LP[c] = (1 - alpha) * LP[c] + alpha * dE[c]
            win = set(); prev = E.copy()
            log["t"].append(t)
            log["E_start"].append(float(E[start0]))        # empowerment realizable from the start
            # how many switches the model has learned (modal next-state on switch = gate opens)
            known = 0
            for swc, gid in arena.switches.items():
                sc = arena.cidx[swc]
                for si, (ci, gb) in enumerate(arena.states):
                    if ci == sc and not (gb >> arena.gate_ids.index(gid) & 1):
                        # find an in-neighbour action that lands on the switch; check model opens gate
                        pass
                known += 1 if m.N[:, :, [j for j, (cj, gj) in enumerate(arena.states)
                                if cj == sc and (gj >> arena.gate_ids.index(gid) & 1)]].sum() > 2 else 0
            log["switches_known"].append(known)
            for r in rooms:
                cells = [i for i in range(S) if arena.region[i] == r]
                log["room_E"][r].append(float(np.max([E[i] for i in cells])))   # best reachable config

        u = m.uncertainty()
        d = dist[s]
        drive = np.where(visited, LP, 0.8)         # optimism for unvisited
        val = np.power(0.9, np.minimum(d, 50)) * drive
        val[d > 1e8] = -1
        c = int(np.argmax(val))
        a = 4 if c == s else int(arena.first[s, c])
        if a < 0: a = 4
        if rng.random() < eps: a = int(rng.integers(NA))
        s2 = int(arena.D[a, s].argmax())
        m.update(s, a, s2); m.decay()
        if not visited[s2]: LP[s2] = 0.3; visited[s2] = True
        win.add(s2)
        log["cur_room"].append(arena.region[s2])
        s = s2
    log["rooms"] = rooms
    return log


def build_chain(n_rooms=3):
    """A vertical chain of small rooms, each separated from the next by a gate
    whose switch sits in the upper room. Mastering switch k unlocks room k+1."""
    # each room is 2 rows x 3 cols; gate row between rooms; switch in upper room.
    lines = ["#####"]
    rowmeta = []   # (kind, room) per grid row
    for k in range(n_rooms):
        lines.append("#...#"); rowmeta.append(("room", k))
        lines.append("#.S.#" if k < n_rooms - 1 else "#...#"); rowmeta.append(("room", k))
        if k < n_rooms - 1:
            lines.append("##G##"); rowmeta.append(("gate", k))
    lines.append("#####")
    grid = lines

    switches, gates = {}, {}
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch == "S": switches[(r, c)] = (r - 1) // 3   # room index k
            if ch == "G": gates[(r, c)] = r // 3 - 1        # gate below room k -> id k
    # normalize gate ids to 0..NG-1 by order
    ids = sorted(set(list(switches.values()) + list(gates.values())))
    remap = {g: i for i, g in enumerate(ids)}
    switches = {k: remap[v] for k, v in switches.items()}
    gates = {k: remap[v] for k, v in gates.items()}
    A = GatedArena(grid, switches, gates)
    # region labels: which room each augmented state's cell is in
    for si, (ci, gb) in enumerate(A.states):
        r = A.cells[ci][0]
        A.region[si] = "R%d" % ((r - 1) // 3)
    return A
