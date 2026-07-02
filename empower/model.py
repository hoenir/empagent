"""A tabular Bayesian model of the WORLD transition map, learned online.

The agent knows its own body channel g(b) (proprioception of its actuators) and
that its actions cause LOCAL moves, but it must learn the layout: which moves are
blocked (self-loop) and which lead where. Dirichlet counts N[s,a,s'] over the local
candidate outcomes, a uniform prior (at an unvisited cell every action looks alike,
so the learned empowerment there is ~0 until structure is discovered), and
multiplicative forgetting so model uncertainty -- hence epistemic empowerment gain
-- regenerates during absence.

Empowerment computed on this model is E(s,b,p_hat); the epistemic face of the gain
drive is the LITERAL increment of that quantity when an observation is folded in.
"""

import numpy as np

from .arena import ACTIONS


class LearnedDynamics:
    def __init__(self, arena, alpha=0.25, lam=0.997):
        self.arena = arena
        self.S = arena.S
        self.NA = arena.D.shape[0]
        self.alpha = alpha
        self.lam = lam
        self.N = np.zeros((self.S, self.NA, self.S))
        # candidate outcomes of any action from s: s itself + free orthogonal
        # neighbours (the local support the agent must distribute mass over).
        self.cand = np.zeros((self.S, self.S))
        for s, (r, c) in enumerate(arena.cells):
            self.cand[s, s] = 1.0
            for dr, dc in ACTIONS[:4]:
                rc = (r + dr, c + dc)
                if rc in arena.cell_index:
                    self.cand[s, arena.cell_index[rc]] = 1.0

    def world_matrices(self):
        """W[a] (S,S): learned posterior-mean map command a -> next cell."""
        post = self.N + self.alpha * self.cand[:, None, :]
        post /= post.sum(axis=2, keepdims=True)
        return np.transpose(post, (1, 0, 2))       # (A,S,S)

    def uncertainty(self):
        """Per-cell model uncertainty in [0,1]: 1 - mean_a max_s' p_hat(s'|s,a).
        ~high where the local map is unlearned, ~0 once it is pinned down."""
        post = self.N + self.alpha * self.cand[:, None, :]
        post /= post.sum(axis=2, keepdims=True)
        return 1.0 - post.max(axis=2).mean(axis=1)  # (S,)

    def update(self, s, a, s2):
        self.N[s, a, s2] += 1.0

    def decay(self):
        self.N *= self.lam
