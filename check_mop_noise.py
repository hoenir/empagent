"""Step 0 (H4): is MOP drawn to the noisy TV only through its state-entropy term?

Figure 16 rewarded next-state entropy H(S'|s,a) with action entropy at
temperature 0.5 -- in MOP's own normalisation (action entropy weight 1) that is
beta = 2.  The calibrated MOP of section 22, which is also the setting of the
paper we compare against, is beta = 0.  Here MOP is run in its own form,

    V(s) = ln sum_a exp[ beta H(S'|s,a) + gamma sum_s' P(s'|s,a) V(s') ],

in the same noise arena as Figure 16, for beta in {0, 0.25, 0.5, 1, 2}, both
wall conventions, and gamma 0.95 (Figure 16) and 0.99 (theirs).  Baselines: a
uniform random walk over the available actions, and the exact Figure 16 recipe.

Prediction written before running: at beta = 0 MOP's time on the noisy TV is
close to a random walk's; it climbs towards Figure 16's 98% as beta grows.

Writes data/mop_noise_check.json.
"""

import json

import numpy as np

from empower.empowerment import empowerment_field
from empower.progression import build_arena_A

An = build_arena_A(noise=0.6)                    # the Figure 16 arena
S, NA = An.S, An.D.shape[0]
T = An.transition_matrices(1.0)                  # full body: no actuator noise
STAY = NA - 1


def ent(p):
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum())


Hsa = np.array([[ent(T[a, s]) for a in range(NA)] for s in range(S)])
Enat = empowerment_field(An, 1.0, n=3) * np.log(2)

AVAIL = {"legal": np.ones((S, NA), dtype=bool),
         "redundant": np.ones((S, NA), dtype=bool)}
for s in range(S):
    for a in range(NA - 1):
        if An.D[a, s, s] > 0.5:                  # a wall bump: leaves you in place
            AVAIL["redundant"][s, a] = False


def soft_policy(r, alpha, gamma, avail, iters=2000, tol=1e-9):
    V = np.zeros(S)
    for _ in range(iters):
        Q = r + gamma * np.einsum('asx,x->sa', T, V)
        Q = np.where(avail, Q, -np.inf)
        m = Q.max(1, keepdims=True)
        Vn = m[:, 0] + alpha * np.log(np.exp((Q - m) / alpha).sum(1))
        if np.abs(Vn - V).max() < tol:
            V = Vn
            break
        V = Vn
    Q = np.where(avail, r + gamma * np.einsum('asx,x->sa', T, V), -np.inf)
    m = Q.max(1, keepdims=True)
    pi = np.exp((Q - m) / alpha)
    return pi / pi.sum(1, keepdims=True)


def dwell(pi, seeds=20, steps=6000, burn=500):
    labels = sorted(set(An.region.values()))
    counts = {lab: 0 for lab in labels}
    tot = 0
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        s = An.resource_s
        for t in range(steps):
            a = rng.choice(NA, p=pi[s])
            s = rng.choice(S, p=T[a, s])
            if t > burn:
                tot += 1
                lab = An.region.get(s)
                if lab in counts:
                    counts[lab] += 1
    return {lab: 100.0 * c / tot for lab, c in counts.items()}


if __name__ == "__main__":
    out = {}
    noise_cells = len(An.region_cells("noise"))
    print("arena: %d cells, %d in the noisy chamber (%.0f%% of the arena)"
          % (S, noise_cells, 100.0 * noise_cells / S))

    ref = dwell(soft_policy(Hsa, 0.5, 0.95, AVAIL["legal"]))
    out["figure16_recipe"] = ref
    print("Figure 16 recipe (reward H(S'), alpha 0.5, gamma 0.95): noise %.0f%%" % ref["noise"])
    co = dwell(soft_policy(np.repeat(Enat[:, None], NA, 1), 0.5, 0.95, AVAIL["legal"]))
    out["control_occupancy"] = co
    print("control-occupancy (reward E, alpha 0.5, gamma 0.95):    noise %.0f%%" % co["noise"])

    for walls in ("legal", "redundant"):
        rw = AVAIL[walls] / AVAIL[walls].sum(1, keepdims=True)
        d = dwell(rw)
        out["random_walk|%s" % walls] = d
        print("\nuniform random walk, walls=%-9s: noise %4.1f%%   (%s)"
              % (walls, d["noise"], "  ".join("%s %.0f%%" % kv for kv in sorted(d.items()))))
        for gamma in (0.95, 0.99):
            for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
                d = dwell(soft_policy(beta * Hsa, 1.0, gamma, AVAIL[walls]))
                out["mop|%s|%.2f|%.2f" % (walls, gamma, beta)] = d
                print("  MOP walls=%-9s gamma %.2f beta %.2f: noise %4.1f%%   (%s)"
                      % (walls, gamma, beta, d["noise"],
                         "  ".join("%s %.0f%%" % kv for kv in sorted(d.items()))),
                      flush=True)

    with open("data/mop_noise_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote data/mop_noise_check.json")
