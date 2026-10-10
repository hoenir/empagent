"""Independent checks on the Mylonas replication.

The fast path in empower.mylonas propagates a factored joint with scatters and
reads every belief off one pass as a linear combination.  The reference here
enumerates trajectories explicitly with dicts, shares no code with it beyond
the environment constants, and is only tractable for small n.
"""

import itertools

import numpy as np

from empower.mylonas import (FOOD, MOVE, NA, THEIRS, World, empowerment_at,
                             energy_next, gate_g)
from empower.empowerment import blahut_arimoto


def _h_step(h, x, world):
    """p(h' | h, x) as an explicit dict, built from the paper's equations."""
    out = {}
    per = []
    for i in (0, 1):
        hi = (h >> (1 - i)) & 1
        stay = world.rho if x == FOOD[i] else world.lam
        p1 = stay if hi else world.mu
        per.append({1: p1, 0: 1.0 - p1})
    for a, pa in per[0].items():
        for b, pb in per[1].items():
            out[a * 2 + b] = pa * pb
    return out


def _obs(x, h):
    """q(omega | x, h) as an explicit dict."""
    out = {}
    per = []
    for i in (0, 1):
        hi = (h >> (1 - i)) & 1
        per.append({hi: 1.0} if x == FOOD[i] else {0: 0.5, 1: 0.5})
    for a, pa in per[0].items():
        for b, pb in per[1].items():
            out[a * 2 + b] = pa * pb
    return out


def brute_force_outcomes(x0, E0, b, n, gated=False, world=THEIRS):
    """P(o_{t+n} | a^n) by explicit enumeration, for every action sequence."""
    b1, b2 = b
    init = {}
    for h in range(4):
        w = (b1 if (h >> 1) else 1 - b1) * (b2 if (h & 1) else 1 - b2)
        if w > 0:
            init[(x0, E0, h)] = init.get((x0, E0, h), 0.0) + w

    rows = []
    for seq in itertools.product(range(NA), repeat=n):
        dist = dict(init)
        for a in seq:
            nxt = {}
            for (x, E, h), p in dist.items():
                for h2, ph in _h_step(h, x, world).items():
                    if ph == 0.0:
                        continue
                    if E <= 0:
                        moves = {x: 1.0}
                    elif gated:
                        g = gate_g(E)
                        moves = {}
                        moves[MOVE[a, x]] = moves.get(MOVE[a, x], 0.0) + (1 - g)
                        for aa in range(NA):
                            q = MOVE[aa, x]
                            moves[q] = moves.get(q, 0.0) + g / NA
                    else:
                        moves = {MOVE[a, x]: 1.0}
                    for x2, pm in moves.items():
                        E2 = energy_next(E, x2, h2)
                        k = (x2, E2, h2)
                        nxt[k] = nxt.get(k, 0.0) + p * ph * pm
            dist = nxt
        row = {}
        for (x, E, h), p in dist.items():
            for w, pw in _obs(x, h).items():
                if pw:
                    row[(x, E, w)] = row.get((x, E, w), 0.0) + p * pw
        rows.append(row)

    cols = sorted({k for r in rows for k in r})
    P = np.array([[r.get(c, 0.0) for c in cols] for r in rows])
    return P


def _cap(P):
    return blahut_arimoto(P, max_iter=5000, tol=1e-13)[0]


def check(x0, E0, b, n, gated=False, world=THEIRS, tol=1e-6):
    P = brute_force_outcomes(x0, E0, b, n, gated=gated, world=world)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9), "reference rows not normalised"
    ref = _cap(P)
    fast = float(empowerment_at(x0, E0, [b], n=n, gated=gated,
                                world=world, max_iter=5000, tol=1e-13)[0])
    ok = abs(ref - fast) < tol
    print("  x0=%2d E0=%2d b=%-10s n=%d %-7s %-7s ref %.6f  fast %.6f  %s"
          % (x0, E0, str(b), n, "gated" if gated else "plain", world.name, ref, fast,
             "OK" if ok else "MISMATCH"))
    return ok


PATCH = World(lam=0.99, mu=0.05, rho=0.3, name="patch")


if __name__ == "__main__":
    cases = [
        (12, 20, (1.0, 1.0), 2, False, THEIRS),
        (12, 20, (0.5, 0.5), 2, False, THEIRS),
        (0, 20, (0.3, 0.7), 2, False, THEIRS),
        (0, 20, (1.0, 0.0), 3, False, THEIRS),
        (6, 10, (0.5, 0.5), 2, True, THEIRS),
        (0, 16, (0.8, 0.2), 2, True, THEIRS),
        (0, 18, (1.0, 1.0), 2, False, PATCH),
        (0, 12, (0.5, 0.9), 3, True, PATCH),
        (24, 20, (0.0, 1.0), 2, False, PATCH),
        (24, 3, (0.5, 0.5), 2, True, THEIRS),      # death reachable in horizon
        (24, 31 - 1, (1.0, 1.0), 2, False, THEIRS),  # energy cap active
    ]
    print("brute-force reference vs fast path:")
    allok = all([check(*c) for c in cases])
    print("ALL OK" if allok else "FAILURES PRESENT")
    raise SystemExit(0 if allok else 1)
