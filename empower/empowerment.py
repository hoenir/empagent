"""n-step empowerment via Blahut-Arimoto on the body-gated transition operator.

E_n(s, b) = max_{p(a^n)} I(A^n ; X_{t+n} | s, b)

where X is either the future cell (full-state empowerment) or the future
sensor reading (sensor-state empowerment). Capacity is reported in bits.
"""

import itertools
import numpy as np

from .arena import Arena, NA


def blahut_arimoto(P, max_iter=200, tol=1e-9):
    """Channel capacity (bits) for conditional distributions P (K inputs x M outputs).

    Returns capacity and the optimal input distribution.
    """
    K = P.shape[0]
    P = np.clip(P, 1e-12, None)
    P = P / P.sum(axis=1, keepdims=True)
    logP = np.log(P)
    r = np.ones(K) / K
    cap = 0.0
    for _ in range(max_iter):
        q = r @ P                      # output marginal (M,)
        q = np.clip(q, 1e-12, None)
        # D_k = sum_j P[k,j] log(P[k,j]/q_j)  (nats)
        D = (P * (logP - np.log(q)[None, :])).sum(axis=1)
        r_new = r * np.exp(D)
        r_new /= r_new.sum()
        new_cap = float((r_new * D).sum())  # nats (lower bound term)
        if abs(new_cap - cap) < tol:
            r = r_new
            cap = new_cap
            break
        r, cap = r_new, new_cap
    # Final capacity from the converged input distribution.
    q = np.clip(r @ P, 1e-12, None)
    D = (P * (logP - np.log(q)[None, :])).sum(axis=1)
    cap_nats = float((r * D).sum())
    return cap_nats / np.log(2.0), r


def body_gate(W, b, g_b0, g_k):
    """Compose a world operator W (NA,S,S) with the body actuator channel at b:
    with prob g(b) the command is replaced by a uniform-random primitive."""
    from .arena import g_noise
    g = g_noise(b, g_b0, g_k)
    U = W.mean(axis=0)
    return (1.0 - g) * W + g * U[None, :, :]


def _sequence_output_tensor(arena: Arena, b, n, sensor=False, T=None):
    """For every start cell, the output distribution of each length-n action seq.

    Returns Out with shape (S, NA**n, M): Out[s, k] = p(X_{t+n} | start=s, seq_k).
    M = S for full-state, n_obs for sensor-state. T (NA,S,S) overrides the true
    body-gated operator -- used to compute empowerment on a LEARNED model.
    """
    if T is None:
        T = arena.transition_matrices(b)          # (NA, S, S) true dynamics
    S = arena.S
    seqs = list(itertools.product(range(NA), repeat=n))
    M = arena.n_obs if sensor else S
    Out = np.empty((S, len(seqs), M))
    proj = arena.proj if sensor else None
    for k, seq in enumerate(seqs):
        Mat = T[seq[0]]
        for a in seq[1:]:
            Mat = Mat @ T[a]                      # chronological compose; rows = start state
        Out[:, k, :] = Mat @ proj if sensor else Mat
    return Out


def empowerment_field(arena: Arena, b, n=3, sensor=False, T=None, max_iter=200):
    """E_n(s, b) for all cells at body integrity b (bits). T overrides dynamics."""
    Out = _sequence_output_tensor(arena, b, n, sensor=sensor, T=T)
    E = np.empty(arena.S)
    for s in range(arena.S):
        E[s], _ = blahut_arimoto(Out[s], max_iter=max_iter)
    return E


def empowerment_cells(arena: Arena, b, cells, n=3, T=None):
    """E_n(s, b) for a SUBSET of start cells (local recompute on a learned model)."""
    Out = _sequence_output_tensor(arena, b, n, sensor=False, T=T)
    return np.array([blahut_arimoto(Out[s])[0] for s in cells])


def empowerment_table(arena: Arena, b_bins, n=3, sensor=False):
    """E_n(s, b) over a grid of body-integrity bins. Shape (len(b_bins), S)."""
    return np.array([empowerment_field(arena, b, n=n, sensor=sensor)
                     for b in b_bins])


def field_to_grid(arena: Arena, field):
    """Embed a per-cell vector into a rows x cols grid (NaN at walls) for plotting."""
    g = np.full((arena.rows, arena.cols), np.nan)
    for s, (r, c) in enumerate(arena.cells):
        g[r, c] = field[s]
    return g
