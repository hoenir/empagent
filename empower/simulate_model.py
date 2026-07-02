"""Model-based agent: empowerment is computed on the LEARNED dynamics p_hat.

The epistemic face of the drive is now the literal increment in computed
empowerment when observations are folded into the model -- not a visit-count
proxy. The agent first FAMILIARIZES (a curiosity-driven coverage walk with the
body paused) so that empowerment-on-p_hat builds toward the true field; then it
runs the gain dynamics, with forgetting regenerating epistemic gain. The
empowerment table is recomputed on the current p_hat every `recompute_every`
steps (the brief's caching prescription).
"""

import numpy as np

from .arena import g_noise, NA
from .empowerment import empowerment_field, body_gate
from .model import LearnedDynamics


def _plan_step(arena, s, value):
    c = int(np.argmax(value))
    if c == s:
        return 4, c
    a = arena.first_step[s, c]
    return (int(a) if a >= 0 else 4), c


def simulate_model(arena, n=3, steps=2000, warmup=600, delta=0.006, intake=0.04,
                   lam_model=0.998, gamma=0.94, alpha=0.25, kappa=1.0,
                   recompute_every=20, n_bins=7, b0=1.0, seed=0):
    rng = np.random.default_rng(seed)
    model = LearnedDynamics(arena, alpha=alpha, lam=lam_model)
    res = arena.resource_s
    S = arena.S
    bins = np.linspace(0.0, 1.0, n_bins)
    dist = arena.dist
    d_home = dist[:, res]

    def recompute():
        W = model.world_matrices()
        return np.array([empowerment_field(arena, bb, n=n,
                         T=body_gate(W, bb, arena.g_b0, arena.g_k)) for bb in bins])

    def Eidx(bvec):
        return np.rint(np.clip(bvec, 0, 1) * (n_bins - 1)).astype(int)

    # ---- familiarization: curiosity coverage, body paused -------------
    s = res
    for t in range(warmup):
        u = model.uncertainty()
        tau = dist[s]
        a, c = _plan_step(arena, s, (gamma ** tau) * u)     # nearest most-uncertain
        s2 = arena.cell_index[arena.step_cell(arena.cells[s], a)]
        model.update(s, a, s2); model.decay()
        s = s2

    Ehat = recompute()
    Emax_prev = Ehat[-1].copy()

    # ---- main phase: gain dynamics on the LEARNED model ---------------
    s = res
    b = b0
    log = {k: [] for k in ("cell", "row", "col", "b", "somatic", "epistemic",
                           "phase", "Ehat_mean", "u_mean", "realized_epi",
                           "proxy_epi", "dist_home")}
    realized = np.nan
    for t in range(steps):
        if t > 0 and t % recompute_every == 0:
            Ehat = recompute()
            realized = float(np.sum(np.maximum(0.0, Ehat[-1] - Emax_prev)))
            Emax_prev = Ehat[-1].copy()
        else:
            realized = np.nan

        u = model.uncertainty()
        tau = dist[s]
        barr = np.clip(b - delta * tau, 0.0, 1.0)
        b_viab = np.clip(b - delta * (tau + d_home), 0.0, 1.0)

        som = np.zeros(S)
        som[res] = max(0.0, Ehat[-1, res] - Ehat[Eidx(barr[res]), res])
        Epot_viab = Ehat[Eidx(b_viab), np.arange(S)]
        epi = u * (Epot_viab + kappa * b_viab)
        disc = gamma ** tau
        val = disc * (som + epi)

        a, c = _plan_step(arena, s, val)
        som_c = float(disc[c] * som[c]); epi_c = float(disc[c] * epi[c])
        if rng.random() < g_noise(b, arena.g_b0, arena.g_k):
            a = int(rng.integers(NA))
        s2 = arena.cell_index[arena.step_cell(arena.cells[s], a)]

        b = b - delta
        if s2 == res:
            b = min(1.0, b + intake + delta)
        b = float(np.clip(b, 0.0, 1.0))

        model.update(s, a, s2); model.decay()

        rc2 = arena.cells[s2]
        log["cell"].append(s2); log["row"].append(rc2[0]); log["col"].append(rc2[1])
        log["b"].append(b); log["somatic"].append(som_c); log["epistemic"].append(epi_c)
        log["phase"].append("consolidate" if som_c >= epi_c else "venture")
        log["Ehat_mean"].append(float(Ehat[-1].mean()))
        log["u_mean"].append(float(u.mean()))
        log["realized_epi"].append(realized)
        log["proxy_epi"].append(float(u[s2] * arena.S))   # visit-count-style proxy at visited cell
        log["dist_home"].append(int(abs(rc2[0] - arena.resource_rc[0]) +
                                    abs(rc2[1] - arena.resource_rc[1])))
        s = s2

    out = {k: np.array(v, dtype=float) if k != "phase" else np.array(v)
           for k, v in log.items()}
    out["true_potential"] = float(empowerment_field(arena, 1.0, n=n).mean())
    return out
