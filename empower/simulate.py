"""Simulation loop + logging for all experiment variants."""

import numpy as np

from .arena import Arena, g_noise, NA
from .agent import EmpowermentLUT, UncertaintyModel, Planner, RolloutPlanner


def simulate(arena: Arena, lut: EmpowermentLUT, steps=1500, drive="gain",
             delta=0.02, intake=0.08, lam=0.97, gamma=0.85,
             depletion=True, forgetting=True,
             use_somatic=True, use_epistemic=True, beta=0.5,
             roll_H=16, roll_gamma=0.9,
             start_cell=None, b0=1.0, seed=0):
    """Run one episode. drive in {level, gain, hybrid, random}.

    hybrid is the two-term baseline R = (1-beta)*empowerment + beta*novelty.

    depletion   : body integrity decays (somatic feedback on/off)
    forgetting  : model uncertainty regenerates (epistemic feedback on/off)
    Returns a dict of logged time series.
    """
    rng = np.random.default_rng(seed)
    s = arena.resource_s if start_cell is None else start_cell
    b = b0
    model = UncertaintyModel(arena, lam=lam if forgetting else 1.0)
    if not forgetting:
        model.lam = 1.0
    planner = Planner(arena, lut, gamma=gamma, delta=(delta if depletion else 0.0))
    rollout = RolloutPlanner(arena, lut, gamma=gamma, delta=delta, intake=intake,
                             gamma_roll=roll_gamma, horizon=roll_H) \
        if drive == "gain_rollout" else None

    log = {k: [] for k in ("cell", "row", "col", "b", "u_mean", "u_home",
                           "E_here", "somatic", "epistemic", "total",
                           "phase", "speed", "action")}
    prev_rc = arena.cells[s]

    for t in range(steps):
        u = model.u()
        # --- choose action under the active drive ---------------------
        if drive == "random":
            a = int(rng.integers(NA))
            som_c = epi_c = tot_c = 0.0
        elif drive == "level":
            val = planner.value_level(s, b)
            a = planner.act(s, val)
            c = int(np.argmax(val))
            som_c = lut.E(s, b); epi_c = 0.0; tot_c = val[c]
        elif drive == "hybrid":
            val, vex, vxp = planner.value_hybrid(s, b, u, beta)
            a = planner.act(s, val)
            c = int(np.argmax(val))
            som_c = float(vex[c]); epi_c = float(vxp[c]); tot_c = float(val[c])
        elif drive == "gain_rollout":
            a, som_c, epi_c = rollout.act(s, b, u)      # viability emerges, no d_home
            tot_c = som_c + epi_c
        else:  # gain
            val, vsom, vepi = planner.value_gain(
                s, b, u, use_somatic=use_somatic, use_epistemic=use_epistemic)
            a = planner.act(s, val)
            # gain components realized at the chosen target
            c = int(np.argmax(val))
            som_c = float(vsom[c]); epi_c = float(vepi[c]); tot_c = float(val[c])

        # --- execute with body-gated actuator noise -------------------
        a_exec = a
        if rng.random() < g_noise(b, arena.g_b0, arena.g_k):
            a_exec = int(rng.integers(NA))
        rc = arena.cells[s]
        s2 = arena.cell_index[arena.step_cell(rc, a_exec)]

        # --- body + model updates -------------------------------------
        if depletion:
            b = b - delta
            if s2 == arena.resource_s:
                b = min(1.0, b + intake + delta)  # net intake when feeding
            b = float(np.clip(b, 0.0, 1.0))
        if forgetting or drive != "random":
            model.step(s2)

        # --- phase label (which face dominates the realized gain) -----
        rc2 = arena.cells[s2]
        speed = abs(rc2[0] - prev_rc[0]) + abs(rc2[1] - prev_rc[1])
        if drive in ("gain", "hybrid", "gain_rollout"):
            phase = "consolidate" if som_c >= epi_c else "venture"
        elif drive == "level":
            phase = "exploit"
        else:
            phase = "random"

        log["cell"].append(s2); log["row"].append(rc2[0]); log["col"].append(rc2[1])
        log["b"].append(b); log["u_mean"].append(float(u.mean()))
        log["u_home"].append(float(u[arena.resource_s]))
        log["E_here"].append(float(lut.E(s2, b)))
        log["somatic"].append(som_c); log["epistemic"].append(epi_c)
        log["total"].append(tot_c); log["phase"].append(phase)
        log["speed"].append(speed); log["action"].append(a)

        prev_rc = rc2
        s = s2

    out = {k: np.array(v) if k != "phase" else np.array(v) for k, v in log.items()}
    out["params"] = dict(drive=drive, delta=delta, intake=intake, lam=lam,
                         gamma=gamma, depletion=depletion, forgetting=forgetting,
                         use_somatic=use_somatic, use_epistemic=use_epistemic)
    return out
