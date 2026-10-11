"""Control-occupancy in the two worlds of section 22.

MOP's machinery (their Eq. 6: soft value iteration over position, energy and
belief, gamma = 0.99, zero value at death, calibrated wall convention) with
empowerment C(x, E, b) as the per-step reward -- the same empowerment tables as
section 22, read on MOP's 0.1 belief grid.

  greedy  maximise E[sum gamma^t C(s_t)] with a hard max: cumulative empowerment.
          Separates HOW you count (cumulative, long horizon) from WHAT you count.
  soft    maximise E[sum gamma^t (C(s_t) + H(pi))], control and action entropy
          both in bits with weight one (alpha = 1/ln 2): MOP plus control.

The question it answers: is MOP's advantage in section 22 (an eating set point
without a body gate) a property of its objective or of its solver?

Writes data/control_occupancy.json; Q tables cached as data/co_*.npz.
"""

import json
import os

import numpy as np

from empower.mop_belief import (BG, NB, NE, MOPAgent, build_transitions,
                                value_iteration)
from empower.mylonas import NPOS, WORLDS, MylonasLUT, run_policy
from diag_mylonas import meal_stats

NEP = int(os.environ.get("MYL_EPISODES", 30))
MAXSTEPS = 3000
ALPHA_BITS = 1.0 / np.log(2.0)
ALPHA_SWEEP = [0.25, 0.5, ALPHA_BITS, 2.0]


def empowerment_reward(world_name, gated):
    suffix = "" if world_name == "theirs" else "_" + world_name
    lut = MylonasLUT.load("data/mylonas_C_n5_%s%s.npz"
                          % ("gated" if gated else "plain", suffix))
    r = np.zeros((NPOS, NE, NB, NB))
    for x in range(NPOS):
        for E in range(1, NE):
            for i in range(NB):
                for j in range(NB):
                    r[x, E, i, j] = lut.C(x, E, BG[i], BG[j])
    return r.ravel()


def solve(world_name, gated, alpha, cache={}):
    tag = "greedy" if alpha == 0 else "a%.3f" % alpha
    path = "data/co_%s_%s_%s.npz" % (world_name, "gated" if gated else "plain", tag)
    if os.path.exists(path):
        return np.load(path)["Q"]
    key = (world_name, gated)
    if key not in cache:
        cache[key] = (build_transitions(WORLDS[world_name], gated),
                      empowerment_reward(world_name, gated))
    M, r = cache[key]
    V, Q, it = value_iteration(M, gamma=0.99, tol=1e-2, max_iter=5000,
                               walls="redundant", reward=r, alpha=alpha)
    np.savez_compressed(path, Q=Q, V=V, iters=it)
    print("    solved %s in %d iterations" % (path, it), flush=True)
    return Q


def evaluate(world_name, gated, alpha):
    Q = solve(world_name, gated, alpha)
    agent = MOPAgent(Q, greedy=(alpha == 0), alpha=max(alpha, 1e-9))
    agg, rs = run_policy(agent, gated, world=WORLDS[world_name], n_episodes=NEP,
                         max_steps=MAXSTEPS)
    ms = meal_stats(rs)
    return dict(life=agg["steps"], life_sem=agg["steps_sem"],
                survived=int(sum(not r["died"] for r in rs)),
                trips=agg["trips_per_1k"], on_food=agg["frac_on_food"],
                alternation=ms["alternation"], meal_E=ms["E_pct"])


def fmt(s):
    return ("life %5.0f +- %-4.0f survive %2d/%d  trips/1k %5.1f  meals alternate %s  "
            "meal E %s" % (s["life"], s["life_sem"], s["survived"], NEP, s["trips"],
                           "%.2f" % s["alternation"] if s["alternation"] is not None else " -  ",
                           [round(v) for v in s["meal_E"]] if s["meal_E"] else "-"))


if __name__ == "__main__":
    out = {"main": {}, "alpha_sweep": {}}
    print("control-occupancy, %d episodes per cell, cap %d" % (NEP, MAXSTEPS))
    for wname in ("theirs", "patch"):
        for gated in (False, True):
            for label, alpha in (("greedy", 0.0), ("soft", ALPHA_BITS)):
                s = evaluate(wname, gated, alpha)
                out["main"]["%s|%d|%s" % (wname, gated, label)] = s
                print("  %-6s %-6s %-6s %s" % (wname, "gated" if gated else "plain",
                                               label, fmt(s)), flush=True)
    print("\nsoft control-occupancy, temperature sweep")
    for wname in ("theirs", "patch"):
        for gated in (False, True):
            for alpha in ALPHA_SWEEP:
                s = evaluate(wname, gated, alpha)
                out["alpha_sweep"]["%s|%d|%.3f" % (wname, gated, alpha)] = s
                print("  %-6s %-6s alpha %.2f  %s" % (wname, "gated" if gated else "plain",
                                                     alpha, fmt(s)), flush=True)
    with open("data/control_occupancy.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote data/control_occupancy.json")
