"""MOP (their Eq. 6) in both worlds, with and without the body gate.

Calibrated against their reported MOP behaviour in their world before use:
with wall-bumping actions unavailable (walls='redundant') our MOP gives
11.6 trips/1k, 27% on food, 41% two or more steps away, 92% reaching both
sources, against their 13.3 / 26% / 42% / 94%.  With wall-bumping legal it
gives 3.5 / 49% / 19% / 89%, so 'redundant' is used throughout and 'legal' is
kept as a robustness variant.

Writes data/mop_<world>_<plain|gated>_<walls>.npz (Q tables) and
data/mop_results.json.
"""

import json
import os
import sys

import numpy as np

from empower.mop_belief import MOPAgent, build_transitions, value_iteration
from empower.mylonas import WORLDS, run_policy
from diag_mylonas import meal_stats

NEP = int(os.environ.get("MYL_EPISODES", 30))
MAXSTEPS = int(os.environ.get("MYL_MAXSTEPS", 3000))


def solve(world, gated, walls):
    path = "data/mop_%s_%s_%s.npz" % (world.name, "gated" if gated else "plain", walls)
    if os.path.exists(path):
        return np.load(path)["Q"]
    V, Q, it = value_iteration(build_transitions(world, gated), gamma=0.99,
                               tol=1e-2, walls=walls)
    np.savez_compressed(path, Q=Q, V=V, iters=it)
    print("  solved %s (%d iterations)" % (path, it), flush=True)
    return Q


if __name__ == "__main__":
    walls_list = sys.argv[1:] or ["redundant", "legal"]
    out = {}
    for walls in walls_list:
        for wname in ("theirs", "patch"):
            for gated in (False, True):
                Q = solve(WORLDS[wname], gated, walls)
                agg, rs = run_policy(MOPAgent(Q), gated, world=WORLDS[wname],
                                     n_episodes=NEP, max_steps=MAXSTEPS)
                ms = meal_stats(rs)
                key = "%s|%s|%d" % (walls, wname, gated)
                out[key] = dict(agg=agg, meals=ms,
                                died=int(sum(r["died"] for r in rs)))
                print("%-9s %-6s %-6s life %5.0f +- %-4.0f died %2d/%d  trips/1k %5.1f  "
                      "on food %2.0f%%  far %2.0f%%  meals alternate %s  meal E %s"
                      % (walls, wname, "gated" if gated else "plain",
                         agg["steps"], agg["steps_sem"], out[key]["died"], NEP,
                         agg["trips_per_1k"], 100 * agg["frac_on_food"],
                         100 * agg["frac_far"],
                         "%.2f" % ms["alternation"] if ms["alternation"] is not None else "-",
                         [round(v) for v in ms["E_pct"]] if ms["E_pct"] else "-"),
                      flush=True)
    with open("data/mop_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote data/mop_results.json")
