"""Robustness checks on the empowerment-vs-MOP comparison.

A. MOP's beta (their Eq. 6): beta * H(s', b' | s, b, a) added to the action
   entropy.  beta < 0 penalises unpredictable transitions, which they note
   approximates empowerment.  Swept in all four conditions.

B. Why MOP loses in the gated patch world.  Hypothesis: its stochastic policy
   keeps it wandering at moderate energy, and the gate's motor noise compounds
   that.  Intervention: MOP's own values, argmax policy (greedy).  If greedy MOP
   survives like empowerment, the deficit is the policy's variability, not its
   values.  Plus a death anatomy for each agent.

Writes data/robust_mop.json and paper/figs/myl6_robustness.png.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from empower.mop_belief import MOPAgent, build_transitions, value_iteration
from empower.mylonas import (DIST_FOOD, MOVE, WORLDS, MylonasLUT, run_arm,
                             run_policy)
from diag_mylonas import excursion_by_energy, meal_stats

NEP = int(os.environ.get("MYL_EPISODES", 30))
MAXSTEPS = 3000
BETAS = [-1.0, -0.5, 0.0, 0.5]
WALLS = "redundant"

INK = "#1a1a2e"; ACCENT = "#e94560"; COOL = "#0f7173"; GREY = "#8a8a9e"
GOLD = "#d98e04"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def solve(world, gated, beta):
    tag = "" if beta == 0.0 else "_b%+.1f" % beta
    path = "data/mop_%s_%s_%s%s.npz" % (world.name, "gated" if gated else "plain",
                                        WALLS, tag)
    if os.path.exists(path):
        return np.load(path)["Q"]
    M, H = build_transitions(world, gated, return_entropy=True)
    V, Q, it = value_iteration(M, gamma=0.99, tol=1e-2, walls=WALLS, beta=beta, H=H)
    np.savez_compressed(path, Q=Q, V=V, iters=it)
    return Q


def summarise(agg, recs):
    ms = meal_stats(recs)
    return dict(life=agg["steps"], life_sem=agg["steps_sem"],
                survived=int(sum(not r["died"] for r in recs)),
                trips=agg["trips_per_1k"], on_food=agg["frac_on_food"],
                far=agg["frac_far"], alternation=ms["alternation"],
                meal_E=ms["E_pct"])


def death_anatomy(recs):
    """For each death: energy at the last meal, how far away it died, and how
    many of its moves after the last meal the body corrupted."""
    rows = []
    for r in recs:
        if not r["died"]:
            continue
        x, E, act = r["x"], r["E"], r["act"]
        meals = [t for t in range(1, len(x)) if E[t] > E[t - 1]]
        k = meals[-1] if meals else 0
        corrupted = sum(1 for t in range(k, len(x) - 1)
                        if x[t + 1] != MOVE[act[t], x[t]])
        below = [t for t in range(k, len(x)) if E[t] < 12]
        rows.append(dict(E_last_meal=int(E[k]),
                         steps_last_meal_to_death=int(len(x) - k),
                         dist_at_death=int(DIST_FOOD[x[-1]]),
                         dist_when_E_below_12=int(DIST_FOOD[x[below[0]]]) if below else None,
                         corrupted_after_last_meal=int(corrupted)))
    return rows


def mop_entropy_by_energy(agent, recs, bins=(0, 4, 8, 12, 16, 20, 24, 28, 31)):
    """Normalised policy entropy of MOP along its own trajectories, by energy."""
    E = np.concatenate([r["E"] for r in recs])
    Hs = np.concatenate([[-(p * np.log(np.where(p > 0, p, 1))).sum() / np.log(5)
                          for p in (agent.policy(xx, ee, b1, b2) for xx, ee, b1, b2 in
                                    zip(r["x"], r["E"], r["b1"], r["b2"]))]
                         for r in recs])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        out.append(float(Hs[m].mean()) if m.sum() > 50 else None)
    return out


def main():
    res = dict(beta_sweep={}, anatomy={})

    # ------------------------------------------------------------------ A
    print("A. MOP beta sweep (walls=%s, %d episodes, cap %d)" % (WALLS, NEP, MAXSTEPS))
    for wname in ("theirs", "patch"):
        for gated in (False, True):
            for beta in BETAS:
                Q = solve(WORLDS[wname], gated, beta)
                agg, rs = run_policy(MOPAgent(Q), gated, world=WORLDS[wname],
                                     n_episodes=NEP, max_steps=MAXSTEPS)
                s = summarise(agg, rs)
                res["beta_sweep"]["%s|%d|%+.1f" % (wname, gated, beta)] = s
                print("  %-6s %-6s beta %+.1f  life %5.0f +- %-4.0f survive %2d/%d  "
                      "trips/1k %5.1f  meals alternate %.2f  meal E %s"
                      % (wname, "gated" if gated else "plain", beta, s["life"],
                         s["life_sem"], s["survived"], NEP, s["trips"],
                         s["alternation"] or 0, [round(v) for v in s["meal_E"]]),
                      flush=True)

    # ------------------------------------------------------------------ B
    print("\nB. gated patch world: anatomy and the argmax intervention")
    W = WORLDS["patch"]
    lut = MylonasLUT.load("data/mylonas_C_n5_gated_patch.npz")
    Q0 = solve(W, True, 0.0)
    agents = {
        "empowerment gain + gate": None,
        "MOP (softmax, beta 0)": MOPAgent(Q0),
        "MOP values, argmax policy": MOPAgent(Q0, greedy=True),
    }
    trajs = {}
    for name, ag in agents.items():
        if ag is None:
            agg, rs = run_arm(lut, "gain", True, n_episodes=NEP, gamma=0.9,
                              world=W, max_steps=MAXSTEPS)
        else:
            agg, rs = run_policy(ag, True, world=W, n_episodes=NEP, max_steps=MAXSTEPS)
        trajs[name] = rs
        s = summarise(agg, rs)
        da = death_anatomy(rs)
        ex = excursion_by_energy(rs, bins=(0, 4, 8, 12, 16, 20, 24, 28, 31))
        s["excursion"] = [row["mean_dist"] for row in ex]
        s["deaths"] = da
        if da:
            s["death_summary"] = dict(
                E_last_meal=float(np.median([d["E_last_meal"] for d in da])),
                dist_at_death=float(np.median([d["dist_at_death"] for d in da])),
                dist_when_E_below_12=float(np.median([d["dist_when_E_below_12"]
                                                      for d in da
                                                      if d["dist_when_E_below_12"] is not None])),
                corrupted_after_last_meal=float(np.median([d["corrupted_after_last_meal"]
                                                           for d in da])))
        if isinstance(ag, MOPAgent) and not ag.greedy:
            s["policy_entropy_by_E"] = mop_entropy_by_energy(ag, rs[:10])
        res["anatomy"][name] = s
        print("  %-26s life %5.0f +- %-4.0f survive %2d/%d  meals alternate %.2f  "
              "meal E %s" % (name, s["life"], s["life_sem"], s["survived"], NEP,
                             s["alternation"] or 0, [round(v) for v in s["meal_E"]]))
        if da:
            d = s["death_summary"]
            print("      deaths %2d: median E at last meal %.0f, distance from food when "
                  "E fell below 12: %.0f, at death: %.0f, corrupted moves after last "
                  "meal: %.0f" % (len(da), d["E_last_meal"], d["dist_when_E_below_12"],
                                  d["dist_at_death"], d["corrupted_after_last_meal"]))
        print("      mean distance from food by energy (0-4 ... 28-30): %s"
              % " ".join("%.2f" % v if v is not None else "-" for v in s["excursion"]))
        if "policy_entropy_by_E" in s:
            print("      MOP normalised policy entropy by energy: %s"
                  % " ".join("%.2f" % v if v is not None else "-"
                             for v in s["policy_entropy_by_E"]))

    with open("data/robust_mop.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote data/robust_mop.json")
    fig_robust(res)


def fig_robust(res):
    """Supplementary figure: MOP's beta, in all four conditions."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    conds = [("theirs", 0, ":"), ("theirs", 1, "-"), ("patch", 0, ":"), ("patch", 1, "-")]
    cols = {"theirs": COOL, "patch": ACCENT}
    names = {"theirs": "their world", "patch": "patch world"}
    for wname, g, ls in conds:
        key = lambda b: "%s|%d|%+.1f" % (wname, g, b)
        alt = [res["beta_sweep"][key(b)]["alternation"] or 0 for b in BETAS]
        life = [res["beta_sweep"][key(b)]["life"] for b in BETAS]
        lab = "%s, %s" % (names[wname], "body gate" if g else "no gate")
        ax[0].plot(BETAS, alt, ls, marker="D", mfc="none", color=cols[wname], lw=2,
                   label=lab)
        ax[1].plot(BETAS, life, ls, marker="D", mfc="none", color=cols[wname], lw=2)
    ax[0].set_xlabel("MOP beta  (< 0: penalise unpredictable transitions)", fontsize=9)
    ax[0].set_ylabel("share of consecutive meals " + chr(10) + "at the OTHER source", fontsize=9)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) switching", fontsize=10)
    ax[1].set_xlabel("MOP beta", fontsize=9)
    ax[1].set_ylabel("mean lifespan (steps, cap 3000)", fontsize=9)
    ax[1].set_title("(b) survival", fontsize=10)
    fig.tight_layout()
    fig.savefig("paper/figs/myl6_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figs/myl6_robustness.png")


if __name__ == "__main__":
    main()
