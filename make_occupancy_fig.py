"""What you count vs how you count it (Figure 21).

Five agents in the four conditions of section 22:
  empowerment, one-shot gain planner        (what: control;   how: one hop, 5-step C)
  MOP as prescribed (softmax)               (what: all variety; how: cumulative, soft)
  MOP's values, argmax                      (what: all variety; how: cumulative, hard)
  control-occupancy, soft                   (what: control + action entropy; cumulative, soft)
  cumulative empowerment, greedy            (what: control;   how: cumulative, hard)

Reads data/mylonas_results[_patch].json, data/robust_mop.json and
data/control_occupancy.json; writes paper/figs/myl7_occupancy.png.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a2e"; ACCENT = "#e94560"; COOL = "#0f7173"; GREY = "#8a8a9e"
GOLD = "#d98e04"; MOPC = "#55556e"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

R = {"theirs": json.load(open("data/mylonas_results.json")),
     "patch": json.load(open("data/mylonas_results_patch.json"))}
ARG = json.load(open("data/robust_mop.json"))["argmax_all"]
CO = json.load(open("data/control_occupancy.json"))["main"]
CONDS = [("theirs", 0), ("theirs", 1), ("patch", 0), ("patch", 1)]
CLAB = ["their world\nno gate", "their world\ngate", "patch world\nno gate",
        "patch world\ngate"]


def stat(agent, w, g):
    """(lifespan, alternation, meal-energy percentiles)."""
    r = R[w]
    if agent == "emp":
        return (r["arms"]["gain|%d" % g]["steps"], r["meals"]["gain|%d" % g]["alternation"],
                r["meals"]["gain|%d" % g]["E_pct"])
    if agent == "mop":
        return (r["refs"]["mop|%d" % g]["steps"], r["meals"]["mop|%d" % g]["alternation"],
                r["meals"]["mop|%d" % g]["E_pct"])
    if agent == "mop_arg":
        a = ARG["%s|%d|argmax" % (w, g)]
        return a["life"], a["alt"], a["mealE"]
    a = CO["%s|%d|%s" % (w, g, "greedy" if agent == "co_greedy" else "soft")]
    return a["life"], a["alternation"], a["meal_E"]


AGENTS = [("emp", "empowerment, one-shot planner (§22)", ACCENT, ACCENT, ""),
          ("mop", "MOP as prescribed (softmax)", MOPC, "white", ""),
          ("mop_arg", "MOP's values, argmax", MOPC, MOPC, ""),
          ("co_soft", "control-occupancy (empowerment + action entropy, soft)", COOL, "white", ""),
          ("co_greedy", "cumulative empowerment (greedy)", COOL, COOL, "")]


def main():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6),
                           gridspec_kw={"width_ratios": [1.5, 0.75, 0.75]})
    w = 0.16
    for j, (key, label, edge, face, _) in enumerate(AGENTS):
        for i, (wn, g) in enumerate(CONDS):
            life = stat(key, wn, g)[0]
            ax[0].bar(i + (j - 2) * w, life, width=w * 0.92, color=face, edgecolor=edge,
                      lw=1.5, label=label if i == 0 else None)
    for i, (wn, g) in enumerate(CONDS):
        for name, mk in (("parker", "_"), ("switcher", "x")):
            v = R[wn]["refs"]["%s|%d" % (name, g)]["steps"]
            ax[0].plot(i + 0.44, v, marker=mk, ms=10, mew=2, color=GOLD, ls="none",
                       label=("%s (objective-free)" % name) if i == 0 else None)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(CLAB, fontsize=9)
    ax[0].set_ylabel("mean lifespan (steps, cap 3000)", fontsize=9)
    ax[0].set_title("(a) survival", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2),
                 ncol=2)

    # (b) when to eat, no body gate
    ung = [("theirs", 0), ("patch", 0)]
    for j, (key, label, edge, face, _) in enumerate(AGENTS):
        for i, (wn, g) in enumerate(ung):
            lo, med, hi = stat(key, wn, g)[2]
            x = i + (j - 2) * w
            ax[1].bar(x, med, width=w * 0.92, color=face, edgecolor=edge, lw=1.5)
            ax[1].errorbar(x, med, yerr=[[med - lo], [hi - med]], color=INK, lw=1,
                           capsize=2)
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(["their world", "patch world"], fontsize=9)
    ax[1].set_ylim(0, 31)
    ax[1].set_ylabel("energy just before a meal\n(median, 10-90%)", fontsize=9)
    ax[1].set_title("(b) when to eat, NO body gate", fontsize=10)

    # (c) switching follows the payoff: the two soft agents, gated
    for key, label, edge, face, _ in AGENTS[1:4:2]:
        alt = [stat(key, wn, 1)[1] or 0.0 for wn in ("theirs", "patch")]
        ax[2].plot([0, 1], alt, "-", marker="D" if key == "mop" else "o", color=edge,
                   mfc=face, lw=2, ms=8)
    for key, label, edge, face, _ in (AGENTS[2], AGENTS[4]):
        alt = [stat(key, wn, 1)[1] or 0.0 for wn in ("theirs", "patch")]
        ax[2].plot([0, 1], alt, "--", marker="D" if key == "mop_arg" else "o",
                   color=edge, mfc=face, lw=1.5, ms=7, alpha=0.8)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["their world\n(switching lethal)",
                           "patch world\n(switching pays)"], fontsize=9)
    ax[2].set_xlim(-0.3, 1.3)
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_ylabel("share of consecutive meals\nat the OTHER source", fontsize=9)
    ax[2].set_title("(c) where to eat, gate on\n(solid: soft agents, dashed: greedy)",
                    fontsize=10)

    fig.tight_layout()
    fig.savefig("paper/figs/myl7_occupancy.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figs/myl7_occupancy.png")


if __name__ == "__main__":
    main()
