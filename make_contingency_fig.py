"""Cross-world summary for the paper (Figure 20).

(a) where to eat: share of consecutive meals at the other source, their world
    vs patch world, gated condition.
(b) survival, same agents.
(c) when to eat: energy at meals, all four conditions, empowerment vs MOP.

MOP is shown twice: as prescribed (softmax) and with its own values under an
argmax policy, which separates what MOP values from how it behaves.

Reads data/mylonas_results.json, data/mylonas_results_patch.json and
data/robust_mop.json; writes paper/figs/myl5_contingency.png.
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
WORLDS = ("theirs", "patch")


def emp(w, g, what):
    r = R[w]
    if what == "alt":
        return r["meals"]["gain|%d" % g]["alternation"] or 0.0
    if what == "E":
        return r["meals"]["gain|%d" % g]["E_pct"]
    return r["arms"]["gain|%d" % g]["steps"]


def ref(w, name, g, what):
    r = R[w]
    if what == "alt":
        return r["meals"]["%s|%d" % (name, g)]["alternation"] or 0.0
    if what == "E":
        return r["meals"]["%s|%d" % (name, g)]["E_pct"]
    return r["refs"]["%s|%d" % (name, g)]["steps"]


def argmax(w, g, what):
    a = ARG["%s|%d|argmax" % (w, g)]
    return {"alt": a["alt"] or 0.0, "E": a["mealE"], "life": a["life"]}[what]


def main():
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4),
                           gridspec_kw={"width_ratios": [1, 1, 1.35]})
    xs = [0, 1]
    series = [
        ("empowerment (gain planner)", lambda w, k: emp(w, 1, k), ACCENT, "-", "o", ACCENT),
        ("MOP as prescribed (softmax)", lambda w, k: ref(w, "mop", 1, k), MOPC, "-", "D", "none"),
        ("MOP's values, argmax policy", lambda w, k: argmax(w, 1, k), MOPC, "--", "D", MOPC),
        ("switcher", lambda w, k: ref(w, "switcher", 1, k), GOLD, "-", "x", GOLD),
        ("parker", lambda w, k: ref(w, "parker", 1, k), GOLD, ":", "|", GOLD),
    ]
    for label, f, col, ls, mk, mfc in series:
        ax[0].plot(xs, [f(w, "alt") for w in WORLDS], ls=ls, marker=mk, color=col,
                   mfc=mfc, lw=2, ms=8, label=label)
        ax[1].plot(xs, [f(w, "life") for w in WORLDS], ls=ls, marker=mk, color=col,
                   mfc=mfc, lw=2, ms=8)
    for a, yl, t in ((ax[0], "share of consecutive meals at the OTHER source",
                      "(a) where to eat"),
                     (ax[1], "mean lifespan (steps, cap 3000)", "(b) survival")):
        a.set_xticks(xs)
        a.set_xticklabels(["their world\n(switching lethal)",
                           "patch world\n(switching pays)"], fontsize=9)
        a.set_xlim(-0.3, 1.3)
        a.set_ylabel(yl, fontsize=9)
        a.set_title(t + ", body gate on", fontsize=10)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left",
                 bbox_to_anchor=(0.0, -0.2), ncol=3)

    # (c) when to eat
    conds = [("theirs", 0, "their world\nno gate"), ("theirs", 1, "their world\ngate"),
             ("patch", 0, "patch world\nno gate"), ("patch", 1, "patch world\ngate")]
    who = [("empowerment", lambda w, g: emp(w, g, "E"), ACCENT, ACCENT),
           ("MOP, softmax", lambda w, g: ref(w, "mop", g, "E"), MOPC, "white"),
           ("MOP's values, argmax", lambda w, g: argmax(w, g, "E"), MOPC, MOPC)]
    width = 0.26
    for j, (name, f, col, face) in enumerate(who):
        for i, (w, g, _) in enumerate(conds):
            lo, med, hi = f(w, g)
            xpos = i + (j - 1) * width
            ax[2].bar(xpos, med, width=width * 0.92, color=face, edgecolor=col, lw=1.5,
                      label=name if i == 0 else None)
            ax[2].errorbar(xpos, med, yerr=[[med - lo], [hi - med]], color=INK,
                           lw=1, capsize=2)
    ax[2].set_xticks(range(len(conds)))
    ax[2].set_xticklabels([c[2] for c in conds], fontsize=9)
    ax[2].set_ylim(0, 31)
    ax[2].set_ylabel("energy just before a meal\n(median, 10-90%)", fontsize=9)
    ax[2].set_title("(c) when to eat", fontsize=10)
    ax[2].legend(frameon=False, fontsize=8, loc="upper left",
                 bbox_to_anchor=(0.0, -0.2), ncol=3)

    fig.tight_layout()
    fig.savefig("paper/figs/myl5_contingency.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figs/myl5_contingency.png")


if __name__ == "__main__":
    main()
