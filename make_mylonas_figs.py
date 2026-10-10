"""Mylonas & Moreno-Bote (arXiv:2609.39342) replication: four empowerment arms.

Run `python build_mylonas_tables.py` first (it caches the two C tables).

    MYL_WORLD=theirs  python make_mylonas_figs.py   -> paper/figs/myl*.png
    MYL_WORLD=patch   python make_mylonas_figs.py   -> paper/figs/mylp*.png

Results go to data/mylonas_results[_patch].json.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from empower.mylonas import (DIST_FOOD, EMAX, FOOD, GRID, NPOS, WORLDS,
                             MylonasLUT, Parker, Switcher, gate_g, run_arm,
                             run_policy)
from diag_mylonas import excursion_by_energy, meal_stats, switching
from empower.mop_belief import MOPAgent

FIG = "paper/figs"
os.makedirs(FIG, exist_ok=True)
os.makedirs("data", exist_ok=True)

INK = "#1a1a2e"; ACCENT = "#e94560"; COOL = "#0f7173"; GREY = "#8a8a9e"
GOLD = "#d98e04"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
EMAP = LinearSegmentedColormap.from_list(
    "emp", ["#10132b", "#26418f", "#1f9e89", "#7fd34e", "#f6e27a"])

WNAME = os.environ.get("MYL_WORLD", "theirs")
WORLD = WORLDS[WNAME]
SUFFIX = "" if WNAME == "theirs" else "_" + WNAME
PFX = "myl" if WNAME == "theirs" else "myl" + WNAME[0]

# What their paper reports (their world only).
REPORTED = {
    "MOP": dict(trips_per_1k=13.3, frac_on_food=0.26, frac_far=0.42,
                both=0.94, src="their Fig. 3, Fig. 6"),
    "EFE (d=0.8)": dict(trips_per_1k=3.8, frac_on_food=0.41, frac_far=0.22,
                        both=None, src="their Fig. 6c"),
}

ARMS = [("level", False, "greedy ascent, no gate"
         + (" (their arm)" if WNAME == "theirs" else "")),
        ("gain", False, "gain planner, no gate"),
        ("level", True, "greedy ascent + body gate"),
        ("gain", True, "gain planner + body gate")]
REFS = [("parker", Parker()), ("switcher", Switcher())]
# MOP, calibrated to their reported numbers (see run_mop.py); it has an
# objective, so it is drawn as a comparison, not as an objective-free reference.
MOP_WALLS = "redundant"

NEP = int(os.environ.get("MYL_EPISODES", 30))
MAXSTEPS = int(os.environ.get("MYL_MAXSTEPS", 3000))
GAMMA = float(os.environ.get("MYL_GAMMA", 0.9))


def key_of(drive, gated):
    return drive + "|" + str(int(gated))


def load_luts(n=5):
    return {g: MylonasLUT.load("data/mylonas_C_n%d_%s%s.npz"
                               % (n, "gated" if g else "plain", SUFFIX))
            for g in (False, True)}


def heat(x):
    h = np.zeros(NPOS)
    np.add.at(h, x, 1.0)
    return h.reshape(GRID, GRID) / h.sum()


# ====================================================================== fig 1
def fig_mechanism(luts):
    """Why their empowerment arm never eats, and what the body gate changes."""
    Es = np.arange(1, EMAX + 1)
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.6),
                           gridspec_kw={"width_ratios": [0.9, 1, 1, 1]})

    F = np.array([luts[False].C(x, 20, 1.0, 1.0) for x in range(NPOS)])
    F = F.reshape(GRID, GRID)
    im = ax[0].imshow(F, cmap=EMAP)
    for f in FOOD:
        ax[0].plot(f % GRID, f // GRID, marker="s", ms=11, mfc="none",
                   mec="white", mew=1.8)
    for r in range(GRID):
        for c in range(GRID):
            ax[0].text(c, r, "%.2f" % F[r, c], ha="center", va="center",
                       fontsize=7, color="white" if F[r, c] < 4.9 else INK)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("%s world, E=20: C peaks NEXT TO food" % WNAME, fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)

    nb = 1                                   # the cell beside food source 0
    for gated, col, lab in [(False, COOL, "no gate"),
                            (True, ACCENT, "+ graded body gate")]:
        ax[1].plot(Es, [luts[gated].C(nb, E, 1.0, 1.0) for E in Es], color=col,
                   lw=2, label=lab)
    ax[1].set_xlabel("energy E")
    ax[1].set_ylabel("empowerment C (bits)")
    ax[1].set_title("C beside a source, vs energy", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")

    ax[2].plot(Es, [gate_g(E) for E in Es], color=ACCENT, lw=2)
    ax[2].axvline(0.40 * EMAX, color=GREY, ls=":", lw=1)
    ax[2].set_xlabel("energy E")
    ax[2].set_ylabel("actuator corruption g(E)")
    ax[2].set_title("the only change to the world", fontsize=10)

    for gated, col, lab in [(False, COOL, "no gate"),
                            (True, ACCENT, "+ graded body gate")]:
        lut = luts[gated]
        g = [lut.C(FOOD[0], min(E + 15, EMAX), 1.0, 1.0)
             - lut.C(FOOD[0], E, 1.0, 1.0) for E in Es]
        ax[3].plot(Es, g, color=col, lw=2, label=lab)
    ax[3].axhline(0, color=GREY, lw=0.8)
    ax[3].set_xlabel("energy E before the meal")
    ax[3].set_ylabel("empowerment restored by a meal (bits)")
    ax[3].set_title("the somatic face (belief held fixed)", fontsize=10)
    ax[3].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig("%s/%s1_mechanism.png" % (FIG, PFX), bbox_inches="tight")
    plt.close(fig)


# ====================================================================== fig 2
def fig_arms(results, recs):
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6),
                             gridspec_kw={"height_ratios": [1, 0.85]})
    for k, (drive, gated, label) in enumerate(ARMS):
        key = key_of(drive, gated)
        m = results[key]
        H = sum(heat(r["x"]) for r in recs[key]) / len(recs[key])
        a = axes[0, k]
        a.imshow(H, cmap=EMAP)
        for f in FOOD:
            a.plot(f % GRID, f // GRID, marker="s", ms=9, mfc="none",
                   mec="white", mew=1.6)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(label.replace(" (", "\n("), fontsize=10)
        a.set_xlabel("%.1f trips/1k    %.0f%% on food\nmean lifespan %.0f steps"
                     % (m["trips_per_1k"], 100 * m["frac_on_food"], m["steps"]),
                     fontsize=9)

        lives = np.array([len(q["x"]) for q in recs[key]])
        r = recs[key][int(np.argmin(np.abs(lives - np.median(lives))))]
        T = min(400, len(r["x"]))
        b = axes[1, k]
        b.plot(np.arange(T), r["E"][:T], color=COOL, lw=1.0)
        b.set_ylim(0, EMAX + 1)
        b.set_xlabel("step (median-lifespan episode)", fontsize=9)
        if k == 0:
            b.set_ylabel("energy", color=COOL)
        b2 = b.twinx()
        b2.plot(np.arange(T), DIST_FOOD[r["x"][:T]], color=ACCENT, lw=0.9,
                alpha=0.85)
        b2.set_ylim(-0.3, 5)
        b2.spines["top"].set_visible(False)
        if k == len(ARMS) - 1:
            b2.set_ylabel("distance to nearest source", color=ACCENT)
        else:
            b2.set_yticks([])
    fig.suptitle("%s world" % WNAME, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig("%s/%s2_arms.png" % (FIG, PFX), bbox_inches="tight")
    plt.close(fig)


# ====================================================================== fig 3
def fig_compare(results, refs, meals):
    """Arms against the objective-free references of the same gate condition."""
    labels = [lab for _, _, lab in ARMS]
    keys = [key_of(d, g) for d, g, _ in ARMS]
    gates = [g for _, g, _ in ARMS]
    cols = [COOL, COOL, ACCENT, ACCENT]
    hatch = ["", "//", "", "//"]
    y = np.arange(len(keys))

    def val(k, metric):
        if metric == "alternation":
            return meals[k]["alternation"] or 0.0
        if metric == "meal_E":
            return meals[k]["E_pct"][1] if meals[k]["E_pct"] else np.nan
        return (results.get(k) or refs.get(k))[metric]

    panels = [("steps", "mean lifespan (steps, cap %d)" % MAXSTEPS),
              ("trips_per_1k", "trips between sources per 1000 steps"),
              ("alternation", "share of consecutive meals " + chr(10) + "taken at the OTHER source"),
              ("meal_E", "energy just before a meal " + chr(10) + "(median; bars 10-90%)")]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.9))
    for j, (metric, name) in enumerate(panels):
        vals = [val(k, metric) for k in keys]
        errs = None
        if metric in ("steps", "trips_per_1k"):
            errs = [results[k][metric + "_sem"] for k in keys]
        elif metric == "meal_E":
            lo = [v - meals[k]["E_pct"][0] if meals[k]["E_pct"] else 0
                  for k, v in zip(keys, vals)]
            hi = [meals[k]["E_pct"][2] - v if meals[k]["E_pct"] else 0
                  for k, v in zip(keys, vals)]
            errs = [lo, hi]
        ax[j].barh(y, vals, xerr=errs, color=cols, hatch=hatch,
                   edgecolor="white", height=0.62)
        for yy, g in zip(y, gates):
            for (rname, _), mk in zip(REFS, ("|", "x")):
                v = val("%s|%d" % (rname, g), metric)
                ax[j].plot(v, yy, marker=mk, ms=11, mew=2, color=GOLD, ls="none")
            v = val("mop|%d" % g, metric)
            ax[j].plot(v, yy, marker="D", ms=8, mew=1.5, mfc="none", color=INK, ls="none")
        if WNAME == "theirs" and metric == "trips_per_1k":
            for nm, style in [("MOP", "-"), ("EFE (d=0.8)", "--")]:
                v = REPORTED[nm][metric]
                ax[j].axvline(v, color=GREY, ls=style, lw=1.2)
                ax[j].text(v, -0.75, " " + nm, color=GREY, fontsize=8)
        if metric == "meal_E":
            ax[j].set_xlim(0, EMAX)
        ax[j].set_yticks(y)
        ax[j].set_yticklabels(labels if j == 0 else [""] * len(y), fontsize=9)
        ax[j].invert_yaxis()
        ax[j].set_xlabel(name, fontsize=9)
    ax[0].plot([], [], marker="|", ms=11, mew=2, color=GOLD, ls="none",
               label="parker (same gate)")
    ax[0].plot([], [], marker="x", ms=9, mew=2, color=GOLD, ls="none",
               label="switcher (same gate)")
    ax[0].plot([], [], marker="D", ms=8, mew=1.5, mfc="none", color=INK, ls="none",
               label="MOP, calibrated (same gate)")
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("%s world: empowerment arms vs objective-free references"
                 % WNAME, fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig("%s/%s3_compare.png" % (FIG, PFX), bbox_inches="tight")
    plt.close(fig)


# ====================================================================== fig 4
def fig_diagnostics(results, recs, refs):
    """Lifespan against the best objective-free policy, and excursion by energy."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    keys = [key_of(d, g) for d, g, _ in ARMS]
    labels = [lab for _, _, lab in ARMS]
    best = {g: max(refs["parker|%d" % g]["steps"], refs["switcher|%d" % g]["steps"])
            for g in (0, 1)}
    best_name = {g: ("parker" if refs["parker|%d" % g]["steps"]
                     >= refs["switcher|%d" % g]["steps"] else "switcher")
                 for g in (0, 1)}
    frac = [results[k]["steps"] / best[int(g)] for k, (_, g, _) in zip(keys, ARMS)]
    y = np.arange(len(keys))
    ax[0].barh(y, frac, color=[COOL, COOL, ACCENT, ACCENT],
               hatch=["", "//", "", "//"], edgecolor="white", height=0.62)
    for yy, f in zip(y, frac):
        ax[0].text(f + 0.01, yy, "%.0f%%" % (100 * f), va="center", fontsize=9)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].invert_yaxis(); ax[0].set_xlim(0, max(1.05, max(frac) + 0.1))
    ax[0].set_xlabel("lifespan / best objective-free policy, same gate", fontsize=9)
    ax[0].set_title("best: no gate = %s (%.0f), gated = %s (%.0f)"
                    % (best_name[0], best[0], best_name[1], best[1]),
                    fontsize=9, color=GREY)

    for (drive, gated, label), ls in [(ARMS[2], "-"), (ARMS[3], "--")]:
        ex = excursion_by_energy(recs[key_of(drive, True)],
                                 bins=(0, 4, 8, 12, 16, 20, 24, 28, 31))
        mids = [np.mean([int(v) for v in row["E"].split("-")]) for row in ex
                if row["n"] > 50]
        dist = [row["mean_dist"] for row in ex if row["n"] > 50]
        ax[1].plot(mids, dist, ls, marker="o", color=ACCENT, lw=2, label=label)
    ax[1].axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax[1].text(29, 1.03, "the cell beside the food", color=GREY, fontsize=8,
               ha="right", va="bottom")
    ax[1].set_xlabel("energy E")
    ax[1].set_ylabel("mean distance to nearest source", fontsize=9)
    ax[1].set_title("excursion by energy", fontsize=9, color=GREY)
    ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("%s/%s4_diagnostics.png" % (FIG, PFX), bbox_inches="tight")
    plt.close(fig)
    return best, best_name


def main():
    luts = load_luts()
    todo = sys.argv[1:] or (["mechanism", "arms"] if WNAME == "theirs" else ["arms"])
    print("world:", WORLD, flush=True)

    if "mechanism" in todo:
        print("fig: mechanism", flush=True)
        fig_mechanism(luts)

    if "arms" in todo:
        results, recs, refs, sw, meals = {}, {}, {}, {}, {}
        for drive, gated, label in ARMS:
            agg, rs = run_arm(luts[gated], drive, gated, n_episodes=NEP,
                              gamma=GAMMA, world=WORLD, max_steps=MAXSTEPS)
            k = key_of(drive, gated)
            results[k], recs[k], sw[k], meals[k] = agg, rs, switching(rs), meal_stats(rs)
            print("  %-34s life %6.0f +- %-4.0f trips/1k %5.1f  meals alternate %s  "
                  "meal E 10/50/90 %s"
                  % (label, agg["steps"], agg["steps_sem"], agg["trips_per_1k"],
                     meals[k]["alternation"], meals[k]["E_pct"]), flush=True)
        for rname, pol in REFS:
            for g in (False, True):
                agg, rs = run_policy(pol, g, world=WORLD, n_episodes=NEP,
                                     max_steps=MAXSTEPS)
                k = "%s|%d" % (rname, g)
                refs[k], sw[k], meals[k] = agg, switching(rs), meal_stats(rs)
                print("  %-34s life %6.0f +- %-4.0f trips/1k %5.1f  on food %3.0f%%"
                      % ("[ref] %s %s" % (rname, "gated" if g else "no gate"),
                         agg["steps"], agg["steps_sem"], agg["trips_per_1k"],
                         100 * agg["frac_on_food"]), flush=True)
        for g in (False, True):
            Qm = np.load("data/mop_%s_%s_%s.npz" % (WNAME, "gated" if g else "plain",
                                                    MOP_WALLS))["Q"]
            agg, rs = run_policy(MOPAgent(Qm), g, world=WORLD, n_episodes=NEP,
                                 max_steps=MAXSTEPS)
            k = "mop|%d" % g
            refs[k], sw[k], meals[k] = agg, switching(rs), meal_stats(rs)
            print("  %-34s life %6.0f +- %-4.0f trips/1k %5.1f  meals alternate %s"
                  % ("[cmp] MOP %s" % ("gated" if g else "no gate"), agg["steps"],
                     agg["steps_sem"], agg["trips_per_1k"], meals[k]["alternation"]),
                  flush=True)
        fig_arms(results, recs)
        fig_compare(results, refs, meals)
        best, best_name = fig_diagnostics(results, recs, refs)
        out = "data/mylonas_results%s.json" % SUFFIX
        with open(out, "w") as fh:
            json.dump(dict(world=repr(WORLD), arms=results, refs=refs, switching=sw,
                           meals=meals,
                           reported=REPORTED if WNAME == "theirs" else None,
                           gamma=GAMMA, n_episodes=NEP, max_steps=MAXSTEPS,
                           best_reference=dict(steps=best, name=best_name)),
                      fh, indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
