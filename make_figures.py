"""Generate all figures for the paper and the JSON payload for the live demo."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from empower.arena import Arena, NA
from empower.agent import EmpowermentLUT
from empower.empowerment import empowerment_field, field_to_grid, body_gate
from empower.model import LearnedDynamics
from empower.simulate import simulate
from empower import diagnostics as dg

FIG = "paper/figs"
os.makedirs(FIG, exist_ok=True)
os.makedirs("demo", exist_ok=True)

INK = "#1a1a2e"; ACCENT = "#e94560"; COOL = "#0f7173"; GOLD = "#d98e04"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
EMAP = LinearSegmentedColormap.from_list(
    "emp", ["#10132b", "#26418f", "#1f9e89", "#7fd34e", "#f6e27a"])

print("building arena + empowerment LUT ...")
A = Arena()
LUT = EmpowermentLUT(A, n=3, n_bins=21)
RES = A.resource_s
HEAD = dict(delta=0.006, intake=0.04, lam=0.996, gamma=0.94)


def occupancy(out):
    occ = np.zeros((A.rows, A.cols))
    for r, c in zip(out["row"][200:], out["col"][200:]):
        occ[r, c] += 1
    occ[A.wall] = np.nan
    return occ


# ====================================================================== fig1
def fig_field():
    grid = field_to_grid(A, empowerment_field(A, 1.0, n=3))
    bs = np.linspace(0, 1, 41)
    oc, rs, co = A.cell_index[(4, 3)], RES, A.cell_index[(1, 1)]
    cur = {k: [] for k in ("open", "res", "corner")}
    for b in bs:
        Eb = empowerment_field(A, b, n=3)
        cur["open"].append(Eb[oc]); cur["res"].append(Eb[rs]); cur["corner"].append(Eb[co])

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax[0].imshow(grid, cmap=EMAP)
    rr, rc = A.resource_rc
    ax[0].scatter([rc], [rr], marker="*", s=240, c="white", edgecolor=INK, lw=1.2, zorder=5)
    ax[0].annotate("resource\n(moderate $E$)", (rc, rr), (rc + 2.7, rr + 0.1),
                   color="white", fontsize=8.5, ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color="white"))
    ax[0].text(3, 4.2, "basin", color="white", ha="center", fontsize=9, style="italic")
    ax[0].text(13, 4.2, "basin", color="white", ha="center", fontsize=9, style="italic")
    ax[0].text(8, 6.6, "corridor", color="#dddddd", ha="center", fontsize=8, style="italic")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("(a)  3-step empowerment field $E(s)$ at full body $b{=}1$", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax[0], fraction=0.035, pad=0.02); cb.set_label("bits", fontsize=9)

    ax[1].plot(bs, cur["open"], color=ACCENT, lw=2.2, label="open basin (high ceiling)")
    ax[1].plot(bs, cur["res"], color=COOL, lw=2.2, label="resource (moderate)")
    ax[1].plot(bs, cur["corner"], color=GOLD, lw=2.2, label="corner (low)")
    ax[1].axvspan(0.78, 1.0, color="#cfe8cf", alpha=0.55, lw=0)
    ax[1].text(0.89, 0.45, "satiation\nplateau\n$\\partial E/\\partial b\\to0$", ha="center", fontsize=8, color="#2a6b2a")
    ax[1].axvspan(0.30, 0.50, color="#f6d2d2", alpha=0.55, lw=0)
    ax[1].text(0.40, 3.5, "hunger\ncliff", ha="center", fontsize=8.5, color="#9b2226")
    ax[1].set_xlabel("body integrity  $b$"); ax[1].set_ylabel("empowerment  $E(s,b)$ (bits)")
    ax[1].set_title("(b)  body integrity is a serial capacity bottleneck", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_field.png", bbox_inches="tight"); plt.close(fig)
    print("  fig1_field done")


# ====================================================================== fig2
def fig_matrix():
    cfg = {
        ("level", "no feedback"): dict(drive="level", depletion=False, forgetting=False),
        ("level", "body + forgetting"): dict(drive="level", depletion=True, forgetting=True),
        ("gain", "no feedback"): dict(drive="gain", depletion=False, forgetting=False),
        ("gain", "body + forgetting"): dict(drive="gain", depletion=True, forgetting=True),
    }
    verdict = {
        ("level", "no feedback"): "parks at a high-$E$ cell\n(fixed point)",
        ("level", "body + forgetting"): "refuel limit cycle\n(both phases exploit)",
        ("gain", "no feedback"): "sweeps, then quiesces\n(de Abril/Kanai)",
        ("gain", "body + forgetting"): "consolidate$\\leftrightarrow$venture\n(the result)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    rows, cols = ["level", "gain"], ["no feedback", "body + forgetting"]
    for i, rdr in enumerate(rows):
        for j, cfb in enumerate(cols):
            ax = axes[i, j]
            p = dict(HEAD); p.update(cfg[(rdr, cfb)])
            out = simulate(A, LUT, steps=1600, seed=5, **p)
            occ = occupancy(out)
            ax.imshow(np.log1p(occ), cmap=EMAP)
            rr, rc = A.resource_rc
            ax.scatter([rc], [rr], marker="*", s=120, c="white", edgecolor=INK, lw=0.8, zorder=5)
            ncell = len(set(out["cell"][200:].tolist()))
            ax.set_xticks([]); ax.set_yticks([])
            hl = ("gain" in rdr and "forget" in cfb)
            ax.set_title(verdict[(rdr, cfb)], fontsize=9,
                         color=ACCENT if hl else INK, fontweight="bold" if hl else "normal")
            ax.text(0.5, -0.07, "cells occupied: %d / %d" % (ncell, A.S),
                    transform=ax.transAxes, ha="center", fontsize=8, color="#555")
            if hl:
                for sp in ax.spines.values():
                    sp.set_edgecolor(ACCENT); sp.set_linewidth(2); sp.set_visible(True)
    for j, c in enumerate(cols):
        axes[0, j].annotate(c, (0.5, 1.28), xycoords="axes fraction", ha="center",
                            fontsize=11, fontweight="bold")
    for i, r in enumerate(rows):
        axes[i, 0].annotate("%s\ndrive" % r, (-0.18, 0.5), xycoords="axes fraction",
                            ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")
    fig.suptitle("Experiment matrix: occupancy maps (log dwell). Only the gain drive "
                 "with feedback both consolidates and explores.", fontsize=10.5, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_matrix.png", bbox_inches="tight"); plt.close(fig)
    print("  fig2_matrix done")


# ====================================================================== fig3
def fig_headline():
    out = simulate(A, LUT, steps=1400, drive="gain", seed=3, **HEAD)
    t = np.arange(len(out["b"]))
    dH = np.abs(out["row"] - A.resource_rc[0]) + np.abs(out["col"] - A.resource_rc[1])
    ph = out["phase"]; win = (300, 820)

    fig, ax = plt.subplots(3, 1, figsize=(10, 6.0), sharex=True)
    def shade(a):
        v = (ph == "venture").astype(int)
        d = np.diff(np.concatenate([[0], v, [0]]))
        for s0, s1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            a.axvspan(s0, s1, color=ACCENT, alpha=0.09, lw=0)
    shade(ax[0]); ax[0].plot(t, out["b"], color=INK, lw=1.5)
    ax[0].set_ylabel("body  $b$"); ax[0].set_ylim(0.35, 0.95)
    ax[0].set_title("One empowerment-gain drive; shaded = venture phase", fontsize=11)
    shade(ax[1])
    ax[1].plot(t, out["somatic"], color=COOL, lw=1.4, label="somatic gain (consolidate)")
    ax[1].plot(t, out["epistemic"], color=ACCENT, lw=1.4, label="epistemic gain (venture)")
    ax[1].set_ylabel("realized gain\n(bits)")
    ax[1].legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    shade(ax[2]); ax[2].plot(t, dH, color=GOLD, lw=1.3)
    ax[2].set_ylabel("distance from\nresource"); ax[2].set_xlabel("step")
    ax[2].set_xlim(*win)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_headline.png", bbox_inches="tight"); plt.close(fig)
    print("  fig3_headline done (cons %.0f%% / vent %.0f%%)"
          % (100*np.mean(ph == "consolidate"), 100*np.mean(ph == "venture")))


# ====================================================================== fig4
def _delay(x, tau):
    return x[:-tau], x[tau:]

def fig_phaseplane():
    g = simulate(A, LUT, steps=2500, drive="gain", seed=4, **HEAD)
    rnd = simulate(A, LUT, steps=2500, drive="random", seed=4, **HEAD)
    lvl = simulate(A, LUT, steps=2500, drive="level", depletion=False, forgetting=False, seed=4, **HEAD)

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.3))
    s, e = g["somatic"][200:], g["epistemic"][200:]
    sc = ax[0].scatter(s, e, s=7, c=np.arange(len(s)), cmap="viridis", alpha=0.6)
    ax[0].set_xlabel("somatic gain (consolidate face)")
    ax[0].set_ylabel("epistemic gain (venture face)")
    ax[0].set_title("(a)  the two faces never co-fire", fontsize=10.5)
    ax[0].text(0.97, 0.95, "empty:\nboth faces high", transform=ax[0].transAxes,
               ha="right", va="top", fontsize=8.5, color="#777")
    ax[0].plot([0, 2.4], [2.4, 0], ls=":", color="#bbb", lw=1)
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.04, pad=0.02); cb.set_label("step", fontsize=8)

    def dH(o):
        return np.abs(o["row"][200:] - A.resource_rc[0]) + np.abs(o["col"][200:] - A.resource_rc[1])
    maxlag = 90
    ax[1].axhline(0, color="#ccc", lw=0.8)
    for out, col, lab in [(lvl, INK, "level no-fb (no oscillation)"),
                          (rnd, "#9aa0a6", "random (monotonic decay)"),
                          (g, ACCENT, "gain drive (oscillatory)")]:
        ac = dg.phase_autocorr(dH(out), maxlag=maxlag)
        ax[1].plot(np.arange(maxlag), ac, color=col, lw=1.7, label=lab)
    ax[1].set_xlabel("lag (steps)"); ax[1].set_ylabel("autocorrelation of\ndistance-from-resource")
    ax[1].set_title("(b)  only the gain drive is periodic", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Phase-space evidence for a limit cycle", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_phaseplane.png", bbox_inches="tight"); plt.close(fig)
    print("  fig4_phaseplane done")


# ====================================================================== fig5
def fig_decomposition():
    out = simulate(A, LUT, steps=4000, drive="gain", seed=7, **HEAD)
    onsets = dg.venture_onsets(out["phase"], refractory=4, burn=200)
    # align cycles to venture onset and average somatic/epistemic share
    L = 34
    som = out["somatic"]; epi = out["epistemic"]
    stacks_s, stacks_e = [], []
    for o in onsets:
        if o - 12 >= 0 and o + L < len(som):
            seg_s = som[o-12:o+L]; seg_e = epi[o-12:o+L]
            tot = seg_s + seg_e + 1e-9
            stacks_s.append(seg_s/tot); stacks_e.append(seg_e/tot)
    ms = np.mean(stacks_s, axis=0); me = np.mean(stacks_e, axis=0)
    x = np.arange(-12, L)

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax[0].fill_between(x, 0, ms, color=COOL, alpha=0.85, label="somatic share")
    ax[0].fill_between(x, ms, 1, color=ACCENT, alpha=0.85, label="epistemic share")
    ax[0].axvline(0, color=INK, ls="--", lw=1)
    ax[0].text(0.2, 0.06, "venture onset", rotation=90, fontsize=8, color=INK)
    ax[0].set_xlim(-12, L-1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("steps relative to venture onset")
    ax[0].set_ylabel("share of the single gain stream")
    ax[0].set_title("(a)  dominance hands off — no mixing weight", fontsize=10.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="center left")

    # per-phase dominance histogram
    isv = out["phase"][200:] == "venture"
    s = out["somatic"][200:]; e = out["epistemic"][200:]
    share_e = e/(s+e+1e-9)
    ax[1].hist(share_e[~isv], bins=24, range=(0,1), color=COOL, alpha=0.8, label="consolidate steps")
    ax[1].hist(share_e[isv], bins=24, range=(0,1), color=ACCENT, alpha=0.8, label="venture steps")
    ax[1].set_xlabel("epistemic share of gain at this step")
    ax[1].set_ylabel("# steps")
    ax[1].set_title("(b)  bimodal: each phase is purely one face", fontsize=10.5)
    ax[1].legend(fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_decomposition.png", bbox_inches="tight"); plt.close(fig)
    print("  fig5_decomposition done")


# ====================================================================== fig6
def fig_scaling():
    deltas = np.array([0.005, 0.006, 0.007, 0.009, 0.011, 0.013, 0.016, 0.019])
    lams = np.array([0.985, 0.990, 0.993, 0.996, 0.998, 0.999])
    seeds = (1, 2, 3, 4)

    seeds6 = (1, 2, 3, 4, 5, 6)
    def measure(**kw):
        ps, cov = [], []
        for sd in seeds6:
            o = simulate(A, LUT, steps=3000, drive="gain", seed=sd, **kw)
            p, n = dg.cycle_period(o)
            if not np.isnan(p): ps.append(p)
            cov.append(len(set(o["cell"][200:].tolist())) / A.S)
        return (np.mean(ps) if ps else np.nan, np.std(ps) if ps else 0,
                np.mean(cov), np.std(cov))

    pd = dict(intake=0.04, lam=0.996, gamma=0.94)
    Pd = [measure(delta=d, **pd) for d in deltas]
    md = np.array([p[0] for p in Pd]); sd_ = np.array([p[1] for p in Pd])
    pl = dict(intake=0.04, delta=0.008, gamma=0.94)
    Pl = [measure(lam=l, **pl) for l in lams]
    clm = np.array([p[2] for p in Pl]); cls = np.array([p[3] for p in Pl])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax[0].errorbar(deltas, md, yerr=sd_, fmt="o", color=COOL, capsize=3, ms=6)
    dd = np.linspace(deltas.min(), deltas.max(), 100)
    k = np.median(md * deltas)
    ax[0].plot(dd, k/dd, color=ACCENT, lw=1.6, ls="--", label="$T\\propto1/\\delta$  ($T\\delta\\approx%.2f$)" % k)
    ax[0].set_xlabel("metabolic rate  $\\delta$"); ax[0].set_ylabel("cycle period  $T$ (steps)")
    ax[0].set_title("(a)  $\\delta$ sets the period (metabolic clock)", fontsize=10.5)
    ax[0].legend(fontsize=9, frameon=False)

    ax[1].errorbar(lams, clm, yerr=cls, fmt="s", color=GOLD, capsize=3, ms=6)
    ax[1].set_xlabel("model retention  $\\lambda$"); ax[1].set_ylabel("arena coverage")
    ax[1].set_ylim(0.3, 1.03)
    ax[1].annotate("local\nre-grooving", (lams[0], clm[0]), (lams[0]+0.001, clm[0]-0.18),
                   fontsize=8, color="#777", ha="left")
    ax[1].annotate("wide\nranging", (lams[-1], clm[-1]), (lams[-1]-0.004, clm[-1]-0.16),
                   fontsize=8, color="#777", ha="right")
    ax[1].set_title("(b)  $\\lambda$ sets exploration breadth", fontsize=10.5)
    fig.suptitle("Two physical timescales shape the rhythm — neither is an explore/exploit mixing weight",
                 fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_scaling.png", bbox_inches="tight"); plt.close(fig)
    print("  fig6_scaling done  (Td=%.3f)" % k)


# ====================================================================== fig7
def fig_goldilocks():
    deltas = np.linspace(0.003, 0.024, 14)
    seeds = (1, 2, 3)
    vent, surv, cov = [], [], []
    for d in deltas:
        vv, ss, cc = [], [], []
        for sd in seeds:
            o = simulate(A, LUT, steps=2500, drive="gain", seed=sd,
                         delta=d, intake=0.04, lam=0.996, gamma=0.94)
            vv.append(np.mean(o["phase"][200:] == "venture"))
            ss.append(o["b"][200:].min())
            cc.append(len(set(o["cell"][200:].tolist())) / A.S)
        vent.append(np.mean(vv)); surv.append(np.mean(ss)); cov.append(np.mean(cc))
    vent, surv, cov = map(np.array, (vent, surv, cov))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    safe = surv > 0.15
    dcrit = deltas[~safe].min() if (~safe).any() else deltas.max()
    ax.axvspan(deltas.min() - 0.001, dcrit, color="#eef7ee", lw=0)
    ax.text((deltas.min() + dcrit) / 2, 0.96, "robust oscillation\n(wide margin)",
            ha="center", fontsize=9, color="#2a6b2a")
    ax.axvspan(dcrit, deltas.max() + 0.001, color="#fdeeee", lw=0)
    ax.text((dcrit + deltas.max()) / 2, 0.965, "fragile: margin $\\to$ 0,\nintermittent starvation",
            ha="center", fontsize=9, color="#9b2226")
    ax.plot(deltas, vent, "o-", color=ACCENT, label="venture fraction")
    ax.plot(deltas, cov, "^-", color=COOL, label="arena coverage")
    ax.plot(deltas, surv, "s-", color=GOLD, label="min body integrity (survival margin)")
    ax.axhline(0.15, color=GOLD, ls=":", lw=1)
    ax.set_xlabel("metabolic rate  $\\delta$"); ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("A survival ceiling on metabolic rate: faster metabolism shrinks the body margin",
                 fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_goldilocks.png", bbox_inches="tight"); plt.close(fig)
    print("  fig7_goldilocks done")


# ====================================================================== fig8
def fig_sensor():
    Ef = empowerment_field(A, 1.0, n=3, sensor=False)
    Es = empowerment_field(A, 1.0, n=3, sensor=True)
    r = np.corrcoef(Ef, Es)[0, 1]
    luts = EmpowermentLUT(A, n=3, n_bins=21, sensor=True)
    out = simulate(A, luts, steps=1400, drive="gain", seed=3, **HEAD)
    t = np.arange(len(out["b"])); ph = out["phase"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), gridspec_kw={"width_ratios":[1,1,1.5]})
    im = ax[0].imshow(field_to_grid(A, Es), cmap=EMAP)
    rr, rc = A.resource_rc
    ax[0].scatter([rc], [rr], marker="*", s=160, c="white", edgecolor=INK, lw=1, zorder=5)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("(a)  sensor-state field $E_{\\mathrm{sensor}}(s)$", fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.035, pad=0.02)

    ax[1].scatter(Ef, Es, s=14, color=COOL, alpha=0.7)
    ax[1].set_xlabel("position empowerment (bits)"); ax[1].set_ylabel("sensor empowerment (bits)")
    ax[1].set_title("(b)  different observables ($r=%.2f$)" % r, fontsize=10)

    def shade(a):
        v=(ph=="venture").astype(int); d=np.diff(np.concatenate([[0],v,[0]]))
        for s0,s1 in zip(np.where(d==1)[0], np.where(d==-1)[0]):
            a.axvspan(s0,s1,color=ACCENT,alpha=0.09,lw=0)
    shade(ax[2])
    ax[2].plot(t, out["somatic"], color=COOL, lw=1.2, label="somatic")
    ax[2].plot(t, out["epistemic"], color=ACCENT, lw=1.2, label="epistemic")
    ax[2].set_xlim(300, 820); ax[2].set_xlabel("step"); ax[2].set_ylabel("gain (bits)")
    ax[2].set_title("(c)  the same alternation under sensor-state empowerment", fontsize=10)
    ax[2].legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Robustness: the rhythm does not depend on which empowerment the agent maximizes",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig8_sensor.png", bbox_inches="tight"); plt.close(fig)
    print("  fig8_sensor done (r=%.3f)" % r)


# ====================================================================== fig9
def fig_headtohead():
    betas = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95])
    seeds = (1, 2, 3, 4)
    bV, bB = [], []                                  # baseline venture frac, bmin
    for beta in betas:
        vv, bb = [], []
        for sd in seeds:
            o = simulate(A, LUT, steps=2500, drive="hybrid", beta=beta, seed=sd, **HEAD)
            vv.append(np.mean(o["phase"][200:] == "venture")); bb.append(o["b"][200:].min())
        bV.append(np.mean(vv)); bB.append(np.mean(bb))
    bV, bB = np.array(bV), np.array(bB)
    # ours
    ov, ob = [], []
    for sd in seeds:
        o = simulate(A, LUT, steps=2500, drive="gain", seed=sd, **HEAD)
        ov.append(np.mean(o["phase"][200:] == "venture")); ob.append(o["b"][200:].min())
    ourV, ourB = np.mean(ov), np.mean(ob)

    # co-firing contributions (alive steps only)
    hb = simulate(A, LUT, steps=2500, drive="hybrid", beta=0.25, seed=2, **HEAD)
    gn = simulate(A, LUT, steps=2500, drive="gain", seed=2, **HEAD)
    alive = hb["b"][200:] > 0.12
    hx, hy = hb["somatic"][200:][alive], hb["epistemic"][200:][alive]
    gx, gy = gn["somatic"][200:], gn["epistemic"][200:]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    # (a) co-firing plane (each drive normalized to its own max -> unit square)
    ax[0].scatter(hx/hx.max(), hy/hy.max(), s=8, color="#9aa0a6", alpha=.5, label="baseline (blend)")
    ax[0].scatter(gx/gx.max(), gy/gy.max(), s=8, color=ACCENT, alpha=.6, label="gain (alternation)")
    ax[0].set_xlabel("exploit / somatic contribution (norm.)")
    ax[0].set_ylabel("explore / epistemic contribution (norm.)")
    ax[0].set_title("(a)  blend vs alternate", fontsize=10.5)
    ax[0].legend(fontsize=8, frameon=False, loc="upper center")
    ax[0].text(0.62, 0.45, "baseline fills\nthe interior:\nboth at once", transform=ax[0].transAxes,
               fontsize=8.5, color="#6b6b76", ha="left")
    ax[0].text(0.02, 0.62, "gain hugs\nthe axes", transform=ax[0].transAxes,
               fontsize=8.5, color=ACCENT, ha="left")

    # (b) beta is the knob; tuning starves it
    axr = ax[1].twinx()
    ax[1].plot(betas, bV, "o-", color=ACCENT, label="venture fraction")
    axr.plot(betas, bB, "s--", color=GOLD, label="survival margin (min b)")
    axr.axhline(0.12, color=GOLD, ls=":", lw=1)
    ax[1].set_xlabel("baseline mixing weight  $\\beta$")
    ax[1].set_ylabel("venture fraction", color=ACCENT); ax[1].set_ylim(0, 1.05)
    axr.set_ylabel("survival margin (min $b$)", color=GOLD); axr.set_ylim(0, 1.05)
    ax[1].set_title("(b)  $\\beta$ trades exploration against survival", fontsize=10.5)
    ax[1].tick_params(axis="y", colors=ACCENT); axr.tick_params(axis="y", colors=GOLD)

    # (c) achievability frontier
    ax[2].plot(bB, bV, color="#bbb", lw=1, zorder=2)
    sc = ax[2].scatter(bB, bV, c=betas, cmap="cividis", s=65, zorder=4, edgecolor="w", lw=.6)
    ax[2].scatter([ourB], [ourV], marker="*", s=440, color=ACCENT, edgecolor=INK, lw=1.2, zorder=6)
    ax[2].annotate("our gain drive\n(no $\\beta$): balanced\nand alive", (ourB, ourV),
                   (ourB - 0.02, ourV - 0.30), fontsize=8.5, color=ACCENT, ha="center",
                   arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax[2].axvspan(-0.02, 0.12, color="#fdeeee", lw=0)
    ax[2].text(0.045, 0.62, "starves", rotation=90, color="#9b2226", fontsize=9, ha="center")
    ax[2].set_xlabel("survival margin (min $b$)"); ax[2].set_ylabel("venture fraction")
    ax[2].set_xlim(-0.02, 0.5); ax[2].set_ylim(-0.03, 1.08)
    ax[2].set_title("(c)  no $\\beta$ reaches balanced + alive", fontsize=10.5)
    cb = fig.colorbar(sc, ax=ax[2], fraction=0.045, pad=0.02); cb.set_label("$\\beta$", fontsize=9)
    fig.suptitle("Head-to-head with the two-term baseline $R=(1{-}\\beta)\\,E+\\beta\\,$novelty",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig9_baseline.png", bbox_inches="tight"); plt.close(fig)
    print("  fig9_baseline done (ours: vent=%.2f bmin=%.2f)" % (ourV, ourB))


# ====================================================================== fig10
def fig_learned():
    """A2: the epistemic face is a literal empowerment increment on a LEARNED
    model -- not a visit-count proxy. (a) empowerment-on-p_hat converges to the
    true field as coverage grows; (b) the realized dE-on-p_hat fires in venture
    phases; (c) the paper's proxy faithfully tracks that literal increment."""
    Etrue = empowerment_field(A, 1.0, n=3)

    def field(m):
        W = m.world_matrices()
        return empowerment_field(A, 1.0, n=3, T=body_gate(W, 1.0, A.g_b0, A.g_k), max_iter=60)

    # (a) estimator convergence under growing uniform coverage
    mC = LearnedDynamics(A, alpha=0.25, lam=1.0)
    obs, ccorr, cmean = [], [], []
    for k in range(0, 13):
        if k > 0:
            for s in range(A.S):
                for act in range(NA):
                    mC.update(s, act, A.cell_index[A.step_cell(A.cells[s], act)])
        E = field(mC)
        obs.append(k); ccorr.append(np.corrcoef(E, Etrue)[0, 1]); cmean.append(E.mean())

    # (b,c) realized dE along the working agent's run + proxy validation
    out = simulate(A, LUT, steps=2600, drive="gain", seed=3, **HEAD)
    cells = out["cell"].astype(int); acts = out["action"].astype(int); ph = out["phase"]
    mB = LearnedDynamics(A, alpha=0.25, lam=0.997)
    K = 16; ts, dE, vf, proxy = [], [], [], []
    prev = np.zeros(A.S); wp, wc = [], []; s_prev = RES
    for t in range(len(cells)):
        s_now = cells[t]
        mB.update(s_prev, acts[t], s_now); mB.decay()
        wp.append(mB.uncertainty()[s_now]); wc.append(s_now)
        if t % K == 0 and t > 0:
            EB = field(mB); vis = np.unique(wc)
            ts.append(t); dE.append(np.sum(np.maximum(0.0, EB[vis] - prev[vis])))
            vf.append(np.mean(ph[t-K:t] == "venture")); proxy.append(np.mean(wp))
            wp, wc = [], []; prev = EB
        s_prev = s_now
    ts = np.array(ts); dE = np.array(dE); vf = np.array(vf); proxy = np.array(proxy)
    isv = vf > 0.5
    r = np.corrcoef(proxy, dE)[0, 1]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    # (a)
    axr = ax[0].twinx()
    ax[0].plot(obs, ccorr, "o-", color=COOL, lw=2)
    axr.plot(obs, cmean, "s--", color=GOLD, lw=1.6)
    axr.axhline(Etrue.mean(), color=GOLD, ls=":", lw=1)
    axr.text(8.5, Etrue.mean()+0.06, "true mean", color=GOLD, fontsize=8)
    ax[0].set_xlabel("observations per state-action")
    ax[0].set_ylabel("corr($\\hat E_{p}$, $E_{\\mathrm{true}}$)", color=COOL); ax[0].set_ylim(-0.35, 1.03)
    axr.set_ylabel("mean $\\hat E_{p}$ (bits)", color=GOLD); axr.set_ylim(0, 4.2)
    ax[0].tick_params(axis="y", colors=COOL); axr.tick_params(axis="y", colors=GOLD)
    ax[0].set_title("(a)  empowerment-on-$\\hat p$ converges to truth", fontsize=10.5)

    # (b) realized dE split by phase
    groups = [dE[~isv], dE[isv]]
    cols = [COOL, ACCENT]
    rng_j = np.random.default_rng(0)
    for i, (g, col) in enumerate(zip(groups, cols)):
        x = i + 1 + (rng_j.random(len(g)) - 0.5) * 0.22
        ax[1].scatter(x, g, s=14, color=col, alpha=0.45, edgecolor="none", zorder=2)
        ax[1].plot([i+1-0.28, i+1+0.28], [g.mean(), g.mean()], color=col, lw=3, zorder=4)
    ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["consolidate", "venture"])
    ax[1].set_xlim(0.5, 2.5)
    ax[1].set_ylabel("realized $\\Delta\\hat E_{p}$ per window (bits)")
    ax[1].annotate("%.1f$\\times$ more\nempowerment learned" % (dE[isv].mean()/max(dE[~isv].mean(),1e-6)),
                   (2, dE[isv].mean()), (1.45, dE.max()*0.82), fontsize=9, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax[1].set_title("(b)  the epistemic face = literal $\\Delta\\hat E$, fired in venture", fontsize=10)

    # (c)
    ax[2].scatter(proxy, dE, s=22, color=COOL, alpha=0.55, edgecolor="none")
    m, c0 = np.polyfit(proxy, dE, 1)
    xs = np.linspace(proxy.min(), proxy.max(), 50)
    ax[2].plot(xs, m*xs + c0, color=ACCENT, lw=1.8, label="$r=%.2f$" % r)
    ax[2].set_xlabel("visit-count proxy (paper's signal)")
    ax[2].set_ylabel("realized $\\Delta\\hat E_{p}$ (bits)")
    ax[2].set_title("(c)  the proxy faithfully tracks $\\Delta\\hat E$", fontsize=10.5)
    ax[2].legend(fontsize=10, frameon=False, loc="upper left")
    fig.suptitle("The epistemic drive is empowerment gain on a learned model, "
                 "not a relabelled novelty bonus", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_learned.png", bbox_inches="tight"); plt.close(fig)
    print("  fig10_learned done (conv corr->%.2f, proxy r=%.2f, dE ratio %.1fx)"
          % (ccorr[-1], r, dE[isv].mean()/max(dE[~isv].mean(), 1e-6)))


# ====================================================================== fig11
def fig_viability():
    """A3: is the viability gate a smuggled survival objective? No. (a) The
    'obvious' alternative -- let viability emerge from forward rollouts in (s,b)
    -- fails: the body depletes over ~1/delta steps, beyond any tractable,
    near-undiscounted horizon, so rollouts over-explore and starve at every H
    while the O(1) gate never does. (b) The gate is the paper's serial bottleneck
    (Fig.1) applied over time: realizable empowerment is gated by the minimum
    body integrity along the trajectory (just before refuelling)."""
    invd = 1.0 / HEAD["delta"]

    def starve_vent(drive, **extra):
        sf, vf = [], []
        for sd in (1, 2, 3):
            o = simulate(A, LUT, steps=600, drive=drive, seed=sd, **HEAD, **extra)
            sf.append(np.mean(o["b"][150:] < 0.05)); vf.append(np.mean(o["phase"][150:] == "venture"))
        return np.mean(sf), np.mean(vf)

    Hs = [15, 60, 150]
    labels = ["gate\n($H{=}1$)"] + ["rollout\n$H{=}%d$" % H for H in Hs]
    sfs, vfs = [], []
    g_sf, g_vf = starve_vent("gain"); sfs.append(g_sf); vfs.append(g_vf)
    for H in Hs:
        sf, vf = starve_vent("gain_rollout", roll_H=H, roll_gamma=0.995)
        sfs.append(sf); vfs.append(vf)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    # (a) grouped bars: venture fraction (balance) + starvation (consequence)
    x = np.arange(len(labels)); w = 0.38
    cols = [COOL] + [ACCENT]*len(Hs)
    ax[0].bar(x - w/2, [100*v for v in vfs], w, color=cols, label="venture %")
    ax[0].bar(x + w/2, [100*s for s in sfs], w, color="#cfa600", label="starving %")
    ax[0].axhline(100*g_vf, color=COOL, ls=":", lw=1.2)
    ax[0].text(len(labels)-1, 100*g_vf+2, "gate's balance", ha="right", fontsize=8, color=COOL)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8.5)
    ax[0].set_ylabel("% of steps"); ax[0].set_ylim(0, 105)
    ax[0].set_title("(a)  rollout never recovers the balanced rhythm", fontsize=10.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    ax[0].text(0.5, -0.27, "even at $H{\\approx}1/\\delta$ and ~%d$\\times$ the per-step cost,"
               " the rollout over-explores ($\\sim$90%% venture) and starves" % (5*Hs[-1]),
               transform=ax[0].transAxes, ha="center", fontsize=8, color="#666")

    # (b) the gate = bottleneck over time
    far = A.cell_index[(4, 13)]                       # a tempting far basin cell
    tau = A.dist[RES, far]; dh = A.dist[far, RES]
    bs = np.linspace(0, 1, 60)
    arr = [LUT.E(far, np.clip(b - HEAD["delta"]*tau, 0, 1)) for b in bs]
    gate = [LUT.E(far, np.clip(b - HEAD["delta"]*(tau+dh), 0, 1)) for b in bs]
    ax[1].plot(bs, arr, color="#9aa0a6", lw=2, label="no metabolic cost\n(what the rollout policy uses)")
    ax[1].plot(bs, gate, color=COOL, lw=2.2, label="gate: body at trajectory min\n(after the round-trip cost)")
    ax[1].fill_between(bs, gate, arr, color=GOLD, alpha=0.18)
    ax[1].text(0.62, 0.6, "metabolic cost\nof the excursion", color=GOLD, fontsize=8.5,
               transform=ax[1].transAxes)
    ax[1].set_xlabel("current body integrity  $b$")
    ax[1].set_ylabel("realizable empowerment at a far cell (bits)")
    ax[1].set_title("(b)  the gate is the §2 bottleneck, applied over time", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("The viability gate is metabolic-cost accounting, not a smuggled survival reward",
                 fontsize=11, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig11_viability.png", bbox_inches="tight"); plt.close(fig)
    print("  fig11_viability done (gate vent %.0f%%, rollout vent %.0f-%.0f%%)"
          % (100*g_vf, 100*min(vfs[1:]), 100*max(vfs[1:])))


# ====================================================================== fig12
def fig_reduction():
    """B1: the dynamical reduction. The agent is a homeostat -- it regulates body
    integrity to a setpoint where the two faces balance (a), the body relaxes at
    two fast-switched rates (b), and because the setpoint geometry is
    delta-independent the period is the metabolic clock, T ~ 1/delta (c)."""
    bs = np.linspace(0, 1, 21)
    SOM = np.array([LUT.Emax[RES] - LUT.E(RES, bb) for bb in bs])
    far = A.cell_index[(4, 13)]
    FAR = np.array([LUT.E(far, bb) for bb in bs])

    def sigma(b): return np.interp(b, bs, SOM)
    def efar(b): return np.interp(b, bs, FAR)

    def reduced(delta, intake=0.04, a_learn=0.05, f=0.004, steps=6000):
        b, w = 0.9, 0.6; v = []
        for _ in range(steps):
            vent = w * efar(b) > sigma(b)
            w = w + f * (1 - w)
            if vent:
                b -= delta; w -= a_learn * w
            else:
                b = min(1.0, b + intake)
            b = min(1.0, max(0.0, b)); w = min(1.0, max(0.0, w))
            v.append(vent)
        v = np.array(v[1000:], int)
        on = np.where(np.diff(v) == 1)[0]
        return float(np.median(np.diff(on))) if len(on) > 2 else np.nan

    # sim body distribution + db/dt, clean seeds
    bvals, dbvals = [], []
    for sd in (1, 2, 4, 5):
        o = simulate(A, LUT, steps=4000, drive="gain", seed=sd, **HEAD)
        b = o["b"][300:]
        if b.mean() > 0.3:                       # drop the occasional starved seed
            bvals.append(b); d = np.diff(b); dbvals.append(d[np.abs(d) > 1e-6])
    bsim = np.concatenate(bvals); dbsim = np.concatenate(dbvals)
    bstar_w = 0.25

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
    # (a) setpoint -- sim body density behind, drive curves on top
    diff = SOM - bstar_w * FAR
    bstar = bs[np.where(np.diff(np.sign(diff)) != 0)[0][0]]
    axt = ax[0].twinx()
    axt.hist(bsim[bsim > 0.05], bins=36, range=(0.05, 1), color="#b9bdc4", alpha=0.45,
             density=True, zorder=0)
    axt.set_yticks([])
    ax[0].axvspan(0, bstar, color="#dff0ef", alpha=0.45, lw=0)
    ax[0].axvspan(bstar, 1, color="#fde3e7", alpha=0.45, lw=0)
    ax[0].set_zorder(axt.get_zorder() + 1); ax[0].patch.set_visible(False)
    ax[0].plot(bs, SOM, color=COOL, lw=2.4, label="somatic $\\sigma(b)$  (hunger)")
    ax[0].plot(bs, bstar_w * FAR, color=ACCENT, lw=2.4, label="epistemic $w\\,E_{far}(b)$")
    ax[0].axvline(bstar, color=INK, ls="--", lw=1)
    ax[0].annotate("setpoint $b^*$", (bstar, 2.6), (bstar - 0.30, 2.9), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color=INK))
    ax[0].text(bstar/2, 0.25, "consolidate", ha="center", fontsize=8.5, color=COOL)
    ax[0].text((bstar+1)/2, 0.25, "venture", ha="center", fontsize=8.5, color=ACCENT)
    ax[0].set_xlabel("body integrity $b$"); ax[0].set_ylabel("drive (bits)")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 3.9)
    ax[0].set_title("(a)  a homeostatic setpoint, not a weight", fontsize=10.5)
    ax[0].legend(fontsize=8, frameon=False, loc="upper right")

    # (b) two-rate relaxation
    ax[1].hist(dbsim, bins=60, color=GOLD, alpha=0.85)
    ax[1].axvline(-HEAD["delta"], color=COOL, lw=1.5, ls="--")
    ax[1].axvline(HEAD["intake"], color=ACCENT, lw=1.5, ls="--")
    ax[1].text(-HEAD["delta"], ax[1].get_ylim()[1]*0.7, " deplete\n $-\\delta$", fontsize=8.5, color=COOL)
    ax[1].text(HEAD["intake"], ax[1].get_ylim()[1]*0.7, "refuel\n$+\\iota$", fontsize=8.5,
               color=ACCENT, ha="right")
    ax[1].set_yscale("log"); ax[1].set_xlabel("$\\dot b$ per step"); ax[1].set_ylabel("# steps")
    ax[1].set_title("(b)  the body relaxes at two rates", fontsize=10.5)

    # (c) T ~ 1/delta  (agent over the fig-6 range; reduced model wider)
    dr = np.array([0.004, 0.006, 0.009, 0.013, 0.018, 0.024])
    Tr = np.array([reduced(d) for d in dr])
    ds = np.array([0.005, 0.007, 0.010, 0.014, 0.019])
    Ts = []
    for d in ds:
        ps = []
        for sd in (1, 2, 3, 4):
            o = simulate(A, LUT, steps=3500, drive="gain", seed=sd, delta=d,
                         intake=0.04, lam=0.996, gamma=0.94)
            p, _ = dg.cycle_period(o)
            if not np.isnan(p): ps.append(p)
        Ts.append(np.mean(ps) if ps else np.nan)
    Ts = np.array(Ts)
    ax[2].loglog(dr, Tr, "s-", color=GOLD, lw=1.8, ms=6, label="reduced model")
    ax[2].loglog(ds, Ts, "o", color=ACCENT, ms=8, label="full agent")
    xx = np.array([0.0035, 0.026])
    ax[2].loglog(xx, Tr[0]*dr[0]/xx, ":", color="#888", lw=1.2, label="slope $-1$  ($T\\propto1/\\delta$)")
    ax[2].set_xlabel("metabolic rate $\\delta$"); ax[2].set_ylabel("period $T$ (steps)")
    ax[2].set_title("(c)  $\\delta$ sets the clock, not the cycle", fontsize=10.5)
    ax[2].legend(fontsize=8, frameon=False)
    fig.suptitle("The explore/exploit cycle reduced: a homeostat at a $\\delta$-independent setpoint, "
                 "clocked at rate $\\delta$", fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig12_reduction.png", bbox_inches="tight"); plt.close(fig)
    print("  fig12_reduction done (b*=%.2f, sim b_mean=%.2f)" % (bstar, bsim.mean()))


# ====================================================================== demo
def export_demo():
    step = np.array([[A.cell_index[A.step_cell(A.cells[s], a)] for a in range(5)]
                     for s in range(A.S)])
    payload = dict(
        rows=A.rows, cols=A.cols, S=A.S,
        wall=A.wall.astype(int).tolist(),
        cells=[[int(r), int(c)] for r, c in A.cells],
        resource=int(RES), resource_rc=[int(A.resource_rc[0]), int(A.resource_rc[1])],
        dist=np.where(A.dist > 1e8, 999, A.dist).astype(int).tolist(),
        first_step=A.first_step.astype(int).tolist(),
        step=step.astype(int).tolist(),
        b_bins=[round(float(x), 4) for x in LUT.b_bins],
        Etable=np.round(LUT.table, 4).tolist(),
        Emax=np.round(LUT.Emax, 4).tolist(),
        g_b0=A.g_b0, g_k=A.g_k, head=HEAD,
    )
    with open("demo/data.json", "w") as f:
        json.dump(payload, f)
    with open("demo/data.js", "w") as f:
        f.write("window.EMP_DATA = ")
        json.dump(payload, f)
        f.write(";")
    print("  demo/data.json + data.js written (%d cells)" % A.S)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["field", "matrix", "headline", "phaseplane",
                            "decomposition", "scaling", "goldilocks", "sensor",
                            "headtohead", "learned", "viability", "reduction", "demo"]
    fns = dict(field=fig_field, matrix=fig_matrix, headline=fig_headline,
               phaseplane=fig_phaseplane, decomposition=fig_decomposition,
               scaling=fig_scaling, goldilocks=fig_goldilocks, sensor=fig_sensor,
               headtohead=fig_headtohead, learned=fig_learned,
               viability=fig_viability, reduction=fig_reduction, demo=export_demo)
    for t in todo:
        fns[t]()
    print("done")
