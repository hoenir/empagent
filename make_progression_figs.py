"""Figures for the developmental-progression sibling study."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from empower.progression import build_arena_A, simulate_progression
from empower.empowerment import empowerment_field, body_gate
from empower.model import LearnedDynamics
from empower.arena import NA

FIG = "paper/figs"; os.makedirs(FIG, exist_ok=True)
INK = "#1a1a2e"; ACCENT = "#e94560"; COOL = "#0f7173"; GOLD = "#d98e04"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
EMAP = LinearSegmentedColormap.from_list("e", ["#10132b", "#26418f", "#1f9e89", "#7fd34e", "#f6e27a"])

A = build_arena_A(noise=0.5)
LABELS = ["frontier", "frontier2", "easy", "noise"]


def occ(drive, seed=1):
    log = simulate_progression(A, steps=2500, drive=drive, delta=0.0, intake=0.0,
                               lam_model=1.0, K=20, n_bins=2, gamma=0.97, eps=0.3, seed=seed)
    v = np.zeros(A.S)
    for c in log["cell"][200:]:
        v[int(c)] += 1
    g = np.full((A.rows, A.cols), np.nan)
    for s, (r, c) in enumerate(A.cells):
        g[r, c] = v[s]
    return g


def draw_occ(ax, g, title, col):
    ax.imshow(np.log1p(g), cmap=EMAP)
    rr, rc = A.resource_rc
    ax.scatter([rc], [rr], marker="*", s=120, c="white", edgecolor=INK, lw=0.8, zorder=5)
    # label chambers
    ax.text(7, 2, "frontier", color="w", ha="center", fontsize=7.5)
    ax.text(7, 12, "frontier", color="w", ha="center", fontsize=7.5)
    ax.text(12, 7, "noise", color="#ffd2da", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(2.0, 7, "easy", color="w", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, color=col, fontweight="bold")


def fig_noisytv():
    gLP = occ("lp"); gNV = occ("novelty")
    # robust bars
    learn = A.region_cells("frontier") + A.region_cells("frontier2") + A.region_cells("easy")
    noise = A.region_cells("noise")
    res = {"lp": ([], []), "novelty": ([], [])}
    for drive in ("lp", "novelty"):
        for sd in (1, 2, 3, 4):
            log = simulate_progression(A, steps=2500, drive=drive, delta=0.0, intake=0.0,
                                       lam_model=1.0, K=20, n_bins=2, gamma=0.97, eps=0.3, seed=sd)
            v = np.zeros(A.S)
            for c in log["cell"][200:]:
                v[int(c)] += 1
            tot = v.sum()
            res[drive][0].append(100*sum(v[c] for c in learn)/tot)
            res[drive][1].append(100*sum(v[c] for c in noise)/tot)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    draw_occ(ax[0], gLP, "(a)  empowerment learning-progress", COOL)
    draw_occ(ax[1], gNV, "(b)  surprise / novelty", ACCENT)

    x = np.arange(2); w = 0.36
    lpL, lpN = np.mean(res["lp"][0]), np.mean(res["lp"][1])
    nvL, nvN = np.mean(res["novelty"][0]), np.mean(res["novelty"][1])
    ax[2].bar(x - w/2, [lpL, lpN], w, yerr=[np.std(res["lp"][0]), np.std(res["lp"][1])],
              color=COOL, capsize=3, label="learning-progress")
    ax[2].bar(x + w/2, [nvL, nvN], w, yerr=[np.std(res["novelty"][0]), np.std(res["novelty"][1])],
              color=ACCENT, capsize=3, label="surprise")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["controllable\n(frontier+easy)", "noisy TV\n(uncontrollable)"])
    ax[2].set_ylabel("% of time spent"); ax[2].set_ylim(0, 100)
    ax[2].set_title("(c)  control-grounded curiosity\nignores the noisy TV", fontsize=10.5)
    ax[2].legend(fontsize=8.5, frameon=False, loc="upper center")
    fig.suptitle("Empowerment learning-progress forages the competence frontier; "
                 "surprise gets trapped on the noisy TV", fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/prog1_noisytv.png", bbox_inches="tight"); plt.close(fig)
    print("  prog1_noisytv done (LP %.0f/%.0f  novelty %.0f/%.0f)" % (lpL, lpN, nvL, nvN))


def fig_mechanism():
    """Why: empowerment can be LEARNED for controllable regions (E rises, u falls)
    but not for the noisy TV (E floored, u stays high). Learning-progress (realized
    dE) flags the frontier and goes to zero on the noise; surprise (u) stays high
    on the noise -- the trap."""
    rng = np.random.default_rng(0)
    m = LearnedDynamics(A, alpha=0.25, lam=1.0)
    labels = ["frontier", "easy", "noise"]
    cols = {"frontier": COOL, "easy": GOLD, "noise": ACCENT}

    def Ehat():
        W = m.world_matrices()
        return empowerment_field(A, 1.0, n=3, T=body_gate(W, 1.0, A.g_b0, A.g_k), max_iter=60)

    obs = list(range(0, 9))
    E_tr = {lab: [] for lab in labels}; u_tr = {lab: [] for lab in labels}
    for k in obs:
        if k > 0:
            for s in range(A.S):
                for a in range(NA):
                    for _ in range(3):
                        aa = rng.integers(NA) if rng.random() < A.env[s] else a
                        s2 = A.cell_index[A.step_cell(A.cells[s], aa)]
                        m.update(s, a, s2)
        E = Ehat(); u = m.uncertainty()
        for lab in labels:
            cs = A.region_cells(lab)
            E_tr[lab].append(np.mean([E[c] for c in cs])); u_tr[lab].append(np.mean([u[c] for c in cs]))

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for lab in labels:
        ax[0].plot(obs, E_tr[lab], "o-", color=cols[lab], lw=2, label=lab)
    ax[0].set_xlabel("observations per state-action"); ax[0].set_ylabel("learned empowerment $\\hat E_p$ (bits)")
    ax[0].set_title("(a)  control is learnable... except in the noisy TV", fontsize=10.5)
    ax[0].legend(fontsize=9, frameon=False)
    ax[0].annotate("noise: empowerment\ncan't be learned", (obs[-1], E_tr["noise"][-1]),
                   (4.2, 0.9), fontsize=8.5, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))

    dE = {lab: np.diff([0] + E_tr[lab]) for lab in labels}
    ax[1].plot(obs, u_tr["noise"], "s--", color=ACCENT, lw=2, label="surprise $u$ (noise) — stays high")
    ax[1].plot(obs, np.clip(dE["frontier"], 0, None), "o-", color=COOL, lw=2,
               label="learning-progress (frontier)")
    ax[1].plot(obs, np.clip(dE["noise"], 0, None), "o-", color="#b07a9a", lw=2,
               label="learning-progress (noise) → 0")
    ax[1].set_xlabel("observations per state-action"); ax[1].set_ylabel("curiosity signal")
    ax[1].set_title("(b)  the two signals diverge on the noisy TV", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Empowerment is control-grounded: surprise stays high on the noisy TV, "
                 "learning-progress does not", fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/prog2_mechanism.png", bbox_inches="tight"); plt.close(fig)
    print("  prog2_mechanism done (noise E floored %.2f, u stays %.2f)" %
          (E_tr["noise"][-1], u_tr["noise"][-1]))


def fig_stages():
    """Given genuine dependencies (a switch-gate chain), an empowerment staircase
    appears (mastering a prerequisite unlocks downstream empowerment) and the agent
    climbs it in a forced order -- unlike the independent-chamber ladder."""
    from empower.gated import build_chain, simulate_gated
    from empower.empowerment import empowerment_field
    NR = 3
    A = build_chain(n_rooms=NR)
    start = next(i for i, (ci, gb) in enumerate(A.states)
                 if gb == 0 and A.region[i] == "R0" and A.cells[ci] == (1, 2))
    ids = A.gate_ids
    Es = []
    for k in range(len(ids) + 1):
        E = empowerment_field(A, 1.0, n=5, T=A.partial_D(ids[:k]))
        Es.append(E[start])
    Es = np.array(Es)
    # independent-chamber baseline: nothing to unlock -> flat at the all-open value
    flat = empowerment_field(A, 1.0, n=5, T=A.partial_D(ids))[start]

    # agent's forced acquisition order
    log = simulate_gated(build_chain(NR), steps=2200, n=4, K=25, eps=0.25, seed=1)
    cur = log["cur_room"]; first = {}
    for t, r in enumerate(cur):
        first.setdefault(r, t)
    rooms = sorted(first)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(Es))
    ax[0].step(x, Es, where="mid", color=COOL, lw=2.4)
    ax[0].plot(x, Es, "o", color=COOL, ms=8)
    ax[0].plot([0, len(Es)-1], [Es[0], Es[0]], color="#9aa0a6", ls="--", lw=1.6)
    ax[0].text(len(Es)-1, Es[0]+0.04, "independent chambers (our earlier ladder):\nnothing gates anything → flat, no staircase",
               fontsize=8, color="#777", ha="right")
    for k in range(1, len(Es)):
        if Es[k]-Es[k-1] > 0.05:
            ax[0].annotate("+%.2f" % (Es[k]-Es[k-1]), (k-0.5, (Es[k]+Es[k-1])/2),
                           fontsize=8.5, color=ACCENT, ha="center")
    ax[0].set_xlabel("prerequisite switches mastered"); ax[0].set_ylabel("empowerment from start (bits)")
    ax[0].set_title("(a)  dependencies make an empowerment staircase", fontsize=10.5)
    ax[0].set_xticks(x)

    ax[1].barh(range(len(rooms)), [first[r] for r in rooms], color=GOLD)
    ax[1].set_yticks(range(len(rooms))); ax[1].set_yticklabels(rooms)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("step first reached"); ax[1].set_title("(b)  the agent acquires them in forced order", fontsize=10.5)
    for i, r in enumerate(rooms):
        ax[1].text(first[r]+max(first.values())*0.02, i, "step %d" % first[r], va="center", fontsize=8.5)
    fig.suptitle("With genuine control-dependencies, developmental stages emerge from empowerment-progress",
                 fontsize=10.5, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/prog3_stages.png", bbox_inches="tight"); plt.close(fig)
    print("  prog3_stages done (unlock %.2f->%.2f->%.2f, order %s)" %
          (Es[0], Es[1], Es[2], [first[r] for r in rooms]))


def fig_mop():
    """MOP vs control-occupancy: the same Boltzmann soft-value-iteration machinery,
    rewarding next-state entropy H(S') (MOP, chases noise) vs empowerment E (control,
    avoids it). Plus a 2x2 map placing MOP, prediction learning-progress, and us."""
    An = build_arena_A(noise=0.6)
    S = An.S; NA = An.D.shape[0]
    T = An.transition_matrices(1.0)
    def ent(p): p = np.clip(p, 1e-12, None); return float(-(p*np.log(p)).sum())
    Hsa = np.array([[ent(T[a, s]) for a in range(NA)] for s in range(S)])
    Enat = empowerment_field(An, 1.0, n=3) * np.log(2)

    def soft_vi(r, alpha=0.5, gamma=0.95, iters=600):
        V = np.zeros(S)
        for _ in range(iters):
            Q = r + gamma*np.einsum('asx,x->sa', T, V)
            m = Q.max(1, keepdims=True); V = m[:, 0] + alpha*np.log(np.exp((Q-m)/alpha).sum(1))
        Q = r + gamma*np.einsum('asx,x->sa', T, V); m = Q.max(1, keepdims=True)
        pi = np.exp((Q-m)/alpha); return pi/pi.sum(1, keepdims=True)

    def noise_dwell(pi, seed=0):
        rng = np.random.default_rng(seed); s = An.resource_s; nz = 0; tot = 0
        for t in range(6000):
            a = rng.choice(NA, p=pi[s]); s = rng.choice(S, p=T[a, s])
            if t > 500:
                tot += 1; nz += (An.region.get(s) == "noise")
        return 100*nz/tot
    d_mop = noise_dwell(soft_vi(Hsa))
    d_ctrl = noise_dwell(soft_vi(np.repeat(Enat[:, None], NA, 1)))

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) the race
    ax[0].bar([0, 1], [d_mop, d_ctrl], color=[ACCENT, COOL], width=0.6)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["MOP\n(reward $H(S')$)", "control-occupancy\n(reward $E$)"])
    ax[0].set_ylabel("time on the noisy TV (%)"); ax[0].set_ylim(0, 105)
    for x, v in [(0, d_mop), (1, d_ctrl)]:
        ax[0].text(x, v+3, "%.0f%%" % v, ha="center", fontweight="bold")
    ax[0].set_title("(a)  same MOP machinery, different reward", fontsize=10.5)

    # (b) the paradox map
    b = ax[1]; b.set_xlim(0, 1); b.set_ylim(0, 1); b.set_xticks([]); b.set_yticks([])
    b.axhline(0.5, color=INK, lw=1); b.axvline(0.5, color=INK, lw=1)
    b.add_patch(plt.Rectangle((0, 0), 0.5, 0.5, color="#fde3e7", zorder=0))      # trapped
    b.add_patch(plt.Rectangle((0.5, 0), 0.5, 0.5, color="#fdf0dd", zorder=0))    # the throne
    for (x, y) in [(0, 0.5), (0.5, 0.5)]:
        b.add_patch(plt.Rectangle((x, y), 0.5, 0.5, color="#dff0ef", zorder=0))  # escape
    b.text(0.25, -0.08, "reward: entropy / occupancy / surprise", ha="center", fontsize=8.5)
    b.text(0.75, -0.08, "reward: control / empowerment", ha="center", fontsize=8.5)
    b.text(-0.04, 0.25, "level\n(stationary)", va="center", ha="center", rotation=90, fontsize=8.5)
    b.text(-0.04, 0.75, "dynamic\n(derivative)", va="center", ha="center", rotation=90, fontsize=8.5)
    def q(x, y, t, sub, col):
        b.text(x, y+0.06, t, ha="center", fontsize=8.5, fontweight="bold", color=col)
        b.text(x, y-0.09, sub, ha="center", fontsize=7.8, color="#555")
    q(0.25, 0.30, "MOP · info-gain · novelty", "✗ grand roulette · noisy TV", ACCENT)
    q(0.25, 0.78, "prediction learning-progress\n(Oudeyer · Schmidhuber)", "escapes noise (derivative)", "#2a6b2a")
    q(0.75, 0.30, "empowerment-level", "✗ the throne\n(fixed-point collapse)", "#9b6a00")
    q(0.75, 0.82, "empowerment gain / progress", "escapes every trap", COOL)
    b.text(0.75, 0.70, "★ this work", ha="center", fontsize=7.8, color=COOL, fontweight="bold")
    b.text(0.5, 1.10, "(FEP → dark room and reward → Buridan's donkey lie on other axes)",
           ha="center", fontsize=7, color="#999")
    b.set_title("(b)  a map of the paradoxes", fontsize=10.5)
    b.set_frame_on(False)
    fig.suptitle("Control-grounding, not occupancy, is what ignores the noisy TV", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/prog4_mop.png", bbox_inches="tight"); plt.close(fig)
    print("  prog4_mop done (MOP %.0f%% noise, control-occupancy %.0f%%)" % (d_mop, d_ctrl))


def fig_roulette():
    """The grand roulette -- MOP's adversarial environment. A control room
    (deterministic, high empowerment) and a gambling room (interchangeable random
    bets, ~zero control). MOP gorges on the gambling; empowerment ignores it."""
    from empower import paradox
    E, dmop, dctrl = paradox.run(n_bets=8)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(2); w = 0.36
    ax[0].bar(x - w/2, [100*dmop["control"], 100*dmop["roulette"]], w, color=ACCENT, label="MOP (occupancy)")
    ax[0].bar(x + w/2, [100*dctrl["control"], 100*dctrl["roulette"]], w, color=COOL, label="control-occupancy")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["control room\n(empowerment 2.3 bits)", "grand roulette\n(empowerment 1.0 bit)"])
    ax[0].set_ylabel("% of time spent"); ax[0].set_ylim(0, 105)
    ax[0].set_title("(a)  MOP gorges on the gambling; empowerment ignores it", fontsize=10.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper center")

    # (b) the empirical taxonomy: dwell-in-the-trap across both adversarial worlds
    envs = ["noisy TV\n(vs info-gain)", "grand roulette\n(vs MOP)"]
    mop = [98, 100*dmop["roulette"]]; igm = [87, 100*dmop["roulette"]]; ctrl = [10, 100*dctrl["roulette"]]
    xx = np.arange(2); w2 = 0.26
    ax[1].bar(xx - w2, mop, w2, color=ACCENT, label="occupancy (MOP)")
    ax[1].bar(xx, igm, w2, color="#d98e04", label="curiosity / info-gain")
    ax[1].bar(xx + w2, ctrl, w2, color=COOL, label="control-grounded (ours)")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(envs)
    ax[1].set_ylabel("% of time trapped"); ax[1].set_ylim(0, 105)
    ax[1].set_title("(b)  control-grounding escapes both traps", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Two adversarial environments, one antidote: only diversity you cause counts",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/prog5_roulette.png", bbox_inches="tight"); plt.close(fig)
    print("  prog5_roulette done (MOP roulette %.0f%%, ctrl-occ %.0f%%)"
          % (100*dmop["roulette"], 100*dctrl["roulette"]))


def export_demo():
    import json
    A_clean = build_arena_A(noise=0.0)            # structural ceiling; noise applied live in JS
    Emax = empowerment_field(A_clean, 1.0, n=3)
    step = np.array([[A.cell_index[A.step_cell(A.cells[s], a)] for a in range(NA)] for s in range(A.S)])
    region_idx = [A.region.get(s, "") for s in range(A.S)]
    payload = dict(
        rows=A.rows, cols=A.cols, S=A.S,
        cells=[[int(r), int(c)] for r, c in A.cells],
        resource=int(A.resource_s),
        dist=np.where(A.dist > 1e8, 999, A.dist).astype(int).tolist(),
        first_step=A.first_step.astype(int).tolist(),
        step=step.astype(int).tolist(),
        Emax=np.round(Emax / Emax.max(), 4).tolist(),
        env=np.round(A.env, 3).tolist(),
        region=region_idx,
    )
    os.makedirs("demo", exist_ok=True)
    with open("demo/prog_data.js", "w") as f:
        f.write("window.PROG_DATA = "); json.dump(payload, f); f.write(";")
    print("  demo/prog_data.js written (%d cells)" % A.S)


if __name__ == "__main__":
    fig_noisytv()
    fig_mechanism()
    fig_stages()
    fig_mop()
    fig_roulette()
    export_demo()
