"""Grand-roulette MDP: the adversarial environment for MOP (maximum occupancy).

A hub connects a CONTROL room (actions lead to distinct states -> high empowerment,
zero outcome noise) and a ROULETTE room (many interchangeable 'bet' actions, each
giving a uniform-random outcome -> high action-state path entropy but ~zero control).
MOP gorges on the roulette (its reward is path entropy, and the bets multiply it);
an empowerment / control-occupancy agent ignores it (the bets control nothing).
Complements the noisy-TV result, which is the adversarial environment for info-gain.
"""

import numpy as np

from .empowerment import blahut_arimoto


def build(n_bets=8, nc=4, nr=4):
    S = 1 + nc + nr
    hub = 0
    control = list(range(1, nc + 1))
    roulette = list(range(nc + 1, nc + nr + 1))

    def d(pairs):
        v = np.zeros(S)
        for k, p in pairs.items():
            v[k] = p
        return v

    A = {hub: [d({control[0]: 1}), d({roulette[0]: 1}), d({hub: 1})]}
    for s in control:                                  # actions -> distinct control states + return
        A[s] = [d({control[j]: 1}) for j in range(nc)] + [d({hub: 1})]
    unif = d({r: 1.0 / nr for r in roulette})
    for s in roulette:                                 # n_bets identical random bets + return
        A[s] = [unif.copy() for _ in range(n_bets)] + [d({hub: 1})]
    return dict(S=S, hub=hub, control=control, roulette=roulette, A=A)


def empowerment(mdp):
    E = np.zeros(mdp["S"])
    for s in range(mdp["S"]):
        E[s] = blahut_arimoto(np.array(mdp["A"][s]))[0]   # capacity over this state's actions (bits)
    return E


def H_sa(mdp):                                          # next-state entropy per (s,a), bits
    out = {}
    for s in range(mdp["S"]):
        out[s] = []
        for p in mdp["A"][s]:
            q = np.clip(p, 1e-12, None)
            out[s].append(float(-(q * np.log2(q)).sum()))
    return out


def soft_vi(mdp, reward, alpha=0.4, gamma=0.9, iters=600):
    S = mdp["S"]; V = np.zeros(S)
    for _ in range(iters):
        Vn = np.zeros(S)
        for s in range(S):
            Q = np.array([reward[s][a] + gamma * (mdp["A"][s][a] @ V) for a in range(len(mdp["A"][s]))])
            m = Q.max(); Vn[s] = m + alpha * np.log(np.exp((Q - m) / alpha).sum())
        V = Vn
    pi = {}
    for s in range(S):
        Q = np.array([reward[s][a] + gamma * (mdp["A"][s][a] @ V) for a in range(len(mdp["A"][s]))])
        m = Q.max(); p = np.exp((Q - m) / alpha); pi[s] = p / p.sum()
    return pi


def dwell(mdp, pi, steps=30000, seed=0):
    rng = np.random.default_rng(seed); s = mdp["hub"]; occ = np.zeros(mdp["S"])
    for t in range(steps):
        a = rng.choice(len(pi[s]), p=pi[s]); s = rng.choice(mdp["S"], p=mdp["A"][s][a])
        if t > 2000:
            occ[s] += 1
    occ /= occ.sum()
    frac = lambda grp: float(occ[grp].sum())
    return dict(control=frac(mdp["control"]), roulette=frac(mdp["roulette"]), hub=float(occ[mdp["hub"]]))


def run(n_bets=8, alpha=0.4, gamma=0.9):
    mdp = build(n_bets=n_bets)
    E = empowerment(mdp); H = H_sa(mdp)
    rew_mop = {s: H[s] for s in range(mdp["S"])}                  # path entropy reward
    rew_ctrl = {s: [E[s]] * len(mdp["A"][s]) for s in range(mdp["S"])}   # empowerment reward
    return (E, dwell(mdp, soft_vi(mdp, rew_mop, alpha, gamma)),
            dwell(mdp, soft_vi(mdp, rew_ctrl, alpha, gamma)))


if __name__ == "__main__":
    E, dmop, dctrl = run(n_bets=8)
    print("empowerment: hub %.2f  control %.2f  roulette %.2f bits"
          % (E[0], E[1], E[5]))
    print("MOP            dwell:", {k: round(100*v) for k, v in dmop.items()})
    print("ctrl-occupancy dwell:", {k: round(100*v) for k, v in dctrl.items()})
    print("\nroulette dwell vs #bets:")
    for nb in (1, 2, 4, 8, 16, 32):
        _, dm, dc = run(n_bets=nb)
        print("  bets=%2d  MOP %3.0f%%   ctrl-occ %3.0f%%" % (nb, 100*dm["roulette"], 100*dc["roulette"]))
