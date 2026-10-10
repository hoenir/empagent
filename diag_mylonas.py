"""Diagnostics for interpreting the Mylonas replication.

1. Survival ceiling: a trivial forager that walks to the nearest source and
   stands on it.  Tells us whether leaving a source ever pays in this world.
2. Excursion by energy: mean distance from the nearest source, binned by E.
   The analogue of their Fig. 3a (MOP's regime switch near E = 15).
3. Death anatomy for the gated arms: what the agent was doing when it died.
4. Switching: after finding a source empty, does the agent try the other one?
"""

import json
import os

import numpy as np

from empower.mylonas import (DIST_FOOD, FOOD, THEIRS, MylonasAgent, MylonasLUT,
                             Parker, episode_metrics, simulate)


def runs(agent, gated, n=30, max_steps=3000, world=THEIRS):
    return [simulate(agent, gated=gated, world=world, seed=k, max_steps=max_steps)
            for k in range(n)]


def summary(recs):
    m = [episode_metrics(r) for r in recs]
    life = np.array([x["steps"] for x in m], float)
    return dict(life_mean=float(life.mean()),
                life_sem=float(life.std(ddof=1) / np.sqrt(len(life))),
                died=float(np.mean([x["died"] for x in m])),
                on_food=float(np.mean([x["frac_on_food"] for x in m])),
                trips=float(np.mean([x["trips_per_1k"] for x in m])))


def excursion_by_energy(recs, bins=(0, 6, 12, 18, 24, 31)):
    E = np.concatenate([r["E"] for r in recs])
    d = np.concatenate([DIST_FOOD[r["x"]] for r in recs])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        out.append(dict(E=f"{lo}-{hi - 1}", n=int(m.sum()),
                        mean_dist=float(d[m].mean()) if m.any() else None,
                        frac_off_food=float((d[m] > 0).mean()) if m.any() else None))
    return out


def death_anatomy(recs):
    """For episodes that ended in death: state at the last visit to food, and
    how the final approach went."""
    rows = []
    for r in recs:
        if not r["died"]:
            continue
        x, E = r["x"], r["E"]
        on = np.flatnonzero(np.isin(x, FOOD))
        if len(on) == 0:
            rows.append(dict(last_food_step=None))
            continue
        k = on[-1]
        src = 0 if x[k] == FOOD[0] else 1
        b_src_after = (r["b1"] if src == 0 else r["b2"])[k]   # observation made at k
        rows.append(dict(
            steps_from_last_food_to_death=int(len(x) - k),
            E_at_last_food=int(E[k]),
            source_seen_empty_at_last_visit=bool(b_src_after < 0.5),
            dist_at_death=int(DIST_FOOD[x[-1]]),
            max_dist_after_last_food=int(DIST_FOOD[x[k:]].max()),
        ))
    return rows


def switching(recs):
    """What the agent does each time it sees a source empty.

    The belief recorded at step t already includes the observation made on
    entering that cell, so "standing on source i at t with b_i[t] == 0" is
    exactly "has just seen source i empty".  For every such event:

      wait   the next step stays on the same source
      leave  it moves off; then 'other' if the next source it stands on is
             the other one, 'back' if it returns to the same one

    frac_other counts the next source observation: other / (all events), with
    waiting counted as staying with the same source.
    """
    wait = back = other = 0
    for r in recs:
        x = r["x"]
        bs = (r["b1"], r["b2"])
        T = len(x)
        for t in range(T - 1):
            if x[t] not in FOOD:
                continue
            i = 0 if x[t] == FOOD[0] else 1
            if bs[i][t] > 0.0:
                continue                                  # not an empty observation
            if x[t + 1] == x[t]:
                wait += 1
                continue
            u = t + 1
            while u < T and x[u] not in FOOD:
                u += 1
            if u < T:
                if x[u] == x[t]:
                    back += 1
                else:
                    other += 1
    tot = wait + back + other
    leaves = back + other
    return dict(empty_events=tot, wait=wait, back=back, other=other,
                frac_wait=(wait / tot) if tot else None,
                frac_other=(other / tot) if tot else None,
                frac_other_given_leave=(other / leaves) if leaves else None)


def meal_stats(recs):
    """Where and when the agent eats.

    alternation  share of consecutive meals taken at the OTHER source
    E_pct        energy just before a meal: 10th / 50th / 90th percentile
    """
    alt = same = 0
    E_meal = []
    for r in recs:
        x, E = r["x"], r["E"]
        srcs = []
        for t in range(1, len(x)):
            if E[t] > E[t - 1]:
                E_meal.append(int(E[t - 1]))
                srcs.append(int(x[t]))
        for a, b in zip(srcs, srcs[1:]):
            alt += a != b
            same += a == b
    return dict(meals=len(E_meal),
                alternation=(alt / (alt + same)) if (alt + same) else None,
                E_pct=[float(v) for v in np.percentile(E_meal, [10, 50, 90])]
                if E_meal else None)


if __name__ == "__main__":
    luts = {g: MylonasLUT.load("data/mylonas_C_n5_%s.npz" % ("gated" if g else "plain"))
            for g in (False, True)}
    report = {}

    print("== survival ceiling: parker (walk to nearest source, stand on it) ==")
    for g in (False, True):
        s = summary(runs(Parker(), g))
        report["parker|%d" % g] = s
        print("  %-7s lifespan %.0f +- %.0f  died %.0f%%  on food %.0f%%"
              % ("gated" if g else "plain", s["life_mean"], s["life_sem"],
                 100 * s["died"], 100 * s["on_food"]))

    print("\n== gated arms: excursion by energy, death anatomy, switching ==")
    for drive in ("level", "gain"):
        recs = runs(MylonasAgent(luts[True], drive=drive, gamma=0.9), True)
        key = drive + "|1"
        ex = excursion_by_energy(recs)
        da = death_anatomy(recs)
        sw = switching(recs)
        report[key] = dict(summary=summary(recs), excursion=ex, deaths=da, switching=sw)
        print("\n  [%s + gate]" % drive)
        for row in ex:
            if row["n"]:
                print("    E %-6s n=%6d  mean dist %.2f  off food %.0f%%"
                      % (row["E"], row["n"], row["mean_dist"], 100 * row["frac_off_food"]))
        d = [r for r in da if r.get("E_at_last_food") is not None]
        if d:
            print("    deaths: %d   E at last food visit: median %.0f   "
                  "steps from last food to death: median %.0f"
                  % (len(d), np.median([r["E_at_last_food"] for r in d]),
                     np.median([r["steps_from_last_food_to_death"] for r in d])))
            print("            source seen empty at last visit: %.0f%%   "
                  "max distance reached after last visit: median %.0f   "
                  "dist at death: median %.0f"
                  % (100 * np.mean([r["source_seen_empty_at_last_visit"] for r in d]),
                     np.median([r["max_dist_after_last_food"] for r in d]),
                     np.median([r["dist_at_death"] for r in d])))
        print("    after seeing a source empty (%d events): wait %d, leave+return %d, "
              "leave+other %d" % (sw["empty_events"], sw["wait"], sw["back"], sw["other"]))

    with open("data/mylonas_diag.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote data/mylonas_diag.json")
