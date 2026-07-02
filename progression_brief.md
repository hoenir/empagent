# Brief: developmental progression from empowerment-progress alone

A sibling study to the explore/exploit paper. Separate arena, separate write-up
(`progression.html`), separate demo. Shares the `empower/` library.

## 0. The contribution we're testing (north star)

Curiosity that maximizes *prediction-error reduction* (Oudeyer–Kaplan learning
progress; Schmidhuber compression progress) famously needs the learning-progress
*derivative* to avoid two traps: the **mastered** region (nothing left to learn)
and the **noisy-TV** region (unlearnable stochasticity that keeps prediction error
high forever). The claim here is that **empowerment-progress** — the expected
increment in *realizable control* from learning — gets the same Goldilocks /
zone-of-proximal-development behaviour, but from a single control-grounded
quantity, and with one extra property prediction-based curiosity lacks for free:

1. it **ignores the noisy TV** structurally — an uncontrollable region has ~zero
   empowerment, so learning it yields ~zero empowerment-progress (not a tuned
   penalty, a consequence of grounding progress in control rather than prediction);
2. it **progresses** — because empowerment is computed on the *learned* model
   (§12 of the main paper), a region whose control routes through an unmastered
   region reads as low-gain *now*, so the agent does prerequisites first, and
   mastering them raises the downstream region's gain. The competence frontier
   marches, and the **order is set by the world's control-dependency structure,
   not a designed schedule.**

Falsifiable hook: the curriculum order tracks *control complexity*, not physical
distance. That is the thing the design must be built to prove (see §4).

## 1. The drive: expected empowerment learning-progress

Replace the level proxy `u(c)·E(c)` (a stock — "how much is left") with a rate —
the expected increment in realizable empowerment from mastering region `c`:

    LP(c) = Σ_s max(0,  Ê(s, p̂ | c reliable) − Ê(s, p̂))

i.e. recompute the empowerment field on the learned model with `c`'s transitions
collapsed to a low-entropy (mastered) estimate, and sum the realizable
empowerment it unlocks — **locally at `c` and downstream** at every region whose
n-step control routes through `c`. This is the curriculum driver: the downstream
term is large only when `c` is a live prerequisite.

- **Mastered `c`** (already low-entropy): field barely changes → LP ≈ 0. (too easy)
- **Noisy/uncontrollable `c`** (high intrinsic action-noise → low `E_max`):
  collapsing uncertainty unlocks little control → LP ≈ 0. (noisy TV)
- **Frontier `c`** (uncertain, controllable, prerequisites met): LP large.

Tractability: LP needs one field recompute per candidate. Restrict to a frontier
set (cells with non-trivial `u`), recompute on the §12 cadence, cache. Somatic /
viability / body system is unchanged — the curriculum runs in the well-fed
("venture") budget and pauses to refuel, exactly the "as long as body isn't
depleted" caveat.

Approximation honesty: "collapse to mastered" assumes the visit *will* teach the
true structure (true in a near-deterministic world). State this; it is the
expected-information-gain estimate the main brief flagged as the escalation path.

## 2. Environments (built to defeat the distance↔difficulty confound)

**Arena A — equidistant typed chambers (defeats the confound for the *preference*).**
A central hub holds the resource. Three (or more) chambers hang off the hub by
**equal-length** corridors, so distance and viability cost are identical:
- `EASY`: small/closed → low `E_max`, learnable in a few visits, then mastered.
- `FRONTIER`: open → high `E_max`, learnable, sustained gain until mastered.
- `NOISE`: high intrinsic action-noise (region-tied, like the body gate but
  environmental) → low `E_max` even when "learned" → the noisy TV.
Because all chambers are equidistant, any preference is about *difficulty/control*,
not distance.

**Arena B — the competence ladder (shows the *marching frontier*).**
A chain `R1→R2→R3(→R4)` where each `Rk`'s n-step control routes through `Rk−1`,
so `Ê(Rk, p̂)` is gated until `Rk−1` is mastered. This yields the staircase. Note
honestly: a spatial chain entangles depth with distance (R3 is farthest). Arena B
shows the *sequencing*; Arena A is what proves the sequencing is complexity-driven
(equidistant) rather than distance-driven. The two together make the case.

**(Stretch) Arena C — horizon as the developmental axis.** Regions that "light up"
only at long control horizons (n=4–5) vs short (n=2), placed independent of
distance. The agent realizes short-horizon control first and longer-horizon
control as the model sharpens — "motor babbling → coordinated skills." Requires
multi-horizon empowerment on the learned model. Only if A/B land.

## 3. Experiments and predictions

| test | prediction | what it shows |
|---|---|---|
| Arena A dwell | FRONTIER ≫ EASY(after onset) ≫ NOISE | Goldilocks preference, equidistant |
| **noisy-TV control** | LP-agent ignores NOISE; a novelty/prediction-error agent is drawn to it | control-grounded ≠ prediction-grounded |
| mastery dynamics | EASY gain spikes then →0; FRONTIER sustained then →0 | abandons the mastered pole |
| Arena B staircase | per-region LP peaks hand off R1→R2→R3; occupancy marches | progression / curriculum |
| **confound defeat** | mastery *order* uncorrelated with distance-from-hub (Arena A); driven by dependency (Arena B) | it's complexity, not distance |
| inverted-U | LP vs difficulty (control-dependency depth) peaks at intermediate | the ZPD signature |

Baselines to run alongside the LP-agent: (i) **novelty/count** drive `u(c)` — gets
trapped by the noisy TV; (ii) **level** drive `u·E_max` (the main paper's proxy) —
ignores noise but does *not* progress (no prerequisite gating in the increment).
The LP-agent should do both: ignore noise *and* progress.

## 4. The decisive confound-defeat test

Place a **hard region near the hub and an easy region far** (Arena A is symmetric;
add an asymmetric variant). If the agent masters near→far it's distance; if it
masters easy/frontier→hard regardless of placement, it's complexity. Report the
correlation between mastery order and (a) distance, (b) control-dependency depth.
The claim earns its keep only if mastery order ⟂ distance and ∝ depth.

## 5. Figures (for `progression.html`)

1. **Goldilocks / noisy-TV**: Arena A dwell maps for LP vs novelty agent — LP
   concentrates on FRONTIER, novelty smears into NOISE. The headline contrast.
2. **The ZPD inverted-U**: LP (and integrated dwell) vs difficulty axis, peak at
   intermediate; annotate the two poles (mastered, noisy TV).
3. **The developmental staircase**: per-region LP(t) handing off; occupancy(t)
   marching through R1→R2→R3.
4. **Confound defeat**: mastery-order vs distance (no correlation) and vs
   dependency-depth (strong) — the falsification-grade panel.

## 6. Demo (separate)

The ladder arena, live: cells coloured by current LP (the competence frontier as a
bright band), per-region gain bars lighting up in sequence (the staircase in real
time), a body meter (curriculum pauses to refuel). Sliders: NOISE level of the hard
region (watch the agent start/stop avoiding it — the noisy-TV knob) and metabolic
rate δ (how much curriculum fits between refuels). A toggle LP ↔ novelty to *show*
the noisy-TV trap appear.

## 7. Build order

0. Expected-LP drive: `LP(c)` via collapse-and-recompute on `p̂`; validate it is
   inverted-U on a hand-set difficulty gradient (cheap sanity check).
1. Arena A + LP-agent → Goldilocks dwell; add the novelty baseline → noisy-TV
   contrast. *First real result.*
2. Arena B → the staircase. Measure marching + per-region handoff.
3. Confound-defeat: asymmetric placement; mastery-order ⟂ distance.
4. Figures + `progression.html` + the demo.
5. (Stretch) Arena C, horizon axis.

## 8. Risks & honest limits

- **Perf**: LP is a field recompute per candidate; restrict to the frontier set,
  cache, recompute on a cadence. Main cost risk.
- **Depth**: n=3 caps dependency depth at ~3 cells → the staircase may be 2–3
  stages and could be mushy. Mitigate with tight chambers / corridors sized to the
  horizon; escalate to larger n if flat.
- **Collapse approximation**: "assume `c` gets mastered" is an estimate; honest in
  a near-deterministic world, weaker if world noise is added.
- **Confound**: Arena B alone can't separate depth from distance — Arena A is
  load-bearing for that. Do not ship B without A.
- **Easy vs frontier distinction from the level proxy**: showing LP *progresses*
  where `u·E_max` does not is the subtler claim; if the difference is small, lead
  with the noisy-TV result (robust) and present progression as supporting.

## 9. Separate-artifact layout

- `empower/progression.py` — Arena A/B builders, the expected-LP drive, the
  curriculum sim loop (reuses `model.py`, `empowerment.body_gate`, the planner,
  the body system).
- `make_progression_figs.py` — figures + demo data, one-command reproducible.
- `progression.html` — the short paper + integrated demo.
- `demo/progression_data.js`, `demo/progression.js`.
- One-line cross-link from the main paper (§14 "what's new" / future).

## 10. Effort

≈ A2 + B1 combined. New: expected-LP computation, two arenas, the staircase +
inverted-U + confound figures, a second demo. Reuses the learned model, the
body-gated empowerment, the planner, the viability system. Medium–high effort,
medium risk concentrated in (depth) and (perf).
