# Explore, exploit, and develop — from empowerment alone

A minimal-model study in three parts, from one quantity — **empowerment** (an agent's
control over its own future):

- **Part I — a rhythm.** A single empowerment-**gain** drive, with actuators throttled by
  a depleting body, produces a spontaneous *consolidate*↔*venture* cycle — explore/exploit
  as a temporal consequence, with the balance a physical timescale (*T* ∝ 1/δ), not a weight.
- **Part II — a direction.** Empowerment **learning-progress** on a learned model forages
  the frontier of what can be *controlled*, ignores the "noisy TV," and (given prerequisite
  structure) climbs a developmental curriculum with no schedule.
- **Part III — the fuller picture.** Because empowerment counts only the futures you *cause*,
  it sidesteps the noisy-TV and grand-roulette traps of info-gain and Maximum Occupancy; the
  paradox map situates it among the reward-free principles. §22 then tests the claims on a
  foraging POMDP built by MOP's authors (Mylonas & Moreno-Bote, arXiv:2609.39342), against a
  MOP calibrated to their reported numbers.

**Read the paper / play with the two live demos:** open [`index.html`](index.html) in a
browser (the demos run the actual agents in JavaScript — nothing is pre-recorded).

## What's here

| path | what |
|---|---|
| `index.html` | the full three-part paper + two integrated interactive demos |
| `empower/` | the simulation (Python) |
| `make_figures.py` | reproduces every Part I figure + demo data from one script |
| `make_progression_figs.py` | reproduces the Part II/III figures + demo data |
| `build_mylonas_tables.py`, `select_patch_world.py`, `make_mylonas_figs.py`, `run_mop.py`, `robust_mop.py`, `make_contingency_fig.py` | the §22 foraging replication (tables, world selection, figures `myl*`/`mylp*`, MOP, robustness) |
| `tests/test_mylonas.py` | brute-force check of the §22 empowerment computation |
| `paper/figs/` | generated figures (`fig*` = Part I, `prog*` = Part II/III) |
| `demo/` | `data.js`/`demo.js` (Part I cycle) and `prog_data.js`/`progression.js` (Part II curiosity) |
| `brainstorm.md` | the original design brief for Part I |
| `progression_brief.md` | the design brief for Part II/III |
| `progression.html` | redirect into the merged paper (kept for old links) |

### `empower/`
- `arena.py` — gridworld, body-gated actuator dynamics, sensors, BFS distances.
- `empowerment.py` — *n*-step empowerment via Blahut–Arimoto (position **or** sensor state).
- `progression.py` — equidistant/ladder arenas + the learning-progress agent (Part II).
- `gated.py` — switch–gate dependency environment + the unlock staircase (Part II, §19).
- `paradox.py` — grand-roulette MDP + MOP-vs-control soft value iteration (Part III, §21).
- `mylonas.py` — the two-food POMDP of §22 (their world and the patch world), exact belief-POMDP
  empowerment, the greedy and gain-planner arms, and the objective-free reference policies.
- `mop_belief.py` — MOP (their Eq. 6) by value iteration over position, energy and belief.
- `agent.py` — empowerment lookup, tabular uncertainty model, the single gain drive
  (somatic + viability-gated epistemic faces), the two-term baseline
  `R=(1−β)·E+β·novelty` for the head-to-head, shortest-path planner, and the
  `RolloutPlanner` (the §13 forward-rollout viability experiment).
- `model.py` — Dirichlet world model `p̂(s'|s,a)`; `simulate_model.py` — agent that
  plans on empowerment computed from the learned model (the §12 proof).
- The dynamical reduction (homeostatic setpoint + `T∝1/δ`, §9 / Fig. 12) is built
  inline in `make_figures.py` (`fig_reduction`).
- `simulate.py` — the episode loop and logging.
- `diagnostics.py` — period, phase, orbit, and spectrum measures.

## Reproduce

```bash
pip install numpy matplotlib
python make_figures.py            # all figures -> paper/figs/, demo data -> demo/
# or a subset:
python make_figures.py field headline scaling demo

# section 22 (empowerment tables take ~45 min on 8-10 cores; cached in data/)
python build_mylonas_tables.py && MYL_WORLD=patch python build_mylonas_tables.py
MYL_WORLD=theirs python make_mylonas_figs.py && MYL_WORLD=patch python make_mylonas_figs.py
python run_mop.py && python robust_mop.py && python make_contingency_fig.py
```

Then serve the folder and open `index.html`:

```bash
python -m http.server 8000        # http://localhost:8000
```

## The idea in one paragraph

Empowerment is the channel capacity from an agent's actions to its future
observations — a reward-free measure of control. A drive that only climbs a fixed
empowerment landscape ends at a fixed point; alternation needs delayed negative
feedback. Here the feedback is the agent's **body**: an integrity variable `b` that
depletes with time and gates the actuators, so realizable empowerment is bottlenecked
by `b` (data-processing inequality). The agent maximizes the *gain* `ΔE`, which has two
faces of the same functional — refuelling (large only when hungry) and reducing model
uncertainty (gated by the body budget, so you can't explore on a depleted platform).
Two slow variables (body integrity, model uncertainty) hand off which face is binding,
giving a relaxation oscillation: consolidate, then venture. See the paper for the
critical discussion of what is and isn't novel.
