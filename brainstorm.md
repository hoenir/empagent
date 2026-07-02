# Me:

I want to explore whether it would be possible to implement the empowerment principle in very simple model systems (eg lenia, braitenberg), to ultimately show that drives to exploit and explore alternate each other just from empowerment alone. Has anyone done something like this in the  literature? Would it be feasible?

## Claude:


Here's the design brief. I'm building it around one honest constraint that should govern everything downstream: a drive that only ever climbs a static empowerment landscape ends at a fixed point. Alternation is a *limit cycle*, and limit cycles need delayed negative feedback. So the real design question isn't "how do we compute empowerment" — that's solved — it's "what makes occupying a high-empowerment state lower its empowerment, with a recovery lag." Everything below is organized around that.

# Brief: explore/exploit alternation from an empowerment-derived drive in a Braitenberg vehicle

## 0. The contribution we're testing (north star for the coding agent)

The established alternatives — active inference's expected free energy (epistemic + pragmatic terms) and de Abril & Kanai's curiosity + empowerment — both *add two terms with a weighting parameter* that trades exploration against exploitation. Our claim is structurally different: a **single** empowerment-derived value, with no second term and no tradeoff knob, produces both behaviors as temporal *phases*. If it works, the discriminating signature is that the explore/exploit balance is set by physical timescales of the agent–environment coupling (how fast control degrades and recovers), not by a hand-tuned mixing coefficient. That's the falsifiable difference and the paper hook.

I'll flag now (speculation): I expect the literal "from empowerment *alone*, nothing added" version to require *some* negative feedback coupling. The cleanest version keeps the agent's drive purely empowerment and pushes the feedback into the environment, where it shows up as a literal drop in channel capacity. That keeps "empowerment alone" honest at the level of the agent's objective.

## 1. Environment

Discrete first (exact empowerment, no estimator noise to confound the dynamics); continuous as a follow-up only after the discrete phenomenon is solid.

A 2D grid arena with designed heterogeneity — this is non-negotiable, because in a uniform open arena empowerment is flat and there's nothing to forage. We want **multiple separated high-empowerment basins** (open patches) connected by **low-empowerment constrictions** (corridors, near-wall cells, hazard/absorbing cells that collapse future reachability). Classic result to expect: empowerment is high in open interior cells (many distinguishable reachable futures) and low near walls, dead-ends, and absorbing states. The multi-basin structure gives the agent somewhere to *go* when one basin is spent.

Agent: minimal differential-drive Braitenberg vehicle — state $(x, y, \theta)$, a small discrete action set (e.g. {forward, forward-left, forward-right, pivot-left, pivot-right} = 5 primitives). Keep $|A|$ small because $n$-step empowerment enumerates $|A|^n$ sequences.

Decision to make explicitly and early: **empowerment over true position vs. over sensor states.** Faithful Braitenberg empowerment is the channel capacity from motor commands to *future sensor readings*, not to ground-truth position. These differ under partial observability and the difference may matter for behavior. Recommendation: implement both, debug on full-state empowerment, and run the headline experiment on sensor-state empowerment for fidelity.

## 2. Empowerment computation

$n$-step empowerment $E_n(s) = \max_{p(a^n)} I(A^n; S_{t+n}\mid s)$, computed by Blahut–Arimoto on the $n$-step transition matrix $p(s_{t+n}\mid s, a^n)$, with $n \in \{2,\dots,5\}$. Two distinct transition operators in play, and the coding agent must never conflate them:

- $p(s'\mid s,a)$ — the **true** environment dynamics, used to *generate* transitions.
- $\hat p_t(s'\mid s,a)$ — the agent's **learned model**, used to *compute* the empowerment the agent acts on.

For the level-objective experiments the agent may use the true dynamics (no learning); for the gain experiments it must use $\hat p_t$. Cache empowerment per cell; recompute only locally where the model or the noise field changed (full recompute every step is the obvious performance trap).

## 3. The two drive variants

Define empowerment evaluated under the agent's model at time $t$ as $\hat E_t(s)$.

**Level (baseline / exploitation pole):**
$$R_{\text{level}}(s) = \hat E_t(s)$$
The agent moves toward states of high estimated empowerment. This is Keep-Your-Options-Open. Prediction: parks at the global empowerment max — a fixed point.

**Gain (the thesis-critical drive — Schmidhuberian):**
$$R_{\text{gain}}(s_t) = \hat E_t^{\text{post}}(s_t) - \hat E_t^{\text{pre}}(s_t)$$
where $\hat E^{\text{pre}}$ uses the model *before* folding in the transition just observed at $s_t$, and $\hat E^{\text{post}}$ uses it *after*. Both are evaluated at the *same* state $s_t$, which isolates the contribution of *knowing more* (control you can now realize) from the contribution of *moving somewhere better*. This is empowerment playing the role of compression and $R_{\text{gain}}$ playing the role of compression *progress*: self-exhausting, because once the local model converges, $\hat E^{\text{post}} \approx \hat E^{\text{pre}}$ and the reward for staying drops to zero, pushing the agent out.

(Intermediate variant worth logging but not central: $R_{\Delta\text{state}} = \hat E_t(s_t) - \hat E_t(s_{t-1})$ — pure hill-climbing on the current landscape. It's still exploitation, just framed as ascent; useful as a sanity bridge between level and gain.)

## 4. The negative feedback — two mechanisms, both delayed

Without one of these, every variant settles. These are the heart of the experiment.

**(D) Dynamical depletion — "empowerment depletes itself."** Maintain a per-cell noise field $c(x)\in[0,1]$. Occupancy raises it: $c \leftarrow c + \kappa(1-c)$ where the agent sits. The true dynamics get noisier with $c$ — e.g. with probability $c$ the commanded action is replaced by a uniform-random action. Higher $c$ → lower channel capacity → lower *true* empowerment there. Recovery: $c \leftarrow c(1-\rho)$ each step. This has direct precedent in Salge & Polani's empowerment-under-state-dependent-noise work, so it's not an arbitrary bolt-on. The agent's objective stays *purely empowerment*; the alternation comes from the agent–environment coupling. This is the cleanest "empowerment alone" existence proof.

**(E) Epistemic regeneration — "you've learned all there is here."** Tabular Bayesian model: Dirichlet counts $N(s,a,s')$ that decay toward the prior, $N \leftarrow \lambda N$ ($\lambda<1$, forgetting), and are refreshed by visitation. Gain regenerates wherever counts have decayed. This is the model-uncertainty route — pure empowerment-under-uncertainty, no environmental change at all. Pairs naturally with $R_{\text{gain}}$.

D feeds back through the world (capacity literally falls); E feeds back through the agent's knowledge (gain falls, then recovers). In both, the recovery *lag* plus the *travel time* between basins is what sets the oscillation period — which is exactly the "timescale, not tradeoff parameter" claim from §0.

## 5. Controller

Tabular world + tabular model → receding-horizon greedy planning under the current model: at each step, plan to depth $\approx n$ maximizing expected discounted reward (level or gain), execute one step, replan. Don't reach for full belief-MDP planning — the gain reward is formally non-Markovian in base state (it depends on the model/belief), but a short-horizon planner against the current model is the right first approximation and the phenomenon should be robust to controller details if the drive structure is right.

Tractable proxy for the gain reward during planning (since true expected gain requires simulating the Bayesian update for every candidate): use model uncertainty at $(s,a)$ — Dirichlet posterior entropy or inverse visit count — weighted by local empowerment potential, as a stand-in for "how much will visiting here raise my realized control." Start with the proxy; only escalate to simulated-update expected information gain if the proxy fails to produce structure.

## 6. Experiment matrix and predictions

The 2×2 of {objective} × {feedback}, plus no-feedback controls. State predictions up front so the run is falsifiable:

| | no feedback | (D) dynamical | (E) epistemic |
|---|---|---|---|
| **level** | park at global $E$-max (fixed point) | **foraging limit cycle** — *Version α, the minimal claim* | ≈ park (level ignores knowledge) — control |
| **gain** | sweep regions, then go quiescent | cycle (richer: capacity + knowledge) | **patrol/re-explore** — *Version β, the compression-progress story* | 

The cell that most purely supports "alternation from empowerment alone, no second drive, no resource counter, no tradeoff knob" is **(level, D)**: the only quantity is empowerment, the only drive is to climb it, and alternation emerges because climbing it locally destroys it. Headline that. **(gain, E)** is the deeper intrinsic-motivation result and the one that connects to your Schmidhuber/MaxEnt program. The off-diagonal controls earn their keep by showing the level objective *can't* exploit epistemic regeneration and the gain objective *needs* regeneration to sustain.

Note honestly: **(gain, no feedback)** terminating is a feature, not a bug — it's the de Abril/Kanai "empowerment alone under-explores then settles" result reproduced. Don't let the coding agent treat quiescence as a failure to debug.

## 7. Diagnostics — proving it's a cycle, not diffusion

The money figure is a **phase portrait**: plot $\hat E(s_t)$ against $\dot{\hat E}$ (or against the gain signal). A point attractor (level, no feedback) collapses to a dot; a genuine limit cycle traces a closed orbit. Supporting measures:

Operationalize phases — *exploit* = low speed + high local $E$ + low model-update magnitude (parked in a basin); *explore* = transit / high displacement + low local $E$ or rising model-update. Then: autocorrelation and power spectrum of the phase signal (a limit cycle gives a dominant frequency); dwell-time distribution (expect bimodal — long basin dwells, short transits); and the decisive one — **vary $\kappa,\rho$ (or $\lambda$) and confirm the period scales predictably**, the signature of a relaxation oscillation rather than an artifact.

Controls against trivial explanations: a random-policy baseline (alternation must beat diffusion), and the §0 decomposition test — show that the *single* reward stream is dominated in different phases by the "move to higher $E$" component vs. the "improve estimate of $E$" component, with no mixing parameter doing the work.

## 8. Milestones (build order)

0. Grid + agent + Blahut–Arimoto on true dynamics → visualize the empowerment field; confirm high in open cells, low near walls/hazards (replicates Keep-Your-Options-Open).
1. Level controller → confirm fixed-point parking.
2. Add (D) depletion + recovery → look for the (level, D) limit cycle. *First real result.*
3. Add tabular Bayesian model + $R_{\text{gain}}$ → (gain, none) sweep-then-quiesce, then (gain, E) sustained patrol.
4. Full 2×2 + diagnostics + timescale-scaling + controls.

## 9. Risks that could kill it (watch these)

A uniform or too-coarsely-discretized arena flattens empowerment → no foraging; the heterogeneity and the discretization must preserve the "open = more reachable futures" gradient. Sensor-empowerment may behave qualitatively differently from position-empowerment — decide deliberately, don't let it be an accident. Blahut–Arimoto per-cell-per-step is the performance trap; cache aggressively. And the conceptual risk I'd weight highest: a reviewer arguing (level, D) is "just foraging with empowerment as a relabeled resource." The defense is that the depleting quantity is *literally channel capacity falling under occupancy-induced noise*, not an arbitrary food counter — and that (gain, E), which has no environmental resource at all, alternates for the same structural reason. Make sure the code can demonstrate both so the argument holds.

---

Two things I'd want your call on before the coding agent starts: whether to commit to sensor-state empowerment from the outset (more faithful, more confounds) or earn it after the full-state version works; and whether the headline you want for the paper is the minimal **(level, D)** existence proof or the richer **(gain, E)** compression-progress story — because that determines which cell gets the careful treatment and which is a supporting control.

## Me:

Sensor-state seems the way to go. The richer gain story is what i want to work towards but we can start with the minimal existence proof. I like the idea of pushing the feedback into the environment, but I feel the environment could be the agent('s body). Not sure what that would map unto in your feedback types. The idea is that the body degrades (the channel capacity?) when certain patterns (returning to resource) aren't revisited. However, when well-fed, habitual cycles (like 'feeding') don't deliver the empowerment gains, so the mode implicitly shifts to explorative. Does that make sense?
Conceptually, at a high level, it should be a minimal proof (ie in a minimal system) of this idea:

To explore effectively, one must first consolidate, maintaining the very constraints and patterns that enable future growth. A child masters basic motor skills through repetition before attempting new challenges. There’s a natural progression whenever there’s no empowerment gain left to be made relative to current goals. This creates a characteristic rhythm: horizon-expanding alternates with horizon-narrowing (aka the explore-exploit cycle). We venture into uncertainty, then retreat to reliable patterns to recharge before the next exploratory arc. The conservative phase isn’t opposed to growth, it’s its precondition.

I *think* what you are proposing reflects this idea, but i'm still working through the mapping to the implementation/computation.

## Claude:

Yes, it makes sense — and it's a better version than what I sketched, because internalizing the feedback into the body does real theoretical work for you that an external resource-cell can't. But it forces one correction up front, and it's important: **what you're describing is not a small version of the minimal (level, D) proof — it's the gain story, and the minimal proof doesn't contain its defining feature.** Let me justify that, then give you the mapping and the computation.

## Why this is intrinsically a gain story

The hinge of your description is "when well-fed, feeding delivers no empowerment gain, so the mode shifts to exploration." That flip is only available to a drive that tracks the *derivative*. A level drive, once sated, sits at the (now-high) empowerment maximum and does nothing interesting — it has no reason to leave, because leaving lowers $E$. The satiation-flips-to-exploration mechanism is the signature of $R_{\text{gain}}$, full stop. So:

- (level, somatic-depletion) still gives you a *limit cycle* — hungry, feed, deplete, feed — but both phases are exploitation. There is no horizon-*expanding* phase. It proves "oscillation, not fixed point," which is the weak existence claim, and it's worth building as scaffolding to de-risk the body↔channel coupling. But it is **not** your idea.
- Your idea — consolidate, then venture into *uncertainty* — needs gain, and it needs two distinct sources of gain to fill the two phases. The exploit phase is fed by *somatic* gain; the explore phase is fed by *epistemic* gain. Both are $\Delta\hat E$ of the same functional, so it's still "empowerment alone." That's the thing to be precise about with a referee: one quantity, two faces, alternating because of slow body/uncertainty dynamics — not two terms with a weight.

So your proposal introduces a feedback type I didn't name. Call it **(B) somatic/viability depletion**: the agent's own actuator→sensor channel degrades with an internal body state, restored by resource contact. It supersedes my (D), and it pairs with (E) rather than competing with it.

## The mapping (your words → mechanism)

| Your phrasing | Mechanism |
|---|---|
| "the body degrades when patterns aren't revisited" | body state $b_t \in [0,1]$ decays each step ($b \leftarrow b - \delta$); resource contact restores it ($b \leftarrow \min(1, b+\iota)$) |
| "the channel capacity?" *(your guess)* | yes — $b$ sets the **self-state-dependent motor-noise floor**: with prob $\sim g(b)$ the commanded action is corrupted, $g$ decreasing in $b$. Low $b$ → noisy actuators → lower $I(A^n;S_{t+n})$. So $E$ is now a function $E(s,b,\hat p)$ and scales with $b$ |
| "well-fed → feeding doesn't deliver empowerment gains" | $b\!\to\!1 \Rightarrow \partial E/\partial b \to 0$; feeding produces $\Delta\hat E \approx 0$ |
| "mode implicitly shifts to explorative" | with somatic gain exhausted, the **only remaining positive $\Delta\hat E$ is epistemic** — improving $\hat p$ where the model is uncertain — so the gain-greedy agent moves toward novelty |
| "to explore effectively one must first consolidate" | realized empowerment $\approx$ potential $\times$ body integrity $\times$ model reliability; exploring on a depleted platform yields no *realizable* gain, so the system restores the platform first |
| "no empowerment gain left relative to current goals" | local somatic + epistemic gain both saturated → progression to the next niche. (Watch the word "goals" — there are none; the apparent goal is just wherever gain currently concentrates) |
| "horizon-expanding / horizon-narrowing" | explore = enlarging the region over which control is realizable; consolidate = narrowing onto the resource to rebuild the platform |

## The computation, and the part that's genuinely elegant

The strongest move in your version is that **homeostasis falls out of empowerment gain rather than being added as a hunger drive.** Because $E(s,b)$ rises steeply with $b$ when $b$ is low (steep $g$ near zero), the empowerment gain from feeding is *automatically largest exactly when the agent is hungriest*. You never write a "hunger" term. "Hunger" is just the name for the region of body-state where feeding is gain-rich. That's a real reduction, and it's the kind of thing your MaxEnt/empowerment program wants: a viability drive derived from, not bolted onto, the information-theoretic objective.

So the whole system is one drive over an augmented state:

$$R_{\text{gain}} = \hat E(s', b', \hat p') - \hat E(s, b, \hat p), \qquad E = \max_{p(a^n)} I(A^n; S_{t+n} \mid s, b, \hat p)$$

with three additive sources of positive gain, all the same functional:
1. **somatic** — $b' > b$ while $\partial E/\partial b > 0$ (drives return-and-consolidate; self-largest when hungry);
2. **epistemic** — $\hat p'$ more accurate where it constrains $E$ (drives exploration toward uncertainty);
3. positional — moving to a context with a higher ceiling (static in a fixed arena; not a driver of alternation, just present).

Phase 1 is dominated by (1), phase 2 by (2), and the slow dynamics of $b$ (and of model-uncertainty $u$) shuttle dominance between them. **No mixing coefficient.** The explore/exploit balance is set by $\delta$ (metabolic rate), $\iota$ (intake), and the model forgetting rate $\lambda$, against arena travel times — your §0 "timescale, not tradeoff" claim, now made literal.

The dynamical-systems framing to put in the paper: this is a **relaxation oscillator** in the two slow variables $(b, u)$ — fast action-selection riding two slow recovery/depletion processes with hysteresis, the van der Pol motif. That's what gives you a *two-phase rhythm with sharp transitions* rather than a smooth blend, and the period scales predictably with $\delta$ and $\lambda$. It also lets you make a sharper claim than active inference: AIF sums an epistemic and a pragmatic term *every step* with a fixed relative weight; here the same two quantities (epistemic gain ≈ epistemic value; somatic gain ≈ a viability-grounded pragmatic value) emerge from a *single* empowerment-gain functional and dominate in *alternating phases* because of the slow dynamics. You'd be re-deriving the EFE decomposition and explaining the temporal structure the static sum leaves out. (Flagging that as the ambitious, speculative version of the contribution — worth stating carefully, not overclaiming.)

## Four subtleties that decide whether it's deep or shallow

**Quiescence vs. exploration in the well-fed phase.** This is the one that can sink it. When sated, somatic gain is zero; the agent only ventures out if epistemic gain is *live and adequately scaled* and the planning horizon is long enough to see that uncertain regions are gain-rich. If epistemic gain is too weak or the horizon too short, the well-fed agent just sits — you reproduce the de Abril/Kanai "empowerment under-explores then settles" result instead of your cycle. The epistemic term must be a first-class part of $\Delta\hat E$, not an afterthought. (Encouragingly, the well-fed window is exactly when the agent has body reserves to *spend* on travel — "recharge before the next exploratory arc" is mechanically self-justifying. But it only fires if the epistemic gradient is there.)

**"Consolidation as precondition" risks collapsing into mere refueling.** If the local model around the resource is fully learned after the first visit, every later return adds zero epistemic value — the consolidation phase is just a gas station, and your motor-skill analogy is unearned. To make consolidation do genuine work, let model-uncertainty about the *home/near region regenerate during long absences* (the same forgetting $\lambda$, applied at home). Then the return phase performs real epistemic re-consolidation — rebuilding the reliable substrate — not just topping off the tank. That's the difference between "refuel and leave" and "the child re-grooving basics before the next attempt."

**The starvation cliff and planning horizon.** Because $g(b)$ should be steep near $b=0$ (empowerment collapses as the body fails), the gain from feeding-when-critically-hungry is enormous, which self-corrects the return — *provided the planner's horizon reaches the resource from wherever exploration took it.* Horizon vs. arena scale is therefore a hard constraint, not a nicety. Too myopic and the agent explores into a region it can't afford to leave.

**Goldilocks band for $\delta$.** Fast metabolism relative to travel time traps the agent in a tight feeding cycle (pure exploit, never explores); too slow and the body never drops enough to pull it back (it wanders off and starves or never returns). There's a window where both phases have meaningful duration. Mapping that window *is* a result — it's the relaxation oscillator's existence region, and showing the period scales with $\delta$ inside it is your falsification-grade evidence.

## What changes in the brief

Build order, revised: (0) sensor-state empowerment field with $E(s,b)$, confirm body scales capacity; (1) somatic relaxation oscillator under a *level* drive — scaffolding, proves cycling, explicitly *not yet the thesis*; (2) introduce the tabular Bayesian model and $R_{\text{gain}}$ with both somatic and epistemic sources, home-region forgetting on; (3) demonstrate the consolidate→venture cycle with genuine epistemic exploration in the well-fed phase; (4) phase portrait in $(b,u)$, period-vs-$\delta$/$\lambda$ scaling, controls. The headline is step 3, and the diagnostic that proves it isn't refueling-plus-diffusion is showing the exploration phase is driven by the *epistemic* component of the single gain stream while the consolidation phase is driven by the *somatic* component — same functional, no weight between them.

One thing I'd want you to decide, because it changes the cleanest claim: do you want the body to gate **actuator** reliability (motor noise — hungry agents act sloppily) or **sensor** resolution (hungry agents perceive coarsely)? Both lower $I(A^n;S_{t+n})$, but they tell different biological stories and may give different phase geometry. Actuator-gating is the more conventional reading of "empowerment as control"; sensor-gating couples more directly to your sensor-state choice. I'd lean actuator for the first pass and treat sensor-gating as the variant — but it's your enactivist intuition that should call it.


## Me:

yes actuator reliability seems most intuitive (most costly/resource-dependent)


## Claude:

Good — and actuator-gating pays off more than sensor-gating would, because it gives empowerment-as-channel-capacity an almost literal Shannon reading. Worth making that precise before we lock the brief, because it upgrades the "hunger emerges from gain" claim from plausible to derivable.

## Why actuator-gating has the clean form: the body is a serial bottleneck

The commanded action passes through two channels in series before it shows up in future sensors:

$$A^n \;\xrightarrow{\text{body, noise } g(b)}\; \tilde A^n \;\xrightarrow{\text{world + sensors}}\; S_{t+n}$$

The body stage corrupts the command (with prob $g(b)$, replace with a uniform-random primitive; $g$ decreasing in $b$). The world stage is the environment dynamics composed with the sensor map. Empowerment is the capacity of the *composition*, and the data processing inequality across the Markov chain $A^n \to \tilde A^n \to S_{t+n}$ pins it from both sides:

$$E(s,b) \;\le\; \min\big(\underbrace{C_{\text{body}}(b)}_{\max I(A^n;\tilde A^n)},\; \underbrace{E_{\max}(s)}_{\max I(\tilde A^n;S_{t+n})}\big)$$

— and in practice it tracks that $\min$ closely. $C_{\text{body}}$ runs monotonically from $\log|A|^n$ at $b{=}1$ down to $0$ at $b{=}0$; $E_{\max}(s)$ is the body-independent positional ceiling we already had. So body integrity is a *serial capacity bottleneck on realizable empowerment*, derived, not posited.

This is what makes the satiation-flip rigorous rather than hand-wavy. The gradient $\partial E/\partial b$ is just **which stage is binding**:

- **Hungry** ($C_{\text{body}} < E_{\max}$): the body is the bottleneck, feeding directly lifts the binding constraint, $\partial E/\partial b$ is large. "Hunger" is the name for this regime — you never write a hunger term, it's the region where the body stage is binding.
- **Well-fed** ($C_{\text{body}} > E_{\max}$): the environment is binding, more body integrity buys nothing, $\partial E/\partial b \to 0$, somatic gain dies, and the only live $\Delta\hat E$ left is epistemic. The mode flips to exploration on its own.

The flip is a **bottleneck handover**, and the steepness of $g$ near $b{=}0$ sets how sharp the cliff is. That's the cleanest possible statement of your "to explore effectively, first consolidate": exploring while the body stage is binding yields no *realizable* empowerment gain no matter how rich the world, because the series can't carry more than its tightest stage. The precondition isn't a preference; it's the data processing inequality.

## Two consequences the coding agent must respect

**A hard design constraint, or the cycle never starts.** Satiation only completes — somatic gain only reaches zero — if $C_{\text{body}}(b{=}1) > E_{\max}(\text{feeding site})$. If the resource sits in a maximally open basin where $E_{\max}$ is huge, the body stays binding even when full, the agent never sates, and it never leaves to explore. So **place the resource in a moderate-empowerment region** (or scale $C_{\text{body}}(1)$ above the local ceiling). This is the precise existence condition for the explore phase, and it's easy to get wrong.

**A free testable prediction (flag: speculative, second-order).** The handover point $C_{\text{body}}(b) = E_{\max}(s)$ is position-dependent, so effective hunger threshold should depend on *where the agent is*: in open basins the body becomes binding at higher $b$, so the agent should "get hungry sooner" there than in cramped regions. If that shows up, it's a non-obvious signature that the drive really is the bottleneck structure and not a relabeled hunger variable. Don't build for it; just log enough to check it post hoc.

---

## Consolidated brief (handoff version — supersedes the scattered design above)

**Goal.** Minimal proof that a *single* empowerment-gain drive, with no second term and no explore/exploit mixing coefficient, produces a two-phase consolidate↔venture rhythm, where the balance is set by physical timescales (metabolic rate, model-forgetting rate) rather than a tuned weight.

**Environment.** 2D grid, heterogeneous: ≥2 open high-$E$ basins joined by low-$E$ constrictions; walls/dead-ends/absorbing cells as natural empowerment sinks. One resource cell in a *moderate*-$E$ region (see constraint above).

**Agent.** Differential-drive vehicle, state $(x,y,\theta,b)$ with body integrity $b\in[0,1]$; action set ~5 primitives. Partial observability: empowerment computed over **future sensor states**, not ground-truth position (debug on full-state first, switch to sensor-state for the headline run).

**Body dynamics.** Decay $b \leftarrow b-\delta$ each step; resource contact $b \leftarrow \min(1,b+\iota)$. Actuator gating: with prob $g(b)$ the command is replaced by a uniform-random primitive, $g$ steep near $b{=}0$. This makes the true transition operator body-dependent: $p(s'\mid s,a,b)$.

**Empowerment.** $E(s,b,\hat p)=\max_{p(a^n)} I(A^n;S_{t+n}\mid s,b)$ via Blahut–Arimoto on the $n$-step sensor-transition matrix, $n\in\{2..4\}$. Keep two operators strictly separate: **true** $p$ (generates transitions, includes real body noise) vs. **learned** $\hat p_t$ (tabular Dirichlet counts, forgetting $N\leftarrow\lambda N$, the drive is computed against this). Cache per (cell, $b$-bin); recompute locally only where model or $b$ changed.

**Drive (single functional).**
$$R_{\text{gain}} = \hat E(s',b',\hat p') - \hat E(s,b,\hat p)$$
Three additive faces, same functional, no weights: **somatic** ($b'\!>\!b$ while body is binding — drives return/consolidate, self-largest when hungry), **epistemic** ($\hat p'$ more accurate where it constrains $E$ — drives exploration toward uncertainty), positional (static in fixed arena). Apply forgetting $\lambda$ to the **home/resource neighbourhood too**, so the consolidation phase performs real epistemic re-grooving (motor-skill repetition) rather than mere refueling.

**Controller.** Receding-horizon greedy on the current model, plan depth $\gtrsim n$ and **horizon must reach the resource from the deepest exploration reachable** (else starvation-trap). For the epistemic face during planning, use a tractable proxy — Dirichlet posterior entropy (or inverse visit count) at $(s,a)$ weighted by local $E$-potential — before escalating to simulated-update expected gain.

**Build order.**
0. $E(s,b)$ field on true dynamics → confirm it's high in open cells, low at walls, and *scales with $b$* (body as bottleneck visible).
1. Level drive → confirm fixed-point parking (control).
2. Level + somatic depletion → relaxation oscillator (proves cycling; explicitly *not yet the thesis* — both phases are exploit).
3. **Headline:** $R_{\text{gain}}$ with somatic + epistemic faces, home-region forgetting on → consolidate→venture cycle with genuine exploration in the well-fed phase.
4. Full diagnostics + controls.

**Diagnostics (prove cycle, not diffusion, not refueling).** Phase portrait in the two slow variables $(b, u{=}\text{model uncertainty})$ — closed orbit vs. dot. Phase labels: exploit = low speed + body binding + low model-update; explore = high displacement + epistemic-dominated gain. Decisive tests: (i) **period scales predictably with $\delta$ and $\lambda$** (relaxation-oscillator signature); (ii) the consolidation phase is dominated by the *somatic* component of the single gain stream and the venture phase by the *epistemic* component, with **no mixing parameter** anywhere in the code; (iii) beats a random-policy baseline. Map the Goldilocks band of $\delta$ (too fast → trapped feeding; too slow → wanders off/starves) — that band *is* the oscillator's existence region, and reporting it is a result.

**Known non-failures (don't let the agent "fix" these).** $R_{\text{gain}}$ with *no* feedback going quiescent after a sweep is the de Abril/Kanai under-exploration result reproduced, not a bug. Well-fed quiescence *with* feedback, however, *is* a failure — it means the epistemic face is too weak or the horizon too short; that's the first thing to tune, not the depletion rates.

The framing payoff to write up: active inference sums an epistemic and a pragmatic term every step at fixed relative weight; here both quantities fall out of one empowerment-gain functional and *alternate in time* because two slow variables hand off which face is binding. You're re-deriving the EFE decomposition and explaining the temporal structure the static sum omits. (Ambitious version — state it carefully, don't overclaim until the period-scaling holds.)

That's everything locked. The only thing I'd still want your eye on before handoff: the home-region forgetting in step 2/3 is the single load-bearing assumption that turns "consolidation" from refueling into genuine re-grooving — if you're uneasy about it being a thumb on the scale, we should think about whether near-resource uncertainty can regenerate for a *principled* reason (e.g. body-state-dependent sensor aliasing that changes what "home" looks like across body states) rather than by imposed forgetting.