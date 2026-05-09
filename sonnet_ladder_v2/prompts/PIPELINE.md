# Ladder Pipeline v2 — Runbook

A multi-agent ideation pipeline that explores a research solution space by
spawning specialized Sonnet sub-agents at each stage, accumulating empirical
knowledge across rounds, and converging on testable experiments. The v2
ladder revises the v1 cascade by importing the Opus pipeline's parallel-
orientation Stage 1 generation pattern, preserving all the math-correctness
gates v1 earned through past failures, and adding a new skeptic-controls
gate for transfer / consistency claims.

This file is a full replacement runbook. Anyone reading PIPELINE.md should
be able to run the v2 ladder without referring to the v1 PIPELINE.md.

---

## Philosophy

**The pipeline characterizes a solution space rather than narrowing it.**
Every killed idea is a piece of empirical knowledge — not a reduction of
available ideas. The number of generatable ideas is effectively unbounded;
what bounds the search is the quality of the empirical map being built. Each
round adds boundary markers: "this region looks promising but failed for
reason X." Future ideators reason from those markers to find unexplored
frames.

**Killing ideas with new knowledge is a feature, not failure.** A round that
falsifies an entire reasoning template (e.g., "less load-bearing → more
compressible") is more valuable than a round that confirms a marginal
improvement, because the falsification constrains a *class* of future ideas
rather than a single point. The pipeline is optimized for class-level
findings.

**Ideas are limitless; the bottleneck is empirical traction.** The agent
stack costs are dominated by stages that fire ideas without testing. The
point of the pipeline is to convert ideation into experiments at high
throughput while preserving rigor (Stage 6 red-teaming has caught multiple
math errors that would have wasted real implementation time).

**Every result reshapes the search space, not closes it.** Mass pre-emption
is signal, not failure — it maps the publishing-saturated boundary, beyond
which novel territory lives. The pipeline does not "saturate" in the sense
of running out of ideas; it learns where the well-trodden grooves are and
generates further from them.

---

## Why v2 imports the parallel-orientation pattern

The v1 ladder's known structural weakness was Sonnet's training-distribution
prior: ideators stayed close to the published ML-compression vocabulary,
re-deriving what was already known. The Opus pipeline addressed this by
spawning **multiple parallel agents under different creative orientations**
(Reach / Composition / First-Principles / Unconventional Angles), each with
the same shared empirical substrate but a different generative pose. The
empirical signal was clear: Opus pipeline R1 produced measurably more
pre-emption-resistant proposals than v1 single-stream rounds, *and* the
convergences across orientations were themselves a load-bearing structural
signal — when two creative poses independently surface the same primitive,
the model's empirical structure is asserting itself across vocabularies.

v2 imports this pattern at Stage 1 (Sonnet ideators bound to orientations),
preserves the v1 cascade structure for Stages 2–6 (Judge → Deep Researcher
→ Skeptic Explorer → Selector → Red Team), and modifies the downstream
stages to handle the richer Stage 1 output without:
- prematurely killing audacious-but-grounded proposals just because they
  arrive in unfamiliar vocabulary (the v1 Judge's standard-ML-playbook
  bias), AND
- weakening the math-correctness gates v1 earned through CSMF, LFAO, CRMS,
  CSKQ, RFSK, THWR, and WDLA failures.

The v2 result should be: more ideas at a higher quality floor, with the
same rigor on math correctness, and an additional rigor on transfer/
consistency claims via the new skeptic-controls gate.

---

## Round structure (per round)

Each round runs 6 stages. Each stage's agent receives a self-contained brief
that includes the cumulative `KILL_LIST.md`, `SUMMARY.md`, and the relevant
prior stage's output for that round.

| Stage | Role | Agent count | Output |
|-------|------|-------------|--------|
| 1 | Orientation-bound Ideator (parallel) | K = 4–5 (one per active orientation) | 4–8 ideas per orientation, 6-section format, track-classified |
| 2 | Judge (cross-orientation union) | 1 | Union score table, convergence map, frame-novelty bonus advancers, hand-off to Stage 3 |
| 3 | Deep Researcher (per advancer) | One per advancer (parallel) | Per-mechanism prior-art status, frame-mismatch verification, calibration pre-flight, residency arithmetic, sharpened smallest-test, refined go/no-go, verdict |
| 4 | Skeptic Explorer (cross-orientation gap-finder) | 1 | Frame-exhaustion map, orientation-saturation diagnostics, cross-pollination opportunities, 6–10 new ideas in unexplored frames |
| 5 | Selector (per track) | 1 per track | One experiment per track with hypothesis, smallest test, go/no-go, ambiguous-zone follow-up, kill criteria, skeptic-controls declaration, runtime, script |
| 6 | Red Team (adversarial review) | 1 per Stage 5 selection | Frame-mismatch re-check, calibration re-check, skeptic-controls check, hidden-prior-art search, biased-framing audit, runtime sanity, CF13–CF15 verification, verdict |

After Stage 6, ACCEPT-AS-IS / ACCEPT-WITH-AMENDMENTS selections are appended
to `SELECTED.md`. All killed ideas (Stages 2, 3, 4, 6) are appended to
`KILL_LIST.md` with citation and rationale.

---

## Stage 1 — parallel orientation-bound ideators

Owned by Agent A. See `STAGE1_IDEATOR.md` and `ORIENTATIONS.md`. The active
orientation set in any round is determined by `ROUND_OVER_ROUND.md`'s
rotation rule (KEEP / TIGHTEN / DROP / INTRODUCE), with a **diversity floor
of 3 active orientations**.

Stage 1 is the only stage where the v2 structure differs from v1 by
*spawning*. Stages 2–6 see the *union* of orientation outputs and are
single-agent (or per-advancer / per-track) as in v1.

The per-orientation falsifiable-grounding constraint applies: every
load-bearing claim of every Stage 1 idea must connect to either (A) a
measured number from `SUMMARY.md` (a CF1–CF12 finding) or (B) a primitive
on the actual stack. Mechanisms that float free of either anchor are
inadmissible.

CF13–CF15 (the in-flight Opus-pipeline measurements) are NOT type-A
anchors. Stage 1 ideators may cite them as motivation but may NOT use their
numerical values as load-bearing. See "Treatment of in-flight findings"
section below.

---

## Stage 2 — Judge (cross-orientation union scoring)

See `STAGE2_JUDGE.md`. Scores every Stage 1 proposal on 5 axes (residency
potential, frame novelty, falsifiability, mechanism specificity, pre-emption
resistance), each 0–3, max 15. Surfaces three classes of advancers:

- **Path 1 — top of union**: top 3–4 per track by total score.
- **Path 2 — convergence representatives**: when two or more orientations
  converge on the same load-bearing primitive, advance the strongest
  representative AND write a convergence note. Convergences advance even
  below the Path 1 score threshold.
- **Path 3 — frame-novelty bonus**: up to 2 ideas with A2 (frame novelty) =
  3 advance even below the Path 1 score threshold, provided A3 (smallest
  test) ≥ 1 and A4 (mechanism specificity) ≥ 1.

The frame-novelty bonus is the explicit anti-bias mechanism. The v1 Judge's
reflex was to score down unfamiliar frames; Path 3 reserves slots for them
provided they pass the math-correctness floor.

The hard rejections (regardless of total score) are the math-correctness
gates: CF9 frame-mismatch (A4 = 0), CF10 unmitigated calibration ill-
conditioning, kill-list re-treads, no-smallest-test, and SGD-required
proposals.

---

## Stage 3 — Deep Researcher (pressure-test)

See `STAGE3_DEEP_RESEARCHER.md`. Per advancer, one Sonnet researcher
produces ~1500–2500 words covering:
- Mechanism decomposition into 3–6 load-bearing claims
- Per-claim prior-art status (PRE-EMPTED / ADJACENT / NOVEL / PARTIAL with
  date-stamped citations)
- Frame-mismatch verification (CF9): for every imported theorem, verify
  the precondition holds for the target object
- Calibration ill-conditioning pre-flight (CF10): n_params, n_independent_
  samples, mitigation
- Residency arithmetic in detail
- Smallest-test sharpening (≤8h wall-clock; otherwise propose smaller
  smoke variant)
- Refined go/no-go thresholds (numerical, tied to CF numbers)
- Updated risk profile
- Two-paper composition flag (Opus pipeline import: name the closest
  cousin pair; articulate the value-add beyond the obvious composition)
- Verdict: REFINE / REGENERATE / KILL / DOWNGRADE

WebSearch usage is non-optional for Track A advancers; the publication-
velocity policy makes date-stamped novelty load-bearing.

---

## Stage 4 — Skeptic Explorer (cross-orientation gap-finder)

See `STAGE4_SKEPTIC_EXPLORER.md`. Reads the full state of the round (every
orientation's Stage 1 output, Stage 2's convergence map, every Stage 3
verdict, the cumulative kill list) and produces:

- **Frame-exhaustion map**: which frames are saturation-exhausted, CF9-
  exhausted, or CF10-exhausted. For each, name the stripped primitive (what
  remains after the dead machinery is removed).
- **Orientation-saturation diagnostics**: for each active orientation, did
  it advance? did it land in a convergence cluster? what should be appended
  to its anti-frame list? recommend KEEP / TIGHTEN / DROP / INTRODUCE.
- **Cross-pollination opportunities**: when an idea died in orientation A
  for a CF9 / CF10 / saturation reason, can a transposition into orientation
  B's pose rescue it?
- **Cross-domain seed exploration**: pick 2–3 professional priors (database
  engineer / FPGA designer / cryptographer / signal processor / control
  theorist / etc.) most likely to produce mechanisms whose preconditions
  match LLM-compression targets given the kill list.
- **Constraint-driven reframing**: pose the problem under alien constraints
  (no continuous numbers; content-addressable; 10 MB seed reconstruction;
  no FP at inference; no persistent state; sequential-only storage). Each
  produces 0–2 candidate ideas.
- **6–10 new ideas** in the gap, six-section format, track-classified,
  CF9- and CF10-prechecked.

The v2 skeptic operates on a *richer* substrate than v1 (multi-orientation
kill set + Stage 3 kills + cumulative kill list); the gap-finding is more
discriminating.

---

## Stage 5 — Selector (per track, one experiment)

See `STAGE5_SELECTOR.md`. Per track (Track A and Track B independently),
picks ONE experiment from the Stage 3 REFINE pool + Stage 4 gap ideas +
Stage 2 frame-novelty advancers.

Selection algorithm:
1. Re-verify novelty (publication-velocity policy — Track A's half-life
   is short enough that a paper that did not exist when Stage 3 ran 24
   hours ago might exist now).
2. Apply convergence multiplier (1.0 / 1.2 / 1.5 for solo / 2-orientation /
   3+-orientation convergence).
3. Apply frame-novelty bonus (0 / +1 / +2).
4. Apply NO-GO-finding bonus (+1 if the NO-GO outcome constrains a class
   of future ideas; +2 if it's a CF candidate).
5. Pick the maximum, tie-break on lower runtime then cleaner threshold.

Output is the Stage 5 plan with 13 sections, including the new
skeptic-controls declaration (section 9).

---

## Stage 6 — Red Team (adversarial review)

See `STAGE6_RED_TEAM.md`. Runs:
- Frame-mismatch re-check (CF9, second pair of eyes after Stage 3)
- Calibration ill-conditioning re-check (CF10, second pair of eyes)
- **Skeptic-controls check (NEW v2 gate)** — see "Skeptic-controls gate"
  section below
- Hidden prior-art search (third pass after Stages 3 and 5)
- Biased-framing audit
- Runtime / scope sanity
- CF13–CF15 verification check
- Verdict: ACCEPT-AS-IS / ACCEPT-WITH-AMENDMENTS / REJECT-PICK-RUNNER-UP /
  REJECT-AND-TRIGGER-ROUND-N+1

---

## Saturation as signal

When all advancing ideas in a round get pre-empted by recent papers (Track
A R2 in v1: CRANK→SmoothQuant, TRED→SimLens/ConfLayers, DLRA→arXiv:
2604.20682, RECS→RetNet/RADLADS), it's not failure — **it's a signal that
the search has reached a publishing-saturated region**. The closer the
pipeline drifts to those well-trodden grooves, the more the kill rate
climbs. **Genuinely novel space lives just past the saturation boundary.**

The v1 PIPELINE.md identified five anti-saturation tools. v2 implements all
five and adds a sixth:

1. **Anti-frame lists, not just kill lists**. Block whole flavors of
   mechanism, not just specific instances. v2: each orientation in
   `ORIENTATIONS.md` carries an anti-frame list that grows round-over-round
   per `ROUND_OVER_ROUND.md`.
2. **Frame-orthogonal generation directives**. v2: each orientation has a
   distinct cross-domain seed list. Stage 4 explicitly cross-domains beyond
   any orientation's seed list.
3. **Constraint-driven reframing**. v2: Stage 4 has a constraint enumeration
   step (no continuous numbers, content-addressable, 10 MB seed, etc.).
4. **Cross-engineer thinking**. v2: each orientation seeds a different
   professional prior; Stage 4 targets specific professional priors most
   likely to produce CF9-clean mechanisms.
5. **Stage 5 frame-novelty weighting**. v2: explicit numerical bonus in the
   selection algorithm.
6. **(NEW) Cross-orientation convergence as positive signal**. v2: Stage 2
   detects convergences; Stage 5 applies a convergence multiplier in the
   selection algorithm. See "Cross-orientation synthesis" below.

Mass pre-emption is the moment to *push harder*, not retreat. Ideas are
limitless; the bottleneck is steering Sonnet's prior past its training
distribution. Each round of pre-emption maps the boundary more precisely.

---

## Theoretical frame mismatch check (CF9)

When an ideator imports machinery from another field (cryptography, coding
theory, signal processing, statistical mechanics, etc.) into LLM
compression, Stage 2/3/6 must verify the imported theory's preconditions
hold for the target object. Three Round 6 v1 kills exemplified the failure
pattern:

- **CSKQ**: Count-Sketch's median-estimator error bound is `O(||E||_1 / w)`.
  Useful only when signal is sparse. INT4 quantization residuals are
  DENSE. Bound becomes vacuous (5470× signal magnitude in the empirical
  check).
- **RFSK**: Bochner's theorem approximates shift-invariant kernels. Weight
  matrices are not kernels. The "RFF" framing reduced to JL random
  projection, which is strictly worse than SVD; CLUBS already proved
  SVD-rank-k fails on Qwen3 W_up.
- **THWR**: Tabulation hashing's collision-resistance is useful when c-bit
  chunks have many possible values mapped to fewer hashed positions. With
  c=4 and 16 exhaustive entries, no collisions exist by definition. The
  "hash" reduces to a 16-entry codebook lookup — VPTQ territory.

**Check item for Stages 2/3/6**: before scoring or accepting an idea that
imports a named theorem or named algorithm, verify the structural
precondition. If the precondition (sparsity, shift-invariance, collision-
resistance, etc.) doesn't hold for the target, the imported machinery is
decorative. Strip it; what's left is usually well-known prior art under a
different name.

**Implication for ideators**: cross-domain frame seeding works (it generates
pre-empt-resistant ideas) BUT must preserve mathematical correctness.
Generate from frames whose preconditions structurally match the target —
e.g., for full-rank dense rectangular weight matrices, don't import
sparsity-assuming algorithms.

The CF9 gate fires at three points: Stage 2 A4 (mechanism specificity)
zero-grade rejections; Stage 3 section 3 explicit verification; Stage 6
section 1 second-pair-of-eyes re-check.

---

## Calibration ill-conditioning check (CF10)

When a Stage 5 / 6 selection involves fitting parameters via least-squares
on calibration data, the selector AND red-team MUST explicitly check:

- `n_params_to_fit` (e.g., for affine A: d_target × d_source + d_target)
- `n_independent_samples × n_output_dims_per_sample`

If these are within an order of magnitude, the fit is underdetermined and
will memorize calibration with arbitrary held-out behavior. WDLA failed
because A had 2.1M parameters fit on ~2M calibration values with ridge=1e-3.
The result was R²=1.0 on calibration, R²=-118000 on held-out.

Mitigations selectors should require:
- Calibration corpus N >> n_params_to_fit / d_output (typically 10K–100K
  tokens for 1.7B-class targets)
- Strong ridge (1e-1 or higher when underdetermined)
- Force low-rank by construction (e.g., A = U·V with U ∈ R^{d_t × r},
  V ∈ R^{r × d_s}, parameter count drops to (d_t + d_s) × r — well-
  conditioned at small calibration)
- Explicitly report calibration vs held-out R²; if they differ by > 0.2,
  the fit is overfitting

This check applies to: any affine/linear projection fit, any codebook
centroid fit, any low-rank decomposition with calibration residual fit, any
kernel-feature-map fit (e.g., CFPK).

The CF10 gate fires at Stage 3 section 4 (pre-flight), Stage 5 section 4
(smallest-test must report cal-vs-eval R²), Stage 6 section 2 (second pair
of eyes).

---

## Skeptic-controls gate (added 2026-05-09)

**Load-bearing v2 addition.** The user's "Skeptic controls first" feedback
is now a Stage 6 hard gate.

When a Stage 5 experiment plan's hypothesis claims any of:
- "X works on this stack"
- "X transfers from model A to model B"
- "X is consistent across layers / heads / strata / models"
- "X generalizes from calibration to held-out"

the experiment plan MUST include:

- **Permutation control**: run the same measurement with the load-bearing
  signal randomly permuted. The GO threshold must be met by the real
  signal AND substantially exceed the permuted signal. The real-vs-permuted
  gap is stated explicitly in the GO criterion.

- **Random-init control**: run the same measurement on a random-initialized
  model of the same architecture. The GO threshold must be met by the
  trained model AND substantially exceed the random-init baseline. This
  catches cases where the structure is architectural rather than trained.

- **Multi-seed**: at least 3 random seeds (where randomness applies). Mean
  ± std reported; the GO threshold must be met by the mean and the worst-
  of-3 must not be below NO-GO.

If one or more controls is genuinely not applicable (the experiment is a
structural-property measurement; or random init is uninformative; or the
experiment is fully deterministic), the Stage 5 plan documents WHY in the
skeptic-controls declaration. "Not applicable" is allowed; "not bothered"
is not.

Stage 6 enforces the gate. Missing controls without justification → REJECT-
PICK-RUNNER-UP or ACCEPT-WITH-AMENDMENTS that adds the control upfront.

The gate's purpose: catch false-positive structural findings. A
precision@K that beats random on the trained model might also beat random
on a random-init model; the "structural finding" would then be an
architectural artifact, not a trained-model property. The v1 ladder caught
these inconsistently. v2 makes the controls a hard gate at Stage 6.

---

## Strong preference signals (Round 1+ baseline)

**Signals, not gates.** Two orientation-spanning preferences shape every
round's generation and selection without being requirements for any
individual proposal.

**(a) Aggressive compression bias.** The pipeline pushes for harder-to-find
compression — not the easy quantize / sparsify / offload moves, but
mechanisms exploiting structure intrinsic to what the trained model is.
The bar is "structurally present but nobody happened to look there yet."
Stage 1 ideators are oriented toward intrinsic structure; Stage 2's
frame-novelty axis (A2) and Path 3 frame-novelty bonus reward this; Stage
4 cross-domain seeding biases toward priors that surface intrinsic
structure rather than generic compression.

**(b) Elegant computational equivalence.** Strong preference for solutions
to existing LLM processes that achieve the same result but are
computationally cheaper for an elegant reason. Archetype: AERO-style
activation removal — drop SiLU and W_gate algebraically folds into W_up,
same I/O behavior, fewer FLOPs and bytes. The class includes algebraic
identities, calibration-derived restructurings, mathematical equivalences,
no-SGD reformulations, gauge / subspace / conserved-quantity exploitation.

The elegant-equivalence preference fires at four points in the cascade:
- **Stage 2 Path 4**: up to 1 additional advancer slot for an
  elegant-equivalence representative (small +1 weight on A2 / A4 when the
  elegance is fresh / constructive).
- **Stage 3 elegance-class tagging**: NOVEL claims in the elegant-
  equivalence class get a sub-class tag (algebraic-identity /
  calibration-fit / gauge-exploitation / subspace-alignment /
  conserved-quantity / no-SGD-reformulation).
- **Stage 4 elegant-equivalence prompt**: cross-domain seeding asks "what
  algebraic identity / conservation law / gauge symmetry does this
  professional prior reveal?" and tags generated ideas accordingly.
- **Stage 5 elegant-equivalence multiplier**: 1.1–1.3 multiplier on tagged
  candidates, capped so it cannot single-handedly flip a structurally
  weak proposal to selection.

These are signals, not gates. The math-correctness gates (CF9, CF10,
skeptic-controls) are unchanged. Elegance cannot paper over a math error;
Stage 6 section 6c rejects an elegant-equivalence claim whose elegance
depends on a precondition that fails the frame-mismatch check.

---

## Wildcard slot — `[FREE SWING]` and `[OUT-OF-ORIENTATION]`

**Creative freedom preserved.** The strong-preference signals above are
signals, not gates. Ideators retain the freedom to surface ideas that
have nothing to do with prior CFs, prior orientations, or the
elegant-equivalence class. Two markers grant downstream protection:

- **`[FREE SWING]`**: an ideator's per-round wildcard slot. Stage 1
  ideators may tag up to 1 proposal per orientation as `[FREE SWING]`;
  Stage 4 may tag up to 2 of its 6–10 generated ideas as `[FREE SWING]`.
  These are proposals that connect to no CF1–CF12 finding and to no
  surfaced cross-orientation primitive.
- **`[OUT-OF-ORIENTATION]`**: an idea an ideator surfaces outside its
  assigned creative pose. Used when the orientation's frame turns out
  not to be the natural home for the idea but the idea is worth
  surfacing anyway.

**Protection these markers grant:**
- **Stage 2** (`STAGE2_JUDGE.md`): not scored down for lacking a CF
  tether on A2 or A5; not penalized on Path 1 / 3 / 4 advancement for
  wildcard status. Verdict tag: `ADVANCE-wildcard`.
- **Stage 3** (`STAGE3_DEEP_RESEARCHER.md`): admissible if it connects
  to a primitive on the actual stack (CPU / RAM / NVMe / trained-weights
  / standard-library / known-math-op); CF-tether requirement suspended.
- **Stage 4** (`STAGE4_SKEPTIC_EXPLORER.md`): may generate up to 2
  `[FREE SWING]` proposals among its 6–10 ideas; these inherit the same
  downstream protection.
- **Stage 5** (`STAGE5_SELECTOR.md`): no CF-tether-deficit penalty in
  the selection algorithm; competes on equal footing.
- **Stage 6** (`STAGE6_RED_TEAM.md`): not rejected for lacking a CF
  tether or for being unfamiliar relative to the standard ML playbook.

**What the markers do NOT relax:**
- The math-correctness floor (CF9 frame-mismatch, CF10 calibration
  ill-conditioning).
- The smallest-test ≤ 8h gate.
- The skeptic-controls gate for transfer / consistency claims.
- The connect-to-a-primitive-on-the-stack requirement (a wildcard that
  floats free of the actual stack is rejected).
- The kill-list re-tread check, the no-SGD constraint.

Wildcards die or survive on **structural grounds only**. The CF tether
is the only requirement that suspends.

Future round briefs (`ROUND_OVER_ROUND.md`, per-stage briefs) should
reference this section by name when issuing wildcard guidance to
ideators. The protection is cascade-wide; downstream stages must honor it.

---

## Cross-orientation synthesis

**Load-bearing v2 addition.** When two or more orientations independently
arrive at the same load-bearing primitive (different vocabulary, same
mathematical structure), the convergence is the strongest single signal in
the pipeline. The v1 ladder noted convergence-as-signal anecdotally
(Round 3's Idea G ≈ Round 1's deferred Idea 5); v2 surfaces convergence as
a first-class output of Stage 2 and weights it explicitly in Stage 5's
selection.

**Elegant-equivalence convergences are particularly strong signals.**
When two or more orientations independently arrive at the same elegant-
equivalence-class mechanism (different priors, same algebraic identity /
gauge symmetry / conserved-quantity / no-SGD reformulation), the
convergence multiplier and the elegant-equivalence multiplier stack
multiplicatively at Stage 5 (e.g., 1.5 × 1.2 = 1.8). This is the rare
case where the empirical structure asserts itself across vocabularies AND
the asserted primitive is computationally cheaper for a structural reason
— the loudest signal the v2 cascade can produce.

The convergence-detection mechanism is imported from the Opus pipeline's
Stage 4 synthesis. In Opus R1, four orientations converged on:
- C1: outlier-channel identity as routing primitive (Composition ∩
  Unconventional ∩ Reach)
- C2: storage-layout-IS-the-predictor (Unconventional ∩ Reach)
- C3: joint structure across "different" matrices (First-Principles ∩
  Composition)
- C4: per-token routing as the unifying inference-time primitive (all four)

v2 Stage 2's convergence map produces analogous output in vocabulary-neutral
terms ("the load-bearing primitive across orientations A and B is the
outlier-channel pattern at layer entry"). v2 Stage 5's selection algorithm
applies a convergence multiplier:
- 1.0 for solo
- 1.2 for 2-orientation convergence
- 1.5 for 3+-orientation convergence

A 3+-orientation convergence is the loudest signal v2 can produce. When it
fires, the pipeline's empirical structure is independently asserting itself
across vocabularies; the strongest representative of the cluster gets
priority over higher-raw-score solos.

---

## Treatment of in-flight findings (CF13–CF15)

The v1 ladder produced findings CF1–CF12 through end-to-end Sonnet-ladder
experiments with user-confirmed results. CF13 (Windows mmap page-fault
profile + NEXTENT prefetch validation), CF14 (massive-activation channel
dominance), and CF15 (stratified residual-stream intrinsic dimension) were
produced by the in-flight Opus pipeline and are NOT yet user-confirmed in
the v2 substrate.

**v2 ladder treatment**:

- Stage 1 ideators may CITE CF13–CF15 as motivation but may NOT treat their
  numerical values as load-bearing.
- Any Stage 1 idea whose feasibility math uses a CF13/14/15 number must
  include "verify CF13/14/15 independently on this stack" as an explicit
  precondition with its own go/no-go in the smallest-experiment section.
- Stage 3 residency arithmetic includes a "what if CF13–CF15 don't
  replicate" branch — the conservative version of the math.
- Stage 5 plans that cite CF13/14/15 in load-bearing roles must include a
  re-derivation rung as the FIRST step of the smallest test.
- Stage 6 verifies that the re-derivation rung is in the plan; if absent,
  REJECT.

If a v2 round runs an experiment that re-derives one of CF13–CF15 and the
result is confirmed, it is appended to the v2 `SUMMARY.md` as a v2-numbered
finding (CF13', CF14', CF15') with explicit citation of the v2 round that
produced it. This keeps the v2 empirical map self-consistent and prevents
cross-contamination from unverified parallel-pipeline outputs.

---

## Track A publication-velocity policy

The arch-transpose space publishes faster than the kill list can be
refreshed. Mamba (Dec 2023), Performer, RADLADS (May 2025), LOLCATS, CMoE
(Feb 2025), AERO (Oct 2024), ProcrustesGPT (June 2025), LLVQ (Mar 2026)
and many KV-side methods all landed within recent windows. Track A stages
must:

1. **Re-check novelty at every stage** (not just Stage 3). Stage 5 must
   re-verify the top picks before commitment; Stage 6 runs a third pass.
2. **Date-stamp claims**. "No paper found doing X *as of <date>*" rather
   than absolute novelty.
3. **Prefer structural/algebraic novelty** over specific-architecture
   imitation. A math-style idea has longer half-life than "calibration-only
   Mamba-from-FFN" because the latter gets pre-empted the moment a no-SGD
   variant of the named architecture publishes.
4. **Frame contributions as operating windows**. We're after empirical
   demonstrations on this hardware tier with specific structural-finding
   constraints, not pure first-mover credit.
5. **Favor pre-empt-resistant ideas**. Ideas tied to empirical statistics
   on a specific model family are harder to pre-empt than clean
   architectural ideas because the empirical characterization is the moat.

Track B publication velocity is also high but the field is more saturated;
novelty claims there have to navigate ~30 known competitors. Track B's risk
is "subsumed by published method" rather than "pre-empted by next week."

---

## Two parallel tracks

The pipeline runs two parallel tracks after the v1 fork at Round 3:

- **Track A — `arch-transpose`**: operator-structure changes (FFN→MoE,
  softmax→linear attention, AERO-style activation-removal+folding,
  polynomial nonlinearity replacement, SSM/Mamba substitution, tensor
  reshapes that change computation). No SGD; calibration-driven fitting OK.
- **Track B — `compression`**: weights compress (quantization, low-rank,
  codebook, sparsity, paging) but the operator structure stays (still
  SwiGLU, still softmax attention).

Classification rule for ideators: if the *computation graph at inference*
changes shape (different ops, different routing), it's Track A. If only
the *bytes-per-weight* changes, it's Track B.

Both tracks share the cumulative `KILL_LIST.md` and `SUMMARY.md`. Each
track maintains its own per-round transcript and its own SELECTED queue:

- `track_A_arch/SELECTED.md`, `track_A_arch/ROUND_<N>_TRANSCRIPT.md`
- `track_B_compress/SELECTED.md`, `track_B_compress/ROUND_<N>_TRANSCRIPT.md`

Tracks run concurrently. After both tracks pick a Round-N experiment, the
cheaper / more informative one runs first; the other queues.

---

## Round-over-round behavior

- Round N+1's Stage 1 receives the cumulative kill list. Cannot re-propose
  any killed idea.
- Round N+1's Stage 1 receives explicit framing-diversity instructions
  ("previous rounds converged on classes X, Y, Z; generate from
  underexplored frames {...}").
- Each orientation's anti-frame list grows monotonically per
  `ROUND_OVER_ROUND.md` rules (NO-GO at Stage 5; Stage 3 kill; mass
  pre-emption at Stage 2).
- The orientation set itself rotates per the KEEP / TIGHTEN / DROP /
  INTRODUCE rule, with a diversity floor of 3 active orientations.
- The kill list grows monotonically.

The v2 ladder treats orientation rotation as a small evolutionary system:
the rotation rule is selection pressure; anti-frame growth is mutation
pressure; cross-orientation convergence is the fitness signal; the
diversity floor is the stop-gap that prevents collapse.

---

## Convergence / stop conditions

Stop when one of these holds:

1. Stage 1 of Round N produces ≥3 ideas already on the kill list across the
   union of orientations (search has saturated within Sonnet's prior even
   with multi-orientation parallelism). v2 threshold is the same as v1
   despite the larger candidate set — the orientations are supposed to be
   diverse enough that 3 collisions across the union signals genuine
   exhaustion.
2. Stage 2 scores all of Round N's ideas ≤ 11/15 across the union (no
   convergence advancers, no frame-novelty advancers, no Path 1 advancers).
3. Stage 5 of Round N selects an experiment ≥80% mechanistically
   overlapping a prior-round selection.
4. The user manually stops after a particular SELECTED experiment looks
   compelling enough to commit to.

When stop condition 1 fires, the v2 ladder's first response is to introduce
a round-specific orientation aimed at the under-explored frame the round
transcript identified — not to declare saturation. If the introduced
orientation also fails to escape, the run actually saturates and the loop
should stop or escalate to a different model family.

---

## Diversity / bias-handling

The pipeline uses Claude Sonnet for all stages. This means the search
inherits Sonnet's training-data biases — there are blindspots no amount of
rounds will surface. To mitigate:

- v2 Stage 1 spawns multiple orientations, each with a distinct anti-frame
  list, so no single agent's bias dominates.
- Stage 2's frame-novelty bonus (Path 3) explicitly reserves slots for
  unfamiliar frames.
- Stage 4 explicitly enumerates frames the previous stages didn't use AND
  cross-pollinates orientations for stripped primitives.
- Stage 6 red-team is asked to look for biased framing in Stage 5.
- Round N+1's orientation set evolves per `ROUND_OVER_ROUND.md`.

For genuine diversity beyond Sonnet's prior, manually inject:
- A different model family (Opus or external API) every K rounds
- A user-written prompt that names an angle Sonnet wouldn't naturally pick

---

## Idea-class taxonomy (for kill-list classification)

Carried forward verbatim from v1 PIPELINE.md. Each killed/selected idea is
tagged with one or more classes so future ideators can see which classes
are exhausted vs untouched.

- `compression-quant` — quantization, low-bit codecs, codebooks
- `compression-lr` — low-rank decomposition, basis sharing, SVD variants
- `compression-entropy` — Huffman, ANS, arithmetic coding, dictionary
- `compression-tensor` — tensor train / ring / Tucker / hierarchical
- `sparsity-static` — magnitude pruning, structured pruning
- `sparsity-dynamic` — contextual sparsity, hot/cold neuron prediction
- `offload-disk` — NVMe/flash residency, page swapping
- `offload-tier` — DRAM/cache/disk tiering
- `prediction-shadow` — small-model predicts large-model behavior
- `prediction-historical` — frequency tables, count-min sketches
- `prediction-internal` — using model's own activations to predict its own
  state
- `runtime-fused` — decode-fused-with-compute kernels
- `systems-fs` — filesystem-level tricks (CoW, dedup, reflink)
- `systems-os` — OS paging, mmap, prefetch hints
- `kv-side` — KV cache focused (compress KV to free RAM for weights)
- `arch-transpose` — architectural redesign achievable via no-SGD
  transposition from trained weights (in scope as of 2026-05-08)
- `train-required` — anything requiring SGD/fine-tuning (out of scope)

---

## Constraint clarification — what "no retraining" admits

The "no retraining" constraint admits architectural transposition. What is
forbidden is SGD against a loss function. What is permitted:

- **Mathematical equivalences** that produce the same input/output behavior
  with a different computation graph (e.g., Performer's random-feature
  approximation of softmax attention, exact in expectation)
- **Calibration-derived restructurings** that fit an alternative operator
  to the trained model's behavior on calibration data using closed-form or
  non-gradient methods (k-means clustering, NMF, generalized eigenvalue,
  polynomial fitting, etc.)
- **Algorithmic reformulations** that replace one operator with another
  whose output matches on the relevant input distribution (silu → degree-3
  polynomial fit on the active range; dense attention → structured-sparse
  pattern derived from calibration; FFN → MoE via expert-clustering with a
  calibration-fit router)
- **Tensor / weight reshapes** that produce algebraically equivalent (or
  provably bounded-error) computations: Kronecker, Tucker, hierarchical
  Tucker, factor-graph reformulations

The constraint is "the trained Qwen3-N weights can be transposed into the
new shape via a no-SGD procedure." This widens the search beyond pure
compression / sparsity / offload ideas. Track A is the explicit space for
arch-transposition.

---

## Output files (per the v2 substrate)

- `KILL_LIST.md` — cumulative ledger of killed ideas. One-line per kill:
  name, round, stage that killed, citation/reason, idea class. v2 uses
  the same format as v1.
- `SELECTED.md` — queue of selected experiments with status (queued /
  running / done) and result. v2 uses the v1 format with the addition of
  the skeptic-controls declaration as a section.
- `ROUND_<N>_TRANSCRIPT.md` — per-round summary of all 6 stages (concise,
  not full agent output).
- `SUMMARY.md` — empirical structural findings (CF1–CF12 confirmed; new
  v2 findings appended as confirmed).
- `ORIENTATIONS.md` — orientation registry (5 default + per-round custom).
- `ROUND_OVER_ROUND.md` — orientation rotation rules.
- `STAGE1_IDEATOR.md` — Stage 1 prompt (Agent A).
- `STAGE2_JUDGE.md` through `STAGE6_RED_TEAM.md` — stage prompts.
- This `PIPELINE.md` — the runbook.

---

## Portability notes

The pipeline is generalizable beyond this specific use case. To port to a
different research domain:

1. **Replace the problem statement** at the top of `SUMMARY.md` and in
   each Stage 1 brief. The framing must specify: hardware/resource
   constraints, hard constraints (e.g., "no retraining"), success
   criteria.
2. **Replace the published landscape list** in Stage 1's brief — the kill
   list and competitor enumeration are domain-specific.
3. **Define the idea-class taxonomy** for the new domain. Track A/B split
   is optional; single-track works for narrower problems.
4. **Re-tune the orientation set**. The R/C/F/U/A defaults work for
   compression-shaped problems with high empirical density; other domains
   may need different orientations (e.g., for software security: Threat /
   Counterexample / Implementation-aware / Composition / Constraint-Alien).
5. **The 6-stage structure is invariant**: Ideator → Judge → Deep
   Researcher → Skeptic Explorer → Selector → Red Team. Each stage's role
   is domain-agnostic.
6. **The math-correctness gates (CF9, CF10, skeptic-controls) are
   invariant.** Every domain has imported-theorem-precondition mismatches
   and calibration-overfit failures and false-positive transfer claims.
   Keep these gates.
7. **The cumulative kill list, selected queue, and orientation registry
   persist across rounds** regardless of domain. The format generalizes.

The pipeline is a search algorithm where the LLM is both the generator and
the evaluator. Its convergence properties depend on:
- Diversity of independent samples (parallel orientations per round)
- Memory between rounds (the kill list as immutable knowledge)
- Adversarial review (red-teaming as a math-correctness gate)
- Empirical grounding (every selected experiment must produce numbers)
- Skeptic discipline (controls on every transfer / consistency claim)

These are all preserved in the portable version.

---

## Why v2 is structured this way (the design rationale)

A short closing argument so the next person modifying this runbook
understands what's load-bearing:

1. **Parallel orientations at Stage 1 are the diversity primitive.** Five
   independent Sonnet draws under different creative poses produce more
   coverage than one Sonnet draw with a long anti-frame list. The Opus
   pipeline's R1 evidence is the basis.

2. **Stage 2's job is harder in v2.** It scores across orientations and
   surfaces convergences. The frame-novelty bonus (Path 3) is the explicit
   anti-bias mechanism — without it, Stage 2's training-distribution
   reflex would kill the audacious-but-grounded proposals the orientations
   produce.

3. **Stage 3's per-claim status taxonomy and two-paper composition flag
   are imported from the Opus pipeline.** Both make the prior-art pass
   sharper without changing the Stage 3 role.

4. **Stage 4 cross-pollinates orientations.** A primitive killed in
   orientation A may be alive in orientation B. v2 makes this an explicit
   step; v1 left it implicit.

5. **Stage 5's selection algorithm has explicit weights.** Convergence
   multiplier + frame-novelty bonus + NO-GO-finding bonus are the three
   levers. v1 used these implicitly; v2 makes them numerical.

6. **Stage 6 keeps the v1 math-correctness gates AND adds skeptic-controls.**
   The user's "Skeptic controls first" feedback is now a hard gate. The
   v1 gates (CF9, CF10) are preserved at full strength.

7. **CF13–CF15 are quarantined.** The v2 substrate is self-consistent
   from CF1–CF12 only. In-flight findings from the parallel pipeline are
   re-derived before being relied on.

The v2 ladder's intended performance: more ideas at higher quality floor,
same rigor on math correctness, additional rigor on transfer/consistency
claims, faster cycle through the kill list (because frame-novelty advancers
explore beyond Sonnet's bias) without sacrificing the structural-finding
yield.
