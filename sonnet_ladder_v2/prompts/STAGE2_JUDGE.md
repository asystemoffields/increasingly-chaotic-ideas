# Stage 2 — Judge (Cross-Orientation Union Scoring + Convergence Detection)

One Sonnet judge agent reads the **union of all orientation outputs** from Stage 1
(typically 4–5 orientations × 4–8 proposals each = 16–40 proposals) and scores
them on the same rubric. Cross-orientation comparison happens HERE, before any
prior-art deepening — which means convergence detection and frame-novelty
weighting both execute upstream of any literature-shaped narrowing.

This is a deliberate structural change vs the v1 ladder. v1 ran one Stage 1
ideator producing ideas in a single pose; the judge ranked a homogeneous set.
v2 produces a heterogeneous union, and the judge's hardest job is **scoring
across orientations without flattening their distinctive value** — a Reach
proposal aimed at a 13× ceiling is not the same shape of object as a Composition
proposal expressing a single-equation coupling, and a uniform rubric must give
each its due.

The v1 Judge's known failure mode is the standard-ML-playbook bias: when a
proposal arrives in unfamiliar vocabulary (sheaves, NTFS extents, coding
theory), Sonnet scores it down because it cannot easily place it in the training
distribution. v2 corrects this with an explicit **frame-novelty bonus** and a
**convergence bonus** that override raw rubric scores when triggered.

---

## Inputs

1. The full Stage 1 output of every active orientation. Each proposal is in
   the six-section form (Ambition / Mechanism / Why our findings make this
   work / What would have to be true / Experiment cascade / What you're NOT
   proposing) and tagged with its orientation letter (R, C, F, U, A, or a
   round-specific custom letter) and its track (A or B).
2. `ORIENTATIONS.md` — so you understand what creative pose each orientation
   was supposed to take, and what its anti-frame list explicitly forbids. Use
   the anti-frame lists to detect orientation drift (a Reach proposal that
   smells like Composition is a flag).
3. `SUMMARY.md` — confirmed empirical findings CF1..CF12. CF13–CF15 are
   in-flight; treat them as motivation only, not as ground truth.
4. Cumulative `KILL_LIST.md` — every previously killed idea. You may not
   advance any proposal that is a re-tread.
5. The round-specific saturated-frame list from `ROUND_OVER_ROUND.md` (named
   flavors that hit publishing-saturation in recent rounds).
6. The track classification each proposal self-declared.

---

## What this stage IS and IS NOT

**IS:**
- A union-set scoring pass on 5 axes, 0–3 each, max 15.
- A cross-orientation convergence detector. Two orientations independently
  arriving at the same load-bearing primitive (different vocabulary, same math)
  is the strongest single signal in the pipeline.
- A frame-mismatch hard gate (CF9). An idea that imports a named theorem or
  algorithm whose preconditions don't hold for the target object is scored down
  to zero on Mechanism Specificity, regardless of other axes.
- A first pass — Stage 3 (Deep Researcher) does the deep prior-art and
  residency arithmetic. You triage; Stage 3 pressure-tests.

**IS NOT:**
- Not a prior-art literature search. Cite obvious overlaps (Deja Vu for
  cross-layer activation prediction; LLM.int8 for outlier-channel claims;
  AERO for activation-removed FFN folding) but do not chase deep citations —
  Stage 3 owns that.
- Not the place to demand new experiments. You score what was proposed.
- Not a per-orientation tournament. There is no per-orientation cap; advance
  the strongest ideas of the union plus convergences plus frame-novelty
  bonuses.

---

## The 5 scoring axes (0–3 each, max 15)

For each idea, score:

### A1. Residency potential (0–3)
The honest answer to: *if this idea works exactly as proposed, how much closer
does it move us to 70B-class on 7.28 GiB at usable tok/s?*

- **0**: no residency or speed effect, or movement <5%.
- **1**: 1.2–1.5× residency reduction or 1.2–1.5× tok/s on the target tier.
- **2**: 1.5–3× on the relevant axis, or unlocks a previously-blocked tier
  (e.g., enables NVMe-streamed 70B at usable speed).
- **3**: ≥3× on the relevant axis, or directly delivers the audacity floor
  (70B-class on 7.28 GiB at ≥3 tok/s within ~0.3 nat).

Reach proposals will live at 2–3 here by construction. First-Principles
proposals may score lower on raw residency but score high on A2/A3 via
unifying-object claims.

### A2. Frame novelty (0–3)
The dimension v1 was structurally weak on. Score the *frame*, not the
mechanism: a scrappy idea in a frame the literature has not chewed has longer
half-life than a polished idea in a saturated frame.

- **0**: standard ML compression playbook frame (incremental quantization,
  generic low-rank, generic sparsity, generic offload). Even if the specific
  mechanism is novel, the *frame* is saturated.
- **1**: in-vocabulary but with a non-obvious twist (e.g., LLM.int8-shaped
  claim with a per-token Jaccard measurement instead of a hand-wave).
- **2**: imports a frame from another field whose preconditions plausibly
  apply (database engineering, OS paging, signal processing on an actual
  shift-invariant target, control theory on a control-shaped object). Frame
  is unrepresented in recent ML compression literature.
- **3**: genuinely fresh frame whose analog has not been written for LLM
  compression — e.g., "NTFS extent layout as routing primitive,"
  "tensor-network bond dimension as KV redundancy bound," "content-addressable
  weight storage from a 10 MB seed." If the round-specific saturated-frame
  list does not contain a named cousin, you are at 3.

**Anti-bias note.** When you don't recognize the frame, the v1 failure mode
was to score it down. Reverse the prior here: a frame you cannot place is more
likely to be 2–3 than 0–1, *provided* it passes the A4 (mechanism specificity)
gate. Unrecognizable + specific math = high frame novelty. Unrecognizable +
hand-waved math = A4 zero, no advancement.

### A3. Falsifiability and decisive-experiment quality (0–3)
Can a single experiment in ≤8 hours produce a number that decides go/no-go?

- **0**: no smallest-test specified, or the smallest test is "implement the
  whole thing first then measure." Reject for advancement regardless of other
  axes.
- **1**: smallest test exists but is large (≥1 day) or the go/no-go threshold
  is qualitative ("looks better").
- **2**: ≤8h experiment with a numerical go/no-go; the experiment produces a
  structural finding even on NO-GO (the proposal carries information whichever
  way the bit lands).
- **3**: clean binary settler in ≤4h with a quantitative threshold tied to a
  CF1–CF12 number, AND the NO-GO outcome itself constrains a class of future
  ideas (a class-level kill).

Examples of 3-grade settlers from the v1 ladder: the W_gate rank sweep
(CF4), the AQFKV per-head vs global disambiguation (CF11), the LHQD
embed/lm_head spectrum (CF12). Each was binary, fast, and on NO-GO would
still have produced a load-bearing class boundary.

### A4. Mechanism specificity and frame-mismatch survival (0–3)
This axis carries the CF9 gate. A proposal that imports a named theorem,
algorithm, or technique must verify the theorem's structural precondition for
the target object before claiming the mechanism applies.

- **0**: hard fail on CF9. The mechanism imports a theorem whose precondition
  does not hold for the target. Strip the imported machinery and what's left
  is well-known prior art under a different name. Examples: Count-Sketch on
  dense INT4 residuals (sparsity precondition fails), Bochner's theorem on
  weight matrices (shift-invariance precondition fails), tabulation hashing
  on c=4 with 16 entries (collision-resistance precondition vacuous). **Score
  zero on A4 sets the proposal to non-advancement regardless of total.**
- **1**: mechanism is specified but with hand-waved transitions; one or more
  steps are gestured rather than derived. Could be sharpened by Stage 3 but
  is not yet a checkable claim.
- **2**: mechanism is checkable — every load-bearing step is a one-line
  calculation or an explicit dependency on a measured number from CF1–CF12.
  Imported theorems cite their precondition and verify it for the target
  object.
- **3**: the mechanism is *constructive* and the construction is auditable
  in 1–2 paragraphs. A reader can simulate the calculation in their head or
  in 10 lines of NumPy.

Important: A4 is the math-correctness gate that the v1 ladder's red team
caught CSMF (polynomial-substitution-eliminates-W_gate), LFAO (RMSNorm-as-
constant), and CRMS (stateless-FFN-fitted-to-recurrent-SSM) on. v2 catches
those at Stage 2 by gating advancement on A4 ≥ 1, and zero-A4 proposals do
not advance even if A1 = 3.

### A5. Pre-emption resistance (0–3)
How likely is this idea to be subsumed by a paper published this quarter?

- **0**: a recent paper (≤24 months) does substantively the same thing on
  similar models. Cite it explicitly. Examples from the v1 kill list: CRANK
  → SmoothQuant; TRED → SimLens/ConfLayers; DLRA → arXiv:2604.20682.
- **1**: adjacent published work exists; the proposal could be sharpened into
  a non-trivial differentiation but is currently a re-derivation.
- **2**: no direct prior found; the proposal's load-bearing claim is a
  structural-finding-driven moat (e.g., "fold gates that calibration shows
  are within ε of constant on Qwen3 family" — the empirical characterization
  is the moat).
- **3**: pre-emption-resistant by construction. Either (a) the load-bearing
  number is a private empirical finding from CF1–CF12 not on arXiv, or
  (b) the proposal is a constructive derivation whose theorem is unstated in
  the literature, or (c) the proposal couples two findings whose joint use
  has no published cousin.

Date-stamp every "no paper found" claim. The Track A publication-velocity
policy demands "no paper found doing X *as of <date>*" rather than absolute
novelty.

---

## Score → advancement mapping

After all proposals are scored, advancement proceeds as follows.

### Path 1 — top-of-union by total score
Advance the **top 3–4 ideas per track** by total score (Track A separately
from Track B), subject to:
- A4 ≥ 1 (no frame-mismatch killers advance, no exceptions)
- A3 ≥ 1 (must have a smallest test; "implement-first" proposals do not
  advance)
- Total ≥ 9/15

### Path 2 — convergence bonus advancers (additional slots)
A **convergence cluster** is two or more proposals from different orientations
that share a load-bearing primitive. The primitive can wear different
vocabulary across orientations — that's the point. You are detecting
*structural* convergence, not literal text overlap.

What counts as a load-bearing convergence:
- Same mathematical object (e.g., "outlier channel pattern at layer entry as
  routing key" appearing in Composition as alignment claim, in Reach as
  predictor input, in Unconventional as storage-layout key).
- Same dependency (e.g., two proposals both gated by the same fresh
  measurement — the measurement is now a high-leverage pre-program target).
- Same constraint exploited (e.g., two proposals leveraging CF11's W_Q
  head-redundancy from different angles).

For each cluster you identify, advance **the strongest single representative**
(highest A1+A2+A3+A4 sum across cluster members) PLUS write a one-paragraph
**convergence note** that names the shared primitive in primitive-neutral
vocabulary and cites which orientations contributed. This note carries forward
to Stages 4 and 5.

A convergence representative advances even if its raw total is below the Path 1
threshold (provided A3 ≥ 1 and A4 ≥ 1). The convergence is the signal.

### Path 3 — frame-novelty bonus advancers (up to 2 additional slots)
Up to **2** ideas with A2 = 3 advance even if their total score is below the
Path 1 threshold, provided A3 ≥ 1 and A4 ≥ 1. This is the explicit anti-bias
mechanism: the v1 Judge's reflex is to drop unfamiliar frames, and Path 3
disables that reflex by reserving slots for them.

If more than 2 ideas have A2 = 3, pick the two with the strongest A4 — frame
novelty AND specificity matter together; an unfamiliar frame with hand-waved
math fails A4 and is not eligible.

### Path 4 — elegant-equivalence representative (up to 1 additional slot)
A positive signal, not a gate. Up to **1** idea whose mechanism class is
"computationally cheaper for an elegant reason" (algebraic identity,
calibration-derived restructuring, mathematical equivalence, no-SGD
reformulation, gauge / subspace / conserved-quantity exploitation) advances
even if its raw total is below the Path 1 threshold, provided A3 ≥ 1 and
A4 ≥ 2 (the elegance must be auditable, not gestured). Archetype: AERO-style
activation removal — drop SiLU and W_gate algebraically folds into W_up;
same I/O behavior, fewer FLOPs and bytes.

When scoring, if the proposal's mechanism is in this class, add a small
positive weight: **+1 on A2 (frame novelty)** if the elegance is fresh
relative to the published landscape, **+1 on A4 (mechanism specificity)**
if the elegance is constructive (auditable in 1–2 paragraphs). The weight
is preferred, not required; it does not flip a structurally weak proposal
to advancement, and the math-correctness floor (A4 ≥ 1, CF9, CF10) is
unchanged.

In your Stage 2 transcript, tag the elegant-equivalence advancer with its
sub-class: `algebraic-identity` / `calibration-fit` / `gauge-exploitation` /
`subspace-alignment` / `conserved-quantity` / `no-SGD-reformulation`. Stage 3
carries this tag forward; Stage 5 uses it for the elegant-equivalence
multiplier.

### Wildcard non-penalization (`[FREE SWING]` / `[OUT-OF-ORIENTATION]`)

A proposal tagged `[FREE SWING]` (Stage 1's per-ideator wildcard slot) or
`[OUT-OF-ORIENTATION]` (an idea the ideator surfaced outside its assigned
creative pose) is **not scored down** for lacking a CF tether or for missing
the elegant-equivalence preference signal. The CF-connection requirement
that applies to standard proposals is suspended for these flagged proposals.

Wildcards still die or survive on **structural grounds only**:
- A4 ≥ 1 (math is checkable, no broken theorem imports — CF9 still applies)
- A3 ≥ 1 (a smallest test exists)
- The mechanism connects to a primitive on the actual stack (CPU / RAM /
  NVMe / trained-weights / standard-library / known-math-op)
- No kill-list re-tread, no SGD requirement, no unmitigated CF10

If the wildcard meets the structural floor, it competes for advancement on
equal footing under Paths 1, 3, or 4. Do not penalize A2 (frame novelty) or
A5 (pre-emption resistance) downward simply because the proposal does not
ground in a CF — for wildcards, "no CF tether" is by design, not a defect.

In the rejection rationale or per-advancer rationale, note the wildcard
status explicitly: "advanced as `[FREE SWING]` — structural floor met,
CF-tether requirement suspended."

### Hard rejections (do not advance regardless of total)
- A4 = 0 (frame-mismatch failure on imported theorem). **Always note this in
  the rejection rationale and propose a Stage 4 reframe target** ("the
  surviving primitive after stripping the imported machinery is X; Stage 4
  may want to attack X from an orientation that doesn't import the broken
  precondition").
- A3 = 0 (no smallest-test or "implement-first" smallest-test).
- Any direct kill-list re-tread (cite the kill-list line).
- Any proposal that requires SGD or fine-tuning (out-of-scope per the no-
  retraining constraint).
- Calibration-fitted proposals where n_params is within an order of magnitude
  of n_independent_samples × d_output and ridge / low-rank-by-construction
  is not specified (CF10 enforcement). Note the failure and pass the
  primitive's surviving structure to Stage 4.

---

## What you write — the Stage 2 transcript

Markdown report, 800–1500 words, with these sections.

### 1. Union score table

A single table with columns:
`ID | Orientation | Track | Title | A1 | A2 | A3 | A4 | A5 | Total | Verdict`

Where Verdict is one of: `ADVANCE`, `ADVANCE-convergence`, `ADVANCE-frame-novelty`,
`ADVANCE-elegant-equivalence`, `ADVANCE-wildcard`, `REJECT-CF9`, `REJECT-CF10`,
`REJECT-killlist`, `REJECT-low-score`, `REJECT-no-smallest-test`.

`ADVANCE-wildcard` indicates a `[FREE SWING]` or `[OUT-OF-ORIENTATION]`
proposal that met the structural floor and competes under Path 1/3/4 with the
CF-tether requirement suspended. `ADVANCE-elegant-equivalence` indicates a
Path 4 advancer; tag it with its elegance sub-class.

### 2. Per-advancer rationale (1 paragraph each)
For every advancing idea: which axes carried it, which CF findings ground it,
what the smallest test is in one line, what the load-bearing primitive is.

### 3. Convergence map
For each detected convergence cluster: name the shared primitive in
orientation-neutral vocabulary, cite the orientations that contributed,
identify the strongest representative, name the cheapest measurement that
would settle whether the convergence is real (this often becomes a Stage 5
pre-program experiment).

Example shape (illustrative, modeled on Opus R1 synthesis):
> **Convergence C1 — outlier-channel identity as routing primitive.**
> Composition X1 (M2 alignment claim), Unconventional X3 (NEXTENT prefetch
> key), Reach X5 (per-token bond predictor) all converge on outlier-channel-
> pattern-at-layer-entry as the universal per-token routing signal. Strongest
> rep: Composition X1 (highest A4, constructive math). Cheapest settler:
> outlier-span × W_Q-rowspan principal-angle measurement (~4h).

### 4. Frame-novelty bonus advancers
Up to 2. For each: name the frame, name the cousin (if any) in the literature
or kill list, explain why this is genuinely past the saturation boundary.

### 5. Rejected with rationale (one bullet each)
Per rejected idea: ID, axis(es) that failed, kill-list-style one-liner that
will be appended to KILL_LIST.md if the proposal does not survive to a later
stage. CF9 rejections must include the **stripped-primitive note** ("after
removing the imported X, the remaining structure is Y; Stage 4 may rescue Y
from a different orientation").

### 6. Hand-off to Stage 3
A short numbered list of advancers (3–4 per track + convergence reps + frame-
novelty bonuses), each with: title, orientation, track, total score, the one
specific question Stage 3 should pressure-test most aggressively. Stage 3 has
finite time; you focus its energy.

---

## Calibration anchors from v1 — useful illustrations

These are the standards you are calibrating against. Cite them implicitly
(don't quote them in your output) but use them to check your own judgments.

- **R1 Idea D — zero-overhead cross-layer SwiGLU predictor**: A1=2, A2=2,
  A3=3 (clean binary settler with within-layer baseline), A4=3 (constructive,
  10-line NumPy auditable), A5=2 (Deja Vu owns ReLU; SwiGLU is open).
  Total 12. ADVANCE. Stage 5 picked it; Stage 6 added the within-layer baseline.
- **R3 CSMF (polynomial silu)**: A4=0 (Stage 6 caught polynomial-substitution-
  does-not-eliminate-W_gate; the d^3 merged tensor would be hundreds of GB
  per layer). v2 should catch this at Stage 2: the Mechanism reads as
  constructive but is not auditable — the d^3 cost is invisible until you do
  the dimension count, which is exactly what A4=2-or-3 demands.
- **R3-A AQFKV**: A1=2, A2=2, A3=3 (per-head vs global disambiguation rule was
  load-bearing for the structural finding), A4=3, A5=3 (no published Qwen3
  W_Q-isolated rank curve). Total 13. ADVANCE. Subsequent Stage 6 catch was
  about disambiguation rule; Stage 2 should already be flagging the
  per-head/global ambiguity in its rationale.
- **R6-B LHQD embed/lm_head**: would have been A1=2, A2=2, A3=3, A4=3,
  A5=2 in tied form — the kicker is that the "tied" precondition was missed
  before Stage 6. v2 rule: any proposal that operates on `lm_head` must
  declare whether the target uses tied or untied embeddings (CF12 dependency).
  Stage 2 enforces this declaration; absent declaration → A4 capped at 1.

These v1 outcomes are the calibration set. If your scoring of a v2 idea is
inconsistent with how you'd score these, recalibrate before producing the
report.

---

## Length and tone

800–1500 words total. Terse and opinionated. Cite specific CF numbers and
specific kill-list entries. Score generously on A2 (frame novelty) for
genuinely unfamiliar frames that pass A4; score harshly on A4 for any
imported-theorem hand-wave. The goal is to surface the audacious-and-grounded
proposals and kill the math-broken ones in the same pass.
