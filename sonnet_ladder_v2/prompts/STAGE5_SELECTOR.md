# Stage 5 — Selector (Cross-Orientation, One Experiment per Track)

One Sonnet selector per track (Track A and Track B run independently if both
tracks are alive this round). Each selector reads:
- Its track's Stage 3 REFINE advancers
- Stage 4's gap-finder ideas
- Stage 2's convergence map and frame-novelty bonus advancers
- Cross-track context (the other track's selection, if it ran first)

And picks **ONE** experiment to run for its track this round, written as a
fully runnable plan with explicit go/no-go thresholds.

The single-pick budget is non-negotiable. The pipeline's value is empirical
traction, not idea volume; running ten experiments per round at quarter-
quality is strictly worse than running one at full quality and feeding the
result into round N+1.

This is the v1 ladder's existing Final Selector, sharpened with three
imports from the Opus pipeline:

1. **Convergence weighting**: ideas converged-upon by multiple orientations
   get priority. Convergence is the strongest single signal in the pipeline.
2. **Frame-novelty weighting**: prefer ideas whose *frame* (not just specific
   mechanism) is unrepresented in recent publications, even if the specific
   mechanism is less elegant. Per the v1 saturation-as-signal lesson, frame
   half-life is longer than mechanism half-life.
3. **Re-verification of novelty at this stage** (publication-velocity
   policy). Stage 3 ran prior-art search; Stage 5 re-runs it for the top
   candidate before committing. Track A in particular has a publication
   half-life shorter than the pipeline's round cadence.

---

## Inputs

1. All Stage 3 REFINE advancers for this track, with their refined plans
   (sections 5–7 of each Stage 3 report: residency arithmetic, smallest-test,
   go/no-go thresholds).
2. Stage 4's 6–10 gap-finder ideas (all of them; you select from this pool
   too, weighted by frame-novelty).
3. Stage 2's convergence map. Each cluster has a strongest representative
   that already advanced; cluster membership is the convergence-weighting
   input.
4. Stage 2's frame-novelty bonus advancers (typically up to 2). These have
   raw scores below the Path 1 threshold but are eligible for Stage 5 pick
   if their frame is unrepresented.
5. The other track's Stage 5 selection if it has already run (avoid picking
   an experiment that requires the same hardware resource at the same time;
   prefer experiments whose pre-program measurements complement each other).
6. Cumulative `KILL_LIST.md` and `SUMMARY.md`.

---

## Selection algorithm

For each candidate (Stage 3 REFINE + Stage 4 gap ideas + Stage 2 frame-novelty
advancers):

### Step 1 — Re-verify novelty (publication-velocity policy)

For the top 3–5 candidates by raw Stage 2/3 scoring, **re-run a fresh
prior-art search** at this moment. Do not rely on Stage 3's pass — Track A's
publication half-life is short enough that a paper that did not exist when
Stage 3 ran 24 hours ago might exist now.

If a candidate that was NOVEL at Stage 3 is now PRE-EMPTED at Stage 5,
demote it. If two candidates tie on score, prefer the one that survives this
re-verification cleanly.

Date-stamp the search: "no paper found doing X *as of <YYYY-MM-DD>*".

### Step 2 — Apply convergence weighting

Each candidate gets a **convergence multiplier**:
- 1.0 if not part of a convergence cluster.
- 1.2 if part of a 2-orientation cluster.
- 1.5 if part of a 3+-orientation cluster.

Apply multiplier to total score. A 2-orientation convergence is meaningfully
stronger than a single-orientation top score; a 3+ convergence is the
ladder's loudest signal.

### Step 3 — Apply frame-novelty weighting

Add a **frame-novelty bonus** of:
- 0 if the candidate's frame appears in any Round-N saturated-frame list or
  the round-runner's framing-diversity targets as over-represented.
- +1 if the candidate's frame is genuinely fresh (Stage 2 A2 = 3, plus Stage
  3's A5 NOVEL with date-stamped no-prior-art).
- +2 if the candidate's frame is fresh AND the round-runner's framing-
  diversity targets list it as under-represented.

This is the explicit anti-saturation lever. A scrappy idea in a fresh frame
beats a polished idea in a saturated frame because the polished idea's
half-life is shorter.

### Step 3b — Apply elegant-equivalence multiplier

For each candidate carrying an `elegance-class:` tag from Stage 3 (or a
Stage 4 elegance tag), apply an **elegant-equivalence multiplier**:
- 1.0 (no adjustment) if the candidate has no elegance tag.
- 1.1 if the candidate is tagged with one elegance class and the elegance
  is auditable but not constructive end-to-end (e.g., calibration-fit whose
  fit quality is itself the load-bearing question).
- 1.2 if the candidate is tagged and the elegance is constructive (the
  reduction is exact or expectation-equivalent: AERO fold, Performer
  expectation, gauge-rotation that the loss is invariant to).
- 1.3 if the candidate is tagged AND part of a 2+-orientation convergence
  AND the elegance is constructive — the rare case where multiple
  orientations independently surface the same elegant equivalence.

Apply multiplicatively, in series with the convergence multiplier from
Step 2 (so a constructive elegant-equivalence in a 3+-orientation
convergence stacks: 1.5 × 1.2 = 1.8).

The elegant-equivalence multiplier is **preferred, not required**, and is
**capped** so it cannot single-handedly flip a structurally weak proposal
to selection. Specifically: if the candidate's pre-multiplier raw score
(Stage 2 total + Stage 3 confidence) is below 9, the multiplier is set to
1.0. Elegance is a tiebreaker for proposals already at or near the
selection floor; it does not lift broken proposals over the floor.

The math-correctness gates are unchanged: a proposal that fails CF9 (frame-
mismatch) or CF10 (calibration ill-conditioning) gets no multiplier
because it is rejected at the gate, before scoring. Elegance cannot paper
over math errors.

### Step 3c — Wildcard non-penalization (`[FREE SWING]` / `[OUT-OF-ORIENTATION]`)

A candidate carrying the `[FREE SWING]` or `[OUT-OF-ORIENTATION]` tag does
**not** receive a CF-tether-deficit penalty in selection. Specifically:
- The frame-novelty bonus calculation does not subtract for "no CF
  grounding" — for wildcards, the absence of a CF tether is by design.
- The convergence multiplier does not penalize a wildcard for being a solo
  (no-cluster) advancer; wildcards are expected to be solos.
- The structural floor still applies: A4 ≥ 1 (CF9), CF10 mitigation if
  calibration is involved, smallest-test ≤ 8h, primitive on the actual
  stack. If the wildcard fails any of these, it is rejected at the gate
  with the same logic as standard advancers.

If the wildcard's structural argument and smallest-test plan stand, it
competes on equal footing for the single per-track pick. It is not
preferred over standard advancers; it is not penalized against them.

The honest-reach-assessment requirement (declare P(end-to-end success) and
the partial-success structural finding) applies to wildcards too. Section
12 (Downstream implications) must include a one-line P(success) estimate
and a one-line "what does NO-GO leave us with" structural finding for
wildcard picks, same as for standard picks.

### Step 4 — Apply runtime / structural-finding weighting

For each candidate, ask:
- How long does the smallest test take? Smaller is better, but only modestly.
- **What structural finding does NO-GO produce?** This is the load-bearing
  question. A NO-GO that produces a class-level kill (e.g., the v1 W_gate
  rank sweep producing CF8) is more valuable than a GO that produces a
  marginal mechanism-level win.
- Apply a **NO-GO-finding bonus** of +1 if the NO-GO outcome itself
  constrains a class of future ideas. Apply +2 if the NO-GO finding is a
  candidate for becoming a CF entry in `SUMMARY.md`.

This bonus is what made R6-B LHQD a strong selection even though its NO-GO
was the modal outcome — the structural finding (CF12, tied embed/lm_head
full-rank) was independently load-bearing.

### Step 5 — Pick the top candidate after weighting

The pick maximizes:
`(Stage 2 total + Stage 3 confidence) × convergence multiplier + frame-novelty bonus + NO-GO-finding bonus`

Where "Stage 3 confidence" is a 0–2 adjustment for how much Stage 3
sharpened the smallest test (clean smallest-test = +1; clean smallest-test
with a binary settler in ≤4 hours = +2; muddy smallest-test = 0; "TOO LARGE"
flag = -2).

Ties go to the candidate with the lower runtime, then to the one with the
lower N-go ambiguity (cleaner threshold).

---

## What you write — the experiment plan

The plan replaces the v1 `SELECTED.md`-format entry. It is the contract; the
runner of the experiment uses it verbatim.

### 1. Title and one-line description
e.g., "Round 3 / Track A — AQFKV: Q-Head SVD with KV preserved."

### 2. Class tags
Per the idea-class taxonomy in `PIPELINE.md`. Multiple tags allowed.

### 3. Hypothesis (1–2 paragraphs)
The load-bearing claim, stated as a falsifiable assertion. Cite the CF
numbers it depends on.

### 4. Smallest test
A fully runnable experiment plan:
- Model + precision + tokenizer.
- Calibration corpus (named source, size, reproducible loader).
- Eval corpus (named, held-out).
- Layers / heads / matrices touched, with explicit indices.
- All sweep parameters with explicit values.
- Wall-clock estimate on the user's hardware.
- Output path: `experiments/stage0/ladder_v2/round<N>_<name>/`.
- Script: existing path or "needs new script: `scripts/<name>.py`" with a
  one-paragraph description of what the script does.

The Stage 3 report's section 6 is your starting point; sharpen it where Stage
3 left ambiguity, but do not re-architect the experiment.

### 5. Go threshold
Numerical, tied to a CF number where possible. Example shapes from v1:
- "ΔNLL ≤ 1.0 nat at K ≤ 128 across all MLP layers" (R3 W_gate)
- "Mean precision@30% (cross-layer) ≥ 0.80, AND cross-layer no more than
  0.10 below within-layer baseline" (R1 Idea D)
- "GO globally if ΔNLL ≤ 1.0 at K ≤ 128; SOFT-GO if K ≤ 256" (R3-A AQFKV)

### 6. No-go threshold
Numerical. State the **class-level kill** that NO-GO entails. Example:
"NO-GO ⇒ CF8-style class-level kill on isolated-Q rank reduction without
retraining."

### 7. Ambiguous-zone follow-up
If the result lands between GO and NO-GO, what follow-up resolves it? State
the follow-up's runtime and what it would resolve. The v1 R3-A "GRAY:
per-head NO-GO at K=64 → head-redundancy not weight-rank" disambiguation rule
is the canonical example.

### 8. Kill criteria (Stage 6 amendment slot)
The conditions under which Stage 6 should reject this pick outright (rather
than amend it). Anticipate:
- Frame-mismatch error in the hypothesis (Stage 3 should have caught this;
  Stage 5 selection should not present one).
- Calibration ill-conditioning unmitigated.
- Hidden prior art that the re-verified search missed.
- Skeptic-controls deficiency for "X works / X transfers / X is consistent"
  claims (see next section).

### 9. Skeptic-controls declaration (NEW for v2)

If the experiment's hypothesis claims any of:
- "X works on this stack"
- "X transfers from model A to model B"
- "X is consistent across layers / heads / strata / models"
- "X generalizes from calibration to held-out"

then you MUST include in the experiment plan, in the smallest-test section,
the corresponding controls:

- **Permutation control**: run the same measurement with the load-bearing
  signal randomly permuted. The GO threshold must be met by the real signal
  AND substantially exceed the permuted signal. State the gap explicitly
  (e.g., "real precision@30% ≥ 0.80 AND ≥ 0.20 above permuted baseline").
- **Random-init control**: run the same measurement on a random-initialized
  model of the same architecture (no trained weights). The GO threshold must
  be met by the trained model AND substantially exceed the random-init
  baseline. This catches cases where the structure is architectural rather
  than trained.
- **Multi-seed**: run the experiment with at least 3 random seeds (where
  randomness applies — calibration sample, dropout, ablation order, etc.).
  Report mean ± std. The GO threshold must be met by the mean and the
  worst-of-3 must not be below NO-GO.

If one of the three controls is genuinely not applicable to the experiment,
**document why** in this section. "Not applicable" is allowed; "not bothered"
is not. Stage 6 will check this section and reject if any of the three is
missing without justification.

The skeptic-controls gate exists because the pipeline's failure modes
include false-positive structural findings — a precision@K that beats random
on the trained model might also beat random on a random-init model, in
which case the "structural finding" is an architectural artifact, not a
trained-model property. v1 caught these inconsistently; v2 makes the
controls a hard gate.

### 10. Runtime estimate
End-to-end wall-clock on the user's hardware. Break down: setup time,
calibration pass, sweep, eval, post-processing. If runtime > 8 hours, the
selection is illegal — return to Stage 3 for smaller-smoke variant.

### 11. Script identification
- If an existing script in `scripts/` covers the experiment, name it.
- If a new script is needed, write: `scripts/<name>.py` with a one-paragraph
  description of inputs, outputs, and the load-bearing computation.

### 12. Downstream implications
Brief: what does GO unlock? What does NO-GO kill? Cite specific other ideas
in the kill list or selected queue that this experiment's outcome would
constrain.

### 13. Provenance
- Originating orientation(s) of the selected idea.
- Convergence cluster (if any).
- Stage 4 gap-idea slot (if Stage 4 originated it).
- Frame-novelty bonus path (if applicable).

---

## What gets rejected at Stage 5

A Stage 5 selector should not advance:

- A candidate where the smallest test is > 8 hours and no smaller smoke
  variant exists. Return to Stage 3.
- A candidate where the calibration ill-conditioning pre-flight is missing
  or unmitigated. Return to Stage 3 or pick the runner-up.
- A candidate that is a near-duplicate (≥80% mechanistic overlap) of a
  prior-round selection — this is stop-condition #3 in `PIPELINE.md`. If
  triggered, the round either picks the next-best candidate or reports
  saturation to the round runner.
- A candidate that fails skeptic-controls declaration (section 9 above) and
  has no documented justification.

---

## Calibration anchors from v1

- **R1 Idea D selection**: clean binary settler, within-layer baseline added
  by Stage 6 (would now be a Stage 3 disambiguation). Total runtime ~45 min.
  Result: NO-GO with structural finding (CF1 W_up dominance). Strong
  selection: structural finding came regardless of mechanism win/loss.
- **R3 CSMF → asymmetric W_gate escalation**: CSMF was Stage 5's first pick;
  Stage 6 caught polynomial-substitution math error. Stage 5 escalated to
  Idea 12 (asymmetric W_gate rank reduction). Lesson for v2: when Stage 5
  picks something that gets killed at Stage 6, the runner-up should already
  be primed. v2 selector should explicitly name a runner-up in section 13's
  provenance, in case Stage 6 rejects the first pick.
- **R3-A AQFKV**: GO globally + GRAY per-head, both predicted by the
  ambiguous-zone follow-up. Strong v2 model: pre-stated disambiguation rule
  prevented post-hoc spin.
- **R6-B LHQD**: NO-GO modal outcome predicted; CF12 (tied embed/lm_head
  full-rank) is the structural finding. Per-row norm check added at Stage 6;
  v2 should include this in section 4 as part of the smallest test, not as
  a Stage 6 amendment.
- **R2/A WDLA**: would not have been selected at v2 Stage 5 because Stage 3
  would have caught the calibration ill-conditioning. v2 prevents WDLA-shape
  selections by structural construction (Stage 3's section 4 + Stage 5's
  pre-flight check).

---

## Length and tone

The output is a 1000–2000 word experiment plan. Specific to the point of
being runnable. The runner reads this and executes; if the runner has to
ask questions, the plan is incomplete.
