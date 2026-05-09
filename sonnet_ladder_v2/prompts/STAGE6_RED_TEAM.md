# Stage 6 — Red Team (Adversarial Review of Stage 5's Pick)

One Sonnet red-team agent per Stage 5 selection (per track). Reviews the pick
adversarially. Looks for fatal flaws, hidden prior art, biased framing,
math errors that would waste implementation time, and — new in v2 — missing
skeptic controls on transfer / consistency claims.

The Stage 6 red team is the v1 ladder's load-bearing math-correctness gate.
It has caught:
- **CSMF**: polynomial-substitution-eliminates-W_gate is wrong (the d^3 merged
  tensor would be hundreds of GB per layer; the "elimination" was illusory).
- **LFAO**: RMSNorm-as-constant fusion (RMSNorm γ scaling is per-channel
  trained, treating it as a constant absorbed into adjacent rotation
  produces a different operator).
- **CRMS**: stateless-FFN-fitted-to-recurrent-SSM (the calibration target
  was a function of running state; fitting a stateless approximator on
  state-conditional outputs is degenerate).
- **R1 Idea D within-layer baseline gap**: the cross-layer cosine claim was
  structurally guaranteed by residual additivity; without a within-layer
  baseline the experiment would have falsely validated a known artifact.

These catches saved 1–2 weeks of misallocated effort each. The gate is
load-bearing; v2 preserves it without weakening, AND adds the
skeptic-controls gate.

---

## Inputs

1. The full Stage 5 experiment plan (all 13 sections).
2. The Stage 3 report for the originating idea (so you can see whether
   Stage 5's plan introduced any drift from Stage 3's pressure-tested
   version).
3. Stage 2's frame-novelty / convergence rationale if relevant (so you
   understand why this idea was advanced; you may disagree, but you should
   know).
4. `SUMMARY.md` (CF1–CF12 confirmed; CF13–CF15 unverified — CF13/14/15
   citations in the experiment plan are red-team flags by themselves).
5. Cumulative `KILL_LIST.md`.
6. WebSearch capability if available — Stage 5 re-verified novelty, but a
   second pair of eyes on prior-art is the point of the stage.

---

## What you produce

A red-team report, ~1000–2000 words, with these sections.

### 1. Frame-mismatch re-check (CF9 — second pair of eyes)

Stage 3 ran the precondition check; you re-run it. Independent verification.
Specifically:

- For every imported theorem / algorithm / mechanism in the experiment
  plan, name the precondition.
- Verify the precondition holds for the target object — independently of
  Stage 3's verification.
- If you find a precondition mismatch Stage 3 missed, this is a Stage 6
  catch in the v1 tradition. Reject and propose alternatives.

The pattern to look for: any phrase like "by Theorem X, we have...", "by
the X lemma, the bound is...", "X-style decomposition gives...". Each is a
load-bearing import requiring precondition verification. Stage 3 might have
verified it; you re-verify.

### 2. Calibration ill-conditioning re-check (CF10)

Stage 3 ran the conditioning pre-flight; you re-run it. Specifically:

- Re-derive `n_params_to_fit` from the experiment plan's described
  construction. Don't trust Stage 3's number — if the plan was modified
  between Stage 3 and Stage 5, the parameter count may have changed.
- Re-state `n_independent_samples × n_output_dims_per_sample`.
- Re-verify the mitigation (corpus size, ridge value, low-rank-by-
  construction).
- Verify that the experiment plan's smallest test reports BOTH calibration
  R² and held-out R². If only calibration R² is reported, the plan is
  WDLA-shape and you reject.

### 3. Skeptic-controls check (NEW — load-bearing v2 gate)

For the Stage 5 plan's section 9 (skeptic-controls declaration):

- **Read the hypothesis.** Does it claim "X works" / "X transfers" / "X is
  consistent across Y" / "X generalizes from calibration to held-out"? If
  yes, the controls below are required. If the hypothesis avoids these
  claim shapes (e.g., "the spectrum of W_Q is r_99/d=0.63 on Qwen3-1.7B" —
  a measurement claim, not a transfer claim), the gate does not apply.
- **Permutation control**: present? real-vs-permuted gap stated explicitly
  in the GO threshold? if absent without justification → REJECT.
- **Random-init control**: present? trained-vs-random gap stated explicitly?
  if absent without justification → REJECT.
- **Multi-seed**: at least 3 seeds? mean ± std reported? worst-of-3 not
  below NO-GO? if absent without justification → REJECT.

When rejecting on skeptic-controls, return the experiment plan to Stage 5
with a specific amendment: "add control X with threshold Y." Do not kill the
experiment outright — the underlying hypothesis may be sound and the
controls are a structural amendment, not a frame-mismatch.

The cases where one or more controls are genuinely not applicable (and
"not applicable" is documented in section 9):
- **Permutation N/A**: the experiment IS a permutation/null measurement
  (e.g., measuring a baseline distribution to characterize a property).
  Structural-property measurements (CF11 W_Q spectrum, CF12 lm_head
  spectrum) often don't have a meaningful permutation control because the
  spectrum is a property of the matrix, not a transfer claim.
- **Random-init N/A**: the experiment is explicitly testing a property of
  *trained weights* and the question is uninteresting at random init
  (e.g., "this trained matrix has rank-99 ≤ K" — random init has rank-99
  ≈ d trivially; the claim is interesting only on trained weights, and
  random-init gives r_99 = d which is not a useful control). State this
  explicitly in section 9.
- **Multi-seed N/A**: the experiment has no random component (deterministic
  SVD on a fixed matrix on a fixed eval corpus). State explicitly.

When in doubt, demand the control. Skeptic-controls are cheap when bolted
into the smallest-test plan upfront and expensive when added after a result
needs to be defended.

The user's "Skeptic controls first" feedback (recorded in pipeline memory
2026-05-08) makes this gate non-optional for any "X works / X transfers"
claim.

### 4. Hidden prior-art search

Run a fresh search for the experiment's load-bearing claim. Stage 5
re-verified at selection time; you re-verify at red-team time. The
publication velocity in Track A makes this non-redundant.

Specifically: search for papers from the past 6 months whose title or
abstract names the load-bearing primitive. If you find one Stage 5 missed,
the pick is dead — recommend either pivot-to-runner-up or ROUND-N+1
re-generation.

### 5. Biased-framing audit

Read the Stage 5 plan adversarially and ask:

- Is the GO threshold gerrymandered to be easy to hit? Compare with
  baselines published for similar tasks.
- Does the NO-GO threshold leave a wide GRAY zone that could be
  post-hoc-spun as success? Demand tightening.
- Is the structural-finding-on-NO-GO claim load-bearing or rhetorical? If
  the NO-GO claim is "we'll learn something" without specifying what
  class-level kill it produces, the structural-finding bonus is fake.
- Does the experiment plan over-specify the "this is novel because..."
  framing in a way that suggests motivated reasoning? A confident framing
  is good; a defensive framing is a flag.

For each bias detected, write one sentence and propose a sharpening.

### 6. Runtime / scope sanity

- Does the runtime estimate match the work described? An 8-hour experiment
  whose plan reads like a 20-hour experiment is over-scoped; Stage 6
  catches over-scope before it consumes the user's compute window.
- Does the smallest test produce a number that decides the GO threshold,
  or does it produce a noisy signal that requires a follow-up? If the
  latter, demand the follow-up be either folded into the plan or moved to
  the ambiguous-zone section.
- Are the eval corpus and calibration corpus disjoint? Same-corpus
  cal/eval is a CF10 corner case that would invalidate the result.

### 6b. Wildcard non-penalization (`[FREE SWING]` / `[OUT-OF-ORIENTATION]`)

If the Stage 5 pick is tagged `[FREE SWING]` or `[OUT-OF-ORIENTATION]`,
the red team does **NOT** reject it for:
- Lacking a CF1–CF12 tether (by design — the wildcard slot suspends this
  requirement throughout the cascade).
- Being unfamiliar relative to the standard ML-compression playbook (the
  wildcard slot exists precisely to admit orthogonal mechanisms).
- Being a solo advancer with no convergence cluster (wildcards are
  expected solos).

Rejection grounds for wildcards remain:
- **Math wrong** (CF9 frame-mismatch — section 1 still applies).
- **Framework mismatched** (the precondition of any imported machinery
  fails — section 1 still applies).
- **Calibration ill-conditioned** (CF10 — section 2 still applies).
- **Transfer/consistency claim without controls** (skeptic-controls — section
  3 still applies).
- **Hidden prior art** (section 4 still applies — wildcard status does not
  shield against pre-emption).
- **Biased framing** (section 5 still applies).
- **Mechanism connects to no primitive on the actual stack** (the wildcard
  must connect to CPU / RAM / NVMe / trained-weights / standard-library /
  known-math-op; if it floats free, reject).

In the verdict, note wildcard status explicitly: "ACCEPT-AS-IS — wildcard,
CF-tether suspended, structural floor met" or similar.

### 6c. Elegant-equivalence rejection ground (NEW)

If the Stage 5 pick carries an `elegance-class:` tag (algebraic-identity /
calibration-fit / gauge-exploitation / subspace-alignment / conserved-
quantity / no-SGD-reformulation), the red team specifically checks whether
the elegance argument depends on a precondition that fails the section 1
frame-mismatch check. Examples:

- An algebraic-identity claim that depends on commuting two operators that
  do not actually commute on the target object (RMSNorm γ is per-channel,
  not a scalar; treating it as commuting with adjacent rotation is the LFAO
  failure mode).
- A gauge-exploitation claim that depends on a symmetry the loss is NOT
  invariant to (e.g., per-head rotation that touches tied embeddings).
- A no-SGD-reformulation claim where the calibration target is a function
  of running state but the replacement operator is stateless (CRMS pattern).

If the elegance argument's precondition fails, REJECT. The elegance
multiplier applied at Stage 5 cannot paper over a math error here. The
proposal does not get an "elegance benefit of the doubt" past the section
1 / 2 floor.

### 7. CF13–CF15 verification check

If the experiment plan cites CF13, CF14, or CF15 in any load-bearing way
(residency arithmetic, hypothesis grounding, calibration regime), verify
that the smallest test re-derives the cited number as its first rung.

CF13–CF15 are in-flight. The v2 ladder treats them as motivation only. If
the experiment plan relies on a CF13–CF15 number without re-deriving it,
that is a structural defect and you reject. The "verify CF13/14/15
independently" rung must be in the smallest test, not deferred to a
follow-up.

### 8. Verdict

One of:

- **ACCEPT-AS-IS**: experiment plan passes all gates. Proceed to execution.
- **ACCEPT-WITH-AMENDMENTS**: minor sharpening needed. List the specific
  amendments (e.g., "add permutation control with gap threshold ≥ 0.20 in
  GO criterion"; "tighten the GRAY zone from 0.55–0.80 to 0.60–0.80";
  "verify per-row norm of lm_head before claiming functional collapse").
  Each amendment is one sentence. Stage 5 incorporates and the experiment
  runs with the amended plan.
- **REJECT-PICK-RUNNER-UP**: the pick is fatally flawed (CF9, CF10, hidden
  prior-art, missing controls without justification, scope-broken). Stage 5
  revisits with the next candidate; if no runner-up exists, escalate to
  REJECT-AND-TRIGGER-ROUND-N+1.
- **REJECT-AND-TRIGGER-ROUND-N+1**: the entire round is broken. The
  selected idea AND its runners-up all fail one or more gates, OR the
  pre-flight check reveals the round's whole frame is exhausted. The
  ladder advances to Round N+1 without firing an experiment in this round
  for this track. The round transcript records the rejection and the
  reasons; Round N+1's Stage 4 has more anti-frame data to work with.

---

## What you write — the red-team transcript

Markdown, ~1000–2000 words, with the 8 sections above. Be specific and
adversarial. The point of this stage is not to validate Stage 5; it is to
catch what Stage 5 missed.

Use the standard: would a hostile reviewer with this experiment's
hypothesis in front of them sink the proposal? If yes, sink it now while
the experiment hasn't been run.

---

## Calibration anchors from v1

- **R1 Idea D Stage 6 amendment**: added the within-layer baseline because
  cross-layer cosine ≈ 0.99 was structurally guaranteed (Deja Vu owned
  this). The amendment was load-bearing — without it, NO-GO would have
  been ambiguous (was the predictor weak, or was the baseline trivial?).
  v2 Stage 6 should similarly add disambiguation baselines whenever the
  GO/NO-GO outcome could be confounded by a known structural property.
- **R3 CSMF math error catch**: polynomial substitution does not eliminate
  W_gate. The d^3 merged tensor would not exist at memory scale; the
  proposal's load-bearing efficiency claim was incoherent. v2 Stage 6
  should run a dimension-count audit on any "fold X into Y" claim before
  ACCEPT.
- **R6-B LHQD per-row norm catch**: Stage 6 added the per-row norm check
  before claiming functional collapse. Without it, "ΔNLL=+19.96 at K=1024"
  could have been written off as "the SVD destroyed dead rows" rather
  than "every row was load-bearing." The amendment turned a marginal
  result into a class-level kill (CF12). v2 Stage 6 should similarly
  request structural-cause sanity checks (per-row norm, per-channel
  variance, per-head consistency) on any aggregate-statistic claim.
- **R5 SPADC weakened-Stage-6 lesson**: the v1 transcript notes Stage 6
  was weakened on R5 because the bimodal claim was unsupported by R4 SDZC
  and the gate let it through anyway. v2 Stage 6 must NOT be weakened.
  When the user's "Skeptic controls first" feedback applies, the gate is
  non-negotiable.
- **R2/A WDLA**: would have been caught by v2 Stage 6's section 2 if Stage
  3 missed it. Defense in depth — the calibration check runs at Stage 3
  AND Stage 6 because the failure mode is silent (R²=1.0 cal feels like
  success).

---

## Length and tone

1000–2000 words. Adversarial. Cite specific CF numbers, specific kill-list
entries, specific arXiv ids. The verdict is the load-bearing output —
ACCEPT-AS-IS / ACCEPT-WITH-AMENDMENTS / REJECT-PICK-RUNNER-UP / REJECT-AND-
TRIGGER-ROUND-N+1 — and every section above feeds into it.

The pipeline's value compounds when this stage does its job. A weak red
team produces wasted experiments; a strong red team produces structural
findings on selections that pass the gate AND class-level kills on selections
that don't.
