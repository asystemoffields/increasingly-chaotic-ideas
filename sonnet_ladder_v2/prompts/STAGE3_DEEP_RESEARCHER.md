# Stage 3 — Deep Researcher (Pressure-Test Stage 2's Advancers)

One Sonnet researcher per advancing idea (parallel agents are fine; ideas are
independent at this stage). For each idea, run the full pressure-test:
prior-art literature search with date-stamped novelty, residency arithmetic in
detail, smallest-test sharpening, frame-mismatch verification, calibration
ill-conditioning pre-flight, and a refined go/no-go.

The Stage 3 output is what Stage 4 uses to decide which orientations are
saturated and which still have territory, and what Stage 5 picks from. A weak
Stage 3 produces a Stage 5 selection that Stage 6 has to amend or kill —
expensive. A strong Stage 3 prevents the Stage 6 catch by doing the math
correctness work earlier.

This stage is the v1 ladder's existing Deep Researcher, sharpened with two
imports from the Opus pipeline's Stage 2 prior-art filter:
- The **per-mechanism status taxonomy** (PRE-EMPTED / ADJACENT / NOVEL /
  PARTIAL) applied to each load-bearing claim of the proposal, not to the
  proposal as a whole.
- The **two-paper composition flag** — naming the closest published cousin
  pair and explaining what value the proposal adds beyond the obvious
  composition of the two.

---

## Inputs (per advancing idea)

1. The full Stage 1 proposal in its six-section form.
2. The Stage 2 score and rationale (which axes carried it, which were weak;
   the one specific question Stage 2 asked you to pressure-test most
   aggressively).
3. Any convergence-cluster note from Stage 2 if this idea was the strongest
   representative of a cluster — you must consider the cluster's other
   members when running prior-art (a published method that pre-empts cluster
   member B may also pre-empt this idea even if A2 says "no direct prior on
   the named mechanism").
4. `SUMMARY.md` (CF1–CF12 confirmed; CF13–CF15 unverified).
5. Cumulative `KILL_LIST.md`.
6. WebSearch capability if available — use it. Run real searches. The
   v1 ladder's published kill rate (Round 2 mass pre-emption: CRANK,
   TRED, DLRA, RECS) was a wake-up call: Sonnet's training-distribution
   prior is also Sonnet's prior-art-recall ceiling, and live searches are
   the only way past it.

---

## What you produce per idea

A refined version of the proposal, ~1500–2500 words, with these sections.

### 1. Mechanism decomposition into load-bearing claims

Restate the proposal as a numbered list of **load-bearing claims** (typically
3–6). Each claim is one sentence, isolatable, and individually falsifiable.
Example shape (from Opus R1 Composition prior-art structure):
> M1: Activation outlier decomposition into static/dynamic/bulk subspaces.
> M2: Outlier-channel-to-W_Q-head-shared-subspace alignment.
> M3: AVX2 PEXT/PDEP for sparse outlier channel gather.
> M4: NVMe-resident MLP weights with bus-signal-driven prefetch.
> ...

The decomposition is the substrate for everything else in your report. A
proposal whose claims you cannot decompose this way is hand-waved and should
be flagged for return-to-Stage-2.

### 2. Per-claim prior-art status

For each load-bearing claim, run a search and assign a status:

- **PRE-EMPTED**: a published method does substantively the same thing on
  similar models. Cite arXiv id and date. The proposal must drop this claim
  or transpose it to a regime where the prior art does not apply.
- **ADJACENT**: a published method does something nearby but materially
  different. Cite the closest cousin and articulate the differentiation in
  ≤2 sentences. The proposal can keep the claim but must explicitly position
  against the cousin.
- **PARTIAL**: precedent exists for some sub-claim but the load-bearing
  combination is unpublished. Articulate exactly which sub-claim is
  precedented and which is not.
- **NOVEL**: no published method addresses the claim. Date-stamp this
  ("no paper found doing X *as of <YYYY-MM-DD>*"). Track A claims have a
  shorter publication-velocity half-life than Track B; aggressively search
  arXiv listings from the last 6 months for Track A.

A proposal where ≥50% of load-bearing claims are PRE-EMPTED is not refinable;
it is salvage. Note this and recommend Stage 4 attack on the surviving novel
sub-claim from a different orientation.

**Elegance-class tag (when status is NOVEL).** When a load-bearing claim is
NOVEL AND the mechanism class is "computationally cheaper for an elegant
reason," append an `elegance-class:` tag. Sub-classes:
- `algebraic-identity` — exact computation-graph reduction (AERO fold,
  Performer expectation-equivalent kernel, Kronecker / block-diagonal
  exploitation, commuting-operator reorder)
- `calibration-fit` — closed-form / NMF / generalized-eigenvalue / polynomial
  fit replacing an operator with a cheaper one matching its measured behavior
- `gauge-exploitation` — gauge symmetry, head / channel / layer redundancy
  group, basis-rotation that the loss is invariant to
- `subspace-alignment` — subspace identity between heads / layers / tied
  embeddings exploited to fold or share weights
- `conserved-quantity` — a measured invariant (norm, eigenvalue mass,
  outlier index) used as the load-bearing primitive
- `no-SGD-reformulation` — replacement of an SGD-trained component with a
  closed-form construction matching its behavior

The tag is consumed by Stage 5's elegant-equivalence multiplier. Tag at most
the dominant elegance class per claim; if the mechanism is plain compression
without an elegance argument, omit the tag.

### 2b. Wildcard handling (`[FREE SWING]` / `[OUT-OF-ORIENTATION]`)

If the advancer carries the `[FREE SWING]` or `[OUT-OF-ORIENTATION]` tag,
the pressure-test does NOT demand connection to existing CF1–CF12 findings.
The proposal is admissible if it connects to a **primitive on the actual
stack**: CPU instruction or feature, RAM / NVMe substrate behavior, trained-
weight property accessible without retraining, standard-library or known-
math operation. Document the primitive in one sentence.

What does NOT loosen for wildcards: section 3 (CF9 frame-mismatch) and
section 4 (CF10 calibration ill-conditioning) are math-correctness floors
and apply unchanged. Section 6's smallest-test ≤ 8h gate applies unchanged.
Section 9's two-paper composition flag applies unchanged. The same
structural rigor still kills math errors and engineering-integration; only
the CF-tether requirement is suspended.

In section 10 (Verdict), wildcards that meet the structural floor get
REFINE; wildcards that fail CF9 / CF10 / smallest-test get KILL with the
same rationale shape as standard advancers. Note the wildcard status in the
verdict line: "REFINE — wildcard, CF-tether-suspended, structural floor met."

### 3. Frame-mismatch check (CF9 — verify imported theory's preconditions)

For every load-bearing claim that imports a named theorem, named algorithm, or
machinery from another field, write:

- **Imported object**: e.g., "Bochner's theorem", "Count-Sketch median
  estimator", "tabulation hashing", "Tucker decomposition", "Performer's RFF
  approximation", "tensor network bond dimension".
- **Precondition the imported object requires**: e.g., shift-invariant kernel,
  signal sparsity, large alphabet, low-rank tensor structure, expectation-
  level approximation, low entanglement entropy.
- **Whether the precondition holds for the target object**: literally
  verifiable. State the test ("the residual stream is not sparse; CF8 says
  W_up is full-rank") and the result.

If the precondition does not hold, the imported machinery is decorative.
Strip it. What's left is usually well-known prior art under a different name —
identify that prior art and add it to the prior-art status table.

This is the gate that the v1 ladder caught CSKQ, RFSK, and THWR on at Round 6.
v2 wants Stage 3 to catch these before Stage 5 selection wastes the cycle on
a Stage 6 amendment.

Examples of the failure pattern from v1:
- **CSKQ**: imported Count-Sketch (sparsity precondition), target is dense
  INT4 residuals → bound vacuous (5470× signal magnitude in empirical check).
  The "what's left after stripping the import": symmetric quantization. Not
  novel.
- **RFSK**: imported Bochner's theorem (shift-invariance precondition), target
  is weight matrices (not kernels) → reduces to JL random projection, which
  is strictly worse than SVD; CLUBS already proved SVD-rank-k fails on
  Qwen3 W_up.
- **THWR**: imported tabulation hashing (collision-resistance over a large
  alphabet), target has c=4 with 16 exhaustive entries → no collisions exist
  by definition; reduces to a 16-entry codebook lookup (VPTQ territory).

### 4. Calibration ill-conditioning pre-flight (CF10)

If the proposal involves any calibration fit — affine projection, codebook
centroid fit, low-rank decomposition with calibration residual fit, kernel
feature-map fit, ridge regression, NNLS — you must compute and report:

- `n_params_to_fit`: explicit count from the proposal's described structure.
  For an affine A: d_target × d_source + d_target. For a codebook: K × d_centroid.
  For a low-rank correction: state the construction explicitly.
- `n_independent_samples`: number of distinct calibration data points, NOT
  total floats.
- `n_output_dims_per_sample`.
- `n_independent_samples × n_output_dims_per_sample`.

If `n_params_to_fit` is within one order of magnitude of `n_independent_samples
× n_output_dims_per_sample`, the fit is underdetermined. The proposal MUST
specify mitigation:
- N >> n_params / d_output (typical 10K–100K tokens for 1.7B-class targets);
  state the corpus.
- Strong ridge (1e-1 or higher when underdetermined); state the value.
- Force low-rank by construction (e.g., A = U·V with U ∈ R^{d_t × r},
  V ∈ R^{r × d_s}, parameter count drops to (d_t + d_s) × r — well-conditioned
  at small calibration); state the rank.
- Cal-vs-eval R² gap reporting requirement; state that the experiment plan
  reports both numbers.

WDLA (CF10's source) had A with 2.1M parameters fit on ~2M calibration values
with ridge=1e-3. Result: R²=1.0 on calibration, R²=-118000 on held-out. v2's
Stage 3 catches this before Stage 5 picks it. If a calibration-fitted
proposal does not pass this pre-flight or specify a mitigation, mark it
"REQUIRES-MITIGATION" and pass to Stage 4 (the unmitigated version is dead;
the well-conditioned variant may be alive in a different orientation).

### 5. Residency arithmetic in detail

A line-by-line accounting of:

- Per-layer parameter footprint with the proposal's compression applied,
  in bytes. Cite CF1–CF12 numbers explicitly when load-bearing (e.g., CF11
  W_Q r_99/d=0.63 means K=512 truncation gives a specific dNLL=+0.20 cost).
- KV-cache footprint at the relevant context length (state the context
  length).
- Activation footprint for the smallest deployable batch.
- Total residency at the 70B target, with the breakdown.
- Per-token compute envelope (DRAM bandwidth, NVMe random-read latency at
  the relevant queue depth, AVX2 throughput) and the bottleneck.

Use only confirmed numbers (CF1–CF12). If the proposal cites CF13–CF15 in its
arithmetic, your residency arithmetic must include a "what if CF13–CF15
don't replicate" branch — the conservative version of the math.

### 6. Smallest-test sharpening

The Stage 1 proposal's smallest-test gets refined into a runnable plan. Output:
- Concrete script name (e.g., `scripts/<name>.py`) — a new script if needed.
- Model + precision + tokenizer.
- Calibration corpus (named source; size).
- Eval corpus (held-out, named source; size).
- Layers / heads / matrices touched.
- All sweep parameters with explicit values.
- Wall-clock estimate on the user's Ryzen 5 7530U / 7.28 GiB tier.
- Output path under `experiments/stage0/ladder_v2/round<N>_<name>/`.

The wall-clock estimate is a hard gate: if the smallest test exceeds 8 hours
end-to-end, mark it "TOO LARGE" and either (a) propose a smaller smoke
variant that produces a structural signal in ≤2 hours, or (b) recommend that
Stage 4 reframe the cascade so the first rung is faster. The v1 ladder's
"smallest test is actually small" gate is non-negotiable; an "8-hour
cascade" smallest test is a code smell.

### 7. Refined go/no-go thresholds

State three thresholds, all numerical:
- **GO**: the experiment passes; advance the cascade. Number tied to a CF
  number where possible.
- **NO-GO**: the experiment fails; class-level kill. Name the class.
- **GRAY / ambiguous-zone**: what to do if the result lands between. Typically
  a follow-up sweep (per-layer adaptive K, alternative stratum, different
  calibration corpus). Specify the follow-up's runtime and what it would
  resolve.

The GO threshold should be tight enough that the experiment is decisive — not
"some improvement vs baseline" but "ΔNLL ≤ 1.0 nat at K ≤ 128 across all
MLP layers." Vague thresholds produce ambiguous outcomes that don't constrain
future ideators.

### 8. Updated risk profile

A short table:
- Top 3 risks, each one sentence.
- For each, a quantitative or structural mitigation.
- The "what would falsify the whole proposal in the cheapest way" — usually
  a single measurement that, if it lands wrong, kills the entire cascade.
  This is the experiment Stage 5 should consider running first.

### 9. Two-paper composition flag (Opus pipeline import)

Identify the **closest published cousin pair** (two papers whose composition
approximates this proposal). Articulate what value the proposal adds beyond
the obvious composition. If the proposal IS the obvious composition of two
published methods with no additional structural insight, mark it "ENGINEERING-
INTEGRATION" rather than research; engineering integrations are not what
this pipeline is for and should be downgraded.

Example shape from the Opus R1 Composition prior-art file:
> Systematic Outliers + FlexGen ≈ "use persistent outlier channels to route
> to NVMe-resident weights" — value-add: signal specificity (Jaccard-cliff
> + W_Q alignment) + overlapped prefetch.

If your proposal's value-add is "we pick better hyperparameters" or "we
combine them on a different model family," that's engineering. Research
value-add must be a structural claim the cousin pair leaves on the table.

### 10. Verdict

One of:

- **REFINE** — proposal advances to Stage 4/5 with the refined plan from
  sections 5–7. Stage 5 selects from the REFINE pool.
- **REGENERATE** — proposal is salvage, not progress; ≥50% of load-bearing
  claims are PRE-EMPTED or fail CF9. Pass the surviving primitive (the part
  of the mechanism that survived all gates) to Stage 4 as a target for
  reframing in a different orientation.
- **KILL** — proposal fails CF9, CF10 unmitigated, or all mitigations
  available render the proposal trivial. Append to KILL_LIST.md with cited
  rationale.
- **DOWNGRADE** — proposal is real but is engineering-integration of cousin
  pair without structural value-add. Park; not a Stage 5 target this round
  unless the cascade is otherwise empty.

---

## Calibration anchors from v1

- **R1 Idea D refined**: at Stage 3 the within-layer baseline was added
  because Deja Vu's cross-layer cosine result is structurally guaranteed.
  Stage 3's value-add was a baseline that disambiguated trivial-cosine from
  load-bearing-prediction. v2 Stage 3 should similarly add disambiguation
  baselines whenever the proposal's claim could be confounded by a known
  structural property.
- **R6 CSKQ/RFSK/THWR triple kill**: all three would have been caught at v2
  Stage 3 by section 3 (frame-mismatch). The pattern is "named theorem
  imported, precondition not verified" — if section 3 is filled out
  honestly, the import dies before Stage 5.
- **R2/A WDLA**: would have been caught at v2 Stage 3 by section 4. The
  proposal had n_params=2.1M and n_independent_samples~2M. The unmitigated
  fit was guaranteed to overfit; section 4 forces the math.
- **R3-A AQFKV refinement**: Stage 3's job was to articulate the per-head vs
  global disambiguation. v2 Stage 3 does this in section 7's GRAY-zone
  follow-up — naming what the experiment cannot disambiguate at the first
  pass and what the follow-up sweep is.

---

## Length and tone

1500–2500 words per idea. Cite arXiv ids and dates. Be specific. The output
is what Stage 5 picks from; a vague Stage 3 produces a vague selection.
Live searches are not optional — the v1 publication-velocity policy makes
date-stamped novelty load-bearing.
