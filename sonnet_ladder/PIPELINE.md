# Ladder Pipeline — Multi-Agent Idea Search

A recursive multi-agent ideation pipeline that explores a research solution
space by spawning specialized sub-agents at each stage, accumulating empirical
knowledge across rounds, and converging on testable experiments.

## Philosophy

**The pipeline characterizes a solution space rather than narrowing it.** Every
killed idea is a piece of empirical knowledge — not a reduction of available
ideas. The number of generatable ideas is effectively unbounded; what bounds
the search is the quality of the empirical map being built. Each round adds
boundary markers: "this region looks promising but failed for reason X." Future
ideators reason from those markers to find unexplored frames.

**Killing ideas with new knowledge is a feature, not failure.** A round that
falsifies an entire reasoning template (e.g., "less load-bearing → more
compressible") is more valuable than a round that confirms a marginal
improvement, because the falsification constrains a *class* of future ideas
rather than a single point. The pipeline is optimized for class-level findings.

**Ideas are limitless; the bottleneck is empirical traction.** The agent stack
costs are dominated by stages that fire ideas without testing. The point of the
pipeline is to convert ideation into experiments at high throughput while
preserving rigor (Stage 6 red-teaming has caught multiple math errors that
would have wasted real implementation time).

## Use case (this instance)

This instance ran on the Nocturnal One project — searching for ways to make
70B-class LLMs runnable on a 7 GiB Ryzen laptop with no retraining. The same
pipeline structure is portable to any open-ended research question with
testable empirical experiments and an active publication landscape.

## Why this works

Empirically observed properties of the pipeline (after 4+ rounds across two
parallel tracks on a hard problem):

- **Independent sampling produces real diversity**: each Sonnet instance is a
  fresh draw, so different rounds reach different parts of the hypothesis
  space. Round 3's ideator generated a hybrid (Idea G ≈ Round 1's deferred
  Idea 5) that the Round 1 ideator missed — the convergence signal itself
  was useful.
- **Red-teaming saves implementation time**: Stage 6 has caught at least
  three load-bearing math errors before they reached experiment stage
  (CSMF's polynomial-substitution-eliminates-W_gate claim, LFAO's
  RMSNorm-as-constant fusion, CRMS's stateless-FFN-fitted-to-recurrent-SSM
  degeneracy). One catch saves 1-2 weeks of misallocated effort.
- **Empirical anchors compound**: by Round 4, structural findings from
  Rounds 1-3 (e.g., "W_up dominates firing-rank") are constraining ideators
  toward genuinely new frames rather than refinements of dead lines. The
  search visibly steers toward unexplored regions over time.
- **Negative results have positive value**: a class-level kill (e.g.,
  "low-rank decomposition of trained transformer MLP weights doesn't work
  without retraining" — proven across W_gate and W_up) is more valuable
  than three small wins, because it eliminates an entire reasoning
  template for all future rounds. The kill list is the moat.

## Round structure (per round)

Each round runs 6 stages of Sonnet sub-agents. Each agent gets a self-contained
brief that includes the *cumulative* `KILL_LIST.md` and `SELECTED.md`.

| Stage | Role | Output |
|-------|------|--------|
| 1 | Ideator | 8-12 candidate ideas with mechanism / residency / novelty / first experiment / risk |
| 2 | Judge | Score each idea 0-15 across 5 dimensions; advance top 3-4 |
| 3 | Deep researcher | Pressure-test surviving ideas: prior-art search, residency arithmetic, smallest viable test |
| 4 | Skeptic explorer | Given full kill list (incl. Stage 1's rejects + Stage 3's kills), generate 6-10 *new* ideas in unexplored frames |
| 5 | Final selector | Pick a single experiment to run; concrete plan with hypothesis, test, go/no-go, kill criteria |
| 6 | Red team | Adversarial review of Stage 5's pick. Looks for fatal flaws, biased framing, hidden prior art |

After Stage 6, the final selected experiment is appended to `SELECTED.md`. All
killed ideas (from Stages 2, 3, 4) are appended to `KILL_LIST.md` with citations
and rationale.

## Saturation as signal (added 2026-05-08, after Track A R2 mass pre-emption)

When all advancing ideas in a round get pre-empted by recent papers (Track A R2: CRANK→SmoothQuant, TRED→SimLens/ConfLayers, DLRA→arXiv:2604.20682, RECS→RetNet/RADLADS), it's not failure — **it's a signal that the search has reached a publishing-saturated region**. The closer the pipeline drifts to those well-trodden grooves, the more the kill rate climbs. **Genuinely novel space lives just past the saturation boundary.**

The challenge is that LLM ideators (Sonnet, in this instance) have a training-distribution prior that *biases* them toward published mechanisms — they re-derive what's already known because that's what they've seen. Mass pre-emption means the agent is correctly re-discovering the field, not exploring beyond it.

To push agents *past* the saturation boundary, briefs and prompts must:

1. **Anti-frame lists, not just kill lists**. Block whole flavors of mechanism, not just specific instances. "Don't propose anything that *feels like* selective scale absorption" is more useful than "don't propose CRANK."
2. **Frame-orthogonal generation directives**. Instead of "8-12 ideas," instruct "1 idea each from {coding theory, database compression, signal processing, control theory, neuroscience, distributed systems, formal verification, cryptography, …}." Forces cross-domain seeding.
3. **Constraint-driven reframing**. Pose the problem under arbitrary alien constraints: "what if continuous numbers were forbidden? what if everything had to be content-addressable? what if the model had to be reconstructible from a 10 MB seed?" Constraints not in the published prior force novel mechanisms.
4. **Cross-engineer thinking**. "What would a database engineer / compiler writer / FPGA designer propose, as distinct from an ML researcher?" Different professional priors generate different mechanisms.
5. **Stage 5 frame-novelty weighting**. Prefer ideas whose *frame* (not just specific mechanism) is unrepresented in recent publications, even if the specific mechanism is less elegant. A scrappy idea in a fresh frame has longer half-life than a polished idea in a saturated frame.

Mass pre-emption is the moment to *push harder*, not retreat. Ideas are limitless; the bottleneck is steering Sonnet's prior past its training distribution. Each round of pre-emption maps the boundary more precisely.

## Theoretical frame mismatch check (added 2026-05-08, after Track B R6 triple kill)

When an ideator imports machinery from another field (cryptography, coding theory, signal processing, statistical mechanics, etc.) into LLM compression, Stage 2/3 must verify the imported theory's preconditions hold for the target object. Three Round 6 kills exemplified the failure pattern:

- **CSKQ**: Count-Sketch's median-estimator error bound is `O(||E||_1 / w)`. Useful only when signal is sparse. INT4 quantization residuals are DENSE. Bound becomes vacuous (5470× signal magnitude in the empirical check).
- **RFSK**: Bochner's theorem approximates shift-invariant kernels. Weight matrices are not kernels. The "RFF" framing reduced to JL random projection, which is strictly worse than SVD and CLUBS already proved SVD-rank-k fails on Qwen3 W_up.
- **THWR**: Tabulation hashing's collision-resistance is useful when c-bit chunks have many possible values mapped to fewer hashed positions. With c=4 and 16 exhaustive entries, no collisions exist by definition. The "hash" reduces to a 16-entry codebook lookup — VPTQ territory.

**Check item for Stage 2/3**: before scoring an idea that imports a named theorem or named algorithm, verify the structural precondition. If the precondition (sparsity, shift-invariance, collision-resistance, etc.) doesn't hold for the target, the imported machinery is decorative. Strip it; what's left is usually well-known prior art under a different name.

**Implication for ideators**: cross-domain frame seeding works (it generates pre-empt-resistant ideas) BUT must preserve mathematical correctness. Generate from frames whose preconditions structurally match the target — e.g., for full-rank dense rectangular weight matrices, don't import sparsity-assuming algorithms.

## Calibration ill-conditioning check (added 2026-05-08, after WDLA failure)

When a Stage 5 / 6 selection involves fitting parameters via least-squares on calibration data, the selector AND red-team MUST explicitly check:

- `n_params_to_fit` (e.g., for affine A: d_target × d_source + d_target)
- `n_independent_samples × n_output_dims_per_sample`

If these are within an order of magnitude, the fit is underdetermined and will memorize calibration with arbitrary held-out behavior. WDLA failed because A had 2.1M parameters fit on ~2M calibration values with ridge=1e-3. The result was R²=1.0 on calibration, R²=-118000 on held-out.

Mitigations selectors should require:
- Calibration corpus N >> n_params_to_fit / d_output (typically 10K-100K tokens for 1.7B-class targets)
- Strong ridge (1e-1 or higher when underdetermined)
- Force low-rank by construction (e.g., A = U·V with U ∈ R^{d_t × r}, V ∈ R^{r × d_s}, parameter count drops to (d_t + d_s) × r — well-conditioned at small calibration)
- Explicitly report calibration vs held-out R²; if they differ by > 0.2, the fit is overfitting

This check applies to: any affine/linear projection fit, any codebook centroid fit, any low-rank decomposition with calibration residual fit, any kernel-feature-map fit (e.g., CFPK).

## Track A publication-velocity policy (added 2026-05-08)

The arch-transpose space publishes faster than the kill list can be refreshed. Mamba (Dec 2023), Performer, RADLADS (May 2025), LOLCATS, CMoE (Feb 2025), AERO (Oct 2024), ProcrustesGPT (June 2025), LLVQ (Mar 2026) and many KV-side methods all landed within recent windows. Track A stages must:

1. **Re-check novelty at every stage** (not just Stage 3). Stage 5 must re-verify the top picks before commitment.
2. **Date-stamp claims**. "No paper found doing X *as of <date>*" rather than absolute novelty.
3. **Prefer structural/algebraic novelty** over specific-architecture imitation. A math-style idea (calibration NNLS for positive feature maps; algebraic folding under specific empirical conditions) has longer half-life than "calibration-only Mamba-from-FFN" because the latter gets pre-empted the moment a no-SGD variant of the named architecture publishes.
4. **Frame contributions as operating windows**. We're after empirical demonstrations on this hardware tier with specific structural-finding constraints (R1, R2, R3), not pure first-mover credit.
5. **Favor pre-empt-resistant ideas**. Ideas tied to empirical statistics on a specific model family (e.g., "fold gates that calibration shows are within ε of constant on Qwen3 family") are harder to pre-empt than clean architectural ideas because the empirical characterization is the moat.

Track B publication velocity is also high (NanoQuant Feb 2026, LLVQ Mar 2026, TesseraQ ICLR 2026) but the field is more saturated; novelty claims there have to navigate ~30 known competitors. Track B's risk is "subsumed by published method" rather than "pre-empted by next week."

## Two parallel tracks (added 2026-05-08, after Round 3)

The pipeline forked into two parallel tracks after Round 3:

- **Track A — `arch-transpose`**: operator-structure changes (FFN→MoE, softmax→linear attention, AERO-style activation-removal+folding, polynomial nonlinearity replacement, SSM/Mamba substitution, tensor reshapes that change computation). No SGD; calibration-driven fitting OK.
- **Track B — `compression`**: weights compress (quantization, low-rank, codebook, sparsity, paging) but the operator structure stays (still SwiGLU, still softmax attention).

Classification rule for ideators: if the *computation graph at inference* changes shape (different ops, different routing), it's Track A. If only the *bytes-per-weight* changes, it's Track B.

Both tracks share the cumulative `KILL_LIST.md` and the empirical structural findings. Each track maintains its own per-round transcript and its own SELECTED queue:

- `track_A_arch/SELECTED.md`, `track_A_arch/ROUND_<N>_TRANSCRIPT.md`
- `track_B_compress/SELECTED.md`, `track_B_compress/ROUND_<N>_TRANSCRIPT.md`

Tracks run concurrently (Stage 1 of both can be spawned in parallel since they are independent agents). After both tracks pick a Round-N experiment, the cheaper / more informative one runs first; the other queues.

## Round-over-round behavior

- Round N+1's Stage 1 receives the cumulative kill list. Cannot re-propose any
  killed idea.
- Round N+1's Stage 1 receives explicit framing-diversity instructions: "previous
  rounds converged on classes X, Y, Z; generate from underexplored frames {...}".
- The kill list grows monotonically. Search space contracts each round.

## Portability notes (for extracting as a self-contained project)

The pipeline is generalizable beyond this specific use case. To port to a
different research domain:

1. **Replace the problem statement** at the top of `PIPELINE.md` and in each
   Stage 1 brief. The framing must specify: hardware/resource constraints,
   hard constraints (e.g., "no retraining"), success criteria (e.g.,
   "dramatically impactful," "testable").
2. **Replace the published landscape list** in Stage 1's brief — the kill
   list and competitor enumeration are domain-specific.
3. **Define the idea-class taxonomy** for the new domain (the
   compression / sparsity / offload / arch-transpose split here is
   project-specific). Track A/B split is optional; single-track works
   for narrower problems.
4. **The 6-stage structure is invariant**: Ideator → Judge → Deep
   Researcher → Skeptic Explorer → Selector → Red Team. Each stage's
   role is domain-agnostic. The same prompts can be templated.
5. **The cumulative kill list and selected queue persist across rounds**
   regardless of domain. The format generalizes.
6. **Optional: parallel tracks** when the design space has cleanly
   separable axes (here: arch-transpose vs compression). Tracks share the
   kill list but maintain separate SELECTED queues.

The pipeline is a search algorithm where the LLM is both the generator and
the evaluator. Its convergence properties depend on:
- Diversity of independent samples (fresh agents per stage)
- Memory between rounds (the kill list as immutable knowledge)
- Adversarial review (red-teaming as a math-correctness gate)
- Empirical grounding (every selected experiment must produce numbers)

These are all preserved in the portable version.

## Convergence / stop conditions

Stop when one of these holds:

1. Stage 1 of Round N proposes ≥3 ideas that are already on the kill list
   (search has saturated within Sonnet's prior).
2. Stage 2 scores all of Round N's ideas ≤ 11/15 (no idea is strong enough).
3. Stage 5 of Round N selects an experiment that is structurally similar
   (>= 80% mechanistic overlap) to a prior round's selection.
4. The user manually stops after a particular SELECTED experiment looks
   compelling enough to commit to.

## Diversity / bias-handling

The pipeline uses Claude Sonnet for all stages. This means the search inherits
Sonnet's training-data biases — there are blindspots no amount of rounds will
surface. To partially mitigate:

- Stage 4 explicitly enumerates *frames* the previous stages didn't use.
- Stage 6 red-team is asked to look for biased framing in Stage 5.
- Round N+1's Stage 1 brief calls out which frames Round N over-used.

For genuine diversity beyond Sonnet's prior, manually inject:
- A different model family (Opus or external API) every K rounds
- A user-written prompt that names an angle Sonnet wouldn't naturally pick

## Output files

- `KILL_LIST.md` — cumulative ledger of killed ideas. One-line per kill: name, round, stage that killed, citation/reason, idea class.
- `SELECTED.md` — queue of selected experiments with status (queued/running/done) and result.
- `ROUND_<N>_TRANSCRIPT.md` — per-round summary of all 6 stages (concise, not full agent output).
- This `PIPELINE.md` — the runbook itself (you're reading it).

## Constraint clarification (added 2026-05-08)

**The "no retraining" constraint admits architectural transposition.** What is forbidden is SGD against a loss function. What is permitted:

- **Mathematical equivalences** that produce the same input/output behavior with a different computation graph (e.g., Performer's random-feature approximation of softmax attention, exact in expectation)
- **Calibration-derived restructurings** that fit an alternative operator to the trained model's behavior on calibration data using closed-form or non-gradient methods (k-means clustering, NMF, generalized eigenvalue, polynomial fitting, etc.)
- **Algorithmic reformulations** that replace one operator with another whose output matches on the relevant input distribution (silu → degree-3 polynomial fit; dense attention → structured-sparse pattern derived from calibration; FFN → MoE via expert-clustering with a calibration-fit router)
- **Tensor / weight reshapes** that produce algebraically equivalent (or provably bounded-error) computations: Kronecker, Tucker, hierarchical Tucker, factor-graph reformulations

Round 3 onwards: ideators may propose architectural-transposition ideas. The constraint is "the trained Qwen3-N weights can be transposed into the new shape via a no-SGD procedure." This widens the search beyond pure compression / sparsity / offload ideas.

## Idea-class taxonomy (for kill-list classification)

We tag each killed/selected idea with one or more classes so future ideators
can see which classes are exhausted vs untouched.

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
- `prediction-internal` — using model's own activations to predict its own state
- `runtime-fused` — decode-fused-with-compute kernels
- `systems-fs` — filesystem-level tricks (CoW, dedup, reflink)
- `systems-os` — OS paging, mmap, prefetch hints
- `kv-side` — KV cache focused (compress KV to free RAM for weights)
- `arch-transpose` — architectural redesign achievable via no-SGD transposition from trained weights (in scope as of 2026-05-08)
- `train-required` — anything requiring SGD/fine-tuning (out of scope)
