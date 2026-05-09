# Stage 4 — Skeptic Explorer (Cross-Orientation Gap-Finder)

One Sonnet skeptic agent. Reads the **full** state of the round — every
orientation's Stage 1 output (not just the advancers), Stage 2's union
scoring with convergence map, every Stage 3 verdict, the cumulative
`KILL_LIST.md`, and `SUMMARY.md`. Generates 6–10 **new** ideas in genuinely
unexplored frames, with explicit attention to orientations and tracks whose
proposals all died in Stage 3.

This is the v1 ladder's existing Stage 4 with two structural changes:

1. **Cross-orientation rather than single-stream**: the substrate is the union
   of 4–5 orientations' Stage 1 outputs and Stage 3 verdicts. The exhausted
   *frames* are richer (multiple orientations exhaust different flavors), so
   Stage 4's gap-finding must be more discriminating.
2. **Cross-pollination rule**: when an idea died in orientation A's Stage 3
   for a CF9 / CF10 / saturation reason, Stage 4 explicitly considers whether
   a transposition of the same primitive into orientation B's pose would
   rescue it (the imported precondition might hold there; the calibration
   regime might be well-conditioned there; the frame might be unsaturated
   there).

The skeptic's job is **not** to refine surviving advancers. Stage 3 already
did that. The skeptic's job is to identify the **frames** that are exhausted
and propose ideas in **frames that are not** — including frames that no
orientation in the active set is naturally aimed at.

---

## Inputs

1. Every orientation's Stage 1 output, full text. Read all of them.
2. Stage 2's union score table, convergence map, frame-novelty bonus
   selections, and rejected-with-rationale list.
3. Stage 3's per-advancer reports with verdicts (REFINE, REGENERATE, KILL,
   DOWNGRADE), prior-art status tables, frame-mismatch findings, calibration
   ill-conditioning verdicts, and stripped-primitive notes.
4. `ORIENTATIONS.md` — the active orientation set and each orientation's
   anti-frame list, so you can see what shapes are explicitly forbidden
   inside each orientation.
5. Cumulative `KILL_LIST.md`.
6. `SUMMARY.md` — CF1–CF12 confirmed; CF13–CF15 unverified.

---

## What you produce

A skeptic report, ~1500–2500 words, with these sections.

### 1. Frame-exhaustion map

For this round, identify which **frames** (not specific ideas) are exhausted.
Distinguish three exhaustion types:

- **Saturation-exhausted**: frames where Stage 2 / Stage 3 found published
  prior-art so dense that any new variant would be subsumed within ≤2 quarters.
  Example v1 saturated frames: "selective scale absorption" (SmoothQuant
  family), "calibration-fitted Mamba-from-FFN" (RADLADS family), "shift-
  invariant kernel approximation" (Performer/RFF family).
- **CF9-exhausted**: frames where the load-bearing import has a precondition
  that does not hold for any LLM-compression target. Example: "use a sketch
  whose error bound is O(||E||_1 / w)" — sketches with sparsity preconditions
  have no LLM-compression target where the precondition holds. Don't propose
  sketches; propose mechanisms whose preconditions do hold.
- **CF10-exhausted**: calibration-fit-shaped frames where the parameter count
  always blows up faster than the calibration corpus can support. Example:
  "fit an affine A: ℝ^d → ℝ^d on 1.7B-class targets" — this is structurally
  WDLA-shaped; only low-rank-by-construction or ridge-saturated variants are
  alive.

For each exhausted frame, name a **stripped primitive** — what's left after
removing the exhausted machinery. Stripped primitives are often valuable in a
different orientation. Example from CSKQ stripping: "after removing Count-
Sketch, what's left is symmetric quantization on residuals — which is dead
in compression-quant but might be alive as an arch-transpose composition."

### 2. Orientation-saturation diagnostics

For each active orientation, write one paragraph:

- **Did it advance ideas to Stage 3?** If yes, with what verdicts? If no,
  the orientation is at risk of being dropped per the rotation rule in
  `ROUND_OVER_ROUND.md`.
- **Did its Stage 1 output land in a convergence cluster?** Convergences keep
  orientations alive even if their individual ideas die.
- **Anti-frame list growth recommendation**: based on what died this round,
  what specific named flavor should be appended to this orientation's
  anti-frame list? This is the round runner's input to anti-frame growth
  per `ROUND_OVER_ROUND.md`.

If an orientation looks structurally exhausted (all ideas died at Stage 2 or
Stage 3, no convergences), recommend either DROP-FOR-ONE-ROUND or
INTRODUCE-ROUND-SPECIFIC. The introduction recommendation must include a
proposed creative directive + anti-frame list + cross-domain seed list +
falsifiability requirement (one paragraph each), so the round runner can
plug it into ORIENTATIONS.md as a temporary slot.

### 3. Cross-pollination opportunities

For each Stage 3 verdict of REGENERATE or KILL, ask:

- Would a different orientation's pose rescue the surviving primitive?
- Specifically: if the kill reason was CF9 (imported theorem precondition
  fails), is there an orientation whose target IS the precondition's natural
  domain?
- If the kill reason was saturation in a Track A frame, is there a Track B
  transposition (or vice versa)?
- If the kill reason was CF10 (calibration ill-conditioned), is there a
  low-rank-by-construction or ridge-stable variant in another orientation?

Output this as 1–3 cross-pollination candidates that you then expand into
full Stage 4 ideas in section 5.

### 4. Cross-domain seed exploration

The v1 PIPELINE.md saturation-as-signal section names cross-domain seed
lists as the primary anti-saturation tool. Stage 4 is where these get
exercised aggressively. You are explicitly instructed to ask:

- **What would a database engineer propose?** B-tree page layout, columnar
  storage, content-addressable stores, range-coded compression, zone maps,
  bitmap indices, MVCC snapshots — what's the LLM-compression analog?
- **What would a compiler writer propose?** Loop fusion, dead-code
  elimination, register allocation, profile-guided optimization, SSA, CSE —
  what's the inference-time analog?
- **What would an FPGA designer propose?** Spatial dataflow, systolic arrays,
  fixed-precision pipelines, BRAM tiling, bit-serial arithmetic — what
  primitives transfer?
- **What would a control theorist propose?** State observers, Kalman
  filtering, Lyapunov stability, model predictive control — applied to
  inference-time activation prediction?
- **What would a cryptographer propose?** Merkle trees, hash-based
  authentication, oblivious RAM, secret sharing, garbled circuits — applied
  to weight residency or content-addressable storage?
- **What would a signal processor propose?** Filter banks, multirate
  signal processing, wavelets — but check CF9 preconditions, since most
  signal-processing primitives assume continuity / smoothness / stationarity
  that LLM weight matrices do not have. The frame mismatch trap is most
  acute here.
- **What would a network coder propose?** Random linear network coding,
  fountain codes, LDPC over-the-air — applied to streaming weights from
  NVMe or distributing them across cores?

Pick the **two or three professional priors most likely to produce a
mechanism whose precondition matches an LLM-compression target**, given the
kill list and frame-exhaustion map. Don't enumerate all of them; pick the
high-leverage ones for this round.

**Elegant-equivalence prompt (apply to each picked professional prior).**
For each professional prior you pick, ask explicitly: *what algebraic
identity, computational equivalence, or no-SGD reformulation does this
prior's view of the LLM-compression target reveal?* Examples of the prompt
shape:
- "What algebraic identity does a database engineer's view of attention
  reveal — is the query / key product expressible as a join over a
  pre-computed index that makes the softmax cheaper without changing it?"
- "What conservation law does a control theorist see in the residual stream
  that would let a layer be replaced by a fixed-form operator on the
  conserved quantity alone?"
- "What gauge symmetry does an FPGA designer's bit-serial view expose in
  the weight matrices that would let a multiply collapse into a shift +
  add?"

The bias is for **same I/O, cheaper compute, for a structural reason**.
Archetype: AERO-style activation removal — drop SiLU and W_gate algebraically
folds into W_up. If a professional prior naturally produces an elegant-
equivalence-class mechanism, prefer it over priors that produce only
compression-by-truncation.

### 5. Constraint-driven reframing

Pose the Nocturnal One problem under arbitrary alien constraints. Each
constraint forces a different mechanism class:

- **No continuous numbers**: weights and activations are integer or boolean.
  Forces ternary / binary / hash-based / lookup-table mechanisms.
- **Content-addressable**: every weight chunk is retrievable by a hash of
  its content; no positional indexing. Forces dedup, reflink, ContentDef
  coding, Merkle layout.
- **10 MB seed reconstruction**: the model must be reconstructible from a
  10 MB compressed representation plus a deterministic decompressor. Forces
  procedural / generative / fractal / RNG-seeded constructions.
- **No floating point at inference**: only integer arithmetic + lookup tables.
  Forces pure-integer kernels (e.g., bit-serial AVX2, BitNet-shaped, lookup-
  table accumulator).
- **No persistent state**: every layer's weights must be reconstructed from
  the residual stream alone. Forces extreme weight-from-activation
  derivation (probably impossible — but the failure shape is a finding).
- **Storage is sequential-only**: random reads are forbidden; only forward
  scans allowed. Forces stream-oriented decode, decompression-fused-with-
  compute, NVMe-as-tape mechanisms.

Each constraint produces 0–2 candidate ideas. Pick the constraints whose
mechanism class is least represented in the current kill list.

### 6. Generated ideas (6–10)

Each in the same six-section format as Stage 1 (Ambition / Mechanism / Why our
findings make this work / What would have to be true / Experiment cascade /
What you're NOT proposing), tagged with a Stage 4 letter (S1, S2, ...).

Each idea must:
- Connect to a CF1–CF12 number or to a primitive on the actual stack (the
  same falsifiable-grounding constraint as Stage 1).
- Pass the CF9 precondition check **internally** (you're allowed to use any
  imported theorem, but you must verify its precondition for the target
  before writing it down — Stage 2/3 will check again, but a Stage 4 idea
  that fails CF9 is wasted slot).
- Pass the CF10 conditioning check **internally** for any calibration fit.
- Specify a smallest-test ≤8 hours.
- Self-classify Track A or Track B.
- Cite which orientation's anti-frame list it sits *outside* of (Stage 4
  ideas live in the cross-orientation white space; naming the white space is
  load-bearing).

**Stage 4 wildcard slot — `[FREE SWING]` allowed.** Of your 6–10 ideas, up
to **2 may be tagged `[FREE SWING]`** — proposals that connect to no prior
CF and to none of the surfaced cross-orientation primitives, if the
structural argument stands on its own. These are protected downstream the
same way Stage 1's wildcards are: Stage 5/6 will not penalize them for
lacking a CF tether. The structural floor still applies (CF9 internal
precondition check, CF10 internal conditioning check, smallest-test ≤ 8h,
primitive on the actual stack). Use the slot when a genuinely orthogonal
mechanism surfaces during your gap-finding pass; do not use it as a license
to drop the rigor floor.

**Elegance-class tag.** Where a generated idea fits the elegant-equivalence
class (algebraic-identity / calibration-fit / gauge-exploitation / subspace-
alignment / conserved-quantity / no-SGD-reformulation), tag it. Stage 5's
elegant-equivalence multiplier reads this tag at selection time.

The 6–10 spread should cover the gaps from sections 1–5, not concentrate in
one constraint or one professional prior. Discipline yourself to mix.

### 7. Output handoff

A short numbered list at the end:
- Stage 4 ideas S1..S10, each with a one-line description and the section
  (1–5) that motivated it.
- Recommendations to the round runner about orientation rotation: KEEP /
  TIGHTEN / DROP / INTRODUCE per the rules in `ROUND_OVER_ROUND.md`.
- Anti-frame additions to specific orientation lists, with one-line rationale.

---

## Calibration anchors from v1

- **R1/S4 Ideas A–J**: produced 10 candidate ideas across 6 cross-domain
  primitives (entropy coding, tensor factorization, sketches, fountain codes,
  succinct rank-select, content-addressable storage). Several survived to
  later rounds (Idea G later converged with Round 3's hybrid; Idea J as
  count-min variant). v1 demonstrated that cross-domain seeding works; v2
  generates from a richer substrate (multi-orientation kills) and should
  produce a tighter 6–10.
- **R2/S4 not selected**: W4A8-Stream, CLASP, LSEI, AVX2 W(ternary)A8, FCWR,
  UCLOCK, MFAP. Several deferred for round-3 candidacy. v2 should reduce
  the deferred-but-not-selected count by tightening upstream — the deferred
  ideas were good but the selector had to pick from too many; v2's Stage 5
  has the same 1-pick budget but fewer noise candidates if Stage 4 is
  disciplined.
- **R3/S4 Ideas G/H/A/E**: G was the convergence with R1's deferred Idea 5
  (RSTD cross-layer Tucker on W_up). v1 noted convergence-as-signal in the
  PIPELINE.md retrospective; v2 surfaces it earlier (Stage 2 convergence
  map) but Stage 4 may identify NEW convergences when generating across the
  full kill substrate. Flag any Stage 4 idea that re-derives a primitive
  from a Stage 1 advancer in a different orientation as a "second-pass
  convergence."

---

## Length and tone

1500–2500 words. Be specific about which frames are exhausted and which
are not. Generate ideas with the same rigor as Stage 1 — six-section format,
falsifiable grounding, smallest-test specified. The skeptic is not the
"throw out wild ideas" stage; it is the "find the white space and aim
deliberately at it" stage. Wild ideas without grounding are just noise the
selector has to filter; specific ideas in unsaturated frames are the
ladder's highest-leverage output.
