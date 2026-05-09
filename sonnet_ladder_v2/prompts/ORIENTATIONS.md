# Orientation Registry — Sonnet Ladder v2

Five orientations. Each Stage 1 round spawns one Sonnet ideator per active orientation, in parallel. Each ideator reads ONLY its own orientation block plus the shared inputs (`SUMMARY.md`, `KILL_LIST.md`, framing-diversity targets, saturated-frame list).

The five are tuned to be mutually distinct enough that two ideators in different orientations should rarely converge on the same idea by chance. When they DO converge, the convergence is a signal — the model's structure is independently asserting itself across creative orientations (this is the convergence-map mechanism from `opus_pipeline/round1/stage4_opus_synthesis_round1.md`).

**Why this set, and why five.** The Opus pipeline used four (Reach, Composition, First-Principles, Unconventional Angles). Those four cover the ambitious-cascade / cross-finding / unifying-derivation / substrate-primitive axes well, but they leave one explicit PIPELINE.md anti-saturation directive uncovered — **constraint-driven reframing** ("what if continuous numbers were forbidden? what if everything had to be content-addressable?"). The v2 ladder adds a fifth orientation, **Constraint-Alien**, that bites off that directive directly. Five is the cap: more orientations would dilute per-ideator depth without adding distinctive frames.

**These orientations bias generation; they don't constrain it.** An ideator that genuinely sees an idea outside its orientation should produce that idea anyway, marked as such (`[OUT-OF-ORIENTATION]` after the Name). The orientation is a creative prior, not a fence. If an idea hits on something that has nothing to do with anything we've done — including any prior round's findings — that's admissible too, especially if the structural argument works.

Every orientation block below has the same five fields, in this order:
1. **Name + tagline**
2. **Creative directive** (paragraph that biases generation)
3. **Anti-frame list** (mechanism flavors that suppress the orientation's known *failure modes* — not orthogonal good ideas)
4. **Cross-domain seed list** (which professional priors / fields to sample from inside this orientation)
5. **Falsifiability requirement** (the orientation-specific way that every proposal must connect to a primitive on the actual stack — CF1–CF12 anchors are a strong form of this but are not the only form)

---

## Orientation R — Reach

**Tagline.** Articulate the most ambitious useful outcome the empirical findings credibly support, and construct a multi-stage cascade that produces a structural finding at every rung.

**Creative directive.** State the audacious claim first — the one-line summary of what CF1–CF12 make tractable that the published landscape has not. Build a residency-arithmetic case for a 5–15× target on the 7.28 GiB Ryzen tier (Qwen3-72B / Llama-3-70B at ≥3 tok/s with quality within ~0.3 nat). Construct a cascade where every rung yields a number — even on NO-GO. Stack mechanisms multiplicatively only when the residency math forces it; flag every multiplicative composition as a P(success) deduction. The R1 Opus Reach proposal targeted 13× and the ambition was load-bearing: it forced choices the smaller ambition would not have surfaced. Reach is for the ceiling, not the floor.

**Anti-frame list.**
- "Incremental improvement on baseline" framings (1.2× targets — wrong orientation; pass to a different ideator).
- Single-experiment binary settlers — that is First-Principles' shape.
- Cascades whose intermediate stages have no informational value on NO-GO.
- Reach-via-stacking: 4+ independent novel sub-claims so that P(end-to-end) drops below 5% — instead, fold redundant claims into one mechanism, or split the proposal in two.
- Aspirations whose floor is what other orientations would consider their ceiling.
- **Cascades that hit their ratio purely by stacking quantize + prune + offload.** Admissible, but ambitious cascades benefit from at least one rung being an algebraic identity (AERO-style fold, structural commutation, calibration-derived closed-form substitution) — that rung buys its compute reduction without trading against quality, and P(end-to-end) survives better. This is a positive signal, not a gate: a cascade with a different kind of structural argument at one of its rungs (a substrate primitive doing unexpected work, a constraint-forced reformulation) is equally welcome.

**Cross-domain seed list.**
- Systems performance engineering and end-to-end ML deployment
- Hardware-software co-design (DRAM bandwidth, NVMe queue depth, cache hierarchy)
- Throughput-constrained optimization (closed-form per-operation budget arithmetic)
- Database-engine tier design (hot/warm/cold under cost-model)
- Compiler scheduling and pipeline-stage overlap
- Operations research on staged cascades

**Falsifiability requirement.** Every Reach proposal must include explicit numerical residency arithmetic anchored to CF1–CF12 numbers and to measured hardware constants (DRAM bandwidth, NVMe random-read latency, AVX2 throughput). State P(end-to-end success) explicitly. The first cascade rung must be a ≤1-day experiment that produces a measurable number whose failure mode is itself a structural finding. CF13–CF15 may be cited as motivation; if their numbers appear in the residency math, the first cascade rung must re-derive them.

---

## Orientation C — Composition

**Tagline.** Find non-obvious mechanisms that emerge from interactions between two or more empirical findings — mechanisms nobody who measured only one finding would predict.

**Creative directive.** Treat `SUMMARY.md` as a graph and look for edges other ideators missed. Your output is a coupling claim: "if both CF_i and CF_j hold on the same model, then a third structure C is forced." Write the coupling as math — an inequality, an alignment angle, a subspace intersection, a compositional bound. The R1 Opus Composition proposal coupled CF3's K-dependent outlier dynamicity with CF11's W_Q head-shared subspace into a single dim-alignment claim (`dim(D_ℓ^{1%} ∩ rowspan(W_Q^shared)) ≥ 0.6 × |D_ℓ^{1%}|`). That coupling is a falsifiable equation whose number is not on arXiv. Yours should have the same shape: two findings, one equation, one experiment that settles whether the equation holds.

**Anti-frame list.**
- **Stacking three published methods to claim a ratio.** Quantize × prune × distill is integration arithmetic, not composition — that's the failure mode. The orientation's strongest move is composing primitives whose *interaction* produces a structural identity (outlier highway + persistent channels coupling into a static partition + dynamic remainder; AERO-style activation removal composing with W_gate / W_up tied structure to yield a single fold; subspace-alignment between two findings producing a smaller joint operator than either alone). **Compositions whose components happen to be conventional are NOT excluded** — what matters is that the *interaction claim* between them is non-obvious and falsifiable. A composition of two conventional primitives whose joint behavior on this trained model is the load-bearing claim is in-orientation.
- Naming the composition without the math (gesturing).
- Stacking compressions multiplicatively without checking destructive interaction (e.g., assuming a 1.4× rotation gain composes with a 4-bit grid without measuring grid distortion).
- "Mixture of expert structures" / "hybrid attention" / other saturated frames the literature has chewed.
- Compositions whose preconditions hold for finding A but not finding B (CF9 risk — verify both findings' structural preconditions before composing them).

**Cross-domain seed list.**
- Statistical mechanics composition rules (mean-field interaction terms, replica composition)
- Tensor-network bond decomposition (MERA, MPS, PEPS bond-fusion rules)
- Information-bottleneck stacking (rate-distortion across compositional stages)
- Sheaf gluing and presheaf restriction (when one finding's local data restricts another's)
- Multi-tier storage hierarchy design (hot/warm/cold under joint constraints)
- Operator algebra (compositions of bounded operators, joint diagonalization)

**Falsifiability requirement.** Every Composition proposal must include the coupling claim as a labeled equation or inequality and must specify the cheapest measurement that tests the coupling — typically a single dot-product, projection, or alignment-angle computation that runs in minutes. Each component finding (the two CFs being composed) must have its precondition independently verified for the target object before the composition is asserted (CF9 enforcement). If either finding is CF13–CF15, the composition is double-conditional — both findings AND the unverified one must verify before the composition can be relied on.

---

## Orientation F — First-Principles

**Tagline.** Derive what should work from the structure already measured. Treat the empirical findings as theorems and look for the mathematical object — or the algebraic identity — that the trained model already satisfies but the prescribed computation graph does not exploit.

**Creative directive.** **The orientation's strongest move is an algebraic identity or no-SGD equivalence that reduces compute or storage without changing the trained net's input/output behavior on the calibration regime.** Weaker moves are admissible if grounded — what makes a proposal in-orientation is that it reasons from structure already measured (or already provable on this stack) toward a mechanism, rather than tuning an engineering knob. The archetype is AERO-style activation removal — once SiLU is shown to be effectively the identity on the relevant slice, W_gate and W_up algebraically fold into one matrix. Adjacent shapes: a unifying-object proposal ("CF_i, CF_j, CF_k are shadows of one structure X; compute in X-coordinates"); a calibration-derived closed-form substitution (NMF / generalized eigenvalue / polynomial-fit) that replaces an operator with a measurably equivalent cheaper one; a gauge-fixing argument that exposes a redundancy. The R1 Opus First-Principles proposal predicted an orthogonal Q* that simultaneously low-ranks W_gate, W_up, AND tied W_E — that is one strong shape, not the only one. The math is your moat when math applies; when the structural argument is non-mathematical (e.g., a thermodynamic-style accounting argument, a topological obstruction) that's still in-orientation as long as the derivation is checkable.

**Anti-frame list.**
- **Plain quantization or plain magnitude pruning as the load-bearing move.** These are engineering wins with a quality/cost tradeoff; they are not first-principles wins. They are admissible only as a *consequence* of an algebraic identity (e.g., "after the fold, the surviving matrix has spectrum X, so it tolerates lower-bit storage"), never as the move itself.
- Importing a named theorem without verifying its preconditions for the target (CF9 — Bochner on weight matrices, Count-Sketch on dense residuals, etc.).
- Math-first proposals with no measurable falsification path in <1 day. If E1 takes 8 hours you are doing it right; if E1 is "implement the whole thing first then measure" you are doing it wrong.
- Calibration-fitted parameter counts that exceed independent samples (CF10 — WDLA failure mode). Always state n_params / n_samples explicitly.
- Beautiful math with no engineering payoff. The orientation is *grounded* derivation; if the cascade does not connect to a measured tok/s or a measured ΔNLL, the orientation has drifted into pure math.
- Multi-step derivations with hand-waved transitions. Each step must be a one-line check (e.g., "Q* preserves softmax invariance because Q*Q*^T = I"). No magic.

**Cross-domain seed list.**
- Riemannian geometry (manifold structure on residual streams, geodesic computation)
- Representation theory (group symmetries on weight matrices; Z_n head-symmetry)
- Operator algebras (joint diagonalization, simultaneous decomposition)
- Statistical mechanics phase transitions (criticality on neural-network avalanche scaling)
- Differential geometry (sheaf / fibration / stratification of activation spaces)
- Algebraic information theory (rate-distortion as a derived quantity)
- Spectral graph theory (when the residual stream's covariance is a graph Laplacian)

**Falsifiability requirement.** Every First-Principles proposal must include a binary settler experiment — one number that comes out cleanly and either confirms or refutes the unifying-object hypothesis. The settler must run in ≤1 day on Qwen3-1.7B (or smaller) on the Ryzen target. The proposal must explicitly state the number that distinguishes "unifying object exists" from "does not exist" (e.g., "Q* such that ‖W_E Q* − low-rank-r‖_F < threshold for r=256 vs the +19.96 ambient-basis catastrophe"). The number must be derivable from CF1–CF12 (or from CF13–CF15 with the unverified-tag and a re-derivation rung).

---

## Orientation U — Unconventional Substrate

**Tagline.** The standard ML playbook is exhausted in the saturation region. Look at what the model runs on (OS, filesystem, NVMe controller, CPU microarchitecture, format codecs, network stack) and ask what those layers can do that ML hasn't asked them to do.

**Creative directive.** Your output is a "let the substrate do the work" proposal. Identify a primitive that exists in a non-ML stack (NTFS extent prefetch, working-set trimmer LRU, AVX2 PEXT/PDEP, NVMe queue depth scheduling, denormal sentinels, cache-line locality, memory-mapped I/O ordering, Windows VirtualLock, Linux madvise, XOR-list compression, content-addressable storage hashes, NIC zero-copy paths) and ask: what could it do for inference if we laid the bytes / control flow / access pattern out for it? The R1 Opus Unconventional proposal recognized that the OS demand-paging path with NTFS extent prefetch is *itself* the predictor for which weights to load. The audacity is in **recognition** — the substrate has been doing the work for 30 years and ML didn't notice.

**This orientation has the widest creative range of the five.** Substrate-level moves often have no connection to model-internal findings (CF1–CF12 or otherwise) — that is the orientation working as designed. A substrate primitive that nobody has pointed at LLM inference yet does not need to map onto any current finding to be in-orientation; it needs to be a real primitive on the real stack with a measurable engagement claim.

**Anti-frame list.**
- Custom kernels or new SIMD intrinsics. The orientation is about USING what exists, not building new low-level code.
- "Hardware-aware ML" framings that mean "tile the matmul to L2." That is standard practice. Unconventional means using a substrate primitive ML does not invoke.
- iGPU compute / SIMD micro-optimizations as the load-bearing mechanism. These are fine as free side-channels but cannot be the core.
- Anything requiring an OS kernel modification or custom NVMe firmware — the orientation exploits standard substrate behavior, not modified substrate.
- Substrate primitives whose engagement on the actual workload is unverified — always state the empirical measurement that confirms the primitive engages on this hardware (and if that measurement is CF13–CF15, mark unverified and re-derive in cascade rung 1).

**Cross-domain seed list.**
- Kernel-level memory management (page cache, working-set trimmer, demand paging)
- Storage-engine internals (NTFS Cache Manager, ext4 extent allocator, btrfs CoW, B-tree page layout)
- Database-engine page layout and cost models (PostgreSQL TOAST, MySQL InnoDB row format)
- Codec design (zstd dictionaries, ANS, Huffman, LZ4 frame format)
- CPU microarchitecture (branch prediction, prefetcher, store buffer, cache-line ownership)
- Virtual memory subsystems (madvise, VirtualLock, page table walks)
- File format on-disk layouts (Parquet, Arrow, ORC, GGUF tensor offset tables)
- Filesystem features (reflinks, dedup, compressed clusters)

**Falsifiability requirement.** Every Unconventional Substrate proposal must (a) name the substrate primitive precisely (down to the API call or the kernel subsystem), (b) include a smallest-experiment that measures whether the primitive engages on the actual workload (not the spec-sheet behavior — measured behavior under Qwen3-class inference on the Ryzen 5 7530U), and (c) state the empirical fallback if the substrate primitive does not engage as predicted. The proposal must NOT use CF13's NTFS extent-prefetch numbers as a load-bearing premise without including a v2 re-derivation in the cascade.

---

## Orientation A — Constraint-Alien

**Tagline.** Pose the problem under a constraint the published prior does not impose, and let the constraint force a mechanism the saturated literature has no vocabulary for.

**Creative directive.** Pick ONE constraint from the menu below (or propose a new constraint that the round runner has not yet seen) and let it bind generation through the entire mechanism — not as a fig-leaf preface, but as a hard binding that rules out the standard moves. The constraint's job is to force the ideator into a frame the published prior does not have words for. **The whole point of this orientation is to surface things from outside the current frame** — proposals here may have no connection to current findings (CF1–CF12 or anything else), and that is the orientation working as designed. The constraint is what does the grounding work; CF connection is welcome but not required. The PIPELINE.md "constraint-driven reframing" anti-saturation directive is this orientation's entire orientation.

**Constraint menu (pick one and let it bind hard).**
- **No continuous numbers during inference.** Everything integer / rational / content-addressable / table-lookup. (Forces: discrete codebooks, integer paths, tropical algebra, table dispatch.)
- **The model must be reconstructible from a 10 MB seed.** All weights derivable from a small generative procedure. (Forces: hypernetwork-style synthesis, fractal/self-similar weight construction, procedurally-generated layer parameters.)
- **Every weight must be content-addressable by its action.** No spatial layout; weights retrieved by hash of "what they do." (Forces: action-keyed dispatch, learned hashing of weight semantics.)
- **No RAM during decode — only registers, L1, L2, L3, NVMe.** (Forces: streaming-only computation, register-allocation-as-residency, cache-line as state unit.)
- **The inference loop must fit in a state machine of N states for small N.** (Forces: discretization of internal state, FSM-as-decoder.)
- **The model cannot hold its own weights — only a peer's.** Inference is a swap operation. (Forces: distributed-state primitives, gossip / vector-clock / CRDT framings.)
- **Answer in O(1) tokens regardless of context — no autoregression.** (Forces: parallel decoding, single-shot reformulation.)
- **All state is append-only.** No mutation during inference. (Forces: log-structured attention, monotonic KV growth with idempotent retrieval.)
- **The model must be debuggable to a step level by reading bytes.** No opaque kernels. (Forces: explicit per-step state representation, transparent on-disk layout.)
- **The computation graph must algebraically simplify under <some symmetry>.** Pick a symmetry the trained weights might respect — head-permutation invariance, residual-stream gauge rotation, channel sign-flip, layer-tie equivalence, softmax shift-invariance — and require that the inference computation be expressible in coordinates where that symmetry collapses out. (Forces: gauge-fixing arguments, joint-diagonalization moves, fold/unfold identities — the constraint pressures the ideator toward elegant computational equivalences rather than engineering tweaks.)

**Anti-frame list.**
- Picking a constraint and then sneaking back to the standard playbook. ("No continuous numbers... so I'll quantize" — that is the standard playbook with a fig leaf. The constraint must remain binding.)
- Constraints that are slogans rather than mathematical bindings. "What if it were elegant" is not a constraint. The constraint must be mechanizable.
- Multiple constraints stacked into a vacuous problem. Pick ONE and let it bind hard.
- Cosmetic constraints that don't actually rule out any standard mechanism (e.g., "what if the model used SIMD" — every CPU model already does).
- Constraints whose binding is only at proposal-time but not at inference-time (the constraint must affect the actual computation, not just the framing).

**Cross-domain seed list.**
- Algorithmic information theory (Kolmogorov complexity, minimum description length)
- Tropical algebra (max-plus / min-plus semirings as alternative to floating-point)
- Content-addressable storage systems (Git objects, IPFS, BLAKE3-keyed dispatch)
- FPGA fixed-resource computing (no dynamic allocation, registers and LUTs only)
- Hypernetwork compression (small generator network synthesizing large target)
- Distributed systems consistency models (CRDTs, vector clocks, gossip protocols)
- Finite automata theory (Myhill-Nerode, minimization, regular-language inference)
- Reversible computation (Bennett-style, no-information-loss inference)

**Falsifiability requirement.** Every Constraint-Alien proposal must (a) name the constraint precisely and state it as a mechanizable binding (not a slogan), (b) calibrate the constraint difficulty — under your constraint, what fraction of the standard mechanisms become inadmissible? Aim for 30–95%; outside that range the constraint is too easy or too hard, recalibrate. (c) Connect the resulting mechanism to a primitive on the actual stack (CPU, RAM, NVMe, the trained model file, an existing library, an existing math operation, an OS subsystem). A CF1–CF12 connection is welcome but not required for this orientation. The constraint forces the frame; the stack-primitive connection keeps the proposal testable.

---

## Coverage check

The five orientations cover:

| Axis | Orientation |
|---|---|
| Ambitious-cascade-builder | R — Reach |
| Cross-finding-importer | C — Composition |
| Math-grounding | F — First-Principles |
| Engineering-substrate | U — Unconventional Substrate |
| Constraint-driven reframing | A — Constraint-Alien |

Reach owns the multi-stage stacked play. Composition owns the cross-finding edge. First-Principles owns the unifying-object derivation. Unconventional Substrate owns the substrate primitive. Constraint-Alien owns frames the literature has no vocabulary for.

The C / F edge is the closest pair (both reason from `SUMMARY.md` directly) and is intentionally distinguished by output shape: Composition produces interaction claims (two findings, one equation, one experiment); First-Principles produces a unifying-object hypothesis (many findings, one mathematical object, one binary settler).

The U / A edge is the next-closest (both bypass the standard ML playbook) and is distinguished by direction: Unconventional reaches outward to existing engineering substrate (the move is recognition); Constraint-Alien reaches inward by imposing an alien constraint (the move is invention).

---

## Universal "think big" directive (applies to every orientation)

Within your orientation, reach into adjacent intellectual territory. Information theory, statistical mechanics, quantum information primitives, abstract math, hardware/OS reformulations, codecs, coding theory, database engineering, FPGA design, distributed systems, signal processing. The published ML compression playbook is a small subset of the available primitives. Sample broadly inside your orientation; let the cross-domain seed list above guide the sampling.

The audacity floor is **not "improve baseline by 20%."** The audacity floor is **"run a 70B-class model at usable tok/s on 7.28 GiB without retraining."** Frame your idea against that floor. If a smaller idea is the right one, frame it as a stage in a cascade aimed at the floor.

The hard constraint that keeps audacity honest: every load-bearing claim must connect to a primitive that exists on the actual stack — CPU, RAM, NVMe, the trained model file, an existing library, an existing math operation, an OS subsystem. A measured number from `SUMMARY.md` is a strong form of grounding when present, but it is not the only form; an idea orthogonal to every CF can still be admissible if it is testable on this stack. CF13–CF15 are NOT confirmed anchors and must be re-derived in the cascade if their numbers appear in your residency math.

---

## Round-over-round evolution

See `ROUND_OVER_ROUND.md` for the rotation rule. In short: orientations whose Stage-5 selections produced GO results in Round N stay; orientations whose proposals all died at Stage 2 / 3 get tightened anti-frame lists for Round N+1; new orientations may be defined per-round if the previous round's transcript flagged a frame as under-explored. The orientation set must remain ≥3 to preserve diversity.
