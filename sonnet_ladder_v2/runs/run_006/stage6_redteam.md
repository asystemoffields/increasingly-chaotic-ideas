# Stage 6 — Red Team — Run 006

**Agent**: Sonnet claude-sonnet-4-6 | **Date**: 2026-05-09
**Inputs read**: stage5_selector.md, stage3_deep_research.md (sections A/C/E), SUMMARY.md, KILL_LIST.md, prompts/STAGE6_RED_TEAM.md
**Prior-art search**: conducted 2026-05-09 (WebSearch)
**Candidates under review**: Track A — F6-WALIGN; Track B — C-WVOK
**Runner-up references**: Track A — C-JOINT-DELNLL; Track B — C-DEEP-SPREAD-QBUD

---

## TRACK A: F6-WALIGN

---

### 1. Frame-Mismatch Re-check (CF9)

**Imported mechanism**: the algebraic identity H1 — "if cos(W_gate[i,:], W_up[i,:]) = 1 then W_up[i,:] = λ_i · W_gate[i,:] and W_up is eliminable." This is a trigonometric fact about collinear vectors, not an imported theorem. No external preconditions.

**Independent verification**: for any two vectors u, v in R^d: cos(u,v) = 1 iff u = λv for λ = ||u||/||v|| > 0. For the SwiGLU neuron: output_i = silu(u^T x) · (v^T x). If v = λu then output_i = silu(u^T x) · λ(u^T x) = λ · z_i · σ(z_i) where z_i = u^T x. W_up row is redundant and can be replaced by the scalar λ_i stored alongside the retained W_gate row. This is verified independently of Stage 3 and holds without any further structural assumption. **No CF9 issue.**

**One subtlety Stage 3 noted but did not stress-test**: the λ_i scalar multiplication must be applied at inference time to the silu output, not folded into W_down. If W_down[i,:] is used to fold λ_i at load time (scaling W_down's corresponding column by λ_i), the identity holds. If the implementation instead scales W_up by 1/λ_i to normalize it to unit norm before folding, that changes nothing. Both are correct. The plan does not specify which folding path to use; the amendment slot should require this to be specified in the script before ΔNLL measurement. Minor implementation detail, not a frame mismatch.

---

### 2. Calibration Ill-Conditioning Re-check (CF10)

The experiment has **no calibration fitting**. Step 1 is deterministic weight arithmetic (cosine of row pairs). Step 2 ΔNLL is a forward-pass measurement, not a regression. CF10 does not apply. No re-derivation needed.

Step 2 does draw a calibration corpus slice (WikiText-2, 512 tokens) — this is for ΔNLL, not for fitting parameters. The eval corpus (3 seeds × 512 tokens) is disjoint from the calibration specification because the experiment has no calibration step. **CF10 clean.**

---

### 3. Skeptic-Controls Check

**Hypothesis shape**: H2 is a "structure is present in trained model" claim, not a transfer/consistency claim. However it does assert trained structure exceeds random-init structure. The controls are present.

**Permutation control**: present and correctly specified. The plan permutes W_gate rows (not columns), which changes the pairing (row i of W_gate no longer paired with row i of W_up) while preserving the row-vector population of both matrices. The permuted cosine distribution tests whether the alignment is a paired property or a global property of the row-vector populations. The GO threshold requires trained-vs-permuted gap ≥ 0.20 in the fraction with |cos| ≥ 0.90. **Adequate.**

**Random-init control**: present. Torch random init of same shape, same histogram. GO requires trained fraction exceeds random-init fraction by ≥ 0.10 absolute. **Adequate.**

**Multi-seed**: correctly marked N/A for the weight-arithmetic step (deterministic). For Step 2 ΔNLL, 3-seed multi-slice is specified with mean ΔNLL < 0.05 and worst-of-3 < 0.10. **Adequate.**

**One gap**: the permutation control gap threshold of ≥ 0.20 is stated in Section 9 but NOT integrated into the GO criterion in Section 5. Section 5's GO criterion is "≥ 5% of neurons with |cos_trained| ≥ 0.90" — no mention of the permutation gap. A result could technically pass Section 5's GO while failing the Section 9 permutation gap, which would mean the alignment is a population property (W_gate rows are generally similar to W_up rows regardless of pairing) rather than a paired property. The permutation control must be a blocking condition in the GO criterion, not just documented in Section 9.

**Required amendment**: add to Section 5 GO criterion: "AND gap(fraction_trained - fraction_permuted) ≥ 0.20 at |cos| ≥ 0.90 threshold." This is a structural amendment, not a frame issue.

---

### 4. Hidden Prior-Art Search

**Primary search conducted 2026-05-09.**

**Critical finding — ICLR 2025 Smooth-SwiGLU paper (Yadav et al., "Scaling FP8 Training to Trillion-Token LLMs")**: This paper explicitly documents that W_gate and W_up row vectors **become mutually aligned during extended training** in SwiGLU networks. The paper observes that w_1 and w_2 (gate and up projections for outlier channels) show initially low correlation, but correlation **increases drastically** between 125B and 210B tokens of training, producing collinear outlier channels that cause FP8 training instability. The paper does NOT report a per-neuron cosine histogram across all neurons (it focuses on the outlier-channel subset that causes instability), and it does not measure the fraction of aligned pairs as a compression primitive. This paper is adjacent to F6-WALIGN's claim: it demonstrates that gate/up alignment **does occur** in trained models (supporting H2's hypothesis that alignment fraction ≥ 5% is possible), but it characterizes this as a stability problem rather than a compression opportunity, and it does not measure the global per-neuron cosine distribution.

**Assessment**: This paper strengthens WALIGN's prior probability (gate/up alignment happens in trained models, at least for outlier channels) rather than pre-empting it. The WALIGN experiment asks a different question: what is the global fraction of aligned neurons (not just outlier channels), and can the aligned ones be losslessly folded? The Yadav et al. paper does NOT measure this global fraction, does NOT propose compression via folding, and focuses on FP8 instability mitigation. WALIGN's M2 claim (global fraction ≥ 5%) and its compression application are NOT pre-empted.

**However, Stage 3 and Stage 5 missed this paper entirely.** The red team flags it as required reading for the experiment: if the per-neuron cosine histogram shows alignment concentrated in the same channels Yadav et al. identified as FP8-outlier channels, the "alignment = compression opportunity" interpretation must be nuanced (those same neurons produce extreme activations; folding them may change the model's outlier-activation statistics in ways that affect downstream INT4/INT8 quantization). This is a downstream concern, not a GO/NO-GO gate for the current histogram experiment. **No pre-emption. Flag as context for interpretation.**

**Secondary search**: no paper found that measures per-neuron cos(W_gate[i,:], W_up[i,:]) as a global distribution statistic across all neurons. WALIGN's M2 remains NOVEL. Date-stamped 2026-05-09.

**AERO prior art**: unchanged from Stage 3. AERO (arXiv:2410.13060) folds globally after full activation removal; WALIGN folds per-neuron without removing the activation. Not pre-empted.

---

### 5. Biased-Framing Audit

**GO threshold (≥ 5% aligned neurons)**: grounded in residency arithmetic (34 MB at 1.7B, 920 MB at 70B). The 5% threshold is not gerrymandered; it is the minimum at which the mechanism produces measurable residency benefit at 70B. Clean.

**NO-GO threshold (< 1% AND mean |cos| < 0.10)**: the compound AND condition creates a potential gray zone. If < 1% but mean |cos| = 0.15, the experiment falls into GRAY without triggering the class-level NO-GO structural finding. Given the ICLR 2025 Yadav et al. finding (alignment occurs in at least outlier channels), a mean |cos| near zero across all neurons is quite possible since the aligned fraction may be small. The AND condition is appropriate — both conditions are required for the structural claim "SGD maximizes angular diversity." If only the < 1% condition holds but mean |cos| = 0.30, the correct interpretation is "alignment is diffuse but not peaked; no large aligned cluster." The NO-GO structural finding statement covers this in the second sentence. **Framing adequate.**

**Elegant-equivalence multiplier (1.2)**: justified. The algebraic identity is exact at cos=1 and near-exact at cos≥0.90. The approximation error at cos=0.90 is bounded by ||W_up[i,:] - λ_i W_gate[i,:]||/||W_up[i,:]|| = sqrt(1 - cos²) ≈ 0.44 of the W_up row norm. That is NOT small — at cos=0.90, the residual is 44% of the W_up row norm. The plan acknowledges this implicitly in Section 6's ΔNLL tolerance (< 0.05 nats), but Stage 5 should not claim the fold is "near-exact" at cos=0.90 — it is 44% error in the row vector approximation. The elegance claim is valid only at cos ≥ 0.99 (residual ≤ ~14%) or cos = 1 (exact). The experiment should report separately: fraction at |cos| ≥ 0.99 (near-exact fold), |cos| ≥ 0.95 (10% residual), |cos| ≥ 0.90 (44% residual). The current plan bins the last two together. This is not a frame issue, but the claim "near-exact at cos≥0.90" is misleading and should be amended.

**Required amendment**: add a |cos| ≥ 0.99 bin to the histogram (separate from the ≥ 0.90 bin); report fraction for each. The "lossless per-neuron compression" language applies only to the ≥ 0.99 subpopulation; the ≥ 0.90 population should be described as "approximate fold, ΔNLL measured empirically."

---

### 6. Runtime / Scope Sanity

**Step 1 (pure tensor arithmetic, 28 layers × 6144 pairs)**: 5-min estimate is plausible. Load 1.4 GB BF16 (~30 sec), row-normalize + einsum (168K cosine computations, each over 2048-dim vectors): ~4 min on Ryzen 5 7530U. Permutation control: same arithmetic, negligible overhead (row shuffle + re-einsum). Random-init control: 2 min to generate + compute. Total Step 1 with controls: ~8 min. Section 10's runtime table states ~8 min total for Step 1 with controls. **Matches the work described. Well within 8-hour ceiling.**

**Step 2 (GO path, single-layer fold + ΔNLL)**: 10-15 min estimate for one layer forward pass is plausible. Multi-seed (3 slices): 20 min. Total GO path: ~38 min. **Reasonable.**

**Disjoint calibration/eval corpora**: no calibration corpus. WikiText-2 held-out for ΔNLL only. 3 seeds draw disjoint 512-token slices. **Clean.**

**Smallest test produces the GO/NO-GO number**: Step 1 produces the primary gate (fraction ≥ 5%). This is a single number from a deterministic computation. Step 2 is gated on GO. **No ambiguous zone that requires a follow-up to resolve the primary gate.**

---

### 6c. Elegant-Equivalence Ground

Elegance tag: `algebraic-identity`. The identity depends on: (a) W_up[i,:] = λ W_gate[i,:] exactly (cos=1); (b) the SwiGLU activation is applied element-wise. Both preconditions hold trivially — (a) is the definition of the alignment claim, (b) is the architecture specification. The elegance claim is sound for cos=1. **No section 1 precondition failure on the algebraic identity itself.**

The 6c risk is the approximation at cos=0.90 (44% residual in the row approximation), which inflates the elegance multiplier's value at the empirically tested threshold. As noted in Section 5, the ≥ 0.99 bin should be the lossless-compression anchor; the ≥ 0.90 bin is an approximation experiment. The elegance multiplier was applied to the whole proposal at Stage 5; Section 6c says this multiplier cannot paper over a math error. Here the math is not wrong — cos=1 is exactly treated, cos<1 is an approximation and the plan correctly calls for ΔNLL measurement to characterize it. **No rejection ground.** The biased-framing amendment (separate ≥ 0.99 bin) is sufficient.

---

### 7. CF13–CF15 Verification Check

No CF13–CF15 dependence anywhere in the plan. The residency arithmetic uses known model dimensions (172K neurons, 2048 d_model). **Clean.**

---

### 8. Verdict — Track A

**ACCEPT-WITH-AMENDMENTS**

Amendments required before execution:

1. **A1 (skeptic-controls gap into GO criterion)**: Add to Section 5 GO criterion: "AND (fraction_trained − fraction_permuted) ≥ 0.20 at the |cos| ≥ 0.90 threshold." The permutation gap is currently in Section 9 only; it must be a blocking GO condition, not a decorative check.

2. **A2 (histogram binning for lossless vs approximate fold)**: Add a |cos| ≥ 0.99 bin to the histogram output. Report separately: fraction at |cos| ≥ 0.99 (lossless fold, ~0% approximation error) vs ≥ 0.90 (44% row residual, ΔNLL-characterized). The experiment plan must not describe the ≥ 0.90 population as "near-exact" — it is a lossy approximation whose acceptable threshold is determined empirically by ΔNLL.

3. **A3 (fold path specification)**: Script must specify which folding path applies the λ_i scalar at inference: either (a) scale W_down's corresponding column by λ_i at load time, or (b) store λ_i alongside W_gate row and apply at inference. Both are correct; the choice affects memory layout and must be documented before the ΔNLL step to ensure the measurement reflects actual deployment overhead.

**Context flag (not a blocking amendment)**: The ICLR 2025 Yadav et al. paper demonstrates gate/up alignment in trained SwiGLU, but only for FP8-outlier channels. If the WALIGN histogram shows alignment concentrated in the same channels as quantization outliers (per CF3/PDAP data on Qwen3-1.7B), the compression opportunity and the quantization-stability concern coincide. This is not a GO/NO-GO gate; it is an interpretive note for post-experiment analysis.

---

---

## TRACK B: C-WVOK

---

### 1. Frame-Mismatch Re-check (CF9)

**H1 coupling equation**: "CF11's head-redundancy finding (16 Q heads → 128-dim subspace) implies dim(effective_value_subspace) ≤ 128, predicting r_99(W_V)/d < 0.80."

This is a **derived inequality, not an imported theorem**. The Stage 3 derivation is:

> "if attention weights are dominated by ~128 effective query directions (CF11), then the effective value outputs accessed per token lie in the span of at most 128 token-positions × W_V, which compresses under SVD if W_V itself has concentrated spectrum."

**Red-team verification of this coupling**:

The argument conflates two distinct objects: (1) the low-rank structure of W_Q (query projection), and (2) the rank of W_V (value projection). The coupling assumes that because queries live in a 128-dim subspace, the value output v = W_V · h_i for attended positions h_i is also effectively low-dimensional. But this is not correct in general:

- W_Q head-redundancy means the *attention weight vector* a = softmax(QK^T/√d) is determined by a 128-dim projection of each query. This constrains the **weighted sum** ∑_i a_i v_i.
- However, the individual value vectors v_i = W_V · h_i are determined by W_V, not by W_Q. W_V can have full rank regardless of W_Q's rank.
- The span of the weighted sum ∑_i a_i (W_V h_i) = W_V (∑_i a_i h_i) is bounded by the rank of W_V applied to the weighted average of residual streams, not by the rank of W_Q.

**The coupling equation's claim — that W_Q head-redundancy implies W_V spectrum concentration — is NOT algebraically derived from CF11.** It is a hypothesis about how training dynamics correlated the two projections, not a structural necessity. The correct logical status of H1 is: "it is plausible that SGD jointly compressed both Q and V subspaces because they interact in the same attention operation" — but this is a training-dynamics conjecture, not a mathematical coupling.

This is NOT a fatal frame mismatch because:
- The plan does not actually rely on the coupling equation being mathematically necessary; it uses H1 as motivation for a prior probability and then empirically tests H2 (r_99 < 0.80) and H3 (ΔNLL < 0.40).
- The experiment is valid regardless of whether the coupling equation is correct.
- Stage 3 correctly identified the main risk as "the assumption may fail if W_V encodes content-specific routing independent of W_Q" — which it does.

**But the Stage 5 plan mislabels the coupling equation as a "specific quantitative prediction" from CF11 in H1.** The quantitative prediction (r_99 < 0.80) is a hypothesis, not a CF11-derived prediction. The plan should represent this as an empirical hypothesis motivated by CF11 and 2604.22778, not as a coupling-equation derivation. The framing in the experiment plan overstates the theoretical grounding. **This is a biased-framing issue (see Section 5) not a CF9 kill.**

**2604.22778 (Spectral Lifecycle) frame mismatch check**: the paper is cited as supporting H1. The paper documents that V/O projections compress uniformly during training on GPT-2/Pythia — not Qwen3. Architectural differences (GQA, RoPE, grouped keys, different d_model/d_head ratios) make this a weak analog. The paper also reports training-time spectral behavior, not post-training measurement. No precondition failure from importing this paper's claim; it is used as prior probability, not as a theorem. **No CF9 issue from 2604.22778.**

---

### 2. Calibration Ill-Conditioning Re-check (CF10)

No calibration fitting. Pure SVD + ΔNLL. CF10 not applicable.

ΔNLL Step 2: forward pass with SVD reconstruction. No parameter fitting; the U, Σ, V^T factors from SVD are computed analytically. No calibration corpus conditioning. **CF10 clean.**

---

### 3. Skeptic-Controls Check

**Hypothesis shape**: H1/H2 are "W_V has concentrated spectrum, structurally different from random" claims. Controls are present.

**Permutation control**: Section 9 correctly identifies that row permutation does NOT change singular values (permutation is a unitary transformation on the left factor; singular values are invariant). The plan correctly documents this as not applicable and proposes the random-init control instead. **Permutation N/A: documented and correctly justified.**

**Random-init control**: present. Marchenko-Pastur bulk edge for 1024×2048 (aspect ratio 0.5) predicts r_99 ≈ 0.95–0.98 for random init. GO requires trained r_99 < 0.80 AND random-init r_99 ≥ 0.90. The gap (0.80 vs 0.90) is tight — only 0.10 absolute separation is required. If trained r_99 = 0.81 (just above GO boundary) and random-init r_99 = 0.93, the random-init control passes but the GO criterion fails. This is acceptable — the control is checking whether the trained structure is distinguishable from random, and the GO criterion is stricter. **Adequate, but see biased-framing note below.**

**Multi-seed**: correctly marked N/A for SVD (deterministic). Step 2 ΔNLL: 3 seeds × 2 K values specified. **Adequate.**

**One gap**: the GO criterion in Section 5 does NOT explicitly include the random-init control separation. It states "r_99(W_V[14])/d < 0.80 AND ΔNLL(K_V=512) < 0.40 nats" — the random-init gap is implicit. The Section 9 specification should be a blocking condition in Section 5.

**Required amendment**: Add to Section 5 GO criterion: "AND r_99(W_V_random_init)/d ≥ 0.90 (confirming trained structure is distinguishable from Marchenko-Pastur baseline)."

---

### 4. Hidden Prior-Art Search

**Primary A3 pre-emption check (arXiv:2505.12942)**:

Stage 5 explicitly flags: "Stage 6 must verify A3 does NOT report Qwen3-specific r_99(W_V) numbers in its paper body or appendix." The A3 paper ("A3: an Analytical Low-Rank Approximation Framework for Attention," arXiv:2505.12942) focuses on LLaMA architectures (MHA without RoPE, MHA-RoPE for LLaMA 1&2, GQA-RoPE for LLaMA 3.1) and demonstrates 4.69 perplexity on WikiText-2 at 70B scale. The paper works on the **fused W_VO product** and GQA-RoPE architectures but evaluates primarily on LLaMA 3.1-70B. Web search does not confirm Qwen3-specific r_99(W_V) tables; A3's target family appears to be LLaMA, not Qwen3. **A3 does not appear to report Qwen3-specific W_V spectral numbers. C-WVOK's Qwen3-specific M2 measurement remains novel.**

**Second critical prior-art finding — W_V compression resistance literature**:

Multiple independent SVD-LLM papers (ARA, arXiv:2510.19389; SVD-LLM V2, NAACL 2025; ARA literature review, Oct 2025) consistently report that **W_V resists compression** in empirical studies, with value and down projections left uncompressed for capability preservation. Specifically:

- ARA (arXiv:2510.19389, Oct 2025): "the value, gate, and down projections are often left uncompressed, with original matrices retained to preserve essential information." Tests include Qwen3-8B.
- Multiple NAACL/ICLR 2025 papers consistently confirm W_V in LLaMA2-7B has high rank requirement.

This empirical literature does NOT pre-empt C-WVOK's measurement (those papers do not report per-model r_99/d numbers that would directly answer the question for Qwen3-1.7B). But it constitutes a **strong external signal that the NO-GO outcome is likely**: published SVD compression papers consistently find W_V requires higher rank retention than W_Q/W_K, opposite to H1's prediction.

**This is not a pre-emption kill** — C-WVOK's result is still novel for Qwen3-1.7B specifically and the coupling-equation framing (M1) is not in the literature. But the prior probability for H2 (r_99 < 0.80) is significantly lower than Stage 5 estimated based on 2604.22778 alone.

**Spectral Lifecycle 2604.22778 re-check**: confirmed as "V/O compress uniformly during training" — this is training-time spectral behavior. Published post-training SVD compression results (ARA, SVD-LLM) override this as the stronger signal about post-training W_V rank properties. The 2604.22778 finding likely describes a different property (within-training spectral evolution) that does not predict post-training r_99 for compression purposes.

**Net prior-art verdict**: no pre-emption, but the prior probability for GO is lower than Stage 5 estimated. The NO-GO outcome (W_V full-rank like ARA/SVD-LLM literature finds empirically) is the mode outcome. **The experiment is even more valuable in the NO-GO case** — it would confirm and extend the ARA finding to the Qwen3 family with the CF11-coupling interpretation.

---

### 5. Biased-Framing Audit

**GO threshold (r_99 < 0.80 AND ΔNLL < 0.40)**: The 0.80 threshold is set at W_K's observed r_99 (0.79), described as "H1 predicts W_V should be MORE concentrated than W_K." But the coupling equation in H1 does not algebraically predict this — as noted in Section 1, the coupling is a training-dynamics hypothesis, not a structural derivation. The threshold is therefore not derived from a prediction; it is set at "at least as concentrated as W_K" as a pragmatic level. This is defensible framing as long as it is not presented as a theorem-derived threshold. Amend Section 5 to state: "threshold set at W_K's empirical boundary (0.79) as a conservative baseline; H1's prediction motivates but does not prove this level."

**"H1 predicts W_V should be MORE concentrated than W_K"** (Section 5): this is the most problematic framing statement in the plan. The coupling equation argument does not algebraically force r_99(W_V) < r_99(W_K). It hypothesizes that query-side redundancy might propagate to value-side redundancy through training dynamics. The published compression literature (ARA, SVD-LLM) suggests W_V is LESS concentrated than W_K in practice. The GO threshold should NOT be described as derived from H1; it should be described as the empirical test point for H1.

**NO-GO threshold (r_99 > 0.90)**: GRAY zone spans 0.80–0.90. This is an unusually wide gray zone (10% of normalized rank). The gray follow-up (ΔNLL at K_V=1024) is specified and adds 30 min. Given that published literature suggests W_V is likely above 0.80, a substantial fraction of outcomes would land in GRAY. The gray follow-up plan is adequate.

**"Most information-dense NO-GO in the run"** claim: confirmed — it simultaneously gates R6-WVOP and R6-ATTNQ-FULL, sharpens CF11, and anchors the 70B residency arithmetic. Even if W_V turns out to be difficult to compress (the literature-predicted NO-GO outcome), the measurement is the correct next step given the convergence cluster. **No biased framing that over-promises; the NO-GO bonus is genuinely large.**

**Per-head vs full-matrix concern (Section 8 kill criterion 3)**: Stage 5's plan correctly identifies this as a potential structural mismatch but treats it as an escalation criterion rather than a baseline requirement. Given GQA's architecture (n_kv_heads=8, each head's W_V is 128×2048), the compression granularity that matters for deployment is the per-head block (128×2048), not the full concatenated matrix (1024×2048). If per-head SVD is needed for actual compression but the experiment only reports full-matrix SVD, the GO criterion (r_99 of the full matrix < 0.80) may not translate to deployable compression. The per-head bonus SVD (Section 4) should be promoted from a bonus to a required measurement to ensure the GO criterion is actionable.

**Required amendment**: Promote the per-head SVD sub-experiment from "Bonus" to required (Step 1b). Add to GO criterion: "AND r_99 of per-head W_V block (128×2048) / d_head < 0.80" — or explicitly document why full-matrix r_99 is the actionable compression unit.

---

### 6. Runtime / Scope Sanity

**Step 1 (SVD of 1024×2048 float32 matrix)**: 2-min estimate. SVD of a 1024×2048 matrix via torch.linalg.svd on CPU (Ryzen 5 7530U, no LAPACK GPU offload): for a float32 matrix, this is O(min(m,n)² × max(m,n)) = O(1024² × 2048) ≈ 2.1B FLOP. At ~5 GFLOP/s single-threaded CPU (conservative): ~0.4 sec. Even with overhead, 2 min is generous. W_O[14] SVD (2048×2048): O(2048³) ≈ 8.6B FLOP, ~2 sec to 1 min depending on LAPACK threading. **Runtime estimates plausible.**

**Step 2 (ΔNLL, forward pass on 512 tokens)**: 12 min for 2 K values. Qwen3-1.7B forward pass at 512 tokens on CPU: ~4-6 min per pass at 6 tok/s (from reference ik_llama.cpp benchmark). Two K values (256, 512) = 2 passes = ~8-12 min. **Matches estimate.**

**Total GO path (41 min with W_O bonus and multi-seed)**: consistent with work described. Well within 8-hour ceiling.

**Disjoint calibration/eval**: no calibration corpus. WikiText-2 held-out for ΔNLL. 3 seeds specified. **Clean.**

**GQA shape concern (Kill Criterion 2 in Section 8)**: the plan specifies W_V[14] shape as 1024×2048 (n_kv_heads × d_head × d_model = 8×128×2048 reshaped). This is correct for Qwen3-1.7B-Base with GQA. However, loading this from safetensors requires knowing the actual key name and shape in the Qwen3 safetensors checkpoint. The script must verify: `model.safetensors` stores the V projection as `model.layers.14.self_attn.v_proj.weight` with shape `[1024, 2048]` (not `[2048, 2048]`). This should be an explicit assertion in the script. **Add as an implementation note, not a blocking amendment.**

---

### 6c. Elegant-Equivalence Ground

C-WVOK carries no elegance tag (Stage 5 Section 3b: "None explicit"). Section 6c applies only if an elegance tag is present. **Not applicable for C-WVOK.**

---

### 7. CF13–CF15 Verification Check

The plan's H3 refers to "70B residency arithmetic" but this is Section 12 downstream implications, not a Section 3 hypothesis. No CF13–CF15 numbers are load-bearing in the GO/NO-GO criterion. The 70B arithmetic is motivational and re-derives 70B dimensions from the Qwen3-72B model card (8192 d_model, 80 layers, 8 KV heads). No CF13–CF15 dependence. **Clean.**

---

### 8. Verdict — Track B

**ACCEPT-WITH-AMENDMENTS**

Amendments required before execution:

1. **B1 (random-init control into GO criterion)**: Add to Section 5 GO criterion: "AND r_99(W_V_random_init)/d ≥ 0.90 (trained spectrum distinguishable from Marchenko-Pastur baseline)."

2. **B2 (per-head SVD promoted to required)**: Promote the per-head W_V[14][0:128, :] SVD from "Bonus" to required sub-step within Step 1. Add to GO criterion: "AND r_99(W_V_per_head[0])/d_head < 0.80 (confirming compression is achievable at the GQA deployment granularity)." If full-matrix r_99 < 0.80 but per-head r_99 ≥ 0.80, the result is a GRAY case requiring investigation of whether the concentration is a cross-head artifact.

3. **B3 (H1 framing correction)**: Remove the phrase "H1 predicts W_V should be MORE concentrated than W_K" from Section 5. Replace with: "H1 motivates the hypothesis that r_99(W_V) may be below r_99(W_K) (0.79) due to training dynamics; this is an empirical test of that hypothesis, not a derivation from CF11." The coupling equation is a training-dynamics conjecture, not an algebraic necessity.

**Prior-art context for interpretation (not blocking)**: Published SVD compression literature (ARA arXiv:2510.19389, SVD-LLM family) consistently finds W_V requires high rank retention in empirical studies across LLaMA and Qwen3-8B architectures. The mode outcome predicted by the literature is NO-GO (r_99 > 0.80). If the Qwen3-1.7B result matches the literature (NO-GO), the experiment produces the most information-dense outcome in the run as claimed. The experiment should proceed with this expectation calibrated: the NO-GO outcome is not a failure, it is the structurally valuable outcome.

---

---

## Summary Table

| Gate | Track A (F6-WALIGN) | Track B (C-WVOK) |
|---|---|---|
| CF9 frame-mismatch | PASS (identity exact; no imported theorem) | PASS (coupling is hypothesis, not theorem; experiment valid regardless) |
| CF10 calibration | N/A (no fitting) | N/A (no fitting) |
| Skeptic controls | PASS with A1 (permutation gap must enter GO criterion) | PASS with B1 (random-init gap must enter GO criterion) |
| Prior art | PASS — Yadav et al. (ICLR 2025) supports prior probability; no pre-emption | PASS — A3 does not report Qwen3 W_V r_99; ARA/SVD-LLM literature signals likely NO-GO |
| Biased framing | PASS with A2 (separate ≥0.99 bin; avoid "near-exact at 0.90" language) | PASS with B2+B3 (per-head as required; H1 framing corrected) |
| Runtime | PASS (~8 min Step 1; 38 min GO path) | PASS (~9 min Step 1; 41 min GO path with bonus) |
| 6c elegance | PASS (identity exact at cos=1; approximation quality requires A2 binning) | N/A (no elegance tag) |
| CF13–CF15 | Clean | Clean |
| **Verdict** | **ACCEPT-WITH-AMENDMENTS (A1, A2, A3)** | **ACCEPT-WITH-AMENDMENTS (B1, B2, B3)** |
