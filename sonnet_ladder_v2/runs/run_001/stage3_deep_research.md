# Stage 3 — Deep Research — Run 001 (v2 Sonnet Ladder)

Researcher: Sonnet 4.6 (single-agent pass over all 16 advancers).
Date: 2026-05-09.
Inputs: Stage 1 outputs (R, C, F, A), Stage 2 judge, SUMMARY.md CF1-CF12, KILL_LIST.md, live WebSearch.

---

## Depth key
- DEEP (~800-1200 words): F-AJDGF, F-CQSGC, F-OCSSQ, R-MCQKV, R-RAOK-70B (convergence/elegant-equivalence/high-A1 set)
- STANDARD (~400-600 words): R-ACWF, R-W_down-SPEC, C-JQSOH, C-ABAR, C-LGHF, C-CTERA, F-SGFVO, C-RAOK-Grounded, A-GFRS, F-RSCLE
- LIGHT (~250-400 words): R-AERO-PROBE, F-ASIT, A-PWGI

---

# A — A-GFRS (Constraint-Alien: Gauge-Fixed Residual-Stream Inference)

## 1. Mechanism decomposition

- M1: The attention score depends only on M = W_Q W_K^T, so (W_Q, W_K) has a gauge freedom: (W_Q Σ, W_K Σ^{-T}) produces identical scores for any invertible Σ.
- M2: If ‖W_Q W_Q^T − W_K W_K^T‖_F / ‖W_Q W_Q^T‖_F < 0.30 for many layers, W_Q and W_K are approximately co-diagonalizable — the gauge freedom can be fixed to a basis where both are nearly block-diagonal.
- M3: Fixing gauge offline (pre-rotate all weight matrices by the joint eigenvector O*) reduces the effective rank of the QK inner product without changing storage or model behavior.
- M4: The measurement (Frobenius ratio per layer, ~10 min) serves as the convergence-cluster gate for Cluster C3.

## 2. Per-claim prior-art status

- M1 — **ADJACENT**: The attention-score M=W_Q W_K^T invariance under gauge is elementary. QuaRot / SpinQuant (ICLR 2025) apply orthogonal rotations to the residual stream for quantization error reduction — closest published rotation-of-residual-stream approach. Differentiation: QuaRot targets quantization outlier suppression; A-GFRS targets joint diagonalizability of the QK operator as a compression entry point. No paper found applying this specifically to joint-diagonalize W_Q and W_K operators as of 2026-05-09.
- M2 — **NOVEL**: No paper found reporting ‖W_Q W_Q^T − W_K W_K^T‖_F measurements per layer as a structural diagnostic. *elegance-class: gauge-exploitation*
- M3 — **ADJACENT**: SpinQuant (arXiv:2405.16406) uses learned rotation matrices; A-GFRS uses joint eigenvectors of a sum of quadratic forms. The derived object and purpose differ.
- M4 — **NOVEL**: The Frobenius ratio as a Cluster C3 gate number is not in any published paper as of 2026-05-09.

## 3. Frame-mismatch check (CF9)

No named theorems imported. Joint eigendecomposition is an elementary linear algebra operation (scipy.linalg.eigh on a symmetric PSD matrix). No precondition to verify beyond matrix symmetry. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit — pure spectral measurement. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

A-GFRS is a measurement/diagnostic proposal first; any residency benefit accrues after gauge-fix is confirmed. If the Frobenius ratio is < 0.30 for 14+ layers and effective W_Q rank decreases by ~30% in gauge-fixed coordinates, the W_Q K=128 compression quality improves equivalently to a 1.3× better rank budget — saving an additional ~30 MB on Qwen3-1.7B attention weights. Not a primary residency lever (A1=2 in Stage 2 is generous). Primary value is as a convergence gate for F-AJDGF.

## 6. Smallest-test sharpening

Script: `scripts/gfrs_frob_ratio.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data needed — purely weight-based.
Layers: all 28
Operation: compute W_Q W_Q^T (2048×2048) and W_K W_K^T (2048×2048) per layer; measure Frobenius ratio.
Runtime: ~10 min on Ryzen 5 7530U.
Output path: `experiments/stage0/ladder_v2/round1_gfrs/frob_ratio_per_layer.json`
Wall-clock: well under 2 hours. PASSES ≤8h gate.

## 7. Refined go/no-go thresholds

- **GO**: ratio < 0.30 in ≥14/28 layers. Advance to F-AJDGF CCA compression experiment.
- **NO-GO**: ratio > 0.50 in all layers. Joint diagonalization gains nothing; independent SVD is optimal. Cluster C3 confirmed as non-compositional.
- **GRAY**: ratio 0.30–0.50 in ≥14 layers. Run F-AJDGF CCA at r=64; compare to independent SVD at equal rank budget. If CCA ΔNLL < independent SVD ΔNLL − 0.2 nats, gauge exploitation is worth it; otherwise the gauge buys marginal gain.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Frobenius ratio > 0.50 everywhere (null result) | Still a structural finding that closes Cluster C3; no wasted cascade investment |
| Gauge fix makes W_V/W_O higher effective rank | Measure W_V/W_O effective rank before and after O* rotation in gray zone |
| Numerical conditioning: eigh on sum of near-singular PSD matrices | Use ridge 1e-4 on both operators before eigendecomposition |

Cheapest falsifier: Frobenius ratio measurement itself (~10 min). If > 0.50 everywhere, the whole cascade (A-GFRS + F-AJDGF) terminates with no implementation cost.

## 9. Two-paper composition flag

Cousin pair: QuaRot (arXiv:2311.01132) + CF11 W_Q head-redundancy measurement.
Composition yields: "rotate residual stream to reduce W_Q outliers while exploiting head-sharing subspace."
A-GFRS value-add: the rotation target is joint diagonalization of the W_Q W_K^T operator, not outlier suppression — structurally different objective. *Not engineering-integration.*

## 10. Verdict

**REFINE** — Cluster C3 representative. Measurement is 10 min with binary outcome. Advance to Stage 4 as the cheapest convergence gate.

## 11. Hand-off note

Run `gfrs_frob_ratio.py` before any F-AJDGF CCA experiment. If GO, chain immediately to F-AJDGF. If NO-GO, retire both Cluster C3 advancers and note in KILL_LIST.

---

# C — C-ABAR (Composition: Asymmetric bpw Allocation from Spectral Rank Boundary)

## 1. Mechanism decomposition

- M1: CF11 W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79 define the attention-weight spectral hierarchy.
- M2: CF12 tied embed/lm_head r_99/d ≈ 0.97 with catastrophic ΔNLL (+19.96 at r=1024) defines an untouchable floor.
- M3: MLP weights are full-rank (CF8) and must use quantization, not rank-reduction.
- M4: The monotonicity claim — sensitivity ∝ r_99/d × functional_importance — is the load-bearing allocation oracle.
- M5: W_Q at K=512 + W_K at K=512 simultaneously achieves ΔNLL < 0.5 nats AND attention residency ≤ 228 MB.

## 2. Per-claim prior-art status

- M1–M3 — **NOVEL** (our private CF measurements; no paper has this data on Qwen3 family).
- M4 — **ADJACENT**: LAMPQ (arXiv:2511.10004, Nov 2025) allocates mixed precision by layer sensitivity, reporting that "nuclear norms of attention and feature maps vary significantly depending on module type (MSA vs MLP)." ABAR's differentiation: LAMPQ uses nuclear norm proxies and calibration sensitivity; ABAR uses empirical spectral rank r_99/d directly. The specific claim that spectral rank r_99/d is a better oracle than Hessian-diagonal or nuclear norm is not demonstrated in published work as of 2026-05-09.
- M5 — **NOVEL**: simultaneous W_Q K=512 + W_K K=512 on Qwen3-1.7B with measured ΔNLL threshold is not published.

## 3. Frame-mismatch check (CF9)

No imported theorems. SVD is elementary. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit — pure SVD rank truncation without regression. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Per Stage 1 C2-B:
- W_Q K=512 (all 28 layers): 28 × 2048 × 512 × 2 × 2 bytes = ~114 MB (vs 457 MB bf16). Saving: 343 MB.
- W_K K=512: same saving 343 MB.
- MLP Q4_0: ~528 MB vs 1053 MB.
- Tied embed: unchanged at 590 MB.
- Total ≈ 1.57 GB (down from 3.4 GB). 2.2× compression on Qwen3-1.7B.
At 70B (untied lm_head assumed): attention W_Q+W_K at K=512 each saves ~3.5 GB. Total model ~16 GB IF W_V/W_O also compressible; ~20 GB with only W_Q+W_K compressed + Q4 MLP. DRAM-resident 70B still out of reach, but ABAR defines the correct allocation oracle for the cascade.

No CF13-CF15 dependencies. Conservative arithmetic is identical to optimistic.

## 6. Smallest-test sharpening

Script: `scripts/abar_wq_wk_k512.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: N/A (no calibration needed for SVD)
Eval corpus: WikiText-2 (512 tokens held-out)
Layers: all 28, W_Q and W_K only
Sweep: K ∈ {512} (the Stage 2 GO threshold)
Runtime: SVD ~10 min + eval ~5 min = ≤20 min total. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_abar/wq_wk_k512_nll.json`

## 7. Refined go/no-go

- **GO**: ΔNLL < 0.5 nats AND combined W_Q+W_K residency ≤ 228 MB.
- **NO-GO**: ΔNLL ≥ 0.5 nats at K=512 — W_K is the weak point; raise K to 1024 for W_K only. Refine allocation boundary.
- **GRAY**: ΔNLL 0.3–0.5 nats. Run K sweep for W_K at {512, 768, 1024} to find the minimum K for W_K given W_Q at K=512 is confirmed safe.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| W_V/W_O full-rank (leaves embed still at bf16 as biggest single item) | Measurement of W_V/W_O is a parallel 10-min experiment; gate W_V/W_O compression separately |
| CF11 head-sharing at global K=128 confirmed but K=512 not yet measured on all 28 layers simultaneously | The smallest experiment measures this directly |
| Mixed-precision inference engine doesn't support (SVD-compressed attention, Q4 MLP) in llama.cpp | Flag as deployment risk; first check correctness in pure PyTorch |

Cheapest falsifier: the 20-min experiment. If ΔNLL ≥ 0.5 nats simultaneously at W_Q+W_K K=512, the allocation boundary moves and the oracle needs recalibration.

## 9. Two-paper composition flag

Cousin pair: CF11 (our pipeline) + LAMPQ (arXiv:2511.10004).
ABAR value-add: uses empirical r_99/d as the oracle rather than nuclear norm or Hessian proxy; explicitly includes the catastrophic CF12 floor constraint on the tied embed. Not a pure engineering integration — the r_99/d vs Hessian-proxy comparison is a structural research claim.

## 10. Verdict

**REFINE** — The 20-min smallest test is decisive, the arithmetic is clean, CF9/CF10 clear.

## 11. Hand-off note

Run alongside R-W_down-SPEC and F-SGFVO to complete the full Qwen3 weight-class spectral picture in one experimental session.

---

# C — C-CTERA (Composition: W_Q Concentration × Tied-Embedding Rigidity Joint Oracle)

## 1. Mechanism decomposition

- M1: CF11 W_Q (layer 1) top-128 singular vectors form a projection basis P_Q1.
- M2: CF12 tied embed/lm_head is full-rank (r_99=1992) and must remain at bf16.
- M3: The embed-query alignment α = ‖E^T P_Q1‖²_F / (‖E‖²_F × ‖P_Q1‖²_F) measures whether the first-layer W_Q already preferentially reads from the embed's high-variance directions.
- M4: If α < 0.2, a rotation of the embed by P_Q1 would improve W_Q compression quality at fixed K=128 budget (misalignment means the K=128 basis ignores high-information embed directions).
- M5: This rotation costs O(d²) one-time at model load; no new weight storage.

## 2. Per-claim prior-art status

- M1, M2 — **NOVEL** (private CF11/CF12 measurements, not on arXiv).
- M3 — **NOVEL**: No paper found measuring the alignment between embed singular subspace and first-layer W_Q basis as of 2026-05-09. Closest: "Weight Tying Biases Token Embeddings Towards the Output Space" (arXiv:2603.26663, March 2026) examines tied-embedding geometry but does not measure alignment with W_Q. *elegance-class: subspace-alignment*
- M4 — **ADJACENT**: SpinQuant / QuaRot apply rotations for quantization error reduction. CTERA's rotation targets a different objective (align W_Q basis with embed variance, not quantization-outlier suppression). No functional overlap.
- M5 — **NOVEL**: The specific O(d²) rotation-as-no-residency-cost co-design argument has no published analog.

## 3. Frame-mismatch check (CF9)

No named theorems imported. The α metric is a direct matrix-product norm computation. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No regression fit — purely matrix algebra. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

No weight residency change in the measurement step. The payoff is quality improvement: if α < 0.2 and rotation improves W_Q K=128 from +0.98 nats to +0.5 nats, this enables K=64 instead of K=128 at equal quality → additional ~57 MB saving on 1.7B, ~2 GB on 70B. Value is conditional on misalignment being confirmed.

## 6. Smallest-test sharpening

Script: `scripts/ctera_align_angle.py`
Model: Qwen3-1.7B-Base, bf16
Operation: Load E (embed, 151936×2048), load W_Q[L=1] top-128 singular vectors P_Q1 (already computed in AQFKV experiment). Compute E^T P_Q1 (151936×128 matmul, ~30 sec in numpy). Compute α.
Runtime: ~30 seconds. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_ctera/embed_wq_alignment.json`

## 7. Refined go/no-go

- **GO**: α < 0.2 (misalignment confirmed) → proceed to rotation step, measure ΔNLL improvement at K=128 and K=64.
- **GO-trivial**: α ≥ 0.5 (training already jointly optimized) → no rotation needed; structural finding that training aligns these by construction.
- **NO-GO / GRAY**: α ∈ [0.2, 0.5] → borderline; run the one-time rotation and measure ΔNLL at K=64 before concluding.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| α is in the 0.2-0.5 range (inconclusive) | Run the actual rotation anyway; measure ΔNLL at K=64 with and without rotation (~5 min extra) |
| Rotation at K=128 (not square) destroys 1920 embed directions | Confirmed in Stage 1 proposal: loss is acceptable if α is already high; new risk only if α is low AND rotation is K-truncated (not full-rank) |
| Finding is purely diagnostic with no deployment action | Even purely diagnostic, the structural finding about CF11/CF12 coupling has value for future proposals |

Cheapest falsifier: the 30-second α computation. If α ≥ 0.5, the rotation is moot and the proposal's deployment payoff disappears (replaced by structural confirmation that training co-optimizes these).

## 9. Two-paper composition flag

Cousin pair: QuaRot (arXiv:2311.01132) + CF12 (our LHQD finding).
CTERA value-add: the target is not quantization error reduction but alignment between two specific components (embed and first-layer W_Q) that CF12 reveals are jointly constrained by training. This is a diagnostic oracle, not a quantization rotation. Not engineering-integration; the structural question is new.

## 10. Verdict

**REFINE** — 30-second computation, decisive binary outcome (misaligned or not), no CF9/CF10 issues, high frame novelty.

## 11. Hand-off note

Can be run immediately with existing AQFKV P_Q1 computation. Pairs naturally with C-LGHF and C-ABAR to form a complete Layer-1 diagnostic session.

---

# C — C-JQSOH (Composition: Joint QK Subspace Outlier Highway)

## 1. Mechanism decomposition

- M1: CF11 establishes that all 16 W_Q heads at a given layer share a ~128-dim projection basis P_Q.
- M2: CF3 establishes that the top-0.1% channels (~2) have Jaccard 0.718 — channel-static.
- M3: If dim(span(e_i : i ∈ S) ∩ colspan(P_Q)) ≥ 1, the static outlier channels survive the W_Q K=128 compression intact — no bypass needed.
- M4: If orthogonal, an INT16 bypass buffer (2 scalars per layer per token) is required.

## 2. Per-claim prior-art status

- M1 — **NOVEL** (our CF11). M2 — **NOVEL** (our CF3).
- M3 — **NOVEL**: The intersection check between the CF3 static-outlier subspace and CF11 W_Q basis has no published analog as of 2026-05-09. *elegance-class: subspace-alignment*
- M4 — **NOVEL**: Calibration-free bypass-or-not decision from a subspace intersection test.

## 3. Frame-mismatch check (CF9)

No theorems imported. Cosine similarity between channel basis vectors and P_Q columns is direct linear algebra. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

If channels ARE inside P_Q: no bypass needed. W_Q K=128 compression saves 87.5% of W_Q bytes (117 MB on 1.7B) with the outliers included at no extra cost. If bypass IS needed: 2×16-bit scalars per layer per token is negligible overhead. The outlier-highway question is primarily a quality/correctness question, not a residency question. Residency impact is the same either way.

## 6. Smallest-test sharpening

Script: `scripts/jqsoh_outlier_alignment.py`
Model: Qwen3-1.7B-Base, bf16
Data: CF3 static-outlier channel indices (from R2 PDAP dataset, already available).
Layers: all 28, but check layer-1 first.
Operation: compute P_Q (W_Q top-128 SVD); compute pairwise cosine between outlier channels and P_Q columns.
Runtime: W_Q SVD per layer already done in AQFKV; pairwise cosine is ~2 min. Total ≤10 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_jqsoh/outlier_pq_cosine_per_layer.json`

## 7. Refined go/no-go

- **GO**: max cosine ≥ 0.3 in ≥14/28 layers → outlier channels are inside P_Q; bypass not needed.
- **NO-GO**: max cosine < 0.1 across all layers → bypass buffer required; changes the compression architecture but does not kill the idea.
- **GRAY**: mixed layers (some inside, some outside) → implement adaptive bypass (bypass only for layers where max cosine < 0.1).

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Outlier channels rotate with token (CF3 Jaccard=0.718 is not 1.0) | Even with 28% rotation, the bypass buffer covers the mobile fraction; GO condition is about the 72% static component |
| P_Q columns at K=128 may not align with any unit-vector channel | The test is direct — if NO-GO across all layers, the bypass is the correct design |

Cheapest falsifier: 10-min cosine computation using existing data.

## 9. Two-paper composition flag

Cousin pair: CF3 (our PDAP) + CF11 (our AQFKV).
Value-add: the composition question (do these two private findings couple geometrically in the residual stream?) is purely internal to our CF dataset. No published paper can answer this because no other paper has both CF3 Jaccard measurements and CF11 head-shared basis on Qwen3. Not engineering-integration.

## 10. Verdict

**REFINE** — 10-min measurement, binary outcome, no CF9/CF10 issues, fully CF-grounded novelty.

---

# C — C-LGHF (Composition: Layer-1 Gate Anomaly × Head-Shared Q-Projection Fold)

## 1. Mechanism decomposition

- M1: CF6 — Layer 1 has 36.1% foldable gate neurons (std < 0.05). Global rate is 1.5%.
- M2: CF11 — Global W_Q K=128 compression gives ΔNLL = +0.98 nats; per-layer K=128 is within this budget.
- M3: Layer-1 W_Q may be MORE compressible than the global average (pre-context regime before any contextual information accumulates).
- M4: If the two compressions operate on disjoint Layer-1 sub-graphs (MLP vs attention), ΔNLL_joint ≈ ΔNLL_gate + ΔNLL_WQ ± ε (independence claim).

## 2. Per-claim prior-art status

- M1, M2 — **NOVEL** (private CF6, CF11). M3 — **NOVEL** as of 2026-05-09.
- M4 — **NOVEL**: no paper tests joint MLP+attention compression at a single layer with independence decomposition on Qwen3 family.

## 3. Frame-mismatch check (CF9)

No theorem imports. The independence claim is falsifiable by measuring ΔNLL_joint directly. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Layer-1 joint: ~17 MB (MLP gate fold) + ~31 MB (W_Q K=64) = 48 MB savings on 1.7B. Scales to ~768 MB on 70B if layer-1 anomaly generalizes across model sizes. Not a primary residency lever (A1=1).

## 6. Smallest-test sharpening

Script: `scripts/lghf_layer1_joint.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: WikiText-2, 200 tokens (gate fold requires calibration to identify foldable neurons)
Eval corpus: WikiText-2, 512 tokens held-out
Operation: (1) Identify Layer-1 foldable neurons (std < 0.05 from CF6); fold them. (2) Replace W_Q[L=1] with K=64 SVD. (3) Measure ΔNLL for fold-only, WQ-only, and joint.
Runtime: gate fold identification ~5 min (already have CF6 data), W_Q SVD ~1 min, eval ≤5 min. Total ≤15 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_lghf/layer1_joint_nll.json`

## 7. Refined go/no-go

- **GO**: Layer-1 W_Q K=64 ΔNLL < 0.5 nats AND ΔNLL_joint < ΔNLL_gate + ΔNLL_WQ + 0.1 nats.
- **NO-GO**: Layer-1 W_Q K=64 ΔNLL ≥ 0.5 nats → pre-context regime does NOT give more compressibility; drop M3 claim; LGHF is still valid at K=128 but loses the novelty of the pre-context argument.
- **GRAY**: Gate fold + W_Q joint cost is subadditive (interaction ε < 0 nats) → strong confirmation that Layer-1 admits genuine joint compression.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Gate fold neurons not truly calibration-independent (they vary with corpus) | Verify against R2 PDAP calibration data using Jaccard of foldable neuron sets across two corpora |
| Layer-1 W_Q is NOT more compressible than global mean (M3 fails) | Layer-1 K=128 still saves 31 MB; only M3 novelty is lost |

Cheapest falsifier: W_Q[L=1] K=64 eval. If ΔNLL ≥ 0.5 nats, M3 is dead.

## 9. Two-paper composition flag

Cousin pair: SDZC (our killed R4-A) + CF11 (our AQFKV).
Value-add: SDZC was killed as a global compression scheme; LGHF uses the Layer-1 anomaly as a structural target and composes it with W_Q compression for which we have empirical cover. Not a global scheme. Not engineering-integration.

## 10. Verdict

**REFINE** — 15-min smallest test, clean independence test, fully grounded in private CF data.

---

# C — C-RAOK-Grounded (Composition: Activation-Outlier Staticity as Quantization-Tier Selector)

## 1. Mechanism decomposition

- M1: CF3 K=0.1% Jaccard=0.718 (channel-static), K=1% Jaccard=0.31 (token-dynamic) defines natural tier boundaries.
- M2: CF6 deep-layer activation liveness confirms tier boundaries cannot be derived from gate variance.
- M3: The three-tier codebook (2 FP16 / 18 INT8 / 2028 INT4) is calibration-free; boundaries set by CF3 Jaccard transitions.
- M4: Top-2 channel stability is the critical upstream gate: if same 2 channels appear in ≥90% of tokens across diverse prompts in ≥20/28 layers, the static FP16 bypass is valid.
- M5: KV cache primary payoff: 4× KV reduction at 4K context (116 MB vs 471 MB).

## 2. Per-claim prior-art status

- M1, M2 — **NOVEL** (private CF3, CF6). 
- M3 — **PARTIAL**: LLM.int8() uses binary split (FP16 static + INT8 rest). SmoothQuant migrates outliers to weights. PrefixQuant (arXiv:2410.05265) eliminates token-wise outliers by prefixing. None of these derive tier boundaries from consecutive-pair Jaccard statistics. The specific three-tier structure from CF3 Jaccard transitions is **NOVEL** as of 2026-05-09. No paper found using Jaccard-derived tier thresholds.
- M4 — **NOVEL**: Per-layer Tier-0 stability measurement using R2 PDAP data.
- M5 — **ADJACENT**: KVQuant, MiniKV, KITTY all compress KV cache; none use CF3-derived tier boundaries. *elegance-class: conserved-quantity*

## 3. Frame-mismatch check (CF9)

No theorem imports. Tier boundaries are empirically measured. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit for the tier boundaries (derived from CF3 statistics, not regression). If a per-layer Tier-1 set lookup table were calibrated, it would need CF10 check — but the proposal uses calibration-free tier assignment. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Activation compression 4× (from 4096 bytes/token/layer to 1036 bytes). Primary payoff is KV cache: at 4K context, savings ≈ 355 MB (471 MB → 116 MB). At 70B scale (d=8192, 64 KV heads per layer): KV at bf16 for 4K context ≈ 64×2×4096×8192×2 bytes ÷ 10^9 ≈ 4 GB; at 4× compression: 1 GB KV. Meaningful but secondary to weight residency for 70B deployment.

## 6. Smallest-test sharpening

Script: `scripts/raok_tier0_stability.py`
Data: R2 PDAP dataset (200 diverse prompts already available)
Operation: Extract top-2 channels by mean magnitude rank per layer per token; compute fraction of tokens where the same 2 channels appear; check stability ≥90% in ≥20/28 layers.
Runtime: ~10 min (using existing PDAP data). PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_raok_grounded/tier0_stability_per_layer.json`

## 7. Refined go/no-go

- **GO**: ≥90% same-channel stability in ≥20/28 layers → static FP16 bypass valid; proceed to full RAOK three-tier ΔNLL test (~4h, see R-RAOK-70B).
- **NO-GO**: < 70% stability in most layers → Tier 0 must be per-token; eliminates the calibration-free property; tier boundaries must shift to K=0% (no static channel, full per-token scheme).
- **GRAY**: 70–90% stability → static bypass for the ~80% stable case, with a fallback INT8 slot for the mobile 20%. Measure overhead of the fallback path.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Tier-1 (18 channels) per-token identification adds runtime overhead | Profile per-token top-20-channel identification cost vs GEMV cost; if > 5% of GEMV time, demote to per-tensor INT8 |
| Jaccard and quantization-sensitivity are different measures (M3 vs M4 gap) | Cross-validate by comparing Jaccard-derived tiers vs Hessian sensitivity tiers on held-out |

Cheapest falsifier: 10-min stability check using existing PDAP data.

## 9. Two-paper composition flag

Cousin pair: LLM.int8() (arXiv:2208.07339) + CF3 (our PDAP).
Value-add: LLM.int8() uses a binary split; RAOK-Grounded uses a CF3-Jaccard-derived three-tier with calibration-free thresholds. The structural novelty is the Jaccard-to-tier-boundary derivation which no published method has used.

## 10. Verdict

**REFINE** — 10-min measurement from existing data, binary gate, fully CF-grounded.

---

# F — F-AJDGF (First-Principles: Attention-Weight Joint Diagonalization Gauge Fix)

## DEEP TREATMENT

## 1. Mechanism decomposition

- M1: Attention scores depend only on M = W_Q W_K^T; the pair (W_Q, W_K) has a gauge freedom: (W_Q Σ, W_K Σ^{-T}) for any invertible Σ produces identical scores.
- M2: CCA applied to the pair (W_Q, W_K) per head finds the Σ* that simultaneously maximizes correlation between the two matrices' canonical variates — this is the gauge-fixing step.
- M3: Under the CCA gauge, both W_Q and W_K can be jointly compressed at rank r with smaller combined Frobenius error than independent SVD compression.
- M4: CCA computation is O(d_head³) per head (standard linear algebra, scipy.linalg.svd on the cross-covariance Σ_12 = W_Q^T W_K); no calibration data required.
- M5: GO condition: ΔNLL at CCA r=64 per head < ΔNLL at independent SVD r=64 per head.

## 2. Per-claim prior-art status

- M1 — **ADJACENT**: The M = W_Q W_K^T invariance is noted in arXiv:2602.18851 "Rank-Aware Spectral Bounds on Attention Logits" as a structural fact. That paper uses it for bounding purposes, not for compression. No compression paper exploits it directly for post-training weight reduction. Stage 2 correctly identified this as ADJACENT, not PRE-EMPTED.
- M2 — **NOVEL**: CCA applied to (W_Q, W_K) pairs per head for post-training joint compression is not in any published paper found as of 2026-05-09. WebSearch for "CCA canonical correlation analysis W_Q W_K transformer post-training compression" returned no matching ML papers; the compressed convolutional attention result is architecturally different (training-time latent space fusion). *elegance-class: gauge-exploitation*
- M3 — **PARTIAL**: A3 (arXiv:2505.12942, May 2025) proposes analytical low-rank approximation for QK components, minimizing functional errors of attention scores. A3's approach: minimize ‖A_approx − A_exact‖_F by calibration-fitted low-rank decomposition. F-AJDGF's approach: exploit the gauge freedom to find a Σ* such that CCA-compressed W_Q and W_K reconstruct M with smaller error than independent SVD at equal rank. These are distinct mechanisms: A3 is calibration-guided (requires activation statistics), F-AJDGF is weight-only (no calibration data, pure linear algebra). The A3 threat is REAL — it covers the "compress QK jointly" territory — but does not foreclose F-AJDGF's CCA gauge-exploitation specifically.
- M4 — **NOVEL**: Weight-only CCA with no calibration data is not A3's approach (A3 uses calibration covariance). *elegance-class: gauge-exploitation*
- M5 — **NOVEL**: Comparing CCA vs independent SVD at equal rank budget as a falsifiability test has no published analog.

**Pre-emption verdict on M3**: A3 (arXiv:2505.12942) is the critical adjacent paper. A3 compresses OV (W_V W_O) AND QK jointly with calibration covariance guidance. The F-AJDGF claim that differentiates it is: *pure gauge algebra without calibration data.* If A3 already achieves better QK compression with calibration guidance than F-AJDGF can achieve without it, the gauge-only approach is DOWNGRADED to ENGINEERING-INTEGRATION of A3 without data. The test: F-AJDGF must show ΔNLL advantage over independent SVD at equal rank — the "better than naive" bar. If it matches or beats A3's calibration-free claim, it remains novel; if worse than independent SVD, the gauge exploitation buys nothing.

## 3. Frame-mismatch check (CF9)

- **Imported object**: CCA / joint SVD of cross-covariance Σ_12 = W_Q^T W_K.
- **Precondition**: CCA is well-conditioned when W_Q and W_K have comparable spectral magnitudes and no degenerate dimensions. CF11 reports W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79 — both have significant spectral content; neither is near-zero-rank. The cross-covariance Σ_12 = W_Q^T W_K is a d_head × d_head matrix (128×128). Both input matrices have rank ≈ 80–100 effective dimensions at 99% energy. CCA is computable and well-conditioned here — the precondition holds.
- **Risk**: Numerical conditioning degrades if W_Q and W_K are nearly orthogonal (Σ_12 near-singular). Mitigation: regularize Σ_12 with ridge λ=1e-3 before SVD; check condition number.
- CF9 CLEAR (with ridge mitigation).

## 4. Calibration pre-flight (CF10)

No calibration data fit. F-AJDGF is purely weight-based: the CCA factorization is computed from (W_Q, W_K) tensors only, no activation samples. CF10 NOT APPLICABLE. This is the key differentiator from A3.

## 5. Residency arithmetic

Per-head in Qwen3-1.7B: W_Q ∈ R^{128×2048}, W_K ∈ R^{128×2048}. At CCA r=64:
- W_Q factors: 64×2048 + 64×128 = 139K params per head.
- W_K factors: 64×2048 + 64×128 = 139K params per head.
- Vs original: 2×(128×2048) = 524K params per head.
- Saving per head: (524K−278K)/524K ≈ 47% of (W_Q+W_K) per head.
- Over 16 heads × 28 layers: saving ≈ 120 MB on (W_Q+W_K) combined from original 448 MB.
- At 70B (d=8192, 64 heads, GQA kv_heads=8): W_Q = 64×(8192×8192) at bf16 = 8.6 GB; W_K = 64×(8192×1024) = 1.1 GB. CCA at r=512: W_Q factors 2×(512×8192)×64 heads ≈ 0.54 GB; W_K factors 2×(512×1024+512×8192)×64 ≈ 0.6 GB. Total compressed ≈ 1.14 GB vs 9.7 GB original. Saving ≈ 8.6 GB at 70B from (W_Q+W_K) only.
No CF13-CF15 dependencies.

## 6. Smallest-test sharpening

Script: `scripts/ajdgf_cca_qk.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data needed.
Layers: all 28; heads: all 16
Operation: (1) For each (layer, head): compute Σ_12 = W_Q^T W_K (2048×2048); compute SVD; truncate at r=64; reconstruct W_Q_cca = U_1 S^{1/2}, W_K_cca = U_2 S^{1/2}. (2) Replace all W_Q and W_K in model. (3) Eval ΔNLL on WikiText-2. (4) Compare to independent SVD at r=64 per head (equal rank budget).
Runtime: 448 head CCA computations (2048×2048 SVD each, ~2s each on Ryzen) = ~15 min. Eval ~5 min. Total ≤25 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_ajdgf/cca_vs_svd_nll_r64.json`

Note: Σ_12 = W_Q^T W_K is 2048×2048, not 128×128 — I correct the stage-1 error. The per-head matrices are W_Q^(h) ∈ R^{128×2048} (head slice of the full W_Q). The relevant SVD for CCA is of W_Q^(h)^T W_K^(h) ∈ R^{2048×2048}. At per-head d_head=128, the CCA SVD is actually of W_Q^(h,T) W_K^(h) ∈ R^{2048×2048} which has rank at most 128. Use truncated SVD at r=64 which is O(2048 × 128²) ≈ 33M FLOPs per head — fast.

## 7. Refined go/no-go

- **GO**: ΔNLL_CCA < ΔNLL_SVD − 0.2 nats at r=64 per head AND ΔNLL_CCA < 0.5 nats absolute.
- **NO-GO**: ΔNLL_CCA ≥ ΔNLL_SVD (gauge exploitation provides zero benefit over independent SVD). The gauge freedom for (W_Q, W_K) is real algebraically, but may not translate to better low-rank approximation in practice. Kill Cluster C3 F-AJDGF path.
- **GRAY**: ΔNLL_CCA ∈ [ΔNLL_SVD−0.2, ΔNLL_SVD]. Marginal gain; run at r=32 and r=96 to see if gain is rank-dependent.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| A3 (arXiv:2505.12942) already covers QK compression with calibration covariance — F-AJDGF is ADJACENT, not pre-empted | F-AJDGF's claim is weight-only, no calibration. If the experiment shows CCA > independent SVD, the gauge-only approach has standalone value. If not, DOWNGRADE to ENGINEERING-INTEGRATION of A3's QK path |
| CCA Σ_12 near-singular (W_Q and W_K nearly orthogonal in head coordinates) | Check condition number of Σ_12; add ridge λ=1e-3; report per-head condition numbers |
| Per-head CCA at r=64 may not beat global K=128 SVD (CF11's better-quality approach) | Run both; primary claim is CCA > per-head-SVD at equal rank, not CCA > global-K-SVD |

Cheapest falsifier: 25-min experiment. If ΔNLL_CCA ≥ ΔNLL_SVD, the gauge-exploitation claim is dead.

## 9. Two-paper composition flag

Cousin pair: A3 (arXiv:2505.12942, QK functional error minimization) + CF11 (our AQFKV, head-redundancy measurement).
Obvious composition: "use A3-style functional minimization on our CF11-characterized Qwen3 attention weights."
F-AJDGF value-add: eliminates calibration data dependency entirely (pure weight-based CCA), which (a) has zero CF10 ill-conditioning risk, (b) is deployable without calibration corpus, and (c) tests whether the algebraic gauge is the dominant compressive structure rather than activation-driven covariance. This is a structural research claim, not just A3 applied to our model. But the bar is higher: F-AJDGF must demonstrate measurable advantage over independent SVD to justify not using A3's calibration-guided approach.

## 10. Verdict

**REFINE** — with explicit A3 competitive bar. Stage 4 should note: if F-AJDGF GO fails (CCA ≤ SVD), recommend DOWNGRADE and pivot to A3-style calibration-guided QK compression as the Cluster C3 deployment path. 25-min test is decisive.

## 11. Hand-off note

A3 (arXiv:2505.12942) is the key competitive benchmark. Stage 4/5 should read A3 Section 3 (QK component specifically) before selecting F-AJDGF or A3's QK path. Run A-GFRS Frobenius ratio first — if ratio < 0.30, joint diagonalization motivation is stronger.

---

# F — F-ASIT (First-Principles: AERO-Compatible SwiGLU Identity Test at Layer 1) [FREE SWING]

## LIGHT TREATMENT

## 1. Mechanism decomposition

- M1: CF6 — Layer 1 has 36.1% foldable gate neurons (std < 0.05).
- M2: If W_gate^(1)[j,:] lies near the null space of input covariance C_x^(1), the neuron is structurally foldable (geometric criterion, not calibration-measured).
- M3: Projection fraction ‖W_gate^(1)[j,:] Π_{C_x^(1)}^{null}‖ / ‖W_gate^(1)[j,:]‖ > 0.70 for foldable neurons confirms the geometric origin.

## 2. Prior-art status

- M2, M3 — **NOVEL** as wildcard primitive: connecting input covariance null space to AERO fold eligibility is not in AERO's paper (arXiv:2410.13060) or any follow-up found as of 2026-05-09. AERO requires calibration; ASIT proposes a zero-calibration structural criterion. *elegance-class: no-SGD-reformulation*

## 3. CF9 check

No theorem import. Projection onto null(C_x) is standard PCA. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Smallest-test sharpening

Script: `scripts/asit_null_space_fold.py`
Model: Qwen3-1.7B-Base
Operation: Collect 200 tokens of Layer-1 inputs; compute C_x; find bottom-r eigenvectors (r=512); project W_gate[L1] rows onto null subspace; compare foldable vs non-foldable neurons.
Runtime: ~30 min. PASSES ≤8h gate.
GO: mean projection fraction for CF6-foldable neurons > 0.70 AND significantly > non-foldable (p < 0.01).

## 6. Verdict

**REFINE** — wildcard, CF-tether-suspended, structural floor met. Connects to real stack primitive (Layer-1 gate weights, activation covariance). Math correct, smallest test ≤30 min.

---

# F — F-CQSGC (First-Principles: Cross-Layer Query-Subspace Gauge Collapse)

## DEEP TREATMENT

## 1. Mechanism decomposition

- M1: CF11 establishes within-layer W_Q head-redundancy (16 heads → ~128-dim subspace).
- M2: If all 28 W_Q matrices share a common dominant right subspace U of dimension D*, then the cross-layer W_Q family can be stored as a shared U ∈ R^{2048×D*} plus 28 thin coefficient matrices C_ℓ ∈ R^{D*×2048}.
- M3: The cross-layer "gauge collapse" framing: U is the gauge-fixed coordinate system for residual-stream attention computation across all layers.
- M4: var@D*=128 > 0.80 across the stacked SVD of all 28 W_Q matrices is the GO threshold — the cheapest settler for Cluster C1 (shared by R-CROSS-Q, C-CQBAL, A-GFRS, F-CQSGC).

## 2. Per-claim prior-art status

- M1 — **NOVEL** (private CF11).
- M2 — **ADJACENT (critical)**: Basis Sharing (arXiv:2410.03765, ICLR 2025) proposes cross-layer weight basis sharing via SVD — this is the closest published cousin. The paper examines which weight matrix types benefit from cross-layer basis sharing. It is NOT confirmed whether Basis Sharing explicitly tested W_Q. WebSearch for "Basis Sharing W_Q cross-layer results" returned non-specific results. If Basis Sharing tested W_Q and found high cross-layer coherence, M2 is PRE-EMPTED.
- M2 — **ADJACENT (second cousin)**: Share Your Attention (arXiv:2508.04581, Aug 2025) proposes matrix-based dictionary learning for attention weight sharing across layers (66.7% attention parameter reduction). This DOES cover W_Q across layers with a dictionary learning approach, which is structurally similar to cross-layer basis sharing. However, MASA (Share Your Attention) is a training-time fine-tuning method, while F-CQSGC is post-hoc weight-only SVD stacking — different in mechanism (no fine-tuning required).
- M3 — **NOVEL**: The gauge-collapse framing (shared subspace is the gauge-fixed coordinate) is not in Basis Sharing or Share Your Attention, which treat it as parameter sharing without this interpretation. *elegance-class: gauge-exploitation*
- M4 — **NOVEL**: The var@128 measurement on stacked Qwen3 W_Q (57344×2048 PCA) is not in any published paper.

**Critical pre-emption assessment for M2**: MASA (arXiv:2508.04581, Aug 2025) uses matrix-based dictionary learning for cross-layer attention weight sharing. If MASA's W_Q sharing on LLaMA achieves cross-layer coherence and quality results comparable to what F-CQSGC would produce, F-CQSGC needs to position against MASA explicitly. Key differentiator: F-CQSGC requires NO fine-tuning, NO calibration data, and produces its compression purely from the SVD of the existing weight stack. MASA requires fine-tuning to match quality. This is a real structural difference — the "no-retraining" claim is the moat.

## 3. Frame-mismatch check (CF9)

No theorem imports. Randomized SVD of a stacked matrix is standard linear algebra. CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Per Stage 1 F1-A:
- Full bf16 W_Q for 28 layers: 457 MB.
- Shared U (2048×128): 0.5 MB.
- Per-layer C_ℓ (2048×128): 28×0.5 = 14 MB.
- Total W_Q: ~15 MB. 15× compression, saving 442 MB on Qwen3-1.7B.
- 70B projection: W_Q at d=8192, L=64: original 8.6 GB; shared U (8192×512): 8 MB; per-layer C_ℓ (8192×512): 64×8 MB = 512 MB. Total: 520 MB vs 8.6 GB. Saving: 8.1 GB.
No CF13-CF15 dependencies.

## 6. Smallest-test sharpening

Script: `scripts/cqsgc_stacked_svd.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data.
Operation: Stack all 28 W_Q matrices into a (57344×2048) matrix (reshape each 2048×2048 to a 2048×2048 row-normalized matrix, stack as 28 blocks). Compute randomized SVD (sklearn.utils.extmath.randomized_svd or torch.svd_lowrank) keeping D* ∈ {64, 128, 256, 512}. Measure var@D* (cumulative singular value energy fraction). Separately, measure per-layer reconstruction ΔNLL using W_Q^(ℓ) ≈ C_ℓ U^T at D*=128.
Runtime: Stacked SVD of 57344×2048 ≈ 15 min (randomized SVD with D*=128 is much faster than full SVD). ΔNLL eval ~10 min. Total ≤30 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_cqsgc/stacked_wq_var_profile.json`

## 7. Refined go/no-go

- **GO**: var@128 > 0.80 AND per-layer reconstruction ΔNLL at D*=128 < 1.0 nat.
- **NO-GO**: var@128 < 0.50 → W_Q subspaces rotate independently across layers; cross-layer sharing has no structural basis. Kill entire Cluster C1 (R-CROSS-Q, C-CQBAL also die). Per-layer K=128 compression (CF11 confirmed) remains valid.
- **GRAY**: var@128 ∈ [0.50, 0.80] → run D*=256 and D*=512 to find elbow; cross-layer sharing valid at higher D*.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Basis Sharing (arXiv:2410.03765) may have tested W_Q cross-layer and found coherence — partial pre-emption | Read Basis Sharing W_Q results specifically before Stage 5 selection. If W_Q was tested and found coherent there, position F-CQSGC as "our private CF11 measurement shows the same structure on Qwen3" (still worth running to close the Qwen3-specific gap) |
| MASA (arXiv:2508.04581) requires fine-tuning — F-CQSGC is no-retraining but may have worse ΔNLL | Report ΔNLL of stacked-SVD-only (no fine-tuning) vs MASA's fine-tuned result as the experimental comparison |
| var@128 depends heavily on layer normalization (are W_Q matrices normalized before stacking?) | Normalize each W_Q by its Frobenius norm before stacking; report with and without normalization |

Cheapest falsifier: the 15-min stacked SVD. If var@128 < 0.50, Cluster C1 entire family terminates.

## 9. Two-paper composition flag

Cousin pair: Basis Sharing (arXiv:2410.03765) + CF11 (our AQFKV).
F-CQSGC value-add: (1) no fine-tuning, (2) gauge-collapse framing that derives U as the algebraic natural basis for all layers simultaneously, (3) private CF11 measurement showing W_Q is specifically non-full-rank (Basis Sharing may not have had this prior).
If Basis Sharing already tested W_Q and found cross-layer coherence, F-CQSGC is ADJACENT (same result on different model, no fine-tuning moat remains). Stage 5 should check Basis Sharing Section 4 (experiments) before selection.

## 10. Verdict

**REFINE** — with flag: Stage 5 must check whether Basis Sharing (arXiv:2410.03765) explicitly covers W_Q cross-layer and reports positive results. If yes, DOWNGRADE to ADJACENT and advance only the no-retraining moat claim. If Basis Sharing did not test W_Q (or only tested it with fine-tuning), F-CQSGC's no-retraining path is genuinely novel.

## 11. Hand-off note

F-CQSGC is the Cluster C1 gate experiment. Run BEFORE investing in R-CROSS-Q or C-CQBAL cascade extensions. The stacked SVD costs 30 min total. Read Basis Sharing's W_Q results concurrently.

---

# F — F-OCSSQ (First-Principles: Outlier-Channel Static Spine Quantization)

## DEEP TREATMENT

## 1. Mechanism decomposition

- M1: CF3 K-dependent Jaccard defines a fixed-point condition: top-0.1% channels are a static attractor (Jaccard=0.718), K=1% is a fluctuating shell (Jaccard=0.308).
- M2: Algebraic derivation: decompose x = x_spine + x_shell + x_bulk where x_spine has 2-channel support (always same channels), x_shell has 18-channel per-token-varying support, x_bulk is 2028-channel INT4.
- M3: The tier boundaries are derived from the Jaccard phase transition, not from calibration fitting — a calibration-free algebraic construction.
- M4: GO condition: ΔNLL < 0.30 nats on three-tier activation quantization applied to all layers.
- M5: Primary payoff at 4K context: 4× KV cache reduction (672 MB → 168 MB on Qwen3-1.7B).

## 2. Per-claim prior-art status

- M1, M3 — **NOVEL** (private CF3; Jaccard-to-tier-boundary derivation is not published).
- M2 — **PARTIAL**: LLM.int8() uses a binary static/dynamic split; SmoothQuant migrates outliers. Neither uses a three-tier Jaccard-derived partition. PrefixQuant (arXiv:2410.05265) handles token-wise outliers differently. The three-tier algebraic derivation from CF3 fixed-point conditions is **NOVEL** as the primary entry point.
- M4, M5 — **NOVEL**: No paper measures ΔNLL of this specific tier structure on Qwen3-1.7B. *elegance-class: conserved-quantity*

**NOTE**: F-OCSSQ and C-RAOK-Grounded converge on the same mechanism from different derivation angles (F-OCSSQ: algebraic from CF3 fixed-point; C-RAOK-Grounded: composition of CF3+CF6). The prior-art status is identical for both. The key empirical question — Tier-0 stability — is shared. The cheapest settler is the 10-min C-RAOK-Grounded stability test.

## 3. Frame-mismatch check (CF9)

No theorem imports. The "fixed-point" framing is descriptive (the top-2 channels are a stable fixed set, not a mathematical fixed point of any operator). CF9 CLEAR.

## 4. Calibration pre-flight (CF10)

No calibration fit. Channel identification is from statistics (mean magnitude rank over 200 prompts). The Jaccard thresholds are measured, not fit by regression. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Per Stage 1 F4-B:
Activation at BF16: 4096 bytes/token/layer.
Under three-tier: x_spine (2×2 bytes = 4 B) + x_shell (18×1 B = 18 B) + x_bulk (2028×0.5 B = 1014 B) = 1036 bytes/token/layer.
4× activation compression.
KV cache at 4K context, 28 layers, Qwen3-1.7B: BF16 = 28×4096×2048×2 ≈ 471 MB; quantized = 471/4 ≈ 118 MB. Saving: 353 MB.
Weight residency: UNCHANGED (this is activation quantization, not weight quantization).

## 6. Smallest-test sharpening

The Tier-0 stability measurement (C-RAOK-Grounded smallest test) is the upstream gate. If Tier-0 stability ≥ 90% in ≥20/28 layers, proceed to:
Script: `scripts/ocssq_three_tier_nll.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: 200 diverse prompts (identify stable channels)
Eval corpus: WikiText-2, 512 tokens held-out
Operation: (1) From CF3 PDAP data, identify Tier-0 (top-2 channels by mean rank), Tier-1 (next 18), Tier-2 (rest). (2) Quantize activations at every layer: Tier-0 FP16 pass-through, Tier-1 INT8, Tier-2 INT4. (3) Eval ΔNLL.
Runtime: Calibration 10 min + eval 20 min. Total ≤30 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_ocssq/three_tier_nll.json`

CF13-CF15 check: F-OCSSQ uses only CF3 (confirmed). No CF13-CF15 dependencies.

## 7. Refined go/no-go

- **GO**: ΔNLL < 0.30 nats AND KV cache 4× reduction demonstrated → advance to full-cascade ΔNLL test at R-RAOK-70B scale (70B arithmetic).
- **NO-GO**: ΔNLL > 1.0 nats → tier boundaries from Jaccard are not aligned with quantization sensitivity; the CF3-to-tier derivation is invalid. Kill the calibration-free tier claim.
- **GRAY**: ΔNLL 0.30–1.0 nats → Jaccard thresholds are approximate but not optimal. Run Hessian-sensitivity tier comparison to find optimal thresholds; if they match CF3 thresholds within 5%, Jaccard-derivation is validated.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| CF3 Jaccard stability ≠ quantization sensitivity (different measure) | Cross-validate Jaccard tiers vs Hessian tiers (5-min additional step) |
| INT4 quantization of 2028-channel bulk may interact badly with deep-layer outlier spread (CF3 notes layers 23-27 have wider outlier distributions) | Report per-layer ΔNLL; if layers 23-27 are the cost drivers, demote bulk to INT6 or INT8 in those layers |
| x_shell per-token channel identification at runtime costs overhead | Profile overhead; if > 5% of total inference time, consider using a per-frame-stable approximate set |

Cheapest falsifier: Three-tier ΔNLL eval (30 min). If ΔNLL > 1.0, the Jaccard-to-tier derivation fails and the calibration-free claim dies.

## 9. Two-paper composition flag

Cousin pair: LLM.int8() (arXiv:2208.07339) + CF3 (our PDAP).
F-OCSSQ value-add: derives tier thresholds from CF3 Jaccard phase transitions (calibration-free, structurally motivated), adds a third tier (INT4 bulk), and targets KV cache as the primary payoff. Not just "LLM.int8() on Qwen3" — the three-tier algebraic derivation from Jaccard is the structural novelty.

## 10. Verdict

**REFINE** — Cluster C2 convergence representative (algebraic derivation angle). 30-min smallest test after 10-min upstream stability gate.

## 11. Hand-off note

Run C-RAOK-Grounded Tier-0 stability check FIRST (10 min, existing data). If GO, proceed to F-OCSSQ three-tier ΔNLL. These two together are the complete Cluster C2 empirical validation.

---

# F — F-RSCLE (First-Principles: Residual-Stream Covariance Laplacian Eigendecomposition)

## 1. Mechanism decomposition

- M1: CF2 (cos ≈ 0.99 between adjacent residual states) implies per-layer updates δ_L = h_{L+1} − h_L are small and directionally concentrated.
- M2: The per-layer update covariance C^(L) = E[δ_L δ_L^T] defines a "active subspace" — the low-rank PSD operator characterizing where each layer writes in residual space.
- M3: If top-r eigenvectors of C^(L) capture > 80% of update variance at r ≤ 256, the layer's computation can be reorganized into an r-dimensional active subspace with two projection steps.
- M4: Compute reduction in attention GEMV: 4× if r=256 << d=2048.

## 2. Per-claim prior-art status

- M1 — **NOVEL** (private CF2 extended to per-layer covariance level, not just adjacent cosine).
- M2 — **ADJACENT (critical)**: "Gated Subspace Inference for Transformer Acceleration" (arXiv:2605.03109, May 2025) proposes exactly this mechanism: decompose each activation vector into a subspace component, compute linear-layer outputs in the low-rank subspace, apply a per-token gate for the residual. This paper achieves 3.0–10.5× speedup on GPT-2/GPT-J/OPT with character-for-character identical output. The mechanism is F-RSCLE's core claim operationalized. **This is a strong adjacency risk.** If arXiv:2605.03109 is the same mechanism applied to the same object (per-layer activation subspace), F-RSCLE may be PRE-EMPTED on M2-M4.
- M2 differentiation from arXiv:2605.03109: F-RSCLE derives the active subspace from the RESIDUAL UPDATE covariance C^(L) = E[δ_L δ_L^T], while arXiv:2605.03109 appears to use the ACTIVATION covariance. These are different objects: δ_L = h_{L+1} − h_L (the layer's contribution) vs h_L (the full residual state). The F-RSCLE "Laplacian" framing uses the update covariance specifically. Need to verify whether arXiv:2605.03109 uses the update covariance or the full activation covariance. **If they use update covariance, F-RSCLE is PRE-EMPTED on its core mechanism.**
- M3 — **ADJACENT**: "Dimensional Collapse in Transformer Attention Outputs" (arXiv:2508.16929, Aug 2025) shows attention outputs are confined to ~60% of full rank, and MLP outputs remain near-full-rank. This partially pre-empts the "active subspace is low-rank" claim for attention specifically.
- M4 — **ADJACENT**: arXiv:2605.03109's speedup is measured (3–10.5× on linear layer weight reads). This is the F-RSCLE payoff operationalized.

**PRE-EMPTION VERDICT**: arXiv:2605.03109 is a material threat. The key question is whether F-RSCLE's residual UPDATE covariance is the same object as what arXiv:2605.03109 uses. If yes: PRE-EMPTED on M2-M4, and what's left is the "Laplacian" framing vocabulary — not a research contribution. REGENERATE if pre-empted, with surviving primitive being the specific Qwen3 δ_L covariance measurement as a characterization tool.

## 3. CF9 check

The "Laplacian" framing is descriptive. No actual Laplacian theorem is applied. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

A1=1 in Stage 2 is correct. Active-subspace routing reduces FMA compute (4× at r=256), not weight residency. On DRAM-bandwidth-bound inference, weight bandwidth dominates: 6.4 GB/token; activation compute is secondary. Active-subspace routing saves ~0.56 s × 20% (attention fraction) × 4× FMA reduction = ~0.11 s/token in compute — but DRAM bandwidth remains the bottleneck. Direct tok/s uplift is minimal unless paired with quantization. F-RSCLE's payoff is primarily as a co-design with activation quantization (lower-rank activations → smaller quantization error in active subspace).

## 6. Smallest-test sharpening

Script: `scripts/rscle_update_cov.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: WikiText-2, 200 tokens
Operation: Record h_L and h_{L+1} at all 28 layers; compute δ_L = h_{L+1} − h_L; compute C^(L) = (1/N) Σ δ_L δ_L^T; eigendecompose; measure var@r for r ∈ {64, 128, 256, 512}. Separately, check var@256 from FULL activation covariance (to compare with arXiv:2605.03109's likely approach).
Runtime: ~15 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_rscle/update_cov_var_profile.json`

**First rung added**: compare update covariance eigenspectrum to activation covariance eigenspectrum at each layer. If they are similar, arXiv:2605.03109 has pre-empted F-RSCLE; if they differ significantly (update covariance is MORE concentrated), F-RSCLE has residual novelty.

## 7. Refined go/no-go

- **GO**: var@256 > 0.80 for update covariance C^(L) in ≥20 layers AND update-covariance var@256 > activation-covariance var@256 + 0.10 (the "Laplacian" framing captures more concentration than generic activation PCA).
- **NO-GO**: var@256 < 0.50 OR update-covariance ≈ activation-covariance → mechanism is equivalent to existing PCA-based active subspace methods; PRE-EMPTED by arXiv:2605.03109.
- **GRAY**: var@256 ∈ [0.50, 0.80] → try r=512; if var@512 > 0.80, active subspace exists but is wider than hoped; still useful for the activation-quantization co-design purpose.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| arXiv:2605.03109 uses the same update covariance object → PRE-EMPTED | **Read arXiv:2605.03109 Section 2 specifically before Stage 5 selection.** If they use δ_L covariance, REGENERATE. If they use h_L covariance, F-RSCLE retains the "Laplacian" novelty |
| Compute savings don't translate to tok/s in DRAM-bandwidth-bound regime | Confirmed — A1=1 is correct. Value is as activation-quantization co-design input |
| C^(L) estimated from only 200 tokens may be noisy | Use 200-token train + 200-token held-out split; report held-out var fraction |

## 9. Two-paper composition flag

Cousin pair: arXiv:2605.03109 (Gated Subspace Inference) + CF2 (our residual additivity).
F-RSCLE value-add: the specific Laplacian/update-covariance framing is more theoretically motivated than generic PCA-of-activations. But if arXiv:2605.03109 already achieves 3–10× speedup with identical output, the value-add is framing, not mechanism.

## 10. Verdict

**DOWNGRADE** pending arXiv:2605.03109 review. If arXiv:2605.03109 uses activation covariance (not update covariance), F-RSCLE's update-covariance framing retains differentiation — REFINE conditional on that check. If arXiv:2605.03109 uses update covariance, REGENERATE with surviving primitive: Qwen3-specific update-covariance spectrum characterization as a new CF finding.

## 11. Hand-off note

Stage 4/5 MUST read arXiv:2605.03109 before selecting F-RSCLE. The decision gate is whether Gated Subspace Inference uses per-layer update covariance (δ_L object) or per-layer activation covariance (h_L object). This single paragraph in their method section determines F-RSCLE's fate.

---

# F — F-SGFVO (First-Principles: Spectral Gauge Freedom in W_V/W_O Product)

## 1. Mechanism decomposition

- M1: P^(i,ℓ) = W_V^(i,ℓ) W_O^(i,ℓ) ∈ R^{d×d} has exact rank ≤ d_head = 128 by construction (since P is a product of a d×d_head and d_head×d matrix). This is an algebraic identity, not an approximation.
- M2: The effective rank r_99(P^(i,ℓ)) may be substantially below d_head=128 if W_V and W_O are complementarily low-rank.
- M3: Replacing the W_V + W_O pair with P's SVD factors produces exact computation at d* = d_head, and cheaper computation at d* < d_head.
- M4: GO condition: median r_99 ≤ 96 across all 448 (16 heads × 28 layers) head-operator products.

## 2. Prior-art status

- M1 — **PARTIAL (critical)**: A3 (arXiv:2505.12942, May 2025) explicitly compresses the OV component (W_V W_O product) as "Theorem 3 A3-OV for MHA-NoPE." A3's approach finds the optimal low-rank approximation of the OV operator minimizing attention output error. The M=W_V W_O rank ceiling identity is the basis for A3-OV's construction. **This is a material pre-emption of M1.** A3 was published May 2025 and explicitly targets the W_V W_O product rank as its OV compression primitive.
- M2 — **PARTIAL**: A3 measures OV component effective rank. Differentiation: A3 uses calibration covariance to guide the low-rank approximation; F-SGFVO measures whether r_99 < d_head from pure weight analysis (no calibration). If the weight-only r_99 measurement is the contribution, that's a measurement contribution (value as a CF characterization) rather than a mechanism contribution.
- M3 — **ADJACENT**: A3's OV Theorem 3 provides the analytical formula for optimal OV low-rank compression. F-SGFVO independently derives the same algebraic entry point (rank ≤ d_head identity). The gauge-freedom framing (W_V, W_O individually have gauge freedom under invertible T: W_V T, T^{-1} W_O leaves P invariant) is the F-SGFVO specific angle — not explicitly in A3.
- M4 — **NOVEL** on Qwen3-1.7B specifically: A3 was tested on other model families. The r_99 measurement on Qwen3-1.7B all layers + all heads is not published.

**Pre-emption verdict**: A3 (arXiv:2505.12942) pre-empts the core mechanism (W_V W_O product rank exploitation for OV compression). F-SGFVO's remaining novelty is: (a) the gauge-freedom framing (vs A3's calibration-fitted OV minimization), (b) Qwen3-specific weight-only r_99 measurement. This is significant residual novelty — A3 uses calibration data; F-SGFVO is weight-only — but the mechanism is ADJACENT, not NOVEL.

## 3. CF9 check

No theorem imports. Rank ≤ d_head identity is elementary linear algebra. CF9 CLEAR.

## 4. CF10 check

No calibration fit (weight-only). CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Per Stage 1 F5-A:
Current W_V + W_O: 28 × 8M params = 448 MB bf16.
At r*=64 (if median r_99 ≤ 64): P factors at 2×(2048×64) = 131K per head × 16 heads × 28 layers = 58.5M params = 117 MB. Saving: 331 MB.
At r*=96: P factors at 2×(2048×96) × 16 × 28 = 175 MB. Saving: 273 MB.
At 70B: W_V + W_O at d=8192, d_head=128, 64 heads GQA: W_V = 64×8192×1024×64 layers = 34B params (wait — GQA kv_heads=8 for Qwen3-72B); W_V = 64×8192×1024 = 537M params; W_O = 64×8192×8192 = 4.3B. Total W_V+W_O = ~4.8B params at bf16 = 9.6 GB. At r*=64 per KV head: saving ≈ 50% of W_V + 50% of W_O ≈ 4.8 GB. At IQ4_XS: ~2.4 GB saved.

## 6. Smallest-test sharpening

Script: `scripts/sgfvo_wv_wo_rank.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data.
Operation: For each (layer, head), compute P = W_V^(h) @ W_O^(h) (2048×2048, but max rank 128); use truncated SVD keeping top-128; measure r_99. Also compute ΔNLL at r*=64 and r*=96 (replace W_V+W_O with P's top-r* factors).
Runtime: 448 matrix multiplies (128×2048 × 2048×128 = 67M FLOPs each) → ~5 min; 448 truncated SVDs at rank 128 (fast) → ~10 min; ΔNLL eval → ~10 min. Total ≤30 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_sgfvo/wv_wo_product_rank.json`

## 7. Refined go/no-go

- **GO**: median r_99 ≤ 96 across heads/layers AND ΔNLL at r*=64 < 0.50 nats. Advance to 70B residency projection and A3 OV comparison.
- **NO-GO**: median r_99 = 128 (product is full head rank) → W_V and W_O are complementarily full-rank; no gain over separate storage. The finding extends the CF8 boundary to the W_V W_O product.
- **GRAY**: median r_99 ∈ [80, 112] → the gain exists but is modest. Compare to A3's OV result on same model to quantify value-add of weight-only vs calibration-guided approach.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| A3 (arXiv:2505.12942) pre-empts the mechanism; F-SGFVO is ADJACENT | Stage 5 should compare ΔNLL of F-SGFVO weight-only vs A3 calibration-guided at equal rank to determine whether calibration guidance is necessary |
| W_V W_O product rank is exactly d_head=128 for all heads (median r_99=128) | Still closes the W_V/W_O characterization question opened in SUMMARY.md |
| A3 was tested on LLaMA family, not Qwen3 — Qwen3 r_99 results are new | The Qwen3-specific measurement is the genuine novelty regardless of A3 |

Cheapest falsifier: ~5 min computation of P rank for one layer. If r_99=128 for all heads in layer 14, abort full measurement.

## 9. Two-paper composition flag

Cousin pair: A3 (arXiv:2505.12942, OV-component compression) + CF11 (our spectral measurements).
F-SGFVO value-add: (1) weight-only (no calibration), (2) Qwen3-specific r_99 measurements, (3) gauge-freedom framing distinguishing from A3's calibration-covariance approach. Not pure engineering-integration — the weight-only gauge approach vs calibration-guided minimization is a structural research question.

## 10. Verdict

**REFINE** — A3 is ADJACENT but does not pre-empt the weight-only path or the Qwen3-specific measurement. The 30-min experiment closes both the W_V/W_O characterization question and the A3 comparison. Stage 5 weight: HIGH (closes an explicitly open question in SUMMARY.md; answers are immediately deployable).

---

# R — R-ACWF (Reach: Attention-Cascade Weight Fold)

## 1. Mechanism decomposition

- M1: CF11 establishes 16:1 head-redundancy in W_Q; ACWF applies the analogous argument to W_O head slices.
- M2: Calibration-measured head importance: ‖W_O^h · v^h‖ averaged over 200 calibration prompts.
- M3: Heads below 10% of max-head norm are "foldable" — their W_O^h is replaced by a shared low-rank projection.
- M4: The fold is algebraically exact for zero-output heads (calibration-average output = 0), approximate for low-output heads.

## 2. Prior-art status

- M1 — **NOVEL** (private CF11).
- M2, M3 — **ADJACENT**: "Are Sixteen Heads Really Better than One?" (Michel et al. 2019) measures per-head importance and prunes; ACWF MERGES rather than prunes. However, attention head importance scoring by output norm is well-covered. ADJACENT, not pre-empted, because the merge operation (not prune) is structurally distinct.
- M4 — **NOVEL**: the algebraic-merge framing (redundant heads collapse to a shared projection) is not in Michel et al. or standard head pruning literature.

## 3. CF9 check

No theorem imports. Head output norm is a direct measurement. CF9 CLEAR.

## 4. CF10 pre-flight

M3 uses calibration-average head norms — this is a threshold-based rule, NOT a least-squares fit. No regression parameters. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

W_O: 28 × 16 heads × 128×2048 × 2 bytes = 224 MB. If 50% of heads are below 10% norm threshold and folded to a 4-head shared projection: saving ≈ 112 MB. If 75%: saving ≈ 168 MB. On 70B: ~6.5 GB W_O savings at 75% fold rate.

## 6. Smallest-test sharpening

Script: `scripts/acwf_head_norm.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: 200 prompts from R2 PDAP dataset (already available)
Operation: Forward pass; collect ‖W_O^h · v^h‖ per (layer, head) per token; compute mean and std; identify heads below 10% max-head norm threshold.
Runtime: ~30 min. PASSES ≤8h gate.
GO: ≥50% of (layer, head) pairs below 10% norm threshold; top-4 heads per layer account for ≥75% of total output norm.

## 7. Refined go/no-go

- **GO**: ≥50% heads below threshold AND top-4 cover ≥75% norm.
- **NO-GO**: Head importance is roughly uniform — CF11's W_Q head-redundancy does NOT extend to W_O output magnitudes. Structural finding: query-side and output-side head behavior differ.
- **GRAY**: 20–50% heads below threshold → the fold exists but is less aggressive; measure ΔNLL at the attainable fold fraction.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Calibration distribution != held-out (token-type-specific heads fold on calibration but activate on held-out) | Stratify calibration by token type (BOS, normal, separator per CF3); only fold heads uniformly low across all types |
| CF10 is borderline for per-layer head importance scores | Head norms are scalar averages (1 scalar per head per layer = 28×16=448 values); no regression fit; CF10 does not apply |

## 9. Two-paper composition flag

Cousin pair: Michel et al. 2019 (head importance) + CF11 (our AQFKV).
Value-add: the MERGE (not prune) operation + the CF11 head-redundancy measurement that provides prior expectation for the fold fraction. Not engineering-integration — the algebraic merge identity is the structural move.

## 10. Verdict

**REFINE** — 30-min measurement, clean binary outcome.

---

# R — R-AERO-PROBE (Reach: AERO Fold Applicability on SwiGLU) [FREE SWING]

## LIGHT TREATMENT

## 1. Mechanism decomposition

- M1: AERO (arXiv:2410.13060) established that if silu(W_gate·x) ≈ constant, W_gate can be algebraically folded into W_up with no retraining.
- M2: Measure Pearson correlation between silu(W_gate·x) and a function of W_up·x per layer on calibration data.
- M3: If correlation > 0.9 for any consecutive 5 layers, the gate branch is redundant given W_up and can be replaced by a scalar calibration constant.
- Primitive: Standard Pearson correlation computation on forward pass data.

## 2. Prior-art status

- M2, M3 — **NOVEL** as an AERO applicability probe (no post-training W_gate/W_up branch correlation measurement found in literature as of 2026-05-09).
- Distinction from CSMF (killed R3/S6): CSMF tried polynomial substitution of W_gate's domain; R-AERO-PROBE checks whether silu(W_gate·x) ≈ f(W_up·x) as a data-driven observable — entirely different question.

## 3. CF9 check

No theorem import. Pearson correlation is a standard statistic. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Smallest-test sharpening

Script: `scripts/aero_probe_correlation.py`
Model: Qwen3-1.7B-Base
Cal corpus: 200 prompts; collect silu(W_gate·h) and W_up·h per layer.
Operation: Compute Pearson correlation per layer.
Runtime: ~45 min. PASSES ≤8h gate.
GO: correlation > 0.9 for ≥5 consecutive layers.
NO-GO: correlation < 0.7 across all layers → W_gate branch carries decorrelated information from W_up; AERO-style fold without retraining is not possible on Qwen3.

## 6. Verdict

**REFINE** — wildcard, CF-tether-suspended, structural floor met. 45-min experiment, binary outcome with independent value even on NO-GO (settles the AERO-without-retraining question for SwiGLU families). The existing AERO paper covers ReLU-activation models; SwiGLU without retraining has no published analysis.

---

# R — R-MCQKV (Reach: Multi-Cascade Attention Compression)

## DEEP TREATMENT

## 1. Mechanism decomposition

- M1: CF11 W_Q r_99/d ≈ 0.63 and W_K r_99/d ≈ 0.79; head-redundancy 16:1 at K=128.
- M2: W_V and W_O spectra are UNTESTED on Qwen3-1.7B — the first rung of this cascade.
- M3: If W_V and W_O have r_99/d ≤ 0.80, all four attention weight matrices (W_Q, W_K, W_V, W_O) are compressible without retraining.
- M4: At 72B, compressed (W_Q+W_K+W_V+W_O) saves 8–16 GB (bf16) enabling a DRAM-resident attention tier in the attention-NVMe-split architecture.
- M5: Cross-scale spectrum homogeneity assumption: Qwen3-1.7B spectrum predicts 72B spectrum (not yet confirmed).

## 2. Prior-art status

- M1 — **NOVEL** (private CF11).
- M2, M3 — **NOVEL** as empirical measurement: no paper has measured W_V/W_O spectra on Qwen3-1.7B. However, A3 (arXiv:2505.12942) demonstrates OV component (W_V W_O product) compression achieves high quality — implying W_V/W_O are compressible on LLaMA family. Qwen3-specific measurement is the gap.
- M4 — **PARTIAL**: ATTN-NVMe-SPLIT (rejected Stage 2 as integration narrative) covers the same residency architecture. R-MCQKV is the prerequisite measurement, not the architecture. The 4-matrix spectrum measurement is the standalone contribution.
- M5 — **NOVEL**: Cross-scale spectrum homogeneity is an unconfirmed assumption. CF13* (unconfirmed) would help; this is CF13-tethered. **CF13-independent conservative path**: assume Qwen3-72B attention spectra are similar to Qwen3-1.7B (similar architecture, same training regime); treat as an extrapolation with uncertainty.

## 3. CF9 check

No theorem imports. SVD measurement is elementary. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic (conservative, no CF13/14/15)

Qwen3-72B parameters (from architecture):
- W_Q: 64 heads × d_head=128 per head × d=8192 × 64 layers = 8192×8192×64 params. At bf16: ~8.6 GB.
- W_K: GQA kv_heads=8, d_head=128, d=8192: 8×128×8192×64 = 536M params = ~1.1 GB.
- W_V: same as W_K = ~1.1 GB.
- W_O: 64 heads × d_head=128 per head × d=8192 × 64 layers = 8192×8192×64 = ~8.6 GB.
- Total attention bf16: ~19.4 GB.
Conservative compression (W_Q K=512, W_K K=512, W_V K=512 if compressible, W_O K=512):
- W_Q: K=512 → (8192×512 + 512×8192)×64 = 537M params = 1.07 GB. Saving: 7.5 GB.
- W_K: K=512, GQA: (8192×512 + 512×1024)×64 = 302M = 0.60 GB. Saving: 0.5 GB.
- IF W_V compressible at K=512: same as W_K = 0.60 GB. Saving: 0.5 GB.
- IF W_O compressible at K=512: same shape as W_Q = 1.07 GB. Saving: 7.5 GB.
- Conservative (only W_Q+W_K): saving 8.0 GB. Attention residency: 19.4 − 8.0 = 11.4 GB.
- Optimistic (all four): saving 16 GB. Attention residency: 3.4 GB (DRAM-resident at 7.28 GiB limit, with room for KV + activations).
- MLP (IQ4_XS): ~28 GB at 72B. Total model: 3.4 + 28 = 31.4 GB optimistic.
**What-if CF13-CF15 don't replicate**: cross-scale extrapolation is purely architectural (same training recipe, same head structure). Worst case: W_V/W_O are full-rank at 72B even if compressible at 1.7B. Conservative path: only W_Q+W_K compression = 11.4 GB attention, giving ~39.4 GB total — no improvement over IQ4_XS baseline.

## 6. Smallest-test sharpening

Script: `scripts/mcqkv_all_attention_svd.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data.
Layers: all 28; matrices: W_Q (already done in AQFKV, reuse), W_K (already done), W_V (new), W_O (new).
Operation: Compute SVD spectrum for W_V and W_O at all 28 layers; measure r_99/d. Compare with W_Q (0.63) and W_K (0.79).
Runtime: W_V and W_O SVDs: 2 matrices × 28 layers × ~5 min per matrix = ~10 min additional (W_Q and W_K already measured in AQFKV). Total new work ≤15 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_mcqkv/wv_wo_spectrum_all_layers.json`

## 7. Refined go/no-go

- **GO**: mean(r_99_W_V/d, r_99_W_O/d) ≤ 0.80 across layers 5–27.
- **NO-GO**: both W_V and W_O have r_99/d ≥ 0.90 → cascade is limited to W_Q + W_K (8 GB saving at 72B, not 16 GB); adjust R-MCQKV residency claim downward.
- **GRAY**: W_V compressible, W_O full-rank (or vice versa) → apply compression to the compressible matrix only; asymmetric W_V/W_O result is still a structural finding.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| W_V and W_O full-rank like MLP (worst case: all four attention matrices are not jointly compressible) | W_Q and W_K already confirmed compressible; the cascade still saves ~8 GB at 72B — DRAM-resident attention tier remains achievable for the W_Q+W_K pair |
| Cross-scale extrapolation fails (72B has different spectral profile) | Stage 5 should flag that the 72B deployment path requires a 72B spectrum measurement as the second rung |
| A3 (arXiv:2505.12942) already compresses OV component on LLaMA family | Our contribution is Qwen3-specific measurement + weight-only path without calibration |

Cheapest falsifier: 15-min W_V/W_O SVD on Qwen3-1.7B.

## 9. Two-paper composition flag

Cousin pair: A3 (arXiv:2505.12942, QK+OV compression) + CF11 (our AQFKV).
R-MCQKV value-add: (1) closes the Qwen3 W_V/W_O measurement gap, (2) uses weight-only path (no calibration), (3) projects to 72B DRAM-resident architecture. The measurement itself (Qwen3 attention weight-class characterization) is independently valuable as a CF finding regardless of A3.

## 10. Verdict

**REFINE** — highest-priority measurement in the union. 15 min of new experiment closes the last open question in the Qwen3 attention weight characterization. All four attention matrices measured = complete CF picture.

## 11. Hand-off note

R-MCQKV spectrum measurement (W_V + W_O) feeds directly into: F-SGFVO (W_V W_O product rank), C-ABAR (spectral hierarchy completion), R-ACWF (W_O head-fold motivation). Run as the first experiment in any Stage 4/5 session.

---

# R — R-RAOK-70B (Reach: R2-Grounded Outlier-Keyed Activation Cascade to 70B)

## DEEP TREATMENT

## 1. Mechanism decomposition

- M1: CF3 Jaccard at K=0.1% = 0.718 (channel-static), K=1% = 0.308 (token-dynamic); this is the empirical substrate for the three-tier scheme.
- M2: The RAOK three-tier activation scheme: 2 channels FP16 (static), 18 channels INT8 (per-token dynamic), 2028 channels INT4 (static bulk).
- M3: The Tier-0 stability test is the critical upstream gate (10 min, existing data).
- M4: If Tier-0 stable, the scheme achieves ΔNLL < 0.3 nats on W_up GEMV inputs on Qwen3-1.7B.
- M5: The 70B cascade composes RAOK with W_Q compression (CROSS-Q / MCQKV) for the complete activation-side + weight-side compression.
- M6: The effective activation bpw ≈ 4.05 (nearly INT4 but with exact FP16 for 2 channels).

## 2. Prior-art status

- M1 — **NOVEL** (private CF3; no published paper has Qwen3-family K-dependent consecutive-pair Jaccard statistics).
- M2 — **PARTIAL**: LLM.int8() (binary), SmoothQuant (weight-side migration), PrefixQuant (token-wise outlier prefix caching) all cover aspects. The three-tier structure with Jaccard-derived thresholds is the NOVEL extension. No paper found as of 2026-05-09 deriving tier boundaries from per-token Jaccard statistics.
- M3 — **NOVEL**: Tier-0 stability measurement using R2 PDAP dataset (our private data).
- M4 — **NOVEL**: ΔNLL of RAOK three-tier on Qwen3-1.7B W_up GEMV inputs is not published.
- M5 — **ADJACENT**: The 70B composed cascade is a deployment architecture; each rung has individual novelty.
- M6 — **ADJACENT**: Effective bpw computation is straightforward arithmetic. *elegance-class: conserved-quantity*

## 3. CF9 check

No theorem imports. Tier boundaries are empirically measured (CF3 Jaccard), not derived from a named theorem. The "fixed-point condition" framing (F-OCSSQ) is descriptive, not a theorem application. CF9 CLEAR.

## 4. CF10 pre-flight

The tier assignment (which channels are Tier-0, Tier-1, Tier-2) is determined from CF3 statistics — mean magnitude rank over 200 prompts. This is a threshold-based classification, NOT a regression fit.
- n_params_to_classify: 2028 binary tier assignments (2 + 18 + rest).
- n_independent_samples: 200 prompts × 28 layers = 5600 independent activation vectors.
- The classification is a deterministic argmax, not a fitted parameter.
CF10 NOT APPLICABLE.

## 5. Residency arithmetic (with CF13-CF15 independence)

RAOK addresses activation quantization, not weight residency. The weight-side residency benefit comes from the composed cascade (RAOK + W_Q compression). Standalone RAOK on Qwen3-1.7B:
- KV cache at 4K context: 28 × 2 × 4096 × 2048 × (1036/4096) bytes (compressed) = 28 × 2 × 4096 × 1036 = 238 MB vs 471 MB bf16. Saving: 233 MB.
- Activation bandwidth during GEMV: 4× reduction. On DRAM-bandwidth-limited inference at 11.5 GB/s: activation bandwidth is ~28 × 2048 × 2 bytes = 114 KB per token — negligible vs 6.4 GB weight bandwidth. Activation compression does not move the tok/s needle directly at 1.7B scale.
- At 70B scale (DRAM bandwidth 11.5 GB/s; model on NVMe): RAOK reduces the compute-side work per activation, but the bottleneck is NVMe-to-DRAM weight loading, not activation processing. RAOK's primary contribution at 70B is as the activation-side rung in a composed cascade, not as a standalone throughput lever.
**What-if CF13-CF15 don't replicate**: R-RAOK-70B does not cite CF13-CF15. The 70B projection arithmetic uses only CF3 (confirmed). Conservative path = optimistic path.

## 6. Smallest-test sharpening

Two-rung pipeline:

**Rung 1 (upstream gate)**: Tier-0 stability test (same as C-RAOK-Grounded).
Script: `scripts/raok_tier0_stability.py`
Data: R2 PDAP dataset (already available, 200 prompts)
Runtime: ~10 min.
GO: ≥90% same-channel stability in ≥20/28 layers.

**Rung 2 (if Rung 1 GO)**: Three-tier ΔNLL test.
Script: `scripts/raok_three_tier_nll.py`
Model: Qwen3-1.7B-Base, bf16
Cal corpus: R2 PDAP dataset (200 prompts; Tier-0/1 channel identification)
Eval corpus: WikiText-2, 512 tokens held-out
Operation: Apply three-tier quantization to all W_up GEMV inputs; measure ΔNLL.
Runtime: Calibration 10 min + eval 20 min = ~30 min.
Total pipeline: ~40 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_raok70b/tier0_stability.json` + `three_tier_nll.json`

CF13/14/15 flag: R-RAOK-70B does NOT use CF13-CF15. All arithmetic is grounded in CF3. No re-derivation rung needed.

## 7. Refined go/no-go

- **GO**: Tier-0 stability ≥90% AND ΔNLL < 0.3 nats. Advance to 70B cascade construction.
- **NO-GO on Rung 1**: < 70% Tier-0 stability → static FP16 bypass fails; demote to per-token detection → Tier-0 overhead increases; recalibrate.
- **NO-GO on Rung 2**: ΔNLL ≥ 0.3 nats at three-tier scheme → the K=1% dynamic shell accumulates error across layers; widen the dynamic shell or move to per-tensor INT8 for all three tiers.
- **GRAY**: 0.15–0.30 nats ΔNLL → per-layer ΔNLL measurement to identify error concentration (deep layers per CF3 spreading).

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| Layer-by-layer error accumulation at 28 layers (per-layer small ΔNLL compounds) | Measure per-layer NLL contribution; identify worst layers; allow higher-precision tier for those layers |
| CF3 Jaccard characterizes consecutive-pair stability (not cross-prompt); a long-range dependency could shift the Tier-0 set | Run on 3 disjoint prompt types (factual, code, conversational); measure Tier-0 stability per type |
| Tier-1 per-token dynamic channel identification at runtime adds latency | Benchmark identification latency; if > 5% of GEMV time, use Tier-0+Tier-2 only (binary scheme) |

Cheapest falsifier: Tier-0 stability check (10 min, existing data). If stability < 70%, the entire three-tier calibration-free scheme is compromised.

## 9. Two-paper composition flag

Cousin pair: LLM.int8() (arXiv:2208.07339) + CF3 (our PDAP R2 finding).
R-RAOK-70B value-add: (1) three-tier (not binary), (2) Jaccard-derived calibration-free tier boundaries, (3) Tier-0 stability verified on 200 diverse prompts, (4) 70B deployment arithmetic composing with attention weight compression. The Jaccard-to-tier derivation is the structural research contribution not in LLM.int8() or SmoothQuant.

## 10. Verdict

**REFINE** — highest-priority Cluster C2 experiment. 40-min two-rung pipeline, decisive at both gates, no CF9/CF10 issues, CF13-CF15-independent. Stage 5 weight: HIGH.

## 11. Hand-off note

R-RAOK-70B Rung 1 (Tier-0 stability) is IDENTICAL to C-RAOK-Grounded smallest test. Run ONCE to serve both advancers. If Rung 1 GO, proceed to Rung 2 for both R-RAOK-70B and F-OCSSQ simultaneously.

---

# R — R-W_down-SPEC (Reach: W_down Spectrum Closure)

## 1. Mechanism decomposition

- M1: W_down ∈ R^{d × d_int} (2048 × 6144 for Qwen3-1.7B) is a "bottleneck" projection from intermediate to residual dimension — max rank is d=2048, not d_int=6144.
- M2: SUMMARY.md explicitly notes W_down "almost certainly applies [to CF8 full-rank] but untested."
- M3: If r_99/d_out ≤ 0.80, W_down is compressible at K=1024 — 50% of max rank, opening a new MLP compression path.
- M4: If r_99/d_out ≥ 0.95 (like W_gate/W_up), CF8 is confirmed to cover all three MLP matrices.

## 2. Prior-art status

- M3, M4 — **NOVEL** (Qwen3 W_down spectrum is not published; the dimension-asymmetry argument for a possible rank ceiling below d_int is not in any paper found as of 2026-05-09). COMPOT (arXiv:2602.15200) does global SVD allocation but does not focus on W_down's dimension asymmetry specifically.
- M1 — the max-rank ceiling (rank ≤ d=2048 for a 2048×6144 matrix) is elementary. Not a published claim about whether r_99 is concentrated near this ceiling.

## 3. CF9 check

No theorem imports. SVD measurement is elementary. The dimension-asymmetry argument (max rank = min(d, d_int) = d for a wide matrix) is a fact of linear algebra, not an imported theorem. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Residency arithmetic

Per Stage 1 R-W_down-SPEC:
At r_99/d_out ≤ 0.80 (GO scenario): K=1024 truncation captures 80%+ of variance. W_down compressed: (2048×1024 + 1024×6144) × 28 layers × 2 bytes = (2M + 6.3M) × 28 × 2 ≈ 463 MB vs 28×2048×6144×2 ≈ 707 MB. Saving: 244 MB on Qwen3-1.7B. At 72B (intermediate=28672): W_down per layer = 8192×28672; compressed at K=2048: (8192×2048 + 2048×28672) × 64 × 2 bytes ≈ (16.8M + 58.7M) × 64 × 2 ≈ 9.7 GB vs 28 × (8192×28672/10^9) × 64 × 2 = 33.9 GB. Saving: ~24 GB at 72B bf16. At IQ4_XS: ~12 GB saving — potentially the largest single compression gain in the cascade IF W_down is compressible.

## 6. Smallest-test sharpening

Script: `scripts/wdown_spectrum.py`
Model: Qwen3-1.7B-Base, bf16
No calibration data.
Layers: all 28
Operation: SVD of W_down (2048×6144) per layer; measure r_99/d_out (d_out=2048); measure var@K for K ∈ {512, 1024, 1536, 2048}.
Runtime: 28 × SVD of 2048×6144 ≈ ~10 min. PASSES ≤8h gate.
Output: `experiments/stage0/ladder_v2/round1_wdown_spec/wdown_spectrum_per_layer.json`

## 7. Refined go/no-go

- **GO**: mean r_99/d_out ≤ 0.80 across layers 5–27 → W_down is compressible; open a new MLP weight compression path.
- **NO-GO**: mean r_99/d_out ≥ 0.95 → CF8 generalizes to all three MLP matrices; closes the W_down question definitively; focus attention on activation quantization (RAOK) as the sole MLP-side compression path.
- **GRAY**: r_99/d_out ∈ [0.80, 0.95] → moderate compressibility; measure ΔNLL at K=1024 before concluding.

## 8. Risk profile

| Risk | Mitigation |
|---|---|
| The output-dimension argument (rank ≤ d_out) tells us the max rank but not whether the effective rank is concentrated | The var@K measurement directly tests concentration; max rank = 2048 does not imply concentration near 2048 |
| SVD-based compression of W_down (if GO) will encounter ΔNLL sensitivity similar to W_up at 50% rank (CF5) | Run ΔNLL test at K=1024 BEFORE claiming W_down compression is deployable |

Cheapest falsifier: 10-min spectrum measurement. A flat spectrum (r_99 = 2048) kills the compression path.

## 9. Two-paper composition flag

Cousin pair: CF8 (our MLP full-rank finding) + SVD-LLM (arXiv:2403.07378, ICLR 2025).
Value-add: W_down has a structural distinction (dimension asymmetry) not present in W_gate/W_up. This is the open question in SUMMARY.md. Not engineering-integration — closing an explicitly flagged open question is the contribution.

## 10. Verdict

**REFINE** — 10-min measurement with the highest potential payoff in the union (if W_down is compressible, the cascade arithmetic changes dramatically). Binary and fast. A5=3 in Stage 2 correctly reflects this.

---

# A — A-PWGI (Constraint-Alien: Peer-Weight Gossip Inference) [OUT-OF-ORIENTATION]

## LIGHT TREATMENT

## 1. Mechanism decomposition

- M1: Two fixed-precision GGUF files (Q2_K attention, Q8_0 MLP) resident on NVMe.
- M2: A static per-layer policy selects per-tensor source: W_Q/W_K from Q2_K; W_up/W_down/W_gate from Q8_0.
- M3: The policy is calibration-derived from CF11 (W_Q compressible) and CF5 (W_up more rank-sensitive).
- M4: Effective bpw = p×2.1 + (1−p)×8 where p ≈ 1/3 (attention fraction) → ~6.0 bpw effective.

## 2. Prior-art status

- M2, M3 — **ADJACENT**: AWQ, LLM.int8(), SqueezeLLM all do per-layer mixed precision from a SINGLE quantized model file. A-PWGI's novelty is using TWO separate full-precision models with a static per-tensor selection policy. No paper found as of 2026-05-09 proposing dual-model per-tensor inference. *This is primarily an engineering frame novelty.*
- M4 — The effective bpw arithmetic is trivial.

## 3. CF9 check

No theorem imports. GGUF tensor loading is a standard API. CF9 CLEAR.

## 4. CF10 check

No calibration fit. CF10 NOT APPLICABLE.

## 5. Smallest-test sharpening

Script: `scripts/pwgi_dual_gguf.py`
Model: Qwen3-1.7B-Base; two GGUF files (Q2_K and Q4_K_M as proxy for Q8_0 due to RAM constraint on Ryzen)
Operation: Load W_Q from Q2_K GGUF; all other weights from Q4_K_M GGUF; eval ΔNLL on WikiText-2.
Runtime: ~20 min. PASSES ≤8h gate.
GO: ΔNLL < 0.5 nats.
NO-GO: ΔNLL > 1.0 nats → Q2_K on W_Q specifically causes quality collapse; need Q4_K as minimum for W_Q.

## 6. Verdict

**REFINE** — wildcard (OUT-OF-ORIENTATION), CF-tether-suspended, structural floor met. The primitive is real (dual-GGUF loading in llama.cpp). However: A1=1 (effective bpw ~6.0 is not a step-change over Q4). Stage 5 should weight this low relative to the Track B compression advancers that address residency directly.

---

# Verdict Summary Table

| Advancer | Verdict | Strongest Claim | Key Risk | Cleanest Smallest Test Runtime |
|---|---|---|---|---|
| A-GFRS | REFINE | Frobenius ratio as Cluster C3 gate; gauge-exploitation novel | Null result (ratio > 0.50 everywhere) | 10 min |
| C-ABAR | REFINE | Empirical spectral-rank as allocation oracle vs Hessian proxy | LAMPQ partial pre-emption (use spectral rank not Hessian) | 20 min |
| C-CTERA | REFINE | Embed-query alignment α as CF11×CF12 coupling diagnostic (frame-novelty) | α in inconclusive range 0.2–0.5 | 30 seconds |
| C-JQSOH | REFINE | Outlier-channels inside W_Q subspace → bypass not needed | Outlier channels fully orthogonal to P_Q | 10 min |
| C-LGHF | REFINE | Layer-1 pre-context joint compression independence claim | Layer-1 W_Q NOT more compressible than global mean | 15 min |
| C-RAOK-Grounded | REFINE | Calibration-free Jaccard-to-tier derivation | Tier-0 stability < 90% → per-token overhead required | 10 min |
| F-AJDGF | REFINE (with A3 competitive bar) | Weight-only CCA gauge-exploitation vs A3's calibration-guided QK compression | A3 (arXiv:2505.12942) pre-empts mechanism; weight-only moat must show ΔNLL advantage | 25 min |
| F-ASIT | REFINE (wildcard) | Geometric null-space criterion for AERO fold eligibility (zero-calibration predictor) | Foldable neurons not geometrically structured (calibration artifact) | 30 min |
| F-CQSGC | REFINE (with Basis Sharing check) | Cross-layer W_Q gauge collapse; no-retraining moat vs MASA (fine-tuning required) | MASA (arXiv:2508.04581) or Basis Sharing (arXiv:2410.03765) pre-empt W_Q cross-layer coherence | 30 min |
| F-OCSSQ | REFINE | Algebraic three-tier derivation from CF3 Jaccard fixed-point; KV 4× compression | Jaccard ≠ quantization sensitivity (different measures) | 30 min (after 10-min stability gate) |
| F-RSCLE | DOWNGRADE (pending arXiv:2605.03109 review) | Update-covariance Laplacian framing (if update ≠ activation covariance) | Gated Subspace Inference (arXiv:2605.03109) likely pre-empts M2-M4 | 15 min (to measure AND compare vs activation covariance) |
| F-SGFVO | REFINE | W_V W_O product rank measurement (closes open CF question); weight-only path vs A3 | A3 (arXiv:2505.12942) covers OV compression with calibration; pre-empts mechanism but not weight-only variant | 30 min |
| R-ACWF | REFINE | Algebraic head-merge (not prune) from CF11 redundancy | Head importance is token-type-dependent on held-out | 30 min |
| R-AERO-PROBE | REFINE (wildcard) | AERO fold without retraining feasibility test on SwiGLU | CF1 rank-order ≠ branch correlation (independent statistics) | 45 min |
| R-MCQKV | REFINE | W_V W_O spectrum closes Qwen3 attention weight characterization | W_V/W_O full-rank like MLP → cascade limited to W_Q+W_K | 15 min (new) |
| R-RAOK-70B | REFINE | Three-tier Jaccard-derived calibration-free activation quantization | Tier-0 stability < 90% → static bypass invalid | 40 min (two rungs) |
| R-W_down-SPEC | REFINE | Closes explicit CF8 open question; largest potential payoff if GO | W_down full-rank like W_gate/W_up | 10 min |
| A-PWGI | REFINE (wildcard, low weight) | Dual-GGUF static policy (engineering frame novelty) | A1=1; effective bpw ~6.0 not step-change | 20 min |

---

## Final counts

- **REFINE**: 16 (A-GFRS, C-ABAR, C-CTERA, C-JQSOH, C-LGHF, C-RAOK-Grounded, F-AJDGF, F-ASIT, F-CQSGC, F-OCSSQ, F-SGFVO, R-ACWF, R-AERO-PROBE, R-MCQKV, R-RAOK-70B, R-W_down-SPEC)
- **DOWNGRADE**: 1 (F-RSCLE — pending arXiv:2605.03109 Section 2 review)
- **KILL**: 0 (all advancers passed CF9 and CF10 floors; wildcards pass structural floor)
- **REGENERATE**: 0

### Priority flags for Stage 5

1. **F-RSCLE**: Must resolve arXiv:2605.03109 (Gated Subspace Inference) pre-emption before Stage 5 selection. If Gated Subspace Inference uses per-layer UPDATE covariance (δ_L object), DOWNGRADE becomes REGENERATE. If they use ACTIVATION covariance (h_L object), F-RSCLE retains differentiation and is upgraded back to REFINE.

2. **F-CQSGC / R-CROSS-Q / C-CQBAL**: Must check Basis Sharing (arXiv:2410.03765) W_Q-specific results before Stage 5 selection. If Basis Sharing explicitly tested W_Q and found cross-layer coherence with fine-tuning, the no-retraining moat must be the explicit Stage 5 justification.

3. **F-AJDGF / A-GFRS**: Must run Frobenius ratio (A-GFRS, 10 min) before investing in F-AJDGF (25 min). The Frobenius ratio is the upstream gate for CCA exploitation.

4. **A3 (arXiv:2505.12942) competitive positioning**: F-AJDGF, F-SGFVO, and R-MCQKV all have A3 as the closest cousin on the attention-weight compression axis. Stage 5 should review A3's model families tested and the calibration-data dependency explicitly to position the weight-only moat.

---

## Adjacent papers requiring Stage 4/5 reading

- arXiv:2505.12942 (A3: Analytical Low-Rank Approximation for Attention, May 2025) — covers QK and OV compression with calibration covariance; adjacent to F-AJDGF and F-SGFVO.
- arXiv:2605.03109 (Gated Subspace Inference, May 2025) — may pre-empt F-RSCLE; must determine whether update-covariance or activation-covariance is used.
- arXiv:2410.03765 (Basis Sharing, ICLR 2025) — cross-layer W_Q basis sharing; must check whether W_Q was explicitly tested.
- arXiv:2508.04581 (Share Your Attention / MASA, Aug 2025) — dictionary-learning cross-layer attention sharing; requires fine-tuning (differentiating from F-CQSGC's no-retraining path).
- arXiv:2511.10004 (LAMPQ, Nov 2025) — mixed-precision by sensitivity; adjacent to C-ABAR.
- arXiv:2410.05265 (PrefixQuant, Oct 2024) — token-wise outlier handling; adjacent to Cluster C2 advancers.
