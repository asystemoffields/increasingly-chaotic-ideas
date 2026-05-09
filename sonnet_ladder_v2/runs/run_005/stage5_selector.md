# Stage 5 Selector — Run 005 (v2 Sonnet Ladder)

Completed: 2026-05-09. Selecting ONE experiment per track from the Stage 3 REFINE pool, Stage 4 gap ideas, and Stage 2 frame-novelty advancers. Selection algorithm applied in full: novelty re-verification, convergence weighting, frame-novelty weighting, elegant-equivalence multiplier, wildcard non-penalization, runtime/structural-finding weighting.

---

## Pre-Selection: Re-Verified Novelty (2026-05-09)

**Track A top candidates re-checked:**

- **A1-GFRS-2 (Z_2^d sign-flip gauge for residual-stream quantization codebook):** Stage 3 confirmed NOVEL as of 2026-05-09. Re-check: searched "residual stream sign flip Z2 gauge symmetry LLM quantization codebook design 2026." No new paper found doing this as of 2026-05-09. Status: NOVEL (confirmed at Stage 5).

- **F2-HPGF (head-permutation gauge fixing via S_16 symmetry):** Stage 3 confirmed NOVEL as of 2026-05-09. Re-check: searched "head permutation gauge fixing multi-head attention compression 2026 exact symmetry." No new paper exploiting S_16 permutation invariance as a compression coordinate choice found as of 2026-05-09. Status: NOVEL (confirmed at Stage 5).

- **C3-ZOIA (W_Q subspace × outlier-channel alignment):** Stage 3 confirmed NOVEL. Re-check: searched "outlier channel absorbed W_Q compressed subspace projection residual attention quantization 2026." No new paper measuring projection residual of outlier channels inside W_Q's rank-128 subspace found as of 2026-05-09. Status: NOVEL (confirmed at Stage 5).

**Track B top candidates re-checked:**

- **R5-RAOK-FULL (3-tier Jaccard-stratified activation codebook):** Stage 3 confirmed NOVEL. Re-check: searched "activation quantization three tier Jaccard dynamicity K=0.1 static K=0.9 dynamic stratification 2026." No new paper using per-token consecutive-pair Jaccard at K=0.1%/0.9% as the stratification criterion found as of 2026-05-09. Status: NOVEL (confirmed at Stage 5).

- **S6-SEQFILT (sequential-scan inline T2 filter for RAOK, Stage 4):** Not yet published by definition (Stage 4 origin from this run). Status: NOVEL (no prior analog found as of 2026-05-09 — inline 20-dot-product row classification during sequential scan as scatter-gather replacement is not in the activation quantization literature).

- **S3-SPECACT (activation covariance spectral entropy as per-layer bpw proxy, Stage 4):** Transposition of F5-SECP from weight space to activation space. Re-check: searched "activation covariance spectral entropy per-layer bpw schedule LLM quantization 2026." arXiv:2604.18085 covers weight-matrix spectral statistics, not activation covariance spectral entropy. Status: NOVEL (confirmed at Stage 5).

**DOWNGRADE pre-emption note (confirmed):** R5-WVWK-MLA and F1-JQKLR remain DOWNGRADED per Stage 3 (PALU, TransMLA, ReCalKV pre-empt the within-layer Q/K joint subspace compression application). No reconsideration — the new search confirms no novelty gaps opened since Stage 3.

---

## Scoring Summary

### Track A

| Candidate | Base (S2+S3 conf) | Conv mult | Elegance mult | Frame-novelty bonus | NO-GO bonus | Final score |
|---|---|---|---|---|---|---|
| A1-GFRS-2 | 12+2=14 | 1.0 | 1.2 (constructive gauge) | +2 (A2=3, fresh frame) | +1 | **19.8** |
| F2-HPGF | 12+2=14 | 1.0 | 1.2 (constructive gauge) | +2 (A2=3, fresh frame) | +1 | **19.8** |
| C3-ZOIA | 11+2=13 | 1.0 | 1.2 (constructive subspace-alignment) | +1 | +1 | **17.6** |
| S1-QFUSE | 11+1=12 | 1.0 | 1.1 | +1 | +1 | **16.3** |

**A1-GFRS-2 and F2-HPGF tie at 19.8.** Tie-breaking: (1) runtime — both 5-10 min, negligible difference; (2) NO-GO ambiguity — A1-GFRS-2's NO-GO (H(sign)≈1.0 → gauge-fixing produces no structured sign plane) is a cleaner binary structural finding with broader downstream kill scope: it closes S2-BITSER and the entire "sign-plane as a separate compressible object" family. F2-HPGF's NO-GO (equidistributed head energies) closes the head-elimination gauge approach but does not close the track-level head-compression question (CF11 head-sharing survives). **A1-GFRS-2 selected for Track A.** F2-HPGF is the named runner-up (Stage 6 kill-and-escalate target).

### Track B

| Candidate | Base (S2+S3 conf) | Conv mult | Elegance mult | Frame-novelty bonus | NO-GO bonus | Final score |
|---|---|---|---|---|---|---|
| S6-SEQFILT | 12+2=14 | 1.0 | 1.2 (algebraic identity) | +1 | +1 | **18.8** |
| R5-RAOK-FULL | 12+2=14 | 1.0 | 1.1 (conserved-quantity, auditable) | +1 | +2 | **18.4** |
| S3-SPECACT | 12+1=13 | 1.0 | 1.1 (conserved-quantity) | +2 | +2 | **18.3** |

**S6-SEQFILT selected for Track B.** R5-RAOK-FULL is the named runner-up. Note: S6-SEQFILT depends on the same CF3 grounding as RAOK and gates on A6-CNTR's precision@20% — the pre-condition check (A6-CNTR) is the first 20-min experiment; if NO-GO, the Track B selection falls back to R5-RAOK-FULL.

---

---

## PICK 1 — Track A

---

### 1. Title and One-Line Description

**Run 005 / Track A — A1-GFRS-2: Z_2^d Residual-Stream Gauge Fixing for Sign-Plane Entropy Measurement**

Exploit the exact Z_2^d sign-flip gauge symmetry of the transformer residual stream to fix coordinates so that each residual-stream channel's max-abs element is positive, then measure whether the resulting sign-plane Shannon entropy H(sign) falls below 0.8 bits/weight — the threshold at which the sign plane is structured enough to compress independently from the magnitude plane.

---

### 2. Class Tags

`compression-quant`, `gauge-exploitation`, `frame-novelty`, `arch-transpose`

---

### 3. Hypothesis

The transformer residual stream admits an exact Z_2^d gauge symmetry: for any assignment of per-channel sign flips s ∈ {−1, +1}^d, flipping the sign of every residual-stream channel c simultaneously in every weight matrix that reads or writes that channel leaves the network's function unchanged. This is not an approximation — it is an algebraic identity whose pre-condition is simply that all weight interactions with channel c come in pairs (a write and a read) with the sign flip threading through. For the transformer residual stream this holds exactly: each channel participates in RMSNorm (scale-invariant, sign-agnostic), W_Q/W_K/W_V (linear, sign threads through), W_up/W_gate (linear at input; the SwiGLU nonlinearity is applied after the projection, and both W_gate and W_up see the same sign-flipped input, so the element-wise product preserves the gauge invariance).

After choosing the canonical gauge (max-abs element of each channel positive), the sign bits of the gauge-fixed weight matrix W_up_gauged form a pattern that may be structured — i.e., Shannon entropy H(sign) = −Σ p log p where p = fraction of weights in each row that are positive, averaged over rows, is below 1.0 bit/weight. The specific hypothesis is H(sign) < 0.8 bits/weight in at least one layer of Qwen3-1.7B-Base W_up after gauge-fixing. If true: the sign plane is a separate compressible object from the magnitude plane. At 4 bpw, the sign bit contributes 1 of 4 bits; compressing it to 0.8 bits/weight effective via a sign-plane codebook yields 3.8 bpw average with no loss in function. At scale, this saves ~5% storage at 4 bpw. The downstream cascade (S2-BITSER) converts the sign plane's near-constancy into a GEMV decomposition saving ~1/4 of sign-GEMV compute. The hypothesis depends on CF1 (W_up is the load-bearing matrix for SwiGLU firing — compressing it is on the critical residency path) but does not require any rank structure in W_up (CF8 remains valid — we are compressing the sign plane independently of the singular value spectrum).

---

### 4. Smallest Test

**Model:** Qwen3-1.7B-Base, bf16 precision. Tokenizer: Qwen3 tokenizer (from HuggingFace).

**Calibration corpus:** 50 tokens from WikiText-2 validation set (tokens 0:50), loaded via `datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')`. No held-out eval needed for the measurement phase (the sign entropy is a property of the weight matrix, not of activations on eval data).

**Measurement procedure:**
1. Load W_up for all 28 layers (shape 6144 × 2048 per layer).
2. For each layer L, for each row r of W_up[L]: find the index i = argmax |W_up[L, r, :]|; if W_up[L, r, i] < 0, flip sign of row r (this is the gauge-fix).
3. For the gauge-fixed W_up_gauged[L]: compute p_r = (number of positive weights in row r) / 2048 for each row.
4. Compute per-row entropy H_r = −p_r log2(p_r) − (1−p_r) log2(1−p_r).
5. Compute layer entropy H_L = mean(H_r over all rows of layer L).
6. Report H_L for all 28 layers.
7. Also compute the full-model entropy H_global = mean(H_L over 28 layers).

**Layers / matrices touched:** W_up for all 28 layers of Qwen3-1.7B-Base. No W_gate, W_down, or attention matrices modified or measured in this smallest test.

**Wall-clock estimate:** < 10 minutes on Ryzen 5 7530U (load bf16 model ~5 min, gauge-fix + entropy calculation ~1 min, total < 10 min).

**Output path:** `experiments/stage0/ladder_v2/round5_gfrs2/gfrs2_sign_entropy_results.json`

**Script:** `scripts/gfrs2_sign_entropy.py`. This script loads Qwen3-1.7B-Base in bf16, iterates over all 28 W_up matrices, applies the per-row canonical gauge fix (flip sign if max-abs element is negative), computes per-row binary entropy H_r, aggregates to per-layer H_L and global H_global, and saves results to JSON with a per-layer table and summary statistics.

---

### 5. Go Threshold

**GO:** H_global < 0.8 bits/weight AND at least 18 of 28 layers have H_L < 0.8 bits/weight.

Interpretation: the gauge-fixed W_up sign plane is structured across most of the model, justifying a sign-plane codebook. The 0.8 threshold is motivated by the compression break-even: at H=0.8 bits, a Huffman code over 2-symbol (sign) alphabet with p=0.8 positive achieves 0.72 bits/symbol — meaningful savings vs storing the raw sign bit at 1 bit.

**SOFT-GO:** H_global < 0.9 AND at least 14 of 28 layers have H_L < 0.8 bits/weight — sign-plane compression is viable for a subset of layers; motivates per-layer entropy-aware bpw scheduling.

---

### 6. No-Go Threshold

**NO-GO:** H_global ≥ 0.95 bits/weight OR fewer than 10 layers have H_L < 0.8 bits/weight.

**Class-level kill from NO-GO:** "Sign-flip gauge freedom in the residual stream, when used to define a canonical coordinate, produces structured sign patterns that are independently compressible" — KILLED. This closes S2-BITSER (bit-serial GEMV decomposition gated on H(sign) < 0.8). It also closes the general "sign plane as a separate object from magnitude plane in quantization" family. The residual-stream Z_2^d symmetry is real regardless — it remains useful as a coordinate-fixing pre-processing step for other quantization schemes (SmoothQuant-style scale absorption is more natural with a sign-fixed representation) — but the sign-compression application specifically is dead.

NO-GO does NOT kill: (a) the gauge-fixing as a pre-processing step for A3/AWQKV W_Q compression (sign-fixing before SVD may still improve spectral concentration — that is F6-GFNM's claim); (b) the Z_2^d symmetry argument itself.

---

### 7. Ambiguous-Zone Follow-Up

**GRAY zone:** H_global ∈ [0.8, 0.95) with a bimodal per-layer distribution (some layers well below 0.8, others near 1.0).

**Follow-up:** Run the layer-stratified analysis — compute H_L for all 28 layers and identify the low-entropy cluster (H_L < 0.75) vs high-entropy cluster (H_L > 0.90). If the low-entropy cluster contains ≥ 8 contiguous early layers (consistent with CF6 Layer-1 anomaly — deep-layer neurons more active), proceed with S2-BITSER scoped to those layers only. Runtime of follow-up: 10 additional minutes (already computed from the base experiment, just thresholding).

**Resolution:** GRAY resolves to PARTIAL-GO (layer-selective sign compression) if the low-entropy cluster is ≥ 8 layers with H_L < 0.75.

---

### 8. Kill Criteria (Stage 6 Amendment Slot)

Stage 6 should reject this pick if:

1. **Frame-mismatch:** The Z_2^d gauge argument is invalidated — i.e., there exists a weight interaction with the residual-stream channel that is NOT linear (and thus does not thread the sign flip through correctly). The specific risk: LayerNorm with learned bias (additive shift) breaks sign-flip invariance because h → −h + bias ≠ −(h + bias). Qwen3-1.7B uses RMSNorm (no bias); this risk is mitigated. Stage 6 should verify: (a) Qwen3-1.7B uses no additive LayerNorm bias; (b) no residual-stream position-embedding (rotary embeddings are applied inside attention, not on the residual stream directly). If either fails, the gauge argument is broken.

2. **Prior art found after submission:** A 2026 paper applying Z_2^d sign-flip gauge to LLM weight quantization codebook design would be a pre-emption. Stage 6 should run one final search before execution.

3. **Skeptic-controls deficiency:** See Section 9 — if the sign entropy measurement is shown to be dominated by random-initialization structure (not trained structure), the gauge exploitation is an architectural artifact.

4. **No baseline comparison:** Stage 6 should add to the smallest test: measure H(sign) of W_up BEFORE gauge-fixing (the un-normalized case) to confirm that gauge-fixing actually reduces entropy (otherwise the "canonical gauge" is not doing useful work).

---

### 9. Skeptic-Controls Declaration

**Claim being tested:** "After Z_2^d gauge-fixing, the sign plane of W_up in trained Qwen3-1.7B has structured (low-entropy) patterns that are not present in random-initialization or permuted weights."

**Permutation control:** For each layer L, randomly permute the column indices of W_up_gauged[L] within each row (scrambles the position of weights while preserving per-row sign distribution). Compute H_L on the permuted matrix. The GO threshold requires: H_L(real, gauge-fixed) < 0.8 AND H_L(permuted) is not substantially different from H_L(real). Wait — the sign entropy H_L is derived from per-row positive fractions p_r; permuting columns within a row does NOT change p_r. Therefore column permutation is not the right permutation control for sign entropy. Correct permutation control: randomly REASSIGN rows of W_up_gauged (permute row order). This tests whether the low-entropy pattern is row-specific or row-order-irrelevant. Since H_L = mean(H_r), row permutation does not change H_L either. The correct permutation control for sign entropy is: for each row r, independently sample a new sign assignment (B(2048, p_r) random draw). If H_L(real) ≈ H_L(random-sign-draw with same p_r), the gauge-fixing is not producing structured sign patterns — it is just choosing the most-positive-skewed representation, and p_r is the informative statistic, not the spatial sign pattern. Document this: the permutation control reduces to "does p_r vary meaningfully across rows?" If std(p_r) < 0.05, the sign plane has no within-layer structure beyond global p̄; if std(p_r) > 0.15, rows have heterogeneous sign distributions (structural). Report std(p_r) alongside H_L.

**Random-init control:** Measure H(sign) on a randomly initialized Qwen3-1.7B-Base (torch.manual_seed(42), model.init_weights() or equivalently xavier_uniform_ on all weight matrices). If H_L(random-init) < 0.8 bit, the sign-plane entropy is a property of the architecture (uniform initialization tends to have p_r ≈ 0.5 → H_r ≈ 1.0 bit). Expected: random-init should have H_L ≈ 1.0 (Gaussian weights → p_r ≈ 0.5 per row). If trained W_up has H_L < 0.8 and random-init has H_L ≈ 1.0, the gap confirms the trained structure is real. GO threshold must be met by trained model AND random-init baseline must show H_L ≥ 0.90 (gap ≥ 0.10 bits). Wall-clock: 3 additional minutes (load random-init model, run same gauge-fix + entropy computation).

**Multi-seed:** The gauge-fix and entropy measurement are deterministic given fixed weights. Randomness enters via: (a) which WikiText-2 calibration tokens are used (not relevant — sign entropy is a weight property, not an activation property). Multi-seed is NOT APPLICABLE to this experiment because the measurement is a weight-property computation, not a stochastic process. Document: multi-seed control not applicable because sign entropy is a deterministic function of the loaded bf16 weight matrix; no random seed affects the result.

---

### 10. Runtime Estimate

| Phase | Time |
|---|---|
| Model load (Qwen3-1.7B-Base bf16) | 5 min |
| Gauge-fix all 28 W_up matrices | <1 min |
| Per-row entropy computation (28 × 6144 rows × 2048 cols) | <2 min |
| Random-init control (load + compute) | 3 min |
| JSON output + summary | <1 min |
| **Total** | **~11 min** |

Well within the 8-hour limit. Estimated total including any debugging: 30 min.

---

### 11. Script Identification

**New script needed:** `scripts/gfrs2_sign_entropy.py`

**Description:** Loads Qwen3-1.7B-Base in bf16 via `transformers.AutoModelForCausalLM.from_pretrained`. For each of the 28 transformer layers, extracts `model.layers[L].mlp.up_proj.weight` (shape 6144 × 2048 in HuggingFace convention). Applies per-row canonical gauge fix: for each row r, finds the column index i = argmax of absolute values; if the sign of weight[r, i] is negative, multiplies the entire row by −1. Computes per-row binary entropy H_r = −p log2(p) − (1−p) log2(1−p) where p = mean(weight[r] > 0). Aggregates to per-layer H_L = mean(H_r over rows), per-layer std(p_r), and global H_global. Repeats the measurement on a random-initialized model (same architecture, `torch.manual_seed(42)` + `init_weights()`). Outputs a JSON with: `trained_H_per_layer`, `trained_H_global`, `trained_std_pr_per_layer`, `randinit_H_per_layer`, `randinit_H_global`. Also outputs a side-by-side console table for quick inspection.

---

### 12. Downstream Implications

**GO (H_global < 0.8):**
- Unlocks **S2-BITSER**: bit-serial GEMV decomposition with sign-plane precomputation, up to ~0.75× GEMV cost for W_up. This becomes Track B's Round 6 candidate if S6-SEQFILT NO-GO and R5-RAOK-FULL advances.
- Motivates **F6-GFNM** follow-up: gauge-fixing W_Q (multiply by RMSNorm γ) as an SVD pre-processing step may also reduce H(sign) for W_Q — a 10-min addon.
- Supports **A1-GFRS-2 compression path**: after sign-entropy confirmation, measure ΔNLL of 4→3.8 effective bpw compression (encode sign plane at H_global bits/symbol via arithmetic coding). This is a 30-min follow-up experiment.
- **What entering KILL_LIST means:** any "sign-flip coordinate choice improves quantization codebook design" derivative would cite this as the founding experiment.

**NO-GO (H_global ≥ 0.95):**
- Kills **S2-BITSER** (sign-plane precompute requires structured signs).
- Kills the general "sign plane as separate compressible object" family for W_up in Qwen3 post-training.
- Does NOT kill: Z_2^d gauge-fixing as a pre-processing step for SVD or SmoothQuant-style operations (those use the gauge freedom for a different purpose — magnitude alignment, not sign-plane compression).
- Informs **KILL_LIST** addition: "Z_2^d gauge-fixing of residual stream produces structured W_up sign patterns in trained Qwen3" — KILLED. CF-class-kill scope: within-layer trained-weight sign-structure for W_up Qwen3 family.

---

### 13. Provenance

- **Originating orientation:** A (Constraint-Alien). A1-GFRS-2 generated in run_005 Stage 1 Orientation A.
- **Convergence cluster:** None — solo advancer. Path 3 (frame-novelty) advancer, Stage 2 Slot 2.
- **Stage 4 gap-idea:** Stage 4 generated S2-BITSER as the downstream of A1-GFRS-2; S2-BITSER gates on this experiment.
- **Frame-novelty bonus path:** A2=3 (Stage 2 judge). Z_2^d gauge symmetry of the transformer residual stream applied to quantization codebook design — no published analog as of 2026-05-09.
- **Runner-up:** F2-HPGF (head-permutation gauge fixing via S_16 symmetry, 5-min experiment, same score 19.8). If Stage 6 kills A1-GFRS-2 (e.g., gauge argument fails due to undiscovered non-linear interaction), escalate to F2-HPGF directly.

---

---

## PICK 2 — Track B

---

### 1. Title and One-Line Description

**Run 005 / Track B — S6-SEQFILT: Sequential-Scan Inline T2 Channel Filter for RAOK (No Scatter-Gather)**

Replace RAOK's per-token scatter-gather to load dynamic T2 channels with an inline 20-dot-product row classification that runs during the sequential W_up scan, eliminating random-access overhead while maintaining the same T2/T3 stratification accuracy.

---

### 2. Class Tags

`compression-quant`, `runtime-fused`, `algebraic-identity`, `sparsity-dynamic`

---

### 3. Hypothesis

RAOK-FULL's 3-tier activation quantization (T1: 2 FP16 static channels; T2: 18 INT8 dynamic channels; T3: 2028 INT4 static) requires knowing, per token, which 18 channels are T2 before reading the corresponding rows of W_up. This creates a per-token scatter-gather: precompute the T2 channel index set from the input activation outlier signature, then load only those rows at INT8 precision. The scatter-gather breaks the sequential access pattern that the Zen 3 CPU stride prefetcher relies on (U5-HPST's finding), introducing up to 18 random reads per W_up scan.

The S6-SEQFILT hypothesis: the T2/T3 classification can be computed INLINE as W_up rows are read sequentially. The load-bearing claim is an algebraic identity — the scatter-gather's result (a set of row indices) is equivalent to a sequential threshold test: for each W_up row w_i read in order, compute the inner product ⟨w_i[outlier_indices], x_outlier⟩ where x_outlier is the 20-element outlier-channel activation vector extracted from the current input h_t. If |⟨w_i[outlier_indices], x_outlier⟩| exceeds a calibrated threshold τ, classify row i as T2 (INT8 decode); otherwise T3 (INT4 decode). This 20-element dot product costs 1-2 AVX2 instructions per row and introduces no additional memory access (w_i is already in cache from the sequential read). The identity: (RAOK scatter-gather T2 classification) ≡ (SEQFILT inline threshold test), provided the threshold τ is calibrated so that the two methods agree on ≥ 90% of T2/T3 assignments.

The hypothesis depends on A6-CNTR's precision@20% claim: that the top-20 outlier channels of the residual-stream activation predict W_up firing row identities at precision > 0.50. Without A6-CNTR GO, the T2 classification has no reliable activation-based routing key. The experiment therefore has TWO phases: (Phase 0, prerequisite) A6-CNTR precision@20% check (20 min); (Phase 1, main) SEQFILT agreement rate measurement (30 min).

CF grounding: CF3 (K=1% Jaccard=0.308 — dynamic outlier channels rotate per token, providing per-token T2 identity at K=0.9%). CF1 (W_up firing dominance — the T2 rows are the ones most affecting SwiGLU output quality, so getting them right matters). The mechanism requires no trained-model properties beyond what CF3 already measured.

---

### 4. Smallest Test

**Model:** Qwen3-1.7B-Base, bf16. Tokenizer: Qwen3 tokenizer.

**Calibration corpus:** 200 tokens from WikiText-2 validation set (same corpus used in R2 PDAP data, tokens 0:200 from 10 diverse passages). This enables reuse of precomputed activation data from R2.

**Eval corpus:** WikiText-2 validation tokens 201:400 (held-out, 200 tokens, not used in threshold calibration).

**Procedure:**

*Phase 0 — A6-CNTR prerequisite check (20 min):*
For each of 200 calibration tokens, extract the residual-stream activation h_t entering each MLP layer. Identify the top-20 outlier channels by absolute magnitude (these are the T2 candidate routing keys per CF3). For each W_up row w_i, measure whether w_i is in the top-20% of post-SwiGLU magnitude (this is the "firing" ground truth from CF1). Compute precision@20%: fraction of predicted T2 rows (predicted by top-20 outlier-channel inner product threshold) that are actual top-20% firing rows. Gate: precision@20% > 0.50 across layers 0–27 on calibration. If NO-GO: do not proceed to Phase 1; report A6-CNTR-style kill.

*Phase 1 — SEQFILT agreement rate (30 min):*
On held-out 200 tokens: for each token t and each W_up layer L:
  1. Compute RAOK scatter-gather T2 set: the 18 rows with highest |⟨w_i[top20_channels], x_outlier⟩| (precomputed per-token T2 ground truth).
  2. Compute SEQFILT inline T2 set: all rows w_i where |⟨w_i[top20_channels], x_outlier⟩| > τ_L (τ_L calibrated on calibration tokens as the 18th-percentile threshold of the dot-product distribution).
  3. Measure agreement rate: Jaccard(RAOK_T2_set, SEQFILT_T2_set).
  4. Across all 200 tokens × 28 layers: report mean agreement rate and 10th-percentile agreement rate.

**Sweep parameters:** τ_L calibrated per layer (median of the 18th-largest dot-product value across calibration tokens). No other free parameters.

**Skeptic controls inline:** Run Phase 1 with x_outlier PERMUTED (channels randomly permuted within each token) to produce a permuted-SEQFILT T2 set. Report agreement rate of permuted SEQFILT with RAOK T2 (should be near Jaccard of random 18-of-2048 vs random 18-of-2048 ≈ 18/(2048-18+18) ≈ 0.0088). The real agreement rate must exceed permuted baseline by ≥ 0.50.

**Wall-clock estimate:** Phase 0: 20 min (reuses PDAP activation extraction code). Phase 1: 30 min (Python simulation on existing activation data). Total: 50 min.

**Output path:** `experiments/stage0/ladder_v2/round5_seqfilt/`

**Script:** `scripts/seqfilt_agreement.py`. See Section 11.

---

### 5. Go Threshold

**GO:** Phase 0 precision@20% > 0.50 (A6-CNTR gate met) AND Phase 1 mean agreement rate ≥ 0.90 AND 10th-percentile agreement rate ≥ 0.80 AND permuted-baseline agreement rate < 0.10.

Interpretation: the inline 20-dot-product sequential classifier reproduces RAOK's scatter-gather T2 selection at ≥ 90% mean Jaccard agreement, confirming that the scatter-gather can be replaced by the sequential scan without loss of T2 stratification quality.

**SOFT-GO:** Mean agreement rate ∈ [0.80, 0.90) → T2 misclassification rate ≤ 20%; acceptable with a slightly larger T2 window (24 channels instead of 18) to absorb misclassified rows. Follow-up: 10-min threshold adjustment experiment.

---

### 6. No-Go Threshold

**NO-GO:** Phase 0 precision@20% ≤ 0.50 (A6-CNTR gate fails) OR Phase 1 mean agreement rate < 0.80.

**Class-level kill from NO-GO:**

If Phase 0 fails: kills S6-SEQFILT AND confirms A6-CNTR's null hypothesis — "CF3 outlier-channel signatures in the residual stream do not predict W_up firing row identity." This closes the "activation-outlier-channel routing key for MLP row dispatch" family (both CNTR and SEQFILT). R5-RAOK-FULL's T2 stratification survives (RAOK uses Jaccard-based dynamicity for STORAGE tier selection, not row-dispatch prediction), but the scatter-gather acceleration is lost.

If Phase 0 GO but Phase 1 NO-GO: kills S6-SEQFILT specifically — the outlier-channel key predicts firing correctly but the 20-element dot-product threshold is too coarse to reproduce the scatter-gather's row selection. Structural finding: T2 row identity requires the full 2048-dim activation signature, not the 20-dim outlier projection. This constrains A6-CNTR's downstream (CNTR dispatch requires full-signature routing, not cheap outlier proxy) and closes the "inline lightweight T2 classifier" family. RAOK scatter-gather survives as-is.

---

### 7. Ambiguous-Zone Follow-Up

**GRAY zone:** Mean agreement rate ∈ [0.80, 0.90) with high layer-to-layer variance (some layers ≥ 0.90, deep layers < 0.80, consistent with CF3's deep-layer outlier spread finding).

**Follow-up:** Compute per-layer agreement rate. If layers 0–19 achieve ≥ 0.90 agreement but layers 20–27 achieve < 0.80 (consistent with v2-CF1's deep-layer outlier spread widening to 14-18% of channels): apply SEQFILT to layers 0–19 only; retain RAOK scatter-gather for layers 20–27 (where the outlier routing key is less predictive). Measure whether the hybrid scheme (SEQFILT early, scatter-gather deep) achieves ≥ 5% mean bandwidth reduction vs full scatter-gather. Follow-up runtime: 10 min (subset layer analysis from Phase 1 data).

---

### 8. Kill Criteria (Stage 6 Amendment Slot)

Stage 6 should reject this pick if:

1. **A6-CNTR Phase 0 pre-flight not runnable:** If the PDAP R2 activation data is not available for reuse, or if the 20-channel outlier index set is ambiguous (per-layer vs global top-20), Stage 6 should require the pre-flight experiment before Phase 1.

2. **Hidden assumption on T2 threshold stability:** The threshold τ_L calibrated on calibration tokens may not generalize to held-out tokens if the activation distribution shifts. Stage 6 should verify: report τ_L calibration variance (std of the 18th-largest dot product over calibration tokens). If std/mean > 0.5, the threshold is unstable and the SOFT-GO path is not reliable.

3. **Prior art post-submission:** A paper proposing inline activation-based T2 row classification during sequential weight scan for quantized LLM inference would pre-empt. Stage 6 should run one final search before execution.

4. **Skeptic-controls deficiency:** Phase 1 must include the permuted-x_outlier control (see Section 9). If Stage 6 finds Section 9 missing its permutation control, reject.

---

### 9. Skeptic-Controls Declaration

**Claim being tested:** "The inline 20-dot-product sequential T2 classifier (SEQFILT) produces the same T2/T3 row assignments as RAOK's precomputed scatter-gather, at ≥ 90% mean Jaccard agreement on held-out tokens — and this agreement is driven by the trained activation-outlier structure (CF3), not by architectural priors."

**Permutation control:** Mandatory (see Section 4). For each evaluation token t, randomly permute the 2048-dim activation vector x_outlier's channel indices (destroying the CF3 outlier-channel identity) before computing the inline dot product. Measure agreement rate of the permuted SEQFILT with RAOK's ground-truth T2 set. Expected baseline: ≈ 18/(2048) ≈ 0.0088 (random 18-of-2048 vs ground-truth 18-of-2048). The GO threshold requires: real agreement rate ≥ 0.90 AND permuted agreement rate ≤ 0.10 (gap ≥ 0.80). This confirms that the agreement is driven by the outlier-channel activation structure, not by any systematic bias in the dot-product threshold.

**Random-init control:** Load a randomly initialized Qwen3-1.7B-Base (same architecture, `torch.manual_seed(42)`). Repeat Phase 1: compute SEQFILT T2 sets from the random-init model's activations and random-init W_up rows. Compare with the trained model's RAOK T2 sets (ground truth remains the trained-model RAOK T2 — we are testing whether the random-init model produces similar T2 assignments). Expected: random-init activations are not structured (no CF3 outlier pattern), so random-init SEQFILT T2 agreement with trained-RAOK T2 should be near random (≈ 0.0088). GO threshold requires: trained-model SEQFILT ≥ 0.90 AND random-init SEQFILT agreement with trained-RAOK ≤ 0.10. This confirms the CF3 outlier structure in trained model is load-bearing for the T2 classification. Wall-clock: 10 additional minutes.

**Multi-seed:** Randomness in this experiment comes from: (a) which calibration tokens are used for τ_L calibration; (b) the ordering of token pairs in the held-out evaluation. Run with 3 calibration-token seeds (seed 0: tokens 0-200; seed 1: tokens 201-400; seed 2: tokens 400-600 from WikiText-2 validation). Report mean ± std of Phase 1 agreement rate across seeds. GO threshold requires: mean agreement ≥ 0.90 AND worst-of-3 agreement ≥ 0.80 (not below NO-GO threshold). Wall-clock: 3× Phase 1 = +60 min. Combined total runtime: ~2 hours with all controls.

---

### 10. Runtime Estimate

| Phase | Time |
|---|---|
| Phase 0 — A6-CNTR precision@20% prerequisite | 20 min |
| Phase 1 — SEQFILT agreement rate (200 held-out tokens, 28 layers) | 30 min |
| Permutation control (per token: permute x_outlier, recompute) | 15 min |
| Random-init control (load random model + Phase 1 on random activations) | 10 min |
| Multi-seed (2 additional calibration splits × Phase 1) | 60 min |
| JSON output + summary | 5 min |
| **Total** | **~140 min (~2.3 hours)** |

Within the 8-hour limit. Estimated with 30-min buffer for debugging: 3 hours.

---

### 11. Script Identification

**New script needed:** `scripts/seqfilt_agreement.py`

**Description:** Loads Qwen3-1.7B-Base in bf16. Phase 0: for each calibration token, forward-pass through all MLP layers with activation hooks; extract pre-MLP residual h_t; identify top-20 outlier channels by |h_t|; for each W_up row w_i compute |⟨w_i[outlier_indices], h_t[outlier_indices]⟩|; rank rows by this score; compute precision@20% against actual post-SwiGLU W_up firing magnitude top-20%. Phase 1 (if Phase 0 GO): calibrate per-layer threshold τ_L as the 18th-largest dot product over calibration tokens; on held-out tokens, compute both RAOK scatter-gather T2 set (top-18 rows by dot product, precomputed) and SEQFILT T2 set (rows exceeding τ_L); compute Jaccard agreement between the two sets per token per layer; aggregate mean and 10th-percentile; repeat with permuted x_outlier and random-init model for controls; repeat for 3 calibration seeds for multi-seed control. Outputs JSON with all metrics and a per-layer agreement table.

---

### 12. Downstream Implications

**GO (agreement ≥ 0.90):**
- Directly enables **RAOK-FULL production path**: replace scatter-gather T2 dispatch with inline sequential classifier, composing with U5-HPST's stride-prefetcher optimization (sequential access pattern now preserved end-to-end). The combined system (SEQFILT + HPST stride alignment) is the core Track B deployment candidate for run 006.
- Unlocks **S2-BITSER composition**: if A1-GFRS-2 also GO, the sequential W_up scan already in SEQFILT naturally decomposes into sign-GEMV + magnitude-GEMV per row.
- Informs **A6-CNTR advancement**: GO on precision@20% (Phase 0) is a CNTR partial-GO, motivating a CNTR deployment experiment at round 6.
- Constrains **RAOK vs SEQFILT at 70B**: SEQFILT's 20-element dot product cost is fixed per row regardless of d_int; at 70B (d_int=28672 per layer) the sequential scan is even more valuable vs scatter-gather. The structural finding generalizes.

**NO-GO (Phase 0, precision ≤ 0.50):**
- Kills **S6-SEQFILT** and **A6-CNTR** simultaneously. Both rely on the CF3 outlier-channel signature predicting W_up firing rows.
- **KILL_LIST addition:** "CF3 outlier-channel identity predicts W_up firing row selection — KILLED." The structural finding: in trained Qwen3, the activation outlier channels entering an MLP are not predictive of which W_up rows dominate the output. This is consistent with CF1 (W_up dominates firing) but shows that the W_up row identity cannot be routed from the residual-stream outlier signature alone.
- **Track B escalation:** Fall back to R5-RAOK-FULL (Stage 5 runner-up) as the round-6 Track B experiment.

**NO-GO (Phase 1, agreement < 0.80):**
- Kills **S6-SEQFILT** specifically. RAOK scatter-gather survives as the implementation path.
- Structural finding: 20-dim dot-product threshold is too coarse for T2 classification accuracy — the full 2048-dim activation vector is needed. This is a hardware feasibility constraint (full-activation dot product per row during scan costs 2048 ops/row, not 20) that kills the "lightweight inline classifier" family.

---

### 13. Provenance

- **Originating orientation:** Stage 4 Skeptic (Section 5, "storage is sequential-only" constraint reframing of RAOK's scatter-gather). Not from any Stage 1 orientation — pure Stage 4 generation.
- **Convergence cluster:** None as a Stage 4 idea; however, it operationalizes the CF3+CF1 primitive that is the core of both RAOK-FULL (Stage 3 top-3) and A6-CNTR (Stage 3 REFINE). It is a composition of two CF-grounded primitives into a runtime-efficiency improvement.
- **Stage 4 gap-idea:** S6-SEQFILT, generated in Stage 4 Section 5 (sequential-only constraint reframing), explicitly named as Stage 4's top candidate.
- **Frame-novelty bonus:** +1 (the frame — inline sequential classification as scatter-gather equivalent — is not in the activation quantization literature; no published paper proposes this specific reformulation).
- **Runner-up:** R5-RAOK-FULL (Stage 5 second-scored Track B candidate at 18.4). If Stage 6 kills S6-SEQFILT or if Phase 0 A6-CNTR prerequisite fails, escalate directly to R5-RAOK-FULL with the standard RAOK-FULL plan from Stage 3 (the 30-min Jaccard sub-band re-run at K=18 channels). If both S6-SEQFILT and R5-RAOK-FULL are killed, escalate to S3-SPECACT (Track B third candidate).
