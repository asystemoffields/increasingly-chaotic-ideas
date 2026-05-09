# Stage 5 — Selector — Run 003
Run: 003 | Date: 2026-05-09 | Selector: Sonnet claude-sonnet-4-6

---

## Selection Algorithm Trace

### Step 1 — Re-verification of novelty (as of 2026-05-09)

Top-5 Track B candidates by raw Stage 2/3 scoring:
F4-WAOS (14), F1-VOFR (14), A2-RTGF (13), C5-WUPDOWN (12), C1-QAOC (12).

Top-5 Track A candidates:
F3-SRSC (12), F2-RGAS (12), F5-HPGO (12), AERO-QW3 (12), S6-WDKV (Stage 4 gap, Track A, FREE SWING).

Re-verified against publication landscape as of 2026-05-09:

**F1-VOFR**: Stage 3 confirmed NOVEL as of 2026-05-09 for M_L = Σ_h W_O^h W_V^h spectrum measurement. LatentLLM (arXiv:2505.18413) and SVD-LLM (arXiv:2403.07378) confirmed ADJACENT not PRE-EMPTING. No paper found forming the head-summed fused composition on a trained GQA model and measuring its spectrum. Re-verified: **NOVEL as of 2026-05-09.**

**F4-WAOS**: Stage 3 confirmed NOVEL for output-subspace SVD of Delta_L = W_down @ A_L under SwiGLU-gated activations. GPTVQ and ASVD/ARHQ are INPUT-side weighting; WAOS is OUTPUT-span measurement. Re-verified: **NOVEL as of 2026-05-09.** ARHQ (arXiv:2605.00140) was added at Stage 3 as ADJACENT — confirms it does NOT form Delta_L = W @ A. No new preempting paper.

**A2-RTGF**: OS+ (arXiv:2304.09145) confirmed ADJACENT for combined scale+sign; the residual-stream consistency enforcement (same Z₂ gauge across all matrices sharing channel d) was not in OS+ or FPTQuant (arXiv:2506.04985). Re-verified: **NOVEL framing as of 2026-05-09.**

**F3-SRSC**: Stage 3 assigned A5=3. Mechanism: softmax shift-invariance applied to W_Q row mean. SmoothQuant and OS+ do not target W_Q row-mean as a query dynamic-range reduction mechanism. No paper measures W_Q per-row mean magnitude as a static Q-offset for inference. Re-verified: **NOVEL as of 2026-05-09.**

**F5-HPGO**: Stage 3 assigned A5=2 (DeltaLLM adjacent but missing permutation alignment). A2=3. "Quantifying LLM Attention-Head Stability" (arXiv:2602.16740) is circuit-analysis, not compression. No paper in head-permutation gauge fixing for delta compression found. Re-verified: **NOVEL as of 2026-05-09.**

**S6-WDKV** (Stage 4 gap): KV-side compute reduction via cross-layer W_V × W_down output-subspace precomputation. This idea is gated on F4-WAOS GO and has no independent smallest-test — it inherits its feasibility from F4-WAOS's outcome. No paper found on cross-layer W_V @ U_L precomputation for KV compute reduction. **NOVEL framing, but structurally dependent on F4-WAOS.**

**AERO-QW3**: Stage 3 confirmed NOVEL for zero-SGD SwiGLU activation removal on trained Qwen3. arXiv:2410.13060 (AERO) covers trained-with-removal only. Re-verified: **NOVEL as of 2026-05-09.**

### Step 2 — Convergence multipliers

Track B:
- F4-WAOS: Convergence cluster 2 (2-orient: F4-WAOS/F + C5-WUPDOWN/C) → **1.2×**
- F1-VOFR: Convergence cluster 1 (2-orient: F1-VOFR/F + WAVQ/R) → **1.2×**
- A2-RTGF: No convergence cluster → **1.0×**
- C5-WUPDOWN: Convergence cluster 2 representative (same cluster as F4-WAOS; F4-WAOS is the stronger rep) → **1.2×**
- S3-QCMI (Stage 4 gap): No convergence cluster → **1.0×**

Track A:
- F3-SRSC: No convergence cluster → **1.0×**
- F2-RGAS: No convergence cluster → **1.0×**
- F5-HPGO: No convergence cluster → **1.0×**
- AERO-QW3: No convergence cluster → **1.0×**
- S6-WDKV: No convergence cluster (FREE SWING, wildcard non-penalization applies) → **1.0×**

### Step 3 — Frame-novelty bonuses

Track B:
- F4-WAOS: A2=3, output-subspace under SwiGLU activations is unrepresented in the saturated-frame list (no output-subspace W_down frame in prior rounds). Frame-diversity targets list W_down compression as under-represented after CF8 class kill → **+2**
- F1-VOFR: A2=3, fused operator M_L frame is unrepresented. Attention fused-composition frame not in any prior-round saturated list → **+2**
- A2-RTGF: A2=3, gauge-exploitation frame. Not on saturated list → **+2**
- S3-QCMI: A2 equivalent ~3 (per-channel V_Q participation as mixed-precision criterion is structurally novel, no prior-round precedent). Frame: spectral-participation-guided mixed-precision → **+2**

Track A:
- F3-SRSC: A2=3 per Stage 2. Softmax shift-invariance applied to W_Q row-mean: no prior-round precedent → **+2**
- F2-RGAS: A2=3 per Stage 2 (RMSNorm gain absorption spectrum shift) → **+2**
- F5-HPGO: A2=3 per Stage 2 (S_{n_heads} permutation group orbit, no prior-round precedent) → **+2**
- AERO-QW3: A2=3 per Stage 2 (zero-SGD SwiGLU activation removal) → **+2**
- S6-WDKV: FREE SWING. Frame: KV-side COMPUTE (not storage) reduction via cross-layer precomputation — explicitly flagged as under-represented gap in Stage 4 → **+2**

### Step 3b — Elegant-equivalence multipliers

Track B:
- F4-WAOS: elegance-class `conserved-quantity`. The concentration mechanism (CF1 soft-sparsity concentrates Delta_L) is constructive — the output span Delta_L = W_down @ A_L is a derived measurement, not a calibration fit. Constructive elegant-equivalence → **1.2×**. Pre-multiplier raw = 14 + 2 (confidence) = 16 ≥ 9: apply.
- F1-VOFR: elegance-class `algebraic-identity`. M_L = Σ_h W_O^h W_V^h is the exact residual-stream update operator — no approximation in forming it. Constructive → **1.2×**. Pre-multiplier = 14 + 2 = 16 ≥ 9: apply.
- A2-RTGF: elegance-class `gauge-exploitation`. Z₂ transform is exact (auditable in 2 lines: flip signs, update all reading/writing matrices). Constructive → **1.2×**. Pre-multiplier = 13 + 2 = 15 ≥ 9: apply.
- S3-QCMI: elegance-class `subspace-alignment`. The partition is derived from the same SVD that CF11 compression already requires — no additional calibration. Constructive → **1.2×**. Pre-multiplier estimated 12 ≥ 9: apply.

Track A:
- F3-SRSC: elegance-class `algebraic-identity`. softmax(z+c) = softmax(z) is a one-line algebraic identity; the W_Q row-mean subtraction is an exact consequence. Constructive → **1.2×**. Pre-multiplier = 12 + 2 = 14 ≥ 9: apply.
- F2-RGAS: elegance-class `algebraic-identity`. W_Q_gauged = W_Q @ diag(g) is auditable in one line. Constructive → **1.2×**. Pre-multiplier = 12 + 2 = 14 ≥ 9: apply.
- F5-HPGO: elegance-class `algebraic-identity`. Bipartite matching on head-permutation cost matrix is a standard auditable algorithm. Constructive → **1.2×**. Pre-multiplier = 12 + 2 = 14 ≥ 9: apply.
- AERO-QW3: no elegance-class tag in Stage 2. The monkeypatch-forward-pass is a measurement, not an algebraic identity. No elegant-equivalence multiplier → **1.0×**.
- S6-WDKV: elegance-class `algebraic-identity`. The cross-layer decomposition W_V^{L+1} h = W_V^{L+1} h_L + W_V^{L+1} U_L (U_L^T δh_L) is an exact algebraic identity given W_down output in U_L. Constructive → **1.2×**. But gated on F4-WAOS GO. Pre-multiplier estimated 11 ≥ 9: apply.

### Step 3c — Wildcard non-penalization

S6-WDKV carries [FREE SWING] tag. The absence of an independent CF tether (it requires F4-WAOS GO to have its first rung) does not penalize it under the wildcard rule. However, the structural floor requires a smallest-test ≤ 8h that is primitive on the actual stack. S6-WDKV's smallest test is 30 min but ONLY after F4-WAOS runs. As a selection for a run-003 experiment, this means the runner cannot execute S6-WDKV until F4-WAOS returns. This is a runtime-dependency that makes it structurally dependent, not just wildcard. It is not a first-class Track A standalone pick for this round. Retain as runner-up but do not select over a standalone experiment.

### Step 4 — NO-GO-finding bonuses

Track B:
- F4-WAOS NO-GO: closes "activation-soft-sparsity → output-subspace compression" as a class → constrained class of future ideas → candidate CF entry for SUMMARY.md → **+2**
- F1-VOFR NO-GO: extends CF8 to the fused attention operator M_L → closes the attention-weight compression class cleanly → candidate CF entry → **+2**
- A2-RTGF NO-GO: structural finding about W_up sign distribution (trained Qwen3 has or has not sign bias per channel) → constrains gauge-exploitation family → **+1**
- S3-QCMI NO-GO: V_Q column norms are uniform → per-channel mixed-precision from spectrum participation is not available → constrains mixed-precision allocation class → **+1**

Track A:
- F3-SRSC NO-GO: W_Q row-mean magnitude is negligible → softmax shift-invariance is not a load-bearing reduction mechanism → constrains row-mean-as-Q-offset class → **+1**
- F2-RGAS NO-GO: RMSNorm gain absorption does not improve W_Q spectrum concentration → closes gain-absorption spectral pre-conditioning class → **+1**
- F5-HPGO NO-GO: head-permutation alignment does not reduce layer-delta Frobenius norm → closes permutation-gauge-fixing for delta compression class → **+1**
- AERO-QW3 NO-GO: zero-SGD activation removal produces unacceptable ΔNLL on Qwen3 SwiGLU → closes zero-SGD AERO-style activation removal on trained SwiGLU without fine-tuning → **+2** (candidate CF entry for "AERO requires fine-tuning on SwiGLU; zero-SGD path is dead")
- S6-WDKV NO-GO: cross-layer W_V × W_down output-subspace precomputation fails → closes KV-side compute reduction via output-subspace cross-layer product → **+1**

### Step 5 — Final scoring

Formula: `(Stage 2 total + Stage 3 confidence) × convergence multiplier × elegance multiplier + frame-novelty bonus + NO-GO-finding bonus`

**Track B candidates:**

| Candidate | Base (S2+conf) | Conv× | Eleg× | Frame+ | NOGO+ | Total |
|---|---|---|---|---|---|---|
| F1-VOFR | 14+2=16 | 1.2 | 1.2 | +2 | +2 | 16×1.44+4 = **27.04** |
| F4-WAOS | 14+2=16 | 1.2 | 1.2 | +2 | +2 | 16×1.44+4 = **27.04** |
| A2-RTGF | 13+2=15 | 1.0 | 1.2 | +2 | +1 | 15×1.2+3 = **21.0** |
| S3-QCMI | ~10+2=12 | 1.0 | 1.2 | +2 | +1 | 12×1.2+3 = **17.4** |

F1-VOFR and F4-WAOS tie at 27.04. **Tiebreak: lower runtime.** F1-VOFR = 25 min; F4-WAOS = 40 min. **F1-VOFR wins.**

**Track A candidates:**

| Candidate | Base (S2+conf) | Conv× | Eleg× | Frame+ | NOGO+ | Total |
|---|---|---|---|---|---|---|
| F3-SRSC | 12+2=14 | 1.0 | 1.2 | +2 | +1 | 14×1.2+3 = **19.8** |
| F2-RGAS | 12+2=14 | 1.0 | 1.2 | +2 | +1 | 14×1.2+3 = **19.8** |
| F5-HPGO | 12+2=14 | 1.0 | 1.2 | +2 | +1 | 14×1.2+3 = **19.8** |
| AERO-QW3 | 12+1=13 | 1.0 | 1.0 | +2 | +2 | 13×1.0+4 = **17.0** |
| S6-WDKV | ~11+2=13 | 1.0 | 1.2 | +2 | +1 | 13×1.2+3 = **18.6** |

Three-way tie at 19.8 among F3-SRSC, F2-RGAS, F5-HPGO. **Tiebreak: lower runtime.** F3-SRSC = 5 min; F2-RGAS = 20 min; F5-HPGO = 15 min. **F3-SRSC wins.**

### Final picks

- **Track A: F3-SRSC** — Softmax Row-Shift Calibration (5 min, algebraic-identity, W_Q row-mean as static Q-offset)
- **Track B: F1-VOFR** — W_V / W_O Fused Composition Rank (25 min, algebraic-identity, M_L = Σ_h W_O^h W_V^h)

Runner-up Track A: F5-HPGO (Stage 6 fallback if F3-SRSC is killed).
Runner-up Track B: F4-WAOS (Stage 6 fallback; already feeds S6-WDKV if F3-SRSC wins Track A and F4-WAOS runs Track B in a separate round).

---

## PICK 1 — TRACK A

---

### 1. Title and one-line description

**Round 3 / Track A — F3-SRSC: Softmax Row-Shift Calibration**

Exploit the softmax shift-invariance identity (softmax(z + c·1) = softmax(z)) to subtract the per-row mean of W_Q from the weight matrix at model-load time, reducing Q activation dynamic range without loss.

---

### 2. Class tags

`arch-transpose`, `algebraic-identity`, `gauge-exploitation`

---

### 3. Hypothesis (1–2 paragraphs)

The softmax that produces attention scores operates on Q K^T / √d products. Since softmax(z + c·1) = softmax(z) for any scalar c, any constant row-mean offset in W_Q's output space (i.e., in q = W_Q h_normed) is invisible to the attention pattern. Concretely, let μ_row ∈ R^{d_model} be the per-row mean of W_Q (shape n_heads×d_head × d_model). Since q = W_Q h_normed, the component q_constant = μ_row × 1^T h_normed = μ_row · (1^T h_normed) is a scalar-times-row-mean contribution that shifts all d_head query dimensions by the same amount per head — and this uniform shift is entirely absorbed by softmax shift-invariance. Absorbing it into the weight at model-prep time eliminates this component from the per-token Q computation, reducing Q activation dynamic range by the mean-contribution fraction.

This depends on CF11 (W_Q global K=128 at +0.98 nats): if W_Q has a structurally significant row-mean, absorbing it is a meaningful coordinate change that reduces the per-token dynamic range of Q projections, improving INT4/INT8 quantization quality at no quality cost. The load-bearing claim — CF11 reduction — is unaffected because the shift-invariance is exact. The two-part test is: (A) is the per-row mean magnitude large enough to matter (static measurement, 5 min), and (B) does subtracting it improve Q-side quantization quality (ΔNLL at INT4 with vs without row-mean absorption)?

---

### 4. Smallest test

**Model:** Qwen3-1.7B-Base bf16  
**Precision:** bf16 → INT4 for quantized eval pass  
**Tokenizer:** Qwen3 tokenizer (Hugging Face)  
**Calibration corpus:** N/A — this is a weight-static measurement requiring no forward-pass calibration. The row-mean μ_row is computed from W_Q weight values directly.  
**Eval corpus:** WikiText-2 test split, 512 tokens (consistent with prior experiments).  
**Layers/matrices touched:** All 28 W_Q matrices. No other matrices modified.  
**Sweep parameters:** No sweep. Single measurement: compute μ_row per W_Q layer, measure ||μ_row||_F / ||W_Q||_F as the magnitude ratio. If > 0.05 (5% of total weight norm), proceed to INT4 eval.

**Procedure:**
1. Load Qwen3-1.7B-Base bf16. For each of 28 layers, extract W_Q (2048×2048 shape = 16 heads × 128 × 2048). Compute per-row mean: μ_row[L][head][dim_q] = mean over d_model of W_Q[L][head][dim_q, :]. Measure relative magnitude: ||μ_row||_F / ||W_Q||_F per layer. This takes < 1 min.
2. Check GO on magnitude: if ≥ 15/28 layers have magnitude ratio > 0.05, continue. This is the cheapest falsification (if magnitudes are sub-threshold, kill immediately).
3. Construct W_Q_shifted[L] = W_Q[L] - outer(μ_row[L], ones(d_model)) — subtract the rank-1 row-mean component. Verify ΔNLL = 0 on 50-token eval (lossless check): the absorbed mean must be added back at inference as a static bias per head, or equivalently, the key matrix and position encoding must compensate. **Implementation note**: the row-mean is constant per head per token, so the inference modification is: q_shifted = W_Q_shifted h; q = q_shifted + μ_row · sum(h) — except sum(h) is not constant per token. Revisit: softmax shift-invariance states softmax(z + c·1) = softmax(z) only if c is the SAME scalar for all d_head dimensions of a given head. The row-mean μ_row[head] ∈ R^{d_head} only acts as a shift if it is proportional to the all-ones vector within each head's d_head dimensions. If the per-row mean within a head varies across the d_head=128 dimensions, the shift is NOT uniform and softmax shift-invariance does NOT apply exactly.

**Skeptic amendment to procedure** (Stage 6 pre-emption): Verify uniformity of μ_row within each head's d_head block. Measure std(μ_row[head_d_head_block]) across d_head=128 dimensions per head per layer. GO on the algebraic-identity claim requires: mean(std(μ_row_block) / |mean(μ_row_block)|) < 0.20 (the mean is 5× larger than its intra-block variation — close to uniform shift). If this uniformity gate fails, the mechanism is not an exact algebraic identity and must be re-framed as an approximation.

4. If lossless check passes: quantize W_Q_shifted to INT4 (symmetric per-block, standard Q4_K_M). Compare ΔNLL at INT4 vs original W_Q INT4 on 512-token eval. Measure improvement.

**Wall-clock on Ryzen 5 7530U:**
- Step 1 (magnitude measurement): ~1 min
- Step 2 (GO check): < 1 min
- Step 3 (W_Q_shifted construction + lossless check): ~3 min
- Step 4 (INT4 eval, 2× ΔNLL passes): ~15–20 min (5–10 min per forward pass at Q4 on 1.7B)
- Total: **~25 min end-to-end.** Within 8h gate.

**Output path:** `experiments/stage0/ladder_v2/round3_srsc/`

**Script:** `scripts/srsc_row_shift.py` — Inputs: Qwen3-1.7B-Base model path, WikiText-2 eval file. Outputs: per-layer magnitude ratio table, uniformity check table, lossless ΔNLL check, ΔNLL_INT4_original vs ΔNLL_INT4_shifted. Load-bearing computation: per-row mean extraction from W_Q weight tensors; row-mean subtraction; uniformity measurement (std/|mean| within each head's d_head block); forward-pass NLL evaluation at bf16 and INT4.

---

### 5. Go threshold

**Stage 1 GO (mechanism is load-bearing):** ≥ 15/28 layers have ||μ_row||_F / ||W_Q||_F > 0.05 AND std(μ_row_block) / |mean(μ_row_block)| < 0.20 in ≥ 15/28 layers (uniform shift confirmed). Advance to Stage 2.

**Stage 2 GO (quantization improvement):** ΔNLL_INT4_shifted < ΔNLL_INT4_baseline by ≥ 0.05 nats on 512-token eval. Advance to cascade: compose with CF11 K=128 global W_Q compression (do row-shift first, then SVD compress) and report combined ΔNLL.

---

### 6. No-go threshold

**Magnitude NO-GO:** < 15/28 layers have magnitude ratio > 0.05. Structural finding: **W_Q row-mean offsets are negligible in trained Qwen3-1.7B — SGD approximately zero-centers W_Q rows.** This is a new structural constraint on W_Q's weight distribution: trained Qwen3 W_Q rows are approximately mean-zero, leaving no room for softmax-shift exploitation. Class-level constraint: "W_Q row-mean absorption as a Q dynamic-range reduction mechanism" is dead on any model where SGD produces approximately mean-zero W_Q rows.

**Uniformity NO-GO:** Magnitude is large but uniformity fails (row mean within each head's d_head block varies > 20%). Structural finding: **W_Q row-mean offsets are not uniform within attention heads — the shift-invariance identity does not apply exactly, and the approximation error must be characterized before use.** This downgrades F3-SRSC from an algebraic identity to a calibration-approximation, which falls into CF9 territory (imported theorem whose precondition — uniform-shift — does not hold). Re-classify as DOWNGRADE if this path is hit; Stage 6 can attempt the approximate version with explicit error quantification.

**Quantization NO-GO:** Uniformity passes but ΔNLL improvement < 0.05 nats. Structural finding: **W_Q row-mean absorption exists but does not materially improve INT4 quantization quality — the row-mean component is small relative to the per-row variation, or INT4 grid placement is not sensitive to mean shifts at this magnitude.** Constrains "static pre-quantization coordinate changes from algebraic symmetries" as a class: exact symmetry exploitation may not translate to quantization improvement at INT4 precision.

---

### 7. Ambiguous-zone follow-up

**GRAY zone:** Stage 1 GO (magnitude large + uniform) but Stage 2 quantization improvement is 0.02–0.05 nats. This is below the GO threshold but above noise. Follow-up: compose F3-SRSC with A2-RTGF (sign-flip gauge). If the two coordinate changes are independent (which they are — row-shift and sign-flip act on different aspects of W_Q), their combined effect on INT4 dynamic range should be additive. Apply both transforms; measure ΔNLL_INT4_combined vs ΔNLL_INT4_baseline. If combined improvement ≥ 0.08 nats, advance the composition not the individual. **Follow-up runtime: ~20 min** (A2-RTGF sign audit: 3 min + combined INT4 eval: 15 min). This disambiguation determines whether coordinate-change gauge-exploitation is compositionally additive on Qwen3 W_Q.

---

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick outright if:
1. **Uniformity precondition math error**: if Stage 6 identifies that softmax shift-invariance requires uniform shift not just within each head's d_head block but also across all KV positions (i.e., the absorbed component is sequence-position-dependent via positional encoding interactions), the mechanism is broken at a deeper level than the Stage 5 uniformity check catches. Stage 6 must verify the RoPE interaction: RoPE applies a per-position rotation to Q; the row-mean μ_row is a static weight component, not a per-position component. After RoPE rotation, the "constant" row-mean contribution becomes a per-position constant vector — which is NOT shift-invariant across heads' per-position dot products. This is a subtle interaction. **Stage 6 kill condition:** if the RoPE interaction analysis shows that the absorbed row-mean re-acquires position-dependence post-RoPE, the algebraic-identity claim is broken and the mechanism reverts to an approximation requiring calibration-fit error analysis (CF10 zone).
2. **Near-duplicate of A1-GFSM**: A1-GFSM was killed in Stage 2 for negligible residency gain (sub-0.5%). Stage 2 noted: "Stage 4 may want to check whether F3-SRSC's row-mean subtraction and A1-GFSM's gauge-fixing are the same operation under different names." Stage 6 must verify mechanistic overlap: A1-GFSM targets softmax bias absorption into a bias term (adding a static log-scale to attention logits); F3-SRSC targets W_Q row-mean subtraction to reduce Q dynamic range before quantization. These are structurally different operations (A1-GFSM changes the attention score; F3-SRSC changes the Q projection weights). If Stage 6 finds they are mechanistically identical, escalate to F5-HPGO as runner-up.
3. **Calibration ill-conditioning**: not applicable here — no calibration fit. This kill criterion is vacuous for F3-SRSC but is included per protocol.

---

### 9. Skeptic-controls declaration

F3-SRSC claims: "W_Q row-mean magnitude is load-bearing (not negligible)" — this is a structural property of the trained weight matrices, not a "works / transfers" claim over distributions. Strictly, the claim is a static measurement of trained weights. The skeptic-controls framework applies as follows:

**Permutation control:** For the magnitude measurement (Step 1), randomly permute the d_model dimension of W_Q (permute columns), recompute the row mean. The per-row mean of a column-permuted W_Q is identical (mean is order-invariant) — permutation does not apply to mean. **N/A — permutation control is degenerate for row-mean measurements.** Documented justification: row-mean is invariant to column permutation; the control would trivially match the real signal, not distinguish structure from noise. What does apply: compare the measured row-mean magnitudes to the row-standard-deviation (i.e., is the mean a significant fraction of the within-row spread, not just large in absolute terms). Reported as: mean(||μ_row||) / mean(std_row) per layer.

**Random-init control:** Compute the same row-mean magnitude on a randomly initialized Qwen3-1.7B model (random Gaussian weights, same architecture). A random Gaussian W_Q has zero-mean rows in expectation; the row-mean is O(1/√d_model) ≈ O(1/45). For trained W_Q, if the row mean is substantially larger than O(1/45) relative to the weight norm, this is a trained-model property not an architectural artifact. **Required:** run the magnitude measurement on a random-init W_Q (single matrix, layer 14 for comparability with CF11). Report gap: trained_mean / randinit_mean. GO requires trained gap > 3× random-init magnitude.

**Multi-seed:** The measurement is deterministic given the trained weights — there is no randomness in the calibration, no dropout, no ablation order. **N/A — multi-seed control is degenerate for deterministic weight measurements.** Documented justification: row-mean of a fixed weight matrix is a deterministic quantity; seeding produces identical results across all seeds. What does apply instead: report results across 5 representative layers (layers 0, 7, 14, 21, 27) to check depth-variation. The "seed" analog for weight measurements is layer index.

---

### 10. Runtime estimate

- Model load: ~5 min (Qwen3-1.7B-Base bf16 from DRAM-resident cache; NVMe load on first access ~10 min)
- Step 1 (magnitude + uniformity measurement): ~2 min
- Step 3 (W_Q_shifted construction + 50-token lossless check): ~3 min
- Step 4 (INT4 quantization of W_Q only): ~5 min
- Step 4 eval (2× forward passes, 512 tokens at Q4 on 1.7B): ~15 min
- Post-processing (table generation, random-init control): ~5 min
- **Total: ~30–35 min end-to-end.** Within 8h gate. Well within.

---

### 11. Script identification

**New script required:** `scripts/srsc_row_shift.py`

Description: Load Qwen3-1.7B-Base bf16. For each of 28 layers, extract W_Q tensor. Compute per-row mean (mean over d_model axis, yielding n_heads×d_head mean vector). Compute (a) magnitude ratio ||μ_row||_F / ||W_Q||_F, (b) uniformity: for each head's d_head block, compute std(μ_row_block) / |mean(μ_row_block)|, (c) random-init control: generate single random-Gaussian W_Q layer of same shape, compute same statistics. Apply row-mean subtraction to all 28 W_Q matrices. Verify ΔNLL = 0 on 50-token eval (with static bias compensation at inference: add μ_row dot sum(h) per token per head — except this requires runtime access to h; alternatively, verify that the inference identity holds by forward-pass comparison). Quantize original and shifted W_Q to INT4 (per-block symmetric). Evaluate NLL on WikiText-2 512-token eval for both. Output: CSV table with per-layer statistics, terminal summary of GO/NO-GO verdict.

---

### 12. Downstream implications

**GO:** Compose with CF11 K=128 global W_Q compression. The shift-invariant row-mean component is absorbed as a static bias; the remaining W_Q has smaller per-row dynamic range, potentially improving the CF11 K=128 approximation quality (the singular values of the centered W_Q may decay faster). Also enables composition with A2-RTGF (sign-flip gauge): the two coordinate changes are independent (mean-shift and sign-flip act on different W_Q properties). Combined application to all quantized attention matrices could stack to 0.1–0.2 nats total improvement at INT4.

**NO-GO:** Constrains the class "static pre-quantization coordinate changes from algebraic symmetries." Specifically closes: (a) "W_Q row-mean as a Q dynamic-range reduction mechanism," (b) "softmax shift-invariance exploitation on trained Qwen3" as a meaningful quantization gain. Does not kill A2-RTGF (different mechanism). Does not kill CF11 K=128 compression (different mechanism). Kill list entry: "F3-SRSC Softmax Row-Shift on Qwen3-1.7B: row-mean either negligible or not uniform-shift-invariant at [outcome]."

---

### 13. Provenance

- **Originating orientation:** F (First-Principles), run_003 Stage 1
- **Convergence cluster:** None (solo advancer)
- **Stage 4 gap-idea slot:** Not a Stage 4 idea; Stage 3 REFINE advancer
- **Frame-novelty path:** A2=3 in Stage 2 (no paper measures W_Q per-row mean magnitude as a Q dynamic-range reduction mechanism; novel framing as of Stage 2 search)
- **Runner-up for Stage 6:** F5-HPGO (S_{n_heads} permutation gauge; 15 min; same score 19.8; second tiebreak by runtime)

---

## PICK 2 — TRACK B

---

### 1. Title and one-line description

**Round 3 / Track B — F1-VOFR: W_V / W_O Fused Composition Rank**

Measure the spectrum of M_L = Σ_h W_O^h W_V^h (the exact residual-stream attention-update operator) to determine if the fused composition is low-rank, enabling a two-factor compression that bypasses per-matrix SVD limitations in GQA models.

---

### 2. Class tags

`compression-lr`, `algebraic-identity`, `arch-transpose`

---

### 3. Hypothesis (1–2 paragraphs)

In a GQA attention block, the actual linear map from value projections to residual-stream update is M_L = Σ_h W_O^h W_V^{kv(h)}, where kv(h) maps each Q-head to its shared KV-head. This 2048×2048 matrix is the exact residual-stream update operator; compressing it at rank K is algebraically equivalent to compressing the attention block's effective output, not any individual weight matrix. The critical structural prediction: even if individual W_V and W_O matrices are full-rank (consistent with CF8's finding on MLP weights), the head-summed M_L may be low-rank because the 16 Q-heads may produce cancelling contributions in their V × O products, concentrating the net update in a lower-dimensional subspace than any individual head.

This prediction is grounded in CF11 (W_Q K=128 global spectrum, showing that 16 heads collectively span ~128 effective dimensions in the query space). By analogy, the 16-head value-output compositions may exhibit similar head-redundancy at the operator level. The fused M_L measurement is the correct object to test this: SVD-LLM tests per-matrix SVD (input: W_V and W_O independently); VOFR tests the actual computation (output: M_L as the measurement object). If r_99(M_L)/d ≤ 0.75, post-training two-factor compression of the full attention value-output path is feasible with a structural finding directly analogous to CF11 for the output side. This depends on CF11 by analogy and on the GQA head-grouping structure (which is explicitly handled in the M_L construction).

---

### 4. Smallest test

**Model:** Qwen3-1.7B-Base bf16  
**Precision:** bf16 throughout (spectrum measurement; no quantization in this experiment)  
**Tokenizer:** Qwen3 tokenizer  
**Calibration corpus:** Not required for spectrum measurement. W_Q-side comparison uses only model weights.  
**Eval corpus:** WikiText-2 test split, 512 tokens (for ΔNLL if spectrum GO passes).  
**Layers/matrices touched:** All 28 attention layers. W_V (8 KV-heads × 128×2048) and W_O (16 Q-heads × 2048×128) per layer.  
**Sweep parameters:** K ∈ {256, 512, 1024} for ΔNLL measurement. Fixed spectrum measurement at full SVD.

**Procedure:**
1. Load Qwen3-1.7B-Base bf16. For each of 28 layers:
   - Extract W_V tensors: 8 KV-heads, shape (128, 2048) each.
   - Extract W_O tensors: 16 Q-heads, shape (2048, 128) each.
   - For each Q-head h (0..15): kv_head = h // 2 (GQA-2 grouping for Qwen3-1.7B with 16Q/8KV).
   - Compute M_h = W_O[h] @ W_V[kv_head], shape (2048, 2048).
   - Sum M_L = Σ_h M_h, shape (2048, 2048).
   - Run `torch.linalg.svd(M_L, full_matrices=False)`. Record singular values S_L.
   - Compute r_99(M_L) = min rank capturing 99% of sum(S_L^2). Record r_99/d and cumvar@{256,512,1024}.
2. **Cheapest falsification gate:** Check layer 14 alone first. If r_99(M_{L=14})/d > 0.90, M_L is approximately full-rank; kill without full sweep (~2 min, no forward pass).
3. If layer 14 passes, sweep all 28 layers (~12 min SVD sweep).
4. If ≥ 20/28 layers pass r_99/d ≤ 0.75: apply K=512 truncation. Replace forward-pass attention output computation with: output = U_L @ torch.diag(S_L[:K]) @ V_L[:K,:]^T @ value_projected, where value_projected = [concatenated V-head projections per token]. Measure ΔNLL on 512-token WikiText-2 eval (~10 min).
5. Measure both: (a) M_h per single head (individual head composition rank), (b) M_L summed. If M_L is higher-rank than individual M_h, this indicates head-summation spreads rank — important structural nuance.
6. **Two-subset consistency check** (analog of F4-WAOS calibration consistency, adapted for weight-only measurement): verify M_L spectrum is deterministic (weight-only, no randomness). N/A — deterministic. Instead: verify by re-computing M_L for layer 14 with explicit GQA grouping verification (confirm kv(h) = h//2 matches the model config).

**Wall-clock on Ryzen 5 7530U:**
- Model load: ~5 min
- Layer 14 cheapest falsification: ~2 min
- Full 28-layer SVD sweep: ~14 min (28 × ~30s per 2048×2048 SVD)
- ΔNLL eval at K=512 (single forward pass, 512 tokens): ~10 min
- Post-processing: ~3 min
- **Total: ~25 min end-to-end** (assuming layer 14 passes). Within 8h gate.

**Output path:** `experiments/stage0/ladder_v2/round3_vofr/`

**Script:** Existing partial spec from Stage 3: `scripts/vofr_spectrum.py`. Contents: load model, compute M_L = Σ_h W_O^h @ W_V^{kv(h)} per layer, run SVD, record spectrum statistics (r_99/d, cumvar@K), apply K=512 truncation to forward pass, evaluate NLL. Include layer-14 early-exit gate.

---

### 5. Go threshold

**Spectrum GO:** r_99(M_L)/d ≤ 0.75 in ≥ 20/28 layers. This establishes M_L is concentrated enough for meaningful two-factor compression at K=512.

**Quality GO:** ΔNLL at K=512 ≤ +0.35 nats (by analogy to W_K K=512 = +0.29 nats from CF11; VOFR's threshold is slightly looser to account for M_L being a composed object rather than a single-matrix decomposition).

**Combined GO:** Both spectrum GO AND quality GO. Advance to: (a) residency arithmetic for 70B (see Stage 3 §5: 37.5 GB W_V+W_O → 3.1 GB at K=512), (b) KV-cache compression opportunity (V-cache can optionally store the K=512-dimensional intermediate rather than full d_model vectors, reducing V-cache by 4×).

**SOFT-GO:** r_99/d ≤ 0.75 spectrum GO, but ΔNLL at K=512 > 0.35 nats. Measure ΔNLL at K=1024 (less compression, higher quality). SOFT-GO threshold: K=1024 ΔNLL ≤ +0.20 nats. Advance to 70B with K=1024 compression factor only.

---

### 6. No-go threshold

**NO-GO:** r_99(M_L)/d > 0.85 in majority of layers (> 14/28). Structural finding: **the fused attention output operator M_L = Σ_h W_O^h W_V^h is full-rank in trained Qwen3. Head-summation does NOT produce cancellation that concentrates the output — the 16 heads contribute additively to the full d_model output subspace. This extends CF8 to the attention output operator and closes the "low-rank fused attention output" class.** Combined with CF11 (W_Q concentrated) and this result, the attention block picture becomes: input subspace (W_Q) is concentrated at K=128; output operator (M_L) is full-rank. Post-training compression must target the query-projection side (CF11) but cannot target the value-output fused side.

NO-GO constrains: WAVQ (W_V and W_O per-matrix SVD, same cluster), any attention output compression, the S6-WDKV gated idea (which requires F4-WAOS GO, not this result, but would also benefit from knowing M_L structure).

---

### 7. Ambiguous-zone follow-up

**GRAY:** r_99/d between 0.75 and 0.85 for M_L. Per-head M_h test: measure r_99(M_h) for individual heads. If individual M_h is more concentrated than summed M_L, the head-summation is SPREADING rank (destructive from a compression standpoint). Follow-up mechanism: compress per-head M_h independently rather than the summed M_L. Per-head K_per_head at analogous thresholds. **Follow-up runtime: ~20 min** (already computed in step 5; just requires evaluating per-head K threshold). This disambiguation determines whether per-head or per-layer fused compression is the correct granularity. Note: per-head at K=64 was NO-GO in CF11 for W_Q; the per-head M_h spectrum may differ from W_Q per-head spectrum because M_h = W_O^h W_V^{kv(h)} compounds two matrix products.

---

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should reject this pick outright if:
1. **GQA grouping error:** The kv(h) = h // 2 grouping assumption is incorrect for Qwen3-1.7B's specific GQA implementation. Stage 6 must verify from the model config that the head-sharing grouping matches the M_L construction. If kv(h) is not a simple integer division (e.g., if Qwen3 uses a different GQA grouping pattern), M_L is formed incorrectly and the spectrum measurement is invalid.
2. **Prior-art emergence:** If a paper appears between Stage 5 selection and Stage 6 red-team that explicitly measures Σ_h W_O^h W_V^h spectrum on a GQA transformer, the frame is pre-empted. Date stamp: **no such paper found as of 2026-05-09.**
3. **Near-duplicate of run_002 F1/JVOC:** Run_002 Stage 5 also selected a Track B experiment involving W_V/W_O fused composition (F1/JVOC). Stage 6 must verify that run_002 F1/JVOC and run_003 F1-VOFR are not the same experiment. If run_002 F1/JVOC has already been run and produced a result, Stage 6 must check whether F1-VOFR is a near-duplicate (≥80% mechanistic overlap) per PIPELINE.md stop-condition #3. If run_002's JVOC result exists, this run_003 selection should either adopt that result or run a distinct follow-up (e.g., varying K sweep not yet done).

---

### 9. Skeptic-controls declaration

F1-VOFR claims: "M_L = Σ_h W_O^h W_V^h has concentrated spectrum — the fused operator is low-rank in trained Qwen3."

This is a structural-property claim about trained weights ("X is consistent across layers" for spectrum concentration). Controls:

**Permutation control:** Randomly permute the column indices of each W_V^{kv(h)} before forming M_h. This breaks the structural relationship between W_O^h and W_V^{kv(h)} — the permuted M_h = W_O^h @ permuted(W_V^{kv(h)}) should have a spectrum consistent with a random matrix product. The GO threshold requires: real r_99(M_L)/d substantially below the permuted baseline. Expected permuted baseline: for random matrix products of shape (2048, 128) and (128, 2048), the product has rank 128 and r_99/d ≈ 128/2048 = 0.0625. This is artificially low because it is bounded by the intermediate dimension. Better permuted control: permute columns of W_V^{kv(h)} to destroy the row-correspondence with W_O^h. In this control, M_h = W_O^h @ permuted(W_V), which has the same rank as the unmodified product (rank 128, both are rank-128 matrices) but destroys any calibration-derived co-structure between the two factors. The meaningful comparison: does the real M_L have LOWER r_99/d than the permuted M_L? If they are comparable, the concentration is architectural (GQA rank bound of 16×128=2048), not a trained structure. **Required gap:** real r_99(M_L)/d ≤ permuted r_99(M_L)/d - 0.10 (at least 10 percentage points below the permuted baseline). Note: if GQA arithmetic dictates r_99(M_L) ≤ 128 (the rank of each M_h product), then r_99/d ≤ 128/2048 = 0.0625 for ALL GQA models regardless of training — this would make the spectrum concentration trivially architectural, not a trained property. **Stage 6 must verify:** is r_99(M_L) bounded by the intermediate dimension 128 (the d_head), making concentration guaranteed by architecture rather than training? If so, the "spectrum concentration" is an architectural artifact of GQA's d_head<d_model structure, and the finding is about GQA geometry, not trained weights. This is not a kill — it is still useful for compression — but the framing changes from "trained model property" to "GQA architecture property."

**Random-init control:** Compute M_L on a randomly initialized Qwen3-1.7B model (random weights, same GQA architecture). Measure r_99(M_L_randinit)/d. If the real model's r_99/d is comparable to the random-init baseline, the concentration is architectural. Required: real r_99(M_L)/d < randinit r_99(M_L)/d - 0.05 (at least 5 ppt below random-init) OR the randinit value confirms the architectural bound (in which case, document the architectural vs trained split). Random-init control runtime: ~5 min (generate random weights for one layer, compute M_L, SVD — no model load needed).

**Multi-seed:** The spectrum measurement is deterministic (weight-only, no randomness). Multi-seed control is not applicable for the same reasons as F3-SRSC. Documented justification: the SVD of a fixed weight matrix is deterministic; there is no randomness in the measurement. Instead: report spectrum across 5 representative layers (0, 7, 14, 21, 27) to characterize depth variation. The depth profile of r_99(M_L)/d is the analog of multi-seed variance for weight-only measurements.

**Architectural bound check (mandatory):** Before claiming the spectrum result as a trained-model finding, explicitly verify whether the GQA rank ceiling (16 heads × d_head=128 = 2048, which equals d_model=2048 for Qwen3-1.7B) implies that r_99(M_L) ≤ 2048 trivially (no concentration possible above rank 2048 for this model). Compare to a larger model where d_model > n_heads × d_head to determine whether the finding is architecture-universal.

---

### 10. Runtime estimate

- Model load: ~5 min (NVMe first access) / ~2 min (DRAM-resident)
- Layer 14 cheapest falsification check: ~2 min
- Full 28-layer SVD sweep: ~14 min
- ΔNLL eval (K=512 forward pass, 512 tokens): ~10 min
- Random-init control (layer 14 only): ~3 min
- Post-processing: ~3 min
- **Total: ~32 min end-to-end.** Within 8h gate. Well within.

---

### 11. Script identification

**New script required:** `scripts/vofr_spectrum.py` (per Stage 3 spec, not yet created)

Description: Load Qwen3-1.7B-Base bf16. For each of 28 layers: extract W_V (8 KV-heads × 128×2048) and W_O (16 Q-heads × 2048×128). Apply GQA grouping kv(h) = h // 2. Compute M_h = W_O[h] @ W_V[kv(h)] for h in 0..15. Sum M_L = Σ_h M_h. Run torch.linalg.svd(M_L, full_matrices=False). Record S_L (2048 singular values). Compute cumvar@K for K in {64, 128, 256, 512, 1024, 2048}. Also compute per-head M_h spectra for depth-5-layer sample. Apply random-init control at layer 14. Apply permuted-W_V control at layer 14. Implement K=512 truncated forward pass for ΔNLL eval. Output: per-layer spectrum table (CSV), spectrum plot (optional), ΔNLL result table, control comparison table.

---

### 12. Downstream implications

**GO:** Opens: (a) Two-factor compression of attention V+O path at Qwen3-72B: 13.6 GB → 3.1 GB at K=512. (b) V-cache in compressed K=512 space: V-cache size reduced 4× at no additional approximation. (c) S6-WDKV becomes more motivated (if M_L is concentrated AND F4-WAOS gives GO, the cross-layer product precomputation for KV-side compute is grounded in two confirmed findings). (d) Extends CF11 from "query-side concentration" to "query-and-output-side concentration" — a clean structural map of attention weight compressibility.

**NO-GO:** Extends CF8 to the fused attention output operator. Combined with CF11 (W_Q concentrated) this produces the definitive attention-weight structure map: compression is available on the query-input side (CF11) but not the value-output fused side (VOFR NO-GO). Kill list entry: "F1-VOFR: fused M_L = Σ_h W_O^h W_V^h full-rank in trained Qwen3-1.7B. Class kill on low-rank fused attention output compression post-training." This closes the Convergence Cluster 1 (WAVQ also killed by proxy — if M_L is full-rank, per-matrix W_V and W_O SVDs are even less likely to show concentration useful for compression).

---

### 13. Provenance

- **Originating orientation:** F (First-Principles), run_003 Stage 1
- **Convergence cluster:** Cluster 1 — W_V / W_O spectrum measurement (2-orientation: F1-VOFR/F + WAVQ/R). F1-VOFR is the cluster representative (score 14 vs WAVQ score 11; fused composition is the theoretically stronger object).
- **Stage 4 gap-idea slot:** Not a Stage 4 idea; Stage 3 REFINE advancer
- **Frame-novelty path:** A2=3 in Stage 2 (no paper measures spectrum of Σ_h W_O^h W_V^h on any GQA architecture, novel as of 2026-05-09)
- **Elegant-equivalence path:** `algebraic-identity` — M_L is the exact residual-stream update operator; forming it is not an approximation. Multiplier 1.2× applies constructively.
- **Runner-up for Stage 6:** F4-WAOS (same score 27.04, 40 min; already motivated as the strongest Track B runner-up and opens S6-WDKV when it runs)
