# Stage 6 — Red Team — Run 003
Date: 2026-05-09 | Red-team model: claude-sonnet-4-6

---

<!-- ============================================================ -->
<!-- TRACK A — F3-SRSC (Softmax Row-Shift Calibration, 5 min)    -->
<!-- ============================================================ -->

## ═══ TRACK A — F3-SRSC ═══

---

### 1. Frame-mismatch re-check (CF9)

**Imported identity:** softmax(z + c·1) = softmax(z) for scalar c.

**Precondition:** c must be THE SAME scalar added uniformly to ALL d_head dimensions of a given head's logit vector for a given key position. Equivalently, the absorbed component of q must be a scalar multiple of the all-ones vector in the d_head space.

**Stage 5 self-caught version of the precondition:** The plan correctly identifies in §4 step 3 that μ_row[head] ∈ R^{d_head} is only a valid shift if it is proportional to the all-ones vector within the d_head block. It then adds the uniformity gate: std(μ_row_block) / |mean(μ_row_block)| < 0.20. This is a correct operationalization.

**Stage 6 independent verification — the RoPE interaction:**

Here the Stage 5 plan partially self-catches a deeper failure in §8 kill criterion 1, but does not fully resolve it. The red team's independent analysis:

After RoPE, the query vector for token at position p becomes:
```
q_p = RoPE(p) · (W_Q h + bias_correction)
```
where RoPE(p) is a position-dependent rotation matrix (block-diagonal, 2×2 Givens rotations). The row-mean subtraction makes q = W_Q_shifted h + μ_row · sum(h) at the pre-RoPE stage. After RoPE rotation, the μ_row · sum(h) term becomes RoPE(p) · μ_row · sum(h) — a POSITION-DEPENDENT vector, NOT a constant shift across all key positions that softmax sees.

**Specifically:** softmax shift-invariance operates on the attention logit vector for a given query position p, which is q_p · K^T / √d_head. The "constant" that must be absorbed is q_p · e_const where e_const is the same for all key positions. After RoPE rotation of q_p, the μ_row · sum(h) component of q_p becomes RoPE(p) · μ_row · sum(h). This is a fixed vector at query position p (given h and p), and it DOES add a constant to ALL key-position dot products (since the keys also have fixed positions). Formally, the contribution to each attention logit is (RoPE(p) · μ_row · sum(h)) · k_j^T / √d_head — which is different for every key j because k_j differs.

**Conclusion on CF9:** The RoPE interaction does NOT salvage the algebraic identity. The absorbed row-mean contribution is NOT a scalar constant c uniformly added to all logits of the attention distribution. It adds (RoPE(p) · μ_row · sum(h)) · k_j^T which varies with j (key index). The softmax shift-invariance identity DOES NOT APPLY.

This is a CF9 frame-mismatch. The precondition — "uniform additive shift to all logits of the softmax distribution" — fails because the transformed row-mean interacts differently with each key vector.

**What survives the CF9 kill:** The static measurement claim (is μ_row large? is it uniform within d_head?) remains valid as a characterization of W_Q's weight distribution. What is DEAD is the claim that subtracting μ_row is lossless (algebraic-identity). It becomes an APPROXIMATION claim: the absorbed component contributes a "near-constant" pollution term that varies mildly across keys. Whether this approximation is useful depends on how much the residual variation matters — a calibration-fit question, not an algebraic-identity question.

**The algebraic-identity elegance tag is therefore DEFEATED.** The 1.2× elegant-equivalence multiplier applied at Stage 5 was applied to a claim that does not survive precondition verification.

---

### 2. Calibration ill-conditioning re-check (CF10)

No calibration fit in F3-SRSC. Weight-static measurement and direct forward-pass comparison. CF10 **does not apply**. PASS.

---

### 3. Skeptic-controls check

**Hypothesis shape:** "W_Q row-mean magnitude is large and uniform" — this is a structural property measurement, not a transfer claim in the strict sense. The follow-on "this improves INT4 quantization" IS a claim of the form "X improves Y."

**Permutation control:** Stage 5 documents N/A correctly — row-mean is invariant to column permutation. The substitute control (mean(||μ_row||) / mean(std_row) per layer) is appropriate. Accepted.

**Random-init control:** Stage 5 requires this and sets a GO threshold (trained gap > 3× random-init). This is appropriate. The expected random-init row-mean is O(1/√d_model) ≈ 0.022 for d_model=2048. The trained model must show row-means substantially above this to claim a trained structure. Accepted.

**Multi-seed:** N/A documented correctly for deterministic weight measurement. Accepted.

**Additional control required by red team:** Given the CF9 finding above (RoPE breaks the algebraic identity), if the experiment is reframed as an APPROXIMATION, the quantization improvement claim (Stage 2 GO: ΔNLL_INT4 improvement ≥ 0.05 nats) becomes a transfer claim over tokens. This requires a permutation control on the TOKEN dimension: does the ΔNLL improvement at INT4 persist on permuted-order tokens (permuting token order in the eval corpus removes positional structure)? Without this, any measured improvement could be attributed to a positional-structure correlation artifact. This control is cheap: run the INT4 eval with shuffled token order and report gap.

**Skeptic-controls overall:** PARTIAL PASS with required amendment (add permuted-token eval if experiment is reframed as approximation).

---

### 4. Hidden prior-art search — third pass

**Track A search result:** arXiv:2309.01729 "Softmax Bias Correction for Quantized Generative Models" (Pandey et al., ICCV 2023) is the most adjacent paper found. This paper proposes an offline bias correction for quantized softmax inputs. It is NOT the same mechanism: it corrects quantization-induced bias in the softmax input (post-quantization correction), whereas F3-SRSC proposes pre-quantization subtraction of a static W_Q row-mean to reduce dynamic range. These are structurally different: 2309.01729 targets activation-level quantization noise; F3-SRSC targets weight-level mean subtraction. **ADJACENT, NOT PRE-EMPTING.**

**arXiv:2503.01832 "Rotary Offset Features in Large Language Models"** (Jonasson, March 2025) is the most concerning find. This paper specifically studies large static features in queries and keys that arise from rotary positional embeddings — "rotary offset features" that exhibit large activations and appear as outliers. The paper derives bounds predicting which rotary frequencies give rise to these features and verifies them across architectures. This directly overlaps with the mechanism F3-SRSC is trying to absorb (a static large-magnitude component of q that is "constant" across tokens). Jonasson's paper frames this as a positional-encoding artifact, not a W_Q mean property. **Degree of overlap:** substantial. If the "large row-mean" that F3-SRSC would measure is primarily the rotary-offset-feature phenomenon identified by arXiv:2503.01832, then the paper has already characterized the mechanism. **This is a PARTIAL PRE-EMPTION** — the measurement framing (W_Q weight-level mean vs query activation-level offset) differs, but the underlying phenomenon may be the same.

**arXiv:2510.00028 "Rethinking RoPE Scaling in Quantized LLM"** and related FireQ/Q-ROAR papers are ADJACENT to the quantization-improvement part of F3-SRSC but do not pre-empt the weight-level row-mean measurement.

**Summary:** No full pre-emption found. arXiv:2503.01832 is a significant ADJACENT paper that Stage 3 and Stage 5 missed and that weakens the novelty claim if the measured row-mean is primarily a rotary-offset-feature artifact.

---

### 5. Biased-framing audit

1. **GO threshold (magnitude):** "≥ 15/28 layers have ||μ_row||_F / ||W_Q||_F > 0.05" — this 5% threshold is not gerrymandered but also not grounded in any published baseline. What is the expected magnitude for a uniform-distribution Gaussian W_Q? O(1/√d_model) ≈ 0.022 in absolute per-row mean. The 5% threshold on the RATIO ||μ_row||_F / ||W_Q||_F is not the same as the per-row mean magnitude. The ratio can be large even when individual row-means are small, if the W_Q Frobenius norm is large. **Amendment required:** normalize properly. The correct quantity is mean(||μ_row[head]||_1) / mean(||W_Q[head] row||_1) per head — mean of row means vs mean of row norms.

2. **GO threshold (uniformity):** std/|mean| < 0.20 within d_head block is reasonable but not motivated. The Stage 5 plan notes "the mean is 5× larger than its intra-block variation." This framing assumes the mechanism is "uniform shift" but in light of the CF9 finding above, this framing is moot — the RoPE interaction means the shift is NOT uniform in the logit space regardless of d_head uniformity.

3. **NO-GO structural finding:** "W_Q rows are approximately mean-zero in trained Qwen3" — this is a clean class-level finding if it holds, and the framing is not motivated. PASS.

4. **Quantization improvement claim:** ΔNLL ≥ 0.05 nats on 512-token eval is a reasonable minimum-detectable-effect threshold for this model/eval setup. Not gerrymandered.

---

### 6. Runtime / scope sanity

Stage 5 estimates 30–35 min total (§10). The Stage 1 description says "5 min." The discrepancy is because the original Stage 1 framing covered only the magnitude measurement (step 1). The full plan including INT4 eval runs ~30 min. The 5 min still appears as the selection runtime label. **This is a scope mismatch**: the Track A "5 min" label sells a 30-min experiment. Not a kill criterion but should be corrected in any downstream documentation — the actual runtime is 30–35 min, placing it in the same bin as Track B (not a 5 min quick-check).

Eval and calibration corpus disjointness: no calibration corpus used. PASS.

---

### 6c. Elegant-equivalence rejection ground

**Tag:** `algebraic-identity`. The algebraic-identity claim is softmax(z + c·1) = softmax(z) with c = μ_row · sum(h) after subtraction.

**Section 1 finding:** The precondition for the algebraic identity is that the absorbed component adds a SCALAR constant to ALL attention logits for a given query position. The RoPE interaction breaks this: after position-dependent rotation, the μ_row · sum(h) component becomes a position-dependent vector that contributes a variable amount to each key's attention logit.

**Conclusion:** The algebraic-identity precondition FAILS for Qwen3 (RoPE architecture). The 1.2× elegant-equivalence multiplier at Stage 5 was applied to a mechanism that does not survive the CF9/6c precondition check. **The elegant-equivalence rejection ground is triggered.**

Under 6c: this is the LFAO failure pattern — "an algebraic-identity claim that depends on a commutation / uniformity condition that does not hold on the target architecture." The correct classification is: F3-SRSC is an APPROXIMATION mechanism (not an identity), and must be reframed with explicit error quantification and appropriate controls.

---

### 7. CF13–CF15 verification check

F3-SRSC does not cite CF13–CF15. PASS.

---

### 8. Verdict

**REJECT-PICK-RUNNER-UP**

**Grounds:**
1. **CF9 / 6c frame-mismatch (FATAL):** The algebraic-identity claim (softmax shift-invariance) fails its precondition on any RoPE architecture. RoPE rotation converts the absorbed static row-mean into a key-position-dependent term, NOT a uniform scalar offset. The mechanism is demoted from algebraic-identity to approximation. This is precisely the LFAO failure pattern from the v1 ladder calibration anchors.

2. **Partial prior-art (arXiv:2503.01832):** Jonasson (March 2025) characterized "rotary offset features" — large static-magnitude components of query/key arising from RoPE — as a known structural phenomenon across architectures. The F3-SRSC measurement may be re-deriving this known phenomenon under a different name.

3. **Runtime label mismatch:** The 5 min label is from Stage 1's minimum framing; the actual experiment is 30–35 min.

**Runner-up escalation:** Advance **F5-HPGO** (S_{n_heads} permutation gauge; 15 min; algebraic-identity; head-permutation alignment before delta compression). F5-HPGO's algebraic-identity claim (bipartite matching on a cost matrix is the exact S_{n_heads}-optimal permutation) does not involve softmax or RoPE interaction, and its precondition — that the permutation group acts exactly on concatenated head outputs — is straightforward to verify. Proceed to Stage 5 re-evaluation of F5-HPGO.

---
---

<!-- ============================================================ -->
<!-- TRACK B — F1-VOFR (W_V/W_O Fused Composition Rank, 25 min) -->
<!-- ============================================================ -->

## ═══ TRACK B — F1-VOFR ═══

---

### 1. Frame-mismatch re-check (CF9)

**Imported claim:** M_L = Σ_h W_O^h W_V^{kv(h)} is the exact residual-stream attention-update operator. This is a matrix identity, not a theorem with preconditions — it follows directly from the linearity of the attention output computation (before the activation-nonlinearity, which is none — attention output is linear in V). **PASS — no imported theorem, just linear algebra.**

**Sub-claim: forming M_h = W_O^h @ W_V^{kv(h)} is well-defined.**

Precondition: the GQA head grouping kv(h) = h // 2 is correct for Qwen3-1.7B. Stage 5 §8 kill criterion 1 lists this as a Stage 6 verification task. Independent verification: Qwen3-1.7B architecture specifies num_attention_heads=16, num_key_value_heads=8. GQA grouping is the canonical h // (num_heads / num_kv_heads) = h // 2. This is standard and verifiable from the model's config.json. The plan's §4 step 1 already specifies "kv_head = h // 2 (GQA-2 grouping for Qwen3-1.7B with 16Q/8KV)." **PASS — grouping is correct.**

**Sub-claim: M_L's SVD is meaningful for compression.**

Precondition: M_L must have computable SVD. M_L is 2048×2048 (d_model × d_model). This is always possible via `torch.linalg.svd`. **PASS.**

**Key frame question: is M_L's rank bounded architecturally?**

Stage 5 §9 raises this explicitly in the "Architectural bound check (mandatory)." Red team's independent analysis: each M_h = W_O^h (2048×128) @ W_V^{kv(h)} (128×2048) has rank ≤ 128 (the intermediate d_head dimension). The sum of 16 such matrices can have rank up to min(16×128, 2048) = min(2048, 2048) = 2048. For Qwen3-1.7B specifically, 16 heads × d_head=128 = 2048 = d_model. So M_L can in principle be full rank (rank=2048). This means the architectural ceiling does NOT trivially guarantee concentration — the sum CAN be full rank.

However, the inverse question matters: does M_L's rank guarantee concentration even in a trivial sense? The PERMUTATION CONTROL raised in Stage 5 §9 is crucial here. If each M_h has rank 128, and 16 heads are summed, a randomly-weighted sum of 16 rank-128 matrices spanning diverse subspaces would have rank 2048 (full). The spectrum of M_L relative to the permuted baseline distinguishes trained co-structure from the architectural sum-of-rank-128-products structure.

**The frame-mismatch risk here is a FALSE POSITIVE risk, not a mathematical error:** if the permuted M_L (with W_V columns permuted to destroy co-structure) has r_99/d similar to the real M_L, then any "concentration" is architectural (rank-bounded per M_h), not a trained property. Stage 5 correctly identifies this risk and requires the permutation control. **The frame is sound; the permutation control is load-bearing.**

---

### 2. Calibration ill-conditioning re-check (CF10)

No calibration fit. Compression via truncated SVD of fixed weight matrices. CF10 **does not apply**. PASS.

---

### 3. Skeptic-controls check

**Hypothesis:** "M_L = Σ_h W_O^h W_V^h has concentrated spectrum — the fused operator is low-rank in trained Qwen3."

This is a "X has property Y" claim about trained weights. Controls:

**Permutation control:** Stage 5 §9 defines this correctly and identifies the subtle issue: the permuted M_L may have the same rank ceiling (128 per head) as the real M_L, making the permuted r_99/d comparison meaningful only if it tests TRAINED CO-STRUCTURE, not just rank. The required gap (real r_99/d ≤ permuted r_99/d - 0.10) is appropriate IF the permuted M_L does not trivially saturate r_99/d near 1.0. Red team confirms: a permuted M_L where W_V columns are shuffled independently per head should spread the spectrum more uniformly, increasing r_99/d toward the rank ceiling. **PRESENT AND ADEQUATE.** Required gap ≥ 0.10 ppt is in the GO criterion. PASS.

**Random-init control:** Stage 5 §9 requires this (layer 14, single layer). The GO requires real r_99/d < randinit r_99/d - 0.05. For random-init weights, M_h = random(2048,128) @ random(128,2048) has spectrum determined by random matrix theory (Marchenko-Pastur distribution); the summed M_L of 16 such terms has a predictable spectrum. The 5 ppt gap requirement is conservative (likely the gap will be larger if training produces co-structure). **PRESENT AND ADEQUATE.** PASS.

**Multi-seed:** N/A documented correctly for deterministic weight measurement. Accepted.

**Architectural bound check:** Stage 5 §9 explicitly requires this before claiming results as trained-model findings. This is correctly flagged. PASS — the check is in the plan.

**Skeptic-controls overall:** PASS. All required controls present and documented, with appropriate N/A justifications.

---

### 4. Hidden prior-art search — third pass

**F1-VOFR load-bearing claim:** "Σ_h W_O^h W_V^h spectrum has not been measured on any GQA architecture as of Stage 5 verification (2026-05-09)."

**LatentLLM (arXiv:2505.18413, MERL, May 2025):** This is the most critical adjacent paper. LatentLLM proposes "activation-aware joint tensor compression" that includes jointly decomposing the value-output projection pair (W_V, W_O). It is activation-aware (Hessian-weighted), not a pure weight-only spectrum measurement. Its goal is joint SVD of the pair to minimize activation-weighted error. This is ADJACENT but NOT THE SAME as measuring the spectrum of the explicit fused product M_L = Σ_h W_O^h W_V^h. LatentLLM does not form the head-summed fused operator and measure its singular value decay curve. The novelty of F1-VOFR's M1+M2 mechanism (fused operator spectrum measurement on trained GQA) is intact. **ADJACENT, NOT PRE-EMPTING.** Stage 5 already documented this paper.

**Low-Rank Key Value Attention (arXiv:2601.11471, January 2026):** This paper finds that "attention heads are not fully independent: head-specific KV projections are highly correlated and occupy overlapping subspaces, indicating substantial redundancy." This is structurally related to F1-VOFR's prediction that M_L is low-rank due to head cancellation/redundancy. However, this paper targets KV projections (the W_K and W_V input-side), not the W_V × W_O output-side fused operator. It is ADJACENT — it motivates the same qualitative intuition — but does not form or measure M_L = Σ_h W_O^h W_V^h. **ADJACENT, NOT PRE-EMPTING.**

**Multi-matrix Factorization Attention (arXiv:2412.19255, December 2024):** Low-rank factorization in the Q-K circuit. Not the value-output operator. **NOT ADJACENT enough to matter.**

**Summary:** Third-pass prior-art search confirms novelty. The fused M_L = Σ_h W_O^h W_V^h spectrum measurement on a trained GQA model is not covered by any paper found in this or prior searches. Two new papers (arXiv:2601.11471, arXiv:2505.18413) are ADJACENT and should be added to the Stage 5 plan's prior-art section for intellectual honesty, but neither pre-empts the Frame.

---

### 5. Biased-framing audit

1. **GO threshold (spectrum):** r_99(M_L)/d ≤ 0.75 in ≥ 20/28 layers. Red team compares with CF11 baseline: W_Q r_99/d ≈ 0.63, W_K r_99/d ≈ 0.79. The threshold of 0.75 is calibrated to W_K's analogous result. This is a reasonable grounding. Not gerrymandered. PASS.

2. **Quality GO threshold:** ΔNLL at K=512 ≤ +0.35 nats, described as "slightly looser than W_K K=512 = +0.29 nats." The loosening is justified in the plan as M_L being a composed object. This is marginally acceptable but the loosening should be explicitly bounded: the threshold should not be looser than the W_Q global K=128 result (+0.98 nats). +0.35 nats is well within this bound. PASS.

3. **NO-GO structural finding:** If M_L is full-rank, this "extends CF8 to the attention output operator and closes the low-rank fused attention output class." This is a genuine class-level kill claim. The structural-finding-on-NO-GO is load-bearing, not rhetorical — it produces a CF entry that constrains S6-WDKV and the WAVQ cluster. PASS.

4. **SOFT-GO:** K=1024 ΔNLL ≤ +0.20 nats as a secondary threshold. The SOFT-GO path exists and is defined with a specific threshold. Not a GRAY-zone-as-success abuse — the SOFT-GO produces a weaker but still useful result (less compression, lower quality hit). PASS.

5. **One framing flag:** The GO criterion states "M_L has concentrated spectrum enabling two-factor compression." The compression follow-up (residency arithmetic for 70B) is in §5 downstream. The plan should clarify that the GO produces a MEASUREMENT finding (M_L is low-rank at K=X) which ENABLES a subsequent compression experiment — it does not directly produce a compression result. This is implicit but could be clearer to avoid outcome inflation. Minor amendment.

---

### 6. Runtime / scope sanity

Stage 5 §10 estimates 32 min end-to-end. Stage 1 description says "25 min." The discrepancy: Stage 10 includes random-init control (~3 min) not in the original Stage 1 estimate. The 25 min remains a plausible floor (no random-init control). The 32 min ceiling is within the 8h gate by a large margin. **PASS.**

Eval and calibration corpus disjointness: no calibration corpus. WikiText-2 test split for NLL eval only. PASS.

Smallest test decision quality: Layer 14 early-exit (step 2 in §4) produces a binary GO/NO-GO on spectrum at a single layer before committing to the full 28-layer sweep. This is a correctly structured cheapest-falsification gate. PASS.

**One scope concern:** Step 4 in §4 describes the K=512 truncated forward-pass implementation as "Replace forward-pass attention output computation with: output = U_L @ diag(S_L[:K]) @ V_L[:K,:]^T @ value_projected." This requires modifying the model's forward pass to use the truncated M_L instead of the full W_V and W_O matrices. This is non-trivial surgery on the attention computation in a GQA model (requires patching the attention output computation, not just the weight matrices). The plan does not fully describe how this patching is done in ik_llama.cpp or the HuggingFace implementation. This is a SCOPE UNDER-SPECIFICATION, not a kill criterion. **Amendment required:** spell out the forward-pass patching strategy (e.g., monkeypatch the forward method to replace the attention output projection with M_L @ value_vector, where value_vector = full concatenated V output).

---

### 6c. Elegant-equivalence rejection ground

**Tag:** `algebraic-identity`. The identity is: the actual residual-stream update operator equals M_L = Σ_h W_O^h W_V^h (exact, no approximation in forming it). This is a consequence of linearity of matrix multiplication, and has no preconditions beyond the architecture being standard attention (no learned nonlinearities inside the V→O path).

**Verification:** In a standard Transformer attention block, the contribution of layer L to the residual stream is:
Attn_L(x) = Σ_h W_O^h · (V^h A^h) = Σ_h W_O^h W_V^h · (X A^h) = M_L · (X A^h)
where A^h are the attention weights. M_L = Σ_h W_O^h W_V^h is an exact algebraic factoring. No approximation. **The algebraic-identity claim is VALID and survives the precondition check.** The elegant-equivalence multiplier is correctly applied. PASS.

---

### 7. CF13–CF15 verification check

F1-VOFR cites no CF13–CF15 numbers in any load-bearing position. PASS.

---

### 8. Verdict

**ACCEPT-WITH-AMENDMENTS**

The core hypothesis and experimental design are sound. The algebraic-identity claim survives. No fatal math errors. Novel frame confirmed through third-pass prior-art search. Skeptic controls present and adequate. Three amendments required before execution:

**Amendment 1 (Load-bearing — scope):** Specify the forward-pass patching strategy for the K=512 truncated ΔNLL eval in §4. The plan must clarify how M_L truncation replaces W_V + W_O in the attention computation (e.g., monkeypatching HuggingFace attention forward, or equivalently restructuring the value projection). Ambiguity here would make the ΔNLL result unreproducible.

**Amendment 2 (Load-bearing — architectural bound):** The §9 architectural bound check must be folded into the GO/NO-GO criterion as an EXPLICIT RUNG, not just a note. Before reporting any spectrum concentration result as a "trained-model property," the experiment must report: (a) r_99(M_L) for real model, (b) r_99(M_L_permuted) for column-permuted W_V control, (c) r_99(M_L_randinit) for random-init control. The GO criterion for "trained structure" requires (a) < (b) - 0.10 AND (a) < (c) - 0.05. This is already in §9 but must appear in §5 GO threshold to be operationally binding.

**Amendment 3 (Minor — framing):** Add arXiv:2601.11471 (Low-Rank Key Value Attention, Jan 2026) and arXiv:2505.18413 (LatentLLM, May 2025) to the prior-art section (§13 or §8 kill criterion 2) with explicit ADJACENT classification. These papers share the qualitative intuition motivating F1-VOFR but do not pre-empt the fused operator measurement.

---

## Summary table

| Track | Pick | Verdict | Key kill/amendment |
|---|---|---|---|
| A | F3-SRSC | **REJECT-PICK-RUNNER-UP** | CF9/6c: RoPE interaction breaks algebraic-identity precondition; softmax shift-invariance does NOT apply after RoPE rotation converts absorbed row-mean into a key-position-dependent term. Advance F5-HPGO. |
| B | F1-VOFR | **ACCEPT-WITH-AMENDMENTS** | Three amendments: (1) specify forward-pass patching for truncated eval, (2) fold architectural-bound check into GO criterion operationally, (3) add arXiv:2601.11471 and 2505.18413 to prior-art section. |
