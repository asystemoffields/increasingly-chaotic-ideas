# Stage 3 — Deep Research Report — Run 003
Date: 2026-05-09

Pipeline context: CF1–CF12 confirmed; v2-CF1 (all-layer Jaccard) and v2-CF2 (cross-layer W_Q kill) added. CF13–CF15 unconfirmed, quarantined. Kill list: v2-CHEAP-TEST-001 (cross-layer W_Q, class-dead).

Depth allocation: ~1000w each for F1-VOFR, F4-WAOS, A2-RTGF, U6-NQDS; ~500–600w for WAVQ, C5-WUPDOWN, C1-QAOC, F5-HPGO, AERO-QW3; ~400w for MHRS, RAOK-72, SOBIB-V, U1-GECP; ~300w wildcards (F3-SRSC, F2-RGAS, F6-ASFA, F7-TEGRA, C3-KDJB, C2-LAYERFOLD, C6-SPECDRIFT, NTQW, A4-APPA, A5-TCRD, A3-NTCA, A6-SSCF, U7-NAWP).

---

## PATH 2 REPRESENTATIVES (DEEP)

---

### F1-VOFR — W_V / W_O Fused Composition Rank

**Score: 14 | Track B | Convergence Cluster 1**

#### 1. Mechanism decomposition

- M1: For each layer L, form the fused operator M_L = Σ_h W_O^h W_V^h (shape d_model × d_model = 2048×2048 on 1.7B). This is the exact linear map the attention block applies to the residual stream.
- M2: SVD of M_L measures the effective rank of the residual-stream attention update. If r_99(M_L)/d ≤ 0.70, M_L is low-rank in the operationally relevant object.
- M3: Store M_L ≈ U_L Σ_L V_L^T at rank K via two-factor split: (U_L Σ_L^{1/2}) and (Σ_L^{1/2} V_L^T). No retraining. Purely algebraic compression.
- M4: ΔNLL at K=512 tested as proxy for quality gate (analogy to W_Q K=512 = +0.20 nats, W_K K=512 = +0.29 nats).

#### 2. Per-claim prior-art status

**M1 (fused operator M_L = Σ_h W_O^h W_V^h as measurement object):**
Status: **NOVEL** as of 2026-05-09.

Closest cousin: SVD-LLM (arXiv:2403.07378, Mar 2024) applies SVD independently to each weight matrix including W_V and W_O; it does NOT form the head-summed composition M_L. LatentLLM (arXiv:2505.18413, May 2025) performs joint tensor compression of Q and K projections concurrently but does not form the summed W_O W_V composition. The GPTVQ paper (arXiv:2402.15319) applies Hessian-weighted codebooks to individual matrices. MLA (DeepSeek-V2) decomposes per-token KV into a shared low-rank at training time — different object, different regime.

No paper found as of 2026-05-09 that (a) forms M_L = Σ_h W_O^h W_V^h on a trained GQA model, (b) measures its spectrum, and (c) applies zero-SGD post-training compression to this fused operator.
elegance-class: `algebraic-identity` — M_L is the exact residual-stream update operator; no approximation in forming it. The compression (SVD of M_L) is algebraically equivalent to a low-rank approximation of the exact computation rather than a per-matrix heuristic.

**M2 (SVD spectrum of M_L measures true attention output rank):**
Status: **ADJACENT** to GPTVQ's activation-Hessian weighting philosophy (GPTVQ says "weight the input directions by activation importance"; M2 says "measure the output directions exactly via M_L"). Differentiation: GPTVQ's Hessian is X^T X (input-side weighting); M2 is direct output-operator spectrum (output-side). These are dual problems and not the same measurement.

**M3 (two-factor storage at rank K, purely algebraic):**
Status: **ADJACENT** to arXiv:2512.03062 (PivGa, Nov 2025), which exploits "gauge freedom in the parametrization of low-rank factors" — specifically, the fact that for any rank-K factorization A = U V, one can insert any invertible K×K matrix G such that A = (UG)(G^{-1}V), and find G to minimize storage. PivGa applies this to SVD factors after per-matrix decomposition. VOFR's M3 is the prior decomposition step (form M_L first, then decompose); the gauge freedom PivGa identifies applies equally to M_L's factors. The differentiation: PivGa does not form M_L — it applies the gauge trick to W_V and W_O independently, which as VOFR argues, is theoretically suboptimal for GQA where the head-sum is the actual computation.

Two-paper composition check: SVD-LLM + PivGa ≈ "apply per-matrix SVD and then gauge-compress the factors." Value-add of VOFR beyond this composition: the fused M_L object is structurally different from independent per-matrix decompositions — if heads have cancellation in their V×O products when summed, M_L's effective rank is LOWER than any individual W_V or W_O rank. This is the algebraic argument that neither SVD-LLM nor PivGa captures. Research value-add is structural.

**M4 (ΔNLL threshold by analogy to CF11):**
Status: **PARTIAL** — the analogy to W_Q/W_K is reasonable (all are attention weight matrices on the same architecture) but the fused operator M_L has never been tested. The sub-claim "M_L behaves spectrally like W_K because both are attention matrices" is an analogy, not a derivation.

#### 3. Frame-mismatch check (CF9)

Imported object: SVD (always valid for real matrices). No theorem precondition to verify. GQA structure: W_O is 2048×(16×128) and W_V is (8×128)×2048 (8 GQA kv-heads); the computation of M_h is per Q-head, not per KV-head, so the head-sharing structure in GQA changes how M_L is formed. Specifically: for GQA, 16 Q-heads share 8 KV-heads, so W_V has only 8 variants while W_O has 16. The fused M_h = W_O^h W_V^{kv(h)} where kv(h) maps Q-head to its KV-head group (h//2 for GQA-2). M_L = Σ_{h=0}^{15} W_O^h W_V^{kv(h)}. This is still a 2048×2048 matrix. The SVD is valid. Precondition: none beyond real matrices. PASS.

#### 4. CF10 calibration check

No calibration fit in F1-VOFR. Compression is by truncated SVD of M_L. No parameters fitted. CF10 does NOT apply. PASS.

#### 5. Residency arithmetic

Qwen3-1.7B: d_model=2048, 16 Q-heads, 8 KV-heads. W_V per KV-head: 128×2048. W_O per Q-head: 2048×128. M_h = W_O^h (2048×128) × W_V^{kv(h)} (128×2048) = 2048×2048 per Q-head. M_L = sum of 16 such matrices = 2048×2048.

Storage baseline: W_V (8 heads × 128×2048 × 2 bytes × 28 layers) + W_O (16 heads × 2048×128 × 2 bytes × 28 layers) = 8×524288×28 + 16×524288×28 = 117 MB + 235 MB = 352 MB (combined W_V+W_O in standard form). Actually using total W_V = 8 × 128 × 2048 × 2 × 28 = 117 MB and W_O = 16 × 2048 × 128 × 2 × 28 = 235 MB → 352 MB total.

At K=512 compression of M_L: two-factor storage U_L (2048×512) + V_L^T (512×2048) per layer = 2 × 2048×512×2 × 28 = 2 × 56 MB = 112 MB. Savings: 352 MB → 112 MB = 240 MB on 1.7B model.

Qwen3-72B scaling: d_model=8192, 64 Q-heads, 8 GQA kv-heads. W_V per KV-head: 128×8192. W_O per Q-head: 8192×128. M_L = Σ W_O^h W_V^{kv(h)} = 8192×8192 per layer × 80 layers. Standard W_V+W_O: ~3.4 GB + ~10.2 GB = 13.6 GB. At K=512 compressed M_L: 2 × 8192×512×2 × 80 = 2 × 0.671 GB = 1.34 GB. Savings: 12.3 GB → 1.34 GB on attention V+O alone.

KV cache at context 4K: M_L compression does NOT directly compress KV cache. At inference, V-cache stores runtime W_V·x vectors (not M_L). However, if inference uses the compressed M_L factorization, the value projection at runtime is the 512-dimensional intermediate rather than the full 2048-dimensional V. This means K-cache and V-cache can optionally be stored in the compressed 512-dim space, reducing KV cache by 4× on the V side at no additional approximation loss.

CF13*/CF15* dependencies: none. The residency math uses only CF11 analogies for quality estimates (conservative). No CF13–CF15 numbers are load-bearing.

Per-token compute: at 70B, M_L forward pass uses the two-factor form: (V_L^T × value_in) at 512×8192 = 4M MACs, then (U_L × result) at 8192×512 = 4M MACs, vs original M_L at 8192×8192 = 67M MACs. 16× compute savings per attention output block in addition to residency savings.

#### 6. Smallest-test sharpening

Script: `scripts/vofr_spectrum.py`
Model: Qwen3-1.7B-Base bf16
Procedure:
1. Load model. For each of 28 layers: extract all W_O^h (2048×128, 16 heads) and W_V^{kv(h)} (128×2048, mapped by GQA grouping). Compute M_L = Σ_h W_O^h @ W_V^{kv(h)} (2048×2048). Run torch.linalg.svd. Record r_99/d and cumvar@512.
2. Apply K=512 truncation: replace forward pass attention output with U_L @ (Σ_L V_L^T @ value_projected). Measure ΔNLL on 512-token WikiText-2 held-out.

Wall-clock on Ryzen 5 7530U: SVD of 28 × 2048×2048 matrix ≈ 28 × ~30s = ~14 min for SVD; + ~10 min forward pass ΔNLL eval = ~25 min total. Well within 8h gate.

Output path: `experiments/stage0/ladder_v2/round3_vofr/`

#### 7. Refined go/no-go

GO: r_99(M_L)/d ≤ 0.75 in ≥ 20/28 layers AND ΔNLL at K=512 ≤ +0.35 nats. Advance to 70B scaling arithmetic and NTQW cascade.
NO-GO: r_99/d > 0.85 in majority of layers → extends CF8 to all attention weight objects. M_L is full-rank; compression class kill on fused composition route.
GRAY: r_99/d between 0.75 and 0.85. Follow-up: test per-head M_h (single head composition) rather than summed — if individual M_h is more concentrated, decompose per head. Also test K=256 vs K=512 ΔNLL to find the quality elbow. Follow-up runtime: ~20 min.

#### 8. Updated risk profile

| Risk | Mitigation |
|---|---|
| Head summation cancels concentration: individual M_h low-rank but sum full-rank | Measure both Σ_h M_h and single-head M_h in the same script; if sum is worse, pivot to per-head |
| GQA head-grouping changes M_L structure vs MHA | Explicit GQA-aware computation already specified above; no structural ambiguity |
| K=512 suboptimal for this operator | Sweep K=256,512,1024 in the ΔNLL measurement; pick knee |

Cheapest falsification: measure r_99(M_L) at a single layer (layer 14 for CF11 comparability). If r_99/d > 0.90 there, kill without full sweep.

#### 9. Two-paper composition flag

SVD-LLM (arXiv:2403.07378) + PivGa/2512.03062 ≈ "apply per-matrix SVD with gauge-compressed factors to all attention weights."
Value-add of VOFR: M_L as the measurement object is structurally different from independent W_V and W_O SVDs, because head-summed cross-cancellation can reduce effective rank below any per-matrix bound. This is a genuine structural insight, not engineering-integration. RESEARCH.

#### 10. Verdict

**REFINE.** Highest-scored advancer (14). Prior art does not cover the fused M_L = Σ_h W_O^h W_V^h spectrum measurement. PivGa is ADJACENT not PRE-EMPTED. 25-min experiment with clean binary outcome. Advance to Stage 4/5.

---

### F4-WAOS — W_down Activation-Weighted Output Subspace

**Score: 14 | Track B | Convergence Cluster 2**

#### 1. Mechanism decomposition

- M1: Define A_L = silu(W_gate^L·x) ⊙ (W_up^L·x) as post-SwiGLU activation vector (shape d_FFN per token).
- M2: Calibration matrix Delta_L = W_down^L @ [A_L^{(1)} ... A_L^{(N)}] (shape d_model × N). SVD of Delta_L measures the effective OUTPUT subspace of W_down under calibration distribution.
- M3: CF1 (W_up dominates firing rank) + CF3 (K=1% Jaccard=0.31, token-dynamic but bounded concentration) → fewer than d_FFN distinct firing patterns → Delta_L effective rank r_eff ≪ d_model.
- M4: Store W_down compressed as (U_L, W_down_compact) where U_L is the output basis (d_model × r_eff) and W_down_compact = U_L^T @ W_down^L (r_eff × d_FFN). Forward pass: δh_L = U_L @ (W_down_compact @ A_L).
- M5: Crucially bypasses CF8: CF8 kills weight-rank compression of W_down; M4 compresses the output subspace of W_down under calibration activations — a different object entirely.

#### 2. Per-claim prior-art status

**M1+M2 (output subspace measurement via Delta_L = W_down A_L):**
Status: **ADJACENT** to GPTVQ (arXiv:2402.15319). GPTVQ uses input Hessian X^T X (a weighted metric on the INPUT space of W). M2 uses the actual output span Delta_L (a measurement of the OUTPUT space of W_down under activations). These are dual problems. Key differentiation: GPTVQ's Hessian weights INPUT directions for quantization; WAOS identifies OUTPUT directions for rank-compression. No published paper applies output-subspace SVD to W_down specifically under SwiGLU-gated activations.

ASVD (arXiv:2312.xx, activation-aware SVD) applies activation-scaled SVD where each singular value is weighted by the activation scale of its input channel — this is still an INPUT-side weighting, not an OUTPUT-subspace compression. Status: ADJACENT not PRE-EMPTED. Differentiation in one sentence: ASVD rescales the input channels by activation magnitude before SVD of W; WAOS computes the actual output span Delta_L = W·A as the measurement object and takes its SVD.

ARHQ (arXiv:2605.00140) performs "truncated SVD on the scaled weight matrix W G^{1/2}_x" — this is G^{1/2}_x = diag of activation scales, essentially ASVD variant. Still INPUT-side. ADJACENT.

**M3 (CF1+CF3 → r_eff ≪ d_model for Delta_L):**
Status: **NOVEL** — the prediction that CF1 soft-sparsity concentrates Delta_L's output subspace is not in any cited paper. The chain CF1→CF3→r_eff(Delta_L) is a specific prediction about W_down's output space that requires our particular empirical measurements. elegance-class: `conserved-quantity` — CF1's firing-rank dominance acts as a conserved statistical structure that concentrates Delta_L.

**M4 (compressed storage bypassing CF8):**
Status: **NOVEL** as of 2026-05-09 — no paper stores W_down in output-subspace factored form (U_L, W_down_compact) motivated by calibration-output-span analysis. The CF8-bypass argument (weight rank ≠ output subspace rank under concentrated activations) is the specific structural insight.

Two-paper composition flag: GPTVQ + SVD-LLM ≈ "use activation information to weight the SVD of W_down." WAOS value-add: the object being decomposed is Delta_L (output span), not W_down itself. The distinction matters because a full-rank W_down (CF8) can still produce a low-rank output span if activations concentrate. This is a structural insight neither GPTVQ nor SVD-LLM addresses.

#### 3. Frame-mismatch check (CF9)

Imported object: SVD on Delta_L = W_down @ A_L. No theorem precondition beyond real-matrix SVD. Valid always.

One potential import: "activation soft-sparsity concentrates output subspace." This is not a theorem import — it is an empirical claim (CF1, CF3). No precondition to verify beyond CF1/CF3 holding for post-SwiGLU activations, which they do (CF1 confirmed).

PASS — no frame-mismatch risk.

#### 4. CF10 calibration check

Delta_L = W_down @ A_L (shape d_model × N, where N = 200 calibration tokens). This is NOT a parameter-fitting operation — it is a measurement of the output span under calibration data. The SVD of Delta_L reveals the output subspace structure without fitting any parameters. CF10 does not apply to subspace measurement.

However, a second-order concern: the calibration corpus (N=200 tokens) must adequately cover the activation distribution. If N is too small, Delta_L is artificially low-rank due to calibration concentration (not true output-subspace concentration). The Stage 1 proposal already identifies this and proposes the two-subset test (compute cumvar on two disjoint 100-token subsets; require < 5 ppt difference). This mitigation is sound. PASS WITH CAVEAT: report the two-subset consistency check in the experiment.

#### 5. Residency arithmetic

Qwen3-1.7B: W_down per layer: d_FFN × d_model = 6144×2048 × 2 bytes = 24.6 MB. Total 28 layers: 689 MB.

At r_eff=512 (90% variance threshold): compressed form = U_L (2048×512) + W_down_compact (512×6144) = 2 × 2048×512×2 × 28 = 112 MB. Wait — W_down_compact = U_L^T @ W_down is 512×6144; storage = 512×6144×2×28 = 176 MB. U_L = 2048×512×2×28 = 56 MB. Total compressed: 232 MB. Savings: 689 MB → 232 MB = 457 MB on W_down alone for 1.7B.

Qwen3-72B: W_down per layer = 29568×8192 × 2 bytes = 469 MB × 80 = 37.5 GB. At r_eff=512: U_L (8192×512×2×80) + W_compact (512×29568×2×80) = 0.671 GB + 2.41 GB = 3.08 GB. Savings: 37.5 GB → 3.1 GB = 34.4 GB. This is the single largest residency reduction in the entire pipeline — IF r_eff ≤ 512 holds.

Conservative branch (if CF13–CF15 don't replicate): CF13'' NVMe bandwidth is unconfirmed. The W_down compression is purely weight-side (no NVMe mechanics), so this branch is not affected.

Per-token compute: at r_eff=512, MLP forward: W_down_compact (512×6144) @ A_L = 3.1M MACs + U_L (8192×512) @ result = 4.2M MACs vs original W_down (8192×29568) @ A_L = 242M MACs at 72B. ~35× compute reduction on the W_down GEMV — the most expensive single operation in the MLP.

#### 6. Smallest-test sharpening

Script: `scripts/waos_output_subspace.py`
Model: Qwen3-1.7B-Base bf16
Calibration corpus: 200-token diverse sample from WikiText-2 training split (first 200 non-overlapping 1-token contexts for coverage)
Eval corpus: WikiText-2 test, 512 tokens

Procedure:
1. Run 200-token forward pass; cache post-SwiGLU activations A_L per layer (d_FFN × 200 matrix). Cache two disjoint 100-token subsets for consistency check.
2. For each layer: compute Delta_L = W_down^L @ A_L (d_model × 200 = 2048×200). SVD of Delta_L (cost: negligible — 2048×200 matrix). Compute cumvar@K for K ∈ {64, 128, 256, 512, 1024}.
3. Consistency check: cumvar@512 on subset A vs B; require |diff| < 0.05.
4. If GO: construct compressed W_down_compact per layer; run ΔNLL on 512-token held-out with compressed forward pass.

Wall-clock: ~40 min (forward pass 15 min + SVD 5 min + ΔNLL eval 20 min). Within 8h gate.

Output path: `experiments/stage0/ladder_v2/round3_waos/`

Layers/matrices touched: all 28 W_down matrices.

#### 7. Refined go/no-go

GO: cumvar(K=512) ≥ 0.90 in ≥ 15/28 layers AND ΔNLL ≤ +0.25 nats. Advance to 70B cascade.
NO-GO: cumvar(K=512) < 0.80 in majority of layers → SwiGLU soft-sparsity does NOT concentrate W_down's output span; closes "activation-soft-sparsity → output-subspace compression" class.
GRAY: cumvar@512 passes but ΔNLL > 0.40 nats. Follow-up: test K=1024 (50% compression), which may close quality gap while still saving ~50% W_down storage. Follow-up runtime: ~15 min additional.

#### 8. Updated risk profile

| Risk | Mitigation |
|---|---|
| 200 tokens insufficient → artificially low Delta_L rank | Two-subset consistency check required; extend to 1000 tokens if diff > 5 ppt |
| Output-subspace compression changes GEMV structure → requires new kernel | At 1.7B scale, implement via standard matmul chain; kernel optimization is Stage 5 implementation concern |
| CF1 soft-sparsity sufficient to concentrate A_L but not Delta_L (W_down can spread it) | Direct measurement answers this; no mitigation needed pre-experiment |

Cheapest falsification: cumvar@512 at single layer (layer 14). If < 0.70, stop.

#### 9. Two-paper composition flag

GPTVQ (arXiv:2402.15319) + ASVD (activation-weighted SVD, training-free) ≈ "use activation distribution to guide W_down compression via input-side weighting."
WAOS value-add: Delta_L = W_down @ A_L is the OUTPUT-side measurement, not input-weighted. The critical structural insight is that CF8 (W_down full-rank as a weight matrix) does not preclude low-effective-rank output span. This is a novel bypass of the CF8 class kill. RESEARCH.

#### 10. Verdict

**REFINE.** Highest-scored advancer tied at 14. CF8-bypass argument is structurally valid and not pre-empted. ADJACENT to GPTVQ/ASVD but the output-vs-input distinction is the load-bearing claim. 40-min experiment. Advance to Stage 4/5.

---

### A2-RTGF — Residual-Stream Sign-Flip Gauge Fold

**Score: 13 | Track B | Path 4 elegant-equivalence**

#### 1. Mechanism decomposition

- M1: The residual stream has a Z₂ × Z₂ × ... × Z₂ (d_model times) gauge symmetry: flipping channel d's sign in h accompanied by flipping all W_in[:,d] and W_out[d,:] is an exact identity.
- M2: SGD does not break this symmetry; post-training, the sign conventions of all 2048 channels are arbitrary artifacts of initialization and optimization path.
- M3: For each channel d, compute the "majority sign" of W_up[:,d] across all 28 layers — the sign convention that makes most W_up entries positive. Apply the flip to channel d and update all matrices that read/write channel d (W_Q, W_K, W_V, W_O per layer + W_gate, W_up, W_down per layer + RMSNorm gains + embedding).
- M4: Post-flip, W_up[:,d] entries are more concentrated on the positive half-line. Unsigned INT4 quantization grids waste less range → lower quantization error at fixed bpw.
- M5: The transform is lossless (exact gauge), verifiable by ΔNLL = 0 check before quantization. The quantization improvement is measured as ΔNLL reduction at INT4 vs the non-gauge-fixed baseline.

#### 2. Per-claim prior-art status

**M1 (Z₂ sign-flip as exact gauge symmetry of the residual stream):**
Status: **ADJACENT** to two known classes of work:

(a) SmoothQuant (arXiv:2211.10438): migrates per-channel activation scale (continuous, multiplicative) into weight matrices. This is a scaling gauge, not a sign gauge. The Z₂ sign flip is a discrete subgroup (order-2) of the continuous scaling group; SmoothQuant applies the full scaling transform, which subsumes sign-flip in the limit of scale → -1. However, SmoothQuant does not identify the per-channel sign convention as an independently exploitable degree of freedom — it treats scale and sign together.

(b) OS+ (arXiv:2304.09145): applies per-channel scaling + sign-alignment to minimize quantization error, specifically including sign flips as part of its per-channel transform. The OS+ mechanism: for each output channel, finds scale s and sign b ∈ {+1, -1} that minimizes rounding error when combined with outlier suppression.

**OS+ is ADJACENT to M1–M4.** OS+ flips signs and applies scales jointly; RTGF isolates the discrete Z₂ component (sign only, no scale). The question is whether the sign-only component is independently load-bearing vs the combined scale+sign transform in OS+. Differentiation: RTGF applies the sign fix to the residual-stream representation (all reading/writing matrices updated simultaneously), while OS+ applies it per-output-channel of individual weight matrices independently. The residual-stream framing ensures consistency across all matrices that share channel d — OS+ does NOT enforce this consistency.

Search result finding: FPTQuant (arXiv:2506.04985, June 2025) introduces "function-preserving transforms" including equivariant transforms for quantization — this is ADJACENT to RTGF's gauge framing but FPTQuant targets per-layer transformations, not per-channel residual-stream sign conventions.

ParoQuant (arXiv:2511.10645) applies "pairwise rotation quantization" for quantization quality. ADJACENT but rotation is a continuous transform, not discrete sign.

**Summary for M1**: ADJACENT to OS+ (not PRE-EMPTED). The residual-stream consistency argument (enforcing the same sign convention across all matrices sharing channel d, not per-matrix independently) is the distinguishing structural claim.

**M2 (SGD arbitrariness of sign convention):**
Status: **NOVEL** — no paper frames this as "trained weights land in random gauge of Z₂^d symmetry." The observation that SGD does not fix discrete sign conventions is implicit in OS+ but never stated as the mechanism. elegance-class: `gauge-exploitation` — the Z₂^d gauge group is a fundamental discrete symmetry of the residual stream architecture that SGD leaves in random gauge.

**M3 (majority-sign across layers as canonical gauge):**
Status: **NOVEL** — applying the majority-sign rule across layers as a canonicalization is not in OS+ or any cited paper.

**M4 (post-flip INT4 quantization improvement):**
Status: **PARTIAL** — OS+ already demonstrates that per-channel sign+scale transforms improve INT4 quality. The partial precedent is OS+'s combined transform. The isolated Z₂ component's contribution to quantization improvement (separate from scale) is unpublished.

Two-paper composition flag: SmoothQuant (arXiv:2211.10438) + OS+ (arXiv:2304.09145) ≈ "continuous scaling + discrete sign flip per channel for pre-quantization weight redistribution." RTGF's value-add: explicitly framing this as a residual-stream gauge symmetry enforcement, which (a) guarantees cross-layer consistency that OS+'s per-matrix approach misses, and (b) identifies the sign component as independently exploitable in the Z₂ orbit, potentially enabling smaller per-channel perturbation than the full OS+ combined transform. The residual-stream consistency is a structural claim OS+ leaves on the table.

Overall assessment: M1 is ADJACENT (OS+ does something similar); M2+M3 are NOVEL framing; M4 is PARTIAL. Two of 5 load-bearing claims are fully novel; the mechanism is not PRE-EMPTED. The pipeline RTGF adds beyond OS+ is small but non-trivial (residual-stream consistency vs per-matrix independence). Not engineering-integration — the residual-stream framing is a structural claim.

#### 3. Frame-mismatch check (CF9)

Imported object: "Z₂ gauge symmetry" from physics/group theory. Precondition: the residual stream must be additively structured so that channel-d sign flip with compensating matrix updates is an exact identity. Precondition holds: the residual stream is an additive accumulation (h_{L+1} = h_L + δh_L), and all operations on channel d are linear in h_d. The Z₂ transform is exact. PASS.

No imported theorem whose precondition is violated. The elegance argument (SGD leaves Z₂ gauge in random configuration) is an empirical claim, not a theorem import. The precondition is "all weights that read/write channel d are updated consistently" — verifiable in the implementation.

#### 4. CF10 calibration check

M5 requires measuring per-channel sign distribution across layers. This is a pure measurement (majority sign of W_up[:,d] over 28 × 6144 entries), not a parameter fit. CF10 does not apply. The subsequent INT4 quantization is a standard operation. PASS.

#### 5. Residency arithmetic

RTGF is a pre-quantization transform, not a direct residency reduction. The gain is in quantization quality at fixed bpw.

At INT4 (4 bpw) without gauge-fix: if 40% of W_up[:,d] entries are negative for some channels (worst case for unsigned INT4), the quantization grid misses 40% of the signal range. After RTGF with majority-sign enforcement, if f_positive rises to ≥ 70% for most channels, the quantization grid is better utilized. Expected ΔNLL improvement at INT4: 0.1–0.3 nats (estimated from unsigned-INT4 asymmetry analysis). Equivalent to ~0.1–0.2 bpw improvement — costs nothing in storage.

Practically: RTGF is applied once at model-load or model-preparation time; the stored weights ARE the gauge-fixed weights. No runtime cost. The benefit accumulates across all INT4-quantized matrices: W_Q, W_K, W_V, W_O, W_gate, W_up, W_down (all attention + MLP weights). Full-model quantization improvement.

#### 6. Smallest-test sharpening

Script: `scripts/rtgf_sign_audit.py`
Model: Qwen3-1.7B-Base bf16
Procedure:
1. For each of the 2048 residual channels d: compute fraction f_positive of W_up[:,d] entries > 0 across all 28 layers (28 × 6144 values = 172K per channel). Record distribution of f_positive.
2. Measure bimodality: fraction of channels with f_positive < 0.35 or f_positive > 0.65.
3. If bimodal: apply majority-sign flip to all affected channels, updating W_Q, W_K, W_V, W_O (all layers), W_gate, W_up, W_down (all layers), RMSNorm gains, embedding. Measure ΔNLL = 0 (lossless check). Then apply INT4 quantization to all matrices and measure ΔNLL at INT4 with vs without RTGF.

Wall-clock: 3 min for audit + 2 min for sign updates + 20 min for INT4 quantization + eval = ~25 min. Within 8h gate.

Output path: `experiments/stage0/ladder_v2/round3_rtgf/`

#### 7. Refined go/no-go

GO: ≥ 30% of channels have f_positive > 0.70 or f_positive < 0.30 AND ΔNLL_INT4_RTGF < ΔNLL_INT4_baseline by ≥ 0.05 nats. Advance to cascade integration.
NO-GO: f_positive distribution is near-uniform (bell around 0.50) → SGD approximately gauge-fixes sign conventions during training. Structural finding: trained Qwen3 W_up has near-zero sign bias per channel.
GRAY: bimodal sign distribution but < 0.05 nats improvement at INT4 → the sign bias exists but OS+-style per-channel scale already captures the main effect. Follow-up: test RTGF composed with SmoothQuant scaling to check whether sign-fix adds value on top.

#### 8. Updated risk profile

| Risk | Mitigation |
|---|---|
| RMSNorm undoes sign fix at runtime (re-imposes sign ambiguity per layer) | RMSNorm is sign-invariant (operates on magnitude); the Z₂ transform commutes with RMSNorm |
| OS+ already captures the same effect; RTGF adds nothing | If ΔNLL improvement equals OS+ literature values, RTGF is engineering-integration; the distinguishing test is the residual-stream cross-layer consistency |
| Sign flip propagates to embedding lookup (INT4 embedding also affected) | Embedding update is included in M3; apply flip to W_E columns for channel d |

Cheapest falsification: 3-minute sign-audit shows uniform distribution. Kill.

#### 9. Two-paper composition flag (see §2 above)

SmoothQuant + OS+ ≈ RTGF minus the residual-stream framing. Value-add: cross-layer consistency of the Z₂ gauge. Not trivial engineering-integration.

#### 10. Verdict

**REFINE.** M1 ADJACENT to OS+ (not PRE-EMPTED); M2+M3 NOVEL framing; no CF9/CF10 violations. 3-min falsification test. The elegance-class `gauge-exploitation` tag is load-bearing for Stage 5 elegant-equivalence multiplier. Advance to Stage 4/5.

---

### U6-NQDS — NVMe Queue-Depth Saturation for Weight Streaming

**Score: 12 | Track B | Convergence Cluster 4**

#### 1. Mechanism decomposition

- M1: NVMe SSDs achieve rated sequential throughput only at QD ≥ 4–8 outstanding I/O requests (QD=1 typically 50–60% of rated throughput; QD=4 achieves 90–95%+).
- M2: LLM autoregressive inference has deterministic layer access order → predictable future I/O schedule known at layer N compute time.
- M3: Windows IOCP (I/O Completion Ports) with explicit overlapped I/O (`CreateIoCompletionPort`, `ReadFileEx`) allows application-controlled QD up to the driver limit, bypassing OS demand-paging QD=1 baseline.
- M4: At QD=4 with 3-layer look-ahead: effective NVMe throughput approaches 4–5 GB/s vs baseline ~1.75 GB/s (QD=1 mmap), producing ~2.5× speedup in layer-load time.
- M5: At 70B (191 MB/layer), with 3-layer prefetch overlap during layer-N compute (~16 ms at DRAM bandwidth): effective per-layer I/O time = 191 MB / 4.5 GB/s / 3 ≈ 14 ms, matching the compute bound. Result: near-compute-bound inference at 70B NVMe-resident.

#### 2. Per-claim prior-art status

**M1 (NVMe QD behavior for LLM weight streaming):**
Status: **ADJACENT** to a 2025 study: "An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD" (CHEOPS '25, ACM 2025). This paper characterizes NVMe I/O behavior for LLM workloads but focuses on GPU-based systems and KV cache offloading, not CPU-based weight streaming with application-controlled IOCP. ADJACENT not PRE-EMPTED.

**M2 (deterministic access schedule enables accurate prefetch):**
Status: **NOVEL** — no published paper applies this observation specifically to Windows IOCP for LLM weight streaming. PRESERVE (arXiv:2501.08192) prefetches model weights in GPU-based distributed serving; NQDS targets CPU/NVMe desktop inference with IOCP. Different target.

**M3 (Windows IOCP for application-controlled QD):**
Status: **NOVEL** as of 2026-05-09 — no LLM inference paper uses IOCP for NVMe QD saturation. Database engines (PostgreSQL async I/O, MySQL InnoDB AIO) use overlapped I/O, but not in the LLM serving context. The CHEOPS '25 study characterizes I/O patterns but does not propose IOCP-based QD control. Primitive: Windows IOCP API (CreateIoCompletionPort, OVERLAPPED structures).

**M4+M5 (throughput estimate at QD=4):**
Status: **PARTIAL** — the QD=4 throughput improvement is hardware-dependent. For the user's specific NVMe device and Windows StorNVMe driver, this is an empirical claim requiring the standalone measurement. The arithmetic is plausible but not verified on this exact hardware.

Two-paper composition flag: PRESERVE (arXiv:2501.08192, GPU-based weight prefetch) + CHEOPS '25 NVMe characterization ≈ "prefetch LLM weights from storage." Value-add: IOCP-based application-controlled QD is a qualitatively different mechanism from OS-hint prefetch (PrefetchVirtualMemory / mmap) — it bypasses the VM subsystem entirely and issues explicit reads with precise QD control. On Windows, this is the only way to guarantee QD > 1 for file I/O in user space. Research value-add: the explicit QD control mechanism for Windows CPU-only LLM inference is unpublished.

#### 3. Frame-mismatch check (CF9)

No imported theorem. The NVMe QD behavior is a hardware/firmware property (documented in NVMe spec). Precondition: "this specific NVMe device and Windows StorNVMe driver support QD > 1 for sequential reads via IOCP." This is the empirical question the experiment answers. No precondition violation — the failure mode is "device already saturates at QD=1" (some consumer drives optimize for sequential access at QD=1), which the experiment directly tests. PASS.

#### 4. CF10 check

No calibration fit. Pure I/O scheduling mechanism. PASS.

#### 5. Residency arithmetic

No weight residency change. The mechanism improves THROUGHPUT at fixed weight size.

Qwen3-72B at 0.55 bpw (5.35 GiB): per-layer weight bytes = 5.35 GiB / 80 = ~68 MB. At QD=1 mmap baseline: ~68 MB / 1.75 GB/s = 39 ms/layer I/O. At QD=4: 68 MB / 4.5 GB/s = 15 ms/layer. With 3-layer overlapped: effective 15/3 = 5 ms/layer from the I/O perspective. Compute per layer at DRAM bandwidth: ~68 MB / 11.5 GB/s = 5.9 ms/layer. Bottleneck flips from I/O-bound to compute-bound. Total: ~28 layers × max(5.9, 5) = ~165 ms/token → ~6 tok/s at 70B NVMe resident. This is a ~6× improvement over the baseline QD=1 path (~1 tok/s).

CF13*/CF14* dependency: U6 is measuring the CF13'' number (NVMe sequential throughput on this device). This is a self-measuring proposal — it produces the number it needs. Conservative branch: if QD=4 only achieves 2× improvement (pessimistic consumer SSD), 70B goes from ~1 tok/s to ~2 tok/s. Still meaningful.

#### 6. Smallest-test sharpening

Script: `scripts/nqds_benchmark.c` (or `scripts/nqds_benchmark.py` via ctypes)
Procedure: Write a 50-line C program that reads 4 × 50 MB sequential chunks of the Qwen3-1.7B GGUF file using (a) sequential `ReadFile` (QD=1 baseline) and (b) 4 outstanding `ReadFileEx` calls via IOCP (QD=4). Flush page cache between runs (`SetSystemFileCacheSize` or mapped view unmap). Measure throughput in MB/s for each condition. Runtime: ~2h (includes program writing + measurement). This is the standalone program test; ik_llama.cpp integration is not required at this stage.

Wall-clock: 2h. Within 8h gate.

Output path: `experiments/stage0/ladder_v2/round3_nqds/`

#### 7. Refined go/no-go

GO: QD=4 throughput ≥ 2× QD=1 throughput on this device. Advance to GECP convergence cluster integration.
NO-GO: QD=4 ≈ QD=1 → device achieves peak throughput at QD=1 for sequential reads (some consumer NVMe drives do). This produces the confirmed v2-CF13'' NVMe bandwidth number and is a high-value structural finding even on NO-GO. GECP (U1, PrefetchVirtualMemory) becomes the preferred path if NQDS fails.
GRAY: QD=4 is 1.5–2× better. Follow-up: test QD=8 and QD=16 to find the saturation point. Runtime: +30 min.

#### 8. Risk profile

| Risk | Mitigation |
|---|---|
| Windows StorNVMe driver throttles IOCP requests at QD=1 despite explicit overlapped I/O | Test with both `ReadFileEx` + IOCP and `ReadFile` + `OVERLAPPED` flag; report both |
| ik_llama.cpp uses mmap exclusively; integrating IOCP requires non-trivial refactor | Standalone benchmark first; integration is Stage 5+ concern |
| Interference with GECP (U1): both do layer-ahead prefetch | Measured independently; composition analysis in Stage 5 |

Cheapest falsification: run standalone benchmark. If QD=4 < 1.2× QD=1, kill.

#### 10. Verdict

**REFINE.** M3 NOVEL (no LLM paper uses IOCP for NVMe QD); M1+M2 ADJACENT to CHEOPS '25 (not PRE-EMPTED); no CF9/CF10 issues. 2h standalone test that produces a confirmed CF13'' number regardless of sign. Advance to Stage 4/5.

---

## PATH 3 (FRAME-NOVELTY)

---

### AERO-QW3 — AERO-Style SwiGLU Activation Removal, Zero-SGD

**Score: 12 | Track A | Frame-novelty**

#### 1. Mechanism decomposition

- M1: Replace SiLU(W_gate·x) with identity in trained Qwen3-1.7B (zero-SGD monkeypatch). Measure ΔNLL.
- M2: If ΔNLL ≤ +2.0 nats (weak GO), the bilinear form (W_gate·x) ⊙ (W_up·x) is the surviving computation. Explore Khatri-Rao factorization.
- M3: If ΔNLL ≤ +0.5 nats (strong GO), the nonlinearity is not critical post-training; W_gate and W_up together encode a structured bilinear that admits joint factorization cheaper than 2×full-rank.

#### 2. Per-claim prior-art status

**M1 (zero-SGD activation removal on trained Qwen3):**
Status: **NOVEL** as of 2026-05-09. AERO (arXiv:2410.13060, Oct 2024) covers trained-with-activation-removed models. The zero-SGD question (apply to a trained Qwen3-1.7B without any fine-tuning) is explicitly NOT in AERO. Searches confirm no paper applies zero-SGD activation removal to a trained SwiGLU model.

**M2 (bilinear structure after activation removal):**
Status: **NOVEL** — Khatri-Rao factorization of the resulting bilinear has no published treatment.

**M3 (strong GO threshold):**
Status: likely NO-GO given CF4 (W_gate full-rank), CF6 (gate std 0.91 at layer 27), and CF8. The nonlinearity is load-bearing. Stage 2 estimated P(ΔNLL ≤ +0.5) = 0.20. The weak threshold (ΔNLL ≤ +2.0) is more plausible at ~0.40.

Two-paper composition flag: AERO (arXiv:2410.13060) + Stage 1 CSMF polynomial substitution kill ≈ "replace the activation function." AERO's value-add over CSMF: AERO removes the activation entirely (not substitutes); CSMF tried polynomial substitution and failed. AERO-QW3's value-add over AERO: zero-SGD application to trained SwiGLU with the empirical apparatus (CF4, CF6, CF8) to measure the actual quality hit.

CF9 check: No theorem import. The monkeypatch is a direct empirical test. PASS.
CF10 check: No calibration fit in M1. PASS.

#### 3. Smallest-test sharpening

Script: `scripts/aero_qw3_monkeypatch.py`
Procedure: Monkeypatch SwiGLU forward pass (replace silu(gate_out) with gate_out). Run ΔNLL on 512-token WikiText-2. 25 min. Within 8h.

#### 4. Verdict

**REFINE.** M1 NOVEL (zero-SGD application not in AERO or any paper found). The CF adversarial priors (CF4, CF6, CF8) make this a hard test — a positive result would be structurally surprising and scientifically valuable; a negative result cleanly closes the zero-SGD activation removal route and confirms AERO requires training. 25-min experiment. Advance to Stage 4/5.

---

### F5-HPGO — Head Permutation Group Orbit Collapse

**Score: 12 | Track A | Frame-novelty**

#### 1. Mechanism decomposition

- M1: The attention output is invariant under S_{16} permutation of heads if W_Q, W_K, W_V triples are co-permuted. SGD does not break this symmetry → trained heads live in a random gauge of S_{16}.
- M2: Optimal head alignment π*_L ∈ S_{16} per layer minimizes ||W_Q^{L, π*(h)} - W_Q^{L-1, π*(h)}||_F.
- M3: Aligned inter-layer delta D_aligned has smaller Frobenius norm than unaligned D_naive → more compressible delta.

#### 2. Prior-art status

**M1 (S_{16} gauge freedom in trained heads):**
Status: **ADJACENT** to NeuPerm (arXiv:2510.20367, Nov 2025): "Disrupting Malware Hidden in Neural Network Parameters by Leveraging Permutation Symmetry." NeuPerm applies permutation symmetry to detect/disrupt malware in LLM weights — not for compression. The permutation symmetry property itself is documented. Differentiation: NeuPerm exploits the symmetry for security; HPGO exploits it for inter-layer delta compression.

**M2+M3 (head-alignment before delta compression):**
Status: **NOVEL** — DeltaLLM (arXiv:2501.18596, Jan 2025) explicitly does NOT apply head-alignment before computing inter-layer deltas. Searches for "head alignment permutation before delta compression" returned no match. As of 2026-05-09 this is the distinguishing structural step not in any delta-compression paper.

Additionally: "Linear Predictability of Attention Heads in Large Language Models" (arXiv:2603.13314, Mar 2026) finds heads are linearly predictable from peer heads within a layer. This is ADJACENT — it establishes inter-head correlations within a layer, not cross-layer permutation alignment. The predictability might LOWER the expected benefit of permutation alignment (if heads already converge to a predictable cross-layer order, the random-gauge assumption weakens). The experiment must measure the actual delta-norm reduction rather than assuming the gain.

CF9 check: Hungarian algorithm is a standard combinatorial optimization. No imported theorem with preconditions. PASS.
CF10 check: No calibration fit. PASS.

Two-paper composition flag: DeltaLLM (arXiv:2501.18596) + NeuPerm (arXiv:2510.20367) ≈ "apply permutation symmetry knowledge to layer-delta compression." Value-add: the specific step of using Hungarian matching on W_Q inter-layer cost matrix before delta computation is not in DeltaLLM or NeuPerm. Research, not engineering-integration.

**Important caveat from search results:** The "Linear Predictability" paper (arXiv:2603.13314) reports that across Llama-3.1 8B, Falcon3-10B, OLMo-2-7B, and Qwen3-32B, "just 2–5 reference heads recover many target heads with high fidelity." This is within-layer predictability. Cross-layer is the untested question for HPGO. If heads DO settle into a consistent cross-layer order (which linear predictability weakly suggests), the S_{16} gauge assumption (that heads are in random order) may be partially wrong — and the actual delta-norm reduction from permutation alignment might be smaller than expected. This weakens the proposal's audacity score but does not kill it; the 15-min experiment answers the question directly.

#### 3. Smallest-test

Script: `scripts/hpgo_head_alignment.py`
Procedure: For each layer pair (L-1, L), compute Hungarian optimal assignment on 16×16 cost matrix of ||W_Q^{L,h} - W_Q^{L-1,h'}||_F. Compute ratio ||D_aligned||_F / ||D_naive||_F. Runtime: ~15 min. Within 8h.

#### 4. Verdict

**REFINE.** M2+M3 NOVEL (not in DeltaLLM or NeuPerm or any found paper). The "Linear Predictability" paper is ADJACENT but not PRE-EMPTED for the cross-layer permutation-delta question. Experiment is 15 min and purely weight-manipulation — no forward pass needed. Advance to Stage 4/5 with the caveat that the random-gauge assumption may be partly broken by training (which would still show a positive delta-norm reduction if the alignment is non-trivial).

---

## PATH 4 (ELEGANT-EQUIVALENCE)

Already covered above under A2-RTGF. Additional note: the elegance-class tag `gauge-exploitation` is valid — the Z₂ orbit is a discrete gauge symmetry, and fixing it before quantization is a computation-graph reduction by coordinate choice. Stage 5 should weight this accordingly.

---

## PATH 1 LEADS — STANDARD DEPTH

---

### WAVQ — W_V / W_O Attention Spectrum Cascade

**Score: 11 | Track A | Convergence Cluster 1**

Prior-art: No paper found measuring per-matrix W_V or W_O r_99/d on Qwen3-family. SVD-LLM applies SVD to all attention matrices including W_V/W_O but does not publish the spectrum concentration results in a way that produces the r_99/d value. LatentLLM (arXiv:2505.18413) performs joint Q+K compression but not per-matrix W_V/W_O measurement on GQA models. **NOVEL** measurement claim as of 2026-05-09.

CF9: SVD is always valid. PASS. CF10: no fitting. PASS.

Key relationship to F1-VOFR: WAVQ measures W_V and W_O individually (per-matrix). F1-VOFR measures M_L = Σ_h W_O^h W_V^h (fused). The two experiments are complementary, not redundant — F1-VOFR's M_L can be low-rank even if individual W_V/W_O are full-rank (if head cross-cancellation reduces the sum's rank). Run F1-VOFR first; if M_L is low-rank, WAVQ is superseded. If M_L is full-rank, run WAVQ to see if individual matrices are lower-rank (potential per-head approach). WAVQ as a standalone is the simpler experiment and should run in parallel if budget allows.

Verdict: **REFINE.** NOVEL measurement; clean experiment; structural complement to F1-VOFR.

---

### C5-WUPDOWN — W_up Full-Rank × W_down Spectrum Bridge

**Score: 12 | Track B | Convergence Cluster 2**

Prior-art: No paper tests the inequality r_99(W_down^T)/d_model < r_99(W_up^T)/d_FFN as a prediction. CF8 (weight-rank full) applies to W_gate and W_up but W_down has never been directly tested. The "residual near-constancy → W_down output-space bottleneck" argument (CF2 × CF5) is a novel composition.

CF9: No imported theorem. The argument from CF2 (cos=0.99 residual additivity) to "W_down output space is bottlenecked" is an inference, not a theorem import. CF2 says residual increments are small relative to the stream magnitude — this does not imply W_down is low-rank output-space; it only implies the increments are small vectors. CF8's full-rank finding could apply to W_down too even if increments are small. The mechanism is an empirical prediction, not a derived identity.

This is a **CF9 soft-flag**: the coupling argument is plausible but CF2→W_down-spectral-bottleneck is not a theorem, it is an analogy. The experiment tests it directly; no structural guarantee.

CF10: SVD measurement, no fitting. PASS.

Note: C5-WUPDOWN is adjacent to F4-WAOS but tests a different object: C5 tests r_99(W_down^T) (weight spectrum), while F4-WAOS tests the output span Delta_L = W_down·A_L (activation-weighted output). These are complementary: C5 may find W_down is SPECTRALLY full-rank (as CF8 predicts) while F4-WAOS may still find OPERATIONALLY low-rank output span. C5 provides the "what if F4-WAOS fails" scenario (establishes the CF8 boundary).

Verdict: **REFINE.** NOVEL measurement. 25-min SVD sweep. Complement to F4-WAOS.

---

### C1-QAOC — Q-Activation Outlier Channel Coupling

**Score: 12 | Track B | Convergence Cluster 3**

Prior-art: No paper couples W_Q global K=128 basis to static outlier channel identity via principal-angle measurement. SmoothQuant and OS+ handle activation outliers and weight scaling separately from attention rank reduction. The coupling claim (D_static ∩ colspan(V_Q)) is an unpublished composition of CF11 and CF3.

CF9: Principal-angle measurement is always valid (standard linear algebra). PASS.
CF10: No fitting. PASS.

The primary risk: the W_Q subspace is shaped by token-routing (query-key attention), which may be geometrically orthogonal to the activation outlier directions (which are shaped by W_up·x per CF1). The experiment is 20 min and decisive.

Verdict: **REFINE.** NOVEL coupling claim; 20-min experiment; convergence with F4-WAOS on outlier-channel-as-routing-key primitive.

---

### MHRS — Multi-Head Residency Staircase

**Score: 10 | Track A**

Prior-art: No paper applies depth-stratified W_Q rank reduction (different K per depth tier) to Qwen3 family without retraining. A3 (arXiv:2505.12942) applies attention-logit-error minimization globally. MHRS is zero-SGD with layer-tier hypothesis.

Stage 2 flag (critical): "the staircase direction hypothesis is not yet derived from data (per-layer r_99 unsampled)." The assumption that early layers tolerate lower K may be WRONG — early layers process raw token embeddings and may need MORE query capacity, not less. Per-layer r_99 sweep (C6-SPECDRIFT prerequisite, ~30 min) must run before MHRS to verify the staircase direction. If early r_99 > deep r_99, reverse the staircase.

CF9: SVD is always valid. PASS. CF10: no fitting. PASS.

Verdict: **REFINE** conditionally. Must gate on per-layer r_99 sweep first. Cheapest test: 3-min SVD of W_Q[0] and W_Q[27] to check direction.

---

### RAOK-72 — R2-Aware Outlier-Keyed Activation Codebook @70B

**Score: 11 | Track B**

Prior-art: PrefixQuant (arXiv:2410.05265) decomposes by token type (BOS vs normal); RAOK-72 decomposes by channel tier using the empirically measured K-dependent Jaccard (CF3 / v2-CF1). PrefixQuant is ADJACENT but does not use the three-tier channel partition grounded in per-channel Jaccard stability. **NOVEL** partition key.

**Critical prior-art check**: v2-CF1 confirmed that the K=1% Jaccard structure generalizes across ALL 28 layers (not just the 7 sampled in CF3). This strengthens RAOK-72's foundation — the partition key is layer-universal.

CF9: No theorem import. CF10: partition key is a lookup (not a fitted parameter). PASS.

One technical concern: RAOK-72 applies the CF3 Jaccard profile from RESIDUAL STREAM activations to W_down INPUT activations (post-SwiGLU vectors). These are different objects. The Stage 1 proposal flags this and includes a precondition gate (measure Jaccard on post-SwiGLU vectors directly). This mitigation is sound; the precondition gate belongs in the experiment plan.

Verdict: **REFINE.** NOVEL partition key; ADJACENT to PrefixQuant. Stage 3 gate: measure CF3-Jaccard on post-SwiGLU A_L vectors in the experiment (not residual stream h_L) to verify the tier structure applies at the W_down input.

---

### SOBIB-V — Selective Outlier INT2 + FP16 on W_down

**Score: 11 | Track B**

Prior-art: SpQR (Hessian-weighted outlier detection), SmoothQuant, LLM.int8() use activation-weighted or Hessian-based outlier column selection. SOBIB-V uses CF3 K-dependent Jaccard as the column partition key — a qualitatively different criterion (temporal stability vs sensitivity). **ADJACENT** to SpQR (different criterion, similar mechanism).

CF9: No theorem import. CF10: partition is a lookup (binary threshold on Jaccard), not a fitted parameter. PASS.

Note: SOBIB-V killed in R5-B for "calibration-only sensitivity may underperform Hessian-based; AVX2 scatter overhead." SOBIB-V here is a materially different proposal — the partition key is CF3 Jaccard (not calibration sensitivity), and the smallest test measures ΔNLL vs uniform INT2 directly. The R5-B kill was for a different variant.

Verdict: **REFINE.** ADJACENT to SpQR but partition criterion (CF3 Jaccard) is novel. The comparison vs uniform INT2 and vs Hessian-based selection is the key experimental question.

---

### U1-GECP — GGUF Extent-Coalescence Prefetch

**Score: 12 | Track B | Convergence Cluster 4**

Prior-art: Apple LLM-in-a-Flash uses reactive mmap + demand-paging. PRESERVE (arXiv:2501.08192) prefetches weights in GPU-based distributed serving. No paper uses `PrefetchVirtualMemory` for layer-ahead prefetch in a CPU-only NVMe LLM serving context.

**Important check from Stage 2 handoff**: "check ik_llama.cpp source for existing mmap prefetch flags before assuming the baseline is unoptimized." This is a legitimate pre-experiment check. If ik_llama.cpp already uses `mmap_populate` or `posix_fadvise(SEQUENTIAL)` equivalents on Windows, the baseline may not be QD=1. The Stage 3 recommendation: inspect ik_llama.cpp's mmap call before running the experiment. If it already applies `FILE_FLAG_SEQUENTIAL_SCAN`, the GECP improvement is measured against that baseline.

CF9: `PrefetchVirtualMemory` is a documented Win32 API. No theorem import. PASS.
CF10: No fitting. PASS.

Relationship to NQDS (U6): GECP uses OS hint (`PrefetchVirtualMemory`); NQDS uses application-controlled explicit IOCP. These are at different abstraction levels. NQDS is more aggressive (bypasses VM subsystem); GECP is simpler to integrate. Run NQDS standalone benchmark first to measure the QD improvement; if significant, NQDS + GECP compose as the full layer-prefetch stack.

Verdict: **REFINE.** NOVEL application of `PrefetchVirtualMemory` for LLM weight streaming. 30-min cold-NVMe experiment. Gate on ik_llama.cpp baseline check.

---

## WILDCARD AND STANDARD DEPTH (~300-400w)

---

### F2-RGAS — RMSNorm Gain Absorption Spectrum Shift

**Score: 12 | Track A**

Prior-art: CRANK (killed R2 as SmoothQuant prior art) absorbed scale for INT8 activation quantization. RGAS uses RMSNorm gain absorption to improve W_Q spectral concentration — a different motivation and measurement. No paper measures SVD spectrum before vs after RMSNorm gain absorption. **NOVEL** measurement.

CF9 check: W_Q_gauged = W_Q @ diag(g) is always valid (column scaling of a matrix). No precondition. PASS. Pre-check: std(g_L) per layer — if near-constant, abort (the fold is a no-op). This is the correct gating step.

The absorption is lossless: W_Q h_normed = W_Q_gauged (h / ‖h‖). The spectral-concentration hypothesis rests on whether SGD distributes scale between g and W_Q in a way that inflates W_Q's effective rank. This is an empirical question; 20-min measurement.

Verdict: **REFINE.** NOVEL measurement; 20-min; no CF9/CF10 issues. Value depends on whether g has enough variance to affect W_Q spectrum — the pre-check gates the experiment.

---

### F3-SRSC — Softmax Row-Shift Calibration

**Score: 12 | Track A**

Prior-art: PrefixQuant handles token-wise outliers in Q/K activations (token-dynamic). F3-SRSC handles the STATIC per-head W_Q row mean — a different object (weight-level mean, not token-level outlier). No paper measures W_Q row-mean magnitude as a Q dynamic-range reduction mechanism. **NOVEL** measurement.

CF9 check: Softmax shift invariance softmax(z+c·1) = softmax(z) is an exact identity, no precondition. PASS.

Risk: RoPE rotation may nullify the static W_Q row-mean after application to each token position. This risk is noted in Stage 1. The experiment measures static W_Q row mean (no forward pass needed — 5 min). If ‖μ_row‖ / median(‖W_Q row‖) < 0.05 across all layers, the mechanism is inert.

Verdict: **REFINE.** 5-min measurement. NOVEL. If RoPE nullifies it (GRAY outcome), the follow-up is to check RoPE-rotated Q means per position bucket.

---

### F7-TEGRA — Tied Embedding Gradient-Path Rank Asymmetry

**Score: 11 | Track B | [OUT-OF-ORIENTATION]**

CF12 killed global SVD truncation of the tied embed/lm_head. TEGRA proposes frequency-stratified decomposition where frequent tokens stay full-precision and tail tokens are compressed.

CF12 finding: r_99=1992/2048 — the FULL matrix is full-rank including tail tokens. The Stage 3 question: does the spectral concentration differ between top-50% vs bottom-50% frequency blocks?

Prior-art: No paper exploits BPE token frequency as a spectral concentration predictor for tied embedding matrices. **NOVEL** composition.

CF9 check: SVD on frequency-stratified sub-matrices is always valid. PASS. No theorem precondition.

The Go threshold (r_99/d of bottom-50% ≤ 0.80) requires a measurable spectral difference. Given CF12's flat-spectrum finding (var@r=256=0.28 globally), the tail-frequency block's spectrum is likely ALSO flat — the two-path gradient (input embedding + output projection) trains every token direction. However, the tail-block specifically receives weaker input-gradient (low frequency → rare lookup), so output-projection gradient dominates there — which is Godey & Artzi's gradient-bottleneck argument, applicable to the tail. The structural argument is real; the empirical question is whether the magnitude difference is measurable.

Verdict: **REFINE.** 25-min SVD of two frequency-stratified blocks. NOVEL application of frequency stratification to CF12-kill extension. High probability of NO-GO but structurally informative either way.

---

### C2-LAYERFOLD — Layer-1 Gate Fold × W_Q Head-Redundancy Cascade

**Score: 11 | Track A**

CF6 (36% foldable gates at layer 1) × CF11 (W_Q K=128 global) coupling. Smallest test: 3-min SVD of W_Q[0]. The Stage 2 handoff noted that the coupling hypothesis (layer-1 compression anomaly extends to attention) may be wrong.

CF9: SVD always valid. PASS. CF10: SVD measurement, not fitting. PASS.

Prior-art: "Early-layer pruning" papers remove whole layers; no paper tests within-layer cross-tensor spectrum heterogeneity at layer 1. **NOVEL** coupling test.

Verdict: **REFINE.** 3-min falsification test. If W_Q[0] r_99/d > 0.63 (same as L14), the layer-1 anomaly is MLP-specific (CF6) and does not generalize to attention.

---

### C3-KDJB — K-Dependent Jaccard × W_K Spectrum Bridge

**Score: 11 | Track B**

Spearman ρ(J(t), cos(θ_K(t))) > 0.50. This couples CF3 (K-dependent Jaccard) to CF11 extended to W_K.

CF9 check: principal-angle computation is standard linear algebra. No theorem import. The "per-token outlier Jaccard predicts W_K subspace stability" is an empirical hypothesis, not a theorem. PASS.

Runtime concern: principal-angle computation per token pair is O(d × K²) per layer per token — at d=2048, K=64, 28 layers, 200 prompts × 100 tokens = 20000 token pairs: 2048 × 64² × 28 × 20000 ≈ 5×10^10 FLOPs. This is ~40 min on CPU (plausible). The Stage 1 estimate of "40 min" is likely tight; flag as POSSIBLY TOO LARGE if token count is higher.

With 200 PDAP prompts × mean 100 tokens per prompt = 20K token pairs total: at 28 layers, this is ~40 min. At the border; acceptable.

Verdict: **REFINE.** NOVEL coupling. At the edge of 8h gate but within it. If the runtime proves > 8h, propose smoke variant: 20 prompts × 100 tokens = 2K pairs, ~4 min.

---

### C6-SPECDRIFT — Per-Layer Spectral Drift × K-Tier Codebook Locality [FREE SWING]

**Score: 10 | Track B | Wildcard**

Prerequisite: measure r_99 of W_Q across all 28 layers. This is a free-rider experiment on the per-layer W_Q sweep that multiple proposals (MHRS, C2, HPGO) depend on. The 30-min sweep settles the heterogeneity question.

CF9/CF10: pure measurement. PASS. CF-tether suspended (FREE SWING).

The primary experiment (r_99 variance across layers) is load-bearing for C6, MHRS direction, and C2 simultaneously. It should run as a single shared experiment.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** The per-layer r_99 sweep is the first shared experiment that settles direction for multiple proposals.

---

### F6-ASFA — Attention Sink Fixed-Point Absorption [FREE SWING]

**Score: 9 | Track A | Wildcard**

BOS contribution as a fixed-point bias b_L per layer. CF10 check: 115K params (28×2048×2 bytes), fit from 200 calibration forward passes. n_params = 115K; n_independent_samples = 200 tokens. n_output_dims_per_sample = d_model = 2048. n_samples × d_output = 200 × 2048 = 409K >> 115K. CF10 passes — well-conditioned.

CF9 check: "V_BOS = W_V^h e_BOS is a fixed vector" — this is correct only if BOS is always the same token (position 0 of every sequence). In Qwen3 tokenizer, BOS is always token ID 151643 — fixed vector from the embedding. This precondition HOLDS. PASS.

Stage 1 correctly notes the risk: attention weight assigned to BOS is query-dependent (not fixed). The experiment measures the variance of this weight. If attn_weight_BOS varies by > 50% across calibration tokens, the fixed-bias approximation fails. This is the clean falsification.

Prior-art: StreamingLLM retains sink KV entries; SnapKV manages sink allocation. No paper absorbs sink contribution as a FIXED bias computed at model-prep time. **NOVEL** framing. elegance-class if GO: `no-SGD-reformulation`.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** 30-min experiment; CF10 passes; CF9 PASS.

---

### NTQW — Nested-Tier Q-K-V-O Cascade [FREE SWING]

**Score: 9 | Track B | Wildcard**

Cascade composition of CF11 (Rungs 1+2) + WAVQ/F1-VOFR (Rung 3) + SOBIB-V (Rung 4). The smallest test is Rungs 1+2 simultaneously (CF11 confirmed, 30 min). This is the cheapest rung-independence test.

CF10: no fitting. CF9: no theorem imports. PASS for both.

The rung-independence assumption (NLLs add approximately) is the key unverified claim. Stage 1 estimates +0.98 + 0.29 = +1.27 nats for joint W_Q@K128 + W_K@K512; if actual is < +1.5 nats, rungs are approximately independent. This is testable in 30 min.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.** Priority: run Rung 1+2 independence test before committing to the full cascade.

---

### A4-APPA — Append-Only Persistent KV Accumulator

**Score: 9 | Track A**

K-cache compression at write time using CF11 W_K spectrum (r_99/d ≈ 0.79). The claim is that K-cache vectors projected at rank-512 preserve quality at the K-cache level (not just at the weight level).

CF9 check: the K-cache compression uses the W_K right singular vectors V_K (from weight-matrix SVD). Precondition: K-cache vectors lie approximately in the row-span of W_K. This holds exactly: K_vec = W_K · h, so K_vec lies in the column space of W_K, which is the left singular vectors — not the right. The right singular vectors V_K span the INPUT space of W_K (applied to h). The compression at rank-512 projects K_vec = W_K·h onto the top-512 left singular vectors U_K[:, :512]. This is valid (always possible via SVD). But the quality is governed by how much of the K-cache energy lies in the top-512 left singular directions — which is NOT the same as the weight matrix's right singular value concentration. CF11's W_K at K=512 was +0.29 nats — this was WEIGHT compression (replacing W_K with its rank-512 approximation). K-CACHE compression at rank-512 is different: it projects the runtime K vectors into a 512-dim space. The distortion on the K-vectors depends on the RUNTIME distribution of K_vec = W_K·h, not just on W_K's spectrum.

This is a **soft CF9 flag**: CF11 measured the weight matrix spectrum; A4-APPA applies a K-cache compression whose quality depends on the RUNTIME K-vector spectrum. These are related but not identical. The Stage 1 proposal acknowledges this explicitly ("CF11 measured weights, not runtime K-cache vectors"). The experiment measures K-cache projected error directly.

CF10: no fitting. PASS.

Prior-art: KIVI quantizes K/V in-place; GQA reduces head count. Append-only projection at write-time using CF11 W_K spectrum is NOVEL.

Verdict: **REFINE.** NOVEL. The CF9 soft-flag is addressed by the experiment measuring K-cache error directly. 20-min experiment.

---

### A5-TCRD — Tropical-Coordinate Residual Dispatch

**Score: 10 | Track A**

Top-K neuron cache hit rate as the load-bearing measurement. CF3 K=1% Jaccard=0.308 is the prior; within a generated sequence the Jaccard may be higher (coherent context → more stable outliers).

CF9 check: "tropical algebra" framing is imported but not as a theorem — the actual mechanism is a standard top-K argmax cache, which is always valid. The tropical framing is motivational, not load-bearing. No precondition to verify beyond "top-K cache hit rate is ≥ 0.15." PASS.

Prior-art: Deja Vu (arXiv:2310.17157) uses a trained cross-layer predictor — KILLED for this pipeline. A5-TCRD uses a ZERO-overhead cache (previous token's actual top-K set). The Idea J deferred (Count-Min Sketch on gate activations) was similar but keyed on gate-only firing, which CF1 says fails in deep Qwen3. A5-TCRD keys on W_up firing magnitude — the correct predictor per CF1.

Note: the savings collapse analysis (A_L) requires full W_up GEMV to identify top-K; the savings on W_down rows are conditional on skipping the identification step via caching. The savings are ~30% of W_up GEMV at K=1% Jaccard=0.308 (cache hit fraction). This is a bandwidth win, not a residency reduction.

Verdict: **REFINE.** NOVEL (zero-overhead top-K cache keyed on W_up firing, not gate). 15-min measurement.

---

### A3-NTCA — No-Continuous-Numbers Token Codec [FREE SWING]

**Score: 11 | Track A | Wildcard**

INT8 AVX2 throughput vs BF16 on Zen3 is the load-bearing measurement. The "no-continuous-numbers" constraint is motivational; the actual test is throughput comparison.

Primitive: AVX2 INT8 SIMD on Ryzen 5 7530U Zen3. This is a documented hardware property. CF-tether suspended.

CF9: INT8 GEMV throughput is a hardware measurement. No theorem import. PASS. CF10: PASS.

Smallest test: 2h coding + 30 min benchmarking. Within 8h gate.

Prior-art: No LLM inference paper focuses on INT8 AVX2 throughput on Zen3 specifically. **NOVEL** hardware characterization in the LLM context.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.**

---

### A6-SSCF — Seed-Reconstructible Small-Codebook Factorization

**Score: 11 | Track B | Frame-novelty**

Minimum-description-length framing: W_up = R + ΔW_up where R is a structured random basis. Is ΔW_up low-rank?

CF9 check: CF8 shows W_up is full-rank. ΔW_up = W_up - R where R has different spectral structure. If R is chosen to match the GLOBAL mean spectral profile of W_up (e.g., Marchenko-Pastur distributed), then ΔW_up captures the deviation from the random-matrix baseline — which could be low-rank if W_up's structure is a small perturbation from random. This is the Kolmogorov-residual framing.

The Stage 1 proposal correctly notes: "CF8 shows W_up is full-rank — if R also has flat spectrum, W_up - R is also full-rank." The mitigation (choose a STRUCTURED R with different spectral concentration, e.g., block-diagonal) is sound. The experiment measures whether this holds.

Prior-art: Hypernetworks (Ha et al. 2016) generate large networks from small seeds at training time. Post-training Kolmogorov-residual approach (W_trained = R + Δ with R structured random) has no published treatment. **NOVEL** framing.

CF10: computing rank-256 approximation of ΔW_up is SVD, not parameter fitting. PASS.

Verdict: **REFINE.** NOVEL framing. Very speculative (CF8 strongly suggests flat spectrum for W_up - R unless R is carefully chosen). 30-min single-layer test. Value is in the structural finding either way.

---

### U7-NAWP — NUMA-Aware CCX-Local Weight Partitioning [FREE SWING]

**Score: 9 | Track B | Wildcard**

Q4_K_M scale-separation for hardware stride prefetcher on Zen3.

CF9: AMD Zen3 hardware prefetcher stride behavior is documented. The claim that interleaved Q4_K_M layout defeats the stride prefetcher is verifiable by the experiment. No theorem import beyond "hardware prefetchers engage on regular strides." PASS.

CF10: PASS.

Prior-art: ASPI (killed R5 as "kernel surgery cost prohibitive"). NAWP is a GGUF post-processor, not kernel surgery. **Structurally different from ASPI.** The separated-scales layout is read by existing dequantization code if the format supports it (K8 supertile format already separates scales).

However: this proposal requires modifying the GGUF layout on disk (ik_llama.cpp must still be able to read the modified layout). This is a non-trivial engineering step — the existing ik_llama.cpp dequantization kernels expect the interleaved Q4_K_M layout. The NAWP GGUF post-processor must produce a layout that ik_llama.cpp's dequant kernel handles correctly. This is a runtime-correctness risk, not a theoretical issue.

Smallest test: 1.5h (benchmark only, no ik_llama.cpp modification needed — use a standalone AVX2 GEMV benchmark that simulates both layouts). Within 8h gate.

Verdict: **REFINE — wildcard, CF-tether-suspended, structural floor met.**

---

## HARD GATES (CF13/CF14/CF15)

CF13, CF14, CF15 are unconfirmed in v2. None of the advancers above use these numbers as load-bearing inputs. RAOK-72's 70B arithmetic cites "CF13*" (NVMe sequential 3 GB/s) but marks it unverified — the RAOK-72 smallest test does NOT depend on CF13. U6-NQDS produces the v2-confirmed CF13'' number as its primary output. PASS for all advancers.

---

## VERDICT SUMMARY TABLE

| ID | Score | Verdict | Reason |
|---|---|---|---|
| F1-VOFR | 14 | REFINE | NOVEL M_L=Σ W_O W_V fused operator; PivGa ADJACENT not PRE-EMPTED; 25-min test |
| F4-WAOS | 14 | REFINE | NOVEL output-subspace bypass of CF8; GPTVQ/ASVD ADJACENT (dual problem); 40-min test |
| A2-RTGF | 13 | REFINE | ADJACENT to OS+ (not PRE-EMPTED); M2+M3 NOVEL Z₂ gauge framing; 3-min falsification |
| AERO-QW3 | 12 | REFINE | NOVEL zero-SGD application; AERO paper covers trained-only; 25-min test |
| C5-WUPDOWN | 12 | REFINE | NOVEL W_down spectrum test; CF8-boundary extension; 25-min test |
| C1-QAOC | 12 | REFINE | NOVEL D_static ∩ colspan(V_Q) coupling; 20-min measurement |
| U1-GECP | 12 | REFINE | NOVEL PrefetchVirtualMemory application; check ik_llama.cpp baseline first |
| U6-NQDS | 12 | REFINE | NOVEL IOCP QD saturation; produces CF13'' regardless; 2h standalone |
| F5-HPGO | 12 | REFINE | NOVEL S_{16} head-alignment before delta; DeltaLLM ADJACENT; 15-min test |
| F2-RGAS | 12 | REFINE | NOVEL RMSNorm gain absorption spectral effect; 20-min test |
| F3-SRSC | 12 | REFINE | NOVEL W_Q row-mean magnitude; 5-min measurement |
| WAVQ | 11 | REFINE | NOVEL W_V/W_O spectrum measurement; complement to F1-VOFR |
| RAOK-72 | 11 | REFINE | NOVEL CF3-Jaccard partition key; PrefixQuant ADJACENT; verify Jaccard on A_L not h_L |
| SOBIB-V | 11 | REFINE | ADJACENT to SpQR (partition criterion novel); direct ΔNLL test |
| F7-TEGRA | 11 | REFINE | NOVEL frequency-stratified CF12 extension; likely NO-GO but informative |
| C6-SPECDRIFT | 10 | REFINE (wildcard) | Per-layer r_99 sweep; free-rider; settles direction for MHRS and C2 |
| C3-KDJB | 11 | REFINE | NOVEL Jaccard-W_K coupling; runtime at edge of 8h gate |
| C2-LAYERFOLD | 11 | REFINE | NOVEL cross-tensor layer-1 anomaly test; 3-min falsification |
| F6-ASFA | 9 | REFINE (wildcard) | NOVEL fixed-point sink absorption; CF10 PASS; 30-min |
| MHRS | 10 | REFINE (conditional) | Gate on per-layer r_99 sweep (C6 free-rider) before assuming staircase direction |
| A5-TCRD | 10 | REFINE | NOVEL zero-overhead top-K cache; 15-min |
| A4-APPA | 9 | REFINE | NOVEL; CF9 soft-flag resolved by direct K-cache measurement; 20-min |
| NTQW | 9 | REFINE (wildcard) | Gate on Rung 1+2 independence test (30 min) |
| A3-NTCA | 11 | REFINE (wildcard) | NOVEL INT8 AVX2 throughput measurement; 2.5h |
| A6-SSCF | 11 | REFINE | NOVEL Kolmogorov-residual framing; speculative; 30-min single-layer |
| U7-NAWP | 9 | REFINE (wildcard) | NOVEL Zen3 prefetcher layout; structural floor met |

**KILL count: 0.** No advancer from Stage 2 fails CF9 frame-mismatch (no imported theorem with violated preconditions), CF10 calibration ill-conditioning, or ≥50% PRE-EMPTED load-bearing claims. All 26 advancers advance with REFINE.

**REGENERATE count: 0.**

**DOWNGRADE count: 0** — no pure engineering-integration of obvious cousin pairs identified.

---

## NOTABLE CROSS-ADVANCER DEPENDENCIES

**Shared prerequisite experiments:**
1. Per-layer W_Q r_99 sweep (all 28 layers, ~30 min): required for C6-SPECDRIFT, gates MHRS direction, informative for C2-LAYERFOLD and HPGO. Run once, share results.
2. F1-VOFR SVD of M_L: if M_L is low-rank, WAVQ per-matrix SVD is superseded (though still informative for per-head decomposition). Run F1-VOFR first.
3. U6-NQDS standalone benchmark: produces CF13'' NVMe bandwidth number needed for U1-GECP and RAOK-72 70B throughput arithmetic.

**Prior-art summary for Stage 5:**
- Most critical gaps: OS+ (arXiv:2304.09145) is ADJACENT to A2-RTGF; verify whether OS+ applies residual-stream consistency (cross-layer channel convention) — if yes, RTGF's differentiation weakens. The search did not find direct evidence OS+ does cross-layer consistency enforcement; it appears to apply per-matrix.
- PivGa (arXiv:2512.03062): ADJACENT to F1-VOFR M3 (gauge freedom in low-rank factors) but does not form M_L = Σ_h W_O^h W_V^h. The M1+M2 novelty is intact.

---

*Search date: 2026-05-09. ArXiv papers cited: 2403.07378 (SVD-LLM), 2402.15319 (GPTVQ), 2512.03062 (PivGa/FermiGrad), 2410.13060 (AERO), 2501.18596 (DeltaLLM), 2510.20367 (NeuPerm), 2603.13314 (Linear Predictability of Heads), 2505.18413 (LatentLLM), 2506.04985 (FPTQuant), 2511.10645 (ParoQuant), 2410.05265 (PrefixQuant), 2211.10438 (SmoothQuant), 2304.09145 (OS+), CHEOPS '25 (NVMe I/O characterization), 2501.08192 (PRESERVE).*
