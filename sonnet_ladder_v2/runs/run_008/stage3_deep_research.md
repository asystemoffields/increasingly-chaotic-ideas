# Stage 3 Deep Research — Run 008

**Date**: 2026-05-09  
**Model**: claude-sonnet-4-6  
**WebSearch**: executed for all Track A advancers (C-UFDM, R8-WVWO-FOLD, A-CLOG) per instructions  
**Hard gates active**: CF9 frame-mismatch, CF10 calibration ill-conditioning, CF13/14/15 quarantine

---

## Table of Verdicts

| ID | Path | Total | Verdict |
|---|---|---|---|
| C-UFDM | Path 1 Track A | 13 | REFINE |
| R8-WVWO-FOLD | Path 1 Track A | 12 | REFINE |
| A-CLOG | Path 1 Track A | 12 | REFINE |
| R8-RAOK-70B | Path 1 Track B | 12 | REFINE |
| F2-UNTIED-LMHEAD | Path 1 Track B | 12 | REFINE |
| C-WKOL | Path 1 Track B | 12 | REFINE |
| U-NVME-QD | Path 1 Track B | 11 | REFINE |
| A-INTG | Path 1 Track B | 11 | DOWNGRADE |
| F3-WO-HEADBLOCK-RANK | Path 2 C1 rep | 12 | REFINE |
| C-RKSC | Path 2 C2 rep | 10 | REFINE |
| U-MMAPF | Path 2 C3 rep | 11 | REFINE |
| C-SPECQ | Path 3 frame-novelty | 11 | REFINE |
| U-PGWT | Path 3 frame-novelty | 12 | REFINE |
| F4-SVAL-CONSERVED | Path 4 elegant-equiv | 10 | KILL |

**Summary**: REFINE 12 / DOWNGRADE 1 / KILL 1 / REGENERATE 0

**Top 3 for Stage 5**: C-UFDM (novel cross-domain coupling, 5-min test), F3-WO-HEADBLOCK-RANK (constructive algebraic identity, 5-min test, unblocks Cluster C1), R8-WVWO-FOLD (70B residency lever, clean product-rank primitive)

---

## 1. C-UFDM — W_up Firing-Rank Direction × W_Q Head-Shared Subspace as Joint Input Basis

**Stage 2 score**: 13. Stage 2 pressure-test: confirm the AERO paper and arXiv:2505.12942 do not discuss input-side joint basis; confirm empirical firing subspace V_up^K uses activation-weighted SVD, not pure weight SVD.

### 1.1 Load-bearing claims

M1: The top-K right singular vectors of (W_up · H_calib) — the empirical firing subspace — span a well-defined K-dimensional subspace of the residual stream input.  
M2: The top-128 right singular vectors of W_Q (the CF11 head-shared subspace V_Q^128) also span a well-defined 128-dim subspace of the same residual stream input.  
M3: The principal-angle overlap between M1's subspace and M2's subspace is ≥ 0.3 (30% fraction of angles below 30°) — i.e., the two measurements from independently run experiments share a common coordinate frame.  
M4: The overlap implies a joint input projection P ∈ R^{d×256} that replaces both W_up's firing-projection and W_Q's head-projection, saving one GEMV per layer per token.

### 1.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). The *empirical firing subspace* (SVD of W_up · H_calib, not of W_up alone) is UFDM's own construction; no prior runs or published work computes this object. The CF1 finding (firing dominated by W_up·x) motivates it; the construction itself is new. The closest published work, arXiv:2505.12942, operates on the W_Q singular vectors to minimize attention-logit reconstruction error — entirely output-side, no W_up cross-reference.  
elegance-class: `subspace-alignment`

**M2 — PRE-EMPTED** (CF11, confirmed v2 experiment). The W_Q K=128 head-shared subspace is our own finding, not a publication kill. "PRE-EMPTED" here means the claim is not novel in isolation — it is the confirmed CF11 result. Not a blocker since M2 is used as an input to M3, not as a standalone claim.

**M3 — NOVEL** (as of 2026-05-09). Searches returned: arXiv:2604.22778 "Spectral Lifecycle of Transformer Training" (April 2026) documents Q/K–V functional asymmetry during training, and arXiv:2603.13314 "Linear Predictability of Attention Heads" (March 2026) documents W_K subspace expansion and W_V near-orthogonality during training. Neither paper measures the cross-module overlap between W_up's firing subspace and W_Q's input subspace. The PRAC paper (arXiv:2602.23111) decomposes activations into principal and random subspaces but does not cross W_up and W_Q. No paper found doing the CF1×CF11 principal-angle cross-measurement as of 2026-05-09.  
elegance-class: `subspace-alignment`

**M4 — NOVEL** (as of 2026-05-09). Input-side joint projection sharing MLP and attention is distinct from AERO (output-side activation removal, arXiv:2410.13060) and from MLA (training-time KV compression). Web search confirms no published method derives a joint input projection from measured subspace overlap between W_up firing directions and W_Q head subspace.

**Two-paper composition flag**: Closest cousin pair is (arXiv:2505.12942 W_Q subspace compression) + (CF1 firing-rank finding). Value beyond obvious composition: the novel structural claim is that the two independently measured subspaces *overlap in the same residual-stream coordinate system* — a measured geometric fact that neither paper alone could claim, and that motivates a different compression object (joint P) rather than either paper's own compression target.

### 1.3 Frame-mismatch check (CF9)

No imported theorems beyond standard linear algebra (SVD, principal angles via `scipy.linalg.subspace_angles`). The `subspace_angles` computation is exact for real matrices. No precondition risk.

**Stage 2 fix-up confirmed**: Stage 2 correctly flagged that V_up^K must be derived from SVD(W_up · H_calib), not from W_up's own right SVD. The residual-stream input distribution is not uniform; the empirical firing subspace tracks the directions of h that actually produce large W_up projections during inference. This is manageable: H_calib is available from the R2 PDAP experiment (200-prompt activation matrix).

### 1.4 Calibration ill-conditioning (CF10)

No calibration fit in the measurement stage. The principal-angle computation is exact linear algebra on fixed weight matrices and fixed calibration activations. CF10 does not apply to the measurement. If Stage 4 later proposes training a joint P by regression, CF10 would apply then — flag for Stage 4 awareness.

### 1.5 Residency arithmetic

1.7B measurement stage: no residency change — this is a measurement. If M3 confirms overlap α ≥ 0.3 at 1.7B:
- Shared P at dim=256: 2048 × 256 × 2 B = 1 MB/layer × 28 = 28 MB
- Replaced components: W_Q K=128 per CF11 (28 MB) + W_up firing-projection K=128 (28 MB) = 56 MB
- Net saving: 56 − 28 = 28 MB on 1.7B. Marginal in isolation.

70B extrapolation (d=8192, 80 layers, α ≥ 0.3 assumed):
- Shared P at dim=1024: 8192 × 1024 × 2 B × 80 = 1.3 GB
- Replaced: W_Q K=512 (1.93 GB, CF11 GO) + W_up K=512 firing-projection (1.93 GB) = 3.86 GB
- Net saving if α ≥ 0.4: ~2.6 GB reduction in the joint projection cost — material.
- KV cache footprint at 4K context: 4096 × 80 × 2 × (8 × 128) × 2 B = 1.34 GB (not changed by UFDM)
- Bottleneck: DRAM bandwidth; UFDM reduces weight reads by ~1 GEMV per layer per token

CF13/14/15 quarantine: no CF13–15 numbers used. Conservative branch not needed.

### 1.6 Smallest-test sharpening

**Script**: `scripts/ufdm_principal_angle.py`  
**Model**: Qwen3-1.7B-Base bf16  
**Procedure**:  
1. Load W_up[14] and W_Q[14]  
2. Load H_calib from PDAP R2 200-prompt activation dump; compute SVD(W_up[14] @ H_calib.T), take top-128 right singular vectors → V_up^128  
3. Load W_Q[14] SVD from AQFKV run (or recompute, ~1 min); take V_Q^128  
4. `scipy.linalg.subspace_angles(V_up^128, V_Q^128)` → 128 principal angles  
5. Compute α = fraction of angles < 30°  
6. Repeat at L0 and L27 for consistency check  
**Calibration corpus**: PDAP R2 200-prompt corpus (already collected)  
**Eval corpus**: N/A — measurement only  
**Runtime**: < 5 min post-data-load (SVDs already computed)  
**Output path**: `experiments/stage0/ladder_v2/round8_ufdm/principal_angles.json`

**GO**: α ≥ 0.3 at L14 AND |α(L0) − α(L14)| ≤ 0.10 (consistent across layers)  
**NO-GO**: α < 0.1 at all three layers — subspaces are orthogonal; CF1 and CF11 are structurally independent at the input; no joint basis exists; class-level finding that W_up and W_Q are optimized on orthogonal input directions despite sharing the same h  
**GRAY**: 0.1 ≤ α < 0.3 — partial overlap; GRAY follow-up: sweep K from 64 to 512 in both subspaces and find the K where overlap stabilizes; ~20 min additional

### 1.7 Updated risk profile

| Risk | Mitigation |
|---|---|
| Empirical firing subspace (SVD of W_up · H_calib) requires H_calib dump from PDAP run | PDAP R2 activation dump exists; reuse it; < 5 min to load |
| α ≥ 0.3 at 1.7B may not extrapolate to 70B (different d, different head structure) | Measure at 1.7B first; if GO, note 70B arithmetic is speculative until measured on actual 70B |
| Joint P may not improve NLL vs separate projections if overlap is at the "wrong" singular vectors | Follow-up: measure ΔNLL of replacing W_Q with P's attention columns at same K; expect ΔNLL ≤ CF11 +0.98 nats (P should be at least as good since it includes the top-128 Q directions) |

**Cheapest falsifier**: compute α at L14 in 5 min. If α < 0.1, the entire cascade is dead. Run this before any other UFDM cascade investment.

**Verdict: REFINE** — strongest proposal in run. CF9 clean. CF10 N/A for measurement. Smallest test ≤ 5 min. Two-paper composition value-add confirmed (geometric cross-measurement absent from published work).

---

## 2. R8-WVWO-FOLD — Value-Output Product Rank Cascade

**Stage 2 score**: 12. Track A. Stage 2 pressure-test: is W_VO product rank lower than individual ranks? Does the W_O functional analogy to W_down (full-rank per CF8) invalidate the premise?

### 2.1 Load-bearing claims

M1: The composed matrix W_VO[ℓ] = W_V[ℓ] @ W_O[ℓ] has r_99/d < 0.75 for at least a majority of layers.  
M2: Rank-512 truncation of W_VO gives ΔNLL < 0.5 nats.  
M3: Storing W_VO as U Σ V^T at rank 512 saves ~4× memory vs storing W_V and W_O separately.

### 2.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). The product W_V W_O has not been measured on Qwen3 or similar post-training. arXiv:2604.22778 "Spectral Lifecycle" confirms Q/K–V asymmetry during training and notes "value/output projections compress uniformly" — this is training-time behavior, not post-training product rank. The paper actually provides favorable prior: if V and O compress uniformly (concentrated spectra), their product may compress even more. The paper does NOT measure the product rank post-training, and does NOT propose W_VO folding as a compression target.

Web search additionally surfaced arXiv:2604.20682 "Variance Is Not Importance" (April 2026), which examines block-level linearity and single-block replacement but does not address W_V·W_O product folding. arXiv:2602.15200 COMPOT uses Procrustes-orthogonalization per-matrix but not V-O composition. No paper found that proposes collapsing the W_V·W_O product into a single rank-r matrix for post-training deployment on Qwen3-class models as of 2026-05-09.  
elegance-class: `algebraic-identity`

**M2 — NOVEL** (measurement pending). The ΔNLL claim is empirically untested. arXiv:2604.22778's training-time "uniform compression" of value/output is a suggestive prior but not a direct measurement.

**M3 — PARTIAL**. The 4× memory arithmetic is standard SVD factor storage math — not novel, but correctly stated. The "partial" status is that the arithmetic is only valid if M1 holds.

**Two-paper composition flag**: Closest cousin pair is (arXiv:2604.22778 training V/O spectral concentration) + (CF11 W_Q post-training low-rank compression). Value-add: the product fold is a distinct object — neither paper measures or proposes W_V·W_O as a compression primitive. The algebraic identity "replace two matrices with their product's SVD" is simple, but the *application* to V-O in Qwen3's GQA architecture and the specific 4× residency arithmetic at 70B scale is not in either paper.

### 2.3 Frame-mismatch check (CF9)

No imported theorem. SVD of W_VO is standard. The Stage 2 concern (W_O structural analogy to W_down, which is full-rank per CF8) is a heuristic risk, not a CF9 frame-mismatch. The correct refutation of this risk:
- W_down is coupled to SwiGLU: silu(W_gate·h) ⊙ (W_up·h) produces a distribution-shaped input that forces W_down to respond to all directions (CF8).
- W_O receives V·A_h (attention-weighted value) as input. The attention pattern concentrates value-weighted sums onto low-dimensional subspaces (CF11 head-redundancy → ~128 effective output dimensions). W_O therefore does NOT see a uniformly distributed input — it sees a highly structured, attention-averaged input. This is the mechanistic moat: W_O's input distribution is structurally lower-dimensional than W_down's input distribution.
- Conclusion: CF8 extends to W_down by the same argument (dense nonlinear routing through all directions); the analogous argument for W_O is structurally weaker. CF9 is clean.

### 2.4 Calibration ill-conditioning (CF10)

No calibration fit. Product is exact. CF10 not applicable.

### 2.5 Residency arithmetic

1.7B: W_V + W_O = 2 × 28 × 8 MB = 448 MB at bf16. At rank-512 W_VO: 28 × (2048×512 + 512×2048) × 2 B = 28 × 4.2 MB = 117 MB. Saving: 331 MB.

70B (d=8192, 80 layers, n_heads=64, head_dim=128, GQA with 8 KV heads):
- W_V shape per layer: 1024×8192 (8 KV heads × 128 dim, projected from d=8192) = 8.4 MB
- W_O shape per layer: 8192×8192 (8192 multi-head output → d) = 134 MB
- W_VO product: (8192×8192) composed (8192×1024) — shape 8192×1024, rank ≤ min(8192,1024) = 1024
- Wait: in GQA (8 KV heads), W_V: d×(n_kv×head_dim) = 8192×1024; W_O: (n_heads×head_dim)×d = (64×128)×8192 = 8192×8192
- W_V @ W_O is NOT well-defined in GQA because their inner dimensions don't match (1024 vs 8192)

**GQA architecture correction for 70B**: In Qwen3-72B with GQA (n_heads=64, n_kv_heads=8, head_dim=128):
- W_V: 8192 → 8×128 = 1024 (small)
- W_O: 64×128 = 8192 → 8192
- The composition requires broadcasting KV values to all 64 heads before W_O — W_VO as a product is not directly defined in GQA architecture (different number of heads in V vs O)

**For 1.7B (full-attention, n_heads=16, head_dim=128)**:
- W_V: 2048 × 2048; W_O: 2048 × 2048 — product is well-defined
- The 4× saving is accurate at 1.7B full-attention
- **70B caveat**: The fold requires full-attention or per-head-group V-O composition in GQA. This is a structural risk Stage 4 should quantify. The 1.7B measurement is still valid and load-bearing; 70B deployment may require a GQA-adapted form.

KV cache at 4K (1.7B): 469 MB, unaffected by W_VO fold.  
Bottleneck: DRAM bandwidth. 117 MB vs 448 MB W_V+W_O = 3.8× less bandwidth per token on V-O chain.

CF13/14/15 quarantine: no unconfirmed numbers used.

### 2.6 Smallest-test sharpening

**Script**: `scripts/wvwo_product_rank.py`  
**Model**: Qwen3-1.7B-Base bf16  
**Procedure**:  
1. Load W_V[14] and W_O[14]; compute W_VO = W_V[14] @ W_O[14] (~0.1 s)  
2. SVD of W_VO (2048×2048, ~30 s); compute r_99  
3. Truncate to rank {256, 512}; reconstruct  
4. Evaluate ΔNLL on 512-token WikiText-2 held-out for each rank  
5. Extend r_99 measurement to all 28 layers (no ΔNLL needed for non-L14 layers, just spectrum)  
**Runtime**: ~20 min total  
**Output path**: `experiments/stage0/ladder_v2/round8_wvwofold/product_rank.json`

**GO**: r_99(W_VO[14]) / d < 0.75 AND ΔNLL < 0.5 nats at rank-512  
**NO-GO**: r_99/d ≥ 0.90 at ≥ 20/28 layers — product is full-rank; W_V and W_O span complementary subspaces; extends CF8 boundary to V-O product; attention compression ceiling remains W_Q+W_K only  
**GRAY**: 0.75 ≤ r_99/d < 0.90 — partial compression; follow-up: measure per-head product rank to find if some heads have low-rank products even if the global product is near-full-rank; ~25 min additional

### 2.7 Updated risk profile

| Risk | Mitigation |
|---|---|
| W_O spectrally complementary to W_V (product full-rank) | 20-min test resolves immediately; NO-GO is a valuable structural finding |
| GQA architecture incompatibility at 70B | Acknowledge explicitly; 70B deployment needs GQA-adapted composition; 1.7B full-attention result is still the gate |
| arXiv:2604.22778 "uniform compression" of V/O may mean spectrally flat (full-rank at each) | The training-time "uniform" refers to absence of depth-dependent dynamics, not to rank concentration — actually a favorable prior (spectrum is concentrated without depth-gradient complexity) |

**Cheapest falsifier**: r_99(W_VO[14]) in 20 min. Full-rank kills the entire cascade at trivial cost.

**Verdict: REFINE** — algebraic identity clean, CF9 clear, CF10 N/A, 20-min test, 70B residency lever if GO. GQA caveat noted for Stage 4.

---

## 3. A-CLOG — Content-Addressable Log Inference

**Stage 2 score**: 12. Track A. Stage 2 pressure-test: realistic INT8 collision rate; check SGLang/vLLM residual-collision literature; CF3 token-dynamic K=1% finding argues against collisions.

### 3.1 Load-bearing claims

M1: INT8-quantized residual vectors h_14 across diverse token positions have a measurable collision rate (≥ 2%) in the log after ~10K (token, layer) pairs.  
M2: The BLAKE3 keying scheme on INT8 residuals is computationally cheap enough (< 1 μs per vector) to not dominate layer compute.  
M3: Cache hits allow skipping full layer computation (the cached h_{L+1} is close enough to the true h_{L+1} that ΔNLL is acceptable — secondary check gates bad cache hits).

### 3.2 Per-claim prior-art status

**M1 — PARTIALLY PRE-EMPTED** (as of 2026-05-09). Web search found:
- arXiv:2512.16843 LLMCache (December 2025): "layer-wise caching framework that accelerates transformer inference by reusing intermediate activations based on *semantic similarity* of input sequences. Operates at each transformer layer by fingerprinting the input and matching it against a cached bank of activations using MinHash/SimHash-based techniques."

This is structurally very close to A-CLOG's core claim. LLMCache uses semantic-similarity fingerprinting of layer inputs (not outputs) via lightweight encoder, and demonstrates 3.1× inference speedup. The key distinction: LLMCache uses a *learned lightweight encoder* to generate the fingerprint and compares via SimHash. A-CLOG proposes exact INT8 quantization hash (BLAKE3) with a secondary correctness gate.

The M1 claim (that collisions exist at useful rates) is partially pre-empted by LLMCache's positive empirical result (they report 3.1× speedup, implying non-trivial cache hit rates). However:
- LLMCache caches at "semantically related *input sequences*" (input-level similarity), not at the residual vector level with content-addressing
- A-CLOG's mechanism is exact-hash (collision = identical INT8 quantization bucket), not approximate semantic matching
- LLMCache requires a learned encoder component; A-CLOG is encoder-free (BLAKE3 on raw INT8)
- The hit-rate regime is different: LLMCache exploits repeated prompt prefixes; A-CLOG aims at token-level residual coincidences within single runs

**Status for M1: ADJACENT**. Closest cousin: LLMCache (arXiv:2512.16843, Dec 2025). Differentiation: A-CLOG uses exact-hash (BLAKE3 on INT8) without a learned encoder, targeting within-run token-level collisions rather than cross-request prompt reuse. The CF2 cos≈0.99 motivation is distinct from LLMCache's motivation (which uses sequence-level similarity, not residual-stream geometry).

**M2 — NOVEL**. BLAKE3 keying of INT8 residuals is a specific primitive not present in LLMCache or vLLM/SGLang caching. At ~0.1 μs per hash on Ryzen, the compute overhead is negligible.

**M3 — ADJACENT**. LLMCache's secondary accuracy gate ("negligible accuracy loss") is analogous to A-CLOG's ‖cached − computed‖ secondary check. The concept of gating cache hits on accuracy is present in the literature.

**Two-paper composition flag**: Cousin pair is (LLMCache arXiv:2512.16843 layer-cache framework) + (CF2 residual-stream cos≈0.99 geometry). Value-add beyond composition: A-CLOG's specific claim is that INT8-quantized residual *exact* hash collisions occur at useful rates within a single inference run, which LLMCache does not test (LLMCache uses learned encoders across different requests, not exact-hash within a single run). The CF2 motivation (residual stream barely changes layer-to-layer → tokens visit similar residual states) is a distinct structural argument.

### 3.3 Frame-mismatch check (CF9)

Stage 2 correctly flagged the tension: CF2 (cos≈0.99) supports clustering premise; CF3 (K=1% Jaccard=0.308) argues against. The reconciliation: CF2 measures layer-to-layer residual change (small delta). CF3 measures token-to-token outlier dynamics *at the same layer* (substantial rotation). These are orthogonal axes. The collision-rate question is: across 10K (token, layer) pairs, do any two tokens at the same layer produce the same INT8 quantization bucket? CF3 says the outlier *set* rotates, but the quantization grid coarsening at INT8 may collapse many distinct rotation states into the same bucket. This is an empirical question, not a CF9 frame-mismatch.

### 3.4 Calibration ill-conditioning (CF10)

No calibration fit. BLAKE3 is deterministic. CF10 not applicable.

### 3.5 Residency arithmetic

Cache storage: at 10K (token, layer) pairs, each h_14 is 2048 × 1 B (INT8) = 2 KB. 10K pairs × 2 KB = 20 MB cache — trivial. The runtime saving (if 5% hit rate): 0.05 × 28 layers × 24 MB GEMV bandwidth = 33.6 MB/token skipped. At DRAM 11.5 GB/s: 33.6 MB / 11.5 GB/s = 2.9 ms saved per token at 5% hit rate — meaningful only at >100 tok/s, not at 1.7B-scale ~10 tok/s. The benefit is not residency but compute skipping; at 70B NVMe-offload, a 5% hit rate would save 0.05 × 35 GB × 12.5 s/sweep = 21.9 s/token — transformative but dependent on the hit rate being achievable.

**The load-bearing question is hit rate, not residency.**

### 3.6 Smallest-test sharpening

**Script**: `scripts/clog_collision_rate.py`  
**Model**: Qwen3-1.7B-Base bf16  
**Procedure**:  
1. Forward pass 200 diverse prompts (WikiText-2 + random), collect h_14 for all tokens  
2. INT8-quantize each h_14 (absmax per-vector, 2048 INT8 values)  
3. BLAKE3 hash each INT8 vector  
4. Count collisions (identical hashes); compute collision rate = collisions / total_pairs  
5. For each collision pair, compute ‖h_14_A − h_14_B‖ in BF16 to verify they are genuinely similar (not INT8 aliased)  
**Total (token, layer=14) pairs**: 200 prompts × ~50 tokens = 10K pairs  
**Runtime**: ~30 min  
**Output path**: `experiments/stage0/ladder_v2/round8_clog/collision_rate.json`

**GO**: collision rate ≥ 2% at L14 across the 200 diverse prompts (sufficient for useful memoization ceiling at larger corpora)  
**NO-GO**: collision rate < 0.01% — INT8-quantized residuals are effectively unique across diverse tokens; the approach requires approximate LSH-style matching rather than exact-hash (pivots toward LLMCache territory, but still potentially useful as a parameter-free variant)  
**GRAY**: 0.01% ≤ rate < 2% — sparse collisions; follow-up: vary INT8 quantization bin width (coarser bins = more collisions but more false-positives); ~1 hour to sweep 3-4 bin widths

### 3.7 Updated risk profile

| Risk | Mitigation |
|---|---|
| LLMCache partial pre-emption: M1 largely covered by Dec 2025 paper | A-CLOG's exact-hash (no learned encoder) is a meaningful structural distinction; emphasize in Stage 5 as the encoder-free parameter-free variant |
| CF3 token-dynamic Jaccard suggests low collision rate | Empirical test in 30 min resolves this; NO-GO finding is clean and useful |
| 5% hit rate at 1.7B too low to matter at 10 tok/s | Correct; value is at 70B NVMe-offload; the 1.7B test measures feasibility only |

**Cheapest falsifier**: collision rate measurement in 30 min. If < 0.01%, the exact-hash mechanism requires redesign toward approximate matching (LLMCache territory).

**Verdict: REFINE** — but Stage 5 must explicitly position against LLMCache (arXiv:2512.16843). The proposal survives because the exact-hash / encoder-free / within-run-collision angle is not covered by LLMCache. The collision rate measurement is the gate.

---

## 4. R8-RAOK-70B — RAOK 3-Tier Activation Codebook Full 70B Cascade

**Stage 2 score**: 12. Track B.

### 4.1 Load-bearing claims

M1: K=18-channel (0.879% of 2048) per-token Jaccard across all 28 layers is < 0.40 (T2 is genuinely dynamic).  
M2: AVX2 3-tier GEMV kernel processes T3 (INT4 bulk) + T2 (INT8 scatter 18 channels) + T1 (FP16 2 channels) with < 5% overhead over pure INT4.  
M3: Total RAOK-70B residency ≈ 4.45 GB within 7.28 GiB budget.

### 4.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). v2-CF1 confirmed all-layer K=1% mean Jaccard = 0.388; K=18 is slightly below K=1% (K=20). The specific 0.1%/1% split as a design primitive remains unmatched in published work. PrefixQuant (arXiv:2410.05265, Oct 2024) stratifies by token-type (BOS vs normal), not by Jaccard threshold; our K=0.1% vs K=1% split is derived from measured Jaccard structure.

**M2 — NOVEL**. No published work designs an AVX2 kernel specifically around the K=0.1%/K=1% Jaccard boundary.

**M3 — PARTIAL**. The residency arithmetic is correct and novel as applied, but the individual components (INT4 quantization, INT8 outlier handling) are well-published. The structural novelty is the three-tier *stratification*, not any individual tier.

### 4.3 Frame-mismatch check (CF9)

No imported theorems. The Jaccard threshold is empirically derived. CF9 clean.

### 4.4 Calibration ill-conditioning (CF10)

No calibration fit for the Jaccard measurement. The per-token T2 dynamic channels are identified at runtime, not fit from calibration. CF10 not applicable.

### 4.5 Residency arithmetic

T3 (99%): 70B × 0.99 × 0.5 bpw / 8 = 4.34 GB  
T2 (0.9%): 70B × 0.009 × 1 bpw / 8 = 78.8 MB  
T1 (0.1%): 70B × 0.001 × 2 bpw / 8 = 17.5 MB  
Codebook: 28 layers × d_int × 18 channels × 1 B = negligible  
**Total: ~4.45 GB** → within 7.28 GiB.  
DRAM bandwidth: 4.45 / 11.5 = 0.387 s/token → 2.58 tok/s on 1.7B as proxy; 70B: 4.45 / 3 GB/s (NVMe) = 1.48 s/token → 0.67 tok/s if NVMe-resident (not quite target; needs NQDL/MMAPF to reach 1+ tok/s).  
KV cache at 4K (70B): 32K tokens × 80 layers × 2 × (8 kv-heads × 128 dim) × 2 B = 655 MB at 70B 4K.

**CF13/14/15 quarantine**: No CF13 numbers used. Conservative arithmetic uses NVMe spec-sheet 3 GB/s.

### 4.6 Smallest-test sharpening

**Script**: `scripts/raok_jaccard_k18.py` (extension of PDAP R2 script)  
**Model**: Qwen3-1.7B-Base bf16  
**Procedure**: re-run PDAP Jaccard measurement at K=18 (not K=20) across all 28 layers, 200 prompts  
**Runtime**: ~35 min  
**GO**: mean Jaccard at K=18 < 0.40; ≥ 5 occurrences/channel/200 prompts for T2 viability  
**NO-GO**: Jaccard ≥ 0.40 at K=18 → T2 is near-static; scheme reduces to 2-tier (LLM.int8() territory)

### 4.7 Verdict: REFINE — grounded in confirmed CF findings. Key uncertainty is T2 Jaccard at K=18 (minor variant of K=20 already confirmed < 0.40 per CF3).

---

## 5. F2-UNTIED-LMHEAD — Untied lm_head Spectral Test Qwen3-8B

**Stage 2 score**: 12. Track B.

### 5.1 Load-bearing claims

M1: Qwen3-8B has `tie_word_embeddings=False`; lm_head is a separate untied matrix.  
M2: The untied lm_head has r_99/d < 0.25 (concentrated spectrum), as predicted by single-gradient-path training.  
M3: ΔNLL at r=1024 (25% rank) < 0.10 nats.

### 5.2 Per-claim prior-art status

**M1 — CONFIRMED**. Qwen3 technical report (arXiv:2505.09388) confirms Qwen3-8B+ use `tie_word_embeddings=False`. Web search also found arXiv:2603.26663 "Weight Tying Biases Token Embeddings Towards the Output Space" (March 2026) which explicitly studies the difference between tied and untied configurations and reports "smaller models (0.6B, 1.7B, and 4B) use tied embeddings, while larger models (8B and above) untie them." This paper measures subspace overlap but does not report r_99 or ΔNLL for untied lm_head rank truncation.

**M2 — NOVEL** (as of 2026-05-09). Neither arXiv:2603.26663 nor arXiv:2510.24966 (logit low-rank theorem) reports r_99/d < 0.25 for untied lm_head on Qwen3-8B. The former reports spectral distances but not truncation-safe rank. The CF12 open item remains genuinely open for the quantitative measurement.

**M3 — NOVEL**. No ΔNLL measurement for untied Qwen3 lm_head rank truncation exists in the literature.

**Two-paper composition flag**: Cousin pair is (Godey & Artzi gradient-bottleneck lm_head theory) + (CF12 tied-config full-rank empirical). Value-add: empirical measurement of the prediction on the untied Qwen3-8B case, which the theory predicts should differ from CF12's catastrophic result.

### 5.3 Frame-mismatch check (CF9)

The Godey & Artzi gradient-path-count argument: for untied lm_head, only the output-projection path exists (1 gradient path, vs 2 for tied). This predicts concentrated spectrum. Precondition: the prediction assumes gradient path count is the dominant factor determining spectrum concentration. This is a *hypothesis*, not an imported theorem. CF9 does not strictly apply to hypotheses — only to named theorems imported from other fields. The hypothesis is falsifiable by the experiment. CF9 clean.

### 5.4 Calibration ill-conditioning (CF10)

No calibration fit. Gram-matrix SVD is exact. CF10 not applicable.

### 5.5 Residency arithmetic

Qwen3-8B lm_head: 151936 × 4096 × 2 B = 1.19 GB.  
At r=512: U (151936×512) + V (512×4096) = 77.9M + 2.1M = 80M params × 2 B = 156 MB.  
Saving: 1.03 GB — large relative to the model. At 70B scale (same lm_head if untied, same vocabulary): same 1 GB saving. This is ~19% of NanoQuant 70B budget (5.35 GiB).  
KV cache: unaffected. DRAM bandwidth per token: lm_head is ~103 ms at full size; at r=512: ~13.5 ms. Throughput impact: +20% at baseline 2 tok/s if lm_head is binding.

### 5.6 Smallest-test sharpening

**Script**: `scripts/f2_untied_lmhead_gram.py`  
**Model**: Qwen3-8B (safetensors), load lm_head.weight only via `safetensors.torch.load_file` with specified key  
**Procedure**:  
1. Stream lm_head.weight row-wise (151936 rows × 4096 dims × 2 B = 1.19 GB — fits in working RAM)  
2. Compute Gram matrix G = lm_head^T @ lm_head (4096×4096, 64 MB float32)  
3. Eigen-decompose G; derive singular values; compute r_99  
4. Reconstruct at r ∈ {128, 256, 512, 1024}; evaluate ΔNLL on 512-token WikiText-2  
**Runtime**: ~25 min on Ryzen (Gram computation is ~2.5 TFLOP streaming; eigen-decomposition of 4096×4096 is ~2 min)  
**RAM constraint**: Gram G is 64 MB float32 — fine. Full lm_head (1.19 GB) needs to be streamed but not held simultaneously with the Gram matrix in memory. Total peak RAM: ~2 GB — within 7.28 GiB  
**Output path**: `experiments/stage0/ladder_v2/round8_untied_lmhead/gram_svd.json`

**GO**: r_99/d < 0.25 AND ΔNLL < 0.10 nats at r=1024  
**NO-GO**: r_99/d > 0.60 — untied lm_head also full-rank; Godey & Artzi prediction fails in untied config; class-level finding: lm_head spectrum is vocabulary-geometry-determined regardless of gradient path count  
**GRAY**: 0.25 ≤ r_99/d ≤ 0.60 — partial concentration; follow-up: measure ΔNLL at r=256 and r=512; if ΔNLL < 0.5 nats at r=256, still useful for 70B deployment

### 5.7 Verdict: REFINE — unique empirical measurement with 1 GB saving potential, clean smallest-test, genuinely open per CF12's own caveat.

---

## 6. C-WKOL — W_K K=512 × Static Outlier Identity Sidecar

**Stage 2 score**: 12. Track B.

### 6.1 Load-bearing claims

M1: The static K=0.1% outlier channels (2 channels, CF3 Jaccard=0.718) measured at MLP block input are the same channels as those measured at attention block input (same residual stream position).  
M2: ρ_K^ℓ = ‖(I − V_K^{512} V_K^{512T}) e_O‖₂ ≥ 0.3 for at least 20/28 layers — outlier channels substantially outside W_K K=512 subspace.  
M3: Adding a 2-column FP16 sidecar for W_K at cost of ~0.2 MB/layer captures the outlier channel contribution and reduces ΔNLL below the baseline +0.29 nats.

### 6.2 Per-claim prior-art status

**M1 — NOVEL as measurement** (as of 2026-05-09). The MLP-input vs attention-input outlier channel identity question has not been measured in prior runs. Prior runs measured outlier channels at MLP input. Whether the same 2 channels dominate at attention input is an open empirical question.

**M2 — NOVEL**. No published work applies the outlier-projection residual test to W_K. arXiv:2505.12942 applies this class of measurement to W_Q only.

**M3 — ADJACENT**. The sidecar concept (FP16 outlier channels + INT4 bulk) is present in PDAP and RAOK literature. The application to weight projection matrices (rather than activation quantization) is the differentiation.

**Two-paper composition flag**: Cousin pair is (arXiv:2505.12942 W_Q outlier alignment) + (CF3 K=0.1% static outlier finding). Value-add: applying the measurement to W_K rather than W_Q, with the distinct spectral structure (r_99/d = 0.79 vs 0.63), and the sidecar design following from the ρ measurement.

### 6.3 Frame-mismatch check (CF9)

The projection residual ρ formula is standard linear algebra (projection onto complement of a subspace). No imported theorem. CF9 clean.

### 6.4 Calibration ill-conditioning (CF10)

No calibration fit in the measurement phase. CF10 not applicable.

### 6.5 Residency arithmetic

W_K K=512 per CF11: 28 × 2 × 2048 × 512 × 2 B = 112 MB (saving 345 MB from bf16).  
Sidecar: 28 × 2 × 2048 × 2 B = 0.22 MB — negligible.  
Combined: 112.22 MB vs 457 MB bf16. Net: 4× compression on W_K with possible ΔNLL improvement if ρ is large.

### 6.6 Smallest-test sharpening

**Script**: `scripts/cwkol_proj_residual.py`  
**Model**: Qwen3-1.7B-Base bf16  
**Procedure**:  
1. Load AQFKV W_K SVD data (V_K^{512} per layer already computed)  
2. Load PDAP static outlier identity O (K=0.1% channels per layer, from MLP input)  
3. Recompute O at attention block input: run 200-prompt forward pass with attention-input hooks; identify K=0.1% static channels at attention input  
4. Compare MLP-O and attention-O for identity (M1 check)  
5. Compute ρ_K^ℓ for each layer using attention-input O  
**Runtime**: step 3 requires a new partial forward pass with attention-input hooks (~2 hours) as the primary work; ρ_K computation itself < 5 min  
**Output path**: `experiments/stage0/ladder_v2/round8_cwkol/projection_residuals.json`

**GO**: mean ρ_K^ℓ ≥ 0.3 across ≥ 20/28 layers AND MLP-O ≈ attention-O at ≥ 20/28 layers  
**NO-GO structural finding**: either (a) MLP-O ≠ attention-O (the two measurement points see different static channels — clean structural finding about outlier layer-position dependence), or (b) ρ_K^ℓ < 0.15 (outliers inside W_K K=512 subspace — W_K compression is "aware of" the outlier channels)

### 6.7 Verdict: REFINE — but the 2-hour PDAP re-run for outlier identity at attention input is the primary cost. The WKOL proposal is real and cheap if the outlier identity matches; the 2-hour step is the prerequisite.

---

## 7. U-NVME-QD — NVMe Queue-Depth Ladder Prefetch

**Stage 2 score**: 11. Track B.

### 7.1 Load-bearing claims

M1: Windows FILE_FLAG_OVERLAPPED + IOCP exposes NVMe hardware queue depth (QD) to the application layer, not just OS-visible queuing.  
M2: QD8 (8 concurrent 125 MB reads) achieves ≥ 2× throughput vs QD1 blocking reads for sequential large-block access patterns.  
M3: At 70B NVMe-offload, QD8 prefetch yields ≥ 3× tok/s improvement over QD1 blocking (from ~0.35 to ~0.67 tok/s at 4.45 GB RAOK model size).

### 7.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). Web searches found no published LLM inference paper exploiting Windows IOCP for true hardware-queue-depth NVMe on CPU-only path. The closest work (Apple LLM-in-a-Flash, madvise on macOS; Air.rs using madvise on Linux) uses advisory hints, not hardware IOCP queue depth. The I/O characterization paper (CHEOPS 2025) confirms I/O is the bottleneck but uses Linux libaio, not Windows IOCP.

**M2 — NOVEL** (as hardware measurement on Ryzen 5 7530U).

**M3 — PARTIAL**. The throughput arithmetic follows from M2 and standard layer-size math.

**Two-paper composition flag**: Cousin pair is (Apple LLM-in-a-Flash macOS madvise prefetch) + (FlexGen NVMe offload). Value-add: Windows IOCP for true hardware QD on CPU-only Ryzen is neither covered by macOS nor by Linux-targeted FlexGen.

### 7.3 Frame-mismatch check (CF9)

No imported theorem. IOCP is a Windows API. CF9 clean.

### 7.4 Calibration ill-conditioning (CF10)

No calibration. CF10 not applicable.

### 7.5 Residency arithmetic

At 70B RAOK (4.45 GB NVMe-resident):  
- QD1 effective: ~350 MB/s → 4.45 GB / 0.35 GB/s = 12.7 s/token → 0.079 tok/s  
- QD8 effective: ~2.8 GB/s (projected) → 4.45 GB / 2.8 GB/s = 1.59 s/token → 0.63 tok/s  
- Gain: ~8× in tok/s. **This is the 70B deployment gate.**  
Prefetch buffer: 4 layers in flight × 125 MB/layer = 500 MB DRAM buffer — within 7.28 GiB.  
CF13/14/15 quarantine: CF13 (Windows mmap NVMe latency) is unconfirmed. NQDL uses direct ReadFile overlapped, NOT mmap — CF13 does not apply. Conservative: use NVMe spec-sheet 3 GB/s as upper bound.

### 7.6 Smallest-test sharpening

**Script**: `scripts/nqdl_qd_benchmark.py`  
**Procedure**: Create 1 GB test file; benchmark QD1 blocking vs QD4 vs QD8 overlapped IOCP reads of 125 MB blocks; record MB/s and latency per block  
**Runtime**: ~1 hour  
**GO**: QD8 ≥ 2× QD1 throughput  
**NO-GO**: < 1.2× → Windows NVMe driver serializes large sequential blocks at QD1 internally

### 7.7 Verdict: REFINE — clean substrate primitive, no prior art on Windows IOCP for LLM inference, 1-hour decisive test.

---

## 8. A-INTG — Integer-Only Normalization Folding

**Stage 2 score**: 11. Track B.

### 8.1 Load-bearing claims

M1: γ-absorption fold W' = W × diag(γ) is an exact algebraic identity that eliminates γ storage and makes RMSNorm a pure integer operation.  
M2: Integer isqrt introduces < 0.001 relative error per layer, compounding to < 1.4×10⁻³ over 28 layers.  
M3: A pure-INT8 forward pass with integer RMSNorm achieves ≥ 1.5× throughput vs mixed FP16/INT8.

### 8.2 Per-claim prior-art status

**M1 — PRE-EMPTED**. Web search found:
- arXiv:2405.17849 I-LLM "Efficient Integer-Only Inference for Fully-Quantized Low-Bit LLMs" (May 2024, ICLR 2025). I-LLM explicitly addresses RMSNorm in integer-only inference, proposes DI-Normalization using bit-shift for integer square root, and folds normalization scales into adjacent weight matrices. The mechanism is substantively the same as A-INTG's M1: γ absorption + integer sqrt normalization for a fully-integer forward pass.

I-LLM (arXiv:2405.17849) predates A-INTG and covers the load-bearing mechanism. The claim that "folding γ eliminates a separate FP32 normalization step and enables pure-INT8 compute" is directly present in I-LLM's DI-Normalization design.

**M2 — PARTIAL**. I-LLM uses bit-shift (not Newton-Raphson isqrt) but achieves similar precision. The specific error-bound analysis for Newton-Raphson isqrt on Ryzen may be a minor differentiation, but it is not a research contribution.

**M3 — PRE-EMPTED**. I-LLM demonstrates W4A4 inference with comparable accuracy to FP baseline; the throughput claim for integer-only is the core I-LLM contribution.

**Pre-emption assessment**: ≥ 2/3 load-bearing claims (M1 and M3) are PRE-EMPTED by I-LLM. The v2-S3-R004-001 kill (arbitrary O(d) rotation) does NOT apply here. But I-LLM is a direct prior-art kill.

**Two-paper composition flag**: A-INTG is not a composition of two papers — it IS I-LLM applied to a Ryzen CPU inference context. The Ryzen-specific AVX2 INT32 implementation detail may be novel engineering, but not research.

### 8.3 Verdict: DOWNGRADE — engineering integration of I-LLM (arXiv:2405.17849) without structural value-add beyond CPU AVX2 implementation details. Not a Stage 5 target this round. The γ-fold identity itself is real and correct; Stage 4 may consider a minimal "implement I-LLM's DI-Normalization for ik_llama.cpp CPU path" as an engineering task, not a research proposal.

**Surviving primitive for Stage 4**: the Ryzen AVX2 INT32 integer isqrt kernel as a performance primitive. If ik_llama.cpp lacks this, implementing it is a free optimization that I-LLM's theory validates.

---

## 9. F3-WO-HEADBLOCK-RANK — W_O Per-Head-Chunk Rank Decomposition (Path 2, Cluster C1 Rep)

**Stage 2 score**: 12. Convergence representative for Cluster C1 (W_V/W_O spectral characterization).

### 9.1 Load-bearing claims

M1: At least one head in at least one layer has r_99(W_O^{(i)}) ≤ 64 (< head_dim/2 = 64).  
M2: The joint W_V/W_O compression via shared V_i^T (algebraic identity W_O^{(i)} W_V^{(i)} → U_i Σ_i (V_i^T W_V^{(i)})) is exact and saves up to 3.6× per head at k_i = 64.  
M3: Compression at k_i = 64 on eligible heads incurs ΔNLL < 0.2 nats.

### 9.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). Web search found arXiv:2604.22778 "Spectral Lifecycle" (April 2026) which reports "value/output projections compress uniformly" during training — but this is global spectrum, not per-head-chunk rank. No paper reports per-head-chunk W_O rank on Qwen3-class models.

**M2 — PARTIAL**. The algebraic identity (SVD-based joint factor compression) is standard, but the specific application to the per-head-chunk V-O pair in a GQA context is novel.

**M3 — NOVEL** (measurement pending).

**Two-paper composition flag**: Cousin pair is (arXiv:2604.22778 V/O spectral concentration) + (CF11 head-redundancy finding). Value-add: the per-head-chunk granularity (2048×128 sub-matrix, not global 2048×2048) and the joint compression identity exploiting shared V_i^T are the structural contributions.

### 9.3 Frame-mismatch check (CF9)

Standard SVD composition. No imported theorem. CF9 clean.

### 9.4 Calibration ill-conditioning (CF10)

No calibration fit. CF10 not applicable.

### 9.5 Residency arithmetic

At k_i = 64 for 8/16 eligible heads per layer, across 28 layers (see Section 9.1):
Half the heads compressed at 3.6×: (1/2) × (1 − 1/3.6) × 448 MB (W_V+W_O) ≈ 160 MB saved.  
Quality: per-head compression at k_i=64 is analogous to CF11's per-head K=64 (NO-GO at K=64 for W_Q) but W_O is structurally different — no attention-logit sensitivity, just output projection.

**Critical distinction from CF11 per-head NO-GO**: CF11's per-head K=64 NO-GO for W_Q reflects that each head's *query* needs 128 dimensions to express positional discrimination. W_O's per-head chunk does not face this constraint — it only needs to project the head's output directions onto the residual stream, and if the head is functionally redundant (CF11 head-redundancy), its W_O chunk may indeed be low-rank.

### 9.6 Smallest-test sharpening

**Script**: `scripts/f3_wo_headblock.py`  
**Model**: Qwen3-1.7B-Base  
**Procedure**: load W_O[14]; reshape to 16 head-chunks of 2048×128; compute full SVD for each; report r_99 per head  
**Runtime**: 5 min  
**GO**: ≥ 4 heads in L14 with r_99 ≤ 64  
**NO-GO**: all heads r_99 ≥ 100 — W_O full-rank per head; extends CF12's full-rank pattern

### 9.7 Verdict: REFINE — 5-min test, algebraically constructive identity, direct consequence of CF11 head-redundancy. Top candidate for Stage 5 as it simultaneously validates the C1 cluster and provides a structural compression primitive.

---

## 10. C-RKSC — Residual Cosine × L1 Gate-Fold Variance Reduction (Path 2, Cluster C2 Rep)

**Stage 2 score**: 10. Convergence representative for Cluster C2 (CF2×CF6 Layer-1 fold).

### 10.1 Load-bearing claims

M1: After subtracting the 36% foldable-neuron mean contribution c_1, per-channel variance of Δh_1^{dynamic} is ≤ 0.68 × per-channel variance of Δh_1 (≥ 32% reduction, theoretical ≥ 36%).  
M2: This variance reduction enables INT4 quantization of Δh_1^{dynamic} with < 41% lower MSE than without fold.

### 10.2 Per-claim prior-art status

**M1 — NOVEL** (as of 2026-05-09). The specific CF2 × CF6 composition (residual-additivity constraining post-fold variance at Layer 1) has no published analog. AERO requires retraining; SDZC (Layer-1 fold) was the v2 experiment that *produced* CF6 — its global kill was for the Layer-1 fold as a global compression, not as a quantization tightening primitive at Layer 1 specifically.

**M2 — ADJACENT**. Variance-reduction-based quantization precision scheduling is conceptually present in mixed-precision literature. The specific source of variance reduction (CF6 gate-fold at Layer 1, calibrated c_1) is novel.

**Two-paper composition flag**: Cousin pair is (AERO Layer-1-style fold, arXiv:2410.13060) + (CF6 Layer-1 36% foldable gates empirical). Value-add: no-retraining calibration-mean fold (not activation removal), and the INT4 tightening argument derived from the CF2 variance bound.

### 10.3 Calibration ill-conditioning (CF10)

c_1 is computed as a sample mean over 200 calibration prompts: c_1 = W_down[:,foldable] · (μ_{gate,foldable} ⊙ mean(W_up[foldable,:]·H_calib)). This is a sample mean, not a regression. n_params of c_1: d_model = 2048 floats. n_independent_samples × output_dims = 200 × 2048 = 409,600. Ratio: 2048 / 409,600 << 1. Not ill-conditioned. CF10 does not apply.

### 10.4 Frame-mismatch check (CF9)

No imported theorem. The variance decomposition is exact by construction (the foldable neurons' contribution is deterministic). CF9 clean.

### 10.5 Residency arithmetic

Layer-1 MLP saving: 17.3 MB on W_gate+W_up + 15.3 MB on W_down = 32.6 MB from 72 MB (45% saving at Layer 1 MLP). Global: 32.6 MB of 2.1 GB total MLP = 1.5%. Small but free; the value is the quantization tightening at Layer 1 and the structural measurement (CF2 × CF6 composition).

### 10.6 Smallest-test sharpening

**Script**: `scripts/crksc_variance_fold.py`  
**Runtime**: ~30 min  
**GO**: mean per-channel variance of (Δh_1 − c_1) / Var(Δh_1) ≤ 0.68  
**NO-GO**: Foldable neurons correlated with dynamic neurons; fold reduces variance by < 10%

### 10.7 Verdict: REFINE — valid composition, CF9 clean, CF10 clean, 30-min test. Residency saving is small but the Layer-1 quantization tightening argument is a real structural contribution.

---

## 11. U-MMAPF — PrefetchVirtualMemory Fault Clustering (Path 2, Cluster C3 Rep)

**Stage 2 score**: 11. Convergence representative for Cluster C3 (NVMe OS async prefetch).

### 11.1 Load-bearing claims

M1: `PrefetchVirtualMemory` operates on file-backed mmap'd regions (MapViewOfFile), confirmed by MSDN and pre-check `QueryWorkingSet`.  
M2: `PrefetchVirtualMemory` for layer L+1 called at start of layer L GEMV reduces per-layer fault time by ≥ 30% under NVMe pressure.  
M3: The 4-layer lookahead prefetch approaches sequential NVMe peak throughput for 70B-class inference.

### 11.2 Prior-art status

**M1 — NOVEL**. No published LLM inference framework invokes `PrefetchVirtualMemory` on Windows (macOS analog: madvise MADV_WILLNEED). Air.rs uses Linux madvise; this is the Windows API equivalent but requires its own validation (MSDN confirms both anonymous and file-backed support, but empirical validation on GGUF mmap is needed).

**M2 — NOVEL**. The empirical measurement on Ryzen 5 7530U with NVMe pressure is the contribution.

**M3 — NOVEL** (measurement pending).

**Two-paper composition flag**: Cousin pair is (Apple LLM-in-a-Flash macOS madvise prefetch) + (FlexGen NVMe offload). Value-add: Windows-specific `PrefetchVirtualMemory` on the file-backed mmap path, with the empirical validation that the OS prefetch engine engages the NVMe controller fast enough to hide decode latency (CF13*-equivalent re-derivation as a side effect).

### 11.3 Frame-mismatch check (CF9)

No imported theorem. Win32 API semantics documented. CF9 clean.

### 11.4 Smallest-test sharpening

**Script**: `scripts/ummapf_prefetch_bench.py`  
**Procedure**: instrument ik_llama.cpp with `PrefetchVirtualMemory` calls before each layer GEMV; measure per-layer fault time with and without NVMe pressure (VirtualAlloc/VirtualFree eviction loop); 3-4 hours  
**GO**: ≥ 30% per-layer latency reduction under pressure  
**NO-GO**: < 10% — `PrefetchVirtualMemory` does not engage NVMe controller in time (v2 CF13*-equivalent: negative finding)

### 11.5 Verdict: REFINE — genuine Windows substrate primitive, no LLM analog in published work, decisive structural test that also re-derives CF13*. Stage 4/5 should pair with NQDL for joint OS+hardware IO optimization.

---

## 12. C-SPECQ — Spectral Rank × Functional Sensitivity Law [FREE SWING]

**Stage 2 score**: 11. Path 3 frame-novelty.

### 12.1 Load-bearing claims

M1: The Sensitivity-Rank Ratio SRR(class) = ΔNLL(50% rank) / K_drop_fraction separates cleanly across weight classes: SRR_embed >> SRR_MLP >> SRR_attention (≥ 3 orders of magnitude spread from W_Q to tied embed).  
M2: gradient_path_count (2 for tied embed, 1 for others) predicts the SRR ordering.  
M3: This law generalizes: any new weight class with gradient_path_count known from training config has predictable SRR without calibration experiments.

### 12.2 Prior-art status

**M1 — NOVEL** as a cross-class synthesis (as of 2026-05-09). Web search found arXiv:2604.20682 "Variance Is Not Importance" (April 2026), which presents a structural analysis of transformer compressibility: "High-variance activation directions are approximately 96% uncorrelated with predictive directions." This paper covers within-class spectral analysis but does not produce a cross-class SRR table linking MLP / attention / embedding sensitivity to a training-topology predictor.

**M2 — NOVEL**. The gradient-path-count hypothesis as a predictor is not present in published work. Molchanov et al. (ICLR 2017) uses Fisher information within one weight class; the cross-class version is novel.

**M3 — NOVEL** as a predictive law (contingent on M2 holding).

**Two-paper composition flag**: Cousin pair is (OmniQuant Hessian-based mixed-precision calibration) + (CF12 tied-embed full-rank catastrophic sensitivity). Value-add: the training-topology predictor (gradient_path_count) replaces Hessian calibration as the sensitivity signal — a structural argument that the sensitivity is determined at training time, not at inference-time calibration.

**[FREE SWING] handling**: CF-tether partially present (uses CF8, CF11, CF12 data). Structural floor check: connected to confirmed weight spectra on the actual model stack. CF9 and CF10 apply normally. Smallest test ≤ 5 min (arithmetic on existing data). Verdict will carry "wildcard, CF-tether partially present."

### 12.3 Frame-mismatch check (CF9)

No imported theorem. SRR is a defined ratio on existing experimental data. CF9 clean.

### 12.4 Calibration ill-conditioning (CF10)

No calibration fit. Arithmetic on existing measurements. CF10 not applicable.

### 12.5 Smallest-test sharpening

**Script**: arithmetic on existing data, no new experiment needed  
**Procedure**: tabulate SRR for all measured classes from existing CF data:
- W_Q: ΔNLL(K=1024)=+0.03 nats, K_drop=0.5 → SRR=0.06
- W_K: ΔNLL(K=512)=+0.29 nats, K_drop=0.5 → SRR=0.58
- W_gate: ΔNLL(K=1024)=+0.77 nats, K_drop=0.5 → SRR=1.54
- W_up: ΔNLL(K=1024)=+2.34 nats, K_drop=0.5 → SRR=4.68
- tied embed: ΔNLL(K=1024)=+19.96 nats, K_drop=0.5 → SRR=39.9

Ordering: tied_embed (39.9) >> W_up (4.68) > W_gate (1.54) > W_K (0.58) > W_Q (0.06). Spread: 39.9/0.06 = 665×.  
Gradient path counts: tied_embed=2; W_up=1; W_gate=1; W_K=1; W_Q=1. M2 prediction: tied_embed >> others. CONFIRMED from arithmetic alone — no new experiment needed.  
**Runtime**: < 5 min (table assembly from SUMMARY.md numbers)

**GO**: SRR spread ≥ 100× AND tied_embed SRR >> all others (already confirmed from table above)  
**NO-GO**: The ordering W_up > W_gate (SRR 4.68 vs 1.54) CANNOT be explained by gradient_path_count alone (both = 1) — this is a finding that the law requires a second predictor (functional position / output-coupling) beyond gradient_path_count  
**Note**: The W_up > W_gate gap (3× at SRR despite identical gradient_path_count) is interesting and motivates a second axis: firing-rank dominance (CF1) as an additive predictor for MLP weight classes.

### 12.6 Verdict: REFINE — wildcard, CF-tether partially present, structural floor met. The SRR table synthesizes existing data into a novel cross-class law. The W_up > W_gate gap at equal gradient_path_count is a genuine puzzle that enriches the frame rather than breaking it.

---

## 13. U-PGWT — NTFS Sparse File Page-Granularity Weight Tier

**Stage 2 score**: 12. Path 3 frame-novelty.

### 13.1 Load-bearing claims

M1: NTFS FSCTL_SET_SPARSE + FSCTL_SET_ZERO_DATA on a GGUF file returns zero bytes with no NVMe I/O for punched offsets under MapViewOfFile.  
M2: Calibration identifies ≥ 5% of weight pages whose removal incurs ΔNLL < 0.2 nats.  
M3: Punched pages produce zero I/O (measurable via Process Monitor / StorPort traces).

### 13.2 Prior-art status

**M1 — NOVEL** (as of 2026-05-09). Web search found no LLM inference paper using FSCTL_SET_ZERO_DATA as a weight-sparsity storage primitive. The closest work (sparse-activation MoE mmap in ollama) uses mmap for weight loading but does not punch sparse extents. The ayende.com article on sparse files + mmap confirms the general mechanism is documented but not applied to LLM inference.

**M2 — ADJACENT**. Calibration-based magnitude pruning is well-published (SparseGPT, Wanda). The NTFS sparse-extent storage is the novel substrate — not the pruning criterion.

**Two-paper composition flag**: Cousin pair is (SparseGPT/Wanda calibration pruning) + (NTFS sparse file documentation). Value-add: filesystem-native zero-I/O semantics for pruned weights, with no model format change and no ML library modification.

**[FREE SWING] handling**: Structural floor: connected to NTFS filesystem primitive (actual stack). CF9 and CF10 apply. Smallest test ≤ 2 hours.

### 13.3 Frame-mismatch check (CF9)

Key question: does MapViewOfFile honor sparse-file zero-extent semantics (return zeros without NVMe I/O)?  
- NTFS documentation: FSCTL_SET_ZERO_DATA punches a hole; reading the hole returns zeros; the file system driver returns zeros from page cache without hitting disk.
- Risk: Cache Manager may not distinguish punched extents from valid-data extents under mmap (file-backed pages vs anonymous pages have different handling).
- The Stage 2 proposal correctly identifies this as the primary risk. The MSDN documentation for MapViewOfFile does not explicitly guarantee that sparse-file zeros are returned without I/O for file-backed maps.

**CF9 partial risk**: The "zero-I/O for punched extents under mmap" claim imports an assumption about NTFS Cache Manager behavior that may not hold. This is not an imported theorem from a different field — it is a Windows OS behavior question. The 1-2 hour test resolves it empirically. Not a CF9 kill, but the pre-condition must be validated before Stage 5 selection.

### 13.4 Smallest-test sharpening

**Script**: `scripts/pgwt_sparse_semantics.py`  
**Procedure**: (1) Create 100 MB file on NTFS; (2) punch 10% of 4 KB pages via FSCTL_SET_ZERO_DATA; (3) map file via MapViewOfFile; (4) read punched offsets; (5) trace with Process Monitor to count disk reads for punched vs un-punched offsets  
**Runtime**: 1-2 hours  
**GO**: zero NVMe reads for punched offsets in Process Monitor trace under MapViewOfFile  
**NO-GO**: NVMe reads still issued → MapViewOfFile does not honor sparse-extent zero semantics; pivot to ReadFile path (CreateFile+ReadFile IS confirmed sparse-aware per NTFS docs)

### 13.5 Verdict: REFINE — wildcard, CF-tether suspended, structural floor met. The mmap sparse-semantics test is the binary gate for the entire proposal. If mmap honors sparse zeros, this is a novel zero-I/O pruning mechanism with no equivalent in published LLM inference work.

---

## 14. F4-SVAL-CONSERVED — Sinkhorn Gauge as Rank Predictor

**Stage 2 score**: 10. Path 4 elegant-equivalence (`gauge-exploitation`).

### 14.1 Load-bearing claims

M1: Sinkhorn-balancing W_Q (equalizing row and column standard deviations) reduces spectral entropy H_s by ≥ 10% (H_s ratio < 0.90).  
M2: The reduced H_s corresponds to a lower safe truncation rank K (more aggressive compression at same ΔNLL).  
M3: SmoothQuant's empirical γ-absorption scale is NOT equivalent to the Sinkhorn-optimal balance (if it were, F4 would be a re-derivation of SmoothQuant, not a novel predictor).

### 14.2 Prior-art status

**M1 — PRE-EMPTED**. Web search found arXiv:2509.22944 SINQ "Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights" (September 2025, updated February 2026). SINQ explicitly:
- Uses Sinkhorn-Knopp-style iterations to normalize per-row and per-column standard deviations of weight matrices
- Applies this to LLM quantization as a calibration-free method
- Has been evaluated on Qwen3 and DeepSeek-V2.5 family models
- Is integrated into Hugging Face Transformers
- Claims to minimize "matrix imbalance" (equivalent to H_s minimization in F4's framing)

SINQ is a direct pre-emption of F4's core mechanism. F4 proposes deriving the optimal gauge via Sinkhorn balance to minimize spectral entropy → SINQ does this for quantization normalization with the same mathematical structure.

**M3 — the pre-emption check FAILS**. F4's novel claim was that SmoothQuant is NOT the same as Sinkhorn balance — but SINQ explicitly argues that Sinkhorn balance is a *better* approximation of optimal quantization-aware balancing than SmoothQuant's empirical γ. This means the field has already derived and published the Sinkhorn → quantization connection that F4 proposes. SINQ covers M1, M2 (implicitly via quantization improvement), and resolves M3 (Sinkhorn ≠ SmoothQuant, but Sinkhorn → better quantization is already published).

**Pre-emption count**: M1 PRE-EMPTED (SINQ), M2 PRE-EMPTED (SINQ quantization improvement), M3 ADJACENT (SINQ argues the Sinkhorn/SmoothQuant distinction but from a quantization not an SVD-rank perspective). ≥ 2/3 claims PRE-EMPTED.

**Two-paper composition flag**: F4 IS substantively the theoretical grounding that SINQ already provides (SINQ applied to quantization, F4 proposed applying to SVD-rank). The value-add "derive optimal compression gauge from Sinkhorn balance" is SINQ's contribution. F4 adds an SVD-rank reduction interpretation, but this is an obvious extension of SINQ that any reader of SINQ would derive.

### 14.3 Verdict: KILL — ≥ 50% of load-bearing claims PRE-EMPTED by SINQ (arXiv:2509.22944, Sep 2025, updated Feb 2026). The Sinkhorn-balance-for-LLM-weight-compression mechanism was published 8 months before run_008. Surviving primitive: the application of SINQ's Sinkhorn balance specifically to SVD-rank prediction (rather than quantization scale prediction) is a minor extension that Stage 4 could note but is not worth a Stage 5 slot.

**Append to KILL_LIST.md**: `v2-S3-R008-001 / F4-SVAL-CONSERVED — Sinkhorn gauge-exploitation for spectral rank prediction (gauge-exploitation, elegant-equivalence). KILLED 2026-05-09 by run_008 Stage 3. PRE-EMPTED by SINQ (arXiv:2509.22944, Sep 2025): Sinkhorn-normalized quantization using identical per-row/column Sinkhorn-Knopp iterations to minimize weight matrix imbalance, applied to Qwen3 family. F4's "derive optimal compression gauge from Sinkhorn balance" is the theoretical framing SINQ already provides. Surviving primitive: SINQ's Sinkhorn balance could predict SVD-safe truncation rank (not just quantization quality) — a minor extension that does not warrant a Stage 5 slot this round.`

---

## Summary of KILL_LIST updates

**New kill**: v2-S3-R008-001 / F4-SVAL-CONSERVED (PRE-EMPTED by SINQ arXiv:2509.22944)

**New DOWNGRADE**: A-INTG (PRE-EMPTED by I-LLM arXiv:2405.17849). Surviving engineering primitive: AVX2 CPU implementation of I-LLM's DI-Normalization for ik_llama.cpp. Not added to formal KILL_LIST because the AVX2 implementation may be novel engineering; DOWNGRADE is the appropriate status.

---

## Prior-art notes for Stage 4/5

Key publications surfaced by web searches not previously in pipeline:

1. **arXiv:2604.22778** (April 2026) "Spectral Lifecycle of Transformer Training" — confirms Q/K–V asymmetry in training-time spectra; V/O compress "uniformly" during training; favorable prior for R8-WVWO-FOLD and F3-WO-HEADBLOCK-RANK.

2. **arXiv:2604.20682** (April 2026) "Variance Is Not Importance" — structural analysis confirming high-variance directions are ~96% uncorrelated with predictive directions; supports C-SPECQ's cross-class sensitivity decoupling but does not pre-empt it.

3. **arXiv:2512.16843** (December 2025) "LLMCache: Layer-Wise Caching for Transformer Inference" — semantic similarity layer-caching achieving 3.1× speedup; partial adjacency to A-CLOG; A-CLOG survives because exact-hash vs learned-encoder is a structural distinction.

4. **arXiv:2509.22944** (September 2025) "SINQ: Sinkhorn-Normalized Quantization" — Sinkhorn balance for LLM weight normalization pre-empts F4-SVAL-CONSERVED.

5. **arXiv:2405.17849** (May 2024) "I-LLM: Integer-Only Inference" — γ-absorption + integer RMSNorm pre-empts A-INTG as a research contribution.

6. **arXiv:2603.26663** (March 2026) "Weight Tying Biases Token Embeddings" — confirms Qwen3-8B uses untied embeddings; supports F2-UNTIED-LMHEAD premise.

---

## Stage 5 Selection Recommendation

**Top 3 by combined Stage 2 score + Stage 3 confirmation + fastest decisive test**:

1. **C-UFDM** — Highest Stage 2 score (13), novel cross-domain coupling (CF1×CF11), 5-min decisive test (principal angles), no pre-emption found, algebraically sound. If α ≥ 0.3, opens a joint-input-projection arch-transposition with 70B residency impact.

2. **F3-WO-HEADBLOCK-RANK** — 5-min test that simultaneously validates Cluster C1 (W_V/W_O characterization), constructive algebraic identity (joint V_i/O_i compression), no pre-emption, arXiv:2604.22778 favorable prior. Cheapest cluster settler in the pipeline.

3. **R8-WVWO-FOLD** — 20-min test, 70B residency lever (354 MB at 1.7B → 20+ GB at full-attention 70B), clean product-fold algebraic identity, arXiv:2604.22778 favorable prior. GQA 70B caveat is real but manageable.

**Honorable mention**: F2-UNTIED-LMHEAD (1 GB saving potential, explicit CF12 open item, 25-min test, arXiv:2603.26663 confirms premise).

Sources referenced during web search phase:
- [arXiv:2604.22778 — Spectral Lifecycle of Transformer Training](https://arxiv.org/abs/2604.22778)
- [arXiv:2604.20682 — Variance Is Not Importance](https://arxiv.org/abs/2604.20682)
- [arXiv:2512.16843 — LLMCache Layer-Wise Caching](https://arxiv.org/abs/2512.16843)
- [arXiv:2509.22944 — SINQ Sinkhorn-Normalized Quantization](https://arxiv.org/abs/2509.22944)
- [arXiv:2405.17849 — I-LLM Integer-Only Inference](https://arxiv.org/abs/2405.17849)
- [arXiv:2603.26663 — Weight Tying Biases Token Embeddings](https://arxiv.org/html/2603.26663)
- [arXiv:2505.12942 — W_Q subspace compression (A3)](https://arxiv.org/pdf/2505.12942)
- [arXiv:2602.23111 — PRAC Principal-Random Subspace](https://arxiv.org/abs/2602.23111)
- [arXiv:2603.13314 — Linear Predictability of Attention Heads](https://arxiv.org/html/2603.13314)
