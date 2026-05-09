# Stage 3 — Deep Researcher — Run 005 (v2 Sonnet Ladder)

Completed: 2026-05-09. Researched: 26 advancers across Path 1 (main REFINE pool), Path 2 (convergence clusters), Path 3 (frame-novelty), Path 4 (elegant-equivalence), and substrate cluster. Live searches run on all key novelty claims. Hard gates applied: CF9 (frame-mismatch), CF10 (calibration ill-conditioning), CF13/CF14/CF15 quarantine.

---

## Path 2 — Convergence Cluster Detailed Research (~800–1200w per rep)

### C1: R5-WVWK-MLA / F1-JQKLR — Within-Layer Attention Joint Subspace

**Mechanism decomposition:**
- M1: Within-layer stacked W_K across GQA kv-heads has r_99/(n_kv × d_head) < 0.70, exploitable as a low-rank KV projection.
- M2: Post-hoc calibration-fit W_K_down (d_model → r_K) + W_K_up (r_K → n_kv × d_head) replaces full W_K with r_K-dim KV-cache representation.
- M3: For F1-JQKLR: canonical correlation between top-256 left singular vectors of W_Q and W_K has ≥ 30 angles < 10°, enabling a joint projector P that compresses both simultaneously.
- M4 (WVWK-MLA): KV-cache per token compresses from n_kv × d_head to r_K dims, enabling 32K-context in 7.28 GiB alongside quantized weights.

**Prior-art status:**

*M1 (within-layer kv-head stacked W_K low rank):*
ADJACENT. A3 (arXiv:2505.12942, May 2025, NeurIPS 2025 accepted) does analytical low-rank approximation of the QK component, minimizing pre-softmax attention-score error. It compresses W_Q and W_K jointly as the QK functional pair. A3 addresses the functional-error-minimization framing, not the GQA kv-head stacking spectral measurement. The within-layer GQA-specific stacking (reshape W_K as n_kv × d_head × d_model → (n_kv × d_head) × d_model, then SVD) is the structural primitive A3 does not name — A3 treats the per-head W_K as a single matrix, not a GQA-stacked matrix. Differentiation: R5-WVWK-MLA's claim is that the GQA stacking reveals a joint rank lower than individual kv-heads, which is an empirical question A3 does not answer.

*M2 (post-hoc calibration-fitted KV projection without retraining):*
PRE-EMPTED by two converging papers. **PALU** (arXiv:2407.21118, July 2024, ICLR 2025) decomposes linear KV layers into low-rank matrices, caches compressed intermediate states, and reconstructs on the fly — precisely the M2 mechanism for KV weight matrices. **TransMLA** (arXiv:2502.07864, Feb 2025, NeurIPS 2025 Spotlight) converts GQA-based pre-trained models (LLaMA, Qwen, Mixtral) to MLA-based models post-training with only light fine-tuning (~6B tokens). TransMLA's key finding: "GQA can always be represented by MLA while maintaining the same KV cache overhead" — the algebraic identity M2 relies on is already published. **ReCalKV** (arXiv:2505.24357, May 2025) goes further: Head-wise Similarity-aware Reordering (HSR) clusters structurally similar heads into groups and applies grouped SVD — this is directly analogous to M1's kv-head stacking.

*M3 (canonical correlation between W_Q and W_K for joint projector):*
ADJACENT. A3's QK functional pair is the cousin; A3 minimizes attention-score error analytically (closed-form) rather than measuring canonical angles. The canonical-angle measurement as a structural diagnostic is not in A3; A3 directly solves for the optimal low-rank Q-K pair. The F1-JQKLR first-principles angle-measurement experiment is a structural diagnostic that A3's analytical framework supersedes for the compression objective, but the geometric measurement itself may reveal structure A3's formulation obscures.

*M4 (KV-cache dimension reduction for long context):*
PRE-EMPTED. PALU explicitly targets KV-cache size reduction for long-context inference via low-rank projection, reducing KV cache by 50% with up to 1.89× speedup. The KV-cache residency arithmetic in WVWK-MLA maps exactly onto PALU's result.

**Two-paper composition flag:** PALU + TransMLA ≈ R5-WVWK-MLA + F1-JQKLR. The composition is exact: TransMLA does the GQA→MLA conversion post-training; PALU does the KV low-rank compression post-training. The value-add these proposals offer beyond the PALU+TransMLA composition: (a) measurement of whether the GQA kv-head stacking on Qwen3-1.7B specifically has lower rank than PALU assumes, and (b) the canonical-angle diagnostic (F1-JQKLR) as a compression-agnostic structural measurement. These are incremental engineering/measurement contributions, not structural research claims.

**CF9 frame-mismatch:** M2's calibration fit: n_params = r_K × d_model + n_kv × d_head × r_K = 128 × 2048 + 1024 × 128 = 394K; n_independent_samples × output_dims = 5K tokens × 2048 = 10M. Well-conditioned per CF10. No frame-mismatch issue.

**Verdict: DOWNGRADE (both R5-WVWK-MLA and F1-JQKLR).** M2 (the load-bearing mechanism) is PRE-EMPTED by PALU (2024) and TransMLA (2025). The GQA-specific stacking measurement has ADJACENT status but is not a research contribution beyond PALU/ReCalKV territory. The proposals are engineering-integration of PALU + A3 applied to Qwen3 without structural value-add. Surviving primitive: the per-GQA-stacked-W_K rank measurement (a 15-min diagnostic) is worth running as a structural characterization of Qwen3-1.7B, feeding F5-SECP's calibration-free proxy validation.

---

### C2: C1-ODSP / C6-DSAF — Depth-Stratified Outlier Schedule

**Mechanism decomposition:**
- M1 (ODSP): Per-layer outlier-channel spread_ℓ (fraction of channels covering 90% of outlier events) monotonically increases with depth (2× ratio layers 23-27 vs 2-19). Spearman ρ ≥ 0.6.
- M2 (ODSP): spread_ℓ drives a bpw schedule: 3 bpw for early/mid layers, 4-5 bpw for deep layers.
- M3 (DSAF): spread_ℓ bounds the minimum viable W_Q rank at each layer: K_safe(ℓ) ≥ spread_ℓ × d.
- M4 (DSAF): Per-layer K schedule from CF3's depth gradient improves ΔNLL at fixed total W_Q budget vs uniform K=128.

**Prior-art status:**

*M1 (depth-gradient of outlier-channel spread):*
NOVEL as of 2026-05-09. The CF3 measurement (within-PDAP result, run_002) established this depth gradient. Published outlier quantization papers (SmoothQuant, OS+, DuQuant, PrefixQuant) characterize per-token vs per-channel outliers but do NOT characterize a depth-gradient of outlier spread across layers as a structural property. v2-CF1 confirms K=1% Jaccard across all 28 layers (mean 0.39, range 0.22-0.531). The spread_ℓ as a per-layer scalar derived from the PDAP Jaccard measurement at operational K is not published. `elegance-class: conserved-quantity` (the spread_ℓ depth profile is a measurable model-structural invariant used as the scheduling primitive).

*M2 (spread_ℓ → bpw schedule, ODSP):*
ADJACENT. MDL-based mixed-precision quantization (killed R2/S2) is the closest cousin. ODSP's structural distinction is that the bpw schedule is derived from activation statistics (CF3 Jaccard-spread) rather than weight-Hessian proxies. No paper uses Jaccard-spread as a per-layer bpw scheduling primitive. The differentiation holds.

*M3/M4 (spread_ℓ bounds W_Q rank, DSAF):*
NOVEL as of 2026-05-09. The coupling from activation-space outlier distribution to weight-space compression rank is a new constraint arrow. No paper derives minimum viable attention rank from activation outlier statistics.

**CF9 check:** No imported theorem. The spread_ℓ → bpw/rank coupling is a first-order inequality (the quantization safe-zone width equation). The precondition is just that spread_ℓ is measurable, which v2-CF1 confirms. No frame-mismatch.

**CF10 check:** Neither ODSP nor DSAF involves calibration fitting (the schedule is derived from statistics, not fitted parameters). CF10 safe.

**Two-paper composition flag:** SmoothQuant (per-channel outlier scaling) + MDL-per-layer scheduling ≈ ODSP/DSAF. Value-add: the SCHEDULING PRIMITIVE (spread_ℓ from Jaccard-based outlier-spread measurement) is not in the cousin pair. SmoothQuant does not measure Jaccard spread across depth. MDL uses perplexity-calibrated compression, not activation statistics. The arrow from CF3's depth-gradient empirical measurement to a per-layer schedule is the novel coupling.

**Residency arithmetic:** DSAF increases W_Q storage by ~6 MB at 1.7B vs uniform K=128 (34 MB vs 28 MB) in exchange for lower ΔNLL. At 70B: saves ~800 MB vs uniform-high-K. ODSP at 3.4 bpw average on MLP activations: modest bandwidth reduction at 4K context, meaningful at 32K.

**Smallest test (sharpened):**
- Script: `scripts/depth_spread_schedule.py`. Extend PDAP to all 28 layers; compute spread_ℓ = fraction of channels covering 90% of outlier events per layer at K=1%; compute Spearman ρ vs layer depth. Wall-clock: ≤ 2 hours (PDAP extension run on Qwen3-1.7B-Base, Ryzen 5 7530U).
- GO: Spearman ρ ≥ 0.6 AND spread(layers 23-27)/spread(layers 2-10) ≥ 2.0.
- NO-GO: depth-gradient is a 7-layer sampling artifact.
- GRAY: ρ ∈ [0.3, 0.6) → run the DSAF K-schedule eval (20 min) to test whether the gradient (even if noisy) predicts W_Q compression sensitivity; if ΔNLL gain ≥ 0.2 nats, DSAF advances regardless.
- Output: `experiments/stage0/ladder_v2/round5_dsaf_odsp/`.

**Verdict: REFINE (strongest convergence C2 representative: C6-DSAF).** M1 is NOVEL (spread_ℓ depth gradient), M3/M4 are NOVEL. C1-ODSP REFINE as secondary (same gate experiment, complementary scheduling direction). Both conditional on spread_ℓ depth gradient confirmation. No CF9/CF10 failures.

---

### C3: C5-RKDB / A4-APND — CF2 Residual-Parallelism as Amplitude Primitive

**Mechanism decomposition:**
- M1 (shared): cos(h_L, h_{L+1}) ≈ 0.99 implies ‖Δ_L‖ / ‖h_L‖ ≤ √(2(1−0.99)) ≈ 0.141. The amplitude claim: Δ_L is a small fraction of h_L in L2 norm.
- M2 (RKDB): CF11 W_K K=512 operator-norm reduction (21% vs full BF16 W_K) tightens the recomputation error bound. Storing KV only at even layers and recomputing odd layers from adjacent residual is bounded.
- M3 (APND): Δ_L magnitude < 0.15 × ‖h_L‖ → int16 delta compression of residual-stream log is lossless at 16-bit range.

**Critical pre-flight — CF2 amplitude claim:**
CF2's cos≈0.99 does NOT directly imply small ‖Δ_L‖/‖h_L‖. The geometry: ‖Δ_L‖² = ‖h_{L+1}‖² + ‖h_L‖² − 2⟨h_{L+1}, h_L⟩. If ‖h_L‖ ≈ ‖h_{L+1}‖ = C (norms roughly constant), then ‖Δ_L‖ = C × √(2(1−0.99)) ≈ 0.141 × C. This looks small, BUT: C is not normalized to 1. In Qwen3's residual stream, ‖h_L‖ grows across layers (RMSNorm re-scales before each sub-block but the absolute magnitudes can be large). If ‖h_L‖ ≈ 50 (a plausible scale for a bf16 residual stream with accumulated magnitudes), then ‖Δ_L‖ ≈ 7 in absolute units — not small in absolute terms, only small relative to the norm. For INT16 delta compression, the relevant quantity is the absolute range, not the relative ratio. The GO threshold (< 0.15 relative) is necessary but may not be sufficient — the experiment must report both relative and absolute delta magnitude.

**Prior-art status:**

*M1 (CF2 amplitude claim):*
PARTIAL. DeltaKV (arXiv:2602.08005, Feb 2026) is the closest cousin: it uses inter-token long-range KV similarity (not inter-layer residual similarity) to compute residuals for KV compression. The CF2-based inter-LAYER residual idea is distinct from DeltaKV's inter-TOKEN residual. However, the delta-compression of KV-cache entries using residuals has now been published (DeltaKV achieves 29% of original KV size). The residual-stream inter-layer delta (APND) is different from KV-cache inter-token delta (DeltaKV), but the framing convergence is notable.

*M2 (RKDB KV recompute from adjacent layer with compressed W_K):*
ADJACENT. Layer-Condensed KV Cache (arXiv:2405.10637) reuses KV across layers at training time. RKDB's no-retraining empirical-bound version (using CF2 + CF11 jointly) is not published. The dual-CF composition is novel.

*M3 (APND residual-stream delta logging):*
NOVEL as of 2026-05-09. Log-structured KV with inter-layer delta compression grounded in CF2 has no named entry. DeltaKV's cousin is inter-token, not inter-layer.

**CF9 check (RKDB):** The operator-norm × CF2-delta composition is an algebraic inequality, not an imported theorem with preconditions. No frame-mismatch.

**CF9 check (APND):** CF2 imported as "‖Δ_L‖ is small." The precondition (that cosine~0.99 implies small L2 delta, not just small angle) is UNVERIFIED. This is the critical risk flagged by Stage 2. The experiment must test whether the relative-ratio GO threshold (< 0.15) is achieved.

**Two-paper composition flag (RKDB):** Layer-Condensed KV Cache + CF11 compressed W_K ≈ RKDB. Value-add: the no-retraining empirical bound from CF2 + CF11 operator-norm reduction. This is a structural claim (the bound exists and is tight enough to be practical) that neither paper alone establishes.

**Two-paper composition flag (APND):** DeltaKV + CF2 ≈ APND. Value-add: inter-layer residual-stream delta compression (vs DeltaKV's inter-token KV delta). The append-only constraint forces a design orthogonal to DeltaKV's in-place KV management.

**Smallest test (sharpened):**
- Script: `scripts/cf2_delta_amplitude.py`. Extract h_L, h_{L+1} for 50 tokens × all 28 layers on Qwen3-1.7B-Base. Compute ‖Δ_L‖₂ / ‖h_L‖₂ AND ‖Δ_L‖₂ (absolute). Report both.
- GO: mean relative ratio < 0.15 over all (layer, token) pairs AND 90th percentile absolute magnitude < 5.0 (rough; calibrate against INT16 range ±32767 × scale).
- NO-GO: relative ratio > 0.15 OR absolute magnitude implies INT16 range overflow → closes both RKDB and APND's amplitude-coupling axis.
- GRAY: relative ratio 0.10–0.20 → per-layer adaptive INT16 scale resolves; follow-up 10-min experiment.
- Wall-clock: ≤ 15 min total.
- Output: `experiments/stage0/ladder_v2/round5_cf2_delta/`.

**Verdict: REFINE (A4-APND stronger; C5-RKDB REFINE conditional on delta amplitude).** A4-APND's delta-compression claim is NOVEL (M3) and the inter-layer framing is distinct from DeltaKV. C5-RKDB's dual-CF composition is NOVEL (M2). Both are contingent on the delta-amplitude measurement. The experiment is the cheapest decisive test in the entire run (15 min, binary structural finding). Stage 5 should run this before any other experiment.

---

## Path 1 — Main Pool Individual Advancers (~400–600w each)

### R5-RAOK-FULL

**Mechanism decomposition:**
- M1: K=0.1% static channels (Jaccard=0.718) → T1 FP16 static tier.
- M2: K=0.9% dynamic channels (Jaccard~0.31) → T2 INT8 per-token codebook.
- M3: Remaining 99% → T3 INT4 static.
- M4: AVX2 GEMV processes T3 bulk + T2 INT8 scatter + T1 FP16 residual.

**Prior-art status:** M1+M2+M3 (the 3-tier stratification keyed by CF3 Jaccard split) is NOVEL as of 2026-05-09. Published outlier quantization: LLM.int8() handles only M1 (static pinning at K=0.1%). SmoothQuant redistributes outliers but does not stratify by Jaccard dynamicity. PrefixQuant (arXiv:2410.05265) decomposes into channel-wise and token-wise outliers — adjacent but uses BOS/delimiter token frequency as the stratification criterion, not per-token Jaccard-based dynamicity at K=0.9%. The K=0.1%/0.9% split point grounded in the measured Jaccard cliff is the novel primitive. M4 (AVX2 GEMV with 3 scatter tiers) is NOVEL implementation. `elegance-class: conserved-quantity` (Jaccard-at-K as the stratification invariant).

**CF10 check:** T2 codebook: ≤256 entries, calibration-free (entries derived from per-token activation statistics, not fitted). No CF10 issue.

**Two-paper composition flag:** LLM.int8() + PrefixQuant ≈ RAOK's three-tier. Value-add: the JACCARD-BASED DYNAMICITY CRITERION at K=0.9% is not in either cousin. PrefixQuant uses token-type statistics; RAOK uses the empirical per-token consecutive-pair Jaccard at operational K. This is a structural claim the cousin pair leaves on the table.

**Verdict: REFINE.** Core mechanism NOVEL. The strongest Track B activation compression candidate in the run.

---

### R5-VOWN

**Mechanism decomposition:**
- M1: W_V × W_O product matrix has r_99/d < 0.70 (lower rank than either factor alone).
- M2: Post-hoc SVD of the product W_VO replaces two matmuls with one rank-r matmul.
- M3: 7.3 GB saved at 70B scale (rank-512 product vs two bf16 matrices).

**Prior-art status:**

*M1/M2 (V-O product folding):*
NOVEL as of 2026-05-09. A3 (arXiv:2505.12942) addresses the OV component: "the OV component involving W_O and W_V, minimizing the attention output error." A3's analytical solution for the OV component minimizes a FUNCTIONAL error (attention output error, after multiplying by attention weights), NOT the spectral rank of the product W_V × W_O. A3 decomposes OV as a training-free analytical solution but does not measure whether W_V × W_O is a lower-rank object than its factors. The VOWN claim (SGD drives W_V into a subspace W_O then maps to a smaller subspace) is an empirical prediction that A3's analytical framework neither confirms nor denies. Structural distinctiveness: M1's falsifiable claim (product rank < factor rank, i.e., "cooperating low-rank channel" hypothesis) is independent of A3's error-minimization approach.

*M3 (residency arithmetic at 70B):*
PARTIAL. A3 reports FLOPs and KV-cache reduction but the specific 7.3 GB W_VO weight residency saving at 70B scale is not computed in A3.

**CF9 check:** M1 imports the "AERO analog" framing but the mechanism reduces to "SVD of the product matrix" — no exotic imported theorem, just matrix multiplication + SVD. Precondition (low product rank) is the empirical claim being tested. No frame-mismatch.

**Two-paper composition flag:** A3 + AERO ≈ VOWN. Value-add: VOWN measures whether the W_V × W_O product is lower rank than each factor (a structural claim A3 assumes away analytically), and explicitly derives the 70B residency saving from a post-hoc weight-space compression (not KV-cache compression). The A3 cousin handles functional-error minimization at inference; VOWN targets weight residency reduction at load time.

**Verdict: REFINE.** M1 is NOVEL. The adjacent A3 paper changes the landscape — Stage 5 must position VOWN against A3's OV analytical solution explicitly, but the product-rank empirical question is still open and distinctly valuable.

---

### R5-AWQKV

**Mechanism decomposition:**
- M1: CF11 W_Q/W_K spectra measured. W_V/W_O spectra untested.
- M2: Per-layer global SVD of W_V and W_O; measure r_99/d.
- M3: 4-matrix joint compression at respective K thresholds.

**Prior-art status:** M1/M2: A3 (arXiv:2505.12942) compresses the OV component analytically; PALU compresses W_K/W_V via low-rank projection (ICLR 2025). W_V and W_O spectra on Qwen3-1.7B specifically are not measured in A3 or PALU. The spectrum measurement as a structural diagnostic remains NOVEL for this model family.

However: this proposal is essentially "extend CF11 to W_V and W_O," which A3 and PALU already act on for the compression application. The structural-measurement value-add (what are r_99/d for W_V, W_O?) is real but the compression downstream is ADJACENT to existing work.

**Two-paper composition flag:** A3 + CF11 = AWQKV. Value-add: r_99/d measurement for W_V/W_O on Qwen3 family, informing F5-SECP's calibration-free proxy. Engineering-integration framing for the compression; research value in the spectrum measurement.

**Verdict: DOWNGRADE.** The 4-matrix compression is engineering-integration of A3/PALU applied to Qwen3. The spectrum measurement (r_99/d for W_V/W_O) is worth running as a 15-min diagnostic, primarily to feed SECP and VOWN. Park the compression cascade; advance the diagnostic.

---

### R5-LAYERGATE

**Prior-art status:** CF6 Layer-1 anomaly (36% foldable neurons) is unpublished. No published method exploits the Layer-1 gate-variance anomaly specifically. M1 (Layer-1 fold with ΔNLL < 0.02 nats) is NOVEL. As a standalone compression, it is tiny (169 MB at 70B). As a quality-reallocation cascade rung, it is a free quality contribution. ADJACENT: AERO (arXiv:2410.13060) does activation removal; this is per-neuron gate-fold for a specific Layer-1 anomaly only.

**Verdict: REFINE.** Small but structurally clean. ΔNLL < 0.02 on held-out is a tight gate; if it fails (held-out distribution activates dormant neurons), the CF6 36% figure is calibration-specific and that is itself a structural finding. Not a Stage 5 top-3 pick but advances as a cascade rung.

---

### C2-TESOC

**Prior-art status:**

M1 (K=0.1% static outlier channels → disproportionate logit variance in tied embed): PARTIAL. SpQR (arXiv:2306.03078, ICLR 2024) does mixed-precision quantization with sensitive-weight isolation. SpQR's sensitive-weight identification comes from weight-sensitivity calibration (Hessian proxy), not from activation-outlier channel identity (CF3). SpQR does not specifically analyze the tied embed matrix, and does not use the CF3 K=0.1% outlier channel identity as the precision-management criterion. The TESOC distinction holds: activation-statistics-derived precision priority for the tied embed is not in SpQR.

M2 (INT4 quantization of 2046 of 2048 columns, BF16 for 2 outlier columns): ADJACENT to SpQR's per-weight mixed-precision scheme. The 2-column BF16 sidecar is structurally simpler than SpQR's general sparse-mixed-precision.

**Two-paper composition flag:** SpQR + CF3 ≈ TESOC. Value-add: the ACTIVATION-STATISTICS-DERIVED precision criterion (not Hessian-derived) for the tied embed specifically. The non-obvious coupling (residual-stream outlier channels predict embed-column logit variance) is the load-bearing novel claim.

**CF10 check:** No calibration fitting. The logit-variance measurement is a one-pass statistic. Safe.

**Verdict: REFINE.** M1 (coupling residual-stream outlier identity to embed logit variance) is PARTIAL/NOVEL. The coupling is directly testable in 15 min. Strong Stage 5 candidate.

---

### C3-ZOIA

**Prior-art status:**

M1 (outlier-projection residual ρ_ℓ = ‖(I − P_Q^ℓ P_Q^{ℓT}) e_{O_ℓ}‖₂ ≤ 0.3): NOVEL as of 2026-05-09. No published work measures whether the K=0.1% static outlier channel lies inside the W_Q rank-128 subspace. SmoothQuant and SpQR handle outlier channels via scale migration or mixed-precision; neither measures the projection residual within W_Q's compressed subspace. The "zero INT16 sidecar for W_Q GEMV" claim follows algebraically if ρ_ℓ ≤ 0.3. `elegance-class: subspace-alignment` (outlier channel absorbed by W_Q subspace → no bypass needed).

**CF9 check:** The ρ_ℓ measurement is a projection norm — no imported theorem, standard linear algebra. No frame-mismatch.

**Two-paper composition flag:** CF11 (W_Q K=128 global GO) + CF3 (static outlier channel identity) ≈ ZOIA. Value-add: the projection-residual ρ_ℓ is the structural claim that neither finding alone establishes. Whether the outlier "disappears" inside the compressed subspace is a genuinely new question.

**Verdict: REFINE.** M1 NOVEL. 5-min measurement. High elegance coefficient. Top Stage 5 candidate.

---

### C4-GCQI

**Prior-art status:** M1 (independence interaction term ε at Layer 1 between gate-fold and W_Q compression): NOVEL. No published work measures the additivity of MLP-gate and attention-weight compressions at the same layer. The interaction coefficient ε is load-bearing for any future multi-path compression. `elegance-class: no-SGD-reformulation` marginal (the interaction measurement is empirical, not a closed-form construction). More accurately NOVEL without elegance tag.

**Verdict: REFINE.** Fast (15 min, 3 evals). Structural-discovery value: the interaction term ε is a primitive any future multi-path proposal needs.

---

### F2-HPGF (Path 3 — Frame-Novelty)

**Mechanism decomposition:**
- M1: S_16 permutation invariance of multi-head attention (exact algebraic identity).
- M2: Energy-sorted gauge fixing: sort heads by W_Q head-block Frobenius norm.
- M3: Bottom-M heads (low energy) eliminated without retraining; ΔNLL bounded by their contribution.

**Prior-art status:**

*M1 (S_16 permutation invariance):*
NOVEL as of 2026-05-09. The permutation invariance of multi-head attention under simultaneous reordering of W_Q rows and W_O columns is a known mathematical property (acknowledged in theoretical analyses) but no published paper exploits it as a COMPRESSION COORDINATE CHOICE. The search for "head permutation gauge fixing" returned: learnable permutation for structured sparsity (arXiv:2601.22980) — this applies learned permutations before N:M sparsity masks, a structurally different application (sparsity, not energy-sorted gauge fixing for head elimination). No paper frames the head-permutation group as a gauge freedom for compression.

*M2/M3 (energy-sorted gauge fixing for head elimination):*
NOVEL. Published head-pruning (Michel et al. 2019) uses gradient-based importance. "High-Layer Attention Pruning with Rescaling" (arXiv:2507.01900) prunes heads via activation-level gradients. Neither uses the exact permutation-invariance symmetry as the compression coordinate. The gauge-fixing move is structurally distinct from heuristic pruning.

**CF9 check:** M1's algebraic identity has no preconditions — permutation invariance is exact for any multi-head attention with the same aggregation structure. No frame-mismatch.

**Two-paper composition flag:** Michel et al. 2019 head pruning + CF11 (16 heads span 1-head worth of subspace) ≈ HPGF. Value-add: the EXACT gauge freedom (not a gradient heuristic) as the coordinate choice. CF11 shows that 16 heads are redundant globally; HPGF asks whether the redundancy is concentrated in specific heads after gauge-fixing (which CF11 does not ask).

**Verdict: REFINE (frame-novelty Path 3 Slot 1).** M1 and M2 NOVEL. 5-min experiment. Clean algebraic grounding. Top Stage 5 candidate.

---

### A1-GFRS-2 (Path 3 — Frame-Novelty)

**Mechanism decomposition:**
- M1: Z_2^d residual-stream sign-flip gauge symmetry (exact algebraic identity).
- M2: Gauge-fixing: normalize each channel so max-abs element is positive.
- M3: Post-gauge sign entropy H(sign) < 0.8 bits/weight → sign plane compressible → effective bpw 3.5 instead of 4.0.

**Prior-art status:**

*M1 (Z_2^d gauge symmetry of residual stream):*
NOVEL as of 2026-05-09. The search for "residual stream sign flip Z2 gauge symmetry neural network quantization" returned: Z(2) gauge neural networks from condensed matter physics (theoretical, not LLM compression), and standard quantization surveys. No paper applies the Z_2^d sign-flip symmetry of the transformer residual stream to quantization codebook design. SmoothQuant migrates scales; LLM.int8() pins channels. Neither frames sign-flip invariance as a gauge symmetry exploitable for codebook compression.

*M2/M3 (gauge-fixed sign entropy measurement):*
NOVEL. The claim that after gauge-fixing, sign patterns are structured (H < 0.8 bits/weight) is an empirical prediction with no published analog.

**CF9 check:** M1 is an algebraic identity — no imported theorem preconditions. The Z_2^d symmetry holds for any transformer residual stream with the described weight-sign co-transformation. No frame-mismatch.

**Two-paper composition flag:** LLM.int8() (channel-static outlier pinning) + standard INT4 quantization ≈ A1-GFRS-2. Value-add: the GAUGE-FIXING MOVE (coordinate transformation that makes the sign representation structured) is not in the cousin pair. The cousin pair handles magnitudes; GFRS-2 handles the sign plane as a separate compressible object.

**Verdict: REFINE (frame-novelty Path 3 Slot 2).** M1 NOVEL, M3 empirically conditional. 10-min experiment. Algebraically exact gauge argument. Top Stage 5 candidate.

---

### A6-CNTR (Path 1 — Frame-Novelty, Track B)

**Prior-art status:**

M1 (CF3 K=1% outlier-channel signature predicts top-20% W_up firing rows, precision@20% > 0.50): PARTIAL. CF1 already showed that gate-side predictors fail in deep layers (precision below random in layers 18-27). The Stage 2 critical risk: "content-addressable dispatch may fail for the same reason the row-herding predictor failed." This is the load-bearing risk. The mechanism imports the CF3 outlier-channel as the routing key (not the gate-side predictor), which CF1 distinguishes as structurally different (W_up dominates firing, and the outlier channels are W_up-side inputs). The precision@20% experiment directly tests whether the CF3 routing key works where the CF1 gate-side predictor fails.

Published adjacent: Deja Vu (arXiv:2310.17157) predicts ReLU neuron firing from attention outputs. Content-addressable dispatch keyed by residual-stream activation-outlier signatures is not in Deja Vu (which uses attention outputs, not activation outlier channels). NOVEL frame.

**CF1 risk assessment:** The CF1 finding is that W_up·x dominates firing, NOT that residual-stream outlier channels don't predict firing. These are different questions. The routing key in CNTR is the top-20 outlier channels of the residual stream entering the MLP (not the gate projection). This is a plausible routing key since the same outlier channels drive the W_up GEMV's dominant directions. The experiment is decisive.

**Verdict: REFINE.** Frame NOVEL. The precision@20% experiment (20 min) is the cheapest falsifier. If NO-GO, surviving primitive: wider routing key (K=5%, 100 dims) may still achieve precision > 0.50. Stage 5 should note the CF1 risk explicitly.

---

### F3-AEROT

**Prior-art status:** M1 (per-neuron gate linearizability fraction ≥ 10% in any layer with max residual < 0.05): PARTIAL. AERO (arXiv:2410.13060) does global activation removal as a training step. Per-neuron, post-hoc, calibration-fitted linearization is not in AERO. CF6 (1.5% globally foldable) bounds the near-constant case; AEROT asks whether a larger fraction is linearly approximable (not constant, but linear). Structurally distinct.

**CF9 check:** The linearization ε is a measurement, not a theorem import. No frame-mismatch.

**Verdict: REFINE.** The 30-min experiment is cheap. If only the CF6 1.5% passes (near-dead neurons), AEROT collapses into a slight extension of CF6 with minimal novel value. The GO threshold (≥ 10% linearizable) is achievable only if SwiGLU admits a significant linear regime beyond the dead-zone. This is a genuine structural question.

---

### F5-SECP

**Prior-art status:**

M1 (spectral entropy H_s as calibration-free rank proxy predicting K* within 0.10 nats of CF11): ADJACENT. **CRITICAL find:** arXiv:2604.18085 ("Predicting LLM Compression Degradation from Spectral Statistics", April 2026) analyzes Qwen3 and Gemma3 specifically, using "stable rank" and "information density (bits per parameter)" as predictors of compression degradation from SVD compression. The paper finds that compression ratio × stable rank is a robust predictor (Pearson ρ=0.890 for attention layers). Stable rank = (‖A‖_F / ‖A‖_op)² is closely related to spectral entropy (both are summary statistics of the singular value distribution). The paper covers four low-rank compression methods on Qwen3.

Differentiation: arXiv:2604.18085 uses STABLE RANK (not spectral entropy H_s = −Σ p_i log p_i). These are related but distinct functionals. Stable rank is the Frobenius-to-operator norm ratio squared; spectral entropy is the Shannon entropy of the normalized squared singular values. F5-SECP's specific claim (H_s at τ=0.99 threshold predicts K* within 0.10 nats) is a different claim from the 2604.18085 regression-based predictor. However, the 2604.18085 paper substantially pre-empts the CORE motivation of SECP: "calibration-free spectral statistics predict SVD compression degradation" is now published for Qwen3 specifically.

**Two-paper composition flag:** SVD-LLM + arXiv:2604.18085 ≈ SECP. The 2604.18085 paper covers the Qwen3 attention-layer prediction claim. Value-add of SECP: spectral entropy specifically (vs stable rank), and the threshold-based K* prediction (vs regression-based prediction). These are incremental refinements, not structural value-adds.

**Verdict: DOWNGRADE.** arXiv:2604.18085 (April 2026) substantially pre-empts the calibration-free spectral predictor for Qwen3. The stable rank metric in that paper addresses the same practical need as H_s. SECP's contribution would be a comparison of H_s vs stable rank as predictors — valuable engineering but not pipeline-level research. The W_V/W_O spectrum measurement (a 15-min diagnostic) should proceed as a byproduct of AWQKV/VOWN, not as a standalone SECP cascade.

---

### F6-GFNM [FREE SWING]

**Prior-art status:** M1 (gauge-fixed W_Q_gauged = W_Q @ diag(gamma) has lower spectral entropy than W_Q): PARTIAL. SmoothQuant (arXiv:2211.10438) absorbs per-channel scale from activations into weights for quantization. GFNM applies the same absorption before SVD to test whether spectral concentration increases. The structural question (does γ absorption change W_Q's effective rank?) is not in SmoothQuant. NOVEL measurement.

**CF9 check (wildcard):** CF-tether suspended. Structural floor met: the mechanism is W_Q @ diag(gamma), a documented algebraic operation. No frame-mismatch.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** 10-min experiment. If H_s decreases meaningfully (> 5% reduction in ≥ 10 layers), it motivates gauge-fixing as a pre-processing step before A3/AWQKV compression. If no change, confirms Qwen3's RMSNorm γ has near-unit dynamic range.

---

### A4-APND (Path 2, also Path 1)

Already covered under Convergence C3 above. Verdict: **REFINE.**

---

### R5-ATTNBAND (Path 4 — Elegant Equivalence)

**Mechanism decomposition:**
- M1: Softmax shift-invariance: softmax(x + c·1) = softmax(x). Exact algebraic identity.
- M2: Row-centering S → S − row_mean(S) · 1^T before INT4 quantization of attention scores.
- M3: ΔNLL < 0.10 nats at INT4 centered scores vs FP16.

**Prior-art status:**

*M1 (softmax shift-invariance exploited for centered quantization):*
ADJACENT. The softmax shift-invariance is a well-known algebraic fact documented in FlashAttention implementations (online softmax normalization uses it). QKV projection quantization papers have explored INT8 attention scores. However: the EXPLICIT framing as a gauge freedom exploitable to CENTER attention scores before quantization — as a lossless coordinate change that maximizes INT4 dynamic range — is not found in published work as of 2026-05-09. The "centered INT4 scores" approach distinguishes from INT8 approaches by using the gauge freedom as the justification for centering, not just empirically measuring whether centering helps. `elegance-class: gauge-exploitation` (softmax shift-invariance = gauge freedom; centering = canonical gauge choice).

*M2/M3:*
NOVEL for attention SCORE quantization specifically (as opposed to KV cache quantization or weight quantization). The 30-min experiment directly tests whether the centering matters.

**CF9 check:** M1's algebraic identity is exact, no preconditions. No frame-mismatch.

**Two-paper composition flag:** FlashAttention (implicit online softmax normalization) + INT8 attention (standard quantization papers) ≈ ATTNBAND. Value-add: USING THE GAUGE FREEDOM EXPLICITLY as the coordinate choice for centering, not just empirically applying centering. The framing distinguishes it from "try centering and see if it helps."

**Verdict: REFINE (Path 4 elegant-equivalence).** M1 NOVEL-as-explicitly-framed-gauge-choice. 30-min experiment. Primary payoff at long context / prefill. Elegance argument is auditable and concise.

---

## Substrate Cluster (U1, U2, U3, U5, U6) — ~250–400w each

### U1-NCCG

**Prior-art status:** M1 (NTFS transparent LZNT1 compression on GGUF metadata stratum): NOVEL as of 2026-05-09. No published LLM inference paper uses OS-level transparent file compression on weight files. Apple LLM-in-a-Flash uses DRAM staging, not OS transparent compression. The search returned no LLM inference work specifically targeting Windows NTFS FSCTL_SET_COMPRESSION for GGUF metadata. The wildcard is whether LZNT1 achieves > 2× on fp16 scale values — a pure empirical question.

**Verdict: REFINE.** The 2-hour smallest test directly measures compressibility. If LZNT1 < 1.3×, KILL with empirical rationale. Otherwise, zero model-quality impact, zero user-mode overhead.

---

### U2-SPHF

**Prior-art status:** M1 (PrefetchVirtualMemory for first-token cold-start latency): NOVEL as of 2026-05-09. No published LLM inference work targets the first-token setup window specifically as a prefetch opportunity. The search found no adjacent papers. The mechanism is a standard Win32 API call applied in a novel context.

**Verdict: REFINE.** 1.5-hour experiment. Binary go/no-go (TTFT ≥ 10% reduction). Risk: setup window may already be saturated by sequential reads ik_llama.cpp already issues.

---

### U3-XDFR

**Prior-art status:** M1 (NTFS extent fragmentation → NVMe firmware stride detection): NOVEL. No LLM inference paper discusses NTFS extent fragmentation and NVMe firmware prefetch interaction. The mechanism exploits the NVMe firmware's internal stride detector (documented in NVMe spec), triggered by contiguous sequential access patterns that fragmentations break.

**Verdict: REFINE.** 3-hour experiment, binary gate (≥ 10% throughput difference between 1-extent and 50-extent). Primary risk: DRAM-cached NVMe drives may absorb fragmentation overhead internally.

---

### U5-HPST

**Prior-art status:** M1 (Q4_K_M block-stride split-field layout for Zen 3 stride prefetcher): NOVEL. No published work addresses cache-line stride alignment of quantized weight block layouts for CPU stride prefetcher training. The specific Q4_K_M 20-byte block structure creating non-power-of-two strides is a hardware-level observation.

**Verdict: REFINE.** 4-hour experiment. Binary gate (≥ 5% GEMV throughput improvement). Risk: decode is compute-bound (not load-bound) on DRAM-resident models.

---

### U6-LPMT [FREE SWING]

**Prior-art status:** M1 (MEM_LARGE_PAGES for weight tensor TLB optimization): NOVEL for LLM inference specifically. HPC BLAS uses large pages (MKL internally) but no published LLM inference paper applies this to transformer GEMV specifically.

**CF-tether:** Suspended. Structural floor: MEM_LARGE_PAGES is a documented Win32 primitive. Gate: SeLockMemoryPrivilege may be unavailable on Windows 11 Home.

**Verdict: REFINE — wildcard, CF-tether-suspended, structural floor met.** 3-hour experiment. Gain likely small at 1.7B (0.3%); structural finding re-tests at 70B where TLB thrash is larger.

---

## KILL / DOWNGRADE Consolidation

**KILL — none from the advancing pool.** No advancer fails CF9 outright (frame-mismatch with unverified imported theorem preconditions) or CF10 unmitigated. The kills from this stage are DOWNGRADES.

**DOWNGRADE:**
- **R5-AWQKV**: Engineering-integration of A3 + PALU applied to Qwen3. The 4-matrix compression cascade is ADJACENT to published work. Preserve the W_V/W_O spectrum measurement as a 15-min diagnostic feeding VOWN and SECP.
- **R5-WVWK-MLA / F1-JQKLR (Convergence C1 reps)**: M2 (post-hoc calibration-fitted KV projection) PRE-EMPTED by PALU (ICLR 2025) and TransMLA (NeurIPS 2025). The GQA-stacking rank measurement is ADJACENT to ReCalKV (May 2025). Engineering-integration of PALU + A3. Surviving primitive: the GQA stacked-W_K r_99 measurement feeds VOWN's product-rank diagnostic.
- **F5-SECP**: Core motivation PRE-EMPTED by arXiv:2604.18085 (April 2026, Qwen3-specific). Downgrade; the H_s vs stable-rank comparison is incremental.

---

## Two-Paper Composition Summary (All Advancing Ideas)

| ID | Cousin Pair | Value-Add |
|---|---|---|
| R5-RAOK-FULL | LLM.int8() + PrefixQuant | K=0.9% Jaccard-dynamicity split point (not in either cousin) |
| R5-VOWN | A3 (OV) + AERO | W_VO product rank < factor rank (empirical prediction A3 doesn't make) |
| R5-LAYERGATE | AERO + SDZC | Layer-1 anomaly-specific fold (global schemes killed; this is isolated) |
| R5-ATTNBAND | FlashAttention + INT8 attn | Explicit gauge-freedom framing for centering (not just empirical centering) |
| C1-ODSP | SmoothQuant + MDL | Jaccard-spread depth gradient as bpw-scheduling primitive (not Hessian) |
| C2-TESOC | SpQR + CF3 | Activation-statistics-derived precision priority for tied embed columns |
| C3-ZOIA | CF11 + CF3 | Projection residual ρ_ℓ: whether outlier is absorbed by W_Q subspace |
| C4-GCQI | CF6 + CF11 | Independence interaction term ε (needed for any multi-path cascade) |
| C5-RKDB | Layer-Condensed KV + CF11 | No-retraining empirical bound from CF2 + CF11 operator-norm |
| C6-DSAF | SmoothQuant + MDL | Depth-stratified rank schedule from activation spread (not Hessian) |
| F2-HPGF | Michel et al. head pruning + CF11 | Exact permutation-gauge freedom (not gradient heuristic) |
| F3-AEROT | AERO + CF6 | Per-neuron post-hoc linearization subset (not global activation removal) |
| F6-GFNM | SmoothQuant + F5-SECP | Gauge absorption as SVD pre-processing (not quantization pre-processing) |
| A1-GFRS-2 | LLM.int8() + INT4 quant | Sign-flip gauge freedom → structured sign plane |
| A4-APND | DeltaKV + CF2 | Inter-layer residual delta (vs inter-token KV delta in DeltaKV) |
| A6-CNTR | Deja Vu + CF3 | CF3 outlier-channel routing key for W_up dispatch (not attention-output predictor) |
| U1-NCCG | Apple LLM-in-Flash + GGUF format | OS-transparent LZNT1 on metadata stratum (no user-mode overhead) |
| U2-SPHF | Apple LLM-in-Flash + Win32 | First-token cold-start setup-window prefetch (not steady-state prefetch) |
| U3-XDFR | NVMe sequential access + GGUF | NVMe firmware stride detector activation via extent defrag |
| U5-HPST | Q4_K_M format + Zen3 microarch | Split-field layout for stride-prefetcher training (no prior analog) |
| U6-LPMT | HPC BLAS large-pages + llama.cpp | Weight-tensor TLB optimization for transformer GEMV specifically |

---

## Verdict Summary Table

| ID | Verdict | Primary Status | Key Risk |
|---|---|---|---|
| R5-RAOK-FULL | REFINE | M1/M2 NOVEL | AVX2 T2 scatter overhead |
| R5-VOWN | REFINE | M1 NOVEL | W_VO product may be full-rank |
| R5-WVWK-MLA | DOWNGRADE | M2 PRE-EMPTED (PALU, TransMLA) | — |
| R5-LAYERGATE | REFINE | M1 NOVEL | Held-out generalization of 36% figure |
| R5-ATTNBAND | REFINE | M1 NOVEL (gauge framing) | Score heavy tails at INT4 |
| R5-AWQKV | DOWNGRADE | ADJACENT (A3, PALU) | — |
| C1-ODSP | REFINE | M1 NOVEL | Depth gradient may be sampling artifact |
| C2-TESOC | REFINE | M1 PARTIAL/NOVEL | Outlier channel ≠ embed column logit contribution |
| C3-ZOIA | REFINE | M1 NOVEL | Outlier outside W_Q subspace (ρ > 0.3) |
| C4-GCQI | REFINE | M1 NOVEL | Interaction term ε dominated by residual coupling |
| C5-RKDB | REFINE | M2 NOVEL | Δ_L amplitude large despite cosine~0.99 |
| C6-DSAF | REFINE | M1/M3 NOVEL | Depth gradient sampling artifact (shares gate with ODSP) |
| F1-JQKLR | DOWNGRADE | M2 ADJACENT (A3, PALU) | — |
| F2-HPGF | REFINE | M1/M2 NOVEL | Head energy equidistributed |
| F3-AEROT | REFINE | M1 PARTIAL | Linearizable fraction < 10% |
| F5-SECP | DOWNGRADE | PRE-EMPTED (arXiv:2604.18085) | — |
| F6-GFNM | REFINE (wildcard) | M1 NOVEL | γ near-unit dynamic range |
| A1-GFRS-2 | REFINE | M1 NOVEL | H(sign) remains ~1.0 bit post-gauge |
| A4-APND | REFINE | M3 NOVEL | CF2 cosine ≠ small L2 delta |
| A6-CNTR | REFINE | M1 NOVEL | CF1 precedent: W_up firing hard to predict |
| U1-NCCG | REFINE | M1 NOVEL | LZNT1 < 1.3× on fp16 scales |
| U2-SPHF | REFINE | M1 NOVEL | Setup window already saturated |
| U3-XDFR | REFINE | M1 NOVEL | DRAM-cached drive absorbs fragmentation |
| U5-HPST | REFINE | M1 NOVEL | Decode is compute-bound not load-bound |
| U6-LPMT | REFINE (wildcard) | M1 NOVEL | SeLockMemoryPrivilege unavailable |
| R5-ATTNBAND | REFINE | gauge-exploitation | (listed above) |

---

## Experiment Sequencing Recommendation for Stage 5

**Run first (decisive, fast):**
1. **CF2 delta amplitude** (A4-APND / C5-RKDB gate) — 15 min. Single number that either validates or closes the CF2 amplitude-coupling assumption for both proposals simultaneously. If NO-GO, both proposals survive in weaker form (APND's log-structured I/O win remains; RKDB loses its error-bound motivation).

2. **C3-ZOIA ρ_ℓ measurement** — 5 min. Uses existing W_Q SVD data. If ρ_ℓ ≤ 0.3, eliminates the need for INT16 sidecar and confirms the elegance of the W_Q + INT8 activation path. If ρ > 0.3, the bypass IS needed (structural finding).

3. **F2-HPGF head energy distribution** — 5 min. The cheapest structural finding in the run. Either confirms the energy cliff (proceeding to ΔNLL measurement) or kills the idea cleanly.

**Top 3 candidates for Stage 5 selection:**
1. **C3-ZOIA** — 5-min experiment, algebraically grounded, NOVEL, elegance-class subspace-alignment. Cheapest decisive structural finding.
2. **A1-GFRS-2** — 10-min experiment, algebraically exact gauge argument, NOVEL frame with no published analog, cleanest novelty claim.
3. **R5-RAOK-FULL** — strongest deployment-path candidate; NOVEL mechanism, CF3-grounded, no imported theorem, best residency arithmetic. 30-min experiment.

F2-HPGF is a strong alternate for slot 3 if its 5-min head-energy experiment shows a cliff.

---

## KILL_LIST Additions

No new kills from the advancing pool in this stage. The DOWNGRADE entries (AWQKV, WVWK-MLA, JQKLR, SECP) do not meet the kill threshold — they are engineering-integration downgrades, not CF9/CF10/math-correctness failures. KILL_LIST entries should await Stage 4 if those surviving primitives are confirmed dead via experiment.

**Pre-emption note for future ideators:** As of 2026-05-09:
- Post-hoc GQA→MLA-style KV projection compression is covered by PALU (ICLR 2025) and TransMLA (NeurIPS 2025).
- Calibration-free spectral statistics predicting SVD compression degradation on Qwen3 is covered by arXiv:2604.18085.
- A3 (arXiv:2505.12942) covers analytical low-rank approximation of QK and OV components with functional-error minimization.
- ReCalKV (arXiv:2505.24357) covers head-reordering + grouped SVD for KV compression.
- DeltaKV (arXiv:2602.08005) covers inter-token residual-based KV compression.
