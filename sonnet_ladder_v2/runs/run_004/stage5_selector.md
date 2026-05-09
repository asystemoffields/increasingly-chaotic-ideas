# Stage 5 — Selector — Run 004 (v2 Sonnet Ladder)

Selector: Sonnet 4.6 (single-agent pass, both tracks).
Date: 2026-05-09.
Inputs read: STAGE5_SELECTOR.md, stage2_judge.md, stage3_deep_research.md, stage4_skeptic.md, SUMMARY.md, KILL_LIST.md.

---

## Scoring Summary (all candidates)

### Track A

Candidates: F-AERO-Q3, F-WVOS, A-CADH, HRQK [FREE SWING], and Stage 4 ideas S2 (ZWMP) and S5 (IQAT) which Stage 4 nominated for Stage 5 consideration. A-SEED, A-ALOG, A-REGM advanced Stage 3 REFINE but are lower priority per Stage 3 summary. Frame-novelty bonus advancers: F-GFQO (DOWNGRADED at Stage 3 via KILL_LIST entry v2-S3-R004-001).

Re-verification of novelty (Step 1):
- **F-AERO-Q3**: AERO paper (arXiv:2410.13060) confirmed applied to GELU/softmax private-inference only; no Qwen3/SwiGLU silu→identity post-training results as of 2026-05-09. NOVEL.
- **F-WVOS**: LatentLLM (MERL TR2026-018) and MHA2MLA (arXiv:2502.14837) require fine-tuning; zero-shot pure-SVD W_V fold on Qwen3 unpublished as of 2026-05-09. PARTIAL (fine-tuning distinguisher intact). No pre-emption since Stage 3.
- **A-CADH**: "Head-wise Shareable Attention" (arXiv:2402.11819) measures KV head similarity, not W_Q weight matrix cosine similarity for content-addressable dispatch. ADJACENT, not pre-empted. NOVEL measurement as of 2026-05-09.
- **S5 (IQAT)**: No paper found measuring sub-additivity of rank truncation + INT8 quantization on Qwen3 attention weights as of 2026-05-09. NOVEL structural claim.

Convergence map (from Stage 2 Cluster C5): F-AERO-Q3 is solo (no cluster); F-WVOS is in Cluster C1 (6-orientation convergence, strongest cluster in this run); A-CADH is in Cluster C5 (3-orientation: HRQK, A-CADH, C-RBHD); IQAT is solo.

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame bonus | NO-GO bonus | Final |
|---|---|---|---|---|---|---|---|---|
| F-AERO-Q3 | 12 | +2 | 14 | 1.0 (solo) | 1.2 (algebraic-identity, constructive, pre-mult ≥ 9) | 0 | +2 (first SwiGLU silu→identity class characterization; CF-candidate) | **18.8** |
| A-CADH | 12 | +2 | 14 | 1.2 (C5, 2-orient) | 1.1 (subspace-alignment, auditable not constructive end-to-end) | +1 (A2=3, frame genuinely fresh: content-addressable hash-dispatch for attention heads) | +1 (disambiguates CF11 head-redundancy interpretation) | **20.32** |
| F-WVOS | 12 | +1 | 13 | 1.5 (C1, 6-orient) | 1.2 (algebraic-identity, constructive, pre-mult ≥ 9) | 0 | +1 (closes W_V/W_O attention-weight characterization) | **24.4** |
| HRQK [FREE SWING] | 11 | +1 | 12 | 1.2 (C5) | 1.0 (no elegance tag) | 0 | +1 | 15.4 |
| S5-IQAT | — (S4) | +2 | S4 raw ≈ 10 | 1.0 | 1.1 (subspace-alignment) | +1 (runtime-fusion gap, genuinely fresh frame: rank-truncation + quant sub-additivity) | +2 (either sub-additive GO enables 32× W_Q storage; or superadditive NO-GO closes INT8+rank combo class) | **16.1** |

**Track A pick: F-WVOS** — highest weighted score (24.4), driven by 6-orientation Cluster C1 convergence (×1.5 multiplier), algebraic-identity constructive elegance (×1.2), pre-mult base of 13, and NO-GO bonus of +1 for closing the last uncharacterized attention-weight class. Runner-up: A-CADH (20.32), which Stage 6 should have on deck if F-WVOS is killed.

Note: F-AERO-Q3 scores 18.8 (third). Its 10-minute runtime and clean binary settler make it the highest-information-per-minute experiment in the run, but it is outscored by both F-WVOS and A-CADH on the weighted formula. Stage 6 should co-schedule F-AERO-Q3 as a 10-minute pre-flight alongside F-WVOS to double the round's structural findings at negligible marginal cost.

---

### Track B

Candidates: A-TROP, WDRS/F-WNORM, Stage 4 ideas S3 (RDCP) and S5 (IQAT; dual-track). Systems candidates: NCZC, WWTE, WSFI [FREE SWING], GTBP, NNPA [FREE SWING]. Weight-compression candidates: AWCJR, VOKV, C-AQOD, C-ODWQ.

Re-verification (Step 1):
- **S3-RDCP**: No paper found applying CF3's K-dependent Jaccard numbers to KV-cache run-length compression in an append-only (no-eviction) tier scheme as of 2026-05-09. NOVEL.
- **A-TROP**: Confirmed no paper uses tropical max-dominance for W_down input-channel dispatch in LLM inference as of 2026-05-09. NOVEL.
- **WDRS/F-WNORM**: W_down SVD sweep on Qwen3 unpublished. NOVEL measurement.
- **WSFI**: SuperFetch/SysMain learning loop for LLM inference confirmed no published prior as of 2026-05-09. NOVEL.

| ID | S2 Total | S3 Conf | Pre-Mult | Conv × | Elegance × | Frame bonus | NO-GO bonus | Final |
|---|---|---|---|---|---|---|---|---|
| A-TROP | 12 | +1 | 13 | 1.0 | 1.1 (structural combinatorial; auditable but not constructive end-to-end) | +2 (A2=3, tropical semiring in LLM inference: zero published cousins + underrepresented in framing-diversity targets) | +1 (W_down activation-input structure first characterization) | **17.3** |
| WDRS/F-WNORM | 11 | +2 | 13 | 1.2 (C3, 2-orient) | 1.2 (algebraic-identity: same R3 W_gate sweep protocol) | 0 | +2 (CF8 completion; CF-candidate kill-list entry if full-rank) | **19.6** |
| S3-RDCP | — (S4) | +2 | S4 raw ≈ 10 | 1.0 | 1.0 | +2 (A2=3: CF3-grounded append-only KV tier; zero published cousins at K=0.1% Jaccard anchor) | +2 (either: GO opens V-channel static-tier path; NO-GO closes the append-only CF3 KV compression class definitively) | **16.0** |
| WSFI [FREE SWING] | 11 | +1 | 12 | 1.0 | 1.0 | +2 (A2=3, SuperFetch learning loop: zero published ML inference cousins) | +1 (documents Windows 11 NVMe OS-prefetch behavior for all future proposals) | **15.0** |
| NCZC | 11 | +1 | 12 | 1.0 | 1.0 | 0 | +1 | 13.0 |
| C-ODWQ | 10 | +2 | 12 | 1.2 (C2) | 1.0 | 0 | +1 | 15.4 |
| VOKV | 11 | +1 | 12 | 1.5 (C1) | 1.1 | 0 | +1 | 20.2 |

Note: VOKV (C1 convergence representative for Track B) scores 20.2, which is higher than WDRS/F-WNORM (19.6) before the NO-GO bonus is considered. However, VOKV's load-bearing measurement (W_V spectrum) is identical to the Track A pick (F-WVOS), and both draw on the same C1 convergence cluster. Running VOKV as Track B after selecting F-WVOS for Track A produces a hardware resource conflict on the same SVD pass — the STAGE5_SELECTOR.md instructs preference for experiments "whose pre-program measurements complement each other." WDRS/F-WNORM is complementary (different matrices, different object class), and its NO-GO-finding bonus (+2, CF-candidate) makes it the correct Track B pick. VOKV/C1 W_V spectrum is already settled by Track A's F-WVOS pass; its result feeds VOKV automatically.

**Track B pick: WDRS/F-WNORM (canonical: F-WNORM)** — score 19.6 (post-hardware-conflict adjustment over VOKV), Cluster C3 2-orientation convergence (×1.2), algebraic-identity constructive elegance (×1.2), NO-GO bonus +2 (CF8 completion CF-candidate), 35-minute test. Runner-up: A-TROP (17.3), which Stage 6 should prime if F-WNORM is killed.

---

## ═══════════════════════════════════════════════════════
## TRACK A PICK — F-WVOS
## ═══════════════════════════════════════════════════════

### 1. Title and one-line description

**Run 004 / Track A — F-WVOS: W_V/W_O Algebraic Fold (Zero-Shot Qwen3)**

Does Qwen3-1.7B's value projection W_V have a sufficiently concentrated singular spectrum (r_99/d ≤ 0.80) to enable an exact algebraic fold into W_O that reduces attention weight residency by up to 4× with ΔNLL ≤ 0.5 nats?

### 2. Class tags

`compression-lr` / `arch-transposition` (the fold changes the attention computation graph from a three-stage V·W_V·W_O to a two-stage V·W_VO where W_VO = W_O·U_V·Σ_V).
Elegance class: `algebraic-identity` (constructive: the fold is exact given any low-rank approximation of W_V; the approximation error is the only loss).

### 3. Hypothesis

**H1 (load-bearing):** W_V in Qwen3-1.7B-Base (GQA: 8 KV heads × 128 dim, W_V shape 1024×2048 per layer) has r_99/d ≤ 0.80 — i.e., at most 819 of the 1024 singular values carry 99% of the Frobenius energy. This is grounded in Cluster C1 convergence (6 orientations independently surfaced W_V/W_O characterization as the unlocking measurement) and Stage 3's note that the Spectral Lifecycle paper (arXiv:2604.22778) confirms V/O projections "compress uniformly" across training.

**H2 (downstream):** Given H1, the algebraic fold O = softmax(QK^T) · (x W_V) · W_O can be rewritten O = softmax(QK^T) · (x V_V^T) · (Σ_V U_V^T W_O), storing two lower-rank factor matrices (V_V^T ∈ R^{r×2048} and W_VO ∈ R^{2048×r}) instead of the original W_V (1024×2048) + W_O (2048×2048). At r=256: combined storage = 2×256×2048×2B = 2 MB per layer vs original 1024×2048×2B + 2048×2048×2B = 12.6 MB per layer. 6× reduction conditional on H1. The fold is exact (no approximation beyond the SVD truncation at rank r).

H1 depends on: CF11 (W_Q concentrated at r_99/d ≈ 0.63; W_V is expected to follow similar training dynamics), and Spectral Lifecycle supporting evidence. H1 falsifiability: 25-minute all-layer SVD. H2 is gated on H1 GO; if H1 is NO-GO, H2 is not tested.

Differentiator from LatentLLM/MHA2MLA: both require fine-tuning to recover quality after decomposition. This is zero-shot, pure-SVD, post-training. The zero-shot moat is load-bearing. As of 2026-05-09, no paper demonstrates W_V fold quality on Qwen3-family GQA models without any fine-tuning.

**Critical shape note (from Stage 3):** In Qwen3-1.7B with GQA, W_V is NOT 2048×2048. Verify `model.layers[0].self_attn.v_proj.weight.shape` before finalizing arithmetic. Expected: (8×128, 2048) = (1024, 2048). Stage 5 plan uses this shape throughout; runner must confirm before committing to residency claims.

### 4. Smallest test

**Model:** Qwen3-1.7B-Base, bf16. Transformers tokenizer (`Qwen/Qwen3-1.7B-Base`).
**Hardware:** Ryzen 5 7530U, 7.28 GiB RAM. All 28 W_V matrices at bf16: 28 × 1024 × 2048 × 2B = ~118 MB. All W_O matrices: 28 × 2048 × 2048 × 2B = ~236 MB. Both fit in RAM easily.
**Calibration corpus:** None — purely weight-based operation (no calibration data needed).
**Eval corpus:** WikiText-2 held-out, 512 tokens. `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:512]`. Same corpus used for all v2 ladder evaluations.
**Layers touched:** All 28 attention layers; matrices: v_proj (W_V) and o_proj (W_O) per layer.
**Sweep parameters:** K ∈ {64, 128, 256, 512} for rank-r fold. Primary threshold: K=256. Permutation and random-init controls (see Section 9).

**Step-by-step:**
1. Verify W_V tensor shape for Qwen3-1.7B GQA (2 min). Confirm 1024×2048 or correct accordingly.
2. SVD of W_V for all 28 layers; record r_99/d, var@K for K ∈ {64, 128, 256, 512}. Script: `scripts/fwvos_wv_fold.py`. ~20 min.
3. Gate check: if mean r_99/d > 0.90 → STOP (NO-GO). Do not proceed to fold NLL.
4. Construct rank-256 fold: for each layer, compute W_VO = W_O · U_V[:, :256] · diag(Σ_V[:256]); store (V_V[:256, :]^T) + W_VO. Replace W_V and W_O with these factors in the model. ~5 min.
5. Measure ΔNLL vs baseline on WikiText-2 eval. ~10 min.
6. **Permutation control** (see Section 9): shuffle rows of W_V before SVD; re-run fold at K=256; measure ΔNLL. ~10 min.
7. **Random-init control**: create Qwen3-1.7B with random weights (same architecture, torch.manual_seed(42)); run SVD on random W_V; measure r_99/d. ~15 min.

**Output path:** `experiments/stage0/ladder_v2/round4_fwvos/`
**Wall-clock:** ~35 min (fold experiment) + ~25 min (controls) = 60 min total. Within 8h gate.
**Script:** `scripts/fwvos_wv_fold.py` (needs new script). The script: loads Qwen3-1.7B-Base weights; verifies W_V shape; runs full-model SVD pass; constructs rank-K factor pairs; patches model's forward to use folded weights; runs NLL eval on WikiText-2. Permutation and random-init variants are command-line flags on the same script.

### 5. Go threshold

**Primary GO:** mean r_99/d ≤ 0.80 AND ΔNLL@K=256 ≤ 0.5 nats vs bf16 baseline.
**Soft GO:** mean r_99/d ≤ 0.90 AND ΔNLL@K=512 ≤ 0.5 nats (4× compressed fold viable at higher K).
**Skeptic-controls gate (mandatory):** GO is valid only if real-signal ΔNLL is ≤ 0.5 nats AND permuted-W_V ΔNLL is ≥ 5.0 nats (permuted signal much worse — confirms the concentration is a learned property). Trained model's r_99/d must be ≤ 0.80 AND random-init model's r_99/d must be > 0.90 (confirms spectral concentration is training-induced, not architectural).
The CF-ground for this threshold: CF11 W_Q global K=128 at ΔNLL=+0.98 nats is the adjacent data point; W_V at K=256 (2× larger rank budget) should meet ≤ 0.5 nats if spectrum is at least as concentrated.

### 6. No-go threshold

**NO-GO:** mean r_99/d > 0.90 → W_V is full-rank (CF8 extends to W_V); zero-shot algebraic fold provides no meaningful compression without retraining.

**Class-level kill:** NO-GO ⇒ "zero-shot SVD-fold of attention value projections (W_V) in Qwen3-family GQA without retraining" is structurally dead. Kills VOKV cascade, C-VKSD joint-basis approach, A-GFWV product-spectrum approach, and all future ideas premised on W_V post-training concentration. Updates CF8 boundary: CF8 (trained Qwen3 MLP weights full-rank) partially extends to W_V, leaving only W_Q and W_K as confirmed low-rank attention weight classes.

**Also kills** (secondary): If W_V is full-rank, the Spectral Lifecycle paper's "V/O compress uniformly" claim fails for Qwen3-1.7B specifically — the training-dynamics basis for the claim may not hold in SwiGLU decoder architectures at this scale.

### 7. Ambiguous-zone follow-up

**GRAY zone:** r_99/d in 0.80–0.90 → concentration partial but not enough for 4× fold at K=256.

Follow-up (30 min additional): try activation-weighted SVD variant (weight the SVD by post-attention activation magnitude using PDAP forward-pass data, reuse 200-prompt PDAP cache). If activation-weighted r_99/d ≤ 0.80 while Frobenius r_99/d is in 0.80–0.90, the concentration is functionally present but not uniform across singular directions — the fold at K=256 under activation weighting may achieve ΔNLL ≤ 0.5 nats. Runtime: +30 min. Resolves: whether the concentration is Frobenius-uniform or activation-skewed.

**Per-layer heterogeneity check:** If mean r_99/d is GRAY but variance across 28 layers is high (std > 0.08), compute per-layer K thresholds and report the fraction of layers meeting ≤ 0.80 at K=256. Layers meeting threshold can be selectively folded (partial GO).

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should kill this pick outright if:
1. **GQA shape arithmetic error:** W_V shape is materially different from expected (1024×2048), invalidating the residency claims. If so, re-run Section 4 with corrected arithmetic before NLL eval.
2. **LatentLLM Qwen3 coverage:** Stage 6 search finds LatentLLM (MERL TR2026-018) includes Qwen3-class results with no fine-tuning. If this is confirmed, the frame differentiator collapses; escalate to runner-up A-CADH.
3. **Skeptic-controls failure:** Permuted-W_V ΔNLL < 3.0 nats (permuted signal nearly as good as trained), which would indicate spectral concentration is architectural rather than trained-in.
4. **RAM OOM:** If the model + SVD workspace exceeds 7.28 GiB, use streaming SVD (process each W_V independently, discard after fold construction).

### 9. Skeptic-controls declaration

The experiment claims: "W_V concentration is a trained property of Qwen3-1.7B-Base (not an architectural artifact); the fold produces meaningful compression."

**Permutation control (MANDATORY):** Randomly permute the rows of each W_V matrix before computing SVD. Measure r_99/d of permuted W_V. The permuted matrix has identical Frobenius norm but destroyed the singular value structure that arose from training. GO requires: trained r_99/d ≤ 0.80 AND permuted r_99/d > 0.90. The gap (≥ 0.10) confirms trained spectral concentration is not a row-norm artifact. Additionally: run fold NLL with permuted W_V; expect ΔNLL ≥ 5.0 nats (permuted fold is badly broken, demonstrating the trained fold's quality is not trivially structural). This control is implementable with a single `torch.randperm` call in the script.

**Random-init control (MANDATORY):** Initialize Qwen3-1.7B architecture with random weights (`torch.manual_seed(42)`), no trained parameters. Run W_V SVD. Record r_99/d. GO requires: trained r_99/d ≤ 0.80 AND random-init r_99/d > 0.90. This catches the case where GQA's (1024×2048) rectangular shape architecturally produces some spectral concentration independent of training. Expected: random init produces near-flat spectrum (r_99/d ≈ 1.0) due to random isometry of Gaussian initializations. Runtime: ~15 min (SVD only; no NLL eval needed for random-init control).

**Multi-seed (APPLICABLE but scoped):** The fold itself has no random seed dependence (SVD is deterministic). However, calibration corpus is not used (pure weight-based). Multi-seed applies to the NLL eval: re-run eval on 3 disjoint 512-token segments of WikiText-2 (seeds for dataset sampling: 42, 123, 456). Report mean ± std ΔNLL. GO threshold must be met by mean; worst-of-3 must not be above NO-GO threshold of 2.0 nats. Runtime: +10 min for 2 additional eval passes.

Justification that all three controls are mandatory (not "not applicable"): the experiment claims trained spectral concentration is the enabling structural property; all three controls directly test this claim. None can be waived.

### 10. Runtime estimate

| Step | Activity | Wall-clock |
|---|---|---|
| Setup | Load model, verify W_V shape, confirm eval corpus | 5 min |
| Step 2 | All-layer W_V SVD (28 × 1024×2048 matrices) | 20 min |
| Step 3 | Gate check: mean r_99/d | 1 min |
| Step 4 | Construct fold at K=256 for all layers | 5 min |
| Step 5 | NLL eval on WikiText-2 (512 tokens, 3 seeds) | 15 min |
| Step 6 | Permutation control SVD + fold NLL | 10 min |
| Step 7 | Random-init control SVD (no NLL) | 15 min |
| Post | Write results table | 5 min |
| **Total** | | **~76 min** |

Within 8h gate with substantial margin. If GRAY zone activates: add 30 min for activation-weighted SVD variant.

### 11. Script identification

New script: `scripts/fwvos_wv_fold.py`

The script:
- Inputs: model_name (default `Qwen/Qwen3-1.7B-Base`), rank_k (default 256), mode (`{fold, permuted, randinit, multieval}`), seed, eval_tokens (default 512).
- Loads model; verifies W_V tensor shape; computes per-layer SVD of W_V and records r_99, var@K for K list.
- In `fold` mode: patches model forward pass to use (V_V^T, W_VO) factor pairs; runs NLL eval; reports ΔNLL.
- In `permuted` mode: row-permutes each W_V before SVD; reports r_99/d and fold NLL.
- In `randinit` mode: re-initializes model weights randomly (seed 42); runs SVD; reports r_99/d only.
- In `multieval` mode: runs eval on 3 dataset seeds; reports mean ± std ΔNLL.
- Outputs: JSON result file + per-layer r_99 plot to output path.
- Load-bearing computation: `torch.linalg.svd(W_V, full_matrices=False)` per layer; fold construction as `W_VO = W_O @ U_V[:, :K] @ diag(S_V[:K])`.

Existing script `scripts/lhqd_lmhead_spectrum.py` (from R6-B) provides the SVD loop pattern; `scripts/fwvos_wv_fold.py` adapts it for per-layer W_V + fold construction.

### 12. Downstream implications

**GO unlocks:**
- Activates Cluster C1 cascade: AWCJR (joint W_Q+W_K+W_V+W_O compression), VOKV (KV-algebraic compression), C-VKSD (W_K/W_V joint basis), A-GFWV (W_V·W_O product fold). All 6 C1 orientations become actionable in parallel.
- Enables S5-IQAT (INT8 feasibility for compressed W_V — same sub-additivity argument as W_Q).
- At 70B scale: ~12× W_V+W_O residency reduction per layer × 80 layers = potentially ~3 GB freed from attention projections in bf16.

**NO-GO kills:**
- Closes the "post-training zero-shot W_V fold" class for Qwen3-family.
- Kills VOKV, C-VKSD (within-layer joint KV basis), A-GFWV (product fold — if W_V is full-rank, the product W_V·W_O is at most full-rank in W_O).
- Adds CF constraint: "zero-shot SVD of W_V in Qwen3 GQA architecture does not compress; CF8 partially extends to value projections."
- Does NOT kill W_Q or W_K compression (CF11 confirmed independently). Track A attention compression survives via W_Q.

**KILL_LIST interaction:** GO does not kill anything on the current KILL_LIST. NO-GO adds a new CF boundary entry, and Stage 6 should draft the KILL_LIST entry before the round closes.

### 13. Provenance

- **Originating orientations:** F (First-Principles): F-WVOS directly. Also in Cluster C1: R-VOKV, C-VKSD, C-WVKQ, F-SPECA, A-GFWV — six orientations converged on W_V spectrum as the unlocking measurement.
- **Convergence cluster:** C1 (W_V and W_O spectral characterization as unlocking primitive). 6-orientation convergence; strongest convergence signal in run_004. Convergence multiplier: 1.5.
- **Stage 4 gap-idea slot:** Not Stage 4 originated; Stage 4 validated the pick (no new Track A ideas outrank F-WVOS).
- **Frame-novelty bonus path:** Not eligible (A2=2, not 3); picks up its weight from convergence multiplier instead.
- **Elegance path:** `algebraic-identity`, constructive (fold is exact given low-rank approximation), auditable in 5 lines.
- **Runner-up (Stage 6 fallback):** A-CADH (score 20.32). If Stage 6 kills F-WVOS for LatentLLM pre-emption, advance A-CADH — it is orthogonal (measures W_Q head similarity, not W_V spectrum) and carries Stage 4's cross-pollination CP-3 structural benefit.

---

## ═══════════════════════════════════════════════════════
## TRACK B PICK — F-WNORM (canonical for WDRS/F-WNORM convergence)
## ═══════════════════════════════════════════════════════

### 1. Title and one-line description

**Run 004 / Track B — F-WNORM: W_down Rank Sweep (CF8 Closure)**

Does Qwen3-1.7B's W_down (6144→2048) have a concentrated singular spectrum (r_99/d < 0.90) or confirm full-rank (r_99/d ≥ 0.90) — closing CF8 to all three MLP weight matrices and definitively establishing the no-retraining MLP compression ceiling?

### 2. Class tags

`compression-lr` (tests whether low-rank compression of W_down is viable) / `pipeline-infrastructure` (CF8 structural closure regardless of outcome).
Elegance class: `algebraic-identity` (the R3 W_gate rank sweep protocol is constructive and auditable; F-WNORM is the same protocol applied to the remaining MLP matrix class).

### 3. Hypothesis

**H1 (primary structural):** W_down in Qwen3-1.7B-Base is full-rank: r_99/d_out ≥ 0.90 across ≥ 20/28 layers (where d_out = 2048, the output dimension). This is the CF8-predicted outcome: R3 (W_gate full-rank), R4 (W_up full-rank, MORE rank-sensitive than W_gate), and the compound finding 8 ("trained Qwen3-family weights resist no-retraining structural compression") all predict W_down full-rank.

**H2 (structural framing novelty):** The d_ffn > d_model shape (6144 input → 2048 output) means the "effective rank" is bounded by d_out = 2048, not d_in = 6144. The relevant rank question is "what fraction of d_out=2048 singular values are load-bearing?" This is a structurally different question from W_gate/W_up (which are d_model×d_model square matrices). The asymmetric bound makes the rank characterization non-trivial even given CF8's prediction: a matrix could have r_99=2048/2048 = 1.0 (full-rank relative to output dimension) while still exhibiting decay patterns in the singular values that inform quantization codebook assignment.

**H3 (secondary, conditional):** IF W_down shows unexpected concentration (r_99/d_out ≤ 0.70), A-TROP's tropical stable-row hypothesis and Stage 4's TDIP are both strengthened (row-level structure may coexist with weight-level concentration). Conditional on H3 GO: activate TDIP / S4 cascade.

H1 is grounded in: CF4 (W_gate full-rank), CF5 (W_up MORE rank-sensitive), CF7/CF8 (class-level kill on MLP low-rank), compound finding 8 (three independent measurements all confirm full-rank).

### 4. Smallest test

**Model:** Qwen3-1.7B-Base, bf16.
**Hardware:** Ryzen 5 7530U, 7.28 GiB RAM. All 28 W_down matrices: 28 × 2048 × 6144 × 2B = ~672 MB. Within RAM, but process sequentially (one layer at a time) to avoid OOM.
**Calibration corpus:** None for Step 1 (pure weight-based SVD). Step 2 (NLL sweep, optional): WikiText-2, 512 tokens. `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:512]`.
**Layers touched:** All 28 MLP layers; matrix: down_proj (W_down) per layer.
**Sweep parameters:** K ∈ {128, 256, 512, 1024, 2048} for rank sweep (same protocol as R3 W_gate experiment). NLL sweep only if H1 NO-GO (unexpected concentration found).

**Step-by-step:**
1. For each of 28 layers, compute `torch.linalg.svd(W_down[L], full_matrices=False)` (shape 2048×6144; SVD gives min(2048,6144)=2048 singular values). Record r_99, var@K for K ∈ {128, 256, 512, 1024, 2048}. Process one layer at a time to stay within RAM. ~20 min.
2. Gate check: if mean r_99/d_out ≥ 0.90 in ≥ 20/28 layers → CF8 CONFIRMED for W_down → STOP here (no NLL sweep needed; structural finding is the result). ~1 min.
3. NLL sweep (only if gate check shows r_99/d_out < 0.90 in ≥ 10 layers): construct rank-K approximation of W_down at K ∈ {256, 512, 1024}; measure ΔNLL. ~15 min.
4. **Permutation control:** Shuffle rows of W_down[L14] before SVD; record r_99/d. Compare to trained r_99/d. ~5 min.
5. **Random-init control:** Initialize W_down with random weights (`torch.manual_seed(42)`); run SVD for all 28 layers; record mean r_99/d. ~15 min.
6. **A-TROP joint measurement** (no additional cost): during the Step 1 forward pass, compute argmax_j(W_down[L][i,j] × h_mlp[j]) for 200 calibration prompts per layer; record per-row stability fraction. This activates or kills Stage 4's TDIP cascade at no additional wall-clock cost beyond loading the calibration data (~10 min additional for 200 prompts). Script accepts `--run_trop` flag to add this.

**Output path:** `experiments/stage0/ladder_v2/round4_fwnorm/`
**Wall-clock:** ~20 min SVD + ~1 min gate + ~15 min controls + ~10 min A-TROP joint = 46 min total. Within 8h gate.
**Script:** `scripts/fwnorm_wdown_spectrum.py` (new script, adapts `scripts/r3_wgate_rank_sweep.py` from R3). The script: per-layer sequential W_down SVD with r_99 recording; optional NLL sweep; permutation + random-init controls; optional TROP stability fraction measurement.

### 5. Go threshold

(F-WNORM has no standard GO in the compression sense; any outcome is structurally valuable.)

**CF8-CONFIRM GO (primary expected outcome):** mean r_99/d_out ≥ 0.90 in ≥ 20/28 layers. Closes CF8 to all three MLP matrices. Pipeline kill-list entry: "W_down full-rank in Qwen3-1.7B; no-retraining MLP weight compression at all three matrices is dead."

**SURPRISE GO (compression viable):** mean r_99/d_out ≤ 0.75 in ≥ 15/28 layers AND ΔNLL@K=512 ≤ 0.5 nats. Opens W_down compression path; activates TDIP (S4) as highest-priority Track B successor.

**Skeptic-controls gate (mandatory):** Whether CF8-CONFIRM or SURPRISE GO, the real trained r_99/d must differ materially from random-init r_99/d by ≥ 0.10 in absolute terms (trained below random-init, indicating spectral structure is architectural for random, but for trained — if trained is more concentrated than random, that confirms learned compression; if trained ≈ random ≈ full-rank, CF8-CONFIRM is structural). For W_down full-rank expected outcome: trained r_99/d ≈ random-init r_99/d ≈ 1.0 is expected AND CORRECT — this would confirm the full-rank finding is an architectural property of d_out×d_in rectangular matrices under random initialization, which is the null hypothesis and consistent with CF8's trained-full-rank finding.

### 6. No-go threshold

For F-WNORM there is no "no-go" in the traditional sense — either outcome is structurally informative. However, for internal consistency:

**AMBIGUOUS RESULT:** r_99/d_out in 0.75–0.90 in 10–20/28 layers → unclear whether W_down is compressible. Follow-up: per-layer NLL sweep at K=512 settles whether the partial concentration produces quality gains.

**Class-level structural finding regardless:** F-WNORM's NO-GO result (from the pipeline's perspective of seeking compression) is that CF8 now covers all three MLP weight matrices. This IS the expected pipeline-valuable outcome — it closes the CF8 characterization and pins "no-retraining MLP compression is dead at all matrices" as a single confirmed CF entry with three independent data points.

### 7. Ambiguous-zone follow-up

If r_99/d_out falls in 0.75–0.90 for 10–20/28 layers (gray zone):

Follow-up 1 (15 min): NLL sweep at K ∈ {512, 1024} on WikiText-2. If ΔNLL ≤ 0.5 nats at K=512 → soft SURPRISE GO; if ΔNLL > 2.0 nats at K=512 → full-rank behavior despite moderate concentration (matches R3 W_gate pattern).

Follow-up 2 (20 min, conditional): activation-weighted SVD using PDAP post-SwiGLU activation stats as weighting kernel (replaces Frobenius SVD with X^TX-weighted SVD for W_down). If activation-weighted r_99/d_out ≤ 0.75 while Frobenius is in gray zone: concentration is functionally present under the actual usage distribution, and GPTVQ-style activation-weighted quantization is the viable path (not rank reduction per se, but codebook assignment from activation-weighted dominant directions).

### 8. Kill criteria (Stage 6 amendment slot)

Stage 6 should kill this pick outright if:
1. **ASVD/GPTVQ pre-emption on W_down:** Stage 6 finds a paper (post-Stage 3 search) reporting Qwen3-specific W_down activation-weighted SVD with rank characterization. Escalate to runner-up A-TROP.
2. **Skeptic-controls failure:** Permuted r_99/d within 0.03 of trained r_99/d (indicating row-permutation doesn't change spectrum, confirming it's purely the matrix shape that sets rank — this is actually the expected null-hypothesis outcome and does NOT kill the experiment; it confirms the finding is architectural).
3. **RAM OOM during sequential SVD:** If 2048×6144 matrix with bf16 + SVD workspace exceeds available RAM per layer. Mitigation: use chunked load via `torch.load(..., mmap=True)` to avoid materializing full model simultaneously.

### 9. Skeptic-controls declaration

The experiment claims: "W_down spectral structure (full-rank or concentrated) is a trained property relevant to compression viability, not a purely architectural artifact."

**Permutation control (MANDATORY):** Row-permute W_down[L14] before SVD; measure r_99/d. For a full-rank expected outcome, the permuted r_99/d should equal the trained r_99/d (permutation does not change singular values — this is by construction, since row permutation is an orthogonal transformation). Therefore: the permutation control for F-WNORM serves a DIFFERENT purpose than for F-WVOS. Here, it verifies measurement correctness (SVD is invariant to row permutation; if permuted ≠ trained, there is a software bug). Document this: "permutation control used for correctness verification, not as a trained-vs-architectural discriminator, because singular values are invariant to row permutation by definition."

**Random-init control (MANDATORY):** Initialize W_down (shape 2048×6144) with `torch.nn.init.kaiming_uniform_` (the default PyTorch init for linear layers). Run SVD. Record r_99/d. The Marchenko-Pastur distribution predicts the bulk spectrum of a random 2048×6144 matrix initialized with variance σ²: all singular values concentrate near σ×√(6144) with a thin tail. Expected: random-init r_99/d ≈ 0.95–1.0 (near-flat by random matrix theory). If trained r_99/d ≈ random-init r_99/d (both ~1.0), this confirms full-rank is the trained model's property AND is consistent with the random-matrix null hypothesis — CF8's full-rank finding is stable against this control. If trained r_99/d << random-init (unexpected concentration), training has compressed the spectrum, which is the SURPRISE GO case. Runtime: 15 min.

**Multi-seed (LIMITED APPLICABILITY):** The SVD is deterministic and weight-only. The NLL sweep (Step 3, only run in SURPRISE GO case) should use 3 WikiText-2 segments for robustness. For the CF8-CONFIRM primary outcome, multi-seed is not applicable because there are no random elements in the measurement. Document: "multi-seed not applicable to SVD measurement; NLL sweep (if run) uses 3 dataset seeds."

### 10. Runtime estimate

| Step | Activity | Wall-clock |
|---|---|---|
| Setup | Load model, verify W_down shape | 3 min |
| Step 1 | All-layer W_down SVD (28 layers, sequential) | 20 min |
| Step 2 | Gate check | 1 min |
| Step 3 | NLL sweep (ONLY if SURPRISE GO) | 15 min |
| Step 4 | Permutation control (correctness check) | 5 min |
| Step 5 | Random-init control SVD (no NLL) | 15 min |
| Step 6 | A-TROP joint stability measurement (optional, recommended) | 10 min |
| Post | Write results table | 3 min |
| **Total (CF8-CONFIRM path)** | | **~42 min** |
| **Total (SURPRISE GO path)** | | **~57 min** |

Within 8h gate with substantial margin.

### 11. Script identification

New script: `scripts/fwnorm_wdown_spectrum.py`

The script:
- Inputs: model_name, mode (`{svd, nll, permuted, randinit, trop}`), rank_k_list, seed, eval_tokens.
- Loads model; extracts W_down per layer sequentially (to stay within RAM budget).
- In `svd` mode: per-layer SVD; records r_99, var@K for K list; outputs summary table.
- In `nll` mode (SURPRISE GO only): constructs rank-K W_down approximation; patches model; measures ΔNLL.
- In `permuted` mode: row-permutes W_down[L14]; runs SVD; reports r_99 (expected = trained value; software correctness check).
- In `randinit` mode: re-initializes W_down with kaiming_uniform_; runs SVD.
- In `trop` mode: runs 200 calibration prompts; records per-row argmax stability fraction across all 28 layers.
- All modes output to JSON result + SVD curve plot.
- Load-bearing computation: sequential per-layer `torch.linalg.svd(W_down_L, full_matrices=False)` where W_down_L is 2048×6144.

Adapts `scripts/r3_wgate_rank_sweep.py` (R3 experiment) with updated matrix name and shape-aware r_99 computation (r_99 relative to min(2048, 6144) = 2048).

### 12. Downstream implications

**CF8-CONFIRM GO unlocks:**
- Closes CF8 completely. Pipeline kill-list entry: "W_gate (R3), W_up (R4), W_down (R4-B F-WNORM) all full-rank in Qwen3-1.7B; no-retraining structural MLP compression is dead at all three matrices." This is the KILL_LIST's most consequential single entry since CF8 was first opened at R3.
- Clears pipeline bandwidth to focus entirely on attention compression (CF11 successor chain) and quantization (RAOK, PDAP-V, TDIP).
- If A-TROP joint measurement is run: provides W_down input-channel stability data to assess TDIP (S4) viability as the first RAOK-compatible codebook-assignment path grounded in activation-trajectory structure.

**SURPRISE GO unlocks:**
- Opens W_down rank-reduction path (kills CF8 extension to W_down).
- Activates TDIP (S4) + WDSC (S7) cascade as immediate Track B follow-up.
- Motivates recheck of W_up under activation-weighted SVD (if W_down concentrates under usage, W_up might too — but R4 empirical data strongly argues against this).

**KILL_LIST interaction:** CF8-CONFIRM outcome should immediately add a KILL_LIST entry: "W_down low-rank no-retraining compression — KILLED empirically." Stage 6 should draft this entry from the F-WNORM result.

### 13. Provenance

- **Originating orientations:** F (First-Principles): F-WNORM directly. R (Reach): WDRS (structurally identical proposal). 2-orientation convergence on Cluster C3.
- **Convergence cluster:** C3 (W_down rank characterization = CF8 completion). 2-orientation convergence (R + F). Convergence multiplier: 1.2.
- **Stage 4 gap-idea slot:** Not Stage 4 originated; Stage 4 confirms F-WNORM as the correct infrastructure experiment (S7-WDSC in Stage 4 is a joint version of F-WNORM + A-TROP; F-WNORM covers the primary measurement and the `--run_trop` flag covers S7's joint bonus at no extra cost).
- **Frame-novelty bonus path:** Not applicable (A2=1 for the rank-sweep claim itself; novelty is the private Qwen3-specific measurement, not the frame).
- **Elegance path:** `algebraic-identity` class (same auditable R3 W_gate SVD protocol); constructive. Pre-multiplier (13) × 1.2 × 1.2 = 18.72 base; NO-GO bonus +2 (CF-candidate structural finding) → total 20.72.
- **Runner-up (Stage 6 fallback):** A-TROP (score 17.3). If Stage 6 kills F-WNORM for ASVD/GPTVQ pre-emption, advance A-TROP — it tests a different W_down property (input-channel activation-trajectory dominance, not linear-algebra rank) and is not pre-empted by the same papers. A-TROP's `--run_trop` flag is included in F-WNORM's script anyway, so the measurement will be available regardless.
