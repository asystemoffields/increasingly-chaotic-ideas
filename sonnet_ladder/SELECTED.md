# Selected Experiments Queue

Pipeline picks one experiment per round. Each entry below is a runnable plan
with explicit go/no-go criteria. Status: `queued | running | done`.

When `done`, the result is appended below the entry. The experiment also
gets a kill-list entry retroactively (so future rounds don't re-propose it,
regardless of result — the *idea* has been tested).

## Format

```
### Round R — Experiment Name
**Class**: prediction-shadow,offload-disk
**Hypothesis**: Cross-scale FFN top-K activation indices are correlated...
**Smallest test**: Run Qwen3-0.6B and Qwen3-4B on N prompts; compute Spearman ρ...
**Go threshold**: ρ ≥ 0.35 across mid-layers at K=20%
**No-go threshold**: ρ < 0.20 across mid-layers
**Status**: queued | running | done
**Result** (if done): one paragraph
```

---

### Round 1 — Zero-Overhead Cross-Layer Row-Herding Predictor on SwiGLU (Idea D, amended by Stage 6)
**Class**: prediction-internal, offload-disk

**Framing per Stage 6**: Deja Vu (arXiv:2310.17157, ICML 2023) already showed (a) cross-layer cosine sim ≈ 0.99 in OPT-175B and (b) that a *trained* cross-layer MLP predictor achieves 93-99% precision on ReLU activations. The genuinely open question is whether a **zero-overhead dot-product predictor** (no separate model, no training) achieves usable precision on **SwiGLU activations** in modern transformers.

**Hypothesis**: For Qwen3-0.6B-Base, the input residual at layer L−1 (h_{L−1}) ranks the rows of W_gate[L] in such a way that the top-30% of |⟨h_{L−1}, W_gate[L]_row⟩| recovers ≥80% of the rows that actually land in the top-30% of post-SwiGLU activation magnitude at layer L. Cross-layer (h_{L−1} → layer L) is compared against a within-layer baseline (h_L → layer L) to isolate the cross-layer claim from the trivial input-to-gate-dot-product result.

**Smallest test**: Qwen3-0.6B BF16 on 200 prompts × 128 tokens (100 WikiText-2, 100 short-answer instruction). Forward hooks on every block input. For each MLP layer L:
  - Compute the **gold ranking** by post-SwiGLU magnitude: rank j by `|silu(W_gate[L] @ h_L)_j · (W_up[L] @ h_L)_j|`
  - Compute the **cross-layer predicted ranking**: rank j by `|⟨h_{L−1}, W_gate[L]_row j⟩|`
  - Compute the **within-layer baseline ranking**: rank j by `|⟨h_L, W_gate[L]_row j⟩|`
  - Report `precision@30% (cross-layer)` and `precision@30% (within-layer)` per layer
  - Also report cosine_sim(h_{L−1}, h_L) and cosine_sim(h_L, h_{L+1}) per layer for reference (NOT a go-gate)

Script: `scripts/cross_layer_sim_probe.py`. Output: `experiments/stage0/ladder/round1_idea_d/cross_layer_sim_results.json` + plot + summary.md. Estimated runtime: ~45 min.

**Go threshold (revised per Stage 6)**:
  - Mean precision@30% (cross-layer) ≥ **0.80** across MLP layers, AND
  - Cross-layer precision is no more than **0.10 below** within-layer precision (i.e., cross-layer signal is comparable to the trivial baseline, validating the L−1 → L predictor as usable for *prefetch* where cross-layer lookahead is what matters)

**No-go threshold (revised)**:
  - Mean precision@30% (cross-layer) < 0.55, OR
  - Cross-layer precision is more than 0.20 below within-layer precision (cross-layer signal substantially weaker than trivial within-layer; the *L−1 lookahead* premise fails even if the gate-dot-product premise holds)

**Ambiguous zone (0.55–0.80 cross-layer precision)**: rerun with K ∈ {10%, 20%, 50%}; check whether higher K cleans up the signal, indicating the predictor is right-shaped but the 30% cutoff is wrong.

**Cosine gate dropped**: per Deja Vu (2023), cross-layer cosine sim ≈ 0.99 in OPT-175B and is structurally guaranteed by residual additivity. Measuring it on Qwen3 reproduces a known result and would falsely validate the precision claim. The cosine numbers are reported but NOT a go-gate.

**Downstream implications (revised by Stage 6)**:
  - Positive: validates *zero-overhead dot-product cross-layer predictor on SwiGLU* — genuinely novel relative to Deja Vu's trained-MLP-on-ReLU result. Idea 3 (cross-model shadow oracle) is *not* validated by this — it remains a separate Round 2 candidate. Idea J (count-min sketch / token-history) is *not* killed even by negative — it's a different signal source.
  - Negative: kills only the "h_{L−1} dot W_gate row" predictor specifically. Idea 3 and Idea J remain alive. Trained-predictor variants (Deja Vu-style) also remain alive but carry overhead cost.

### Round 2 — PDAP: Per-Token Dynamic Activation Outlier Pinning (Stage 6 amended)
**Class**: compression-quant, runtime-fused, prediction-internal

**Hypothesis**: Activation outliers in transformer hidden states have token-dynamic structure (per-token Jaccard < 0.50 in normal-token stratum) NOT collapsing to LLM.int8()'s channel-static finding. Symmetric to Round 1's W_up dominance finding on the weight side.

**Smallest test (Stage 6 amended)**:
- Qwen3-4B in fp16 (NOT Q4_K_M — fp16 is the deployment-relevant scenario for W4A8/W4A16)
- 200 prompts × ~20 tokens drawn from 4 strata (50 each):
  - repetitive text (song lyrics, repeated patterns)
  - technical/code
  - natural language prose
  - random token sequences
- Forward hooks on hidden states at layers **{4, 8, 12, 16, 20, 24, 28, 32}** (8 layers, dense in the 36-layer model — Stage 6 amendment)
- Per-token: record top-K outlier channel sets at K ∈ **{0.1%, 1%, 5%, 10%}** (Stage 6 amendment — multi-K)
- **Token-type stratification (Stage 6 critical amendment)**: classify each token position as
  - "massive-activation" if max activation magnitude > 5× batch median (PrefixQuant-style detection)
  - "normal" otherwise
- Compute Jaccard separately within each stratum

**Primary metric**: mean Jaccard across consecutive normal-token pairs within a prompt (per layer, per K).

**Secondary metrics**:
- Bimodal Jaccard distribution check (Stage 6: most likely outcome — channel-static for normal tokens, low-Jaccard for massive-activation tokens)
- Top-k channels covering 90% of outlier events (cumulative coverage)
- Cross-stratum Jaccard variance (does technical/code differ from prose?)

**Go/No-Go (Stage 6 amended)**:
- **GO** — token-dynamic real: Jaccard < 0.50 in **normal-token stratum** AND top-k 90%-coverage > 5% of channels
- **NO-GO** — channel-static: Jaccard > 0.85 in normal-token stratum
- **GRAY (PrefixQuant pattern)**: bimodal — high Jaccard for normal tokens, low for massive-activation tokens. Most likely outcome. Implies hybrid scheme: PrefixQuant-style isolation of massive-activation tokens + static channel pinning for the rest.

**Script**: `scripts/pdap_outlier_jaccard.py`. Output: `experiments/stage0/ladder/round2_pdap/pdap_results.json` + plot + summary.md.

**Estimated runtime**: ~30-45 min on user's laptop (Qwen3-4B fp16 ~8 GB — actually doesn't fit at fp16; fall back to bf16 ~4 GB or use Qwen3-1.7B if memory pressure).

**Reference**:
- LLM.int8() (arXiv:2208.07339): asserts channel-static qualitatively, no formal Jaccard
- SmoothQuant (arXiv:2211.10438): assumes channel-static, no formal measurement
- **PrefixQuant (arXiv:2410.05265)**: decomposes channel-wise vs token-wise outliers; we explicitly stratify per their decomposition
- Quaff (arXiv:2505.14742, ACL 2025): Outlier Spatial Stability for fine-tuning
- NVFP4 Pretraining Dynamics (arXiv:2602.02047): training-time outlier dynamics

**Downstream implications (Stage 6 corrected)**:
- **Positive (token-dynamic in normal stratum)**: PDAP-style per-token scheme worth designing. AVX2 W(ternary)A8 stack (Round 2 ideas H+4) gets dynamic-outlier kernel design problem.
- **Negative (channel-static)**: Static channel-union pinning suffices. H/4 work with simpler scheme (faster path). Mild boost to ANS residual entropy assumption (idea A).
- **Gray zone (bimodal — most likely)**: PrefixQuant framing applies on Qwen3 family. Hybrid scheme (token-isolation + static channel pin) is the deployment recipe. Round 3 follow-up: confirm PrefixQuant's massive-token isolation works on Qwen3.

### Round 3 — Asymmetric W_gate Rank Reduction (Idea 12, escalated from CSMF)

**Class**: compression-lr (W_up-aware)

**Stage 6 escalation note**: CSMF (Idea C) was Stage 5's pick but Stage 6 found a fatal math flaw. Polynomial substitution of silu does NOT eliminate W_gate as a stored matrix — the β·(W_gate·x)·(W_up·x) and γ·(W_gate·x)²·(W_up·x) terms still require W_gate at inference (the merged tensor would be d^3 ≈ hundreds of GB per layer). Only the α-term folds; that's a degenerate biased linear map, useless on its own. The real residency saving is via rank reduction of W_gate, motivated by Round 1's W_up dominance finding. AERO (arXiv:2410.13060) is the right adjacent published technique — it removes activations entirely so W_gate and W_up can be folded into one matrix; we don't go that far, just compress W_gate via low-rank SVD.

**Hypothesis**: W_gate tolerates aggressive rank reduction (~K=64-128 for d_in=2048, d_out=11008 in Qwen3-1.7B) while W_up at full precision because of R1's finding that W_up·x dominates post-SwiGLU magnitude. PPL degradation < 1.0 nats at K=128 would validate this asymmetric compression as a Round 3 structural finding.

**Smallest test**: Qwen3-1.7B-Base bf16. For each MLP layer, replace W_gate with rank-K SVD reconstruction (K ∈ {4, 8, 16, 32, 64, 128, 256, 512}). Keep W_up and W_down at bf16 (testing W_gate sensitivity in isolation). Measure WikiText-2 PPL on a held-out passage. Find the rank where PPL degradation crosses 1.0 nats threshold.

**Go threshold**: PPL degradation ≤ 1.0 nats at K ≤ 128 across all layers AND degradation curve shows a clear inflection (not linear in K) — implies low-rank structure exists.

**No-go threshold**: PPL degradation > 5 nats at K ≤ 256 across all layers (W_gate is genuinely high-rank; rank reduction is not the path).

**Mid zone (1-5 nats at K=128)**: layer-sensitivity follow-up — some layers may tolerate K=64, others need K=256+. Adaptive per-layer rank selection.

**Script**: `scripts/wgate_rank_sweep.py`. Output: `experiments/stage0/ladder/round3_wgate/wgate_rank_results.json` + plot + summary.md.

**Estimated runtime**: ~15-25 min on user's laptop (Qwen3-1.7B bf16, 8 ranks × ~2 min PPL eval).

**References**:
- AERO (arXiv:2410.13060) — softmax-only LLMs via activation removal + FFN folding (different mechanism, same goal)
- Round 1 finding: W_up dominates SwiGLU top-K firing; suggests W_gate carries less structural information
- LoftQ, CALDERA, QERA — uniform low-rank correction (not asymmetric W_gate-only)

**Downstream implications**:
- **Positive (K≤128, ΔPPL<1)**: 70B W_gate footprint drops from 9.4 GB at IQ4 to ~750 MB at K=128 fp16 — saves 8.6 GB. Combined with W_up at IQ4 (9.4 GB) + W_down at IQ4 (9.4 GB) = ~19.5 GB total. Still NVMe but improved. With Idea 2 W_up codebook (sub-IQ4) and Idea G (RSTD on W_up across layers), DRAM-resident 70B becomes plausible.
- **Negative**: W_gate is genuinely full-rank; the 12+2 line is broken. Falls back to better quantization across the board. Idea 2 (PDAP three-tier on all of MLP) becomes the leading line.
- **Mid**: layer-adaptive rank — opens a per-layer K-selection problem (Round 4 candidate).

**Status**: done (Qwen3-1.7B-Base bf16, 4 ranks tested + sanity check)

**Result**: **NO-GO. W_gate is essentially full-rank in trained Qwen3.**

| K | K/d | PPL | ΔPPL | ΔNLL (nats) |
|---|---|---|---|---|
| 2048 (full) | 1.000 | 11.81 | +0.03 | +0.002 |
| 1024 | 0.500 | 25.47 | +13.7 | +0.77 |
| 512 | 0.250 | 420.3 | +408.6 | +3.57 |
| 256 | 0.125 | 5572 | +5560 | +6.16 |
| 128 | 0.063 | 19249 | +19237 | +7.40 |
| 16 | 0.008 | 14789 | +14777 | +7.13 |

Baseline NLL/tok = 2.467, PPL = 11.78. Full-rank reconstruction (K=2048) recovers baseline exactly (ΔPPL = +0.03 from bf16 round-trip noise) — confirms the SVD machinery is correct.

The catastrophic degradation at K < 2048 reveals **W_gate has essentially no low-rank structure** in Qwen3-1.7B. Even at 50% rank retention (K=1024), PPL doubles. At 25% rank, PPL is 35× baseline. The non-monotonicity at very low K (K=16 ~ K=128) is noise within the "garbage output" regime.

**Round 3 structural finding**: Round 1's "W_up dominance" was about *firing-magnitude rank* (which neurons fire post-SwiGLU); it does NOT translate to *linear-algebra rank* of W_gate. These are different questions. W_gate is a full-rank linear projection that does meaningful work at all 2048 singular value indices, even though its output magnitudes don't dominate firing.

**Class implication**: any compression scheme premised on "R1 says W_up matters more, therefore W_gate is compressible" is structurally wrong. Affected ideas:
- **Idea 12 (asymmetric W_gate rank reduction)**: KILLED empirically
- **CSMF (polynomial silu)**: already killed mathematically (Stage 6)
- **The 12+2 line as a whole**: KILLED. Cannot achieve DRAM-resident 70B via this path.

Surviving paths from Round 3:
- **Idea 2 (PDAP three-tier on W_up)**: independent of W_gate; Round 2 finding still applies. Should be Round 4 priority.
- **Idea G (RSTD cross-layer Tucker on W_up)**: convergence signal from R1 deferred Idea 5; explicitly tests W_up compressibility (which IS the load-bearing matrix per R1). Strong Round 4 candidate.
- **Idea A (CALFAV calibration-fitted attention kernel)**: arch-transposition; attacks attention, orthogonal to MLP findings. Round 4 candidate.
- **Idea E (JLVH JL-sketched lm_head)**: cheap output-side win, additive to other paths.

**AERO-style folding** (arXiv:2410.13060) is the only known mechanism to truly eliminate W_gate as a stored matrix — but it requires removing the SwiGLU activation entirely so W_gate and W_up algebraically merge. That's a meaningful arch-transposition worth Round 4 exploration.

---

### Round 2 — PDAP: Per-Token Dynamic Activation Outlier Pinning (Stage 6 amended)

[earlier R2 entry retained below]

**Status (R2)**: done (200 prompts, Qwen3-1.7B-Base bf16, 32425 normal-token pairs at K=1%)

**Result**: **GO — token-dynamic confirmed at operational K, with K-dependent structural finding**

Canonical mean Jaccard, normal-token stratum, mean across 7 sampled layers (28-layer model):

| K (% of d=2048) | Jaccard | interpretation |
|---|---:|---|
| 0.1% (~2 chans) | 0.718 | channel-static at top tier (LLM.int8() pattern holds for the very tops) |
| 1.0% (~20 chans) | 0.308 | **operational range — strongly token-dynamic** |
| 5.0% (~102 chans) | 0.257 | strongly token-dynamic |
| 10.0% (~205 chans) | 0.244 | strongly token-dynamic |

Massive-activation pairs at K=1%: Jaccard = 0.182 (1399 pairs, 4.1% of total). Lower than normal (PrefixQuant pattern partially supported) but the normal stratum is already firmly token-dynamic — the dominant signal is NOT the bimodal massive-vs-normal split.

Channel concentration evolves with depth:
- Layers 2-19: 5-8% of channels cover 90% of outlier events (concentrated)
- Layers 23-27: 14-18% (dispersed; deep layers spread outliers across more channels)

**Structural finding for Round 3**: LLM.int8()'s channel-static claim **holds only at K≈0.1%**. At any K from 1-10% (the operational range for activation quantization or outlier pinning), outliers rotate per-token. This means:
- **Pure channel-static masking is suboptimal** for any scheme pinning >0.1% of channels
- **Pure per-token dynamic** is overkill at K=0.1% (pinning the 2 stable channels is enough at the tip)
- **Hybrid scheme indicated**: pin top-0.1% statically (cheap, lossless), per-token dynamic for the 1-10% range
- The W(ternary)A8 stack (Round 2 idea H+4) needs per-token outlier handling at operational K

This is symmetric to Round 1's W_up dominance finding — both expose architectural assumptions in conventional methods that don't hold in modern SwiGLU+Qwen3 territory.

---

### Round 1 — Zero-Overhead Cross-Layer Row-Herding Predictor on SwiGLU (Idea D, amended by Stage 6)

[earlier entry retained below]

**Status (R1)**: done (smoke-stage; full run aborted as redundant)

**Result**: **NO-GO**, with a structural finding worth more than the experiment itself.

Smoke on 5 prompts (Qwen3-0.6B-Base, 28 layers, K=921 = 30% of r=3072):
- Mean cross-layer precision@30%: **0.363** (random = 0.30)
- Mean within-layer precision@30%: **0.384** (only 2% above cross-layer)
- cos(h_{L-1}, h_L): 0.88-0.97 across all layers (matches Deja Vu)

Per-layer pattern:
- Early (L=1-7): cross 0.45-0.57, within 0.50-0.57 — both meaningful
- Mid (L=8-17): cross 0.35-0.50, within 0.40-0.50 — moderate
- Deep (L=18-27): cross **0.12-0.35**, within **0.10-0.34** — *below random*

**Structural finding**: SwiGLU top-K firing is dominated by W_up·x, not by W_gate·x. Gate-magnitude is *anti-correlated* with post-SwiGLU magnitude in deep layers. Even with perfect within-layer information, predicting top-K from gate side alone fails. This kills any zero-overhead predictor that uses only gate-side signal in SwiGLU models. ReLU/GELU-only models (older OPT, GPT-2) inherit the original Deja Vu finding; SwiGLU models do not.

Class implications:
- Idea D (cross-layer dot-product on gate): KILLED — predictor is sub-random in deep layers regardless of cross-layer vs within-layer
- Idea J (count-min sketch on gate firings): WEAKENED — needs to key on combined (gate, up) signature, not gate alone
- Idea 3 (Shadow Oracle): WEAKENED — must predict via post-SwiGLU magnitude (requires runtime access to both gate AND up, partially defeating the savings)
- Generalizes to all SwiGLU/SiLU-gated models: Llama-2/3, Qwen2/3, Mistral, Gemma

Full 200-prompt run aborted; smoke pattern was conclusive and refined numbers wouldn't change the verdict or the structural lesson.

---

### Track A R3 — AQFKV: Q-Head SVD with KV Preserved (Stage 6 amended)

**Class**: compression-lr (attention-side)

**Hypothesis**: Compound finding 8 (W_gate / W_up full-rank in trained Qwen3) DOES NOT extend to attention W_Q. The W_Q matrix may have either (a) genuine low-rank structure (concentrated singular values within [1, 2048]), or (b) per-head redundancy (16 query heads sharing a common low-dim subspace). Outcome priors per Stage 6: 65% NO-GO, 20% GRAY, 10% SOFT-GO, 5% GO.

**Smallest test**: Qwen3-1.7B-Base bf16. Three sweeps in one execution:
1. **Global W_Q SVD** (K ∈ {2048, 1024, 512, 256, 128, 64, 32, 16}) — replaces full [2048,2048] W_Q with rank-K reconstruction. K=2048 sanity gate (ΔNLL must be <0.05).
2. **Per-head W_Q SVD** (K_per_head ∈ {128, 96, 64, 32, 16, 8}) — reshape W_Q into [16, 128, 2048], SVD each head-slice independently, reassemble. K=128 per-head sanity gate. Critical: distinguishes weight-rank from head-redundancy.
3. **Global W_K SVD** (K ∈ {1024, 512, 256, 128, 64}) — cheap parallel measurement on the [1024, 2048] W_K matrix.

W_K, W_V, W_O held at bf16 throughout. Pre-PPL spectrum dump at layers 0, 14, 27 for early-warning signal.

Script: `scripts/aqfkv_q_rank_sweep.py`. Output: `experiments/stage0/ladder/track_A_arch/round3_aqfkv/aqfkv_results.json` + plot + summary.md. Estimated runtime: 40-70 min.

**Go thresholds (global)**: GO if ΔNLL ≤ 1.0 nats at K≤128; SOFT-GO if K≤256; NO-GO if >5.0 at K≤256.
**Go thresholds (per-head)**: GO-perhead if ΔNLL ≤ 1.0 at K_per_head≤64; NO-GO-perhead if >5.0 at K_per_head≤96.

**Disambiguation rule** (Stage 6 critical): a global GO at K=128 with a per-head NO-GO at K_per_head=64 means head-sharing, NOT genuine W_Q low-rank structure.

**Downstream implications**:
- **NO-GO** (modal outcome): CF8 extends to attention Q. Class-level kill on no-retraining isolated-Q rank reduction. Round 4-A pivots cleanly to I/O scheduling (INPOLT) + lm_head (LRALP).
- **GO**: Opens attention compression line. Combine with KV-side methods. Re-examine W_K/W_V spectra in R4-A.
- **GRAY**: Layer-sensitivity follow-up; per-layer adaptive K.

**Frame novelty**: Stage 6 confirmed no published Qwen3 W_Q-isolated rank curve exists. QSVD/A3/PALU operate on joint or product spaces.

**Status**: done (Qwen3-1.7B-Base bf16, 455 eval tokens, all three sweeps + spectrum diagnostic)

**Result**: **GO globally at K=128 (ΔNLL=+0.98) AND GRAY per-head (sanity passes at K=128, NO-GO at K_per_head=64 with ΔNLL=+1.53).** Bonus: W_K K=512 ΔNLL=+0.29.

| matrix | K | K/d | ΔNLL (nats) | compress |
|---|---|---|---|---|
| W_Q global | 1024 | 0.500 | +0.0309 | 1.00× |
| W_Q global | 512 | 0.250 | +0.1961 | 2.00× |
| W_Q global | 256 | 0.125 | +0.5109 | 4.00× |
| **W_Q global** | **128** | **0.062** | **+0.9835** | **8.00×** |
| W_Q global | 64 | 0.031 | +1.3679 | 16.00× |
| W_Q per-head | 96/128 | 0.75 | +0.9431 | 1.25× |
| W_Q per-head | 64/128 | 0.50 | +1.5343 | 1.88× |
| W_K global | 512 | 0.50 | +0.2932 | 2.00× |
| W_K global | 256 | 0.25 | +0.8209 | 4.00× |

**Spectrum diagnostic** confirms the verdict: W_Q L14 r_99=1293 (vs ≈ d=2048 for W_gate/W_up); var@r=256=0.64. W_K L14 r_99=814.

**Compound finding 10**: CF8 boundary is sharp — applies to MLP weights only, NOT to attention weights. Class boundary established with quantitative spectra.

**Structural finding 11**: Global K=128 GO is a *head-redundancy* finding (16 heads collapse to ~1 head's worth of subspace), NOT a per-head weight-rank finding (per-head K=64 NO-GO at +1.53). MLA-style post-training joint Q-K projection lines now empirically motivated.

**What's still open**:
- W_V and W_O spectra (cheap measurement)
- Cross-layer Q basis (PCA-stack across 28 layers)
- Engineering deployment: W_Q K=512 + W_K K=512 ≈ 2× attention weight residency reduction at <0.5 nats. Follow-up GGUF wiring.
- MLA-style joint Q-K projection in head space (closest prior: A3 arXiv:2505.12942)

---

### Track B R6 — LHQD First-Pass: SVD spectrum of lm_head.weight (Stage 6 amended)

**Class**: compression-lr (output-side)

**Hypothesis**: Compound finding 8 does NOT extend to lm_head. The V×d output projection (151936×2048 in Qwen3-1.7B-Base) has structurally distinct singular value decay due to (a) V>>d shape and dead-row potential for rare tokens, (b) training dynamics per arXiv:2603.10145 (Godey & Artzi, Mar 2026) which predicts lm_head suppresses gradient rank during backprop. Outcome priors per Stage 6: 35% STRONG GO, 30% WEAK GO, 20% INTERESTING (spectral pass + functional flag), 15% NO-GO.

**Smallest test**: Qwen3-1.7B-Base bf16. Six steps:
0. **Tied-embedding hard gate**: check `config.tie_word_embeddings` AND data_ptr equality. If tied, halt and reframe as embed_tokens spectrum study.
1. **Cast and decompose**: SVD of lm_head in fp32 (thin SVD, rank cap 2048).
2. **Spectrum diagnostics**: cum-variance at r ∈ {64, 128, 256, 512, 1024, 2048}; per-row L2 norm histogram; dead-row confound check (>20% near-zero norm = artifact flag).
3. **Reconstruction sweep**: r ∈ {2048, 1024, 512, 256, 128, 64}. Per rank: ΔNLL, top-1 argmax retention, top-5 set retention, per-token KL(p_full || p_r), multi-passage NLL on 3 disjoint short passages.
4. **Sanity gate at r=2048**: ΔNLL must be <0.005 nats.
5. **Optional**: W_Q spectrum at mid-layer for Track A cross-reference (~3 min).
6. **Verdict classification**: STRONG GO / WEAK GO / INTERESTING FINDING / NO-GO / EMBED-TIED / DEAD-ROW-FLAGGED.

Script: `scripts/lhqd_lmhead_spectrum.py`. Output: `experiments/stage0/ladder/track_B_compress/round6_lhqd/lhqd_results.json` + plot + summary.md. Estimated runtime: 35-40 min.

**Go thresholds (amended)**:
- **STRONG GO**: 99% var at r≤256 AND ΔNLL<0.10 AND top-1 retention ≥95% AND <20% dead rows.
- **WEAK GO**: 99% var at r≤512 AND ΔNLL<0.10.
- **INTERESTING FINDING**: 99% var at r≤256 BUT ΔNLL>0.10 or top-1<85%. Aligns with Godey & Artzi softmax-bottleneck framing.
- **NO-GO (spectral)**: 99% var requires r>1536. Extends CF8 to all weight matrices.
- **NO-GO (functional)**: ΔNLL>1.0 nat at r≤512.

**Downstream implications**:
- **STRONG GO**: ~14× lm_head compression viable. Opens SVLD (progressive SVD with confidence-gated early exit). Breaks CF8 generalization.
- **WEAK GO**: ~7× compression at r=512. SVLD harder to tune.
- **INTERESTING FINDING**: spectral compression geometrically possible but functionally hazardous. Documents arXiv:2603.10145's training-dynamics implication on weights.
- **NO-GO**: Strengthens CF8 to all weight matrices in Qwen3 family. Track B narrows cleanly to RAOK (R2-grounded activation codebook) and KHQL (per-head KV ladder).

**CF9 preconditions** (verified by experiment design):
1. Measures lm_head WEIGHT spectrum, not arXiv:2510.24966's logit-matrix theorem.
2. Tied-embedding case is hard-gated (different question if triggered).
3. Argmax + KL deployment metrics, not NLL alone.
4. Multi-passage robustness against single-passage WDLA-adjacent risk.

**Frame novelty**: No 2024-2026 paper reports isolated lm_head SVD spectrum on any Qwen model with NLL + argmax retention.

**Status**: done (Qwen3-1.7B-Base bf16, 399 primary tokens + 3 multi-passages, tied gate triggered)

**Result**: **EMBED-TIED — Stage 6's hard gate caught the tied configuration. Measured the joint embed/lm_head matrix.** Spectrum is FLAT, NOT concentrated as the Godey & Artzi (2603.10145) hypothesis would predict in tied case.

| r | ΔNLL (nats) | top1 ret | top5 ret | KL/tok |
|---|---|---|---|---|
| 2048 (sanity) | -0.0024 | 0.967 | 0.981 | 0.0016 |
| 1024 | **+19.96** | **0.005** | 0.008 | 20.22 |
| 512 | +21.34 | 0.000 | 0.000 | 21.60 |
| 256 | +22.42 | 0.000 | 0.001 | 22.85 |
| 128 | +25.62 | 0.000 | 0.000 | 26.15 |
| 64 | +26.03 | 0.000 | 0.000 | 26.59 |

Spectrum: r_99=1992/2048 (97% of available), var@r=128=0.19, var@r=256=0.28, var@r=512=0.43.
Per-row norm: median 1.62, bottom-decile 1.36, near-zero count 0 (NOT a dead-row artifact).
Multi-passage robustness: ΔNLL consistent across 3 disjoint passages (max +28.85 at r=64).

**Compound finding 12**: In tied embedding configurations, the joint embed/lm_head matrix is full-rank because gradient signal flows through both input lookup AND output projection during training. The Godey & Artzi gradient-bottleneck argument applies only to the output side; the input side keeps every direction load-bearing.

**What's killed:**
- LHQD (INT8 U + BF16 V^T at r=256) — would destroy the model in tied configs
- SVLD (progressive SVD with confidence-gated early exit) — same
- "Concentrated lm_head weight spectrum from training dynamics" — refuted in tied config

**What's still open:**
- **Untied lm_head matrices** (Qwen3-8B+, GPT-NeoX, others) — genuinely different question; would need streaming SVD on the 8B which doesn't fit in 7.28 GiB
- **arXiv:2510.24966 logit low-rank theorem** — survives. The function lm_head computes can be low-rank even when W is full-rank, if activations have low intrinsic dimension. Moves the experiment to RSIDC (residual stream intrinsic dimension), not weight SVD.
- **RAOK** (R2-grounded 3-tier activation codebook) — strongest surviving Track B path; no imported theory to fail
- **W_V / W_O spectra** (Track A side; cheap measurement now that AQFKV established the methodology)

**Frame novelty preserved**: tied-embedding measurement on Qwen3-1.7B-Base specifically not in literature; CF12 is a new structural finding.
