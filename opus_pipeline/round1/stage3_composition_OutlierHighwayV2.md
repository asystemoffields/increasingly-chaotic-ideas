# Stage 3 Composition v2 — The Outlier Highway is the Q-Projection's Native Channel

## 1. Path Selection — A (refine) with structural reweighting

PATH A. M2 ⊗ M5 ⊗ M6 jointly carry differentiation no surviving 2-paper composition reaches. Title shift: alignment claim moves to headline.

## 2. Refined Core Claim

Residual stream factors as h_ℓ = h_hwy ⊕ h_bus ⊕ h_bulk, **simultaneously** an information-rate decomposition (M6: I(h_hwy)+I(h_bus) ≥ 0.85·I(h) at ≤1.1% cardinality), a temporal-stability decomposition (M5: h_hwy basis Jaccard ≥ 0.7 across sessions; h_bus Jaccard ≈ 0.3 within session, basis-stable but index-volatile), and a **W_Q-aligned** decomposition (M2: dim(span(D^1%) ∩ rowspan(W_Q^shared)) / |D^1%| ≥ 0.5, principal angles concentrated below π/4).

The non-obvious mechanism is (3). Channels through which information flows are the same channels the head-shared Q-projection attends to. AQFKV (head-shared K/V latent) is not architectural happenstance — model converged on outlier channels as native query substrate. Outliers and attention geometry **co-evolved during pretraining**, alignment detectable post hoc.

Routing on outlier identity ≡ routing on attention-relevant identity, **for free**, no learned predictor, no auxiliary signal. Differentiator vs PowerInfer + Massive Activations: those compose to "outliers predict hot weights via trained MLP." We claim outliers ARE the predictor with relation given by W_Q's rowspace, not by training.

## 3. Cascade Reordered

**E1 (load-bearing, ~4 CPU-hrs).** Qwen3-4B-Thinking layers {4, 12, 20, 28}. Compute principal angles between top-1% outlier channel basis (D^1%, ~30 channels over 10k WikiText tokens) and rowspan(W_Q) per head, then for head-shared component via SVD across heads.
- Predict: median principal angle ≤ 35° for head-shared component; intersection-fraction ≥ 0.5 (relaxed from 0.6 absorbing prior-art realistic range).
- Decisive: kills the composition if returns 75°.

**E2 (concentration, ~2 CPU-hrs).** Project h onto orthogonal complement of span(h_hwy ∪ h_bus) — zero h_bulk in outlier basis — measure ΔNLL on held-out 1k tokens × 8 prompt categories.
- Predict: ΔNLL ≤ 0.15 nats.
- Inverse: zero h_hwy ∪ h_bus, keep h_bulk; predict ΔNLL ≥ 1.5 nats.

**E3 (persistence, ~30 min × 5 sessions).** Run same 1k-token corpus through model in 5 independent process invocations, log h_hwy basis indices (top-K=0.1%) per session.
- Predict: pairwise Jaccard ≥ 0.65; max member-shift ≤ 2 channels.
- Excludes position-fixed (BOS-only) channels per CF9 concern. Confirms L1d-pinning premise.

**E4 (prefetch lead time, ~1 hr).** Wall-clock between (h_bus computed at end of attention sub-layer ℓ) and (W_down,ℓ first read by MLP). On Ryzen CPU-only.
- Predict: ≥ 80μs lead time per layer at d=3072, MLP=8192. Combined with realistic NVMe page-fault budget determines feasibility of bus-driven prefetch.

**E5 (gather primitive, ~2 hrs).** Bench three implementations of 1%-channel gather: (a) PEXT/PDEP chained, (b) sorted-index scalar loop, (c) AVX2 gather with precomputed mask.
- Predict: at K=20-30 (1% of d=3072), (b) wins by 1.5-3×; at K=200 (10% safety margin), (a) wins. Pick per-layer based on measured K_eff.

E1→E2→E3 are science. E4→E5 are engineering. Run E1 first; if intersection-fraction < 0.3 or median angle > 60°, abandon and regenerate.

## 4. Defense Against Crowded Compositions

**Crowd 1: Systematic Outliers + FlexGen.** Differentiator: W_Q-alignment claim. Persistence + W_Q-rowspace coincidence makes routing protocol attention-native; prefetch decisions and attention computation share state.

**Crowd 2 (dangerous): PowerInfer + Massive Activations.** Two-pronged differentiator:
(a) Zero-cost predictor: h_bus IS routing signal, not input to predictor. PowerInfer pays forward pass through predictor MLP; we pay nothing because h_bus already computed as part of residual.
(b) AQFKV alignment (M2): no published work measures principal-angle entanglement between outlier span and W_Q^shared.

**Crowd 3: PrefixQuant + MLA.** Differentiator: descriptive of vanilla transformers, not prescriptive of architecture variant. We don't redesign architecture (MLA) or requantize (PrefixQuant); observe alignment in pretrained models having neither.

## 5. Page-Fault Reality Budget

v1 budget assumed 5.6μs page-fault. Realistic Windows 11 NTFS 100-300μs on consumer NVMe. At 80 layers × 2 MLP weight tensors × 70B-class, 100μs/fault = 16ms/token I/O serialization — incompatible with 3-5 tok/s.

Three responses, increasing aggressiveness:

**(R1) Drop lazy-fault model. Use overlapped ReadFileEx with FILE_FLAG_NO_BUFFERING + FILE_FLAG_OVERLAPPED.** Prefetch as queued async I/O, not on-demand fault. With ≥80μs lead time and 3.5 GB/s sustained NVMe, 32MB W_down chunk arrives ~9ms. Per-layer MLP compute Q4 ~12-20ms for 70B-class. **I/O fits inside compute** iff E4 succeeds.

**(R2) Tier the bulk.** Not every MLP weight goes to NVMe. Reserve DRAM for top-30% most frequently h_bus-attended W_down rows (measurable per-layer offline). NVMe holds cold 70%. PowerInfer-shaped but routed by h_bus identity not trained predictor — differentiation preserved. Effective fault rate per token: 30% of W_down → 3.3× compute headroom.

**(R3) Revise ambition.** 70B at 3-5 tok/s → **2-3 tok/s on 70B at Q4 reach goal**, **5-8 tok/s on 32B probable**. 32B at 5 tok/s on CPU-only no-GPU not currently published. 70B target reach contingent on R1+R2 fitting jointly.

## 6. Operational Definitions

- **Concentration (M6, direct):** Let h_bulk = h − Π_{D^1%} h. Predict E_x[NLL(model | h_bulk ← 0)] − E_x[NLL(model)] ≤ 0.15 nats averaged over WikiText-103 validation, layers ablated independently per forward.
- **Alignment (M2, direct):** U = top-K left singular vectors of stacked W_Q^{(h)} across heads (head-shared subspace, dim ~128). V = canonical basis vectors for D^1%. σ_i = singular values of U^T V. Predict Σ σ_i² / |D^1%| ≥ 0.5; equivalently, median principal angle ≤ 35°.
- **Persistence (M5, direct):** Inter-session Jaccard of D^0.1% ≥ 0.65; channel identity stable to ±2 members across 5 sessions on identical input distribution but distinct random sampling seeds. Position-fixed (BOS-only) channels EXCLUDED from D^0.1% (addresses CF9: massive activations may be position-fixed, we count channel identity over non-trivial token positions).

## 7. Path / Changes from v1

1. **Front-loaded M2.** 0.6 alignment claim moved to load-bearing first-experiment; relaxed to 0.5 absorbing prior-art realistic range; median-principal-angle ≤ 35° as geometric counterpart.
2. **Demoted M1/M3/M4.** Decomposition taxonomy, PEXT/PDEP gather, NVMe routing → engineering substrate beneath M2⊗M5⊗M6 scientific core. Prior art acknowledged; no longer claim novelty for taxonomy.
3. **Reordered cascade.** E1 (alignment) first and decisive vs v1 infrastructure-first.
4. **Direct measurement for M6.** v1 cited KVQuant; v2 specifies direct ablation (zero h_bulk, ΔNLL) at ≤0.15 nats threshold.
5. **Differentiation from PowerInfer + Massive Activations made explicit** (zero-cost predictor + AQFKV alignment).
6. **Acknowledged "Rethinking the Outlier Distribution"** (arXiv:2505.21670) as closest observational precursor.
7. **Page-fault budget revised.** 5.6μs → 100-300μs realistic. R1 queued ReadFileEx + R2 tiered bulk introduced. Target: 70B 3-5 → 70B 2-3 tok/s reach / 32B 5-8 tok/s probable.
8. **Title shift.** Alignment claim in headline.

V2 composition stands or falls on E1 returning non-trivial alignment signal. Cheap, single afternoon on existing Qwen3-4B checkpoint. Run E1 first.
