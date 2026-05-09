# Stage 1 — Reach (R) — Run 009

Orientation: R — Reach
Track mix: 2 Track A, 2 Track B, 1 Track A/B cross
KILL applied: cross-layer W_Q, arbitrary rotation, within-layer Q/K joint rank reduction, softmax × RoPE temporal differencing.

---

## R9-R1 — ATQKV — Attention-Weight Cascade (W_Q → W_K → W_V → W_O full-spectrum audit)

**R / Track B**

### Mechanism

Track B compression cascade. CF11 confirmed W_Q at global K=128 gives ΔNLL=+0.98 and W_K at K=512 gives +0.29; W_V and W_O spectra are UNTESTED. If W_V and W_O have r_99/d in the 0.6–0.8 range (predicted by the pattern W_Q < W_K < MLP), then compressing all four attention weight matrices per layer is the cleanest Track B path to DRAM-resident 70B without touching MLP weights at all. The mechanism is: for each layer, replace W_Q at K=128 (8× compression, +0.98 nats), W_K at K=256 (4× compression, +0.82 nats), W_V and W_O at empirically-determined K producing ΔNLL < 0.3 each. These are parallel compressions on distinct matrices with no destructive interaction (each matrix affects an independent weight-stream). The attention weight budget is 4 × d_model × d_head × n_heads = 4 × 2048 × 128 × 16 ≈ 2.1B params/layer on Qwen3-1.7B, 4 × 4096 × 128 × 32 on 70B-class. Compressing attention weights alone at mean 4× across all four matrices yields 3× reduction on the attention-weight slice. For a 70B model where attention weights are approximately 33% of total weight bytes, 3× on that slice = 2× net on total weights (from 70B bf16 = 140 GB down to ~100 GB for that slice; in practical terms at 3–4 bpw baseline, a further 2× on the attention slice saves ~15–20% residency). The cascade's payoff compounds with RAOK-style quantization on MLP; the Reach claim is that joint W_Q@128 + W_K@256 + W_V@K_V + W_O@K_O compression plus RAOK tiering on activations closes within reach of DRAM-resident 70B at 4 bpw = 5.35 GiB if K_V, K_O turn out favorable (≥4× each).

Does not collide with: CF8 (MLP weights, not these); v2-CHEAP-TEST-001 (cross-LAYER W_Q killed, this is per-layer within-matrix SVD, not stacking W_Q across layers); v2-S3-R004-001 arbitrary rotation killed (this is a non-rotation SVD truncation, no RMSNorm conflict). Does not hit per-head W_Q kill (this is GLOBAL W_Q truncation, which is CF11 GO at K=128).

### Residency arithmetic

Target: Qwen3-72B-equivalent at ≤7.28 GiB.
- Qwen3-72B at IQ4_XS (GGUF) ≈ 43 GB (estimated from ~0.55 bpw × 72B params = 4.95 GB / 10B params baseline × 7.2 = 35.6 GB; realistically ~43 GB at IQ4_XS).
- DRAM resident = 7.28 GiB; gap = ~36 GB from NVMe, 11.5 GB/s DRAM bandwidth ceiling.
- Current DRAM path: ~11.5 GB/s / 43 GB × 1 token = 0.27 tok/s — unusable.
- Attention-weight compression target: W_Q×8, W_K×4, W_V×4, W_O×4 (pending measurement; conservative). Attention slice is ≈4 × d_model × d_kv × n_layers / total_params. For 72B: ~40% of weights are attention. 4× mean on attention slice = 2.7× net compression on those bytes, saving ~20 GB.
- Combined with 2-bpw MLP quantization (RAOK-style), total weight bytes at ~2.5 bpw average = 72B × 2.5 / 8 = 22.5 GB — still NVMe territory.
- DRAM-resident floor: at ~0.55 bpw (NanoQuant) = 4.95 GB. Adding attention cascade's ~1.5× multiplicative on the NanoQuant floor → ~3.3 GiB — comfortably DRAM-resident with 7.28 GiB budget.
- Realistic path: IQ4 baseline + attention cascade 2× on attention slice = 43 GB → ~33 GB; RAOK tiered activation quantization (separate mechanism, ~1.2× throughput). Still NVMe tier. Full path to DRAM needs NanoQuant + this cascade stacked.
- Tok/s at DRAM-resident (33 GB on NVMe, 11.5 GB/s DRAM): if 7.28 GiB fits in RAM with NanoQuant + attention cascade, ~11.5 GB/s / 7.28 GB = 1.58 tok/s. At 3 tok/s target: need 11.5 / 3 = 3.83 GiB — only achievable with NanoQuant (IQ4_XS 0.55 bpw = 4.95 GiB for 72B).
- Quality cost: W_Q@128 (+0.98 nats) + W_K@256 (+0.82 nats) + W_V + W_O (unknown) — sum is the critical unknown. If W_V and W_O each contribute ≤0.3 nats, total attention-side quality cost is ~2.4 nats — above target of 0.3 nats for the full system. This is the main risk. First cascade rung resolves W_V / W_O costs.

### Novelty gloss vs the kill list and the published landscape

Closest kill-list item: AQFKV (selected experiment in prior rounds). AQFKV tested W_Q and W_K. This proposal extends the sweep to W_V and W_O (the remaining two uncharacterized attention weight matrices) and proposes a four-matrix joint deployment as a compression cascade — novel in that the full-four-matrix interaction quality cost has not been measured.

Closest published: MLA (DeepSeek-V2) — trains a joint low-rank KV projection from scratch. This is post-training, no-SGD, targeting the same matrices via SVD measurement without retraining. Also adjacent to A3 (arXiv:2505.12942) which uses attention-logit-error minimization — structurally related but this is a full-spectrum audit + deployment cascade.

The frame-novel element is the cascade arithmetic: four matrices measured jointly, quality costs summed, residency arithmetic closed on this stack.

### Smallest experiment

Claim: W_V and W_O have r_99/d ≤ 0.80 in Qwen3-1.7B (consistent with W_K's 0.79 pattern), and joint W_Q@128 + W_K@256 + W_V@K_V + W_O@K_O produces total ΔNLL ≤ 1.5 nats.

Test: SVD spectrum of W_V and W_O (Qwen3-1.7B-Base bf16) — same protocol as AQFKV. Plot singular value decay. Then measure ΔNLL at K_V ∈ {128, 256, 512} and K_O ∈ {128, 256, 512} with W_Q fixed at K=128 and W_K fixed at K=256.

Runtime: ~30 min (two spectra + grid of ΔNLL evaluations, each ~5 min on Ryzen 5 7530U).
Go threshold: W_V r_99/d ≤ 0.85 AND W_O r_99/d ≤ 0.85 AND joint ΔNLL ≤ 1.5 nats.
NO-GO structural finding: W_V and/or W_O are full-rank like MLP weights — extends CF8 boundary characterization; kills attention-weight cascade as a Reach path; confirms MLP is the true bottleneck even within attention.

### Primary risk

W_V is full-rank (r_99/d ≈ 1.0) because it directly indexes the value content — full-rank preservation of all value directions may be needed. Mitigation: test W_O first (purely linear projection without value semantics); if W_O is concentrated, partial cascade (W_Q + W_K + W_O) still yields compression even if W_V fails.

---

## R9-R2 — RAOK-REACH — RAOK Tiered Activation Codebook as 70B Cascade Anchor

**R / Track B**

### Mechanism

Track B. RAOK (R2-Aware Outlier-Keyed Activation Codebook) was deferred as the strongest surviving Track B path in R6 and R4. This Reach proposal frames RAOK not as a standalone measurement but as the anchor rung of a 70B cascade. The mechanism: CF3 and v2-CF1 establish that at K=0.1% (~2 channels per 2048), outlier channel identity is near-static (Jaccard 0.718); at K=1% (~20 channels), it is token-dynamic (0.31). The RAOK tiering exploits this: 2 channels pinned at FP16 statically (identified once at calibration), next 18 channels (K=1% minus K=0.1%) handled with INT8 dynamically per token, remaining 2026 channels at INT4 statically. This is an activation representation compression, not a weight compression — it reduces the effective bits consumed per GEMV call by narrowing the activation storage format. The Reach framing: RAOK's activation tiering reduces per-token DRAM read bandwidth by a factor proportional to the activation-dominated portion of compute at 70B scale. At 70B on NVMe (43 GB weight bytes), the bandwidth cost is in the weight stream (weights dominate activations by ~70B/4096 per layer). However, RAOK's contribution is to enable a higher-compression weight format by matching the activation precision to the quantization grid: if activations entering each GEMV are in a known mixed-INT format, the weight quantization grid can be chosen to match, eliminating the precision mismatch that drives quantization error at 2–3 bpw. This is a precondition enabler, not a standalone multiplier. The cascade: RAOK tiered activations (rung 1, establishes precision matching) + 2-bpw weight quantization tuned to RAOK activation format (rung 2, leverages rung 1 to achieve lower quantization error) + attention-weight compression from ATQKV (rung 3, orthogonal).

CF anchors: CF3 (K-dependent Jaccard, load-bearing), v2-CF1 (all 28 layers, per-layer generalization). Does not rely on CF13/14/15.

### Residency arithmetic

RAOK itself does not reduce weight bytes — it changes activation representation. The Reach claim is indirect:
- At 2 bpw weight quantization, current quantization error comes partly from activation outliers mismatched to the INT2 grid.
- RAOK tiering reduces outlier-driven quantization error, enabling 2 bpw quality to match current 3–4 bpw quality.
- Net: equivalent quality at 2 bpw instead of 4 bpw = 2× weight residency reduction.
- 72B at 4 bpw = 36 GB → at effective 2 bpw = 18 GB — still NVMe, but at 11.5 GB/s DRAM would be 0.63 tok/s resident. NVMe path: 18 GB / NVMe 3 GB/s = 0.17 tok/s streaming (CF13*-conditioned).
- Combined with NanoQuant-class 0.55 bpw target: 72B × 0.55 / 8 = 4.95 GiB. RAOK as enabler for NanoQuant quality preservation = DRAM-resident at 4.95 GiB, 11.5 GB/s / 4.95 GB = 2.32 tok/s. Approaches 3 tok/s target.
- Quality cost: depends on RAOK's precision-matching benefit at 2 bpw. Measurable as ΔNLL(2bpw + RAOK) vs ΔNLL(2bpw without RAOK). Target: ΔNLL recovery of ≥0.3 nats from activation tiering.

### Novelty gloss vs the kill list and the published landscape

Closest kill-list item: PDAP three-tier on W_up (deferred in R4, untested). PDAP targets weight quantization directly; RAOK targets activation quantization and uses it to enable weight quantization improvement. Structurally different: PDAP is weight-codebook; RAOK is activation-tiering as precision-matching enabler.

Closest published: SmoothQuant (arXiv:2211.10438) migrates quantization difficulty from activations to weights via per-channel scaling. RAOK does not use per-channel scaling — it uses a three-tier dynamic partition based on the measured K-dependent Jaccard structure. The structural difference is that RAOK's tier boundary is empirically derived from CF3's Jaccard crossover (K=0.1% static, K=1% dynamic) rather than a calibration-fitted scale.

LLM.int8() channel-static assumption is empirically broken at K>0.1% (CF3 confirmed). RAOK is the first activation-quantization scheme directly grounded in the K-dependent Jaccard measurement.

### Smallest experiment

Claim: RAOK three-tier activation representation (FP16 top-2 static / INT8 top-20 dynamic / INT4 rest) reduces quantization error (measured as ΔNLL) vs uniform INT4 activation quantization by ≥0.3 nats on Qwen3-1.7B-Base at 4 bpw weight quantization.

Test: implement activation quantization (three tiers as described) in Python/PyTorch evaluation harness on Qwen3-1.7B-Base; compare ΔNLL vs uniform INT4 activation baseline at same weight bpw. Static channel identification from 500-token calibration set.

Runtime: ~2 hours (calibration + two NLL evaluations).
Go threshold: ΔNLL improvement ≥0.3 nats relative to uniform INT4 activation baseline.
NO-GO structural finding: outlier tiering provides no benefit at 4 bpw weight precision → quantization error is weight-dominated, not activation-dominated at this bpw; shifts priority to weight codebook design (PDAP-style).

### Primary risk

The precision-matching benefit requires the weight quantization grid to be re-optimized for the RAOK activation format — simply applying RAOK to an existing quantized model without grid adjustment provides no gain. Mitigation: test the codebook-joint version (recompute weight codebook centroids with activation-aware objective) rather than post-hoc activation tiering.

---

## R9-R3 — HSMLA — Head-Sharing MLA Post-Training Joint Q-K Projection

**R / Track A**

### Mechanism

Track A arch-transposition. CF11 established that 16 W_Q heads collectively span a ~128-dim subspace — a 16:1 head-redundancy ratio. The mechanism is to exploit this by replacing the per-head W_Q stack with a shared low-rank basis across all heads, in the style of MLA (DeepSeek-V2), but post-training and without SGD. MLA at training time learns a joint W_Q^{compress}: R^{d_model} → R^{d_KV_compress} followed by per-head uncompressor W_Q^{uncompress,h}: R^{d_KV_compress} → R^{d_head}. Post-training: if the 16 heads' W_Q matrices already span a ~128-dim joint subspace (CF11's head-redundancy finding), we can extract that basis via stacked SVD of [W_Q^1; W_Q^2; ...; W_Q^{16}] (shape 16×d_head × d_model = 2048 × 2048 in Qwen3-1.7B), take top-128 left singular vectors as the shared basis B ∈ R^{d_model × 128}, and express each head as W_Q^h ≈ W_Q^h · B · B^T (Frobenius approximation). At inference, compute q_shared = B^T · x (one 128-dim projection), then q_h = B^h · q_shared for each head where B^h = W_Q^h · B is the per-head extraction (precomputed). Cost: one d_model × 128 projection instead of 16 × d_head × d_model projections = 128/2048 = 6.25% of original W_Q compute. Weight storage: B (128 × d_model) + 16 × B^h (d_head × 128) = 128×2048 + 16×128×128 = 262144 + 262144 = 524288 params vs original 16 × 128 × 2048 = 4194304 params — 8× reduction. This is the cascade anchor: W_Q arch-transposition yielding 8× on the W_Q weight slice with an algebraic justification (head-redundancy is a structural fact, not a statistical approximation). The key algebraic constraint is that B be an exact or near-exact reconstruction of the head-shared subspace under Frobenius; the CF11 measurement (K=128 global GO at +0.98 nats, per-head K=64 NO-GO at +1.53 nats) implies the 128-dim shared basis IS load-bearing but the per-head 64-dim subsets are not individually sufficient — consistent with the shared basis interpretation.

Not the cross-layer W_Q kill (v2-CHEAP-TEST-001 killed cross-layer stacking; this is within-layer cross-head factorization). Not arbitrary rotation (v2-S3-R004-001; this factorization does not involve O(d) arbitrary rotation — it is a structured projection). Not per-head rank reduction (CF11 kills per-head K=64; this uses the global shared basis, which CF11 confirms as GO at K=128).

### Residency arithmetic

W_Q contribution: for Qwen3-1.7B, each layer's W_Q is 16 × 128 × 2048 = 4.19M params = 8.39 MB bf16. With HSMLA: B = 128 × 2048 = 262K params + 16 × B^h = 262K = total 524K = 1.05 MB bf16. Saving: 7.34 MB/layer × 28 layers = 205 MB. At 70B scale (d_model=8192, 64 heads, d_head=128): W_Q per layer = 64 × 128 × 8192 = 67M params = 134 MB bf16. With HSMLA and K_shared=512 (head-redundancy ratio 64:1 would give K_shared=128, but 70B's multi-head structure may have different ratio): ~2× saving on W_Q slice. At 70B, W_Q is ~10% of total attention weights (~4% of total weights). 2× on 4% = ~2% net — not a Reach-level saving alone.

Cascade: HSMLA (W_Q 8×) + ATQKV W_K compression (4×) + W_V, W_O compression (pending) + NanoQuant MLP (0.55 bpw). The Reach value of HSMLA is its algebraic quality: the W_Q reduction is structurally exact on the measured subspace, with known quality cost (≤+0.98 nats, matching CF11's GO result), and it composes cleanly with other attention compressions because the shared-basis projection is a linear map that does not interact destructively with W_K.

For 70B DRAM-resident: NanoQuant (4.95 GiB) + HSMLA saves negligible absolute bytes on a 70B model (attention slice small relative to MLP). The Reach cascade here is a quality-arithmetic win: replacing W_Q with an algebraically exact shared-basis projection incurs quality cost at training distribution, but the CF11 measurement shows the subspace IS low-dimensional — the cost is bounded at +0.98 nats.

Quality cost: +0.98 nats (CF11-grounded, the global K=128 measurement IS this decomposition's quality ceiling).

### Novelty gloss vs the kill list and the published landscape

Closest kill-list: AQFKV tested W_Q SVD in isolation. HSMLA is an architectural replacement — it replaces the per-head projection with a factored shared-basis inference-time computation, not merely truncating singular values.

Closest published: MLA (DeepSeek-V2, arXiv:2405.04434) — trains joint Q-K low-rank from scratch. Post-training variant via CF11's measured head-redundancy is novel. A3 (arXiv:2505.12942) uses attention-logit-error minimization; HSMLA uses Frobenius-optimal shared-basis extraction, a different objective. The key structural difference: HSMLA is algebraically motivated by the measured head-redundancy ratio (16:1 compression of the 16-head subspace to 128 dims) rather than by a training objective.

### Smallest experiment

Claim: the top-128 left singular vectors of stacked [W_Q^1; ...; W_Q^{16}] in Qwen3-1.7B-Base reconstruct each per-head W_Q^h with Frobenius relative error ≤15% (consistent with CF11's +0.98 nats global result), and the HSMLA inference computation (shared B^T projection + 16 B^h uncompressors) gives the same ΔNLL as the AQFKV K=128 global result (≤+1.1 nats) on Qwen3-1.7B.

Test: stack W_Q matrices (28 × 16 × 128 × 2048), compute top-128 SVD, extract shared basis, compute per-head reconstruction error, measure ΔNLL on 512-token eval. PyTorch/NumPy. 

Runtime: ~45 min (SVD of 4M × 2048 stack: feasible on CPU with numpy.linalg.svd in blocks; eval ~10 min).
Go threshold: ΔNLL ≤+1.1 nats (within 10% of AQFKV K=128 result).
NO-GO structural finding: Frobenius reconstruction error is large → the 16 heads' W_Q matrices do NOT share a common 128-dim subspace despite the rank measurement; the CF11 global-K result is not decomposable into a per-head shared basis. This itself is a structural finding about the geometry of the multi-head subspace.

### Primary risk

The CF11 global-K=128 GO measures the global matrix's spectrum, not the geometry of the 16 per-head matrices stacked together. The joint subspace might be 128-dim in aggregate but with each head needing its own 64-dim slice that doesn't compose into a shared 128-dim basis. Mitigation: compute the 16 per-head Grassmannian principal angles first (5 min) before running the full ΔNLL evaluation; early abort if principal angles between heads are all close to 90°.

---

## R9-R4 — NVME-ATTN-STREAM — NVMe-Resident Attention Weight Streaming with AVX2 Scatter-Gather

**R / Track B**

### Mechanism

Track B, systems-grounded. The confirmed finding from CF11 is that W_Q is compressible to K=128 global (8× compression, 524K params/layer) while W_V and W_O are untested. The hypothesis motivating this Reach proposal: even if W_V and W_O are NOT compressible (full-rank, like MLP), the attention weight slice can be handled with a different streaming strategy from the MLP slice. The mechanism: pack per-layer attention weights (W_Q SVD basis + W_K at K=256 + W_V bf16 + W_O bf16) into a contiguous NVMe extent, ordered by token-step dependency. During autoregressive decode, each token requires one layer's attention weights before it requires the same layer's MLP weights. The streaming schedule: load attention weights for layer L while computing MLP for layer L-1 (overlapped I/O). This requires the GGUF weight layout to interleave per-layer attention-then-MLP rather than the default all-attention-then-all-MLP layout. On Windows 11 with NTFS, sequential NVMe reads achieve ~3 GB/s (CF13* — unverified on this stack; the first cascade rung must re-derive this). For a 70B model, per-layer attention weight bytes ≈ 4 × 8192 × 128 × 64 / 8 × 4 bpw = 2 GB/layer; MLP bytes ≈ 3 × 8192 × 28672 / 8 × 4 bpw = 4.4 GB/layer. Total per token = 80 layers × (2 + 4.4) GB = 512 GB — far above any streaming budget. At 0.55 bpw: 80 layers × (0.14 + 0.31) GB = 36 GB/token still exceeds DRAM. The Reach arithmetic only closes at NanoQuant territory. BUT: the interleaved layout's value is that it enables prefetching attention weights for layer L+1 during the MLP compute of layer L, hiding latency. The throughput gain is the fraction of time attention weights arrive "free" due to compute overlap. At 70B 0.55 bpw, per-layer attention = 0.14 GB, per-layer MLP = 0.31 GB; MLP compute time at 4096×28672 GEMV on Ryzen 5 7530U ≈ 0.31 GB × (CPU GFLOPS ratio / bandwidth ratio) — if MLP compute time ≥ attention load time, attention arrives free. MLP at 0.55 bpw = 0.31 GB; at NVMe 3 GB/s = 103 ms load; attention = 0.14 GB = 47 ms load — attention can be fully prefetched during MLP load. Net gain: eliminates attention weight I/O from the critical path. Attention is 31% of total weight bytes; hiding it = theoretical 1.44× throughput improvement at NVMe-bottleneck 70B.

Re-derives CF13* (NVMe read throughput) as its first cascade rung.

### Residency arithmetic

70B at 0.55 bpw = 4.95 GiB → DRAM-resident. NVMe streaming only needed above 7.28 GiB.
70B at 2 bpw = 17.5 GiB → NVMe; at NVMe 3 GB/s = 5.83 s/token = 0.17 tok/s (unusable).
With interleaved attention prefetch (hiding 31% attention weight load): effective bandwidth = total_weight / (total_weight - attn_weight + max(mlp_load_time, attn_load_time)) ≈ 1.44× at NVMe; 0.17 tok/s × 1.44 = 0.24 tok/s — still unusable at 2 bpw NVMe.

This proposal's Reach value is not standalone — it is a cascade component that contributes 1.44× throughput improvement when the model is NVMe-tier (2–4 bpw range). At 0.55 bpw DRAM-resident, the throughput is DRAM-bandwidth-limited (not NVMe), and this mechanism is irrelevant. The cascade target: 70B at 1.5 bpw NVMe-resident → 70B × 1.5 / 8 = 13.1 GiB → NVMe; with prefetch 1.44× = achieves ~0.37 tok/s at 1.5 bpw NVMe. Approaches 3 tok/s only if NVMe bandwidth is substantially higher than CF13*'s 3 GB/s value, or if model fits in DRAM.

This is an honest framing: the mechanism produces a real 1.44× improvement on the NVMe-tier path but does not close the 3 tok/s gap alone; it is a rung in a larger cascade.

Quality cost: zero (layout reorder, no mathematical approximation applied to weights).

### Novelty gloss vs the kill list and the published landscape

Closest kill-list: NVPF (NVMe Layer Prefetch via Static Access Fingerprint, Track B R6 deferred). NVPF was deferred because it "produces a single scalar baseline rather than a structural finding." This proposal extends beyond NVPF by specifying the interleaved layout scheme and the arithmetic for the compute-overlap gain. The structural novelty is the per-layer attention-MLP interleave producing a deterministic prefetch schedule derivable from the model's own layer structure.

Closest published: Apple LLM-in-a-flash (arXiv:2312.11514) exploits sequential NVMe reads for transformer inference. That work uses sliding-window weight caching; this proposal uses a different schedule — interleaved per-layer attention-then-MLP layout matched to the autoregressive execution order. The structural difference is the attention-MLP interleave (enabled by CF11's demonstration that attention weights are smaller than MLP weights and can be prefetched within the MLP compute window).

### Smallest experiment

CF13* re-derivation (first rung, required): Measure NVMe sequential read throughput on the actual Ryzen 5 7530U stack under ik_llama.cpp load (not OS benchmark — actual inference weight load). Run a timed weight load of a 1 GB GGUF file with mmap sequential access pattern; record GB/s.

Runtime: ~10 min.
Go threshold for CF13* re-derivation: ≥2.5 GB/s sequential NVMe read under inference workload.
NO-GO finding: NVMe reads under inference are slower than 2.5 GB/s (e.g., due to random-access patterns from non-sequential GGUF layout) → the prefetch overlap arithmetic doesn't close; priority shifts to DRAM-resident path exclusively.

Second rung (if CF13* GO): implement interleaved per-layer attention-MLP weight layout in GGUF format (Python script to repack); measure inference tok/s on Qwen3-1.7B at bf16 NVMe-simulated load vs standard GGUF layout. Runtime: ~4 hours.
Go threshold: ≥1.3× tok/s improvement with interleaved layout.

### Primary risk

The compute-overlap gain assumes MLP GEMV time ≥ attention weight load time. At low bpw (0.55–1 bpw), weights are bandwidth-limited and attention weights load in 47 ms/layer while MLP loads in 103 ms/layer at NVMe 3 GB/s — the overlap holds. But at DRAM-resident regime, DRAM bandwidth dominates and the NVMe prefetch path is never engaged. The mechanism is only useful in the NVMe-resident regime (>7.28 GiB), which requires 70B at ≤1 bpw to reach DRAM-resident and >0.55 bpw to remain NVMe. Mitigation: test specifically at 2–4 bpw where NVMe path is clearly engaged.

---

## R9-R5 — LAYERONE-FOLD — Layer-1 Gate Constant Fold + Full-Stack Cascade [FREE SWING]

**R / Track A / [FREE SWING]**

### Mechanism

Track A arch-transposition, exploiting CF6's Layer-1 anomaly. CF6 confirmed that 36.1% of Layer-1 gate neurons in Qwen3-1.7B have variance < 0.05 across calibration tokens (near-constant). For these neurons, silu(W_gate^{L1}·x) ≈ c_i (constant, measurable per-neuron on calibration data). For constant gates, the SwiGLU output simplifies: silu(W_gate^{L1}·x)_i ⊙ (W_up^{L1}·x)_i ≈ c_i · (W_up^{L1}·x)_i — the gate's activation is absorbed into a per-neuron scale on W_up. This is an exact algebraic fold for near-constant neurons: W_up^{L1}_{scaled,i} = c_i · W_up^{L1}_{i,:}. For the 36% of neurons where c_i is near-constant, the gate projection W_gate^{L1}_{i,:} is deleted from the inference graph (saving one row of W_gate per foldable neuron). This saves 36% × d_intermediate × d_model weights in W_gate at Layer 1 only. On Qwen3-1.7B: d_intermediate = 6144, d_model = 2048, bf16. W_gate per layer = 6144 × 2048 × 2 bytes = 25.2 MB. 36% fold = 9.07 MB saved at Layer 1. Negligible in absolute terms (~9 MB / total ~3.4 GB). However: the FOLD IS ALGEBRAICALLY EXACT on the calibration distribution — it saves the gate projection computation for 36% of Layer 1 neurons at ZERO quality cost on the calibration distribution. The Reach framing: this is the only known zero-quality-cost weight fold confirmed by v2 measurements. It establishes a cascade template: find layers with high near-constant fraction → fold → delete → propagate. The Reach question is whether this template extends to higher layers with lower base rate if we condition on context type (e.g., specific input patterns may push layers 2–5 to higher near-constant fractions). The [FREE SWING] tag reflects that this cascade extrapolation is speculative — the base finding (Layer-1 36%) is CF6-confirmed, but the context-conditional extension to higher layers is not.

The Reach proposal: measure per-layer near-constant fraction as a function of (a) threshold, (b) context type (code vs. prose vs. math), (c) temperature. If mid-layers reach ≥20% near-constant under any context partition, a context-conditional fold (fold different neurons per context type) produces a multi-context cache of pre-folded W_up variants — each variant is valid for its context type with zero quality cost on that distribution.

### Residency arithmetic

Layer-1 fold alone: 9.07 MB / 3.4 GB = 0.27% reduction. Negligible.
Context-conditional fold at 20% rate across 28 layers: 20% × 25.2 MB × 28 = 141 MB — still <5% of 3.4 GB. Negligible on Qwen3-1.7B.

At 70B scale: d_intermediate = 28672, d_model = 8192, W_gate per layer = 28672 × 8192 × 2 = 470 MB. 36% fold at Layer 1 = 169 MB. If 20% fold at all 80 layers = 20% × 470 MB × 80 = 7.52 GB — this becomes meaningful (7.52 GB / 140 GB total bf16 = 5.4% reduction). At 0.55 bpw, 5.4% of 4.95 GiB = 0.27 GiB savings — puts the 70B model at 4.68 GiB from 4.95 GiB. Meaningful for a tight DRAM budget.

Quality cost: zero on calibration distribution (algebraically exact fold for near-constant neurons). Non-zero on out-of-distribution inputs that activate the folded neurons strongly (the c_i approximation fails). Measured as ΔNLL on held-out set vs calibration set.

### Novelty gloss vs the kill list and the published landscape

Closest kill-list: R4-A SDZC (SwiGLU Dead-Zone Gate Collapse) — KILLED globally because 1.5% near-constant rate is too low. Layer-1 anomaly was noted as interesting but isolated. This proposal focuses exclusively on Layer-1 as the first fold rung and extends to context-conditional multi-layer folds, framing the anomaly as a cascade template rather than a global scheme.

Closest published: AERO (arXiv:2410.13060) removes the activation entirely and algebraically folds W_gate and W_up into one matrix — this requires retraining. The Layer-1 fold is a weaker, no-retraining version that only applies to near-constant neurons (not the full activation). Structurally different: AERO is global and requires fine-tuning; LAYERONE-FOLD is per-neuron, no-retraining, calibration-derived, with exact quality cost bounds.

The [FREE SWING] designation is because the context-conditional extension (the Reach component) lacks empirical grounding beyond the single-layer CF6 measurement.

### Smallest experiment

Claim: at Layer 1, the 36% of near-constant neurons fold exactly (ΔNLL ≈ 0 on calibration distribution) and Jacobian-bound error on held-out ≤0.05 nats; AND at least 5 additional layers have ≥10% near-constant fraction.

Test: (a) Apply fold to Layer-1 neurons with std < 0.05 (from CF6 calibration); evaluate ΔNLL on Qwen3-1.7B held-out set. (b) Run CF6 variance probe on all 28 layers (same script, extended) to measure per-layer near-constant fraction.

Runtime: ~1.5 hours total (fold implementation + eval + extended variance probe).
Go threshold: ΔNLL ≤ 0.05 nats on held-out AND ≥3 additional layers with ≥10% near-constant.
NO-GO structural finding: fold incurs >0.05 nats on held-out → the c_i approximation fails on the token distribution; near-constant at calibration means temporally static for the calibration tokens only, not globally; limits the approach to calibration-matched deployment.

### Primary risk

The calibration-held-out generalization gap is the key risk: CF6 measured variance across 566 tokens × 10 passages — a narrow calibration. A held-out distribution that activates the "constant" neurons' token-specific variation would incur quality cost. Mitigation: measure distribution of c_i values across a wider calibration set (10K tokens) before committing to the fold; if the variance distribution shifts substantially, the near-constant fraction drops and the mechanism loses its zero-cost property.

---

## Convergence handles

- W_V and W_O spectra (untested attention matrices — multiple ideas depend on whether they follow W_K's 0.79 r_99/d pattern or MLP's 1.0 pattern)
- Activation precision-matching as an enabler for lower-bpw weight quantization (RAOK-REACH and ATQKV cascade arithmetic)
- Head-shared subspace geometry: do the 16 W_Q per-head matrices jointly span a 128-dim subspace with shared basis, or do their individual principal directions not align (HSMLA falsifiability)
- NVMe sequential read throughput under ik_llama.cpp inference workload (CF13* re-derivation required by NVME-ATTN-STREAM)
- Per-layer near-constant gate fraction as a function of context type (LAYERONE-FOLD cascade extension; CF6 measures only calibration aggregate)
