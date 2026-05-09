# Stage 1 — Orientation A (Constraint-Alien) — Run 9

Kill applied: gauge constraints producing arbitrary orthogonal rotation (F-GFQO). All ideas below avoid O(d) rotation as the operative constraint. Each binds ONE constraint hard through the full mechanism.

---

## A1-TROPFFN — Tropical-Semiring FFN Dispatch

**Orientation:** A — Constraint-Alien | **Track:** A (arch-transposition)

**Constraint:** No continuous multiplication during the MLP forward pass — only max and addition (tropical semiring ℝ_max).

**Constraint difficulty calibration:** Eliminates standard GEMV (which requires multiply-accumulate), codebook look-up with dot-product scoring, and any activation function with a continuous nonlinearity. Retains integer addition, table dispatch, argmax, log-quantized weights. Roughly 80% of standard inference primitives are inadmissible.

### Mechanism

Track A. Under the tropical semiring (ℝ_max, ⊕=max, ⊗=+), matrix-vector "multiplication" W ⊗ x = [max_j(W_{ij} + x_j)]_i. This is an argmax-weighted routing operation: the output neuron i activates at the maximum log-weight-plus-log-input across all input channels j. For log-quantized inputs (x stored as log₂|x| in INT8) and log-quantized weights (W stored as log₂|W_{ij}| in INT8), the tropical GEMV reduces to: (1) INT8 vector addition (broadcast W_row + x), (2) a horizontal max across 2048 channels — both operations achievable with AVX2 vpaddb + vpmaxsb in ~4 cycles per neuron vs ~128 FP16 FMAs for standard GEMV. SwiGLU's gate can be recast as tropical gating: gate neuron fires when its tropical product exceeds a threshold; W_up contribution selected by max over channels. This is not quantization of a standard GEMV — it is a different algebraic computation that approximates the top-K firing structure identified in CF1 (W_up·x dominates post-SwiGLU magnitude). Tropical GEMV naturally implements a "winner-take-most" computation that mimics the empirical firing concentration CF1 documents. Not CF8 MLP rank reduction; does not rely on W_gate/W_up rank structure.

### Residency arithmetic

Weights stored as INT8 log-magnitudes + 1-bit sign: 9 bits/weight. At 70B parameters: 70B × 9 / 8 = 78.75 GB — worse than bf16 baseline. The residency story is not the bit count but the GEMV throughput: tropical GEMV on AVX2 runs at ~4 cycles/neuron vs ~32 cycles for bf16 GEMV on the same hardware. On the Ryzen 5 7530U (AVX2, ~3.5 GHz, 6 cores), tropical GEMV throughput ~21B "neurons/s" vs ~6.5B for bf16. For a DRAM-resident 7B model at INT8 (7B): weight bandwidth is the bottleneck at 11.5 GB/s DRAM. Tropical doesn't help residency directly. **Payoff is compute-bound regime**: once weights are in L2/L3 (small models, hot layers), tropical dispatch is ~5× faster per neuron than FP16 GEMV. Primary metric: per-neuron throughput on AVX2 in register-hot scenario, measured in cycles/output-neuron.

### Novelty gloss

Closest kill list item: none directly. Closest published: Tropical Neural Networks (Charisopoulos & Maragos, 2020) use tropical layers as a training objective — they retrain from scratch. The structural difference here is post-hoc substitution of trained weights via log-quantization, no retraining, tested against CF1's firing-concentration structure. Tropical NN literature does not target post-training inference approximation on existing LLM weights.

### Smallest experiment

**Claim:** For a single W_up layer of Qwen3-1.7B-Base, tropical GEMV with INT8 log-magnitude weights achieves top-K neuron-firing recall ≥ 0.70 at K=5% vs bf16 standard GEMV, on the Ryzen AVX2 path.
**Runtime:** ~2 hours. Load W_up bf16 → log2-quantize → implement numpy tropical_gemv → measure recall@K=5% on 200 calibration tokens.
**Go threshold:** recall@K=5% ≥ 0.70 (would indicate tropical dispatch approximates the load-bearing firing structure).
**NO-GO finding:** Characterizes how badly tropical semiring approximates SwiGLU firing at various K — itself a structural map of the firing-concentration geometry. If recall is 0.50, tropical only approximates a winner-take-most shell.

### Primary risk

Log-quantization of weights with full dynamic range (W_up has per-row variation across ~3 orders of magnitude, CF5 implies high rank-sensitivity) may destroy the approximation accuracy. Mitigation: use per-row log-scale normalization before INT8 quantization, storing the per-row scale factor as a bf16 multiplier applied after the tropical max.

---

## A2-APPENDKV — Append-Only KV Accumulator with Idempotent Retrieval

**Orientation:** A — Constraint-Alien | **Track:** A (arch-transposition)

**Constraint:** All KV state is append-only. No mutation of existing KV entries during inference. Each new token may only APPEND to the KV log; it may not update or delete any prior entry.

**Constraint difficulty calibration:** Eliminates sliding-window KV eviction, in-place KV update, GQA sharing that mutates a cache slot, and any scheme that overwrites older tokens' K or V projections. Retains log-structured append, indexed lookup, idempotent reads. ~60% of KV management schemes are inadmissible.

### Mechanism

Track A. Standard KV cache eviction (H2O, StreamingLLM, MiniKV) selects which tokens to drop and overwrites the cache. The append-only constraint forces a different structure: the KV log grows monotonically; retrieval is a read-only operation over the full log. This forces the attention computation to become a *selective read* over an immutable log, which maps onto a content-addressable retrieval problem. Concretely: keys K_t are stored in an append-only buffer with an accompanying BLAKE3-keyed index (32-byte hash of K_t ∈ R^128 quantized to INT8). At attention time, the query Q_t hashes to the nearest-neighbor in the key index via locality-sensitive hashing (LSH) over INT8 K vectors — an operation implementable with AVX2 hamming-distance on binary projections. The append-only constraint converts the KV management problem from "which entries to evict" to "which entries to retrieve," decoupling memory layout from retrieval policy. The KV log can live on NVMe (append-only sequential writes are the fastest NVMe access pattern — full sequential bandwidth, no seek), with the hash index in DRAM. At 128K context, 70B model: KV at INT4 ≈ 2× heads × 70 layers × 128 dim × 128K tokens × 0.5 bytes = ~1.4 GB NVMe-resident, hash index ~50 MB DRAM-resident. Not CF3/CF11 dependent — orthogonal to weight structure.

### Residency arithmetic

For the 7.28 GiB Ryzen target at 4K context, KV is ~670 MB at 4B (not the binding constraint). For 32K context: KV ≈ 5.3 GB at bf16 (exceeds the budget). Append-only INT4 KV on NVMe at 32K: ~670 MB NVMe. Hash index: ~5 MB DRAM. Weight residency unchanged. This idea's payoff is **long-context enablement** (32K+ without DRAM blowup) rather than model-size compression. Metric: supported context length at 7.28 GiB, measured in tokens before OOM.

### Novelty gloss

Closest kill list item: APPENDKV is structurally distinct from KV Temporal Differencing (R1/S2, killed because RoPE rotates keys, making deltas non-small) — this idea doesn't delta-encode, it appends unmodified K/V and retrieves by hash. Closest published: InfLLM / RetrievalAttention do retrieval-based KV but permit eviction and mutation. SnapKV and H2O mutate the cache (not append-only). The constraint-forced novelty: the append-only binding forces the retrieval mechanism to be the entire policy, eliminating the eviction-policy design space.

### Smallest experiment

**Claim:** LSH-based retrieval over an append-only INT4 KV log on NVMe achieves attention-output cosine similarity ≥ 0.92 vs full bf16 KV cache on a 4K-context Qwen3-1.7B forward pass.
**Runtime:** ~3 hours. Implement append-only INT4 KV buffer + binary LSH index in Python/numpy, measure similarity vs exact attention on 50 prompts.
**Go threshold:** cosine similarity ≥ 0.92 on attention outputs (implies the approximate retrieval preserves the dominant attention pattern).
**NO-GO finding:** Characterizes approximation gap of LSH-based key retrieval at various context lengths — useful for calibrating whether ANN-based attention is viable on this model.

### Primary risk

RoPE key rotation means K_t vectors are token-position-dependent; the LSH index built on INT8-quantized K_t may not give good retrieval for Q_s at different positions even when the semantic content matches. Mitigation: index on RoPE-stripped key vectors (remove positional encoding before hashing, re-apply at retrieval) — this is a measurable design choice.

---

## A3-FSMDEC — Finite-State-Machine Decoder with N ≤ 256 States

**Orientation:** A — Constraint-Alien | **Track:** A (arch-transposition)

**Constraint:** The token-generation loop must be implementable as a finite state machine with N ≤ 256 states. The decoder's routing decisions — which computation path runs for this token — must be determined by FSM state alone.

**Constraint difficulty calibration:** Eliminates continuous per-token hidden-state routing, attention-score-dependent computation graphs, and any scheme that requires reading the full residual stream to decide which ops to run. Retains static per-state dispatch tables, precomputed state-transition functions, and any computation path that can be labeled by a symbol in a 256-symbol alphabet. ~75% of dynamic-routing schemes are inadmissible.

### Mechanism

Track A. The constraint forces a separation between *state label* (which of N=256 FSM states are we in?) and *residual stream value* (the actual continuous hidden state). The FSM state is determined by a discrete projection of the residual stream onto N clusters, computed cheaply — a single nearest-centroid lookup with N=256 INT8 centroids of dimension d=2048, costing 256 dot products per token (<<1% of one layer's compute). The FSM transition function maps (current state, current token) → next state via a 256×V table (256 × 151936 × 1 byte = 38 MB, DRAM-resident). For each FSM state, a precomputed computation policy is loaded: which attention heads are active, which FFN neurons are in the "hot" path, which layers can be early-exited. This is a learned MoE routing table, but the routing key is the FSM state (a 1-byte integer) rather than a continuous gating vector. The payoff: the per-token routing cost collapses from O(d) (continuous gating) to O(1) (FSM table lookup). The FSM-state assignment step (nearest-centroid in INT8) runs in ~10 µs on AVX2 vs ~100 µs for a standard gating network. CF3's outlier-channel structure motivates the centroid parameterization: outlier channels at K=0.1% are static (Jaccard 0.718), so the FSM state can be derived from those ~2 static channels alone — making the state assignment O(1) rather than O(d).

### Residency arithmetic

FSM routing table: 38 MB DRAM. Per-state computation policies (head masks, layer-skip flags): 256 × 28 layers × ~100 bytes = ~0.7 MB. No weight change; full-precision weights retained. Payoff: if the FSM correctly routes to a 30% sparse computation graph (30% heads/neurons skipped) for 60% of tokens, the effective DRAM bandwidth drops by ~18%. At 11.5 GB/s, a 70B model at 0.55 bpw = 5.35 GiB: 5.35 GB / 11.5 GB/s = 465 ms/token baseline; at 18% reduction: ~381 ms/token (1.22× speedup). Small absolute gain, but the routing cost goes to near-zero — no gating overhead.

### Novelty gloss

Closest kill list item: none directly. The FSM-decoder framing has no kill-list analog. Closest published: Deja Vu (arXiv:2310.17157) predicts neuron firing via a 2-layer MLP predictor; the kill-list entry (R1/S6) shows it's pre-empted for cross-layer cosine similarity but not for FSM-state-keyed routing. The structural difference: Deja Vu trains a *continuous* predictor; FSMDEC uses a precomputed *discrete* state table derived from calibration data — no predictor network at runtime. Closest living idea: SGHH (deferred Track A R3) routes based on sink heads; FSMDEC routes based on residual-stream cluster membership.

### Smallest experiment

**Claim:** 256-centroid k-means clustering of residual-stream vectors (layer 14 output, 2048-dim) on 1000 calibration tokens captures ≥ 80% of the variance in per-token firing patterns (top-30% neurons active vs not).
**Runtime:** ~1 hour. Run k-means with N=256 on calibration residual vectors; measure neuron-firing recall@30% for tokens routed to each centroid.
**Go threshold:** mean per-centroid firing recall ≥ 0.80 (implies the 256-state FSM captures most of the routing-relevant structure).
**NO-GO finding:** Quantifies the minimum FSM size needed to capture firing-pattern diversity — itself a structural measurement of the residual stream's effective state complexity.

### Primary risk

The residual stream's continuous dynamics (CF2: cos(h_L, h_{L+1}) ≈ 0.99) may mean that centroid clusters do not correspond to meaningfully distinct computation paths — adjacent tokens in the same cluster may actually require different computation. Mitigation: measure per-centroid firing-pattern variance; if within-centroid variance is large, increase N or switch from k-means to a fine-grained hierarchical clustering.

---

## A4-SEEDNET — 10 MB Generative Seed for Full-Layer Weight Synthesis

**Orientation:** A — Constraint-Alien | **Track:** B (compression)

**Constraint:** The full inference weight set must be reconstructible from a 10 MB generative seed — a deterministic procedure applied to a small seed tensor produces every weight matrix at inference time.

**Constraint difficulty calibration:** Eliminates storing trained weights directly, any quantization scheme that stores residuals from a base, and standard codebook approaches that store per-group centroids. Retains deterministic pseudorandom generation, calibration-fitted synthesis procedures, structured weight parameterizations. ~85% of standard compression schemes are inadmissible (they all store learned weight values, just more compactly).

### Mechanism

Track B. The constraint forces the question: what structure in the trained Qwen3 weights is predictable from a compact description? CF11 (W_Q r_99/d ≈ 0.63, head-redundancy 16:1) and CF12 (tied embed full-rank) give partial answers. The seed-net approach: (1) A small deterministic network G_θ (the "seed," 10 MB = ~5M fp16 parameters) takes as input a layer index L ∈ [0,27] and a role label r ∈ {W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, embed} and outputs the full weight matrix W_{L,r} ∈ R^{d×d} or R^{d×d_int}. G_θ is a hypernetwork. (2) The seed is fit post-hoc to minimize the reconstruction error ‖G_θ(L, r) − W_{L,r}‖_F across all (L, r) pairs. This is a calibration fit — but the CF10 condition (n_params ≫ n_independent_fit_dims) must be checked. For Qwen3-1.7B: total weight parameters ≈ 1.7B; G_θ has 5M parameters. The compression ratio is 1.7B / 5M = 340:1 in parameter count. The fit quality depends on how much structure the trained weights share across layers and roles. CF11 (W_Q low-rank and head-redundant) and CF2 (adjacent residual streams nearly parallel) both suggest cross-layer regularity. CF8 (MLP full-rank) suggests MLP weights are harder to predict. **CF10 check**: n_params to fit = 5M; n_independent_samples ≈ 1.7B weight values treated as signal. The fit is over-constrained (5M parameters fitting 1.7B values) — CF10's failure mode (under-constrained) does NOT apply here. The risk is under-expressivity, not overfitting.

### Residency arithmetic

Seed net G_θ: 10 MB on NVMe. At inference: G_θ generates one layer's weights at a time, held in DRAM for that layer's computation, then discarded. DRAM residency: max(one layer's weights). For Qwen3-1.7B: one layer = 28 weight matrices each ~d×d_int bf16 = ~100 MB. Plus G_θ = 10 MB. Total DRAM: ~110 MB vs full model ~3.4 GB bf16. Effective compression: 30×. For 70B at same structure: DRAM = one 70B layer (~400 MB) + 10 MB seed. **Quality caveat**: the compression ratio is brutal; reconstruction quality is the entirely open question. If ΔNLL < 1.0 nat, this is a qualitatively different point on the residency-vs-quality curve.

### Novelty gloss

Closest kill list item: none. The seed-net / hypernetwork framing does not appear in the kill list. Closest published: HyperNetworks (Ha et al., 2016) use small networks to generate weights — but for training, not for post-hoc compression of a trained model. Neural Implicit Representations for weights (NeRF-style, e.g., smolgen) are closer but target tiny models (GPT-2-scale). The structural difference: fitting G_θ to a trained 1.7B model's weights post-hoc with no retraining, and exploiting the empirically measured cross-layer regularity (CF2, CF11) as a structural prior on G_θ's architecture.

### Smallest experiment

**Claim:** A 1M-parameter MLP G_θ(L, role) can reconstruct Qwen3-1.7B W_Q matrices across all 28 layers with mean ‖G_θ(L, W_Q) − W_Q^L‖_F / ‖W_Q^L‖_F ≤ 0.20.
**Runtime:** ~4 hours. Fit a 2-layer MLP (input: [L/27, role_one_hot], output: flattened W_Q^L) via Adam on all 28 W_Q matrices. Measure relative reconstruction error.
**Go threshold:** Relative Frobenius error ≤ 0.20 on held-out layers (train on 22 layers, test on 6).
**NO-GO finding:** Characterizes the cross-layer smoothness / predictability of W_Q — itself a structural measurement that informs whether any cross-layer structure exists beyond CF11's within-layer concentration.

### Primary risk

MLP weights (CF8: full-rank, CF5: W_up more rank-sensitive than W_gate) have no obvious cross-layer regularity that G_θ could exploit — reconstruction error on W_up/W_gate may be catastrophically high. Mitigation: first-experiment targets W_Q only (CF11 gives structural motivation); MLP weight reconstruction is secondary and can be abandoned without killing the attention-side win.

---

## A5-REGONLY — Register-Only Decode (No RAM During Decode)

**Orientation:** A — Constraint-Alien | **Track:** A (arch-transposition) [FREE SWING]

**Constraint:** During the autoregressive decode step, no values may be loaded from or written to DRAM. All state fits in CPU registers, L1, L2, and L3 cache — and NVMe for weights. DRAM is forbidden during decode.

**Constraint difficulty calibration:** Eliminates standard DRAM-resident KV cache, standard bf16 GEMV that exceeds L3 bandwidth, and any activation buffer that spills to DRAM. Retains NVMe-streamed weight loading, L3-cached KV for recent tokens, register-allocated residual stream. ~70% of standard decode operations are inadmissible.

### Mechanism

Track A (architecture is restructured to fit the constraint). The Ryzen 5 7530U has 16 MB L3 cache shared across 6 cores. The residual stream for one token: 2048 × bf16 = 4 KB — fits in L1. One attention layer's KV for the current token: 28 layers × 128 dim × 2 heads × bf16 ≈ 14 KB — fits in L1. For the KV cache, the constraint forces a bounded-history design: L3 = 16 MB; KV per token per layer = 128 × 2 × bf16 = 512 bytes; tokens that fit in L3 = 16 MB / (28 layers × 512 B) = ~1120 tokens. For tokens beyond 1120, KV must come from NVMe — but that's a read, not a DRAM access. The constraint-forced design: weights stream from NVMe (sequential, full 3 GB/s bandwidth), KV for recent 1120 tokens lives in L3, residual stream stays in registers/L1. The key restructuring: attention computation is reordered to be NVMe-bounded (load W_Q, W_K, W_V from NVMe; compute Q·K in-flight; accumulate attention-weighted V in L1 registers). This matches the streaming-compute model of a GPU kernel — but on CPU with NVMe as the weight bus. The constraint makes explicit what is implicit in the Apple LLM-in-a-Flash paper: treat NVMe as the weight bus and CPU registers/cache as the only working memory.

### Residency arithmetic

L3 KV budget: 16 MB / (28 × 512 B/token/layer) = 1120 tokens. NVMe weight stream for 70B at 0.55 bpw: 5.35 GB. At 3 GB/s NVMe: 1.78 s/full-model pass = 0.56 tok/s. Matching Apple LLM-in-a-Flash architecture but on Windows/NVMe. At 7B model 0.55 bpw = 535 MB NVMe: 535 MB / 3 GB/s = 178 ms/token = 5.6 tok/s in DRAM-free mode. The constraint forces exactly the architecture Apple described — but as a consequence of the constraint, not as an engineering choice. This is the "recognize the substrate primitive" pattern, forced by the constraint.

### Novelty gloss

Closest kill list item: OS-Paging-Aware Weight Layout (R1/S2) was killed for not reducing bpw. REGONLY doesn't claim bpw reduction — it claims a DRAM-free architecture. The structural difference: R1/S2 was a layout optimization over standard DRAM-resident inference; REGONLY eliminates DRAM from the critical path entirely. Closest published: Apple LLM-in-a-Flash (arXiv:2312.11514) does NVMe-streamed weight loading — structurally the same mechanism but framed as an engineering optimization, not a constraint-forced derivation. The constraint-alien value here is not novelty of mechanism but novelty of *why*: the "no DRAM" constraint forces the architecture, and the constraint language surfaces the design tradeoffs (L3 KV bound, NVMe bandwidth budget) more clearly than the engineering framing does.

### Smallest experiment

**Claim:** On the Ryzen 5 7530U, a single-layer bf16 GEMV (2048×2048) that reads weights from NVMe via mmap and writes outputs to L1-allocated buffers achieves throughput ≥ 1.5 GB/s (NVMe-bandwidth-bound, not DRAM-bound).
**Runtime:** ~2 hours. Implement a Python/ctypes loop that mmap-reads weight pages and forces L1-resident output via tight compute loop; measure throughput with perf or time.
**Go threshold:** Throughput ≥ 1.5 GB/s sustained (indicates NVMe is the binding constraint, not DRAM).
**NO-GO finding:** If throughput is DRAM-bound rather than NVMe-bound, it means the OS is caching the mmap reads in DRAM — which is a real measurement of the NTFS page-cache behavior on this workload, itself a v2 re-derivation of the CF13* number.

### Primary risk

The OS (Windows 11 NTFS) inevitably stages NVMe reads through DRAM via the page cache, making "no DRAM" physically impossible without VirtualLock + madvise bypasses. Mitigation: test with VirtualLock to pin KV buffers in L3 and measure whether NVMe reads stage through a DRAM buffer or hit cache directly — this distinguishes "logically DRAM-free" (OS manages the staging transparently) from "physically DRAM-free" (not achievable without kernel modification, which is inadmissible per Orientation U rules).

---

## Convergence handles

- CF1 / CF3 outlier-channel structure as a routing key (A1-TROPFFN, A3-FSMDEC both converge on this)
- Cross-layer W_Q regularity (A4-SEEDNET: is G_θ predictable across layers? A3-FSMDEC: is FSM state shared across layers?)
- NVMe as weight bus / L3 as working KV store (A5-REGONLY, A2-APPENDKV both force NVMe into the compute path)
- Discrete state representation of continuous residual stream (A3-FSMDEC: FSM centroids; A1-TROPFFN: log-quantized dispatch)
- Content-addressable retrieval vs eviction as KV policy (A2-APPENDKV)
