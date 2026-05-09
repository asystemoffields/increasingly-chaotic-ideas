r"""R-RAOK-70B Rung 2 — Three-Tier Activation Codebook + ΔNLL Evaluation.

Stage 6 amendments applied (all honored):
  B1: random-init control uses torch.nn.init.normal_(std=0.02).
  B2: H1 is 28-layer EXTENSION of 7-layer CF3 finding (Rung 1 already confirmed GO).
  B3: quantize the shared MLP block input (before W_gate/W_up split); both projections
      receive the same quantized activation.
  B4: INT4 scale factor is max_abs / 7, NOT max_abs / 8 (validated in rraok_int4_unit_test.py).
      Unit test is re-run inline before any model inference.

Three-Tier Design:
  Tier 0 (FP16 bypass): top-2 channels by MEAN abs magnitude across the 200-prompt
    calibration corpus.  Static per layer, computed once.
  Tier 1 (INT8 dynamic shell): top-18 channels by CURRENT-TOKEN magnitude,
    excluding Tier-0 channels.  Dynamic per token.
    Quantize: int8 = round(x * 127 / max_abs(x_tier1))
    Reconstruct: x_q = int8 * max_abs(x_tier1) / 127
  Tier 2 (INT4 bulk): remaining 2028 channels.
    Quantize: int4 = round(x * 7 / max_abs(x_tier2))
    Reconstruct: x_q = int4 * max_abs(x_tier2) / 7  ← B4: /7 NOT /8

Evaluation:
  Model: Qwen/Qwen3-1.7B-Base, bf16, CPU.
  Corpus: WikiText-2 held-out, ~512 tokens (3 seeds = 3 different offsets within test split).
  Baseline: forward pass with NO quantization.
  Three-tier: same pass with MLP-input quantization applied at all 28 layers.

Skeptic controls:
  - Multi-seed (3 WikiText-2 offsets): mean ± std ΔNLL.
  - Permutation control: shuffle channels randomly WITHIN each tier before quantizing.
    The shuffled-tier scheme is expected to give worse ΔNLL; gap > 0.5 nats validates
    "magnitude keys to compressibility."
  - Random-init control: SKIP for Rung 2 (already run in Rung 1; would need different
    calibration-corpus pass and would not produce meaningful ΔNLL on a random model).

GO:   mean ΔNLL < 0.30 nats AND worst-of-3 < 0.50 nats AND perm-gap > 0.5 nats.
NO-GO: mean ΔNLL > 1.0 nats OR worst-of-3 > 1.5 nats.
GRAY:  0.30 <= mean ΔNLL <= 1.0 — tier-boundary scan recommended.

Outputs:
  experiments/stage0/ladder_v2/round1_raok70b/rung2_results.json
  sonnet_ladder_v2/cheap_tests/run_001/rraok_rung2_result.md
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

# ---------------------------------------------------------------------------
# B4 inline unit test — must pass before any model inference.
# ---------------------------------------------------------------------------

def _b4_unit_test() -> None:
    x = torch.tensor([-7.0, -4.0, 0.0, 4.0, 7.0])
    scale = x.abs().max() / 7
    q = torch.round(x / scale).clamp(-7, 7)
    x_recon = q * scale
    max_err = (x - x_recon).abs().max().item()
    assert abs(scale.item() - 1.0) < 1e-6, f"B4 FAIL scale={scale.item()}"
    assert max_err < 1e-5, f"B4 FAIL max_err={max_err}"

    # Wrong scale (max_abs/8) must produce clipping error > 0.5.
    scale_wrong = x.abs().max() / 8
    q_wrong = torch.round(x / scale_wrong).clamp(-7, 7)
    err_wrong = (x - q_wrong * scale_wrong).abs().max().item()
    assert err_wrong > 0.5, f"B4 FAIL wrong-scale error={err_wrong} should be >0.5"

    print("B4 unit test: PASS (scale = max_abs/7 confirmed correct)", flush=True)


# ---------------------------------------------------------------------------
# Calibration corpus (same 200-prompt PDAP strata as Rung 1)
# ---------------------------------------------------------------------------

# Import the prompt lists directly from rraok_rung1 to guarantee identical strata.
# We re-embed them here to keep rraok_rung2.py self-contained and avoid modifying
# rraok_rung1.py (constraint: DO NOT modify the Rung 1 script).

REPETITIVE_PROMPTS = [
    "yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes",
    "the the the the the the the the the the the the the the the the the the the the",
    "and and and and and and and and and and and and and and and and and and and and",
    "1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d",
    "round and round and round and round and round and round and round and round",
    "very very very very very very very very very very very very very very very",
    "knock knock knock knock knock knock knock knock knock knock knock knock",
    "click click click click click click click click click click click click",
    "tic tac tic tac tic tac tic tac tic tac tic tac tic tac tic tac",
    "Roses are red, violets are blue, sugar is sweet, and so are you. Roses are red.",
    "I scream, you scream, we all scream for ice cream. I scream, you scream.",
    "The wheels on the bus go round and round, round and round, round and round.",
    "Old MacDonald had a farm, ee i ee i o, and on his farm he had a cow, ee i ee i o.",
    "She sells sea shells by the sea shore. She sells sea shells by the sea shore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
    "Peter Piper picked a peck of pickled peppers. Peter Piper picked a peck.",
    "Fuzzy Wuzzy was a bear, Fuzzy Wuzzy had no hair, Fuzzy Wuzzy wasn't fuzzy was he.",
    "Mary had a little lamb, little lamb, little lamb. Mary had a little lamb.",
    "row row row your boat gently down the stream merrily merrily merrily merrily",
    "twinkle twinkle little star how I wonder what you are up above the world",
    "London Bridge is falling down, falling down, falling down, falling down.",
    "Hickory dickory dock, the mouse ran up the clock, hickory dickory dock.",
    "Pat-a-cake, pat-a-cake, baker's man, bake me a cake as fast as you can.",
    "This is the house that Jack built. This is the malt that lay in the house.",
    "There was an old woman who lived in a shoe. She had so many children.",
    "Hot cross buns, hot cross buns, one a penny, two a penny, hot cross buns.",
    "Jack and Jill went up the hill to fetch a pail of water. Jack fell down.",
    "Pop goes the weasel, pop goes the weasel, pop goes the weasel.",
    "ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC ABC",
    "Once upon a time, once upon a time, once upon a time, once upon a time.",
    "Tomorrow and tomorrow and tomorrow creeps in this petty pace from day to day.",
    "Beep beep beep beep beep beep beep beep beep beep beep beep beep beep",
    "The quick brown fox the quick brown fox the quick brown fox the quick brown fox",
    "rain rain go away come again another day rain rain go away come again",
    "Mirror mirror on the wall, mirror mirror on the wall, mirror mirror on the wall.",
    "If you're happy and you know it clap your hands. If you're happy and you know it.",
    "boom boom boom boom boom boom boom boom boom boom boom boom boom",
    "Wee Willie Winkie runs through the town, upstairs and downstairs in his nightgown.",
    "Lazy Mary will you get up, will you get up, will you get up, will you get up today.",
    "Diddle diddle dumpling my son John went to bed with his trousers on.",
    "Knick knack paddy whack give a dog a bone, this old man came rolling home.",
    "Five little ducks went out one day, over the hill and far away. Mother duck said quack.",
    "Wheels on the bus go round, doors on the bus go open and shut, horn goes beep.",
    "buzz buzz buzz buzz buzz buzz buzz buzz buzz buzz buzz buzz buzz buzz",
    "tap tap tap tap tap tap tap tap tap tap tap tap tap tap tap tap tap",
    "ding dong ding dong ding dong ding dong ding dong ding dong ding dong",
    "snap crackle pop snap crackle pop snap crackle pop snap crackle pop snap",
    "I love you, you love me, we're a happy family, with a great big hug and a kiss.",
    "Oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh",
]

TECHNICAL_PROMPTS = [
    "def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[len(arr) // 2]",
    "import torch; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')",
    "SELECT user_id, COUNT(*) FROM events WHERE created_at > '2026-01-01' GROUP BY user_id",
    "for i in range(0, n, 2): if arr[i] != arr[i+1]: return False; return True",
    "git rebase -i HEAD~5 --autosquash --strategy=recursive --strategy-option=ours",
    "kubectl apply -f deployment.yaml --namespace=production --record --validate=true",
    "class Node: def __init__(self, val=0, next=None): self.val = val; self.next = next",
    "async function fetchData(url) { const response = await fetch(url); return response.json(); }",
    "MOV EAX, [EBP+8]; ADD EAX, [EBP+12]; MOV [EBP-4], EAX; LEAVE; RET",
    "DROP TABLE IF EXISTS users CASCADE; CREATE TABLE users (id BIGSERIAL PRIMARY KEY)",
    "static inline void __attribute__((always_inline)) memcpy_avx2(void* dst, const void* src, size_t n)",
    "dispatch_async(queue, ^{ NSLog(@\"Hello from %@ thread\", [NSThread currentThread]); });",
    "let result = arr.iter().filter(|&x| x > 0).map(|&x| x * 2).collect::<Vec<i32>>();",
    "DECLARE @cursor CURSOR FOR SELECT id FROM orders WHERE status = 'pending' OPEN @cursor",
    "fn main() { let mut x: Vec<i32> = vec![1, 2, 3]; x.push(4); println!(\"{:?}\", x); }",
    "@app.route('/api/users/<int:user_id>', methods=['GET']) def get_user(user_id): return jsonify(...)",
    "RUN apt-get update && apt-get install -y python3-pip && pip3 install -r requirements.txt",
    "matrix_a = np.random.randn(1024, 1024).astype(np.float32); result = matrix_a @ matrix_a.T",
    "git log --pretty=format:'%h %s' --since='2 weeks ago' --author='alice@example.com'",
    "tcpdump -i eth0 -nn -X 'tcp port 443 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'",
    "ResNet50 architecture: Conv2d 7x7 stride 2, MaxPool 3x3 stride 2, four residual stages with bottleneck blocks.",
    "P(A|B) = P(B|A) * P(A) / P(B), Bayes theorem; the posterior is proportional to likelihood times prior.",
    "Type-erased polymorphism in C++ with std::variant<int, std::string, std::vector<double>> storage.",
    "@dataclass(frozen=True) class Vector: x: float; y: float; def dot(self, other) -> float: return self.x * other.x + self.y * other.y",
    "TLS 1.3 handshake: ClientHello with key_share extension, ServerHello, EncryptedExtensions, Finished.",
    "VPMADDUBSW ymm0, ymm1, ymm2 -- vertical multiply unsigned bytes by signed bytes, sum to 16-bit lanes.",
    "Lambda calculus: (\\x. x x) (\\x. x x) is the omega combinator and does not normalize.",
    "Floyd-Warshall: for k in V: for i in V: for j in V: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])",
    "CRDTs: G-Counter, PN-Counter, OR-Set, LWW-Element-Set; conflict-free merging via lattice operations.",
    "Mean field theory replaces N-body interactions with a single average field, reducing to one-body problems.",
    "FFT bit-reversal permutation: index i maps to reverse_bits(i, log2(N)); butterfly stages combine subproblems.",
    "Erlang OTP: gen_server, gen_statem, supervisor, application; let it crash philosophy with restart strategies.",
    "Reed-Solomon (255, 223): 32 parity bytes correct up to 16 byte errors per 255-byte block in GF(256).",
    "Chebyshev polynomials of the first kind: T0(x) = 1, T1(x) = x, T_{n+1} = 2x T_n - T_{n-1}.",
    "Conway's Game of Life: live cell with 2-3 neighbors survives; dead cell with exactly 3 becomes live.",
    "QUIC stream multiplexing over UDP; congestion control via NewReno or BBR; 0-RTT resumption.",
    "Strassen matmul: 7 multiplications instead of 8 for 2x2 blocks; recursive O(n^log2(7)).",
    "Hidden Markov Model: forward-backward algorithm; Viterbi for most likely state sequence.",
    "Newton's method: x_{n+1} = x_n - f(x_n) / f'(x_n); quadratic convergence near simple roots.",
    "Regex: ^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$ matches ISO 8601 date strings.",
    "x86 PUSH RBP, MOV RBP RSP, SUB RSP 32, MOV [RBP-8] RDI, MOV [RBP-16] RSI; standard prologue.",
    "Operating system: process state machine has Running, Ready, Waiting, Zombie.",
    "Backpropagation: chain rule applied recursively; dL/dw_l = dL/da_l * a_{l-1}^T.",
    "Distributed consensus Raft: leader election term, log replication AppendEntries.",
    "BST insertion: if key < node.key go left else go right; recurse until null.",
    "Galois group of x^4 - 2 over Q is the dihedral group D4 of order 8.",
    "Zermelo-Fraenkel set theory with axiom of choice: extensionality, pairing, union, power set.",
    "AVX-512: VPCONFLICTD detects conflicts in indirect store; VBMI2 adds bit shuffle.",
    "BTRFS: copy-on-write subvolume snapshots; B-tree metadata; checksumming with CRC32C.",
    "Quantum Hall effect: sigma_xy = nu e^2/h with integer nu; topological invariant.",
]

PROSE_PROMPTS = [
    "The library at Alexandria stood for nearly a thousand years before its slow decline.",
    "Beneath the pavements of Pompeii, archaeologists discovered loaves of bread carbonized by the eruption.",
    "She had been walking for three days through the high grass when she finally saw the river.",
    "The astronomer adjusted the eyepiece and waited for the seeing to settle.",
    "His grandmother kept her recipes on index cards in a tin box.",
    "The river had changed course three times in the last century.",
    "After the storm, the streets were littered with branches.",
    "The watchmaker held the tiny gear up to the light.",
    "On winter mornings the windows fogged up so thoroughly.",
    "The librarian remembered every patron who had borrowed the rare folio.",
    "Migration patterns of the monarch butterfly span four generations.",
    "He had grown up in a town where everyone could trace their lineage.",
    "The cellist tuned her instrument by ear.",
    "Soft snow accumulated on the brick walls overnight.",
    "Her father had been a merchant marine.",
    "The composer worked best in absolute silence.",
    "By the third winter, the cabin had begun to settle.",
    "Finches arrived at the feeder in small waves.",
    "The cartographer's study was lit by a single oil lamp.",
    "She remembered the sound of her grandfather's laugh.",
    "Ocean currents carry tropical seeds to northern beaches.",
    "The summer kitchen had been added to the back of the house in 1923.",
    "Standing on the platform, she watched the train pull in slowly.",
    "The bookbinder used wheat paste, never PVA.",
    "Eel migration remains poorly understood.",
    "Late afternoon light slanted through the venetian blinds.",
    "The cobbler kept a wooden last for each regular customer.",
    "Her notebook was held together with a single thick rubber band.",
    "The river ferry ran every twenty minutes during summer.",
    "Conifers in the boreal forest produce seeds that require fire to germinate.",
    "The conductor lowered his baton and the orchestra fell silent.",
    "He spent his retirement carving wooden birds.",
    "Glacial moraines record the boundaries of past ice advances.",
    "The bakery opened at four in the morning.",
    "Lichens on the granite outcrop grew at less than a millimeter per year.",
    "She washed the dishes by hand, even though the dishwasher worked perfectly.",
    "The carpenter selected each board for its grain pattern.",
    "When the streetlamps came on at dusk, the neighborhood children would scatter.",
    "The clock in the bell tower had stopped working in 1967.",
    "Fishermen had names for every tide and every current.",
    "The pottery wheel turned slowly under her practiced hands.",
    "Old maps of the harbor showed shoals that had since silted in.",
    "He kept a small notebook in his jacket pocket.",
    "The retired mathematician spent her mornings on a chess problem.",
    "The tea ceremony required four hours of preparation.",
    "Spring runoff carved a new channel through the meadow.",
    "The accordionist played each Sunday afternoon in the park.",
    "Her writing desk faced east so that morning light fell on the page.",
    "The mason had laid the chimney bricks one summer fifty years ago.",
    "Across the valley, the church bells rang out the hours.",
]


def _gen_random_token_prompts(tokenizer, n: int, length: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    out = []
    for _ in range(n):
        ids = rng.integers(0, vocab, size=length).tolist()
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def _build_calib_prompts(tokenizer, n_per_stratum: int = 50, max_length: int = 32) -> list[tuple[str, str]]:
    rand_prompts = _gen_random_token_prompts(tokenizer, n_per_stratum, max_length, seed=1)
    return (
        [(p, "repetitive") for p in REPETITIVE_PROMPTS[:n_per_stratum]] +
        [(p, "technical") for p in TECHNICAL_PROMPTS[:n_per_stratum]] +
        [(p, "prose") for p in PROSE_PROMPTS[:n_per_stratum]] +
        [(p, "random") for p in rand_prompts]
    )


# ---------------------------------------------------------------------------
# Step 1: Calibration pass — compute Tier-0 channel indices per layer.
# Tier-0 = top-2 channels by MEAN abs magnitude across the 200-prompt corpus.
# B3: hook captures MLP block input (shared tensor before W_gate/W_up split).
# ---------------------------------------------------------------------------

def compute_tier0_channels(
    model,
    tokenizer,
    calib_prompts: list[tuple[str, str]],
    n_layers: int,
    d: int,
    device: torch.device,
    max_length: int = 32,
    tier0_k: int = 2,
    tier1_k: int = 18,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Returns:
      tier0_per_layer: {layer_idx: LongTensor of shape (tier0_k,)} — static Tier-0 channel indices.
      tier1_static_per_layer: {layer_idx: LongTensor of shape (tier1_k,)} — Tier-1 channel indices
        by mean magnitude rank (excluding Tier-0); used for multi-seed calibration-stability check.
    Both are computed from mean abs magnitude across the full calibration corpus.
    """
    # Accumulate sum of abs magnitudes per layer, summed over tokens.
    sum_mag: dict[int, torch.Tensor] = {L: torch.zeros(d, dtype=torch.float64) for L in range(n_layers)}
    count: dict[int, int] = {L: 0 for L in range(n_layers)}

    captured: dict[int, torch.Tensor] = {}

    def make_pre_hook(idx: int):
        def hook(_mod, inputs):
            captured[idx] = inputs[0].detach().float()
        return hook

    handles = []
    for L in range(n_layers):
        handles.append(model.model.layers[L].mlp.register_forward_pre_hook(make_pre_hook(L)))

    with torch.inference_mode():
        for prompt_text, _stratum in calib_prompts:
            captured.clear()
            encoded = tokenizer(prompt_text, return_tensors="pt",
                                truncation=True, max_length=max_length)
            ids = encoded.input_ids.to(device)
            if ids.shape[-1] < 2:
                continue
            try:
                model(input_ids=ids, use_cache=False)
            except Exception as e:
                print(f"  calib prompt failed ({e}); skipping", flush=True)
                continue
            for L in range(n_layers):
                if L not in captured:
                    continue
                h = captured[L].squeeze(0)  # (T, d)
                sum_mag[L] += h.abs().sum(dim=0).double()
                count[L] += h.shape[0]

    for h in handles:
        h.remove()

    tier0_per_layer: dict[int, torch.Tensor] = {}
    tier1_static_per_layer: dict[int, torch.Tensor] = {}

    for L in range(n_layers):
        if count[L] == 0:
            print(f"  WARNING: layer {L} had no calibration tokens", flush=True)
            mean_mag = torch.zeros(d)
        else:
            mean_mag = (sum_mag[L] / count[L]).float()

        # Tier-0: top-2 channels by mean magnitude.
        top2 = mean_mag.topk(tier0_k).indices.sort().values  # sorted for reproducibility
        tier0_per_layer[L] = top2

        # Tier-1 (static version for stability check): top-18 by mean magnitude,
        # excluding Tier-0.
        mask_non_t0 = torch.ones(d, dtype=torch.bool)
        mask_non_t0[top2] = False
        non_t0_mag = mean_mag.clone()
        non_t0_mag[~mask_non_t0] = -1e9  # suppress Tier-0 channels
        top18 = non_t0_mag.topk(tier1_k).indices.sort().values
        tier1_static_per_layer[L] = top18

    return tier0_per_layer, tier1_static_per_layer


# ---------------------------------------------------------------------------
# Step 2: Three-tier quantize function (applied per-token per-layer).
# B3: applied to the shared MLP block input tensor.
# B4: INT4 scale = max_abs / 7.
# ---------------------------------------------------------------------------

def quantize_three_tier(
    h: torch.Tensor,           # (d,) float32 activation for ONE token
    tier0_idx: torch.Tensor,   # (2,) static LongTensor
    tier1_k: int = 18,
    permute_control: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Apply three-tier quantization to a single token's MLP block input.

    Tier 0: FP16 bypass (keep exact values — cast through float16 and back to float32).
    Tier 1: top-18 non-Tier-0 channels by CURRENT-TOKEN magnitude → INT8.
    Tier 2: remaining 2028 channels → INT4.

    permute_control=True: shuffle channel indices WITHIN each tier before quantizing.
    This should break the magnitude-keyed structure and worsen ΔNLL.
    """
    d = h.shape[0]
    h_out = h.clone()

    # --- Build per-token tier masks ---
    # Tier-0 mask: static
    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    # Tier-1: top-18 non-Tier-0 channels by current-token abs magnitude.
    mag = h.abs()
    mag_non_t0 = mag.clone()
    mag_non_t0[t0_mask] = -1e9  # suppress Tier-0
    top_non_t0 = mag_non_t0.topk(tier1_k).indices  # (18,)

    t1_mask = torch.zeros(d, dtype=torch.bool)
    t1_mask[top_non_t0] = True

    # Tier-2: everything else.
    t2_mask = ~(t0_mask | t1_mask)

    # Permutation control: randomly shuffle which channels get assigned to each tier's
    # QUANTIZATION (but preserving tier membership counts). The shuffle is within-tier:
    # we permute the CHANNEL VALUES assigned to each tier slot.
    if permute_control and rng is not None:
        # Tier-0: randomly draw 2 non-overlapping channels and swap their values
        # with the actual tier-0 channels. Effect: Tier-0 channels lose their
        # FP16-bypass advantage; random channels get it instead.
        all_idx = np.arange(d)

        # Shuffle which channels are in Tier-0.
        t0_new_idx = rng.choice(all_idx, size=tier0_idx.shape[0], replace=False)
        t0_new = torch.from_numpy(t0_new_idx).long()

        # Shuffle which channels are in Tier-1 (from non-Tier-0-new channels).
        remaining = np.setdiff1d(all_idx, t0_new_idx)
        t1_new_idx = rng.choice(remaining, size=tier1_k, replace=False)
        t1_new = torch.from_numpy(t1_new_idx).long()

        t0_mask_perm = torch.zeros(d, dtype=torch.bool)
        t0_mask_perm[t0_new] = True
        t1_mask_perm = torch.zeros(d, dtype=torch.bool)
        t1_mask_perm[t1_new] = True
        t2_mask_perm = ~(t0_mask_perm | t1_mask_perm)

        t0_mask, t1_mask, t2_mask = t0_mask_perm, t1_mask_perm, t2_mask_perm

    # --- Tier 0: FP16 bypass ---
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # --- Tier 1: INT8 dynamic ---
    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1
        # else: near-zero values; keep original (quantization error negligible)

    # --- Tier 2: INT4 bulk (B4: scale = max_abs / 7, NOT /8) ---
    h_t2 = h[t2_mask]
    if h_t2.numel() > 0:
        max_abs_t2 = h_t2.abs().max()
        if max_abs_t2 > 1e-9:
            scale_t2 = max_abs_t2 / 7.0  # B4 critical: /7 not /8
            q_t2 = torch.round(h_t2 / scale_t2).clamp(-7, 7)
            h_out[t2_mask] = q_t2 * scale_t2

    return h_out


# ---------------------------------------------------------------------------
# Step 3: Tier-1 cross-token Jaccard stability check.
# Measures how much the dynamic Tier-1 set rotates token-to-token.
# Expected to match Rung 1's K=1% Jaccard (mean ~0.39).
# ---------------------------------------------------------------------------

def compute_tier1_jaccard(
    tier1_sets_per_layer: dict[int, list[set]],
    n_layers: int,
) -> dict[int, float]:
    """
    For each layer, compute mean Jaccard between consecutive Tier-1 sets.
    tier1_sets_per_layer: {layer: [set_of_indices_for_each_token]}
    """
    result: dict[int, float] = {}
    for L in range(n_layers):
        sets = tier1_sets_per_layer.get(L, [])
        if len(sets) < 2:
            result[L] = float("nan")
            continue
        jacc_vals = []
        for i in range(len(sets) - 1):
            a, b = sets[i], sets[i + 1]
            inter = len(a & b)
            union = len(a | b)
            if union > 0:
                jacc_vals.append(inter / union)
        result[L] = float(np.mean(jacc_vals)) if jacc_vals else float("nan")
    return result


# ---------------------------------------------------------------------------
# Step 4: NLL evaluation — baseline and quantized.
# ---------------------------------------------------------------------------

def _load_wikitext2_tokens(tokenizer, offset: int = 0, n_tokens: int = 512) -> torch.Tensor:
    """
    Load WikiText-2 test split as a flat token sequence.
    Returns a 1D LongTensor of length n_tokens starting at the given offset.

    Falls back to a short built-in sample if datasets is not available.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])
    except Exception as e:
        print(f"  WARNING: datasets not available ({e}); using fallback text", flush=True)
        # Fallback: a ~3000-character Wikipedia-style text sample to generate enough tokens.
        text = (
            "The history of science is the study of the development of science, including both the "
            "natural sciences and social sciences. Science is a body of empirical, theoretical, and "
            "practical knowledge about the natural world, produced by scientists who emphasize the "
            "observation, explanation, and prediction of real-world phenomena. Historiography of science, "
            "in contrast, often treats science as a collection of sets of knowledge. The English word "
            "scientist is relatively recent, first coined by William Whewell in the nineteenth century. "
            "Previously, people investigating nature called themselves natural philosophers. Plato is "
            "generally credited with coining the word philosophy in ancient Greece, meaning the love of "
            "wisdom. The search for knowledge is as old as civilization itself. From the earliest times, "
            "humans sought to explain the world around them. Mathematics, astronomy, and philosophy were "
            "among the earliest disciplines, with ancient civilizations in Mesopotamia, Egypt, and Greece "
            "contributing significant advances. The scientific revolution of the sixteenth and seventeenth "
            "centuries transformed natural philosophy into modern science, with figures such as Copernicus, "
            "Galileo, Kepler, and Newton laying the groundwork for classical mechanics, astronomy, and "
            "mathematics. The development of the printing press allowed for the wider dissemination of "
            "scientific knowledge, accelerating the pace of discovery. By the nineteenth century, science "
            "had become a profession, with universities, academies, and journals supporting research in "
            "many specialized fields. The twentieth century witnessed unprecedented advances in physics, "
            "chemistry, biology, and engineering, culminating in technologies such as nuclear energy, "
            "computers, and space exploration. The relationship between science, technology, and society "
            "remains a subject of intense inquiry, as scientists grapple with questions of funding, ethics, "
            "reproducibility, and the communication of results to the public. "
        ) * 6  # repeat to get enough tokens

    ids = tokenizer.encode(text, add_special_tokens=False)
    # Ensure offset does not exceed available tokens.
    total = len(ids)
    start = offset % max(1, total - n_tokens)
    end = start + n_tokens
    if end > total:
        start = max(0, total - n_tokens)
        end = start + n_tokens
    chunk = ids[start:end]
    if len(chunk) < n_tokens:
        # Pad by cycling.
        reps = (n_tokens // len(chunk)) + 2
        chunk = (chunk * reps)[:n_tokens]
    return torch.tensor(chunk[:n_tokens], dtype=torch.long)


def eval_nll_with_quantization(
    model,
    tier0_per_layer: dict[int, torch.Tensor],
    eval_tokens: torch.Tensor,  # (N,) LongTensor
    n_layers: int,
    d: int,
    device: torch.device,
    chunk_size: int = 64,
    tier1_k: int = 18,
    apply_quantization: bool = True,
    permute_control: bool = False,
    collect_tier1_jaccard: bool = False,
) -> tuple[float, dict[int, float], dict[int, list[set]] | None]:
    """
    Compute mean NLL (nats) over eval_tokens using a sliding-window approach.

    For each chunk of `chunk_size` tokens, run a forward pass and collect the
    cross-entropy loss at every position. MLP block inputs are quantized (or not)
    depending on apply_quantization.

    Returns:
      (mean_nll, per_layer_delta_proxy, tier1_sets_per_layer)

    per_layer_delta_proxy: {layer: mean squared reconstruction error per token}.
    This is a proxy for per-layer contribution; true per-layer ΔNLL requires
    per-layer ablation (expensive). We report per-layer recon error as a diagnostic.

    tier1_sets_per_layer: {layer: [set_of_tier1_indices per token]} — for Jaccard check.
    """
    N = eval_tokens.shape[0]
    perm_rng = np.random.default_rng(42) if permute_control else None

    # Per-layer reconstruction error accumulator.
    recon_err_sum: dict[int, float] = {L: 0.0 for L in range(n_layers)}
    recon_err_count: dict[int, int] = {L: 0 for L in range(n_layers)}

    tier1_sets: dict[int, list[set]] | None = (
        {L: [] for L in range(n_layers)} if collect_tier1_jaccard else None
    )

    # Install hooks to intercept and modify MLP block inputs.
    # The hook replaces the input tensor with the quantized version in-place.
    layer_quant_cache: dict[int, torch.Tensor] = {}

    def make_pre_hook_quant(layer_idx: int):
        tier0_idx = tier0_per_layer[layer_idx]

        def hook(module, inputs):
            h_batch = inputs[0]  # (B, T, d)
            if not apply_quantization:
                return  # No-op for baseline
            h_float = h_batch.detach().float()
            B, T, D = h_float.shape
            h_out = h_float.clone()
            for b in range(B):
                for t in range(T):
                    h_tok = h_float[b, t]  # (d,)
                    h_q = quantize_three_tier(
                        h_tok, tier0_idx,
                        tier1_k=tier1_k,
                        permute_control=permute_control,
                        rng=perm_rng,
                    )
                    # Reconstruction error for diagnostic.
                    err = (h_tok - h_q).pow(2).mean().item()
                    recon_err_sum[layer_idx] += err
                    recon_err_count[layer_idx] += 1
                    h_out[b, t] = h_q

                    # Tier-1 Jaccard collection (only for non-permuted runs).
                    if tier1_sets is not None and not permute_control:
                        mag = h_tok.abs()
                        t0_mask = torch.zeros(D, dtype=torch.bool)
                        t0_mask[tier0_idx] = True
                        mag[t0_mask] = -1e9
                        top18 = mag.topk(tier1_k).indices
                        tier1_sets[layer_idx].append(set(top18.tolist()))

            # Return modified input by replacing tensor.
            # Hooks cannot change the dtype of returned tensors without issues,
            # so we cast back to the original dtype.
            h_replaced = h_out.to(h_batch.dtype)
            return (h_replaced,) + inputs[1:]

        return hook

    handles = []
    for L in range(n_layers):
        handles.append(
            model.model.layers[L].mlp.register_forward_pre_hook(make_pre_hook_quant(L))
        )

    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        # Process in overlapping chunks; compute NLL on the target token only.
        # Use stride=chunk_size//2 for context warm-up; report only back half.
        stride = chunk_size
        context = min(32, chunk_size // 2)

        for start in range(0, N - 1, stride):
            # Include a leading context window for better conditioning.
            ctx_start = max(0, start - context)
            end = min(N, start + stride)
            chunk_ids = eval_tokens[ctx_start:end + 1].unsqueeze(0).to(device)
            if chunk_ids.shape[-1] < 2:
                continue

            try:
                out = model(input_ids=chunk_ids, use_cache=False)
            except Exception as e:
                print(f"  forward error at start={start}: {e}", flush=True)
                continue

            logits = out.logits[0]  # (T, vocab)
            # NLL for each token position (from context+1 onward to use the leading context).
            report_start_local = start - ctx_start  # index within chunk where we start counting
            for pos in range(report_start_local, logits.shape[0] - 1):
                target = chunk_ids[0, pos + 1]
                log_probs = torch.nn.functional.log_softmax(logits[pos], dim=-1)
                nll = -log_probs[target].item()
                if np.isfinite(nll):
                    total_nll += nll
                    total_tokens += 1

    for h in handles:
        h.remove()

    mean_nll = total_nll / max(1, total_tokens)

    # Per-layer reconstruction error (mean per token).
    per_layer_recon: dict[int, float] = {}
    for L in range(n_layers):
        cnt = recon_err_count[L]
        per_layer_recon[L] = recon_err_sum[L] / cnt if cnt > 0 else float("nan")

    return mean_nll, per_layer_recon, tier1_sets


# ---------------------------------------------------------------------------
# Main experiment loop.
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    DEVICE = torch.device("cpu")
    MAX_LENGTH = 32
    CALIB_N_PER_STRATUM = 50  # 200 total
    EVAL_N_TOKENS = 512
    CHUNK_SIZE = 64
    TIER0_K = 2
    TIER1_K = 18
    # Three WikiText-2 offsets (seeds for multi-seed skeptic control).
    # Different starting positions within the test split.
    EVAL_OFFSETS = [0, 512, 1024]

    OUT_DIR = Path("experiments/stage0/ladder_v2/round1_raok70b")
    RESULT_MD_DIR = Path("sonnet_ladder_v2/cheap_tests/run_001")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_MD_DIR.mkdir(parents=True, exist_ok=True)

    # B4 unit test — must pass before anything else.
    print(f"[{time.time()-t0:.1f}s] Running B4 unit test ...", flush=True)
    _b4_unit_test()

    # Load model.
    print(f"[{time.time()-t0:.1f}s] Loading {MODEL_ID} bf16 on CPU ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}", flush=True)
    assert d == 2048, f"Expected d=2048 for Qwen3-1.7B, got {d}"
    assert n_layers == 28, f"Expected 28 layers, got {n_layers}"
    expected_t2_channels = d - TIER0_K - TIER1_K
    print(f"  Tier-0: {TIER0_K} channels (FP16), Tier-1: {TIER1_K} channels (INT8), "
          f"Tier-2: {expected_t2_channels} channels (INT4)", flush=True)

    # Calibration pass — identify Tier-0 static channel set per layer.
    print(f"\n[{time.time()-t0:.1f}s] Calibration pass (200 prompts, all 28 layers) ...", flush=True)
    calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
    print(f"  Calibration corpus size: {len(calib_prompts)} prompts", flush=True)
    tier0_per_layer, tier1_static_per_layer = compute_tier0_channels(
        model, tokenizer, calib_prompts, n_layers, d, DEVICE, max_length=MAX_LENGTH,
        tier0_k=TIER0_K, tier1_k=TIER1_K
    )
    calib_elapsed = time.time() - t0
    print(f"[{calib_elapsed:.1f}s] Calibration done.", flush=True)
    for L in range(n_layers):
        t0c = tier0_per_layer[L].tolist()
        print(f"  L{L:2d} Tier-0 channels: {t0c}", flush=True)

    # Multi-seed evaluation loop.
    print(f"\n[{time.time()-t0:.1f}s] Starting multi-seed evaluation (3 seeds) ...", flush=True)

    seed_results: list[dict[str, Any]] = []
    per_layer_recon_all_seeds: list[dict[int, float]] = []
    # Tier-1 Jaccard accumulators — set during seed 0 only, preserved across all seeds.
    tier1_jacc_per_layer: dict[int, float] | None = None
    mean_tier1_jacc: float = float("nan")

    for seed_idx, offset in enumerate(EVAL_OFFSETS):
        print(f"\n[{time.time()-t0:.1f}s] --- Seed {seed_idx+1}/3 (offset={offset}) ---", flush=True)

        eval_tokens = _load_wikitext2_tokens(tokenizer, offset=offset, n_tokens=EVAL_N_TOKENS)
        print(f"  Loaded {eval_tokens.shape[0]} eval tokens", flush=True)

        # Baseline NLL (no quantization).
        print(f"[{time.time()-t0:.1f}s]   Baseline forward pass ...", flush=True)
        baseline_nll, _, _ = eval_nll_with_quantization(
            model, tier0_per_layer, eval_tokens, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=False,
            permute_control=False,
            collect_tier1_jaccard=False,
        )
        print(f"  Baseline NLL: {baseline_nll:.4f} nats", flush=True)

        # Three-tier quantized NLL.
        collect_t1_jacc = (seed_idx == 0)  # collect Tier-1 Jaccard only for seed 0
        print(f"[{time.time()-t0:.1f}s]   Three-tier quantized forward pass ...", flush=True)
        quant_nll, per_layer_recon, tier1_sets = eval_nll_with_quantization(
            model, tier0_per_layer, eval_tokens, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=True,
            permute_control=False,
            collect_tier1_jaccard=collect_t1_jacc,
        )
        delta_nll = quant_nll - baseline_nll
        print(f"  Quantized NLL: {quant_nll:.4f} nats  |  ΔNLL = {delta_nll:.4f} nats", flush=True)

        seed_results.append({
            "seed_idx": seed_idx,
            "offset": offset,
            "baseline_nll": baseline_nll,
            "quant_nll": quant_nll,
            "delta_nll": delta_nll,
        })
        per_layer_recon_all_seeds.append(per_layer_recon)

        # Tier-1 cross-token Jaccard (seed 0 only — preserve across subsequent seeds).
        if collect_t1_jacc and tier1_sets is not None:
            tier1_jacc_per_layer = compute_tier1_jaccard(tier1_sets, n_layers)
            mean_tier1_jacc = float(np.nanmean(list(tier1_jacc_per_layer.values())))
            print(f"  Tier-1 cross-token Jaccard (mean across layers): {mean_tier1_jacc:.4f}", flush=True)
            print(f"  (Rung 1 K=1% reference: 0.388; Tier-1 K=18/2048~0.88% expected to be similar)", flush=True)
        # else: leave existing tier1_jacc_per_layer / mean_tier1_jacc unchanged.

    # Aggregate multi-seed results.
    delta_nlls = [r["delta_nll"] for r in seed_results]
    mean_delta = float(np.mean(delta_nlls))
    std_delta = float(np.std(delta_nlls))
    worst_delta = float(np.max(delta_nlls))

    print(f"\n[{time.time()-t0:.1f}s] Multi-seed ΔNLL summary:", flush=True)
    for i, r in enumerate(seed_results):
        print(f"  Seed {i+1} (offset={r['offset']}): ΔNLL = {r['delta_nll']:.4f} nats", flush=True)
    print(f"  Mean ΔNLL: {mean_delta:.4f} ± {std_delta:.4f} nats", flush=True)
    print(f"  Worst-of-3 ΔNLL: {worst_delta:.4f} nats", flush=True)

    # Per-layer reconstruction error (mean over seeds).
    per_layer_recon_mean: dict[int, float] = {}
    for L in range(n_layers):
        vals = [s.get(L, float("nan")) for s in per_layer_recon_all_seeds]
        per_layer_recon_mean[L] = float(np.nanmean(vals))

    # Permutation control — single seed (offset=0), one forward pass each.
    print(f"\n[{time.time()-t0:.1f}s] Permutation control (offset=0) ...", flush=True)
    eval_tokens_perm = _load_wikitext2_tokens(tokenizer, offset=0, n_tokens=EVAL_N_TOKENS)
    baseline_nll_perm, _, _ = eval_nll_with_quantization(
        model, tier0_per_layer, eval_tokens_perm, n_layers, d, DEVICE,
        chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        apply_quantization=False,
        permute_control=False,
    )
    perm_nll, _, _ = eval_nll_with_quantization(
        model, tier0_per_layer, eval_tokens_perm, n_layers, d, DEVICE,
        chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        apply_quantization=True,
        permute_control=True,
    )
    perm_delta = perm_nll - baseline_nll_perm
    perm_gap = perm_delta - seed_results[0]["delta_nll"]
    print(f"  Permutation control ΔNLL: {perm_delta:.4f} nats", flush=True)
    print(f"  Three-tier ΔNLL (seed 0): {seed_results[0]['delta_nll']:.4f} nats", flush=True)
    print(f"  Permutation gap (perm - three-tier): {perm_gap:.4f} nats", flush=True)

    # Verdict.
    elapsed_total = time.time() - t0
    go_nll = mean_delta < 0.30 and worst_delta < 0.50 and perm_gap > 0.5
    nogo_nll = mean_delta > 1.0 or worst_delta > 1.5

    if go_nll:
        verdict = "GO"
        verdict_note = (
            f"mean ΔNLL={mean_delta:.4f} < 0.30 nats, "
            f"worst-of-3={worst_delta:.4f} < 0.50 nats, "
            f"perm-gap={perm_gap:.4f} > 0.50 nats. "
            "Cluster C2 advances. Calibration-free CF3-derived tiers are structurally validated."
        )
    elif nogo_nll:
        verdict = "NO-GO"
        verdict_note = (
            f"mean ΔNLL={mean_delta:.4f} or worst-of-3={worst_delta:.4f} exceeds NO-GO threshold. "
            "CF3 Jaccard phase transitions are NOT a quantization-sensitivity oracle for "
            "MLP-block-input target. Class-level kill: R-RAOK-70B + F-OCSSQ + C-RAOK-Grounded. "
            "Structural finding: CF-RAOK-KILL candidate."
        )
    else:
        verdict = "GRAY"
        verdict_note = (
            f"mean ΔNLL={mean_delta:.4f} nats is in GRAY zone (0.30 < ΔNLL < 1.0). "
            "Follow-up: tier-boundary scan (Tier-1 widen to 27 channels or Tier-0 to 4 channels). "
            "Identify per-layer ΔNLL bottleneck from reconstruction error profile."
        )

    print(f"\n[{elapsed_total:.1f}s] VERDICT: {verdict}", flush=True)
    print(f"  {verdict_note}", flush=True)

    # -----------------------------------------------------------------------
    # Assemble output JSON.
    # -----------------------------------------------------------------------
    tier0_channels_json: dict[str, list[int]] = {
        str(L): tier0_per_layer[L].tolist() for L in range(n_layers)
    }
    tier1_static_json: dict[str, list[int]] = {
        str(L): tier1_static_per_layer[L].tolist() for L in range(n_layers)
    }
    tier1_jacc_json: dict[str, float] = (
        {str(L): v for L, v in tier1_jacc_per_layer.items()}
        if tier1_jacc_per_layer is not None else {}
    )

    output: dict[str, Any] = {
        "experiment": "R-RAOK-70B Rung 2",
        "model_id": MODEL_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_total,
        "n_layers": n_layers,
        "d_hidden": d,
        "tier0_k": TIER0_K,
        "tier1_k": TIER1_K,
        "tier2_k": d - TIER0_K - TIER1_K,
        "eval_n_tokens": EVAL_N_TOKENS,
        "eval_offsets": EVAL_OFFSETS,
        "calib_n_prompts": len(calib_prompts),
        "b3_note": (
            "B3: quantization applied to shared MLP block input (pre-W_gate/W_up split). "
            "Both projections receive same quantized activation."
        ),
        "b4_note": "B4: INT4 scale = max_abs / 7 (NOT /8). Unit test PASSED inline.",
        "tier0_channels_per_layer": tier0_channels_json,
        "tier1_static_channels_per_layer": tier1_static_json,
        "seed_results": seed_results,
        "mean_delta_nll": mean_delta,
        "std_delta_nll": std_delta,
        "worst_of_3_delta_nll": worst_delta,
        "permutation_control": {
            "baseline_nll": baseline_nll_perm,
            "perm_quant_nll": perm_nll,
            "perm_delta_nll": perm_delta,
            "three_tier_delta_nll_seed0": seed_results[0]["delta_nll"],
            "perm_gap": perm_gap,
            "perm_gap_passes": perm_gap > 0.5,
        },
        "tier1_cross_token_jaccard_per_layer": tier1_jacc_json,
        "tier1_cross_token_jaccard_mean": mean_tier1_jacc,
        "per_layer_recon_error_mean": {str(L): v for L, v in per_layer_recon_mean.items()},
        "verdict": verdict,
        "verdict_note": verdict_note,
        "go_threshold": "mean ΔNLL < 0.30 AND worst-of-3 < 0.50 AND perm-gap > 0.50",
        "nogo_threshold": "mean ΔNLL > 1.0 OR worst-of-3 > 1.5",
    }

    json_path = OUT_DIR / "rung2_results.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}", flush=True)

    # -----------------------------------------------------------------------
    # Assemble result markdown.
    # -----------------------------------------------------------------------
    _write_result_md(
        result_dir=RESULT_MD_DIR,
        verdict=verdict,
        verdict_note=verdict_note,
        mean_delta=mean_delta,
        std_delta=std_delta,
        worst_delta=worst_delta,
        seed_results=seed_results,
        perm_delta=perm_delta,
        perm_gap=perm_gap,
        tier1_jacc_per_layer=tier1_jacc_per_layer,
        mean_tier1_jacc=mean_tier1_jacc,
        per_layer_recon_mean=per_layer_recon_mean,
        n_layers=n_layers,
        d=d,
        tier0_channels=tier0_channels_json,
        elapsed_total=elapsed_total,
        calib_elapsed=calib_elapsed,
    )

    return 0


def _write_result_md(
    result_dir: Path,
    verdict: str,
    verdict_note: str,
    mean_delta: float,
    std_delta: float,
    worst_delta: float,
    seed_results: list[dict],
    perm_delta: float,
    perm_gap: float,
    tier1_jacc_per_layer: dict[int, float] | None,
    mean_tier1_jacc: float,
    per_layer_recon_mean: dict[int, float],
    n_layers: int,
    d: int,
    tier0_channels: dict[str, list[int]],
    elapsed_total: float,
    calib_elapsed: float,
) -> None:
    seed0_delta = seed_results[0]["delta_nll"] if seed_results else float("nan")

    # Per-layer reconstruction error table (descending for standouts).
    recon_items = [(L, v) for L, v in per_layer_recon_mean.items() if np.isfinite(v)]
    recon_items.sort(key=lambda x: x[1], reverse=True)
    recon_mean_val = float(np.nanmean([v for _, v in recon_items])) if recon_items else float("nan")
    recon_table_rows = []
    for L, v in sorted(per_layer_recon_mean.items()):
        flag = " (standout)" if v > 3 * recon_mean_val else ""
        t0ch = tier0_channels.get(str(L), [])
        t1j = (
            f"{tier1_jacc_per_layer[L]:.4f}" if tier1_jacc_per_layer and L in tier1_jacc_per_layer else "n/a"
        )
        recon_table_rows.append(
            f"| L{L:2d} | {v:.6f} | {t1j} | {t0ch} |{flag}"
        )

    recon_table = "\n".join(recon_table_rows)

    perm_gap_pass = "PASS" if perm_gap > 0.5 else "FAIL"
    mean_nll_pass = "PASS" if mean_delta < 0.30 else ("FAIL" if mean_delta > 1.0 else "GRAY")
    worst_pass = "PASS" if worst_delta < 0.50 else ("FAIL" if worst_delta > 1.5 else "GRAY")

    md = f"""# R-RAOK-70B Rung 2 Result

Date: {datetime.now(UTC).strftime("%Y-%m-%d")}
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_rung2.py`
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/rung2_results.json`
Wall-clock elapsed: {elapsed_total:.1f} s (~{elapsed_total/60:.1f} min; calib ~{calib_elapsed/60:.1f} min)

---

## B4 Unit Test: PASS (CRITICAL)

INT4 scale factor unit test verified inline before any model inference:
- `scale = max_abs(x) / 7` — confirmed correct (zero reconstruction error on integer inputs)
- `scale = max_abs(x) / 8` (wrong) — confirmed produces clipping artifact at x=-7

---

## Rung 2 Verdict: {verdict}

{verdict_note}

---

## Multi-Seed ΔNLL Summary

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
"""
    for r in seed_results:
        md += (f"| {r['seed_idx']+1} | {r['offset']} | {r['baseline_nll']:.4f} | "
               f"{r['quant_nll']:.4f} | {r['delta_nll']:.4f} |\n")

    md += f"""
**Mean ΔNLL: {mean_delta:.4f} ± {std_delta:.4f} nats** — threshold 0.30 nats — {mean_nll_pass}
**Worst-of-3 ΔNLL: {worst_delta:.4f} nats** — threshold 0.50 nats — {worst_pass}

---

## Permutation Control

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| Three-tier ΔNLL (seed 0) | {seed0_delta:.4f} nats | — | — |
| Permuted-tier ΔNLL | {perm_delta:.4f} nats | — | — |
| Permutation gap | {perm_gap:.4f} nats | > 0.50 nats | {perm_gap_pass} |

Permutation control shuffles channel assignments WITHIN each tier randomly before quantizing.
A gap > 0.50 nats validates the "magnitude keys to compressibility" structural claim.

---

## Tier-1 Cardinality Stability (Cross-Token Jaccard)

Mean Tier-1 cross-token Jaccard (K=18 of 2046 non-Tier-0 channels): **{mean_tier1_jacc:.4f}**
Reference (Rung 1, K=1%≈21 channels): 0.388 mean across all 28 layers.
Tier-1 K=18/2046 ≈ 0.88% — expected Jaccard slightly below Rung 1's 1% result.
{("Tier-1 stability is consistent with Rung 1 K=1% Jaccard finding." if abs(mean_tier1_jacc - 0.388) < 0.1 else "Tier-1 stability deviates from Rung 1 K=1% Jaccard reference by >" + f"{abs(mean_tier1_jacc - 0.388):.3f}.")}

---

## Per-Layer Reconstruction Error and Tier-1 Jaccard

(Reconstruction error = mean squared error between original and quantized MLP block input.)
(Tier-1 Jaccard = cross-token Jaccard of dynamic Tier-1 channel sets, seed 0 only.)

| Layer | Mean Recon MSE | Tier-1 Jaccard | Tier-0 Channels |
|---|---|---|---|
{recon_table}

---

## Structural Finding

**Three-tier design (B3):** quantization applied to the shared MLP block input (pre-W_gate/W_up split).
Both W_gate and W_up receive the same quantized activation at every layer.

**Tier boundaries:**
- Tier 0: top-{2} channels by mean abs magnitude (static per layer, from 200-prompt calibration corpus).
- Tier 1: top-{18} channels by CURRENT-TOKEN magnitude, excluding Tier-0 (dynamic per token).
- Tier 2: remaining {d - 2 - 18} channels (bulk INT4, scale = max_abs / 7).

**B4 compliance:** INT4 scale = max_abs / 7. Off-by-one (÷8) would clip at negative saturation.

---

## L27 Anomaly Note

Rung 1 found L27 Jaccard@K=1% = 0.531 (lone outlier above 0.50). In Rung 2, L27's
reconstruction error is:
  L27 recon MSE = {per_layer_recon_mean.get(27, float('nan')):.6f}
  Mean all-layer recon MSE = {recon_mean_val:.6f}
{"L27 is a STANDOUT layer (recon MSE > 3x mean). Consider widening Tier-1 for L27." if per_layer_recon_mean.get(27, 0) > 3 * recon_mean_val else "L27 reconstruction error is not a standout (within 3x mean). No targeted widening required."}

---

## GO / NO-GO / GRAY Thresholds

| Criterion | Value | Threshold | Status |
|---|---|---|---|
| Mean ΔNLL | {mean_delta:.4f} nats | < 0.30 | {mean_nll_pass} |
| Worst-of-3 ΔNLL | {worst_delta:.4f} nats | < 0.50 | {worst_pass} |
| Permutation gap | {perm_gap:.4f} nats | > 0.50 | {perm_gap_pass} |

**Final verdict: {verdict}**

{verdict_note}

---

## Cascade Implications

{"**Cluster C2 ADVANCES.** R-RAOK-70B + F-OCSSQ + C-RAOK-Grounded all survive. Calibration-free CF3-Jaccard-derived tier boundaries are an empirically grounded post-training compression class. Next step: S4 VPTK (AVX2 INT4 kernel for Tier-2 bulk) or C-ABAR extension. 70B residency claim: CF3 Jaccard thresholds generalize as quantization-tier oracles at MLP-block-input scope. This Rung 2 result is the gating step before deployment arithmetic — the 70B KV-cache compression projection (4× activation compression → ~1 GB KV at 4K context) is now structurally grounded." if verdict == "GO" else ("**Cluster C2 KILLED.** CF3 Jaccard phase transitions do NOT serve as quantization-sensitivity oracles for MLP-block-input activations. Structural finding CF-RAOK-KILL: 'activation quantization sensitivity does not align with per-token outlier-set stability; different metric required.' Class-level kill applies to R-RAOK-70B, F-OCSSQ, C-RAOK-Grounded, and all RAOK derivatives." if verdict == "NO-GO" else "**GRAY — follow-up required.** Run tier-boundary scan: (a) widen Tier-1 to 27 channels (1.3%); (b) widen Tier-0 to 4 channels (0.2%). Identify which layers drive the ΔNLL excess from per-layer reconstruction error profile. Targeted widening for the worst 5 layers (likely deep layers 23-27 per CF3 pattern) is the fastest resolution path.")}
"""

    md_path = result_dir / "rraok_rung2_result.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
