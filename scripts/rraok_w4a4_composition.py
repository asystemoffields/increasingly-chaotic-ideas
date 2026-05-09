r"""W4A4 Composition Test — Per-Channel INT4 Weights × RAOK Three-Tier Activations.

Deployment-critical test: does per-channel INT4 weight quantization compose cleanly
with RAOK three-tier activation quantization?

The 70B-on-laptop deployment claim requires:
  W4 weights (~0.5 bpw) + RAOK activations + KV compression = fits in 7.28 GiB.
This test settles whether the W4×A4 matmul is viable.

Three forward passes per seed:
  1. Baseline: FP16 weights + FP16 activations (identical to rraok_rung2.py baseline).
  2. W4-only: per-channel INT4 weight quantization, FP16 activations.
  3. W4A4: per-channel INT4 weights + RAOK three-tier activations.

Per-channel INT4 weight quantization (pure torch, no external libs):
  For each MLP weight matrix W (W_gate, W_up, W_down), per output-channel:
    scale_c = max_abs(W[c, :]) / 7
    W_q[c, :] = round(W[c, :] / scale_c).clamp(-7, 7) * scale_c
  Applied in-place as a dequantized FP copy (simulate INT4 at inference time).

Applied to: all 28 layers, W_gate, W_up, W_down in MLP blocks.

Evaluation: WikiText-2, 512 tokens, 3 seeds (offsets 0/512/1024).

Composition delta = (W4A4 ΔNLL) - (W4-only ΔNLL) - (RAOK-only ΔNLL=0.083)
  ~0: composes linearly (GO)
  0.10-0.50: slippage (WARN)
  >0.50: destructive interference (FAIL)

Outputs:
  experiments/stage0/ladder_v2/round1_raok70b/w4a4_composition_results.json
  sonnet_ladder_v2/cheap_tests/run_001/rraok_w4a4_composition_result.md
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

# ---------------------------------------------------------------------------
# RAOK-only ΔNLL reference from Rung 2.
# ---------------------------------------------------------------------------
RAOK_ONLY_DELTA_NLL = 0.0830  # nats, mean over 3 seeds from rraok_rung2_result.md

# ---------------------------------------------------------------------------
# B4 inline unit test (same as rraok_rung2.py — verified before any inference).
# ---------------------------------------------------------------------------

def _b4_unit_test() -> None:
    x = torch.tensor([-7.0, -4.0, 0.0, 4.0, 7.0])
    scale = x.abs().max() / 7
    q = torch.round(x / scale).clamp(-7, 7)
    x_recon = q * scale
    max_err = (x - x_recon).abs().max().item()
    assert abs(scale.item() - 1.0) < 1e-6, f"B4 FAIL scale={scale.item()}"
    assert max_err < 1e-5, f"B4 FAIL max_err={max_err}"

    scale_wrong = x.abs().max() / 8
    q_wrong = torch.round(x / scale_wrong).clamp(-7, 7)
    err_wrong = (x - q_wrong * scale_wrong).abs().max().item()
    assert err_wrong > 0.5, f"B4 FAIL wrong-scale error={err_wrong} should be >0.5"

    print("B4 unit test: PASS (scale = max_abs/7 confirmed correct)", flush=True)


# ---------------------------------------------------------------------------
# Per-channel INT4 weight quantization (pure torch).
# ---------------------------------------------------------------------------

def _quantize_weight_per_channel_int4(W: torch.Tensor) -> torch.Tensor:
    """
    Simulate per-channel INT4 weight quantization by quantizing and dequantizing
    each output channel of W.

    W shape: (out_features, in_features)
    Returns a new float32 tensor of the same shape with INT4-level precision.

    Per-channel: for each output channel c:
      scale_c = max_abs(W[c, :]) / 7
      W_q[c, :] = round(W[c, :] / scale_c).clamp(-7, 7) * scale_c

    Channels with all-zero weights are left as zero (no divide-by-zero).
    """
    W_float = W.detach().float()
    out_ch = W_float.shape[0]

    # Per-channel max abs magnitude → scales shape (out_ch,)
    scales = W_float.abs().max(dim=1).values / 7.0  # (out_ch,)

    # Avoid division by zero for dead channels.
    safe_scales = scales.clone()
    safe_scales[safe_scales < 1e-9] = 1.0  # effectively no-op for zero rows

    # Quantize and dequantize.
    # Broadcast: W_float (out, in) / scales (out, 1)
    W_q = torch.round(W_float / safe_scales.unsqueeze(1)).clamp(-7, 7) * safe_scales.unsqueeze(1)

    # Restore zero rows.
    zero_mask = (scales < 1e-9)
    W_q[zero_mask] = 0.0

    return W_q


def apply_w4_to_model(model: nn.Module, n_layers: int, t0: float) -> nn.Module:
    """
    Apply per-channel INT4 weight quantization to all MLP W_gate, W_up, W_down
    in all n_layers. Modifies a COPY of the model and returns it.

    We work on a copy so the original FP model stays intact for baseline.
    """
    print(f"[{time.time()-t0:.1f}s] Creating deep copy of model for W4 quantization ...", flush=True)
    model_w4 = copy.deepcopy(model)

    total_params_quantized = 0
    for L in range(n_layers):
        mlp = model_w4.model.layers[L].mlp
        for attr_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, attr_name, None)
            if proj is None:
                print(f"  WARNING: layer {L} has no {attr_name}", flush=True)
                continue
            W_orig = proj.weight.data  # (out, in)
            W_q = _quantize_weight_per_channel_int4(W_orig)
            proj.weight.data.copy_(W_q.to(W_orig.dtype))
            total_params_quantized += W_q.numel()

    print(f"  W4 quantized {total_params_quantized:,} weight params across {n_layers} layers "
          f"(gate_proj + up_proj + down_proj)", flush=True)
    return model_w4


# ---------------------------------------------------------------------------
# Calibration corpus (same structure as rraok_rung2.py for Tier-0 calculation).
# Copied here to keep this script self-contained; do NOT modify rraok_rung2.py.
# ---------------------------------------------------------------------------

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
# Calibration: compute Tier-0 channel indices (identical logic to rraok_rung2.py).
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
) -> dict[int, torch.Tensor]:
    """Returns tier0_per_layer: {layer_idx: LongTensor of shape (tier0_k,)}."""
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
                h = captured[L].squeeze(0)
                sum_mag[L] += h.abs().sum(dim=0).double()
                count[L] += h.shape[0]

    for h in handles:
        h.remove()

    tier0_per_layer: dict[int, torch.Tensor] = {}
    for L in range(n_layers):
        if count[L] == 0:
            mean_mag = torch.zeros(d)
        else:
            mean_mag = (sum_mag[L] / count[L]).float()
        top2 = mean_mag.topk(tier0_k).indices.sort().values
        tier0_per_layer[L] = top2

    return tier0_per_layer


# ---------------------------------------------------------------------------
# Three-tier activation quantization (identical to rraok_rung2.py).
# ---------------------------------------------------------------------------

def quantize_three_tier(
    h: torch.Tensor,
    tier0_idx: torch.Tensor,
    tier1_k: int = 18,
) -> torch.Tensor:
    """Apply three-tier quantization to a single token's MLP block input (d,)."""
    d = h.shape[0]
    h_out = h.clone()

    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    mag = h.abs()
    mag_non_t0 = mag.clone()
    mag_non_t0[t0_mask] = -1e9
    top_non_t0 = mag_non_t0.topk(tier1_k).indices

    t1_mask = torch.zeros(d, dtype=torch.bool)
    t1_mask[top_non_t0] = True

    t2_mask = ~(t0_mask | t1_mask)

    # Tier 0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Tier 1: INT8 dynamic.
    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1

    # Tier 2: INT4 bulk (B4: /7).
    h_t2 = h[t2_mask]
    if h_t2.numel() > 0:
        max_abs_t2 = h_t2.abs().max()
        if max_abs_t2 > 1e-9:
            scale_t2 = max_abs_t2 / 7.0
            q_t2 = torch.round(h_t2 / scale_t2).clamp(-7, 7)
            h_out[t2_mask] = q_t2 * scale_t2

    return h_out


# ---------------------------------------------------------------------------
# WikiText-2 loader (identical to rraok_rung2.py).
# ---------------------------------------------------------------------------

def _load_wikitext2_tokens(tokenizer, offset: int = 0, n_tokens: int = 512) -> torch.Tensor:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])
    except Exception as e:
        print(f"  WARNING: datasets not available ({e}); using fallback text", flush=True)
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
        ) * 6

    ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(ids)
    start = offset % max(1, total - n_tokens)
    end = start + n_tokens
    if end > total:
        start = max(0, total - n_tokens)
        end = start + n_tokens
    chunk = ids[start:end]
    if len(chunk) < n_tokens:
        reps = (n_tokens // len(chunk)) + 2
        chunk = (chunk * reps)[:n_tokens]
    return torch.tensor(chunk[:n_tokens], dtype=torch.long)


# ---------------------------------------------------------------------------
# NLL evaluation engine.
# Supports three modes:
#   apply_raok=False, (model passed in): baseline or W4-only.
#   apply_raok=True, (model passed in): W4A4 or RAOK-only depending on which model is passed.
# ---------------------------------------------------------------------------

def eval_nll(
    model,
    eval_tokens: torch.Tensor,
    n_layers: int,
    device: torch.device,
    tier0_per_layer: dict[int, torch.Tensor] | None = None,
    apply_raok: bool = False,
    chunk_size: int = 64,
    tier1_k: int = 18,
) -> float:
    """
    Compute mean NLL (nats) over eval_tokens.

    If apply_raok=True, tier0_per_layer must be provided.
    The RAOK hook quantizes MLP block inputs using the three-tier scheme.
    """
    N = eval_tokens.shape[0]
    handles = []

    if apply_raok and tier0_per_layer is not None:
        def make_pre_hook_quant(layer_idx: int):
            tier0_idx = tier0_per_layer[layer_idx]

            def hook(module, inputs):
                h_batch = inputs[0]  # (B, T, d)
                h_float = h_batch.detach().float()
                B, T, D = h_float.shape
                h_out = h_float.clone()
                for b in range(B):
                    for t in range(T):
                        h_out[b, t] = quantize_three_tier(
                            h_float[b, t], tier0_idx, tier1_k=tier1_k
                        )
                h_replaced = h_out.to(h_batch.dtype)
                return (h_replaced,) + inputs[1:]

            return hook

        for L in range(n_layers):
            handles.append(
                model.model.layers[L].mlp.register_forward_pre_hook(make_pre_hook_quant(L))
            )

    total_nll = 0.0
    total_tokens = 0
    stride = chunk_size
    context = min(32, chunk_size // 2)

    with torch.inference_mode():
        for start in range(0, N - 1, stride):
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
            report_start_local = start - ctx_start
            for pos in range(report_start_local, logits.shape[0] - 1):
                target = chunk_ids[0, pos + 1]
                log_probs = torch.nn.functional.log_softmax(logits[pos], dim=-1)
                nll = -log_probs[target].item()
                if np.isfinite(nll):
                    total_nll += nll
                    total_tokens += 1

    for h in handles:
        h.remove()

    return total_nll / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    DEVICE = torch.device("cpu")
    MAX_LENGTH = 32
    CALIB_N_PER_STRATUM = 50
    EVAL_N_TOKENS = 512
    CHUNK_SIZE = 64
    TIER0_K = 2
    TIER1_K = 18
    EVAL_OFFSETS = [0, 512, 1024]

    OUT_DIR = Path("experiments/stage0/ladder_v2/round1_raok70b")
    RESULT_MD_DIR = Path("sonnet_ladder_v2/cheap_tests/run_001")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_MD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.time()-t0:.1f}s] W4A4 Composition Test — Qwen3-1.7B-Base", flush=True)
    print(f"  Per-channel INT4 weights x RAOK three-tier activations", flush=True)

    # B4 unit test.
    print(f"[{time.time()-t0:.1f}s] Running B4 unit test ...", flush=True)
    _b4_unit_test()

    # Load FP model (bf16, CPU).
    print(f"[{time.time()-t0:.1f}s] Loading {MODEL_ID} bf16 on CPU ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_fp = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    n_layers = int(model_fp.config.num_hidden_layers)
    d = int(model_fp.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}", flush=True)
    assert d == 2048, f"Expected d=2048 for Qwen3-1.7B, got {d}"
    assert n_layers == 28, f"Expected 28 layers, got {n_layers}"

    # Calibration pass on the FP model.
    print(f"\n[{time.time()-t0:.1f}s] Calibration pass (200 prompts, FP model) ...", flush=True)
    calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
    tier0_per_layer = compute_tier0_channels(
        model_fp, tokenizer, calib_prompts, n_layers, d, DEVICE,
        max_length=MAX_LENGTH, tier0_k=TIER0_K, tier1_k=TIER1_K,
    )
    calib_elapsed = time.time() - t0
    print(f"[{calib_elapsed:.1f}s] Calibration done.", flush=True)

    # Build W4 model: deep copy + per-channel INT4 quantize MLP weights.
    print(f"\n[{time.time()-t0:.1f}s] Building W4 model ...", flush=True)
    model_w4 = apply_w4_to_model(model_fp, n_layers, t0)
    model_w4.eval()
    print(f"[{time.time()-t0:.1f}s] W4 model ready.", flush=True)

    # Multi-seed evaluation: 3 modes x 3 seeds.
    print(f"\n[{time.time()-t0:.1f}s] Starting 3-mode x 3-seed evaluation ...", flush=True)

    baseline_nlls: list[float] = []
    w4_nlls: list[float] = []
    w4a4_nlls: list[float] = []

    seed_results: list[dict[str, Any]] = []

    for seed_idx, offset in enumerate(EVAL_OFFSETS):
        print(f"\n[{time.time()-t0:.1f}s] --- Seed {seed_idx+1}/3 (offset={offset}) ---", flush=True)
        eval_tokens = _load_wikitext2_tokens(tokenizer, offset=offset, n_tokens=EVAL_N_TOKENS)

        # 1. Baseline: FP model, no RAOK.
        print(f"[{time.time()-t0:.1f}s]   [1/3] Baseline (FP weights, FP activations) ...", flush=True)
        nll_baseline = eval_nll(
            model_fp, eval_tokens, n_layers, DEVICE,
            tier0_per_layer=None, apply_raok=False,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        )
        baseline_nlls.append(nll_baseline)
        print(f"  Baseline NLL: {nll_baseline:.4f} nats", flush=True)

        # 2. W4-only: W4 model, no RAOK.
        print(f"[{time.time()-t0:.1f}s]   [2/3] W4-only (INT4 weights, FP activations) ...", flush=True)
        nll_w4 = eval_nll(
            model_w4, eval_tokens, n_layers, DEVICE,
            tier0_per_layer=None, apply_raok=False,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        )
        w4_nlls.append(nll_w4)
        delta_w4 = nll_w4 - nll_baseline
        print(f"  W4-only NLL: {nll_w4:.4f} nats  |  ΔNLL(W4) = {delta_w4:.4f} nats", flush=True)

        # 3. W4A4: W4 model + RAOK activations.
        print(f"[{time.time()-t0:.1f}s]   [3/3] W4A4 (INT4 weights + RAOK activations) ...", flush=True)
        nll_w4a4 = eval_nll(
            model_w4, eval_tokens, n_layers, DEVICE,
            tier0_per_layer=tier0_per_layer, apply_raok=True,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        )
        w4a4_nlls.append(nll_w4a4)
        delta_w4a4 = nll_w4a4 - nll_baseline
        print(f"  W4A4 NLL: {nll_w4a4:.4f} nats  |  ΔNLL(W4A4) = {delta_w4a4:.4f} nats", flush=True)

        seed_results.append({
            "seed_idx": seed_idx,
            "offset": offset,
            "baseline_nll": nll_baseline,
            "w4_nll": nll_w4,
            "w4a4_nll": nll_w4a4,
            "delta_w4": delta_w4,
            "delta_w4a4": delta_w4a4,
            "composition_delta_seed": delta_w4a4 - delta_w4 - RAOK_ONLY_DELTA_NLL,
        })

    # Aggregate.
    mean_baseline = float(np.mean(baseline_nlls))
    mean_w4_delta = float(np.mean([r["delta_w4"] for r in seed_results]))
    std_w4_delta = float(np.std([r["delta_w4"] for r in seed_results]))
    worst_w4_delta = float(np.max([r["delta_w4"] for r in seed_results]))

    mean_w4a4_delta = float(np.mean([r["delta_w4a4"] for r in seed_results]))
    std_w4a4_delta = float(np.std([r["delta_w4a4"] for r in seed_results]))
    worst_w4a4_delta = float(np.max([r["delta_w4a4"] for r in seed_results]))

    composition_delta = mean_w4a4_delta - mean_w4_delta - RAOK_ONLY_DELTA_NLL

    elapsed_total = time.time() - t0

    print(f"\n[{elapsed_total:.1f}s] === Results Summary ===", flush=True)
    print(f"  Baseline NLL: {mean_baseline:.4f} nats", flush=True)
    print(f"  W4-only  ΔNLL: {mean_w4_delta:.4f} ± {std_w4_delta:.4f} nats "
          f"(worst={worst_w4_delta:.4f})", flush=True)
    print(f"  W4A4     ΔNLL: {mean_w4a4_delta:.4f} ± {std_w4a4_delta:.4f} nats "
          f"(worst={worst_w4a4_delta:.4f})", flush=True)
    print(f"  RAOK-only ΔNLL reference: {RAOK_ONLY_DELTA_NLL:.4f} nats (from Rung 2)", flush=True)
    print(f"  Composition delta: {composition_delta:.4f} nats", flush=True)
    print(f"    = (W4A4 ΔNLL={mean_w4a4_delta:.4f}) - (W4 ΔNLL={mean_w4_delta:.4f}) "
          f"- (RAOK ΔNLL={RAOK_ONLY_DELTA_NLL:.4f})", flush=True)

    # Verdict.
    if mean_w4_delta > 1.0:
        verdict = "W4 ALONE FAILS"
        verdict_note = (
            f"W4-only ΔNLL={mean_w4_delta:.4f} > 1.0 nats. Per-channel INT4 weight quantization "
            "is not viable on Qwen3-1.7B without more sophisticated quantization (GPTQ/AWQ). "
            "Deployment claim broken at the weight side; activation composition moot."
        )
    elif composition_delta < 0.10:
        verdict = "COMPOSES CLEANLY"
        verdict_note = (
            f"Composition delta={composition_delta:.4f} nats < 0.10. "
            f"W4-only ΔNLL={mean_w4_delta:.4f}, W4A4 ΔNLL={mean_w4a4_delta:.4f}. "
            "W4 and RAOK stack approximately additively. "
            "Deployment arithmetic holds: predicted W4A4 NLL penalty ≈ W4 + RAOK."
        )
    elif composition_delta < 0.50:
        verdict = "COMPOSES WITH SLIPPAGE"
        verdict_note = (
            f"Composition delta={composition_delta:.4f} nats (0.10-0.50 range). "
            f"W4-only ΔNLL={mean_w4_delta:.4f}, W4A4 ΔNLL={mean_w4a4_delta:.4f}. "
            "W4 and RAOK compose with worse-than-additive stacking. "
            "Still deployable but quality cost is super-additive; budget ~{composition_delta:.2f} extra nats."
        )
    else:
        verdict = "DESTRUCTIVE INTERFERENCE"
        verdict_note = (
            f"Composition delta={composition_delta:.4f} nats > 0.50. "
            f"W4-only ΔNLL={mean_w4_delta:.4f}, W4A4 ΔNLL={mean_w4a4_delta:.4f}. "
            "W4 and RAOK schemes do not compose; activation quantization on already-W4-quantized "
            "weights creates large additional error. Deployment claim broken."
        )

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  {verdict_note}", flush=True)

    # Write JSON output.
    output: dict[str, Any] = {
        "experiment": "W4A4 Composition Test",
        "model_id": MODEL_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_total,
        "n_layers": n_layers,
        "d_hidden": d,
        "tier0_k": TIER0_K,
        "tier1_k": TIER1_K,
        "eval_n_tokens": EVAL_N_TOKENS,
        "eval_offsets": EVAL_OFFSETS,
        "w4_quant_method": "per-channel INT4 (scale=max_abs(W[c,:])/7), applied to gate_proj/up_proj/down_proj",
        "raok_method": "three-tier activation quantization (Tier0=FP16 bypass top-2, Tier1=INT8 top-18, Tier2=INT4 bulk)",
        "raok_only_delta_nll_reference": RAOK_ONLY_DELTA_NLL,
        "raok_only_source": "rraok_rung2_result.md Rung 2 mean ΔNLL over 3 seeds",
        "seed_results": seed_results,
        "mean_baseline_nll": mean_baseline,
        "w4_only": {
            "mean_delta_nll": mean_w4_delta,
            "std_delta_nll": std_w4_delta,
            "worst_of_3_delta_nll": worst_w4_delta,
        },
        "w4a4": {
            "mean_delta_nll": mean_w4a4_delta,
            "std_delta_nll": std_w4a4_delta,
            "worst_of_3_delta_nll": worst_w4a4_delta,
        },
        "composition_delta_nll": composition_delta,
        "composition_delta_formula": "mean_w4a4_delta - mean_w4_delta - raok_only_delta",
        "verdict": verdict,
        "verdict_note": verdict_note,
        "verdict_thresholds": {
            "COMPOSES_CLEANLY": "composition_delta < 0.10 AND w4_only_mean < 1.0",
            "COMPOSES_WITH_SLIPPAGE": "composition_delta 0.10-0.50",
            "DESTRUCTIVE_INTERFERENCE": "composition_delta > 0.50",
            "W4_ALONE_FAILS": "w4_only_mean > 1.0",
        },
    }

    json_path = OUT_DIR / "w4a4_composition_results.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}", flush=True)

    # Write result markdown.
    _write_result_md(
        result_dir=RESULT_MD_DIR,
        seed_results=seed_results,
        mean_w4_delta=mean_w4_delta,
        std_w4_delta=std_w4_delta,
        worst_w4_delta=worst_w4_delta,
        mean_w4a4_delta=mean_w4a4_delta,
        std_w4a4_delta=std_w4a4_delta,
        worst_w4a4_delta=worst_w4a4_delta,
        composition_delta=composition_delta,
        verdict=verdict,
        verdict_note=verdict_note,
        elapsed_total=elapsed_total,
    )

    return 0


def _write_result_md(
    result_dir: Path,
    seed_results: list[dict],
    mean_w4_delta: float,
    std_w4_delta: float,
    worst_w4_delta: float,
    mean_w4a4_delta: float,
    std_w4a4_delta: float,
    worst_w4a4_delta: float,
    composition_delta: float,
    verdict: str,
    verdict_note: str,
    elapsed_total: float,
) -> None:
    seed_rows = ""
    for r in seed_results:
        seed_rows += (
            f"| {r['seed_idx']+1} | {r['offset']} | "
            f"{r['baseline_nll']:.4f} | {r['delta_w4']:.4f} | "
            f"{r['delta_w4a4']:.4f} | {r['composition_delta_seed']:.4f} |\n"
        )

    # Composition interpretation.
    if verdict == "W4 ALONE FAILS":
        comp_interp = "W4 weight quantization is not viable — composition moot."
    elif verdict == "COMPOSES CLEANLY":
        comp_interp = (
            "W4 and RAOK stack approximately additively. Deployment arithmetic holds.\n"
            "Predicted penalty for W4+RAOK ≈ W4_delta + RAOK_delta (± noise)."
        )
    elif verdict == "COMPOSES WITH SLIPPAGE":
        comp_interp = (
            "Super-additive stacking: W4A4 penalty exceeds the sum of parts by "
            f"{composition_delta:.3f} nats. Still deployable but budget the extra cost."
        )
    else:
        comp_interp = (
            "Destructive interference: W4+RAOK is not a simple sum. "
            "Activation quantization noise amplified by already-quantized weights."
        )

    md = f"""# W4A4 Composition Test Result

Date: {datetime.now(UTC).strftime("%Y-%m-%d")}
Model: Qwen/Qwen3-1.7B-Base (bf16 loaded, CPU)
Script: `scripts/rraok_w4a4_composition.py`
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/w4a4_composition_results.json`
Wall-clock elapsed: {elapsed_total:.1f} s (~{elapsed_total/60:.1f} min)

---

## Verdict: {verdict}

{verdict_note}

---

## Summary Table

| Scheme | Mean ΔNLL | Std | Worst-of-3 |
|---|---|---|---|
| W4-only (INT4 weights, FP activations) | {mean_w4_delta:.4f} | {std_w4_delta:.4f} | {worst_w4_delta:.4f} |
| W4A4 (INT4 weights + RAOK activations) | {mean_w4a4_delta:.4f} | {std_w4a4_delta:.4f} | {worst_w4a4_delta:.4f} |
| RAOK-only reference (Rung 2) | {RAOK_ONLY_DELTA_NLL:.4f} | 0.0159 | 0.1052 |

**Composition delta = (W4A4 ΔNLL) - (W4-only ΔNLL) - (RAOK-only ΔNLL)**
= {mean_w4a4_delta:.4f} - {mean_w4_delta:.4f} - {RAOK_ONLY_DELTA_NLL:.4f} = **{composition_delta:.4f} nats**

{comp_interp}

---

## Per-Seed Breakdown

| Seed | Offset | Baseline NLL | W4 ΔNLL | W4A4 ΔNLL | Comp. Delta |
|---|---|---|---|---|---|
{seed_rows}
---

## Composition Delta Thresholds

| Zone | Range | This result |
|---|---|---|
| COMPOSES CLEANLY | < 0.10 nats | {"<-- HERE" if verdict == "COMPOSES CLEANLY" else ""} |
| COMPOSES WITH SLIPPAGE | 0.10 – 0.50 nats | {"<-- HERE" if verdict == "COMPOSES WITH SLIPPAGE" else ""} |
| DESTRUCTIVE INTERFERENCE | > 0.50 nats | {"<-- HERE" if verdict == "DESTRUCTIVE INTERFERENCE" else ""} |
| W4 ALONE FAILS | W4-only ΔNLL > 1.0 | {"<-- HERE" if verdict == "W4 ALONE FAILS" else ""} |

---

## Weight Quantization Method

**Per-channel INT4 (pure torch, no external libs):**
- Applied to: `gate_proj`, `up_proj`, `down_proj` in all 28 MLP blocks.
- Per output channel c: `scale_c = max_abs(W[c, :]) / 7`
- `W_q[c, :] = round(W[c, :] / scale_c).clamp(-7, 7) * scale_c`
- B4 compliance: scale denominator is 7 (not 8).
- Dequantized FP copy stored in weight tensors; simulates INT4 inference error.

**Activation quantization:** RAOK three-tier (identical to rraok_rung2.py):
- Tier 0: top-2 channels by mean calibration magnitude → FP16 bypass.
- Tier 1: top-18 current-token channels → INT8 dynamic.
- Tier 2: remaining 2028 channels → INT4 bulk.

---

## Deployment Implication

The 70B-on-laptop claim requires W4 weights + RAOK activations + KV compression.
This test determines whether W4 and RAOK stack without destructive interaction.

{"DEPLOYMENT CLAIM HOLDS: W4A4 composition is viable. The combined NLL penalty is approximately additive; operators can budget W4_penalty + RAOK_penalty independently." if verdict == "COMPOSES CLEANLY" else ("DEPLOYMENT CLAIM HOLDS WITH CAVEAT: W4A4 composes but with super-additive penalty of " + f"{composition_delta:.3f} nats. Budget the extra overhead." if verdict == "COMPOSES WITH SLIPPAGE" else ("DEPLOYMENT CLAIM BROKEN: " + verdict_note))}
"""

    md_path = result_dir / "rraok_w4a4_composition_result.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
