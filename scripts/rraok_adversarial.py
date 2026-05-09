r"""R-RAOK-70B Adversarial Baselines — Three skeptic tests for the three-tier novelty claim.

This script runs three adversarial baselines against the Rung 2 three-tier result.
Each baseline destroys one structural component of the RAOK design. If any baseline
matches RAOK's ΔNLL, the corresponding component is not load-bearing and the novelty
claim collapses to published prior art.

Adversarial 1 — Flat INT4 (--flat-int4):
  ALL 2048 channels quantized to INT4. No Tier-0 bypass, no Tier-1 INT8 shell.
  scale = max_abs(h_full) / 7 per token.
  KILL threshold: ΔNLL <= 0.10 nats → tier structure not load-bearing.
  Reduces to: LLM-FP4 / SmoothQuant-style flat INT4 activation quant.

Adversarial 2 — Random Tier-0 (--random-tier0):
  Tier-0 channels chosen randomly per layer (seeded) instead of top-2 by mean magnitude.
  Tier-1 (top-18 per-token dynamic, excluding random Tier-0) and Tier-2 (INT4) unchanged.
  KILL threshold: ΔNLL within 0.05 nats of RAOK → specific channel selection not load-bearing.
  Reduces to: CF3-derived calibration-free Tier-0 claim collapses.

Adversarial 3 — Static Tier-1 (--static-tier1):
  Tier-1 identified ONCE by top-18 mean magnitude (excluding Tier-0). Same channels every token.
  Tier-0 (FP16 bypass, top-2 static) and Tier-2 (INT4 bulk) unchanged.
  KILL threshold: ΔNLL within 0.05 nats of RAOK → dynamic rotation not load-bearing.
  Reduces to: CF3's K=1% dynamicity claim is not exploited.

Same eval protocol as Rung 2: Qwen/Qwen3-1.7B-Base, 28 layers, 3 seeds (WikiText-2 offsets
0/512/1024), 512 tokens per seed, chunk_size=64.

Outputs:
  experiments/stage0/ladder_v2/round1_raok70b/adversarial_results.json
  sonnet_ladder_v2/cheap_tests/run_001/rraok_adversarial_result.md

Usage:
  python scripts/rraok_adversarial.py --flat-int4
  python scripts/rraok_adversarial.py --random-tier0
  python scripts/rraok_adversarial.py --static-tier1
  python scripts/rraok_adversarial.py --all   (run all three sequentially)
"""

from __future__ import annotations

import argparse
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
# (Copied verbatim from rraok_rung2.py per self-contained constraint.)
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
# Calibration corpus — identical strata to rraok_rung2.py (copy-of-record).
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
# Calibration pass — same as Rung 2.
# Returns tier0_per_layer and tier1_static_per_layer (mean-magnitude ranked).
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
    """Returns tier0_per_layer (top-2 by mean mag) and tier1_static_per_layer (top-18 by mean mag, excl. T0)."""
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
    tier1_static_per_layer: dict[int, torch.Tensor] = {}

    for L in range(n_layers):
        if count[L] == 0:
            mean_mag = torch.zeros(d)
        else:
            mean_mag = (sum_mag[L] / count[L]).float()
        top2 = mean_mag.topk(tier0_k).indices.sort().values
        tier0_per_layer[L] = top2

        mask_non_t0 = torch.ones(d, dtype=torch.bool)
        mask_non_t0[top2] = False
        non_t0_mag = mean_mag.clone()
        non_t0_mag[~mask_non_t0] = -1e9
        top18 = non_t0_mag.topk(tier1_k).indices.sort().values
        tier1_static_per_layer[L] = top18

    return tier0_per_layer, tier1_static_per_layer


# ---------------------------------------------------------------------------
# WikiText-2 loader — identical to Rung 2.
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
# Per-token quantization functions for each adversarial baseline.
# ---------------------------------------------------------------------------

def quantize_flat_int4(h: torch.Tensor) -> torch.Tensor:
    """Adversarial 1: flat INT4 across ALL d channels. No tiers."""
    max_abs = h.abs().max()
    if max_abs < 1e-9:
        return h.clone()
    scale = max_abs / 7.0
    q = torch.round(h / scale).clamp(-7, 7)
    return q * scale


def quantize_random_tier0(
    h: torch.Tensor,
    random_tier0_idx: torch.Tensor,  # (2,) randomly chosen, fixed per layer
    tier1_k: int = 18,
) -> torch.Tensor:
    """
    Adversarial 2: random Tier-0 (FP16 bypass), dynamic Tier-1 (INT8), bulk Tier-2 (INT4).
    Tier-0 channels are randomly assigned (not magnitude-selected).
    Tier-1 is still top-18 per-token magnitude among non-random-Tier-0 channels.
    """
    d = h.shape[0]
    h_out = h.clone()

    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[random_tier0_idx] = True

    # Tier-1: top-18 per-token among non-Tier-0 channels (same dynamic logic as RAOK).
    mag = h.abs().clone()
    mag[t0_mask] = -1e9
    top18 = mag.topk(tier1_k).indices
    t1_mask = torch.zeros(d, dtype=torch.bool)
    t1_mask[top18] = True

    t2_mask = ~(t0_mask | t1_mask)

    # Tier 0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Tier 1: INT8.
    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1

    # Tier 2: INT4 (B4: /7).
    h_t2 = h[t2_mask]
    if h_t2.numel() > 0:
        max_abs_t2 = h_t2.abs().max()
        if max_abs_t2 > 1e-9:
            scale_t2 = max_abs_t2 / 7.0
            q_t2 = torch.round(h_t2 / scale_t2).clamp(-7, 7)
            h_out[t2_mask] = q_t2 * scale_t2

    return h_out


def quantize_static_tier1(
    h: torch.Tensor,
    tier0_idx: torch.Tensor,       # (2,) magnitude-selected, same as RAOK
    tier1_static_idx: torch.Tensor,  # (18,) static mean-magnitude selection
) -> torch.Tensor:
    """
    Adversarial 3: static Tier-1 (same 18 channels every token, no dynamic rotation).
    Tier-0 identical to RAOK. Tier-1 fixed from corpus mean rather than per-token top-18.
    """
    d = h.shape[0]
    h_out = h.clone()

    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    t1_mask = torch.zeros(d, dtype=torch.bool)
    t1_mask[tier1_static_idx] = True
    # Exclude any Tier-0 channels that may overlap (should be none, but be safe).
    t1_mask[t0_mask] = False

    t2_mask = ~(t0_mask | t1_mask)

    # Tier 0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Tier 1: INT8 (static channels, not dynamically chosen).
    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1

    # Tier 2: INT4 (B4: /7).
    h_t2 = h[t2_mask]
    if h_t2.numel() > 0:
        max_abs_t2 = h_t2.abs().max()
        if max_abs_t2 > 1e-9:
            scale_t2 = max_abs_t2 / 7.0
            q_t2 = torch.round(h_t2 / scale_t2).clamp(-7, 7)
            h_out[t2_mask] = q_t2 * scale_t2

    return h_out


# ---------------------------------------------------------------------------
# Generic eval loop — installs a hook that applies a per-token quantizer.
# ---------------------------------------------------------------------------

def eval_nll_with_hook(
    model,
    eval_tokens: torch.Tensor,   # (N,) LongTensor
    n_layers: int,
    device: torch.device,
    hook_fn_per_layer: dict[int, Any],  # {layer_idx: callable(h_tok) -> h_q}
    apply_quantization: bool = True,
    chunk_size: int = 64,
) -> float:
    """
    Compute mean NLL (nats) over eval_tokens.
    hook_fn_per_layer[L] is called per (token, layer) when apply_quantization=True.
    """
    N = eval_tokens.shape[0]
    total_nll = 0.0
    total_tokens = 0

    def make_pre_hook_quant(layer_idx: int):
        fn = hook_fn_per_layer.get(layer_idx)

        def hook(module, inputs):
            h_batch = inputs[0]  # (B, T, d)
            if not apply_quantization or fn is None:
                return
            h_float = h_batch.detach().float()
            B, T, D = h_float.shape
            h_out = h_float.clone()
            for b in range(B):
                for t in range(T):
                    h_out[b, t] = fn(h_float[b, t])
            h_replaced = h_out.to(h_batch.dtype)
            return (h_replaced,) + inputs[1:]

        return hook

    handles = []
    for L in range(n_layers):
        handles.append(
            model.model.layers[L].mlp.register_forward_pre_hook(make_pre_hook_quant(L))
        )

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
            logits = out.logits[0]
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
# Run one adversarial baseline across 3 seeds.
# Returns (mean_delta, std_delta, seed_deltas, seed_records).
# ---------------------------------------------------------------------------

def run_baseline(
    name: str,
    model,
    tokenizer,
    tier0_per_layer: dict[int, torch.Tensor],
    tier1_static_per_layer: dict[int, torch.Tensor],
    n_layers: int,
    d: int,
    device: torch.device,
    eval_offsets: list[int],
    eval_n_tokens: int,
    chunk_size: int,
    tier1_k: int,
    t0_global: float,
) -> tuple[float, float, list[float], list[dict]]:
    seed_deltas: list[float] = []
    seed_records: list[dict] = []

    # For adversarial-2: fixed random Tier-0 indices per layer (seeded at 999).
    rng_t0 = np.random.default_rng(999)
    random_tier0_per_layer: dict[int, torch.Tensor] = {}
    for L in range(n_layers):
        idx = rng_t0.choice(d, size=2, replace=False)
        idx.sort()
        random_tier0_per_layer[L] = torch.from_numpy(idx).long()

    for seed_idx, offset in enumerate(eval_offsets):
        print(f"\n[{time.time()-t0_global:.1f}s] [{name}] Seed {seed_idx+1}/3 (offset={offset}) ...", flush=True)
        eval_tokens = _load_wikitext2_tokens(tokenizer, offset=offset, n_tokens=eval_n_tokens)

        # Build per-layer hook function.
        if name == "flat_int4":
            hook_fn_per_layer = {L: quantize_flat_int4 for L in range(n_layers)}
        elif name == "random_tier0":
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_random_tier0(h, random_tier0_per_layer[_L], tier1_k=tier1_k))
                for L in range(n_layers)
            }
        elif name == "static_tier1":
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_static_tier1(
                    h, tier0_per_layer[_L], tier1_static_per_layer[_L]))
                for L in range(n_layers)
            }
        else:
            raise ValueError(f"Unknown baseline: {name}")

        # Baseline NLL (no quantization).
        print(f"[{time.time()-t0_global:.1f}s]   baseline forward pass ...", flush=True)
        baseline_nll = eval_nll_with_hook(
            model, eval_tokens, n_layers, device,
            hook_fn_per_layer={}, apply_quantization=False, chunk_size=chunk_size,
        )
        print(f"  baseline NLL: {baseline_nll:.4f} nats", flush=True)

        # Adversarial quantized NLL.
        print(f"[{time.time()-t0_global:.1f}s]   {name} quantized forward pass ...", flush=True)
        quant_nll = eval_nll_with_hook(
            model, eval_tokens, n_layers, device,
            hook_fn_per_layer=hook_fn_per_layer, apply_quantization=True, chunk_size=chunk_size,
        )
        delta_nll = quant_nll - baseline_nll
        print(f"  {name} quantized NLL: {quant_nll:.4f} nats  |  ΔNLL = {delta_nll:.4f} nats", flush=True)

        seed_deltas.append(delta_nll)
        seed_records.append({
            "seed_idx": seed_idx,
            "offset": offset,
            "baseline_nll": baseline_nll,
            "quant_nll": quant_nll,
            "delta_nll": delta_nll,
        })

    mean_delta = float(np.mean(seed_deltas))
    std_delta = float(np.std(seed_deltas))
    print(f"\n[{name}] Mean ΔNLL: {mean_delta:.4f} ± {std_delta:.4f} nats", flush=True)
    return mean_delta, std_delta, seed_deltas, seed_records


# ---------------------------------------------------------------------------
# Verdict logic per baseline.
# ---------------------------------------------------------------------------

def verdict_flat_int4(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if mean_delta <= 0.10:
        v = "KILL"
        note = (
            f"Flat INT4 ΔNLL={mean_delta:.4f} nats <= 0.10. Tier structure is NOT load-bearing. "
            f"RAOK reduces to flat INT4 activation quantization — covered by LLM-FP4 (Xiao et al. 2023) "
            f"and SmoothQuant (Xiao et al. 2022). Novelty claim COLLAPSES."
        )
    elif gap >= 0.20:
        v = "SURVIVES"
        note = (
            f"Flat INT4 ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} over RAOK ({raok_mean:.4f}). "
            f"Gap >= 0.20 nats. Tier-0/Tier-1 shell is load-bearing. Three-tier design has structural merit."
        )
    else:
        v = "MARGINAL"
        note = (
            f"Flat INT4 ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} over RAOK ({raok_mean:.4f}). "
            f"Gap 0.00-0.20 nats — tier structure marginally helpful. Inconclusive."
        )
    return v, note


def verdict_random_tier0(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = abs(mean_delta - raok_mean)
    if gap <= 0.05:
        v = "KILL"
        note = (
            f"Random Tier-0 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap <= 0.05 nats. Any 2 FP16 channels perform equally well. "
            f"CF3-derived Tier-0 channel identification is NOT load-bearing. "
            f"The 'calibration-free CF3-derived Tier-0 selection' novelty COLLAPSES."
        )
    elif gap >= 0.10:
        v = "SURVIVES"
        note = (
            f"Random Tier-0 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap >= 0.10 nats. The CF3-identified static-magnitude channels ARE load-bearing. "
            f"Calibration-free Tier-0 selection claim survives."
        )
    else:
        v = "MARGINAL"
        note = (
            f"Random Tier-0 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap 0.05-0.10 nats. Tier-0 channel identity weakly load-bearing. Inconclusive."
        )
    return v, note


def verdict_static_tier1(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = abs(mean_delta - raok_mean)
    if gap <= 0.05:
        v = "KILL"
        note = (
            f"Static Tier-1 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap <= 0.05 nats. Dynamic per-token rotation of the Tier-1 shell is NOT load-bearing. "
            f"CF3 K=1% dynamicity is not exploited. Dynamic shell novelty COLLAPSES."
        )
    elif gap >= 0.10:
        v = "SURVIVES"
        note = (
            f"Static Tier-1 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap >= 0.10 nats. Per-token dynamic rotation IS load-bearing. "
            f"CF3 dynamicity (K=1% Jaccard) is the genuine moat. Dynamic shell claim survives."
        )
    else:
        v = "MARGINAL"
        note = (
            f"Static Tier-1 ΔNLL={mean_delta:.4f} nats, gap={gap:.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap 0.05-0.10 nats. Dynamic rotation marginally helpful. Inconclusive."
        )
    return v, note


def aggregate_verdict(v1: str, v2: str, v3: str) -> tuple[str, str]:
    kills = [v for v in [v1, v2, v3] if v == "KILL"]
    survives = [v for v in [v1, v2, v3] if v == "SURVIVES"]

    if len(kills) >= 2:
        return "COLLAPSED-TO-PRIOR-ART", (
            f"{len(kills)}/3 baselines KILL. RAOK three-tier scheme is not novel over published work. "
            f"Flat INT4 prior: LLM-FP4 (Xiao et al. 2023) / SmoothQuant (Xiao et al. 2022)."
        )
    elif len(kills) == 1 and len(survives) == 0:
        return "COLLAPSED-TO-PRIOR-ART", (
            f"1/3 baseline KILLS, no clear SURVIVES. Core claim undermined."
        )
    elif len(kills) == 0 and len(survives) >= 2:
        return "REAL BREAKTHROUGH", (
            f"{len(survives)}/3 baselines SURVIVE. All tested structural components are load-bearing. "
            f"Tier-0/Tier-1 design is genuinely novel and effective."
        )
    elif len(kills) == 1 and len(survives) >= 1:
        killed_tiers = []
        if v1 == "KILL":
            killed_tiers.append("tier-structure (flat INT4 sufficient)")
        if v2 == "KILL":
            killed_tiers.append("Tier-0 channel selection (random works equally)")
        if v3 == "KILL":
            killed_tiers.append("Tier-1 dynamicity (static sufficient)")
        surviving_tiers = []
        if v1 == "SURVIVES":
            surviving_tiers.append("tier-structure")
        if v2 == "SURVIVES":
            surviving_tiers.append("Tier-0 CF3 selection")
        if v3 == "SURVIVES":
            surviving_tiers.append("Tier-1 dynamicity")
        return "PARTIAL", (
            f"Mixed verdict. KILLED: {', '.join(killed_tiers)}. SURVIVING: {', '.join(surviving_tiers)}."
        )
    else:
        # 0 kills, 0-1 survives, rest marginal
        return "PARTIAL", (
            f"No decisive kills, but insufficient SURVIVES ({len(survives)}/3). "
            f"Results are inconclusive — tiers are marginally helpful but no single component is clearly the moat."
        )


# ---------------------------------------------------------------------------
# Write output files.
# ---------------------------------------------------------------------------

def write_outputs(
    results: dict[str, Any],
    out_dir: Path,
    result_md_dir: Path,
) -> None:
    # JSON.
    json_path = out_dir / "adversarial_results.json"
    # Merge with existing if present.
    existing: dict[str, Any] = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(results)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    # Markdown — only write if all three baselines present.
    b1 = existing.get("flat_int4")
    b2 = existing.get("random_tier0")
    b3 = existing.get("static_tier1")
    if b1 and b2 and b3:
        _write_result_md(existing, result_md_dir / "rraok_adversarial_result.md")
    else:
        present = [k for k in ["flat_int4", "random_tier0", "static_tier1"] if existing.get(k)]
        print(f"  (MD not written yet — only {present} complete; need all 3)", flush=True)


def _write_result_md(data: dict, path: Path) -> None:
    raok_mean = data.get("raok_mean_delta_nll", float("nan"))
    b1 = data.get("flat_int4", {})
    b2 = data.get("random_tier0", {})
    b3 = data.get("static_tier1", {})
    agg = data.get("aggregate_verdict", "UNKNOWN")
    agg_note = data.get("aggregate_verdict_note", "")

    def fmt_row(d_: dict) -> str:
        seeds = d_.get("seed_records", [])
        rows = ""
        for s in seeds:
            rows += f"| {s['seed_idx']+1} | {s['offset']} | {s['baseline_nll']:.4f} | {s['quant_nll']:.4f} | {s['delta_nll']:.4f} |\n"
        return rows

    md = f"""# R-RAOK-70B Adversarial Baseline Results

Date: {datetime.now(UTC).strftime("%Y-%m-%d")}
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_adversarial.py`
RAOK Rung 2 reference mean ΔNLL: **{raok_mean:.4f} nats**

---

## Aggregate Verdict: {agg}

{agg_note}

---

## Adversarial 1 — Flat INT4 (no tier structure)

**Verdict: {b1.get('verdict', 'PENDING')}**

{b1.get('verdict_note', '')}

Mean ΔNLL: **{b1.get('mean_delta_nll', float('nan')):.4f} ± {b1.get('std_delta_nll', float('nan')):.4f} nats**
Gap vs RAOK: **{b1.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_row(b1)}

---

## Adversarial 2 — Random Tier-0 channels

**Verdict: {b2.get('verdict', 'PENDING')}**

{b2.get('verdict_note', '')}

Mean ΔNLL: **{b2.get('mean_delta_nll', float('nan')):.4f} ± {b2.get('std_delta_nll', float('nan')):.4f} nats**
Gap vs RAOK: **{b2.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} nats**

Random Tier-0 seed: 999 (numpy default_rng). Channels drawn without replacement, sorted per layer.

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_row(b2)}

---

## Adversarial 3 — Static Tier-1 (no per-token rotation)

**Verdict: {b3.get('verdict', 'PENDING')}**

{b3.get('verdict_note', '')}

Mean ΔNLL: **{b3.get('mean_delta_nll', float('nan')):.4f} ± {b3.get('std_delta_nll', float('nan')):.4f} nats**
Gap vs RAOK: **{b3.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} nats**

Static Tier-1 = top-18 channels by corpus mean magnitude (same as calibration pass), fixed across all tokens.

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_row(b3)}

---

## Summary Table

| Baseline | Mean ΔNLL | Std | Gap vs RAOK | Verdict |
|---|---|---|---|---|
| RAOK three-tier (Rung 2) | {raok_mean:.4f} | — | 0.0000 | reference |
| Flat INT4 (Adv-1) | {b1.get('mean_delta_nll', float('nan')):.4f} | {b1.get('std_delta_nll', float('nan')):.4f} | {b1.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} | {b1.get('verdict', 'PENDING')} |
| Random Tier-0 (Adv-2) | {b2.get('mean_delta_nll', float('nan')):.4f} | {b2.get('std_delta_nll', float('nan')):.4f} | {b2.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} | {b2.get('verdict', 'PENDING')} |
| Static Tier-1 (Adv-3) | {b3.get('mean_delta_nll', float('nan')):.4f} | {b3.get('std_delta_nll', float('nan')):.4f} | {b3.get('mean_delta_nll', float('nan')) - raok_mean:+.4f} | {b3.get('verdict', 'PENDING')} |

---

## Interpretation Guide

- **KILL**: The attacked component is NOT load-bearing. RAOK's claim for that component fails.
- **MARGINAL**: Component provides minimal benefit (gap < threshold). Inconclusive.
- **SURVIVES**: Component IS load-bearing. Gap clearly exceeds threshold.

Kill thresholds:
- Adv-1 (flat INT4): ΔNLL <= 0.10 nats → KILL; gap >= 0.20 nats → SURVIVES
- Adv-2 (random Tier-0): gap <= 0.05 nats → KILL; gap >= 0.10 nats → SURVIVES
- Adv-3 (static Tier-1): gap <= 0.05 nats → KILL; gap >= 0.10 nats → SURVIVES

If Adv-1 KILLS: RAOK reduces to flat INT4 activation quantization (LLM-FP4, SmoothQuant).
If Adv-2 KILLS: CF3-derived calibration-free channel identification is not the moat.
If Adv-3 KILLS: CF3 K=1% dynamicity claim is not exploited by RAOK's Tier-1 design.
"""
    path.write_text(md, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="R-RAOK-70B adversarial baselines")
    parser.add_argument("--flat-int4", action="store_true", help="Run adversarial 1: flat INT4")
    parser.add_argument("--random-tier0", action="store_true", help="Run adversarial 2: random Tier-0")
    parser.add_argument("--static-tier1", action="store_true", help="Run adversarial 3: static Tier-1")
    parser.add_argument("--all", action="store_true", help="Run all three baselines sequentially")
    parser.add_argument("--raok-mean-nll", type=float, default=None,
                        help="RAOK Rung 2 mean ΔNLL (auto-read from rung2_results.json if not given)")
    args = parser.parse_args()

    if args.all:
        args.flat_int4 = True
        args.random_tier0 = True
        args.static_tier1 = True

    if not any([args.flat_int4, args.random_tier0, args.static_tier1]):
        parser.error("Specify at least one of --flat-int4, --random-tier0, --static-tier1, --all")

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

    # B4 unit test.
    print(f"[{time.time()-t0:.1f}s] Running B4 unit test ...", flush=True)
    _b4_unit_test()

    # Read RAOK Rung 2 mean ΔNLL for gap comparisons.
    raok_mean_delta: float
    if args.raok_mean_nll is not None:
        raok_mean_delta = args.raok_mean_nll
        print(f"[{time.time()-t0:.1f}s] Using supplied RAOK mean ΔNLL: {raok_mean_delta:.4f} nats", flush=True)
    else:
        rung2_json = OUT_DIR / "rung2_results.json"
        if rung2_json.exists():
            rung2_data = json.loads(rung2_json.read_text(encoding="utf-8"))
            raok_mean_delta = float(rung2_data["mean_delta_nll"])
            print(f"[{time.time()-t0:.1f}s] Read RAOK mean ΔNLL from rung2_results.json: {raok_mean_delta:.4f} nats", flush=True)
        else:
            print(f"[{time.time()-t0:.1f}s] WARNING: rung2_results.json not found. Using placeholder 0.0 nats.", flush=True)
            print(f"  Run with --raok-mean-nll <value> to supply the Rung 2 result explicitly.", flush=True)
            raok_mean_delta = 0.0

    # Load model (once, shared across all baselines).
    print(f"\n[{time.time()-t0:.1f}s] Loading {MODEL_ID} bf16 on CPU ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}", flush=True)
    assert d == 2048, f"Expected d=2048 for Qwen3-1.7B, got {d}"
    assert n_layers == 28, f"Expected 28 layers, got {n_layers}"

    # Calibration pass.
    print(f"\n[{time.time()-t0:.1f}s] Calibration pass (200 prompts) ...", flush=True)
    calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
    tier0_per_layer, tier1_static_per_layer = compute_tier0_channels(
        model, tokenizer, calib_prompts, n_layers, d, DEVICE,
        max_length=MAX_LENGTH, tier0_k=TIER0_K, tier1_k=TIER1_K,
    )
    print(f"[{time.time()-t0:.1f}s] Calibration done.", flush=True)

    # Result accumulator (merge with any existing partial results).
    json_path = OUT_DIR / "adversarial_results.json"
    all_results: dict[str, Any] = {}
    if json_path.exists():
        try:
            all_results = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    all_results["raok_mean_delta_nll"] = raok_mean_delta
    all_results["timestamp"] = datetime.now(UTC).isoformat()
    all_results["model_id"] = MODEL_ID

    # Run requested baselines.
    baselines_to_run: list[str] = []
    if args.flat_int4:
        baselines_to_run.append("flat_int4")
    if args.random_tier0:
        baselines_to_run.append("random_tier0")
    if args.static_tier1:
        baselines_to_run.append("static_tier1")

    for baseline_name in baselines_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"[{time.time()-t0:.1f}s] Running baseline: {baseline_name}", flush=True)
        print(f"{'='*60}", flush=True)

        mean_delta, std_delta, seed_deltas, seed_records = run_baseline(
            name=baseline_name,
            model=model,
            tokenizer=tokenizer,
            tier0_per_layer=tier0_per_layer,
            tier1_static_per_layer=tier1_static_per_layer,
            n_layers=n_layers,
            d=d,
            device=DEVICE,
            eval_offsets=EVAL_OFFSETS,
            eval_n_tokens=EVAL_N_TOKENS,
            chunk_size=CHUNK_SIZE,
            tier1_k=TIER1_K,
            t0_global=t0,
        )

        # Verdict.
        if baseline_name == "flat_int4":
            verdict, verdict_note = verdict_flat_int4(mean_delta, raok_mean_delta)
        elif baseline_name == "random_tier0":
            verdict, verdict_note = verdict_random_tier0(mean_delta, raok_mean_delta)
        else:
            verdict, verdict_note = verdict_static_tier1(mean_delta, raok_mean_delta)

        print(f"\n[{baseline_name}] VERDICT: {verdict}", flush=True)
        print(f"  {verdict_note}", flush=True)

        all_results[baseline_name] = {
            "mean_delta_nll": mean_delta,
            "std_delta_nll": std_delta,
            "seed_deltas": seed_deltas,
            "seed_records": seed_records,
            "verdict": verdict,
            "verdict_note": verdict_note,
            "gap_vs_raok": mean_delta - raok_mean_delta,
        }

        # Checkpoint after each baseline.
        json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Checkpointed {json_path}", flush=True)

    # Aggregate verdict (only if all three are present).
    b1 = all_results.get("flat_int4", {})
    b2 = all_results.get("random_tier0", {})
    b3 = all_results.get("static_tier1", {})

    if b1 and b2 and b3:
        agg, agg_note = aggregate_verdict(b1["verdict"], b2["verdict"], b3["verdict"])
        print(f"\n{'='*60}", flush=True)
        print(f"AGGREGATE VERDICT: {agg}", flush=True)
        print(f"  {agg_note}", flush=True)
        print(f"{'='*60}", flush=True)
        all_results["aggregate_verdict"] = agg
        all_results["aggregate_verdict_note"] = agg_note
    else:
        present = [k for k in ["flat_int4", "random_tier0", "static_tier1"] if all_results.get(k)]
        print(f"\nPartial run complete: {present}. Run remaining baselines to get aggregate verdict.", flush=True)

    all_results["elapsed_seconds"] = time.time() - t0

    write_outputs(all_results, OUT_DIR, RESULT_MD_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
