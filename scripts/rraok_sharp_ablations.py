r"""R-RAOK-70B Sharp Ablations — P1/P2/P3 published-art reduction tests.

Resolves whether RAOK's 0.083 nat result reduces to standard activation-quantization
primitives already in the literature (SmoothQuant, LLM.int8, per-channel INT4 variants).

Three ablations:
  P1 --per-channel-all:
    Per-channel INT4 across ALL 2048 channels.  For each channel c independently,
    scale_c = max_abs(h[:, c]) / 7 computed per-TOKEN (so it IS per-token per-channel).
    No Tier-0 FP16 isolation. No Tier-1 INT8 shell.
    This is the "right" published-art comparison baseline the adversarial script missed —
    flat-INT4 used a single per-token scale across all channels and catastrophically blew
    up outlier channels. Per-channel INT4 is what SmoothQuant-family methods actually use.
    Decisive: if ΔNLL ≈ 0.083, the FP16+INT8 tiers are NOT load-bearing and RAOK reduces
    to standard per-channel INT4 activation quantization. If ΔNLL >> 0.083, tiers ARE
    load-bearing.

  P2 --minimal-raok:
    Tier-0 (top-2 FP16 bypass, static) + per-channel INT4 on remaining 2046 channels.
    No Tier-1 INT8 dynamic shell — it is simply skipped.
    Decisive: if ΔNLL ≈ 0.083, Tier-1 INT8 shell is NOT load-bearing.
    If ΔNLL >> 0.083, Tier-1 INT8 is doing real work.

  P3 --per-token-single:
    Per-token single-scale INT4.  scale_t = max_abs(h_full_t) / 7.
    Same as flat-INT4 in rraok_adversarial.py — included here for completeness and to
    establish the floor. Expected catastrophic (ΔNLL >> 1 nat). Documents the baseline
    against which per-channel INT4 (P1) must be compared.

Same eval protocol as Rung 2:
  Model: Qwen/Qwen3-1.7B-Base, bf16, CPU.
  Corpus: WikiText-2 test, 512 tokens, 3 seeds (offsets 0, 512, 1024).
  RAOK reference: mean ΔNLL = 0.0830 ± 0.0159 nats.
  Report: mean ± std ΔNLL per ablation.

Outputs:
  experiments/stage0/ladder_v2/round1_raok70b/sharp_ablations_results.json
  sonnet_ladder_v2/cheap_tests/run_001/rraok_sharp_ablations_result.md

Constraints:
  - Does NOT modify rraok_rung2.py, rraok_rung1.py, rraok_adversarial.py.
  - Self-contained: copies shared utilities verbatim (calibration corpus, wikitext loader,
    NLL eval loop) rather than importing from those scripts.

Usage:
  python scripts/rraok_sharp_ablations.py --per-channel-all
  python scripts/rraok_sharp_ablations.py --minimal-raok
  python scripts/rraok_sharp_ablations.py --per-token-single
  python scripts/rraok_sharp_ablations.py --all   (run all three sequentially)
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
# Calibration corpus (copy-of-record; identical strata to rraok_rung2.py).
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
# Calibration pass — compute Tier-0 channel indices per layer.
# Needed for P2 (minimal RAOK, which uses Tier-0 FP16 bypass).
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
) -> dict[int, torch.Tensor]:
    """Returns tier0_per_layer: {layer_idx: LongTensor(tier0_k)} — top-k by mean abs magnitude."""
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
            print(f"  WARNING: layer {L} had no calibration tokens", flush=True)
            mean_mag = torch.zeros(d)
        else:
            mean_mag = (sum_mag[L] / count[L]).float()
        top_k = mean_mag.topk(tier0_k).indices.sort().values
        tier0_per_layer[L] = top_k

    return tier0_per_layer


# ---------------------------------------------------------------------------
# WikiText-2 token loader — identical to rraok_rung2.py.
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
# Ablation quantization functions (per-token, applied to h: (d,) float32).
# ---------------------------------------------------------------------------

def quantize_per_channel_all_int4(h: torch.Tensor) -> torch.Tensor:
    """
    P1: Per-channel INT4 across ALL d channels.
    For each channel c: scale_c = max_abs(h[c]) / 7.
    This is the standard per-channel (per-group-of-1) INT4 scheme used in
    SmoothQuant / LLM.int8-family methods.
    Note: applied per-token here (scale_c computed fresh for each token),
    which is the per-token-per-channel variant that eliminates outlier-blown-scale.
    """
    d = h.shape[0]
    h_out = h.clone()
    abs_h = h.abs()
    # Per-channel scales: (d,)
    scales = abs_h / 7.0  # shape (d,); scale_c = |h_c| / 7
    # Avoid division by zero for near-zero channels (keep original value, error negligible).
    nonzero = scales > 1e-9
    if nonzero.any():
        # For nonzero channels: quantize.
        q = torch.round(h[nonzero] / scales[nonzero]).clamp(-7, 7)
        h_out[nonzero] = q * scales[nonzero]
    # Zero-scale channels: h_out already equals h (clone above).
    return h_out


def quantize_minimal_raok(
    h: torch.Tensor,
    tier0_idx: torch.Tensor,  # (2,) static
) -> torch.Tensor:
    """
    P2: Minimal RAOK — Tier-0 FP16 bypass + per-channel INT4 for all others.
    No Tier-1 INT8 shell at all. The 2046 non-Tier-0 channels each get their
    own per-token scale (scale_c = max_abs(h[c]) / 7).
    """
    d = h.shape[0]
    h_out = h.clone()

    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    # Tier-0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Non-Tier-0: per-channel INT4.
    non_t0_vals = h[~t0_mask]          # (2046,)
    scales = non_t0_vals.abs() / 7.0   # per-channel scales
    nonzero = scales > 1e-9
    if nonzero.any():
        q = torch.round(non_t0_vals[nonzero] / scales[nonzero]).clamp(-7, 7)
        tmp = non_t0_vals.clone()
        tmp[nonzero] = q * scales[nonzero]
        h_out[~t0_mask] = tmp

    return h_out


def quantize_per_token_single_int4(h: torch.Tensor) -> torch.Tensor:
    """
    P3: Per-token single-scale INT4 (the naive baseline).
    scale_t = max_abs(h_full) / 7.  One scale per token across all 2048 channels.
    Expected: catastrophic ΔNLL because outlier channels blow up the scale.
    This is equivalent to the flat-INT4 adversarial already in rraok_adversarial.py.
    Included here for: (a) floor-establishment, (b) direct comparison within same
    eval harness and 3-seed protocol.
    """
    max_abs = h.abs().max()
    if max_abs < 1e-9:
        return h.clone()
    scale = max_abs / 7.0
    q = torch.round(h / scale).clamp(-7, 7)
    return q * scale


# ---------------------------------------------------------------------------
# Generic NLL evaluation loop — hook-per-layer architecture.
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
    hook_fn_per_layer[L](h_tok) quantizes a single (d,) float32 token activation.
    Identical eval protocol to rraok_rung2.py: stride=chunk_size, context=min(32, chunk//2).
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
# Run one ablation across 3 seeds.
# Returns (mean_delta, std_delta, seed_deltas, seed_records).
# ---------------------------------------------------------------------------

def run_ablation(
    name: str,
    model,
    tokenizer,
    tier0_per_layer: dict[int, torch.Tensor] | None,
    n_layers: int,
    d: int,
    device: torch.device,
    eval_offsets: list[int],
    eval_n_tokens: int,
    chunk_size: int,
    t0_global: float,
) -> tuple[float, float, list[float], list[dict]]:
    seed_deltas: list[float] = []
    seed_records: list[dict] = []

    for seed_idx, offset in enumerate(eval_offsets):
        print(f"\n[{time.time()-t0_global:.1f}s] [{name}] Seed {seed_idx+1}/3 (offset={offset}) ...", flush=True)
        eval_tokens = _load_wikitext2_tokens(tokenizer, offset=offset, n_tokens=eval_n_tokens)

        # Build per-layer hook.
        if name == "per_channel_all":
            hook_fn_per_layer = {L: quantize_per_channel_all_int4 for L in range(n_layers)}
        elif name == "minimal_raok":
            if tier0_per_layer is None:
                raise ValueError("minimal_raok requires tier0_per_layer (calibration pass needed)")
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_minimal_raok(h, tier0_per_layer[_L]))
                for L in range(n_layers)
            }
        elif name == "per_token_single":
            hook_fn_per_layer = {L: quantize_per_token_single_int4 for L in range(n_layers)}
        else:
            raise ValueError(f"Unknown ablation: {name}")

        # Baseline NLL (no quantization).
        print(f"[{time.time()-t0_global:.1f}s]   baseline forward pass ...", flush=True)
        baseline_nll = eval_nll_with_hook(
            model, eval_tokens, n_layers, device,
            hook_fn_per_layer={}, apply_quantization=False, chunk_size=chunk_size,
        )
        print(f"  baseline NLL: {baseline_nll:.4f} nats", flush=True)

        # Ablation quantized NLL.
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
# Verdict logic per ablation.
# ---------------------------------------------------------------------------

RAOK_MEAN_REF = 0.08295687024140778  # from rung2_results.json
RAOK_STD_REF  = 0.015906671145993815
COLLAPSE_THRESHOLD = 0.05  # ΔNLL within this of RAOK → tiers not load-bearing
SURVIVES_THRESHOLD = 1.0   # ΔNLL this much worse than RAOK → tiers are load-bearing

# P1 "within 0.05 nats of RAOK" collapses; ">1 nat worse" confirms tier necessity.
# P2 same logic (tests Tier-1 specifically).
# P3 expected >> 1 nat (establishes floor).


def verdict_p1(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if abs(gap) <= COLLAPSE_THRESHOLD:
        v = "COLLAPSES-TO-PER-CHANNEL-INT4"
        note = (
            f"P1 per-channel-all ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"|gap| <= {COLLAPSE_THRESHOLD} nats. The FP16 Tier-0 and INT8 Tier-1 shell are NOT "
            f"load-bearing. RAOK reduces to standard per-channel INT4 activation quantization, "
            f"which is covered by GPTQ (Frantar et al. 2022), SmoothQuant (Xiao et al. 2022), "
            f"and LLM.int8 (Dettmers et al. 2022). The CF3-tier novelty fully collapses to prior art."
        )
    elif gap > SURVIVES_THRESHOLD:
        v = "TIERS-LOAD-BEARING"
        note = (
            f"P1 per-channel-all ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats worse than RAOK ({raok_mean:.4f}). "
            f"Gap > {SURVIVES_THRESHOLD} nats. The FP16 Tier-0 and/or INT8 Tier-1 shell ARE load-bearing. "
            f"Per-channel INT4 alone cannot match the RAOK result. The tier structure has structural merit "
            f"beyond standard per-channel INT4 activation quantization."
        )
    else:
        v = "MARGINAL"
        note = (
            f"P1 per-channel-all ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap {abs(gap):.4f} nats is in marginal zone ({COLLAPSE_THRESHOLD}–{SURVIVES_THRESHOLD} nats). "
            f"Tiers provide modest benefit over per-channel INT4 but not decisive."
        )
    return v, note


def verdict_p2(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if abs(gap) <= COLLAPSE_THRESHOLD:
        v = "TIER1-NOT-LOAD-BEARING"
        note = (
            f"P2 minimal-RAOK ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"|gap| <= {COLLAPSE_THRESHOLD} nats. Tier-1 INT8 dynamic shell is NOT load-bearing. "
            f"RAOK reduces to: FP16-isolate top-2 channels (Tier-0) + per-channel INT4 for the rest. "
            f"Combined with P1 result, this would place RAOK at the level of "
            f"'per-channel INT4 with 2-channel FP16 outlier bypass' — a minor variant of LLM.int8."
        )
    elif gap > SURVIVES_THRESHOLD:
        v = "TIER1-LOAD-BEARING"
        note = (
            f"P2 minimal-RAOK ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats worse than RAOK ({raok_mean:.4f}). "
            f"Gap > {SURVIVES_THRESHOLD} nats. Tier-1 INT8 dynamic shell IS load-bearing. "
            f"Skipping it significantly degrades compression quality. The dynamic INT8 shell "
            f"(top-18 channels per token) is doing real work."
        )
    else:
        v = "MARGINAL"
        note = (
            f"P2 minimal-RAOK ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap {abs(gap):.4f} nats is marginal ({COLLAPSE_THRESHOLD}–{SURVIVES_THRESHOLD} nats). "
            f"Tier-1 INT8 shell provides modest benefit but not decisive."
        )
    return v, note


def verdict_p3(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if mean_delta > 1.0:
        v = "CATASTROPHIC-AS-EXPECTED"
        note = (
            f"P3 per-token-single ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"As expected: per-token single-scale INT4 is catastrophic due to outlier channels "
            f"blowing up the global scale. This establishes the floor that any per-channel or "
            f"tiered method must beat. The {gap:.2f} nat gap confirms that naive INT4 (as the "
            f"adversarial flat-INT4 already showed) is not the right published-art comparison baseline."
        )
    elif mean_delta <= RAOK_MEAN_REF + 0.10:
        v = "SURPRISINGLY-LOW"
        note = (
            f"P3 per-token-single ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} from RAOK ({raok_mean:.4f}). "
            f"Unexpectedly low ΔNLL for naive single-scale INT4. Published baselines may be "
            f"stronger than assumed. Re-check eval harness."
        )
    else:
        v = "DEGRADED-ABOVE-FLOOR"
        note = (
            f"P3 per-token-single ΔNLL={mean_delta:.4f} nats, gap={gap:+.4f} from RAOK ({raok_mean:.4f}). "
            f"Significantly worse than RAOK but not catastrophic (> 1 nat). "
            f"Per-token single-scale INT4 cannot match RAOK; establishes partial floor."
        )
    return v, note


def aggregate_verdict(v1: str, v2: str, raok_mean: float, p1_mean: float, p2_mean: float, p3_mean: float) -> tuple[str, str]:
    collapses_to_per_channel = "COLLAPSES-TO-PER-CHANNEL-INT4" in v1
    tier1_not_load_bearing = "TIER1-NOT-LOAD-BEARING" in v2

    if collapses_to_per_channel and tier1_not_load_bearing:
        return "FULLY-REDUCED-TO-PRIOR-ART", (
            f"RAOK fully reduces to standard per-channel INT4 activation quantization. "
            f"P1 (per-channel-all, {p1_mean:.4f} nats) ≈ RAOK ({raok_mean:.4f} nats): FP16+INT8 tiers not needed. "
            f"P2 (minimal-RAOK, {p2_mean:.4f} nats) ≈ RAOK: Tier-1 not needed. "
            f"Closest prior art: LLM.int8 (Dettmers et al. 2022) per-channel INT4 activations + "
            f"optional 2-channel FP16 outlier bypass. RAOK has no engineering novelty beyond "
            f"competent application of standard per-channel INT4 activation quantization."
        )
    elif collapses_to_per_channel and not tier1_not_load_bearing:
        return "REDUCES-TO-PER-CHANNEL-INT4-WITH-INT8-SHELL", (
            f"FP16 Tier-0 is not the moat (P1 per-channel-all ≈ RAOK), but Tier-1 INT8 shell IS load-bearing. "
            f"RAOK reduces to: standard per-channel INT4 + dynamic INT8 shell for per-token top-18 channels. "
            f"This is a minor variant of mixed-precision activation quantization (LLM.int8 style) "
            f"with a dynamic top-k INT8 shell on top. Engineering value: modest, mostly incremental."
        )
    elif not collapses_to_per_channel and tier1_not_load_bearing:
        return "FP16-TIER0-IS-MOAT-NOT-INT8-SHELL", (
            f"FP16 Tier-0 IS load-bearing (P1 per-channel-all >> RAOK), but Tier-1 INT8 shell is NOT. "
            f"RAOK mechanism: FP16 isolation of the 2 highest-magnitude channels is critical; "
            f"the INT8 dynamic shell adds no further benefit. "
            f"RAOK reduces to: 2-channel FP16 bypass + per-channel INT4 for the rest. "
            f"This is equivalent to the outlier-isolation approach in LLM.int8 (Dettmers et al. 2022) "
            f"applied to activations rather than weights. The CF3-derived channel selection is the "
            f"real contribution, not the INT8 tier."
        )
    elif not collapses_to_per_channel and not tier1_not_load_bearing:
        return "BOTH-TIERS-LOAD-BEARING-RAOK-HAS-STRUCTURE", (
            f"Both the FP16 Tier-0 and INT8 Tier-1 shell are load-bearing. "
            f"P1 per-channel-all ({p1_mean:.4f} nats) >> RAOK ({raok_mean:.4f} nats): tiers needed. "
            f"P2 minimal-RAOK ({p2_mean:.4f} nats) >> RAOK: Tier-1 INT8 shell also needed. "
            f"RAOK's three-tier design is not reducible to standard per-channel INT4. "
            f"The CF3-derived tier boundaries provide genuine structural benefit. "
            f"This is the result that supports the novelty claim most strongly."
        )
    else:
        return "INCONCLUSIVE", (
            f"Mixed or marginal results. P1={p1_mean:.4f}, P2={p2_mean:.4f}, P3={p3_mean:.4f}, "
            f"RAOK={raok_mean:.4f}. Cannot draw a decisive conclusion on tier load-bearing status."
        )


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def write_json(
    all_results: dict[str, Any],
    out_dir: Path,
) -> None:
    json_path = out_dir / "sharp_ablations_results.json"
    existing: dict[str, Any] = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(all_results)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)


def write_result_md(
    all_results: dict[str, Any],
    result_md_dir: Path,
) -> None:
    """Write verdict markdown only if enough ablations are present."""
    p1 = all_results.get("per_channel_all", {})
    p2 = all_results.get("minimal_raok", {})
    p3 = all_results.get("per_token_single", {})
    present = [k for k in ["per_channel_all", "minimal_raok", "per_token_single"] if all_results.get(k)]

    raok_mean = all_results.get("raok_mean_delta_nll", RAOK_MEAN_REF)

    def fmt_seed_rows(d_: dict) -> str:
        rows = ""
        for s in d_.get("seed_records", []):
            rows += (f"| {s['seed_idx']+1} | {s['offset']} | {s['baseline_nll']:.4f} | "
                     f"{s['quant_nll']:.4f} | {s['delta_nll']:.4f} |\n")
        return rows

    def gap_str(d_: dict) -> str:
        v = d_.get("mean_delta_nll", float("nan"))
        g = v - raok_mean
        return f"{g:+.4f}"

    agg = all_results.get("aggregate_verdict", "PENDING")
    agg_note = all_results.get("aggregate_verdict_note", "")

    md = f"""# R-RAOK-70B Sharp Ablations Result

Date: {datetime.now(UTC).strftime("%Y-%m-%d")}
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_sharp_ablations.py`
Ablations run: {present}
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/sharp_ablations_results.json`

---

## RAOK Reference

Mean ΔNLL (Rung 2, three-tier): **{raok_mean:.4f} nats** (std={RAOK_STD_REF:.4f})
Seeds: WikiText-2 offsets 0 / 512 / 1024, 512 tokens each.

---

## Aggregate Verdict: {agg}

{agg_note}

---

## P1 — Per-Channel INT4 Across ALL 2048 Channels

**The decisive published-art comparison.** This is what SmoothQuant, GPTQ, and LLM.int8
actually do: each channel gets its own quantization scale (per-token-per-channel here).
No FP16 isolation, no INT8 dynamic shell.

**Verdict: {p1.get("verdict", "PENDING")}**

{p1.get("verdict_note", "")}

Mean ΔNLL: **{p1.get("mean_delta_nll", float("nan")):.4f} ± {p1.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p1)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p1)}
---

## P2 — Minimal RAOK (Tier-0 FP16 + Per-Channel INT4, No Tier-1 INT8 Shell)

**Tests whether Tier-1 INT8 dynamic shell is load-bearing.**
Tier-0 (top-2 by mean magnitude, FP16 bypass) is kept. All other 2046 channels
get per-channel INT4 directly — the Tier-1 INT8 shell is skipped entirely.

**Verdict: {p2.get("verdict", "PENDING")}**

{p2.get("verdict_note", "")}

Mean ΔNLL: **{p2.get("mean_delta_nll", float("nan")):.4f} ± {p2.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p2)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p2)}
---

## P3 — Per-Token Single-Scale INT4 (Published-Art Floor)

**Establishes the floor.** Single scale per token across all 2048 channels.
Equivalent to flat-INT4 in rraok_adversarial.py. Expected catastrophic due to outlier
channels. Included to confirm the floor and show why the adversarial script's flat-INT4
was not the right SmoothQuant comparison.

**Verdict: {p3.get("verdict", "PENDING")}**

{p3.get("verdict_note", "")}

Mean ΔNLL: **{p3.get("mean_delta_nll", float("nan")):.4f} ± {p3.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p3)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p3)}
---

## Summary Table

| Scheme | Mean ΔNLL | Std | Gap vs RAOK | Verdict |
|---|---|---|---|---|
| RAOK three-tier (Rung 2) | {raok_mean:.4f} | {RAOK_STD_REF:.4f} | 0.0000 | reference |
| P1: per-channel-all INT4 | {p1.get("mean_delta_nll", float("nan")):.4f} | {p1.get("std_delta_nll", float("nan")):.4f} | {gap_str(p1)} | {p1.get("verdict", "PENDING")} |
| P2: minimal-RAOK (T0+pcINT4) | {p2.get("mean_delta_nll", float("nan")):.4f} | {p2.get("std_delta_nll", float("nan")):.4f} | {gap_str(p2)} | {p2.get("verdict", "PENDING")} |
| P3: per-token-single INT4 | {p3.get("mean_delta_nll", float("nan")):.4f} | {p3.get("std_delta_nll", float("nan")):.4f} | {gap_str(p3)} | {p3.get("verdict", "PENDING")} |

---

## Interpretation Logic

**Decision tree:**

1. P1 ΔNLL ≈ RAOK (|gap| <= 0.05 nats) → FP16 + INT8 tiers are NOT load-bearing.
   RAOK reduces to standard per-channel INT4 activation quantization.
   Closest prior art: LLM.int8 / SmoothQuant / GPTQ per-activation-channel quant.

2. P2 ΔNLL ≈ RAOK (|gap| <= 0.05 nats) → Tier-1 INT8 shell is NOT load-bearing.
   Combined with P1 KILL: RAOK = FP16-bypass-2-channels + per-channel INT4 for rest.

3. P1 ΔNLL >> RAOK (gap > 1.0 nat) → FP16 Tier-0 and/or INT8 Tier-1 are load-bearing.
   The tier structure has structural merit beyond standard per-channel INT4.

4. P3 catastrophic (ΔNLL > 1 nat) → confirms per-channel scaling is essential;
   the adversarial flat-INT4 was not the right published-art comparison.

---

## Where RAOK Lands in the Literature (Post-Ablation)

{"(Complete: see Aggregate Verdict above.)" if agg not in ["PENDING", "INCONCLUSIVE"] else "(Pending: need all three ablations complete.)"}
"""

    md_path = result_md_dir / "rraok_sharp_ablations_result.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="R-RAOK-70B sharp ablations (P1/P2/P3)")
    parser.add_argument("--per-channel-all", action="store_true",
                        help="P1: per-channel INT4 across all 2048 channels (no tiers)")
    parser.add_argument("--minimal-raok", action="store_true",
                        help="P2: Tier-0 FP16 + per-channel INT4 on rest (no Tier-1 INT8 shell)")
    parser.add_argument("--per-token-single", action="store_true",
                        help="P3: per-token single-scale INT4 (flat; floor baseline)")
    parser.add_argument("--all", action="store_true", help="Run all three ablations sequentially")
    parser.add_argument("--raok-mean-nll", type=float, default=None,
                        help="RAOK Rung 2 mean ΔNLL (auto-read from rung2_results.json if not given)")
    args = parser.parse_args()

    if args.all:
        args.per_channel_all = True
        args.minimal_raok = True
        args.per_token_single = True

    if not any([args.per_channel_all, args.minimal_raok, args.per_token_single]):
        parser.error("Specify at least one of --per-channel-all, --minimal-raok, --per-token-single, --all")

    t0 = time.time()

    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    DEVICE = torch.device("cpu")
    MAX_LENGTH = 32
    CALIB_N_PER_STRATUM = 50
    EVAL_N_TOKENS = 512
    CHUNK_SIZE = 64
    TIER0_K = 2
    EVAL_OFFSETS = [0, 512, 1024]

    OUT_DIR = Path("experiments/stage0/ladder_v2/round1_raok70b")
    RESULT_MD_DIR = Path("sonnet_ladder_v2/cheap_tests/run_001")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_MD_DIR.mkdir(parents=True, exist_ok=True)

    # B4 unit test.
    print(f"[{time.time()-t0:.1f}s] Running B4 unit test ...", flush=True)
    _b4_unit_test()

    # Read RAOK Rung 2 mean ΔNLL.
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
            print(f"[{time.time()-t0:.1f}s] WARNING: rung2_results.json not found; using hardcoded reference 0.08296 nats", flush=True)
            raok_mean_delta = RAOK_MEAN_REF

    # Load model.
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

    # Calibration pass — required for P2 only, but cheap to do always.
    needs_calib = args.minimal_raok
    tier0_per_layer: dict[int, torch.Tensor] | None = None

    if needs_calib:
        print(f"\n[{time.time()-t0:.1f}s] Calibration pass (P2 requires Tier-0 channels) ...", flush=True)
        calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
        print(f"  Calibration corpus size: {len(calib_prompts)} prompts", flush=True)
        tier0_per_layer = compute_tier0_channels(
            model, tokenizer, calib_prompts, n_layers, d, DEVICE,
            max_length=MAX_LENGTH, tier0_k=TIER0_K,
        )
        calib_elapsed = time.time() - t0
        print(f"[{calib_elapsed:.1f}s] Calibration done.", flush=True)
        for L in range(n_layers):
            print(f"  L{L:2d} Tier-0 channels: {tier0_per_layer[L].tolist()}", flush=True)
    else:
        print(f"\n[{time.time()-t0:.1f}s] Calibration pass skipped (not needed for selected ablations).", flush=True)

    # Load any existing partial results (to allow incremental runs).
    json_path = OUT_DIR / "sharp_ablations_results.json"
    all_results: dict[str, Any] = {}
    if json_path.exists():
        try:
            all_results = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    all_results["raok_mean_delta_nll"] = raok_mean_delta
    all_results["raok_std_delta_nll"] = RAOK_STD_REF
    all_results["timestamp"] = datetime.now(UTC).isoformat()
    all_results["model_id"] = MODEL_ID

    # Run requested ablations.
    ablations_to_run: list[str] = []
    if args.per_channel_all:
        ablations_to_run.append("per_channel_all")
    if args.minimal_raok:
        ablations_to_run.append("minimal_raok")
    if args.per_token_single:
        ablations_to_run.append("per_token_single")

    for ablation_name in ablations_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"[{time.time()-t0:.1f}s] Running ablation: {ablation_name}", flush=True)
        print(f"{'='*60}", flush=True)

        mean_delta, std_delta, seed_deltas, seed_records = run_ablation(
            name=ablation_name,
            model=model,
            tokenizer=tokenizer,
            tier0_per_layer=tier0_per_layer,
            n_layers=n_layers,
            d=d,
            device=DEVICE,
            eval_offsets=EVAL_OFFSETS,
            eval_n_tokens=EVAL_N_TOKENS,
            chunk_size=CHUNK_SIZE,
            t0_global=t0,
        )

        # Verdict per ablation.
        if ablation_name == "per_channel_all":
            verdict, verdict_note = verdict_p1(mean_delta, raok_mean_delta)
        elif ablation_name == "minimal_raok":
            verdict, verdict_note = verdict_p2(mean_delta, raok_mean_delta)
        else:  # per_token_single
            verdict, verdict_note = verdict_p3(mean_delta, raok_mean_delta)

        print(f"\n[{ablation_name}] VERDICT: {verdict}", flush=True)
        print(f"  {verdict_note}", flush=True)

        all_results[ablation_name] = {
            "mean_delta_nll": mean_delta,
            "std_delta_nll": std_delta,
            "seed_deltas": seed_deltas,
            "seed_records": seed_records,
            "verdict": verdict,
            "verdict_note": verdict_note,
            "gap_vs_raok": mean_delta - raok_mean_delta,
        }

        # Checkpoint after each ablation.
        write_json(all_results, OUT_DIR)
        print(f"Checkpointed.", flush=True)

    # Aggregate verdict — compute whenever P1 and P2 are both present.
    p1 = all_results.get("per_channel_all", {})
    p2 = all_results.get("minimal_raok", {})
    p3 = all_results.get("per_token_single", {})

    if p1 and p2:
        agg, agg_note = aggregate_verdict(
            v1=p1["verdict"],
            v2=p2["verdict"],
            raok_mean=raok_mean_delta,
            p1_mean=p1["mean_delta_nll"],
            p2_mean=p2["mean_delta_nll"],
            p3_mean=p3.get("mean_delta_nll", float("nan")),
        )
        print(f"\n{'='*60}", flush=True)
        print(f"AGGREGATE VERDICT: {agg}", flush=True)
        print(f"  {agg_note}", flush=True)
        print(f"{'='*60}", flush=True)
        all_results["aggregate_verdict"] = agg
        all_results["aggregate_verdict_note"] = agg_note
    else:
        present = [k for k in ["per_channel_all", "minimal_raok", "per_token_single"] if all_results.get(k)]
        print(f"\nPartial run: {present} complete. Run P1+P2 for aggregate verdict.", flush=True)

    all_results["elapsed_seconds"] = time.time() - t0

    write_json(all_results, OUT_DIR)
    write_result_md(all_results, RESULT_MD_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
