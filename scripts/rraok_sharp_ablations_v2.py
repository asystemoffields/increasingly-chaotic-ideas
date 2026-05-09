r"""R-RAOK-70B Sharp Ablations v2 — Calibration-derived per-channel INT4 (bug-fixed).

## Bug fixed vs rraok_sharp_ablations.py

The original P1/P2 functions computed per-channel INT4 scale as:
    scale_c = |h[c]| / 7   (per-token, per-channel)
This is degenerate: the scale is fit to the single value being quantized,
giving bit-exact reconstruction and ΔNLL = 0.0000. It is NOT per-channel INT4;
it is a no-op.

This version uses **calibration-derived** per-channel scales (standard published art):
    max_abs_L_c  = max over all calibration tokens of |h_L[c]|   (per layer, per channel)
    scale_L_c    = max_abs_L_c / 7                                 FIXED — never shrinks to fit token
At eval: q = round(h[c] / scale_L_c).clamp(-7, 7); h_q[c] = q * scale_L_c.

This is the standard SmoothQuant / AWQ-class per-channel quantization.

## Ablations

  P1-fixed  --per-channel-all-fixed
    Calibration-derived per-channel INT4 across ALL 2048 channels.
    No tiers. Direct SmoothQuant-family comparison.

  P2-fixed  --minimal-raok-fixed
    Tier-0 FP16 bypass (static top-2 channels) + calibration-scale per-channel INT4
    on remaining 2046 channels.  No Tier-1 INT8 dynamic shell.

  P4-new    --per-channel-with-tier1-fixed
    Calibration-scale per-channel INT4 on bulk (Tier-2) PLUS dynamic Tier-1 INT8 shell
    (top-18 per-token after excluding Tier-0). Tier-0 FP16 bypass retained.
    Tests whether calibration-scale Tier-2 + dynamic Tier-1 ≈ RAOK (which uses
    per-token-scale Tier-2 + dynamic Tier-1).

Same eval protocol as Rung 2 / rraok_sharp_ablations.py:
  Model: Qwen/Qwen3-1.7B-Base, bf16, CPU.
  Corpus: WikiText-2 test, 512 tokens, 3 seeds (offsets 0, 512, 1024).
  RAOK reference: mean ΔNLL = 0.0830 ± 0.0159 nats.

Outputs:
  experiments/stage0/ladder_v2/round1_raok70b/sharp_ablations_v2_results.json
  sonnet_ladder_v2/cheap_tests/run_001/rraok_sharp_ablations_v2_result.md

Constraints:
  - Does NOT modify rraok_rung2.py, rraok_rung1.py, rraok_sharp_ablations.py.
  - Self-contained; calibration corpus identical to rraok_rung2.py strata.
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
# B4 unit test — calibration-scale version.
# ---------------------------------------------------------------------------

def _b4_unit_test() -> None:
    """Verify calibration-scale INT4 produces non-trivial error (unlike degenerate per-token-per-channel)."""
    # Calibration scale: max_abs / 7 over many tokens.
    calib_data = torch.tensor([[-7.0, -4.0, 0.0, 4.0, 7.0],
                                [ 3.0,  2.0, 1.0, 0.5, 0.1]])
    # max_abs per channel over calibration data
    max_abs_calib = calib_data.abs().max(dim=0).values  # (5,)
    calib_scales = max_abs_calib / 7.0                  # (5,)

    # Eval token — different from calibration max
    h_eval = torch.tensor([3.5, -1.0, 0.5, 2.0, 6.0])
    q = torch.round(h_eval / calib_scales).clamp(-7, 7)
    h_recon = q * calib_scales
    err = (h_eval - h_recon).abs().max().item()

    # For channel 0: calib max = 7, scale = 1.0, h_eval[0] = 3.5 → q=4, recon=4.0, err=0.5
    # This shows non-trivial error (≠ 0) — not degenerate.
    assert err > 0.0, f"B4-v2 FAIL: calibration-scale INT4 should produce nonzero error, got {err}"

    # Also verify degenerate (per-token-per-channel) gives exactly 0 error on same token.
    per_token_scales = h_eval.abs() / 7.0
    q_degen = torch.round(h_eval / per_token_scales).clamp(-7, 7)
    h_degen_recon = q_degen * per_token_scales
    err_degen = (h_eval - h_degen_recon).abs().max().item()
    assert err_degen < 1e-5, f"B4-v2 sanity: degenerate should give ~0, got {err_degen}"

    print(f"B4-v2 unit test: PASS (calib-scale err={err:.4f} > 0; degenerate err={err_degen:.6f} ≈ 0)", flush=True)


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
# Calibration pass — computes BOTH per-channel max-abs scales AND Tier-0 channels.
# ---------------------------------------------------------------------------

def compute_calibration(
    model,
    tokenizer,
    calib_prompts: list[tuple[str, str]],
    n_layers: int,
    d: int,
    device: torch.device,
    max_length: int = 32,
    tier0_k: int = 2,
    tier1_k: int = 18,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Single calibration pass. Returns:
      calib_scales_per_layer:  {L: Tensor(d,)} — per-channel INT4 scales = max_abs / 7
      tier0_per_layer:         {L: LongTensor(tier0_k,)} — top-k channels by mean |h|
      tier1_per_layer:         {L: LongTensor(tier1_k,)} — next-top-k after excluding tier0

    The per-channel scales are the KEY FIX: they are computed as
        max_abs_L_c = max over all calibration tokens |h_L[c]|
        scale_L_c   = max_abs_L_c / 7
    This is identical to SmoothQuant / AWQ-class per-channel INT4 activation quantization.
    """
    # Track both max_abs (for scales) and sum_abs (for Tier-0 selection by mean magnitude).
    max_abs: dict[int, torch.Tensor] = {L: torch.zeros(d, dtype=torch.float64) for L in range(n_layers)}
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
                # max_abs: running per-channel maximum
                max_abs[L] = torch.maximum(max_abs[L], h.abs().max(dim=0).values.double())
                # sum_mag: for mean-magnitude tier selection
                sum_mag[L] += h.abs().sum(dim=0).double()
                count[L] += h.shape[0]

    for h in handles:
        h.remove()

    calib_scales_per_layer: dict[int, torch.Tensor] = {}
    tier0_per_layer: dict[int, torch.Tensor] = {}
    tier1_per_layer: dict[int, torch.Tensor] = {}

    for L in range(n_layers):
        if count[L] == 0:
            print(f"  WARNING: layer {L} had no calibration tokens", flush=True)
            mean_mag = torch.zeros(d)
            max_abs_L = torch.ones(d)  # fallback — won't be used in practice
        else:
            mean_mag = (sum_mag[L] / count[L]).float()
            max_abs_L = max_abs[L].float()

        # Per-channel INT4 scales — KEY FIX over v1.
        # Floors at 1e-9 to avoid div-by-zero on truly-zero channels.
        scales = (max_abs_L / 7.0).clamp(min=1e-9)
        calib_scales_per_layer[L] = scales

        # Tier-0: top-2 channels by mean |h| (same as rraok_rung2.py).
        top_k0 = mean_mag.topk(tier0_k).indices.sort().values
        tier0_per_layer[L] = top_k0

        # Tier-1: next top-18 by mean |h|, excluding Tier-0 channels.
        t0_mask = torch.zeros(d, dtype=torch.bool)
        t0_mask[top_k0] = True
        mean_mag_non_t0 = mean_mag.clone()
        mean_mag_non_t0[t0_mask] = -1.0  # exclude tier0
        top_k1_global = mean_mag_non_t0.topk(tier1_k).indices.sort().values
        tier1_per_layer[L] = top_k1_global

    return calib_scales_per_layer, tier0_per_layer, tier1_per_layer


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
# Fixed quantization functions — calibration-derived scales.
# ---------------------------------------------------------------------------

def quantize_per_channel_all_fixed(
    h: torch.Tensor,              # (d,) float32 — one token's activation
    calib_scales: torch.Tensor,   # (d,) float32 — pre-computed from calibration
) -> torch.Tensor:
    """
    P1-fixed: Per-channel INT4 across ALL d channels using calibration-derived scales.

    scale_L_c = max_abs_calib_L_c / 7   (fixed across all eval tokens)
    q = round(h[c] / scale_L_c).clamp(-7, 7)
    h_q[c] = q * scale_L_c

    This is the standard SmoothQuant / AWQ-class per-channel INT4 activation quantization.
    Unlike the buggy v1, scale is NOT fit to the individual token value, so quantization
    error is non-trivial.
    """
    q = torch.round(h / calib_scales).clamp(-7, 7)
    return q * calib_scales


def quantize_minimal_raok_fixed(
    h: torch.Tensor,              # (d,) float32 — one token's activation
    tier0_idx: torch.Tensor,      # (2,) LongTensor — static Tier-0 channel indices
    calib_scales: torch.Tensor,   # (d,) float32 — per-channel calibration scales
) -> torch.Tensor:
    """
    P2-fixed: Tier-0 FP16 bypass + calibration-scale per-channel INT4 on the rest.
    No Tier-1 INT8 dynamic shell (skipped entirely).

    Tier-0 FP16 bypass: h[tier0] → FP16 → FP32 (outlier protection, identical to RAOK).
    Non-Tier-0: per-channel INT4 with calibration-derived scale (fixed, not per-token).
    """
    d = h.shape[0]
    h_out = h.clone()

    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    # Tier-0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Non-Tier-0: per-channel INT4 with calibration scales.
    non_t0_vals = h[~t0_mask]              # (d - tier0_k,)
    non_t0_scales = calib_scales[~t0_mask] # same mask
    q = torch.round(non_t0_vals / non_t0_scales).clamp(-7, 7)
    h_out[~t0_mask] = q * non_t0_scales

    return h_out


def quantize_per_channel_with_tier1_fixed(
    h: torch.Tensor,              # (d,) float32
    tier0_idx: torch.Tensor,      # (2,) LongTensor — static Tier-0 (FP16 bypass)
    tier1_calib_idx: torch.Tensor,# (18,) LongTensor — static Tier-1 candidates (from calib)
    calib_scales: torch.Tensor,   # (d,) float32 — per-channel calibration scales
    tier1_k_dynamic: int = 18,
) -> torch.Tensor:
    """
    P4-new: Calibration-scale per-channel INT4 (Tier-2) + dynamic Tier-1 INT8 shell + Tier-0 FP16.

    Tier-0 (2 channels): FP16 bypass — identical to RAOK.
    Tier-1 (18 channels, dynamic): top-18 per-token |h| after excluding Tier-0 → INT8.
       INT8 scale per-token: max_abs(tier1_vals) / 127
    Tier-2 (remaining ~2028 channels): per-channel INT4 with CALIBRATION-DERIVED scales.
       (NOT per-token scale — this is the key difference from RAOK's Tier-2.)

    This tests: does swapping RAOK's per-token Tier-2 scales with calibration-derived scales
    while keeping the dynamic Tier-1 shell preserve the ΔNLL ≈ 0.083?

    If P4 ≈ 0.083: calibration-scale Tier-2 + dynamic Tier-1 = RAOK.
    If P4 >> 0.083: RAOK's per-token Tier-2 scale IS doing additional work.
    """
    d = h.shape[0]
    h_out = h.clone()

    # Tier-0 mask.
    t0_mask = torch.zeros(d, dtype=torch.bool)
    t0_mask[tier0_idx] = True

    # Tier-0: FP16 bypass.
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Dynamic Tier-1: top-18 channels by |h| among non-Tier-0.
    h_non_t0_abs = h.abs().clone()
    h_non_t0_abs[t0_mask] = -1.0  # exclude Tier-0 from selection
    top_t1_idx = h_non_t0_abs.topk(tier1_k_dynamic).indices  # (18,)

    t1_mask = torch.zeros(d, dtype=torch.bool)
    t1_mask[top_t1_idx] = True

    # Tier-1: dynamic per-token INT8.
    t1_vals = h[t1_mask]
    t1_max = t1_vals.abs().max()
    if t1_max > 1e-9:
        t1_scale = t1_max / 127.0
        t1_q = torch.round(t1_vals / t1_scale).clamp(-127, 127)
        h_out[t1_mask] = t1_q * t1_scale
    # else: near-zero → keep FP16 pass-through (h_out already has h values)

    # Tier-2: everything not in Tier-0 or Tier-1 — calibration-derived per-channel INT4.
    t2_mask = ~t0_mask & ~t1_mask
    t2_vals = h[t2_mask]
    t2_scales = calib_scales[t2_mask]
    t2_q = torch.round(t2_vals / t2_scales).clamp(-7, 7)
    h_out[t2_mask] = t2_q * t2_scales

    return h_out


# ---------------------------------------------------------------------------
# Generic NLL evaluation loop — hook-per-layer architecture.
# ---------------------------------------------------------------------------

def eval_nll_with_hook(
    model,
    eval_tokens: torch.Tensor,
    n_layers: int,
    device: torch.device,
    hook_fn_per_layer: dict[int, Any],
    apply_quantization: bool = True,
    chunk_size: int = 64,
) -> float:
    """
    Compute mean NLL (nats) over eval_tokens.
    hook_fn_per_layer[L](h_tok) quantizes a single (d,) float32 token activation.
    Identical eval protocol to rraok_rung2.py.
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
# ---------------------------------------------------------------------------

def run_ablation(
    name: str,
    model,
    tokenizer,
    calib_scales_per_layer: dict[int, torch.Tensor],
    tier0_per_layer: dict[int, torch.Tensor],
    tier1_per_layer: dict[int, torch.Tensor],
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

        # Build per-layer hook functions.
        if name == "per_channel_all_fixed":
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_per_channel_all_fixed(h, calib_scales_per_layer[_L]))
                for L in range(n_layers)
            }
        elif name == "minimal_raok_fixed":
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_minimal_raok_fixed(
                        h, tier0_per_layer[_L], calib_scales_per_layer[_L]))
                for L in range(n_layers)
            }
        elif name == "per_channel_with_tier1_fixed":
            hook_fn_per_layer = {
                L: (lambda h, _L=L: quantize_per_channel_with_tier1_fixed(
                        h, tier0_per_layer[_L], tier1_per_layer[_L], calib_scales_per_layer[_L]))
                for L in range(n_layers)
            }
        else:
            raise ValueError(f"Unknown ablation name: {name}")

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
# Verdict logic.
# ---------------------------------------------------------------------------

RAOK_MEAN_REF = 0.08295687024140778
RAOK_STD_REF  = 0.015906671145993815
COLLAPSE_THRESHOLD = 0.05   # |gap| <= this → effectively matches RAOK
SURVIVES_THRESHOLD = 1.0    # gap > this → clearly worse than RAOK


def verdict_p1_fixed(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if abs(gap) <= COLLAPSE_THRESHOLD:
        v = "COLLAPSES-TO-PER-CHANNEL-INT4"
        note = (
            f"P1-fixed calibration-scale per-channel-all ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"|gap| <= {COLLAPSE_THRESHOLD} nats. RAOK's FP16 Tier-0 and INT8 Tier-1 shell are NOT "
            f"load-bearing beyond plain calibration-derived per-channel INT4. "
            f"RAOK reduces to SmoothQuant-class per-channel INT4 activation quantization — prior art."
        )
    elif gap > SURVIVES_THRESHOLD:
        v = "TIERS-LOAD-BEARING"
        note = (
            f"P1-fixed calibration-scale per-channel-all ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats worse than RAOK ({raok_mean:.4f}). "
            f"Gap > {SURVIVES_THRESHOLD} nats. The FP16 Tier-0 and/or INT8 Tier-1 shell ARE doing "
            f"real work that flat calibration-scale per-channel INT4 cannot replicate."
        )
    elif gap < -COLLAPSE_THRESHOLD:
        v = "BETTER-THAN-RAOK"
        note = (
            f"P1-fixed calibration-scale per-channel-all ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats BETTER than RAOK ({raok_mean:.4f}). "
            f"Calibration-scale INT4 outperforms RAOK — tiers add noise rather than value. "
            f"RAOK engineering claim is not just reduced to prior art but is inferior to it."
        )
    else:
        v = "MARGINAL"
        note = (
            f"P1-fixed calibration-scale per-channel-all ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap {abs(gap):.4f} nats is in marginal zone ({COLLAPSE_THRESHOLD}–{SURVIVES_THRESHOLD} nats). "
            f"Tiers provide modest benefit over calibration-scale per-channel INT4 but not decisive."
        )
    return v, note


def verdict_p2_fixed(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if abs(gap) <= COLLAPSE_THRESHOLD:
        v = "TIER1-NOT-LOAD-BEARING"
        note = (
            f"P2-fixed minimal-RAOK ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"|gap| <= {COLLAPSE_THRESHOLD} nats. Tier-1 INT8 dynamic shell is NOT load-bearing. "
            f"RAOK = Tier-0 FP16 bypass + calibration-scale per-channel INT4 for the rest. "
            f"Combined with P1 verdict: if P1 also collapses, RAOK is fully prior art."
        )
    elif gap > SURVIVES_THRESHOLD:
        v = "TIER1-LOAD-BEARING"
        note = (
            f"P2-fixed minimal-RAOK ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats worse than RAOK ({raok_mean:.4f}). "
            f"Gap > {SURVIVES_THRESHOLD} nats. Tier-1 INT8 dynamic shell IS doing real work — "
            f"removing it causes significant degradation even with correct Tier-0 + calibration-scale Tier-2."
        )
    elif gap < -COLLAPSE_THRESHOLD:
        v = "BETTER-THAN-RAOK-WITHOUT-TIER1"
        note = (
            f"P2-fixed minimal-RAOK ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats BETTER than RAOK ({raok_mean:.4f}). "
            f"Tier-1 INT8 shell is actively harmful — removing it improves quality."
        )
    else:
        v = "MARGINAL"
        note = (
            f"P2-fixed minimal-RAOK ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap {abs(gap):.4f} nats is marginal ({COLLAPSE_THRESHOLD}–{SURVIVES_THRESHOLD} nats). "
            f"Tier-1 INT8 shell provides modest benefit but not decisive."
        )
    return v, note


def verdict_p4_new(mean_delta: float, raok_mean: float) -> tuple[str, str]:
    gap = mean_delta - raok_mean
    if abs(gap) <= COLLAPSE_THRESHOLD:
        v = "CALIB-TIER2-PLUS-DYN-TIER1-MATCHES-RAOK"
        note = (
            f"P4-new calib-scale-per-channel + dynamic-Tier-1 ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"|gap| <= {COLLAPSE_THRESHOLD} nats. Calibration-scale Tier-2 + dynamic Tier-1 INT8 = RAOK. "
            f"RAOK's per-token Tier-2 scale is NOT providing benefit over calibration-derived scale. "
            f"RAOK = calibration-derived per-channel INT4 + dynamic INT8 outlier shell (Tier-1). "
            f"This is a minor variant of LLM.int8 / SmoothQuant with a dynamic INT8 shell."
        )
    elif gap > SURVIVES_THRESHOLD:
        v = "RAOK-PERTOK-TIER2-LOAD-BEARING"
        note = (
            f"P4-new calib-scale-per-channel + dynamic-Tier-1 ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats worse than RAOK ({raok_mean:.4f}). "
            f"Gap > {SURVIVES_THRESHOLD} nats. RAOK's per-token Tier-2 scale IS doing extra work "
            f"beyond calibration-derived scale. The per-token adaptive quantization in Tier-2 is "
            f"a genuine contribution that static calibration cannot replicate."
        )
    elif gap < -COLLAPSE_THRESHOLD:
        v = "CALIB-TIER2-BETTER-THAN-RAOK"
        note = (
            f"P4-new calib-scale-per-channel + dynamic-Tier-1 ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats BETTER than RAOK ({raok_mean:.4f}). "
            f"Calibration-scale Tier-2 outperforms RAOK's per-token Tier-2. "
            f"Per-token scale adaptation is suboptimal here — consistent calibration scale is better."
        )
    else:
        v = "MARGINAL"
        note = (
            f"P4-new calib-scale-per-channel + dynamic-Tier-1 ΔNLL={mean_delta:.4f} nats, "
            f"gap={gap:+.4f} nats from RAOK ({raok_mean:.4f}). "
            f"Gap {abs(gap):.4f} nats is in marginal zone. Per-token vs calibration Tier-2 scales "
            f"provide comparable performance."
        )
    return v, note


def aggregate_verdict_v2(
    results: dict[str, dict],
    raok_mean: float,
) -> tuple[str, str]:
    v1 = results.get("per_channel_all_fixed", {}).get("verdict", "")
    v2 = results.get("minimal_raok_fixed", {}).get("verdict", "")
    v4 = results.get("per_channel_with_tier1_fixed", {}).get("verdict", "")
    p1_mean = results.get("per_channel_all_fixed", {}).get("mean_delta_nll", float("nan"))
    p2_mean = results.get("minimal_raok_fixed", {}).get("mean_delta_nll", float("nan"))
    p4_mean = results.get("per_channel_with_tier1_fixed", {}).get("mean_delta_nll", float("nan"))

    p1_collapses = "COLLAPSES-TO-PER-CHANNEL-INT4" in v1 or "BETTER-THAN-RAOK" in v1
    p2_tier1_not_needed = "TIER1-NOT-LOAD-BEARING" in v2 or "BETTER-THAN-RAOK-WITHOUT-TIER1" in v2
    p4_matches = "CALIB-TIER2-PLUS-DYN-TIER1-MATCHES-RAOK" in v4 or "CALIB-TIER2-BETTER-THAN-RAOK" in v4

    if p1_collapses and p2_tier1_not_needed:
        return "FULLY-COLLAPSED-TO-PRIOR-ART", (
            f"RAOK fully reduces to calibration-derived per-channel INT4 activation quantization. "
            f"P1-fixed ({p1_mean:.4f} nats) ≈ RAOK ({raok_mean:.4f} nats): the FP16 Tier-0 and "
            f"INT8 Tier-1 tiers are NOT load-bearing. P2-fixed ({p2_mean:.4f} nats) ≈ RAOK: Tier-1 not needed. "
            f"RAOK is SmoothQuant-class per-channel INT4 activation quantization — covered by "
            f"prior art (SmoothQuant Xiao et al. 2022, LLM.int8 Dettmers et al. 2022). "
            f"No engineering novelty beyond competent application of standard methods."
        )
    elif p1_collapses and not p2_tier1_not_needed and p4_matches:
        return "PER-CHANNEL-INT4-PLUS-DYN-INT8-SHELL", (
            f"RAOK = calibration-derived per-channel INT4 + dynamic INT8 outlier shell. "
            f"P1-fixed ({p1_mean:.4f} nats) ≈ RAOK — FP16 Tier-0 not load-bearing. "
            f"P2-fixed ({p2_mean:.4f} nats) worse — Tier-1 INT8 dynamic shell IS needed. "
            f"P4-new ({p4_mean:.4f} nats) ≈ RAOK — calibration-scale Tier-2 + dynamic Tier-1 works. "
            f"RAOK = SmoothQuant-class per-channel INT4 with LLM.int8-style dynamic INT8 outlier shell. "
            f"Minor engineering variant of published methods; dynamic top-k INT8 shell is the surviving piece."
        )
    elif not p1_collapses and not p2_tier1_not_needed and not p4_matches:
        return "BOTH-TIERS-LOAD-BEARING-RAOK-HAS-GENUINE-STRUCTURE", (
            f"Both FP16 Tier-0 and INT8 Tier-1 shell are load-bearing, AND per-token Tier-2 scale is load-bearing. "
            f"P1-fixed ({p1_mean:.4f} nats) >> RAOK ({raok_mean:.4f} nats): flat calib INT4 insufficient. "
            f"P2-fixed ({p2_mean:.4f} nats) >> RAOK: Tier-1 shell needed. "
            f"P4-new ({p4_mean:.4f} nats) >> RAOK: per-token Tier-2 scale also needed. "
            f"RAOK's three-tier design with per-token Tier-2 adaptation is not reducible to published methods. "
            f"Real engineering novelty — all three tiers contribute."
        )
    elif not p1_collapses and p2_tier1_not_needed:
        return "FP16-TIER0-IS-MOAT-TIER1-REDUNDANT", (
            f"FP16 Tier-0 IS load-bearing (P1-fixed {p1_mean:.4f} >> RAOK {raok_mean:.4f}), "
            f"but Tier-1 INT8 shell is NOT (P2-fixed {p2_mean:.4f} ≈ RAOK). "
            f"RAOK mechanism: 2-channel FP16 isolation is critical; dynamic INT8 shell adds nothing. "
            f"Closest prior art: LLM.int8 outlier-channel isolation applied to activations. "
            f"The CF3-derived tier-0 selection is the main contribution."
        )
    elif not p1_collapses and not p2_tier1_not_needed and p4_matches:
        return "CALIB-TIER2-PLUS-DYN-TIER1-SUFFICIENT-FP16-LOAD-BEARING", (
            f"FP16 Tier-0 IS load-bearing (P1-fixed {p1_mean:.4f} >> RAOK {raok_mean:.4f}). "
            f"Tier-1 INT8 dynamic shell IS load-bearing (P2-fixed {p2_mean:.4f} >> RAOK). "
            f"But calibration-scale Tier-2 + dynamic Tier-1 = RAOK (P4-new {p4_mean:.4f} ≈ RAOK). "
            f"Per-token Tier-2 scale IS doing extra work beyond calibration. "
            f"RAOK = FP16 outlier isolation + dynamic INT8 shell + per-token adaptive INT4 Tier-2. "
            f"The dynamic Tier-1 and per-token Tier-2 together constitute the novel contribution."
        )
    else:
        present = [k for k in ["per_channel_all_fixed", "minimal_raok_fixed", "per_channel_with_tier1_fixed"]
                   if results.get(k)]
        return "INCONCLUSIVE", (
            f"Mixed or marginal results. Ablations present: {present}. "
            f"P1={p1_mean:.4f}, P2={p2_mean:.4f}, P4={p4_mean:.4f}, RAOK={raok_mean:.4f}. "
            f"Cannot draw decisive conclusion."
        )


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def write_json_v2(all_results: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "sharp_ablations_v2_results.json"
    existing: dict[str, Any] = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(all_results)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)


def write_result_md_v2(all_results: dict[str, Any], result_md_dir: Path) -> None:
    p1 = all_results.get("per_channel_all_fixed", {})
    p2 = all_results.get("minimal_raok_fixed", {})
    p4 = all_results.get("per_channel_with_tier1_fixed", {})
    present = [k for k in ["per_channel_all_fixed", "minimal_raok_fixed", "per_channel_with_tier1_fixed"]
               if all_results.get(k)]

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

    md = f"""# R-RAOK-70B Sharp Ablations v2 Result (Calibration-Fixed)

Date: {datetime.now(UTC).strftime("%Y-%m-%d")}
Model: Qwen/Qwen3-1.7B-Base (bf16, CPU)
Script: `scripts/rraok_sharp_ablations_v2.py`
Ablations run: {present}
Output JSON: `experiments/stage0/ladder_v2/round1_raok70b/sharp_ablations_v2_results.json`

## Bug fixed from v1

The original `rraok_sharp_ablations.py` computed per-channel INT4 scale as:
    `scale_c = |h[c]| / 7`  (per-token, per-channel)

This is degenerate: it fits the scale to the individual value being quantized,
giving bit-exact reconstruction (ΔNLL = 0.0000 for all seeds). Not a test of INT4 at all.

**This script uses calibration-derived scales:**
    `max_abs_L_c = max over all 200 calibration tokens of |h_L[c]|`
    `scale_L_c   = max_abs_L_c / 7`  (fixed — does not shrink to fit eval tokens)

This is the standard SmoothQuant / AWQ-class per-channel INT4 activation quantization.

---

## RAOK Reference

Mean ΔNLL (Rung 2, three-tier, per-token Tier-2 scale): **{raok_mean:.4f} nats** (std={RAOK_STD_REF:.4f})
Seeds: WikiText-2 offsets 0 / 512 / 1024, 512 tokens each.

---

## Aggregate Verdict: {agg}

{agg_note}

---

## P1-fixed — Calibration-Scale Per-Channel INT4, ALL 2048 Channels

**SmoothQuant-class published-art baseline.**
No Tier-0 FP16, no Tier-1 INT8 dynamic shell.
Per-channel INT4 scale derived from calibration corpus max-abs (not per-token).

**Verdict: {p1.get("verdict", "PENDING")}**

{p1.get("verdict_note", "")}

Mean ΔNLL: **{p1.get("mean_delta_nll", float("nan")):.4f} ± {p1.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p1)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p1)}
---

## P2-fixed — Tier-0 FP16 + Calibration-Scale Per-Channel INT4 (No Tier-1 Shell)

**Tests whether Tier-1 INT8 dynamic shell is load-bearing.**
Tier-0 (top-2 by mean magnitude, FP16 bypass) retained.
Remaining 2046 channels: calibration-derived per-channel INT4. No Tier-1 shell.

**Verdict: {p2.get("verdict", "PENDING")}**

{p2.get("verdict_note", "")}

Mean ΔNLL: **{p2.get("mean_delta_nll", float("nan")):.4f} ± {p2.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p2)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p2)}
---

## P4-new — Calibration-Scale Tier-2 + Dynamic Tier-1 INT8 Shell + Tier-0 FP16

**New ablation — isolates whether RAOK's per-token Tier-2 scale is load-bearing.**
Same Tier-0 FP16 bypass and dynamic Tier-1 INT8 shell as RAOK.
Key difference: Tier-2 uses calibration-derived per-channel scales (not per-token scale).

If P4 ≈ RAOK (0.083): per-token Tier-2 scale is NOT special; RAOK = calibration-INT4 + dynamic INT8 shell.
If P4 >> RAOK: per-token Tier-2 scale IS doing extra work; RAOK's adaptive quantization matters.

**Verdict: {p4.get("verdict", "PENDING")}**

{p4.get("verdict_note", "")}

Mean ΔNLL: **{p4.get("mean_delta_nll", float("nan")):.4f} ± {p4.get("std_delta_nll", float("nan")):.4f} nats**
Gap vs RAOK: **{gap_str(p4)} nats**

| Seed | WikiText-2 offset | Baseline NLL | Quantized NLL | ΔNLL |
|---|---|---|---|---|
{fmt_seed_rows(p4)}
---

## Summary Table

| Scheme | Mean ΔNLL | Std | Gap vs RAOK | Verdict |
|---|---|---|---|---|
| RAOK three-tier (Rung 2, per-token Tier-2) | {raok_mean:.4f} | {RAOK_STD_REF:.4f} | 0.0000 | reference |
| P1-fixed: calib per-channel INT4, all | {p1.get("mean_delta_nll", float("nan")):.4f} | {p1.get("std_delta_nll", float("nan")):.4f} | {gap_str(p1)} | {p1.get("verdict", "PENDING")} |
| P2-fixed: T0-FP16 + calib per-ch INT4 | {p2.get("mean_delta_nll", float("nan")):.4f} | {p2.get("std_delta_nll", float("nan")):.4f} | {gap_str(p2)} | {p2.get("verdict", "PENDING")} |
| P4-new: T0-FP16 + dyn-T1-INT8 + calib-T2-INT4 | {p4.get("mean_delta_nll", float("nan")):.4f} | {p4.get("std_delta_nll", float("nan")):.4f} | {gap_str(p4)} | {p4.get("verdict", "PENDING")} |

---

## Decision Tree (Post-v2 Fix)

1. **P1-fixed ≈ RAOK** → flat calibration-scale per-channel INT4 matches RAOK.
   FP16 Tier-0 and INT8 Tier-1 are NOT load-bearing.
   **RAOK collapses to SmoothQuant-class prior art.**

2. **P1-fixed >> RAOK AND P2-fixed ≈ RAOK** → Tier-0 FP16 isolation is load-bearing; Tier-1 is not.
   RAOK = LLM.int8-style 2-channel FP16 bypass + per-channel INT4 for rest.

3. **P2-fixed >> RAOK AND P4-new ≈ RAOK** → Tier-1 dynamic INT8 shell IS load-bearing,
   but calibration-scale Tier-2 works as well as per-token Tier-2.
   **RAOK = calibration-INT4 + dynamic INT8 outlier shell — minor variant of published methods.**

4. **P4-new >> RAOK** → per-token Tier-2 scale IS providing benefit over calibration scale.
   **RAOK's adaptive quantization is genuinely novel vs published static-scale methods.**

---

## Where RAOK Lands in the Published Landscape (v2)

{"(Complete: see Aggregate Verdict above.)" if agg not in ["PENDING", "INCONCLUSIVE"] else "(Pending: need ablations complete.)"}
"""

    md_path = result_md_dir / "rraok_sharp_ablations_v2_result.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="R-RAOK-70B sharp ablations v2 — calibration-derived per-channel INT4 (bug-fixed)"
    )
    parser.add_argument("--per-channel-all-fixed", action="store_true",
                        help="P1-fixed: calibration-scale per-channel INT4 across all 2048 channels")
    parser.add_argument("--minimal-raok-fixed", action="store_true",
                        help="P2-fixed: Tier-0 FP16 + calibration-scale per-channel INT4 (no Tier-1)")
    parser.add_argument("--per-channel-with-tier1-fixed", action="store_true",
                        help="P4-new: calibration-scale Tier-2 + dynamic Tier-1 INT8 + Tier-0 FP16")
    parser.add_argument("--all", action="store_true", help="Run all three ablations sequentially")
    parser.add_argument("--raok-mean-nll", type=float, default=None,
                        help="RAOK Rung 2 mean ΔNLL (auto-read from rung2_results.json if not given)")
    args = parser.parse_args()

    if args.all:
        args.per_channel_all_fixed = True
        args.minimal_raok_fixed = True
        args.per_channel_with_tier1_fixed = True

    if not any([args.per_channel_all_fixed, args.minimal_raok_fixed, args.per_channel_with_tier1_fixed]):
        parser.error(
            "Specify at least one of --per-channel-all-fixed, --minimal-raok-fixed, "
            "--per-channel-with-tier1-fixed, or --all"
        )

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

    # B4-v2 unit test.
    print(f"[{time.time()-t0:.1f}s] Running B4-v2 unit test ...", flush=True)
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
            print(f"[{time.time()-t0:.1f}s] WARNING: rung2_results.json not found; using hardcoded {RAOK_MEAN_REF:.5f} nats", flush=True)
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

    # Calibration pass — always needed for the fixed scale computation.
    print(f"\n[{time.time()-t0:.1f}s] Calibration pass (computing per-channel max-abs scales + tier indices) ...", flush=True)
    calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
    print(f"  Calibration corpus: {len(calib_prompts)} prompts", flush=True)

    calib_scales_per_layer, tier0_per_layer, tier1_per_layer = compute_calibration(
        model, tokenizer, calib_prompts, n_layers, d, DEVICE,
        max_length=MAX_LENGTH, tier0_k=TIER0_K, tier1_k=TIER1_K,
    )
    print(f"[{time.time()-t0:.1f}s] Calibration done.", flush=True)

    # Log per-layer calibration scale stats and tier assignments.
    for L in range(n_layers):
        sc = calib_scales_per_layer[L]
        print(
            f"  L{L:2d} calib scale min={sc.min().item():.4f} max={sc.max().item():.4f} "
            f"mean={sc.mean().item():.4f} | "
            f"T0={tier0_per_layer[L].tolist()} | "
            f"T1 (first 4)={tier1_per_layer[L][:4].tolist()}...",
            flush=True,
        )

    # Load any existing partial results.
    json_path = OUT_DIR / "sharp_ablations_v2_results.json"
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
    all_results["script"] = "rraok_sharp_ablations_v2.py"
    all_results["bug_fix_note"] = (
        "v1 used scale_c = |h[c]|/7 per token — degenerate, gives ΔNLL=0. "
        "v2 uses calibration-derived max_abs_L_c / 7 — proper SmoothQuant-class per-channel INT4."
    )

    # Run requested ablations.
    ablations_to_run: list[str] = []
    if args.per_channel_all_fixed:
        ablations_to_run.append("per_channel_all_fixed")
    if args.minimal_raok_fixed:
        ablations_to_run.append("minimal_raok_fixed")
    if args.per_channel_with_tier1_fixed:
        ablations_to_run.append("per_channel_with_tier1_fixed")

    for ablation_name in ablations_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"[{time.time()-t0:.1f}s] Running ablation: {ablation_name}", flush=True)
        print(f"{'='*60}", flush=True)

        mean_delta, std_delta, seed_deltas, seed_records = run_ablation(
            name=ablation_name,
            model=model,
            tokenizer=tokenizer,
            calib_scales_per_layer=calib_scales_per_layer,
            tier0_per_layer=tier0_per_layer,
            tier1_per_layer=tier1_per_layer,
            n_layers=n_layers,
            d=d,
            device=DEVICE,
            eval_offsets=EVAL_OFFSETS,
            eval_n_tokens=EVAL_N_TOKENS,
            chunk_size=CHUNK_SIZE,
            t0_global=t0,
        )

        # Verdict per ablation.
        if ablation_name == "per_channel_all_fixed":
            verdict, verdict_note = verdict_p1_fixed(mean_delta, raok_mean_delta)
        elif ablation_name == "minimal_raok_fixed":
            verdict, verdict_note = verdict_p2_fixed(mean_delta, raok_mean_delta)
        else:  # per_channel_with_tier1_fixed
            verdict, verdict_note = verdict_p4_new(mean_delta, raok_mean_delta)

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
        write_json_v2(all_results, OUT_DIR)
        print(f"Checkpointed.", flush=True)

    # Aggregate verdict — compute whenever at least P1 and P2 are present.
    p1 = all_results.get("per_channel_all_fixed", {})
    p2 = all_results.get("minimal_raok_fixed", {})
    p4 = all_results.get("per_channel_with_tier1_fixed", {})

    if p1 and p2:
        agg, agg_note = aggregate_verdict_v2(all_results, raok_mean_delta)
        print(f"\n{'='*60}", flush=True)
        print(f"AGGREGATE VERDICT: {agg}", flush=True)
        print(f"  {agg_note}", flush=True)
        print(f"{'='*60}", flush=True)
        all_results["aggregate_verdict"] = agg
        all_results["aggregate_verdict_note"] = agg_note
    else:
        present = [k for k in ["per_channel_all_fixed", "minimal_raok_fixed", "per_channel_with_tier1_fixed"]
                   if all_results.get(k)]
        print(f"\nPartial run: {present} complete. Run P1-fixed + P2-fixed for aggregate verdict.", flush=True)

    all_results["elapsed_seconds"] = time.time() - t0

    write_json_v2(all_results, OUT_DIR)
    write_result_md_v2(all_results, RESULT_MD_DIR)

    total_time = time.time() - t0
    print(f"\nDone. Total elapsed: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
