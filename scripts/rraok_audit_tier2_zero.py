r"""R-RAOK-70B Audit: Check 3 (Tier-2 Zero), Check 5 (2K-token eval), Check 6 (code corpus).
Check 7 (noise floor) is computed as a free statistic during Check 3's baseline pass.

Check 3 — Tier-2-zero degeneracy test:
  Replace Tier-2 channel values with LITERAL ZEROS instead of INT4-quantizing them.
  Tier-0 and Tier-1 are unchanged (FP16 bypass and INT8 respectively).
  If Tier-2-zero DNLL is within 0.10 nats of RAOK's 0.083, the 2028 bulk channels
  carry no useful signal — either the model only uses ~20 channels or quantization
  is not actually reaching the matmul.
  Run on same 3 WikiText-2 seeds as Rung 2.

Check 5 — Eval-length scaling (2048 tokens, seed 1 only):
  Re-run RAOK three-tier quantization with 2048-token eval window.
  PASS if DNLL < 0.20 nats.

Check 6 — Corpus specificity (code corpus):
  Eval on Python/code text using the same tier boundaries.
  Use the TECHNICAL_PROMPTS from the calibration corpus joined as a long string,
  or codeparrot/github-code-clean if available.
  Same 3 seeds (offsets 0/512/1024 within the code token stream).

Check 7 — Noise floor (free, computed during Check 3 seed-1 baseline):
  Compute per-token NLL stddev on the baseline (seed 1, 512 tokens, full-precision).
  Noise floor = stddev / sqrt(n_tokens).
  PASS if 0.083 >> noise_floor (i.e., 0.083 > 3 * noise_floor).
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
# Rung 2 REFERENCE VALUES (from completed run)
# ---------------------------------------------------------------------------
RUNG2_MEAN_DELTA_NLL = 0.08295687024140778
RUNG2_SEED1_DELTA_NLL = 0.06897725144477729  # seed idx 0 / offset 0
RUNG2_THRESHOLD = 0.30  # GO threshold

# ---------------------------------------------------------------------------
# Calibration corpus (identical to rraok_rung2.py — copy to stay self-contained)
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
    "Reed-Solomon (255, 223): 32 parity bytes correct up to 16 byte errors per 256-byte block in GF(256).",
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
# Tier-0 calibration (identical logic to rraok_rung2.py)
# ---------------------------------------------------------------------------

def compute_tier0_channels(
    model, tokenizer, calib_prompts, n_layers, d, device,
    max_length=32, tier0_k=2, tier1_k=18,
):
    sum_mag = {L: torch.zeros(d, dtype=torch.float64) for L in range(n_layers)}
    count = {L: 0 for L in range(n_layers)}
    captured: dict[int, torch.Tensor] = {}

    def make_pre_hook(idx):
        def hook(_mod, inputs):
            captured[idx] = inputs[0].detach().float()
        return hook

    handles = []
    for L in range(n_layers):
        handles.append(model.model.layers[L].mlp.register_forward_pre_hook(make_pre_hook(L)))

    with torch.inference_mode():
        for prompt_text, _ in calib_prompts:
            captured.clear()
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            if ids.shape[-1] < 2:
                continue
            try:
                model(input_ids=ids, use_cache=False)
            except Exception as e:
                print(f"  calib skip ({e})", flush=True)
                continue
            for L in range(n_layers):
                if L not in captured:
                    continue
                h = captured[L].squeeze(0)
                sum_mag[L] += h.abs().sum(dim=0).double()
                count[L] += h.shape[0]

    for h in handles:
        h.remove()

    tier0_per_layer = {}
    for L in range(n_layers):
        mean_mag = (sum_mag[L] / max(1, count[L])).float()
        top2 = mean_mag.topk(tier0_k).indices.sort().values
        tier0_per_layer[L] = top2

    return tier0_per_layer


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_three_tier_t2zero(h: torch.Tensor, tier0_idx: torch.Tensor, tier1_k: int = 18) -> torch.Tensor:
    """Three-tier quantization with Tier-2 channels set to ZERO (Check 3)."""
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

    # Tier 0: FP16 bypass
    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    # Tier 1: INT8 dynamic
    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1

    # Tier 2: ZERO (not INT4) — this is the Check 3 ablation
    h_out[t2_mask] = 0.0

    return h_out


def quantize_three_tier_raok(h: torch.Tensor, tier0_idx: torch.Tensor, tier1_k: int = 18) -> torch.Tensor:
    """Standard RAOK three-tier quantization (replica for Check 5 and 6)."""
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

    h_out[t0_mask] = h[t0_mask].to(torch.float16).to(torch.float32)

    h_t1 = h[t1_mask]
    if h_t1.numel() > 0:
        max_abs_t1 = h_t1.abs().max()
        if max_abs_t1 > 1e-9:
            scale_t1 = max_abs_t1 / 127.0
            q_t1 = torch.round(h_t1 / scale_t1).clamp(-127, 127)
            h_out[t1_mask] = q_t1 * scale_t1

    h_t2 = h[t2_mask]
    if h_t2.numel() > 0:
        max_abs_t2 = h_t2.abs().max()
        if max_abs_t2 > 1e-9:
            scale_t2 = max_abs_t2 / 7.0
            q_t2 = torch.round(h_t2 / scale_t2).clamp(-7, 7)
            h_out[t2_mask] = q_t2 * scale_t2

    return h_out


# ---------------------------------------------------------------------------
# Token loading helpers
# ---------------------------------------------------------------------------

def _load_wikitext2_tokens(tokenizer, offset: int = 0, n_tokens: int = 512) -> torch.Tensor:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])
    except Exception as e:
        print(f"  WARNING: datasets not available ({e}); using fallback", flush=True)
        text = (
            "The history of science is the study of the development of science, including both the "
            "natural sciences and social sciences. Science is a body of empirical, theoretical, and "
            "practical knowledge about the natural world, produced by scientists who emphasize the "
            "observation, explanation, and prediction of real-world phenomena. "
        ) * 50

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


def _load_code_tokens(tokenizer, offset: int = 0, n_tokens: int = 512) -> torch.Tensor:
    """Load a code token stream for corpus-specificity check (Check 6).

    Tries codeparrot/github-code-clean first; falls back to TECHNICAL_PROMPTS
    joined as a long string (these are already in the file and represent
    code-like text with diverse syntax).
    """
    code_text = None
    try:
        from datasets import load_dataset
        ds = load_dataset("codeparrot/github-code-clean", split="train", streaming=True)
        samples = []
        for ex in ds:
            samples.append(ex.get("code", ex.get("content", "")))
            if sum(len(s) for s in samples) > 50000:
                break
        code_text = "\n\n".join(samples)
        print(f"  Loaded code corpus from codeparrot/github-code-clean ({len(code_text)} chars)", flush=True)
    except Exception as e:
        print(f"  codeparrot not available ({e}); using TECHNICAL_PROMPTS fallback", flush=True)

    if code_text is None or len(code_text) < 1000:
        # Fallback: concatenate TECHNICAL_PROMPTS many times to get enough tokens.
        code_text = "\n".join(TECHNICAL_PROMPTS) * 20

    ids = tokenizer.encode(code_text, add_special_tokens=False)
    total = len(ids)
    if total < n_tokens:
        reps = (n_tokens // total) + 2
        ids = (ids * reps)[:n_tokens * 3]
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
# Core NLL eval with pluggable quantizer
# ---------------------------------------------------------------------------

def eval_nll(
    model,
    tier0_per_layer: dict[int, torch.Tensor],
    eval_tokens: torch.Tensor,
    n_layers: int,
    d: int,
    device: torch.device,
    chunk_size: int = 64,
    tier1_k: int = 18,
    apply_quantization: bool = True,
    quant_mode: str = "raok",       # "raok" | "tier2_zero"
    collect_per_token_nll: bool = False,
) -> tuple[float, float | None, list[float] | None]:
    """
    Returns (mean_nll, nll_stddev_or_None, per_token_nll_list_or_None).
    nll_stddev and per_token_nll_list are only populated if collect_per_token_nll=True.
    """
    N = eval_tokens.shape[0]

    def make_pre_hook_quant(layer_idx: int):
        tier0_idx = tier0_per_layer[layer_idx]

        def hook(module, inputs):
            h_batch = inputs[0]
            if not apply_quantization:
                return
            h_float = h_batch.detach().float()
            B, T, D = h_float.shape
            h_out = h_float.clone()
            for b in range(B):
                for t in range(T):
                    h_tok = h_float[b, t]
                    if quant_mode == "tier2_zero":
                        h_q = quantize_three_tier_t2zero(h_tok, tier0_idx, tier1_k=tier1_k)
                    else:
                        h_q = quantize_three_tier_raok(h_tok, tier0_idx, tier1_k=tier1_k)
                    h_out[b, t] = h_q
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
    per_token_nlls: list[float] = [] if collect_per_token_nll else None  # type: ignore[assignment]

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
                    if collect_per_token_nll and per_token_nlls is not None:
                        per_token_nlls.append(nll)

    for h in handles:
        h.remove()

    mean_nll = total_nll / max(1, total_tokens)
    nll_std = None
    if collect_per_token_nll and per_token_nlls:
        nll_std = float(np.std(per_token_nlls))

    return mean_nll, nll_std, per_token_nlls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    DEVICE = torch.device("cpu")
    MAX_LENGTH = 32
    CALIB_N_PER_STRATUM = 50
    CHUNK_SIZE = 64
    TIER0_K = 2
    TIER1_K = 18
    EVAL_OFFSETS = [0, 512, 1024]

    OUT_DIR = Path("experiments/stage0/ladder_v2/round1_raok70b")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.time()-t0:.1f}s] Loading {MODEL_ID} bf16 on CPU ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}", flush=True)

    # Calibration pass
    print(f"\n[{time.time()-t0:.1f}s] Calibration pass (200 prompts) ...", flush=True)
    calib_prompts = _build_calib_prompts(tokenizer, n_per_stratum=CALIB_N_PER_STRATUM, max_length=MAX_LENGTH)
    tier0_per_layer = compute_tier0_channels(
        model, tokenizer, calib_prompts, n_layers, d, DEVICE,
        max_length=MAX_LENGTH, tier0_k=TIER0_K, tier1_k=TIER1_K,
    )
    print(f"[{time.time()-t0:.1f}s] Calibration done.", flush=True)

    results: dict[str, Any] = {
        "experiment": "R-RAOK-70B Audit: Check 3 / Check 5 / Check 6 / Check 7",
        "model_id": MODEL_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "rung2_reference_mean_delta_nll": RUNG2_MEAN_DELTA_NLL,
        "rung2_reference_seed1_delta_nll": RUNG2_SEED1_DELTA_NLL,
    }

    # -----------------------------------------------------------------------
    # CHECK 3: Tier-2-zero degeneracy test (3 seeds)
    # Also collects per-token NLL for Check 7 on seed 1 baseline.
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] === CHECK 3: Tier-2-zero degeneracy test ===", flush=True)
    check3_seeds = []
    check7_nll_std = None
    check7_noise_floor = None
    check7_n_tokens = None

    for seed_idx, offset in enumerate(EVAL_OFFSETS):
        print(f"[{time.time()-t0:.1f}s] --- Seed {seed_idx+1}/3 (offset={offset}) ---", flush=True)
        eval_tokens = _load_wikitext2_tokens(tokenizer, offset=offset, n_tokens=512)

        # Baseline NLL — collect per-token NLL for seed 1 (offset=0) for Check 7.
        collect_pt = (seed_idx == 0)
        print(f"[{time.time()-t0:.1f}s]   Baseline forward pass ...", flush=True)
        baseline_nll, baseline_std, pt_nlls = eval_nll(
            model, tier0_per_layer, eval_tokens, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=False,
            quant_mode="raok",
            collect_per_token_nll=collect_pt,
        )
        print(f"  Baseline NLL: {baseline_nll:.4f} nats", flush=True)

        if collect_pt and baseline_std is not None:
            check7_nll_std = baseline_std
            check7_n_tokens = len(pt_nlls) if pt_nlls else 0
            check7_noise_floor = baseline_std / np.sqrt(max(1, check7_n_tokens))
            print(f"  [Check 7] Per-token NLL stddev: {baseline_std:.4f} nats", flush=True)
            print(f"  [Check 7] Noise floor (std/sqrt(n)): {check7_noise_floor:.4f} nats  (n={check7_n_tokens})", flush=True)
            print(f"  [Check 7] RAOK DNLL=0.083 vs noise_floor={check7_noise_floor:.4f}  ratio={0.083/check7_noise_floor:.2f}x", flush=True)

        # Tier-2-zero quantized NLL
        print(f"[{time.time()-t0:.1f}s]   Tier-2-zero forward pass ...", flush=True)
        t2z_nll, _, _ = eval_nll(
            model, tier0_per_layer, eval_tokens, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=True,
            quant_mode="tier2_zero",
        )
        t2z_delta = t2z_nll - baseline_nll
        print(f"  Tier-2-zero NLL: {t2z_nll:.4f} nats  |  DNLL = {t2z_delta:.4f} nats", flush=True)
        check3_seeds.append({
            "seed_idx": seed_idx,
            "offset": offset,
            "baseline_nll": baseline_nll,
            "t2zero_nll": t2z_nll,
            "t2zero_delta_nll": t2z_delta,
        })

    c3_deltas = [s["t2zero_delta_nll"] for s in check3_seeds]
    c3_mean = float(np.mean(c3_deltas))
    c3_worst = float(np.max(c3_deltas))
    # Decisive threshold: if within 0.10 nats of RAOK's 0.083 => DEGENERATE
    c3_diff_from_raok = abs(c3_mean - RUNG2_MEAN_DELTA_NLL)
    c3_degenerate = c3_diff_from_raok <= 0.10
    c3_verdict = "FAIL (degenerate — Tier-2 carries no signal or hook not applying)" if c3_degenerate else "PASS (Tier-2 zeroing catastrophic — bulk channels do carry signal)"

    print(f"\n[{time.time()-t0:.1f}s] Check 3 summary:", flush=True)
    print(f"  Tier-2-zero mean DNLL: {c3_mean:.4f} nats", flush=True)
    print(f"  RAOK mean DNLL:        {RUNG2_MEAN_DELTA_NLL:.4f} nats", flush=True)
    print(f"  Difference:            {c3_diff_from_raok:.4f} nats", flush=True)
    print(f"  Check 3: {c3_verdict}", flush=True)

    results["check3"] = {
        "description": "Tier-2 channels set to zero instead of INT4",
        "seeds": check3_seeds,
        "t2zero_mean_delta_nll": c3_mean,
        "t2zero_worst_delta_nll": c3_worst,
        "raok_mean_delta_nll": RUNG2_MEAN_DELTA_NLL,
        "diff_from_raok": c3_diff_from_raok,
        "degenerate_threshold": 0.10,
        "degenerate": c3_degenerate,
        "verdict": c3_verdict,
    }

    # -----------------------------------------------------------------------
    # CHECK 7: Noise floor (computed from seed-1 baseline above)
    # -----------------------------------------------------------------------
    if check7_nll_std is not None:
        c7_ratio = 0.083 / check7_noise_floor if check7_noise_floor and check7_noise_floor > 0 else float("inf")
        # PASS if DNLL > 3x noise floor; FAIL if <= 3x
        c7_pass = c7_ratio > 3.0
        c7_verdict = f"PASS (ratio={c7_ratio:.2f}x > 3.0)" if c7_pass else f"FAIL (ratio={c7_ratio:.2f}x <= 3.0 — result may not be statistically significant)"
        print(f"\n[{time.time()-t0:.1f}s] Check 7 summary:", flush=True)
        print(f"  Per-token NLL stddev: {check7_nll_std:.4f} nats  n={check7_n_tokens}", flush=True)
        print(f"  Noise floor (std/sqrt(n)): {check7_noise_floor:.4f} nats", flush=True)
        print(f"  RAOK DNLL / noise_floor = {c7_ratio:.2f}x", flush=True)
        print(f"  Check 7: {c7_verdict}", flush=True)
        results["check7"] = {
            "description": "Noise floor: per-token NLL stddev on baseline seed 1, 512 tokens",
            "per_token_nll_stddev": check7_nll_std,
            "n_tokens": check7_n_tokens,
            "noise_floor_sem": check7_noise_floor,
            "raok_mean_delta_nll": RUNG2_MEAN_DELTA_NLL,
            "ratio_dnll_to_noise_floor": c7_ratio,
            "pass_threshold": 3.0,
            "verdict": c7_verdict,
        }
    else:
        print(f"  WARNING: Check 7 per-token NLL collection failed", flush=True)
        results["check7"] = {"verdict": "UNKNOWN — collection failed"}

    # -----------------------------------------------------------------------
    # CHECK 5: Eval-length scaling (2048 tokens, seed 1 / offset=0 only)
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] === CHECK 5: 2048-token eval-length scaling (seed 1) ===", flush=True)
    eval_tokens_2k = _load_wikitext2_tokens(tokenizer, offset=0, n_tokens=2048)

    print(f"[{time.time()-t0:.1f}s]   Baseline 2K forward pass ...", flush=True)
    baseline_2k, _, _ = eval_nll(
        model, tier0_per_layer, eval_tokens_2k, n_layers, d, DEVICE,
        chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        apply_quantization=False,
    )
    print(f"  Baseline 2K NLL: {baseline_2k:.4f} nats", flush=True)

    print(f"[{time.time()-t0:.1f}s]   Three-tier quantized 2K forward pass ...", flush=True)
    quant_2k, _, _ = eval_nll(
        model, tier0_per_layer, eval_tokens_2k, n_layers, d, DEVICE,
        chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
        apply_quantization=True,
        quant_mode="raok",
    )
    c5_delta = quant_2k - baseline_2k
    c5_pass = c5_delta < 0.20
    c5_superlinear = c5_delta > 4 * RUNG2_SEED1_DELTA_NLL
    c5_verdict = (
        f"PASS (2K DNLL={c5_delta:.4f} < 0.20)" if c5_pass
        else f"FAIL (2K DNLL={c5_delta:.4f} >= 0.20 — error compounds over context)"
    )
    if c5_superlinear:
        c5_verdict += f" [SUPERLINEAR: {c5_delta:.4f} > 4x seed1 ({4*RUNG2_SEED1_DELTA_NLL:.4f})]"

    print(f"  2K Quantized NLL: {quant_2k:.4f} nats  |  DNLL = {c5_delta:.4f} nats", flush=True)
    print(f"  512-token seed1 DNLL: {RUNG2_SEED1_DELTA_NLL:.4f}  |  4x = {4*RUNG2_SEED1_DELTA_NLL:.4f}", flush=True)
    print(f"  Superlinear: {c5_superlinear}", flush=True)
    print(f"  Check 5: {c5_verdict}", flush=True)

    results["check5"] = {
        "description": "2048-token eval-length scaling (seed 1 / offset 0)",
        "baseline_2k_nll": baseline_2k,
        "quant_2k_nll": quant_2k,
        "delta_nll_2k": c5_delta,
        "pass_threshold": 0.20,
        "superlinear_threshold": 4 * RUNG2_SEED1_DELTA_NLL,
        "superlinear": c5_superlinear,
        "verdict": c5_verdict,
    }

    # -----------------------------------------------------------------------
    # CHECK 6: Corpus specificity (code, 3 seeds)
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] === CHECK 6: Corpus specificity (code corpus, 3 seeds) ===", flush=True)
    check6_seeds = []
    for seed_idx, offset in enumerate(EVAL_OFFSETS):
        print(f"[{time.time()-t0:.1f}s] --- Code Seed {seed_idx+1}/3 (offset={offset}) ---", flush=True)
        eval_tokens_code = _load_code_tokens(tokenizer, offset=offset, n_tokens=512)

        print(f"[{time.time()-t0:.1f}s]   Baseline code forward pass ...", flush=True)
        baseline_code, _, _ = eval_nll(
            model, tier0_per_layer, eval_tokens_code, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=False,
        )
        print(f"  Code Baseline NLL: {baseline_code:.4f} nats", flush=True)

        print(f"[{time.time()-t0:.1f}s]   Three-tier quantized code forward pass ...", flush=True)
        quant_code, _, _ = eval_nll(
            model, tier0_per_layer, eval_tokens_code, n_layers, d, DEVICE,
            chunk_size=CHUNK_SIZE, tier1_k=TIER1_K,
            apply_quantization=True,
            quant_mode="raok",
        )
        c6_delta = quant_code - baseline_code
        print(f"  Code Quantized NLL: {quant_code:.4f} nats  |  DNLL = {c6_delta:.4f} nats", flush=True)
        check6_seeds.append({
            "seed_idx": seed_idx,
            "offset": offset,
            "baseline_nll": baseline_code,
            "quant_nll": quant_code,
            "delta_nll": c6_delta,
        })

    c6_deltas = [s["delta_nll"] for s in check6_seeds]
    c6_mean = float(np.mean(c6_deltas))
    c6_diff_from_nl = abs(c6_mean - RUNG2_MEAN_DELTA_NLL)
    c6_within_threshold = c6_diff_from_nl <= 0.10
    c6_verdict = (
        f"PASS (code DNLL={c6_mean:.4f}, diff from NL={c6_diff_from_nl:.4f} <= 0.10 — generalizes)"
        if c6_within_threshold
        else f"INFORMATIVE (code DNLL={c6_mean:.4f}, diff from NL={c6_diff_from_nl:.4f} > 0.10 — corpus-specific)"
    )
    print(f"\n[{time.time()-t0:.1f}s] Check 6 summary:", flush=True)
    print(f"  Code mean DNLL: {c6_mean:.4f} nats", flush=True)
    print(f"  NL (WikiText-2) mean DNLL: {RUNG2_MEAN_DELTA_NLL:.4f} nats", flush=True)
    print(f"  Difference: {c6_diff_from_nl:.4f} nats", flush=True)
    print(f"  Check 6: {c6_verdict}", flush=True)

    results["check6"] = {
        "description": "Corpus specificity: code vs natural language (WikiText-2)",
        "seeds": check6_seeds,
        "code_mean_delta_nll": c6_mean,
        "nl_mean_delta_nll": RUNG2_MEAN_DELTA_NLL,
        "diff_code_vs_nl": c6_diff_from_nl,
        "generalization_threshold": 0.10,
        "within_threshold": c6_within_threshold,
        "verdict": c6_verdict,
    }

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    out_json = OUT_DIR / "audit_checks_3_5_6_7.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[{elapsed:.1f}s] Wrote {out_json}", flush=True)

    print(f"\n=== AUDIT SUMMARY ===", flush=True)
    for k in ["check3", "check5", "check6", "check7"]:
        v = results.get(k, {}).get("verdict", "NOT RUN")
        print(f"  {k}: {v}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
