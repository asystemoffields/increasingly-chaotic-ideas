r"""R-RAOK-70B Rung 1 — 28-layer Outlier-Channel Jaccard Extension.

Stage 6 amendments applied:
  B1: random-init control uses torch.nn.init.normal_(std=0.02).
  B2: H1 restated as a 28-layer EXTENSION of the 7-layer CF3 finding,
      not a consequence — Rung 1 is the first full 28-layer measurement.
  B3: MLP block input (pre-gate/up split) is the quantized tensor; both
      W_gate and W_up receive the same quantized activation. Stated explicitly.
  B4: INT4 scale-factor unit test already run and PASSED in rraok_int4_unit_test.py.

Hypothesis (B2-restated):
  H1 (28-layer extension): CF3 found K-dependent outlier dynamicity (Jaccard at
  K=1% ≈ 0.31 mean) across 7 sampled layers. This script extends that measurement
  to all 28 layers. The GO threshold (Jaccard@K=1% < 0.50 in >=80% of layers) is
  what Rung 1 tests — it is NOT assumed to follow from CF3 a priori.

Procedure:
  1. Load Qwen3-1.7B-Base bf16 on CPU.
  2. Forward hooks on MLP block input (shared tensor feeding W_gate AND W_up)
     at ALL 28 layers.
  3. Run 200 diverse prompts (4 strata x 50 each, same as PDAP).
  4. For each layer, for each consecutive normal-token pair:
     compute Jaccard of top-K channel sets at K in {0.1%, 1%, 5%}.
  5. Report per-layer Jaccard at K=1%, verdict, and B3 scope note.

Controls:
  - Permutation control: per-token channel-index permutation; expected Jaccard ≈ 0.
  - Random-init control: normal_(std=0.02) initialization; expected stability near 0.

GO threshold (Rung 1): Jaccard@K=1% < 0.50 in >=80% of 28 layers.
NO-GO: Jaccard@K=1% > 0.65 in >=30% of 28 layers.
GRAY: between.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

# ---------------------------------------------------------------------------
# Eval corpus (same strata as PDAP / pdap_outlier_jaccard.py)
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


def gen_random_token_prompts(tokenizer, n: int, length: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    out = []
    for _ in range(n):
        ids = rng.integers(0, vocab, size=length).tolist()
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


# ---------------------------------------------------------------------------
# Core Jaccard measurement
# ---------------------------------------------------------------------------

def compute_jaccard_per_layer(
    model,
    tokenizer,
    prompts: list[tuple[str, str]],
    all_layers: list[int],
    k_fracs: tuple[float, ...],
    d: int,
    device: torch.device,
    massive_multiplier: float = 5.0,
    max_length: int = 32,
    permute_channels: bool = False,
    perm_seed: int = 777,
) -> dict[int, dict[float, list[float]]]:
    """
    Returns: {layer_idx: {k_frac: [jaccard_values_for_normal_consecutive_pairs]}}

    permute_channels=True: permutation control — shuffle channel indices per token
    before top-K extraction. Expected Jaccard ≈ 0 for any K.
    """
    # Hooks: capture MLP block INPUT (pre-gate/up split — B3 clarification).
    # In Qwen3 the MLP block input is the tensor passed to the MLP module's forward.
    # Both W_gate and W_up receive this same tensor.
    captured: dict[int, torch.Tensor] = {}

    def make_prehook(idx: int):
        def hook(_mod, inputs):
            captured[idx] = inputs[0].detach().to(torch.float32)
        return hook

    handles = []
    for i in all_layers:
        handles.append(model.model.layers[i].mlp.register_forward_pre_hook(make_prehook(i)))

    # Per layer, per k_frac: list of normal-consecutive Jaccard values.
    result: dict[int, dict[float, list[float]]] = {
        L: {k: [] for k in k_fracs} for L in all_layers
    }

    rng_perm = np.random.default_rng(perm_seed)

    n_processed = 0
    with torch.inference_mode():
        for prompt_text, _stratum in prompts:
            captured.clear()
            encoded = tokenizer(prompt_text, return_tensors="pt",
                                truncation=True, max_length=max_length)
            ids = encoded.input_ids.to(device)
            if ids.shape[-1] < 4:
                continue
            try:
                model(input_ids=ids, use_cache=False)
            except Exception as e:
                print(f"  prompt failed ({e}); skipping", flush=True)
                continue

            for L in all_layers:
                if L not in captured:
                    continue
                h = captured[L].squeeze(0)  # (T, d)
                T = h.shape[0]

                # Permutation control: shuffle channels independently per token.
                if permute_channels:
                    perm_indices = np.argsort(rng_perm.random(size=(T, d)), axis=1)
                    perm_t = torch.from_numpy(perm_indices).long()
                    h = h.gather(1, perm_t)

                mag = h.abs()
                max_per_token = mag.max(dim=-1).values  # (T,)
                median = max_per_token.median().item()
                classes = [
                    "massive" if mp.item() > massive_multiplier * median else "normal"
                    for mp in max_per_token
                ]

                for k_frac in k_fracs:
                    K = max(1, int(k_frac * d))
                    top_idx = mag.topk(K, dim=-1).indices  # (T, K)
                    masks = torch.zeros((T, d), dtype=torch.bool)
                    masks.scatter_(1, top_idx, True)
                    for t in range(T - 1):
                        if classes[t] != "normal" or classes[t + 1] != "normal":
                            continue
                        inter = (masks[t] & masks[t + 1]).sum().item()
                        union = (masks[t] | masks[t + 1]).sum().item()
                        if union == 0:
                            continue
                        result[L][k_frac].append(inter / union)

            n_processed += 1

    for h in handles:
        h.remove()

    return result


# ---------------------------------------------------------------------------
# Random-init control
# ---------------------------------------------------------------------------

def make_random_init_model(config, device: torch.device) -> Any:
    """Create a model with the same architecture but random normal_(std=0.02) weights.
    Amendment B1: use normal_(std=0.02) per GPT-style init for near-flat singular spectrum.
    """
    import torch.nn as nn
    model_rand = AutoModelForCausalLM.from_config(config)
    # Re-initialize all linear layers with normal_(std=0.02).
    for module in model_rand.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model_rand.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    DEVICE = torch.device("cpu")
    MAX_LENGTH = 32
    K_FRACS = (0.001, 0.01, 0.05)  # 0.1%, 1%, 5%
    N_PER_STRATUM = 50
    OUT_DIR = Path("experiments/stage0/ladder_v2/round1_raok70b")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.time()-t0:.1f}s] Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()

    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}", flush=True)

    # All 28 layers (B2: extension of CF3's 7-layer finding to full 28).
    all_layers = list(range(n_layers))

    # Build prompts.
    random_prompts = gen_random_token_prompts(tokenizer, N_PER_STRATUM, length=MAX_LENGTH, seed=1)
    prompts: list[tuple[str, str]] = (
        [(p, "repetitive") for p in REPETITIVE_PROMPTS[:N_PER_STRATUM]] +
        [(p, "technical") for p in TECHNICAL_PROMPTS[:N_PER_STRATUM]] +
        [(p, "prose") for p in PROSE_PROMPTS[:N_PER_STRATUM]] +
        [(p, "random") for p in random_prompts]
    )
    print(f"  total prompts: {len(prompts)}", flush=True)

    # -----------------------------------------------------------------------
    # Trained model measurement (all 28 layers).
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Running trained model (all {n_layers} layers) ...", flush=True)
    trained_result = compute_jaccard_per_layer(
        model, tokenizer, prompts, all_layers, K_FRACS, d, DEVICE, max_length=MAX_LENGTH
    )
    print(f"  done.", flush=True)

    # Per-layer summary at K=1%.
    per_layer_k1pct: dict[int, float] = {}
    for L in all_layers:
        vals = trained_result[L][0.01]
        per_layer_k1pct[L] = float(np.mean(vals)) if vals else float("nan")

    print(f"\n[{time.time()-t0:.1f}s] Per-layer Jaccard at K=1% (trained):", flush=True)
    for L, j in per_layer_k1pct.items():
        flag = " <<< >0.65" if j > 0.65 else (" <<< <0.50" if j < 0.50 else "")
        print(f"  L{L:2d}: {j:.4f}{flag}", flush=True)

    # -----------------------------------------------------------------------
    # Permutation control (one representative layer: layer 14).
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Running permutation control (layer 14) ...", flush=True)
    perm_result = compute_jaccard_per_layer(
        model, tokenizer, prompts[:50], [14], K_FRACS, d, DEVICE,
        max_length=MAX_LENGTH, permute_channels=True
    )
    perm_j_k1pct = float(np.mean(perm_result[14][0.01])) if perm_result[14][0.01] else float("nan")
    print(f"  permuted Jaccard@K=1% at layer 14: {perm_j_k1pct:.4f} (expect ≈0)", flush=True)

    # -----------------------------------------------------------------------
    # Random-init control.
    # -----------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Running random-init control ...", flush=True)
    config = model.config
    model_rand = make_random_init_model(config, DEVICE)
    rand_result = compute_jaccard_per_layer(
        model_rand, tokenizer, prompts[:50], all_layers, (0.01,), d, DEVICE, max_length=MAX_LENGTH
    )
    del model_rand
    rand_per_layer = {
        L: float(np.mean(rand_result[L][0.01])) if rand_result[L][0.01] else float("nan")
        for L in all_layers
    }
    rand_mean = float(np.nanmean(list(rand_per_layer.values())))
    print(f"  random-init mean Jaccard@K=1% (all layers): {rand_mean:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------------
    valid = [j for j in per_layer_k1pct.values() if not np.isnan(j)]
    n_below_050 = sum(1 for j in valid if j < 0.50)
    n_above_065 = sum(1 for j in valid if j > 0.65)
    frac_below_050 = n_below_050 / len(valid) if valid else 0.0
    frac_above_065 = n_above_065 / len(valid) if valid else 0.0
    mean_j_k1pct = float(np.nanmean(list(per_layer_k1pct.values())))

    # Multi-K summary (across all layers, normal stratum).
    multi_k: dict[str, float] = {}
    for kf in K_FRACS:
        all_vals: list[float] = []
        for L in all_layers:
            all_vals.extend(trained_result[L][kf])
        multi_k[f"k{kf:.3f}"] = float(np.mean(all_vals)) if all_vals else float("nan")

    # CF3 comparison (CF3 found mean ≈ 0.31 at K=1% across 7 sampled layers).
    cf3_mean_7layer = 0.31
    gap_trained_vs_rand = mean_j_k1pct - rand_mean
    gap_trained_vs_perm = mean_j_k1pct - perm_j_k1pct

    # GO: Jaccard@K=1% < 0.50 in >=80% of layers.
    # NO-GO: Jaccard@K=1% > 0.65 in >=30% of layers.
    if frac_below_050 >= 0.80:
        verdict = "GO"
        verdict_note = (
            f"Jaccard@K=1% < 0.50 in {n_below_050}/{len(valid)} layers "
            f"({frac_below_050*100:.1f}% >= 80% threshold). "
            "CF3 K-dependent dynamicity generalizes to all 28 layers. Proceed to Rung 2."
        )
    elif frac_above_065 >= 0.30:
        verdict = "NO-GO"
        verdict_note = (
            f"Jaccard@K=1% > 0.65 in {n_above_065}/{len(valid)} layers "
            f"({frac_above_065*100:.1f}% >= 30% threshold). "
            "CF3 finding does NOT generalize. Class-level kill: C2 (R-RAOK-70B, F-OCSSQ, C-RAOK-Grounded) + all RAOK derivatives."
        )
    else:
        verdict = "GRAY"
        verdict_note = (
            f"Jaccard@K=1%: {n_below_050}/{len(valid)} layers below 0.50, "
            f"{n_above_065}/{len(valid)} above 0.65. "
            "Gray zone — adaptive bypass follow-up needed (Section 7 of Stage 5 plan)."
        )

    elapsed = time.time() - t0
    print(f"\n[{elapsed:.1f}s] VERDICT: {verdict}", flush=True)
    print(f"  {verdict_note}", flush=True)

    # -----------------------------------------------------------------------
    # Write outputs.
    # -----------------------------------------------------------------------
    summary: dict[str, Any] = {
        "experiment": "R-RAOK-70B Rung 1",
        "model_id": MODEL_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "n_layers": n_layers,
        "d_hidden": d,
        "all_layers_measured": all_layers,
        "n_prompts": len(prompts),
        "k_fracs": list(K_FRACS),
        "b2_note": (
            "H1 restated per amendment B2: this is the FIRST full 28-layer Jaccard measurement. "
            "CF3 was 7 sampled layers; Rung 1 extends, does not assume, that finding."
        ),
        "b3_note": (
            "Amendment B3: hooks capture MLP block INPUT (tensor before W_gate/W_up split). "
            "Both W_gate and W_up receive the same quantized activation in deployment."
        ),
        "b4_note": "Amendment B4: INT4 scale-factor unit test (max_abs/7) PASSED in rraok_int4_unit_test.py.",
        "b1_note": "Amendment B1: random-init control uses torch.nn.init.normal_(std=0.02).",
        "per_layer_jaccard_k1pct": per_layer_k1pct,
        "multi_k_mean_jaccard_all_layers": multi_k,
        "cf3_comparison": {
            "cf3_mean_7layer_k1pct": cf3_mean_7layer,
            "rung1_mean_28layer_k1pct": mean_j_k1pct,
            "delta_vs_cf3": mean_j_k1pct - cf3_mean_7layer,
        },
        "controls": {
            "permutation_k1pct_layer14": perm_j_k1pct,
            "random_init_mean_k1pct_all_layers": rand_mean,
            "gap_trained_vs_perm": gap_trained_vs_perm,
            "gap_trained_vs_rand": gap_trained_vs_rand,
            "perm_control_pass": perm_j_k1pct < 0.10,
            "rand_init_pass": gap_trained_vs_rand > 0.05,
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
        "go_threshold": "Jaccard@K=1% < 0.50 in >=80% of 28 layers",
        "nogo_threshold": "Jaccard@K=1% > 0.65 in >=30% of 28 layers",
        "n_layers_below_050": n_below_050,
        "n_layers_above_065": n_above_065,
        "frac_below_050": frac_below_050,
        "frac_above_065": frac_above_065,
        "mean_k1pct_all_layers": mean_j_k1pct,
    }

    out_json = OUT_DIR / "tier0_stability_per_layer.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
