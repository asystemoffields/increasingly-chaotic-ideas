r"""PDAP outlier-Jaccard probe — Round 2 / Stage 6 amended.

Tests whether activation outliers in transformer hidden states are token-dynamic
(per-token rotating outlier channels) or channel-static (LLM.int8() pattern).

Key amendments per Stage 6:
  - Stratify per-token Jaccard by token type:
      * "massive-activation": max |h[t]| > 5x batch median
      * "normal": otherwise
    PrefixQuant (arXiv:2410.05265) showed token-wise outliers (BOS, delimiters,
    sparse) drive the dynamic component; without stratification they contaminate
    the mean Jaccard.
  - Multi-K reporting: top-{0.1%, 1%, 5%, 10%} outlier sets at each token.
  - 8-layer dense sampling (Qwen3-1.7B has 28 layers; sample {2,6,10,14,18,22,26}).
  - 4 prompt strata: repetitive, technical/code, prose, random tokens.

Go/No-Go (in normal-token stratum):
  - GO     : mean Jaccard < 0.50 AND top-k 90%-coverage > 5% of channels
  - NO-GO  : mean Jaccard > 0.85
  - GRAY   : 0.50-0.85 (bimodal — PrefixQuant pattern, hybrid scheme indicated)

Reference:
  LLM.int8() (arXiv:2208.07339) — claimed channel-static, no formal Jaccard.
  SmoothQuant (arXiv:2211.10438) — assumed channel-static.
  PrefixQuant (arXiv:2410.05265) — channel-wise vs token-wise decomposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


# Stratum definitions: 50 prompts each.
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
    "P(A|B) = P(B|A) * P(A) / P(B), Bayes' theorem; the posterior is proportional to likelihood times prior.",
    "Type-erased polymorphism in C++ with std::variant<int, std::string, std::vector<double>> storage.",
    "@dataclass(frozen=True) class Vector: x: float; y: float; def dot(self, other) -> float: return self.x * other.x + self.y * other.y",
    "TLS 1.3 handshake: ClientHello with key_share extension, ServerHello, EncryptedExtensions, Finished.",
    "VPMADDUBSW ymm0, ymm1, ymm2 — vertical multiply unsigned bytes by signed bytes, sum to 16-bit lanes.",
    "Lambda calculus: (\\x. x x) (\\x. x x) is the omega combinator and does not normalize.",
    "Floyd-Warshall: for k in V: for i in V: for j in V: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])",
    "CRDTs: G-Counter, PN-Counter, OR-Set, LWW-Element-Set; conflict-free merging via lattice operations.",
    "Mean field theory replaces N-body interactions with a single average field, reducing to one-body problems.",
    "FFT bit-reversal permutation: index i maps to reverse_bits(i, log2(N)); butterfly stages combine subproblems.",
    "Erlang OTP: gen_server, gen_statem, supervisor, application; let it crash philosophy with restart strategies.",
    "Reed-Solomon (255, 223): 32 parity bytes correct up to 16 byte errors per 255-byte block in the Galois field GF(256).",
    "Chebyshev polynomials of the first kind: T0(x) = 1, T1(x) = x, T_{n+1} = 2x T_n - T_{n-1}; minimax property.",
    "Conway's Game of Life: live cell with 2-3 neighbors survives; dead cell with exactly 3 becomes live.",
    "QUIC stream multiplexing over UDP; congestion control via NewReno or BBR; 0-RTT resumption with session tickets.",
    "Strassen matmul: 7 multiplications instead of 8 for 2x2 blocks; recursive O(n^log2(7)) ≈ O(n^2.807).",
    "Hidden Markov Model: forward-backward algorithm; Viterbi for most likely state sequence; Baum-Welch for parameter estimation.",
    "Newton's method: x_{n+1} = x_n - f(x_n) / f'(x_n); quadratic convergence near simple roots if f' nonzero.",
    "Regex: ^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$ matches ISO 8601 date strings YYYY-MM-DD.",
    "x86 PUSH RBP, MOV RBP RSP, SUB RSP 32, MOV [RBP-8] RDI, MOV [RBP-16] RSI; standard prologue.",
    "Operating system: process state machine has Running, Ready, Waiting, Zombie; transitions via scheduler and syscalls.",
    "Backpropagation: chain rule applied recursively; ∂L/∂w_l = ∂L/∂a_l * a_{l-1}^T; update via SGD or Adam.",
    "Distributed consensus Raft: leader election term, log replication AppendEntries, safety via election restriction.",
    "BST insertion: if key < node.key go left else go right; recurse until null; create new node and link.",
    "Galois group of x^4 - 2 over Q is the dihedral group D4 of order 8; intermediate fields correspond to subgroups.",
    "Zermelo-Fraenkel set theory with axiom of choice: extensionality, pairing, union, power set, infinity, replacement.",
    "AVX-512: VPCONFLICTD detects conflicts in indirect store; VBMI2 adds bit shuffle and compress; VPMULLD multiplies 32-bit ints.",
    "BTRFS: copy-on-write subvolume snapshots; B-tree metadata; checksumming with CRC32C; balance and scrub.",
    "Quantum Hall effect: σ_xy = ν e^2/h with integer ν; topological invariant; edge states robust to disorder.",
]

PROSE_PROMPTS = [
    "The library at Alexandria stood for nearly a thousand years before its slow decline reduced its collections.",
    "Beneath the pavements of Pompeii, archaeologists discovered loaves of bread carbonized by the eruption of 79 CE.",
    "She had been walking for three days through the high grass when she finally saw the river in the distance.",
    "The astronomer adjusted the eyepiece and waited for the seeing to settle, hoping for a glimpse of Saturn's rings.",
    "His grandmother kept her recipes on index cards in a tin box, each card stained with the ingredient it commemorated.",
    "The river had changed course three times in the last century, each time leaving a new oxbow lake behind.",
    "After the storm, the streets were littered with branches, and a single shoe sat improbably on top of a parked car.",
    "The watchmaker held the tiny gear up to the light, examining the polish with a magnifying loupe in his eye.",
    "On winter mornings the windows fogged up so thoroughly that you had to wipe a porthole to see outside.",
    "The librarian remembered every patron who had borrowed the rare folio, though only seven had ever asked for it.",
    "Migration patterns of the monarch butterfly span four generations to complete a single round-trip from Mexico.",
    "He had grown up in a town where everyone could trace their lineage to one of three founding families.",
    "The cellist tuned her instrument by ear, ignoring the digital tuner her teacher had given her last spring.",
    "Soft snow accumulated on the brick walls overnight, transforming familiar geometry into rounded contours.",
    "Her father had been a merchant marine, and the spice cabinet always carried scents from distant ports.",
    "The composer worked best in absolute silence, with cotton in his ears and a thick wool blanket over the windows.",
    "By the third winter, the cabin had begun to settle, and the floorboards creaked in patterns she had memorized.",
    "Finches arrived at the feeder in small waves, each cohort displacing the previous with subtle aggression.",
    "The cartographer's study was lit by a single oil lamp, and the maps on the walls fluttered in the draft.",
    "She remembered the sound of her grandfather's laugh more clearly than the details of his face.",
    "Ocean currents carry tropical seeds to northern beaches, where they occasionally take root in unexpected places.",
    "The summer kitchen had been added to the back of the house in 1923 to keep cooking heat out of the main rooms.",
    "Standing on the platform, she watched the train pull in slowly, brakes hissing as it eased to a halt.",
    "The bookbinder used wheat paste, never PVA, and her repairs lasted longer than the original commercial bindings.",
    "Eel migration remains poorly understood; juveniles travel from the Sargasso Sea to European rivers via mechanisms we still debate.",
    "Late afternoon light slanted through the venetian blinds, projecting striped patterns across the office wall.",
    "The cobbler kept a wooden last for each regular customer, shaped to the individual peculiarities of their feet.",
    "Her notebook was held together with a single thick rubber band she had carried in her pocket since college.",
    "The river ferry ran every twenty minutes during summer, but only twice an hour after the leaves fell.",
    "Conifers in the boreal forest produce seeds that require fire to germinate, an adaptation to lightning regimes.",
    "The conductor lowered his baton and the orchestra fell silent before the audience realized the piece had ended.",
    "He spent his retirement carving wooden birds, each one painstakingly accurate to the species in its summer plumage.",
    "Glacial moraines record the boundaries of past ice advances, and farmers in upstate New York still hit boulders deposited 18,000 years ago.",
    "The bakery opened at four in the morning so the early commuters could buy hot bread on their way to the train station.",
    "Lichens on the granite outcrop grew at less than a millimeter per year, marking decades in barely visible increments.",
    "She washed the dishes by hand, even though the dishwasher worked perfectly, because the rhythm calmed her nerves.",
    "The carpenter selected each board for its grain pattern, refusing to use any with knots near the visible ends.",
    "When the streetlamps came on at dusk, the neighborhood children would scatter from the playground for dinner.",
    "The clock in the bell tower had stopped working in 1967 and no one in town remembered why it had not been fixed.",
    "Fishermen had names for every tide and every current, vocabulary their grandfathers had taught them on summer afternoons.",
    "The pottery wheel turned slowly under her practiced hands, and a vase began to rise from the gray clay center.",
    "Old maps of the harbor showed shoals that had since silted in, and reefs that had since been dredged.",
    "He kept a small notebook in his jacket pocket and wrote down a single observation each day, dating each entry.",
    "The retired mathematician spent her mornings on a chess problem the rest of the world had abandoned in 1962.",
    "The tea ceremony required four hours of preparation for fifteen minutes of seated quiet with two close friends.",
    "Spring runoff carved a new channel through the meadow, exposing layers of clay that had not seen the surface in centuries.",
    "The accordionist played each Sunday afternoon in the park, and elderly couples danced slowly on the cracked pavement.",
    "Her writing desk faced east so that morning light fell on the page; afternoons were for housework or walks.",
    "The mason had laid the chimney bricks one summer fifty years ago, and his initials were still visible on the lowest course.",
    "Across the valley, the church bells rang out the hours, and farmers planning their day looked up to count.",
]


def gen_random_token_prompts(tokenizer, n: int, length: int, seed: int) -> list[str]:
    """Adversarial baseline: random token IDs decoded back to text."""
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    out = []
    for _ in range(n):
        ids = rng.integers(0, vocab, size=length).tolist()
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=32,
                   help="Per-prompt token cap (we want ~20 tokens average).")
    p.add_argument("--n-per-stratum", type=int, default=50)
    p.add_argument("--k-fracs", default="0.001,0.01,0.05,0.10")
    p.add_argument("--massive-multiplier", type=float, default=5.0,
                   help="Token classified massive-activation if max(|h|) > "
                        "this multiplier × batch median(max(|h|)).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/stage0/ladder/round2_pdap"))
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_fracs = tuple(float(k) for k in args.k_fracs.split(","))

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    print(f"  n_layers={n_layers}  d={d}")

    # Layer sampling: 7-8 evenly spaced layers (Stage 6 amendment).
    sample_layers = sorted(set(int(round(x))
                               for x in np.linspace(2, n_layers - 1, 7)))
    print(f"  sampled layers: {sample_layers}")

    # Build prompts.
    rng = np.random.default_rng(args.seed)
    repetitive = list(REPETITIVE_PROMPTS)[: args.n_per_stratum]
    technical = list(TECHNICAL_PROMPTS)[: args.n_per_stratum]
    prose = list(PROSE_PROMPTS)[: args.n_per_stratum]
    random_text = gen_random_token_prompts(
        tokenizer, args.n_per_stratum, length=args.max_length, seed=args.seed)

    prompts = (
        [(p, "repetitive") for p in repetitive] +
        [(p, "technical") for p in technical] +
        [(p, "prose") for p in prose] +
        [(p, "random") for p in random_text]
    )
    print(f"  total prompts: {len(prompts)}")

    # Hooks.
    captured: dict[int, torch.Tensor] = {}

    def make_prehook(idx: int):
        def hook(_mod, inputs):
            captured[idx] = inputs[0].detach().to(torch.float32)
        return hook

    handles = []
    for i in sample_layers:
        handles.append(
            model.model.layers[i].register_forward_pre_hook(make_prehook(i))
        )

    # Aggregate across prompts.
    # Per (layer, k_frac, stratum, token_class): list of Jaccard values.
    per_key: dict[tuple, list[float]] = {}
    # Per layer: counts and channel-hit-frequency (cumulative coverage).
    channel_hit: dict[int, np.ndarray] = {L: np.zeros(d, dtype=np.int64)
                                          for L in sample_layers}
    # Per (layer, stratum): list of (token_class, top_indices_at_k=1%).
    n_processed = 0
    n_tokens_total = 0

    for prompt_text, stratum in prompts:
        captured.clear()
        encoded = tokenizer(prompt_text, return_tensors="pt",
                            truncation=True, max_length=args.max_length)
        ids = encoded.input_ids.to(device)
        if ids.shape[-1] < 4:
            continue
        try:
            _ = model(input_ids=ids, use_cache=False)
        except Exception as e:
            print(f"  prompt failed ({e}); skipping")
            continue

        for L in sample_layers:
            if L not in captured:
                continue
            h = captured[L].squeeze(0)  # (T, d)
            T = h.shape[0]
            mag = h.abs()
            max_per_token = mag.max(dim=-1).values  # (T,)
            median = max_per_token.median().item()
            # Token-type classification.
            classes = ["massive" if mp.item() > args.massive_multiplier * median
                       else "normal" for mp in max_per_token]

            for k_frac in k_fracs:
                K = max(1, int(k_frac * d))
                top_idx = mag.topk(K, dim=-1).indices  # (T, K)
                # Track per-channel hit frequency for cumulative coverage.
                if abs(k_frac - 0.01) < 1e-6:
                    flat = top_idx.flatten().cpu().numpy()
                    np.add.at(channel_hit[L], flat, 1)
                # Convert to bool masks for Jaccard.
                masks = torch.zeros((T, d), dtype=torch.bool)
                masks.scatter_(1, top_idx, True)
                # Consecutive-pair Jaccard.
                for t in range(T - 1):
                    inter = (masks[t] & masks[t + 1]).sum().item()
                    union = (masks[t] | masks[t + 1]).sum().item()
                    if union == 0:
                        continue
                    j = inter / union
                    pair_class = "normal" if (
                        classes[t] == "normal" and classes[t + 1] == "normal"
                    ) else "massive"
                    key = (L, k_frac, stratum, pair_class)
                    per_key.setdefault(key, []).append(j)

        n_processed += 1
        n_tokens_total += int(ids.shape[-1])

    for h_handle in handles:
        h_handle.remove()

    # Aggregate.
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_prompts_processed": n_processed,
        "n_tokens_total": n_tokens_total,
        "n_layers_total": n_layers,
        "sample_layers": sample_layers,
        "d_hidden": d,
        "k_fracs": list(k_fracs),
        "massive_multiplier": args.massive_multiplier,
        "timestamp": datetime.now(UTC).isoformat(),
        "prefixquant_reference": "arXiv:2410.05265 (Oct 2024) — channel-wise vs token-wise outlier decomposition. We stratify per their pattern.",
        "per_key": {},
        "aggregate": {},
        "cumulative_coverage": {},
    }

    # Cumulative coverage at k_frac=1% per layer.
    for L in sample_layers:
        hits = channel_hit[L]
        sorted_hits = np.sort(hits)[::-1]
        cum = np.cumsum(sorted_hits)
        total = cum[-1]
        if total == 0:
            continue
        # Smallest fraction of channels covering 90% of outlier events.
        frac90 = float((cum >= 0.9 * total).argmax() + 1) / d
        # Largest concentration: top-1% of channels capture what fraction of events.
        top_1pct = max(1, int(0.01 * d))
        cap_top_1pct = float(cum[top_1pct - 1] / total)
        summary["cumulative_coverage"][str(L)] = {
            "frac_channels_for_90pct_events": frac90,
            "events_captured_by_top_1pct_channels": cap_top_1pct,
        }

    # Stratified Jaccard means.
    for key, vals in per_key.items():
        L, k_frac, stratum, cls = key
        a = np.array(vals)
        summary["per_key"][f"L{L}_k{k_frac:.3f}_{stratum}_{cls}"] = {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.median(a)),
            "p10": float(np.quantile(a, 0.1)),
            "p90": float(np.quantile(a, 0.9)),
        }

    # Overall: mean Jaccard within normal-token stratum, all strata, k=1%.
    def agg(k_target: float, cls_target: str, layers=None) -> dict[str, float]:
        vals = []
        for key, v in per_key.items():
            L, k_frac, stratum, cls = key
            if abs(k_frac - k_target) > 1e-6:
                continue
            if cls != cls_target:
                continue
            if layers is not None and L not in layers:
                continue
            vals.extend(v)
        a = np.array(vals) if vals else np.array([0.0])
        return {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else 0.0,
            "p50": float(np.median(a)) if a.size else 0.0,
        }

    summary["aggregate"]["normal_k1pct"] = agg(0.01, "normal")
    summary["aggregate"]["massive_k1pct"] = agg(0.01, "massive")
    summary["aggregate"]["normal_k01pct"] = agg(0.001, "normal")
    summary["aggregate"]["normal_k5pct"] = agg(0.05, "normal")
    summary["aggregate"]["normal_k10pct"] = agg(0.10, "normal")

    # Verdict.
    n_normal = summary["aggregate"]["normal_k1pct"]["mean"]
    n_massive = summary["aggregate"]["massive_k1pct"]["mean"]
    coverage_5pct_layers = sum(
        1 for L in sample_layers
        if summary["cumulative_coverage"].get(str(L), {})
            .get("frac_channels_for_90pct_events", 0) > 0.05
    )
    coverage_pass = coverage_5pct_layers / max(1, len(sample_layers)) > 0.5

    if n_normal < 0.50 and coverage_pass:
        verdict = "GO (token-dynamic)"
    elif n_normal > 0.85:
        verdict = "NO-GO (channel-static; LLM.int8() pattern)"
    else:
        verdict = "GRAY (PrefixQuant pattern likely; bimodal massive/normal)"
    summary["verdict"] = verdict

    out_json = args.output_dir / "pdap_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Per-layer Jaccard at k=1% by token class.
        ax = axes[0, 0]
        for cls in ["normal", "massive"]:
            ys = []
            for L in sample_layers:
                vs = []
                for stratum in ["repetitive", "technical", "prose", "random"]:
                    rec = summary["per_key"].get(f"L{L}_k0.010_{stratum}_{cls}")
                    if rec:
                        vs.append(rec["mean"])
                ys.append(np.mean(vs) if vs else np.nan)
            ax.plot(sample_layers, ys, marker="o", label=cls)
        ax.axhline(0.50, color="green", linestyle="--", alpha=0.5, label="GO 0.50")
        ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="NO-GO 0.85")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean Jaccard (consecutive normal pairs, k=1%)")
        ax.set_title("PDAP per-token outlier-set stability")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        # Per-K Jaccard for normal stratum.
        ax = axes[0, 1]
        for k_frac in k_fracs:
            ys = []
            for L in sample_layers:
                vs = []
                for stratum in ["repetitive", "technical", "prose", "random"]:
                    rec = summary["per_key"].get(
                        f"L{L}_k{k_frac:.3f}_{stratum}_normal")
                    if rec:
                        vs.append(rec["mean"])
                ys.append(np.mean(vs) if vs else np.nan)
            ax.plot(sample_layers, ys, marker="o", label=f"k={k_frac*100:.1f}%")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean Jaccard (normal stratum)")
        ax.set_title("Multi-K per-layer Jaccard")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        # Per-stratum Jaccard at k=1%, normal class.
        ax = axes[1, 0]
        for stratum in ["repetitive", "technical", "prose", "random"]:
            ys = []
            for L in sample_layers:
                rec = summary["per_key"].get(f"L{L}_k0.010_{stratum}_normal")
                ys.append(rec["mean"] if rec else np.nan)
            ax.plot(sample_layers, ys, marker="o", label=stratum)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean Jaccard")
        ax.set_title("Per-stratum normal-token Jaccard, k=1%")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        # Cumulative coverage: at each layer, fraction of channels for 90% of events.
        ax = axes[1, 1]
        ys = []
        for L in sample_layers:
            rec = summary["cumulative_coverage"].get(str(L))
            ys.append(rec["frac_channels_for_90pct_events"] if rec else np.nan)
        ax.plot(sample_layers, ys, marker="^", color="purple")
        ax.axhline(0.05, color="green", linestyle="--", alpha=0.5,
                   label="dynamic threshold 5%")
        ax.set_xlabel("layer")
        ax.set_ylabel("frac channels covering 90% events")
        ax.set_title("Channel concentration (k=1%)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_png = args.output_dir / "pdap_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Summary.md.
    md_lines = [
        f"# PDAP outlier-Jaccard probe — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Prompts: {n_processed}, total tokens: {n_tokens_total}",
        f"Sampled layers ({len(sample_layers)} of {n_layers}): {sample_layers}",
        f"K fractions tested: {[f'{k*100:.1f}%' for k in k_fracs]}",
        f"Massive-activation classifier: max(|h|) > {args.massive_multiplier}× batch median",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Headline numbers (k = 1% of d)",
        "",
        f"- Mean Jaccard (consecutive normal-token pairs): "
        f"**{summary['aggregate']['normal_k1pct']['mean']:.3f}**  "
        f"(n={summary['aggregate']['normal_k1pct']['n']})",
        f"- Mean Jaccard (massive-activation pairs): "
        f"**{summary['aggregate']['massive_k1pct']['mean']:.3f}**  "
        f"(n={summary['aggregate']['massive_k1pct']['n']})",
        "",
        "If the *normal* mean is high (≥0.85) and the *massive* mean is low, "
        "this is the PrefixQuant pattern — channel-static for the bulk of tokens, "
        "with sparse token-wise outliers driving the apparent dynamism.",
        "",
        "## Multi-K Jaccard (normal stratum, mean across layers)",
        "",
        "| K | Jaccard mean |",
        "|---|---|",
        f"| 0.1% | {summary['aggregate']['normal_k01pct']['mean']:.3f} |",
        f"| 1.0% | {summary['aggregate']['normal_k1pct']['mean']:.3f} |",
        f"| 5.0% | {summary['aggregate']['normal_k5pct']['mean']:.3f} |",
        f"| 10.0% | {summary['aggregate']['normal_k10pct']['mean']:.3f} |",
        "",
        "## Channel concentration (top-K=1%)",
        "",
        "| L | frac channels for 90% events | top-1%-channels capture |",
        "|---|---|---|",
    ]
    for L in sample_layers:
        rec = summary["cumulative_coverage"].get(str(L))
        if rec:
            md_lines.append(
                f"| {L} | {rec['frac_channels_for_90pct_events']:.3f} "
                f"| {rec['events_captured_by_top_1pct_channels']:.3f} |"
            )

    md_lines += [
        "",
        "## Reference",
        "",
        "- **PrefixQuant** (arXiv:2410.05265): channel-wise vs token-wise outlier decomposition. We stratify per their pattern.",
        "- LLM.int8() (arXiv:2208.07339): claimed channel-static qualitatively; no formal Jaccard.",
        "- SmoothQuant (arXiv:2211.10438): assumed channel-static.",
        "- Quaff (arXiv:2505.14742): Outlier Spatial Stability Hypothesis (fine-tuning, not inference).",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
