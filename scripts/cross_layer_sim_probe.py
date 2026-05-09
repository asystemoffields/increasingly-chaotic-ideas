r"""Cross-layer activation similarity probe — Round 1 / Idea D (amended).

For each MLP layer L of Qwen3-0.6B-Base, three rankings of the intermediate
neurons (j = 0..r-1) are computed per token:

  1. Gold (actual top-K firing post-SwiGLU):
       g_j = | silu(W_gate[L] @ h_L)_j  *  (W_up[L] @ h_L)_j |

  2. Cross-layer predicted (zero-overhead lookahead from L-1):
       c_j = | <h_{L-1}, W_gate[L]_row j> |

  3. Within-layer baseline (gate pre-activation magnitude at L):
       w_j = | <h_L, W_gate[L]_row j> |     (== |gate_pre_{L,j}|)

Metrics per layer:
  - precision@30% (cross-layer)  = |top30%(c) ∩ top30%(g)| / K
  - precision@30% (within-layer) = |top30%(w) ∩ top30%(g)| / K
  - cosine_sim(h_{L-1}, h_L), cosine_sim(h_L, h_{L+1})    (reference, not gates)

The within-layer baseline isolates the *cross-layer* claim from the trivial
"gate dot product correlates with post-SwiGLU magnitude" effect. The
selection's defensible novelty is whether the L-1 lookahead is comparable
to the within-layer (no-lookahead) signal.

Go threshold: mean precision@30% (cross-layer) >= 0.80 across MLP layers
              AND cross is no more than 0.10 below within-layer.
No-go:        cross < 0.55, OR cross is more than 0.20 below within-layer.

Reference: Deja Vu (Liu et al., ICML 2023, arXiv:2310.17157) showed
trained cross-layer MLP predictors achieve 93-99% on ReLU/OPT-family
models. This experiment asks whether a *zero-overhead dot-product*
predictor on *SwiGLU/Qwen3* preserves enough signal.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--k-frac", type=float, default=0.30,
                   help="Top-K fraction for precision@K metric.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/stage0/ladder/round1_idea_d"))
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


# Hard-coded short-answer instruction prompts (used to mix with WikiText).
INSTRUCTION_PROMPTS = [
    "Explain the difference between viscosity and surface tension in two sentences.",
    "What is a Faraday cage, and why does it block radio signals?",
    "Describe how a sourdough starter is maintained over months.",
    "Name three causes of the 1973 oil crisis and one consequence.",
    "Why does adding salt to water raise its boiling point slightly?",
    "Summarize the plot of Don Quixote in three sentences.",
    "What does it mean for a problem to be NP-complete? Give one example.",
    "Describe how a pneumatic tire absorbs road shock.",
    "Why are mitochondria sometimes called the powerhouse of the cell?",
    "Explain how a transformer (electrical, not LLM) steps voltage up or down.",
    "What is the difference between a virus and a bacterium?",
    "How does a sextant determine latitude at sea?",
    "Describe the Krebs cycle in two sentences.",
    "Why did the Roman Empire adopt Christianity in the fourth century?",
    "Explain how noise-canceling headphones work.",
    "What is the difference between a star and a planet?",
    "How does a refrigerator move heat against a temperature gradient?",
    "Why are Antarctic ice cores useful for studying climate history?",
    "Explain photosynthesis in plants in two sentences.",
    "What is the role of the kidney in maintaining blood pressure?",
    "Describe how a sundial works and why its accuracy varies by season.",
    "Why did the Hagia Sophia change religion three times in its history?",
    "What is the principle behind fiber-optic communication?",
    "How does a CPU's branch predictor reduce pipeline stalls?",
    "Explain how vaccines train the immune system.",
    "What is the difference between weather and climate?",
    "Describe how a pendulum clock keeps time.",
    "Why is the speed of light considered a universal constant?",
    "How does a desalination plant produce fresh water from seawater?",
    "Explain Bayes' theorem in two sentences.",
    "What is gravitational lensing, and what does it tell us?",
    "Describe how an MRI machine produces an image.",
    "Why are honeybees considered keystone species?",
    "Explain how a hydraulic brake system multiplies force.",
    "What caused the Cambrian explosion?",
    "Describe the difference between AC and DC electricity.",
    "How does sourdough fermentation differ from yeasted bread fermentation?",
    "Why is helium scarce on Earth despite being abundant in the universe?",
    "Explain how a barometer measures atmospheric pressure.",
    "What is the role of the cerebellum in motor coordination?",
    "Describe how fingerprints form during fetal development.",
    "Why do we get jet lag, and how can it be mitigated?",
    "Explain the difference between deductive and inductive reasoning.",
    "How does a Fresnel lens work, and where is it used?",
    "What is the Coriolis effect, and how does it affect weather?",
    "Describe how submarines control buoyancy.",
    "Why is Antarctica technically a desert?",
    "Explain how solar panels convert sunlight into electricity.",
    "What is the role of REM sleep in memory consolidation?",
    "Describe the difference between mass and weight.",
]


def load_prompts(n_total: int, seed: int) -> list[str]:
    """Load up to n_total/2 from WikiText-2 + n_total/2 from instructions."""
    rng = np.random.default_rng(seed)
    n_wiki = n_total // 2
    n_instr = n_total - n_wiki

    prompts: list[str] = []

    # WikiText-2.
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t) > 200]
        idx = rng.choice(len(texts), size=min(n_wiki, len(texts)), replace=False)
        for i in idx:
            prompts.append(texts[int(i)])
        print(f"  loaded {len(prompts)} WikiText-2 segments")
    except Exception as e:
        print(f"  WikiText-2 load failed ({e}); using instructions only")

    # Instructions: cycle through INSTRUCTION_PROMPTS until we have n_instr.
    instr_pool = list(INSTRUCTION_PROMPTS)
    rng.shuffle(instr_pool)
    while len(prompts) < n_total:
        prompts.append(instr_pool[len(prompts) % len(instr_pool)])

    rng.shuffle(prompts)
    return prompts[:n_total]


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    r = int(model.config.intermediate_size)
    K = max(1, int(args.k_frac * r))
    print(f"  n_layers={n_layers}  d={d}  r={r}  K={K} ({100*args.k_frac:.0f}% of r)")

    print(f"Loading {args.n_prompts} prompts ...")
    prompts = load_prompts(args.n_prompts, args.seed)
    print(f"  total prompts: {len(prompts)}")

    # Per-layer running stats. We use cross-layer prediction L-1 -> L for
    # L in [1, n_layers-1] (need L-1 to exist).
    layer_indices = list(range(1, n_layers))
    metrics_acc: dict[str, dict[int, list[float]]] = {
        "precision_cross": {L: [] for L in layer_indices},
        "precision_within": {L: [] for L in layer_indices},
        "cos_prev": {L: [] for L in layer_indices},      # cos(h_{L-1}, h_L)
        "cos_next": {L: [] for L in layer_indices[:-1]}, # cos(h_L, h_{L+1})
    }

    captured: dict[int, torch.Tensor] = {}

    def make_prehook(idx: int):
        def hook(_mod, inputs):
            captured[idx] = inputs[0].detach().to(torch.float32)
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_prehook(i)))

    print("Running forward passes ...")
    n_done = 0
    for i, prompt in enumerate(prompts):
        captured.clear()
        encoded = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=args.max_length)
        ids = encoded.input_ids.to(device)
        if ids.shape[-1] < 8:
            continue  # skip very short prompts
        try:
            _ = model(input_ids=ids, use_cache=False)
        except Exception as e:
            print(f"  prompt {i}: forward failed ({e}); skipping")
            continue

        # For each MLP layer L compute the three rankings.
        for L in layer_indices:
            if L not in captured or (L - 1) not in captured:
                continue
            h_prev = captured[L - 1].squeeze(0)  # (T, d)
            h_curr = captured[L].squeeze(0)      # (T, d)
            mlp = model.model.layers[L].mlp
            W_gate = mlp.gate_proj.weight.detach().to(torch.float32)  # (r, d)
            W_up = mlp.up_proj.weight.detach().to(torch.float32)      # (r, d)

            # Predictors (T, r) magnitudes.
            gate_pre = h_curr @ W_gate.T          # (T, r) — within baseline
            up_act = h_curr @ W_up.T              # (T, r)
            cross_pre = h_prev @ W_gate.T         # (T, r) — cross-layer
            gold = (F.silu(gate_pre) * up_act).abs()
            cross_mag = cross_pre.abs()
            within_mag = gate_pre.abs()

            # Top-K indices per token (T, K).
            gold_top = gold.topk(K, dim=-1).indices
            cross_top = cross_mag.topk(K, dim=-1).indices
            within_top = within_mag.topk(K, dim=-1).indices

            # Precision@K = |intersection| / K, vectorized via boolean masks.
            T = gold_top.shape[0]
            r_dim = gold.shape[-1]
            mask_gold = torch.zeros((T, r_dim), dtype=torch.bool)
            mask_gold.scatter_(1, gold_top, True)
            cross_hits = mask_gold.gather(1, cross_top).sum(dim=-1).float() / K
            within_hits = mask_gold.gather(1, within_top).sum(dim=-1).float() / K

            metrics_acc["precision_cross"][L].extend(cross_hits.tolist())
            metrics_acc["precision_within"][L].extend(within_hits.tolist())

            # Cosine sims (per-token, then mean across token dim for this prompt).
            cos_pl = F.cosine_similarity(h_prev, h_curr, dim=-1)
            metrics_acc["cos_prev"][L].extend(cos_pl.tolist())
            if L + 1 in captured and L in metrics_acc["cos_next"]:
                h_next = captured[L + 1].squeeze(0)
                cos_nl = F.cosine_similarity(h_curr, h_next, dim=-1)
                metrics_acc["cos_next"][L].extend(cos_nl.tolist())

        n_done += 1
        if (n_done) % 20 == 0:
            print(f"  {n_done}/{len(prompts)} prompts processed")

    for h in handles:
        h.remove()

    # Aggregate per-layer.
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_prompts_attempted": len(prompts),
        "n_prompts_processed": n_done,
        "n_layers": n_layers,
        "d": d, "r": r,
        "k_frac": args.k_frac, "K": K,
        "max_length": args.max_length,
        "seed": args.seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "deja_vu_reference": "arXiv:2310.17157 (Liu et al., ICML 2023) — trained cross-layer MLP predictor on ReLU/OPT achieves 93-99% precision; this experiment is the SwiGLU + zero-overhead dot-product analog.",
        "per_layer": {},
        "aggregate": {},
    }

    layer_means_cross: list[float] = []
    layer_means_within: list[float] = []
    for L in layer_indices:
        c = np.array(metrics_acc["precision_cross"][L])
        w = np.array(metrics_acc["precision_within"][L])
        cp = np.array(metrics_acc["cos_prev"][L])
        cn = np.array(metrics_acc["cos_next"].get(L, []))
        if c.size == 0:
            continue
        rec = {
            "n_samples": int(c.size),
            "precision_cross_mean": float(c.mean()),
            "precision_cross_p10": float(np.quantile(c, 0.1)),
            "precision_cross_p50": float(np.quantile(c, 0.5)),
            "precision_cross_p90": float(np.quantile(c, 0.9)),
            "precision_within_mean": float(w.mean()),
            "precision_within_p50": float(np.quantile(w, 0.5)),
            "cross_minus_within_mean": float(c.mean() - w.mean()),
            "cos_prev_mean": float(cp.mean()) if cp.size else None,
            "cos_next_mean": float(cn.mean()) if cn.size else None,
        }
        summary["per_layer"][str(L)] = rec
        layer_means_cross.append(c.mean())
        layer_means_within.append(w.mean())

    lc = np.array(layer_means_cross)
    lw = np.array(layer_means_within)
    summary["aggregate"] = {
        "mean_precision_cross_over_mlp_layers": float(lc.mean()),
        "mean_precision_within_over_mlp_layers": float(lw.mean()),
        "mean_cross_minus_within": float((lc - lw).mean()),
        "frac_layers_cross_ge_0.80": float((lc >= 0.80).sum() / lc.size),
        "frac_layers_cross_ge_0.55": float((lc >= 0.55).sum() / lc.size),
        "min_layer_cross_precision": float(lc.min()),
        "max_layer_cross_precision": float(lc.max()),
    }

    # Verdict per Stage 5/6 amended thresholds.
    cross_mean = summary["aggregate"]["mean_precision_cross_over_mlp_layers"]
    cross_minus_within = summary["aggregate"]["mean_cross_minus_within"]
    if cross_mean >= 0.80 and cross_minus_within >= -0.10:
        verdict = "GO"
    elif cross_mean < 0.55 or cross_minus_within < -0.20:
        verdict = "NO-GO"
    else:
        verdict = "AMBIGUOUS"
    summary["verdict"] = verdict

    out_json = args.output_dir / "cross_layer_sim_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        layers = sorted(int(L) for L in summary["per_layer"])
        cross = [summary["per_layer"][str(L)]["precision_cross_mean"] for L in layers]
        within = [summary["per_layer"][str(L)]["precision_within_mean"] for L in layers]
        cos_prev = [summary["per_layer"][str(L)]["cos_prev_mean"] for L in layers]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(layers, cross, marker="o", label="cross-layer (h_{L-1} -> W_gate[L])")
        ax1.plot(layers, within, marker="s", label="within-layer (h_L -> W_gate[L])")
        ax1.axhline(0.80, color="green", linestyle="--", alpha=0.6, label="GO 0.80")
        ax1.axhline(0.55, color="red", linestyle="--", alpha=0.6, label="NO-GO 0.55")
        ax1.axhline(args.k_frac, color="gray", linestyle=":", alpha=0.6,
                    label=f"random ({args.k_frac:.2f})")
        ax1.set_ylabel("precision @ 30%")
        ax1.set_title(f"{args.model_id} cross-layer row predictor (K=30%)")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_ylim(0, 1)

        ax2.plot(layers, cos_prev, marker="^", color="purple",
                 label="cos(h_{L-1}, h_L)")
        ax2.set_ylabel("cosine sim")
        ax2.set_xlabel("layer L")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        out_png = args.output_dir / "cross_layer_sim_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Summary.md
    md_lines = [
        f"# Cross-layer activation similarity probe — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Prompts: {n_done}/{len(prompts)} processed, max_length={args.max_length}",
        f"K = {K} ({100*args.k_frac:.0f}% of r={r}), model n_layers={n_layers}",
        "",
        f"## Aggregate verdict: **{verdict}**",
        "",
        f"- Mean precision@30% (cross-layer): "
        f"{summary['aggregate']['mean_precision_cross_over_mlp_layers']:.3f}",
        f"- Mean precision@30% (within-layer): "
        f"{summary['aggregate']['mean_precision_within_over_mlp_layers']:.3f}",
        f"- cross − within (mean): "
        f"{summary['aggregate']['mean_cross_minus_within']:+.3f}",
        f"- Layers passing cross ≥ 0.80: "
        f"{summary['aggregate']['frac_layers_cross_ge_0.80']:.0%}",
        f"- Layers passing cross ≥ 0.55: "
        f"{summary['aggregate']['frac_layers_cross_ge_0.55']:.0%}",
        f"- min/max cross precision across layers: "
        f"{summary['aggregate']['min_layer_cross_precision']:.3f} / "
        f"{summary['aggregate']['max_layer_cross_precision']:.3f}",
        "",
        "## Per-layer table",
        "",
        "| L | cross | within | Δ | cos(L-1,L) | cos(L,L+1) |",
        "|---|---|---|---|---|---|",
    ]
    for L in sorted(int(L) for L in summary["per_layer"]):
        rec = summary["per_layer"][str(L)]
        cn = rec["cos_next_mean"]
        cn_str = f"{cn:.3f}" if cn is not None else "—"
        cp = rec["cos_prev_mean"]
        cp_str = f"{cp:.3f}" if cp is not None else "—"
        md_lines.append(
            f"| {L} | {rec['precision_cross_mean']:.3f} "
            f"| {rec['precision_within_mean']:.3f} "
            f"| {rec['cross_minus_within_mean']:+.3f} "
            f"| {cp_str} | {cn_str} |"
        )
    md_lines += [
        "",
        "## Reference",
        "",
        f"- Deja Vu (Liu et al., ICML 2023, arXiv:2310.17157): trained "
        f"cross-layer MLP predictor on ReLU/OPT achieves 93-99% precision. "
        f"This experiment tests the SwiGLU + zero-overhead dot-product analog.",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nVERDICT: {verdict}  "
          f"(cross_mean={cross_mean:.3f}  cross-within={cross_minus_within:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
