r"""SDZC gate-variance histogram — Track A R1 selected experiment.

For each MLP layer in Qwen3-1.7B-Base, measure per-neuron statistics of
silu(W_gate · x) across a calibration corpus:

  - mean: E[silu(W_gate[i]·x)]
  - std: Std[silu(W_gate[i]·x)]
  - CV: std / |mean|  (relative variability)
  - max: max |silu(W_gate[i]·x)|

Per Track A R1 S3: SDZC posits that a substantial fraction of SwiGLU
neurons have *near-constant* gate output across the calibration
distribution. For those neurons, silu(W_gate[i]·x) ≈ c is a constant,
so the gate's contribution can be folded into W_up by rescaling
(W_up[i,:] ← c · W_up[i,:]) and W_gate row i can be eliminated from
the stored model.

This experiment does NOT do the folding — it measures the variance
distribution. The fold itself comes later if the histogram supports it.

Go threshold: ≥15% of neurons across Qwen3-1.7B's 28 layers have
              std < 0.1 AND |mean| < 1.0 (i.e., near-constant near-zero
              or near-constant moderate-magnitude — both are foldable).

No-go: <5% of neurons qualify at any threshold ≤ 0.2 std.

References:
  - SCAP (arXiv:2412.07174): notes GLU activations are "centered"
  - R1 finding (Round 1): W_up dominates SwiGLU firing-rank
  - R3 finding (Round 3): W_gate is full-rank (linear-algebra) — but
    full rank doesn't preclude near-constant *function output* on the
    actual data manifold
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


CALIBRATION_PASSAGES = [
    "The art of celestial navigation rests on a small handful of "
    "instruments and a great deal of patience. A sextant measures the "
    "angle between a celestial body and the horizon; a chronometer "
    "marks Greenwich Mean Time; a nautical almanac gives the precise "
    "position of the sun, moon, and major stars at every hour of every "
    "day. With these in hand, a navigator can determine latitude from a "
    "noon sun sight, and longitude from a timed observation.",

    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x "
    "for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in "
    "arr[1:] if x >= arr[0]]). The Python implementation is concise but "
    "not in-place; for in-place quicksort one uses partition with two "
    "pointers and a pivot index, recursing on the two halves.",

    "Ocean acidification is the steady decrease in seawater pH caused "
    "by the ocean's absorption of atmospheric carbon dioxide. Since the "
    "start of the industrial era, the average surface pH of the ocean "
    "has dropped by roughly 0.1, a 26 percent increase in hydrogen-ion "
    "concentration. The effects on calcifying organisms are documented.",

    "Bayesian inference treats unknown parameters as random variables "
    "with prior distributions that are updated by observed data via the "
    "likelihood function. The posterior is proportional to the prior "
    "times the likelihood, with the marginal likelihood serving as a "
    "normalization constant.",

    "Glass-blowing originated in the Sidonian region in the first "
    "century BCE and spread rapidly through the Roman Empire. The "
    "technique relies on a blowpipe through which the glassblower "
    "forces breath into a gather of molten glass.",

    "Lithography is the process of printing from a flat surface treated "
    "to repel ink except where it is required. Modern photolithography "
    "uses similar principles but at the nanometer scale to manufacture "
    "integrated circuits.",

    "Sourdough fermentation depends on a stable culture of wild yeasts "
    "and lactobacilli sustained in a flour-water medium. Successful "
    "starters exhibit characteristic acidity, gas production, and a "
    "fruity aroma from ester compounds.",

    "When trained on the right corpus, recurrent networks of moderate "
    "width can solve sequence modeling tasks that were once thought to "
    "require much larger architectures. The Penn Treebank language "
    "model benchmark remained a useful comparison point.",

    "Lambda calculus: (\\x. x x) (\\x. x x) is the omega combinator and "
    "does not normalize. Church encoding represents booleans, numbers, "
    "and pairs as functions, demonstrating that pure lambda calculus is "
    "Turing-complete with no primitive data types required.",

    "The Battle of Lepanto, fought in 1571, was a naval engagement "
    "between the fleets of the Holy League and the Ottoman Empire in "
    "the Gulf of Patras. The Holy League's victory ended Ottoman naval "
    "dominance in the central Mediterranean.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/stage0/ladder/track_A_arch/round1_sdzc"))
    return p.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    r = int(model.config.intermediate_size)
    print(f"  n_layers={n_layers}  d={d}  intermediate={r}")

    # Hooks: capture input to each MLP block (the residual stream input
    # = input to W_gate via gate_proj's pre-hook).
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def make_prehook(idx: int):
        def hook(_mod, inputs):
            captured[idx].append(inputs[0].detach().to(torch.float32).cpu())
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(
            layer.mlp.register_forward_pre_hook(make_prehook(i))
        )

    print(f"Running forward passes on {len(CALIBRATION_PASSAGES)} passages ...")
    n_tokens_total = 0
    for passage in CALIBRATION_PASSAGES:
        encoded = tokenizer(passage, return_tensors="pt",
                            truncation=True, max_length=args.max_length)
        ids = encoded.input_ids.to(device)
        n_tokens_total += int(ids.shape[-1])
        _ = model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()
    print(f"  total tokens: {n_tokens_total}")

    # Compute per-neuron gate statistics.
    print("\nComputing per-neuron gate statistics ...")
    per_layer_stats: dict[int, dict[str, np.ndarray]] = {}
    for layer_idx in range(n_layers):
        # Concatenate all calibration runs for this layer: (T_total, d).
        x_all = torch.cat([t.squeeze(0) for t in captured[layer_idx]], dim=0)
        T_total = x_all.shape[0]
        W_gate = (model.model.layers[layer_idx].mlp.gate_proj
                  .weight.detach().to(torch.float32).cpu())   # (r, d)
        # Compute silu(W_gate · x) for all tokens.
        gate_pre = x_all @ W_gate.T                            # (T_total, r)
        gate_post = F.silu(gate_pre)                           # (T_total, r)

        means = gate_post.mean(dim=0).numpy()                  # (r,)
        stds = gate_post.std(dim=0).numpy()                    # (r,)
        max_abs = gate_post.abs().max(dim=0).values.numpy()    # (r,)
        cv = stds / (np.abs(means) + 1e-6)                     # CV per neuron

        per_layer_stats[layer_idx] = {
            "mean": means,
            "std": stds,
            "max_abs": max_abs,
            "cv": cv,
        }
        # Free this layer's captured buffer.
        captured[layer_idx] = []

    # Aggregate: histograms of std across all layers / neurons.
    all_means = np.concatenate([per_layer_stats[L]["mean"] for L in range(n_layers)])
    all_stds = np.concatenate([per_layer_stats[L]["std"] for L in range(n_layers)])
    all_max = np.concatenate([per_layer_stats[L]["max_abs"] for L in range(n_layers)])
    total_neurons = all_stds.size

    # Foldability classes:
    #  CLASS-DEAD   : std < 0.05 AND |mean| < 0.05 — neuron's gate output
    #                 is near-zero for all tokens; entire neuron contributes
    #                 ~0; can be ELIMINATED (drop W_up row too).
    #  CLASS-CONST  : std < 0.05 AND |mean| ≥ 0.05 — gate is near-constant
    #                 nonzero c; FOLDABLE (W_up[i,:] ← c · W_up[i,:],
    #                 drop W_gate row).
    #  CLASS-LOOSE  : 0.05 ≤ std < 0.1 — borderline; loose threshold.
    #  CLASS-LIVE   : std ≥ 0.1 — gate is genuinely variable; keep as is.

    cls_dead = (all_stds < 0.05) & (np.abs(all_means) < 0.05)
    cls_const = (all_stds < 0.05) & (np.abs(all_means) >= 0.05)
    cls_loose = (all_stds >= 0.05) & (all_stds < 0.1)
    cls_live = all_stds >= 0.1

    frac_dead = cls_dead.sum() / total_neurons
    frac_const = cls_const.sum() / total_neurons
    frac_loose = cls_loose.sum() / total_neurons
    frac_live = cls_live.sum() / total_neurons
    frac_foldable = frac_dead + frac_const  # std < 0.05 total

    # Per-layer breakdown.
    per_layer_summary: list[dict[str, Any]] = []
    for L in range(n_layers):
        stds_L = per_layer_stats[L]["std"]
        means_L = per_layer_stats[L]["mean"]
        n_neurons = stds_L.size
        d_L = ((stds_L < 0.05) & (np.abs(means_L) < 0.05)).sum() / n_neurons
        c_L = ((stds_L < 0.05) & (np.abs(means_L) >= 0.05)).sum() / n_neurons
        per_layer_summary.append({
            "layer": L,
            "frac_dead_neurons": float(d_L),
            "frac_const_neurons": float(c_L),
            "frac_foldable": float(d_L + c_L),
            "frac_live": float((stds_L >= 0.1).sum() / n_neurons),
            "mean_std": float(stds_L.mean()),
            "median_std": float(np.median(stds_L)),
            "p90_std": float(np.quantile(stds_L, 0.9)),
        })

    # Verdict.
    if frac_foldable >= 0.15:
        verdict = "GO (≥15% neurons foldable)"
    elif frac_foldable < 0.05:
        verdict = "NO-GO (<5% neurons foldable)"
    else:
        verdict = f"MID ({frac_foldable*100:.1f}% foldable; layer-adaptive review needed)"

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_layers": n_layers,
        "d_hidden": d,
        "intermediate_size": r,
        "n_tokens_total": n_tokens_total,
        "total_neurons": int(total_neurons),
        "thresholds": {
            "std_dead_const": 0.05,
            "mean_dead": 0.05,
            "std_loose": 0.1,
        },
        "global": {
            "frac_dead_neurons": float(frac_dead),
            "frac_const_neurons": float(frac_const),
            "frac_foldable_total": float(frac_foldable),
            "frac_loose": float(frac_loose),
            "frac_live": float(frac_live),
            "median_std": float(np.median(all_stds)),
            "p10_std": float(np.quantile(all_stds, 0.1)),
            "p90_std": float(np.quantile(all_stds, 0.9)),
        },
        "per_layer": per_layer_summary,
        "verdict": verdict,
        "timestamp": datetime.now(UTC).isoformat(),
        "scap_reference": "arXiv:2412.07174 — GLU activations are 'centered'.",
        "r1_reference": "Round 1 finding: W_up dominates SwiGLU firing-rank.",
        "r3_reference": "Round 3 finding: W_gate is full-rank linear-algebra; SDZC tests function-output near-constancy.",
    }

    out_json = args.output_dir / "sdzc_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram of per-neuron std.
        ax = axes[0]
        ax.hist(all_stds, bins=80, color="steelblue", alpha=0.8)
        ax.axvline(0.05, color="green", linestyle="--",
                   label="dead/const threshold")
        ax.axvline(0.1, color="orange", linestyle="--", label="loose threshold")
        ax.set_yscale("log")
        ax.set_xlabel("per-neuron std of silu(W_gate·x)")
        ax.set_ylabel("count (log)")
        ax.set_title(f"Gate output std distribution\n"
                     f"(n_neurons={total_neurons}, n_tok={n_tokens_total})")
        ax.legend()

        # Per-layer foldable fraction.
        ax = axes[1]
        layers = [s["layer"] for s in per_layer_summary]
        d_frac = [s["frac_dead_neurons"] for s in per_layer_summary]
        c_frac = [s["frac_const_neurons"] for s in per_layer_summary]
        live_frac = [s["frac_live"] for s in per_layer_summary]
        ax.plot(layers, d_frac, marker="o", label="dead (std<0.05, |μ|<0.05)")
        ax.plot(layers, c_frac, marker="s", label="const (std<0.05, |μ|≥0.05)")
        ax.plot(layers, live_frac, marker="^", color="red", label="live (std≥0.1)")
        ax.axhline(0.15, color="green", linestyle=":",
                   label="GO 15% foldable")
        ax.set_xlabel("layer")
        ax.set_ylabel("fraction of neurons")
        ax.set_title("Foldability by layer depth")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Mean std vs layer depth.
        ax = axes[2]
        med = [s["median_std"] for s in per_layer_summary]
        p90 = [s["p90_std"] for s in per_layer_summary]
        ax.plot(layers, med, marker="o", label="median std")
        ax.plot(layers, p90, marker="s", label="p90 std")
        ax.set_xlabel("layer")
        ax.set_ylabel("std of silu(W_gate·x)")
        ax.set_title("Gate variance by depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = args.output_dir / "sdzc_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Summary.md.
    md = [
        f"# SDZC gate-variance probe — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Calibration: {n_tokens_total} tokens across {len(CALIBRATION_PASSAGES)} passages",
        f"n_layers={n_layers}, d={d}, intermediate={r}, total neurons={total_neurons}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Global statistics (all neurons across all layers)",
        "",
        f"- **Foldable** (std < 0.05): **{frac_foldable*100:.1f}%**",
        f"  - dead (also |μ|<0.05): {frac_dead*100:.1f}%",
        f"  - const (|μ|≥0.05): {frac_const*100:.1f}%",
        f"- Loose (0.05 ≤ std < 0.1): {frac_loose*100:.1f}%",
        f"- Live (std ≥ 0.1): {frac_live*100:.1f}%",
        f"- median std: {summary['global']['median_std']:.4f}",
        f"- p10 std: {summary['global']['p10_std']:.4f}",
        f"- p90 std: {summary['global']['p90_std']:.4f}",
        "",
        "## Per-layer breakdown",
        "",
        "| L | foldable | dead | const | live | median std |",
        "|---|---|---|---|---|---|",
    ]
    for s in per_layer_summary:
        md.append(
            f"| {s['layer']} | {s['frac_foldable']*100:.1f}% "
            f"| {s['frac_dead_neurons']*100:.1f}% "
            f"| {s['frac_const_neurons']*100:.1f}% "
            f"| {s['frac_live']*100:.1f}% "
            f"| {s['median_std']:.4f} |"
        )
    md += [
        "",
        "## Class definitions",
        "",
        "- **dead**: std < 0.05 AND |mean| < 0.05 — gate output is near-zero on all tokens. The neuron contributes near-nothing; can be ELIMINATED entirely (drop both W_gate row and W_up row).",
        "- **const**: std < 0.05 AND |mean| ≥ 0.05 — gate output is near-constant nonzero c. FOLDABLE: rescale W_up[i,:] ← c · W_up[i,:] and drop W_gate row.",
        "- **loose**: 0.05 ≤ std < 0.1 — gate has small but non-trivial variation. Borderline; might admit polynomial fit per neuron or stay as is.",
        "- **live**: std ≥ 0.1 — gate genuinely depends on input. Keep full SwiGLU path.",
        "",
        "## References",
        "",
        "- SCAP (arXiv:2412.07174): GLU activations are 'centered'.",
        "- Round 1 finding: W_up dominates SwiGLU firing-rank ranking.",
        "- Round 3 finding: W_gate is full-rank in linear-algebra sense; SDZC tests whether the *function output* is near-constant on the actual data manifold (compatible with full rank).",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nVERDICT: {verdict}  "
          f"(foldable={frac_foldable*100:.1f}%, dead={frac_dead*100:.1f}%, "
          f"const={frac_const*100:.1f}%, live={frac_live*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
