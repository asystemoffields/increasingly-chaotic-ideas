r"""F6-WALIGN — per-neuron W_gate / W_up row cosine-alignment test on Qwen3-1.7B-Base.

Test: for each neuron i in each MLP layer, compute |cos(W_gate[i,:], W_up[i,:])|.
If sufficient fraction of neurons have |cos| ≈ 1, those neurons admit an exact
algebraic fold: silu(W_gate·x) ⊙ (W_up·x) collapses for aligned rows because
W_gate[i,:] and W_up[i,:] differ only by a sign+scalar.

Per Stage 6 amendment (run 006): separate the |cos| >= 0.99 bin (near-exact fold,
publishable as compression) from the |cos| >= 0.90 bin (approximate, has 44%
row-residual energy and would require ΔNLL characterization).

Output: per-layer histogram of |cos|, fraction of neurons in each threshold bin,
combined-across-layers histogram, and verdict.

GO threshold (publishable compression): >= 5% of neurons across the model have
|cos| >= 0.99.

NO-GO threshold: < 1% of neurons at |cos| >= 0.99 AND < 5% at |cos| >= 0.90.

Random baseline: per Stage 6 amendment, also report baseline using random unit
vectors of same dimension (compare against expected |cos| distribution for
random pairs in d=11008).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

REPO = Path(r"C:\Users\power\documents\increasingly-chaotic-ideas")
OUT_JSON = REPO / "experiments" / "stage0" / "ladder_v2" / "round1_walign" / "f6_walign_results.json"
OUT_MD = REPO / "sonnet_ladder_v2" / "cheap_tests" / "run_001" / "f6_walign_result.md"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("F6-WALIGN: per-neuron W_gate/W_up cosine alignment test", flush=True)
    print(f"[{time.time()-t0:.1f}s] Loading Qwen3-1.7B-Base bf16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
    )
    print(f"[{time.time()-t0:.1f}s] Model loaded.", flush=True)

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}  d_ffn={d_ffn}  d_model={d_model}", flush=True)

    # Per-layer cosine vectors
    per_layer = []
    all_abscos = []
    for L in range(n_layers):
        layer = model.model.layers[L].mlp
        W_gate = layer.gate_proj.weight.data.to(torch.float32)  # (d_ffn, d_model)
        W_up = layer.up_proj.weight.data.to(torch.float32)      # (d_ffn, d_model)
        # Per-row cosine: dot(g, u) / (||g|| ||u||)
        g_norm = W_gate.norm(dim=1).clamp(min=1e-9)
        u_norm = W_up.norm(dim=1).clamp(min=1e-9)
        dot = (W_gate * W_up).sum(dim=1)
        cos = dot / (g_norm * u_norm)
        abscos = cos.abs().numpy()
        per_layer.append({
            "layer": L,
            "n_neurons": int(d_ffn),
            "frac_abscos_ge_0_99": float((abscos >= 0.99).mean()),
            "frac_abscos_ge_0_95": float((abscos >= 0.95).mean()),
            "frac_abscos_ge_0_90": float((abscos >= 0.90).mean()),
            "frac_abscos_ge_0_80": float((abscos >= 0.80).mean()),
            "frac_abscos_ge_0_50": float((abscos >= 0.50).mean()),
            "median_abscos": float(np.median(abscos)),
            "mean_abscos": float(abscos.mean()),
            "max_abscos": float(abscos.max()),
        })
        all_abscos.append(abscos)
        if L % 7 == 0 or L == n_layers - 1:
            print(
                f"  layer {L:2d}: median |cos|={np.median(abscos):.3f}, "
                f"max={abscos.max():.4f}, "
                f">=0.99: {(abscos >= 0.99).mean()*100:.2f}%, "
                f">=0.90: {(abscos >= 0.90).mean()*100:.2f}%",
                flush=True,
            )

    all_abscos_flat = np.concatenate(all_abscos)
    n_total = all_abscos_flat.size
    print(f"\n[{time.time()-t0:.1f}s] All-layer combined statistics", flush=True)
    print(f"  total neurons measured: {n_total}", flush=True)
    print(f"  median |cos|: {np.median(all_abscos_flat):.4f}", flush=True)
    print(f"  mean |cos|: {all_abscos_flat.mean():.4f}", flush=True)
    print(f"  max |cos|: {all_abscos_flat.max():.4f}", flush=True)
    print(f"  fraction |cos| >= 0.99: {(all_abscos_flat >= 0.99).mean()*100:.4f}% ({int((all_abscos_flat >= 0.99).sum())} / {n_total})", flush=True)
    print(f"  fraction |cos| >= 0.95: {(all_abscos_flat >= 0.95).mean()*100:.4f}%", flush=True)
    print(f"  fraction |cos| >= 0.90: {(all_abscos_flat >= 0.90).mean()*100:.4f}%", flush=True)
    print(f"  fraction |cos| >= 0.80: {(all_abscos_flat >= 0.80).mean()*100:.4f}%", flush=True)
    print(f"  fraction |cos| >= 0.50: {(all_abscos_flat >= 0.50).mean()*100:.4f}%", flush=True)

    # Random baseline: in d_model=2048, expected |cos| of random unit vectors
    # is ~ sqrt(2 / (pi * d)) ≈ 0.025 for d=2048. Generate samples for comparison.
    print(f"\n[{time.time()-t0:.1f}s] Random-baseline comparison (d={d_model})", flush=True)
    rng = np.random.default_rng(20260509)
    n_random = 100_000
    a = rng.standard_normal((n_random, d_model)).astype(np.float32)
    b = rng.standard_normal((n_random, d_model)).astype(np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / a_norm
    b_unit = b / b_norm
    rand_cos = (a_unit * b_unit).sum(axis=1)
    rand_abscos = np.abs(rand_cos)
    print(f"  random pair |cos| median: {np.median(rand_abscos):.4f}", flush=True)
    print(f"  random pair |cos| max: {rand_abscos.max():.4f}", flush=True)
    print(f"  random pair |cos| >= 0.99 fraction: {(rand_abscos >= 0.99).mean()*100:.6f}%", flush=True)
    print(f"  random pair |cos| >= 0.90 fraction: {(rand_abscos >= 0.90).mean()*100:.6f}%", flush=True)

    # Verdict
    frac_99 = (all_abscos_flat >= 0.99).mean()
    frac_95 = (all_abscos_flat >= 0.95).mean()
    frac_90 = (all_abscos_flat >= 0.90).mean()
    rand_frac_90 = (rand_abscos >= 0.90).mean()

    if frac_99 >= 0.05:
        verdict = "GO_EXACT"
        verdict_note = (
            f"{frac_99*100:.2f}% of neurons have |cos| >= 0.99 — exact algebraic fold viable for "
            f"that fraction. Real MLP compression mechanism."
        )
    elif frac_99 >= 0.01:
        verdict = "WEAK_GO_EXACT"
        verdict_note = (
            f"{frac_99*100:.2f}% of neurons at |cos| >= 0.99 — small but nonzero exact-fold fraction. "
            f"Worth follow-up with ΔNLL eval on the fold."
        )
    elif frac_90 >= 0.05 and frac_90 > 10 * rand_frac_90:
        verdict = "GO_APPROX"
        verdict_note = (
            f"{frac_90*100:.2f}% at |cos| >= 0.90 (>>random {rand_frac_90*100:.4f}%) — approximate fold "
            f"detectable but not exact. Would require ΔNLL characterization for compression viability."
        )
    else:
        verdict = "NO_GO"
        verdict_note = (
            f"frac_99={frac_99*100:.4f}%, frac_90={frac_90*100:.4f}%, random_baseline_frac_90="
            f"{rand_frac_90*100:.4f}%. Insufficient alignment for exact-fold compression. "
            f"Per-neuron alignment claim killed."
        )

    print(f"\n[{time.time()-t0:.1f}s] VERDICT: {verdict}", flush=True)
    print(f"  {verdict_note}", flush=True)

    out = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "d_model": d_model,
        "n_total_neurons": n_total,
        "all_layer_stats": {
            "median_abscos": float(np.median(all_abscos_flat)),
            "mean_abscos": float(all_abscos_flat.mean()),
            "max_abscos": float(all_abscos_flat.max()),
            "frac_abscos_ge_0_99": float(frac_99),
            "frac_abscos_ge_0_95": float(frac_95),
            "frac_abscos_ge_0_90": float(frac_90),
            "frac_abscos_ge_0_80": float((all_abscos_flat >= 0.80).mean()),
            "frac_abscos_ge_0_50": float((all_abscos_flat >= 0.50).mean()),
        },
        "per_layer": per_layer,
        "random_baseline": {
            "n_samples": n_random,
            "d_model": d_model,
            "median_abscos": float(np.median(rand_abscos)),
            "max_abscos": float(rand_abscos.max()),
            "frac_ge_0_99": float((rand_abscos >= 0.99).mean()),
            "frac_ge_0_95": float((rand_abscos >= 0.95).mean()),
            "frac_ge_0_90": float((rand_abscos >= 0.90).mean()),
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}", flush=True)

    # Markdown report
    md = []
    md.append("# F6-WALIGN — Per-Neuron W_gate/W_up Row-Cosine Alignment Test")
    md.append("")
    md.append(f"Date: 2026-05-09")
    md.append(f"Model: Qwen3-1.7B-Base (n_layers={n_layers}, d_ffn={d_ffn}, d_model={d_model})")
    md.append(f"Source: ICLR 2025 Yadav noted FP8-training-context alignment; this test measures the same")
    md.append(f"object as a post-training compression primitive (no published variant found).")
    md.append("")
    md.append(f"## Verdict: {verdict}")
    md.append("")
    md.append(f"{verdict_note}")
    md.append("")
    md.append("## Aggregate statistics (all 28 layers, {} total neurons)".format(n_total))
    md.append("")
    md.append(f"- median |cos|: **{np.median(all_abscos_flat):.4f}**")
    md.append(f"- mean |cos|: {all_abscos_flat.mean():.4f}")
    md.append(f"- max |cos|: {all_abscos_flat.max():.4f}")
    md.append("")
    md.append("| Threshold | Fraction (model) | Fraction (random baseline d={}) | Ratio |".format(d_model))
    md.append("|---|---|---|---|")
    for thr in [0.99, 0.95, 0.90, 0.80, 0.50]:
        m = (all_abscos_flat >= thr).mean()
        r = (rand_abscos >= thr).mean()
        ratio = m / max(r, 1e-12)
        md.append(f"| |cos| >= {thr:.2f} | {m*100:.4f}% | {r*100:.6f}% | {ratio:.1f}× |")
    md.append("")
    md.append("## Per-layer fractions (|cos| >= 0.99 / 0.95 / 0.90)")
    md.append("")
    md.append("| Layer | >=0.99 | >=0.95 | >=0.90 | median |")
    md.append("|---|---|---|---|---|")
    for s in per_layer:
        md.append(
            f"| L{s['layer']:02d} | {s['frac_abscos_ge_0_99']*100:.3f}% | "
            f"{s['frac_abscos_ge_0_95']*100:.3f}% | {s['frac_abscos_ge_0_90']*100:.3f}% | "
            f"{s['median_abscos']:.3f} |"
        )
    md.append("")
    md.append("## Compression-viability implication")
    md.append("")
    if frac_99 >= 0.01:
        saved_frac = frac_99 / 2  # each folded pair saves one row out of two
        md.append(
            f"If only the |cos| >= 0.99 fraction ({frac_99*100:.2f}%) folds exactly: ~{saved_frac*100:.2f}% "
            f"of MLP weight storage saved (one of W_gate row + W_up row replaced by W_up row + scalar α)."
        )
        md.append("")
        md.append("Composes additively with weight quantization (Q4_K_M etc.).")
    else:
        md.append("Insufficient |cos| >= 0.99 fraction for exact-fold compression. The mechanism is dead.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `scripts/f6_walign.py`")
    md.append("- `experiments/stage0/ladder_v2/round1_walign/f6_walign_results.json`")
    md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
