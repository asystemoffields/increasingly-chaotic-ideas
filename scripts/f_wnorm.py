r"""F-WNORM — W_down weight SVD spectrum across Qwen3-1.7B-Base.

Test whether W_down has lower rank than W_gate / W_up.

CF8 established that W_gate and W_up are essentially full-rank in trained Qwen3-1.7B
(PPL doubles at 50% rank truncation). F-WNORM closes CF8 to all 3 MLP matrices
by measuring W_down's spectrum directly.

W_down shape: (d_model, d_ffn) = (2048, 6144). Min rank = 2048.

Method: per layer, SVD W_down, report var@K for K ∈ {64, 128, 256, 512, 1024, 2048}.

GO threshold: 99% variance at K <= 256 in >= 50% of layers (real compression viable).
SOFT-GO: 99% variance at K <= 512 (4× compression viable on W_down weight footprint).
NO-GO: 99% variance requires K > 1024 (W_down is essentially full-rank like W_gate/W_up).

Per Stage 6 amendment (run 004 F-WNORM verdict and run 003 F4-WAOS): the published
landscape (ARA arXiv:2510.19389, SVD-LLM V2 NAACL 2025) reports W_V resists SVD
on Llama and Qwen3-8B. W_down is structurally different (d_ffn -> d_model shape,
not d_model -> d_model). Empirical test settles whether the resistance generalizes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

REPO = Path(r"C:\Users\power\documents\increasingly-chaotic-ideas")
OUT_JSON = REPO / "experiments" / "stage0" / "ladder_v2" / "round1_wnorm" / "f_wnorm_results.json"
OUT_MD = REPO / "sonnet_ladder_v2" / "cheap_tests" / "run_001" / "f_wnorm_result.md"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("F-WNORM: W_down SVD spectrum across Qwen3-1.7B-Base", flush=True)
    print(f"[{time.time()-t0:.1f}s] Loading model bf16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
    )
    print(f"[{time.time()-t0:.1f}s] Loaded.", flush=True)

    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}  d_ffn={d_ffn}  d_model={d_model}", flush=True)
    print(f"  W_down shape: ({d_model}, {d_ffn})", flush=True)

    K_VALUES = [32, 64, 128, 256, 512, 1024, 1536, 2048]

    per_layer = []
    for L in range(n_layers):
        W_down = model.model.layers[L].mlp.down_proj.weight.data.to(torch.float32).numpy()
        # Shape (d_model, d_ffn). SVD: U (d_model, d_model), S (d_model,), Vh (d_model, d_ffn)
        # Min dim is d_model = 2048.
        s = np.linalg.svd(W_down, compute_uv=False)
        total_energy = (s * s).sum()
        cum_var = np.cumsum(s * s) / total_energy
        var_at_K = {}
        for K in K_VALUES:
            if K <= len(s):
                var_at_K[K] = float(cum_var[K - 1])
            else:
                var_at_K[K] = 1.0
        # Also: rank at 99% variance
        r_99 = int(np.argmax(cum_var >= 0.99)) + 1 if (cum_var >= 0.99).any() else len(s)
        per_layer.append({
            "layer": L,
            "var_at_K": {str(K): v for K, v in var_at_K.items()},
            "r_99": r_99,
            "r_99_div_d": r_99 / d_model,
            "max_singular_value": float(s[0]),
            "min_singular_value": float(s[-1]),
            "condition_number": float(s[0] / s[-1]),
        })
        if L % 7 == 0 or L == n_layers - 1:
            print(
                f"  layer {L:2d}: r_99={r_99} (r_99/d={r_99/d_model:.3f}), "
                f"var@128={var_at_K[128]:.4f}, var@256={var_at_K[256]:.4f}, "
                f"var@512={var_at_K[512]:.4f}, var@1024={var_at_K[1024]:.4f}",
                flush=True,
            )

    # Aggregate
    r_99_values = np.array([s["r_99"] for s in per_layer])
    var_at_256 = np.array([s["var_at_K"]["256"] for s in per_layer])
    var_at_512 = np.array([s["var_at_K"]["512"] for s in per_layer])
    var_at_1024 = np.array([s["var_at_K"]["1024"] for s in per_layer])

    print(f"\n[{time.time()-t0:.1f}s] Aggregate", flush=True)
    print(f"  r_99 across layers: median={int(np.median(r_99_values))}, "
          f"min={int(r_99_values.min())}, max={int(r_99_values.max())}, "
          f"mean={r_99_values.mean():.1f}", flush=True)
    print(f"  r_99 / d_model: median={np.median(r_99_values)/d_model:.3f}", flush=True)
    print(f"  var@256 across layers: mean={var_at_256.mean():.4f}", flush=True)
    print(f"  var@512 across layers: mean={var_at_512.mean():.4f}", flush=True)
    print(f"  var@1024 across layers: mean={var_at_1024.mean():.4f}", flush=True)

    layers_99var_at_256 = (var_at_256 >= 0.99).sum()
    layers_99var_at_512 = (var_at_512 >= 0.99).sum()
    layers_99var_at_1024 = (var_at_1024 >= 0.99).sum()

    print(f"  layers reaching 99%var at K=256: {layers_99var_at_256}/{n_layers}", flush=True)
    print(f"  layers reaching 99%var at K=512: {layers_99var_at_512}/{n_layers}", flush=True)
    print(f"  layers reaching 99%var at K=1024: {layers_99var_at_1024}/{n_layers}", flush=True)

    # Verdict
    if layers_99var_at_256 >= n_layers * 0.5:
        verdict = "GO_STRONG"
        verdict_note = (
            f"99% variance at K=256 in {layers_99var_at_256}/{n_layers} layers. "
            f"W_down has substantial low-rank structure. Real MLP compression candidate."
        )
    elif layers_99var_at_512 >= n_layers * 0.5:
        verdict = "GO_SOFT"
        verdict_note = (
            f"99% variance at K=512 in {layers_99var_at_512}/{n_layers} layers. "
            f"W_down has 4x rank reduction headroom. Worth follow-up activation-weighted SVD."
        )
    elif layers_99var_at_1024 >= n_layers * 0.5:
        verdict = "MARGINAL"
        verdict_note = (
            f"99% variance at K=1024 in {layers_99var_at_1024}/{n_layers} layers. "
            f"Only 2x rank reduction; W_down is moderately full-rank. Limited compression value."
        )
    else:
        verdict = "NO_GO"
        verdict_note = (
            f"99% variance requires K > 1024 in {n_layers - layers_99var_at_1024}/{n_layers} layers. "
            f"W_down is essentially full-rank like W_gate/W_up. CF8 closes to all 3 MLP matrices."
        )

    print(f"\n[{time.time()-t0:.1f}s] === VERDICT: {verdict} ===", flush=True)
    print(f"  {verdict_note}", flush=True)

    out = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "d_model": d_model,
        "K_values_tested": K_VALUES,
        "aggregate": {
            "r_99_median": float(np.median(r_99_values)),
            "r_99_div_d_median": float(np.median(r_99_values) / d_model),
            "var_at_256_mean": float(var_at_256.mean()),
            "var_at_512_mean": float(var_at_512.mean()),
            "var_at_1024_mean": float(var_at_1024.mean()),
            "layers_99var_at_256": int(layers_99var_at_256),
            "layers_99var_at_512": int(layers_99var_at_512),
            "layers_99var_at_1024": int(layers_99var_at_1024),
        },
        "per_layer": per_layer,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}", flush=True)

    # Markdown
    md = []
    md.append("# F-WNORM — W_down Weight SVD Spectrum on Qwen3-1.7B-Base")
    md.append("")
    md.append("Date: 2026-05-09")
    md.append(f"Model: Qwen3-1.7B-Base, {n_layers} layers, W_down shape ({d_model}, {d_ffn})")
    md.append(f"")
    md.append(f"## Verdict: {verdict}")
    md.append("")
    md.append(verdict_note)
    md.append("")
    md.append("## Aggregate")
    md.append("")
    md.append(f"| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| r_99 median | {int(np.median(r_99_values))} |")
    md.append(f"| r_99 / d_model median | {np.median(r_99_values)/d_model:.3f} |")
    md.append(f"| var@256 mean | {var_at_256.mean():.4f} |")
    md.append(f"| var@512 mean | {var_at_512.mean():.4f} |")
    md.append(f"| var@1024 mean | {var_at_1024.mean():.4f} |")
    md.append(f"| Layers reaching 99% var at K=256 | {layers_99var_at_256}/{n_layers} |")
    md.append(f"| Layers reaching 99% var at K=512 | {layers_99var_at_512}/{n_layers} |")
    md.append(f"| Layers reaching 99% var at K=1024 | {layers_99var_at_1024}/{n_layers} |")
    md.append("")
    md.append("## Per-layer r_99 / d_model")
    md.append("")
    md.append("| Layer | r_99 | r_99/d | var@128 | var@256 | var@512 | var@1024 |")
    md.append("|---|---|---|---|---|---|---|")
    for s in per_layer:
        md.append(
            f"| L{s['layer']:02d} | {s['r_99']} | {s['r_99_div_d']:.3f} | "
            f"{s['var_at_K']['128']:.4f} | {s['var_at_K']['256']:.4f} | "
            f"{s['var_at_K']['512']:.4f} | {s['var_at_K']['1024']:.4f} |"
        )
    md.append("")
    md.append("## Files")
    md.append("- `scripts/f_wnorm.py`")
    md.append("- `experiments/stage0/ladder_v2/round1_wnorm/f_wnorm_results.json`")
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
