r"""First-Principles E2 — MLP-saturation inequality test.

Tests the inequality predicted from gradient-decay/NTK signal-propagation argument:
  r_99(W_gate) ≥ k_residual − ε  with ε ≤ 50

where:
  - r_99(W_gate) is the empirical 99th-percentile rank of W_gate (from SVD spectrum)
  - k_residual is the residual-stream intrinsic dimension at the layer's INPUT,
    measured by PRE-2 (M-Strat / RSIDC)
  - ε is a slack constant due to initialization noise in gradient-inert directions

Theoretical claim: under NTK signal propagation, gradients to W_gate columns aligned
with V_residual^⊥ are zero through h. Initialization fluctuations in those columns
are not corrected by training and persist as near-Gaussian noise. Thus r_99(W_gate)
≈ k_residual + (small ε from initialization tail).

Falsified if: r_99(W_gate) is systematically MUCH lower than k_residual (would mean
W_gate is more compressed than the residual stream feeding it — incompatible with
gradient-decay derivation).

Inputs:
  - PRE-2 RSIDC results JSON (per-layer k_99 values for residual stream)
  - Qwen3 model (compute W_gate r_99 directly from SVD)

Output: per-layer comparison table; plot of (r_99(W_gate), k_residual) pairs;
        verdict on whether the inequality holds within ε ≤ 50.
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
from transformers import AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--rsidc-results",
        type=Path,
        default=Path("experiments/stage0/ladder/m_strat_rsidc/rsidc_results.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/fp_e2_mlp_saturation"),
    )
    p.add_argument("--epsilon-threshold", type=float, default=50.0,
                   help="ε threshold for inequality test (k_residual − r_99 ≤ ε passes)")
    return p.parse_args()


def r_99_of_weight(W: torch.Tensor) -> int:
    """Compute 99th-percentile rank of W (singular values).

    Returns smallest k such that cumulative sum of squared singular values
    reaches 99% of total Frobenius norm squared.
    """
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total == 0:
        return -1
    k = int(np.searchsorted(cum, 0.99 * total) + 1)
    return k


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.rsidc_results.exists():
        print(f"ERROR: RSIDC results not found at {args.rsidc_results}")
        print("Run m_strat_rsidc.py first.")
        return 1

    print(f"Loading RSIDC results from {args.rsidc_results} ...")
    rsidc = json.loads(args.rsidc_results.read_text())
    rsidc_per_layer = rsidc["per_layer"]  # {str(layer_idx): {...}}

    print(f"Loading {args.model_id} ...")
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    cfg = model.config
    d = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    print(f"  d={d}  n_layers={n_layers}")

    # Probe the same layers that RSIDC measured
    probe_layer_strs = sorted(rsidc_per_layer.keys(), key=int)
    probe_layers = [int(s) for s in probe_layer_strs]
    print(f"  probing layers: {probe_layers}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "rsidc_source": str(args.rsidc_results),
        "probe_layers": probe_layers,
        "epsilon_threshold": args.epsilon_threshold,
        "per_layer": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    print(f"\nMLP-Saturation Inequality test:")
    print(f"  layer | r_99(W_gate) | r_99(W_up) | k_residual | gap_gate | gap_up | passes?")
    print(f"  ------+--------------+------------+------------+----------+--------+--------")

    pass_count = 0
    fail_count = 0
    for ℓ in probe_layers:
        layer = model.model.layers[ℓ]
        W_gate = layer.mlp.gate_proj.weight.detach()  # [intermediate, d_model]
        W_up = layer.mlp.up_proj.weight.detach()       # [intermediate, d_model]

        # The relevant rank is along the INPUT dimension (right singular subspace),
        # which has dim d_model. SVD gives min(rows, cols) singular values; we want the
        # rank of the right factor. For a [r, d_model] matrix with r > d_model
        # (intermediate > d_model in Qwen3), all 2048 singular values are non-trivial.
        # r_99 is computed on these.
        r99_gate = r_99_of_weight(W_gate)
        r99_up = r_99_of_weight(W_up)

        k_residual = rsidc_per_layer[str(ℓ)]["k_99"]

        gap_gate = k_residual - r99_gate
        gap_up = k_residual - r99_up

        # Inequality passes if r_99(W_gate) ≥ k_residual − ε
        # i.e. k_residual − r_99 ≤ ε
        gate_passes = abs(gap_gate) <= args.epsilon_threshold
        up_passes = abs(gap_up) <= args.epsilon_threshold

        if gate_passes:
            pass_count += 1
        else:
            fail_count += 1

        rec = {
            "layer": int(ℓ),
            "r_99_W_gate": int(r99_gate),
            "r_99_W_up": int(r99_up),
            "k_residual": int(k_residual),
            "gap_gate": int(gap_gate),
            "gap_up": int(gap_up),
            "gate_passes_inequality": gate_passes,
            "up_passes_inequality": up_passes,
        }
        results["per_layer"][str(ℓ)] = rec
        passes_str = "GATE✓" if gate_passes else "GATE✗"
        passes_str += " UP✓" if up_passes else " UP✗"
        print(f"  {ℓ:5d} | {r99_gate:12d} | {r99_up:10d} | {k_residual:10d} | "
              f"{gap_gate:+8d} | {gap_up:+6d} | {passes_str}")

    n_layers_tested = pass_count + fail_count
    pass_rate = pass_count / n_layers_tested if n_layers_tested else 0.0
    print(f"\nGate inequality pass rate: {pass_count}/{n_layers_tested} = {pass_rate:.1%}")

    # Verdict
    if pass_rate >= 0.7:
        verdict = (f"GO — MLP-saturation inequality holds in {pass_rate:.0%} of layers "
                   f"with ε ≤ {args.epsilon_threshold}; gradient-decay derivation supported")
    elif pass_rate >= 0.4:
        verdict = (f"PARTIAL — inequality holds in {pass_rate:.0%} of layers; "
                   f"middle layers may pass while extremes fail; investigate")
    else:
        verdict = (f"NO-GO — inequality fails in {1-pass_rate:.0%} of layers; "
                   f"gradient-decay derivation does not predict observed CF8 pattern")

    results["aggregate"] = {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
    }
    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    out_json = args.output_dir / "fp_e2_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = probe_layers
        r99_gate_vals = [results["per_layer"][str(ℓ)]["r_99_W_gate"] for ℓ in layers]
        r99_up_vals = [results["per_layer"][str(ℓ)]["r_99_W_up"] for ℓ in layers]
        k_res_vals = [results["per_layer"][str(ℓ)]["k_residual"] for ℓ in layers]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(layers, k_res_vals, "k-", label="k_residual (RSIDC)", linewidth=2)
        ax.plot(layers, r99_gate_vals, "b-o", label="r_99(W_gate)")
        ax.plot(layers, r99_up_vals, "r-s", label="r_99(W_up)")
        ax.axhline(d, color="gray", linestyle=":", alpha=0.5, label=f"d_model={d}")
        ax.set_xlabel("layer")
        ax.set_ylabel("rank")
        ax.set_title(f"MLP-Saturation Inequality: r_99(W) vs k_residual on {args.model_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = args.output_dir / "fp_e2_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
