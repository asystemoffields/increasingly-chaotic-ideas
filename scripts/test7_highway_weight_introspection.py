r"""Test 7 — Highway channel weight introspection.

Question: are the persistent highway channels [1999, 1793] (RSIDC v2 finding) special
in the WEIGHTS, not just the activations?

For each layer, examine W_gate, W_up, W_down, W_q, W_k, W_v, W_o:
  - Norms of rows/columns indexed by 1999 and 1793 vs the median row/column norm
  - Are these channels disproportionately "loud" in the weights?

For lm_head and embed_tokens (tied):
  - Norm of column 1999 vs column 1793 vs median column norm
  - Cosine similarity of column 1999 with itself across the two roles (input embed vs output projection — same matrix in tied config but worth verifying)

If weights ALSO show extreme structure on these channels, then the model has dedicated
specialized weights for the highway — strengthens the "communication channel"
interpretation. If weights are normal but activations are extreme, then the highway is
purely an activation-level emergent phenomenon (consistent with attention sink dynamics).

Either outcome is informative.
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


HIGHWAY_CHANNELS_DEFAULT = [1999, 1793]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--highway-channels", default="1999,1793",
                   help="Comma-separated highway channel indices to inspect")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/test7_highway_weights"),
    )
    return p.parse_args()


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Return the percentile rank of value within distribution (0-100)."""
    return float((distribution < value).sum() / len(distribution) * 100)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    highway = [int(x) for x in args.highway_channels.split(",")]

    print(f"Loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device).eval()
    cfg = model.config
    d = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    print(f"  d={d}  n_layers={n_layers}")
    print(f"  highway channels: {highway}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "highway_channels": highway,
        "d_model": d,
        "n_layers": n_layers,
        "per_layer": {},
        "embed_lm_head": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # The relevant weight axes per matrix:
    # W_gate, W_up: shape [intermediate, d_model] -- COLUMNS index residual stream (input)
    # W_down: shape [d_model, intermediate] -- ROWS index residual stream (output)
    # W_q: shape [d_q, d_model] -- COLUMNS index residual stream (input)
    # W_k, W_v: shape [d_kv, d_model] -- COLUMNS index residual stream (input)
    # W_o: shape [d_model, d_q] -- ROWS index residual stream (output)
    # embed_tokens / lm_head (tied): shape [V, d_model] -- COLUMNS index residual stream

    print("\nPer-layer highway channel weight analysis:")
    print(f"  layer | W_gate col[hwy]  | W_up col[hwy]  | W_down row[hwy] | W_q col[hwy]  | W_o row[hwy]")
    print(f"  ------+------------------+----------------+-----------------+----------------+--------------")

    for ℓ in range(n_layers):
        layer = model.model.layers[ℓ]

        Wg = layer.mlp.gate_proj.weight.detach().to(torch.float32)
        Wu = layer.mlp.up_proj.weight.detach().to(torch.float32)
        Wd = layer.mlp.down_proj.weight.detach().to(torch.float32)
        Wq = layer.self_attn.q_proj.weight.detach().to(torch.float32)
        Wk = layer.self_attn.k_proj.weight.detach().to(torch.float32)
        Wv = layer.self_attn.v_proj.weight.detach().to(torch.float32)
        Wo = layer.self_attn.o_proj.weight.detach().to(torch.float32)

        layer_rec = {}

        # For each input-side matrix (gate, up, q, k, v): inspect COLUMN norms at highway indices
        for name, W in [("W_gate", Wg), ("W_up", Wu), ("W_q", Wq), ("W_k", Wk), ("W_v", Wv)]:
            col_norms = torch.linalg.norm(W, dim=0).cpu().numpy()  # [d_model] columns
            median_col = float(np.median(col_norms))
            hwy_norms = [float(col_norms[h]) for h in highway]
            hwy_pct = [percentile_rank(c, col_norms) for c in hwy_norms]
            ratios = [c / median_col for c in hwy_norms]
            layer_rec[name] = {
                "median_col_norm": median_col,
                "highway_col_norms": hwy_norms,
                "highway_pct_rank": hwy_pct,
                "ratio_to_median": ratios,
            }

        # For output-side matrices (down, o): inspect ROW norms at highway indices
        for name, W in [("W_down", Wd), ("W_o", Wo)]:
            row_norms = torch.linalg.norm(W, dim=1).cpu().numpy()  # [d_model] rows
            median_row = float(np.median(row_norms))
            hwy_norms = [float(row_norms[h]) for h in highway]
            hwy_pct = [percentile_rank(c, row_norms) for c in hwy_norms]
            ratios = [c / median_row for c in hwy_norms]
            layer_rec[name] = {
                "median_row_norm": median_row,
                "highway_row_norms": hwy_norms,
                "highway_pct_rank": hwy_pct,
                "ratio_to_median": ratios,
            }

        results["per_layer"][str(ℓ)] = layer_rec

        # Compact per-layer summary line
        gh = layer_rec["W_gate"]["ratio_to_median"]
        uh = layer_rec["W_up"]["ratio_to_median"]
        dh = layer_rec["W_down"]["ratio_to_median"]
        qh = layer_rec["W_q"]["ratio_to_median"]
        oh = layer_rec["W_o"]["ratio_to_median"]
        print(f"  {ℓ:5d} | {gh[0]:.2f}/{gh[1]:.2f}× | {uh[0]:.2f}/{uh[1]:.2f}× | "
              f"{dh[0]:.2f}/{dh[1]:.2f}× | {qh[0]:.2f}/{qh[1]:.2f}× | {oh[0]:.2f}/{oh[1]:.2f}×")

    # Embedding/lm_head
    print("\nEmbedding / lm_head highway analysis:")
    W_E = model.lm_head.weight.detach().to(torch.float32)  # [V, d_model]
    V_size, d_check = W_E.shape
    col_norms_E = torch.linalg.norm(W_E, dim=0).cpu().numpy()
    median_col_E = float(np.median(col_norms_E))
    print(f"  lm_head shape: ({V_size}, {d_check})  median col norm: {median_col_E:.4f}")
    for h in highway:
        col_n = float(col_norms_E[h])
        pct = percentile_rank(col_n, col_norms_E)
        ratio = col_n / median_col_E
        print(f"    column {h}: norm={col_n:.4f}  pctile={pct:.1f}  ratio_to_median={ratio:.2f}×")
        results["embed_lm_head"][f"col_{h}"] = {
            "norm": col_n,
            "percentile_rank": pct,
            "ratio_to_median": ratio,
        }
    results["embed_lm_head"]["median_col_norm"] = median_col_E
    results["embed_lm_head"]["max_col_norm"] = float(col_norms_E.max())
    results["embed_lm_head"]["min_col_norm"] = float(col_norms_E.min())

    # Aggregate stats across layers
    print("\nAggregate (median/mean across all layers):")
    aggregate = {}
    for matrix_name in ["W_gate", "W_up", "W_down", "W_q", "W_k", "W_v", "W_o"]:
        ratios_h0 = []
        ratios_h1 = []
        pct_h0 = []
        pct_h1 = []
        for ℓ in range(n_layers):
            rec = results["per_layer"][str(ℓ)][matrix_name]
            ratios_h0.append(rec["ratio_to_median"][0])
            ratios_h1.append(rec["ratio_to_median"][1])
            pct_h0.append(rec["highway_pct_rank"][0])
            pct_h1.append(rec["highway_pct_rank"][1])
        aggregate[matrix_name] = {
            f"median_ratio_chan_{highway[0]}": float(np.median(ratios_h0)),
            f"median_ratio_chan_{highway[1]}": float(np.median(ratios_h1)),
            f"median_pctile_chan_{highway[0]}": float(np.median(pct_h0)),
            f"median_pctile_chan_{highway[1]}": float(np.median(pct_h1)),
        }
        agg = aggregate[matrix_name]
        print(f"  {matrix_name}: chan {highway[0]} ratio median = {agg[f'median_ratio_chan_{highway[0]}']:.2f}× "
              f"(pct {agg[f'median_pctile_chan_{highway[0]}']:.1f}); "
              f"chan {highway[1]} ratio = {agg[f'median_ratio_chan_{highway[1]}']:.2f}× "
              f"(pct {agg[f'median_pctile_chan_{highway[1]}']:.1f})")

    results["aggregate_across_layers"] = aggregate

    # Verdict
    # Strong signal if highway channels' weights are >2× median or <0.5× median consistently
    extreme_count = 0
    for matrix_name, agg in aggregate.items():
        for h in highway:
            r = agg[f"median_ratio_chan_{h}"]
            if r > 2.0 or r < 0.5:
                extreme_count += 1

    if extreme_count >= 4:
        verdict = (f"STRONG EVIDENCE — highway channels show extreme weight structure in "
                   f"{extreme_count} of {len(aggregate) * len(highway)} matrix×channel slots; "
                   f"the model has dedicated specialized weights for the highway")
    elif extreme_count >= 2:
        verdict = (f"PARTIAL EVIDENCE — {extreme_count}/{len(aggregate)*len(highway)} extreme; "
                   f"some weight specialization but not pervasive")
    else:
        verdict = (f"WEAK EVIDENCE — only {extreme_count}/{len(aggregate)*len(highway)} extreme; "
                   f"highway is primarily an ACTIVATION phenomenon, not a weight-level structure. "
                   f"Consistent with emergent attention sink dynamics from training, not "
                   f"hardcoded specialization in weights.")

    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    out_json = args.output_dir / "test7_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
