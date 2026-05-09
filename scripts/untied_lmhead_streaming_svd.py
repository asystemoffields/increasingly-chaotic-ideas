r"""Untied lm_head streaming SVD on Qwen3-8B (or Llama-3, etc.).

Tests whether CF12 (tied embed/lm_head full-rank) is *configuration-dependent*
or generalizes to untied configs.

Background:
  CF12 in Qwen3-1.7B-Base (tied): joint embed/lm_head matrix has r_99=1992/2048
  (full rank); ΔNLL +19.96 catastrophic at r=1024. The mechanistic explanation:
  in tied configs, gradient flows through both input embedding lookup AND output
  projection during training, keeping every direction load-bearing.

  Godey & Artzi (arXiv:2603.10145, Mar 2026) predict that lm_head suppresses
  95-99% of gradient norm during backprop — a gradient-bottleneck claim that
  implies the trained weight should have concentrated rank IF only the output
  path drives the matrix.

  Qwen3-8B has tie_word_embeddings=False: lm_head is independent of embed_tokens.
  This is the right test for the prediction.

Method:
  1. Stream Qwen3-8B's lm_head weight from safetensors WITHOUT loading the full
     model (saves ~16 GB RAM).
  2. Compute thin SVD on the lm_head matrix (shape ~151936 × 4096).
  3. Plot singular value decay.
  4. Compute r_99, r_95, r_90, var@r∈{64,128,256,512,1024,2048}.
  5. Compare to CF12 tied (r_99=1992/2048) and to MLP weights (r_99 ≈ d).

Predicts (per Godey & Artzi):
  - r_99 << d_model (concentrated spectrum)
  - var@r=256 > 0.85 (heavily concentrated in top components)

Falsifies:
  - r_99 ≈ d_model (extends CF8 to all weight matrices regardless of tie config)
  - var@r=256 < 0.30 (no real concentration)

Output: spectrum diagnostics; CF12 generalization verdict.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-8B",
                   help="HuggingFace model id with untied lm_head")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/untied_lmhead_svd"),
    )
    return p.parse_args()


def stream_lm_head_weight(model_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Stream lm_head.weight from safetensors without loading the full model.

    Returns: (weight_tensor, metadata_dict)
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    metadata = {
        "model_id": model_id,
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
    }
    print(f"  config tie_word_embeddings = {metadata['tie_word_embeddings']}")
    print(f"  vocab_size = {metadata['vocab_size']}")
    print(f"  hidden_size = {metadata['hidden_size']}")

    if metadata["tie_word_embeddings"] is True:
        print(f"  WARNING: {model_id} has tied embeddings; this is a CF12 retest, "
              f"not an untied test. Continuing anyway.")

    # Try to find safetensors index file to locate which shard contains lm_head.weight
    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        lm_head_key = None
        for key in ["lm_head.weight", "model.lm_head.weight", "transformer.lm_head.weight"]:
            if key in weight_map:
                lm_head_key = key
                break
        if lm_head_key is None:
            print(f"  WARNING: could not find lm_head.weight in index; available keys (first 10):")
            for k in list(weight_map.keys())[:10]:
                print(f"    {k}")
            # Try with embed_tokens fallback (tied case)
            for key in ["model.embed_tokens.weight", "embed_tokens.weight"]:
                if key in weight_map:
                    lm_head_key = key
                    print(f"  Falling back to {key} (tied case)")
                    break
        if lm_head_key is None:
            raise ValueError(f"No lm_head or embed_tokens weight found in {model_id}")

        shard = weight_map[lm_head_key]
        shard_path = hf_hub_download(model_id, shard)
        print(f"  found {lm_head_key} in shard {shard}")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            W = f.get_tensor(lm_head_key)
        print(f"  loaded weight shape: {tuple(W.shape)}  dtype: {W.dtype}")
        metadata["lm_head_key"] = lm_head_key
        metadata["shard"] = shard
        return W, metadata
    except Exception as e:
        print(f"  Streaming via index failed: {e}")
        print(f"  Falling back to loading model and extracting lm_head ...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        W = model.lm_head.weight.detach().clone()
        del model
        return W, metadata


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming lm_head from {args.model_id} ...")
    W, metadata = stream_lm_head_weight(args.model_id)
    V, d = W.shape
    print(f"\nlm_head shape: ({V}, {d})")

    # Sanity check on row norms — dead-row confound check
    print("\nPer-row norm distribution:")
    W_fp32 = W.to(torch.float32)
    row_norms = torch.linalg.norm(W_fp32, dim=1).cpu().numpy()
    median_norm = float(np.median(row_norms))
    near_zero_count = int(np.sum(row_norms < 0.01 * median_norm))
    near_zero_frac = near_zero_count / V
    print(f"  median: {median_norm:.4f}")
    print(f"  near-zero (<1% of median): {near_zero_count} ({100*near_zero_frac:.2f}% of {V})")
    print(f"  bottom decile: {float(np.percentile(row_norms, 10)):.4f}")
    print(f"  max: {float(row_norms.max()):.4f}")

    # SVD
    print("\nComputing thin SVD (this may take ~5-15 min for V=151936, d=4096)...")
    t0 = datetime.now(UTC)
    U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
    t1 = datetime.now(UTC)
    print(f"  SVD done in {(t1 - t0).total_seconds():.1f}s")

    S_np = S.cpu().numpy()
    eigvals = S_np ** 2
    cum = np.cumsum(eigvals)
    total = cum[-1]
    cum_frac = cum / total

    spec = {
        "n_singvals": len(S_np),
        "rank_for_99_var": int(np.searchsorted(cum_frac, 0.99) + 1),
        "rank_for_95_var": int(np.searchsorted(cum_frac, 0.95) + 1),
        "rank_for_90_var": int(np.searchsorted(cum_frac, 0.90) + 1),
        "var_at_r_64": float(cum_frac[min(63, len(cum_frac) - 1)]),
        "var_at_r_128": float(cum_frac[min(127, len(cum_frac) - 1)]),
        "var_at_r_256": float(cum_frac[min(255, len(cum_frac) - 1)]),
        "var_at_r_512": float(cum_frac[min(511, len(cum_frac) - 1)]),
        "var_at_r_1024": float(cum_frac[min(1023, len(cum_frac) - 1)]),
        "var_at_r_2048": float(cum_frac[min(2047, len(cum_frac) - 1)]),
        "ratio_s1_s0": float(S_np[1] / S_np[0]) if len(S_np) > 1 else float("nan"),
        "top_50_singvals": S_np[:50].tolist(),
        "rank_for_99_over_d": int(np.searchsorted(cum_frac, 0.99) + 1) / d,
    }

    print(f"\nSpectrum diagnostics:")
    print(f"  rank for 99% variance: {spec['rank_for_99_var']} (of {len(S_np)})")
    print(f"  rank for 95% variance: {spec['rank_for_95_var']}")
    print(f"  rank for 90% variance: {spec['rank_for_90_var']}")
    print(f"  r_99 / d = {spec['rank_for_99_over_d']:.4f}")
    print(f"\n  variance at r=128: {spec['var_at_r_128']:.4f}")
    print(f"  variance at r=256: {spec['var_at_r_256']:.4f}")
    print(f"  variance at r=512: {spec['var_at_r_512']:.4f}")
    print(f"  variance at r=1024: {spec['var_at_r_1024']:.4f}")
    print(f"  variance at r=2048: {spec['var_at_r_2048']:.4f}")

    # Verdict relative to CF12
    # CF12 (tied, Qwen3-1.7B): r_99/d=1992/2048=0.973, var@r=256=0.28
    cf12_r99_over_d = 0.973
    cf12_var_at_r256 = 0.28

    if spec["rank_for_99_over_d"] < 0.5 and spec["var_at_r_256"] > 0.85:
        verdict = (f"GO — UNTIED lm_head shows concentrated spectrum "
                   f"(r_99/d={spec['rank_for_99_over_d']:.3f}, var@r=256={spec['var_at_r_256']:.3f}). "
                   f"Godey & Artzi gradient-bottleneck prediction CONFIRMED in untied config. "
                   f"CF12 is configuration-dependent. LHQD compression line opens for "
                   f"untied models (Qwen3-8B+, Llama-3, GPT-NeoX class).")
    elif spec["rank_for_99_over_d"] < 0.85:
        verdict = (f"PARTIAL — UNTIED lm_head moderately concentrated "
                   f"(r_99/d={spec['rank_for_99_over_d']:.3f} vs CF12's {cf12_r99_over_d}). "
                   f"Some compression viable but not as dramatic as Godey & Artzi predict.")
    else:
        verdict = (f"NO-GO — UNTIED lm_head also full-rank "
                   f"(r_99/d={spec['rank_for_99_over_d']:.3f} ~ CF12's {cf12_r99_over_d}). "
                   f"CF8 generalizes to ALL weight matrices regardless of tie config. "
                   f"Godey & Artzi gradient-bottleneck argument REFUTED at the trained "
                   f"weight level — the bottleneck doesn't translate to weight rank.")

    if near_zero_frac > 0.20:
        verdict += f" CAVEAT: dead-row confound ({100*near_zero_frac:.1f}% of rows near-zero norm)."

    results = {
        "metadata": metadata,
        "spectrum": spec,
        "row_norm_stats": {
            "median": median_norm,
            "near_zero_count": near_zero_count,
            "near_zero_frac": near_zero_frac,
            "bottom_decile": float(np.percentile(row_norms, 10)),
            "max": float(row_norms.max()),
        },
        "comparison_to_cf12": {
            "tied_qwen3_1.7b_r_99_over_d": cf12_r99_over_d,
            "tied_qwen3_1.7b_var_at_r256": cf12_var_at_r256,
        },
        "verdict": verdict,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_json = args.output_dir / "untied_lmhead_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Singular value decay
        ax = axes[0]
        ax.semilogy(range(1, len(S_np) + 1), S_np)
        ax.set_xlabel("singular value index")
        ax.set_ylabel("σ (log)")
        ax.set_title(f"lm_head spectrum: {args.model_id}")
        ax.grid(True, alpha=0.3)

        # Cumulative variance
        ax = axes[1]
        ax.semilogx(range(1, len(cum_frac) + 1), cum_frac, label="this model")
        # Add CF12 reference
        ax.axhline(0.99, color="red", linestyle=":", alpha=0.5, label="99%")
        ax.axhline(0.95, color="orange", linestyle=":", alpha=0.5, label="95%")
        ax.axvline(spec["rank_for_99_var"], color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("rank r")
        ax.set_ylabel("cumulative variance fraction")
        ax.set_title("Cumulative variance explained")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = args.output_dir / "spectrum_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\n=========================================")
    print(f"VERDICT: {verdict}")
    print(f"=========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
