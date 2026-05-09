r"""Composition v2 E1 — Outlier-channel × W_Q-rowspan principal-angle alignment.

Tests the Outlier Highway v2 load-bearing claim:
  dim(span(D^1%) ∩ rowspan(W_Q^shared)) / |D^1%| ≥ 0.5
  AND median principal angle ≤ 35°

where D^1% is the top-1% outlier channel basis from R2-style measurement
and W_Q^shared is the head-shared component extracted via SVD across heads.

If the alignment holds: outliers and the head-shared Q-projection co-evolved
during pretraining; routing on outlier identity ≡ routing on attention-relevant
identity for free, no learned predictor.

If it fails (intersection-fraction < 0.3 or median angle > 60°): the
composition collapses; Outlier Highway reduces to PowerInfer-shaped predictor
work. Either outcome is a publishable structural finding.

Method:
  1. Load Qwen3-1.7B-Base
  2. Forward pass calibration tokens; capture residual stream at each block
  3. Per layer ℓ, compute D^1% basis (canonical basis for top-1% channels by
     persistent magnitude across calibration tokens)
  4. Per layer ℓ, extract W_Q head-shared component:
     W_Q ∈ R^[2048×2048] reshapes to [16 heads × 128 head_dim × 2048 d_in]
     Stack all per-head [128 × 2048] slices → [16*128 × 2048] (already W_Q)
     Take top-K SVD of W_Q across heads — but specifically the *shared* directions
     The "shared subspace" = top-128 right-singular vectors of W_Q
     (since K=128 GO in AQFKV, this is operationally what "head-shared" means)
  5. Compute principal angles between D^1% canonical basis and W_Q top-128 right-singular subspace
  6. Report intersection-fraction (sum of cos² of principal angles, normalized by |D^1%|)
     and median principal angle

References:
  - R2 PDAP K-dependent outlier dynamicity (Jaccard 0.72 → 0.31 at K=0.1% → 1%)
  - AQFKV: W_Q at global K=128 ΔNLL +0.98 (8× compression); head-redundancy
  - Linear Predictability of Attention Heads (arXiv:2603.13314)
  - DeepSeek-V2 MLA architectural cousin
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
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


CALIBRATION_PASSAGES = [
    "The art of celestial navigation rests on a small handful of "
    "instruments and a great deal of patience. A sextant measures the "
    "angle between a celestial body and the horizon; a chronometer "
    "marks Greenwich Mean Time; a nautical almanac gives the precise "
    "position of the sun, moon, and major stars at every hour of every day.",
    "The cytoskeleton is a dynamic network of protein filaments inside "
    "every eukaryotic cell. Three principal classes of filaments — "
    "actin microfilaments, intermediate filaments, and microtubules — "
    "give the cell its shape, partition organelles, and provide the "
    "tracks along which motor proteins haul vesicles.",
    "Quicksort is an in-place comparison sort whose average-case running "
    "time is O(n log n) and whose worst-case running time is O(n²). "
    "The algorithm partitions the input around a pivot element, "
    "recursively sorts the two partitions, and combines them implicitly.",
    "In Renaissance Florence, Brunelleschi's dome on the cathedral of "
    "Santa Maria del Fiore was an engineering marvel. He invented "
    "machines, perfected herringbone brick patterns, and designed "
    "double-shell construction without using traditional centering forms.",
    "The Indus Valley Civilization flourished from approximately 3300 "
    "to 1300 BCE across what is now Pakistan and northwest India. "
    "Its sophisticated urban planning, standardized weights and measures, "
    "and undeciphered script remain subjects of intense study.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--probe-layers", default="4,12,20,27",
                   help="Comma-separated list of layer indices to probe")
    p.add_argument("--K-pct", type=float, default=1.0,
                   help="Top-K%% of channels to use as outlier basis D^K%")
    p.add_argument("--shared-rank", type=int, default=128,
                   help="Rank of W_Q shared subspace (operationally K=128 from AQFKV GO)")
    p.add_argument("--n-tokens-target", type=int, default=10000,
                   help="Target number of tokens for outlier statistics")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/outlier_wq_alignment"),
    )
    return p.parse_args()


@torch.inference_mode()
def collect_residual_activations(
    model, tokenizer, passages: list[str], probe_layers: list[int],
    max_length: int, device: torch.device,
) -> tuple[dict[int, torch.Tensor], int]:
    """Return per-layer concatenated residual-stream activations across all passages.

    Hook the BLOCK INPUT (residual stream entering each transformer block).
    Returns dict: layer_idx -> tensor [total_tokens, d_model] in fp32.
    """
    captures: dict[int, list[torch.Tensor]] = {ℓ: [] for ℓ in probe_layers}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # Block input is the residual stream entering this layer
            x = input[0] if isinstance(input, tuple) else input
            captures[layer_idx].append(x.detach().to(torch.float32).reshape(-1, x.shape[-1]).cpu())
        return hook

    for ℓ in probe_layers:
        h = model.model.layers[ℓ].register_forward_pre_hook(
            lambda m, i, ℓ=ℓ: captures[ℓ].append(
                i[0].detach().to(torch.float32).reshape(-1, i[0].shape[-1]).cpu()
            ) or None
        )
        handles.append(h)

    total_tokens = 0
    for passage in passages:
        encoded = tokenizer(passage, return_tensors="pt",
                            truncation=True, max_length=max_length)
        ids = encoded.input_ids.to(device)
        n = int(ids.shape[-1])
        total_tokens += n
        model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    out = {ℓ: torch.cat(parts, dim=0) for ℓ, parts in captures.items()}
    return out, total_tokens


def compute_outlier_basis(
    activations: torch.Tensor, K_pct: float
) -> tuple[torch.Tensor, np.ndarray]:
    """Compute top-K%% outlier channel basis D^K%.

    activations: [n_tokens, d_model]
    Returns:
      basis: [d_model, k_outliers] canonical basis vectors (one-hot rows)
      indices: indices of top channels (numpy int array)
    """
    d = activations.shape[-1]
    k = max(1, int(round(d * K_pct / 100.0)))
    # Per-channel persistent magnitude metric:
    # mean of |activation| across tokens per channel, then take top-k
    mean_abs = activations.abs().mean(dim=0)
    indices = mean_abs.topk(k).indices.cpu().numpy()
    basis = torch.zeros(d, k, dtype=activations.dtype)
    for j, i in enumerate(indices):
        basis[i, j] = 1.0
    return basis, indices


def compute_wq_shared_subspace(W_Q: torch.Tensor, shared_rank: int) -> torch.Tensor:
    """Extract head-shared component of W_Q via top-K right singular vectors.

    W_Q: [d_out, d_in] = [num_heads*head_dim, d_model]
    Returns: V_shared [d_in, shared_rank] — orthonormal columns spanning
      the head-shared input subspace.

    Operationally we take top-K SVD; if K=128 = d_model/16, this matches
    AQFKV's empirical finding that 16 heads collectively span ~128 dims.
    """
    W_fp32 = W_Q.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
    # Right singular vectors (rows of Vh) span the input subspace where
    # W_Q has support. Top-K covers the dominant head-shared content.
    V_shared = Vh[:shared_rank].T  # [d_in, shared_rank]
    return V_shared


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """Compute principal angles (in radians) between subspaces spanned by columns of A and B.

    Both must have orthonormal columns.
    """
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    # Orthonormalize via QR
    A_q, _ = torch.linalg.qr(A_fp32)
    B_q, _ = torch.linalg.qr(B_fp32)
    M = A_q.T @ B_q
    sigma = torch.linalg.svdvals(M)
    sigma = torch.clamp(sigma, -1.0, 1.0)
    angles = torch.arccos(sigma)
    return angles.cpu().numpy()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_layers = [int(x) for x in args.probe_layers.split(",")]

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    cfg = model.config
    d = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    n_q_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", d // n_q_heads))
    print(f"  d={d}  n_layers={n_layers}  n_q_heads={n_q_heads}  head_dim={head_dim}")

    # Determine number of passage repetitions to hit n_tokens_target
    avg_tokens = sum(len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
    repetitions = max(1, int(args.n_tokens_target / (avg_tokens * len(CALIBRATION_PASSAGES))))
    passages = CALIBRATION_PASSAGES * repetitions
    print(f"  using {len(passages)} passages ({repetitions}× repetition) for ~{args.n_tokens_target} tokens")

    print(f"\nCollecting residual-stream activations at layers {probe_layers} ...")
    activations, total_tokens = collect_residual_activations(
        model, tokenizer, passages, probe_layers, args.max_length, args.device
    )
    print(f"  total tokens collected: {total_tokens}")
    for ℓ, A in activations.items():
        print(f"  layer {ℓ}: {tuple(A.shape)}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "n_tokens": total_tokens,
        "K_pct": args.K_pct,
        "shared_rank": args.shared_rank,
        "probe_layers": probe_layers,
        "per_layer": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    print(f"\nPer-layer alignment analysis (D^{args.K_pct}% × W_Q top-{args.shared_rank} subspace):")
    for ℓ in probe_layers:
        A = activations[ℓ]  # [n_tokens, d_model]
        # Compute outlier basis
        D_basis, D_indices = compute_outlier_basis(A, args.K_pct)
        k_outliers = D_basis.shape[-1]

        # Get W_Q for this layer
        W_Q = model.model.layers[ℓ].self_attn.q_proj.weight.detach()  # [d_out, d_in]
        V_shared = compute_wq_shared_subspace(W_Q, args.shared_rank)

        # Compute principal angles
        angles_rad = principal_angles(D_basis, V_shared)
        angles_deg = np.degrees(angles_rad)
        cos2 = np.cos(angles_rad) ** 2

        # Intersection fraction = Σ cos² σ_i / |D^K%|
        # (Σ cos² is the Frobenius "alignment energy" between D^K% and V_shared)
        intersection_fraction = float(cos2.sum() / k_outliers)
        median_angle_deg = float(np.median(angles_deg))
        min_angle_deg = float(angles_deg.min())
        max_angle_deg = float(angles_deg.max())

        rec = {
            "layer": int(ℓ),
            "n_outlier_channels_K_pct": int(k_outliers),
            "shared_rank": int(args.shared_rank),
            "outlier_indices": D_indices.tolist(),
            "intersection_fraction": intersection_fraction,
            "median_angle_deg": median_angle_deg,
            "min_angle_deg": min_angle_deg,
            "max_angle_deg": max_angle_deg,
            "principal_angles_deg_top10": angles_deg[:10].tolist(),
        }
        results["per_layer"][str(ℓ)] = rec
        print(f"  L{ℓ:2d}: |D^K%|={k_outliers}  intersection_frac={intersection_fraction:.3f}  "
              f"median_angle={median_angle_deg:.1f}°  min={min_angle_deg:.1f}°  max={max_angle_deg:.1f}°")

    # Aggregate
    fracs = [r["intersection_fraction"] for r in results["per_layer"].values()]
    medians = [r["median_angle_deg"] for r in results["per_layer"].values()]
    results["aggregate"] = {
        "mean_intersection_fraction": float(np.mean(fracs)),
        "median_intersection_fraction": float(np.median(fracs)),
        "mean_median_angle_deg": float(np.mean(medians)),
        "median_median_angle_deg": float(np.median(medians)),
    }
    agg = results["aggregate"]
    print(f"\nAggregate across {len(probe_layers)} layers:")
    print(f"  mean intersection_fraction = {agg['mean_intersection_fraction']:.3f}")
    print(f"  median median_angle = {agg['median_median_angle_deg']:.1f}°")

    # Verdict
    if agg["mean_intersection_fraction"] >= 0.5 and agg["median_median_angle_deg"] <= 35:
        verdict = "GO — alignment confirmed; outliers and W_Q head-shared subspace are co-aligned"
    elif agg["mean_intersection_fraction"] >= 0.3:
        verdict = "GRAY — partial alignment (0.3 ≤ frac < 0.5); composition is weakened but salvageable"
    else:
        verdict = "NO-GO — outliers and W_Q rowspan are nearly orthogonal; Outlier Highway composition collapses"
    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    out_json = args.output_dir / "alignment_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
