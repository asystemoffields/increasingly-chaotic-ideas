r"""First-Principles E3 — Joint Q* Construction.

Minimizes Σ_M ||M·Q[:, r:]||²_F over Q ∈ O(d) for M ∈ {W_gate^{(ℓ)}, W_up^{(ℓ)}, W_E (tied)}.

Math: this objective is equivalent to maximizing trace(Q^T C Q · diag([1]*r + [0]*(d-r)))
where C = Σ_M α_M · M^T M. The optimal Q has first r columns spanning top-r eigenvectors
of C. Equivalently, Q* is the right singular basis of the vertically-stacked matrix
[W_gate^{(ℓ)}; W_up^{(ℓ)}; W_E] (with appropriate weights for matrix-size balance).

Novelty per Stage 2 review (vs MoDeGPT): MoDeGPT does intra-module joint SVD; Q* in this
proposal is computed across ALL transformer layers AND includes the tied embedding/lm_head
matrix W_E. The cross-module shared rotation is the contribution.

Output: Q* matrix saved as torch tensor; per-layer r_99(M · Q*) measurements; saved as
input to FP E4 (the binary settling experiment).

Per-layer Q*: there are two options:
  (a) GLOBAL Q*: single Q for all layers. Simplest. Lets us thread Q through residual
      stream cleanly via paired rotations.
  (b) PER-LAYER Q*: each layer ℓ has its own Q*_ℓ. Better compression but breaks the
      paired-rotation argument unless we also rotate the residual stream between layers
      by Q*_{ℓ-1}^T · Q*_ℓ — which is itself an O(d) rotation that could be absorbed
      into the layer's input projections.

For E3 simplicity, we compute GLOBAL Q* (option a): the top-r right singular vectors
of [W_gate^{(0)}; W_up^{(0)}; W_gate^{(1)}; W_up^{(1)}; ...; W_E] vertically stacked.
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
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/fp_e3_joint_qstar"),
    )
    p.add_argument("--include-attention", action="store_true",
                   help="Also include W_Q, W_K, W_V, W_O in the joint stack")
    p.add_argument("--save-q-star", action="store_true", default=True,
                   help="Save Q* as torch tensor for E4 to use")
    return p.parse_args()


def r_99(W: torch.Tensor) -> int:
    """Compute 99th-percentile rank of W's right-singular subspace."""
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total == 0:
        return -1
    return int(np.searchsorted(cum, 0.99 * total) + 1)


def variance_at_r(W: torch.Tensor, r: int) -> float:
    """Fraction of Frobenius energy explained by top-r right-singular components."""
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total == 0:
        return float("nan")
    return float(cum[min(r - 1, len(cum) - 1)] / total)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    cfg = model.config
    d = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    print(f"  d={d}  n_layers={n_layers}")

    # Collect all weight matrices to include in the joint stack.
    # Each matrix M acts as M·h on residual stream h ∈ ℝ^d, so we want their
    # right-singular bases (input subspaces) to align to a common Q*.
    matrices: list[tuple[str, torch.Tensor]] = []

    print("\nCollecting weight matrices...")
    for ℓ in range(n_layers):
        layer = model.model.layers[ℓ]
        Wg = layer.mlp.gate_proj.weight.detach().to(torch.float32)
        Wu = layer.mlp.up_proj.weight.detach().to(torch.float32)
        matrices.append((f"L{ℓ}.gate", Wg))
        matrices.append((f"L{ℓ}.up", Wu))
        if args.include_attention:
            Wq = layer.self_attn.q_proj.weight.detach().to(torch.float32)
            Wk = layer.self_attn.k_proj.weight.detach().to(torch.float32)
            Wv = layer.self_attn.v_proj.weight.detach().to(torch.float32)
            matrices.append((f"L{ℓ}.q", Wq))
            matrices.append((f"L{ℓ}.k", Wk))
            matrices.append((f"L{ℓ}.v", Wv))

    # Tied lm_head / embedding
    W_E = model.lm_head.weight.detach().to(torch.float32)
    matrices.append(("lm_head", W_E))
    print(f"  total matrices: {len(matrices)}")

    # All matrices have d_in = d (right side); shapes vary on d_out
    # Verify
    for name, M in matrices:
        if M.shape[-1] != d:
            print(f"  WARNING: {name} shape {tuple(M.shape)} has d_in != {d}")

    # Stack vertically. To balance contributions across matrices of different sizes
    # (lm_head is 151936 × 2048 vs MLP is 6144 × 2048), normalize each matrix to
    # unit Frobenius norm before stacking — this gives equal weight to each matrix's
    # information content rather than its row count.
    print("\nNormalizing and stacking matrices...")
    normalized = []
    for name, M in matrices:
        fro = torch.linalg.norm(M)
        normalized.append(M / (fro + 1e-12))
    stacked = torch.cat(normalized, dim=0)  # [total_rows, d]
    print(f"  stacked shape: {tuple(stacked.shape)}")

    # Compute SVD of stacked matrix. Top-r right singular vectors define Q*'s first r columns.
    print("\nComputing SVD of stacked matrix (this may take ~1-3 min for d=2048)...")
    t0 = datetime.now(UTC)
    U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
    t1 = datetime.now(UTC)
    elapsed = (t1 - t0).total_seconds()
    print(f"  SVD done in {elapsed:.1f}s")
    print(f"  U: {tuple(U.shape)}  S: {tuple(S.shape)}  Vh: {tuple(Vh.shape)}")

    # Q* = V (right singular vectors of stacked matrix).
    # Vh has shape [d, d]; rows of Vh are right-singular vectors. Q* is V = Vh^T.
    Q_star = Vh.T  # shape [d, d]
    print(f"  Q* shape: {tuple(Q_star.shape)}")

    # Per-matrix energy concentration in top-r columns of M·Q*
    print(f"\nPer-matrix energy concentration after right-rotation by Q*:")
    print(f"  matrix         | r_99(orig) | r_99(rotated) | var@r=1024 (orig) | var@r=1024 (rotated)")
    print(f"  ---------------+------------+---------------+-------------------+---------------------")
    per_matrix_rec: list[dict[str, Any]] = []
    for name, M in matrices[:5] + [matrices[len(matrices)//2], matrices[-1]]:
        # Sample for reporting
        M_rotated = M @ Q_star  # [d_out, d]
        r99_orig = r_99(M)
        r99_rot = r_99(M_rotated)
        var_1024_orig = variance_at_r(M, 1024)
        var_1024_rot = variance_at_r(M_rotated, 1024)
        per_matrix_rec.append({
            "name": name,
            "r_99_orig": r99_orig,
            "r_99_rotated": r99_rot,
            "var_at_1024_orig": var_1024_orig,
            "var_at_1024_rotated": var_1024_rot,
        })
        print(f"  {name:14s} | {r99_orig:10d} | {r99_rot:13d} | "
              f"{var_1024_orig:.4f}            | {var_1024_rot:.4f}")

    # Joint metric: how much energy of EACH matrix lies in the first r columns when
    # using the GLOBAL Q*?
    # For each matrix M, compute ||M · Q*[:, :r]||_F^2 / ||M||_F^2
    print(f"\nFraction of M·Q* energy in first r columns of rotated basis:")
    print(f"  r        | mean_frac | min_frac | max_frac")
    print(f"  ---------+-----------+----------+---------")
    aggregate: list[dict[str, Any]] = []
    for r in [128, 256, 512, 1024, 1536]:
        if r > d:
            continue
        Q_top = Q_star[:, :r]  # [d, r]
        fracs = []
        for name, M in matrices:
            # ||M · Q_top||_F^2 / ||M||_F^2
            M_top = M @ Q_top  # [d_out, r]
            energy_top = torch.linalg.norm(M_top) ** 2
            energy_total = torch.linalg.norm(M) ** 2
            frac = float((energy_top / (energy_total + 1e-12)).item())
            fracs.append(frac)
        fracs_arr = np.array(fracs)
        rec = {
            "r": r,
            "r_over_d": r / d,
            "mean_frac": float(fracs_arr.mean()),
            "median_frac": float(np.median(fracs_arr)),
            "min_frac": float(fracs_arr.min()),
            "max_frac": float(fracs_arr.max()),
        }
        aggregate.append(rec)
        print(f"  r={r:5d} | {rec['mean_frac']:.4f}    | {rec['min_frac']:.4f}   | {rec['max_frac']:.4f}")

    # Compare to identity baseline (no rotation)
    print(f"\nBaseline (no rotation, sum of squared singular values in first r columns vs total):")
    print(f"  r        | mean_frac | min_frac | max_frac")
    print(f"  ---------+-----------+----------+---------")
    baseline_aggregate: list[dict[str, Any]] = []
    for r in [128, 256, 512, 1024, 1536]:
        if r > d:
            continue
        # In ambient basis, "first r columns" = first r input dimensions arbitrarily.
        # But that's not meaningful comparison. Instead compare to "first r dims along
        # M's OWN top right-singular vectors" — which is just var_at_r(M, r).
        fracs = []
        for name, M in matrices:
            fracs.append(variance_at_r(M, r))
        fracs_arr = np.array(fracs)
        rec = {
            "r": r,
            "mean_frac": float(fracs_arr.mean()),
            "median_frac": float(np.median(fracs_arr)),
            "min_frac": float(fracs_arr.min()),
            "max_frac": float(fracs_arr.max()),
        }
        baseline_aggregate.append(rec)
        print(f"  r={r:5d} | {rec['mean_frac']:.4f}    | {rec['min_frac']:.4f}   | {rec['max_frac']:.4f}")

    # Verdict (vs Stage 4 synthesis E3 GO threshold: ≥1.4× compression on summed rank)
    # Compute "summed rank" for Q* and identity:
    #   For each matrix M, find smallest r such that var@r >= 0.99
    # Sum across matrices.
    print(f"\nSummed r_99 across all matrices:")
    summed_r99_orig = sum(r_99(M) for _, M in matrices)
    summed_r99_rot = sum(r_99(M @ Q_star) for _, M in matrices)
    compression_ratio = summed_r99_orig / summed_r99_rot if summed_r99_rot > 0 else float("inf")
    print(f"  identity Q (orig): Σ r_99 = {summed_r99_orig}")
    print(f"  joint Q*:           Σ r_99 = {summed_r99_rot}")
    print(f"  compression ratio: {compression_ratio:.3f}×")

    if compression_ratio >= 1.4:
        verdict = (f"GO — joint Q* achieves {compression_ratio:.2f}× compression on summed r_99; "
                   f"green-light E4 (rotated tied lm_head settling experiment)")
    elif compression_ratio >= 1.05:
        verdict = (f"PARTIAL — modest gain ({compression_ratio:.2f}×); E4 may still produce "
                   f"interesting structural finding but ambition reduced")
    else:
        verdict = (f"NO-GO — joint Q* produces no compression ({compression_ratio:.2f}×); "
                   f"matrices' right-singular bases are essentially orthogonal; "
                   f"hypothesis fails at the joint-rotation level")

    results = {
        "model_id": args.model_id,
        "d_model": d,
        "n_matrices_in_stack": len(matrices),
        "include_attention": args.include_attention,
        "stacked_shape": list(stacked.shape),
        "svd_runtime_seconds": elapsed,
        "summed_r99_identity": summed_r99_orig,
        "summed_r99_q_star": summed_r99_rot,
        "compression_ratio": compression_ratio,
        "per_matrix_sample": per_matrix_rec,
        "energy_concentration_q_star": aggregate,
        "energy_concentration_baseline_per_matrix": baseline_aggregate,
        "verdict": verdict,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    out_json = args.output_dir / "fp_e3_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    if args.save_q_star:
        q_star_path = args.output_dir / "q_star.pt"
        torch.save({
            "Q_star": Q_star,
            "S_singular_values": S,
            "stacked_norms": [float(torch.linalg.norm(M).item()) for _, M in matrices],
            "matrix_names": [name for name, _ in matrices],
            "model_id": args.model_id,
            "d_model": d,
            "timestamp": datetime.now(UTC).isoformat(),
        }, q_star_path)
        print(f"Wrote {q_star_path} ({Q_star.element_size() * Q_star.numel() / 1e6:.1f} MB)")

    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
