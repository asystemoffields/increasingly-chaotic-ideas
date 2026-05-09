r"""M-Strat / RSIDC — Residual-stream stratified intrinsic dimension.

Pre-program measurement that benefits multiple Stage 3 directions:
  - First-Principles E1: tests d* < d_model claim (geometric-fingerprint hypothesis)
  - Composition M6: information concentration at ≤1.1% cardinality dual to intrinsic dim
  - All four directions: per-layer "how much room is in the residual stream" map

Method:
  1. Load Qwen3 model (default 1.7B-Base for compute reasons; can run 4B if RAM allows)
  2. Capture residual-stream activations at every transformer layer
  3. Per layer, compute three intrinsic-dimension estimators:
     - Participation ratio of activation covariance (PR = (Σλ)² / Σλ²)
     - Top-99% rank (k_99 = smallest k explaining 99% of cumulative variance)
     - TwoNN estimator (Facco et al. 2017) for nonlinear ID
  4. Optionally: stratify by token type (BOS/normal/punct/etc) to test sheaf-style
     conditional dimensionality

Go/no-go for First-Principles:
  - GO: median d*/d_model < 0.7 in middle layers (5-22) → stratified geometry
    hypothesis is supported, green-light E2/E3/E4
  - NO-GO: median d*/d_model > 0.95 across all layers → residual stream genuinely
    full-dim; First-Principles claim dead

Output: per-layer (PR, k_99, TwoNN_d) maps; saved to experiments/stage0/ladder/m_strat_rsidc/

References:
  - Facco et al. (2017): "Estimating the intrinsic dimension of datasets by a
    minimal neighborhood information" (TwoNN)
  - "The Shape of Learning" (arXiv:2311.05928, EACL 2024) — bell-shaped
    anisotropy curve peaking middle layers; gives prior expectation
  - "Attention Layers Add Into Low-Dimensional Residual Subspaces" (arXiv:2508.16929)
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
    "every eukaryotic cell. Three principal classes of filaments give "
    "the cell its shape, partition organelles, and provide tracks along "
    "which motor proteins haul vesicles.",
    "Quicksort is an in-place comparison sort whose average-case running "
    "time is O(n log n) and worst-case O(n²). The algorithm partitions "
    "the input around a pivot element and recursively sorts the two "
    "partitions, combining them implicitly.",
    "In Renaissance Florence, Brunelleschi's dome on the cathedral of "
    "Santa Maria del Fiore was an engineering marvel. He invented "
    "machines and perfected herringbone brick patterns to build a "
    "double-shell construction without traditional centering forms.",
    "The Indus Valley Civilization flourished from 3300 to 1300 BCE "
    "across what is now Pakistan and northwest India. Its sophisticated "
    "urban planning, standardized weights, and undeciphered script "
    "remain subjects of intense study.",
    "Quantum entanglement is a physical phenomenon where the quantum state "
    "of each particle in a group cannot be described independently of the "
    "state of the others, including when the particles are separated by "
    "large distances. This non-classical correlation underpins quantum "
    "teleportation, dense coding, and quantum cryptography.",
    "The compiler optimizes the intermediate representation through "
    "dead code elimination, constant folding, common subexpression "
    "elimination, and loop-invariant code motion. Each transformation "
    "preserves program semantics while reducing instruction count and "
    "improving register allocation pressure.",
    "Glaciers form where snow accumulates faster than it ablates, "
    "compressing into firn and then ice over years. Under their own "
    "weight, ice sheets flow plastically downhill, carving U-shaped "
    "valleys, depositing moraines, and gouging fjords during retreat.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-tokens-target", type=int, default=8000,
                   help="Target tokens for ID estimation (more is better up to ~10k)")
    p.add_argument("--probe-every-n-layers", type=int, default=1,
                   help="Probe every Nth layer (1=all, 2=every-other, etc)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder/m_strat_rsidc"),
    )
    return p.parse_args()


@torch.inference_mode()
def collect_residual(
    model, tokenizer, passages: list[str], probe_layers: list[int],
    max_length: int, device: torch.device,
) -> tuple[dict[int, torch.Tensor], int]:
    """Hook each layer's pre_forward to capture residual stream entering the block."""
    captures: dict[int, list[torch.Tensor]] = {ℓ: [] for ℓ in probe_layers}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            captures[layer_idx].append(
                x.detach().to(torch.float32).reshape(-1, x.shape[-1]).cpu()
            )
        return hook

    for ℓ in probe_layers:
        h = model.model.layers[ℓ].register_forward_pre_hook(make_hook(ℓ))
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


def participation_ratio(activations: torch.Tensor) -> float:
    """PR = (Σλ)² / Σλ² where λ are eigenvalues of covariance.

    For Gaussian on R^d, PR ≈ d. For low-dim distribution, PR << d.
    """
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    # SVD-based eigenvalues of covariance
    U, S, Vt = torch.linalg.svd(A_centered, full_matrices=False)
    eigvals = (S ** 2) / (A.shape[0] - 1)
    eigvals = eigvals.cpu().numpy()
    sum_lambda = eigvals.sum()
    sum_lambda_sq = (eigvals ** 2).sum()
    if sum_lambda_sq == 0:
        return float("nan")
    return float(sum_lambda ** 2 / sum_lambda_sq)


def cumulative_variance_rank(activations: torch.Tensor, threshold: float = 0.99) -> int:
    """Smallest k such that cumulative top-k variance ≥ threshold of total."""
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(A_centered)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total == 0:
        return -1
    k = int(np.searchsorted(cum, threshold * total) + 1)
    return k


def twonn_estimator(activations: torch.Tensor, sample_size: int = 2000) -> float:
    """TwoNN (Facco et al. 2017) intrinsic dimension estimator.

    For each point, find r1 (1-NN distance) and r2 (2-NN distance).
    μ_i = r2 / r1. Empirical CDF of μ satisfies F(μ) = 1 - μ^(-d).
    Linear fit log(1 - F) vs log(μ) → slope = -d.
    """
    A = activations.to(torch.float32).cpu().numpy()
    n = A.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        A = A[idx]
        n = sample_size

    # Pairwise distance matrix
    sq_norms = (A ** 2).sum(axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * A @ A.T
    dist_sq = np.maximum(dist_sq, 0)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)  # exclude self

    # For each row, sort and take 1st and 2nd nearest
    sorted_dist = np.sort(dist, axis=1)
    r1 = sorted_dist[:, 0]
    r2 = sorted_dist[:, 1]

    # Filter out zero r1 (duplicates)
    valid = (r1 > 1e-10) & (r2 > 1e-10)
    r1 = r1[valid]
    r2 = r2[valid]
    if len(r1) < 50:
        return float("nan")

    mu = r2 / r1
    mu_sorted = np.sort(mu)
    # Empirical CDF: F_i = i / n (for i=0..n-1)
    n_mu = len(mu_sorted)
    F = np.arange(1, n_mu + 1) / n_mu
    # Use middle 80% to avoid tail noise
    lo = int(0.1 * n_mu)
    hi = int(0.9 * n_mu)
    log_mu = np.log(mu_sorted[lo:hi])
    log_one_minus_F = np.log(np.maximum(1 - F[lo:hi], 1e-10))

    # Linear fit: log_one_minus_F = -d * log_mu + const
    # slope = -d
    slope, intercept = np.polyfit(log_mu, log_one_minus_F, 1)
    d_hat = -slope
    return float(d_hat)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    cfg = model.config
    d = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    print(f"  d={d}  n_layers={n_layers}")

    probe_layers = list(range(0, n_layers, args.probe_every_n_layers))
    if probe_layers[-1] != n_layers - 1:
        probe_layers.append(n_layers - 1)
    print(f"  probing layers: {probe_layers}")

    # Repeat passages to hit token target
    avg_tokens_per_passage = sum(
        len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES
    ) / len(CALIBRATION_PASSAGES)
    repetitions = max(1, int(args.n_tokens_target / (avg_tokens_per_passage * len(CALIBRATION_PASSAGES))))
    passages = CALIBRATION_PASSAGES * repetitions
    print(f"  using {len(passages)} passages ({repetitions}× repetition)")

    print("\nCollecting residual-stream activations...")
    activations, total_tokens = collect_residual(
        model, tokenizer, passages, probe_layers, args.max_length, args.device
    )
    print(f"  total tokens: {total_tokens}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "n_tokens": total_tokens,
        "d_model": d,
        "n_layers": n_layers,
        "probe_layers": probe_layers,
        "per_layer": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    print(f"\nPer-layer intrinsic dimension estimators:")
    print(f"  layer | PR     | k_99 | TwoNN_d | PR/d | k_99/d | TwoNN_d/d")
    print(f"  ------+--------+------+---------+------+--------+----------")
    for ℓ in probe_layers:
        A = activations[ℓ]
        pr = participation_ratio(A)
        k99 = cumulative_variance_rank(A, 0.99)
        twonn = twonn_estimator(A, sample_size=min(2000, A.shape[0]))
        rec = {
            "layer": int(ℓ),
            "n_tokens": int(A.shape[0]),
            "PR": pr,
            "k_99": k99,
            "TwoNN_d": twonn,
            "PR_over_d": pr / d,
            "k_99_over_d": k99 / d if k99 > 0 else float("nan"),
            "TwoNN_d_over_d": twonn / d if not np.isnan(twonn) else float("nan"),
        }
        results["per_layer"][str(ℓ)] = rec
        print(f"  {ℓ:5d} | {pr:6.1f} | {k99:4d} | {twonn:7.1f} | "
              f"{pr/d:.3f} | {k99/d:.3f} | "
              f"{twonn/d if not np.isnan(twonn) else float('nan'):.3f}")

    # Aggregate
    middle_layers = [ℓ for ℓ in probe_layers if 5 <= ℓ <= n_layers - 6]
    middle_recs = [results["per_layer"][str(ℓ)] for ℓ in middle_layers]
    if middle_recs:
        agg = {
            "median_PR_over_d_middle_layers": float(np.median(
                [r["PR_over_d"] for r in middle_recs]
            )),
            "median_k99_over_d_middle_layers": float(np.median(
                [r["k_99_over_d"] for r in middle_recs]
            )),
            "median_TwoNN_over_d_middle_layers": float(np.nanmedian(
                [r["TwoNN_d_over_d"] for r in middle_recs]
            )),
        }
        results["aggregate"] = agg
        print(f"\nMiddle-layer aggregates (layers {middle_layers[0]}-{middle_layers[-1]}):")
        print(f"  median PR/d:     {agg['median_PR_over_d_middle_layers']:.3f}")
        print(f"  median k_99/d:   {agg['median_k99_over_d_middle_layers']:.3f}")
        print(f"  median TwoNN/d:  {agg['median_TwoNN_over_d_middle_layers']:.3f}")

        # Verdict
        # Use k_99 as primary because it's most directly comparable to CF8 r_99/d numbers
        if agg["median_k99_over_d_middle_layers"] < 0.7:
            verdict = ("GO — stratified geometry supported (median k_99/d < 0.7); "
                       "green-light First-Principles E2/E3/E4")
        elif agg["median_k99_over_d_middle_layers"] < 0.85:
            verdict = ("PARTIAL — moderate stratification (0.7 ≤ k_99/d < 0.85); "
                       "First-Principles claim weakened but salvageable")
        else:
            verdict = ("NO-GO — residual stream genuinely full-dim "
                       "(median k_99/d ≥ 0.85); First-Principles geometric-fingerprint "
                       "hypothesis dead in current form")
        results["verdict"] = verdict
        print(f"\nVERDICT: {verdict}")

    out_json = args.output_dir / "rsidc_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Quick markdown
    md = ["# M-Strat / RSIDC — Residual-stream intrinsic dimension",
          "",
          f"Date: {results['timestamp']}",
          f"Model: {args.model_id}",
          f"Tokens: {total_tokens}",
          f"d_model: {d}",
          "",
          f"## Verdict: **{verdict}**" if "verdict" in results else "",
          "",
          "## Per-layer table",
          "",
          "| layer | PR | k_99 | TwoNN_d | PR/d | k_99/d | TwoNN/d |",
          "|---|---|---|---|---|---|---|"]
    for ℓ in probe_layers:
        r = results["per_layer"][str(ℓ)]
        md.append(f"| {ℓ} | {r['PR']:.1f} | {r['k_99']} | {r['TwoNN_d']:.1f} | "
                  f"{r['PR_over_d']:.3f} | {r['k_99_over_d']:.3f} | "
                  f"{r['TwoNN_d_over_d']:.3f} |")
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
