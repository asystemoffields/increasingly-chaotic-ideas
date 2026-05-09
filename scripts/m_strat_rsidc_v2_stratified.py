r"""M-Strat / RSIDC v2 — STRATIFIED residual-stream intrinsic dimension.

Round 1 PRE-2 v1 found that participation ratio collapses to 1.0 in middle layers
of Qwen3-1.7B because 1-2 massive-activation channels dominate variance (consistent
with Sun et al. 2024 "Massive Activations in LLMs"). This is itself a finding —
the residual stream genuinely has 1-2 extreme channels — but it confounds the
"intrinsic dimension of the typical-token subspace" question that First-Principles
E2 needs.

V2 stratifies the measurement:
  1. Capture residual-stream activations across calibration tokens
  2. Identify top-K magnitude channels per layer (K=2 default — the "highway" of
     Composition v2's hwy/bus/bulk decomposition)
  3. Project them out — orthogonalize the activations against the top-K basis
  4. Re-measure k_99, PR, TwoNN on the projected residual

The k_99_stratified value is what First-Principles E2 should compare to
r_99(W_gate) for the MLP-saturation inequality test.

Independently informative: the residual norm of the stratified activations
gives the "remaining typical-token variance" — if it's also ~1-dim, then there's
a SECOND tier of dominance (the bus). If it's ~1000-dim, then the typical-token
subspace is wide as the geometric-fingerprint hypothesis predicts.
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
    "Santa Maria del Fiore was an engineering marvel.",
    "The Indus Valley Civilization flourished from 3300 to 1300 BCE "
    "across what is now Pakistan and northwest India.",
    "Quantum entanglement is a physical phenomenon where the quantum state "
    "of each particle in a group cannot be described independently.",
    "The compiler optimizes the intermediate representation through "
    "dead code elimination, constant folding, common subexpression "
    "elimination, and loop-invariant code motion.",
    "Glaciers form where snow accumulates faster than it ablates, "
    "compressing into firn and then ice over years.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-tokens-target", type=int, default=8000)
    p.add_argument("--probe-every-n-layers", type=int, default=2)
    p.add_argument("--top-k-strip", type=int, default=2,
                   help="Number of top-magnitude channels to strip (the 'highway')")
    p.add_argument("--exclude-first-tokens", type=int, default=4,
                   help="Number of leading tokens per passage to exclude (BOS/attention-sink)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder/m_strat_rsidc_v2"),
    )
    return p.parse_args()


@torch.inference_mode()
def collect_residual_with_token_filter(
    model, tokenizer, passages: list[str], probe_layers: list[int],
    max_length: int, device: torch.device, exclude_first: int,
) -> tuple[dict[int, torch.Tensor], int]:
    """Capture residual stream entering each probed layer, EXCLUDING the first
    `exclude_first` tokens of each passage (where BOS/attention-sink sit).
    """
    captures: dict[int, list[torch.Tensor]] = {ℓ: [] for ℓ in probe_layers}
    handles = []

    def make_hook(layer_idx: int, exclude_first: int):
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            # x shape [B, T, d]; exclude first tokens
            if x.shape[1] > exclude_first:
                x_filtered = x[:, exclude_first:, :]
            else:
                return  # passage too short
            captures[layer_idx].append(
                x_filtered.detach().to(torch.float32).reshape(-1, x.shape[-1]).cpu()
            )
        return hook

    for ℓ in probe_layers:
        h = model.model.layers[ℓ].register_forward_pre_hook(
            make_hook(ℓ, exclude_first)
        )
        handles.append(h)

    total_tokens = 0
    for passage in passages:
        encoded = tokenizer(passage, return_tensors="pt",
                            truncation=True, max_length=max_length)
        ids = encoded.input_ids.to(device)
        n = max(0, int(ids.shape[-1]) - exclude_first)
        total_tokens += n
        model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    out = {ℓ: torch.cat(parts, dim=0) for ℓ, parts in captures.items()}
    return out, total_tokens


def find_top_magnitude_channels(activations: torch.Tensor, k: int) -> np.ndarray:
    """Return indices of top-k channels by L2 norm across tokens."""
    A = activations.to(torch.float32)
    per_channel_norm = torch.linalg.norm(A, dim=0)  # [d]
    indices = per_channel_norm.topk(k).indices.cpu().numpy()
    return indices


def strip_channels(activations: torch.Tensor, channel_indices: np.ndarray) -> torch.Tensor:
    """Project out specified channels — set them to zero (in canonical basis).

    For a more rigorous "project out", we'd Gram-Schmidt against those basis vectors,
    but since they're canonical basis, simply zeroing the channels = orthogonal projection.
    """
    A = activations.clone().to(torch.float32)
    A[:, channel_indices] = 0.0
    return A


def participation_ratio(activations: torch.Tensor) -> float:
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(A_centered)
    eigvals = (S ** 2).cpu().numpy()
    sum_lambda = eigvals.sum()
    sum_lambda_sq = (eigvals ** 2).sum()
    if sum_lambda_sq == 0:
        return float("nan")
    return float(sum_lambda ** 2 / sum_lambda_sq)


def cumulative_variance_rank(activations: torch.Tensor, threshold: float = 0.99) -> int:
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(A_centered)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total == 0:
        return -1
    return int(np.searchsorted(cum, threshold * total) + 1)


def cumulative_variance_first_n(activations: torch.Tensor, n_top: int = 5) -> list[float]:
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(A_centered)
    eigvals = (S ** 2).cpu().numpy()
    total = eigvals.sum()
    if total == 0:
        return []
    return [float(eigvals[:k+1].sum() / total) for k in range(min(n_top, len(eigvals)))]


@torch.inference_mode()
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

    avg_tokens = sum(len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
    repetitions = max(1, int(args.n_tokens_target / (avg_tokens * len(CALIBRATION_PASSAGES))))
    passages = CALIBRATION_PASSAGES * repetitions
    print(f"  using {len(passages)} passages × {repetitions}x; excluding first {args.exclude_first_tokens} tokens of each")

    print("\nCollecting residual-stream activations (BOS-stripped)...")
    activations, total_tokens = collect_residual_with_token_filter(
        model, tokenizer, passages, probe_layers,
        args.max_length, args.device, args.exclude_first_tokens
    )
    print(f"  total tokens (after BOS strip): {total_tokens}")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "n_tokens": total_tokens,
        "d_model": d,
        "n_layers": n_layers,
        "probe_layers": probe_layers,
        "top_k_strip": args.top_k_strip,
        "exclude_first_tokens": args.exclude_first_tokens,
        "per_layer": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    print(f"\nPer-layer stratified intrinsic dimension:")
    print(f"  layer | top-K chans       | k_99 raw | k_99 stripped | PR raw | PR stripped | top-1 var")
    print(f"  ------+-------------------+----------+---------------+--------+-------------+---------")
    for ℓ in probe_layers:
        A = activations[ℓ]

        # Raw measurements (baseline)
        pr_raw = participation_ratio(A)
        k99_raw = cumulative_variance_rank(A, 0.99)

        # Identify and strip top magnitude channels
        top_chans = find_top_magnitude_channels(A, args.top_k_strip)
        A_stripped = strip_channels(A, top_chans)

        # Stratified measurements
        pr_stripped = participation_ratio(A_stripped)
        k99_stripped = cumulative_variance_rank(A_stripped, 0.99)
        cum_var_first5 = cumulative_variance_first_n(A, n_top=5)

        rec = {
            "layer": int(ℓ),
            "n_tokens": int(A.shape[0]),
            "top_magnitude_channels": top_chans.tolist(),
            "k_99_raw": int(k99_raw),
            "k_99_stripped": int(k99_stripped),
            "PR_raw": pr_raw,
            "PR_stripped": pr_stripped,
            "PR_raw_over_d": pr_raw / d,
            "PR_stripped_over_d": pr_stripped / d,
            "k_99_stripped_over_d": k99_stripped / d if k99_stripped > 0 else float("nan"),
            "var_top_1": cum_var_first5[0] if cum_var_first5 else float("nan"),
            "var_top_2": cum_var_first5[1] if len(cum_var_first5) > 1 else float("nan"),
            "var_top_5": cum_var_first5[4] if len(cum_var_first5) > 4 else float("nan"),
        }
        results["per_layer"][str(ℓ)] = rec
        print(f"  {ℓ:5d} | {str(top_chans.tolist()):17s} | {k99_raw:8d} | "
              f"{k99_stripped:13d} | {pr_raw:6.1f} | {pr_stripped:11.1f} | {cum_var_first5[0] if cum_var_first5 else float('nan'):.4f}")

    # Aggregate
    middle_layers = [ℓ for ℓ in probe_layers if 5 <= ℓ <= n_layers - 6]
    middle_recs = [results["per_layer"][str(ℓ)] for ℓ in middle_layers]
    if middle_recs:
        # use stripped values for the geometric-fingerprint test
        k99_stripped_vals = [r["k_99_stripped"] for r in middle_recs]
        pr_stripped_vals = [r["PR_stripped"] for r in middle_recs]
        agg = {
            "median_k99_stripped_middle": float(np.median(k99_stripped_vals)),
            "median_k99_stripped_over_d_middle": float(np.median(k99_stripped_vals) / d),
            "median_PR_stripped_middle": float(np.median(pr_stripped_vals)),
            "median_PR_stripped_over_d_middle": float(np.median(pr_stripped_vals) / d),
            "median_var_top_1_middle": float(np.median([r["var_top_1"] for r in middle_recs])),
            "median_var_top_2_middle": float(np.median([r["var_top_2"] for r in middle_recs])),
        }
        results["aggregate"] = agg
        print(f"\nMiddle-layer stratified aggregates:")
        print(f"  median var_top_1 (raw): {agg['median_var_top_1_middle']:.4f}  "
              f"(fraction of variance in top-1 channel — confirms massive-activation dominance)")
        print(f"  median var_top_2 (raw): {agg['median_var_top_2_middle']:.4f}")
        print(f"  median k_99 (stripped): {agg['median_k99_stripped_middle']:.0f}")
        print(f"  median k_99/d (stripped): {agg['median_k99_stripped_over_d_middle']:.4f}")
        print(f"  median PR (stripped): {agg['median_PR_stripped_middle']:.1f}")

        # Verdict on STRIPPED measurement (the typical-token intrinsic dimension)
        if agg["median_k99_stripped_over_d_middle"] < 0.7:
            verdict = (f"GO — stratified geometry SUPPORTED on typical-token subspace "
                       f"(median k_99/d after top-{args.top_k_strip} strip = "
                       f"{agg['median_k99_stripped_over_d_middle']:.3f} < 0.7). "
                       f"Geometric-fingerprint hypothesis stands when massive activations "
                       f"are accounted for. First-Principles E2 should use stripped k_residual.")
        elif agg["median_k99_stripped_over_d_middle"] < 0.85:
            verdict = ("PARTIAL — stripped k_99/d in [0.7, 0.85]; weakened but not killed.")
        else:
            verdict = (f"NO-GO — even after stripping top-{args.top_k_strip} massive "
                       f"channels, residual stream is full-dim "
                       f"(median k_99/d = {agg['median_k99_stripped_over_d_middle']:.3f}). "
                       f"Geometric-fingerprint hypothesis dies.")

        results["verdict"] = verdict
        print(f"\nVERDICT: {verdict}")

    out_json = args.output_dir / "rsidc_v2_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
