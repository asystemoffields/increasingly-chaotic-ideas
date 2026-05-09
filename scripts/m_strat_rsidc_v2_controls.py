r"""M-Strat / RSIDC v2 — CONTROLS SCRIPT.

Subjects CF15 (the stratified residual-stream intrinsic-dimension claim) to
skeptic-controls discipline.

CF15 claims:
  A. Typical-token residual stream lives in ~223/2048 dim in middle layers
     (k_99/d ≈ 0.11) after stripping top-2 magnitude channels + first 4 BOS tokens.
  B. Top-2 magnitude channels are CONSISTENT across layers 2-27 at indices
     [1999, 1793] (with layer 0 differing at [784, 1371]).

Controls implemented:
  --permute        : shuffle residual activations across token positions per
                     channel within each layer (destroys cross-channel
                     covariance) and re-measure k_99. Permuted stream should
                     give k_99 → d (full-rank random projection).
  --random-init    : same architecture, random weights. Re-measure. If random-
                     init also gives k_99 ≈ 223 OR top-2 channels at [1999,1793],
                     the corresponding sub-claim is architectural artefact.
  --multi-seed N   : N disjoint slices of a longer corpus, re-measure on each
                     slice. Report mean ± stddev of k_99 in middle layers.
  --strat-prompt-dist : partition CALIBRATION_PASSAGES into prompt-distribution
                     strata and re-measure intrinsic dim per stratum.
  --cross-scale    : if Qwen3-0.6B is available locally, re-measure on it.

Each control writes its slice of results into the JSON output. The script is
designed to be re-run with different flags; results merge under their flag key.

Usage:
  python scripts/m_strat_rsidc_v2_controls.py --permute --multi-seed 3 ...
  python scripts/m_strat_rsidc_v2_controls.py --random-init ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import gc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


# Reuse exactly the same calibration passages as the original v2 script for
# direct comparability of any "no-control" baseline measurement we run here.
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


# Prompt-distribution stratification: tag each passage. (Index-aligned with
# CALIBRATION_PASSAGES.) Used by --strat-prompt-dist.
PASSAGE_TAGS = [
    "natural_language",  # celestial navigation
    "natural_language",  # cytoskeleton
    "code_like",         # quicksort (algorithm description; closest we have)
    "natural_language",  # Brunelleschi
    "natural_language",  # Indus
    "natural_language",  # quantum
    "code_like",         # compiler optimizations
    "natural_language",  # glaciers
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-tokens-target", type=int, default=8000)
    p.add_argument("--probe-every-n-layers", type=int, default=2)
    p.add_argument("--top-k-strip", type=int, default=2)
    p.add_argument("--exclude-first-tokens", type=int, default=4)
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "opus_pipeline/round1/m_strat_v2_controls.json"
        ),
    )
    # Control-selection flags
    p.add_argument("--baseline", action="store_true",
                   help="Re-run the original (unmodified) measurement on the "
                        "trained model, as an in-script baseline for comparison.")
    p.add_argument("--permute", action="store_true",
                   help="Permutation control: shuffle each channel "
                        "independently across tokens, then re-measure k_99.")
    p.add_argument("--random-init", action="store_true",
                   help="Random-init control: instantiate the same arch with "
                        "fresh random weights and re-measure.")
    p.add_argument("--multi-seed", type=int, default=0,
                   help="Number of disjoint corpus seeds to evaluate (0=skip).")
    p.add_argument("--multi-seed-tokens", type=int, default=2500,
                   help="Per-seed token target for the multi-seed control. "
                        "Must be >= d_model for full-rank k_99 estimation.")
    p.add_argument("--strat-prompt-dist", action="store_true",
                   help="Partition corpus by prompt-distribution tag and "
                        "re-measure per stratum.")
    p.add_argument("--cross-scale", action="store_true",
                   help="Try to also load Qwen3-0.6B-Base and re-measure.")
    p.add_argument("--cross-scale-model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--seed", type=int, default=20260509,
                   help="Master seed (used for random-init and permutation).")
    return p.parse_args()


# ============================================================================
# Activation collection (same as v2)
# ============================================================================

@torch.inference_mode()
def collect_residual_with_token_filter(
    model, tokenizer, passages: list[str], probe_layers: list[int],
    max_length: int, device: torch.device, exclude_first: int,
    progress_every: int = 25,
) -> tuple[dict[int, torch.Tensor], int]:
    captures: dict[int, list[torch.Tensor]] = {l: [] for l in probe_layers}
    handles = []

    def make_hook(layer_idx: int, exclude_first: int):
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if x.shape[1] > exclude_first:
                x_filtered = x[:, exclude_first:, :]
            else:
                return
            captures[layer_idx].append(
                x_filtered.detach().to(torch.float32)
                .reshape(-1, x.shape[-1]).cpu()
            )
        return hook

    for l in probe_layers:
        h = model.model.layers[l].register_forward_pre_hook(
            make_hook(l, exclude_first)
        )
        handles.append(h)

    total_tokens = 0
    n_passages = len(passages)
    t0 = datetime.now(UTC)
    for i, passage in enumerate(passages):
        encoded = tokenizer(passage, return_tensors="pt",
                            truncation=True, max_length=max_length)
        ids = encoded.input_ids.to(device)
        n = max(0, int(ids.shape[-1]) - exclude_first)
        total_tokens += n
        model(input_ids=ids, use_cache=False)
        if progress_every and (i + 1) % progress_every == 0:
            dt = (datetime.now(UTC) - t0).total_seconds()
            print(f"    passage {i+1}/{n_passages}  total_tokens={total_tokens}  "
                  f"elapsed={dt:.0f}s  rate={total_tokens/max(dt,1):.0f} tok/s",
                  flush=True)

    for h in handles:
        h.remove()

    out = {l: torch.cat(parts, dim=0) for l, parts in captures.items()}
    return out, total_tokens


# ============================================================================
# Activation statistics (same as v2)
# ============================================================================

def find_top_magnitude_channels(activations: torch.Tensor, k: int) -> np.ndarray:
    A = activations.to(torch.float32)
    per_channel_norm = torch.linalg.norm(A, dim=0)
    indices = per_channel_norm.topk(k).indices.cpu().numpy()
    return indices


def strip_channels(activations: torch.Tensor, channel_indices: np.ndarray) -> torch.Tensor:
    A = activations.clone().to(torch.float32)
    A[:, channel_indices] = 0.0
    return A


def participation_ratio(activations: torch.Tensor) -> float:
    A = activations.to(torch.float32)
    A_centered = A - A.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(A_centered)
    eigvals = (S ** 2).cpu().numpy()
    s = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 == 0:
        return float("nan")
    return float(s ** 2 / s2)


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
    return [float(eigvals[:k+1].sum() / total)
            for k in range(min(n_top, len(eigvals)))]


def measure_one(activations: torch.Tensor, top_k_strip: int, d: int) -> dict[str, Any]:
    """Run the standard CF15 measurement on a [N, d] activation matrix."""
    A = activations
    pr_raw = participation_ratio(A)
    k99_raw = cumulative_variance_rank(A, 0.99)
    top_chans = find_top_magnitude_channels(A, top_k_strip)
    A_stripped = strip_channels(A, top_chans)
    pr_stripped = participation_ratio(A_stripped)
    k99_stripped = cumulative_variance_rank(A_stripped, 0.99)
    cum_var_first5 = cumulative_variance_first_n(A, n_top=5)
    return {
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


def aggregate_middle_layers(per_layer: dict[str, dict[str, Any]],
                            probe_layers: list[int],
                            n_layers: int, d: int) -> dict[str, float]:
    middle_layers = [l for l in probe_layers if 5 <= l <= n_layers - 6]
    middle_recs = [per_layer[str(l)] for l in middle_layers]
    if not middle_recs:
        return {}
    k99_stripped_vals = [r["k_99_stripped"] for r in middle_recs]
    pr_stripped_vals = [r["PR_stripped"] for r in middle_recs]
    return {
        "median_k99_stripped_middle": float(np.median(k99_stripped_vals)),
        "median_k99_stripped_over_d_middle":
            float(np.median(k99_stripped_vals) / d),
        "median_PR_stripped_middle": float(np.median(pr_stripped_vals)),
        "median_PR_stripped_over_d_middle":
            float(np.median(pr_stripped_vals) / d),
        "median_var_top_1_middle":
            float(np.median([r["var_top_1"] for r in middle_recs])),
        "median_var_top_2_middle":
            float(np.median([r["var_top_2"] for r in middle_recs])),
    }


# ============================================================================
# Model load helpers
# ============================================================================

def load_trained(model_id: str, device: str):
    print(f"Loading TRAINED weights from {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    return tokenizer, model


def load_random_init(model_id: str, device: str, seed: int):
    print(f"Instantiating RANDOM-INIT model with arch from {model_id} ...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # bf16 to match memory budget; from_config makes a fresh randomly
    # initialized model and never touches the pretrained weights.
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model = model.to(device).eval()
    return tokenizer, model


def free_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()


# ============================================================================
# Per-control runners
# ============================================================================

def measure_all_layers(activations: dict[int, torch.Tensor],
                       probe_layers: list[int], top_k_strip: int,
                       d: int, n_layers: int, label: str) -> dict[str, Any]:
    """Run measure_one over each probed layer; print + aggregate."""
    per_layer: dict[str, dict[str, Any]] = {}
    print(f"  [{label}] layer | top-K chans       | k_99 raw | k_99 stripped | PR raw | PR stripped",
          flush=True)
    for l in probe_layers:
        rec = measure_one(activations[l], top_k_strip, d)
        rec["layer"] = int(l)
        per_layer[str(l)] = rec
        print(f"  {l:5d} | {str(rec['top_magnitude_channels']):17s} | "
              f"{rec['k_99_raw']:8d} | {rec['k_99_stripped']:13d} | "
              f"{rec['PR_raw']:6.1f} | {rec['PR_stripped']:11.1f}",
              flush=True)
    agg = aggregate_middle_layers(per_layer, probe_layers, n_layers, d)
    return {"per_layer": per_layer, "aggregate": agg}


@torch.inference_mode()
def run_baseline(args, tokenizer, model, passages: list[str],
                 probe_layers: list[int], d: int, n_layers: int,
                 label: str = "baseline",
                 cached_activations: dict[int, torch.Tensor] | None = None,
                 cached_total_tokens: int = 0) -> dict[str, Any]:
    """Standard CF15 measurement on the given (model, corpus).

    If `cached_activations` is supplied, skip the forward-pass collection and
    re-use those activations.
    """
    print(f"\n=== Running {label} measurement ===", flush=True)
    if cached_activations is None:
        activations, total_tokens = collect_residual_with_token_filter(
            model, tokenizer, passages, probe_layers,
            args.max_length, args.device, args.exclude_first_tokens,
        )
        print(f"  total tokens: {total_tokens}", flush=True)
    else:
        activations = cached_activations
        total_tokens = cached_total_tokens
        print(f"  using cached activations  total tokens: {total_tokens}",
              flush=True)
    out = measure_all_layers(activations, probe_layers, args.top_k_strip,
                             d, n_layers, label)
    out["n_tokens"] = total_tokens
    return out


@torch.inference_mode()
def run_permute(args, tokenizer, model, passages: list[str],
                probe_layers: list[int], d: int, n_layers: int,
                seed: int,
                cached_activations: dict[int, torch.Tensor] | None = None,
                cached_total_tokens: int = 0) -> dict[str, Any]:
    """Permutation control: per-channel shuffling across token positions.

    For each layer, we capture activations [N, d] and apply an independent
    random permutation of [0..N-1] to each of the d columns. This destroys
    the joint covariance structure (any non-trivial subspace alignment) while
    preserving each channel's marginal distribution. If CF15's ~223-dim claim
    is real structure (not estimator artefact), the permuted matrix should
    have k_99/d → 1 (full-rank random Gaussian-ish matrix has ~all directions
    contributing). If the permuted matrix still gives k_99 ≈ 223, the claim
    is an artefact.
    """
    print(f"\n=== Running permutation control (seed={seed}) ===", flush=True)
    if cached_activations is None:
        activations, total_tokens = collect_residual_with_token_filter(
            model, tokenizer, passages, probe_layers,
            args.max_length, args.device, args.exclude_first_tokens,
        )
        print(f"  total tokens: {total_tokens}", flush=True)
    else:
        activations = cached_activations
        total_tokens = cached_total_tokens
        print(f"  using cached activations  total tokens: {total_tokens}",
              flush=True)

    rng = np.random.default_rng(seed)

    per_layer: dict[str, dict[str, Any]] = {}
    print(f"  layer | k_99 raw (perm) | k_99 stripped (perm) | k_99 stripped/d (perm)",
          flush=True)
    for l in probe_layers:
        A = activations[l].numpy()  # [N, d]
        N = A.shape[0]
        A_perm = np.empty_like(A)
        for c in range(A.shape[1]):
            idx = rng.permutation(N)
            A_perm[:, c] = A[idx, c]
        A_perm_t = torch.from_numpy(A_perm)
        rec = measure_one(A_perm_t, args.top_k_strip, d)
        rec["layer"] = int(l)
        per_layer[str(l)] = rec
        print(f"  {l:5d} | {rec['k_99_raw']:15d} | {rec['k_99_stripped']:20d} | "
              f"{rec['k_99_stripped_over_d']:.4f}", flush=True)
    agg = aggregate_middle_layers(per_layer, probe_layers, n_layers, d)
    return {
        "n_tokens": total_tokens,
        "permutation_seed": seed,
        "per_layer": per_layer,
        "aggregate": agg,
    }


@torch.inference_mode()
def run_random_init(args, passages: list[str], probe_layers: list[int],
                    d: int, n_layers: int) -> dict[str, Any]:
    """Random-init control. Loads a separate model instance with fresh weights."""
    tokenizer, model = load_random_init(args.model_id, args.device, args.seed)
    out = run_baseline(args, tokenizer, model, passages, probe_layers,
                       d, n_layers, label=f"random-init (seed={args.seed})")
    out["random_init_seed"] = args.seed
    free_model(model)
    return out


@torch.inference_mode()
def run_multi_seed(args, tokenizer, model, base_passages: list[str],
                   probe_layers: list[int], d: int, n_layers: int,
                   n_seeds: int,
                   cached_activations: dict[int, torch.Tensor] | None = None,
                   cached_total_tokens: int = 0) -> dict[str, Any]:
    """Multi-seed control: bootstrap-resample the captured activations N times
    and report mean ± stddev of k_99_stripped per layer.

    NOTE on design: forward passes through the trained model are deterministic,
    so simply repeating CALIBRATION_PASSAGES in different orders does NOT
    create different sample sets — the SVD of the resulting activation matrix
    is identical regardless of ordering (and identical up to scaling regardless
    of repetition count). Multi-seed only gives genuine sample variability if
    we either (a) use diverse, non-overlapping passages — but our corpus has
    only 8 unique passages — or (b) bootstrap-resample the captured tokens.
    We do (b): from the cached activation matrix [N, d], for each seed s draw
    a sample of `multi_seed_tokens` indices WITH replacement and re-measure on
    the resampled matrix. This estimates the variance of k_99 under the
    finite-sample induced sampling distribution.
    """
    print(f"\n=== Running multi-seed control (N={n_seeds}, bootstrap-resample) ===",
          flush=True)
    if cached_activations is None:
        # Need cached activations to bootstrap; if not provided, collect first.
        avg_tokens = sum(len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
        reps = max(1, int(args.n_tokens_target / (avg_tokens * len(CALIBRATION_PASSAGES))))
        passages = CALIBRATION_PASSAGES * reps
        print(f"  collecting activations once for bootstrap ({len(passages)} passages)",
              flush=True)
        cached_activations, cached_total_tokens = collect_residual_with_token_filter(
            model, tokenizer, passages, probe_layers,
            args.max_length, args.device, args.exclude_first_tokens,
        )
    print(f"  cached pool: {cached_total_tokens} tokens; "
          f"per-seed bootstrap sample size: {args.multi_seed_tokens}",
          flush=True)

    rng_root = np.random.default_rng(args.seed)
    seeds_per_layer: dict[int, list[int]] = {l: [] for l in probe_layers}
    seeds_per_layer_pr: dict[int, list[float]] = {l: [] for l in probe_layers}
    seed_records: list[dict[str, Any]] = []
    seed_top_chans: list[dict[int, list[int]]] = []
    sample_size = min(args.multi_seed_tokens, cached_total_tokens)
    for s in range(n_seeds):
        seed_rng = np.random.default_rng(rng_root.integers(0, 2**31 - 1))
        # bootstrap-with-replacement indices into the captured activation pool
        idx = seed_rng.integers(0, cached_total_tokens, size=sample_size)
        print(f"  -- seed {s} -- bootstrap sample {sample_size} of {cached_total_tokens} "
              f"(replacement; first 5 idx: {idx[:5].tolist()})", flush=True)

        # Apply the same idx to every layer's activation matrix and re-measure
        per_layer: dict[str, dict[str, Any]] = {}
        for l in probe_layers:
            A = cached_activations[l][idx]  # [sample_size, d]
            rec = measure_one(A, args.top_k_strip, d)
            rec["layer"] = int(l)
            per_layer[str(l)] = rec
        agg = aggregate_middle_layers(per_layer, probe_layers, n_layers, d)
        out = {"per_layer": per_layer, "aggregate": agg, "n_tokens": int(sample_size)}

        seed_records.append({"seed_index": s, "n_tokens": int(sample_size),
                             "aggregate": agg, "per_layer": per_layer})
        layer_chans = {}
        for l in probe_layers:
            seeds_per_layer[l].append(per_layer[str(l)]["k_99_stripped"])
            seeds_per_layer_pr[l].append(per_layer[str(l)]["PR_stripped"])
            layer_chans[int(l)] = per_layer[str(l)]["top_magnitude_channels"]
        seed_top_chans.append(layer_chans)
        # Print one-line summary per seed
        mid = aggregate_middle_layers(per_layer, probe_layers, n_layers, d)
        print(f"    seed {s}: median k_99_stripped middle={mid.get('median_k99_stripped_middle', float('nan')):.1f}",
              flush=True)

    # Per-layer summary
    summary_per_layer: dict[str, dict[str, Any]] = {}
    for l in probe_layers:
        arr = np.array(seeds_per_layer[l], dtype=float)
        prarr = np.array(seeds_per_layer_pr[l], dtype=float)
        # Channel stability: how often is each channel index the top-1 / top-2?
        top_chans_runs = [tc[l] for tc in seed_top_chans]
        summary_per_layer[str(l)] = {
            "k_99_stripped_values": arr.tolist(),
            "k_99_stripped_mean": float(arr.mean()),
            "k_99_stripped_std": float(arr.std(ddof=0)),
            "PR_stripped_values": prarr.tolist(),
            "PR_stripped_mean": float(prarr.mean()),
            "PR_stripped_std": float(prarr.std(ddof=0)),
            "top_channels_per_seed": top_chans_runs,
            "all_seeds_agree_top_2": all(
                set(tc) == set(top_chans_runs[0]) for tc in top_chans_runs
            ),
        }

    # Aggregated middle-layer mean ± std
    middle_layers = [l for l in probe_layers if 5 <= l <= n_layers - 6]
    middle_means = [summary_per_layer[str(l)]["k_99_stripped_mean"] for l in middle_layers]
    middle_stds = [summary_per_layer[str(l)]["k_99_stripped_std"] for l in middle_layers]
    summary = {
        "n_seeds": n_seeds,
        "per_layer_summary": summary_per_layer,
        "per_seed_records": seed_records,
        "middle_layers_mean_of_means": float(np.mean(middle_means)) if middle_means else float("nan"),
        "middle_layers_mean_of_stds": float(np.mean(middle_stds)) if middle_stds else float("nan"),
        "middle_layers_max_std": float(np.max(middle_stds)) if middle_stds else float("nan"),
    }
    print(f"  middle-layer mean of means k_99_stripped: "
          f"{summary['middle_layers_mean_of_means']:.2f}", flush=True)
    print(f"  middle-layer mean of stds: "
          f"{summary['middle_layers_mean_of_stds']:.2f}", flush=True)
    print(f"  middle-layer max stddev: "
          f"{summary['middle_layers_max_std']:.2f}", flush=True)
    return summary


@torch.inference_mode()
def run_strat_prompt_dist(args, tokenizer, model,
                          probe_layers: list[int], d: int, n_layers: int) -> dict[str, Any]:
    """Partition the corpus by PASSAGE_TAGS and re-measure per stratum."""
    print(f"\n=== Running prompt-distribution stratification ===", flush=True)
    by_tag: dict[str, list[str]] = {}
    for p, t in zip(CALIBRATION_PASSAGES, PASSAGE_TAGS):
        by_tag.setdefault(t, []).append(p)

    avg_tokens = sum(len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
    out: dict[str, Any] = {}
    # Use a smaller per-stratum token target to keep wall time bounded.
    target_per_stratum = max(args.n_tokens_target // 2, 2500)
    for tag, passages_tag in by_tag.items():
        # Repeat to hit ~target token count per stratum.
        reps = max(1, int(target_per_stratum / (avg_tokens * len(passages_tag))))
        passages_used = passages_tag * reps
        print(f"  -- stratum '{tag}' -- {len(passages_used)} passages (reps={reps})", flush=True)
        rec = run_baseline(args, tokenizer, model, passages_used, probe_layers,
                           d, n_layers, label=f"strat[{tag}]")
        out[tag] = rec
    return out


@torch.inference_mode()
def run_cross_scale(args, passages: list[str]) -> dict[str, Any]:
    """Re-measure on a smaller-scale Qwen3."""
    print(f"\n=== Running cross-scale control on {args.cross_scale_model_id} ===",
          flush=True)
    try:
        tokenizer2, model2 = load_trained(args.cross_scale_model_id, args.device)
    except Exception as e:
        print(f"  FAILED to load {args.cross_scale_model_id}: {e}", flush=True)
        return {"error": f"load failed: {e}"}
    cfg2 = model2.config
    d2 = int(cfg2.hidden_size)
    n_layers2 = int(cfg2.num_hidden_layers)
    probe_layers2 = list(range(0, n_layers2, args.probe_every_n_layers))
    if probe_layers2[-1] != n_layers2 - 1:
        probe_layers2.append(n_layers2 - 1)
    print(f"  d={d2}  n_layers={n_layers2}  probe={probe_layers2}", flush=True)
    rec = run_baseline(args, tokenizer2, model2, passages, probe_layers2,
                       d2, n_layers2, label=f"cross-scale[{args.cross_scale_model_id}]")
    rec["d_model"] = d2
    rec["n_layers"] = n_layers2
    rec["probe_layers"] = probe_layers2
    free_model(model2)
    return rec


# ============================================================================
# Main
# ============================================================================

def merge_into_existing(out_json: Path, key: str, value: Any) -> dict[str, Any]:
    """Merge `value` under `key` into the existing JSON file (creating if needed)."""
    if out_json.exists():
        try:
            cur = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    else:
        cur = {}
    cur[key] = value
    out_json.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Args: {vars(args)}", flush=True)
    print(f"Output: {args.output_json}", flush=True)

    # The trained-model controls (baseline, permute, multi-seed,
    # strat-prompt-dist) all share one model instance; load once.
    needs_trained = (args.baseline or args.permute or args.multi_seed > 0
                     or args.strat_prompt_dist)
    needs_random = args.random_init
    needs_cross = args.cross_scale

    if not (needs_trained or needs_random or needs_cross):
        print("No controls selected. Pass at least one of "
              "--baseline / --permute / --multi-seed / --random-init / "
              "--strat-prompt-dist / --cross-scale.", flush=True)
        return 2

    # ------------------------------------------------------------------
    # Pre-build standard passage list (~original v2: target tokens / 8 / avg)
    # We delay the avg-tokens computation until we have a tokenizer.
    # ------------------------------------------------------------------
    base_meta: dict[str, Any] = {
        "model_id": args.model_id,
        "exclude_first_tokens": args.exclude_first_tokens,
        "top_k_strip": args.top_k_strip,
        "max_length": args.max_length,
        "n_tokens_target": args.n_tokens_target,
        "probe_every_n_layers": args.probe_every_n_layers,
        "seed": args.seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "claim_under_test": "CF15 / RSIDC v2",
        "claim_A": "typical-token residual stream lives in ~223/2048 dim "
                   "in middle layers (k_99/d ≈ 0.11) after stripping top-2 "
                   "magnitude channels and BOS-4 tokens",
        "claim_B": "top-2 magnitude channels are CONSISTENT across layers "
                   "2-27 at indices [1999, 1793]",
    }
    merge_into_existing(args.output_json, "meta", base_meta)

    if needs_trained:
        tokenizer, model = load_trained(args.model_id, args.device)
        cfg = model.config
        d = int(cfg.hidden_size)
        n_layers = int(cfg.num_hidden_layers)
        probe_layers = list(range(0, n_layers, args.probe_every_n_layers))
        if probe_layers[-1] != n_layers - 1:
            probe_layers.append(n_layers - 1)
        avg_tokens = sum(len(tokenizer(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
        repetitions = max(1, int(args.n_tokens_target / (avg_tokens * len(CALIBRATION_PASSAGES))))
        passages = CALIBRATION_PASSAGES * repetitions
        print(f"  d={d} n_layers={n_layers} probe_layers={probe_layers}", flush=True)
        print(f"  using {len(passages)} passages × {repetitions}x", flush=True)

        merge_into_existing(args.output_json, "trained_meta", {
            "d_model": d,
            "n_layers": n_layers,
            "probe_layers": probe_layers,
            "n_passages": len(passages),
            "repetitions": repetitions,
        })

        # If both baseline and permute were requested, share one activation
        # collection between them (the heavy cost is the forward passes,
        # which are identical for both).
        shared_acts = None
        shared_tot = 0
        if args.baseline or args.permute:
            print(f"\n--- collecting activations once for baseline + permute ---",
                  flush=True)
            shared_acts, shared_tot = collect_residual_with_token_filter(
                model, tokenizer, passages, probe_layers,
                args.max_length, args.device, args.exclude_first_tokens,
            )
            print(f"  total tokens collected: {shared_tot}", flush=True)

        if args.baseline:
            r = run_baseline(args, tokenizer, model, passages,
                             probe_layers, d, n_layers, "baseline",
                             cached_activations=shared_acts,
                             cached_total_tokens=shared_tot)
            merge_into_existing(args.output_json, "baseline", r)

        if args.permute:
            r = run_permute(args, tokenizer, model, passages,
                            probe_layers, d, n_layers, args.seed,
                            cached_activations=shared_acts,
                            cached_total_tokens=shared_tot)
            merge_into_existing(args.output_json, "permute", r)

        if args.multi_seed > 0:
            r = run_multi_seed(args, tokenizer, model, passages,
                               probe_layers, d, n_layers, args.multi_seed,
                               cached_activations=shared_acts,
                               cached_total_tokens=shared_tot)
            merge_into_existing(args.output_json, "multi_seed", r)

        # Free shared activations before strat-prompt-dist (which collects
        # activations afresh per stratum and we want headroom)
        if shared_acts is not None:
            del shared_acts
            shared_acts = None
            gc.collect()

        if args.strat_prompt_dist:
            r = run_strat_prompt_dist(args, tokenizer, model,
                                      probe_layers, d, n_layers)
            merge_into_existing(args.output_json, "strat_prompt_dist", r)

        free_model(model)

    if needs_random:
        # Use the same probe_layers as the trained run (1.7B has 28 layers)
        cfg = AutoConfig.from_pretrained(args.model_id)
        d = int(cfg.hidden_size)
        n_layers = int(cfg.num_hidden_layers)
        probe_layers = list(range(0, n_layers, args.probe_every_n_layers))
        if probe_layers[-1] != n_layers - 1:
            probe_layers.append(n_layers - 1)
        # Build the standard passage list (we need a tokenizer for token count)
        tok_tmp = AutoTokenizer.from_pretrained(args.model_id)
        avg_tokens = sum(len(tok_tmp(p).input_ids) for p in CALIBRATION_PASSAGES) / len(CALIBRATION_PASSAGES)
        repetitions = max(1, int(args.n_tokens_target / (avg_tokens * len(CALIBRATION_PASSAGES))))
        passages = CALIBRATION_PASSAGES * repetitions
        del tok_tmp
        gc.collect()
        r = run_random_init(args, passages, probe_layers, d, n_layers)
        merge_into_existing(args.output_json, "random_init", r)

    if needs_cross:
        # We need a passage list for cross-scale. The cross-scale model has
        # its own tokenizer; use the same calibration passages and let the
        # tokenizer determine token counts.
        passages = CALIBRATION_PASSAGES * max(1, int(args.n_tokens_target / 200))
        r = run_cross_scale(args, passages)
        merge_into_existing(args.output_json, "cross_scale", r)

    print(f"\nWrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
