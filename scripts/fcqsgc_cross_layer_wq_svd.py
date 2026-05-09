r"""F-CQSGC — Cross-Layer W_Q Stacked SVD (Run 001, Track A).

Tests whether all 28 W_Q matrices in Qwen3-1.7B-Base share a common dominant
right singular subspace U ∈ R^{2048×128} that captures > 80% of stacked-matrix
variance (var@128 > 0.80).

Stage 6 amendments applied:
  A1 — random-init uses torch.nn.init.normal_(std=0.02); raw var@128 reported.
  A2 — BLOCKING: matched-spectrum random-orientation baseline. GO requires
       real var@128 > baseline var@128 by > 0.20.
  A3 — deployment GO sub-threshold: ΔNLL < 0.30 nat at the passing K required
       for HOIST cascade justification.
  A4 — COMPOT (arXiv:2602.15200, Feb 2026) cited in kill criteria / context.

References:
  - Basis Sharing (arXiv:2410.03765, ICLR 2025): requires fine-tuning; no-retraining
    moat for F-CQSGC appears to survive per Stage 6.
  - MASA / Share Your Attention (arXiv:2508.04581): fine-tuning-aided recovery.
  - QSVD (arXiv:2510.16292, NeurIPS 2025): cross-layer adaptive rank, calibration-guided.
  - COMPOT (arXiv:2602.15200): single shared subspace degrades at moderate compression.
  - xKV (arXiv:2503.18893): cross-layer SVD on KV-Cache activations, not weights.

Go thresholds (Stage 5 + Stage 6 amendments):
  STRUCTURAL GO: mean var@128 (3 seeds) > 0.80
                 AND gap over matched-spectrum baseline > 0.20  [A2 BLOCKING]
                 AND worst-of-3 var@128 > 0.75
  DEPLOYMENT GO: ΔNLL < 0.30 nat at K=128 (required for HOIST cascade)  [A3]
  Soft GO: var@128 ∈ [0.65, 0.80] → cross-layer sharing needs D*=256.
  NO-GO: var@128 < 0.50 → kills Cluster C1 (R-CROSS-Q, C-CQBAL, F-CQSGC, A-GFRS).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


# ---------------------------------------------------------------------------
# Eval passage (~400 tokens — same celestial-navigation text used in AQFKV)
# ---------------------------------------------------------------------------
EVAL_PASSAGE = (
    "The art of celestial navigation rests on a small handful of "
    "instruments and a great deal of patience. A sextant measures the "
    "angle between a celestial body and the horizon; a chronometer "
    "marks Greenwich Mean Time; a nautical almanac gives the precise "
    "position of the sun, moon, and major stars at every hour of every "
    "day. With these in hand, a navigator can determine latitude from a "
    "noon sun sight, and longitude from a timed observation of any "
    "celestial body, anywhere on the open ocean. The mathematics is "
    "spherical trigonometry, taught for centuries to merchant marine "
    "officers and naval cadets alike. The intuition, however, comes only "
    "with practice — learning to brace against the swell while keeping "
    "the body's reflected image precisely on the horizon, learning to "
    "estimate the moment of meridian passage from the curve of the sun's "
    "altitude over the half hour bracketing local noon. The development "
    "of accurate marine chronometers in the eighteenth century, by John "
    "Harrison and others, changed the practice of navigation more "
    "fundamentally than any innovation since the magnetic compass. "
    "Before Harrison, longitude at sea was determined by lunar distances "
    "or by dead reckoning, both prone to large errors over weeks of "
    "voyaging. After Harrison, longitude could be determined to within a "
    "few miles given a working chronometer, a sextant, and a careful "
    "navigator. The British Admiralty, eventually, awarded Harrison a "
    "substantial portion of the Longitude Prize, though the official "
    "award process took decades and was tangled in politics. The story "
    "of celestial navigation continues into the twentieth century: the "
    "Polynesian wayfinders of Hawaii revived traditional star-path "
    "navigation in the 1970s; commercial airlines used celestial fixes "
    "across the Pacific until satellite navigation made the practice "
    "obsolete; the United States Navy still trains its officers in the "
    "discipline as a backup against GPS jamming."
)

# Rank sweep values for ΔNLL evaluation.
K_VALUES = (32, 64, 128, 256, 512, 1024, 2048)

# Number of seeds for randomized SVD stability check.
SVD_SEEDS = (42, 1337, 2024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder_v2/round1_cqsgc"),
    )
    p.add_argument("--skip-random-init", action="store_true",
                   help="Skip random-init control to save time.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# NLL helpers
# ---------------------------------------------------------------------------
@torch.inference_mode()
def compute_loss(model, ids: torch.Tensor) -> tuple[float, int]:
    """Return (sum_nll_nats, n_tokens)."""
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(nll.sum().item()), int(nll.numel())


# ---------------------------------------------------------------------------
# Stacked-SVD helpers
# ---------------------------------------------------------------------------
def extract_wq_matrices(model) -> list[torch.Tensor]:
    """Return list of W_Q float32 tensors, one per layer."""
    return [
        layer.self_attn.q_proj.weight.detach().clone().float()
        for layer in model.model.layers
    ]


def frobenius_norms(wq_list: list[torch.Tensor]) -> np.ndarray:
    return np.array([float(W.norm(p="fro").item()) for W in wq_list])


def build_stacked_matrix(
    wq_list: list[torch.Tensor],
    norms: np.ndarray | None = None,
) -> np.ndarray:
    """Stack all W_Q into [n_layers*d, d] after optional Frobenius normalization.
    Uses float32 to reduce memory.
    """
    layers = []
    for i, W in enumerate(wq_list):
        w = W.numpy().astype(np.float32)
        if norms is not None:
            w = w / np.float32(norms[i])
        layers.append(w)
    return np.vstack(layers).astype(np.float32)  # [n_layers*d, d]


def compute_var_at_k(
    M_stack: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Compute var@k via torch.svd_lowrank. Returns (var_at_k, singular_values_np)."""
    torch.manual_seed(seed)
    n_components = min(k, min(M_stack.shape))
    T = torch.from_numpy(M_stack.astype(np.float32))
    # niter=4 for accuracy; q parameter is oversampling (n_components + 10)
    _, S, _ = torch.svd_lowrank(T, q=n_components, niter=4)
    S_np = S.numpy()
    total_energy = float(np.sum(M_stack.astype(np.float32) ** 2))
    captured = float(np.sum(S_np ** 2))
    return captured / total_energy, S_np


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random orthogonal matrix from the Haar measure via QR of random normal."""
    A = rng.standard_normal((n, n)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Correct sign to ensure uniform Haar distribution.
    d = np.sign(np.diag(R))
    Q = Q * d[np.newaxis, :]
    return Q


def build_matched_spectrum_baseline(
    wq_list: list[torch.Tensor],
    norms: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    A2 BLOCKING amendment: matched-spectrum random-orientation baseline.

    For each layer ℓ:
      1. Compute singular values of (normalized) W_Q^(ℓ) via torch.linalg.svdvals (fast).
      2. Draw random orthogonal Q_a, Q_b from Haar measure via QR.
      3. Produce W_rand^(ℓ) = Q_a @ diag(s) @ Q_b^T (same singular values, random orientations).

    Stack the random-orientation copies and return [n_layers*d, d].
    Uses torch float32 for SVD (faster via MKL), numpy float32 for construction.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    layers = []
    for i, W in enumerate(wq_list):
        # Fast: only singular values needed (torch.linalg.svdvals is cheaper than full SVD)
        W_norm = W.float() / float(norms[i])
        s = torch.linalg.svdvals(W_norm).numpy().astype(np.float32)  # [d]
        layer_d = len(s)
        # Draw random orthogonal replacements (size = layer_d × layer_d via QR).
        Q_a = random_orthogonal(layer_d, rng).astype(np.float32)  # [d, d] Haar
        Q_b = random_orthogonal(layer_d, rng).astype(np.float32)  # [d, d] Haar
        # Q_a diag(s) Q_b^T  =  (Q_a * s[None, :]) @ Q_b
        W_rand = (Q_a * s[np.newaxis, :]) @ Q_b  # [d, d]
        layers.append(W_rand)
        if (i + 1) % 7 == 0:
            print(f"    A2: completed {i+1}/{len(wq_list)} layers ...", flush=True)
    return np.vstack(layers).astype(np.float32)


def reconstruct_wq_from_shared_basis(
    original_wq: torch.Tensor,
    U_shared: np.ndarray,
) -> torch.Tensor:
    """
    Given original W_Q^(ℓ) (un-normalized) and shared right-basis U ∈ R^{d×k},
    compute W_Q^(ℓ)_approx = C_ℓ @ U^T where C_ℓ = W_Q^(ℓ) @ U.

    Note: normalization is NOT applied here — U is derived from the normalized
    stack but reconstruction uses the original (un-normalized) weight matrix.
    This is the Stage 6 math-correctness check: normalization only affects the
    SVD step, not the per-layer reconstruction.
    """
    W = original_wq.float().numpy()  # [d, d] original
    C = W @ U_shared  # [d, k]
    W_approx = C @ U_shared.T  # [d, d]
    return torch.from_numpy(W_approx).to(original_wq.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.inference_mode()
def main() -> int:
    args = parse_args()
    t_start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[F-CQSGC] Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    d = int(cfg.hidden_size)
    print(f"  n_layers={n_layers}  d={d}")

    # Tokenize eval passage.
    encoded = tokenizer(
        args.passage if hasattr(args, "passage") else EVAL_PASSAGE,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    ids = encoded.input_ids.to(args.device)
    n_tokens = int(ids.shape[-1])
    print(f"  n_eval_tokens={n_tokens}")

    # ---- Baseline NLL ----
    print("\n[1] Baseline NLL ...")
    base_nll, base_n = compute_loss(model, ids)
    base_loss_per_tok = base_nll / base_n
    base_ppl = float(np.exp(base_loss_per_tok))
    print(f"  baseline NLL/tok = {base_loss_per_tok:.4f}  PPL = {base_ppl:.3f}")

    # ---- Extract W_Q matrices ----
    print("\n[2] Extracting W_Q matrices ...")
    wq_list = extract_wq_matrices(model)
    norms = frobenius_norms(wq_list)
    median_norm = float(np.median(norms))
    outlier_layers = [i for i, n_ in enumerate(norms) if n_ < 0.1 * median_norm]
    print(f"  Frobenius norms: min={norms.min():.2f}  max={norms.max():.2f}  "
          f"median={median_norm:.2f}")
    if outlier_layers:
        print(f"  WARNING: outlier near-zero-norm layers (< 0.1×median): {outlier_layers}")
    else:
        print("  No near-zero-norm outlier layers detected.")

    # Save original W_Q for restore.
    original_wq = [
        layer.self_attn.q_proj.weight.detach().clone()
        for layer in model.model.layers
    ]

    def restore_wq() -> None:
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.weight.data.copy_(original_wq[li])

    # ---- Build stacked matrix ----
    print("\n[3] Building stacked matrix [n_layers*d, d] ...")
    M_stack = build_stacked_matrix(wq_list, norms=norms)
    print(f"  M_stack shape={M_stack.shape}  dtype={M_stack.dtype}  "
          f"memory={M_stack.nbytes / 1e6:.1f} MB")

    # ---- Compute var@K profile for multiple K values and 3 seeds ----
    print("\n[4] Stacked SVD var@K profile (3 seeds) ...")
    var_profile: dict[str, Any] = {}
    # We want D* in {64, 128, 256, 512} for profile; compute up to max(512, need)
    profile_ks = [64, 128, 256, 512]
    for k_profile in profile_ks:
        seed_vars = []
        for seed in SVD_SEEDS:
            v, _ = compute_var_at_k(M_stack, k_profile, seed=seed)
            seed_vars.append(v)
        mean_v = float(np.mean(seed_vars))
        std_v = float(np.std(seed_vars))
        var_profile[f"var_at_{k_profile}"] = {
            "mean": mean_v,
            "std": std_v,
            "per_seed": seed_vars,
        }
        print(f"  var@{k_profile}: mean={mean_v:.4f}  std={std_v:.4f}  "
              f"per_seed={[f'{v:.4f}' for v in seed_vars]}")

    # Primary: var@128 over 3 seeds.
    var128_mean = var_profile["var_at_128"]["mean"]
    var128_std = var_profile["var_at_128"]["std"]
    var128_seeds = var_profile["var_at_128"]["per_seed"]
    var128_worst = float(min(var128_seeds))

    # Also store the full singular-value array from seed=42 at k=128 for approximation
    # error check (we'll compute it during basis extraction below).
    _unused_v, S_128_seed42 = compute_var_at_k(M_stack, 128, seed=42)

    # ---- A2 BLOCKING: matched-spectrum random-orientation baseline ----
    print("\n[5] A2 BLOCKING: matched-spectrum random-orientation baseline ...")
    print("  Computing per-layer full SVD for all 28 layers (this may take several minutes)...")
    t_a2_start = time.time()
    M_rand = build_matched_spectrum_baseline(wq_list, norms, seed=42)
    t_a2 = time.time() - t_a2_start
    print(f"  Built random-orientation stack in {t_a2:.1f}s")
    var128_baseline_seeds = []
    for seed in SVD_SEEDS:
        v_rand, _ = compute_var_at_k(M_rand, 128, seed=seed)
        var128_baseline_seeds.append(float(v_rand))
    var128_baseline_mean = float(np.mean(var128_baseline_seeds))
    var128_baseline_std = float(np.std(var128_baseline_seeds))
    gap_vs_baseline = var128_mean - var128_baseline_mean
    print(f"  Baseline var@128: mean={var128_baseline_mean:.4f}  std={var128_baseline_std:.4f}")
    print(f"  Gap (real - baseline): {gap_vs_baseline:+.4f}  "
          f"(GO requires > 0.20)")
    a2_pass = bool(gap_vs_baseline > 0.20)
    print(f"  A2 BLOCKING gate: {'PASS' if a2_pass else 'FAIL'}")

    # ---- Permutation control ----
    print("\n[6] Permutation control ...")
    rng_perm = np.random.default_rng(42)
    layers_perm = []
    for i, W in enumerate(wq_list):
        w = W.numpy() / norms[i]
        perm = rng_perm.permutation(w.shape[0])
        layers_perm.append(w[perm, :])
    M_perm = np.vstack(layers_perm)
    var128_perm, _ = compute_var_at_k(M_perm, 128, seed=42)
    gap_vs_perm = var128_mean - var128_perm
    print(f"  Permuted var@128 = {var128_perm:.4f}  (expected ~0.063)")
    print(f"  Gap real-vs-permuted = {gap_vs_perm:+.4f}  (GO requires >= 0.60)")
    perm_pass = bool(var128_perm < 0.20 and gap_vs_perm >= 0.60)
    print(f"  Permutation control: {'PASS' if perm_pass else 'FAIL'}")

    # ---- Random-init control (A1) ----
    rand_init_var128: float | None = None
    gap_vs_rand_init: float | None = None
    rand_init_pass: bool | None = None
    if not args.skip_random_init:
        print("\n[7] A1: Random-init control (normal_(std=0.02)) ...")
        rand_config = AutoConfig.from_pretrained(args.model_id)
        rand_model = AutoModelForCausalLM.from_config(rand_config).float().eval()
        # Re-initialize all Q projections with normal_(std=0.02) as specified by A1.
        for layer in rand_model.model.layers:
            torch.nn.init.normal_(layer.self_attn.q_proj.weight, std=0.02)
        rand_wq_list = extract_wq_matrices(rand_model)
        rand_norms = frobenius_norms(rand_wq_list)
        M_rand_init = build_stacked_matrix(rand_wq_list, norms=rand_norms)
        rand_init_var128, _ = compute_var_at_k(M_rand_init, 128, seed=42)
        gap_vs_rand_init = var128_mean - rand_init_var128
        rand_init_pass = bool(gap_vs_rand_init > 0.30)
        print(f"  Random-init var@128 = {rand_init_var128:.4f}")
        print(f"  Gap (real - rand_init) = {gap_vs_rand_init:+.4f}  (GO requires > 0.30)")
        print(f"  Random-init control: {'PASS' if rand_init_pass else 'FAIL'}")
        del rand_model, rand_wq_list, M_rand_init
    else:
        print("\n[7] A1: Random-init control skipped (--skip-random-init).")

    # ---- Extract shared right basis U at K=128 (seed=42) ----
    print("\n[8] Extracting shared right basis U at K=128 ...")
    # torch.svd_lowrank returns (U, S, V) where V has shape [n, q] (right singular vectors).
    torch.manual_seed(42)
    T_stack = torch.from_numpy(M_stack.astype(np.float32))
    _, S_shared_t, V_shared_t = torch.svd_lowrank(T_stack, q=128, niter=4)
    U_shared = V_shared_t.numpy()  # [d, 128] — right singular vectors of M_stack
    S_shared = S_shared_t.numpy()
    # Approximation error check (Stage 5, Section 13, item 1).
    M_approx = (M_stack @ U_shared) @ U_shared.T
    approx_err = float(np.linalg.norm(M_stack - M_approx, "fro") / np.linalg.norm(M_stack, "fro"))
    print(f"  Approx error ||M_stack - U S V^T||_F / ||M_stack||_F = {approx_err:.4f}  "
          f"(should be < 0.05 for high-quality basis)")
    if approx_err > 0.05:
        print("  WARNING: approximation error > 0.05; basis may be imprecise.")
    del M_approx  # free memory

    # ---- Rank sweep: reconstruct W_Q^(ℓ) and evaluate ΔNLL ----
    print("\n[9] Rank sweep: K ∈ {32,64,128,256,512,1024,2048} ΔNLL ...")
    rank_results: list[dict[str, Any]] = []

    for K in K_VALUES:
        if K > d:
            continue
        t_k = time.time()
        if K == 2048:
            # Full rank — sanity check: should give ΔNLL ~ 0.
            for li, layer in enumerate(model.model.layers):
                layer.self_attn.q_proj.weight.data.copy_(original_wq[li])
        else:
            # Get shared basis at rank K.
            if K <= 128:
                # Re-use already computed basis truncated to K.
                U_k = U_shared[:, :K]  # [d, K]
            else:
                torch.manual_seed(42)
                _, _, V_k = torch.svd_lowrank(T_stack, q=K, niter=4)
                U_k = V_k.numpy()  # [d, K]

            for li, layer in enumerate(model.model.layers):
                W_approx = reconstruct_wq_from_shared_basis(original_wq[li], U_k)
                layer.self_attn.q_proj.weight.data.copy_(W_approx)

        nll_k, n_k = compute_loss(model, ids)
        loss_per_tok = nll_k / n_k
        delta_loss = loss_per_tok - base_loss_per_tok
        ppl_k = float(np.exp(loss_per_tok))

        # Storage arithmetic: shared U [d×K] + 28 C_ℓ [d×K] → (1+28)×d×K×2 bytes vs 28×d×d×2
        if K < d:
            shared_bytes = (1 + n_layers) * d * K * 2  # U + all C_ℓ
            orig_bytes = n_layers * d * d * 2
            compress = orig_bytes / shared_bytes if shared_bytes > 0 else float("inf")
        else:
            shared_bytes = n_layers * d * d * 2
            orig_bytes = shared_bytes
            compress = 1.0

        rec: dict[str, Any] = {
            "K": int(K),
            "K_frac_d": float(K / d),
            "n_eval_tokens": int(n_k),
            "loss_per_tok_nats": float(loss_per_tok),
            "ppl": float(ppl_k),
            "delta_loss_nats": float(delta_loss),
            "orig_bytes_wq_all_layers": int(orig_bytes),
            "shared_basis_bytes": int(shared_bytes),
            "compression_ratio": float(compress),
            "wall_clock_s": float(time.time() - t_k),
        }
        rank_results.append(rec)
        deployment_flag = " [DEPLOYMENT GO]" if delta_loss < 0.30 else ""
        print(
            f"  K={K:5d}  NLL/tok={loss_per_tok:.4f}  ΔNLL={delta_loss:+.4f}  "
            f"PPL={ppl_k:.3f}  compress={compress:.2f}x{deployment_flag}"
        )

    restore_wq()

    # ---- Verdict synthesis ----
    print("\n[10] Verdict synthesis ...")

    # Structural GO conditions.
    structural_go = (
        var128_mean > 0.80
        and var128_worst > 0.75
        and a2_pass
    )
    soft_go = (
        not structural_go
        and var128_mean >= 0.65
        and var128_mean <= 0.80
    )
    no_go = var128_mean < 0.50

    # Deployment GO: check ΔNLL < 0.30 at K=128.
    rec_128 = next((r for r in rank_results if r["K"] == 128), None)
    deployment_go = bool(rec_128 is not None and rec_128["delta_loss_nats"] < 0.30)

    if structural_go:
        verdict = "GO"
        verdict_detail = (
            f"var@128={var128_mean:.4f} > 0.80, gap_vs_baseline={gap_vs_baseline:.4f} > 0.20, "
            f"worst-of-3={var128_worst:.4f} > 0.75. "
            f"Deployment GO: {'YES' if deployment_go else 'NO (HOIST cascade not yet warranted)'}"
        )
    elif soft_go:
        verdict = "GRAY"
        verdict_detail = (
            f"var@128={var128_mean:.4f} in [0.65, 0.80]. "
            f"Cross-layer sharing needs D*=256. "
            f"Deployment GO at K=128: {'YES' if deployment_go else 'NO'}"
        )
    elif no_go:
        verdict = "NO-GO"
        verdict_detail = (
            f"var@128={var128_mean:.4f} < 0.50. "
            f"W_Q subspaces rotate independently across layers. "
            f"Class-level kill: Cluster C1 (R-CROSS-Q, C-CQBAL, F-CQSGC, A-GFRS). "
            f"CF-C1-KILL candidate for CF registry."
        )
    elif not a2_pass:
        verdict = "NO-GO"
        verdict_detail = (
            f"A2 BLOCKING FAILED: var@128={var128_mean:.4f} but gap vs matched-spectrum "
            f"baseline={gap_vs_baseline:.4f} <= 0.20. "
            f"Result explained by within-layer spectral concentration (CF11) being stacked, "
            f"not genuine cross-layer subspace alignment. "
            f"Class-level kill: Cluster C1."
        )
    else:
        verdict = "GRAY"
        verdict_detail = (
            f"var@128={var128_mean:.4f} in [0.50, 0.65] gray zone. "
            f"Elbow analysis needed at D*=256/512."
        )

    print(f"\n  VERDICT: {verdict}")
    print(f"  {verdict_detail}")

    # ---- Assemble full results ----
    t_total = time.time() - t_start
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_layers": n_layers,
        "d_hidden": d,
        "n_eval_tokens": n_tokens,
        "baseline_loss_per_tok": float(base_loss_per_tok),
        "baseline_ppl": float(base_ppl),
        "frobenius_norms_per_layer": norms.tolist(),
        "outlier_near_zero_layers": outlier_layers,
        # --- Stacked SVD structural result ---
        "var_profile": var_profile,
        "var128_mean_3seeds": float(var128_mean),
        "var128_std_3seeds": float(var128_std),
        "var128_worst_of_3": float(var128_worst),
        # --- A2 baseline ---
        "a2_matched_spectrum_baseline_var128_mean": float(var128_baseline_mean),
        "a2_matched_spectrum_baseline_var128_std": float(var128_baseline_std),
        "a2_gap_real_minus_baseline": float(gap_vs_baseline),
        "a2_pass": bool(a2_pass),
        # --- Permutation control ---
        "permutation_control_var128": float(var128_perm),
        "permutation_gap": float(gap_vs_perm),
        "permutation_control_pass": bool(perm_pass),
        # --- A1 random-init control ---
        "a1_random_init_var128": rand_init_var128,
        "a1_gap_real_minus_rand_init": gap_vs_rand_init,
        "a1_pass": rand_init_pass,
        # --- SVD approximation error ---
        "svd_approx_error_frob_ratio": float(approx_err),
        # --- Rank sweep ---
        "rank_sweep_results": rank_results,
        "deployment_go_at_k128": bool(deployment_go),
        # --- Final verdict ---
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        # --- Meta ---
        "wall_clock_total_s": float(t_total),
        "timestamp": datetime.now(UTC).isoformat(),
        "stage6_amendments_applied": [
            "A1: random-init uses normal_(std=0.02); raw var@128 reported",
            "A2 BLOCKING: matched-spectrum random-orientation baseline; GO requires gap > 0.20",
            "A3: deployment GO sub-threshold ΔNLL < 0.30 nat for HOIST cascade",
            "A4: COMPOT (arXiv:2602.15200) cited in kill criteria context",
        ],
        "references": {
            "COMPOT": "arXiv:2602.15200, Feb 2026 — single shared subspace degrades at moderate compression",
            "BasisSharing": "arXiv:2410.03765, ICLR 2025 — requires fine-tuning",
            "MASA": "arXiv:2508.04581, Aug 2025 — requires fine-tuning",
            "QSVD": "arXiv:2510.16292, NeurIPS 2025 — calibration-guided adaptive rank",
            "xKV": "arXiv:2503.18893, March 2025 — KV-Cache activations, not weights",
        },
    }

    out_json = args.output_dir / "fcqsgc_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Total wall-clock: {t_total/60:.1f} min")
    print(f"\nFINAL VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
