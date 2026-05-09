r"""First-Principles E4 — Rotated Tied lm_head Binary Settling Experiment.

The single highest-information experiment in Round 1. Tests whether the +19.96 NLL
catastrophe at r=1024 (LHQD R6 result) DISSOLVES when truncation is performed in the
joint Q* basis (FP E3 output) rather than the ambient basis.

Comparison conditions (all evaluated on the same held-out passages):
  (1) baseline      — full lm_head, no truncation; ΔNLL = 0
  (2) ambient r=1024 — truncate residual to first 1024 ambient coords; expect +19.96
  (3) random Q r=1024 — truncate in random orthogonal basis; expect ≈ +19.96
  (4) Q* r=1024     — truncate in joint Q* basis; PREDICT ≪ +1.0
  (5) Q* r=512      — same with tighter truncation
  (6) Q* r=256      — same with even tighter truncation

The mechanism: truncate the residual stream entering lm_head to its first r coordinates
in the rotated (Q*) basis. Mathematically:
  logits_ambient = W_E · h
  logits_truncated = W_E · proj_r(h, basis)
where proj_r in basis B is: B[:, :r] @ B[:, :r]^T @ h.

For B = I (ambient): keeps first r coords of h directly.
For B = Q* (joint basis): keeps top-r rotated coords; equivalent to W_E · Q* · diag([1]*r + [0]*(d-r)) · Q*^T · h.

CRITICAL CF9 NOTE: This experiment intentionally ignores the full RMSNorm absorption
chain from First-Principles v2 §4. The simplified version tests whether truncation in
Q* basis ALONE preserves logit ranking. If even this proxy fails, the full paired-fusion
won't work either. If it succeeds, the full paired-fusion is the next experiment.

The experiment is BINARY: predict ΔNLL ≪ +1.0 for condition (4); observed +19.96 for
condition (2/3) is the established catastrophe baseline.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


PASSAGE_NAVIGATION = (
    "The art of celestial navigation rests on a small handful of "
    "instruments and a great deal of patience. A sextant measures the "
    "angle between a celestial body and the horizon; a chronometer "
    "marks Greenwich Mean Time; a nautical almanac gives the precise "
    "position of the sun, moon, and major stars at every hour of every day."
)
PASSAGE_BIOLOGY = (
    "The cytoskeleton is a dynamic network of protein filaments inside "
    "every eukaryotic cell. Three principal classes of filaments — actin "
    "microfilaments, intermediate filaments, and microtubules — give the "
    "cell its shape, partition organelles, and provide tracks along which "
    "motor proteins haul vesicles."
)
PASSAGE_TECHNICAL = (
    "Quicksort is an in-place comparison sort whose average-case running "
    "time is O(n log n) and whose worst-case running time is O(n²). "
    "The algorithm partitions the input around a pivot element, "
    "recursively sorts the two partitions, and combines them implicitly."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--q-star-path",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/fp_e3_joint_qstar/q_star.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/fp_e4_rotated_lmhead"),
    )
    p.add_argument("--ranks", default="2048,1024,512,256",
                   help="Truncation ranks to test")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def hooked_forward_with_lm_head_proj(
    model, ids: torch.Tensor, projection_matrix: torch.Tensor | None
) -> tuple[float, int]:
    """Run forward; just before lm_head, apply (optional) projection to the
    post-RMSNorm hidden state.

    We monkey-patch the lm_head call by capturing model.model output and
    applying lm_head ourselves with the projected hidden state.
    """
    # Trick: call model with output_hidden_states=True to get the post-norm
    # hidden state at the last position.
    out = model.model(input_ids=ids, use_cache=False)
    h = out.last_hidden_state  # [B, T, d] — already post final-RMSNorm

    if projection_matrix is not None:
        # h_proj = projection_matrix @ h
        # projection_matrix has shape [d, d]
        h_flat = h.reshape(-1, h.shape[-1])  # [B*T, d]
        h_proj_flat = h_flat @ projection_matrix.T  # [B*T, d]; row vectors
        # Equivalent to: h_proj[i] = projection @ h[i] for each row vector
        h = h_proj_flat.reshape(h.shape)

    logits = model.lm_head(h)  # [B, T, V]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(nll.sum().item()), int(nll.numel())


def build_projection(basis: torch.Tensor, r: int) -> torch.Tensor:
    """Return projection matrix P = basis[:, :r] @ basis[:, :r].T (rank-r, in given basis)."""
    B_top = basis[:, :r]
    return (B_top @ B_top.T).contiguous()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(",")]

    if not args.q_star_path.exists():
        print(f"ERROR: Q* not found at {args.q_star_path}")
        print("Run fp_e3_joint_qstar.py first.")
        return 1

    print(f"Loading Q* from {args.q_star_path} ...")
    q_star_data = torch.load(args.q_star_path, weights_only=False)
    Q_star = q_star_data["Q_star"]  # [d, d]
    print(f"  Q* shape: {tuple(Q_star.shape)}")

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    cfg = model.config
    d = int(cfg.hidden_size)
    print(f"  d={d}")
    Q_star = Q_star.to(torch.float32)

    # Build a random orthogonal Q for control comparison
    torch.manual_seed(args.seed)
    M = torch.randn(d, d, dtype=torch.float32)
    Q_random, _ = torch.linalg.qr(M)
    print(f"  Q_random shape: {tuple(Q_random.shape)}")

    passages = {
        "primary": PASSAGE_NAVIGATION + " " + PASSAGE_BIOLOGY + " " + PASSAGE_TECHNICAL,
        "navigation": PASSAGE_NAVIGATION,
        "biology": PASSAGE_BIOLOGY,
        "technical": PASSAGE_TECHNICAL,
    }
    encoded_ids = {
        name: tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length).input_ids.to(args.device)
        for name, text in passages.items()
    }
    primary_ids = encoded_ids["primary"]
    n_primary = int(primary_ids.shape[-1])
    print(f"  primary tokens: {n_primary}")

    # Identity basis for "ambient" condition
    I_basis = torch.eye(d, dtype=torch.float32)

    # Conditions to test
    conditions: list[dict[str, Any]] = []

    # (1) Baseline: no projection
    print("\nRunning baseline (no projection)...")
    base_nll, base_n = hooked_forward_with_lm_head_proj(model, primary_ids, None)
    base_loss_per_tok = base_nll / base_n
    base_ppl = float(np.exp(base_loss_per_tok))
    print(f"  baseline NLL/tok: {base_loss_per_tok:.4f}  PPL: {base_ppl:.3f}")
    conditions.append({
        "label": "baseline",
        "basis": "none",
        "r": d,
        "nll_per_tok": base_loss_per_tok,
        "ppl": base_ppl,
        "delta_nll": 0.0,
    })

    # Run truncation conditions
    for r in ranks:
        if r > d:
            continue
        if r == d:
            print(f"\nSkipping r={r} (full rank); already covered by baseline")
            continue

        # (a) Ambient basis truncation
        P_ambient = build_projection(I_basis, r)
        nll_amb, n_amb = hooked_forward_with_lm_head_proj(model, primary_ids, P_ambient)
        lp_amb = nll_amb / n_amb
        delta_amb = lp_amb - base_loss_per_tok
        print(f"\nr={r}:")
        print(f"  AMBIENT  ΔNLL={delta_amb:+.4f}  PPL={float(np.exp(lp_amb)):.3e}")
        conditions.append({
            "label": f"ambient_r{r}",
            "basis": "ambient",
            "r": r,
            "nll_per_tok": lp_amb,
            "ppl": float(np.exp(lp_amb)),
            "delta_nll": delta_amb,
        })

        # (b) Random Q basis truncation
        P_random = build_projection(Q_random, r)
        nll_rnd, n_rnd = hooked_forward_with_lm_head_proj(model, primary_ids, P_random)
        lp_rnd = nll_rnd / n_rnd
        delta_rnd = lp_rnd - base_loss_per_tok
        print(f"  RANDOM Q ΔNLL={delta_rnd:+.4f}  PPL={float(np.exp(lp_rnd)):.3e}")
        conditions.append({
            "label": f"random_q_r{r}",
            "basis": "random_q",
            "r": r,
            "nll_per_tok": lp_rnd,
            "ppl": float(np.exp(lp_rnd)),
            "delta_nll": delta_rnd,
        })

        # (c) Joint Q* basis truncation — THE LOAD-BEARING TEST
        P_qstar = build_projection(Q_star, r)
        nll_qs, n_qs = hooked_forward_with_lm_head_proj(model, primary_ids, P_qstar)
        lp_qs = nll_qs / n_qs
        delta_qs = lp_qs - base_loss_per_tok
        print(f"  JOINT Q* ΔNLL={delta_qs:+.4f}  PPL={float(np.exp(lp_qs)):.3e}")
        conditions.append({
            "label": f"q_star_r{r}",
            "basis": "q_star",
            "r": r,
            "nll_per_tok": lp_qs,
            "ppl": float(np.exp(lp_qs)),
            "delta_nll": delta_qs,
        })

        # Multi-passage robustness on Q* condition only
        multi = {}
        for pname in ["navigation", "biology", "technical"]:
            ids_p = encoded_ids[pname]
            base_n_p, _ = hooked_forward_with_lm_head_proj(model, ids_p, None)
            qs_n_p, _ = hooked_forward_with_lm_head_proj(model, ids_p, P_qstar)
            n_tok = int(ids_p.shape[-1]) - 1
            base_lp_p = base_n_p / n_tok
            qs_lp_p = qs_n_p / n_tok
            multi[pname] = {
                "delta_nll_q_star": qs_lp_p - base_lp_p,
            }
        conditions[-1]["multi_passage"] = multi

    # Verdict
    # Find the best Q* result and corresponding ambient/random comparator
    best_qs = None
    best_qs_delta = float("inf")
    for c in conditions:
        if c["basis"] == "q_star" and c["delta_nll"] < best_qs_delta:
            best_qs = c
            best_qs_delta = c["delta_nll"]

    if best_qs is None:
        verdict = "ERROR — no Q* condition ran"
    else:
        # Find matching ambient at same r
        match_amb = next((c for c in conditions if c["basis"] == "ambient" and c["r"] == best_qs["r"]), None)
        ambient_delta = match_amb["delta_nll"] if match_amb else float("nan")

        if best_qs["delta_nll"] < 1.0:
            verdict = (f"GO — joint Q* DISSOLVES catastrophe. At r={best_qs['r']}, "
                       f"ambient ΔNLL={ambient_delta:.2f} vs Q* ΔNLL={best_qs['delta_nll']:.4f}. "
                       f"First-Principles geometric-fingerprint hypothesis SUPPORTED.")
        elif best_qs["delta_nll"] < ambient_delta * 0.3:
            verdict = (f"PARTIAL — Q* gives major reduction (ΔNLL={best_qs['delta_nll']:.2f} vs "
                       f"ambient {ambient_delta:.2f}, {ambient_delta/best_qs['delta_nll']:.1f}× better) "
                       f"but doesn't dissolve fully; mechanism partially valid.")
        else:
            verdict = (f"NO-GO — Q* does NOT dissolve catastrophe. At r={best_qs['r']}, "
                       f"ambient ΔNLL={ambient_delta:.2f} vs Q* ΔNLL={best_qs['delta_nll']:.2f}. "
                       f"First-Principles geometric-fingerprint hypothesis FAILS at "
                       f"tied lm_head — embedding's information mass occupies different "
                       f"subspace than FFN matrices.")

    results = {
        "model_id": args.model_id,
        "n_primary_tokens": n_primary,
        "d_model": d,
        "ranks_tested": ranks,
        "baseline_nll_per_tok": base_loss_per_tok,
        "baseline_ppl": base_ppl,
        "conditions": conditions,
        "verdict": verdict,
        "q_star_source": str(args.q_star_path),
        "stage6_caveat": (
            "This is a SIMPLIFIED proxy of First-Principles v2 §4 paired-fusion. "
            "The full mechanism would also rotate every linear layer's weights via Q* "
            "and absorb RMSNorm γ. This simplified version tests whether Q* basis "
            "preserves logit ranking under truncation as a first-pass binary settler. "
            "If GO here, full paired-fusion is the next experiment. If NO-GO, full "
            "version unlikely to work."
        ),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    out_json = args.output_dir / "fp_e4_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    print(f"\n=========================================")
    print(f"VERDICT: {verdict}")
    print(f"=========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
