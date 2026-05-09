r"""AQFKV — Q-Head SVD with KV Preserved (Track A R3, Stage 6 amended).

Tests whether compound finding 8 (W_gate and W_up full-rank in trained Qwen3)
extends to attention W_Q. The attention object is the LAST untested weight class
on Qwen3.

Stage 6 amendments folded in:
  1. Pre-PPL singular-value spectrum dump at layers 0, mid, last (cheap signal).
  2. Mandatory K=full-rank sanity gate (K=2048 global, K=128 per-head).
  3. Global W_Q SVD sweep: K ∈ {2048, 1024, 512, 256, 128, 64, 32, 16}.
  4. Per-head W_Q SVD sweep: K_per_head ∈ {128, 96, 64, 32, 16, 8}.
  5. Cheap W_K global SVD sweep parallel measurement.
  6. max_length=512 to match R3/R4 (~455 tokens).

Stage 6 prior: P(NO-GO) ≈ 65% — extending CF8 to attention is the modal outcome.
A NO-GO is itself a strong structural finding (class-level kill on
no-retraining attention rank reduction). A GO at K_per_head=64 would be a
genuine breakthrough opening the attention compression line.

References:
  - QSVD (arXiv:2510.16292): joint [Q,K,V] SVD — different question.
  - A3 (arXiv:2505.12942): per-head QK product compression (different from W_Q alone).
  - arXiv:2410.23819: weight decay drives W_K^T W_Q low-rank, but individual factors
    need not be. Honesty note: we may be testing a structurally suboptimal slice.
  - R3 W_gate sweep (this repo): NO-GO at K≤256, ΔNLL +0.77 at K=1024 (50% rank).
  - R4 W_up sweep: NO-GO at K≤512, ΔNLL +2.34 at K=1024.
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
    "discipline as a backup against GPS jamming. The wayfinder's craft "
    "demonstrates a recurring pattern in human technology: the most "
    "sophisticated tools rest on the most ancient observations, and "
    "they remain useful long after a more powerful replacement has "
    "made them seem unnecessary. A sextant, used carefully, is "
    "accurate to about a tenth of a nautical mile under good "
    "conditions — a precision that is more than sufficient for any "
    "navigational task short of harbor approach, and that the "
    "underlying physics has not improved upon in two centuries."
)

GLOBAL_K_VALUES = (2048, 1024, 512, 256, 128, 64, 32, 16)
PERHEAD_K_VALUES = (128, 96, 64, 32, 16, 8)
WK_GLOBAL_K_VALUES = (1024, 512, 256, 128, 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--passage", default=EVAL_PASSAGE)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder/track_A_arch/round3_aqfkv"),
    )
    p.add_argument("--skip-wk", action="store_true",
                   help="Skip the W_K global sweep to save time.")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def compute_loss(model, ids: torch.Tensor) -> tuple[float, int]:
    """Return (sum_log_loss_in_nats, n_tokens) for greedy next-token loss."""
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    n = nll.numel()
    return float(nll.sum().item()), int(n)


def svd_rank_reconstruct(W: torch.Tensor, k: int) -> torch.Tensor:
    """Rank-K SVD reconstruction in fp32, returned in W's original dtype."""
    W_fp32 = W.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    k_eff = min(k, S.shape[0])
    W_k = (U[:, :k_eff] * S[:k_eff].unsqueeze(0)) @ Vt[:k_eff]
    return W_k.to(W.dtype)


def perhead_svd_reconstruct(
    W_q: torch.Tensor, num_heads: int, head_dim: int, k_per_head: int
) -> torch.Tensor:
    """Per-head rank-K reconstruction for W_q.

    W_q has shape [num_heads*head_dim, d_in]. Reshape into
    [num_heads, head_dim, d_in] heads-as-blocks; SVD each [head_dim, d_in]
    slice independently at rank k_per_head; reassemble.
    """
    d_out, d_in = W_q.shape
    assert d_out == num_heads * head_dim, (
        f"W_q rows {d_out} != num_heads*head_dim "
        f"({num_heads}*{head_dim}={num_heads*head_dim})"
    )
    out = torch.empty_like(W_q)
    for h in range(num_heads):
        slice_h = W_q[h * head_dim : (h + 1) * head_dim, :]
        out[h * head_dim : (h + 1) * head_dim, :] = svd_rank_reconstruct(
            slice_h, k_per_head
        )
    return out


def spectrum_summary(W: torch.Tensor, top_k: int = 50) -> dict[str, Any]:
    """Compute thin SVD singular values and key spectral statistics."""
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    s = S.cpu().numpy()
    cum = np.cumsum(s ** 2)
    total = cum[-1]
    cum_frac = cum / total
    return {
        "n_singvals": int(s.shape[0]),
        "top_50": s[: top_k].tolist(),
        "ratio_s1_s0": float(s[1] / s[0]) if s.shape[0] > 1 else float("nan"),
        "ratio_s_last_s0": float(s[-1] / s[0]) if s.shape[0] > 0 else float("nan"),
        "rank_for_99_var": int(np.searchsorted(cum_frac, 0.99) + 1),
        "rank_for_95_var": int(np.searchsorted(cum_frac, 0.95) + 1),
        "rank_for_90_var": int(np.searchsorted(cum_frac, 0.90) + 1),
        "var_at_r_64": float(cum_frac[min(63, len(cum_frac) - 1)]),
        "var_at_r_128": float(cum_frac[min(127, len(cum_frac) - 1)]),
        "var_at_r_256": float(cum_frac[min(255, len(cum_frac) - 1)]),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    d = int(cfg.hidden_size)
    n_q_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_q_heads))
    head_dim = int(getattr(cfg, "head_dim", d // n_q_heads))
    print(
        f"  n_layers={n_layers}  d={d}  n_q_heads={n_q_heads}  "
        f"n_kv_heads={n_kv_heads}  head_dim={head_dim}"
    )

    # First attention layer to inspect shapes.
    first_attn = model.model.layers[0].self_attn
    Wq0 = first_attn.q_proj.weight
    Wk0 = first_attn.k_proj.weight
    print(f"  W_Q[0] shape={tuple(Wq0.shape)}  dtype={Wq0.dtype}")
    print(f"  W_K[0] shape={tuple(Wk0.shape)}  dtype={Wk0.dtype}")

    encoded = tokenizer(
        args.passage,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    ids = encoded.input_ids.to(device)
    n_tokens = int(ids.shape[-1])
    print(f"  n_eval_tokens={n_tokens}")

    print("\nRunning baseline ...")
    base_nll, base_n = compute_loss(model, ids)
    base_ppl = float(np.exp(base_nll / base_n))
    base_loss_per_tok = base_nll / base_n
    print(f"  baseline: NLL/tok = {base_loss_per_tok:.4f}  PPL = {base_ppl:.3f}")

    # ---- Pre-PPL singular value spectra at layers 0, mid, last (cheap diagnostic) ----
    spectrum_layers = sorted({0, n_layers // 2, n_layers - 1})
    print(f"\nSpectrum diagnostic on W_Q at layers {spectrum_layers} ...")
    spectra: dict[str, Any] = {}
    for li in spectrum_layers:
        Wq_li = model.model.layers[li].self_attn.q_proj.weight
        spec = spectrum_summary(Wq_li)
        spectra[f"W_Q.layer{li}"] = spec
        print(
            f"  W_Q L{li}: r_99={spec['rank_for_99_var']}  "
            f"r_95={spec['rank_for_95_var']}  r_90={spec['rank_for_90_var']}  "
            f"var@r=128={spec['var_at_r_128']:.4f}  "
            f"var@r=256={spec['var_at_r_256']:.4f}"
        )
    if not args.skip_wk:
        for li in spectrum_layers:
            Wk_li = model.model.layers[li].self_attn.k_proj.weight
            spec = spectrum_summary(Wk_li)
            spectra[f"W_K.layer{li}"] = spec
            print(
                f"  W_K L{li}: r_99={spec['rank_for_99_var']}  "
                f"r_95={spec['rank_for_95_var']}  "
                f"var@r=128={spec['var_at_r_128']:.4f}"
            )

    # Save originals once (Q and K).
    print("\nSaving baseline W_Q / W_K weights ...")
    original_q: list[torch.Tensor] = [
        layer.self_attn.q_proj.weight.detach().clone()
        for layer in model.model.layers
    ]
    original_k: list[torch.Tensor] = [
        layer.self_attn.k_proj.weight.detach().clone()
        for layer in model.model.layers
    ]

    def restore_q() -> None:
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.weight.data.copy_(original_q[li])

    def restore_k() -> None:
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.k_proj.weight.data.copy_(original_k[li])

    # ---- Sweep 1: Global W_Q SVD ----
    global_results: list[dict[str, Any]] = []
    print("\n=== Sweep 1: Global W_Q SVD ===")
    for k in GLOBAL_K_VALUES:
        if k > min(Wq0.shape):
            continue
        # Replace.
        for li, layer in enumerate(model.model.layers):
            W_orig = original_q[li]
            W_k = svd_rank_reconstruct(W_orig, k)
            layer.self_attn.q_proj.weight.data.copy_(W_k)
        nll_k, n_k = compute_loss(model, ids)
        ppl_k = float(np.exp(nll_k / n_k))
        loss_per_tok = nll_k / n_k
        delta_loss = loss_per_tok - base_loss_per_tok
        out_bytes = (Wq0.shape[0] + Wq0.shape[1]) * k * 2
        orig_bytes = Wq0.shape[0] * Wq0.shape[1] * 2
        comp = orig_bytes / out_bytes if out_bytes else float("inf")
        rec = {
            "K": int(k),
            "K_frac_d": float(k / d),
            "n_eval_tokens": int(n_k),
            "loss_per_tok_nats": loss_per_tok,
            "ppl": ppl_k,
            "delta_loss_nats": delta_loss,
            "delta_ppl": ppl_k - base_ppl,
            "wq_bytes_per_layer_orig": orig_bytes,
            "wq_bytes_per_layer_rankK": out_bytes,
            "compression_ratio_vs_bf16": comp,
        }
        global_results.append(rec)
        print(
            f"  K={k:5d}  NLL/tok={loss_per_tok:.4f}  ΔNLL={delta_loss:+.4f}  "
            f"PPL={ppl_k:.3f}  compress={comp:.2f}×"
        )
    restore_q()

    # Sanity gate: at K=2048 (full rank), ΔNLL should be ~0.
    sanity_rec = next((r for r in global_results if r["K"] == 2048), None)
    sanity_ok = sanity_rec is not None and abs(sanity_rec["delta_loss_nats"]) < 0.05
    if sanity_rec is not None:
        print(
            f"\n  K=2048 sanity gate: ΔNLL={sanity_rec['delta_loss_nats']:+.4f} "
            f"({'OK' if sanity_ok else 'FAILED'})"
        )

    # ---- Sweep 2: Per-Head W_Q SVD ----
    perhead_results: list[dict[str, Any]] = []
    print("\n=== Sweep 2: Per-Head W_Q SVD ===")
    for kph in PERHEAD_K_VALUES:
        if kph > head_dim:
            continue
        for li, layer in enumerate(model.model.layers):
            W_orig = original_q[li]
            W_kph = perhead_svd_reconstruct(W_orig, n_q_heads, head_dim, kph)
            layer.self_attn.q_proj.weight.data.copy_(W_kph)
        nll_k, n_k = compute_loss(model, ids)
        ppl_k = float(np.exp(nll_k / n_k))
        loss_per_tok = nll_k / n_k
        delta_loss = loss_per_tok - base_loss_per_tok
        # bytes: per head [head_dim + d] * kph * 2; n_q_heads heads.
        out_bytes_perhead = (head_dim + d) * kph * 2 * n_q_heads
        orig_bytes = Wq0.shape[0] * Wq0.shape[1] * 2
        comp = orig_bytes / out_bytes_perhead if out_bytes_perhead else float("inf")
        rec = {
            "K_per_head": int(kph),
            "K_frac_head_dim": float(kph / head_dim),
            "n_eval_tokens": int(n_k),
            "loss_per_tok_nats": loss_per_tok,
            "ppl": ppl_k,
            "delta_loss_nats": delta_loss,
            "delta_ppl": ppl_k - base_ppl,
            "wq_bytes_per_layer_orig": orig_bytes,
            "wq_bytes_per_layer_perheadK": out_bytes_perhead,
            "compression_ratio_vs_bf16": comp,
        }
        perhead_results.append(rec)
        print(
            f"  K_per_head={kph:3d}  NLL/tok={loss_per_tok:.4f}  "
            f"ΔNLL={delta_loss:+.4f}  PPL={ppl_k:.3f}  compress={comp:.2f}×"
        )
    restore_q()

    # Per-head sanity at K=128 (full).
    perhead_sanity = next(
        (r for r in perhead_results if r["K_per_head"] == 128), None
    )
    perhead_sanity_ok = (
        perhead_sanity is not None
        and abs(perhead_sanity["delta_loss_nats"]) < 0.05
    )
    if perhead_sanity is not None:
        print(
            f"\n  K_per_head=128 sanity gate: "
            f"ΔNLL={perhead_sanity['delta_loss_nats']:+.4f} "
            f"({'OK' if perhead_sanity_ok else 'FAILED'})"
        )

    # ---- Sweep 3: Global W_K SVD (cheap parallel measurement) ----
    wk_results: list[dict[str, Any]] = []
    if not args.skip_wk:
        print("\n=== Sweep 3: Global W_K SVD ===")
        for k in WK_GLOBAL_K_VALUES:
            if k > min(Wk0.shape):
                continue
            for li, layer in enumerate(model.model.layers):
                W_orig = original_k[li]
                W_k = svd_rank_reconstruct(W_orig, k)
                layer.self_attn.k_proj.weight.data.copy_(W_k)
            nll_k, n_k = compute_loss(model, ids)
            ppl_k = float(np.exp(nll_k / n_k))
            loss_per_tok = nll_k / n_k
            delta_loss = loss_per_tok - base_loss_per_tok
            wk_results.append(
                {
                    "K": int(k),
                    "n_eval_tokens": int(n_k),
                    "loss_per_tok_nats": loss_per_tok,
                    "ppl": ppl_k,
                    "delta_loss_nats": delta_loss,
                }
            )
            print(
                f"  W_K K={k:4d}  NLL/tok={loss_per_tok:.4f}  "
                f"ΔNLL={delta_loss:+.4f}"
            )
        restore_k()

    # ---- Verdict synthesis ----
    global_at_128 = next((r for r in global_results if r["K"] == 128), None)
    global_at_256 = next((r for r in global_results if r["K"] == 256), None)
    perhead_at_64 = next(
        (r for r in perhead_results if r["K_per_head"] == 64), None
    )
    perhead_at_96 = next(
        (r for r in perhead_results if r["K_per_head"] == 96), None
    )

    global_verdict: str
    if global_at_128 and global_at_128["delta_loss_nats"] <= 1.0:
        global_verdict = "GO (W_Q is low-rank tolerant at K≤128 globally)"
    elif global_at_256 and global_at_256["delta_loss_nats"] > 5.0:
        global_verdict = "NO-GO (W_Q is full-rank globally; CF8 extends to attention Q)"
    else:
        global_verdict = "GRAY (mid-zone; layer-sensitivity follow-up indicated)"

    perhead_verdict: str
    if perhead_at_64 and perhead_at_64["delta_loss_nats"] <= 1.0:
        perhead_verdict = "GO-perhead (per-head Q is compressible at K_per_head≤64)"
    elif perhead_at_96 and perhead_at_96["delta_loss_nats"] > 5.0:
        perhead_verdict = "NO-GO-perhead (per-head Q is full-rank)"
    else:
        perhead_verdict = "GRAY-perhead (mid-zone)"

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_eval_tokens": n_tokens,
        "n_layers": n_layers,
        "d_hidden": d,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "baseline_loss_per_tok": base_loss_per_tok,
        "baseline_ppl": base_ppl,
        "global_verdict": global_verdict,
        "perhead_verdict": perhead_verdict,
        "global_results": global_results,
        "perhead_results": perhead_results,
        "wk_results": wk_results,
        "spectra": spectra,
        "sanity_gate_global_K2048_ok": bool(sanity_ok),
        "sanity_gate_perhead_K128_ok": bool(perhead_sanity_ok),
        "timestamp": datetime.now(UTC).isoformat(),
        "stage6_amendments": [
            "global+per-head sweeps to disambiguate weight-rank vs head-redundancy",
            "K=2048 and K_per_head=128 sanity gates",
            "max_length=512 to match R3/R4 token count (~455)",
            "pre-PPL singular value spectrum at layers 0, mid, last",
            "cheap W_K parallel sweep for second structural finding",
        ],
        "cf9_preconditions": [
            "no calibration fitting (pure SVD reconstruction); WDLA risk N/A",
            "Eckart-Young guarantees rank-K is best Frobenius approx",
            "per-head SVD is the structurally correct test of head-level Q rank",
            "global SVD GO at K=128 conflates head-redundancy with weight-rank",
        ],
    }

    out_json = args.output_dir / "aqfkv_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Plot 1: Global W_Q PPL vs K
        ax = axes[0]
        gks = [r["K"] for r in global_results]
        gppls = [r["ppl"] for r in global_results]
        ax.semilogx(gks, gppls, marker="o", label="W_Q rank-K")
        ax.axhline(base_ppl, color="black", linestyle="--", alpha=0.7,
                   label=f"baseline {base_ppl:.2f}")
        ax.set_xlabel("rank K (global SVD)")
        ax.set_ylabel("PPL")
        ax.set_title("Global W_Q SVD")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Per-head W_Q PPL vs K_per_head
        ax = axes[1]
        phks = [r["K_per_head"] for r in perhead_results]
        phppls = [r["ppl"] for r in perhead_results]
        ax.plot(phks, phppls, marker="s", color="firebrick", label="per-head rank")
        ax.axhline(base_ppl, color="black", linestyle="--", alpha=0.7,
                   label=f"baseline {base_ppl:.2f}")
        ax.set_xlabel("K per head")
        ax.set_ylabel("PPL")
        ax.set_title("Per-Head W_Q SVD")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: spectrum top-50 at three layers
        ax = axes[2]
        for layer_label, spec in spectra.items():
            if not layer_label.startswith("W_Q"):
                continue
            top50 = spec["top_50"]
            ax.semilogy(range(1, len(top50) + 1), top50, label=layer_label, alpha=0.8)
        ax.set_xlabel("singular value index")
        ax.set_ylabel("singular value (log)")
        ax.set_title("W_Q top-50 singular values")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = args.output_dir / "aqfkv_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # summary.md
    md_lines = [
        f"# AQFKV — Q-Head SVD with KV Preserved — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Eval: {n_tokens} tokens; n_layers={n_layers}, "
        f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}",
        f"Baseline: NLL/tok = {base_loss_per_tok:.4f}, PPL = {base_ppl:.3f}",
        "",
        f"## Global verdict: **{global_verdict}**",
        f"## Per-head verdict: **{perhead_verdict}**",
        "",
        f"Sanity gates: global K=2048 {'PASS' if sanity_ok else 'FAIL'}; "
        f"per-head K=128 {'PASS' if perhead_sanity_ok else 'FAIL'}",
        "",
        "## Sweep 1 — Global W_Q SVD",
        "",
        "| K | K/d | PPL | ΔNLL (nats) | compress |",
        "|---|---|---|---|---|",
    ]
    for rec in global_results:
        md_lines.append(
            f"| {rec['K']} | {rec['K_frac_d']:.3f} | {rec['ppl']:.3f} | "
            f"{rec['delta_loss_nats']:+.4f} | "
            f"{rec['compression_ratio_vs_bf16']:.2f}x |"
        )
    md_lines += [
        "",
        "## Sweep 2 — Per-Head W_Q SVD",
        "",
        "| K_per_head | K/head_dim | PPL | ΔNLL (nats) | compress |",
        "|---|---|---|---|---|",
    ]
    for rec in perhead_results:
        md_lines.append(
            f"| {rec['K_per_head']} | {rec['K_frac_head_dim']:.3f} | "
            f"{rec['ppl']:.3f} | {rec['delta_loss_nats']:+.4f} | "
            f"{rec['compression_ratio_vs_bf16']:.2f}x |"
        )
    if wk_results:
        md_lines += [
            "",
            "## Sweep 3 — Global W_K SVD (parallel measurement)",
            "",
            "| K | PPL | ΔNLL (nats) |",
            "|---|---|---|",
        ]
        for rec in wk_results:
            md_lines.append(
                f"| {rec['K']} | {rec['ppl']:.3f} | "
                f"{rec['delta_loss_nats']:+.4f} |"
            )
    md_lines += [
        "",
        "## Spectrum diagnostic (rank for 99% / 95% / 90% variance)",
        "",
        "| Matrix | r_99 | r_95 | r_90 | var@128 | var@256 |",
        "|---|---|---|---|---|---|",
    ]
    for label, spec in spectra.items():
        md_lines.append(
            f"| {label} | {spec['rank_for_99_var']} | {spec['rank_for_95_var']} "
            f"| {spec['rank_for_90_var']} | {spec['var_at_r_128']:.4f} | "
            f"{spec['var_at_r_256']:.4f} |"
        )
    md_lines += [
        "",
        "## Notes",
        "",
        "- Pure SVD reconstruction (no calibration fitting); WDLA ill-conditioning N/A.",
        "- W_K, W_V, W_O held at bf16 throughout.",
        "- Stage 6 critical amendment: per-head SVD distinguishes weight-rank from "
        "head-redundancy. A global GO at K=128 with a per-head NO-GO at "
        "K_per_head=64 implies head-sharing, NOT genuine W_Q low-rank structure.",
        "",
        "## References",
        "",
        "- QSVD (arXiv:2510.16292): joint [Q,K,V] SVD on VLMs.",
        "- A3 (arXiv:2505.12942): per-head QK product compression.",
        "- arXiv:2410.23819: weight decay drives W_K^T W_Q low-rank, but factors not "
        "individually.",
        "- R3 / R4: W_gate and W_up full-rank in Qwen3-1.7B (CF8).",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nGLOBAL VERDICT: {global_verdict}")
    print(f"PER-HEAD VERDICT: {perhead_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
