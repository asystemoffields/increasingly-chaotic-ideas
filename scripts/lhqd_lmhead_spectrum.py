r"""LHQD First-Pass — SVD spectrum of lm_head.weight (Track B R6, Stage 6 amended).

Tests whether compound finding 8 (W_gate / W_up full-rank in trained Qwen3)
generalizes to lm_head, OR whether lm_head has a structurally distinct
concentrated spectrum, opening LHQD (INT8 U + BF16 V^T) and SVLD (progressive
SVD with confidence-gated early exit).

Stage 6 amendments folded in:
  0. Tied-embedding hard gate: halt if lm_head shares storage with embed_tokens.
  1. Per-row L2 norm histogram (V>>d dead-row confound check).
  2. Top-1 / top-5 argmax retention per rank (deployment-relevant metric).
  3. Per-token KL divergence vs full-rank.
  4. Multi-passage NLL check (3 disjoint 170-token passages).
  5. Sanity gate at r=2048 (full thin-SVD; round-trip should be <0.005 nats).
  6. Optional W_Q spectrum comparison from one mid-layer (~3 min cost).

Stage 6 priors:
  - STRONG GO: 35%  (concentrated spectrum + retained argmax)
  - WEAK GO: 30%
  - INTERESTING FINDING (spectral pass, functional flag): 20%
  - NO-GO (extends CF8): 15%

References:
  - arXiv:2603.10145 (Godey & Artzi, Mar 2026): lm_head is a gradient bottleneck;
    training dynamics suppress effective rank during backprop. LHQD empirically tests
    whether that training prediction manifests as a concentrated weight spectrum.
  - arXiv:2510.24966 (Sequences of Logits...): logit *function* low-rank theorem.
    CF9 precondition: this does NOT directly imply weight low-rank — must measure W.
  - SVD-LLM (ICLR 2025), ARA (arXiv:2510.19389): global SVD compression; do NOT
    report isolated lm_head spectrum on Qwen family.
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


# Three disjoint passages for multi-passage NLL check (Stage 6 amendment).
PASSAGE_NAVIGATION = (
    "The art of celestial navigation rests on a small handful of "
    "instruments and a great deal of patience. A sextant measures the "
    "angle between a celestial body and the horizon; a chronometer "
    "marks Greenwich Mean Time; a nautical almanac gives the precise "
    "position of the sun, moon, and major stars at every hour of every "
    "day. With these in hand, a navigator can determine latitude from a "
    "noon sun sight, and longitude from a timed observation of any "
    "celestial body, anywhere on the open ocean."
)

PASSAGE_BIOLOGY = (
    "The cytoskeleton is a dynamic network of protein filaments inside "
    "every eukaryotic cell. Three principal classes of filaments — "
    "actin microfilaments, intermediate filaments, and microtubules — "
    "give the cell its shape, partition organelles, and provide the "
    "tracks along which motor proteins haul vesicles. Microtubules in "
    "particular grow and shrink at their plus ends in a process called "
    "dynamic instability, which allows the mitotic spindle to reorganize "
    "in seconds during each cell division."
)

PASSAGE_TECHNICAL = (
    "Quicksort is an in-place comparison sort whose average-case running "
    "time is O(n log n) and whose worst-case running time is O(n^2). "
    "The algorithm partitions the input around a pivot element, "
    "recursively sorts the two partitions, and combines them implicitly "
    "by virtue of having sorted in place. Pivot selection strategies "
    "include first-element, last-element, median-of-three, and randomized; "
    "randomized pivots make the worst case asymptotically negligible at "
    "the cost of a small expected-case overhead."
)

# Primary 512-token passage (concatenation, used as the main NLL probe).
PRIMARY_PASSAGE = (
    PASSAGE_NAVIGATION
    + " "
    + PASSAGE_BIOLOGY
    + " "
    + PASSAGE_TECHNICAL
    + " "
    + "Across all three vignettes — celestial navigation, cellular "
    "biology, sorting algorithms — the same epistemic posture recurs: "
    "the practitioner reasons backward from observable consequences to "
    "underlying causes. The sailor reads the sun and stars to fix his "
    "position; the cell biologist reads filament dynamics to infer "
    "regulatory signals; the programmer reads runtime traces to infer "
    "the structure of the input. In each case, abstraction makes the "
    "practitioner's life tractable but also encodes assumptions that, "
    "when violated, produce predictably surprising failures."
)

R_VALUES = (2048, 1024, 512, 256, 128, 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder/track_B_compress/round6_lhqd"),
    )
    p.add_argument("--with-wq-comparison", action="store_true",
                   help="Also compute W_Q spectrum from one mid-layer.")
    return p.parse_args()


@torch.inference_mode()
def compute_loss_and_logits(
    model, ids: torch.Tensor
) -> tuple[float, int, torch.Tensor]:
    """Return (sum_log_loss_in_nats, n_tokens, log_probs).

    log_probs has shape (T-1, V) for the shifted prediction positions.
    """
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    n = nll.numel()
    return float(nll.sum().item()), int(n), log_probs.squeeze(0).detach().cpu()


def reconstruct_at_rank(
    U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, r: int, dtype: torch.dtype
) -> torch.Tensor:
    r_eff = min(r, S.shape[0])
    W_r = (U[:, :r_eff] * S[:r_eff].unsqueeze(0)) @ Vt[:r_eff]
    return W_r.to(dtype)


def spectrum_summary(S_np: np.ndarray) -> dict[str, Any]:
    cum = np.cumsum(S_np ** 2)
    total = cum[-1]
    cum_frac = cum / total
    return {
        "n_singvals": int(S_np.shape[0]),
        "top_50": S_np[:50].tolist(),
        "ratio_s1_s0": float(S_np[1] / S_np[0]) if S_np.shape[0] > 1 else float("nan"),
        "rank_for_99_var": int(np.searchsorted(cum_frac, 0.99) + 1),
        "rank_for_95_var": int(np.searchsorted(cum_frac, 0.95) + 1),
        "rank_for_90_var": int(np.searchsorted(cum_frac, 0.90) + 1),
        "rank_for_999_var": int(np.searchsorted(cum_frac, 0.999) + 1),
        "var_at_r_64": float(cum_frac[min(63, len(cum_frac) - 1)]),
        "var_at_r_128": float(cum_frac[min(127, len(cum_frac) - 1)]),
        "var_at_r_256": float(cum_frac[min(255, len(cum_frac) - 1)]),
        "var_at_r_512": float(cum_frac[min(511, len(cum_frac) - 1)]),
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
    print(f"  n_layers={n_layers}  d={d}")

    # ---- Step 0: Tied-embedding hard gate ----
    tie_flag = bool(getattr(cfg, "tie_word_embeddings", False))
    lmhead_ptr = model.lm_head.weight.data_ptr()
    embed_ptr = model.model.embed_tokens.weight.data_ptr()
    storage_tied = lmhead_ptr == embed_ptr
    print(
        f"\nStep 0 — Tied-embedding gate:\n"
        f"  config.tie_word_embeddings = {tie_flag}\n"
        f"  storage tied (data_ptr match) = {storage_tied}"
    )
    if tie_flag or storage_tied:
        print(
            "  GATE TRIPPED: lm_head is tied to embed_tokens. The experiment "
            "measures the embedding matrix, not an independent output projection. "
            "Proceeding but flagging as 'embed_tokens spectrum study' rather than LHQD."
        )

    # Get lm_head W (151936, 2048).
    W_lmhead = model.lm_head.weight
    V, d_check = W_lmhead.shape
    print(f"  lm_head shape = ({V}, {d_check})  dtype={W_lmhead.dtype}")
    assert d_check == d, f"hidden mismatch: {d_check} vs {d}"

    # Eval token IDs.
    encoded = tokenizer(
        PRIMARY_PASSAGE, return_tensors="pt", truncation=True,
        max_length=args.max_length,
    )
    primary_ids = encoded.input_ids.to(device)
    n_primary = int(primary_ids.shape[-1])
    print(f"  primary passage tokens={n_primary}")

    multi_pass_ids: dict[str, torch.Tensor] = {}
    for label, txt in [
        ("nav", PASSAGE_NAVIGATION),
        ("bio", PASSAGE_BIOLOGY),
        ("tech", PASSAGE_TECHNICAL),
    ]:
        e = tokenizer(txt, return_tensors="pt", truncation=True,
                      max_length=200)
        multi_pass_ids[label] = e.input_ids.to(device)
        print(f"  {label} passage tokens={int(e.input_ids.shape[-1])}")

    # Baseline NLL on primary passage and capture log_probs (for KL/argmax later).
    print("\nBaseline forward ...")
    base_nll, base_n, base_logp = compute_loss_and_logits(model, primary_ids)
    base_loss_per_tok = base_nll / base_n
    base_ppl = float(np.exp(base_loss_per_tok))
    print(f"  baseline: NLL/tok = {base_loss_per_tok:.4f}  PPL = {base_ppl:.3f}")

    # baseline argmax for top-1/top-5 retention.
    base_argmax = base_logp.argmax(dim=-1)  # (T-1,)
    base_top5 = base_logp.topk(5, dim=-1).indices  # (T-1, 5)

    # Multi-passage baselines.
    base_multi: dict[str, float] = {}
    for label, ids in multi_pass_ids.items():
        nll_p, n_p, _ = compute_loss_and_logits(model, ids)
        base_multi[label] = nll_p / n_p

    # ---- Step 1: SVD in fp32 ----
    print("\nStep 1 — SVD of lm_head (fp32) ...")
    W_fp32 = W_lmhead.detach().to(torch.float32)
    t0 = datetime.now(UTC)
    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    t1 = datetime.now(UTC)
    print(f"  SVD done in {(t1 - t0).total_seconds():.1f}s")
    print(f"  U={tuple(U.shape)}  S={tuple(S.shape)}  Vt={tuple(Vt.shape)}")

    S_np = S.cpu().numpy()
    spec = spectrum_summary(S_np)
    print(
        f"  rank_for_99_var={spec['rank_for_99_var']}  "
        f"rank_for_95_var={spec['rank_for_95_var']}  "
        f"rank_for_90_var={spec['rank_for_90_var']}\n"
        f"  var@r=128={spec['var_at_r_128']:.4f}  "
        f"var@r=256={spec['var_at_r_256']:.4f}  "
        f"var@r=512={spec['var_at_r_512']:.4f}"
    )

    # ---- Step 2: Per-row norm distribution (dead-row confound check) ----
    print("\nStep 2 — Per-row norm distribution ...")
    row_norms = torch.linalg.norm(W_fp32, dim=1).cpu().numpy()
    median_norm = float(np.median(row_norms))
    near_zero_threshold = 0.01 * median_norm
    near_zero_count = int(np.sum(row_norms < near_zero_threshold))
    near_zero_frac = near_zero_count / V
    bottom_decile = float(np.percentile(row_norms, 10))
    print(
        f"  median_norm={median_norm:.4f}  "
        f"near_zero_count={near_zero_count} ({100*near_zero_frac:.2f}% of {V} rows)\n"
        f"  bottom_decile_norm={bottom_decile:.4f}  "
        f"max={float(row_norms.max()):.4f}"
    )
    dead_row_confound = near_zero_frac > 0.20

    # ---- Step 3: Reconstruction sweep ----
    print("\nStep 3 — Reconstruction sweep ...")
    W_orig = W_lmhead.detach().clone()
    sweep_results: list[dict[str, Any]] = []
    for r in R_VALUES:
        if r > S.shape[0]:
            continue
        W_r = reconstruct_at_rank(U, S, Vt, r, dtype)
        model.lm_head.weight.data.copy_(W_r)

        # Primary passage NLL + log_probs.
        nll_r, n_r, lp_r = compute_loss_and_logits(model, primary_ids)
        loss_per_tok = nll_r / n_r
        ppl_r = float(np.exp(loss_per_tok))
        delta_loss = loss_per_tok - base_loss_per_tok

        # Top-1 retention.
        argmax_r = lp_r.argmax(dim=-1)
        top1_retention = float((argmax_r == base_argmax).float().mean().item())
        # Top-5 set retention.
        top5_r = lp_r.topk(5, dim=-1).indices
        top5_retention_per_tok = (
            (top5_r.unsqueeze(-1) == base_top5.unsqueeze(-2))
            .any(dim=-1).float().mean(dim=-1)
        )
        top5_retention = float(top5_retention_per_tok.mean().item())

        # KL divergence: KL(p_full || p_r) per token, mean.
        # p_full = exp(base_logp), p_r = exp(lp_r). KL = sum p_full * (log p_full - log p_r).
        # base_logp / lp_r are log-probs. p_full * (log p_full - log p_r) = exp(log p_full) * (log p_full - log p_r).
        kl_per_tok = (
            (base_logp.exp() * (base_logp - lp_r)).sum(dim=-1).mean()
        )
        kl_mean = float(kl_per_tok.item())

        # Multi-passage NLL.
        multi_results = {}
        for label, ids in multi_pass_ids.items():
            nll_p, n_p, _ = compute_loss_and_logits(model, ids)
            lpt_p = nll_p / n_p
            multi_results[label] = {
                "loss_per_tok": lpt_p,
                "delta": lpt_p - base_multi[label],
            }

        # Storage byte calc.
        # Full bf16: V * d * 2 bytes
        # Mixed (LHQD-style: U INT8, V^T BF16): V*r*1 + d*r*2
        full_bytes = V * d * 2
        mixed_bytes = V * r * 1 + d * r * 2
        bf16_rank_bytes = (V + d) * r * 2
        comp_full_to_mixed = full_bytes / mixed_bytes if mixed_bytes else float("inf")

        rec = {
            "r": int(r),
            "loss_per_tok_nats": loss_per_tok,
            "ppl": ppl_r,
            "delta_loss_nats": delta_loss,
            "top1_retention": top1_retention,
            "top5_retention": top5_retention,
            "kl_per_token_mean": kl_mean,
            "multi_passage": {
                k: {"delta": v["delta"]} for k, v in multi_results.items()
            },
            "lm_head_bytes_full_bf16": full_bytes,
            "lm_head_bytes_lhqd_mixed": mixed_bytes,
            "lm_head_bytes_bf16_rank": bf16_rank_bytes,
            "compress_ratio_full_to_mixed": comp_full_to_mixed,
        }
        sweep_results.append(rec)
        print(
            f"  r={r:5d}  ΔNLL={delta_loss:+.4f}  PPL={ppl_r:.3f}  "
            f"top1_ret={top1_retention:.3f}  top5_ret={top5_retention:.3f}  "
            f"KL={kl_mean:.4f}  multi_max_Δ="
            f"{max(v['delta'] for v in multi_results.values()):+.4f}"
        )

    # Restore baseline lm_head.
    model.lm_head.weight.data.copy_(W_orig)

    # ---- Step 4: Sanity gate at r=2048 ----
    sanity = next((r for r in sweep_results if r["r"] == 2048), None)
    sanity_ok = sanity is not None and abs(sanity["delta_loss_nats"]) < 0.005
    print(
        f"\nStep 4 — Sanity gate at r=2048: ΔNLL="
        f"{sanity['delta_loss_nats']:+.5f} ({'PASS' if sanity_ok else 'FAIL'})"
        if sanity is not None
        else "\nStep 4 — Sanity gate skipped (r=2048 not in sweep)"
    )

    # ---- Step 5: Optional W_Q spectrum comparison ----
    wq_spec = None
    if args.with_wq_comparison:
        print("\nStep 5 — W_Q spectrum at mid-layer (comparison) ...")
        mid_layer = n_layers // 2
        Wq = model.model.layers[mid_layer].self_attn.q_proj.weight.detach().to(
            torch.float32
        )
        Sq = torch.linalg.svdvals(Wq)
        Sq_np = Sq.cpu().numpy()
        wq_spec = spectrum_summary(Sq_np)
        wq_spec["layer"] = mid_layer
        print(
            f"  W_Q L{mid_layer}: r_99={wq_spec['rank_for_99_var']}  "
            f"r_95={wq_spec['rank_for_95_var']}  "
            f"var@r=128={wq_spec['var_at_r_128']:.4f}  "
            f"var@r=256={wq_spec['var_at_r_256']:.4f}"
        )

    # ---- Step 6: Verdict classification ----
    rec_at_256 = next((r for r in sweep_results if r["r"] == 256), None)
    rec_at_512 = next((r for r in sweep_results if r["r"] == 512), None)

    spectral_strong_go = spec["rank_for_99_var"] <= 256
    spectral_weak_go = spec["rank_for_99_var"] <= 512
    func_strong_go = (
        rec_at_256 is not None
        and rec_at_256["delta_loss_nats"] < 0.10
        and rec_at_256["top1_retention"] >= 0.95
    )
    func_weak_go = (
        rec_at_512 is not None
        and rec_at_512["delta_loss_nats"] < 0.10
    )
    func_no_go = (
        rec_at_512 is not None
        and rec_at_512["delta_loss_nats"] > 1.0
    )

    if dead_row_confound:
        verdict = "FLAGGED — dead-row confound: >20% rows near-zero norm; spectral concentration is artifact"
    elif tie_flag or storage_tied:
        verdict = "EMBED-TIED — measured embed_tokens spectrum, not lm_head; reframe"
    elif spectral_strong_go and func_strong_go:
        verdict = "STRONG GO — concentrated spectrum AND argmax retention; opens LHQD + SVLD"
    elif spectral_strong_go and not func_strong_go:
        verdict = "INTERESTING FINDING — spectral pass at r≤256, functional flag (KL/argmax/NLL); aligns with arXiv:2603.10145 softmax bottleneck"
    elif spectral_weak_go and func_weak_go:
        verdict = "WEAK GO — moderate spectrum (99%@r≤512) with functional retention"
    elif func_no_go:
        verdict = "NO-GO (functional) — ΔNLL>1 nat at r≤512 regardless of variance"
    elif spec["rank_for_99_var"] > 1536:
        verdict = "NO-GO (spectral) — extends CF8 to lm_head; quantization-only path"
    else:
        verdict = "GRAY — between thresholds; review per-rank table"

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_primary_tokens": n_primary,
        "n_layers": n_layers,
        "d_hidden": d,
        "vocab_size": V,
        "tie_word_embeddings_config": tie_flag,
        "storage_ptr_match": storage_tied,
        "baseline_loss_per_tok": base_loss_per_tok,
        "baseline_ppl": base_ppl,
        "lmhead_spectrum": spec,
        "row_norm_stats": {
            "median": median_norm,
            "near_zero_threshold": near_zero_threshold,
            "near_zero_count": near_zero_count,
            "near_zero_frac": near_zero_frac,
            "bottom_decile": bottom_decile,
            "max": float(row_norms.max()),
        },
        "dead_row_confound": dead_row_confound,
        "sweep_results": sweep_results,
        "sanity_gate_r2048_ok": bool(sanity_ok),
        "wq_mid_layer_spectrum": wq_spec,
        "verdict": verdict,
        "stage6_amendments_applied": [
            "tied-embedding hard gate (data_ptr check)",
            "per-row L2 norm histogram for dead-row confound",
            "top-1/top-5 argmax retention per rank",
            "per-token KL divergence vs full-rank",
            "multi-passage NLL (3 disjoint passages)",
            "sanity gate at r=2048 (<0.005 nats round-trip)",
            "W_Q spectrum comparison (optional flag)",
        ],
        "cf9_preconditions": [
            "lm_head WEIGHT spectrum, not logit-matrix spectrum (arXiv:2510.24966)",
            "tied_embeddings hard-gates the experiment interpretation",
            "argmax/KL deployment metrics (NLL alone insufficient)",
            "multi-passage robustness check (single-passage WDLA-adjacent risk)",
            "no calibration fitting; WDLA ill-conditioning N/A",
        ],
        "godey_artzi_note": (
            "arXiv:2603.10145 (March 2026) predicts lm_head suppresses gradient "
            "rank during training. A concentrated weight spectrum here would be "
            "consistent with that mechanism but is not independently caused by it; "
            "framing must be honest about this prior."
        ),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_json = args.output_dir / "lhqd_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Plot 1: singular value decay (log)
        ax = axes[0, 0]
        ax.semilogy(range(1, S_np.shape[0] + 1), S_np, label="lm_head (Qwen3-1.7B-Base)")
        if wq_spec is not None:
            ax.semilogy(
                range(1, len(wq_spec["top_50"]) + 1),
                wq_spec["top_50"],
                "--",
                label=f"W_Q L{wq_spec['layer']} (top-50)",
                alpha=0.7,
            )
        ax.set_xlabel("singular value index")
        ax.set_ylabel("σ (log)")
        ax.set_title("Singular value spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: cumulative variance
        ax = axes[0, 1]
        cum_frac = np.cumsum(S_np ** 2) / np.sum(S_np ** 2)
        ax.semilogx(range(1, len(cum_frac) + 1), cum_frac, label="cum var explained")
        ax.axhline(0.99, color="red", linestyle="--", alpha=0.6, label="99%")
        ax.axhline(0.95, color="orange", linestyle="--", alpha=0.6, label="95%")
        ax.axvline(256, color="green", linestyle=":", alpha=0.6, label="r=256")
        ax.set_xlabel("rank r")
        ax.set_ylabel("cumulative variance fraction")
        ax.set_title("Cumulative variance explained")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: per-row norm histogram (log y)
        ax = axes[1, 0]
        ax.hist(row_norms, bins=80, log=True)
        ax.axvline(median_norm, color="green", linestyle="--",
                   label=f"median {median_norm:.3f}")
        ax.axvline(near_zero_threshold, color="red", linestyle="--",
                   label=f"0.01×median {near_zero_threshold:.4f}")
        ax.set_xlabel("row L2 norm")
        ax.set_ylabel("count (log)")
        ax.set_title(f"Per-row norm distribution ({V} rows)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: NLL/top1/KL vs rank
        ax = axes[1, 1]
        rs = [r["r"] for r in sweep_results]
        dnll = [r["delta_loss_nats"] for r in sweep_results]
        top1 = [r["top1_retention"] for r in sweep_results]
        ax.semilogx(rs, dnll, marker="o", label="ΔNLL (nats)", color="firebrick")
        ax2 = ax.twinx()
        ax2.semilogx(rs, top1, marker="s", color="navy", label="top-1 retention")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("top-1 retention", color="navy")
        ax.axhline(0.10, color="red", linestyle="--", alpha=0.5,
                   label="ΔNLL=0.10 GO")
        ax2.axhline(0.95, color="navy", linestyle="--", alpha=0.5)
        ax.set_xlabel("rank r")
        ax.set_ylabel("ΔNLL (nats)", color="firebrick")
        ax.set_title("Functional metrics vs rank")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = args.output_dir / "lhqd_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # summary.md
    md_lines = [
        f"# LHQD First-Pass — lm_head SVD spectrum — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Eval: {n_primary} primary tokens; vocab={V}, d={d}, n_layers={n_layers}",
        f"Baseline: NLL/tok = {base_loss_per_tok:.4f}, PPL = {base_ppl:.3f}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- tie_word_embeddings (config): {tie_flag}",
        f"- storage tied (data_ptr): {storage_tied}",
        f"- sanity gate r=2048 ΔNLL ≈ 0: {'PASS' if sanity_ok else 'FAIL'}",
        f"- dead-row confound (>20% near-zero): {dead_row_confound}",
        "",
        "## Spectrum",
        "",
        f"- rank for 99% variance: **{spec['rank_for_99_var']}** (of {spec['n_singvals']})",
        f"- rank for 95% variance: {spec['rank_for_95_var']}",
        f"- rank for 90% variance: {spec['rank_for_90_var']}",
        f"- variance at r=128: {spec['var_at_r_128']:.4f}",
        f"- variance at r=256: {spec['var_at_r_256']:.4f}",
        f"- variance at r=512: {spec['var_at_r_512']:.4f}",
        "",
        "## Per-row norm distribution",
        "",
        f"- median row norm: {median_norm:.4f}",
        f"- near-zero count (<0.01×median): {near_zero_count} "
        f"({100*near_zero_frac:.2f}% of {V})",
        f"- bottom-decile norm: {bottom_decile:.4f}",
        "",
        "## Sweep results",
        "",
        "| r | ΔNLL (nats) | PPL | top1 ret | top5 ret | KL/tok | comp (full→mixed) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in sweep_results:
        md_lines.append(
            f"| {rec['r']} | {rec['delta_loss_nats']:+.4f} | {rec['ppl']:.3f} | "
            f"{rec['top1_retention']:.3f} | {rec['top5_retention']:.3f} | "
            f"{rec['kl_per_token_mean']:.4f} | "
            f"{rec['compress_ratio_full_to_mixed']:.2f}x |"
        )
    md_lines += [
        "",
        "### Multi-passage robustness (ΔNLL nats per passage)",
        "",
        "| r | nav | bio | tech |",
        "|---|---|---|---|",
    ]
    for rec in sweep_results:
        mp = rec["multi_passage"]
        md_lines.append(
            f"| {rec['r']} | {mp.get('nav',{}).get('delta',float('nan')):+.4f} | "
            f"{mp.get('bio',{}).get('delta',float('nan')):+.4f} | "
            f"{mp.get('tech',{}).get('delta',float('nan')):+.4f} |"
        )
    md_lines += [
        "",
        "## Notes",
        "",
        "- LHQD-mixed bytes assume INT8 U + BF16 V^T (the LHQD deployment format).",
        "- arXiv:2603.10145 (Mar 2026) predicts lm_head's training dynamics suppress "
        "gradient rank; a concentrated spectrum here is consistent with that paper's "
        "mechanistic implication on the trained weight.",
        "- arXiv:2510.24966 is about logit-matrix rank, NOT weight rank — the precondition "
        "is that we measure W's spectrum directly (CF9).",
        "",
        "## References",
        "",
        "- arXiv:2603.10145 (Godey & Artzi, Mar 2026) — Lost in Backpropagation",
        "- arXiv:2510.24966 — Sequences of Logits / low logit-matrix rank",
        "- SVD-LLM (ICLR 2025), ARA (arXiv:2510.19389) — global SVD baselines",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
