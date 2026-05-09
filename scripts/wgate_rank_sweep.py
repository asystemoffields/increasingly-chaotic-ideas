r"""W_gate rank-reduction sweep — Round 3 / Idea 12 (escalated from CSMF).

Tests whether W_gate tolerates aggressive low-rank approximation while
W_up stays at full precision. Motivated by Round 1's structural finding
that W_up dominates SwiGLU top-K firing — if true, W_gate carries less
load-bearing information and should compress more aggressively.

For each rank K in {4, 8, 16, 32, 64, 128, 256, 512}:
  - Replace every MLP layer's W_gate with rank-K SVD reconstruction
  - Keep W_up and W_down unchanged
  - Measure perplexity on a held-out WikiText-2-style passage
  - Restore baseline before next K

Go threshold: PPL degradation ≤ 1.0 nat at K ≤ 128 across all layers
              AND degradation curve shows clear inflection (low-rank exists)
No-go: PPL degradation > 5 nats at K ≤ 256

References:
  - AERO (arXiv:2410.13060): activation removal + FFN folding
  - LoftQ, CALDERA, QERA: uniform low-rank correction
  - Stage 6 confirmed CSMF math flaw — polynomial silu does not eliminate W_gate
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

K_VALUES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--passage", default=EVAL_PASSAGE)
    p.add_argument("--k-values", default=",".join(str(k) for k in K_VALUES))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/stage0/ladder/round3_wgate"))
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def compute_loss(model, ids: torch.Tensor) -> tuple[float, int]:
    """Return (sum_log_loss_in_nats, n_tokens) for greedy next-token loss."""
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits  # (1, T, V)
    # Shift: predict token t from position t-1.
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather true labels.
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    n = nll.numel()
    return float(nll.sum().item()), int(n)


def replace_gate_with_rank_k(W_gate_full: torch.Tensor, k: int) -> torch.Tensor:
    """Return rank-K SVD reconstruction of W_gate, same dtype as input."""
    # Compute SVD in float32 for stability, then cast back.
    W = W_gate_full.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k_eff = min(k, S.shape[0])
    W_k = (U[:, :k_eff] * S[:k_eff].unsqueeze(0)) @ Vt[:k_eff]
    return W_k.to(W_gate_full.dtype)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_values = tuple(int(k) for k in args.k_values.split(","))

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    r = int(model.config.intermediate_size)
    print(f"  n_layers={n_layers}  d={d}  r={r}")

    encoded = tokenizer(args.passage, return_tensors="pt",
                        truncation=True, max_length=args.max_length)
    ids = encoded.input_ids.to(device)
    n_tokens = int(ids.shape[-1])
    print(f"  n_eval_tokens={n_tokens}")

    # Baseline (no modification).
    print("\nRunning baseline ...")
    base_nll, base_n = compute_loss(model, ids)
    base_ppl = float(np.exp(base_nll / base_n))
    base_loss_per_tok = base_nll / base_n
    print(f"  baseline: NLL/tok = {base_loss_per_tok:.4f}  PPL = {base_ppl:.3f}")

    # Save original W_gate weights for restoration.
    print("\nSaving baseline W_gate weights ...")
    original_gates: list[torch.Tensor] = []
    for layer in model.model.layers:
        original_gates.append(
            layer.mlp.gate_proj.weight.detach().clone()
        )

    # Sweep ranks.
    results: list[dict[str, Any]] = []
    for k in k_values:
        if k > min(d, r):
            continue
        print(f"\nRank K={k} ...")
        # Replace W_gate in every layer with rank-K SVD reconstruction.
        for layer_idx, layer in enumerate(model.model.layers):
            W_orig = original_gates[layer_idx]
            W_k = replace_gate_with_rank_k(W_orig, k)
            layer.mlp.gate_proj.weight.data.copy_(W_k)

        # Evaluate.
        nll_k, n_k = compute_loss(model, ids)
        ppl_k = float(np.exp(nll_k / n_k))
        loss_per_tok = nll_k / n_k
        delta_loss = loss_per_tok - base_loss_per_tok
        delta_ppl = ppl_k - base_ppl

        # bpw arithmetic.
        # Original W_gate at bf16: r * d * 2 bytes per layer
        # Rank-K W_gate at bf16: (r + d) * K * 2 bytes per layer
        original_bytes = r * d * 2
        rank_k_bytes = (r + d) * k * 2
        compression_ratio = original_bytes / rank_k_bytes if rank_k_bytes else float("inf")

        rec = {
            "K": int(k),
            "K_frac_d_in": float(k / d),
            "n_eval_tokens": int(n_k),
            "loss_per_tok_nats": loss_per_tok,
            "ppl": ppl_k,
            "delta_loss_nats": delta_loss,
            "delta_ppl": delta_ppl,
            "wgate_bytes_per_layer_orig": original_bytes,
            "wgate_bytes_per_layer_rankK": rank_k_bytes,
            "compression_ratio_vs_bf16": compression_ratio,
        }
        results.append(rec)
        print(f"  K={k}  NLL/tok={loss_per_tok:.4f}  ΔNLL={delta_loss:+.4f}  "
              f"PPL={ppl_k:.3f}  ΔPPL={delta_ppl:+.3f}  "
              f"compress={compression_ratio:.2f}× vs bf16")

    # Restore baseline (be tidy).
    for layer_idx, layer in enumerate(model.model.layers):
        layer.mlp.gate_proj.weight.data.copy_(original_gates[layer_idx])

    # Verdict.
    rec_at_128 = next((r for r in results if r["K"] == 128), None)
    rec_at_256 = next((r for r in results if r["K"] == 256), None)
    if rec_at_128 and rec_at_128["delta_loss_nats"] <= 1.0:
        verdict = "GO (W_gate is low-rank tolerant at K≤128)"
    elif rec_at_256 and rec_at_256["delta_loss_nats"] > 5.0:
        verdict = "NO-GO (W_gate is genuinely full-rank; rank reduction is not the path)"
    else:
        verdict = "MID (layer-sensitivity follow-up needed; per-layer adaptive K)"

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "n_eval_tokens": n_tokens,
        "n_layers": n_layers,
        "d_hidden": d,
        "intermediate_size": r,
        "k_values": list(k_values),
        "baseline_loss_per_tok": base_loss_per_tok,
        "baseline_ppl": base_ppl,
        "verdict": verdict,
        "results": results,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage6_note": "CSMF (polynomial silu) was Stage 5 pick but Stage 6 found math flaw: polynomial substitution does not eliminate W_gate. Idea 12 (this experiment) is the correct mechanism — low-rank SVD on W_gate while keeping W_up at full precision.",
        "aero_reference": "AERO (arXiv:2410.13060) eliminates activations entirely and folds W_gate+W_up; we do not go that far, only compress W_gate via low-rank.",
    }

    out_json = args.output_dir / "wgate_rank_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r["K"] for r in results]
        ppls = [r["ppl"] for r in results]
        d_loss = [r["delta_loss_nats"] for r in results]
        comp = [r["compression_ratio_vs_bf16"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.semilogx(ks, ppls, marker="o")
        ax.axhline(base_ppl, color="black", linestyle="--", alpha=0.7,
                   label=f"baseline {base_ppl:.2f}")
        ax.set_xlabel("rank K")
        ax.set_ylabel("PPL")
        ax.set_title("PPL vs W_gate rank (W_up bf16, all layers)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogx(comp, d_loss, marker="s", color="firebrick")
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.7,
                   label="GO 1.0 nats")
        ax.axhline(5.0, color="red", linestyle="--", alpha=0.7,
                   label="NO-GO 5.0 nats")
        ax.set_xlabel("compression ratio vs bf16 W_gate")
        ax.set_ylabel("ΔNLL/tok (nats)")
        ax.set_title("Quality cost vs compression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = args.output_dir / "wgate_rank_plot.png"
        plt.savefig(out_png, dpi=120)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Summary.md.
    md_lines = [
        f"# W_gate rank reduction sweep — {args.model_id}",
        "",
        f"Date: {summary['timestamp']}",
        f"Eval: {n_tokens} tokens; n_layers={n_layers}, d_hidden={d}, intermediate={r}",
        f"Baseline: NLL/tok = {base_loss_per_tok:.4f}, PPL = {base_ppl:.3f}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Results",
        "",
        "| K | K/d | PPL | ΔPPL | ΔNLL (nats) | bf16 W_gate (MB/layer) | rank-K (MB/layer) | compress |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in results:
        md_lines.append(
            f"| {rec['K']} | {rec['K_frac_d_in']:.3f} "
            f"| {rec['ppl']:.3f} | {rec['delta_ppl']:+.3f} "
            f"| {rec['delta_loss_nats']:+.4f} "
            f"| {rec['wgate_bytes_per_layer_orig']/1e6:.1f} "
            f"| {rec['wgate_bytes_per_layer_rankK']/1e6:.2f} "
            f"| {rec['compression_ratio_vs_bf16']:.2f}x |"
        )
    md_lines += [
        "",
        "## Notes",
        "",
        "- W_up and W_down kept at bf16 throughout (testing W_gate sensitivity in isolation).",
        "- All MLP layers receive rank-K replacement simultaneously (deployment-relevant case).",
        "- SVD computed in fp32 for stability, cast to bf16 for inference.",
        "",
        "## References",
        "",
        "- AERO (arXiv:2410.13060): activation removal + FFN folding (different mechanism, same goal of eliminating gate-side weight).",
        "- Round 1 finding: W_up dominates SwiGLU top-K firing → W_gate carries less load-bearing information → should compress more aggressively. This experiment tests that hypothesis directly.",
    ]
    out_md = args.output_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
