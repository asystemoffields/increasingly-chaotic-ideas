r"""A1-GFRS-2 — Z_2^d Residual-Stream Sign-Entropy Gauge on W_up.

Test: under canonical sign-flip gauge fix (each row of W_up has its sign flipped
if its max-abs element is negative), measure Shannon entropy H of the sign-bit
distribution per row and aggregate. If H < 0.8 bits/weight globally, ~12.5%
lossless compression via sign-bit Huffman is on the table (composes additively
with weight quantization).

GO threshold: H_global < 0.8 bits/weight AND random-init H_global >= 0.90
              AND std(p_row) > 0.10 (per Stage 6 amendment A2 — sharpens the
              codebook-vs-global-bias interpretation).
NO-GO: H_global >= 0.95 (essentially random sign distribution)
       OR random-init gap < 0.05 (architectural artifact, not trained)

Per Stage 6 amendment A1: also measure sign-agreement fraction between W_gate and
W_up max-abs elements per neuron. If aligned, the joint gauge is smaller than
independent gauges.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig

REPO = Path(r"C:\Users\power\documents\increasingly-chaotic-ideas")
OUT_JSON = REPO / "experiments" / "stage0" / "ladder_v2" / "round1_gfrs2" / "a1_gfrs_2_results.json"
OUT_MD = REPO / "sonnet_ladder_v2" / "cheap_tests" / "run_001" / "a1_gfrs_2_result.md"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def shannon_entropy(p: float) -> float:
    """Binary entropy H(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def measure_sign_entropy(model, label: str):
    """Returns dict of stats for the W_up sign-entropy structure of a model."""
    n_layers = model.config.num_hidden_layers
    d_ffn = model.config.intermediate_size

    per_layer = []
    all_p_pos = []  # per-row probability of '+' sign after canonical gauge
    sign_agree_with_gate = []  # per-row: does W_up max-abs sign match W_gate max-abs sign?

    for L in range(n_layers):
        layer = model.model.layers[L].mlp
        W_up = layer.up_proj.weight.data.to(torch.float32).numpy()       # (d_ffn, d_model)
        W_gate = layer.gate_proj.weight.data.to(torch.float32).numpy()   # (d_ffn, d_model)
        # Canonical gauge: flip each row if its max-abs element is negative
        max_abs_idx = np.argmax(np.abs(W_up), axis=1)
        max_abs_val = W_up[np.arange(d_ffn), max_abs_idx]
        flip = (max_abs_val < 0).astype(np.float32) * -2.0 + 1.0  # -1 if flip else +1
        W_up_canon = W_up * flip[:, None]
        # After canonicalization, the max-abs element is positive in every row.
        # Sign distribution of remaining elements:
        signs = np.sign(W_up_canon)
        p_pos_per_row = (signs == 1).mean(axis=1)
        # Per-row entropy:
        H_per_row = np.array([shannon_entropy(p) for p in p_pos_per_row])
        # Layer-level statistics:
        per_layer.append({
            "layer": L,
            "n_rows": int(d_ffn),
            "p_pos_mean": float(p_pos_per_row.mean()),
            "p_pos_std": float(p_pos_per_row.std()),
            "p_pos_min": float(p_pos_per_row.min()),
            "p_pos_max": float(p_pos_per_row.max()),
            "H_row_mean_bits": float(H_per_row.mean()),
            "H_row_std_bits": float(H_per_row.std()),
            "H_row_min_bits": float(H_per_row.min()),
            "H_row_max_bits": float(H_per_row.max()),
            "rows_at_H_lt_0_8": int((H_per_row < 0.8).sum()),
            "rows_at_H_lt_0_5": int((H_per_row < 0.5).sum()),
        })
        all_p_pos.extend(p_pos_per_row.tolist())

        # Stage 6 A1: W_gate sign agreement check
        gate_max_abs_idx = np.argmax(np.abs(W_gate), axis=1)
        gate_max_abs_sign = np.sign(W_gate[np.arange(d_ffn), gate_max_abs_idx])
        up_max_abs_sign = np.sign(W_up[np.arange(d_ffn), max_abs_idx])
        sign_agree_with_gate.extend((gate_max_abs_sign == up_max_abs_sign).tolist())

    all_p_pos = np.array(all_p_pos)
    sign_agree_with_gate = np.array(sign_agree_with_gate)

    # Global statistics
    p_pos_global_mean = all_p_pos.mean()
    H_global = shannon_entropy(p_pos_global_mean)
    H_per_row_global = np.array([shannon_entropy(p) for p in all_p_pos])
    H_per_row_mean = H_per_row_global.mean()
    p_pos_std = all_p_pos.std()
    sign_agree_frac = sign_agree_with_gate.mean()

    print(f"\n=== {label} ===", flush=True)
    print(f"  n_layers x d_ffn = {len(per_layer)} x {per_layer[0]['n_rows']}", flush=True)
    print(f"  global p(sign=+) mean: {p_pos_global_mean:.4f}", flush=True)
    print(f"  global p(sign=+) std (per-row): {p_pos_std:.4f}", flush=True)
    print(f"  H(global p_pos): {H_global:.4f} bits/weight", flush=True)
    print(f"  H(per-row mean): {H_per_row_mean:.4f} bits/weight", flush=True)
    print(f"  rows with H<0.8: {int((H_per_row_global < 0.8).sum())} / {len(H_per_row_global)} ({(H_per_row_global < 0.8).mean()*100:.2f}%)", flush=True)
    print(f"  rows with H<0.5: {int((H_per_row_global < 0.5).sum())} / {len(H_per_row_global)} ({(H_per_row_global < 0.5).mean()*100:.2f}%)", flush=True)
    print(f"  W_gate/W_up max-abs sign-agreement: {sign_agree_frac:.4f}", flush=True)

    return {
        "label": label,
        "p_pos_global_mean": float(p_pos_global_mean),
        "p_pos_std": float(p_pos_std),
        "H_global_bits": float(H_global),
        "H_per_row_mean_bits": float(H_per_row_mean),
        "frac_rows_H_lt_0_8": float((H_per_row_global < 0.8).mean()),
        "frac_rows_H_lt_0_5": float((H_per_row_global < 0.5).mean()),
        "sign_agreement_with_gate": float(sign_agree_frac),
        "per_layer": per_layer,
    }


def main():
    t0 = time.time()
    print("A1-GFRS-2: Z_2^d sign-entropy gauge on W_up", flush=True)

    print(f"\n[{time.time()-t0:.1f}s] Loading TRAINED Qwen3-1.7B-Base ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
    )
    print(f"[{time.time()-t0:.1f}s] Model loaded.", flush=True)
    trained_stats = measure_sign_entropy(model, "trained")
    del model
    import gc; gc.collect()

    # Random-init control with same architecture
    print(f"\n[{time.time()-t0:.1f}s] Loading RANDOM-INIT same architecture ...", flush=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B-Base")
    rng = torch.Generator().manual_seed(20260509)
    rand_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    # Re-initialize MLP weights with the same default scheme but explicit seed
    for L in range(rand_model.config.num_hidden_layers):
        layer = rand_model.model.layers[L].mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(layer, name).weight.data
            torch.nn.init.normal_(w, mean=0.0, std=0.02, generator=rng)
    print(f"[{time.time()-t0:.1f}s] Random-init done.", flush=True)
    random_stats = measure_sign_entropy(rand_model, "random_init")
    del rand_model
    gc.collect()

    # Verdict logic
    H_trained = trained_stats["H_global_bits"]
    H_random = random_stats["H_global_bits"]
    p_pos_std_trained = trained_stats["p_pos_std"]
    H_per_row_mean_trained = trained_stats["H_per_row_mean_bits"]
    H_per_row_mean_random = random_stats["H_per_row_mean_bits"]

    # Per Stage 6 amendments:
    # Per-row is the actual codebook unit (we'd Huffman-encode each row independently)
    # so H_per_row_mean is the right comparison.
    if H_per_row_mean_trained < 0.8 and H_per_row_mean_random >= 0.90 and p_pos_std_trained > 0.10:
        verdict = "GO_STRONG"
        verdict_note = (
            f"H_per_row(trained)={H_per_row_mean_trained:.3f} < 0.8 AND "
            f"H_per_row(random)={H_per_row_mean_random:.3f} >= 0.90 AND "
            f"p_pos_std={p_pos_std_trained:.3f} > 0.10. Sign-bit Huffman lossless compression viable. "
            f"Compression saving = ~{(1.0 - H_per_row_mean_trained)*100:.1f}% on sign bits."
        )
    elif H_per_row_mean_trained < 0.95 and H_per_row_mean_trained < H_per_row_mean_random - 0.05:
        verdict = "GO_WEAK"
        verdict_note = (
            f"H_per_row(trained)={H_per_row_mean_trained:.3f}, "
            f"H_per_row(random)={H_per_row_mean_random:.3f}, gap={H_per_row_mean_random - H_per_row_mean_trained:.3f}. "
            f"Marginal sign-bit savings (~{(1.0 - H_per_row_mean_trained)*100:.1f}%); structurally non-random "
            f"but not at the >=0.10 bits threshold for strong compression."
        )
    else:
        verdict = "NO_GO"
        verdict_note = (
            f"H_per_row(trained)={H_per_row_mean_trained:.3f}, "
            f"H_per_row(random)={H_per_row_mean_random:.3f}. "
            f"Trained sign distribution is essentially random; canonical-gauge sign-codebook compression infeasible."
        )

    print(f"\n[{time.time()-t0:.1f}s] === VERDICT: {verdict} ===", flush=True)
    print(f"  {verdict_note}", flush=True)

    out = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "trained": trained_stats,
        "random_init": random_stats,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}", flush=True)

    # Markdown report
    md = []
    md.append("# A1-GFRS-2 — Z_2^d Sign-Entropy Gauge on W_up")
    md.append("")
    md.append("Date: 2026-05-09")
    md.append("Model: Qwen3-1.7B-Base bf16, 28 layers x d_ffn=6144")
    md.append("")
    md.append(f"## Verdict: {verdict}")
    md.append("")
    md.append(verdict_note)
    md.append("")
    md.append("## Numbers")
    md.append("")
    md.append(f"| Metric | Trained | Random-init |")
    md.append("|---|---|---|")
    md.append(f"| H(global p_pos) bits | {trained_stats['H_global_bits']:.4f} | {random_stats['H_global_bits']:.4f} |")
    md.append(f"| H(per-row mean) bits | {trained_stats['H_per_row_mean_bits']:.4f} | {random_stats['H_per_row_mean_bits']:.4f} |")
    md.append(f"| p(sign=+) global mean | {trained_stats['p_pos_global_mean']:.4f} | {random_stats['p_pos_global_mean']:.4f} |")
    md.append(f"| p(sign=+) per-row std | {trained_stats['p_pos_std']:.4f} | {random_stats['p_pos_std']:.4f} |")
    md.append(f"| Frac rows H<0.8 | {trained_stats['frac_rows_H_lt_0_8']*100:.2f}% | {random_stats['frac_rows_H_lt_0_8']*100:.2f}% |")
    md.append(f"| Frac rows H<0.5 | {trained_stats['frac_rows_H_lt_0_5']*100:.2f}% | {random_stats['frac_rows_H_lt_0_5']*100:.2f}% |")
    md.append(f"| W_gate/W_up max-abs sign-agreement | {trained_stats['sign_agreement_with_gate']:.4f} | {random_stats['sign_agreement_with_gate']:.4f} |")
    md.append("")
    md.append("## Compression-viability implication")
    md.append("")
    if "GO" in verdict:
        savings = (1.0 - trained_stats["H_per_row_mean_bits"]) * 100
        md.append(f"Sign-bit Huffman per row would save ~{savings:.1f}% of sign-bit storage (vs uniform 1 bit/sign).")
        md.append(f"Total weight-storage savings: ~{savings/8:.2f}% on the model (since each weight has 1 sign bit out of typically 8-16 stored bits at quantized precision).")
    else:
        md.append("Sign-bit compression infeasible — distribution is too close to uniform random.")
    md.append("")
    md.append("## Files")
    md.append("- `scripts/a1_gfrs_2.py`")
    md.append("- `experiments/stage0/ladder_v2/round1_gfrs2/a1_gfrs_2_results.json`")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
