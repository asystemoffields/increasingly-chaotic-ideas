r"""W_V / W_O rank sweep + joint Q-K-V-O SVD diagnostic.

Tests theoretical predictions from the W_V/W_O Opus prior thread:
  - W_V ≈ W_K compressibility, possibly slightly MORE (softmax averages V errors)
  - W_O ≈ W_Q on global axis but WEAKER per-head redundancy (different inputs per head)
  - Joint Q-K-V-O union rank ~1.5× single → ~3× attention block compression

Mirror of `scripts/aqfkv_q_rank_sweep.py` extended to W_V (GQA-grouped, 1024×2048)
and W_O (square, 2048×2048). Plus bonus joint-SVD step at one mid layer that tests
the reach prediction "shared k_residual subspace across all four attention matrices."

Sanity gates: K=full rank must recover baseline within ±0.005 nats.
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
    "award process took decades and was tangled in politics."
)


WV_K_VALUES = (1024, 512, 256, 128, 64)
WO_GLOBAL_K = (2048, 1024, 512, 256, 128, 64)
WO_PERHEAD_K = (128, 96, 64, 32, 16)
JOINT_K = (2048, 1500, 1000, 500)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/stage0/ladder/track_A_arch/round4_vo"),
    )
    p.add_argument("--joint-layer", type=int, default=14,
                   help="Layer for joint Q-K-V-O SVD diagnostic")
    p.add_argument("--skip-perhead-wo", action="store_true",
                   help="Skip per-head W_O sweep (saves time)")
    return p.parse_args()


@torch.inference_mode()
def compute_loss(model, ids: torch.Tensor) -> tuple[float, int]:
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(nll.sum().item()), int(nll.numel())


def svd_reconstruct(W: torch.Tensor, k: int) -> torch.Tensor:
    W_fp32 = W.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    k_eff = min(k, S.shape[0])
    W_k = (U[:, :k_eff] * S[:k_eff].unsqueeze(0)) @ Vt[:k_eff]
    return W_k.to(W.dtype)


def perhead_svd_reconstruct(
    W_o: torch.Tensor, num_heads: int, head_dim: int, k_per_head: int
) -> torch.Tensor:
    """Per-head SVD on W_O. W_O has shape [d_model, num_heads*head_dim].
    Reshape input axis as [num_heads, head_dim, d_model] (transpose), SVD per
    head, reassemble.
    """
    d_out, d_in = W_o.shape
    assert d_in == num_heads * head_dim
    out = torch.empty_like(W_o)
    for h in range(num_heads):
        slice_h = W_o[:, h * head_dim : (h + 1) * head_dim]  # [d_out, head_dim]
        out[:, h * head_dim : (h + 1) * head_dim] = svd_reconstruct(slice_h, k_per_head)
    return out


def r_99(W: torch.Tensor) -> int:
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    return int(np.searchsorted(cum, 0.99 * total) + 1) if total > 0 else -1


def variance_at_r(W: torch.Tensor, r: int) -> float:
    W_fp32 = W.to(torch.float32)
    S = torch.linalg.svdvals(W_fp32)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    return float(cum[min(r-1, len(cum)-1)] / total) if total > 0 else float("nan")


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
    n_layers = int(cfg.num_hidden_layers)
    d = int(cfg.hidden_size)
    n_q_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_q_heads))
    head_dim = int(getattr(cfg, "head_dim", d // n_q_heads))
    print(f"  n_layers={n_layers}  d={d}  n_q_heads={n_q_heads}  n_kv_heads={n_kv_heads}  head_dim={head_dim}")

    encoded = tokenizer(EVAL_PASSAGE, return_tensors="pt",
                        truncation=True, max_length=args.max_length)
    ids = encoded.input_ids.to(args.device)
    n_tokens = int(ids.shape[-1])
    print(f"  eval tokens: {n_tokens}")

    print("\nBaseline...")
    base_nll, base_n = compute_loss(model, ids)
    base_loss = base_nll / base_n
    base_ppl = float(np.exp(base_loss))
    print(f"  baseline NLL/tok: {base_loss:.4f}  PPL: {base_ppl:.3f}")

    # Save originals
    print("\nSaving baselines for W_V and W_O across all layers...")
    original_v = [layer.self_attn.v_proj.weight.detach().clone() for layer in model.model.layers]
    original_o = [layer.self_attn.o_proj.weight.detach().clone() for layer in model.model.layers]

    Wv0 = original_v[0]
    Wo0 = original_o[0]
    print(f"  W_V[0] shape: {tuple(Wv0.shape)}")
    print(f"  W_O[0] shape: {tuple(Wo0.shape)}")

    def restore_v():
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.v_proj.weight.data.copy_(original_v[li])
    def restore_o():
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.o_proj.weight.data.copy_(original_o[li])

    # --- Spectrum diagnostic at three layers ---
    print("\nSpectrum diagnostic for W_V and W_O at layers 0, mid, last:")
    spectrum_layers = [0, n_layers // 2, n_layers - 1]
    spectra = {}
    for li in spectrum_layers:
        Wv = model.model.layers[li].self_attn.v_proj.weight.detach()
        Wo = model.model.layers[li].self_attn.o_proj.weight.detach()
        Wv_r99 = r_99(Wv)
        Wo_r99 = r_99(Wo)
        spectra[f"W_V.L{li}"] = {"r_99": Wv_r99, "r_99_over_d": Wv_r99 / Wv.shape[1]}
        spectra[f"W_O.L{li}"] = {"r_99": Wo_r99, "r_99_over_d": Wo_r99 / Wo.shape[1]}
        print(f"  L{li}: W_V r_99={Wv_r99} ({Wv_r99 / Wv.shape[1]:.3f} of d_in), "
              f"W_O r_99={Wo_r99} ({Wo_r99 / Wo.shape[1]:.3f} of d_in)")

    # --- Sweep 1: W_V global SVD ---
    print("\n=== Sweep 1: W_V global SVD ===")
    wv_results = []
    for k in WV_K_VALUES:
        for li, layer in enumerate(model.model.layers):
            W_orig = original_v[li]
            W_k = svd_reconstruct(W_orig, k)
            layer.self_attn.v_proj.weight.data.copy_(W_k)
        nll_k, n_k = compute_loss(model, ids)
        loss_k = nll_k / n_k
        ppl_k = float(np.exp(loss_k))
        delta = loss_k - base_loss
        rec = {"K": k, "K_frac_d_in": k / Wv0.shape[1],
               "loss_per_tok": loss_k, "ppl": ppl_k, "delta_nll": delta}
        wv_results.append(rec)
        print(f"  K={k:5d}  ΔNLL={delta:+.4f}  PPL={ppl_k:.3f}")
    restore_v()

    # --- Sweep 2: W_O global SVD ---
    print("\n=== Sweep 2: W_O global SVD ===")
    wo_global_results = []
    for k in WO_GLOBAL_K:
        for li, layer in enumerate(model.model.layers):
            W_orig = original_o[li]
            W_k = svd_reconstruct(W_orig, k)
            layer.self_attn.o_proj.weight.data.copy_(W_k)
        nll_k, n_k = compute_loss(model, ids)
        loss_k = nll_k / n_k
        ppl_k = float(np.exp(loss_k))
        delta = loss_k - base_loss
        rec = {"K": k, "K_frac_d": k / d, "loss_per_tok": loss_k,
               "ppl": ppl_k, "delta_nll": delta}
        wo_global_results.append(rec)
        print(f"  K={k:5d}  ΔNLL={delta:+.4f}  PPL={ppl_k:.3f}")
    restore_o()

    # --- Sweep 3: W_O per-head SVD ---
    wo_perhead_results = []
    if not args.skip_perhead_wo:
        print("\n=== Sweep 3: W_O per-head SVD ===")
        for kph in WO_PERHEAD_K:
            if kph > head_dim:
                continue
            for li, layer in enumerate(model.model.layers):
                W_orig = original_o[li]
                W_kph = perhead_svd_reconstruct(W_orig, n_q_heads, head_dim, kph)
                layer.self_attn.o_proj.weight.data.copy_(W_kph)
            nll_k, n_k = compute_loss(model, ids)
            loss_k = nll_k / n_k
            ppl_k = float(np.exp(loss_k))
            delta = loss_k - base_loss
            rec = {"K_per_head": kph, "K_frac_head_dim": kph / head_dim,
                   "loss_per_tok": loss_k, "ppl": ppl_k, "delta_nll": delta}
            wo_perhead_results.append(rec)
            print(f"  K_ph={kph:3d}  ΔNLL={delta:+.4f}  PPL={ppl_k:.3f}")
        restore_o()

    # --- Bonus: Joint Q-K-V-O SVD at mid layer ---
    print(f"\n=== Bonus: Joint Q-K-V-O SVD at layer {args.joint_layer} ===")
    jl = args.joint_layer
    Wq = model.model.layers[jl].self_attn.q_proj.weight.detach().to(torch.float32)
    Wk = model.model.layers[jl].self_attn.k_proj.weight.detach().to(torch.float32)
    Wv = model.model.layers[jl].self_attn.v_proj.weight.detach().to(torch.float32)
    Wo = model.model.layers[jl].self_attn.o_proj.weight.detach().to(torch.float32)
    # Stack so all matrices share input axis = d_model
    # W_Q [d, d], W_K [n_kv*head_dim, d], W_V [n_kv*head_dim, d], W_O has input axis n_q*head_dim = d, output d
    # We need them all to share input axis d_model = 2048. W_O reads from concat-of-heads (size 2048), writes to d_model.
    # For "joint right-singular basis on the input axis = d_model side":
    # W_Q reads from d_model → input axis is d_model
    # W_K reads from d_model → input axis is d_model
    # W_V reads from d_model → input axis is d_model
    # W_O writes to d_model → if we transpose, input axis becomes d_model (reading from heads)
    # So stack as [Wq; Wk; Wv; Wo^T], all share columns dim=d_model
    Wo_t = Wo.T  # [n_q*head_dim, d_model] but that's [2048, 2048] same dim either way for square
    stacked = torch.cat([Wq, Wk, Wv, Wo_t], dim=0)
    print(f"  stacked shape: {tuple(stacked.shape)}")
    print(f"  Computing SVD...")
    U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
    eigvals = (S ** 2).cpu().numpy()
    cum = np.cumsum(eigvals)
    total = cum[-1]
    cum_frac = cum / total

    joint_results = {}
    for k in JOINT_K:
        if k > min(stacked.shape):
            continue
        joint_results[f"k={k}"] = {
            "var_at_r": float(cum_frac[min(k - 1, len(cum_frac) - 1)]),
        }
        print(f"  joint k={k:5d}: var_at_r = {cum_frac[min(k-1, len(cum_frac)-1)]:.4f}")

    joint_r99 = int(np.searchsorted(cum_frac, 0.99) + 1)
    print(f"  joint r_99 = {joint_r99}")
    individual_r99_sum = (
        r_99(Wq) + r_99(Wk) + r_99(Wv) + r_99(Wo)
    )
    overlap_ratio = joint_r99 / individual_r99_sum if individual_r99_sum > 0 else float("nan")
    print(f"  individual r_99 sum: {individual_r99_sum}")
    print(f"  joint/individual ratio: {overlap_ratio:.3f}")
    print(f"  → if <0.5: strong subspace sharing (predicted ~1.5×); >0.7: no sharing")

    # Verdict on each sub-prediction
    wv_at_512 = next((r for r in wv_results if r["K"] == 512), None)
    wo_at_1024 = next((r for r in wo_global_results if r["K"] == 1024), None)
    wo_at_128 = next((r for r in wo_global_results if r["K"] == 128), None)
    wo_perhead_64 = next((r for r in wo_perhead_results if r["K_per_head"] == 64), None) if wo_perhead_results else None

    verdicts = {}
    if wv_at_512:
        d_v = wv_at_512["delta_nll"]
        if d_v < 0.45:
            verdicts["W_V_K512"] = f"GO_predicted ({d_v:.3f} < 0.45 vs W_K's 0.29)"
        elif d_v < 0.65:
            verdicts["W_V_K512"] = f"GRAY ({d_v:.3f})"
        else:
            verdicts["W_V_K512"] = f"REFUTED ({d_v:.3f}; V less compressible than K)"

    if wo_at_128:
        d_o = wo_at_128["delta_nll"]
        if d_o < 1.20:
            verdicts["W_O_K128"] = f"GO_predicted ({d_o:.3f}, similar to W_Q's 0.98)"
        else:
            verdicts["W_O_K128"] = f"WEAKER ({d_o:.3f})"

    if wo_perhead_64 and wo_at_128:
        d_perhead = wo_perhead_64["delta_nll"]
        wq_perhead_64 = 1.53  # AQFKV result
        if d_perhead < wq_perhead_64 - 0.3:
            verdicts["W_O_perhead_64"] = (
                f"PREDICTED_SOFTER ({d_perhead:.3f} < W_Q's {wq_perhead_64:.2f} - 0.3); "
                f"W_O has weaker per-head redundancy than W_Q"
            )
        elif d_perhead > wq_perhead_64 + 0.5:
            verdicts["W_O_perhead_64"] = (
                f"REFUTED_OPPOSITE ({d_perhead:.3f} > W_Q's {wq_perhead_64:.2f} + 0.5); "
                f"W_O is HARDER to compress per-head than W_Q (surprising)"
            )
        else:
            verdicts["W_O_perhead_64"] = f"SIMILAR ({d_perhead:.3f} ≈ W_Q's {wq_perhead_64:.2f})"

    if overlap_ratio == overlap_ratio:
        if overlap_ratio < 0.5:
            verdicts["joint_subspace"] = (
                f"GO_predicted ({overlap_ratio:.3f} < 0.5; strong sharing → "
                f"~3× attention block compression viable)"
            )
        elif overlap_ratio > 0.7:
            verdicts["joint_subspace"] = (
                f"REFUTED ({overlap_ratio:.3f} > 0.7; matrices use orthogonal subspaces)"
            )
        else:
            verdicts["joint_subspace"] = f"GRAY ({overlap_ratio:.3f})"

    summary = {
        "model_id": args.model_id,
        "n_eval_tokens": n_tokens,
        "baseline_loss_per_tok": base_loss,
        "baseline_ppl": base_ppl,
        "spectrum_diagnostic": spectra,
        "wv_global_sweep": wv_results,
        "wo_global_sweep": wo_global_results,
        "wo_perhead_sweep": wo_perhead_results,
        "joint_qkvo_layer": jl,
        "joint_qkvo_results": joint_results,
        "joint_qkvo_r99": int(joint_r99),
        "joint_qkvo_individual_r99_sum": int(individual_r99_sum),
        "joint_qkvo_overlap_ratio": float(overlap_ratio) if overlap_ratio == overlap_ratio else None,
        "verdicts": verdicts,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_json = args.output_dir / "vo_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    print(f"\n=========================================")
    print(f"VERDICTS:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"=========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
