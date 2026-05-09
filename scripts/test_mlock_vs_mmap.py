r"""Test (b): mlock vs default-mmap comparison on Qwen3-4B-IQ4_XS.

Validates whether OS demand-paging (with NEXTENT) keeps up with explicit RAM-pinning
on the actual deployment workload.

For each condition, drop standby cache via memory pressure, then run 3 turns of
chat-style prompts. Conditions:

  - mlock:    --mlock flag passed to llama-cli; model is pinned in RAM, no eviction
  - default:  default mmap behavior; relies on OS page cache + NEXTENT prefetch

If decode tok/s is similar across conditions, the OS demand-paging path with NEXTENT
is sufficient for this deployment. If mlock is meaningfully faster, NEXTENT alone
isn't enough and explicit pinning would be required for production.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


DEFAULT_MODEL = (
    "C:/Users/power/documents/Nocturnal_One/models/cache/gguf/"
    "Qwen_Qwen3-4B-Thinking-2507-IQ4_XS.gguf"
)
DEFAULT_LLAMA_CLI = (
    "C:/Users/power/documents/upstream/ik_llama.cpp/build/bin/llama-cli.exe"
)

CHAT_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: How does the citric acid cycle relate to ATP production?\nAnswer:",
    "Question: Write a short Python function that computes the nth Fibonacci number using memoization.\nAnswer:",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llama-cli", default=DEFAULT_LLAMA_CLI)
    p.add_argument("--n-predict", type=int, default=80)
    p.add_argument("--n-threads", type=int, default=6)
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--evict-cache-mb", type=int, default=5120)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/test_mlock_vs_mmap.json"),
    )
    return p.parse_args()


def evict_cache(buffer_mb: int) -> None:
    print(f"  evicting cache via {buffer_mb} MB pressure buffer...")
    n_bytes = buffer_mb * 1024 * 1024
    buf = bytearray(n_bytes)
    for i in range(0, n_bytes, 4096):
        buf[i] = (buf[i] + 1) & 0xFF
    del buf
    time.sleep(0.5)


def run_llama_cli_turn(
    args: argparse.Namespace, prompt: str, turn_idx: int, mlock: bool
) -> dict[str, Any]:
    cmd = [
        args.llama_cli,
        "-m", args.model,
        "-t", str(args.n_threads),
        "-fa", "1",
        "-c", str(args.ctx_size),
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-n", str(args.n_predict),
        "--no-warmup",
        "-p", prompt,
        "--seed", "42",
    ]
    if mlock:
        cmd.append("--mlock")

    label = "mlock" if mlock else "mmap"
    print(f"\n  Turn {turn_idx} ({label}): launching llama-cli...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    t1 = time.perf_counter()
    wall = t1 - t0

    stderr = proc.stderr or ""
    timing = {}
    for pattern, key in [
        (r"load time\s*=\s*([\d.]+)\s*ms", "load_time_ms"),
        (r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", "prompt_eval"),
        (r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", "decode_eval"),
        (r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", "total"),
    ]:
        m = re.search(pattern, stderr)
        if m:
            if key == "load_time_ms":
                timing[key] = float(m.group(1))
            else:
                ms = float(m.group(1))
                n = int(m.group(2))
                timing[key] = {
                    "ms": ms,
                    "n_tokens": n,
                    "ms_per_token": ms / n if n else float("nan"),
                    "tok_per_sec": 1000.0 * n / ms if ms else float("nan"),
                }

    decode_tok_s = timing.get("decode_eval", {}).get("tok_per_sec", float("nan"))
    decode_ms_per = timing.get("decode_eval", {}).get("ms_per_token", float("nan"))
    load_ms = timing.get("load_time_ms", float("nan"))
    print(f"  Turn {turn_idx} ({label}): wall={wall:.2f}s  load={load_ms:.0f}ms  "
          f"decode={decode_tok_s:.2f} tok/s ({decode_ms_per:.1f} ms/tok)")

    return {
        "turn": turn_idx,
        "condition": label,
        "wall_seconds": wall,
        "timing": timing,
        "returncode": proc.returncode,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        return 1
    if not Path(args.llama_cli).exists():
        print(f"ERROR: llama-cli not found: {args.llama_cli}")
        return 1

    print(f"mlock vs default-mmap deployment comparison")
    print(f"Model: {args.model}")
    print(f"Model size: {Path(args.model).stat().st_size / (1024**3):.2f} GiB")

    all_results = []

    # === Condition 1: default mmap ===
    print(f"\n=== Condition 1: default mmap (relies on NEXTENT + page cache) ===")
    evict_cache(args.evict_cache_mb)
    mmap_results = []
    for i, prompt in enumerate(CHAT_PROMPTS):
        result = run_llama_cli_turn(args, prompt, i, mlock=False)
        mmap_results.append(result)
    all_results.extend(mmap_results)

    # === Condition 2: --mlock (model pinned in RAM) ===
    print(f"\n=== Condition 2: --mlock (model pinned in RAM, no eviction) ===")
    evict_cache(args.evict_cache_mb)
    mlock_results = []
    for i, prompt in enumerate(CHAT_PROMPTS):
        result = run_llama_cli_turn(args, prompt, i, mlock=True)
        mlock_results.append(result)
    all_results.extend(mlock_results)

    # Summarize
    def turn_decode(r):
        return r["timing"].get("decode_eval", {}).get("tok_per_sec", float("nan"))
    def turn_load(r):
        return r["timing"].get("load_time_ms", float("nan"))

    mmap_decodes = [turn_decode(r) for r in mmap_results]
    mlock_decodes = [turn_decode(r) for r in mlock_results]

    print(f"\n=== Comparison ===")
    print(f"{'turn':>5}  {'mmap load(ms)':>15}  {'mlock load(ms)':>15}  "
          f"{'mmap tok/s':>11}  {'mlock tok/s':>12}")
    for i in range(len(CHAT_PROMPTS)):
        print(f"{i:>5}  {turn_load(mmap_results[i]):>15.0f}  "
              f"{turn_load(mlock_results[i]):>15.0f}  "
              f"{turn_decode(mmap_results[i]):>11.2f}  "
              f"{turn_decode(mlock_results[i]):>12.2f}")

    avg_mmap = sum(mmap_decodes) / len(mmap_decodes)
    avg_mlock = sum(mlock_decodes) / len(mlock_decodes)
    delta_pct = 100 * (avg_mlock - avg_mmap) / avg_mmap if avg_mmap > 0 else float("nan")

    print(f"\nAvg decode tok/s:  mmap={avg_mmap:.2f}  mlock={avg_mlock:.2f}  "
          f"Δ={delta_pct:+.1f}%")

    if abs(delta_pct) < 5.0:
        verdict = (f"GO — mlock and default-mmap are within {abs(delta_pct):.1f}% "
                   f"({avg_mmap:.2f} vs {avg_mlock:.2f} tok/s). The OS demand-paging "
                   f"path with NEXTENT prefetch is sufficient for this deployment. "
                   f"NEXTENT is a real deployment substrate, not a synthetic-only effect.")
    elif delta_pct > 5.0:
        verdict = (f"PARTIAL — mlock is {delta_pct:.1f}% faster ({avg_mmap:.2f} → "
                   f"{avg_mlock:.2f} tok/s). NEXTENT helps but explicit pinning is "
                   f"meaningfully better. For production, --mlock is the right default "
                   f"on this hardware when the model fits.")
    else:
        verdict = (f"CURIOUS — mlock is {-delta_pct:.1f}% SLOWER ({avg_mmap:.2f} → "
                   f"{avg_mlock:.2f} tok/s). Possibly due to mlock initial cost or "
                   f"reduced standby cache for KV. Investigate further.")

    print(f"\nVERDICT: {verdict}")

    summary = {
        "model": args.model,
        "model_size_gb": Path(args.model).stat().st_size / (1024**3),
        "n_predict": args.n_predict,
        "n_threads": args.n_threads,
        "ctx_size": args.ctx_size,
        "mmap_results": mmap_results,
        "mlock_results": mlock_results,
        "avg_mmap_tok_s": avg_mmap,
        "avg_mlock_tok_s": avg_mlock,
        "delta_pct": delta_pct,
        "verdict": verdict,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
