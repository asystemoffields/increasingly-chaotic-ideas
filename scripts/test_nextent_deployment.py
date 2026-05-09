r"""NEXTENT validation under genuine deployment conditions.

Test scenario: a real chat/agent loop using ik_llama.cpp on the actual Qwen3-4B-IQ4_XS
GGUF. Cold cache. Measures whether NEXTENT (Cache Manager extent prefetch) makes
the workflow work at expected tok/s without explicit weight loading.

Three conditions:
  1. Cold cache, no pressure — drop standby cache via memory pressure, then run
     a multi-turn chat. Measure: how fast does throughput converge to steady state?
  2. Cold cache, sustained memory pressure — same but with 5 GB pressure buffer
     held throughout the chat. Forces the OS to actively evict and re-fault.
  3. Warm cache control — run the chat after the cache is already populated by run 1.
     Should approximate the published 5.5 tok/s baseline.

The deployment-realistic question:
  Does NEXTENT make the workflow work even when starting cold (no pre-load)?
  Or does cold-start force visible degradation that NEXTENT can't hide?

Output: per-turn tokens/sec, time-to-first-token, prompt-eval and decode timing.
"""

from __future__ import annotations

import argparse
import json
import os
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

# A multi-turn chat-style prompt that simulates an agent loop.
# Using simple base-model completion style since this is the Base/Thinking model.
CHAT_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: How does the citric acid cycle relate to ATP production?\nAnswer:",
    "Question: Write a short Python function that computes the nth Fibonacci number using memoization.\nAnswer:",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llama-cli", default=DEFAULT_LLAMA_CLI)
    p.add_argument("--n-predict", type=int, default=80,
                   help="Tokens to generate per turn")
    p.add_argument("--n-threads", type=int, default=6)
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--evict-cache-mb", type=int, default=5120,
                   help="MB of dummy buffer to allocate (drops most of standby cache)")
    p.add_argument("--sustained-pressure-mb", type=int, default=0,
                   help="If >0, hold this much memory pressure during the run")
    p.add_argument("--condition", choices=["cold", "cold_pressure", "warm"],
                   required=True)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return p.parse_args()


def evict_cache(buffer_mb: int) -> None:
    """Drop standby cache by allocating and touching a large buffer."""
    print(f"  evicting cache via {buffer_mb} MB pressure buffer...")
    n_bytes = buffer_mb * 1024 * 1024
    buf = bytearray(n_bytes)
    for i in range(0, n_bytes, 4096):
        buf[i] = (buf[i] + 1) & 0xFF
    del buf
    time.sleep(0.5)


def run_llama_cli_turn(
    args: argparse.Namespace, prompt: str, turn_idx: int
) -> dict[str, Any]:
    """Run one llama-cli invocation with the given prompt; return timing data."""
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

    print(f"\n  Turn {turn_idx}: launching llama-cli...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    t1 = time.perf_counter()
    wall = t1 - t0

    stderr = proc.stderr or ""
    stdout = proc.stdout or ""

    # Parse llama.cpp's timing output from stderr
    # Lines look like:
    #  llama_print_timings:        load time = 1234.56 ms
    #  llama_print_timings:      sample time = ...
    #  llama_print_timings: prompt eval time = ... ms / N tokens (... ms per token, ... tok/s)
    #  llama_print_timings:        eval time = ... ms / N runs (... ms per token, ... tok/s)
    #  llama_print_timings:       total time = ... ms / N tokens
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
    print(f"  Turn {turn_idx}: wall={wall:.2f}s  load={load_ms:.0f}ms  "
          f"decode={decode_tok_s:.2f} tok/s ({decode_ms_per:.1f} ms/tok)")

    return {
        "turn": turn_idx,
        "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
        "wall_seconds": wall,
        "stdout_excerpt": stdout[-500:] if len(stdout) > 500 else stdout,
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

    print(f"NEXTENT deployment test ({args.condition} condition)")
    print(f"Model: {args.model}")
    print(f"Model size: {Path(args.model).stat().st_size / (1024**3):.2f} GiB")
    print(f"llama-cli: {args.llama_cli}")
    print(f"n_predict: {args.n_predict}, threads: {args.n_threads}, ctx: {args.ctx_size}")

    # Cache eviction step (for cold conditions)
    if args.condition in ("cold", "cold_pressure"):
        evict_cache(args.evict_cache_mb)

    # Sustained pressure: allocate a buffer that stays live during the whole run
    pressure_buf = None
    if args.condition == "cold_pressure" or args.sustained_pressure_mb > 0:
        size_mb = args.sustained_pressure_mb if args.sustained_pressure_mb > 0 else 4096
        print(f"  allocating sustained {size_mb} MB pressure buffer (held during run)...")
        pressure_buf = bytearray(size_mb * 1024 * 1024)
        # Touch every page so it's resident
        for i in range(0, len(pressure_buf), 4096):
            pressure_buf[i] = (pressure_buf[i] + 1) & 0xFF

    # Multi-turn chat run
    turn_results = []
    for i, prompt in enumerate(CHAT_PROMPTS):
        result = run_llama_cli_turn(args, prompt, i)
        turn_results.append(result)
        if pressure_buf is not None:
            # Re-touch pressure buf periodically so it stays resident
            for j in range(0, len(pressure_buf), 65536):
                pressure_buf[j] = (pressure_buf[j] + 1) & 0xFF

    # Summarize
    decode_tok_s_per_turn = [
        t.get("timing", {}).get("decode_eval", {}).get("tok_per_sec", float("nan"))
        for t in turn_results
    ]
    load_per_turn = [
        t.get("timing", {}).get("load_time_ms", float("nan"))
        for t in turn_results
    ]
    print(f"\n=== Summary ({args.condition}) ===")
    print(f"  Turn  load(ms)  decode(tok/s)  ms/tok  wall(s)")
    for t in turn_results:
        d = t["timing"].get("decode_eval", {})
        print(f"  {t['turn']:4d}  {t['timing'].get('load_time_ms', 0):8.0f}  "
              f"{d.get('tok_per_sec', float('nan')):13.2f}  "
              f"{d.get('ms_per_token', float('nan')):6.1f}  {t['wall_seconds']:7.2f}")

    summary = {
        "condition": args.condition,
        "model": args.model,
        "model_size_gb": Path(args.model).stat().st_size / (1024**3),
        "n_predict": args.n_predict,
        "n_threads": args.n_threads,
        "evict_cache_mb": args.evict_cache_mb if args.condition in ("cold", "cold_pressure") else 0,
        "sustained_pressure_mb": (args.sustained_pressure_mb
                                  if args.sustained_pressure_mb > 0
                                  else (4096 if args.condition == "cold_pressure" else 0)),
        "decode_tok_s_per_turn": decode_tok_s_per_turn,
        "load_ms_per_turn": load_per_turn,
        "turns": turn_results,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
