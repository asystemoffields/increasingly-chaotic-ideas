r"""Test 3 — NEXTENT under the actual decode access pattern.

Test 1 (random 4 KB probes) confirmed: NEXTENT engages, same-extent followons are
~free (1.3-1.9 μs vs 137-213 μs first-page).

But that was random 4 KB probes. The actual decode access pattern is different:
each token's forward pass touches a SUBSET OF EXTENTS, fully consuming pages within
each extent (because the layer's weights span a 1+ MB region), then moves to the
next extent.

This script simulates that pattern:
  Pattern A: pure sequential — touch every page in a row, simulating first-load
  Pattern B: random-extent + intra-extent sequential — pick K random 1 MB extents,
             read all 256 pages within each in order, repeat
  Pattern C: random-extent + sparse-within-extent — pick K random 1 MB extents,
             read 16 evenly-spaced pages within each (simulating partial extent
             usage), repeat

Pattern B is the closest model of "PFOR-aligned decode." If it shows the same
~1.3-1.9 μs amortized per-page latency, then NEXTENT is the deployment substrate
we hoped for. If the average per-page latency degrades to ~50-100 μs, then
something else (paging contention, cache thrashing, batch IOPS limits) is interfering.

Pattern C tests whether NEXTENT requires CONSUMING the whole extent or just touching
a single page inside it. If C shows similar latency to B, then prefetch is "all or
nothing" once triggered. If C shows latency ~halfway between random and B, then
we only get the prefetch benefit on pages we actually use.
"""

from __future__ import annotations

import argparse
import json
import mmap
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


PAGE_SIZE = 4096
EXTENT_SIZE = 1024 * 1024  # 1 MB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file-path", type=Path, required=True)
    p.add_argument("--n-extents-per-pattern", type=int, default=200,
                   help="Number of random extents to touch per pattern")
    p.add_argument("--n-sparse-pages-per-extent", type=int, default=16,
                   help="For pattern C, pages to touch per random extent")
    p.add_argument("--evict-cache-mb", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/stage0/opus_pipeline/round1/test3_nextent_decode_pattern.json"),
    )
    return p.parse_args()


def evict_cache_via_pressure(buffer_mb: int) -> None:
    print(f"  evicting cache via {buffer_mb} MB pressure buffer...")
    n_bytes = buffer_mb * 1024 * 1024
    buf = bytearray(n_bytes)
    for i in range(0, n_bytes, PAGE_SIZE):
        buf[i] = (buf[i] + 1) & 0xFF
    del buf


def time_pattern_A_pure_sequential(mm: mmap.mmap, file_size: int, n_pages: int) -> list[float]:
    """Touch the first n_pages of the file sequentially (page by page)."""
    n_pages = min(n_pages, file_size // PAGE_SIZE)
    latencies = []
    accum = 0
    for p in range(n_pages):
        offset = p * PAGE_SIZE
        t0 = time.perf_counter_ns()
        b = mm[offset]
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)
        accum += b
    return latencies


def time_pattern_B_random_extent_full(
    mm: mmap.mmap, file_size: int, n_extents: int, seed: int
) -> tuple[list[float], list[float]]:
    """For each of n_extents randomly chosen 1 MB extents, touch all 256 pages
    within in order. Return (first_page_latencies, intra_extent_followon_latencies)
    separately so we can compare the cost amortization.
    """
    rng = random.Random(seed)
    n_extents_in_file = file_size // EXTENT_SIZE
    pages_per_extent = EXTENT_SIZE // PAGE_SIZE
    first_page_latencies = []
    followon_latencies = []
    accum = 0
    for _ in range(n_extents):
        ext_idx = rng.randrange(0, n_extents_in_file)
        ext_base = ext_idx * EXTENT_SIZE
        # First page of extent (cold fault)
        t0 = time.perf_counter_ns()
        b = mm[ext_base]
        t1 = time.perf_counter_ns()
        first_page_latencies.append((t1 - t0) / 1000.0)
        accum += b
        # All subsequent pages within this extent
        for k in range(1, pages_per_extent):
            offset = ext_base + k * PAGE_SIZE
            t0 = time.perf_counter_ns()
            b = mm[offset]
            t1 = time.perf_counter_ns()
            followon_latencies.append((t1 - t0) / 1000.0)
            accum += b
    return first_page_latencies, followon_latencies


def time_pattern_C_random_extent_sparse(
    mm: mmap.mmap, file_size: int, n_extents: int, n_sparse_pages: int, seed: int
) -> tuple[list[float], list[float]]:
    """For each of n_extents randomly chosen 1 MB extents, touch n_sparse_pages
    evenly-spaced pages within (skipping pages between). Tests whether prefetch
    works when extent is only partially consumed.
    """
    rng = random.Random(seed)
    n_extents_in_file = file_size // EXTENT_SIZE
    pages_per_extent = EXTENT_SIZE // PAGE_SIZE
    stride = max(1, pages_per_extent // n_sparse_pages)
    first_page_latencies = []
    followon_latencies = []
    accum = 0
    for _ in range(n_extents):
        ext_idx = rng.randrange(0, n_extents_in_file)
        ext_base = ext_idx * EXTENT_SIZE
        # First page of extent
        t0 = time.perf_counter_ns()
        b = mm[ext_base]
        t1 = time.perf_counter_ns()
        first_page_latencies.append((t1 - t0) / 1000.0)
        accum += b
        # Sparse subsequent pages within extent
        for k in range(1, n_sparse_pages):
            page_idx = min(k * stride, pages_per_extent - 1)
            offset = ext_base + page_idx * PAGE_SIZE
            t0 = time.perf_counter_ns()
            b = mm[offset]
            t1 = time.perf_counter_ns()
            followon_latencies.append((t1 - t0) / 1000.0)
            accum += b
    return first_page_latencies, followon_latencies


def summarize(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {}
    s = sorted(latencies)
    return {
        "n": len(s),
        "min_us": s[0],
        "p10_us": s[len(s) // 10],
        "median_us": statistics.median(s),
        "p90_us": s[len(s) * 9 // 10],
        "p99_us": s[len(s) * 99 // 100],
        "max_us": s[-1],
        "mean_us": statistics.mean(s),
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.file_path.exists():
        print(f"ERROR: file does not exist: {args.file_path}")
        return 1
    file_size = args.file_path.stat().st_size
    print(f"Test 3 — NEXTENT under decode access pattern")
    print(f"File: {args.file_path}  size: {file_size / (1024**3):.2f} GiB")

    results: dict[str, Any] = {
        "file_path": str(args.file_path),
        "file_size_gb": file_size / (1024**3),
        "n_extents_per_pattern": args.n_extents_per_pattern,
        "page_size": PAGE_SIZE,
        "extent_size": EXTENT_SIZE,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # === Pattern A: Pure sequential first-load ===
    print("\n=== Pattern A: pure sequential (first-load simulation) ===")
    evict_cache_via_pressure(args.evict_cache_mb)
    with open(args.file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Touch enough pages to span ~50 MB of sequential reads (12800 pages)
        n_seq_pages = min(12800, file_size // PAGE_SIZE)
        latencies_A = time_pattern_A_pure_sequential(mm, file_size, n_seq_pages)
        mm.close()
    sA = summarize(latencies_A)
    results["pattern_A_pure_sequential"] = sA
    print(f"  n={sA['n']}  median={sA['median_us']:.2f} μs  "
          f"p10={sA['p10_us']:.2f}  p90={sA['p90_us']:.2f}  p99={sA['p99_us']:.2f}")
    print(f"  Pages touched in order across {n_seq_pages * PAGE_SIZE / (1024**2):.1f} MB")

    # === Pattern B: Random extent + full intra-extent ===
    print("\n=== Pattern B: random extent + full intra-extent sequential ===")
    evict_cache_via_pressure(args.evict_cache_mb)
    with open(args.file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        first_B, follow_B = time_pattern_B_random_extent_full(
            mm, file_size, args.n_extents_per_pattern, args.seed
        )
        mm.close()
    sB_first = summarize(first_B)
    sB_follow = summarize(follow_B)
    results["pattern_B_full_extent"] = {
        "first_page": sB_first,
        "intra_extent_followon": sB_follow,
    }
    print(f"  first-page-of-extent: median={sB_first['median_us']:.2f} μs  p99={sB_first['p99_us']:.2f}")
    print(f"  intra-extent followon: median={sB_follow['median_us']:.2f} μs  p99={sB_follow['p99_us']:.2f}")
    speedup_B = sB_first['median_us'] / sB_follow['median_us'] if sB_follow['median_us'] > 0 else float('inf')
    print(f"  followon speedup: {speedup_B:.1f}×")
    # Average per-page cost across full pattern (this is the deployment-relevant number)
    all_B = first_B + follow_B
    sB_all = summarize(all_B)
    print(f"  AMORTIZED per-page across full extents: median={sB_all['median_us']:.2f}  "
          f"mean={sB_all['mean_us']:.2f}  p99={sB_all['p99_us']:.2f}")
    results["pattern_B_full_extent"]["amortized_per_page"] = sB_all

    # === Pattern C: Random extent + sparse within ===
    print(f"\n=== Pattern C: random extent + sparse-within ({args.n_sparse_pages_per_extent} pages/extent) ===")
    evict_cache_via_pressure(args.evict_cache_mb)
    with open(args.file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        first_C, follow_C = time_pattern_C_random_extent_sparse(
            mm, file_size, args.n_extents_per_pattern,
            args.n_sparse_pages_per_extent, args.seed + 1000
        )
        mm.close()
    sC_first = summarize(first_C)
    sC_follow = summarize(follow_C)
    results["pattern_C_sparse_extent"] = {
        "first_page": sC_first,
        "intra_extent_followon": sC_follow,
        "sparse_pages_per_extent": args.n_sparse_pages_per_extent,
    }
    print(f"  first-page-of-extent: median={sC_first['median_us']:.2f} μs  p99={sC_first['p99_us']:.2f}")
    print(f"  intra-extent followon (sparse): median={sC_follow['median_us']:.2f} μs  p99={sC_follow['p99_us']:.2f}")
    if sC_follow['median_us'] > 0:
        print(f"  followon speedup: {sC_first['median_us'] / sC_follow['median_us']:.1f}×")
    all_C = first_C + follow_C
    sC_all = summarize(all_C)
    print(f"  AMORTIZED per-page across sparse-within: median={sC_all['median_us']:.2f}  "
          f"mean={sC_all['mean_us']:.2f}  p99={sC_all['p99_us']:.2f}")
    results["pattern_C_sparse_extent"]["amortized_per_page"] = sC_all

    # Verdict
    print("\n=== Verdict ===")
    # If amortized B per-page is < 5× the synthetic same-extent followon (1.9 μs),
    # then NEXTENT is the deployment substrate.
    target_pageoff_us = 10.0
    target_amortized_us = 20.0
    if sB_all['median_us'] <= target_amortized_us and sB_follow['median_us'] <= target_pageoff_us:
        verdict = (f"GO — full-extent decode pattern fully exploits NEXTENT. "
                   f"Amortized per-page = {sB_all['median_us']:.2f} μs (target ≤{target_amortized_us}); "
                   f"followon = {sB_follow['median_us']:.2f} μs (target ≤{target_pageoff_us}). "
                   f"NEXTENT is the deployment substrate for PFOR.")
    elif sB_all['median_us'] <= 50.0:
        verdict = (f"PARTIAL — moderate NEXTENT engagement. "
                   f"Amortized per-page = {sB_all['median_us']:.2f} μs.")
    else:
        verdict = (f"WEAKER THAN EXPECTED — amortized per-page = {sB_all['median_us']:.2f} μs. "
                   f"NEXTENT engages but not as efficiently as the pure-random benchmark suggested.")
    results["verdict"] = verdict
    print(verdict)

    # Sparse vs full comparison
    if sC_all["median_us"] > 0 and sB_all["median_us"] > 0:
        sparse_to_full = sC_all["median_us"] / sB_all["median_us"]
        print(f"\nSparse vs full intra-extent comparison: sparse pattern is "
              f"{sparse_to_full:.1f}× the per-page latency of full pattern")
        if sparse_to_full < 1.5:
            print(f"  → NEXTENT is 'all or nothing': prefetch fires regardless of "
                  f"how much of the extent we actually use")
        else:
            print(f"  → Prefetch benefit depends on consumption ratio; partial "
                  f"extent usage costs more per page")

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
