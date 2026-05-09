r"""NVMe page-fault round-trip benchmark on Windows 11 NTFS mmap.

Hard prerequisite for both:
  - Reach v2 (TC-MR): NVMe overlapped-I/O latency drives the 70B@≥1.5 tok/s ceiling
  - Unconventional v2 (PFOR+NEXTENT): branches the cascade between optimistic
    (E0a, ≤30 μs/fault: original PFOR+WSTLO works) and realistic (E0b,
    100-300 μs/fault: must use NEXTENT extent prefetch + ZIPFLOCK + iGPU overlap)

Method:
  1. Allocate sparse 8 GiB file
  2. Memory-map it (MapViewOfFile / mmap.mmap)
  3. Drop standby cache between runs (EmptyStandbyList equivalent — best effort
     since this requires admin; fall back to filling RAM with junk to evict cache)
  4. Touch N random 4 KB pages cold; measure per-fault latency
  5. Repeat at thread counts 1, 2, 4, 8

Branch decision:
  - E0a if median < 30 μs/fault — original PFOR+WSTLO viable
  - E0b if median 100-300 μs/fault — pivot to NEXTENT extent prefetch
  - HARD FAIL if median > 300 μs/fault — entire OS-paging direction dies

Also measures:
  - Sequential-extent prefetch latency (touch one page in a 1 MB extent, then
    measure subsequent same-extent page latencies — should be near-zero if
    Cache Manager engages)
  - QD effect (single-thread vs multi-thread random faults)
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


PAGE_SIZE = 4096
EXTENT_TEST_SIZE = 1024 * 1024  # 1 MB extent
DEFAULT_FILE_SIZE = 8 * 1024 * 1024 * 1024  # 8 GiB sparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file-path", type=Path,
                   default=Path("C:/Users/power/AppData/Local/Temp/nocturnal_pf_bench.bin"))
    p.add_argument("--file-size-gb", type=float, default=8.0,
                   help="Size of sparse test file in GiB")
    p.add_argument("--n-faults", type=int, default=10000,
                   help="Number of cold page faults per measurement")
    p.add_argument("--thread-counts", default="1,2,4,8")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/stage0/opus_pipeline/round1/e0_nvme_pagefault_results.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evict-cache-mb", type=int, default=4096,
                   help="MB of dummy buffer to allocate to evict file cache (best-effort if no admin)")
    return p.parse_args()


def create_sparse_file(path: Path, size_bytes: int) -> None:
    """Create a sparse file of given size, filling with non-trivial bytes
    so each page actually requires a real read.

    On NTFS we set the file size with SetFileValidData where possible,
    otherwise we just write data — the OS/drive should compress zeros so
    we write a deterministic pattern instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size_bytes:
        print(f"  reusing existing {path} ({size_bytes / (1024**3):.1f} GiB)")
        return
    print(f"  creating {path} of size {size_bytes / (1024**3):.1f} GiB (this writes data, may take a minute)...")
    rng = random.Random(0)
    chunk_size = 1024 * 1024
    chunk = bytes((rng.randint(0, 255) for _ in range(chunk_size)))
    with open(path, "wb") as f:
        written = 0
        while written < size_bytes:
            to_write = min(chunk_size, size_bytes - written)
            f.write(chunk[:to_write])
            written += to_write
            if written % (256 * 1024 * 1024) == 0:
                print(f"    {written / (1024**3):.1f} / {size_bytes / (1024**3):.1f} GiB")


def evict_cache_via_pressure(buffer_mb: int) -> None:
    """Best-effort cache eviction: allocate large buffer, touch every page,
    discard. Forces OS to reclaim file cache pages.
    """
    print(f"  evicting cache via {buffer_mb} MB pressure buffer...")
    n_bytes = buffer_mb * 1024 * 1024
    buf = bytearray(n_bytes)
    # Touch every 4 KB page
    for i in range(0, n_bytes, PAGE_SIZE):
        buf[i] = (buf[i] + 1) & 0xFF
    del buf


def time_random_faults(
    mm: mmap.mmap, file_size: int, n_faults: int, seed: int
) -> tuple[list[float], int]:
    """Touch N random pages, measuring per-fault latency."""
    rng = random.Random(seed)
    n_pages = file_size // PAGE_SIZE
    # Pre-generate random page offsets to avoid RNG overhead in the timing loop
    page_indices = [rng.randrange(0, n_pages) for _ in range(n_faults)]

    latencies = []
    accum = 0  # prevent compiler/dead-code elimination
    for pidx in page_indices:
        offset = pidx * PAGE_SIZE
        t0 = time.perf_counter_ns()
        b = mm[offset]  # read 1 byte from a random page → triggers page fault
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # microseconds
        accum += b
    return latencies, accum


def time_extent_prefetch(
    mm: mmap.mmap, file_size: int, n_extents: int, seed: int
) -> dict[str, list[float]]:
    """For each random 1 MB extent, time:
      - first-page fault latency
      - subsequent same-extent page fault latencies (should be near-zero
        if Cache Manager prefetch engaged)
    """
    rng = random.Random(seed)
    pages_per_extent = EXTENT_TEST_SIZE // PAGE_SIZE
    n_extents_in_file = file_size // EXTENT_TEST_SIZE

    first_page_latencies: list[float] = []
    same_extent_latencies: list[float] = []

    accum = 0
    for _ in range(n_extents):
        ext_idx = rng.randrange(0, n_extents_in_file)
        ext_base = ext_idx * EXTENT_TEST_SIZE

        # Touch first page of extent (cold fault)
        t0 = time.perf_counter_ns()
        b = mm[ext_base]
        t1 = time.perf_counter_ns()
        first_page_latencies.append((t1 - t0) / 1000.0)
        accum += b

        # Touch a few subsequent pages in the same extent
        for k in range(1, min(8, pages_per_extent)):
            offset = ext_base + k * PAGE_SIZE
            t0 = time.perf_counter_ns()
            b = mm[offset]
            t1 = time.perf_counter_ns()
            same_extent_latencies.append((t1 - t0) / 1000.0)
            accum += b

    return {
        "first_page_latencies_us": first_page_latencies,
        "same_extent_latencies_us": same_extent_latencies,
    }


def run_threaded_faults(
    file_path: Path, file_size: int, n_faults: int, n_threads: int, seed: int
) -> list[list[float]]:
    """Spawn n_threads each touching n_faults/n_threads random pages.
    Returns list of per-thread latency lists.
    """
    per_thread = n_faults // n_threads
    results: list[list[float]] = [[] for _ in range(n_threads)]

    def worker(tid: int) -> None:
        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            lat, _ = time_random_faults(mm, file_size, per_thread, seed + tid)
            results[tid] = lat
            mm.close()

    threads = [Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
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


def classify_branch(median_us: float) -> str:
    if median_us <= 30:
        return "E0a (optimistic) — original PFOR+WSTLO viable"
    if median_us <= 300:
        return "E0b (realistic) — pivot to NEXTENT+ZIPFLOCK+iGPU overlap"
    return "HARD FAIL — OS paging direction not viable"


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_size = int(args.file_size_gb * 1024 * 1024 * 1024)
    thread_counts = [int(x) for x in args.thread_counts.split(",")]

    print(f"Page-fault benchmark on {args.file_path}")
    create_sparse_file(args.file_path, file_size)

    results: dict[str, Any] = {
        "platform": sys.platform,
        "file_path": str(args.file_path),
        "file_size_gb": args.file_size_gb,
        "n_faults_per_run": args.n_faults,
        "page_size": PAGE_SIZE,
        "extent_test_size": EXTENT_TEST_SIZE,
        "timestamp": datetime.now(UTC).isoformat(),
        "single_thread_random": {},
        "multi_thread_random": {},
        "extent_prefetch": {},
    }

    # --- Single-thread random faults ---
    print("\n=== Single-thread random fault latency ===")
    evict_cache_via_pressure(args.evict_cache_mb)
    with open(args.file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        latencies, _ = time_random_faults(mm, file_size, args.n_faults, args.seed)
        mm.close()
    summary = summarize_latencies(latencies)
    results["single_thread_random"] = summary
    print(f"  n={summary['n']}  median={summary['median_us']:.1f} us  "
          f"p10={summary['p10_us']:.1f}  p90={summary['p90_us']:.1f}  "
          f"p99={summary['p99_us']:.1f}  max={summary['max_us']:.1f}")

    branch = classify_branch(summary["median_us"])
    results["branch_decision"] = branch
    print(f"\n  BRANCH: {branch}")

    # --- Multi-thread random faults ---
    print("\n=== Multi-thread random fault latency ===")
    for nt in thread_counts:
        if nt == 1:
            continue  # already measured
        evict_cache_via_pressure(args.evict_cache_mb)
        thread_results = run_threaded_faults(
            args.file_path, file_size, args.n_faults, nt, args.seed
        )
        flat = [x for sub in thread_results for x in sub]
        summary = summarize_latencies(flat)
        results["multi_thread_random"][f"threads_{nt}"] = summary
        print(f"  threads={nt}  n={summary['n']}  median={summary['median_us']:.1f} us  "
              f"p99={summary['p99_us']:.1f}")

    # --- Extent prefetch test ---
    print("\n=== Extent prefetch latency ===")
    evict_cache_via_pressure(args.evict_cache_mb)
    with open(args.file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ext = time_extent_prefetch(mm, file_size, n_extents=200, seed=args.seed + 100)
        mm.close()
    first = summarize_latencies(ext["first_page_latencies_us"])
    same = summarize_latencies(ext["same_extent_latencies_us"])
    results["extent_prefetch"] = {
        "first_page": first,
        "same_extent_followon": same,
    }
    print(f"  first-page-of-extent:    median={first['median_us']:.1f} us  p99={first['p99_us']:.1f}")
    print(f"  same-extent followon:    median={same['median_us']:.1f} us  p99={same['p99_us']:.1f}")
    if same["median_us"] < first["median_us"] * 0.3:
        print(f"  ✓ Cache Manager prefetch ENGAGED — followon is "
              f"{first['median_us'] / same['median_us']:.1f}× faster than first-page")
        results["nextent_engaged"] = True
    else:
        print(f"  ✗ Cache Manager prefetch DID NOT engage clearly — "
              f"NEXTENT primitive may not work; fall back to PrefetchVirtualMemory")
        results["nextent_engaged"] = False

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"\nFINAL BRANCH: {branch}")
    print(f"NEXTENT engaged: {results['nextent_engaged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
