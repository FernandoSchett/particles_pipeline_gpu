#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys

# .par format:
# int64 total_particles
# repeated total_particles times: int32 mpi_rank, int64 key, float64 coord[3]
DT = np.dtype([("rank",  "<i4"),
               ("key",   "<i8"),
               ("coord", "<f8", (3,))])

def read_par(path):
    with open(path, "rb") as f:
        n = np.fromfile(f, dtype="<i8", count=1)
        if n.size != 1:
            raise ValueError("Invalid .par file (missing header)")
        n = int(n[0])
        data = np.fromfile(f, dtype=DT, count=n)
        if data.size != n:
            raise ValueError(f"Inconsistent size: header={n}, read={data.size}")
        return data

def check_sorted_per_rank(data):
    ranks = np.unique(data["rank"])
    per_rank_ok = {}
    for r in ranks:
        k = data["key"][data["rank"] == r]
        ok = True
        if k.size >= 2:
            ok = np.all(k[:-1] <= k[1:])  # non-decreasing
        per_rank_ok[int(r)] = bool(ok)
    return per_rank_ok

def check_balance(data, tol):
    ranks, counts = np.unique(data["rank"], return_counts=True)
    cmin, cmax = counts.min(), counts.max()
    balanced = (cmax - cmin) <= tol
    return balanced, dict(zip(ranks.astype(int), counts.tolist())), int(cmin), int(cmax)

def main():
    ap = argparse.ArgumentParser(description="Check per-rank ordering and load balance in a .par file")
    ap.add_argument("parfile", help="path to .par file")
    ap.add_argument("--tol", type=int, default=2, help="allowed imbalance (max particle count difference)")
    args = ap.parse_args()

    if not os.path.isfile(args.parfile):
        print(f"Error: file not found: {args.parfile}", file=sys.stderr)
        sys.exit(2)

    data = read_par(args.parfile)

    per_rank_sorted = check_sorted_per_rank(data)
    balanced, counts_map, cmin, cmax = check_balance(data, args.tol)

    print(f"File: {os.path.basename(args.parfile)}")
    print("Ordering per rank (keys non-decreasing):")
    for r in sorted(per_rank_sorted):
        status = "OK" if per_rank_sorted[r] else "FAIL"
        print(f"  rank {r:4d}: {status}")

    print("\nLoad balance:")
    print(f"  tolerance = {args.tol}")
    print(f"  min={cmin}  max={cmax}  diff={cmax - cmin}  -> {'OK' if balanced else 'FAIL'}")
    print("  particle counts per rank:")
    for r in sorted(counts_map):
        print(f"    rank {r:4d}: {counts_map[r]}")

    # exit code for automation/CI
    exit_code = 0
    if not all(per_rank_sorted.values()):
        exit_code |= 1
    if not balanced:
        exit_code |= 2
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
