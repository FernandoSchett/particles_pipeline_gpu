#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
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
    if data.size == 0:
        return False, {}, 0, 0
    ranks, counts = np.unique(data["rank"], return_counts=True)
    cmin, cmax = counts.min(), counts.max()
    balanced = (cmax - cmin) <= tol
    return balanced, dict(zip(ranks.astype(int), counts.tolist())), int(cmin), int(cmax)

def main():
    ap = argparse.ArgumentParser(description="Check per-rank ordering and load balance in .par files")
    ap.add_argument("parfiles", nargs="+", type=Path, help="paths to .par files")
    ap.add_argument("--tol", type=int, default=2, help="allowed imbalance (max particle count difference)")
    ap.add_argument("--key-partitions", type=int, metavar="RANKS",
                    help="verify exact quantile partitions, keeping equal Morton keys together")
    args = ap.parse_args()
    if args.key_partitions is not None and args.key_partitions < 1:
        ap.error("--key-partitions must be positive")

    exit_code = 0
    for index, parfile in enumerate(args.parfiles):
        if index:
            print()

        if not parfile.is_file():
            print(f"Error: file not found: {parfile}", file=sys.stderr)
            exit_code |= 4
            continue

        try:
            data = read_par(parfile)
        except (OSError, ValueError) as error:
            print(f"Error reading {parfile}: {error}", file=sys.stderr)
            exit_code |= 4
            continue

        per_rank_sorted = check_sorted_per_rank(data)
        balanced, counts_map, cmin, cmax = check_balance(data, args.tol)
        if args.key_partitions is not None:
            # Each splitter is the first key reaching ceil(N * rank / ranks).
            # upper_bound retains every occurrence of that key on the left.
            ranks = args.key_partitions
            keys = np.sort(data["key"])
            if keys.size:
                targets = [(keys.size * rank + ranks - 1) // ranks for rank in range(1, ranks)]
                splitters = keys[np.asarray(targets, dtype=np.int64) - 1]
                expected_owner = np.searchsorted(splitters, data["key"], side="left")
                balanced = bool(np.all(data["rank"] == expected_owner))
            else:
                balanced = False
            counts_map = {rank: counts_map.get(rank, 0) for rank in range(ranks)}
            cmin, cmax = min(counts_map.values()), max(counts_map.values())

        print(f"File: {parfile.name}")
        print("Ordering per rank (keys non-decreasing):")
        for r in sorted(per_rank_sorted):
            status = "OK" if per_rank_sorted[r] else "FAIL"
            print(f"  rank {r:4d}: {status}")

        print("\nLoad balance:")
        if args.key_partitions is None:
            print(f"  tolerance = {args.tol}")
        else:
            print(f"  checking exact Morton quantile partitions across {args.key_partitions} ranks")
        print(f"  min={cmin}  max={cmax}  diff={cmax - cmin}  -> {'OK' if balanced else 'FAIL'}")
        print("  particle counts per rank:")
        for r in sorted(counts_map):
            print(f"    rank {r:4d}: {counts_map[r]}")

        if not per_rank_sorted or not all(per_rank_sorted.values()):
            exit_code |= 1
        if not balanced:
            exit_code |= 2

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
