# Tree correctness tests

## Run locally

After configuring and building with `BUILD_TESTING=ON` (the default):

```bash
ctest --test-dir build --output-on-failure
```

This runs the C++ tests, 14 Python validator tests, and CPU end-to-end tests
with 1, 2, and 4 MPI ranks. Each end-to-end test uses five generated PEPC
fixtures and the bundled 10,000-particle snapshot.

Enable GPU end-to-end tests on a machine with working CUDA and CUDA-aware MPI:

```bash
cmake -S . -B build -DP_SFC_TEST_GPU=ON
cmake --build build
ctest --test-dir build -R p_sfc_gpu_tree_reference --output-on-failure
```

These tests use 1, 2, and 4 ranks. To run only one GPU rank:

```bash
python3 py_apps/run_tree_correctness.py --device gpu \
  --executable build/src/gpu_mpi_p_sfc_exe --ranks 1
```

The runner uses temporary output directories. Add `--output-dir results/tests/edge_cases`
to retain fixtures and generated files in a unique subdirectory.

## Checks

- **Particle/tree consistency:** recompute Morton keys from input coordinates;
  check node intervals, counts, leaf ranges, and refinement against Python references.
- **Conservation:** compare coordinate multisets before and after MPI redistribution,
  including multiplicities. Equal total counts alone cannot pass this check.
- **Distributed CPU tree:** check quantile ownership, branch ranges, global skeleton,
  and remote stubs against the corresponding branch on its owner rank.
- **Distributed GPU tree:** compare cornerstone boundaries, node keys, child offsets,
  and level offsets exactly between ranks when `P_SFC_VERIFY_REPLICAS=1`.
  The runner enables this check during export, outside timed tree construction.
- **Independent references:** `tree_reference.py` builds a simple recursive sparse
  hashed tree and a full-child cornerstone tree. Their stopping rules differ,
  so CPU and GPU node lists are not expected to match each other.
- **Edge cases:** one particle with empty ranks, 129 coincident particles, mixed
  duplicate keys, concentrated particles, domain boundaries and adjacent floats.
- **Validator regression tests:** deliberately corrupt counts, ownership, ranges,
  keys, flags, topology, and binary files and require rejection.

## Validate existing files

```bash
python3 py_apps/verify_tree_file.py output.tree --particles output.par \
  --input-pepc input.mpi
python3 py_apps/verify_global_tree_file.py output.gtree --particles output.par \
  --input-pepc input.mpi
```

Alternatively, `--before-par before.par` checks conservation of coordinates and
keys against a particle dump taken before redistribution. Without `--particles`,
the validators perform structural checks only.

The PEPC reader/fixtures assume the project's current binary layout: 128-byte
header and 216-byte records. Particle dumps discard PEPC particle labels;
conservation therefore checks observable coordinates and their multiplicities,
not identity between otherwise identical particles. Empty ranks are exercised
end to end; a globally empty input is covered only by the Python reference tests.

## Slurm

`bash scripts/tests/run_all_tests.sh` checks the normal snapshot runs against
their particle files and PEPC input, and runs the edge cases with up to four
ranks per device. Set `RUN_EDGE_CASES=0` to skip that additional fixture suite.
GPU runs enable the replica check. Artifacts stay under `results/tests/` or
`P_SFC_OUTPUT_DIR`.
