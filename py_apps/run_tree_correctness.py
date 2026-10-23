#!/usr/bin/env python3
"""Run real CPU/GPU binaries on adversarial PEPC fixtures and validate outputs."""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CASES = {
    'single_particle_empty_ranks': [(-3., 7., 2.)],
    'repeated_keys': [(0., 0., 0.)] * 129,
    'mixed_duplicates': [(0., 0., 0.)] * 80 + [(1., 1., 1.)] * 17 + [(2., 0., 1.)] * 3,
    'concentrated': [(1e-8*i, 1e-9*(i % 5), -1e-8*i) for i in range(100)] + [(-10., -10., -10.), (10., 10., 10.)],
    'boundaries': [(x, y, z) for x in (-1., 0., 1.) for y in (-1., 0., 1.) for z in (-1., 0., 1.)]
                  + [(math.nextafter(0., -math.inf), 0., 0.),
                     (math.nextafter(0., math.inf), 0., 0.),
                     (math.nextafter(1., 0.), 1., -1.)],
}


def write_pepc(path, coords):
    """Create the project's PEPC layout without invoking its C++ writer."""
    with path.open('wb') as stream:
        stream.write(struct.pack('<q', len(coords)) + bytes(120))
        for index, xyz in enumerate(coords):
            record = bytearray(216)
            struct.pack_into('<ddd', record, 0, *xyz)
            struct.pack_into('<q', record, 48, index)  # PEPC label; coordinates define the test oracle.
            stream.write(record)


def checked(command, **kwargs):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180, **kwargs)
    if result.returncode:
        raise ValueError(f"Command failed ({result.returncode}): {' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def run_cases(args, output):
    scripts = Path(__file__).resolve().parent
    cases = []
    for name, coords in CASES.items():
        source = output / f'{name}.mpi'
        write_pepc(source, coords)
        cases.append((name, source))
    if args.input_pepc:
        cases.append(('input_snapshot', args.input_pepc.resolve()))
    for name, source in cases:
        directory = output / name
        directory.mkdir()
        env = dict(os.environ, P_SFC_OUTPUT_DIR=str(directory), OMP_NUM_THREADS='1', P_SFC_VERIFY_REPLICAS='1')
        launcher = [args.launcher, *args.launcher_arg, args.rank_flag, str(args.ranks)]
        checked(launcher + [str(args.executable.resolve()), 'pepc_shape', '3', '67',
                            'strong', 'table', str(source)], cwd=directory, env=env)
        pars = list(directory.glob('*.par'))
        suffix = '.tree' if args.device == 'cpu' else '.gtree'
        trees = list(directory.glob('*' + suffix))
        if len(pars) != 1 or len(trees) != 1:
            raise ValueError(f'{name}: expected one particle file and one tree')
        validator = 'verify_tree_file.py' if args.device == 'cpu' else 'verify_global_tree_file.py'
        checked([sys.executable, str(scripts / validator), str(trees[0]),
                 '--particles', str(pars[0]), '--input-pepc', str(source)])
        print(f'{args.device} ranks={args.ranks} {name}: PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--launcher-arg', action='append', default=[])
    parser.add_argument('--rank-flag', default='-np')
    parser.add_argument('--input-pepc', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='parent directory for a unique persistent test run; otherwise use temporary files')
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('--ranks must be positive')
    try:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=f'{args.device}-n{args.ranks}-', dir=args.output_dir.resolve()))
            print(f'Artifacts: {output}', flush=True)
            run_cases(args, output)
        else:
            with tempfile.TemporaryDirectory(prefix='tree-correctness-') as directory:
                run_cases(args, Path(directory))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
