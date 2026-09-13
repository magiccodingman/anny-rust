#!/usr/bin/env python3
"""Run native generation against an existing Python-generated fixture directory.

Only Python's standard library is required. Outputs and detailed comparisons are
written under each fixture directory. Missing fixtures are a failure, never a pass.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('target/release/anny'))
    parser.add_argument('--assets', type=Path, default=Path('data'))
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args()
    if not args.fixtures.is_dir():
        parser.error(f'Missing fixture directory: {args.fixtures}')
    cases = [args.fixtures / name for name in args.cases] if args.cases else sorted(args.fixtures.iterdir())
    if args.cases:
        for case in cases:
            if not case.is_dir():
                parser.error(f'Missing requested fixture case: {case}')
    cases = [case for case in cases if case.is_dir()]
    if not cases:
        parser.error('No fixture cases found; generate them with tools/export_reference.py first.')
    results = []
    for case in cases:
        for name in ('reference.safetensors', 'config.json', 'params.json'):
            if not (case / name).is_file():
                parser.error(f'Missing {case / name}')
        actual = case / 'rust.safetensors'
        report = case / 'comparison.json'
        with (case / 'native.log').open('w') as log:
            cmd = [str(args.binary.resolve()), 'generate', '--assets', str(args.assets.resolve()),
                   '--config', str(case / 'config.json'), '--params', str(case / 'params.json'),
                   '--output', str(actual)]
            build = subprocess.run(cmd, stdout=log, stderr=log, check=False)
            if build.returncode:
                result = {'case': case.name, 'passed': False, 'error': 'generation failed; see native.log'}
            else:
                check = subprocess.run([str(args.binary.resolve()), 'compare', '--actual', str(actual),
                       '--expected', str(case / 'reference.safetensors'), '--report', str(report)],
                       stdout=log, stderr=log, check=False)
                result = {'case': case.name, 'passed': check.returncode == 0}
                if report.is_file():
                    result['comparison'] = json.loads(report.read_text())
        results.append(result)
        print(f"{case.name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(results, indent=2) + '\n')
    raise SystemExit(0 if all(result['passed'] for result in results) else 1)


if __name__ == '__main__':
    main()
