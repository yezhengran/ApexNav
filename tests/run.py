#!/usr/bin/env python3
"""Explicit test groups. Default unit tests never load models or start navigation."""
import argparse
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'docker')]


class LocalTestLoader(unittest.TestLoader):
    """Fixture imports do not re-run the imported module's TestCase classes."""
    def loadTestsFromModule(self, module, *, pattern=None):
        return self.suiteClass(
            self.loadTestsFromTestCase(value)
            for _, value in inspect.getmembers(module, inspect.isclass)
            if issubclass(value, unittest.TestCase) and value.__module__ == module.__name__
        )


def python_suite(group, verbose):
    suite = LocalTestLoader().discover(str(ROOT / 'tests' / group),
                                      top_level_dir=str(ROOT))
    return unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite).wasSuccessful()


def cpp_suite():
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ is required; run inside the existing development container')
    with tempfile.TemporaryDirectory(prefix='apexnav-cpp-tests-') as tmp:
        for source in sorted((ROOT / 'tests/cpp').glob('test_*.cpp')):
            binary = str(Path(tmp) / source.stem)
            subprocess.run([compiler, '-std=c++17', '-O2', '-I/usr/include/eigen3',
                            '-I' + str(ROOT / 'src/planner/exploration_manager/include'),
                            '-I' + str(ROOT / 'src/planner/plan_env/include'),
                            str(source), '-o', binary], check=True)
            subprocess.run([binary], check=True)
            print(f'PASS {source.relative_to(ROOT)}', flush=True)
    return True


def require_idle_planner():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        names = {Path(arg.decode(errors='replace')).name for arg in argv if arg}
        if names & {'habitat_evaluation.py', 'run_batch.py', 'exploration_node', 'controller.py'}:
            raise RuntimeError('Integration tests require an idle evaluation/planner')


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['unit', 'cpp', 'integration', 'all'], nargs='?', default='unit')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    groups = ['unit', 'cpp', 'integration'] if args.group == 'all' else [args.group]
    for group in groups:
        if group == 'integration':
            require_idle_planner()
        passed = cpp_suite() if group == 'cpp' else python_suite(group, args.verbose)
        if not passed:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
