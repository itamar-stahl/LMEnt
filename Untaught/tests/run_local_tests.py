"""Run every local suite and report once.

    python tests/run_local_tests.py

Three suites, three different questions:

    units        does each component behave correctly, in isolation?
    integration  do we still hold up OLMo-core's side of every contract?
    refactoring  is the package structurally complete and coherent today?

Exit code is 0 only if all three pass. Nothing here needs a GPU, the dataset,
Elasticsearch or the cluster -- for those, see tests/remote/run_remote_tests.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("units", "test_local_units.py"),
    ("integration", "test_local_integration.py"),
    ("refactoring", "test_local_refactoring.py"),
]


def main() -> int:
    results = []
    for name, script in SUITES:
        print(f"\n>>> {name} ({script})\n", flush=True)
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)])
        results.append((name, rc))

    width = 74
    print("\n" + "=" * width)
    print("  LOCAL TEST RUN -- summary")
    print("=" * width)
    for name, rc in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name}")
    failed = [n for n, rc in results if rc != 0]
    print("-" * width)
    print(f"  {len(results) - len(failed)}/{len(results)} suites passed"
          + (f" -- failed: {', '.join(failed)}" if failed else ""))
    print("=" * width)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
