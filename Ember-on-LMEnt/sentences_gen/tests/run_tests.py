"""Run every sentences_gen suite and report once.

    python tests/run_tests.py            # all three
    python tests/run_tests.py units      # one or more by name

Three suites, three different questions:

    units       does the logic behave correctly, in isolation?   (offline)
    live_es     does this deployment match what the module assumes about it?
    end_to_end  does the whole pipeline produce a usable corpus?

``units`` runs anywhere. The other two need Elasticsearch, so run them on the
login node (c-003) after sourcing Untaught's ``activate_env.sh``; without it
they report SKIP with the reason, not FAIL.

Exit code is 0 only if nothing failed. Skips do not fail the run -- a laptop
should be able to run this and see two clean skips.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SUITES = [
    ("units", "test_units.py", "offline"),
    ("live_es", "test_live_es.py", "needs Elasticsearch"),
    ("end_to_end", "test_end_to_end.py", "needs Elasticsearch"),
]


def main(argv: list[str]) -> int:
    wanted = set(argv) or {name for name, _, _ in SUITES}
    unknown = wanted - {name for name, _, _ in SUITES}
    if unknown:
        print(f"unknown suite(s): {sorted(unknown)}", file=sys.stderr)
        print(f"available: {[s[0] for s in SUITES]}", file=sys.stderr)
        return 2

    # The suites import the module and testlib by plain name.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(HERE.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    results = []
    for name, script, note in SUITES:
        if name not in wanted:
            continue
        code = subprocess.call([sys.executable, str(HERE / script)], env=env)
        results.append((name, code, note))

    width = 74
    print("\n" + "=" * width)
    print("  sentences_gen -- test run summary")
    print("=" * width)
    for name, code, note in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name:12} ({note})")
    failed = [n for n, c, _ in results if c != 0]
    print("-" * width)
    print(f"  {len(results) - len(failed)}/{len(results)} suites passed"
          + (f" -- failed: {', '.join(failed)}" if failed else ""))
    if not failed:
        print("  (individual SKIPs above are preconditions, not failures)")
    print("=" * width)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
