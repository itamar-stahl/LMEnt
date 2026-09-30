#!/usr/bin/env python
"""Re-score a topic from the frozen sets file and check the committed results
still come back.

Every other check in this repository compares committed artifacts against each
other -- `validate_accwinners.py` runs twelve of them, `audit.py` and
`ensembles/audit_results.py` more. None of them runs a model. They can all pass
while the code that produced the numbers has drifted away from them.

This one closes that loop. It takes two freshly scored trees, recomputes the
full-versus-twin comparison with `compare.py`, and asserts the result against
`results_accwinners/<topic>/results.json` -- the file the paper's Panel B and C
are printed from. Expected values are read from that file, never hardcoded, so
this cannot drift from the published numbers: if they are regenerated, the
target moves with them.

It is a regression test, not a grader instruction. Running it needs both 5 GB
checkpoints and a GPU.

Two steps. First score the two models (about 22 minutes each, one GPU):

    S=ember_eval/nll_kl/slurm/score.slurm
    V=<some run dir>
    sbatch --job-name=vfyFull-rome --export=ALL,ROOT=$PWD,\\
        MODEL=<hf-models>/lment-1b-control-2e-b131k,OUT=$V/scored/full,\\
        SETS=$PWD/ember_eval/nll_kl/sets/rome.json $S
    sbatch --job-name=vfyTwin-rome --export=ALL,ROOT=$PWD,\\
        MODEL=<hf-models>/lment-1b-norome-2e-b131k,OUT=$V/scored/twin,\\
        SETS=$PWD/ember_eval/nll_kl/sets/rome.json $S

Do not add `--exclude` on the command line: it replaces score.slurm's node list
rather than adding to it. Then check, with `--dependency=afterok` on the two:

    python ember_eval/nll_kl/verify_reproduction.py --topic rome \\
        --scored $V/scored --out $V

Exits non-zero if any set fails to reproduce.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMPARE = HERE / "compare.py"

TOL = 5e-4


def committed(topic: str) -> Dict[str, Any]:
    p = HERE / "results_accwinners" / topic / "results.json"
    if not p.is_file():
        raise SystemExit(f"no committed results at {p}")
    return json.loads(p.read_text())


def run_compare(args: List[str], out: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(COMPARE), *args, "--out", str(out)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


def collect(ref: Dict[str, Any], got: Dict[str, Any], field: str,
            label: str) -> List[Tuple[str, float, float]]:
    """(name, got, want) for every set the committed row reports this field on."""
    rows = []
    for name, entry in ref["per_set"].items():
        stat = entry.get(field)
        if not stat or stat.get("n", 0) == 0:
            continue
        mine = got["per_set"].get(name, {}).get(field)
        if not mine:
            rows.append((f"{label} {name} MISSING", float("nan"), stat["mean"]))
            continue
        rows.append((f"{label} {name} mean", mine["mean"], stat["mean"]))
        rows.append((f"{label} {name} sd", mine["sd"], stat["sd"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, choices=["rome", "baseball", "ai"])
    ap.add_argument("--scored", required=True,
                    help="directory holding full/<topic>/ and twin/<topic>/")
    ap.add_argument("--out", required=True, help="where to write the comparison JSONs")
    a = ap.parse_args()

    ref = committed(a.topic)
    scored = Path(a.scored)
    full, twin = scored / "full" / a.topic, scored / "twin" / a.topic
    for d in (full, twin):
        if not (d / "records.json").is_file():
            raise SystemExit(f"not a scored tree: {d}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"topic={a.topic}  full={ref['full']}  twin={ref['twin']}")
    print(f"targets from {HERE / 'results_accwinners' / a.topic / 'results.json'}\n")

    # NLL: compare.py's convention is evaluated - reference, so twin - full.
    nll_ref = next(r for r in ref["rows"] if r["label"] == "Twin vs Full")
    nll_got = run_compare(["--eval", str(twin), "--ref", str(full)],
                          out / f"verify_{a.topic}_nll.json")

    # KL: the convention is KL(kl_reference || evaluated), so the twin is the
    # kl-ref and the full model is evaluated -- kl_rows' "Full" entry.
    kl_ref = next(r for r in ref["kl_rows"] if r.get("label") == "Full")
    kl_got = run_compare(["--eval", str(full), "--ref", str(twin),
                          "--kl-ref", str(twin)],
                         out / f"verify_{a.topic}_kl.json")

    checks = (collect(nll_ref, nll_got, "nll_diff", "NLL twin-full")
              + collect(kl_ref, kl_got, "kl", "KL(twin||full)"))

    print("\n=== reproduction check ===")
    bad = 0
    for name, got, want in checks:
        ok = abs(got - want) <= TOL
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} got {got:.4f}  want {want:.4f}")
    print(f"\n{len(checks) - bad}/{len(checks)} reproduced")

    summary = {"topic": a.topic, "tolerance": TOL, "n_checks": len(checks),
               "n_failed": bad,
               "checks": [{"name": n, "got": g, "want": w} for n, g, w in checks]}
    (out / f"verify_{a.topic}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
