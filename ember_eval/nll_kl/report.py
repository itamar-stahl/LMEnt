#!/usr/bin/env python
"""Assemble one topic's result table from scored models.

Rows (reference is the second model of each pair, per the peers' table):
    Twin      vs Full      the ceiling: what never-having-learned looks like
    <erased>  vs Twin      how far each erasure is from that
    <erased>  vs Full      what each erasure changed (target) and shouldn't have (neighbour, unrelated)

Columns: the three TEST sets. Cells: both metrics with mean / CI / median /
frac-up, from compare.py. Writes <out>/table.md and <out>/results.json, the
latter carrying every per-item value.

    python ember_eval/nll_kl/report.py --topic rome --scored <scored root> \
        --full lment-1b-control-2e-b131k --twin lment-1b-norome-2e-b131k \
        --erased EMBER=lment-1b-rome-erased-b131k RMU=rmu_rome_L5hi_a100 ... --out results/rome
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from compare import compare, load_scored  # same directory

TEST_SETS = ("target_test", "neighbour_test", "unrelated_test")
PRETTY = {"target_test": "Target", "neighbour_test": "Neighbour", "unrelated_test": "Unrelated"}


def cell_nll(stat: Dict[str, Any]) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "—"
    lo, hi = stat["ci95"]
    return f"**{stat['mean']:+.3f}** [{lo:+.3f}, {hi:+.3f}] · med {stat['median']:+.3f} · up {stat['frac_pos']:.0%}"


def cell_kl(stat: Dict[str, Any]) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "—"
    lo, hi = stat["ci95"]
    return f"**{stat['mean']:.4f}** [{lo:.4f}, {hi:.4f}] · med {stat['median']:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="sets slug: rome | baseball | ai")
    ap.add_argument("--scored", required=True, help="scored root: <scored>/<model>/<topic>/")
    ap.add_argument("--full", required=True)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--erased", nargs="*", default=[], help="METHOD=scored-model-name ...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.scored)
    load = lambda name: load_scored(root / name / a.topic)  # noqa: E731
    full, twin = load(a.full), load(a.twin)
    erased = [(kv.split("=", 1)[0], kv.split("=", 1)[1]) for kv in a.erased]

    rows: List[Dict[str, Any]] = []
    rows.append({"label": "Twin vs Full", "res": compare(twin, full)})
    for method, name in erased:
        rows.append({"label": f"{method} vs Twin", "res": compare(load(name), twin)})
    for method, name in erased:
        rows.append({"label": f"{method} vs Full", "res": compare(load(name), full)})

    topic_name = full["meta"]["topic"]
    md = [f"# {topic_name} — answer-token NLL difference and KL, test sets\n",
          f"Full = `{a.full}`, Twin = `{a.twin}`. "
          "Convention: evaluated − reference for NLL; KL(reference ‖ evaluated). "
          "Cell = mean [95% bootstrap CI] · median · fraction of items with Δ>0 (NLL only). "
          "n = 50 per set.\n"]
    for metric, cell in (("nll_diff", cell_nll), ("kl", cell_kl)):
        title = "Answer NLL difference (nats/token)" if metric == "nll_diff" else "KL(ref ‖ eval) (nats)"
        md.append(f"\n## {title}\n")
        md.append("| Evaluated vs Reference | " + " | ".join(PRETTY[s] for s in TEST_SETS) + " |")
        md.append("|---|" + "---|" * len(TEST_SETS))
        for r in rows:
            cells = [cell(r["res"]["per_set"].get(s, {}).get(metric, {})) for s in TEST_SETS]
            md.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    md.append("\n## Erased models\n")
    for method, name in erased:
        md.append(f"- {method}: `{name}`")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(md) + "\n")
    (out / "results.json").write_text(json.dumps(
        {"topic": topic_name, "full": a.full, "twin": a.twin, "erased": dict(erased),
         "rows": [{"label": r["label"], **r["res"]} for r in rows]}, indent=1))
    print("\n".join(md))
    print("\nwrote", out / "table.md")


if __name__ == "__main__":
    main()
