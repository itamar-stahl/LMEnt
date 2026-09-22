#!/usr/bin/env python
"""Assemble one topic's result table from scored models.

Two tables, because the two metrics do not take the same reference.

NLL difference is signed, so its reference fixes which way the sign runs:
    Twin      vs Full      the ceiling: what never-having-learned looks like
    <erased>  vs Twin      how far each erasure is from that
    <erased>  vs Full      what each erasure changed (target) and shouldn't have

KL is asymmetric, so its FIRST argument fixes whose probabilities weight the
deviations, and the protocol makes that the twin for every comparison -- the
question being how closely each model reproduces the twin. So the KL table has
one row per evaluated model, all of them KL(twin || model):
    Full, then each erased model.
Putting the twin second instead would answer a different question and can
change both the values and the ordering: on Rome's target set KL(twin || full)
is 0.955 where KL(full || twin) is 0.761, a 26% difference.

Columns: the three TEST sets. Cells: mean, SD across the 50 questions, 95%
bootstrap CI, median, and (NLL only) the fraction of items that moved up.
Writes <out>/table.md and <out>/results.json, the latter carrying every
per-item value.

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
    return (f"**{stat['mean']:+.3f}** ± {stat['sd']:.3f} · [{lo:+.3f}, {hi:+.3f}]"
            f" · med {stat['median']:+.3f} · up {stat['frac_pos']:.0%}")


def cell_kl(stat: Dict[str, Any]) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "—"
    lo, hi = stat["ci95"]
    return (f"**{stat['mean']:.4f}** ± {stat['sd']:.4f} · [{lo:.4f}, {hi:.4f}]"
            f" · med {stat['median']:.4f}")


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

    loaded = [(method, load(name)) for method, name in erased]

    # NLL: reference sets the sign.  KL: twin is the first argument throughout.
    rows: List[Dict[str, Any]] = []
    rows.append({"label": "Twin vs Full", "res": compare(twin, full, kl_ref=twin)})
    for (method, mod) in loaded:
        rows.append({"label": f"{method} vs Twin", "res": compare(mod, twin, kl_ref=twin)})
    for (method, mod) in loaded:
        rows.append({"label": f"{method} vs Full", "res": compare(mod, full, kl_ref=twin)})

    # One KL row per evaluated model; the twin is the reference for all of them,
    # so the Full row is compare(full, twin) and NOT the Twin-vs-Full row above,
    # whose evaluated model is the twin itself (KL would be identically zero).
    kl_rows: List[Dict[str, Any]] = [{"label": "Full", "res": compare(full, twin, kl_ref=twin)}]
    kl_rows += [{"label": method, "res": compare(mod, twin, kl_ref=twin)} for method, mod in loaded]

    topic_name = full["meta"]["topic"]
    md = [f"# {topic_name} — answer-token NLL difference and KL, test sets\n",
          f"Full = `{a.full}`, Twin = `{a.twin}`. "
          "Cell = mean ± SD · [95% bootstrap CI] · median · fraction of items with Δ>0 "
          "(NLL only). n = 50 per set.\n"]

    md.append("\n## Answer NLL difference (nats/token)\n")
    md.append("Signed, evaluated − reference.\n")
    md.append("| Evaluated vs Reference | " + " | ".join(PRETTY[s] for s in TEST_SETS) + " |")
    md.append("|---|" + "---|" * len(TEST_SETS))
    for r in rows:
        cells = [cell_nll(r["res"]["per_set"].get(s, {}).get("nll_diff", {})) for s in TEST_SETS]
        md.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    md.append("\n## Full-vocabulary KL from the twin (nats)\n")
    md.append("KL(twin ‖ model): the twin is the first argument for every row, so "
              "deviations are weighted by the twin's own distribution.\n")
    md.append("| Model | " + " | ".join(PRETTY[s] for s in TEST_SETS) + " |")
    md.append("|---|" + "---|" * len(TEST_SETS))
    for r in kl_rows:
        cells = [cell_kl(r["res"]["per_set"].get(s, {}).get("kl", {})) for s in TEST_SETS]
        md.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    md.append("\n## Erased models\n")
    for method, name in erased:
        md.append(f"- {method}: `{name}`")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(md) + "\n")
    (out / "results.json").write_text(json.dumps(
        {"topic": topic_name, "full": a.full, "twin": a.twin, "erased": dict(erased),
         "kl_convention": "KL(twin || evaluated) for every kl_row",
         "rows": [{"label": r["label"], **r["res"]} for r in rows],
         "kl_rows": [{"label": r["label"], **r["res"]} for r in kl_rows]}, indent=1))
    print("\n".join(md))
    print("\nwrote", out / "table.md")


if __name__ == "__main__":
    main()
