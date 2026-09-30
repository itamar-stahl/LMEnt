#!/usr/bin/env python
"""Panel A for every cell, standalone and ensemble, on the held-out test split.

Accuracy per group and the efficacy-preservation score, computed exactly as the
selection rule computes it but on the TEST half and against the test-half full
model:

    Acc~_g  = clip01( (Acc_g(M) - 0.25) / (Acc_g(F) - 0.25) )
    H_test  = HM( 1 - Acc~_target , HM( Acc~_neighbour , Acc~_unrelated ) )

Panel B (NLL and KL) comes from report.py and the question-level distances from
question_level_similarity.py; this file does not duplicate either.

    python ember_eval/ensembles/finalize_ensembles.py --scored-test <dir> --out <dir>
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Ember-on-LMEnt"))
from ember.evals.harmonic import harmonic_mean  # noqa: E402

TOPICS = {"rome": "Ancient Rome", "baseball": "Baseball", "ai": "Artificial intelligence"}
GROUPS = ("target_test", "neighbour_test", "unrelated_test")
CHANCE = 0.25
FULL = "FULL_lment-1b-control-2e-b131k"
TWIN = {"rome": "TWIN_lment-1b-norome-2e-b131k",
        "baseball": "TWIN_lment-1b-nobaseball-2e-b131k",
        "ai": "TWIN_lment-1b-noai-2e-b131k"}


def acc(scored: Path, label: str, topic: str) -> dict:
    f = scored / label / topic / "per_question.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    out = {}
    for g in GROUPS:
        sub = [r for r in rows if r["question_group"] == g]
        if sub:
            out[g] = sum(int(r["is_correct"]) for r in sub) / len(sub)
    return out if len(out) == len(GROUPS) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-test", required=True, help="tree with every label's test scoring")
    ap.add_argument("--cells", required=True, help="JSON {topic: {method: label}}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scored, out = Path(a.scored_test), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = json.loads(Path(a.cells).read_text())

    rows, missing = [], []
    for slug, name in TOPICS.items():
        full = acc(scored, FULL, slug)
        if not full:
            raise SystemExit(f"{slug}: full model not scored on the test split")
        for g in GROUPS:
            if full[g] <= CHANCE:
                raise SystemExit(f"{slug}/{g}: full accuracy {full[g]:.3f} <= chance; "
                                 "normalisation undefined")
        for method, label in [("Twin", TWIN[slug])] + list(cells.get(slug, {}).items()):
            ga = acc(scored, label, slug)
            if not ga:
                missing.append(f"{slug}/{method} ({label})")
                continue
            comp = {g: min(1.0, max(0.0, (ga[g] - CHANCE) / (full[g] - CHANCE))) for g in GROUPS}
            eff = 1.0 - comp["target_test"]
            pres = harmonic_mean([comp["neighbour_test"], comp["unrelated_test"]])
            rows.append({
                "topic": name, "method": method, "checkpoint": label,
                **{f"acc_{g.split('_')[0]}": f"{ga[g]:.6f}" for g in GROUPS},
                **{f"norm_{g.split('_')[0]}": f"{comp[g]:.6f}" for g in GROUPS},
                "phi_efficacy": f"{eff:.10f}", "phi_preservation": f"{pres:.10f}",
                "H_test": f"{harmonic_mean([eff, pres]):.10f}"})

    cols = list(rows[0])
    with open(out / "panel_A_accuracy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"{'topic':24s} {'method':16s} {'target':>7s} {'neigh':>7s} {'sciq':>7s} {'H_test':>8s}")
    for r in rows:
        print(f"{r['topic']:24s} {r['method']:16s} {float(r['acc_target']):7.2f} "
              f"{float(r['acc_neighbour']):7.2f} {float(r['acc_unrelated']):7.2f} "
              f"{float(r['H_test']):8.4f}")
    if missing:
        print("\nNOT SCORED (reported, not silently dropped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    (out / "panel_A_missing.json").write_text(json.dumps(missing, indent=1) + "\n")
    print(f"\nwrote {out/'panel_A_accuracy.csv'} ({len(rows)} rows, {len(missing)} missing)")


if __name__ == "__main__":
    main()
