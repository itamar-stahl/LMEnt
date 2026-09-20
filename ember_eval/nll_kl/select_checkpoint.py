#!/usr/bin/env python
"""Pick one checkpoint per (topic, method) from a grid, by the peers' rule.

Selection uses answer NLL only, on the SELECTION sets only. Test-phase items
are ignored even if a candidate's records contain them; the twins are never
candidates and never references here. The rule is fixed in this file and is
not to be edited once any test number has been looked at.

    delta_T(M)     = mean over target_selection    of  nll_M(q) - nll_ctrl(q)   (signed)
    D_neighbour(M) = mean over neighbour_selection of |nll_M(q) - nll_ctrl(q)|   (absolute)
    D_unrelated(M) = mean over unrelated_selection of |nll_M(q) - nll_ctrl(q)|   (absolute)

    S(M) = 0.5 * delta_T(M) - 0.25 * D_neighbour(M) - 0.25 * D_unrelated(M)

The candidate with the largest S wins. Signed on the target (we want the
erased model LESS confident of the right answer), absolute on the other two
(any movement is collateral, and a signed mean would let it cancel).

    python ember_eval/nll_kl/select_checkpoint.py --control <scored>/<ctrl>/rome \
        --candidates <scored>/<cand_a>/rome <scored>/<cand_b>/rome ... --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

W_TARGET, W_NEIGHBOUR, W_UNRELATED = 0.5, 0.25, 0.25
SETS = ("target_selection", "neighbour_selection", "unrelated_selection")


def load_nll(d: Path) -> Dict[str, Any]:
    rec = json.loads((d / "records.json").read_text())
    by_set: Dict[str, Dict[str, float]] = {}
    for r in rec["records"]:
        if r["phase"] != "selection":
            continue
        by_set.setdefault(r["set"], {})[r["id"]] = r["nll"]
    return {"meta": rec["meta"], "by_set": by_set}


def score_candidate(cand: Dict[str, Any], ctrl: Dict[str, Any]) -> Dict[str, Any]:
    comps: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    for s in SETS:
        c, k = cand["by_set"].get(s, {}), ctrl["by_set"].get(s, {})
        ids = sorted(set(c) & set(k))
        if len(ids) < 50:
            raise SystemExit(f"{cand['meta']['model_name']}: {s} has {len(ids)} common items, need 50")
        d = np.array([c[i] - k[i] for i in ids])
        comps[s] = float(d.mean()) if s == "target_selection" else float(np.abs(d).mean())
        ns[s] = len(ids)
    S = (W_TARGET * comps["target_selection"]
         - W_NEIGHBOUR * comps["neighbour_selection"]
         - W_UNRELATED * comps["unrelated_selection"])
    return {"model": cand["meta"]["model_name"], "path": cand["meta"]["model"],
            "delta_T": comps["target_selection"],
            "D_neighbour": comps["neighbour_selection"],
            "D_unrelated": comps["unrelated_selection"],
            "S": S, "n": ns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="scored dir of the control (Full) model, one topic")
    ap.add_argument("--candidates", required=True, nargs="+", help="scored dirs, same topic")
    ap.add_argument("--out", help="write ranking JSON")
    ap.add_argument("--label", default="", help="free text recorded in the output (topic/method)")
    a = ap.parse_args()

    ctrl = load_nll(Path(a.control))
    rows = []
    for c in a.candidates:
        cand = load_nll(Path(c))
        if cand["meta"]["sets_file"] != ctrl["meta"]["sets_file"]:
            raise SystemExit(f"{c}: scored on a different sets file than the control")
        rows.append(score_candidate(cand, ctrl))
    rows.sort(key=lambda r: -r["S"])
    print(f"selection {a.label}  (control = {ctrl['meta']['model_name']})")
    print(f"  {'candidate':44s} {'S':>8s} {'delta_T':>8s} {'D_nb':>7s} {'D_unr':>7s}")
    for r in rows:
        print(f"  {r['model']:44s} {r['S']:+8.4f} {r['delta_T']:+8.4f} {r['D_neighbour']:7.4f} {r['D_unrelated']:7.4f}")
    print(f"  -> winner: {rows[0]['model']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "label": a.label, "control": ctrl["meta"]["model_name"],
            "rule": "S = 0.5*delta_T - 0.25*D_neighbour - 0.25*D_unrelated, selection sets only",
            "ranking": rows, "winner": rows[0]}, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
