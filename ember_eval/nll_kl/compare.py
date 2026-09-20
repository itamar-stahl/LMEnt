#!/usr/bin/env python
"""The two metrics for one (evaluated, reference) model pair on one topic.

Both models must have been scored by score_model.py on the SAME sets file.

Metric 1 -- answer NLL difference, per item:
    delta(q) = nll_eval(q) - nll_ref(q)            (evaluated - reference, always)
Metric 2 -- full-vocabulary KL, per item:
    kl(q) = mean over completion positions t of  KL( P_ref(.|q,t) || P_eval(.|q,t) )
          = mean_t sum_v p_ref(v) * (log p_ref(v) - log p_eval(v))
    needs both models' saved distributions for the item (test items by default).

Per set, both metrics are summarised four ways from the same per-item numbers:
    mean            the headline, what the peers' table reports
    median          the typical item; far from the mean => a few items carry it
    frac_pos        fraction of items with delta > 0 (NLL only; KL is >= 0);
                    ~0.5 with a mean near 0 is cancellation, not "no effect"
    ci95            bootstrap over items, 1000 resamples, fixed seed

    python ember_eval/nll_kl/compare.py --eval <scored>/<model>/rome --ref <scored>/<ctrl>/rome
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BOOT_N = 1000
BOOT_SEED = 0


def load_scored(d: Path) -> Dict[str, Any]:
    rec = json.loads((d / "records.json").read_text())
    rec["meta"]["label"] = Path(d).resolve().parent.name   # <scored>/<name>/<topic>
    out = {"meta": rec["meta"], "records": {r["id"]: r for r in rec["records"]},
           "dists": None, "index": None}
    if (d / "dists.npy").exists():
        out["dists"] = np.load(d / "dists.npy", mmap_mode="r")
        out["index"] = json.loads((d / "dists_index.json").read_text())
    return out


def summarise(values: List[float], signed: bool) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(BOOT_N)])
    out = {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }
    if signed:
        out["frac_pos"] = float((x > 0).mean())
        out["frac_neg"] = float((x < 0).mean())
    return out


def kl_item(ref_rows: np.ndarray, eval_rows: np.ndarray) -> float:
    """mean over positions of KL(ref || eval), rows are log-probs [k, vocab]."""
    if ref_rows.shape != eval_rows.shape:
        raise ValueError(f"shape mismatch {ref_rows.shape} vs {eval_rows.shape}")
    ref = np.asarray(ref_rows, dtype=np.float64)
    ev = np.asarray(eval_rows, dtype=np.float64)
    p = np.exp(ref)
    kl = (p * (ref - ev)).sum(axis=1)           # [k]
    return float(kl.mean())


def compare(ev: Dict[str, Any], ref: Dict[str, Any], sets: Optional[List[str]] = None
            ) -> Dict[str, Any]:
    ids = sorted(set(ev["records"]) & set(ref["records"]))
    if not ids:
        raise SystemExit("no common items between the two scored dirs")
    if ev["meta"]["sets_file"] != ref["meta"]["sets_file"]:
        print(f"WARNING: sets files differ:\n  {ev['meta']['sets_file']}\n  {ref['meta']['sets_file']}")
    per_set: Dict[str, Dict[str, Any]] = {}
    items: List[Dict[str, Any]] = []
    for i in ids:
        a, b = ev["records"][i], ref["records"][i]
        if a["completion_token_ids"] != b["completion_token_ids"]:
            raise SystemExit(f"item {i}: completion tokens differ between models (tokenizer mismatch?)")
        rec = {"id": i, "set": a["set"], "role": a["role"], "phase": a["phase"],
               "n_tokens": a["n_tokens"],
               "nll_eval": a["nll"], "nll_ref": b["nll"], "delta": a["nll"] - b["nll"],
               "delta_first_token": a["token_nlls"][0] - b["token_nlls"][0]}
        have = (ev["dists"] is not None and ref["dists"] is not None
                and i in ev["index"] and i in ref["index"])
        if have:
            s0, e0 = ref["index"][i]
            s1, e1 = ev["index"][i]
            rec["kl"] = kl_item(ref["dists"][s0:e0], ev["dists"][s1:e1])
            rec["kl_first_token"] = kl_item(ref["dists"][s0:s0 + 1], ev["dists"][s1:s1 + 1])
        items.append(rec)
    for name in sorted({r["set"] for r in items}):
        if sets and name not in sets:
            continue
        rs = [r for r in items if r["set"] == name]
        entry = {"nll_diff": summarise([r["delta"] for r in rs], signed=True),
                 "nll_diff_first_token": summarise([r["delta_first_token"] for r in rs], signed=True)}
        kls = [r["kl"] for r in rs if "kl" in r]
        if kls:
            entry["kl"] = summarise(kls, signed=False)
            entry["kl_first_token"] = summarise([r["kl_first_token"] for r in rs if "kl_first_token" in r],
                                                signed=False)
        per_set[name] = entry
    return {
        "evaluated": ev["meta"]["label"], "reference": ref["meta"]["label"],
        "evaluated_path": ev["meta"]["model"], "reference_path": ref["meta"]["model"],
        "topic": ev["meta"]["topic"], "convention": "evaluated - reference; KL(reference || evaluated)",
        "per_set": per_set, "items": items,
    }


def fmt(stat: Dict[str, Any], signed: bool) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "—"
    s = f"{stat['mean']:+.3f}" if signed else f"{stat['mean']:.4f}"
    s += f" [{stat['ci95'][0]:+.3f}, {stat['ci95'][1]:+.3f}]" if signed else \
         f" [{stat['ci95'][0]:.4f}, {stat['ci95'][1]:.4f}]"
    s += f" med {stat['median']:+.3f}" if signed else f" med {stat['median']:.4f}"
    if signed:
        s += f" up {stat['frac_pos']:.0%}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="scored dir of the evaluated model (one topic)")
    ap.add_argument("--ref", required=True, help="scored dir of the reference model (same topic)")
    ap.add_argument("--out", help="write the full result JSON here")
    ap.add_argument("--sets", nargs="*", help="restrict to these set names")
    a = ap.parse_args()
    res = compare(load_scored(Path(a.eval)), load_scored(Path(a.ref)), a.sets)
    print(f"{res['evaluated']}  vs  {res['reference']}   ({res['topic']})")
    for name, e in res["per_set"].items():
        print(f"  {name:22s} NLL diff {fmt(e['nll_diff'], True)}")
        if "kl" in e:
            print(f"  {'':22s} KL       {fmt(e['kl'], False)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
