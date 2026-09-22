#!/usr/bin/env python
"""Check that a finished nll_kl run is actually complete and self-consistent.

Written after the 2026-09-20 sweep, where three separate things were wrong at
once and none of them raised an error: three checkpoints were saved with holes
and scored anyway, three scoring jobs hit a 4 h wall leaving six candidates
unscored, and the selected winners had never been scored on the test phase the
tables are built from. Every job involved reported COMPLETED.

So this asserts the properties a finished run has, rather than trusting job
exit codes:

    1. every candidate on disk has selection-phase records for its topic
    2. every reference model has all three topics, 300 records, distributions
    3. all 9 selections exist, rank every candidate, and their winners are
       test-scored with distributions (report.py cannot build a row otherwise)
    4. every table reports exactly the winners its selection files name
    5. no live candidate checkpoint has unallocated holes
    6. quarantined originals are documented, and the live scores that share
       their names are genuinely different numbers -- the rebuild, not the
       corrupt data still sitting where the results read from

Exit 0 if everything holds, 1 with the failures named otherwise.

    python audit.py --run <run root>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOPICS = ("rome", "baseball", "ai")
METHODS = ("ember", "rmu", "snmf")
CONTROL = "lment-1b-control-2e-b131k"
TWIN = {"rome": "lment-1b-norome-2e-b131k",
        "baseball": "lment-1b-nobaseball-2e-b131k",
        "ai": "lment-1b-noai-2e-b131k"}
RELEASED = {"rome": "lment-1b-rome-erased-b131k",
            "baseball": "lment-1b-baseball-erased-b131k",
            "ai": "lment-1b-ai-erased-b131k"}
FULL_RECORDS = 300      # selection + test, 50 items x 3 roles x 2 phases
MIN_ALLOC = 0.98


def records(p: Path) -> list:
    return json.loads((p / "records.json").read_text())["records"]


def target_mean(p: Path, topic: str) -> float | None:
    v = [x["nll"] for x in records(p / topic) if x["set"] == "target_selection"]
    return sum(v) / len(v) if v else None


def inventory(cand_root: Path) -> list[tuple[str, Path, str, str]]:
    """-> (logical label, model dir, topic, method) for every candidate."""
    raw = []
    for d in sorted(cand_root.iterdir()):
        if not d.is_dir() or d.name == "logs" or d.name.startswith("_"):
            continue
        if (d / "model" / "config.json").exists():
            raw.append((d.name, d / "model"))
        else:                                    # ember_<topic>/d<delta>/model
            for s in sorted(d.iterdir()):
                if (s / "model" / "config.json").exists():
                    raw.append((f"{d.name}_{s.name}", s / "model"))
    out = []
    for name, model in raw:
        # dirs carry provenance the scored/ tree does not: the building job's
        # id, or a _rebuild marker. Both strip to the logical cell.
        label = re.sub(r"_(?:\d{6,}|rebuild)$", "", name)
        parts = label.split("_")
        if len(parts) >= 2 and parts[0] in METHODS and parts[1] in TOPICS:
            out.append((label, model, parts[1], parts[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = Path(a.run)
    scored, results = R / "scored", R / "results"
    ok: list[str] = []
    fail: list[str] = []

    # 1. candidates -------------------------------------------------------
    inv = inventory(R / "candidates")
    ok.append(f"candidates discovered: {len(inv)}")
    missing = [(n, t) for n, _, t, _ in inv if not (scored / n / t / "records.json").exists()]
    (fail if missing else ok).append(f"candidates unscored: {len(missing)} {missing[:5]}")

    # 2. references -------------------------------------------------------
    refs = [CONTROL, *TWIN.values(), *RELEASED.values()]
    for ref in refs:
        for t in TOPICS:
            p = scored / ref / t
            if not (p / "records.json").exists():
                fail.append(f"reference {ref}/{t}: MISSING")
                continue
            n = len(records(p))
            if n != FULL_RECORDS:
                fail.append(f"reference {ref}/{t}: {n} records, expected {FULL_RECORDS}")
            if not (p / "dists.npy").exists():
                fail.append(f"reference {ref}/{t}: no dists.npy")
    ok.append(f"references checked: {len(refs)} x {len(TOPICS)} topics")

    # 3. selections and their winners -------------------------------------
    winners: dict[tuple[str, str], str] = {}
    for t in TOPICS:
        for m in METHODS:
            f = results / t / "selection" / f"{m}.json"
            if not f.exists():
                fail.append(f"selection {t}/{m}: MISSING")
                continue
            rank = json.loads(f.read_text())["ranking"]
            total = sum(1 for _, _, tt, mm in inv if tt == t and mm == m)
            if len(rank) != total:
                fail.append(f"selection {t}/{m}: ranks {len(rank)} of {total} candidates")
            w = rank[0]["model"]
            winners[(t, m)] = w
            wp = scored / w / t
            if not (wp / "records.json").exists():
                fail.append(f"winner {w} ({t}): not scored")
                continue
            if len(records(wp)) != FULL_RECORDS:
                fail.append(f"winner {w} ({t}): {len(records(wp))} records, "
                            f"needs {FULL_RECORDS} for the table")
            if not (wp / "dists.npy").exists():
                fail.append(f"winner {w} ({t}): no dists.npy, KL column would be empty")
    ok.append(f"selections: {len(winners)}/{len(TOPICS) * len(METHODS)} present, "
              "every winner test-scored with dists")

    # 4. tables agree with the selections ---------------------------------
    for t in TOPICS:
        tbl, res = results / t / "table.md", results / t / "results.json"
        if not tbl.exists() or not res.exists():
            fail.append(f"table {t}: MISSING")
            continue
        erased = json.loads(res.read_text())["erased"]
        for m in METHODS:
            if (t, m) in winners and erased.get(m.upper()) != winners[(t, m)]:
                fail.append(f"table {t}: {m.upper()}={erased.get(m.upper())} "
                            f"but selection names {winners[(t, m)]}")
    ok.append(f"tables: {len(TOPICS)}/{len(TOPICS)}, each reporting its selection winners")

    # 5. holes ------------------------------------------------------------
    holed = []
    for _, model, _, _ in inv:
        for f in model.glob("*.safetensors"):
            st = f.stat()
            if st.st_size > 10_000_000 and st.st_blocks * 512 / st.st_size < MIN_ALLOC:
                holed.append(str(f))
    (fail if holed else ok).append(f"holed checkpoints among live candidates: {len(holed)}")

    # 6. quarantine -------------------------------------------------------
    for q in sorted(R.glob("_corrupt_*")):
        (ok if (q / "README.md").exists() else fail).append(f"{q.name}/README.md present")
        if not (q / "scored").exists():
            continue
        stale, checked = [], 0
        for d in (q / "scored").iterdir():
            for t in TOPICS:
                live = scored / d.name
                if not (d / t / "records.json").exists():
                    continue
                if not (live / t / "records.json").exists():
                    continue
                checked += 1
                x, y = target_mean(live, t), target_mean(d, t)
                # A quarantined dir shares its logical name with the rebuild
                # that replaced it, so a name match proves nothing -- only
                # identical numbers would mean the corrupt scoring is still
                # what the results read from.
                if x is not None and y is not None and abs(x - y) < 1e-6:
                    stale.append(f"{d.name}/{t}: live still matches quarantined ({x:.3f})")
        (fail if stale else ok).append(
            f"{q.name}: {checked} live/quarantined pairs compared, {len(stale)} stale")

    print("== PASS")
    for x in ok:
        print("  ", x)
    print("== FAIL" if fail else "== FAIL  (none)")
    for x in fail:
        print("  ", x)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
