#!/usr/bin/env python
"""Take the nll_kl grid from wherever it is to finished tables, idempotently.

Every step checks what is already on disk and does only what is missing, so
this can be killed at any point -- preemption on killable, a timeout -- and
simply re-run. That is the whole design goal: the 2026-09-20 sweep lost three
scoring jobs to a 4 h wall and needed a human to work out what to resubmit.

    1. discover every candidate under <run>/candidates
    2. score any that lack selection-phase records (skipping holed checkpoints)
    3. run the selection rule per (topic, method) -> results/<topic>/selection
    4. score each winner on the TEST sets with distributions, which selection
       never needed and so never produced
    5. build the per-topic table with report.py

Step 4 writes via a temp dir and only replaces the real one once the records
look complete, so an interrupted run cannot leave a half-written records.json
where a good one used to be.

    python finalize.py --run <run root> --root <worktree> [--no-gpu] [--dry-run]

--no-gpu does steps 1, 3 and 5 only: everything computable from what is
already scored. Use it on the login node.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
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

PY = sys.executable
HERE = Path(__file__).resolve().parent


def log(*a):
    print(*a, flush=True)


def discover(cand_root: Path) -> list[tuple[str, Path, str, str]]:
    """-> (name, model dir, topic, method), sorted."""
    found = []
    for d in sorted(cand_root.iterdir()):
        if not d.is_dir() or d.name in ("logs",) or d.name.startswith("_"):
            continue
        if (d / "model" / "config.json").exists():          # rmu_*, snmf_*
            found.append((d.name, d / "model"))
        else:                                                # ember_<topic>/d<delta>
            for sub in sorted(d.iterdir()):
                if (sub / "model" / "config.json").exists():
                    found.append((f"{d.name}_{sub.name}", sub / "model"))
    out = []
    for name, model in found:
        parts = name.split("_")
        method, topic = parts[0], parts[1]
        # The dir carries provenance the scored/ tree does not: the building
        # job's id (rmu_ai_L6hi_a10_913344) or a _rebuild marker. The label is
        # the logical cell, which is what scored/ and the tables are keyed on.
        label = re.sub(r"_(?:\d{6,}|rebuild)$", "", name)
        if method in METHODS and topic in TOPICS:
            out.append((label, model, topic, method))
    return sorted(out)


def run(cmd: list[str], **kw) -> bool:
    log("  $", " ".join(str(c) for c in cmd[:6]), "...")
    return subprocess.run([str(c) for c in cmd], **kw).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    R, W = Path(a.run), Path(a.root)
    scored, cands, results = R / "scored", R / "candidates", R / "results"
    sets = W / "ember_eval" / "nll_kl" / "sets"
    check = HERE / "check_alloc.py"

    inv = discover(cands)
    log(f"== inventory: {len(inv)} candidates")
    for t in TOPICS:
        for m in METHODS:
            n = sum(1 for _, _, tt, mm in inv if tt == t and mm == m)
            log(f"   {t:9s} {m:6s} {n:3d}")

    # --- 1/2. score anything missing on the selection phase -----------------
    missing = [(n, d, t) for n, d, t, _ in inv
               if not (scored / n / t / "records.json").exists()]
    log(f"\n== unscored: {len(missing)}")
    for n, d, t in missing:
        log(f"   {n} ({t})")
    if missing and not a.no_gpu and not a.dry_run:
        for n, d, t in missing:
            if not run([PY, check, d]):
                log(f"  SKIP {n}: holed checkpoint")
                continue
            ok = run([PY, "-u", HERE / "score_model.py", "--model", d,
                      "--sets", sets / f"{t}.json", "--out", scored / n,
                      "--no-dists", "--only-phase", "selection"])
            log(f"  {'scored' if ok else 'FAILED'} {n}")

    # --- 3. selection per (topic, method) -----------------------------------
    log("\n== selection")
    winners: dict[tuple[str, str], str] = {}
    for t in TOPICS:
        for m in METHODS:
            names = [n for n, _, tt, mm in inv if tt == t and mm == m
                     and (scored / n / t / "records.json").exists()]
            total = sum(1 for _, _, tt, mm in inv if tt == t and mm == m)
            if not names:
                log(f"   {t}/{m}: nothing scored, skipped")
                continue
            out = results / t / "selection" / f"{m}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not a.dry_run:
                cmd = [PY, HERE / "select_checkpoint.py",
                       "--control", scored / CONTROL / t, "--candidates"]
                cmd += [scored / n / t for n in names]
                cmd += ["--out", out]
                if not run(cmd, stdout=subprocess.DEVNULL):
                    log(f"   {t}/{m}: FAILED")
                    continue
            if out.exists():
                rank = json.loads(out.read_text())["ranking"]
                winners[(t, m)] = rank[0]["model"]
                flag = "" if len(names) == total else f"  [PARTIAL {len(names)}/{total}]"
                log(f"   {t:9s} {m:6s} {len(names):2d} cands -> {rank[0]['model']:26s} "
                    f"S={rank[0]['S']:+.3f}{flag}")

    # --- 4. test-phase scoring of the winners -------------------------------
    log("\n== winners needing test-phase scores")
    todo = []
    for (t, m), name in sorted(winners.items()):
        rec = scored / name / t / "records.json"
        phases = set()
        if rec.exists():
            phases = {r["phase"] for r in json.loads(rec.read_text())["records"]}
        if "test" not in phases or not (scored / name / t / "dists.npy").exists():
            todo.append((name, t))
            log(f"   {name} ({t})")
    if not todo:
        log("   none")
    if todo and not a.no_gpu and not a.dry_run:
        by_name = {n: d for n, d, _, _ in inv}
        for name, t in todo:
            d = by_name.get(name)
            if d is None:
                log(f"  SKIP {name}: no model dir")
                continue
            tmp = scored / f"{name}__tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            if not run([PY, "-u", HERE / "score_model.py", "--model", d,
                        "--sets", sets / f"{t}.json", "--out", tmp]):
                log(f"  FAILED {name}")
                continue
            rec = tmp / t / "records.json"
            try:
                n_rec = len(json.loads(rec.read_text())["records"])
            except Exception as e:
                log(f"  FAILED {name}: unreadable records ({e})")
                continue
            if n_rec < 300 or not (tmp / t / "dists.npy").exists():
                log(f"  FAILED {name}: incomplete ({n_rec} records)")
                continue
            dest = scored / name / t
            bak = scored / name / f"{t}.selection-only"
            if dest.exists() and not bak.exists():
                dest.rename(bak)
            elif dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            (tmp / t).rename(dest)
            shutil.rmtree(tmp, ignore_errors=True)
            log(f"  scored {name} on test ({n_rec} records)")

    # --- 5. tables ----------------------------------------------------------
    log("\n== tables")
    for t in TOPICS:
        erased = []
        for m in METHODS:
            name = winners.get((t, m))
            if name and (scored / name / t / "dists.npy").exists():
                erased.append(f"{m.upper()}={name}")
        rel = RELEASED[t]
        if (scored / rel / t / "dists.npy").exists():
            erased.append(f"EMBER-released={rel}")
        twin = TWIN[t]
        if not (scored / twin / t / "dists.npy").exists():
            log(f"   {t}: twin {twin} not scored, skipped")
            continue
        if not erased:
            log(f"   {t}: no erased model ready, skipped")
            continue
        if a.dry_run:
            log(f"   {t}: would report {erased}")
            continue
        cmd = [PY, HERE / "report.py", "--topic", t, "--scored", scored,
               "--full", CONTROL, "--twin", twin, "--erased", *erased,
               "--out", results / t]
        log(f"   {t}: {' '.join(erased)}")
        if not run(cmd, stdout=subprocess.DEVNULL):
            log(f"   {t}: report FAILED")

    log("\n== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
