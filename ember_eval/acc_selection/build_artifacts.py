#!/usr/bin/env python
"""Assemble the selection run's output artifacts from the scored tree.

Produces run_manifest.json, candidate_manifest.csv, the aggregated per_option.csv
and per_question.csv, and VALIDATION.md. group_accuracies.csv, normalized_scores.csv,
rankings.csv and selected_checkpoints.json come from select_accuracy.py.

Checkpoint identity: hashing 64 x 5.1 GB is hours of I/O for no benefit here, so each
checkpoint gets a FINGERPRINT rather than a content hash -- sha256 over the
safetensors index plus every weight file's (name, size, st_blocks). st_blocks is
included deliberately: it is what catches a file written with unallocated holes,
which this project has already been bitten by three times. The field is named
checkpoint_fingerprint, not checkpoint_hash, so nobody mistakes it for a content
digest.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

TOPICS = ("rome", "baseball", "ai")
GROUPS = ("target_selection", "neighbour_selection", "unrelated_selection")
EXPECT = {("rome","EMBER"):12, ("rome","RMU"):8, ("rome","SNMF"):4,
          ("baseball","EMBER"):9, ("baseball","RMU"):8, ("baseball","SNMF"):3,
          ("ai","EMBER"):9, ("ai","RMU"):8, ("ai","SNMF"):3}
FULL_LABEL = "FULL_lment-1b-control-2e-b131k"


def fingerprint(model_dir: Path) -> tuple[str, str]:
    h = hashlib.sha256()
    notes = []
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        h.update(idx.read_bytes())
    files = sorted(p for p in model_dir.iterdir() if p.suffix == ".safetensors")
    for p in files:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_blocks}".encode())
        if st.st_blocks * 512 < st.st_size * 0.9:
            notes.append(f"{p.name} allocated {st.st_blocks*512} of {st.st_size} bytes")
    return h.hexdigest(), ("; ".join(notes) if notes else "ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--joblists", required=True, help="dir holding joblist_<topic>.tsv")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scored, out = Path(a.scored), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(Path(a.inventory).read_text())

    # ---- candidate manifest -------------------------------------------------
    cand_rows, missing = [], []
    for topic, meth, label, path in sorted(inv):
        md = Path(path)
        pq = scored / label / topic / "per_question.csv"
        fp, integrity = fingerprint(md) if md.exists() else ("", "model dir absent")
        status = "scored" if pq.exists() else "NOT SCORED"
        if not pq.exists():
            missing.append(f"{topic}/{meth}/{label}")
        p = label.split("_")
        hp = "_".join(p[2:])
        cand_rows.append({"topic": topic, "method": meth, "model_label": label,
                          "hyperparameters": hp, "checkpoint_path": str(md),
                          "checkpoint_fingerprint": fp, "integrity": integrity,
                          "load_status": status})
    with (out / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand_rows[0].keys())); w.writeheader()
        for r in cand_rows: w.writerow(r)

    # ---- aggregate per-option and per-question ------------------------------
    def merge(name: str) -> list[dict]:
        rows = []
        for d in sorted(scored.iterdir()):
            for t in TOPICS:
                f = d / t / name
                if f.exists():
                    rows.extend(csv.DictReader(open(f, encoding="utf-8")))
        return rows
    per_opt, per_q = merge("per_option.csv"), merge("per_question.csv")
    for name, rows in (("per_option.csv", per_opt), ("per_question.csv", per_q)):
        if not rows: continue
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)

    # ---- run manifest -------------------------------------------------------
    def rev(p):
        try: return subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
        except Exception: return "unknown"
    metas = []
    for d in sorted(scored.iterdir()):
        for t in TOPICS:
            m = d / t / "meta.json"
            if m.exists(): metas.append(json.loads(m.read_text()))
    observed = {}
    for topic, meth, label, _ in inv:
        if (scored / label / topic / "per_question.csv").exists():
            observed[f"{topic}/{meth}"] = observed.get(f"{topic}/{meth}", 0) + 1
    man_bytes = Path(a.manifest).read_bytes()
    joblists = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(a.joblists).glob("joblist_*.tsv"))}
    (out / "run_manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": rev(a.root),
        "scorer_version": (metas[0]["scorer_version"] if metas else None),
        "dtype": sorted({m["dtype"] for m in metas}),
        "devices": sorted({str(m["device"]) for m in metas}),
        "gpus_used": sorted({str(m.get("gpu")) for m in metas}),
        "question_manifest": {"path": str(a.manifest),
                              "sha256": hashlib.sha256(man_bytes).hexdigest(),
                              "n_rows": sum(1 for _ in open(a.manifest, encoding="utf-8")) - 1},
        "joblist_sha256": joblists,
        "seeds": {"scoring": "deterministic, no sampling: teacher-forced log-probs only",
                  "set_construction": 20260920},
        "expected_candidates": {f"{k[0]}/{k[1]}": v for k, v in sorted(EXPECT.items())},
        "observed_candidates": dict(sorted(observed.items())),
        "expected_total": sum(EXPECT.values()), "observed_total": sum(observed.values()),
        "full_model_scorings": sum(1 for t in TOPICS
                                   if (scored / FULL_LABEL / t / "per_question.csv").exists()),
        "commands": [
            "python ember_eval/acc_selection/build_manifest.py --out questions_with_options.csv",
            "sbatch ... ember_eval/acc_selection/slurm/score_selection.slurm  (one job per topic)",
            "python ember_eval/acc_selection/select_accuracy.py --scored <dir> --inventory <json> "
            "--full-label " + FULL_LABEL + " --out <dir>",
            "python ember_eval/acc_selection/build_artifacts.py ...",
        ],
        "not_scored": missing,
    }, indent=1))
    print(f"candidates: {len(cand_rows)} | per_option {len(per_opt)} | per_question {len(per_q)}")
    print(f"not scored: {len(missing)}")
    for m in missing[:20]: print("   ", m)


if __name__ == "__main__":
    main()
