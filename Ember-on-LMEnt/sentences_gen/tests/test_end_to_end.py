"""Full pipeline against live Elasticsearch. Login node only (c-003).

Runs the CLI exactly as you would, then inspects the run folder it produced.
Two runs at a reduced sentence count establish reproducibility; one run at the
real count establishes that a usable corpus is actually reachable for this
blacklist on this index.

    python tests/test_end_to_end.py              # reduced + full
    SENTENCES_GEN_E2E_FULL=0 python tests/test_end_to_end.py    # reduced only

The full run costs one Elasticsearch scan per QID plus ~300 chunk fetches.
Harry Potter is a couple of minutes; a large subject such as World War II
(296k chunks) is considerably longer, which is why the reduced runs come first
and the full run is skippable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from testlib import (
    MODULE_DIR, Suite, assert_eq, assert_in_range, assert_true, es_or_skip,
    index_or_skip, require, sample_blacklist,
)

import blacklist_to_concept_sentences as M

suite = Suite("end to end (login node)")

SCRIPT = MODULE_DIR / "blacklist_to_concept_sentences.py"
OUTPUTS = MODULE_DIR / "outputs"
REDUCED_N = 40

_runs: Dict[str, Path] = {}          # cache so the full run happens once


def _run_cli(n: int, seed: int, extra: Optional[List[str]] = None) -> Path:
    """Invoke the CLI and return the run folder it created."""
    before = set(OUTPUTS.glob("*")) if OUTPUTS.exists() else set()
    cmd = [sys.executable, str(SCRIPT), str(sample_blacklist()),
           "-n", str(n), "--seed", str(seed)] + (extra or [])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MODULE_DIR))
    after = set(OUTPUTS.glob("*")) if OUTPUTS.exists() else set()
    created = sorted(after - before)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise AssertionError(
            f"CLI exited {proc.returncode} for -n {n}:\n            "
            + "\n            ".join(detail))
    assert_true(len(created) == 1,
                f"expected exactly one new run folder, got {created}")
    return created[0]


def _cached_run(key: str, n: int, seed: int) -> Path:
    if key not in _runs:
        index_or_skip(es_or_skip(), M.CASE_SENSITIVE_INDEX)
        _runs[key] = _run_cli(n, seed)
    return _runs[key]


def _load(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    assert_true(path.exists(), f"{name} missing from {run_dir.name}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #

@suite.case("reduced run completes and writes every artefact")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    expected = {"chunk_ids.json", "harvested_sentences.json",
                "concept_sentences.json", sample_blacklist().name}
    present = {p.name for p in run_dir.iterdir()}
    missing = expected - present
    assert_true(not missing, f"missing from run folder: {sorted(missing)}")
    print(f"          {run_dir.name}: {sorted(present)}")


@suite.case("the input blacklist is copied in verbatim")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    original = sample_blacklist().read_bytes()
    copied = (run_dir / sample_blacklist().name).read_bytes()
    assert_eq(copied, original, "blacklist copy differs from the input")


@suite.case("intermediate artefact records inputs, pools and rejections")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    artifact = _load(run_dir, "chunk_ids.json")
    for key in ("generated", "blacklist", "index", "case_sensitive",
                "thresholds", "num_chunks", "entities", "concept", "seed",
                "selection", "pools_after_rarest_assignment", "harvest",
                "length_report"):
        assert_true(key in artifact, f"artifact key {key!r} missing")
    assert_eq(artifact["thresholds"], M.DEFAULT_THRESHOLDS, "thresholds recorded")
    assert_eq(artifact["index"], M.CASE_SENSITIVE_INDEX, "index recorded")
    assert_true(artifact["num_chunks"] > 0, "no chunks resolved")
    for entity in artifact["entities"]:
        assert_eq(len(entity["chunk_ids"]), entity["num_chunks"],
                  f"{entity['qid']} count/list disagree")
        assert_eq(entity["chunk_ids"], sorted(set(entity["chunk_ids"])),
                  f"{entity['qid']} chunk_ids not sorted-unique")
    print(f"          pools: {artifact['pools_after_rarest_assignment']}")
    print(f"          rejections: {artifact['harvest']['rejections']}")


@suite.case("no offset mismatches on real data")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    rejections = _load(run_dir, "chunk_ids.json")["harvest"]["rejections"]
    assert_eq(rejections.get("offset_mismatch", 0), 0,
              "entity offsets do not index into the stored text -- see "
              "test_live_es.py's offset invariant")


@suite.case("output matches the concept_sentences.json schema")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    records = _load(run_dir, "concept_sentences.json")
    assert_true(isinstance(records, list) and len(records) == 1,
                "expected a one-record list")
    record = records[0]
    assert_eq(sorted(record), ["concept", "sentences"], "record keys")
    assert_eq(record["concept"], M.concept_name_from_path(sample_blacklist()),
              "concept name")
    assert_eq(len(record["sentences"]), REDUCED_N, "sentence count")
    assert_true(all(isinstance(s, str) and s.strip() for s in record["sentences"]),
                "every sentence is a non-empty string")


@suite.case("output is loadable by the EMBER dataset reader")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    sys.path.insert(0, str(MODULE_DIR.parent))
    try:
        from ember.local_datasets import ConceptDataset
    except ImportError as exc:
        require(False, f"ember package not importable: {exc}")
    concept = M.concept_name_from_path(sample_blacklist())
    dataset = ConceptDataset(
        concept,
        concept_path=run_dir / "concept_sentences.json",
        neutral_path=MODULE_DIR.parent / "data" / "neutral_sentences.json",
    )
    labels = {label for _, label in dataset}
    assert_true(concept in labels, f"{concept!r} absent from dataset labels")
    assert_true("Neutral" in labels, "neutral sentences absent")
    print(f"          ConceptDataset: {len(dataset)} rows, labels={sorted(labels)}")


@suite.case("sentences are on-topic and traceable to their chunk")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    harvested = _load(run_dir, "harvested_sentences.json")
    final = _load(run_dir, "concept_sentences.json")[0]["sentences"]
    assert_eq([h["sentence"] for h in harvested], final,
              "provenance list and final list disagree")
    for entry in harvested:
        for key in ("chunk_id", "qid", "title", "sentence"):
            assert_true(key in entry, f"provenance key {key!r} missing")
    chunk_ids = [h["chunk_id"] for h in harvested]
    assert_eq(len(set(chunk_ids)), len(chunk_ids), "a chunk was used twice")


@suite.case("sentences are deduplicated")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    sentences = _load(run_dir, "concept_sentences.json")[0]["sentences"]
    keys = [M.normalize_key(s) for s in sentences]
    assert_eq(len(set(keys)), len(keys), "duplicate sentences in the output")


@suite.case("every sentence respects the per-sentence length window")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    for sentence in _load(run_dir, "concept_sentences.json")[0]["sentences"]:
        words = len(sentence.split())
        assert_in_range(words, M.SENT_MIN_WORDS, M.SENT_MAX_WORDS,
                        f"{sentence[:48]!r}")


@suite.case("REPRODUCIBILITY: same seed gives byte-identical sentences")
def _():
    first = _cached_run("reduced_a", REDUCED_N, 42)
    second = _run_cli(REDUCED_N, 42)
    try:
        a = _load(first, "concept_sentences.json")
        b = _load(second, "concept_sentences.json")
        assert_eq(b, a, "same seed produced different output")
        ids_a = [h["chunk_id"] for h in _load(first, "harvested_sentences.json")]
        ids_b = [h["chunk_id"] for h in _load(second, "harvested_sentences.json")]
        assert_eq(ids_b, ids_a, "same seed drew different chunks")
        print(f"          {len(ids_a)} chunks drawn identically")
    finally:
        shutil.rmtree(second, ignore_errors=True)


@suite.case("a different seed draws different chunks")
def _():
    first = _cached_run("reduced_a", REDUCED_N, 42)
    other = _run_cli(REDUCED_N, 7)
    try:
        ids_a = [h["chunk_id"] for h in _load(first, "harvested_sentences.json")]
        ids_b = [h["chunk_id"] for h in _load(other, "harvested_sentences.json")]
        assert_true(ids_a != ids_b, "seed 7 drew the same chunks as seed 42")
    finally:
        shutil.rmtree(other, ignore_errors=True)


@suite.case("draws stay balanced across the blacklist's QIDs")
def _():
    run_dir = _cached_run("reduced_a", REDUCED_N, 42)
    per_qid = _load(run_dir, "chunk_ids.json")["harvest"]["per_qid"]
    contributing = {q: n for q, n in per_qid.items() if n}
    print(f"          per QID: {per_qid}")
    if len(per_qid) > 1:
        pools = _load(run_dir, "chunk_ids.json")["pools_after_rarest_assignment"]
        viable = [q for q, size in pools.items() if size >= REDUCED_N // len(per_qid)]
        for qid in viable:
            assert_true(contributing.get(qid, 0) > 0,
                        f"{qid} has {pools[qid]} chunks but contributed nothing")


@suite.case("FULL RUN: 300 sentences pass the distribution gate")
def _():
    require(os.environ.get("SENTENCES_GEN_E2E_FULL", "1") != "0",
            "SENTENCES_GEN_E2E_FULL=0 -- skipping the full-size run")
    es_or_skip()
    run_dir = _run_cli(M.TARGET_SENTENCES, 42)
    _runs["full"] = run_dir

    record = _load(run_dir, "concept_sentences.json")[0]
    assert_eq(len(record["sentences"]), M.TARGET_SENTENCES, "sentence count")

    report = _load(run_dir, "chunk_ids.json")["length_report"]
    problems = M.validate_distribution(report, M.TARGET_SENTENCES)
    assert_true(not problems, f"distribution gate failed: {problems}")
    print(f"          n={report['n']} median={report['median']} "
          f"sd={report['stdev']:.1f} distinct={report['distinct_lengths']}")
    print(f"          deciles   {report['deciles']}")
    print(f"          reference {M.REFERENCE_DECILES}  (Harry Potter, shipped)")
    print(f"          kept: {run_dir}")


@suite.case("FAILURE PATH: an unreachable server writes no run folder")
def _():
    before = set(OUTPUTS.glob("*")) if OUTPUTS.exists() else set()
    env = dict(os.environ, ES_HOST="127.0.0.1", ES_PORT="9", ES_PASSWORD="x")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(sample_blacklist()), "-n", "5"],
        capture_output=True, text=True, cwd=str(MODULE_DIR), env=env, timeout=600)
    after = set(OUTPUTS.glob("*")) if OUTPUTS.exists() else set()
    assert_eq(after - before, set(), "a failed run left an orphan folder behind")


@suite.case("FAILURE PATH: a missing password is refused before any work")
def _():
    env = {k: v for k, v in os.environ.items() if k != "ES_PASSWORD"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(sample_blacklist()), "-n", "5"],
        capture_output=True, text=True, cwd=str(MODULE_DIR), env=env)
    assert_true(proc.returncode != 0, "missing password should be fatal")
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert_true("no Elasticsearch password" in combined,
                f"unclear error for a missing password:\n{combined[-400:]}")


if __name__ == "__main__":
    sys.exit(suite.run())
