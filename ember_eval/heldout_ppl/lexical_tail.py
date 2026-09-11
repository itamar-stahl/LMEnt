#!/usr/bin/env python
"""Is the erasure's shattered tail lexical? Test it per chunk.

ERASURE_RESULTS.md establishes at QUESTION level that EMBER's Rome erasure
damages text containing the tokens it edited -- 3 of 3 damaged cross-concept
questions contain ' distance', one of the 76 edited rows, and 0 damaged
questions lack a swept-in word. n = 3 is not a result.

This runs the same test on 3,000 held-out and 5,004 control chunks, where the
per-chunk loss deltas already exist from the delta sweep. For each chunk it
counts how many of the 76 edited token ids the chunk actually contains, and
relates that to how much the erasure damaged it.

The decisive arm is the CONTROL chunks. They are not about Ancient Rome, so a
semantic account predicts no relationship there at all; a lexical account
predicts the same relationship as in the held-out set. Nothing is loaded onto a
GPU -- this reads token ids from the dataset and joins them to stored losses.
"""
from __future__ import annotations
import argparse, json, glob, re, statistics as st
from pathlib import Path

from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType

DATA_GLOB = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-tokenized/*.npy"
WORK_DIR = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-cache"


def build_dataset(work_dir=WORK_DIR):
    return NumpyDatasetConfig.glob(
        DATA_GLOB, name=NumpyDatasetType.kas_vsl, max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=TokenizerConfig.dolma2(), work_dir=work_dir,
        include_instance_metadata=False,
    ).build()


def losses(path):
    d = json.load(open(path))
    return ({r["chunk_id"]: r["loss"] for r in d["heldout_rows"]},
            {r["chunk_id"]: r["loss"] for r in d["control_rows"]})


def report(label, ids, delta_by_id, count_by_id, tok_by_id, out):
    """Relate edited-token content to per-chunk damage."""
    have = [i for i in ids if i in delta_by_id and i in count_by_id]
    withtok = [i for i in have if count_by_id[i] > 0]
    without = [i for i in have if count_by_id[i] == 0]
    shattered = [i for i in have if delta_by_id[i] > 3.0]
    rest = [i for i in have if delta_by_id[i] <= 3.0]
    m = lambda xs: (sum(delta_by_id[i] for i in xs) / len(xs)) if xs else float("nan")
    rate = lambda xs: (sum(1 for i in xs if count_by_id[i] > 0) / len(xs)) if xs else float("nan")
    dens = lambda xs: (sum(count_by_id[i] / max(tok_by_id[i], 1) for i in xs) / len(xs)) if xs else float("nan")
    block = {
        "n": len(have),
        "n_with_edited_token": len(withtok),
        "n_without": len(without),
        "mean_delta_with_edited_token": m(withtok),
        "mean_delta_without": m(without),
        "n_shattered_gt3": len(shattered),
        "pct_of_shattered_containing_edited_token": rate(shattered) * 100,
        "pct_of_rest_containing_edited_token": rate(rest) * 100,
        "mean_edited_density_shattered": dens(shattered),
        "mean_edited_density_rest": dens(rest),
    }
    print(f"\n===== {label} =====")
    print(f"  chunks {block['n']}   containing >=1 edited token: {block['n_with_edited_token']}"
          f"  ({100*block['n_with_edited_token']/max(block['n'],1):.1f}%)")
    print(f"  mean loss delta  WITH edited token : {block['mean_delta_with_edited_token']:+.4f}")
    print(f"  mean loss delta  WITHOUT           : {block['mean_delta_without']:+.4f}")
    if block["n_shattered_gt3"]:
        print(f"  shattered (>3 nats): {block['n_shattered_gt3']}")
        print(f"     contain an edited token: {block['pct_of_shattered_containing_edited_token']:.1f}%"
              f"   vs rest {block['pct_of_rest_containing_edited_token']:.1f}%")
        print(f"     mean edited-token density: {block['mean_edited_density_shattered']:.5f}"
              f"   vs rest {block['mean_edited_density_rest']:.5f}")
    # dose response by count bucket
    buckets = [(0,0),(1,1),(2,3),(4,7),(8,10**9)]
    rows=[]
    for lo,hi in buckets:
        sel=[i for i in have if lo<=count_by_id[i]<=hi]
        if sel:
            rows.append({"bucket": f"{lo}-{hi if hi<10**9 else '+'}", "n": len(sel),
                         "mean_delta": m(sel)})
            print(f"     tokens {rows[-1]['bucket']:>5s}: n={len(sel):5d}  mean delta {m(sel):+.4f}")
    block["dose_response"] = rows
    out[label] = block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="erasure run report.json (for edited_token_ids)")
    ap.add_argument("--ppl-glob", required=True, help="whitespace-separated globs of erased ppl jsons")
    ap.add_argument("--control-ppl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rep = json.load(open(a.report))
    edited = set(int(x) for x in rep["integrity"]["changed_embedding_rows"])
    print(f"[lex] {len(edited)} edited token ids from {a.report}")

    print("[lex] building dataset", flush=True)
    ds = build_dataset()
    print(f"[lex] dataset: {len(ds)} instances", flush=True)

    ctlH, ctlK = losses(a.control_ppl)
    ids = sorted(set(ctlH) | set(ctlK))
    print(f"[lex] scanning {len(ids)} chunks for edited tokens", flush=True)
    count_by_id, tok_by_id = {}, {}
    for n, i in enumerate(ids):
        toks = ds[i]["input_ids"].tolist()
        count_by_id[i] = sum(1 for t in toks if t in edited)
        tok_by_id[i] = len(toks)
        if (n + 1) % 1000 == 0:
            print(f"    {n+1}/{len(ids)}", flush=True)

    out = {"edited_token_ids": sorted(edited), "source_report": a.report, "cells": {}}
    paths = []
    for pat in a.ppl_glob.split():
        paths.extend(glob.glob(pat))
    if not paths:
        raise SystemExit(f"no ppl files matched: {a.ppl_glob!r}")
    for p in sorted(set(paths)):
        mm = re.search(r"_d(\d+)_", p)
        delta = mm.group(1) if mm else "200"
        H, K = losses(p)
        dH = {i: H[i] - ctlH[i] for i in H if i in ctlH}
        dK = {i: K[i] - ctlK[i] for i in K if i in ctlK}
        report(f"delta {delta} / HELD-OUT (Rome) chunks", list(dH), dH, count_by_id, tok_by_id, out["cells"])
        report(f"delta {delta} / CONTROL (non-Rome) chunks", list(dK), dK, count_by_id, tok_by_id, out["cells"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n[lex] wrote {a.out}")


if __name__ == "__main__":
    main()
