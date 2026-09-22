#!/usr/bin/env python
"""Score one model on one topic's sets: per-item answer NLL, and the full
next-token distributions at the completion positions for KL.

Teacher-forced, one forward pass per item over `stem + completion`:

    ids  = tok(stem, add_special_tokens=True) + tok(completion, add_special_tokens=False)
    the completion's k tokens sit at positions n-k .. n-1 and are predicted by
    the logits at n-k-1 .. n-2 (the first one from the stem's last token).

    nll(item)      = mean over those k tokens of -log p(token | prefix)   [nats]
    token_nlls     = the k per-token values
    dists          = the k full log-softmax rows (fp32), test-phase items only
                     unless --dists-phase says otherwise; --no-dists skips them

Right padding with a causal mask, so a batch scores exactly like one sequence
at a time -- the same convention as completion_eval/evaluate_completion.py's
`score_pairs`, which every accuracy number in this project came from.

Output layout under --out:
    records.json     model, sets file, per-item records (id, role, phase, set,
                     n_tokens, first_context_token, completion_token_ids, nll,
                     token_nlls)
    dists.npy        float32 [total_positions, vocab] for the items that got
                     distributions, in the order records.json lists them
    dists_index.json {item id: [start, end)} into dists.npy

    python ember_eval/nll_kl/score_model.py --model <hf dir> --sets sets/rome.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def encode(tok, item: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    ctx = tok(item["stem"], add_special_tokens=True)["input_ids"]
    cont = tok(item["completion"], add_special_tokens=False)["input_ids"]
    if not ctx:
        raise SystemExit(f"stem tokenised to nothing: {item['stem']!r}")
    if not cont:
        raise SystemExit(f"completion tokenised to nothing: {item['completion']!r}")
    return ctx, cont


@torch.no_grad()
def score(model, tok, items: List[Dict[str, Any]], device, want_dists, batch_size: int):
    """Returns (records, dists_rows, index). Items are processed longest-first
    within batches for padding efficiency, results returned in input order."""
    encoded = [encode(tok, it) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(encoded[i][0]) + len(encoded[i][1]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    records: List[Dict[str, Any]] = [None] * len(items)  # type: ignore[list-item]
    rows: Dict[int, np.ndarray] = {}
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        seqs = [encoded[i][0] + encoded[i][1] for i in chunk]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, :len(s)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for r, i in enumerate(chunk):
            ctx, cont = encoded[i]
            n, k = len(ctx) + len(cont), len(cont)
            pos = torch.arange(n - k - 1, n - 1, device=logprobs.device)
            idx = torch.tensor(cont, device=logprobs.device)
            tok_lp = logprobs[r, pos, idx]
            token_nlls = (-tok_lp).tolist()
            it = items[i]
            records[i] = {
                "id": it["id"], "role": it["role"], "phase": it["phase"], "set": it["set"],
                "n_tokens": k, "first_context_token": ctx[0],
                "completion_token_ids": cont,
                "nll": float(sum(token_nlls) / k), "token_nlls": token_nlls,
            }
            if want_dists(it):
                rows[i] = logprobs[r, pos, :].cpu().numpy().astype(np.float32)
    # assemble dists in record order
    index: Dict[str, List[int]] = {}
    blocks: List[np.ndarray] = []
    cursor = 0
    for i, it in enumerate(items):
        if i in rows:
            blk = rows[i]
            index[it["id"]] = [cursor, cursor + blk.shape[0]]
            blocks.append(blk)
            cursor += blk.shape[0]
    dists = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 0), dtype=np.float32)
    return records, dists, index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True, nargs="+",
                    help="topic sets JSONs from build_sets.py; each is scored into <out>/<its stem>/")
    ap.add_argument("--out", required=True, help="output root; one subdir per sets file")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--no-dists", action="store_true", help="NLL only (grid candidates)")
    ap.add_argument("--dists-phase", default="test", choices=["test", "selection", "all"],
                    help="which items get full distributions saved (default: test)")
    ap.add_argument("--only-phase", choices=["selection", "test"],
                    help="score only this phase's items (grid candidates: selection)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.no_dists:
        want = lambda it: False  # noqa: E731
    elif a.dists_phase == "all":
        want = lambda it: True  # noqa: E731
    else:
        want = lambda it, p=a.dists_phase: it["phase"] == p  # noqa: E731

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    print(f"loaded {a.model} as {a.dtype} on {a.device} in {time.time() - t0:.0f}s; "
          f"vocab={model.get_input_embeddings().weight.shape[0]}")

    for sets_path in a.sets:
        sets = json.loads(Path(sets_path).read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = []
        for name, lst in sets["sets"].items():
            for it in lst:
                if a.only_phase and it["phase"] != a.only_phase:
                    continue
                items.append(dict(it, set=name))
        ids = [it["id"] for it in items]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{sets_path}: duplicate item ids across sets")

        t1 = time.time()
        records, dists, index = score(model, tok, items, a.device, want, a.batch_size)
        print(f"[{sets['topic']}] scored {len(records)} items ({dists.shape[0]} dist rows) "
              f"in {time.time() - t1:.0f}s")

        out = Path(a.out) / Path(sets_path).stem
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": str(Path(a.model).resolve()), "model_name": Path(a.model).name,
            "sets_file": str(Path(sets_path).resolve()), "topic": sets["topic"],
            "dtype": a.dtype, "device": a.device,
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "code_rev": git_rev(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dists": (None if a.no_dists else a.dists_phase), "only_phase": a.only_phase,
            "vocab": int(dists.shape[1]) if dists.size else None,
        }
        (out / "records.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1))
        if not a.no_dists:
            np.save(out / "dists.npy", dists)
            (out / "dists_index.json").write_text(json.dumps(index))
        by_set: Dict[str, List[float]] = {}
        for r in records:
            by_set.setdefault(r["set"], []).append(r["nll"])
        for k, v in by_set.items():
            print(f"  {k:22s} n={len(v):3d} mean nll={np.mean(v):.4f}")
        print("  wrote", out)


if __name__ == "__main__":
    main()
