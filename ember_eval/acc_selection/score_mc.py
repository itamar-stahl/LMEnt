#!/usr/bin/env python
"""Score one model on the frozen SELECTION questions, four options per question.

Scoring is the repository's own primitive: ember.evals.causal_mc.continuation_logprobs
is imported rather than reimplemented, so tokenisation, padding and the choice of
logit positions are identical to causal_mc.py by construction.

Two deliberate differences from evaluate_causal_mc, both required by the protocol:

  context       the DECLARATIVE STEM, not "Question: ...\\nAnswer:". Options are
                scored as continuations of the same stem answer NLL conditions on.
  continuation  " " + option.strip()

The per-character denominator follows causal_mc.py exactly, INCLUDING the leading
space: per_char = summed_logprob / max(len(continuation), 1), where continuation
carries its leading space. Ranking is by per_char; summed and per-token scores are
saved too, but are never the prediction rule -- summed log-probability has a strong
answer-length bias.

    python score_mc.py --model <dir> --manifest questions_with_options.csv \
        --topic rome --out <dir>
"""
from __future__ import annotations

import argparse, csv, json, math, os, subprocess, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCORER_VERSION = "acc_selection/score_mc.py v1"
HERE = Path(__file__).resolve().parent
FORK = HERE.parents[1] / "Ember-on-LMEnt"
sys.path.insert(0, str(FORK))
from ember.evals.causal_mc import continuation_logprobs  # noqa: E402


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--topic", required=True,
                    help="value of the manifest's topic column to score "
                         "(rome | baseball | ai | sciq)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["topic"] == a.topic]
    if not rows:
        raise SystemExit(f"{a.topic}: no rows with that topic in {a.manifest}")
    phases = sorted({r["phase"] for r in rows})
    if not set(phases) <= {"selection", "test"}:
        raise SystemExit(f"unexpected phases in manifest: {phases}")
    groups = sorted({r["question_group"] for r in rows})
    print(f"scoring {len(rows)} items for topic={a.topic!r} "
          f"phases={phases} groups={groups}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    print(f"loaded {a.model} in {time.time()-t0:.0f}s", flush=True)

    label = a.label or Path(a.model).parent.name
    per_option, per_question = [], []
    for r in rows:
        opts = [r[f"option_{i}"] for i in range(4)]
        gold_i = int(r["gold_index"])
        conts = [" " + o.strip() for o in opts]
        scored = continuation_logprobs(model, tok, r["stem"], conts, device=a.device)
        summed = [s for s, _ in scored]
        ntok = [n for _, n in scored]
        per_char = [s / max(len(c), 1) for s, c in zip(summed, conts)]
        per_tok = [s / n for s, n in zip(summed, ntok)]
        best = max(range(4), key=per_char.__getitem__)
        ties = sum(1 for i in range(4) if i != best and per_char[i] == per_char[best])
        others = [per_char[i] for i in range(4) if i != gold_i]
        margin = per_char[gold_i] - max(others)
        order = sorted(range(4), key=lambda i: -per_char[i])
        rank = {i: order.index(i) + 1 for i in range(4)}
        nonfinite = sum(1 for v in summed + per_char + per_tok if not math.isfinite(v))
        for i in range(4):
            cont_ids = tok(conts[i], add_special_tokens=False)["input_ids"]
            per_option.append({
                "model_label": label, "checkpoint_path": str(Path(a.model).resolve()),
                "topic": r["topic"], "question_group": r["question_group"],
                "phase": "selection", "question_id": r["question_id"],
                "source_topic": r["source_topic"], "source_split": r["source_split"],
                "question": r["question"], "stem": r["stem"],
                "option_index": i, "option_text": opts[i],
                "is_correct_option": int(i == gold_i),
                "continuation": conts[i],
                "continuation_token_ids": " ".join(str(x) for x in cont_ids),
                "continuation_token_count": ntok[i],
                "continuation_char_count": len(conts[i]),
                "sum_logprob": f"{summed[i]:.10f}",
                "mean_logprob_per_token": f"{per_tok[i]:.10f}",
                "logprob_per_char": f"{per_char[i]:.10f}",
                "option_rank": rank[i],
                "is_predicted": int(i == best),
            })
        per_question.append({
            "model_label": label, "topic": r["topic"],
            "question_group": r["question_group"], "phase": "selection",
            "question_id": r["question_id"],
            "gold_index": gold_i, "predicted_index": best,
            "is_correct": int(best == gold_i),
            "score_opt0": f"{per_char[0]:.10f}", "score_opt1": f"{per_char[1]:.10f}",
            "score_opt2": f"{per_char[2]:.10f}", "score_opt3": f"{per_char[3]:.10f}",
            "gold_margin": f"{margin:.10f}", "n_ties_at_top": ties,
            "n_nonfinite": nonfinite,
        })

    out = Path(a.out) / label / a.topic
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_option.csv", per_option), ("per_question.csv", per_question)):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader()
            for d in data: w.writerow(d)
    meta = {"model": str(Path(a.model).resolve()), "label": label, "topic": a.topic,
            "dtype": a.dtype, "device": a.device, "scorer_version": SCORER_VERSION,
            "code_rev": git_rev(HERE), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "n_questions": len(rows), "n_options": len(per_option)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    # Iterate the groups actually present. Hardcoding the concept triple made
    # every sciq scoring exit non-zero on an empty slice -- and it did so AFTER
    # the CSVs were written, so 16 models were reported FAILED with complete,
    # correct output on disk.
    for g in sorted({q["question_group"] for q in per_question}):
        sub = [q for q in per_question if q["question_group"] == g]
        if not sub:
            continue
        acc = sum(q["is_correct"] for q in sub) / len(sub)
        print(f"  {g:22s} n={len(sub)} acc={acc:.4f}")
    print("  wrote", out, flush=True)


if __name__ == "__main__":
    main()
