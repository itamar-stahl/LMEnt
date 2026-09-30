"""Does few-shot prompting make these base models answer EMBER's questions?

ember_eval/EVALUATION.md established that zero-shot they do not: given EMBER's
"Answer only with one letter" prompt they emit neither a letter nor an option,
so every letter-parsing evaluator scores them near zero. OLMES avoids this by
prompting few-shot -- the paper's own triviaqa::kas uses 10 shots -- and its
tasks do work on models this size (OLMES_RESULTS.md). This asks whether the same
trick rescues EMBER's questions specifically.

Ten test questions, in two formats, with identical shots:

  open      Question: ...\\nAnswer: <gold>   x5, then the test question
  mc        the same but each item lists A-D and the answer is a letter

Both are scored by GENERATION, which is what EMBER's protocol assumes and what
these models were previously unable to do. Shots come from QA_train and test
items from QA_test, so they are disjoint.

Everything the model writes is printed verbatim. The generations are the point:
a number cannot show whether a model answered the question or wandered off.
"""
import argparse, hashlib, json, random, re, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = ("/home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/completion_eval/"
             "data/completion_questions.json")
LETTERS = "ABCD"


def fmt_open(item, answer=None):
    s = f"Question: {item['q']}\nAnswer:"
    return s + (f" {answer}\n\n" if answer is not None else "")


def fmt_mc(item, answer=None):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    s = f"Question: {item['q']}\n{opts}\nAnswer:"
    return s + (f" {LETTERS[answer]}\n\n" if answer is not None else "")


def shuffle_options(item, concept, split, seed=42):
    """Reorder options exactly as completion_eval/evaluate_completion.py does.

    The raw JSON keeps EMBER's original order, which puts the correct answer at
    index 0 in ALL 50 QA_test items. Reading `options` straight from the file
    therefore asks every question with the answer at "A"; the models answer "B"
    most of the time, so the score is a measurement of that coincidence and
    nothing else. Job 782141 was scored this way and its multiple-choice half is
    void.

    Replicated rather than reimplemented, so item N here is item N in every
    other run: per-question seed from a sha1 of concept/subset/split/question.
    """
    subset, sample_split = split.rsplit("_", 1)
    sid = f"{concept}|{subset}|{sample_split}|{item['q'].strip()}"
    per_q = (seed + int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 32)
    opts = [o.strip() for o in item["options"]]
    random.Random(per_q).shuffle(opts)
    item["options"] = opts[:4]
    return item


def gold_index(item):
    return item["options"].index(item["correct_answer"].strip())


def pick_balanced_shots(pool, concept, split, n=4, seed=42):
    """One shot per gold letter, in a seeded-random order.

    Shots chosen arbitrarily carry a letter-frequency signal the model can copy:
    the first five QA_train items give golds C, C, A, A, B -- C and A doubled, D
    never shown. Taking exactly one of each letter makes the shot block
    uninformative about which letter to answer, so whatever the model does with
    letters comes from the question rather than from the prompt.

    The order is shuffled (deterministically) so the block is not itself the
    fixed sequence A, B, C, D, which would be its own pattern to copy.
    """
    by_letter = {}
    for raw in pool:
        it = shuffle_options(dict(raw), concept, split)
        by_letter.setdefault(LETTERS[gold_index(it)], []).append(it)
    missing = [L for L in LETTERS[:n] if L not in by_letter]
    if missing:
        raise SystemExit(f"no {split} item with gold letter(s) {missing}; "
                         "cannot build a letter-balanced shot block")
    shots = [by_letter[L][0] for L in LETTERS[:n]]
    random.Random(seed).shuffle(shots)
    return shots


def build(shots, item, mode):
    f = fmt_open if mode == "open" else fmt_mc
    ctx = "".join(f(s, s["correct_answer"] if mode == "open" else gold_index(s))
                  for s in shots)
    return ctx + f(item)


@torch.no_grad()
def generate(model, tok, prompt, device, max_new):
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--concept", default="Pornography")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--balanced", action="store_true",
                    help="one shot per gold letter, so the shot block carries no "
                         "letter-frequency signal")
    ap.add_argument("--out")
    a = ap.parse_args()

    data = json.load(open(QUESTIONS))[a.concept]
    if a.balanced:
        shots = pick_balanced_shots(data["QA_train"], a.concept, "QA_train", a.shots)
    else:
        shots = [shuffle_options(dict(x), a.concept, "QA_train") for x in data["QA_train"][:a.shots]]
    tests = [shuffle_options(dict(x), a.concept, "QA_test") for x in data["QA_test"][:a.n]]
    from collections import Counter
    print(f"[probe] shots: {len(shots)}, gold letters in order "
          f"{[LETTERS[gold_index(sh)] for sh in shots]}"
          f"{' (letter-balanced)' if a.balanced else ''}")
    print(f"[probe] gold-letter distribution in the {len(tests)} test items: "
          f"{dict(Counter(LETTERS[gold_index(t)] for t in tests))}")

    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32).to(dev).eval()

    records = []
    for mode, max_new in (("open", 20), ("mc", 5)):
        hits = parsed = 0
        print("\n" + "=" * 78)
        print(f"  {a.label}   {mode.upper()}   {a.shots}-shot   {len(tests)} questions")
        print("=" * 78)
        if not records:
            print("--- prompt for item 0, verbatim ---")
            print(build(shots, tests[0], mode))
            print("--- end prompt ---")
        for i, item in enumerate(tests):
            gi = gold_index(item)
            gen = generate(model, tok, build(shots, item, mode), dev, max_new)
            one = " ".join(gen.split())
            if mode == "open":
                ok = item["correct_answer"].strip().lower() in gen.lower()
                got = None
            else:
                m = re.search(r"\b([A-D])\b", gen)
                got = m.group(1) if m else None
                parsed += got is not None
                ok = got == LETTERS[gi]
            hits += ok
            print(f"\n[{i:02d}] {item['q'][:82]}")
            print(f"     gold : {item['correct_answer'][:60]}"
                  + (f"   ({LETTERS[gi]})" if mode == "mc" else ""))
            print(f"     wrote: {one[:100]!r}")
            print(f"     -> {'CORRECT' if ok else 'wrong'}"
                  + (f"   parsed={got}" if mode == "mc" else ""))
            records.append({"model": a.label, "mode": mode, "i": i,
                            "question": item["q"], "gold": item["correct_answer"],
                            "gold_letter": LETTERS[gi], "generated": gen,
                            "parsed": got, "correct": bool(ok)})
        print(f"\n  {a.label} / {mode}: {hits}/{len(tests)} correct"
              + (f", parse rate {parsed}/{len(tests)}" if mode == "mc" else ""))

    if a.out:
        json.dump(records, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
