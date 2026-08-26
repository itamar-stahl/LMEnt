"""Two Harry Potter questions, both PopQA formats, all three models, verbatim.

A look before building anything. The hypothesis under test is that pornography
was the problem -- EMBER's pornography answers are conceptual ("Sexual arousal",
"1969-1984") where PopQA's are named entities, and Harry Potter's are too
("J.K. Rowling", "Sirius Black", 1.9 words on average). If the models answer
Harry Potter questions they could not answer about pornography, the topic and
answer shape were the obstacle rather than the models.

Fifteen shots from QA_train, tests from QA_test, matching PopQA's num_shots.
Nothing is scored here; the generations are the point.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Q = ("/home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/completion_eval/"
     "data/completion_questions.json")
MODELS = [("control2e",  "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"),
          ("noporn2e",   "/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e"),
          ("released2e", None)]   # filled from the cache at run time
N_SHOT, N_TEST = 15, 2


def qa_block(shots, test):
    s = "".join(f"Q: {i['q']} A: {i['correct_answer']}\n\n" for i in shots)
    return s + f"Q: {test['q']} A:"


def stem_block(shots, test):
    s = "".join(f"{i['stem']} {i['correct_answer']}\n" for i in shots)
    return s + f"{test['stem']}"


@torch.no_grad()
def gen(model, tok, prompt, dev, n=20):
    ids = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    import glob
    d = json.load(open(Q))["Harry Potter"]
    shots, tests = d["QA_train"][:N_SHOT], d["QA_test"][:N_TEST]
    models = [(l, p or glob.glob("/home/dcor/galbarak2/hf_cache/hub/"
               "models--dhgottesman--LMEnt-1B-2E/snapshots/*/step219344")[0])
              for l, p in MODELS]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[peek] device={dev}  shots={N_SHOT}  tests={N_TEST}\n")

    # Load each model ONCE and run both formats and both questions under it.
    # The first version reloaded per format -- six loads instead of three -- which
    # is merely wasteful on a GPU but fatal when CUDA fails to initialise and the
    # job falls back to CPU, as happened on job 782791.
    results = {}
    for label, path in models:
        print(f"[peek] loading {label} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32).to(dev).eval()
        for ti, t in enumerate(tests):
            for fname, fn in (("Q&A", qa_block), ("STEM", stem_block)):
                g = gen(m, tok, fn(shots, t), dev)
                results[(ti, fname, label)] = g
                print(f"[peek]   {label} q{ti+1} {fname}: "
                      f"{' '.join(g.split())[:70]!r}", flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    for ti, t in enumerate(tests):
        print("\n" + "#" * 78)
        print(f"# QUESTION {ti+1}: {t['q']}")
        print(f"#   gold: {t['correct_answer']}")
        print("#" * 78)
        for fname, fn in (("Q&A", qa_block), ("STEM", stem_block)):
            print(f"\n  --- {fname} format, last prompt line: ---")
            print(f"      {fn(shots, t).split(chr(10))[-1]!r}")
            for label, _ in models:
                g = results.get((ti, fname, label), "")
                print(f"      {label:12s} -> {' '.join(g.split())[:88]!r}")


if __name__ == "__main__":
    main()
