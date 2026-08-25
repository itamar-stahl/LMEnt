"""Per-token loss of one model on the held-out chunks, and on matched controls.

This is the measurement ember_eval/EVALUATION.md ranked first after the
completion evaluation came back null (COMPLETION_RESULTS.md). It asks the only
question aimed squarely at the intervention itself: the ablated twin received no
gradient from 2,546 specific chunks -- did that leave a trace in its loss on
exactly those chunks?

No prompt, no answer key, no option scoring, no metric choice, and no reliance
on a question-answering ability these base models do not have. Millions of
tokens instead of 100 questions.

Two sets are scored:

  held-out   the 2,546 chunk_ids in the run's untaught_blacklist.json
  control    a random sample of chunk_ids NOT in the blacklist, length-matched
             by VSL bucket so the two sets have the same sequence-length mix
             (loss varies systematically with position, so an unmatched control
             would confound length with concept)

The comparison of interest is the DIFFERENCE OF DIFFERENCES:

    (ablated_heldout - control_heldout) - (ablated_ctrlset - control_ctrlset)

Each twin's own control-set loss absorbs any global difference between the two
models, so what remains is specific to the masked chunks. Per-chunk losses are
written out so the pairing can be tested without a GPU.

    python heldout_ppl.py --model <hf dir> --out <json> [--n-control 5000]
"""
import argparse, json, random, time

import torch
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType
from transformers import AutoModelForCausalLM

DATA_GLOB = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-tokenized/*.npy"
WORK_DIR = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-cache"


def build_dataset(work_dir):
    return NumpyDatasetConfig.glob(
        DATA_GLOB,
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=work_dir,
        include_instance_metadata=False,
    ).build()


def match_by_length(ds, held, n_control, seed, pool_factor=40):
    """Sample non-blacklisted ids whose length histogram matches the held-out set.

    VSL packs instances into power-of-two buckets and mean token loss falls with
    position, so a control set with a different length mix would differ from the
    held-out set for reasons having nothing to do with the concept.
    """
    rng = random.Random(seed)
    held_set = set(held)
    want = {}
    for i in held:
        want[len(ds[i]["input_ids"])] = want.get(len(ds[i]["input_ids"]), 0) + 1
    scale = n_control / max(len(held), 1)
    want = {L: max(1, round(c * scale)) for L, c in want.items()}

    by_len, seen, picked = {}, 0, set()
    target_total = sum(want.values())
    while seen < target_total * pool_factor and len(by_len) < 10**7:
        j = rng.randrange(len(ds))
        seen += 1
        if j in held_set:
            continue
        if j in picked:
            continue          # rng.randrange can repeat; without this the same
                              # chunk is scored twice and double-counts
        L = len(ds[j]["input_ids"])
        if L in want and len(by_len.setdefault(L, [])) < want[L]:
            by_len[L].append(j)
            picked.add(j)
        if all(len(by_len.get(L, [])) >= c for L, c in want.items()):
            break
    out = [j for L in want for j in by_len.get(L, [])]
    rng.shuffle(out)
    return out


@torch.no_grad()
def chunk_losses(model, ds, ids, device, log_every=500):
    """Mean per-token NLL for each id, plus its token count."""
    rows = []
    t0 = time.time()
    for n, i in enumerate(ids):
        x = ds[i]["input_ids"].to(device).unsqueeze(0)
        if x.shape[1] < 2:
            continue
        logits = model(input_ids=x).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1], x[0, 1:], reduction="mean")
        rows.append({"chunk_id": int(i), "tokens": int(x.shape[1]),
                     "loss": float(loss)})
        if (n + 1) % log_every == 0:
            done = n + 1
            rate = done / (time.time() - t0)
            print(f"    {done}/{len(ids)}  {rate:.1f} chunk/s  "
                  f"eta {(len(ids)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
    return rows


def summarize(rows):
    if not rows:
        return {}
    tot_tok = sum(r["tokens"] - 1 for r in rows)
    tot_nll = sum(r["loss"] * (r["tokens"] - 1) for r in rows)
    macro = sum(r["loss"] for r in rows) / len(rows)
    micro = tot_nll / max(tot_tok, 1)
    return {"n_chunks": len(rows), "n_tokens": tot_tok,
            "mean_loss_macro": macro, "mean_loss_micro": micro,
            "ppl_micro": float(torch.exp(torch.tensor(micro)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--blacklist", default=(
        "/home/morg/NLP_2526b/galbarak2/LMEnt/Untaught/runs/"
        "untaught-no-porn-1b-2e_20260818_183858/untaught_blacklist.json"))
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--n-control", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    held = json.load(open(a.blacklist))["entities"][0]["chunk_ids"]
    print(f"[ppl] {len(held)} held-out chunk ids", flush=True)

    ds = build_dataset(a.work_dir)
    print(f"[ppl] dataset: {len(ds)} instances", flush=True)
    assert max(held) < len(ds), "chunk_id out of range -- wrong dataset build"

    if a.limit:
        held = held[:a.limit]
    ctrl = match_by_length(ds, held, a.n_control if not a.limit else a.limit, a.seed)
    print(f"[ppl] {len(ctrl)} length-matched control chunks", flush=True)

    print(f"[ppl] loading {a.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()

    print("[ppl] scoring held-out set", flush=True)
    held_rows = chunk_losses(model, ds, held, a.device)
    print("[ppl] scoring control set", flush=True)
    ctrl_rows = chunk_losses(model, ds, ctrl, a.device)

    hs, cs = summarize(held_rows), summarize(ctrl_rows)
    print(f"\n  held-out : {hs['n_chunks']} chunks, {hs['n_tokens']} tokens, "
          f"loss {hs['mean_loss_micro']:.4f}, ppl {hs['ppl_micro']:.3f}")
    print(f"  control  : {cs['n_chunks']} chunks, {cs['n_tokens']} tokens, "
          f"loss {cs['mean_loss_micro']:.4f}, ppl {cs['ppl_micro']:.3f}")
    print(f"  held-out minus control: {hs['mean_loss_micro']-cs['mean_loss_micro']:+.4f} nats/token")

    json.dump({"metadata": {"model": a.model, "blacklist": a.blacklist,
                            "seed": a.seed, "dtype": a.dtype,
                            "n_control_requested": a.n_control},
               "heldout": hs, "control": cs,
               "heldout_rows": held_rows, "control_rows": ctrl_rows},
              open(a.out, "w"))
    print(f"[ppl] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
