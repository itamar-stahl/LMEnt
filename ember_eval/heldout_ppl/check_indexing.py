"""Prove that chunk_id indexes the text we think it does, before spending a GPU.

The whole held-out-chunk measurement rests on one claim, from
Untaught/framework/node/exclusion.py: a blacklist chunk_id *is* the OLMo-core
dataset instance index, so DATASET[chunk_id] is the very text that was masked
out of the ablated twin's loss.

That claim was established for the authors' tokenized copy. We trained on
stahli/LMEnt-Dataset. If the two tokenizations differ in file order or content,
the indices silently shift and every number downstream is meaningless while
looking perfectly reasonable. So: decode some blacklisted chunks and some
random ones, and count how often the concept's own vocabulary shows up.

Run with no arguments. CPU only, no model.
"""
import json, random, re, sys

from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType
from transformers import AutoTokenizer

BLACKLIST = ("/home/morg/NLP_2526b/galbarak2/LMEnt/Untaught/runs/"
             "untaught-no-porn-1b-2e_20260818_183858/untaught_blacklist.json")
HF_TOK = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"
N = 12

# Words that should be far more common in Q291 chunks than in random Wikipedia.
MARKERS = re.compile(
    r"\b(porn\w*|pornograph\w*|obscen\w*|erotic\w*|X-rated|adult film\w*|"
    r"hardcore|softcore|nudit\w*|censor\w*|indecen\w*|smut\w*)\b", re.I)


def build_dataset(work_dir):
    cfg = NumpyDatasetConfig.glob(
        "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-tokenized/*.npy",
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=work_dir,
        include_instance_metadata=False,
    )
    return cfg.build()


def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-cache"
    print(f"[check] work_dir = {work_dir}", flush=True)

    ids = json.load(open(BLACKLIST))["entities"][0]["chunk_ids"]
    print(f"[check] {len(ids)} blacklisted chunk ids, max {max(ids)}", flush=True)

    ds = build_dataset(work_dir)
    print(f"[check] dataset built: {len(ds)} instances", flush=True)
    if max(ids) >= len(ds):
        print(f"!! FATAL: max chunk_id {max(ids)} >= dataset length {len(ds)}")
        return 1

    tok = AutoTokenizer.from_pretrained(HF_TOK)
    rng = random.Random(42)
    picked = rng.sample(ids, N)
    randoms = [rng.randrange(len(ds)) for _ in range(N)]

    def decode(i):
        return tok.decode(ds[i]["input_ids"].tolist(), skip_special_tokens=True)

    hits_bl = hits_rand = 0
    print("\n=== BLACKLISTED chunks (expect concept vocabulary) ===")
    for i in picked:
        t = decode(i)
        m = MARKERS.findall(t)
        hits_bl += bool(m)
        print(f"\n[{i}] markers={len(m)} {sorted(set(w.lower() for w in m))[:6]}")
        print("   ", " ".join(t.split())[:220])

    print("\n\n=== RANDOM chunks (expect almost none) ===")
    for i in randoms:
        t = decode(i)
        m = MARKERS.findall(t)
        hits_rand += bool(m)
        print(f"[{i}] markers={len(m)}   {' '.join(t.split())[:110]}")

    print(f"\n=== VERDICT: blacklisted {hits_bl}/{N} contain concept vocabulary, "
          f"random {hits_rand}/{N}")
    if hits_bl >= N * 0.7 and hits_rand <= N * 0.25:
        print("=== INDEXING CONFIRMED: chunk_id resolves to the masked text.")
        return 0
    print("=== INDEXING NOT CONFIRMED. Do not run the measurement on these indices.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
