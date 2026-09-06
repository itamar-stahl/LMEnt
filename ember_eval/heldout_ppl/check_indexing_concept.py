"""Prove that chunk_id indexes the text we think it does, before spending a GPU.

Generalised from check_indexing.py, which hardcoded the Pornography run. Same
claim under test, for any concept: a blacklist chunk_id *is* the OLMo-core
dataset instance index, so DATASET[chunk_id] is the very text that was masked
out of the ablated twin's loss. If the tokenized copy we score differs from the
one we trained on in file order or content, the indices shift silently and every
number downstream is meaningless while looking perfectly reasonable.

    python check_indexing_concept.py --blacklist <json> --markers <regex>

CPU only, no model.
"""
import argparse, json, random, re, sys

from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType
from transformers import AutoTokenizer

DATA_GLOB = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-tokenized/*.npy"
WORK_DIR = "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-cache"
HF_TOK = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blacklist", required=True)
    ap.add_argument("--markers", required=True, help="regex of concept vocabulary")
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    markers = re.compile(a.markers, re.I)
    ids = json.load(open(a.blacklist))["entities"][0]["chunk_ids"]
    print(f"[check] {len(ids)} blacklisted chunk ids, max {max(ids)}", flush=True)

    ds = build_dataset(a.work_dir)
    print(f"[check] dataset built: {len(ds)} instances", flush=True)
    if max(ids) >= len(ds):
        print(f"!! FATAL: max chunk_id {max(ids)} >= dataset length {len(ds)}")
        return 1

    tok = AutoTokenizer.from_pretrained(HF_TOK)
    rng = random.Random(a.seed)
    picked = rng.sample(ids, a.n)
    randoms = [rng.randrange(len(ds)) for _ in range(a.n)]

    def decode(i):
        return tok.decode(ds[i]["input_ids"].tolist(), skip_special_tokens=True)

    hits_bl = hits_rand = 0
    print("\n=== BLACKLISTED chunks (expect concept vocabulary) ===")
    for i in picked:
        t = decode(i)
        m = markers.findall(t)
        hits_bl += bool(m)
        print(f"\n[{i}] markers={len(m)} {sorted(set(w.lower() for w in m))[:6]}")
        print("   ", " ".join(t.split())[:220])

    print("\n\n=== RANDOM chunks (expect almost none) ===")
    for i in randoms:
        t = decode(i)
        m = markers.findall(t)
        hits_rand += bool(m)
        print(f"[{i}] markers={len(m)}   {' '.join(t.split())[:110]}")

    print(f"\n=== VERDICT: blacklisted {hits_bl}/{a.n} contain concept vocabulary, "
          f"random {hits_rand}/{a.n}")
    if hits_bl >= a.n * 0.7 and hits_rand <= a.n * 0.25:
        print("=== INDEXING CONFIRMED: chunk_id resolves to the masked text.")
        return 0
    print("=== INDEXING NOT CONFIRMED. Do not run the measurement on these indices.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
