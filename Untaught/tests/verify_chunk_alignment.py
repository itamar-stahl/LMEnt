"""Empirically verify chunk_id <-> dataset-index alignment on the remote.

The entire Untaught design rests on one identity: the ``chunk_id`` stored in
the Elasticsearch index equals the OLMo-core dataset instance index. That was
verified by reading the code (both sides build the same ``NumpyDatasetConfig``
and ``create_es_index.py`` stores ``'chunk_id': idx``), and path order is
deterministic because config globs are ``sorted()`` -- but code review cannot
prove the *deployed* index and the *deployed* dataset directory actually match
(e.g. a stale snapshot, a missing tokenized file, a renamed part).

This script proves it empirically: sample N random dataset indices, decode each
chunk's tokens, fetch the same chunk_id from ES, and require the texts to be
identical (ES stored ``tokenizer.decode(chunk)`` at index-build time, so equal
strings are expected, not just similar ones).

REMOTE ONLY -- needs the lment conda env, the dataset, and Elasticsearch.
Takes a few minutes (dataset construction dominates). Run it once before any
real training:

    cd $LMENT_ROOT/Untaught && . ./activate_env.sh
    python tests/verify_chunk_alignment.py --config configs/train_170m_control.json -n 25

(activate_env.sh gives you conda, the paths and a running Elasticsearch. If
you skip it, an unresolved config still makes the script read configs/env.sh
for the paths -- but nothing will start Elasticsearch for you.)

Exit code 0 = aligned; 1 = MISALIGNED (do not train until resolved).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNTAUGHT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(UNTAUGHT_ROOT)

sys.path.insert(0, UNTAUGHT_ROOT)

from framework.config_env import load_config, load_env_sh  # noqa: E402

# OLMO_CORE_SRC is read before any olmo import, so fill the environment in first
# for the case where the environment was not sourced.
if "OLMO_CORE_SRC" not in os.environ:
    load_env_sh()
sys.path.insert(
    0, os.environ.get("OLMO_CORE_SRC", os.path.join(REPO_ROOT, "OLMo-core", "src"))
)

from framework.es_blacklist import get_esclient  # noqa: E402


def build_dataset(config_path: str):
    """Build the dataset exactly as training does -- via the upstream
    ``build_config`` -- so every parameter (glob, curriculum, seq lens, dtype)
    is identical by construction, not by copy-paste."""
    from examples.kas.train import build_config

    config = build_config(load_config(config_path))
    dataset = config.dataset.build()
    dataset.prepare()
    return dataset


def make_tokenizer():
    """Mirror ``create_es_index.py``'s tokenizer so decode output is comparable."""
    from olmo_core.data import TokenizerConfig
    from olmo_eval import HFTokenizer

    tc = TokenizerConfig.dolma2()
    return HFTokenizer(
        tc.identifier,
        pad_token_id=tc.pad_token_id,
        eos_token_id=tc.eos_token_id,
        bos_token_id=tc.bos_token_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="a training config JSON")
    parser.add_argument("-n", type=int, default=25, help="chunks to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--es-index", default=os.environ.get("ES_INDEX", "lment_cs")
    )
    args = parser.parse_args()

    print(f"[verify] building dataset from {args.config} ...")
    dataset = build_dataset(args.config)
    n_instances = len(dataset)
    print(f"[verify] dataset has {n_instances:,} chunks")

    tokenizer = make_tokenizer()
    es = get_esclient()

    es_total = es.count(index=args.es_index)["count"]
    print(f"[verify] ES index '{args.es_index}' has {es_total:,} chunks")
    if es_total != n_instances:
        print(
            f"[verify] FAIL: size mismatch -- dataset {n_instances:,} vs "
            f"ES {es_total:,}. The index and the dataset are not the same corpus.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(range(n_instances), min(args.n, n_instances))

    failures = 0
    for cid in sample:
        decoded = tokenizer.decode(dataset[cid]["input_ids"].tolist())

        resp = es.search(
            index=args.es_index,
            body={
                "size": 2,
                "query": {"term": {"chunk_id": cid}},
                "_source": ["chunk_id", "text", "title"],
            },
        )
        hits = resp["hits"]["hits"]
        if len(hits) != 1:
            print(f"  FAIL chunk_id={cid}: expected 1 ES hit, got {len(hits)}")
            failures += 1
            continue

        es_text = hits[0]["_source"]["text"]
        if es_text == decoded:
            print(f"  ok   chunk_id={cid}  ({len(decoded)} chars, title={hits[0]['_source'].get('title')!r})")
        else:
            failures += 1
            print(f"  FAIL chunk_id={cid}: text mismatch")
            print(f"       dataset: {decoded[:120]!r}...")
            print(f"       es     : {es_text[:120]!r}...")

    print()
    if failures:
        print(
            f"[verify] MISALIGNED: {failures}/{len(sample)} chunks differ. "
            "DO NOT TRAIN -- the blacklist would exclude the wrong chunks.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[verify] PASS: {len(sample)}/{len(sample)} sampled chunks are "
        "byte-identical between the dataset and the ES index. chunk_id "
        "alignment holds on this deployment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
