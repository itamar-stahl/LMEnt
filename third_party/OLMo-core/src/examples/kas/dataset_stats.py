import sys
import time
import json
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

# Add OLMo-core to sys.path
olmo_core_path = Path.cwd() / "src"
if olmo_core_path.exists():
    sys.path.insert(0, str(olmo_core_path))
    
from olmo_core.data import TokenizerConfig, NumpyDatasetConfig, NumpyDatasetType
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType

DATASET = None
TOKENIZER = None

def process_init(dataset_config_dict, tokenizer_identifier):
    global DATASET, TOKENIZER
    from olmo_core.data import NumpyDatasetConfig
    from transformers import AutoTokenizer as WorkerAutoTokenizer

    dataset_config = NumpyDatasetConfig(**dataset_config_dict)
    DATASET = dataset_config.build()
    TOKENIZER = WorkerAutoTokenizer.from_pretrained(tokenizer_identifier)

def overlaps(a, b):
    return a["char_start"] < b["char_end"] and b["char_start"] < a["char_end"]

def deduplicated_stats_for_mentions(entity_mentions):
    stats = {
        "total_mentions": 0,
        "total_entities": 0,
        "hyperlink_mentions": 0,
        "entity_linking_mentions": 0,
        "entity_linking_above_thresh": 0,
        "coref_mentions": 0,
        "coref_above_thresh": 0,
        "coref_cluster_mentions": 0,
        "coref_cluster_above_thresh": 0,
        "hyperlinks_by_entity": defaultdict(int),
        "chunks_retrieved_by_entity": defaultdict(int),
    }

    if not isinstance(entity_mentions, list):
        return stats

    # Step 1: group overlapping mentions
    sorted_mentions = sorted(entity_mentions, key=lambda m: m.get("char_start", 0))
    groups = []
    current_group = []

    for mention in sorted_mentions:
        if not current_group:
            current_group.append(mention)
            continue
        last = current_group[-1]
        if overlaps(last, mention):
            current_group.append(mention)
        else:
            groups.append(current_group)
            current_group = [mention]
    if current_group:
        groups.append(current_group)

    unique_qids = set()

    # Step 2: deduplicate counts within each group
    for group in groups:
        stats["total_mentions"] += len(group)
        group_qid_sources = defaultdict(set)

        for mention in group:
            candidates = mention.get("candidates", [])
            stats["total_entities"] += len(candidates)

            for cand in candidates:
                qid = cand.get("qid")
                if not isinstance(qid, str):
                    continue

                unique_qids.add(qid)

                scores = cand.get("scores_by_source", {})
                for source, score in scores.items():
                    try:
                        score = float(score)
                        group_qid_sources[(qid, source)].add(score)
                    except:
                        continue

        counted_qids = set()
        for (qid, source), scores in group_qid_sources.items():
            max_score = max(scores)
            if max_score == 0.0:
                continue
            if source == "hyperlinks":
                stats["hyperlink_mentions"] += 1
                if qid not in counted_qids:
                    stats["hyperlinks_by_entity"][qid] += 1
                    stats["chunks_retrieved_by_entity"]["qid"] += 1
            elif source == "entity_linking":
                stats["entity_linking_mentions"] += 1
                if max_score >= 0.6:
                    stats["entity_linking_above_thresh"] += 1
                    if qid not in counted_qids:
                        stats["chunks_retrieved_by_entity"]["qid"] += 1
            elif source == "coref":
                stats["coref_mentions"] += 1
                if max_score >= 0.6:
                    stats["coref_above_thresh"] += 1
                    if qid not in counted_qids:
                        stats["chunks_retrieved_by_entity"]["qid"] += 1
            elif source == "coref_cluster":
                stats["coref_cluster_mentions"] += 1
                if max_score >= 0.6:
                    stats["coref_cluster_above_thresh"] += 1
                    if qid not in counted_qids:
                        stats["chunks_retrieved_by_entity"]["qid"] += 1
            counted_qids.add(qid)

    return stats, unique_qids

def analyze_chunk(idx):
    global DATASET
    try:
        chunk = DATASET[idx]
        mentions = chunk["metadata"].get("entities", [])
        return deduplicated_stats_for_mentions(mentions)
    except Exception as e:
        print(f"Error processing chunk {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None, set()

def main():
    print("Starting stats collection...", flush=True)

    # Dataset config
    work_dir = Path("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache")
    tokenizer_config_obj = TokenizerConfig.dolma2()

    dataset_config = NumpyDatasetConfig.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=tokenizer_config_obj,
        work_dir=str(work_dir),
        include_instance_metadata=True,
    )

    dataset = dataset_config.build()
    total_chunks = len(dataset)
    print(f"Total chunks: {total_chunks}", flush=True)

    # Aggregate stats
    aggregated = {
        "total_mentions": 0,
        "total_entities": 0,
        "hyperlink_mentions": 0,
        "entity_linking_mentions": 0,
        "entity_linking_above_thresh": 0,
        "coref_mentions": 0,
        "coref_above_thresh": 0,
        "coref_cluster_mentions": 0,
        "coref_cluster_above_thresh": 0,
        "hyperlinks_by_entity": defaultdict(int),
        "chunks_retrieved_by_entity": defaultdict(int),
    }

    # Parallel processing
    max_workers = 128
    start_time = time.time()
    all_unique_qids = set()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=process_init,
        initargs=(dataset_config.__dict__, tokenizer_config_obj.identifier)
    ) as executor:
        futures = {executor.submit(analyze_chunk, idx): idx for idx in range(total_chunks)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_chunks):
            result, unique_qids = future.result()
            if not result:
                continue
            for key in aggregated:
                if key == "hyperlinks_by_entity":
                    for qid, count in result["hyperlinks_by_entity"].items():
                        aggregated["hyperlinks_by_entity"][qid] += count

                elif key == "chunks_retrieved_by_entity":
                    for qid, count in result["hyperlinks_by_entity"].items():
                        aggregated["chunks_retrieved_by_entity"][qid] += count                    
                
                else:
                    aggregated[key] += result[key]
            
            all_unique_qids.update(unique_qids)

    aggregated["total_entities"] = len(all_unique_qids)

    # Print summary
    print("\n=== ENTITY STATS SUMMARY ===")
    for key, val in aggregated.items():
        if key != "hyperlinks_by_entity":
            print(f"{key}: {val}")
    print(f"Unique entities with hyperlinks: {len(aggregated['hyperlinks_by_entity'])}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

    # Save output
    stats_path = Path("overall_stats.json")
    hyperlinks_path = Path("hyperlinks_by_entity.json")
    chunks_retrieved_by_entity_path = Path("chunks_retrieved_by_entity.json")


    # with open(stats_path, "w") as f:
    #     json.dump({k: v for k, v in aggregated.items() if k != "hyperlinks_by_entity"}, f, indent=2)
    # print(f"Saved stats to {stats_path.resolve()}")

    # with open(hyperlinks_path, "w") as f:
    #     json.dump(dict(aggregated["hyperlinks_by_entity"]), f, indent=2)
    # print(f"Saved hyperlinks_by_entity to {hyperlinks_path.resolve()}")

    with open(chunks_retrieved_by_entity_path, "w") as f:
        json.dump(dict(aggregated["chunks_retrieved_by_entity"]), f, indent=2)
    print(f"Saved chunks_retrieved_by_entity to {chunks_retrieved_by_entity_path.resolve()}")

if __name__ == "__main__":
    main()
