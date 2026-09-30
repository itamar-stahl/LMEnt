import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# Add the src directory to Python path
olmo_core_path = Path.cwd() / "src"
if olmo_core_path.exists():
    sys.path.insert(0, str(olmo_core_path))

from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,

)
from olmo_core.data.numpy_dataset import (
    VSLCurriculumType,
    VSLCurriculumConfig,
)

# # Set your new cache base directory (change this to your preferred location)
# cache_base = "/home/morg/students/gottesman3/.cache"

# # Set all relevant Hugging Face cache directories
# os.environ["HF_HOME"] = cache_base
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
# os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
# os.environ["HF_TOKENIZERS_CACHE"] = os.path.join(cache_base, "tokenizers")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from olmo_eval import HFTokenizer


def save_all_batches(dataloader, output_file='/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/batch_indices.npy'):
    all_batches = []

    try:
        for batch in tqdm(dataloader, desc="Saving batch indices"):
            all_batches.append(batch["index"].detach().cpu().numpy())
            del batch
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        np.save(output_file, np.array(all_batches, dtype=object))  # variable-length rows

    
    except Exception as e:
        # Save checkpoint before crashing
        current_batch = len(all_batches)
        np.save(f'/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/batch_indices_checkpoint_{current_batch}.npy', np.array(all_batches, dtype=object))  # variable-length rows

        print(f"Error occurred at batch {current_batch}. Checkpoint saved.")
        print(f"To resume, simply run the script again.")
        raise
        

if __name__ == "__main__":
    print("Loading dataset")
    tokenizer_config = TokenizerConfig.dolma2()
    tokenizer = HFTokenizer(
                tokenizer_config.identifier,
                pad_token_id=tokenizer_config.pad_token_id,
                eos_token_id=tokenizer_config.eos_token_id,
                bos_token_id=tokenizer_config.bos_token_id,
            )

    include_instance_metadata = False # Set to true when you want tp retrieve metadata, during training set this to False
    work_dir = "/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache"

    dataset_config = NumpyDatasetConfig.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",  # can be globs
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        include_instance_metadata=include_instance_metadata,
    )
    dataset = dataset_config.build()

    print("Preparing dataloader")
    data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=32768,
    seed=0,
    num_workers=16,
    prefetch_factor=128,
    )

    dataloader = data_loader_config.build(dataset)
    dataloader.reshuffle(1)

    print("Iterating over the batches")
    save_all_batches(dataloader)