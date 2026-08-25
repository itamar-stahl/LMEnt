"""Independent cross-check of the per-chunk loss in heldout_ppl.py.

heldout_ppl.py computes, for each chunk,

    cross_entropy(logits[0, :-1], x[0, 1:], reduction="mean")

The headline result rests on that one line, so recompute it two other ways and
require all three to agree: HuggingFace's own `labels=` path, which does the
shift internally, and an explicit gather of log-softmax probabilities.
"""
import json, sys
import torch
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.data.numpy_dataset import VSLCurriculumConfig, VSLCurriculumType
from transformers import AutoModelForCausalLM

MODEL = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"
BLACKLIST = ("/home/morg/NLP_2526b/galbarak2/LMEnt/Untaught/runs/"
             "untaught-no-porn-1b-2e_20260818_183858/untaught_blacklist.json")
REF = "/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/heldout/ppl_control2e_776653.json"
N = 20

ds = NumpyDatasetConfig.glob(
    "/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-tokenized/*.npy",
    name=NumpyDatasetType.kas_vsl, max_sequence_length=2048, min_sequence_length=64,
    vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
    tokenizer=TokenizerConfig.dolma2(),
    work_dir="/home/morg/NLP_2526b/stahli/LMEnt-Dataset/dataset-cache",
    include_instance_metadata=False).build()

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()

recorded = {r["chunk_id"]: r["loss"] for r in json.load(open(REF))["heldout_rows"]}
ids = json.load(open(BLACKLIST))["entities"][0]["chunk_ids"][:N]

worst_hf = worst_gather = worst_rec = 0.0
with torch.no_grad():
    for i in ids:
        x = ds[i]["input_ids"].to(dev).unsqueeze(0)
        logits = m(input_ids=x).logits.float()
        a = torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="mean").item()
        b = m(input_ids=x, labels=x).loss.item()                      # HF's own shift
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        c = -lp.gather(1, x[0, 1:].unsqueeze(1)).mean().item()        # explicit gather
        worst_hf = max(worst_hf, abs(a - b))
        worst_gather = max(worst_gather, abs(a - c))
        worst_rec = max(worst_rec, abs(a - recorded[i]))
        print(f"  chunk {i:>9}  ours {a:.6f}  hf {b:.6f}  gather {c:.6f}  recorded {recorded[i]:.6f}")

print(f"\nmax |ours - HF labels=|   {worst_hf:.3e}")
print(f"max |ours - gather|       {worst_gather:.3e}")
print(f"max |ours - recorded run| {worst_rec:.3e}")
ok = worst_hf < 1e-4 and worst_gather < 1e-5 and worst_rec < 1e-4
print("LOSS VERIFIED — three implementations agree" if ok else "MISMATCH — investigate")
sys.exit(0 if ok else 1)
