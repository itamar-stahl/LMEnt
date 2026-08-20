"""Prove the masking actually removes blacklisted chunks from the loss.

Runs against OLMo-core's real ``get_labels``, not a reimplementation, so it
verifies the contract this whole approach depends on:

    batch["instance_mask"] == False  =>  every label of that row is
    label_ignore_index  =>  that chunk contributes nothing to the loss.

    python tests/test_exclusion.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UNTAUGHT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(UNTAUGHT_ROOT)

sys.path.insert(0, UNTAUGHT_ROOT)
sys.path.insert(0, os.environ.get("OLMO_CORE_SRC", os.path.join(REPO_ROOT, "OLMo-core", "src")))

try:
    from olmo_core.data.utils import get_labels, split_batch  # noqa: E402

    SOURCE = "olmo_core.data.utils (real)"
except ImportError:  # olmo_core deps (omegaconf, ...) absent off-cluster
    import math

    SOURCE = "vendored copy of olmo_core.data.utils"

    # Verbatim from OLMo-core/src/olmo_core/data/utils.py so this test still
    # exercises the exact upstream semantics on a laptop without the full env.
    def get_labels(batch, label_ignore_index: int = -100):  # noqa: D103
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, label_ignore_index)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, label_ignore_index)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=label_ignore_index)
        return labels[..., 1:].contiguous()

    def split_batch(batch, num_microbatch_instances: int):  # noqa: D103
        if num_microbatch_instances <= 0:
            raise RuntimeError("microbatch size is too small!")
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= num_microbatch_instances:
            return [batch]
        micro_batches = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                micro_batches[key] = value.split(num_microbatch_instances, dim=0)
            elif isinstance(value, list):
                micro_batches[key] = [
                    value[
                        num_microbatch_instances * i : num_microbatch_instances * i
                        + num_microbatch_instances
                    ]
                    for i in range(math.ceil(batch_size / num_microbatch_instances))
                ]
            else:
                raise RuntimeError(f"unexpected item in batch: '{key}={value}'")
        return [
            {key: value[i] for key, value in micro_batches.items()}
            for i in range(len(micro_batches["input_ids"]))
        ]


IGNORE = -100


def make_batch(chunk_ids, seq_len=8):
    n = len(chunk_ids)
    return {
        "input_ids": torch.randint(5, 100, (n, seq_len), dtype=torch.long),
        "attention_mask": torch.ones(n, seq_len, dtype=torch.float),
        "index": torch.tensor(chunk_ids, dtype=torch.long),
    }


def apply_mask(batch, blacklist):
    """The one line that matters, mirroring ChunkExclusionCallback.pre_step."""
    excluded = torch.isin(batch["index"], torch.tensor(blacklist, dtype=torch.long))
    batch["instance_mask"] = ~excluded
    return excluded


def test_masked_rows_are_fully_ignored():
    ids = [10, 11, 12, 13]
    batch = make_batch(ids)
    apply_mask(batch, [11, 13])

    labels = get_labels(batch, label_ignore_index=IGNORE)

    assert (labels[1] == IGNORE).all(), "blacklisted row 11 still has live labels"
    assert (labels[3] == IGNORE).all(), "blacklisted row 13 still has live labels"
    assert not (labels[0] == IGNORE).any(), "kept row 10 was wrongly masked"
    assert not (labels[2] == IGNORE).any(), "kept row 12 was wrongly masked"
    print("  ok  masked rows are entirely ignore_index, kept rows untouched")


def test_loss_normaliser_excludes_masked_tokens():
    """`_train_batch` divides by this count; it must not include masked chunks."""
    ids = [10, 11, 12, 13]
    batch = make_batch(ids, seq_len=8)
    apply_mask(batch, [11, 13])

    labels = get_labels(batch, label_ignore_index=IGNORE)
    n_tokens = int((labels != IGNORE).sum())

    # 4 rows, seq_len 8 -> labels are shifted to 7 per row; 2 rows survive.
    assert n_tokens == 2 * 7, f"expected 14 live tokens, got {n_tokens}"
    print(f"  ok  loss normaliser counts {n_tokens} tokens (masked chunks excluded)")


def test_control_run_is_a_no_op():
    ids = [10, 11, 12, 13]
    batch_ctl = make_batch(ids)
    baseline = get_labels(dict(batch_ctl), label_ignore_index=IGNORE)

    apply_mask(batch_ctl, [])  # empty blacklist
    after = get_labels(batch_ctl, label_ignore_index=IGNORE)

    assert torch.equal(baseline, after), "empty blacklist changed the labels"
    print("  ok  empty blacklist leaves labels bit-identical")


def test_instance_mask_survives_microbatch_split():
    """split_batch must carry instance_mask, or masking silently dies."""
    ids = list(range(8))
    batch = make_batch(ids)
    apply_mask(batch, [3, 5])
    batch["labels"] = get_labels(batch, label_ignore_index=IGNORE)

    micros = split_batch(batch, 4)
    assert len(micros) == 2

    for m in micros:
        assert "instance_mask" in m, "instance_mask lost during micro-batch split"
        assert "index" in m, "index lost during micro-batch split"

    assert (micros[0]["labels"][3] == IGNORE).all(), "chunk 3 leaked into micro-batch 0"
    assert (micros[1]["labels"][1] == IGNORE).all(), "chunk 5 leaked into micro-batch 1"
    print("  ok  instance_mask and labels survive split_batch")


def test_all_masked_would_break_without_guard():
    """Documents *why* ChunkExclusionCallback.guard_all_masked exists."""
    ids = [10, 11]
    batch = make_batch(ids)
    apply_mask(batch, [10, 11])

    labels = get_labels(batch, label_ignore_index=IGNORE)
    n_tokens = int((labels != IGNORE).sum())
    assert n_tokens == 0

    # This is the divide-by-zero `_train_batch` would hit.
    loss = torch.tensor(3.0) / torch.tensor(float(n_tokens))
    assert torch.isinf(loss) or torch.isnan(loss)

    # With the guard, one row is kept and the loss stays finite.
    keep = batch["instance_mask"].clone()
    keep[-1] = True
    batch["instance_mask"] = keep
    labels = get_labels(batch, label_ignore_index=IGNORE)
    assert int((labels != IGNORE).sum()) > 0
    print("  ok  fully-masked batch is NaN without the guard, finite with it")


def test_blacklist_roundtrip(tmp="_tmp_blacklist.npy"):
    ids = np.array([12, 5, 5, 99], dtype=np.int64)
    np.save(tmp, ids)
    try:
        loaded = np.unique(np.load(tmp).astype(np.int64))
        assert loaded.tolist() == [5, 12, 99], "blacklist not deduped/sorted"
        t = torch.from_numpy(loaded)
        assert t.dtype == torch.int64, "must match batch['index'] dtype"
        print("  ok  blacklist .npy round-trips deduped, sorted, int64")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [
        test_masked_rows_are_fully_ignored,
        test_loss_normaliser_excludes_masked_tokens,
        test_control_run_is_a_no_op,
        test_instance_mask_survives_microbatch_split,
        test_all_masked_would_break_without_guard,
        test_blacklist_roundtrip,
    ]
    print(f"\nrunning {len(tests)} checks against {SOURCE}\n")
    for t in tests:
        t()
    print("\nall checks passed\n")
