"""Chunk-level exclusion callback.

The whole mechanism in one sentence: LMEnt's Elasticsearch index stores a
``chunk_id`` for every chunk, that ``chunk_id`` *is* the OLMo-core dataset
instance index, and every training batch carries those same ids in
``batch["index"]`` -- so holding a concept out is just masking the matching
rows of the batch.

Why this needs no dataset rebuild
---------------------------------
``retrieval-index/create_es_index.py`` builds its ES documents by iterating a
``NumpyDatasetConfig`` that is identical to the one ``train.py`` builds (same
glob, ``kas_vsl``, 2048/64, ``grow_p2``/8/unbalanced) and stores
``'chunk_id': idx`` where ``idx`` is the argument to ``DATASET[idx]``.
``data_loader.py::_get_dataset_item`` attaches that same ``idx`` to every
instance as ``index=idx``.  One identifier, end to end.

Why masking and not dropping
----------------------------
VSL batches are token-constant and instance-variable
(``instances_per_batch = global_batch_size // bucket_seq_len``), and
``_batch_index_to_local_instance_indices`` slices them *per rank*.  Physically
dropping instances would give different ranks different instance counts, which
desynchronises FSDP collectives, and would change the effective batch size
relative to the control run.  Masking keeps batch shapes, step counts and data
order bit-identical between the control and ablated runs -- the only difference
is that excluded chunks contribute no gradient.

OLMo-core supports this natively.  ``data/utils.py::get_labels`` applies
``labels.masked_fill_(~instance_mask.unsqueeze(-1), label_ignore_index)``, and
``trainer.py::_train_batch`` already logs a ``train/masked instances`` metric
when an ``instance_mask`` is present.  We are using the framework as intended,
not patching around it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from olmo_core.train.callbacks import Callback
from olmo_core.train.common import ReduceType

try:
    from .es_blacklist import ARTIFACT_NAME, load_artifact
except ImportError:  # pragma: no cover - file-path launch
    from framework.es_blacklist import ARTIFACT_NAME, load_artifact

log = logging.getLogger(__name__)

EXCLUDED_METRIC = "train/untaught excluded instances"
EXCLUDED_CUMULATIVE_METRIC = "train/untaught excluded cumulative"
GUARD_LEAK_METRIC = "train/untaught guard leaks"


@dataclass
class ChunkExclusionCallback(Callback):
    """Zero out the training signal from the chunks that mention an entity.

    :param blacklist: path to the run's QID list. Only a marker here -- its
        presence says an ablation was configured, so the artifact below is
        required. ``None`` makes this callback a no-op, which is exactly what
        the control run wants.
    :param artifact_name: file inside the trainer's ``save_folder`` holding the
        resolved chunk ids, written before submission by
        ``train_untaught.py --prepare`` on a machine that can reach
        Elasticsearch. Loaded into a ``{chunk_id: qid}`` dict at ``pre_train``.
    :param chunk_ids: chunk ids given directly, skipping the artifact. Takes
        precedence over ``blacklist``; mainly for tests.
    :param enabled: Set ``False`` to keep the callback attached but inert.
    :param guard_all_masked: If every instance in a rank's batch would be masked,
        keep one so the loss normaliser cannot be zero. See note below.
    :param strict: Raise if a batch has no ``index`` key instead of warning once.
    """

    # Run before any callback that might inspect the batch.
    priority = 10

    blacklist: Optional[str] = None
    artifact_name: str = ARTIFACT_NAME
    chunk_ids: Optional[Sequence[int]] = None
    enabled: bool = True
    guard_all_masked: bool = True
    strict: bool = True

    def post_attach(self):
        # Runtime state is deliberately NOT declared as dataclass fields: the
        # trainer serialises callbacks via `config.as_config_dict()` for W&B and
        # the config saver, and a dict of ids would not survive that.
        self._blacklist: Optional[Dict[int, str]] = None
        self._total_excluded: int = 0
        self._warned_missing_index: bool = False

    def pre_train(self):
        self._blacklist = self._load_blacklist()

        if self._blacklist is None:
            log.info("[untaught] no blacklist configured -- this is a CONTROL run")
            return

        log.info(
            "[untaught] holding out %d chunk ids from the loss (source: %s)",
            len(self._blacklist),
            self.blacklist or "explicit chunk_ids",
        )

    def pre_step(self, batch: Dict[str, Any]):
        if not self.enabled or not self._blacklist:
            return

        index = batch.get("index")
        if index is None:
            msg = (
                "[untaught] batch has no 'index' key, cannot exclude chunks. "
                "This dataset/dataloader combination does not expose chunk ids."
            )
            if self.strict:
                raise RuntimeError(msg)
            if not self._warned_missing_index:
                log.warning(msg)
                self._warned_missing_index = True
            return

        # One dict lookup per instance. A batch holds 16-512 of them (VSL:
        # global_batch_size // bucket_seq_len), so this is cheaper than moving a
        # blacklist tensor to the device on every step.
        excluded = torch.tensor(
            [int(i) in self._blacklist for i in index.tolist()],
            dtype=torch.bool,
            device=index.device,
        )
        keep = ~excluded

        # Respect any mask that is already on the batch rather than clobbering it.
        existing = batch.get("instance_mask")
        if existing is not None:
            keep = keep & existing.to(device=keep.device, dtype=torch.bool)

        # `_train_batch` divides the summed loss by the number of non-ignored
        # label tokens in the batch. If a rank masks *everything*, that divisor
        # is 0 and the loss becomes NaN, which then poisons the all-reduce for
        # every other rank. With a realistic blacklist (a few thousand ids out
        # of 10.5M chunks) this cannot happen, but a silent NaN would waste a
        # multi-hour job so we spend one branch on it.
        if self.guard_all_masked and not bool(keep.any()):
            keep[-1] = True
            # This *leaks* one blacklisted chunk into training. Record it so the
            # run's exclusion guarantee is auditable, not just a log line.
            self.trainer.record_metric(GUARD_LEAK_METRIC, 1.0, ReduceType.sum)
            log.warning(
                "[untaught] step %d: every instance in this rank's batch was "
                "blacklisted; keeping 1 (A LEAK) to avoid a divide-by-zero loss. "
                "If you see this often your blacklist is too broad for "
                "chunk-level exclusion.",
                self.step,
            )

        batch["instance_mask"] = keep

        n_excluded = int(excluded.sum().item())
        self._total_excluded += n_excluded
        self.trainer.record_metric(EXCLUDED_METRIC, float(n_excluded), ReduceType.sum)
        self.trainer.record_metric(
            EXCLUDED_CUMULATIVE_METRIC, float(self._total_excluded), ReduceType.sum
        )

    def post_train(self):
        if self._blacklist is not None:
            log.info(
                "[untaught] run finished; %d instance-slots were excluded from the "
                "loss on this rank",
                self._total_excluded,
            )

    def _load_blacklist(self) -> Optional[Dict[int, str]]:
        if not self.enabled:
            return None

        if self.chunk_ids is not None:
            blacklist = {int(c): "" for c in self.chunk_ids}
        elif self.blacklist:
            blacklist = self._load_artifact()
        else:
            return None

        if not blacklist:
            log.warning("[untaught] blacklist is empty -- nothing will be excluded")
        return blacklist

    def _load_artifact(self) -> Dict[int, str]:
        """Read the chunk ids resolved before submission.

        Elasticsearch is not reachable from the GPU nodes, so the lookup cannot
        happen here -- the artifact is the hand-off.
        """
        path = os.path.join(str(self.trainer.save_folder), self.artifact_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[untaught] blacklist artifact not found: {path}\n"
                "Generate it where Elasticsearch is reachable, before submitting:\n"
                f"  python -m framework.train_untaught <config> --prepare"
            )

        blacklist = load_artifact(path)
        log.info("[untaught] loaded %d chunk ids from %s", len(blacklist), path)
        return blacklist
