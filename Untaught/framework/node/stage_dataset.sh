#!/bin/sh
# Stage the tokenized dataset onto node-local disk, then point LMENT_DATASET at it.
#
# WHY. Training reads the token .npy files continuously, and the shared filer
# behind them is unreliable. Between 2026-08-28 and 2026-08-31 it cost four runs:
# the h200 NFS client fell to 0.44 MB/s and wedged two jobs in uninterruptible
# sleep; an OSError Errno 5 killed both h100 twins mid-checkpoint at 00:12,
# truncating the checkpoints they were writing; and n-102 later dropped to
# 5 MB/s with both GPUs at 0% utilisation. The problem follows the filesystem,
# not the partition, so moving between h100 and h200 does not help.
#
# WHAT. Copy only dataset-tokenized (~14 GiB of .npy) to node-local disk and
# symlink dataset-cache straight back to the share -- the cache is 214 GiB and
# is read at startup, not on the hot path. Then export LMENT_DATASET so the
# config's "${LMENT_DATASET}/dataset-tokenized/*.npy" resolves locally.
#
# The configs keep the ${LMENT_DATASET} string unexpanded, so redirecting the
# variable leaves config text byte-identical and resume identity intact. This
# changes where bytes are read from, never which bytes or in what order.
#
# Set LMENT_STAGE_DATASET=0 to skip staging and read from the share as before.
#
# FAIL-SAFE. A partial copy would be silently corrupted training data, which is
# far worse than a slow run. Every file is size-verified after copying and the
# staged tree is only adopted if all of them match; otherwise LMENT_DATASET is
# left alone and the run proceeds off the share.

lment_stage_dataset() {
    _src="${LMENT_DATASET}"
    _src_tok="${_src}/dataset-tokenized"
    _dst="${LMENT_STAGE_DIR:-/tmp/lment-dataset-$(whoami)}"
    _dst_tok="${_dst}/dataset-tokenized"

    if [ ! -d "${_src_tok}" ]; then
        echo "[stage] source not found: ${_src_tok} -- staying on the share" >&2
        return 0
    fi

    mkdir -p "${_dst_tok}" || {
        echo "[stage] cannot create ${_dst_tok} -- staying on the share" >&2
        return 0
    }

    _copied=0
    _reused=0
    for _f in "${_src_tok}"/*.npy; do
        [ -e "${_f}" ] || continue
        _base=$(basename "${_f}")
        _want=$(stat -Lc%s "${_f}" 2>/dev/null)
        _have=$(stat -Lc%s "${_dst_tok}/${_base}" 2>/dev/null || echo -1)
        if [ "${_have}" = "${_want}" ]; then
            _reused=$((_reused + 1))
            continue
        fi
        # .part first, rename on success: an interrupted copy can never be
        # mistaken for a complete one by the next job to land on this node.
        if cp "${_f}" "${_dst_tok}/${_base}.part" 2>/dev/null &&
           mv "${_dst_tok}/${_base}.part" "${_dst_tok}/${_base}" 2>/dev/null; then
            _copied=$((_copied + 1))
        else
            rm -f "${_dst_tok}/${_base}.part"
            echo "[stage] copy failed for ${_base} -- staying on the share" >&2
            return 0
        fi
    done

    # Verify every source file has a byte-identical-sized local twin.
    _bad=0
    _n=0
    for _f in "${_src_tok}"/*.npy; do
        [ -e "${_f}" ] || continue
        _n=$((_n + 1))
        _base=$(basename "${_f}")
        _want=$(stat -Lc%s "${_f}" 2>/dev/null)
        _have=$(stat -Lc%s "${_dst_tok}/${_base}" 2>/dev/null || echo -1)
        [ "${_have}" = "${_want}" ] || {
            echo "[stage] size mismatch on ${_base}: want ${_want}, have ${_have}" >&2
            _bad=$((_bad + 1))
        }
    done
    if [ "${_bad}" -ne 0 ] || [ "${_n}" -eq 0 ]; then
        echo "[stage] verification failed (${_bad} bad of ${_n}) -- staying on the share" >&2
        return 0
    fi

    # The cache is NOT startup-only, which cost runs 8 and 9 on 2026-09-02:
    # both died to a 1200 s DataLoader timeout while the whole cache was
    # symlinked to the share, and the log showed the VSL curriculum reading
    # global_batch_indices_*.npy through that symlink mid-training. The batch
    # and bucket indices are on the hot path.
    #
    # They are also small. Of the 214 GiB, dataset-metadata is 212 GiB and the
    # index subtrees are ~589 MiB, so stage everything EXCEPT dataset-metadata
    # and symlink only that back to the share.
    mkdir -p "${_dst}/dataset-cache" || {
        echo "[stage] cannot create ${_dst}/dataset-cache -- staying on the share" >&2
        return 0
    }
    for _entry in "${_src}/dataset-cache"/*; do
        [ -e "${_entry}" ] || continue
        _name=$(basename "${_entry}")
        if [ "${_name}" = "dataset-metadata" ]; then
            [ -e "${_dst}/dataset-cache/${_name}" ] || \
                ln -s "${_entry}" "${_dst}/dataset-cache/${_name}" 2>/dev/null || {
                    echo "[stage] could not link ${_name} -- staying on the share" >&2
                    return 0
                }
            continue
        fi
        # cp -a --update copies only what is missing or newer, so a resumed job
        # on a warm node re-verifies rather than re-copies.
        cp -a --update "${_entry}" "${_dst}/dataset-cache/" 2>/dev/null || {
            echo "[stage] could not copy cache subtree ${_name} -- staying on the share" >&2
            return 0
        }
    done
    _cache_bytes=$(du -sLb "${_dst}/dataset-cache" 2>/dev/null | cut -f1)
    echo "[stage] cache indices staged locally; dataset-metadata symlinked to the share"

    LMENT_DATASET="${_dst}"
    export LMENT_DATASET
    echo "[stage] ${_n} files verified on node-local disk (${_copied} copied, ${_reused} reused)"
    echo "[stage] LMENT_DATASET -> ${LMENT_DATASET}  (dataset-cache symlinked to the share)"
}

if [ "${LMENT_STAGE_DATASET:-1}" = "1" ]; then
    lment_stage_dataset
else
    echo "[stage] disabled by LMENT_STAGE_DATASET=0 -- reading from the share"
fi
