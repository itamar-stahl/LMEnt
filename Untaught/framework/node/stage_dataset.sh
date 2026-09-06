#!/bin/sh
# Stage the dataset onto node-local disk, then point LMENT_DATASET at it.
#
# WHY. Training reads the token .npy files and the VSL batch/bucket indices
# continuously, and the shared filer behind them is unreliable. Between
# 2026-08-28 and 2026-09-02 it cost nine runs: the h200 NFS client fell to
# 0.44 MB/s and wedged jobs in uninterruptible sleep; an OSError Errno 5 killed
# both twins mid-checkpoint and truncated what they were writing; n-102 dropped
# to 5 MB/s with GPUs at 0%; and twice a DataLoader timed out after 1200 s. The
# problem follows the filesystem, not the partition.
#
# WHAT. Copy dataset-tokenized (~14 GiB of .npy) and every dataset-cache
# subtree EXCEPT dataset-metadata to node-local disk, and symlink only
# dataset-metadata back to the share. Of the 214 GiB cache, dataset-metadata is
# 212 GiB; the index subtrees the dataloader actually reads are ~589 MiB.
#
# An earlier version symlinked the WHOLE cache, on the theory that it is read
# at startup only. That was wrong -- the log showed the VSL curriculum reading
# global_batch_indices_*.npy through the symlink mid-training -- and runs 8 and
# 9 died to it with the tokenized data already local.
#
# The configs keep the ${LMENT_DATASET} string unexpanded, so redirecting the
# variable leaves config text byte-identical and resume identity intact. This
# changes where bytes are read from, never which bytes or in what order.
#
# Set LMENT_STAGE_DATASET=0 to skip staging and read from the share as before.
#
# FAIL-SAFE. A partial copy would be silently corrupted training data, far
# worse than a slow run, so .npy files are copied to .part and renamed on
# success, every one is size-verified against its source, and the staged tree
# is adopted only if all of them match. Any failure leaves LMENT_DATASET alone.
#
# EVERY BAIL-OUT SAYS WHY. A previous version returned silently, so a run that
# never staged looked identical to one that had, and a job went out with full
# NFS exposure before anyone noticed. The final state is always announced.
#
# NEVER WRITE TO THE SOURCE. A stale dataset-cache symlink left by the older
# version pointed at the share; mkdir -p on it succeeds, and copying through it
# would have written into another user's dataset directory. Write targets are
# resolved and refused unless they sit under the staging root, and stale
# symlinks are replaced rather than followed.

lment_stage_skip() {
    echo "[stage] NOT STAGED: $* -- reading from the share" >&2
    return 0
}

lment_stage_dataset() {
    _src="${LMENT_DATASET}"
    _src_tok="${_src}/dataset-tokenized"
    _dst="${LMENT_STAGE_DIR:-/tmp/lment-dataset-$(whoami)}"
    _dst_tok="${_dst}/dataset-tokenized"
    _dst_cache="${_dst}/dataset-cache"

    [ -d "${_src_tok}" ] || { lment_stage_skip "source missing: ${_src_tok}"; return 0; }
    mkdir -p "${_dst}" || { lment_stage_skip "cannot create ${_dst}"; return 0; }

    # Refuse if the staging root itself resolves somewhere shared.
    _real_dst=$(cd "${_dst}" 2>/dev/null && pwd -P)
    _real_src=$(cd "${_src}" 2>/dev/null && pwd -P)
    case "${_real_dst}" in
        /tmp/*|/scratch/*|/local/*) : ;;
        *) lment_stage_skip "staging root ${_dst} resolves to ${_real_dst:-?}, not node-local"
           return 0 ;;
    esac
    # Belt and braces: a prefix match alone would accept a staging root that
    # sits INSIDE the dataset, which is how a copy ends up in the source.
    case "${_real_dst}" in
        "${_real_src}"|"${_real_src}"/*)
            lment_stage_skip "staging root ${_real_dst} is inside the dataset ${_real_src}"
            return 0 ;;
    esac

    # A stale symlink from the older script points dataset-cache at the share.
    # Following it would copy into someone else's directory, so replace it.
    if [ -L "${_dst_cache}" ]; then
        echo "[stage] replacing stale dataset-cache symlink -> $(readlink "${_dst_cache}")"
        rm -f "${_dst_cache}" || { lment_stage_skip "cannot remove stale symlink"; return 0; }
    fi
    mkdir -p "${_dst_tok}" "${_dst_cache}" || {
        lment_stage_skip "cannot create ${_dst_tok} / ${_dst_cache}"; return 0; }

    # Every write target must resolve under the staging root, never the share.
    for _guard in "${_dst_tok}" "${_dst_cache}"; do
        _r=$(cd "${_guard}" 2>/dev/null && pwd -P)
        case "${_r}" in
            "${_real_dst}"/*) : ;;
            *) lment_stage_skip "${_guard} resolves outside the staging root (${_r:-?})"
               return 0 ;;
        esac
    done

    # ---- tokenized .npy: copy via .part, then size-verify every one --------
    _copied=0; _reused=0
    for _f in "${_src_tok}"/*.npy; do
        [ -e "${_f}" ] || continue
        _base=$(basename "${_f}")
        _want=$(stat -Lc%s "${_f}" 2>/dev/null)
        _have=$(stat -Lc%s "${_dst_tok}/${_base}" 2>/dev/null || echo -1)
        if [ "${_have}" = "${_want}" ]; then _reused=$((_reused + 1)); continue; fi
        if cp "${_f}" "${_dst_tok}/${_base}.part" 2>/dev/null &&
           mv "${_dst_tok}/${_base}.part" "${_dst_tok}/${_base}" 2>/dev/null; then
            _copied=$((_copied + 1))
        else
            rm -f "${_dst_tok}/${_base}.part"
            lment_stage_skip "copy failed for ${_base}"; return 0
        fi
    done

    _bad=0; _n=0
    for _f in "${_src_tok}"/*.npy; do
        [ -e "${_f}" ] || continue
        _n=$((_n + 1)); _base=$(basename "${_f}")
        _want=$(stat -Lc%s "${_f}" 2>/dev/null)
        _have=$(stat -Lc%s "${_dst_tok}/${_base}" 2>/dev/null || echo -1)
        [ "${_have}" = "${_want}" ] || {
            echo "[stage] size mismatch ${_base}: want ${_want} have ${_have}" >&2
            _bad=$((_bad + 1)); }
    done
    { [ "${_bad}" -eq 0 ] && [ "${_n}" -gt 0 ]; } || {
        lment_stage_skip "verification failed (${_bad} bad of ${_n})"; return 0; }

    # ---- cache: stage the indices, symlink only dataset-metadata -----------
    for _entry in "${_src}/dataset-cache"/*; do
        [ -e "${_entry}" ] || continue
        _name=$(basename "${_entry}")
        if [ "${_name}" = "dataset-metadata" ]; then
            [ -e "${_dst_cache}/${_name}" ] || \
                ln -s "${_entry}" "${_dst_cache}/${_name}" 2>/dev/null || {
                    lment_stage_skip "cannot link ${_name}"; return 0; }
            continue
        fi
        # NOT cp -a: that implies --preserve=all including ownership, and
        # these files belong to another user, so chown fails and cp returns
        # non-zero. Timestamps are all that matter here -- -u compares them.
        cp -ru --preserve=timestamps "${_entry}" "${_dst_cache}/" 2>/dev/null || {
            lment_stage_skip "cannot copy cache subtree ${_name}"; return 0; }
    done

    # If the indices are not local, the hot path is still on NFS and staging
    # has not achieved its purpose -- say so rather than half-adopting.
    if ! ls "${_dst_cache}"/dataset-*/*/global_batch_indices_*.npy >/dev/null 2>&1; then
        lment_stage_skip "no local batch indices under ${_dst_cache}"; return 0
    fi

    LMENT_DATASET="${_dst}"
    export LMENT_DATASET
    echo "[stage] tokenized: ${_n} files verified (${_copied} copied, ${_reused} reused)"
    echo "[stage] cache: indices local, dataset-metadata symlinked to the share"
    echo "[stage] LMENT_DATASET -> ${LMENT_DATASET}"
}

if [ "${LMENT_STAGE_DATASET:-1}" = "1" ]; then
    lment_stage_dataset
    # Always announce the final state: a silent no-op is how a full-NFS run
    # once passed for a staged one.
    case "${LMENT_DATASET}" in
        /tmp/*|/scratch/*|/local/*)
            echo "[stage] CONFIRMED node-local: ${LMENT_DATASET}" ;;
        *)
            echo "[stage] WARNING: reading from the share (${LMENT_DATASET})" >&2 ;;
    esac
else
    echo "[stage] disabled by LMENT_STAGE_DATASET=0 -- reading from the share"
fi
