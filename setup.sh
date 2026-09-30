#!/usr/bin/env bash
# Prepare a downloaded LMEnt-Dataset for use: decompress the per-chunk metadata
# CSVs, then give the content-hashed .npy index files the part-N names that the
# dataloader looks for.
#
# Usage: bash setup.sh <absolute path to LMEnt-Dataset directory>
#
# Safe to re-run: the symlinks are replaced rather than added to.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <absolute path to LMEnt-Dataset directory>" >&2
  exit 2
fi

LMENT_DATASET_PATH="$1"

if [ ! -d "${LMENT_DATASET_PATH}" ]; then
  echo "error: not a directory: ${LMENT_DATASET_PATH}" >&2
  exit 2
fi

LMENT_DATASET_TOKENIZED_PATH="${LMENT_DATASET_PATH}/dataset-tokenized"
LMENT_DATASET_INDEX_METADATA_PATH="${LMENT_DATASET_PATH}/dataset-cache/dataset-metadata"
LMENT_DATASET_COMMON_PATH="${LMENT_DATASET_PATH}/dataset-cache/dataset-common"

for d in "${LMENT_DATASET_TOKENIZED_PATH}" \
         "${LMENT_DATASET_INDEX_METADATA_PATH}" \
         "${LMENT_DATASET_COMMON_PATH}"; do
  if [ ! -d "$d" ]; then
    echo "error: expected directory is missing: $d" >&2
    echo "       is ${LMENT_DATASET_PATH} really an unpacked LMEnt-Dataset?" >&2
    exit 2
  fi
done

echo "Decompressing .csv.gz from ${LMENT_DATASET_TOKENIZED_PATH} directly into ${LMENT_DATASET_INDEX_METADATA_PATH}"
find "${LMENT_DATASET_TOKENIZED_PATH}" -type f -name "*.csv.gz" -print0 | while IFS= read -r -d '' gzfile; do
  base="$(basename "${gzfile}" .gz)"   # foo.csv
  out_csv="${LMENT_DATASET_INDEX_METADATA_PATH}/${base}"

  echo "  $(basename "${gzfile}") -> ${out_csv}"
  gzip -dc "${gzfile}" > "${out_csv}"
done

# Content hash of each metadata shard, in part order (part-0 .. part-7).
METADATA_HASHES=(
  7c79c698e8357904526b9232939cc8fb2f323c626a92e3f4f6034dd1ff718f7f
  2cae60cd5233288004907adb629fb44341f3cb483e3f1b9def48102d64403577
  0578e6dda5d3affd9e3114d87647af6a1d22b8719929f5ef1107571215fd12c5
  8ed59ee23677c91c2665cfe2e3aed0b27f42d9fede0beb1d40f19824f3b280fd
  79179b22717f0053cef1fdfcf5d8333d72a1f6c5eb47abc777a5e9501662451f
  0be78896e35baf663f09264760c6b2f067d336b9f6bf066c3ada456e05b95f63
  640877e4c5a103035406cfab626e5e74bf59081fe67475ceb83d41c20bdc7971
  b534ac0080a01bd2c6f680e51cf2282033fde5df2a745b7ac38e93de3fa0ee67
)

# Content hash of each bucketed-doc-indices shard, in part order (part-0 .. part-7).
BUCKETED_HASHES=(
  199a692bb41320d5fff4240b51c45b28e75f5eb5be8ad169aa4425a8279403a0
  d6f57a7aff71eaccad01fbe418cfae79b8c830f9de2e479608c39a0d06f620a7
  291e4ea3eec9adfa2d9f1f9f39f09b234451dfd0bc7e8f887fa66ccc986a10cb
  e59a12800fb9de9f6ba5229dd645b2cebf8525b72a32df3d285a29dab541ac4b
  67e5915df159a996c0215a837edd4b701156a2debb44a5e95fad27e7c823d181
  84881decceee71c9324553ac2a9cdc854f6ff25f6cb6c20fac754fdcfa97c2bd
  a9fb4580b983c919577350a32f37f07fbcb549299c9e5f4b3825c2244b68e6f9
  2850148368e9fa80cd8cc4aee1382678faebaa6015d0ef9798cf249ea08dd7a9
)

link_shards() {
  local dir="$1" prefix="$2"
  shift 2
  local hashes=("$@")
  local part=0 src dst
  for h in "${hashes[@]}"; do
    src="${dir}/${prefix}-${h}.npy"
    dst="${dir}/${prefix}-part-${part}-00000.npy"
    if [ ! -e "${src}" ]; then
      echo "error: missing shard: ${src}" >&2
      exit 1
    fi
    ln -sfn "${src}" "${dst}"
    echo "  ${prefix}-part-${part}-00000.npy -> $(basename "${src}")"
    part=$((part + 1))
  done
}

echo "Linking Metadata Index Paths in ${LMENT_DATASET_INDEX_METADATA_PATH}"
link_shards "${LMENT_DATASET_INDEX_METADATA_PATH}" metadata "${METADATA_HASHES[@]}"

echo "Linking Bucketed Doc Indices Paths in ${LMENT_DATASET_COMMON_PATH}"
link_shards "${LMENT_DATASET_COMMON_PATH}" bucketed-doc-indices "${BUCKETED_HASHES[@]}"

echo "Done."
