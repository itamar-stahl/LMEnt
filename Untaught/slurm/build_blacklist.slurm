#!/bin/sh
#SBATCH --job-name=untaught-blacklist
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000

# Querying Elasticsearch needs no GPU. Submit to a CPU partition, e.g.
#   sbatch --partition=cpu-killable --time=60 slurm/build_blacklist.slurm
# For a single entity this takes seconds, so running it directly on the client
# node is usually fine too -- see run_smoke_harry_potter.sh.
#
# Expected environment:
#   UNTAUGHT_ROOT, UNTAUGHT_ENTITIES, UNTAUGHT_BLACKLIST, CONDA_ENV
#   ES_HOST / ES_PORT / ES_INDEX / ES_PASSWORD

set -eu

echo " entities  : ${UNTAUGHT_ENTITIES}"
echo " output    : ${UNTAUGHT_BLACKLIST}"
echo " es index  : ${ES_INDEX:-lment_cs} @ ${ES_HOST:-localhost}:${ES_PORT:-9200}"

if command -v conda >/dev/null 2>&1; then
  # conda's activation scripts trip `set -u` on many versions.
  set +u
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi

cd "${UNTAUGHT_ROOT}"
export PYTHONPATH="${UNTAUGHT_ROOT}:${PYTHONPATH:-}"

python -m untaught.es_blacklist build \
  --entities "${UNTAUGHT_ENTITIES}" \
  --out "${UNTAUGHT_BLACKLIST}" \
  --preview "${UNTAUGHT_PREVIEW:-3}"
