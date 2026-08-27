#!/bin/sh
# One login-node command: prepare, submit, wait, and report the Titan XP suite.
set -u

WAIT_MINUTES=240
: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
PROJECT="${LMENT_ROOT}/Ember-on-LMEnt"
CONFIG="${PROJECT}/configs/ember_lment_slurm_test.yaml"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wait-minutes) WAIT_MINUTES="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    -h|--help)
      echo "Usage: sh $0 [--wait-minutes N] [--config /absolute/test.yaml]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "${WAIT_MINUTES}" in
  ''|*[!0-9]*) echo "--wait-minutes must be a positive integer" >&2; exit 2 ;;
esac
if [ "${WAIT_MINUTES}" -le 0 ]; then
  echo "--wait-minutes must be a positive integer" >&2
  exit 2
fi
if [ ! -f "${CONFIG}" ]; then
  echo "Test YAML not found: ${CONFIG}" >&2
  exit 1
fi

cd "${PROJECT}" || exit 1
if ! . "${PROJECT}/activate_env.sh"; then
  echo "FAIL: could not activate the root lment Conda environment" >&2
  exit 1
fi

PREPARE_ERR="$(mktemp)"
if ! PREPARED="$(python -m slurm.tests.prepare_test_job \
    --project "${PROJECT}" --config "${CONFIG}" 2>"${PREPARE_ERR}")"; then
  echo "FAIL: could not prepare the Slurm test job" >&2
  cat "${PREPARE_ERR}" >&2
  rm -f "${PREPARE_ERR}"
  exit 1
fi
rm -f "${PREPARE_ERR}"
TEST_DIR="$(printf '%s' "${PREPARED}" | python -c 'import json,sys; print(json.load(sys.stdin)["test_dir"])')"
JOB_FILE="$(printf '%s' "${PREPARED}" | python -c 'import json,sys; print(json.load(sys.stdin)["job_file"])')"

echo "Prepared test run: ${TEST_DIR}"
echo "Submitting: ${JOB_FILE}"
if ! SUBMIT_OUTPUT="$(sbatch --parsable "${JOB_FILE}" 2>&1)"; then
  echo "FAIL: sbatch rejected the test job" >&2
  echo "${SUBMIT_OUTPUT}" >&2
  echo "Prepared files remain at: ${TEST_DIR}" >&2
  exit 1
fi
JOB_ID="${SUBMIT_OUTPUT%%;*}"
printf '%s\n' "${JOB_ID}" >"${TEST_DIR}/job_id.txt"
echo "Submitted job ${JOB_ID}; waiting up to ${WAIT_MINUTES} minutes"

waited=0
limit=$((WAIT_MINUTES * 60))
while [ "${waited}" -lt "${limit}" ]; do
  if ! squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q .; then
    break
  fi
  sleep 20
  waited=$((waited + 20))
done
if [ "${waited}" -ge "${limit}" ]; then
  echo "FAIL: job ${JOB_ID} still runs after ${WAIT_MINUTES} minutes" >&2
  echo "The script did not cancel it. Inspect: ${TEST_DIR}" >&2
  exit 1
fi

result_wait=0
while [ ! -f "${TEST_DIR}/test_result.json" ] && [ "${result_wait}" -lt 30 ]; do
  sleep 2
  result_wait=$((result_wait + 2))
done
if [ ! -f "${TEST_DIR}/test_result.json" ]; then
  echo "FAIL: job ${JOB_ID} ended without test_result.json" >&2
  echo "Slurm state: $(sacct -j "${JOB_ID}" --format=State,ExitCode -n -P 2>/dev/null || true)" >&2
  echo "stdout: ${TEST_DIR}/log.out" >&2
  echo "stderr: ${TEST_DIR}/log.err" >&2
  [ -f "${TEST_DIR}/log.err" ] && tail -n 80 "${TEST_DIR}/log.err" >&2
  exit 1
fi

cat "${TEST_DIR}/report.log"
if python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["status"]=="PASS" else 1)' \
    "${TEST_DIR}/test_result.json"; then
  echo "PASS: all local tests and the real Titan XP erasure passed."
  echo "Test package: ${TEST_DIR}"
  exit 0
fi

echo "FAIL: one or more Slurm test phases failed." >&2
echo "Summary: ${TEST_DIR}/report.log" >&2
echo "Full output: ${TEST_DIR}/full_output.log" >&2
echo "Slurm stderr: ${TEST_DIR}/log.err" >&2
exit 1
