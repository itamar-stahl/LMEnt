#!/bin/sh
# Run on the login node. By default this runs local checks, prepares CPU
# factors, submits one real H100 end-to-end smoke job, waits, and verifies it.
set -u

NO_SUBMIT=0
WAIT_MINUTES=240
JUDGE_MODEL="google/gemma-3-12b-it"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-submit) NO_SUBMIT=1 ;;
    --wait-minutes) WAIT_MINUTES="$2"; shift ;;
    --judge-model) JUDGE_MODEL="$2"; shift ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
PROJECT="${LMENT_ROOT}/Ember-on-LMEnt"
cd "${PROJECT}" || exit 1
# shellcheck disable=SC1091
. ./activate_env.sh || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
TEST_DIR="${PROJECT}/slurm_test_runs/test_${STAMP}"
JOB_ROOT="${TEST_DIR}/jobs"
OUTPUT_DIR="${TEST_DIR}/erased"
HF_HOME_DIR="${LMENT_USER_ROOT}/hf-cache"
FULL="${TEST_DIR}/full_output.log"
REPORT="${TEST_DIR}/report.log"
mkdir -p "${TEST_DIR}" "${JOB_ROOT}"
: >"${FULL}"
: >"${REPORT}"

record() {
  status="$1"
  name="$2"
  detail="$3"
  line="[RESULT] ${name} ${status}  ${detail}"
  echo "${line}" | tee -a "${FULL}" "${REPORT}"
}

run_phase() {
  name="$1"
  shift
  phase_log="${TEST_DIR}/${name}.log"
  echo "---- ${name} ----" | tee -a "${FULL}"
  if "$@" >"${phase_log}" 2>&1; then
    cat "${phase_log}" >>"${FULL}"
    record PASS "${name}" "${phase_log}"
    return 0
  else
    rc=$?
    cat "${phase_log}" >>"${FULL}"
    record FAIL "${name}" "exit=${rc}; ${phase_log}"
    return "${rc}"
  fi
}

echo "LMEnt EMBER Slurm test ${STAMP}" | tee -a "${FULL}" "${REPORT}"
echo "project=${PROJECT}" | tee -a "${FULL}" "${REPORT}"
echo "python=$(command -v python) conda=${CONDA_DEFAULT_ENV:-none}" \
  | tee -a "${FULL}" "${REPORT}"

run_phase unit_tests python -m unittest discover -s tests -p 'test_*.py' -v || true
run_phase compile python -m compileall -q ember tests slurm || true
run_phase pip_check python -m pip check || true
run_phase shell_syntax sh -n activate_env.sh slurm/submit_ember.sh \
  slurm/tests/run_test.sh example.sh || true

if grep -q '\[RESULT\].* FAIL' "${REPORT}"; then
  record SKIP h100_e2e "local checks failed; no GPU job submitted"
elif [ "${NO_SUBMIT}" -eq 1 ]; then
  record SKIP h100_e2e "--no-submit"
else
  SUBMIT_LOG="${TEST_DIR}/submit.log"
  if python -m ember.slurm_submit \
      --config configs/ember_lment_slurm.yaml \
      --concept "Culture of Greece" \
      --concept-json data/concept_sentences.json \
      --neutral-json data/neutral_sentences.json \
      --output-dir "${OUTPUT_DIR}" \
      --delta 0.5 \
      --judge-model "${JUDGE_MODEL}" \
      --alpaca-eval --alpaca-max-items 1 \
      --hf-home "${HF_HOME_DIR}" \
      --run-root "${JOB_ROOT}" \
      --job-name ember-lment-test >"${SUBMIT_LOG}" 2>&1; then
    cat "${SUBMIT_LOG}" >>"${FULL}"
    record PASS submit "${SUBMIT_LOG}"
    JOB_RUN="$(ls -1dt "${JOB_ROOT}"/* 2>/dev/null | head -1)"
    JOB_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
      "${JOB_RUN}/client_report.json")"
    record PASS job_id "${JOB_ID}; ${JOB_RUN}"

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
      record FAIL h100_wait "timeout after ${WAIT_MINUTES}m; job ${JOB_ID} was not cancelled"
    elif run_phase h100_verify python -m slurm.tests.verify_e2e_report \
        --report "${OUTPUT_DIR}/report.json" \
        --log-out "${JOB_RUN}/log.out" \
        --log-err "${JOB_RUN}/log.err"; then
      record PASS h100_e2e "job=${JOB_ID}; ${OUTPUT_DIR}/report.json"
    else
      record FAIL h100_e2e "job=${JOB_ID}; inspect ${JOB_RUN}"
    fi
  else
    rc=$?
    cat "${SUBMIT_LOG}" >>"${FULL}"
    record FAIL submit "exit=${rc}; ${SUBMIT_LOG}"
  fi
fi

echo "Full output: ${FULL}" | tee -a "${REPORT}"
echo "Report: ${REPORT}"
if grep -q '\[RESULT\].* FAIL' "${REPORT}"; then
  exit 1
fi
exit 0
