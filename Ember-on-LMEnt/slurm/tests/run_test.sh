#!/bin/sh
# Login-node verification package. Add --no-submit to skip the real H100 run.
set -u

NO_SUBMIT=0
WAIT_MINUTES=240
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-submit) NO_SUBMIT=1 ;;
    --wait-minutes) WAIT_MINUTES="$2"; shift ;;
    -h|--help) sed -n '1,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
PROJECT="${LMENT_ROOT}/Ember-on-LMEnt"
CONFIG="${PROJECT}/configs/ember_lment_slurm_test.yaml"
cd "${PROJECT}" || exit 1
. "${PROJECT}/activate_env.sh" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
TEST_DIR="${PROJECT}/slurm_test_runs/test_${STAMP}"
FULL="${TEST_DIR}/full_output.log"
REPORT="${TEST_DIR}/report.log"
mkdir -p "${TEST_DIR}"
: >"${FULL}"
: >"${REPORT}"

record() {
  line="[RESULT] $2 $1  $3"
  echo "${line}" | tee -a "${FULL}" "${REPORT}"
}

run_phase() {
  name="$1"; shift
  phase_log="${TEST_DIR}/${name}.log"
  if "$@" >"${phase_log}" 2>&1; then
    cat "${phase_log}" >>"${FULL}"
    record PASS "${name}" "${phase_log}"
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

run_phase environment python -c \
  'import os,site,torch,transformers,ember,factorization; assert os.environ.get("CONDA_DEFAULT_ENV")=="lment"; assert not site.ENABLE_USER_SITE; print(torch.__version__, transformers.__version__)' || true
run_phase unit_tests python -m unittest discover -s tests -p 'test_*.py' -v || true
run_phase compile python -m compileall -q ember tests slurm || true
run_phase pip_check python -m pip check || true
run_phase shell_syntax sh -n "${PROJECT}/activate_env.sh" \
  "${PROJECT}/slurm/submit_ember.sh" "${PROJECT}/slurm/tests/run_test.sh" \
  "${PROJECT}/slurm/submit_all_concepts.sh" \
  "${PROJECT}/example.sh" || true

if grep -q '\[RESULT\].* FAIL' "${REPORT}"; then
  record SKIP h100_e2e "local checks failed"
elif [ "${NO_SUBMIT}" -eq 1 ]; then
  record SKIP h100_e2e "--no-submit"
else
  SUBMIT_JSON="${TEST_DIR}/submit.json"
  SUBMIT_ERR="${TEST_DIR}/submit.err"
  if python -m ember.slurm_submit \
      --config "${CONFIG}" --concept "Pornography" \
      >"${SUBMIT_JSON}" 2>"${SUBMIT_ERR}"; then
    cat "${SUBMIT_JSON}" "${SUBMIT_ERR}" >>"${FULL}"
    RUN_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "${SUBMIT_JSON}")"
    JOB_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "${SUBMIT_JSON}")"
    record PASS submit "job=${JOB_ID}; run=${RUN_DIR}"

    waited=0
    limit=$((WAIT_MINUTES * 60))
    while [ "${waited}" -lt "${limit}" ]; do
      if ! squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q .; then break; fi
      sleep 20
      waited=$((waited + 20))
    done
    if [ "${waited}" -ge "${limit}" ]; then
      record FAIL h100_wait "timeout after ${WAIT_MINUTES}m; job ${JOB_ID} not cancelled"
    elif run_phase h100_verify python -m slurm.tests.verify_e2e_report \
        --report "${RUN_DIR}/outputs/report.json" \
        --log-out "${RUN_DIR}/log.out" --log-err "${RUN_DIR}/log.err"; then
      record PASS h100_e2e "job=${JOB_ID}; ${RUN_DIR}/outputs/report.json"
    else
      record FAIL h100_e2e "job=${JOB_ID}; inspect ${RUN_DIR}"
    fi
  else
    rc=$?
    cat "${SUBMIT_JSON}" "${SUBMIT_ERR}" >>"${FULL}"
    record FAIL submit "exit=${rc}; ${SUBMIT_ERR}"
  fi
fi

echo "Full output: ${FULL}" | tee -a "${REPORT}"
echo "Report: ${REPORT}"
if grep -q '\[RESULT\].* FAIL' "${REPORT}"; then exit 1; fi
exit 0
