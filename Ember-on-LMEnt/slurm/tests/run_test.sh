#!/bin/sh
# One login-node command: prepare, submit, wait, and report the Titan XP suite.
set -u

WAIT_MINUTES=240
: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
PROJECT="${LMENT_ROOT}/Ember-on-LMEnt"
CPU_CONFIG="${PROJECT}/configs/ember_lment_real_slurm_cpu.yaml"
GPU_CONFIG="${PROJECT}/configs/ember_lment_real_slurm_gpu.yaml"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wait-minutes) WAIT_MINUTES="$2"; shift ;;
    --cpu-config) CPU_CONFIG="$2"; shift ;;
    --gpu-config|--config) GPU_CONFIG="$2"; shift ;;
    -h|--help)
      echo "Usage: sh $0 [--wait-minutes N] [--cpu-config /absolute/cpu.yaml] [--gpu-config /absolute/gpu.yaml]"
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
for test_config in "${CPU_CONFIG}" "${GPU_CONFIG}"; do
  if [ ! -f "${test_config}" ]; then
    echo "Test YAML not found: ${test_config}" >&2
    exit 1
  fi
done

cd "${PROJECT}" || exit 1
if ! . "${PROJECT}/activate_env.sh"; then
  echo "FAIL: could not activate the root lment Conda environment" >&2
  exit 1
fi

PREPARE_ERR="$(mktemp)"
if ! PREPARED="$(python -m slurm.tests.prepare_test_job \
    --project "${PROJECT}" --config "${GPU_CONFIG}" 2>"${PREPARE_ERR}")"; then
  echo "FAIL: could not prepare the Slurm test job" >&2
  cat "${PREPARE_ERR}" >&2
  rm -f "${PREPARE_ERR}"
  exit 1
fi
rm -f "${PREPARE_ERR}"
TEST_DIR="$(printf '%s' "${PREPARED}" | python -c 'import json,sys; print(json.load(sys.stdin)["test_dir"])')"
JOB_FILE="$(printf '%s' "${PREPARED}" | python -c 'import json,sys; print(json.load(sys.stdin)["job_file"])')"
FULL="${TEST_DIR}/full_output.log"
REPORT="${TEST_DIR}/report.log"
FAILED=0
CPU_REAL_RUN=""
: >"${FULL}"
: >"${REPORT}"
cp "${CPU_CONFIG}" "${TEST_DIR}/login_cpu_config.yaml"

record() {
  phase="$1"
  status="$2"
  detail="$3"
  line="[RESULT] ${status} ${phase}  ${detail}"
  echo "${line}" | tee -a "${FULL}" "${REPORT}"
}

run_login_phase() {
  name="$1"
  shift
  phase_log="${TEST_DIR}/${name}.log"
  echo "[LOGIN PHASE] ${name}" | tee -a "${FULL}"
  if "$@" >"${phase_log}" 2>&1; then
    cat "${phase_log}" | tee -a "${FULL}"
    record "${name}" PASS "${phase_log}"
    return 0
  else
    rc=$?
    cat "${phase_log}" | tee -a "${FULL}"
    record "${name}" FAIL "exit=${rc}; ${phase_log}"
    FAILED=1
    return "${rc}"
  fi
}

echo "Prepared test run: ${TEST_DIR}"
echo "Running Linux login-node tests before any Slurm submission."
run_login_phase login_environment python -c \
  'import json,os,site,torch,transformers,ember,factorization; assert os.environ.get("CONDA_DEFAULT_ENV")=="lment", os.environ.get("CONDA_DEFAULT_ENV"); assert not site.ENABLE_USER_SITE; print(json.dumps({"conda_env":os.environ.get("CONDA_DEFAULT_ENV"),"torch":torch.__version__,"transformers":transformers.__version__,"cuda_available":torch.cuda.is_available()}))' || true
run_login_phase unit_tests python -m unittest discover -s tests -p 'test_*.py' -v || true
run_login_phase compile python -m compileall -q ember tests slurm || true
run_login_phase pip_check python -m pip check || true
run_login_phase shell_syntax sh -c \
  'cd "$1" && git ls-files "*.sh" | while IFS= read -r file; do sh -n "$file" || exit 1; done' \
  shell-check "${PROJECT}" || true
run_login_phase cluster_model_cpu python -m ember.slurm_model \
  --config "${CPU_CONFIG}" || true
run_login_phase cluster_model_gpu python -m ember.slurm_model \
  --config "${GPU_CONFIG}" || true

if [ "${FAILED}" -eq 0 ]; then
  run_login_phase cpu_real_flow python -m ember.real_flow_test \
    --config "${CPU_CONFIG}" --concept Pornography --execution local || true
  CPU_REAL_RUN="$(sed -n 's/^\[real-run\] //p' \
    "${TEST_DIR}/cpu_real_flow.log" | tail -n 1)"
  if [ -z "${CPU_REAL_RUN}" ] || \
      [ ! -f "${CPU_REAL_RUN}/outputs/real_flow_test_report.json" ]; then
    record cpu_real_evidence FAIL \
      "could not locate retained evidence from ${TEST_DIR}/cpu_real_flow.log"
    FAILED=1
  else
    printf '%s\n' "${CPU_REAL_RUN}" >"${TEST_DIR}/cpu_real_run.txt"
    record cpu_real_evidence PASS \
      "${CPU_REAL_RUN}/outputs/real_flow_test_report.json"
  fi
else
  record cpu_real_flow SKIP "fast login-node tests failed"
fi

if [ "${FAILED}" -ne 0 ]; then
  record gpu_submission SKIP "login-node tests failed"
  python -c \
    'import json,sys; from pathlib import Path; Path(sys.argv[3]).write_text(json.dumps({"schema_version":1,"status":"FAIL","stage":"login","exit_code":1,"report":sys.argv[1],"cpu_real_run":sys.argv[2] or None},indent=2)+"\n",encoding="utf-8")' \
    "${REPORT}" "${CPU_REAL_RUN}" "${TEST_DIR}/test_result.json"
  echo "FAIL: login-node tests failed; no Slurm job was submitted." >&2
  echo "Summary: ${REPORT}" >&2
  echo "Full output: ${FULL}" >&2
  exit 1
fi

record login_suite PASS "all login-node checks passed"
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
  echo "CPU real run: ${CPU_REAL_RUN}"
  echo "GPU real run: $(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("gpu_real_run") or "missing")' "${TEST_DIR}/test_result.json")"
  echo "Test package: ${TEST_DIR}"
  exit 0
fi

echo "FAIL: one or more Slurm test phases failed." >&2
echo "Summary: ${TEST_DIR}/report.log" >&2
echo "Full output: ${TEST_DIR}/full_output.log" >&2
echo "Slurm stderr: ${TEST_DIR}/log.err" >&2
exit 1
