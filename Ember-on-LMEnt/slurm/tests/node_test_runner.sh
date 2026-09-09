#!/bin/sh
# Runs only CUDA-dependent tests inside the allocated Titan XP job.
set -u

PROJECT=""
CONFIG=""
TEST_DIR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --project) PROJECT="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --test-dir) TEST_DIR="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "${PROJECT}" ] || [ -z "${CONFIG}" ] || [ -z "${TEST_DIR}" ]; then
  echo "Required: --project, --config, and --test-dir" >&2
  exit 2
fi

FULL="${TEST_DIR}/full_output.log"
REPORT="${TEST_DIR}/report.log"
RESULT="${TEST_DIR}/test_result.json"
FAILED=0
GPU_REAL_RUN=""
touch "${FULL}" "${REPORT}"

record() {
  phase="$1"
  status="$2"
  detail="$3"
  line="[RESULT] ${status} ${phase}  ${detail}"
  echo "${line}" | tee -a "${FULL}" "${REPORT}"
}

run_phase() {
  name="$1"
  shift
  phase_log="${TEST_DIR}/${name}.log"
  echo "[PHASE] ${name}" | tee -a "${FULL}"
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

echo "LMEnt EMBER Slurm GPU tests" | tee -a "${FULL}" "${REPORT}"
echo "project=${PROJECT}" | tee -a "${FULL}" "${REPORT}"
echo "config=${CONFIG}" | tee -a "${FULL}" "${REPORT}"
echo "job=${SLURM_JOB_ID:-not-set}" | tee -a "${FULL}" "${REPORT}"
cd "${PROJECT}" || exit 1

run_phase config python -c \
  'import sys; from pathlib import Path; from ember.lment_pipeline import load_lment_config; from ember.slurm_model import validate_slurm_model_config; p=validate_slurm_model_config(Path(sys.argv[1])); c=load_lment_config(Path(sys.argv[1])); assert c.model_path==p; assert c.slurm["partition"]=="studentkillable"; assert c.slurm["constraint"]=="titan_xp"; print(p)' \
  "${CONFIG}" || true

if [ -f "${TEST_DIR}/config.log" ]; then
  MODEL_PATH="$(tail -n 1 "${TEST_DIR}/config.log")"
  if [ -f "${MODEL_PATH}/config.json" ]; then
    EMBER_LMENT_MODEL_PATH="${MODEL_PATH}"
    export EMBER_LMENT_MODEL_PATH
  fi
fi

run_phase gpu_environment python -c \
  'import json,os,site,torch,transformers,ember,factorization; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""; assert os.environ.get("CONDA_DEFAULT_ENV")=="lment", os.environ.get("CONDA_DEFAULT_ENV"); assert not site.ENABLE_USER_SITE; assert torch.cuda.is_available(); assert "titan xp" in gpu.lower(), gpu; print(json.dumps({"conda_env":os.environ.get("CONDA_DEFAULT_ENV"),"torch":torch.__version__,"transformers":transformers.__version__,"gpu":gpu}))' || true
run_phase gpu_checkpoint python -m unittest tests.test_lment_gpu -v || true

if [ "${FAILED}" -ne 0 ]; then
  record gpu_real_flow SKIP "earlier test phase failed"
else
  run_phase gpu_real_flow python -m ember.real_flow_test \
    --config "${CONFIG}" --concept Pornography --execution slurm || true
  if [ "${FAILED}" -eq 0 ]; then
    GPU_REAL_RUN="$(sed -n 's/^\[real-run\] //p' \
      "${TEST_DIR}/gpu_real_flow.log" | tail -n 1)"
    if [ -z "${GPU_REAL_RUN}" ] || \
        [ ! -f "${GPU_REAL_RUN}/outputs/real_flow_test_report.json" ]; then
      record gpu_real_evidence FAIL \
        "could not locate retained evidence from ${TEST_DIR}/gpu_real_flow.log"
      FAILED=1
    else
      run_phase gpu_real_evidence python -c \
        'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"outputs"/"real_flow_test_report.json"; d=json.loads(p.read_text()); assert d["passed"] is True; assert d["actual_model_device"].startswith("cuda"); assert Path(d["erased_embedding"]).is_file(); print(p)' \
        "${GPU_REAL_RUN}" || true
    fi
  fi
fi

if [ "${FAILED}" -eq 0 ]; then STATUS=PASS; EXIT_CODE=0; else STATUS=FAIL; EXIT_CODE=1; fi
python -c \
  'import json,sys; from pathlib import Path; p={"schema_version":1,"status":sys.argv[1],"exit_code":int(sys.argv[2]),"slurm_job_id":sys.argv[3] or None,"report":sys.argv[4],"full_output":sys.argv[5],"gpu_real_run":sys.argv[6] or None}; Path(sys.argv[7]).write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")' \
  "${STATUS}" "${EXIT_CODE}" "${SLURM_JOB_ID:-}" "${REPORT}" "${FULL}" "${GPU_REAL_RUN}" "${RESULT}"

echo "Test status: ${STATUS}" | tee -a "${FULL}" "${REPORT}"
echo "Report: ${REPORT}" | tee -a "${FULL}"
echo "Full output: ${FULL}" | tee -a "${REPORT}"
exit "${EXIT_CODE}"
