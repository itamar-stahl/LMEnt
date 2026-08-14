#!/bin/sh
# THE remote test entry point. One command, no other shell work required:
#
#     sh tests/remote/run_remote_tests.sh
#
# It sets up the login-node environment itself, runs every phase, optionally
# submits BOTH smoke jobs (control and "Harry Potter" ablated) and waits for
# them, and writes a single report you can hand back:
#
#     runs/remote_test_<date>_<time>/report.log     <- submit this one
#     runs/remote_test_<date>_<time>/full_output.log
#
# Options (all optional):
#   --quick          skip the slow phases (dataset build, chunk alignment)
#   --no-submit      run every check but do not submit real SLURM jobs
#   --wait N         how many minutes to wait for each job to start (default 20)
#   --sample N       chunks to sample in the alignment check (default 10)
#
# Exit code is 0 only if every phase passed.

set -u

# --- options -----------------------------------------------------------------
QUICK=0
SUBMIT=1
WAIT_MINUTES=20
SAMPLE=10
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)     QUICK=1 ;;
    --no-submit) SUBMIT=0 ;;
    --wait)      WAIT_MINUTES="$2"; shift ;;
    --sample)    SAMPLE="$2"; shift ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- environment (the same one a real submission uses) ------------------------
: "${LMENT_ROOT:=/home/morg/NLP_2526b/stahli/LMEnt}"
cd "${LMENT_ROOT}/Untaught" 2>/dev/null || {
  echo "FATAL: cannot cd to ${LMENT_ROOT}/Untaught" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
# activate_env.sh defines UNTAUGHT_RUNS_DIR; until then, derive it the same way.
OUT_DIR="$(pwd)/runs/remote_test_${STAMP}"
mkdir -p "${OUT_DIR}"
FULL="${OUT_DIR}/full_output.log"
REPORT="${OUT_DIR}/report.log"

echo "Untaught remote test run ${STAMP}"
echo "Full output : ${FULL}"
echo "Report      : ${REPORT}"
echo

# Everything from here is teed into the full log.
{
  echo "================================================================"
  echo " Untaught remote test suite"
  echo " started : $(date)"
  echo " host    : $(hostname)"
  echo " user    : $(whoami)"
  echo " options : quick=${QUICK} submit=${SUBMIT} wait=${WAIT_MINUTES}m sample=${SAMPLE}"
  echo "================================================================"

  export UNTAUGHT_TEST_LOGDIR="${OUT_DIR}"

  echo
  echo "---- setting up the login-node environment ----------------------"
  # shellcheck disable=SC1091
  . ./activate_env.sh
  echo "environment ready: python=$(command -v python)"

  PHASES="env local data es identity prepare"
  [ "${QUICK}" -eq 1 ] && PHASES="env local es prepare"

  for phase in ${PHASES}; do
    echo
    echo "---- phase: ${phase} --------------------------------------------"
    if [ "${phase}" = "identity" ]; then
      python tests/remote/remote_checks.py identity "${SAMPLE}"
    else
      python tests/remote/remote_checks.py "${phase}"
    fi
    echo "[PHASE_RC] ${phase} $?"
  done

  # --- optionally submit real jobs and watch them -----------------------------
  # Both configs: the control proves the plumbing, the ablated one proves the
  # blacklist actually reaches the trainer and masks chunks. Only the second
  # exercises Elasticsearch -> artifact -> exclusion end to end.
  if [ "${SUBMIT}" -eq 1 ]; then
    CONTROL_PAIR="configs/train_170m_control.yaml:untaught-control-170m"
    ABLATED_PAIR="configs/train_170m_no_harry_potter.yaml:untaught-no-hp-170m"
    for pair in "${CONTROL_PAIR}" "${ABLATED_PAIR}"; do
      cfg="${pair%%:*}"
      prefix="${pair##*:}"

      echo
      echo "---- phase: submit ${prefix} ------------------------------------"
      sh ./framework/client/sub_builder.sh "${cfg}"
      echo "[PHASE_RC] submit_${prefix} $?"

      RUN_DIR="$(ls -1dt runs/${prefix}_* 2>/dev/null | head -1)"
      if [ -z "${RUN_DIR}" ]; then
        echo "[RESULT] submit.${prefix}_run_folder FAIL  no run folder appeared"
        continue
      fi
      echo "[RESULT] submit.${prefix}_run_folder PASS  ${RUN_DIR}"

      echo
      echo "---- waiting up to ${WAIT_MINUTES}m for ${prefix} to start ------"
      waited=0
      limit=$((WAIT_MINUTES * 60))
      while [ "${waited}" -lt "${limit}" ]; do
        if [ -s "${RUN_DIR}/log.out" ] && grep -q "UNTAUGHT RUN" "${RUN_DIR}/log.out" 2>/dev/null; then
          echo "trainer started after ${waited}s"
          # give it a few steps so the exclusion metrics have something to say
          sleep 60
          break
        fi
        sleep 20
        waited=$((waited + 20))
      done
      [ "${waited}" -ge "${limit}" ] && echo "still waiting after ${WAIT_MINUTES}m (queue is busy)"

      echo
      echo "---- phase: submitted ${prefix} ---------------------------------"
      python tests/remote/remote_checks.py submitted "${RUN_DIR}"
      echo "[PHASE_RC] submitted_${prefix} $?"

      echo
      echo "---- tail of ${RUN_DIR}/log.out ---------------------------------"
      tail -30 "${RUN_DIR}/log.out" 2>/dev/null || echo "(no log.out yet)"
      echo
      echo "---- ${RUN_DIR}/log.err (should hold problems only) -------------"
      if [ -s "${RUN_DIR}/log.err" ]; then
        tail -30 "${RUN_DIR}/log.err"
      else
        echo "(empty -- good: nothing went wrong)"
      fi
    done

    echo
    echo "---- squeue ------------------------------------------------------"
    squeue --me 2>&1 | head -20
  else
    echo
    echo "---- submit phases skipped (--no-submit) ------------------------"
  fi

  echo
  echo " finished : $(date)"
} 2>&1 | tee "${FULL}"

# --- build the report ---------------------------------------------------------
python - "${FULL}" "${REPORT}" <<'PYEOF'
import re
import sys

full, report = sys.argv[1], sys.argv[2]
with open(full, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.read().splitlines()

results, phases, context = [], [], {}
for line in lines:
    m = re.match(r"\[RESULT\] (\S+) (PASS|FAIL|SKIP)\s*(.*)", line)
    if m:
        results.append((m.group(1), m.group(2), m.group(3).strip()))
    m = re.match(r"\[PHASE\] (\S+) finished in (\S+): (.*)", line)
    if m:
        phases.append((m.group(1), m.group(2), m.group(3)))
    for key, pattern in (("host", r"^ host    : (.*)"),
                         ("user", r"^ user    : (.*)"),
                         ("started", r"^ started : (.*)"),
                         ("options", r"^ options : (.*)")):
        m2 = re.match(pattern, line)
        if m2:
            context[key] = m2.group(1).strip()

failed = [r for r in results if r[1] == "FAIL"]
skipped = [r for r in results if r[1] == "SKIP"]
passed = [r for r in results if r[1] == "PASS"]

out = []
w = 72
out.append("=" * w)
out.append("  UNTAUGHT -- REMOTE TEST REPORT")
out.append("=" * w)
for key in ("started", "host", "user", "options"):
    if key in context:
        out.append(f"  {key:<8}: {context[key]}")
out.append("")
out.append(f"  VERDICT : {'PASS' if not failed else 'FAIL'}"
           f"   ({len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped)")
out.append("=" * w)

if failed:
    out.append("")
    out.append("  FAILURES -- these need attention")
    out.append("  " + "-" * (w - 4))
    for name, _s, detail in failed:
        out.append(f"  FAIL  {name}")
        for chunk in (detail[i:i + 66] for i in range(0, len(detail), 66)):
            out.append(f"          {chunk}")

if skipped:
    out.append("")
    out.append("  SKIPPED -- not run (usually: the job had not started yet)")
    out.append("  " + "-" * (w - 4))
    for name, _s, detail in skipped:
        out.append(f"  SKIP  {name}: {detail[:60]}")

out.append("")
out.append("  ALL CHECKS")
out.append("  " + "-" * (w - 4))
for name, status, detail in results:
    out.append(f"  {status:<4}  {name}")
    if detail:
        for chunk in (detail[i:i + 66] for i in range(0, len(detail), 66)):
            out.append(f"          {chunk}")

if phases:
    out.append("")
    out.append("  PHASE TIMING")
    out.append("  " + "-" * (w - 4))
    for name, secs, summary in phases:
        out.append(f"  {name:<12} {secs:>8}  {summary}")

out.append("")
out.append("=" * w)
out.append(f"  Full output: {full}")
out.append("=" * w)

text = "\n".join(out) + "\n"
with open(report, "w", encoding="utf-8") as f:
    f.write(text)
print(text)
sys.exit(1 if failed else 0)
PYEOF
RC=$?

echo
echo "================================================================"
echo " Report written to: ${REPORT}"
echo " Send that file back. Full output: ${FULL}"
echo "================================================================"
exit ${RC}
