#!/bin/bash
#SBATCH --job-name=ens-report
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#
# The watcher that actually survives the night.
#
# An interactive Monitor expires after 30 minutes and dies with its session, so
# it cannot report on a run that finishes at 3am. This hangs off the last stage
# with afterany, which means it fires however the chain ends -- completed,
# failed, or cancelled because something upstream never satisfied a dependency.
#
# It reports state rather than asserting success: sacct for every job in the
# chain, what actually exists on disk, and the tail of anything that failed. A
# stage that reports COMPLETED while writing nothing is exactly the failure this
# is built to catch.
#
# SLURM's --mail-type does nothing on this cluster; smtp.tau.ac.il:25 delivers
# unauthenticated, so the mail goes out directly.
set -uo pipefail
: "${ENS:?}"; : "${CHAIN:?}"; MAILTO="${MAILTO:-}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
R="$ENS/REPORT.md"

{
  echo "# Ensemble run report"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)."
  echo
  echo '## Stages'
  echo
  echo '```'
  sacct -j "$CHAIN" --format=JobID%10,JobName%18,State%14,Elapsed,ExitCode -X 2>/dev/null
  echo '```'
  echo
  echo '## Artifacts on disk'
  echo
  built=$(find "$ENS/candidates" -maxdepth 2 -name model -type d 2>/dev/null | wc -l)
  scored=$(find "$ENS/scored_selection" -maxdepth 3 -name per_question.csv 2>/dev/null | wc -l)
  tested=$(find "$ENS/scored_test" -maxdepth 3 -name per_question.csv 2>/dev/null | wc -l)
  nllkl=$(find "$ENS/scored_nllkl" -maxdepth 3 -name records.json 2>/dev/null | wc -l)
  echo "| artifact | found | expected |"
  echo "|---|---|---|"
  echo "| candidate checkpoints | $built | 43 |"
  echo "| selection scorings | $scored | 46 |"
  echo "| test scorings | $tested | 9 |"
  echo "| NLL/KL scorings | $nllkl | 9 |"
  for f in results_ensembles/selected_checkpoints.json tables/panel_A_accuracy.csv \
           tables/question_level/summary.csv results_final/rome/results.json; do
    echo "| \`$f\` | $([ -f "$ENS/$f" ] && echo present || echo MISSING) | present |"
  done
  echo
  if [ -f "$ENS/results_ensembles/selected_checkpoints.json" ]; then
    echo '## Frozen winners'; echo; echo '```'
    "$PY" -c "
import json
for w in json.load(open('$ENS/results_ensembles/selected_checkpoints.json'))['winners']:
    print(f\"{w['topic']:9s} {w['method']:16s} {w['model_label']:30s} H={w['H_selection']}\")" 2>&1
    echo '```'; echo
  fi
  if [ -f "$ENS/tables/panel_A_accuracy.csv" ]; then
    echo '## Panel A'; echo; echo '```'; column -s, -t "$ENS/tables/panel_A_accuracy.csv" 2>/dev/null | cut -c1-140; echo '```'; echo
  fi
  if [ -f "$ENS/tables/question_level/summary.csv" ]; then
    echo '## Question-level distance to the twin'; echo; echo '```'
    column -s, -t "$ENS/tables/question_level/summary.csv" 2>/dev/null | cut -c1-140; echo '```'; echo
  fi
  bad=$(sacct -j "$CHAIN" --format=JobID,JobName%18,State -X -n 2>/dev/null \
        | grep -viE "COMPLETED|RUNNING|PENDING" | head -20)
  if [ -n "$bad" ]; then
    echo '## Stages that did NOT complete'; echo; echo '```'; echo "$bad"; echo '```'; echo
    echo 'Tails:'; echo; echo '```'
    echo "$bad" | awk '{print $1}' | while read -r j; do
      for f in "$ENS"/../ensembles/*_"$j".err /home/dcor/galbarak2/runs/ensembles/*_"$j".err; do
        [ -f "$f" ] && { echo "--- $(basename "$f")"; tail -8 "$f"; }
      done
    done 2>/dev/null
    echo '```'
  else
    echo '## No stage failed.'
  fi
} > "$R" 2>&1

echo "wrote $R"
cat "$R"

if [ -n "$MAILTO" ]; then
  "$PY" - "$MAILTO" "$R" <<'PYEOF'
import smtplib, ssl, sys
from email.message import EmailMessage
to, path = sys.argv[1], sys.argv[2]
body = open(path, encoding="utf-8").read()
m = EmailMessage()
m["Subject"] = "ensembles run report"
m["From"] = "galbarak2@tau.ac.il"
m["To"] = to
m.set_content(body)
try:
    with smtplib.SMTP("smtp.tau.ac.il", 25, timeout=60) as s:
        s.send_message(m)
    print(f"mailed report to {to}")
except Exception as e:
    print(f"mail failed ({e}) -- the report is still on disk", file=sys.stderr)
PYEOF
fi
exit 0
