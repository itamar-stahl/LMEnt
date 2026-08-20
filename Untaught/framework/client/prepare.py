"""Fill a run folder with everything a training job needs -- and an audit trail.

Called by ``sub_builder.sh`` on the login node, once per submission:

    python -m framework.client.prepare --config configs/train_X.yaml \
        --run-dir runs/<job_name>_<date>_<time>

After it returns, the run folder is self-contained:

    <run_dir>/
    ├── config.yaml               verbatim copy of the submitted config
    ├── untaught_blacklist.json   the exact exclusion (explicitly empty for control)
    ├── job.slurm                 fully materialized -- no environment variables
    ├── run_wrapper.sh            what the GPU node executes (chmod +x)
    └── checkpoints/              the trainer saves parameters here, by step

(log.out / log.err / client.log join them once the job runs.)

The GPU node never resolves anything: it executes ``run_wrapper.sh``, which
runs the trainer on this folder's name. The trainer knows where run folders
live and that each holds its own config.yaml and artifact. Elasticsearch is
only ever touched here.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import stat
import zlib
from typing import Any, Dict, Optional, Sequence

from framework.node.config_env import load_config, resolve_path
from framework.node.run_folder import (
    ARTIFACT_NAME,
    CHECKPOINTS_DIR,
    CONFIG_NAME,
    JOB_SLURM,
    RUN_WRAPPER,
    UNTAUGHT_ROOT,
    artifact_path,
    config_path,
)

from .es_blacklist import (
    build_artifact,
    empty_artifact,
    normalize_thresholds,
    write_artifact,
)


def generate_job_slurm(job: Dict[str, Any], run_dir: str, user: str) -> str:
    """The submission script, as fully parsed strings -- zero env variables.

    Reading it back months later must tell you exactly what was asked of SLURM,
    with no indirection through whatever the environment happened to hold.
    """
    # --constraint selects which GPU features a node must have. An empty
    # constraint means "any card SLURM has free", and then the directive is
    # simply left out rather than written as an empty one.
    constraint = str(job["constraint"]).strip()
    constraint_line = f'#SBATCH --constraint="{constraint}"\n' if constraint else ""

    # --account picks which SLURM association pays for the job, and on this
    # cluster it is what gates the partitions: a student default account only
    # reaches studentkillable, while killable and the gpu-*-killable partitions
    # need the research account. Omitted when empty, exactly like --constraint,
    # so a config that does not name one keeps using the submitter's default.
    account = str(job.get("account", "")).strip()
    account_line = f"#SBATCH --account={account}\n" if account else ""

    # --mail-user / --mail-type ask SLURM to email on the events named. Both are
    # optional and omitted together: a config that names no address gets no
    # directives, exactly like --constraint and --account. Worth knowing before
    # relying on it -- this cluster's `MailProg` is `/bin/mail`, which is not
    # present on the login node, so delivery may silently do nothing. Test with a
    # throwaway job before trusting it for anything that matters.
    mail_user = str(job.get("mail_user", "")).strip()
    mail_type = str(job.get("mail_type", "END,FAIL,TIME_LIMIT")).strip()
    mail_lines = (
        f"#SBATCH --mail-user={mail_user}\n#SBATCH --mail-type={mail_type}\n"
        if mail_user else ""
    )

    return f"""#! /bin/sh
#SBATCH --job-name={job["name"]}
#SBATCH --output={run_dir}/log.out
#SBATCH --error={run_dir}/log.err
{account_line}#SBATCH --partition={job["partition"]}
#SBATCH --time={job["max_time_minutes"]}
#SBATCH --signal=USR1@120
#SBATCH --nodes={job["nodes"]}
#SBATCH --ntasks={job["ntasks"]}
#SBATCH --mem={job["cpu_mem_mb"]}
#SBATCH --cpus-per-task={job["cpus_per_task"]}
#SBATCH --gpus={job["gpus"]}
{constraint_line}{mail_lines}
# Created by: {user}

{run_dir}/{RUN_WRAPPER}
"""


def rendezvous_port(run_dir: str) -> int:
    """A per-run torchrun --master-port, derived from the run folder name.

    20000-29999: above the ports a service is likely to hold, below the range
    Linux hands out for outgoing connections (32768-60999), so nothing else on
    the node is competing for it. crc32 rather than hash() because the value has
    to be the same every time this folder name is hashed -- hash() is salted per
    process, which would make the wrapper disagree with itself between runs.
    """
    return 20000 + zlib.crc32(os.path.basename(run_dir).encode("utf-8")) % 10000


def generate_run_wrapper(job: Dict[str, Any], run_dir: str) -> str:
    """What the GPU node executes -- literal strings, no variables.

    Reading this file must tell you exactly what ran, with nothing to look up:
    no shell variables standing in for paths, no indirection. The trainer takes
    just the run folder's name, because it knows where run folders live and
    that each one holds its own config.yaml -- naming the config as well would
    be saying the same thing twice.

    The resources are the #SBATCH block's business, exactly as
    https://www.cs.tau.ac.il/system/slurm documents them; all this file does is
    start one process per allocated GPU on the node the batch script runs on.

    The one thing it must not leave to chance is the rendezvous port. torchrun's
    static rendezvous binds 29500 unless told otherwise, and a control/ablated
    pair is submitted seconds apart, so SLURM regularly puts both on one node --
    where the second one dies at startup with EADDRINUSE. The port is therefore
    derived from the run folder name, whose timestamp already makes it unique,
    and written into the wrapper as a literal: same rule as every other value in
    this file, decided here and recorded there, nothing looked up at run time.
    """
    launch = f"""torchrun --nproc-per-node={job["gpus"]} --master-port={rendezvous_port(run_dir)} \\
  {UNTAUGHT_ROOT}/framework/node/train_untaught.py {os.path.basename(run_dir)}"""

    return f"""#!/bin/sh
# Generated by framework.client.prepare for job '{job["name"]}' -- do not edit.
# Every path below is literal, so this file is the record of exactly what ran.
# Everything this run used or produced lives in {run_dir}
set -eu

echo "=========================================================="
echo " job      : {job["name"]}"
echo " run dir  : {run_dir}"
echo " node     : $(hostname)"
echo " started  : $(date)"
echo "=========================================================="

# Node-side environment: cd to Untaught/, variables, PYTHONPATH, conda. No
# Elasticsearch -- the blacklist was resolved into the run folder before
# submission.
. {UNTAUGHT_ROOT}/framework/node/set_node_env.sh

nvidia-smi || echo "warning: nvidia-smi unavailable"

{launch}

echo "=========================================================="
echo " finished : $(date)"
echo "=========================================================="
"""


def prepare(config_file: str, run_dir: str) -> Dict[str, str]:
    """Populate ``run_dir``. Returns the paths written, keyed by role."""
    config_file = os.path.abspath(config_file)
    run_dir = os.path.abspath(run_dir)

    config = load_config(config_file)
    job = config["job"]
    untaught_cfg = config.get("untaught", {}) or {}
    # Fail on a threshold typo before anything is written or queried.
    normalize_thresholds(untaught_cfg)

    os.makedirs(os.path.join(run_dir, CHECKPOINTS_DIR), exist_ok=True)
    written: Dict[str, str] = {}

    # 1. The config, byte-for-byte -- comments included. The job trains from
    #    this copy, so later edits to configs/ cannot change what this run was.
    config_copy = config_path(run_dir)
    shutil.copyfile(config_file, config_copy)
    written["config"] = config_copy
    print(f"[prepare] config copy : {config_copy}")

    # 2. The exclusion, resolved. Control runs get an explicitly-empty artifact
    #    so the folder always answers "what was held out?" -- even with "nothing".
    blacklist = untaught_cfg.get("blacklist")
    if blacklist:
        blacklist = resolve_path(blacklist)
        if not os.path.isfile(blacklist):
            raise FileNotFoundError(f"[prepare] blacklist file not found: {blacklist}")
        artifact = build_artifact(blacklist, untaught_cfg)
        print(f"[prepare] blacklist   : {blacklist}")
        print(f"[prepare] index       : {artifact['index']} "
              f"(case_sensitive={artifact['case_sensitive']})")
        print(f"[prepare] thresholds  : {artifact['thresholds']}")
        for entity in artifact["entities"]:
            print(f"[prepare]   {entity['qid']:<12} {entity['num_chunks']:>9,} chunks  "
                  f"{entity['comment']}")
        print(f"[prepare] total       : {artifact['num_chunks']:,} unique chunks "
              f"({100.0 * artifact['num_chunks'] / 10_500_000:.4f}% of the corpus)")
    else:
        artifact = empty_artifact()
        print("[prepare] blacklist   : (none) -- CONTROL run, artifact written as "
              "explicitly empty")
    written["artifact"] = write_artifact(artifact, artifact_path(run_dir))
    print(f"[prepare] artifact    : {written['artifact']}")

    # 3. The submission script, materialized.
    slurm_path = os.path.join(run_dir, JOB_SLURM)
    with open(slurm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_job_slurm(job, run_dir, getpass.getuser()))
    written["job_slurm"] = slurm_path
    print(f"[prepare] job.slurm   : {slurm_path}")

    # 4. The node-side wrapper, executable.
    wrapper_path = os.path.join(run_dir, RUN_WRAPPER)
    with open(wrapper_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_run_wrapper(job, run_dir))
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode
             | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    written["run_wrapper"] = wrapper_path
    print(f"[prepare] run_wrapper : {wrapper_path}")

    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="framework.client.prepare", description=__doc__)
    parser.add_argument("--config", required=True, help="a training run config (YAML)")
    parser.add_argument("--run-dir", required=True,
                        help="this submission's run folder (created if missing)")
    args = parser.parse_args(argv)

    prepare(args.config, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
