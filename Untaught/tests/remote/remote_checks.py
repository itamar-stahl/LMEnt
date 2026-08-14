"""REMOTE SUITE -- validates the real deployment, end to end.

Not run directly: ``tests/remote/run_remote_tests.sh`` drives it, phase by
phase, so that the slow parts (Elasticsearch, dataset construction, an actual
SLURM job) can be skipped or waited on from the shell.

    python tests/remote/remote_checks.py <phase>

Phases, cheapest first -- each assumes the previous one passed:

    env        the environment the login node actually has
    local      the three local suites, but here (real olmo_core, real paths)
    data       the dataset and its caches are where the configs say
    es         Elasticsearch is up, the index exists, and it is the right size
    identity   THE check: ES chunk_id == dataset instance index, empirically
    prepare    a real run folder builds, against the real index
    submitted  a submitted job's run folder is complete and its log is sane

Everything prints a machine-greppable line so the shell driver can summarize:

    [RESULT] <phase>.<check> PASS|FAIL|SKIP  <detail>
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
UNTAUGHT_ROOT = os.path.dirname(TESTS)
sys.path.insert(0, UNTAUGHT_ROOT)
sys.path.insert(0, TESTS)

CONFIGS = os.path.join(UNTAUGHT_ROOT, "configs")
CONTROL_CFG = os.path.join(CONFIGS, "train_170m_control.yaml")
ABLATED_CFG = os.path.join(CONFIGS, "train_170m_no_harry_potter.yaml")

_results = []


def record(phase: str, check: str, status: str, detail: str = "") -> None:
    _results.append((phase, check, status, detail))
    print(f"[RESULT] {phase}.{check} {status}  {detail}", flush=True)


def run_check(phase: str, check: str, fn) -> None:
    """Run one check, turning any exception into a FAIL line (never a crash)."""
    try:
        detail = fn()
        record(phase, check, "PASS", detail or "")
    except SkipCheck as e:
        record(phase, check, "SKIP", str(e))
    except Exception as e:  # noqa: BLE001 - a failing check must not abort the phase
        record(phase, check, "FAIL", f"{type(e).__name__}: {e}")


class SkipCheck(Exception):
    pass


# --------------------------------------------------------------------------- #
# phase: env
# --------------------------------------------------------------------------- #
def phase_env() -> None:
    def python_env():
        import platform

        conda = os.environ.get("CONDA_DEFAULT_ENV", "(none)")
        if conda != os.environ.get("CONDA_ENV", "lment"):
            raise AssertionError(
                f"active conda env is {conda!r}, expected {os.environ.get('CONDA_ENV')!r}"
            )
        return f"python {platform.python_version()} in conda env {conda}"

    def required_variables():
        missing = [
            v for v in ("UNTAUGHT_ROOT", "LMENT_ROOT", "LMENT_DATASET",
                        "OLMO_CORE_SRC", "UNTAUGHT_RUNS_DIR", "ES_HOST", "ES_PORT")
            if not os.environ.get(v)
        ]
        if missing:
            raise AssertionError(f"unset: {missing} -- did activate_env.sh run?")
        return f"UNTAUGHT_ROOT={os.environ['UNTAUGHT_ROOT']}"

    def cwd_is_untaught():
        if os.path.realpath(os.getcwd()) != os.path.realpath(UNTAUGHT_ROOT):
            raise AssertionError(f"cwd is {os.getcwd()}, expected {UNTAUGHT_ROOT}")
        return os.getcwd()

    def imports():
        import torch  # noqa: F401
        import yaml  # noqa: F401
        from examples.kas.train import build_config  # noqa: F401
        import olmo_core  # noqa: F401

        return f"torch {torch.__version__}, olmo_core + examples.kas.train import"

    def elasticsearch_client_installed():
        import elasticsearch

        return f"elasticsearch-py {elasticsearch.__version__}"

    for name, fn in (
        ("python_env", python_env),
        ("required_variables", required_variables),
        ("cwd", cwd_is_untaught),
        ("imports", imports),
        ("es_client", elasticsearch_client_installed),
    ):
        run_check("env", name, fn)


# --------------------------------------------------------------------------- #
# phase: local -- the same three suites, but against the real deployment
# --------------------------------------------------------------------------- #
def phase_local() -> None:
    """Run each local suite, and on failure say WHICH checks failed.

    A summary line ("1 failed") is useless in a report you cannot re-run, so
    the failing check names and their assertion messages are extracted, and the
    suite's whole output is saved next to the report.
    """
    log_dir = os.environ.get("UNTAUGHT_TEST_LOGDIR", "")

    for name, script in (("units", "test_local_units.py"),
                         ("integration", "test_local_integration.py"),
                         ("refactoring", "test_local_refactoring.py")):
        def run(script=script, name=name):
            r = subprocess.run(
                [sys.executable, os.path.join(TESTS, script)],
                capture_output=True, text=True, cwd=UNTAUGHT_ROOT,
            )
            out = (r.stdout or "") + (r.stderr or "")

            if log_dir:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    with open(os.path.join(log_dir, f"local_{name}.log"), "w",
                              encoding="utf-8") as f:
                        f.write(out)
                except OSError:
                    pass

            summary = next((l.strip() for l in reversed(out.splitlines())
                            if "passed," in l), "")
            if r.returncode == 0:
                return summary

            # Name every failing check, with the assertion that broke it.
            details, current = [], None
            for line in out.splitlines():
                stripped = line.strip()
                if stripped.startswith("FAIL "):
                    current = stripped[5:].strip()
                    details.append(current)
                elif current and ("Error:" in stripped or "error:" in stripped):
                    details[-1] = f"{details[-1]} -- {stripped[:150]}"
                    current = None
            raise AssertionError(
                f"{summary} || " + " || ".join(details or ["(no FAIL lines found)"])
            )

        run_check("local", name, run)


# --------------------------------------------------------------------------- #
# phase: data
# --------------------------------------------------------------------------- #
def phase_data() -> None:
    from framework.node.config_env import load_config

    cfg = load_config(CONTROL_CFG)

    def tokenized_parts():
        pattern = cfg["job"]["dataset_paths"]
        parts = sorted(glob.glob(pattern))
        if not parts:
            raise AssertionError(f"no tokenized parts match {pattern}")
        total_gb = sum(os.path.getsize(p) for p in parts) / (1024 ** 3)
        return f"{len(parts)} parts, {total_gb:.1f} GiB, first={os.path.basename(parts[0])}"

    def dataset_cache():
        cache = cfg["job"]["dataset_cache"]
        if not os.path.isdir(cache):
            raise SkipCheck(f"{cache} does not exist yet (built on the first run)")
        entries = os.listdir(cache)
        return f"{cache}: {len(entries)} entries"

    def runs_dir_writable():
        runs = os.environ["UNTAUGHT_RUNS_DIR"]
        os.makedirs(runs, exist_ok=True)
        probe = os.path.join(runs, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return runs

    def dataset_builds():
        """Construct the dataset the way training does -- catches a bad glob."""
        import tempfile

        from examples.kas.train import build_config

        from framework.node.config_env import to_upstream

        experiment = build_config(
            to_upstream(cfg, save_folder=tempfile.mkdtemp(prefix="untaught-remote-"))
        )
        dataset = experiment.dataset.build()
        dataset.prepare()
        return f"{len(dataset):,} instances"

    for name, fn in (("tokenized_parts", tokenized_parts),
                     ("dataset_cache", dataset_cache),
                     ("runs_dir_writable", runs_dir_writable),
                     ("dataset_builds", dataset_builds)):
        run_check("data", name, fn)


# --------------------------------------------------------------------------- #
# phase: es
# --------------------------------------------------------------------------- #
def phase_es() -> None:
    from framework.client.es_blacklist import get_esclient, index_for, load_blacklist
    from framework.node.config_env import load_config, resolve_path

    cfg = load_config(ABLATED_CFG)
    untaught = cfg["untaught"]
    index = index_for(bool(untaught.get("case_sensitive", True)))

    def cluster_up():
        es = get_esclient()
        info = es.info()
        return f"{info['version']['number']} at {os.environ['ES_HOST']}:{os.environ['ES_PORT']}"

    def index_exists():
        es = get_esclient()
        if not es.indices.exists(index=index):
            raise AssertionError(f"index {index!r} missing; restore it per the README")
        count = es.count(index=index)["count"]
        return f"{index}: {count:,} chunks"

    def blacklist_qids_resolve():
        blacklist = resolve_path(untaught["blacklist"])
        entities = load_blacklist(blacklist)
        es = get_esclient()
        from framework.client.es_blacklist import build_entity_query, normalize_thresholds

        thresholds = normalize_thresholds(untaught)
        found = []
        for entity in entities:
            n = es.count(index=index,
                         body={"query": build_entity_query([entity["qid"]], thresholds)}
                         )["count"]
            found.append(f"{entity['qid']}={n:,}")
            if n == 0:
                raise AssertionError(
                    f"{entity['qid']} matches 0 chunks -- wrong QID or wrong index"
                )
        return ", ".join(found)

    for name, fn in (("cluster_up", cluster_up),
                     ("index_exists", index_exists),
                     ("blacklist_qids_resolve", blacklist_qids_resolve)):
        run_check("es", name, fn)


# --------------------------------------------------------------------------- #
# phase: identity -- the assumption everything rests on
# --------------------------------------------------------------------------- #
def phase_identity(sample: int = 10) -> None:
    def sizes_match():
        import tempfile

        from examples.kas.train import build_config

        from framework.client.es_blacklist import get_esclient, index_for
        from framework.node.config_env import load_config, to_upstream

        cfg = load_config(CONTROL_CFG)
        experiment = build_config(
            to_upstream(cfg, save_folder=tempfile.mkdtemp(prefix="untaught-remote-"))
        )
        dataset = experiment.dataset.build()
        dataset.prepare()

        abl = load_config(ABLATED_CFG)
        index = index_for(bool(abl["untaught"].get("case_sensitive", True)))
        es_total = get_esclient().count(index=index)["count"]
        if es_total != len(dataset):
            raise AssertionError(
                f"dataset has {len(dataset):,} chunks, ES has {es_total:,} -- "
                "the index and the dataset are NOT the same corpus"
            )
        return f"both {len(dataset):,} chunks"

    def text_matches():
        r = subprocess.run(
            [sys.executable, os.path.join(TESTS, "verify_chunk_alignment.py"),
             "--config", CONTROL_CFG, "-n", str(sample)],
            capture_output=True, text=True, cwd=UNTAUGHT_ROOT,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            tail = out.strip().splitlines()[-2:]
            raise AssertionError("DO NOT TRAIN -- " + " | ".join(tail))
        return f"{sample} sampled chunks byte-identical between dataset and index"

    run_check("identity", "sizes_match", sizes_match)
    run_check("identity", "text_matches", text_matches)


# --------------------------------------------------------------------------- #
# phase: prepare -- build a real run folder against the real index
# --------------------------------------------------------------------------- #
def phase_prepare() -> None:
    import shutil
    import tempfile

    from framework.client.prepare import prepare
    from framework.node.artifact import ARTIFACT_NAME, load_artifact

    made = {}

    def control_folder():
        run_dir = tempfile.mkdtemp(prefix="untaught-remote-control-")
        made["control"] = run_dir
        written = prepare(CONTROL_CFG, run_dir)
        artifact = json.load(open(written["artifact"], encoding="utf-8"))
        if artifact["num_chunks"] != 0 or "EMPTY" not in artifact["comment"]:
            raise AssertionError("control artifact is not explicitly empty")
        for f in ("config.yaml", ARTIFACT_NAME, "job.slurm", "run_wrapper.sh"):
            if not os.path.getsize(os.path.join(run_dir, f)):
                raise AssertionError(f"{f} missing or empty")
        return f"4 files + checkpoints/ in {run_dir}"

    def ablated_folder():
        run_dir = tempfile.mkdtemp(prefix="untaught-remote-ablated-")
        made["ablated"] = run_dir
        written = prepare(ABLATED_CFG, run_dir)
        blacklist = load_artifact(written["artifact"])
        if not blacklist:
            raise AssertionError("ablated artifact is empty -- QIDs matched nothing")
        artifact = json.load(open(written["artifact"], encoding="utf-8"))
        per_entity = ", ".join(
            f"{e['qid']}={e['num_chunks']:,}" for e in artifact["entities"])
        pct = 100.0 * artifact["num_chunks"] / 10_500_000
        return f"{artifact['num_chunks']:,} chunks ({pct:.4f}% of corpus): {per_entity}"

    def slurm_is_pure_strings():
        text = open(os.path.join(made["control"], "job.slurm"), encoding="utf-8").read()
        if "${" in text or "$(" in text:
            raise AssertionError("job.slurm contains shell variables")
        if not text.startswith("#! /bin/sh\n#SBATCH --job-name="):
            raise AssertionError("job.slurm does not match the required format")
        return "no variables, correct header"

    def config_copy_is_verbatim():
        src = open(CONTROL_CFG, "rb").read()
        cp = open(os.path.join(made["control"], "config.yaml"), "rb").read()
        if src != cp:
            raise AssertionError("the config copy differs from the source")
        return "byte-identical"

    def check_passes_on_the_copy():
        """The trainer must accept the copy -- this is what the node will run."""
        r = subprocess.run(
            [sys.executable, os.path.join(UNTAUGHT_ROOT, "framework", "node",
                                          "train_untaught.py"),
             os.path.join(made["ablated"], "config.yaml"), "--check"],
            capture_output=True, text=True, cwd=UNTAUGHT_ROOT,
        )
        if r.returncode != 0:
            raise AssertionError(((r.stdout or "") + (r.stderr or "")).strip()[-300:])
        return "--check passed on the run folder's config copy"

    for name, fn in (("control_folder", control_folder),
                     ("ablated_folder", ablated_folder),
                     ("slurm_is_pure_strings", slurm_is_pure_strings),
                     ("config_copy_verbatim", config_copy_is_verbatim),
                     ("check_on_copy", check_passes_on_the_copy)):
        run_check("prepare", name, fn)

    for run_dir in made.values():
        shutil.rmtree(run_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# phase: submitted -- inspect a run folder produced by a real submission
# --------------------------------------------------------------------------- #
def phase_submitted(run_dir: str) -> None:
    from framework.node.artifact import ARTIFACT_NAME

    def folder_is_complete():
        required = ["config.yaml", ARTIFACT_NAME, "job.slurm", "run_wrapper.sh",
                    "client.log"]
        missing = [f for f in required
                   if not os.path.exists(os.path.join(run_dir, f))]
        if missing:
            raise AssertionError(f"missing: {missing}")
        if not os.path.isdir(os.path.join(run_dir, "checkpoints")):
            raise AssertionError("no checkpoints/ folder")
        return f"{len(required)} files + checkpoints/"

    def job_was_accepted():
        log = open(os.path.join(run_dir, "client.log"), encoding="utf-8").read()
        if "submitted" not in log.lower():
            raise AssertionError("client.log records no submission")
        line = next(l for l in log.splitlines() if "submitted" in l.lower())
        return line.strip()

    def job_produced_output():
        out = os.path.join(run_dir, "log.out")
        if not os.path.exists(out):
            raise SkipCheck("log.out not created yet -- the job is still queued")
        size = os.path.getsize(out)
        if size == 0:
            raise SkipCheck("log.out is empty -- the job has not started")
        return f"log.out is {size:,} bytes"

    def gpu_recorded():
        out = os.path.join(run_dir, "log.out")
        if not os.path.exists(out):
            raise SkipCheck("job has not started")
        text = open(out, encoding="utf-8", errors="ignore").read()
        for marker in ("NVIDIA", "Tesla", "GeForce", "A100", "L40S"):
            if marker in text:
                line = next(l for l in text.splitlines() if marker in l)
                return line.strip()[:90]
        raise SkipCheck("no nvidia-smi output yet")

    def training_started():
        out = os.path.join(run_dir, "log.out")
        err = os.path.join(run_dir, "log.err")
        text = ""
        for path in (out, err):
            if os.path.exists(path):
                text += open(path, encoding="utf-8", errors="ignore").read()
        if not text:
            raise SkipCheck("no output yet")
        for bad in ("Traceback (most recent call last)", "CUDA out of memory",
                    "torch.distributed.elastic.multiprocessing.errors"):
            if bad in text:
                line = next((l for l in text.splitlines() if bad.split("(")[0] in l), bad)
                raise AssertionError(f"job failed: {line.strip()[:120]}")
        if "UNTAUGHT RUN" not in text:
            raise SkipCheck("the trainer has not printed its summary yet")
        return "trainer started and printed its run summary"

    def exclusion_behaved_as_configured():
        out = os.path.join(run_dir, "log.out")
        err = os.path.join(run_dir, "log.err")
        text = ""
        for path in (out, err):
            if os.path.exists(path):
                text += open(path, encoding="utf-8", errors="ignore").read()
        if "UNTAUGHT RUN" not in text:
            raise SkipCheck("the trainer has not started yet")

        artifact = json.load(
            open(os.path.join(run_dir, ARTIFACT_NAME), encoding="utf-8"))
        is_control = artifact["num_chunks"] == 0

        if is_control:
            if "excluded instances" in text:
                raise AssertionError("a CONTROL run reported excluded instances")
            return "control: no exclusions reported, as expected"

        if "loaded" not in text or "chunk ids" not in text:
            raise SkipCheck("the callback has not loaded the artifact yet")
        line = next(l for l in text.splitlines() if "chunk ids" in l)
        if "guard leaks" in text:
            raise AssertionError("guard leaks recorded -- the blacklist is too broad")
        return line.strip()[-90:]

    for name, fn in (("folder_complete", folder_is_complete),
                     ("job_accepted", job_was_accepted),
                     ("job_output", job_produced_output),
                     ("gpu_recorded", gpu_recorded),
                     ("training_started", training_started),
                     ("exclusion_behaviour", exclusion_behaved_as_configured)):
        run_check("submitted", name, fn)


# --------------------------------------------------------------------------- #
def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    phase = argv[1]
    started = time.time()
    if phase == "env":
        phase_env()
    elif phase == "local":
        phase_local()
    elif phase == "data":
        phase_data()
    elif phase == "es":
        phase_es()
    elif phase == "identity":
        phase_identity(int(argv[2]) if len(argv) > 2 else 10)
    elif phase == "prepare":
        phase_prepare()
    elif phase == "submitted":
        phase_submitted(argv[2])
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        return 2

    failed = sum(1 for _p, _c, s, _d in _results if s == "FAIL")
    print(f"[PHASE] {phase} finished in {time.time() - started:.1f}s: "
          f"{sum(1 for r in _results if r[2] == 'PASS')} passed, {failed} failed, "
          f"{sum(1 for r in _results if r[2] == 'SKIP')} skipped", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
