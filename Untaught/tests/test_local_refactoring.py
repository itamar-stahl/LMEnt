"""LOCAL REFACTORING SUITE -- is this package complete and coherent, today?

    python tests/test_local_refactoring.py

Written against the package as it *should* be, with no regard for how it got
here. Many refactorings have moved, renamed and split things; the risk after
that is not a broken function but a broken *whole*: a dangling path, a name
only half-renamed, two sources of truth that drifted, a file nothing references.

So this suite checks structure and invariants rather than behaviour:

  A. inventory      -- every file that must exist, and nothing vestigial
  B. importability  -- every module imports, no dead imports, no cycles
  C. layering       -- node/ never depends on client/ or elasticsearch
  D. one source of truth -- no duplicated constants, paths or job parameters
  E. wiring         -- every generated/referenced path actually resolves
  F. config schema  -- configs and code agree on every key, both directions
  G. hygiene        -- no stale names, no leftover scripts, docs match reality
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

from testlib import Suite, UNTAUGHT_ROOT  # noqa: E402

suite = Suite(
    "LOCAL REFACTORING: package completeness & integrity",
    "Structural invariants of the package as it stands, history ignored.",
)

FRAMEWORK = os.path.join(UNTAUGHT_ROOT, "framework")
CLIENT = os.path.join(FRAMEWORK, "client")
NODE = os.path.join(FRAMEWORK, "node")
CONFIGS = os.path.join(UNTAUGHT_ROOT, "configs")


def read(*parts: str) -> str:
    with open(os.path.join(*parts), "r", encoding="utf-8") as f:
        return f.read()


def py_files() -> list:
    out = []
    for root, dirs, files in os.walk(FRAMEWORK):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        out += [os.path.join(root, f) for f in files if f.endswith(".py")]
    return out


# --------------------------------------------------------------------------- #
# A. inventory
# --------------------------------------------------------------------------- #
@suite.test
def test_every_expected_file_exists():
    """the package contains exactly the files the design calls for"""
    expected = [
        "activate_env.sh",
        "README.md",
        "SMOKE_TEST.md",
        "framework/__init__.py",
        "framework/env.sh",
        "framework/conda.sh",
        "framework/client/__init__.py",
        "framework/client/sub_builder.sh",
        "framework/client/es_blacklist.py",
        "framework/client/prepare.py",
        "framework/node/__init__.py",
        "framework/node/set_node_env.sh",
        "framework/node/config_env.py",
        "framework/node/run_folder.py",
        "framework/node/exclusion.py",
        "framework/node/train_untaught.py",
        "configs/train_170m_control.yaml",
        "configs/train_170m_no_harry_potter.yaml",
        "blacklists/harry_potter.json",
    ]
    missing = [p for p in expected if not os.path.exists(os.path.join(UNTAUGHT_ROOT, p))]
    assert not missing, f"missing: {missing}"


@suite.test
def test_no_vestigial_files():
    """no leftovers from earlier layouts (old dirs, run scripts, JSON configs)"""
    forbidden = [
        "untaught",                       # pre-rename package
        "framework/client_node",          # pre-rename
        "framework/gpu_node",             # pre-rename
        "framework/gpu_node.sh",
        "gpu_node.sh",
        "slurm",                          # pre-sub_builder slurm files
        "run_smoke_control.sh",
        "run_smoke_harry_potter.sh",
        "configs/env.sh",                 # moved under framework/
        "configs/conda.sh",
        "configs/entities",               # folded into blacklists/
        "configs/train_170m_control.json",
        "configs/train_170m_no_harry_potter.json",
    ]
    present = [p for p in forbidden if os.path.exists(os.path.join(UNTAUGHT_ROOT, p))]
    assert not present, f"vestigial, should be removed: {present}"


# --------------------------------------------------------------------------- #
# B. importability
# --------------------------------------------------------------------------- #
@suite.test
def test_every_module_imports():
    """every module imports standalone (config_env before any heavy dep)"""
    import importlib

    for module in (
        "framework",
        "framework.node",
        "framework.node.config_env",
        "framework.node.run_folder",
        "framework.client",
        "framework.client.es_blacklist",
    ):
        importlib.import_module(module)

    # These two need olmo_core; import them only if it is available.
    try:
        import olmo_core  # noqa: F401
    except ImportError:
        return
    for module in ("framework.node.exclusion", "framework.client.prepare"):
        importlib.import_module(module)


@suite.test
def test_no_unused_imports_in_our_modules():
    """no dead imports left behind by the refactorings"""
    problems = []
    for path in py_files():
        tree = ast.parse(read(path), filename=path)
        source = read(path)
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported[(a.asname or a.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == "*":
                        continue
                    imported[a.asname or a.name] = node.lineno
        for name, lineno in imported.items():
            if name in ("annotations",):
                continue
            # crude but effective: the name must appear somewhere other than
            # its own import line
            uses = [
                i for i, line in enumerate(source.splitlines(), 1)
                if re.search(rf"\b{re.escape(name)}\b", line) and i != lineno
            ]
            if not uses:
                problems.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)}:{lineno} {name}")
    assert not problems, "unused imports:\n  " + "\n  ".join(problems)


# --------------------------------------------------------------------------- #
# C. layering
# --------------------------------------------------------------------------- #
def imported_names(path: str) -> list:
    """Every module actually imported by a file (AST, so prose never counts).

    Includes imports nested inside functions -- our modules deliberately import
    heavy or optional things lazily, and those count just as much.
    """
    names = []
    for node in ast.walk(ast.parse(read(path), filename=path)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * (node.level or 0)
            names.append(prefix + (node.module or ""))
    return names


@suite.test
def test_node_never_depends_on_client():
    """framework/node imports nothing from framework/client (one-way arrow)"""
    offenders = []
    for path in py_files():
        if os.sep + "node" + os.sep not in path:
            continue
        for name in imported_names(path):
            if "client" in name:
                offenders.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)} -> {name}")
    assert not offenders, f"node depends on client: {offenders}"


@suite.test
def test_node_never_imports_elasticsearch():
    """framework/node has no Elasticsearch dependency, even transitively"""
    offenders = []
    for path in py_files():
        if os.sep + "node" + os.sep not in path:
            continue
        for name in imported_names(path):
            if "elasticsearch" in name.lower():
                offenders.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)} -> {name}")
    assert not offenders, f"node imports elasticsearch: {offenders}"

    # and prove it dynamically: import the node half with ES unimportable
    from testlib import no_elasticsearch

    try:
        import olmo_core  # noqa: F401
    except ImportError:
        return
    with no_elasticsearch():
        import importlib

        for module in ("framework.node.run_folder", "framework.node.config_env",
                       "framework.node.exclusion"):
            importlib.reload(importlib.import_module(module))


@suite.test
def test_client_uses_node_for_the_shared_contract():
    """the artifact format has one owner (node), which client imports"""
    prepare = read(CLIENT, "prepare.py")
    assert "from framework.node.run_folder import" in prepare
    # and the name is defined exactly once, in node/artifact.py
    definitions = [
        os.path.relpath(p, UNTAUGHT_ROOT) for p in py_files()
        if re.search(r"^ARTIFACT_NAME\s*=", read(p), re.M)
    ]
    assert definitions == ["framework" + os.sep + "node" + os.sep + "run_folder.py"], (
        f"ARTIFACT_NAME defined in {definitions}"
    )


# --------------------------------------------------------------------------- #
# D. one source of truth
# --------------------------------------------------------------------------- #
@suite.test
def test_no_duplicated_definitions():
    """each key constant is defined in exactly one place"""
    for const in ("DEFAULT_THRESHOLDS", "THRESHOLD_PREFIX", "CASE_SENSITIVE_INDEX",
                  "ARTIFACT_NAME", "CONFIG_NAME", "CHECKPOINTS_DIR", "JOB_SLURM",
                  "RUN_WRAPPER", "ENV_SH"):
        places = [
            os.path.relpath(p, UNTAUGHT_ROOT) for p in py_files()
            if re.search(rf"^{const}\s*(:|=)", read(p), re.M)
        ]
        assert len(places) == 1, f"{const} defined {len(places)}x: {places}"


@suite.test
def test_job_parameters_live_only_in_configs():
    """SLURM resources are in the configs, never hard-coded in the scripts"""
    sub_builder = read(CLIENT, "sub_builder.sh")
    for leaked in ("--partition=", "--mem=", "--cpus-per-task=", "--gpus=",
                   "#SBATCH"):
        assert leaked not in sub_builder, (
            f"sub_builder.sh hard-codes {leaked!r}; job parameters belong in the config"
        )

    # every #SBATCH line the generator emits must come from a config key
    prepare = read(CLIENT, "prepare.py")
    for key in ("name", "partition", "max_time_minutes", "nodes", "ntasks",
                "cpu_mem_mb", "cpus_per_task", "gpus"):
        assert f'job["{key}"]' in prepare, f"generator ignores job.{key}"


@suite.test
def test_paths_are_named_once_in_env_sh():
    """cluster paths appear in framework/env.sh, not scattered in the code"""
    env_sh = read(FRAMEWORK, "env.sh")
    for var in ("STAHLI_ROOT", "LMENT_ROOT", "LMENT_DATASET", "OLMO_CORE_SRC",
                "UNTAUGHT_RUNS_DIR", "ANACONDA_ROOT", "ES_HOME"):
        assert re.search(rf'^: "\$\{{{var}:=', env_sh, re.M), f"{var} not defaulted in env.sh"

    # The one absolute path allowed outside env.sh is the bootstrap LMENT_ROOT
    # in the two entry points (chicken-and-egg: they cd before sourcing).
    hardcoded = []
    for path in py_files():
        text = read(path)
        for m in re.finditer(r"/home/morg/\S*", text):
            hardcoded.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)}: {m.group()}")
    assert not hardcoded, "cluster paths hard-coded in python:\n  " + "\n  ".join(hardcoded)


# --------------------------------------------------------------------------- #
# E. wiring
# --------------------------------------------------------------------------- #
@suite.test
def test_shell_scripts_reference_real_paths():
    """every framework/... path named in a shell script exists"""
    scripts = [
        os.path.join(UNTAUGHT_ROOT, "activate_env.sh"),
        os.path.join(FRAMEWORK, "env.sh"),
        os.path.join(FRAMEWORK, "conda.sh"),
        os.path.join(CLIENT, "sub_builder.sh"),
        os.path.join(NODE, "set_node_env.sh"),
    ]
    missing = []
    for script in scripts:
        for m in re.finditer(r"(?:\./|/)(framework/[\w/\.]+\.(?:sh|py))", read(script)):
            rel = m.group(1)
            if not os.path.exists(os.path.join(UNTAUGHT_ROOT, rel)):
                missing.append(f"{os.path.basename(script)} -> {rel}")
    assert not missing, f"dangling references: {missing}"


@suite.test
def test_module_paths_in_scripts_and_docs_are_importable():
    """every `python -m framework...` invocation names a real module"""
    import importlib.util

    texts = {}
    for name in ("framework/client/sub_builder.sh", "README.md", "SMOKE_TEST.md",
                 "configs/train_170m_control.yaml",
                 "configs/train_170m_no_harry_potter.yaml"):
        texts[name] = read(UNTAUGHT_ROOT, name)

    bad = []
    for where, text in texts.items():
        for m in re.finditer(r"python -m ([\w\.]+)", text):
            module = m.group(1)
            if importlib.util.find_spec(module) is None:
                bad.append(f"{where}: {module}")
    assert not bad, f"non-existent modules referenced: {bad}"


@suite.test
def test_sub_builder_flow_is_complete():
    """sub_builder.sh does env -> run dir -> prepare -> validate -> sbatch"""
    text = read(CLIENT, "sub_builder.sh")
    required = [
        ("sources the client env", "activate_env.sh"),
        ("reads job.name from the config", "job.name"),
        ("timestamped run dir", "date +"),
        ("calls prepare", "framework.client.prepare"),
        ("passes the run dir", "--run-dir"),
        ("validates the generated files", "job.slurm"),
        ("checks for unresolved variables", r"grep -q"),
        ("logs to the run folder", "client.log"),
        ("submits", "sbatch"),
    ]
    missing = [what for what, marker in required if not re.search(marker, text)]
    assert not missing, f"sub_builder.sh is missing: {missing}"


@suite.test
def test_generated_scripts_are_self_contained():
    """job.slurm has no env vars; run_wrapper.sh uses absolute paths"""
    prepare = read(CLIENT, "prepare.py")
    slurm_template = prepare[prepare.index("def generate_job_slurm"):
                             prepare.index("def generate_run_wrapper")]
    # No shell variable references may survive into the generated slurm file
    assert "${" not in slurm_template.replace('{job["', "").replace('{run_dir}', ""), (
        "job.slurm template contains shell variables"
    )
    wrapper = prepare[prepare.index("def generate_run_wrapper"):
                      prepare.index("def prepare(")]
    assert "UNTAUGHT_ROOT" in wrapper, "wrapper must use absolute framework paths"
    assert "set_node_env.sh" in wrapper and "torchrun" in wrapper
    # The wrapper is a record, not a program: no shell variables may stand in
    # for the paths. (${...} in the template would survive into the output.)
    emitted = wrapper[wrapper.index('return f"""'):]
    assert "${" not in emitted.replace("{job[", "").replace("{run_dir}", ""), (
        "run_wrapper.sh template introduces shell variables"
    )


# --------------------------------------------------------------------------- #
# F. config schema, both directions
# --------------------------------------------------------------------------- #
@suite.test
def test_configs_and_code_agree_on_every_key():
    """to_upstream reads exactly the keys the configs define -- both ways"""
    from framework.node.config_env import load_config, to_upstream

    code = read(NODE, "config_env.py")
    mapper = code[code.index("def to_upstream"):code.index("def expand_env")]
    used = set(re.findall(r'train\["(\w+)"\]', mapper)) | set(
        re.findall(r'job\["(\w+)"\]', mapper))

    for name in ("train_170m_control", "train_170m_no_harry_potter"):
        cfg = load_config(os.path.join(CONFIGS, f"{name}.yaml"))
        defined = set(cfg["train"]) | set(cfg["job"])

        # every key the mapper reads must exist in the config
        missing = used - defined
        assert not missing, f"{name}: config lacks keys the code reads: {missing}"

        # and every train key must be consumed (job keys are also read by the
        # slurm generator, so they are checked separately)
        job_only = {"name", "partition", "max_time_minutes", "nodes", "ntasks",
                    "cpu_mem_mb", "cpus_per_task", "gpus"}
        unused = (set(cfg["train"]) - used)
        assert not unused, f"{name}: train keys nothing reads: {unused}"
        assert job_only <= set(cfg["job"]), f"{name}: job section incomplete"

        to_upstream(cfg, save_folder="/tmp/x")  # must not raise


@suite.test
def test_untaught_block_keys_are_all_honoured():
    """every untaught.* key in the configs is read somewhere in the code"""
    from framework.node.config_env import load_config

    all_code = "".join(read(p) for p in py_files())
    for name in ("train_170m_control", "train_170m_no_harry_potter"):
        cfg = load_config(os.path.join(CONFIGS, f"{name}.yaml"))
        for key in cfg.get("untaught", {}):
            if key.startswith("threshold_"):
                assert "THRESHOLD_PREFIX" in all_code
                continue
            assert f'"{key}"' in all_code, f"{name}: untaught.{key} is never read"


@suite.test
def test_configs_have_no_pseudo_comment_fields():
    """comments are YAML comments, never _comment fields"""
    from framework.node.config_env import load_config

    for name in ("train_170m_control", "train_170m_no_harry_potter"):
        cfg = load_config(os.path.join(CONFIGS, f"{name}.yaml"))
        assert list(cfg) == ["job", "train", "untaught"], f"{name}: groups {list(cfg)}"
        for group_name, group in cfg.items():
            bad = [k for k in group if k.startswith("_")]
            assert not bad, f"{name}.{group_name} has pseudo-comment fields: {bad}"
        # and the file really does carry '#' comments
        assert "#" in read(CONFIGS, f"{name}.yaml")


@suite.test
def test_the_pair_differs_only_where_intended():
    """control and ablated configs differ only in job.name and untaught"""
    from framework.node.config_env import load_config

    ctl = load_config(os.path.join(CONFIGS, "train_170m_control.yaml"))
    abl = load_config(os.path.join(CONFIGS, "train_170m_no_harry_potter.yaml"))

    assert ctl["train"] == abl["train"], "train blocks differ -- not a controlled pair"
    ctl_job = {k: v for k, v in ctl["job"].items() if k != "name"}
    abl_job = {k: v for k, v in abl["job"].items() if k != "name"}
    assert ctl_job == abl_job, "job blocks differ beyond the name"
    assert ctl["job"]["name"] != abl["job"]["name"], "both runs share a job name"
    assert ctl["untaught"].get("blacklist") is None
    assert abl["untaught"].get("blacklist")


# --------------------------------------------------------------------------- #
# G. hygiene
# --------------------------------------------------------------------------- #
@suite.test
def test_no_stale_names_anywhere():
    """no references to renamed/removed things survive in code or docs"""
    # Names that must not appear anywhere: old package layout, deleted scripts,
    # deleted functions, the pre-artifact submission style.
    everywhere = [
        "client_node", "gpu_node", "run_smoke", "sbatch --wrap",
        "strip_json_comments", "UNTAUGHT_BLACKLIST=",
        "train_untaught.py --prepare",
    ]
    # Old *config keys*. These are checked only in YAML, because the same words
    # are legitimate Python parameter names (e.g. build_artifact(blacklist_path)).
    config_only = ["blacklist_path:", "save_folder:", "thresholds:", "artifact_name:"]

    shell_and_docs = [
        os.path.join(UNTAUGHT_ROOT, f) for f in
        ("README.md", "SMOKE_TEST.md", "activate_env.sh")
    ] + [
        os.path.join(FRAMEWORK, "env.sh"), os.path.join(FRAMEWORK, "conda.sh"),
        os.path.join(CLIENT, "sub_builder.sh"),
        os.path.join(NODE, "set_node_env.sh"),
    ]
    configs = [os.path.join(CONFIGS, f"{n}.yaml") for n in
               ("train_170m_control", "train_170m_no_harry_potter")]

    hits = []
    for path in py_files() + shell_and_docs + configs:
        text = read(path)
        for name in everywhere:
            if name in text:
                hits.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)}: {name!r}")
    for path in configs:
        text = read(path)
        for name in config_only:
            if name in text:
                hits.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)}: stale key {name!r}")
    assert not hits, "stale references:\n  " + "\n  ".join(hits)


@suite.test
def test_run_folders_are_never_committed():
    """runs/ is gitignored -- run folders are records, not source"""
    gitignore = os.path.join(UNTAUGHT_ROOT, ".gitignore")
    assert os.path.exists(gitignore), "no .gitignore"
    patterns = [l.strip() for l in read(gitignore).splitlines()
                if l.strip() and not l.startswith("#")]
    assert "runs/" in patterns, f"runs/ not ignored: {patterns}"

    tracked = subprocess.run(["git", "ls-files", "runs/"], capture_output=True,
                             text=True, cwd=UNTAUGHT_ROOT)
    assert not tracked.stdout.strip(), (
        "run-folder files are committed: " + tracked.stdout[:300]
    )


@suite.test
def test_shell_scripts_are_valid_posix_sh():
    """every shell script parses under /bin/sh"""
    scripts = [
        os.path.join(UNTAUGHT_ROOT, "activate_env.sh"),
        os.path.join(FRAMEWORK, "env.sh"),
        os.path.join(FRAMEWORK, "conda.sh"),
        os.path.join(CLIENT, "sub_builder.sh"),
        os.path.join(NODE, "set_node_env.sh"),
    ]
    for script in scripts:
        r = subprocess.run(["sh", "-n", script], capture_output=True, text=True)
        assert r.returncode == 0, f"{os.path.basename(script)}: {r.stderr.strip()}"


@suite.test
def test_docs_only_reference_files_that_exist():
    """every python/shell path named in the docs actually exists"""
    import re as _re

    missing = []
    for doc in ("README.md", "SMOKE_TEST.md"):
        text = read(UNTAUGHT_ROOT, doc)
        for m in _re.finditer(r"(?:python |sh |\./)((?:tests|framework|configs)/[\w/\.]+\.(?:py|sh|yaml))", text):
            rel = m.group(1)
            if not os.path.exists(os.path.join(UNTAUGHT_ROOT, rel)):
                missing.append(f"{doc} -> {rel}")
    assert not missing, f"docs reference non-existent files: {missing}"


@suite.test
def test_archived_tests_are_not_referenced():
    """nothing points at tests/archive/ except the archive's own README"""
    archived = ["test_exclusion.py", "test_untaught_units.py"]
    hits = []
    for path in py_files() + [
        os.path.join(UNTAUGHT_ROOT, f) for f in ("README.md", "SMOKE_TEST.md")
    ] + [
        os.path.join(CLIENT, "sub_builder.sh"),
        os.path.join(UNTAUGHT_ROOT, "tests", "run_local_tests.py"),
        os.path.join(UNTAUGHT_ROOT, "tests", "remote", "run_remote_tests.sh"),
        os.path.join(UNTAUGHT_ROOT, "tests", "remote", "remote_checks.py"),
    ]:
        if not os.path.exists(path):
            continue
        text = read(path)
        for name in archived:
            if name in text:
                hits.append(f"{os.path.relpath(path, UNTAUGHT_ROOT)}: {name}")
    assert not hits, f"references to archived tests: {hits}"


@suite.test
def test_docs_describe_the_current_flow():
    """README and SMOKE_TEST describe sub_builder and the run folder"""
    for doc in ("README.md", "SMOKE_TEST.md"):
        text = read(UNTAUGHT_ROOT, doc)
        for marker in ("sub_builder.sh", "run folder", "untaught_blacklist.json",
                       "job.slurm", "checkpoints/"):
            assert marker in text, f"{doc} does not mention {marker!r}"


if __name__ == "__main__":
    sys.exit(suite.run())
