"""Drive a Gemma judge that lives in another interpreter.

This is the pipeline-side half of the split described in
:mod:`ember.judge_server`. It exposes the same four callback seams as
:class:`~ember.gemma_judge.GemmaJudge`, so
:func:`~ember.lment_pipeline.run_lment_pipeline` cannot tell the two apart,
and forwards each call to a worker process started with a different Python.

The worker is a child of this run, not a shared service: it starts inside the
same Slurm allocation, dies with it, and loads the exact revision recorded in
the run snapshot. Nothing external can change the judge between runs.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

# Loading ~23G of bf16 weights off the lab NFS is slow and highly variable.
# Measured ~25-29 MB/s on both /home/dcor and /home/morg, which puts a bare
# read of gemma-4-12B-it at 15 minutes before any GPU transfer, and a real
# load on a contended node past 30. Override per config when a filer is
# slower still; a judge that is merely slow must not look like a dead one.
_DEFAULT_STARTUP_TIMEOUT = 3600.0
# A judge call is one short greedy generation; anything beyond this is a hang.
_DEFAULT_REQUEST_TIMEOUT = 600.0
_STDERR_TAIL_LINES = 40


class JudgeProcessError(RuntimeError):
    """The worker died, never became ready, or stopped answering."""


class SubprocessJudge:
    """Expose one out-of-process Gemma judge through EMBER's callback seams."""

    def __init__(self, *, model_path: str | Path, python_executable: str | Path,
                 device: str = "cuda", max_new_tokens: int = 256,
                 cache_dir: Optional[str | Path] = None,
                 local_files_only: bool = True,
                 project_root: Optional[str | Path] = None,
                 startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
                 request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
                 worker_module: str = "ember.judge_server") -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        interpreter = Path(python_executable).expanduser()
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"Judge interpreter not found: {interpreter}. Set "
                "lment.judge.python to the Python that can import "
                "transformers >= 5 (the Gemma environment).")
        root = (Path(project_root) if project_root is not None
                else Path(__file__).resolve().parents[1])

        command = [
            str(interpreter), "-u", "-m", worker_module,
            "--model", str(model_path),
            "--device", device,
            "--max-new-tokens", str(int(max_new_tokens)),
        ]
        if cache_dir is not None:
            command += ["--cache-dir", str(cache_dir)]
        if local_files_only:
            command.append("--local-files-only")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(root), env.get("PYTHONPATH", ""))))
        # The worker must never emit a progress bar on the protocol stream,
        # and it must not inherit a stale interpreter's site-packages.
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "error"

        self._request_timeout = float(request_timeout)
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_tail: Deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()

        self._process = subprocess.Popen(
            command, cwd=str(root), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

        self._readers = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

        try:
            self._await_ready(startup_timeout)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ #
    # Process plumbing                                                    #
    # ------------------------------------------------------------------ #

    def _pump_stdout(self) -> None:
        try:
            for line in self._process.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _pump_stderr(self) -> None:
        for line in self._process.stderr:
            self._stderr_tail.append(line.rstrip("\n"))

    def _fail(self, reason: str) -> JudgeProcessError:
        tail = "\n".join(self._stderr_tail) or "(no stderr output)"
        code = self._process.poll()
        status = "still running" if code is None else f"exit code {code}"
        return JudgeProcessError(f"{reason} ({status}). Worker stderr:\n{tail}")

    def _receive(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise self._fail(
                f"The judge worker did not answer within {timeout:g}s") from None
        if line is None:
            raise self._fail("The judge worker closed its output stream")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise self._fail(
                f"The judge worker emitted non-JSON output: {line.strip()!r}") from None
        if not isinstance(payload, dict):
            raise self._fail("The judge worker emitted a non-object response")
        return payload

    def _await_ready(self, timeout: float) -> None:
        payload = self._receive(timeout)
        if payload.get("ok") and payload.get("event") == "ready":
            return
        if payload.get("event") == "startup_failed":
            raise self._fail(
                "The judge worker failed to load the model: "
                f"{payload.get('error_type')}: {payload.get('error')}")
        raise self._fail(f"Unexpected judge handshake: {payload!r}")

    def _request(self, op: str, prompt: str) -> str:
        if self._closed:
            raise JudgeProcessError("The judge worker is already closed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string")
        # One loaded model generates serially, and Alpaca scoring calls these
        # from a thread pool. Serialise here as well as in the worker so a
        # second caller cannot interleave two lines on the same pipe.
        with self._lock:
            try:
                self._process.stdin.write(
                    json.dumps({"op": op, "prompt": prompt}) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, ValueError):
                raise self._fail("The judge worker stopped accepting requests") from None
            payload = self._receive(self._request_timeout)

        if payload.get("ok"):
            text = payload.get("text")
            if not isinstance(text, str):
                raise self._fail("The judge worker returned a non-string result")
            return text
        # Preserve the in-process exception type. The pipeline is fail-closed
        # on judge output, and it distinguishes a malformed response
        # (ValueError) from a dead judge, so collapsing both would turn an
        # abort into a misleading infrastructure error.
        message = f"{payload.get('error_type')}: {payload.get('error')}"
        if payload.get("error_type") == "ValueError":
            raise ValueError(message)
        raise JudgeProcessError(message)

    # ------------------------------------------------------------------ #
    # Callback seams, mirroring GemmaJudge                                #
    # ------------------------------------------------------------------ #

    def describe_feature(self, prompt: str) -> str:
        return self._request("describe", prompt)

    def classify_feature(self, prompt: str) -> str:
        return self._request("classify", prompt)

    def score_alpaca_relevance(self, prompt: str) -> str:
        return self._request("alpaca_relevance", prompt)

    def score_alpaca_fluency(self, prompt: str) -> str:
        return self._request("alpaca_fluency", prompt)

    def close(self) -> None:
        """Shut the worker down and free its GPU memory."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError, AttributeError):
                pass
        # Close stdin so the worker sees EOF even if the shutdown write failed.
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        # The pumps end when the pipes reach EOF. Join them before closing
        # those pipes, or they raise on a file this thread closed underneath.
        for reader in self._readers:
            reader.join(timeout=30)
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (BrokenPipeError, ValueError):
                pass

    def __enter__(self) -> "SubprocessJudge":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def resolve_judge_directory(source: str, *, revision: Optional[str] = None,
                            cache_dir: Optional[str | Path] = None,
                            local_files_only: bool = True) -> Path:
    """Resolve a judge to a concrete directory before the worker starts.

    Resolving on this side keeps hub access, and the pinned revision, in the
    process that owns the run snapshot; the worker only ever sees a path.
    """
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_absolute() or source.startswith("."):
        raise FileNotFoundError(f"Judge model directory not found: {candidate}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=source, revision=revision,
        cache_dir=(str(Path(cache_dir).resolve()) if cache_dir is not None else None),
        local_files_only=bool(local_files_only),
    )).resolve()


__all__ = ["JudgeProcessError", "SubprocessJudge", "resolve_judge_directory"]
