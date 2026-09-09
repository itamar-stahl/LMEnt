import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from ember import judge_server
from ember.subprocess_judge import JudgeProcessError, SubprocessJudge


class _StubJudge:
    """Stands in for a loaded GemmaJudge on the worker side."""

    def __init__(self):
        self.calls = []

    def _record(self, op, prompt):
        self.calls.append((op, prompt))
        return f"{op}:{prompt}"

    def describe_feature(self, prompt):
        return self._record("describe", prompt)

    def classify_feature(self, prompt):
        if "bad" in prompt:
            raise ValueError("Gemma classification response contains no JSON object")
        return self._record("classify", prompt)

    def score_alpaca_relevance(self, prompt):
        return self._record("alpaca_relevance", prompt)

    def score_alpaca_fluency(self, prompt):
        return self._record("alpaca_fluency", prompt)


def _serve(requests, judge):
    """Drive judge_server.serve over canned requests, returning its replies."""
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    protocol = io.StringIO()
    with mock.patch.object(judge_server, "_PROTOCOL", protocol):
        judge_server.serve(judge, stdin=stdin)
    return [json.loads(line) for line in protocol.getvalue().splitlines() if line]


class JudgeServerTests(unittest.TestCase):
    def test_serve_routes_every_op_to_its_callback(self) -> None:
        judge = _StubJudge()
        replies = _serve([
            {"op": "describe", "prompt": "tokens"},
            {"op": "classify", "prompt": "description"},
            {"op": "alpaca_relevance", "prompt": "answer"},
            {"op": "alpaca_fluency", "prompt": "answer"},
        ], judge)
        self.assertEqual([r["text"] for r in replies], [
            "describe:tokens", "classify:description",
            "alpaca_relevance:answer", "alpaca_fluency:answer",
        ])
        self.assertTrue(all(r["ok"] for r in replies))
        self.assertEqual(len(judge.calls), 4)

    def test_serve_reports_the_original_exception_type(self) -> None:
        # The pipeline is fail-closed on malformed judge output, so the type
        # has to survive the hop rather than becoming a generic failure.
        reply, = _serve([{"op": "classify", "prompt": "bad tokens"}], _StubJudge())
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error_type"], "ValueError")
        self.assertIn("no JSON object", reply["error"])

    def test_serve_rejects_unknown_ops_and_malformed_requests(self) -> None:
        replies = _serve([{"op": "nope", "prompt": "x"}, {"op": "describe"}],
                         _StubJudge())
        self.assertEqual([r["ok"] for r in replies], [False, False])
        self.assertIn("Unknown judge op", replies[0]["error"])
        self.assertIn("string 'prompt'", replies[1]["error"])

    def test_serve_stops_on_shutdown_without_consuming_later_requests(self) -> None:
        judge = _StubJudge()
        replies = _serve([{"op": "shutdown"}, {"op": "describe", "prompt": "x"}],
                         judge)
        self.assertEqual(replies, [{"ok": True, "event": "shutdown"}])
        self.assertEqual(judge.calls, [])

    def test_importing_the_module_does_not_rebind_stdout(self) -> None:
        # The worker claims stdout for the protocol, but only inside main().
        # Importing it here must leave the test runner's stdout alone.
        self.assertIsNone(judge_server._PROTOCOL)
        self.assertIsNot(sys.stdout, sys.stderr)


# A worker that speaks the protocol without loading any weights, so the client
# half is testable on a login node.
_STUB_WORKER = textwrap.dedent('''
    import json, sys
    protocol = sys.stdout
    sys.stdout = sys.stderr
    def send(payload):
        protocol.write(json.dumps(payload) + "\\n"); protocol.flush()
    args = sys.argv[1:]
    import pathlib
    pathlib.Path("argv.json").write_text(json.dumps(args))
    if "--explode" in args:
        send({"ok": False, "event": "startup_failed",
              "error_type": "RuntimeError", "error": "no CUDA device"})
        raise SystemExit(1)
    print("a library wrote this to stdout", file=sys.stdout)
    send({"ok": True, "event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request["op"] == "shutdown":
            send({"ok": True, "event": "shutdown"}); break
        if request["op"] == "classify":
            send({"ok": False, "error_type": "ValueError", "error": "bad JSON"})
        else:
            send({"ok": True, "text": request["op"] + ":" + request["prompt"],
                  "argv": args})
''')


class SubprocessJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "stub_worker.py").write_text(_STUB_WORKER, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def _judge(self, **overrides):
        kwargs = dict(
            model_path="/models/gemma", python_executable=sys.executable,
            project_root=self.root, worker_module="stub_worker",
            startup_timeout=60.0, request_timeout=60.0,
        )
        kwargs.update(overrides)
        judge = SubprocessJudge(**kwargs)
        self.addCleanup(judge.close)
        return judge

    def test_callbacks_round_trip_through_a_live_worker(self) -> None:
        judge = self._judge()
        self.assertEqual(judge.describe_feature("tokens"), "describe:tokens")
        self.assertEqual(judge.score_alpaca_fluency("answer"),
                         "alpaca_fluency:answer")

    def test_stdout_noise_does_not_corrupt_the_protocol(self) -> None:
        # The stub prints to stdout before announcing readiness; the handshake
        # must still be the first thing the client reads.
        self.assertEqual(self._judge().describe_feature("x"), "describe:x")

    def test_worker_value_errors_arrive_as_value_errors(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._judge().classify_feature("anything")
        self.assertIn("bad JSON", str(caught.exception))

    def test_startup_failure_reports_worker_stderr(self) -> None:
        with self.assertRaises(JudgeProcessError) as caught:
            self._judge(model_path="--explode")
        self.assertIn("failed to load the model", str(caught.exception))

    def test_model_and_generation_settings_reach_the_worker(self) -> None:
        self._judge(max_new_tokens=64, device="cpu", cache_dir="/hf/hub",
                    local_files_only=True)
        argv = json.loads((self.root / "argv.json").read_text())
        self.assertEqual(argv[argv.index("--model") + 1], "/models/gemma")
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertEqual(argv[argv.index("--max-new-tokens") + 1], "64")
        self.assertEqual(argv[argv.index("--cache-dir") + 1], "/hf/hub")
        self.assertIn("--local-files-only", argv)

    def test_local_files_only_is_omitted_when_disabled(self) -> None:
        self._judge(local_files_only=False)
        argv = json.loads((self.root / "argv.json").read_text())
        self.assertNotIn("--local-files-only", argv)

    def test_closing_twice_is_safe_and_blocks_later_calls(self) -> None:
        judge = self._judge()
        judge.close()
        judge.close()
        with self.assertRaises(JudgeProcessError):
            judge.describe_feature("x")

    def test_missing_interpreter_is_reported_before_spawning(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            SubprocessJudge(model_path="/models/gemma",
                            python_executable=self.root / "no-such-python",
                            project_root=self.root, worker_module="stub_worker")
        self.assertIn("lment.judge.python", str(caught.exception))

    def test_rejects_empty_prompts_without_touching_the_worker(self) -> None:
        with self.assertRaises(ValueError):
            self._judge().describe_feature("   ")


if __name__ == "__main__":
    unittest.main()
