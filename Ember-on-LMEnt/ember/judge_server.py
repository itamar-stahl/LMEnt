"""Stdio worker that serves one loaded Gemma judge to another interpreter.

The pipeline and the Gemma 4 judge cannot share one environment: EMBER pins
transformers 4.56.2, while ``gemma-4-12B-it`` reports ``model_type:
gemma4_unified`` and is unrecognised before transformers 5.x. This module is
the half that runs inside the Gemma environment. It loads one
:class:`~ember.gemma_judge.GemmaJudge` and answers newline-delimited JSON
requests on stdin, so the pipeline keeps its own interpreter and its own pins.

It imports nothing from EMBER except ``gemma_judge``, which needs only torch
and transformers. The Gemma environment therefore does not need pandas,
scipy, or python-dotenv.

Protocol, one JSON object per line in each direction::

    -> {"op": "describe|classify|alpaca_relevance|alpaca_fluency", "prompt": str}
    -> {"op": "shutdown"}
    <- {"ok": true, "text": str}
    <- {"ok": false, "error_type": str, "error": str}

Readiness is announced once, after the weights are resident, as
``{"ok": true, "event": "ready"}``. The caller must not send requests before
it arrives.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional


# Anything a library prints on stdout would corrupt the protocol stream, and
# transformers is not shy about it. ``main`` claims the real stdout for the
# protocol and redirects every ordinary write to stderr, which the parent
# drains separately and reports only when something fails. This happens in
# ``main`` rather than at import so that merely importing this module -- as
# the tests do -- never rebinds the caller's stdout.
_PROTOCOL: Optional[Any] = None


def _install_protocol() -> None:
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = sys.stdout
        sys.stdout = sys.stderr


def _send(payload: Dict[str, Any]) -> None:
    stream = sys.stdout if _PROTOCOL is None else _PROTOCOL
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        help="Resolved judge directory. The parent resolves the snapshot so "
             "this process never needs hub access.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Refuse to reach the network while loading.")
    return parser


def serve(judge: Any, *, stdin: Any = None) -> None:
    """Answer requests until shutdown, EOF, or a broken pipe."""
    handlers = {
        "describe": judge.describe_feature,
        "classify": judge.classify_feature,
        "alpaca_relevance": judge.score_alpaca_relevance,
        "alpaca_fluency": judge.score_alpaca_fluency,
    }
    stream = sys.stdin if stdin is None else stdin
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Judge request must be a JSON object")
            op = request.get("op")
            if op == "shutdown":
                _send({"ok": True, "event": "shutdown"})
                return
            handler = handlers.get(op)
            if handler is None:
                raise ValueError(f"Unknown judge op: {op!r}")
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("Judge request needs a string 'prompt'")
            _send({"ok": True, "text": handler(prompt)})
        except Exception as error:  # reported to the parent, never fatal here
            # The pipeline is fail-closed on judge output: a malformed
            # response must reach it as the same exception type the in-process
            # judge would have raised, so the run aborts rather than silently
            # selecting the wrong features.
            _send({
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            })


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    _install_protocol()
    from ember.gemma_judge import GemmaJudge

    try:
        judge = GemmaJudge.from_pretrained(
            args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
        )
    except Exception as error:
        _send({
            "ok": False,
            "event": "startup_failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise

    _send({"ok": True, "event": "ready"})
    try:
        serve(judge)
    finally:
        judge.close()


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main", "serve"]
