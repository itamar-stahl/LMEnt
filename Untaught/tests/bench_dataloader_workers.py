#!/usr/bin/env python3
"""Does batch assembly get faster with more dataloader workers?

One question, because one decision rests on it. The 2-epoch twins spend 40-60%
of their wall clock stalled waiting for batches (measured 2026-09-03, 8 h window
on n-h200), and `data_num_workers` is NOT in RESUME_IGNORED_FIELDS -- so the 12
workers those runs started with cannot be changed without abandoning them and
starting over. Whether that is worth doing depends entirely on whether more
workers would deliver more batches per second. If assembly is already saturated
at 12, a fresh run buys nothing and the question is closed.

This measures the *dataloader alone*: no model, no optimizer, no GPU. That is
the point -- in training the GPU is what waits, so the thing worth timing is how
fast batches can be produced, and that is pure CPU plus filesystem work. It runs
anywhere with cores, which is why it can answer the question today instead of
waiting for an accelerator slot.

The config is read, not reproduced. Everything -- the 32,768-token batch, the
VSL curriculum, prefetch_factor, the dataset paths and cache -- comes from the
same YAML the live runs were submitted with, through the same build_config the
trainer uses. Only `data_num_workers` is overridden, one value per trial.

`--workers 4,12,24,48` includes two settings whose answers are already known, and
that is deliberate: 4 starved catastrophically in the 2026-08-29 01:02 pair
(~13.6 s/step, death by 1200 s DataLoader timeout) and 12 is what is running now.
If this harness does not reproduce that gap it is measuring the wrong thing, and
its numbers for 24 and 48 should not be believed either.

    python tests/bench_dataloader_workers.py configs/train_1b_control_2e.yaml

Filer note: this reads the same dataset the running twins read. The volume is
small -- a few hundred batches is tens of MB, and the cost is random IOPS rather
than bandwidth -- but it is not zero, so keep the batch counts modest while
those runs are live.
"""

import argparse
import copy
import os
import statistics
import sys
import tempfile
import time

UNTAUGHT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if UNTAUGHT_DIR not in sys.path:
    sys.path.insert(0, UNTAUGHT_DIR)

# Importing train_untaught runs its _bootstrap_olmo_core() at module level, which
# is what puts third_party/OLMo-core/src on the path. Do it before the olmo_core imports.
from framework.node import train_untaught  # noqa: F401,E402
from framework.node.config_env import load_config, to_upstream  # noqa: E402

from examples.kas.train import build_config  # noqa: E402
from olmo_core.data import KASDataCollator  # noqa: E402


def _tokens_in(batch) -> int:
    """Token count of a collated batch, without assuming the collator's schema."""
    for key in ("input_ids", "input_ids_list", "tokens"):
        value = batch.get(key) if isinstance(batch, dict) else None
        if value is not None and hasattr(value, "numel"):
            return int(value.numel())
    return 0


def measure(config_dict, workers, warmup, batches):
    """Steady-state batches/s at one worker count.

    The first batches are discarded: worker processes have to start, the first
    reads miss every cache, and prefetch_factor buffers have to fill. What the
    trainer experiences is the steady state after that, so that is what is timed.
    """
    trial = copy.deepcopy(config_dict)
    trial["train"]["data_num_workers"] = workers

    # save_folder has to exist for build_config, and nothing is written to it --
    # no trainer is ever constructed. A temp dir keeps this away from runs/.
    with tempfile.TemporaryDirectory(prefix="bench-dl-") as tmp:
        config = build_config(to_upstream(trial, save_folder=tmp))

    dataset = config.dataset.build()
    loader = config.data_loader.build(
        dataset,
        collator=KASDataCollator(
            pad_token_id=dataset.pad_token_id,
            rank_batch_size=config.trainer.rank_microbatch_size,
        ),
        mesh=None,
    )
    loader.reshuffle(epoch=1)

    iterator = iter(loader)
    try:
        for _ in range(warmup):
            next(iterator)

        waits = []
        tokens = 0
        started = time.perf_counter()
        for _ in range(batches):
            before = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - before)
            tokens += _tokens_in(batch)
        elapsed = time.perf_counter() - started
    finally:
        # Worker processes outlive the iterator otherwise, and the next trial
        # would then be timed against a machine still running the last one's.
        del iterator
        for name in ("shutdown", "close"):
            if hasattr(loader, name):
                getattr(loader, name)()
                break
        del loader

    waits.sort()
    return {
        "workers": workers,
        "batches": batches,
        "elapsed": elapsed,
        "bps": batches / elapsed,
        "s_per_batch": elapsed / batches,
        "tokens_per_s": tokens / elapsed if tokens else 0.0,
        "p50": statistics.median(waits),
        "p90": waits[int(0.90 * len(waits))],
        "max": waits[-1],
        # The training stalls are 60-300 s gaps between steps, so what matters is
        # not the average but whether long waits happen at all.
        "over_1s": sum(1 for w in waits if w > 1.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the run config to read (configs/*.yaml)")
    parser.add_argument(
        "--workers",
        default="4,12,24,48",
        help="comma-separated data_num_workers values to try (default 4,12,24,48)",
    )
    parser.add_argument("--warmup", type=int, default=20, help="batches discarded")
    parser.add_argument("--batches", type=int, default=120, help="batches timed")
    args = parser.parse_args()

    config_dict = load_config(args.config)
    train = config_dict["train"]
    print(f"[bench] config          : {args.config}", flush=True)
    print(f"[bench] global batch    : {train['data_global_batch_size']}", flush=True)
    print(f"[bench] microbatch      : {train['data_rank_microbatch_size']}", flush=True)
    print(f"[bench] prefetch factor : {train['data_prefetch_factor']}", flush=True)
    print(f"[bench] configured      : data_num_workers={train['data_num_workers']}", flush=True)
    print(f"[bench] host            : {os.uname().nodename}, "
          f"{len(os.sched_getaffinity(0))} cores available", flush=True)
    print(f"[bench] warmup/timed    : {args.warmup}/{args.batches} batches\n", flush=True)

    header = (f"{'workers':>7} {'batches/s':>10} {'s/batch':>9} {'tokens/s':>11} "
              f"{'p50':>7} {'p90':>7} {'max':>8} {'>1s':>5}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    results = []
    for workers in [int(w) for w in args.workers.split(",")]:
        try:
            r = measure(config_dict, workers, args.warmup, args.batches)
        except Exception as exc:  # a starved setting can time out; report, continue
            print(f"{workers:>7}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(r)
        print(f"{r['workers']:>7} {r['bps']:>10.2f} {r['s_per_batch']:>9.3f} "
              f"{r['tokens_per_s']:>11,.0f} {r['p50']:>7.3f} {r['p90']:>7.3f} "
              f"{r['max']:>8.2f} {r['over_1s']:>5}", flush=True)

    if len(results) < 2:
        return

    print("\n[bench] scaling relative to the configured 12 workers:", flush=True)
    baseline = next((r for r in results if r["workers"] == 12), results[0])
    for r in results:
        ratio = r["bps"] / baseline["bps"]
        print(f"  {r['workers']:>3} workers: {ratio:5.2f}x", flush=True)

    best = max(results, key=lambda r: r["bps"])
    gain = best["bps"] / baseline["bps"]
    print("", flush=True)
    if gain < 1.15:
        print(f"[bench] VERDICT: no meaningful headroom. Best is {best['workers']} "
              f"workers at {gain:.2f}x the configured 12. Raising data_num_workers "
              f"cannot explain or fix the training stalls, so abandoning the runs "
              f"to change it buys nothing.", flush=True)
    else:
        print(f"[bench] VERDICT: {best['workers']} workers delivers {gain:.2f}x the "
              f"configured 12. Batch assembly is worker-limited, so a fresh run at "
              f"the authors' settings with more workers is worth costing out "
              f"against finishing the current ones.", flush=True)


if __name__ == "__main__":
    main()
