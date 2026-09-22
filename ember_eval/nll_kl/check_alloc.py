#!/usr/bin/env python
"""Refuse a checkpoint whose weight files were written with holes.

Three candidates in the 2026-09-20 grid were saved by jobs that exited 0 but
left 380-710 MB of shard 1 unwritten. The apparent size is correct -- only
st_blocks betrays it -- so nothing downstream noticed, and two of them were
scored and ranked. A holed model reads its missing blocks back as zeros and
looks like a very aggressive erasure: the one that reached the Rome table
degraded UNRELATED answers by 3.7 nats.

Sparse files are legitimate in general, but a safetensors checkpoint written
by torch.save/safetensors.save_file never is: every byte is dense tensor data.
So allocation materially below apparent size means a partial write.

    python check_alloc.py <model dir> [<model dir> ...]

Exits 1 and names the offending files if any are short.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MIN_RATIO = 0.98   # allow a little slack for filesystem accounting
MIN_SIZE = 10_000_000


def check(model_dir: Path) -> list[str]:
    problems = []
    for f in sorted(model_dir.glob("*.safetensors")):
        st = f.stat()
        apparent, alloc = st.st_size, st.st_blocks * 512
        if apparent < MIN_SIZE:
            continue
        ratio = alloc / apparent
        if ratio < MIN_RATIO:
            problems.append(
                f"{f}: {ratio:.1%} allocated, {(apparent - alloc) / 1e6:.0f} MB "
                f"of holes (apparent {apparent}, allocated {alloc})"
            )
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    bad = False
    for d in sys.argv[1:]:
        p = Path(d)
        if not p.is_dir():
            print(f"check_alloc: {d} is not a directory", file=sys.stderr)
            bad = True
            continue
        problems = check(p)
        if problems:
            bad = True
            for line in problems:
                print(f"HOLED {line}", file=sys.stderr)
    if bad:
        print("check_alloc: refusing to use a partially-written checkpoint", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
