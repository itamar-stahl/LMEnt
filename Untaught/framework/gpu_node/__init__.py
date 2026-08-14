"""Untaught, compute-node half: the training run itself.

Deliberately free of any Elasticsearch dependency -- the GPU nodes cannot reach
the index, so this side only ever *reads* the blacklist artifact that
``client_node`` produced. Imports point one way: ``client_node`` may use these
modules, never the reverse.
"""

__all__ = ["ChunkExclusionCallback"]


def __getattr__(name):
    # Lazy so that importing config/artifact helpers does not drag in olmo_core
    # and its training dependencies.
    if name == "ChunkExclusionCallback":
        from .exclusion import ChunkExclusionCallback

        return ChunkExclusionCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
