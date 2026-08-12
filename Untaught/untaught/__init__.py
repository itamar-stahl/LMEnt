"""Untaught: train LMEnt models with specific concepts held out of the loss.

See ``Untaught/README.md`` for the mechanism and how to run it.
"""

__all__ = ["ChunkExclusionCallback"]


def __getattr__(name):
    # Lazy so that `python -m untaught.es_blacklist` (an Elasticsearch-only
    # tool) does not drag in olmo_core and its training dependencies.
    if name == "ChunkExclusionCallback":
        from .exclusion import ChunkExclusionCallback

        return ChunkExclusionCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
