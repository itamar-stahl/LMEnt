"""Untaught: train LMEnt models with specific concepts held out of the loss.

See ``Untaught/README.md`` for the mechanism and how to run it.
"""

__all__ = ["ChunkExclusionCallback"]

from .exclusion import ChunkExclusionCallback
