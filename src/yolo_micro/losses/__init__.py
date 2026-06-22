"""Loss helpers.

Only dependency-free assignment utilities are imported at package import time.
Torch-backed loss modules remain available through their direct module paths.
"""

from .assignment import SizeAwareAssigner

__all__ = ["SizeAwareAssigner"]
