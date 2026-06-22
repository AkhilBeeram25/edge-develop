"""YOLO-Micro package.

The top-level package intentionally avoids importing PyTorch modules so
standard-library utilities can run on edge machines without model dependencies.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

