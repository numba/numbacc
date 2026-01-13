import os.path
from pathlib import Path
from functools import lru_cache

_dir = Path(os.path.dirname(__file__))


@lru_cache(maxsize=None)
def _load_transform_file(filename: str) -> str:
    """Load a transform file on demand and cache the result."""
    with open(_dir / "transform_sequences" / filename, "r") as fin:
        return fin.read()


# Cache for lazy-loaded attributes
_cached_transforms: dict[str, str] = {}


def __getattr__(name: str) -> str:
    """Lazy load transform attributes on demand based on available .mlir files."""
    if name in _cached_transforms:
        return _cached_transforms[name]

    # Check if {name}.mlir exists in transform_sequences directory
    filename = f"{name}.mlir"
    file_path = _dir / "transform_sequences" / filename

    if file_path.exists():
        _cached_transforms[name] = _load_transform_file(filename)
        return _cached_transforms[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
