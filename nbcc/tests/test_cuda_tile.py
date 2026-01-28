import os
import os.path
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any
from subprocess import check_output, PIPE
from io import StringIO

import pytest
from mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)

import nbcc
from nbcc.compiler import compile_to_mlir
try:
    import cuda_tile
except ImportError:
    HAS_CUDA_TILE = False
else:
    HAS_CUDA_TILE = True

if HAS_CUDA_TILE:
    from nbcc.cutile_backend.backend import CuTileBackend


example_dir = Path(os.path.dirname(nbcc.__file__)) / ".." / "examples" / "cuda_tile"


@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


@contextmanager
def compile_mlir(filename: str) -> Generator[Any, None, None]:
    path = example_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        mlir_mod = compile_to_mlir(str(path), be_type=CuTileBackend)
        yield mlir_mod


def test_has_examples():
    assert example_dir.exists()

@pytest.mark.skipif(not HAS_CUDA_TILE, reason="no cuda_tile")
def test_cuda_tile_to_mlir():
    with compile_mlir("tile_example.spy") as mlir_mod:
        mlir_text = mlir_mod.operation.get_asm()
        assert "entry @spy_tile_example$exported$export_foo" in mlir_text
        assert "entry @spy_tile_example$exported$export_vecadd" in mlir_text
        bcbytes = check_output(["cuda-tile-translate", "--bytecode-version=13.1", "--mlir-to-cudatilebc", "--no-implicit-module"], input=mlir_text.encode())