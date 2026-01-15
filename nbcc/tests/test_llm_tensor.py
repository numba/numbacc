import os
import os.path
import tempfile
from contextlib import contextmanager
from ctypes import CDLL, byref, c_double
from pathlib import Path
from typing import Generator

import numpy as np
from mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)

import nbcc
from nbcc.compiler import compile_shared_lib


example_dir = Path(os.path.dirname(nbcc.__file__)) / ".." / "examples"


@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


@contextmanager
def compile_lib(filename: str, libname: str) -> Generator[Path, None, None]:
    path = example_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        outpath = dir / libname
        compile_shared_lib(str(path), str(outpath))
        yield outpath


def test_has_examples():
    assert example_dir.exists()


DIM0 = 700
DIM1 = 2000

benchmark_config = dict(rounds=200, iterations=1, warmup_rounds=1)
"""
NOTE:
- iterations MUST be one for teardown only run once per round.
"""


def golden_softmax(A):
    exp_x = np.exp(A - A.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def test_softmax():
    with compile_lib("llm_tensor.spy", "llm_tensor.so") as libname:
        lib = CDLL(libname)
        export_function = getattr(
            lib, "_mlir_ciface_spy_llm_tensor$exported$export_softmax"
        )
        print(export_function)

        # RUN
        memref_2d_f64 = make_nd_memref_descriptor(2, c_double)

        A = np.arange(DIM0 * DIM1, dtype=np.float64).reshape(DIM0, DIM1)
        # B = np.arange(DIM0 * DIM1, dtype=np.float64).reshape(DIM0, DIM1)

        argA = get_ranked_memref_descriptor(A)
        # argB = get_ranked_memref_descriptor(B)

        out_memref = (memref_2d_f64 * 1)()
        # args = [out_memref, byref(argA), byref(argB)]
        args = [out_memref, byref(argA)]
        export_function(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, golden_softmax(A))


def test_bench_baseline_softmax(benchmark):
    A = np.random.random((DIM0, DIM1)).astype(dtype=np.float64)
    # Requires pytest-benchmark >= 5.2.0 for teardown
    benchmark.pedantic(golden_softmax, args=[A], **benchmark_config)


def test_bench_nbcc_softmax(benchmark):
    with compile_lib("llm_tensor.spy", "llm_tensor.so") as libname:
        lib = CDLL(libname)
        export_function = getattr(
            lib, "_mlir_ciface_spy_llm_tensor$exported$export_softmax"
        )
        print(export_function)

        # RUN

        memref_2d_f64 = make_nd_memref_descriptor(2, c_double)

        A = np.random.random((DIM0, DIM1)).astype(dtype=np.float64)

        argA = get_ranked_memref_descriptor(A)

        out_memref = (memref_2d_f64 * 1)()
        # args = [out_memref, byref(argA), byref(argB)]
        args = [out_memref, byref(argA)]
        export_function(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, golden_softmax(A))

        def cleanup(*args):
            ranked_memref_to_numpy(args[0])  # cleanup

        # Requires pytest-benchmark >= 5.2.0 for teardown
        benchmark.pedantic(
            export_function, args=args, teardown=cleanup, **benchmark_config
        )


def test_bench_nbcc_softmax_fused(benchmark):
    with compile_lib("llm_tensor.spy", "llm_tensor.so") as libname:
        lib = CDLL(libname)

        export_function = getattr(
            lib,
            "_mlir_ciface_spy_llm_tensor$exported$export_softmax$transformed",
        )
        print(export_function)

        # RUN

        memref_2d_f64 = make_nd_memref_descriptor(2, c_double)

        A = np.random.random((DIM0, DIM1)).astype(dtype=np.float64)

        argA = get_ranked_memref_descriptor(A)

        out_memref = (memref_2d_f64 * 1)()
        # args = [out_memref, byref(argA), byref(argB)]
        args = [out_memref, byref(argA)]
        export_function(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, golden_softmax(A))

        def cleanup(*args):
            ranked_memref_to_numpy(args[0])  # cleanup

        # Requires pytest-benchmark >= 5.2.0 for teardown
        benchmark.pedantic(
            export_function, args=args, teardown=cleanup, **benchmark_config
        )
