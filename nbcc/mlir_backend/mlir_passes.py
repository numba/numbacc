import subprocess as subp
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

import mlir.passmanager as passmanager
from mlir import ir


def module_pipeline(*passes) -> str:
    parts = []
    for ps in passes:
        match ps:
            case ModulePass() as mp:
                pss = mp.get_unwrapped()
            case FunctionPass() as fp:
                pss = fp.get_wrapped()
            case _:
                raise ValueError(ps)
        parts.append(pss)

    return f"{ModulePass.anchor}({','.join(parts)})"


@dataclass
class PassOption:
    name: str
    value_ctor: Callable[[Any], str]


def bool_ctor(x: bool) -> str:
    return str(int(bool(x)))


def double_ctor(x: float) -> str:
    return str(x)


def string_ctor(x: str) -> str:
    return str(x)


def string_options_ctor(*options: str) -> Callable[[str], str]:
    opts = set(options)

    def ctor(x: str) -> str:
        assert x in opts
        return x

    return ctor


class Pass:
    anchor: str
    passname: str
    options: list[str]
    _pass_options: dict[str, PassOption]

    def __init_subclass__(cls):
        cls._pass_options = {
            k: v for k, v in cls.__dict__.items() if isinstance(v, PassOption)
        }

    def __init__(self, **kwargs):
        self.options = options = []
        for k, v in kwargs.items():
            field = self._pass_options[k]
            options.append(f"{field.name}={field.value_ctor(v)}")

    def get_unwrapped(self) -> str:
        output = f"{self.passname}{{{self.get_options()}}}"
        return output

    def get_wrapped(self) -> str:
        return f"{self.anchor}({self.get_unwrapped()})"

    def get_options(self) -> str:
        return " ".join(self.options)


class ModulePass(Pass):
    anchor: str = "builtin.module"


class FunctionPass(Pass):
    anchor: str = "func.func"


class Canonicalize(ModulePass):
    passname = "canonicalize"


class CSE(ModulePass):
    passname = "cse"


class Inline(ModulePass):
    passname = "inline"


class SymbolDCE(ModulePass):
    passname = "symbol-dce"


class TransformInterpreter(ModulePass):
    passname = "transform-interpreter"


class LowerVectorMask(FunctionPass):
    passname = "lower-vector-mask"


class EliminateEmptyTensors(ModulePass):
    passname = "eliminate-empty-tensors"


class EmptyTensorToAllocTensor(ModulePass):
    passname = "empty-tensor-to-alloc-tensor"


class OneShotBufferize(ModulePass):
    passname = "one-shot-bufferize"
    bufferize_function_boundaries = PassOption(
        "bufferize-function-boundaries", bool_ctor
    )
    copy_before_write = PassOption("copy-before-write", bool_ctor)


class OwnershipBasedBufferDeallocation(ModulePass):
    passname = "ownership-based-buffer-deallocation"


class BufferizationLowerDeallocations(ModulePass):
    passname = "bufferization-lower-deallocations"


class ConvertBufferizationToMemRef(ModulePass):
    passname = "convert-bufferization-to-memref"


class BufferDeallocationSimplification(ModulePass):
    passname = "buffer-deallocation-simplification"


class BufferHoisting(FunctionPass):
    passname = "buffer-hoisting"


class BufferLoopHoisting(FunctionPass):
    passname = "buffer-loop-hoisting"


class FoldMemRefAliasOps(ModulePass):
    passname = "fold-memref-alias-ops"


class FoldTensorSubsetOps(ModulePass):
    passname = "fold-tensor-subset-ops"


class PromoteBuffersToStack(FunctionPass):
    passname = "promote-buffers-to-stack"


class Mem2Reg(ModulePass):
    passname = "mem2reg"


class LoopInvariantCodeMotion(ModulePass):
    passname = "loop-invariant-code-motion"


class ScfForLoopCanonicalization(ModulePass):
    passname = "scf-for-loop-canonicalization"


class ScfForLoopRangeFolding(ModulePass):
    passname = "scf-for-loop-range-folding"


class ScfForLoopToParallel(ModulePass):
    passname = "scf-forall-to-parallel"


class ScfForLoopSpecialization(ModulePass):
    passname = "scf-for-loop-specialization"


class ScfParallelLoopFusion(ModulePass):
    passname = "scf-parallel-loop-fusion"


class ConvertVectorToSCF(ModulePass):
    passname = "convert-vector-to-scf"


class ConvertVectorToLLVM(ModulePass):
    passname = "convert-vector-to-llvm"
    enable_arm_neon = PassOption("enable-arm-neon", bool_ctor)
    enable_arm_sve = PassOption("enable-arm-sve", bool_ctor)


class FinalizeMemRefToLLVM(ModulePass):
    passname = "finalize-memref-to-llvm"


class ConvertArithToLLVM(ModulePass):
    passname = "convert-arith-to-llvm"


class ConvertSCFToCF(ModulePass):
    passname = "convert-scf-to-cf"


class ConvertCFToLLVM(ModulePass):
    passname = "convert-cf-to-llvm"


class ConvertUbToLLVM(ModulePass):
    passname = "convert-ub-to-llvm"


class ConvertFuncToLLVM(ModulePass):
    passname = "convert-func-to-llvm"


class ConvertMathToLibM(ModulePass):
    passname = "convert-math-to-libm"


class ConvertIndexToLLVM(ModulePass):
    passname = "convert-index-to-llvm"


class ReconileUnrealizedCasts(ModulePass):
    passname = "reconcile-unrealized-casts"


class LinalgFuseElementwiseOps(ModulePass):
    passname = "linalg-fuse-elementwise-ops"


class LinalgGeneralizeNamedOps(ModulePass):
    passname = "linalg-generalize-named-ops"


class ConvertTensorToLinalg(ModulePass):
    passname = "convert-tensor-to-linalg"


class ConvertLinalgToAffineLoops(ModulePass):
    passname = "convert-linalg-to-affine-loops"


class ConvertLinalgToLoops(ModulePass):
    passname = "convert-linalg-to-loops"


class ConvertLinalgToParallelLoops(ModulePass):
    passname = "convert-linalg-to-parallel-loops"


class LowerAffine(FunctionPass):
    passname = "lower-affine"


class AffineSimplifyStructures(FunctionPass):
    passname = "affine-simplify-structures"


class AffineScalrep(FunctionPass):
    passname = "affine-scalrep"


class AffineLoopFusion(ModulePass):
    passname = "affine-loop-fusion"

    mode = PassOption(
        "mode", string_options_ctor("greedy", "producer", "sibling")
    )
    maximal = PassOption("maximal", bool_ctor)
    compute_tolerance = PassOption("compute-tolerance", double_ctor)


class MemRefExpand(FunctionPass):
    passname = "memref-expand"


class NormalizeMemRefs(ModulePass):
    passname = "normalize-memrefs"


class ExpandStridedMetadata(ModulePass):
    passname = "expand-strided-metadata"


@dataclass
class SubProcessOpt:
    passes: Sequence[Pass]
    _last_stderr: str = field(default="", init=False)

    def run(self, mod: ir.Module) -> ir.Module:
        pipeline = module_pipeline(*self.passes)
        ir_mod = mod.operation.get_asm(enable_debug_info=True)
        # run mlir-opt as a subprocess
        proc = subp.Popen(
            [
                "mlir-opt",
                "-mlir-print-ir-after-all",
                "-mlir-disable-threading",
                f"--pass-pipeline={pipeline}",
            ],
            stdin=subp.PIPE,
            stdout=subp.PIPE,
            stderr=subp.PIPE,
        )
        # send the MLIR module to the stdin
        stdout, stderr = proc.communicate(input=ir_mod.encode(), timeout=10)
        self._last_stderr = stderr.decode()
        # the optimized module is printed to stdout
        opt_ir_mod = stdout.decode()
        return ir.Module.parse(opt_ir_mod, context=mod.context)

    @property
    def last_stderr(self) -> str:
        return self._last_stderr


@dataclass
class InProcessOpt:
    passes: Sequence[Pass]

    def run(self, mod: ir.Module) -> ir.Module:
        pipeline = module_pipeline(*self.passes)
        # clone the module
        ir_mod = mod.operation.get_asm(enable_debug_info=True)
        cloned_mod = ir.Module.parse(ir_mod, context=mod.context)
        # run the passes
        pm = passmanager.PassManager.parse(pipeline, context=mod.context)
        pm.run(cloned_mod)
        return cloned_mod


class PassManager:
    """
    Note:
    - `get_log()` is only available when this class is initialized with
      `with_subprocess=True`.
    """

    _opt: Union[SubProcessOpt, InProcessOpt]

    def __init__(self, passes: Sequence[Pass], with_subprocess: bool):
        self._with_subprocess = with_subprocess
        if with_subprocess:
            self._opt = SubProcessOpt(passes)
        else:
            self._opt = InProcessOpt(passes)

    def run(self, mod: ir.Module) -> ir.Module:
        """
        Returns a optimized clone of `mod`
        """
        return self._opt.run(mod)

    def get_log(self) -> str:
        if self._with_subprocess:
            assert isinstance(self._opt, SubProcessOpt)
            return self._opt.last_stderr
        else:
            return ""
