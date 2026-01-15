from typing import TYPE_CHECKING, Annotated, cast

import base64

from spy.fqn import FQN
from spy.vm.builtin import IRTag
from spy.vm.function import (
    FuncParam,
    W_BuiltinFunc,
    W_FuncType,
    W_ASTFunc,
    W_Func,
)
from spy.vm.member import Member
from spy.vm.object import W_Object, W_Type, builtin_method
from spy.vm.struct import W_StructType
from spy.vm.registry import ModuleRegistry
from spy.vm.str import W_Str
from spy.vm.tuple import W_Tuple

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM

MLIR = ModuleRegistry("mlir")


@MLIR.builtin_type("MLIR_Type")
class W_MLIR_Type(W_StructType):
    original_name: str
    size: int

    @builtin_method("__new__", color="blue")
    @staticmethod
    def w_new(
        vm: "SPyVM", w_name: W_Str, *w_argtypes: "W_MLIR_Type"
    ) -> "W_MLIR_Type":
        name = vm.unwrap_str(w_name)

        def fmt(t: "W_MLIR_Type"):
            fn = getattr(t, "w_str", None)
            if fn is not None:
                return vm.unwrap_str(fn(vm, t))
            else:
                raise TypeError

        fqn = FQN(["mlir", "type", name.format(*map(fmt, w_argtypes))])
        w_type = W_MLIR_Type.from_pyclass(fqn, W_Object)
        w_type.original_name = name
        w_type.size = 0  # Fake sizeof for SPy
        return w_type

    @builtin_method("__str__")
    @staticmethod
    def w_str(vm: "SPyVM", w_self: "W_MLIR_Type") -> "W_Str":
        return vm.wrap(str(w_self.original_name))


@MLIR.builtin_func("MLIR_op")
def w_MLIR_op(
    vm: "SPyVM", w_opname: W_Str, w_restype: W_Type, w_argtypes: W_Tuple
) -> W_BuiltinFunc:
    RESTYPE = Annotated[W_Object, w_restype]
    opname = vm.unwrap_str(w_opname)
    argtypes_w = w_argtypes.items_w

    # functype - cast W_Object to W_Type (they should be types)
    params = [FuncParam(cast(W_Type, w_T), "simple") for w_T in argtypes_w]
    w_functype = W_FuncType.new(params, w_restype=w_restype)

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> RESTYPE:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = FQN(["mlir", "op", opname])
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag("mlir.op")  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_asm")
def w_MLIR_asm(
    vm: "SPyVM", w_asm: W_Str, w_restype: W_Type, w_argtypes: W_Tuple
) -> W_BuiltinFunc:
    RESTYPE = Annotated[W_Object, w_restype]
    asm = vm.unwrap_str(w_asm)
    argtypes_w = w_argtypes.items_w

    # Ensure all argument types have fqn attribute
    for at in argtypes_w:
        assert hasattr(at, "fqn"), f"Argument type {at} missing fqn attribute"

    # Cast to W_Type after assertion for type safety
    argtypes_typed = cast(list[W_Type], list(argtypes_w))

    fqn_parts = [
        asm,
        w_restype.fqn.fullname,
        *(at.fqn.fullname for at in argtypes_typed),
    ]
    opname = base64.urlsafe_b64encode(("$".join(fqn_parts)).encode()).decode()

    # functype - use the typed argtypes after assertion
    params = [FuncParam(w_T, "simple") for w_T in argtypes_typed]
    w_functype = W_FuncType.new(params, w_restype=w_restype)

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> RESTYPE:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = FQN(["mlir", "asm", opname])
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag(
        "mlir.asm", asm=asm
    )  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_transform", color="blue")
def w_MLIR_transform(
    vm: "SPyVM",
    fn: W_ASTFunc,
    passes: W_Tuple,
) -> W_ASTFunc:
    newfn = W_ASTFunc(
        w_functype=fn.w_functype,
        fqn=fn.fqn.with_suffix("transformed"),
        funcdef=fn.funcdef,
        closure=fn.closure,
        locals_types_w=fn.locals_types_w,
    )
    passes_list = [vm.unwrap_str(ps) for ps in passes.items_w]
    vm.add_global(
        newfn.fqn,
        newfn,
        irtag=IRTag("mlir.transforms", transforms=" ".join(passes_list)),
    )
    return newfn
