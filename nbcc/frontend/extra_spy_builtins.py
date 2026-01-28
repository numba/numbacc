from typing import TYPE_CHECKING, Annotated, cast, Any

from spy.fqn import FQN
from ..mlir_utils import (
    encode_asm_operation,
    create_mlir_type_fqn,
    parse_composite_type,
    decode_type_name,
)
from spy.vm.builtin import IRTag
from spy.vm.function import (
    FuncParam,
    W_BuiltinFunc,
    W_FuncType,
    W_ASTFunc,
    W_Func,
)
from spy.vm.b import B
from spy.vm.member import Member
from spy.vm.object import W_Object, W_Type, builtin_method
from spy.vm.struct import W_StructType
from spy.vm.registry import ModuleRegistry
from spy.vm.str import W_Str
from spy.vm.tuple import W_Tuple

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM

MLIR = ModuleRegistry("mlir")


class W_MLIR_Value(W_Object):
    """
    Base class for MLIR value objects.
    MLIR values are opaque objects that represent SSA values in MLIR.
    """

    __spy_storage_category__ = "reference"


_type_caches: dict[str, "W_MLIR_Type"] = {}


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
                out = vm.unwrap_str(fn(vm, t))
                return out
            else:
                raise TypeError

        formatted_name = name.format(*map(fmt, w_argtypes))
        fqn = create_mlir_type_fqn(formatted_name)

        if fqn in _type_caches:
            return _type_caches[fqn]
        w_type = W_MLIR_Type.from_pyclass(fqn, W_MLIR_Value)
        w_type.original_name = formatted_name
        w_type.size = 0  # Fake sizeof for SPy
        _type_caches[fqn] = w_type
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

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> W_Object:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = FQN(["mlir", "op", opname])
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag("mlir.op")  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_unpack")
def w_MLIR_unpack(vm: "SPyVM", w_fn: W_Func, w_idx: W_Object) -> W_BuiltinFunc:

    restype = cast(W_MLIR_Type, w_fn.w_functype.w_restype)

    assert restype.original_name.startswith("multivalues$")
    members = parse_composite_type(restype.original_name)

    def decode_type(str_fqn: str) -> str:
        fqn = FQN(str_fqn)
        [enc] = fqn.parts[-1].qualifiers
        return decode_type_name(str(enc))

    assert members is not None

    types = list(
        map(lambda x: W_MLIR_Type.w_new(vm, vm.wrap(decode_type(x))), members)
    )
    idx = vm.unwrap_i32(w_idx)
    retty = types[idx]

    params = [FuncParam(B.w_object, "simple")]
    w_functype = W_FuncType.new(params, w_restype=retty)

    def w_opimpl(vm: "SPyVM", fn: W_Object) -> W_Object:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = (
        FQN(["mlir", "unpack"])
        .with_suffix(str(idx))
        .with_qualifiers([restype.fqn.fullname])
    )
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag(
        "mlir.asm", idx=idx
    )  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_asm")
def w_MLIR_asm(
    vm: "SPyVM", w_asm: W_Str, w_restype: W_Object, w_argtypes: W_Tuple
) -> W_BuiltinFunc:

    RESTYPE: Any
    if isinstance(w_restype, W_Tuple):
        innernames = []
        for it_type in w_restype.items_w:
            assert isinstance(it_type, W_MLIR_Type)
            innernames.append(it_type.fqn.fullname)

        tyname = "multivalues$" + "|".join(innernames)
        fn_retty = W_MLIR_Type.w_new(vm, vm.wrap(tyname))
        RESTYPE = Annotated[W_Object, fn_retty]
        resname = fn_retty.fqn.fullname
    elif isinstance(w_restype, W_Type):
        RESTYPE = Annotated[W_Object, w_restype]
        fn_retty = w_restype
        resname = fn_retty.fqn.fullname
    else:
        raise AssertionError("NOT NEEDED")
        assert w_restype == B.w_None
        RESTYPE = W_Object
        fn_retty = B.w_None
        resname = "void"

    asm = vm.unwrap_str(w_asm)
    argtypes_w = w_argtypes.items_w

    # Ensure all argument types have fqn attribute
    for at in argtypes_w:
        assert hasattr(at, "fqn"), f"Argument type {at} missing fqn attribute"

    # Cast to W_Type after assertion for type safety
    argtypes_typed = cast(list[W_Type], list(argtypes_w))

    fqn_parts = [
        asm,
        resname,
        *(at.fqn.fullname for at in argtypes_typed),
    ]
    opname = encode_asm_operation(fqn_parts)

    # functype - use the typed argtypes after assertion
    params = [FuncParam(w_T, "simple") for w_T in argtypes_typed]
    w_functype = W_FuncType.new(params, w_restype=fn_retty)

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
