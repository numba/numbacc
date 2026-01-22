from __future__ import annotations
from abc import abstractmethod

from typing import Any, Callable, Coroutine, Sequence, cast
from nbcc.mlir_backend.backend import BackendInterface
from nbcc.mlir_utils import decode_type_name, parse_composite_type
from nbcc.frontend import TranslationUnit
from spy.fqn import FQN
from sealir.dispatchtable import DispatchTableBuilder, dispatchtable

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    PointerType,
    TileType,
    TensorViewType,
    writeBytecode,
    register_dialect,
)
import cuda_tile._mlir.dialects._cuda_tile_ops_gen as _cuda_tile

from cuda_tile._mlir.extras import types as _tile_types
import cuda_tile._mlir.ir as ir  # Context, Location, Module, Type

from nbcc.developer import TODO


def entry(
    sym_name,
    function_type,
    *,
    arch=None,
    arg_attrs=None,
    num_cta=None,
    occupancy=None,
    loc=None,
    ip=None,
) -> Tile:
    """
    from https://github.com/NVIDIA/cuda-tile/blob/8a775693b18303d6c696be6ffd06dadad1b32a8e/python/cuda_tile/dialects/cuda_tile_ops.py#L2C37-L2C44
    """
    optimization_hints = None
    # if (arch != None) and ((num_cta != None) or (occupancy != None)):
    #     optimization_hints = OptimizationHintsAttr.getEntryOpHint(
    #         arch,
    #         0 if num_cta is None else num_cta,
    #         0 if occupancy is None else occupancy,
    #         context=_ods_get_default_loc_context(loc),
    #     )
    # elif (num_cta != None) or (occupancy != None):
    #     # (arch == None) and hint values are specified
    #     raise ValueError(
    #         "Expected arch to be specified for OptimizationHint:"
    #         f" num_cta = {num_cta}, occupancy = {occupancy}"
    #     )
    return _cuda_tile.EntryOp(
        sym_name=sym_name,
        function_type=function_type,
        arg_attrs=arg_attrs,
        optimization_hints=optimization_hints,
        loc=loc,
        ip=ip,
    )


class CuTileBackend(BackendInterface):
    _tu: TranslationUnit

    Location = ir.Location
    InsertionPoint = ir.InsertionPoint

    def __init__(self, tu: TranslationUnit):
        self._tu = tu

        self._context = ir.Context()
        register_dialect(self._context, load=True)

        context = self._context
        with (
            context,
            self.Location.name(
                "CuTileBackend.__init__", context=self._context
            ),
        ):
            self.f32 = ir.F32Type.get()
            self._f64 = ir.F64Type.get()
            self.index_type = ir.IndexType.get()
            self.i8 = ir.IntegerType.get_signless(8)
            self._i32 = ir.IntegerType.get_signless(32)
            self._i64 = ir.IntegerType.get_signless(64)
            self._boolean = ir.IntegerType.get_signless(1)
            self._io_type = ir.IntegerType.get_signless(1)
            self._none_type = ir.IntegerType.get_signless(1)

    @classmethod
    def create(cls, tu: TranslationUnit) -> CuTileBackend:
        return cls(tu)

    def make_module(self, module_name: str) -> ir.Module:
        with self._context:

            class ModuleOp(_cuda_tile.ModuleOp):
                """Specialization for the module op class."""

                def __init__(self, sym_name, *, loc=None, ip=None):
                    super().__init__(sym_name, loc=loc, ip=ip)
                    self._body = self.regions[0].blocks.append()

                @property
                def body(self):
                    return self._body

            return ModuleOp("module", loc=ir.Location.name(module_name))

    def create_mlir_asm(self, opname, attr, result_types, args):
        if attr:
            irattrs = ir.Attribute.parse(attr)
            if isinstance(irattrs, ir.DictAttr):
                attrs = {
                    named_attr.name: named_attr.attr for named_attr in irattrs
                }
            else:
                raise ValueError("expects a dictattr")

        else:
            attrs = None

        op = ir.Operation.create(opname, result_types, args, attributes=attrs)
        try:
            op.verify()
        except Exception:
            print(op.get_asm())
            raise
        if len(result_types) == 1:
            return op.result
        elif len(result_types) > 1:
            return op.results

    def create_function(self, function_name: str, input_types, output_types):
        fqn = FQN(function_name)
        export_name = fqn.c_name
        assert output_types == [], output_types
        fnty = ir.TypeAttr.get(ir.FunctionType.get(input_types, []))
        entry_op = entry(sym_name=export_name, function_type=fnty)
        entry_op.sym_visibility = ir.StringAttr.get("public")
        body_start = entry_op.body.blocks.append(*fnty.value.inputs)

        class WrappedFunc:
            def __init__(self, op):
                self.func_op = op

            @property
            def arguments(self):
                return self.func_op.body.blocks[0].arguments

            @property
            def operation(self):
                return self.func_op

        print(entry_op.get_asm())
        return WrappedFunc(entry_op), body_start, body_start

    def create_constant(self, value, type):
        asm = f"cuda_tile.constant <{type}: {value}> : tile<{type}>"
        op = ir.Operation.parse(asm)
        self.InsertionPoint.current.insert(op)
        return op

    def initialize_io(self):
        return self.create_constant(0, self.io_type)

    def create_none(self):
        return self.create_constant(0, self.none_type)

    def create_return(self, values):
        return _cuda_tile.return_(values)

    def run_passes(self, module: Any, transforms: Any) -> Any:
        return module

    def finalize_const_block(self, const_entry, target):
        pass

    # Type Constants - Properties for clean access pattern
    @property
    def context(self) -> ir.Context:
        return self._context

    @property
    def i32(self) -> ir.Type:
        return self._i32

    @property
    def i64(self) -> ir.Type:
        return self._i64

    @property
    def f64(self) -> ir.Type:
        return self._f64

    @property
    def boolean(self) -> ir.Type:
        return self._boolean

    @property
    def none_type(self) -> ir.Type:
        return self._none_type

    @property
    def io_type(self) -> ir.Type:
        return self._io_type

    @property
    def llvm_ptr(self) -> ir.Type:
        return self._llvm_ptr

    # Core Methods
    def lower_type(self, ty) -> ir.Type:
        """Convert SealIR types to backend IR types."""
        res = self._dispatch_lower_type(self, fqn=FQN(ty.name), args=ty.args)
        return res

    @dispatchtable
    @staticmethod
    def _dispatch_lower_type(disp: DispatchTableBuilder) -> None:
        @disp.default
        def _unknown_type(self, fqn: FQN, args: tuple):
            raise NotImplementedError(f"unknown type: {fqn}")

        def type_name_matches(fullname):
            def wrap(self, fqn: FQN, args: tuple) -> bool:
                return fqn.fullname == fullname

            return wrap

        @disp.case(type_name_matches("mlir::type::()"))
        def _handle_void(self, fqn: FQN, args: tuple):
            return None

        @disp.case(
            lambda self, fqn, args: fqn.namespace.fullname == "mlir::type"
        )
        def _handle_mlir_types_by_parsing(self, fqn: FQN, args: tuple):
            [enc] = fqn.parts[-1].qualifiers
            tyname = decode_type_name(enc.fullname)
            items = parse_composite_type(tyname)
            if items is not None:
                tys = []
                for it in items:
                    res = self._dispatch_lower_type(self, FQN(it), ())
                    TODO("proper handling of multiple values")
                    assert not isinstance(res, (tuple, list))
                    tys.append(res)
                return tys
            else:
                return ir.Type.parse(tyname, context=self.context)

        def by_typename(fullname: str):
            def wrap(self, fqn, args):
                return fqn.fullname == fullname

            return wrap

        @disp.case(by_typename("builtins::i32"))
        def _handle_builtins_i32(self, fqn: FQN, args: tuple):
            return self.i32

        @disp.case(by_typename("types::NoneType"))
        def _handle_none(self, fqn: FQN, args: tuple):
            return self.none_type

    def get_ll_type(self, expr, mdmap) -> ir.Type | None:
        """Get backend type for expression with metadata."""
        raise NotImplementedError

    def handle_builtin_op(
        self, op_name: str, args, state, lowering_instance=None
    ):
        """Handle builtin operations during lowering."""
        raise NotImplementedError

    def handle_mlir_op(self, mlir_op: str, resty, args):
        """Handle MLIR-specific operations during lowering."""
        raise NotImplementedError
