from __future__ import annotations

from typing import Any
from nbcc.mlir_lowering import BackendInterface, UnsupportedError
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
from nbcc.mlir_lowering import LowerStates


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
) -> Any:
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
        raise UnsupportedError

    # Core Methods
    def lower_type(self, ty) -> tuple[ir.Type, ...]:
        """Convert SealIR types to backend IR types.

        Returns a tuple of MLIR types. For single types, returns (type,).
        For void/None types, returns (). For composite types, returns
        tuple of all component types.
        """
        res = self._dispatch_lower_type(self, fqn=FQN(ty.name), args=ty.args)

        assert isinstance(res, tuple), str(res)
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
            return ()

        @disp.case(
            lambda self, fqn, args: fqn.namespace.fullname == "mlir::type"
        )
        def _handle_mlir_types_by_parsing(self, fqn: FQN, args: tuple):
            [enc] = fqn.parts[-1].qualifiers
            tyname = decode_type_name(enc.fullname)
            items = parse_composite_type(tyname)
            if items is not None:
                # Handle composite types (multiple values)
                tys: list[ir.Type] = []
                for it in items:
                    res = self._dispatch_lower_type(self, FQN(it), ())
                    # All dispatch methods should return tuples consistently
                    assert isinstance(
                        res, tuple
                    ), f"Expected tuple from dispatch, got {type(res)}: {res}"
                    tys.extend(res)
                return tuple(tys)
            else:
                # Single type - return the type (will be wrapped in tuple by lower_type)
                ty = ir.Type.parse(tyname, context=self.context)
                return (ty,)

        def by_typename(fullname: str):
            def wrap(self, fqn, args):
                return fqn.fullname == fullname

            return wrap

        @disp.case(by_typename("builtins::i32"))
        def _handle_builtins_i32(self, fqn: FQN, args: tuple):
            return (self.i32,)

        @disp.case(by_typename("types::NoneType"))
        def _handle_none(self, fqn: FQN, args: tuple):
            return ()

    def get_ll_type(self, expr, mdmap) -> ir.Type | None:
        """Get backend type for expression with metadata."""
        raise NotImplementedError

    def handle_builtin_op(
        self, op_name: str, args, state, lowering_instance=None
    ):
        """Handle builtin operations during lowering."""
        raise NotImplementedError

    def handle_mlir_op(self, mlir_op: str, result_types, args):
        """Handle MLIR-specific operations during lowering."""
        raise NotImplementedError

    # Constant creation methods
    def create_constant_i32(self, value: int):
        """Create 32-bit integer constant using CuTile."""
        # Using the existing create_constant helper method
        return self.create_constant(value, "i32")

    def create_constant_i64(self, value: int):
        """Create 64-bit integer constant using CuTile."""
        return self.create_constant(value, "i64")

    def create_constant_f64(self, value: float):
        """Create 64-bit float constant using CuTile."""
        return self.create_constant(value, "f64")

    def create_constant_boolean(self, value: bool):
        """Create boolean constant using CuTile."""
        return self.create_constant(int(value), "i1")

    # Control flow methods
    def create_if_op(self, condition, result_types, has_else=True):
        """Create if-else control flow operation - may not be supported in CuTile."""
        raise UnsupportedError(
            "SCF if operations not supported in CuTile backend"
        )

    def create_yield_op(self, operands):
        """Create yield operation - may not be supported in CuTile."""
        raise UnsupportedError(
            "SCF yield operations not supported in CuTile backend"
        )

    def create_while_op(self, result_types, init_args):
        """Create while loop operation - may not be supported in CuTile."""
        raise UnsupportedError(
            "SCF while operations not supported in CuTile backend"
        )

    def create_condition_op(self, condition, args):
        """Create condition operation - may not be supported in CuTile."""
        raise UnsupportedError(
            "SCF condition operations not supported in CuTile backend"
        )

    def get_scf_op_results(self, while_op):
        """Get results from SCF operation - may not be supported in CuTile."""
        raise UnsupportedError(
            "SCF operation results not supported in CuTile backend"
        )

    # Function operation methods
    def create_function_call(self, result_types, callee, args):
        """Create function call operation - limited support in CuTile."""
        raise UnsupportedError(
            "Function calls not supported in CuTile backend"
        )

    def create_function_declaration(
        self, name, arg_types, result_types, visibility="private"
    ):
        """Create function declaration - limited support in CuTile."""
        raise UnsupportedError(
            "Function declarations not supported in CuTile backend"
        )

    # String constant method
    def create_string_constant(self, state: LowerStates, value: str):
        """Create string constant - not supported in CuTile (no LLVM operations)."""
        raise UnsupportedError(
            "String constants not supported in CuTile backend (requires LLVM operations)"
        )
