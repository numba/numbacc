from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast, Coroutine, TYPE_CHECKING

# Type aliases for MLIR types - backends will use their specific ir modules
IRContext = Any
IRType = Any
IRLocation = Any
IRInsertionPoint = Any


from sealir import ase
from sealir.ase import SExpr
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix
from spy.fqn import FQN

from nbcc.developer import TODO
from nbcc.mlir_utils import decode_type_name, decode_asm_operation
from nbcc.frontend import grammar as sg, TranslationUnit


@dataclass(frozen=True)
class LowerStates(ase.TraverseState):
    push: Callable
    get_region_args: Callable
    function_block: Any
    constant_block: Any


class MDMap:
    mdmap: defaultdict[ase.SExpr, list[ase.SExpr]]

    def __init__(self):
        self.mdmap = defaultdict(list)

    def load(self, mdlist):
        for md in mdlist:
            match md:
                case sg.TypeInfo(value=value) | sg.IRTag(value=value):
                    self.mdmap[value].append(md)
                case _:
                    breakpoint()
                    TODO(f"Unknown MD: {md}: {type(md)}")

    def lookup_irtag(self, val: ase.SExpr) -> list[sg.IRTag]:
        return [x for x in self.mdmap[val] if isinstance(x, sg.IRTag)]

    def lookup_typeinfo(self, val: ase.SExpr) -> list[sg.TypeInfo]:
        return [x for x in self.mdmap[val] if isinstance(x, sg.TypeInfo)]

    def lookup_typeinfo_by_fqn(self, fqn: str) -> list[sg.TypeInfo]:
        for md, tis in self.mdmap.items():
            match md:
                case sg.FQN(fullname=fqn_fullname):
                    if fqn_fullname == fqn:
                        return [x for x in tis if isinstance(x, sg.TypeInfo)]
        raise NameError(f"{fqn!r} not found")


class UnsupportedError(Exception):
    """
    Backend do not support the requested operation.
    """


class BackendInterface(ABC):
    """Abstract interface for MLIR compilation backends.

    Defines the contract that Lowering depends on, enabling different
    backend implementations while maintaining MLIR-specific typing.
    """

    # Class attributes for MLIR context management
    Location: Any  # Set to ir.Location by implementations
    InsertionPoint: Any  # Set to ir.InsertionPoint by implementations

    @classmethod
    @abstractmethod
    def create(cls, tu: TranslationUnit) -> BackendInterface:
        raise NotImplementedError

    @abstractmethod
    def make_module(self, module_name: str) -> Any: ...

    @abstractmethod
    def run_passes(self, module: Any, transforms: Any) -> Any: ...

    # Type Constants - Properties for clean access pattern
    @property
    @abstractmethod
    def context(self) -> "IRContext":
        """MLIR context for creating types and operations."""

    @property
    @abstractmethod
    def i32(self) -> "IRType":
        """32-bit integer type."""

    @property
    @abstractmethod
    def i64(self) -> "IRType":
        """64-bit integer type."""

    @property
    @abstractmethod
    def f64(self) -> "IRType":
        """64-bit float type."""

    @property
    @abstractmethod
    def boolean(self) -> "IRType":
        """Boolean (1-bit integer) type."""

    @property
    @abstractmethod
    def none_type(self) -> "IRType":
        """None/void type representation."""

    @property
    @abstractmethod
    def io_type(self) -> "IRType":
        """IO token type for sequencing."""

    @property
    @abstractmethod
    def llvm_ptr(self) -> "IRType":
        """LLVM pointer type for memory operations."""

    # Core Methods
    @abstractmethod
    def lower_type(self, ty) -> tuple["IRType", ...]:
        """Convert SealIR types to backend IR types.

        Returns a tuple of MLIR types. For single types, returns (type,).
        For void/None types, returns (). For composite types, returns
        tuple of all component types.
        """

    @abstractmethod
    def get_ll_type(self, expr, mdmap) -> "IRType | None":
        """Get backend type for expression with metadata."""

    @abstractmethod
    def handle_builtin_op(
        self, op_name: str, args, state, lowering_instance=None
    ):
        """Handle builtin operations during lowering."""

    @abstractmethod
    def handle_mlir_op(self, mlir_op: str, result_types, args):
        """Handle MLIR-specific operations during lowering."""

    @abstractmethod
    def create_function(self, function_name: str, input_types, output_types):
        """Create function and return (func_op, const_block, func_block)."""

    @abstractmethod
    def finalize_const_block(self, const_entry, target):
        """Finalize constant block by adding control flow to target."""

    @abstractmethod
    def initialize_io(self):
        """Initialize IO token for operation sequencing."""

    @abstractmethod
    def create_return(self, values):
        """Create return operation with given values."""

    @abstractmethod
    def create_none(self):
        """Create None/null value."""

    @abstractmethod
    def create_mlir_asm(self, opname, attr, result_types, args):
        """Create MLIR operation from assembly specification."""

    # Constant creation methods
    @abstractmethod
    def create_constant_i32(self, value: int) -> Any:
        """Create 32-bit integer constant."""

    @abstractmethod
    def create_constant_i64(self, value: int) -> Any:
        """Create 64-bit integer constant."""

    @abstractmethod
    def create_constant_f64(self, value: float) -> Any:
        """Create 64-bit float constant."""

    @abstractmethod
    def create_constant_boolean(self, value: bool) -> Any:
        """Create boolean constant."""

    # Control flow methods
    @abstractmethod
    def create_if_op(
        self, condition: Any, result_types: list, has_else: bool = True
    ) -> Any:
        """Create if-else control flow operation."""

    @abstractmethod
    def create_yield_op(self, operands: list) -> Any:
        """Create yield operation for control flow regions."""

    @abstractmethod
    def create_while_op(self, result_types: list, init_args: list) -> Any:
        """Create while loop operation."""

    @abstractmethod
    def create_condition_op(self, condition: Any, args: list) -> Any:
        """Create condition operation for loops."""

    @abstractmethod
    def get_scf_op_results(self, while_op: Any) -> Any:
        """Get results from SCF while operation."""

    # Function operation methods
    @abstractmethod
    def create_function_call(
        self, result_types: list, callee: str, args: list
    ) -> Any:
        """Create function call operation."""

    @abstractmethod
    def create_function_declaration(
        self,
        name: str,
        arg_types: list,
        result_types: list,
        visibility: str = "private",
    ) -> Any:
        """Create function declaration operation."""

    # String constant method
    @abstractmethod
    def create_string_constant(self, state: LowerStates, value: str) -> Any:
        """Create string constant with backend-specific implementation."""


class Lowering:
    be: BackendInterface
    module: Any
    mdmap: MDMap
    loc: Any

    def __init__(
        self,
        be: BackendInterface,
        module: Any,
        mdmap: MDMap,
        func_map: dict[str, rg.Func],
    ):
        self.be = be
        self.module = module
        self.mdmap = mdmap
        self.func_map = func_map
        self._declared: dict[str, Any] = {}

    def get_return_types(self, root) -> list[Any]:
        [retval] = [
            port.value
            for port in root.body.ports
            if port.name == internal_prefix("ret")
        ]
        [ti] = self.mdmap.lookup_typeinfo(retval)
        out_types = self.be.lower_type(ti.type_expr)
        # Convert tuple to list and handle None type
        outs = list(out_types)
        if outs == [self.be.none_type]:
            return []
        return outs

    def irtags(self, root: rg.Func) -> dict:
        out: dict[str, list[tuple[str, str]]] = {}
        if irtags := self.mdmap.lookup_irtag(cast(ase.SExpr, root.body)):
            for irtag in irtags:
                bin = out.setdefault(irtag.tag, [])
                for irtagdata in cast(
                    list[sg.IRTagData],
                    cast(rg.GenericList, irtag.data[0]).children,
                ):
                    bin.append((irtagdata.key, irtagdata.value))
        return out

    def lower(self, root: rg.Func) -> Any:
        """Expression Lowering

        Lower RVSDG expressions to MLIR operations, handling control flow
        and data flow constructs.
        """
        context = self.be.context
        self.loc = loc = self.be.Location.name(
            f"{self.__class__.__name__}.lower()", context=context
        )
        module = self.module

        function_name = root.fname
        # Get the module body pointer so we can insert content into the
        # module.
        self.module_body = module_body = self.be.InsertionPoint(module.body)
        # Convert SealIR types to MLIR types.

        assert isinstance(root.args, rg.Args)
        arguments = root.args.arguments

        input_types = tuple(
            [typ for arg in arguments for typ in self.be.lower_type(arg)]
        )
        output_types = self.get_return_types(root)

        with context, loc, module_body:
            fun, const_block, func_block = self.be.create_function(
                function_name, input_types, output_types
            )

        # Define entry points of both the blocks.
        constant_entry = self.be.InsertionPoint(const_block)
        function_entry = self.be.InsertionPoint(func_block)

        region_args = []

        @contextmanager
        def push(arg_values):
            region_args.append(tuple(arg_values))
            try:
                yield
            finally:
                region_args.pop()

        def get_region_args():
            return region_args[-1]

        with context, loc, function_entry:
            memo = ase.traverse(
                cast(ase.SExpr, root),
                cast(
                    Callable[
                        [ase.SExpr, ase.TraverseState],
                        "Coroutine[ase.SExpr, Any, Any]",
                    ],
                    self.lower_expr,
                ),
                LowerStates(
                    push=push,
                    get_region_args=get_region_args,
                    function_block=fun,
                    constant_block=constant_entry,
                ),
            )

        with context, loc:
            self.be.finalize_const_block(constant_entry, func_block)

        print(fun.operation.get_asm())
        fun.operation.verify()
        return fun

    def _cast_return_value(self, val):
        return val

    def lower_expr(self, expr: SExpr, state: LowerStates):
        """Expression Lowering Implementation

        Implement the core expression lowering logic for various RVSDG
        constructs including functions, regions, control flow, and operations.
        """
        match expr:
            case rg.Func(args=args, body=body, fname=fqn):  # type: ignore[misc]
                func_args = args
                names = {
                    argspec.name: state.function_block.arguments[i]
                    for i, argspec in enumerate(func_args.arguments)
                }
                # Cast body to known type from rg.Func pattern match
                func_body = body
                region_begin = func_body.begin
                argvalues = []
                for k in region_begin.inports:
                    if k == internal_prefix("io"):
                        v = self.be.initialize_io()
                    else:
                        v = names[k]
                    argvalues.append(v)

                with state.push(argvalues):
                    outs = yield func_body

                # Handle SExpr - check for ports attribute or use _args with name filtering
                func_body_ports = func_body.ports
                portnames = [cast(rg.Port, p).name for p in func_body_ports]

                if self.get_return_types(expr) == []:
                    # func.ReturnOp([])
                    self.be.create_return([])
                    return
                try:
                    retidx = portnames.index(internal_prefix("ret"))
                except ValueError as e:
                    assert "!ret" in str(e)
                    self.be.create_return([])
                else:
                    retval = outs[retidx]
                    # func.ReturnOp([self._cast_return_value(retval)])
                    self.be.create_return([self._cast_return_value(retval)])

            case rg.RegionBegin(inports=ins):
                # Convert ins from tuple to list
                inports: list[str] = list(ins)
                portvalues = []
                for i, k in enumerate(inports):
                    pv = state.get_region_args()[i]
                    portvalues.append(pv)
                return tuple(portvalues)

            case rg.RegionEnd(
                begin=rg.RegionBegin() as begin,
                ports=ports,
            ):
                yield begin
                # Convert ports from tuple to list
                region_ports = [cast(rg.Port, p) for p in ports]
                portvalues = []
                for p in region_ports:
                    pv = yield p.value
                    portvalues.append(pv)
                return tuple(portvalues)

            case rg.ArgRef(idx=int(idx), name=str(name)):
                return state.function_block.arguments[idx]

            case rg.Unpack(val=source, idx=int(idx)):
                ports = yield cast(tuple, source)
                return ports[idx]

            case rg.DbgValue(value=value):
                val = yield value
                return val

            case rg.PyInt(int(ival)):
                with state.constant_block:
                    const = self.be.create_constant_i32(
                        ival
                    )  # HACK: select type
                return const

            case rg.PyBool(int(ival)):
                with state.constant_block:
                    const = self.be.create_constant_boolean(bool(ival))
                return const

            case rg.PyFloat(float(fval)):
                with state.constant_block:
                    const = self.be.create_constant_f64(fval)
                return const

            case rg.PyStr(str(strval)):
                with self.module_body:
                    str_addr = self.be.create_string_constant(state, strval)
                return str_addr

            # NBCC specific - BuiltinOp cases handled by dispatch table
            case sg.BuiltinOp(opname=op_name, args=args):
                # Cast args to known tuple type from pattern match
                builtin_args: tuple[SExpr, ...] = args
                # Handle special case for struct_get which has mixed arg types
                if op_name == "struct_get":
                    struct, pos = builtin_args
                    struct_value = yield struct
                    return self.be.handle_builtin_op(
                        op_name, [struct_value, pos], state, self
                    )
                else:
                    # Process arguments first, similar to MLIR ops
                    lowered_args = []
                    for arg in builtin_args:
                        lowered_args.append((yield arg))
                    return self.be.handle_builtin_op(
                        op_name, lowered_args, state, self
                    )

            case rg.PyBool(val):
                return self.be.create_constant_boolean(bool(val))

            case rg.PyInt(val):
                return self.be.create_constant_i64(val)

            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                condval = yield cond
                operand_vals = []
                for op in operands:
                    operand_vals.append((yield op))

                result_tys: list[Any] = []

                # determine result types
                assert isinstance(body, rg.RegionEnd)
                assert isinstance(orelse, rg.RegionEnd)
                for left_port, right_port in zip(
                    body.ports, orelse.ports, strict=True
                ):
                    left_ty = self.get_port_type(left_port)
                    right_ty = self.get_port_type(right_port)
                    if left_ty is None:
                        ty = right_ty
                    elif right_ty is None:
                        ty = left_ty
                    else:
                        assert left_ty == right_ty, f"{left_ty} != {right_ty}"
                        ty = left_ty
                    result_tys.append(ty)

                # Build the MLIR If-else
                if_op = self.be.create_if_op(
                    condition=condval, result_types=result_tys, has_else=True
                )

                with state.push(operand_vals):
                    # Make a detached module to temporarily house the blocks
                    with self.be.InsertionPoint(if_op.then_block):
                        value_body = yield body
                        self.be.create_yield_op([x for x in value_body])

                    with self.be.InsertionPoint(if_op.else_block):
                        value_else = yield orelse
                        self.be.create_yield_op([x for x in value_else])

                return if_op.results

            case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                # Cast operands to expected type from pattern match
                loop_operands = cast("list[SExpr]", operands)
                # process operands
                loop_operand_vals: list[Any] = []
                for op in loop_operands:
                    loop_operand_vals.append(cast(Any, (yield op)))

                loop_result_tys: list[Any] = []
                for op in loop_operand_vals:
                    loop_result_tys.append(cast(Any, op).type)

                while_op = self.be.create_while_op(
                    result_types=loop_result_tys,
                    init_args=[op for op in loop_operand_vals],
                )
                before_block = while_op.before.blocks.append(*loop_result_tys)
                after_block = while_op.after.blocks.append(*loop_result_tys)
                new_ops = before_block.arguments

                # Before Region
                with self.be.InsertionPoint(before_block), state.push(new_ops):
                    values = yield body
                    self.be.create_condition_op(
                        condition=values[0], args=[val for val in values[1:]]
                    )

                # After Region
                with self.be.InsertionPoint(after_block):
                    self.be.create_yield_op(after_block.arguments)

                while_op_res = self.be.get_scf_op_results(while_op)
                return while_op_res

            case rg.Undef(name):
                # HACK
                return self.be.create_constant_i32(0)

            case sg.CallFQN(
                fqn=sg.FQN() as callee_fqn, io=io_val, args=args_vals
            ):
                mdmap = self.mdmap

                [callee_ti] = mdmap.lookup_typeinfo(callee_fqn)

                type_expr: sg.TypeExpr = cast(sg.TypeExpr, callee_ti.type_expr)

                argtys: list[Any] = []
                for arg in args_vals:
                    [ti] = mdmap.lookup_typeinfo(arg)
                    arg_types = self.be.lower_type(
                        cast(sg.TypeExpr, ti.type_expr)
                    )
                    argtys.extend(arg_types)

                lowered_args = []
                for arg in args_vals:
                    lowered_args.append((yield arg))

                c_name = FQN(callee_fqn.fullname).c_name

                callee_fqn_obj: FQN = FQN(callee_fqn.fullname)

                if callee_fqn_obj.namespace.fullname == "mlir::op":
                    TODO("XXX: hardcode support of MLIR::OP ")
                    result_types_tuple = self.be.lower_type(
                        cast(sg.TypeExpr, type_expr.args[0])
                    )

                    res = self.be.handle_mlir_op(
                        callee_fqn_obj.symbol_name,
                        result_types_tuple,
                        lowered_args,
                    )
                    owner = getattr(res, "owner", None)
                    if owner is not None:
                        assert owner.verify()
                    return [io_val, res]
                    # self.declare_builtins(c_name, argtys, [resty])
                elif callee_fqn_obj.namespace.fullname == "mlir::asm":
                    result_types = list(
                        self.be.lower_type(
                            cast(sg.TypeExpr, type_expr.args[0])
                        )
                    )
                    res = self._handle_mlir_asm(
                        callee_fqn_obj.symbol_name,
                        result_types,
                        lowered_args,
                    )
                    return [io_val, res]
                elif (
                    callee_fqn_obj.namespace.fullname == "mlir"
                    and callee_fqn_obj.parts[1].name == "unpack"
                ):
                    idx = int(callee_fqn_obj.parts[-1].suffix)
                    [operand] = lowered_args
                    return [io_val, operand[idx]]
                else:
                    result_types_tuple = self.be.lower_type(
                        cast(sg.TypeExpr, type_expr.args[0])
                    )
                    result_types = list(result_types_tuple)
                # if callee_fqn.fullname == "builtins::print_object":
                #     TODO("XXX: hardcode support of builtins::print_object ")
                #     with self.module_body:
                #         self.declare_builtins(c_name, argtys, [resty])
                #         fntype = ir.FunctionType.get(argtys, [resty])
                #         func.FuncOp(
                #             name=c_name,
                #             type=fntype,
                #             visibility="private",
                #         )

                call = self.be.create_function_call(
                    result_types, c_name, lowered_args
                )
                return [io_val, call]
            case rg.PyNone():
                return self.be.create_none()
            case _:
                raise NotImplementedError(
                    expr, type(expr), ase.as_tuple(expr, depth=3)
                )

    def _handle_mlir_asm(self, mlir_op: str, result_types, args):
        mlir_op = decode_asm_operation(mlir_op)
        try:
            first_split = mlir_op.index("$")
        except ValueError:
            pass
        else:
            mlir_op = mlir_op[:first_split]

        opname, _, attr = mlir_op.partition(" ")
        return self.be.create_mlir_asm(opname, attr, result_types, args)

    def get_port_type(self, port) -> Any:
        if port.name == internal_prefix("io"):
            ty = self.be.io_type
        else:
            ty = self.be.get_ll_type(port.value, self.mdmap)
        return ty

    def declare_builtins(self, sym_name, argtypes, restypes):
        if sym_name in self._declared:
            return self._declared[sym_name]

        with self.module_body:
            ret = self._declared[sym_name] = (
                self.be.create_function_declaration(
                    name=sym_name,
                    arg_types=argtypes,
                    result_types=restypes,
                    visibility="private",
                )
            )
        return ret
