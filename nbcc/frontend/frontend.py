from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence, cast
from pathlib import Path

from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    SyntheticAssignment,
    SyntheticExitBranch,
    SyntheticExitingLatch,
    SyntheticFill,
    SyntheticHead,
    SyntheticReturn,
    SyntheticTail,
)
from numba_scfg.core.datastructures.scfg import SCFG
from sealir import ase
from sealir.rvsdg import format_rvsdg
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix
from spy.fqn import FQN
from spy.vm.function import W_ASTFunc, W_BuiltinFunc, W_FuncType, W_Func
from spy.vm.struct import W_StructType
from spy.vm.object import W_Type
from spy.vm.vm import SPyVM
from spy.location import Loc
from spy.vm.module import W_Module

from . import grammar as sg
from .restructure import SCFG, SpyBasicBlock, _SpyScfgRenderer, restructure
from .spy_ast import Node, convert_to_node
from nbcc.developer import TODO
from . import extra_spy_builtins


@dataclass(frozen=True)
class FunctionInfo:
    fqn: FQN
    region: SCFG
    metadata: list[ase.SExpr]


class TranslationUnit:
    _symtabs: dict[FQN, FunctionInfo]
    _structs: dict[FQN, W_StructType]
    _builtins: dict[FQN, Any]

    def __init__(self):
        self._symtabs = {}
        self._structs = {}
        self._builtins = {}
        self._irtags = {}

    def add_function(self, fi: FunctionInfo) -> None:
        self._symtabs[fi.fqn] = fi

    def add_struct_type(self, fqn: FQN, obj) -> None:
        self._structs[fqn] = obj

    def add_builtin(self, fqn: FQN, obj) -> None:
        self._builtins[fqn] = obj

    def is_struct(self, fqn: FQN) -> bool:
        return fqn in self._structs

    def get_struct(self, fqn: FQN) -> W_StructType:
        return self._structs[fqn]

    def get_function(self, fqn: FQN) -> FunctionInfo:
        return self._symtabs[fqn]

    def list_functions(self) -> list[FQN]:
        return list(self._symtabs)

    def list_builtins(self) -> list[FQN]:
        return list(self._builtins)

    def __repr__(self):
        cname = self.__class__.__name__
        syms = ", ".join(map(str, self._symtabs))
        return f"{cname}([{syms}])"


def redshift(filename: str | Path) -> tuple[SPyVM, W_Module]:
    """
    Perform redshift on the given file

    NOTE: this is adapted from `spy/interop.py`
    """
    filename = Path(filename)
    modname = filename.stem
    builddir = filename.parent
    vm = SPyVM()
    # Install custom builtins here
    vm.make_module(extra_spy_builtins.MLIR)
    # End custom builtins
    vm.path.append(str(builddir))
    w_mod = vm.import_(modname)
    vm.redshift(error_mode="eager")
    return vm, w_mod


def frontend(filename: str, *, view: bool = False) -> TranslationUnit:
    vm, w_mod = redshift(filename)

    tu = TranslationUnit()

    symtab: dict[FQN, Node] = {}
    fn_type: dict[FQN, W_FuncType] = {}

    fqn_to_local_type = {}
    vm.pp_globals()
    for fqn, w_obj in vm.fqns_by_modname(w_mod.name):
        print("?" * 80)
        print(fqn, "|", w_obj, "::", type(w_obj))
        if isinstance(w_obj, W_ASTFunc):
            if w_obj.locals_types_w is not None:
                node = convert_to_node(w_obj.funcdef, vm=vm).insert_fqn(fqn)
                symtab[fqn] = node
                fn_type[fqn] = w_obj.w_functype
                fqn_to_local_type[fqn] = w_obj.locals_types_w
        elif isinstance(w_obj, W_BuiltinFunc):
            tu.add_builtin(fqn, w_obj)
        elif isinstance(w_obj, W_StructType):
            tu.add_struct_type(fqn, w_obj)
        else:
            breakpoint()

    # restructure
    for fqn, func_node in symtab.items():
        print("/" * 80)
        print("///TRANSLATE", fqn)

        scfg = restructure(fqn.fullname, func_node)
        if view:
            _SpyScfgRenderer(scfg).view()
        region, mds = convert_to_sexpr(
            func_node,
            scfg,
            fn_type[fqn],
            fqn_to_local_type[fqn],
            fqn_to_local_type,
            vm,
        )
        print(format_rvsdg(region))
        tu.add_function(FunctionInfo(fqn=fqn, region=region, metadata=mds))

    return tu


def convert_to_sexpr(
    func_node: Node,
    scfg: SCFG,
    fn_type: W_FuncType,
    local_types: dict[str, W_Type],
    global_ns: dict[FQN, dict[str, W_Type]],
    vm: SPyVM,
) -> tuple[SCFG, list]:
    with ase.Tape() as tape:
        cts = ConvertToSExpr(tape, local_types, global_ns, vm)
        with cts.setup_function(func_node) as rb:
            cts.handle_region(scfg)

        region = cts.close_function(rb, func_node, fn_type)
        return region, cts._metadata


@dataclass(frozen=True)
class Scope:
    vardefs: dict[str, FQN] = field(init=False, default_factory=dict)
    local_vars: dict[str, ase.SExpr] = field(init=False, default_factory=dict)


@dataclass(frozen=True)
class ConversionContext:
    grm: sg.Grammar
    local_types: dict[str, W_Type]
    global_ns: dict[FQN, dict[str, W_Type]]
    scope_stack: list = field(init=False, default_factory=list)
    scope_map: dict[Any, Scope] = (
        field(  # Keys are wrapped NamedSExpr[Grammar, RegionBegin]
            init=False, default_factory=dict
        )
    )

    @property
    def loopcond_name(self) -> str:
        d = len(self.scope_stack)
        return internal_prefix(f"_loopcond_{d:03x}")

    @property
    def scope(self) -> Scope:
        return self.scope_stack[-1]

    def store_local(self, target: str, expr: ase.SExpr) -> None:
        assert isinstance(expr, ase.SExpr)
        self.scope.local_vars[target] = expr

    def load_local(self, target: str) -> ase.SExpr:
        return self.scope.local_vars[target]

    def get_io(self) -> ase.SExpr:
        out = self.load_local(internal_prefix("io"))
        assert isinstance(out, ase.SExpr)
        return out

    def set_io(self, value: ase.SExpr) -> None:
        assert isinstance(value, ase.SExpr)
        self.store_local(internal_prefix("io"), value)

    def insert_io_node(self, node: rg.grammar.Rule) -> ase.SExpr:
        grm = self.grm
        written = grm.write(node)
        io, res = (grm.write(rg.Unpack(val=written, idx=i)) for i in range(2))
        self.set_io(io)
        return res

    def update_scope(self, expr: ase.SExpr, vars: Sequence[str]) -> None:
        grm = self.grm

        for i, k in enumerate(vars):
            self.store_local(k, grm.write(rg.Unpack(val=expr, idx=i)))

    # Unwrapping utilities for NamedSExpr -> Rule conversion
    def unwrap_type_expr(self, wrapped_te) -> sg.TypeExpr:
        """Unwrap NamedSExpr[Grammar, TypeExpr] to TypeExpr - still needed for emit_function_type"""
        return sg.TypeExpr(name=wrapped_te.name, args=wrapped_te.args)

    # Removed unused unwrapping utilities that are no longer needed after type annotation fixes

    @contextmanager
    def new_region(self, region_parameters: Sequence[str]):

        write = self.grm.write
        rb = write(
            rg.RegionBegin(
                attrs=write(rg.Attrs(())),
                inports=tuple(region_parameters),
            )
        )

        scope = Scope()
        self.scope_map[rb] = scope
        self.scope_stack.append(scope)

        self.initialize_scope(rb)

        yield rb

        self.scope_stack.pop()

    def initialize_scope(
        self, rb
    ):  # rb is wrapped NamedSExpr[Grammar, RegionBegin]
        write = self.grm.write
        for i, k in enumerate(rb.inports):
            self.store_local(k, write(rg.Unpack(val=rb, idx=i)))

    def compute_updated_vars(
        self, rb
    ) -> set[str]:  # rb is wrapped NamedSExpr[Grammar, RegionBegin]
        return set(self.scope_map[rb].local_vars.keys())

    def close_region(
        self, rb, expected_vars: set[str]
    ) -> ase.SExpr:  # rb is wrapped NamedSExpr[Grammar, RegionBegin]
        scope = self.scope_map[rb]

        write = self.grm.write
        ports: list[ase.SExpr] = []
        for k in sorted(expected_vars):
            v: ase.SExpr
            if k not in scope.local_vars:
                v = write(rg.Undef(name=k))
            else:
                v = scope.local_vars[k]
            p = write(rg.Port(name=k, value=v))
            ports.append(p)

        return write(rg.RegionEnd(begin=rb, ports=tuple(ports)))

    def get_scope_as_operands(self) -> tuple[ase.SExpr, ...]:
        operands = []
        for _, v in sorted(self.scope.local_vars.items()):
            operands.append(v)
        return tuple(operands)

    def get_scope_as_parameters(self) -> tuple[str, ...]:
        return tuple(sorted(self.scope.local_vars))


class ConvertToSExpr:
    def __init__(
        self,
        tape: ase.Tape,
        local_types: dict[str, W_Type],
        global_ns: dict[FQN, dict[str, W_Type]],
        vm: SPyVM,
    ):
        self._tape = tape
        self._context = ConversionContext(
            grm=sg.Grammar(self._tape),
            local_types=local_types,
            global_ns=global_ns,
        )
        self._metadata: list[ase.SExpr] = []
        self._local_types = local_types
        self._global_ns = global_ns
        self._vm = vm
        self._args: list[ase.SExpr] = []
        self._memo_fntypes: dict[Any, Any] = {}

    def insert_typeinfo(self, value: ase.SExpr, type_expr: ase.SExpr) -> None:
        self._metadata.append(
            self._context.grm.write(
                sg.TypeInfo(value=value, type_expr=type_expr)
            )
        )

    def insert_func_typeinfo(
        self, value: ase.SExpr, functype: W_FuncType
    ) -> None:
        tys = [self.emit_type(param.w_T) for param in functype.params]
        restype = self.emit_type(functype.w_restype)
        typexpr = self._context.grm.write(
            sg.TypeExpr(name=".function", args=(restype, *tys))
        )
        return self.insert_typeinfo(value, typexpr)

    def emit_function_type(
        self, resty: sg.TypeExpr, *args: sg.TypeExpr
    ) -> sg.TypeExpr:
        # Convert TypeExpr args to SExprs first
        resty_sexpr = self._context.grm.write(resty)
        args_sexprs = tuple(self._context.grm.write(arg) for arg in args)
        written_type = self._context.grm.write(
            sg.TypeExpr(name=".function", args=(resty_sexpr, *args_sexprs))
        )
        return self._context.unwrap_type_expr(written_type)

    def emit_type(self, ty: W_Type):
        if fqn := ty.fqn:
            return self._context.grm.write(
                sg.TypeExpr(name=fqn.fullname, args=())
            )
        else:
            print("???ty", ty, type(ty))
            breakpoint()

    @contextmanager
    def setup_function(self, func_node: Node):
        argmap = {}
        match func_node:
            case Node("FuncDef", args=args):
                grm = self._context.grm
                for i, arg in enumerate(args):
                    # The names must be defined in local_types
                    assert arg.name in self._local_types
                    arg_sexpr = grm.write(rg.ArgRef(idx=i, name=arg.name))
                    self._args.append(arg_sexpr)
                    argmap[arg.name] = arg_sexpr
            case _:
                raise ValueError(func_node)

        ctx = self._context
        with ctx.new_region([internal_prefix("io")]) as rb:
            for k, v in argmap.items():
                self._context.store_local(k, v)
            yield rb

    def close_function(
        self, rb: rg.RegionBegin, func_node: Node, fn_type: W_FuncType
    ) -> rg.SExpr:
        ctx = self._context
        vars = {internal_prefix("io"), internal_prefix("ret")}

        assert len(func_node.args) == len(self._args)

        # redirect return value
        scope_map = ctx.scope_map[rb]
        if not (retval := scope_map.local_vars.get("__scfg_return_value__")):
            retval = ctx.grm.write(rg.PyNone())

        scope_map.local_vars[internal_prefix("ret")] = retval
        vars.add(internal_prefix("ret"))

        argtypes = []
        for arg_sexpr, param in zip(self._args, fn_type.params, strict=True):
            fqn = param.w_T.fqn
            typexpr = ctx.grm.write(sg.TypeExpr(name=fqn.fullname, args=()))
            argtypes.append(typexpr)
            self.insert_typeinfo(arg_sexpr, typexpr)
        written_args = ctx.grm.write(rg.Args(arguments=tuple(argtypes)))

        retval = scope_map.local_vars[internal_prefix("ret")]
        ret_tyname = fn_type.w_restype.fqn.fullname
        restype = ctx.grm.write(sg.TypeExpr(name=ret_tyname, args=()))
        self.insert_typeinfo(retval, type_expr=restype)

        body = ctx.close_region(rb, vars)
        fnty = ctx.grm.write(
            sg.TypeExpr(name=".function", args=tuple(argtypes))
        )
        self.insert_typeinfo(body, fnty)

        # add IRtags
        irtag = self._vm.irtags[func_node.fqn]
        if irtag.tag:
            datalist = []
            for k, v in irtag.data.items():
                datalist.append(ctx.grm.write(sg.IRTagData(key=k, value=v)))

            self._metadata.append(
                ctx.grm.write(
                    sg.IRTag(value=body, tag=irtag.tag, data=tuple(datalist))
                )
            )

        return ctx.grm.write(
            rg.Func(
                fname=func_node.fqn.fullname,
                args=written_args,
                body=body,
            )
        )

    def handle_region(self, scfg: SCFG):
        ctx = self._context
        crv = list(scfg.concealed_region_view.items())
        by_kinds = defaultdict(list)
        for _, block in crv:
            kind = getattr(block, "kind", None)
            by_kinds[kind].append(block)

        print("--by-kinds", [(k, len(vs)) for k, vs in by_kinds.items()])
        if "branch" in by_kinds:
            [head_block] = by_kinds["head"]
            [then_block, else_block] = by_kinds["branch"]
            [tail_block] = by_kinds["tail"]

            test_expr = self.codegen(head_block)

            operands = ctx.get_scope_as_operands()

            with ctx.new_region(ctx.get_scope_as_parameters()) as rb_then:
                self.codegen(then_block)

            with ctx.new_region(ctx.get_scope_as_parameters()) as rb_else:
                self.codegen(else_block)

            updated_vars = ctx.compute_updated_vars(rb_then)
            updated_vars |= ctx.compute_updated_vars(rb_else)

            region_then = ctx.close_region(rb_then, updated_vars)
            region_else = ctx.close_region(rb_else, updated_vars)

            # type metadata
            assert isinstance(region_then, rg.RegionEnd)
            assert isinstance(region_else, rg.RegionEnd)
            for region in (region_then, region_else):
                # Access ports from wrapped SExpr RegionEnd
                region_ports = cast(list[rg.Port], region.ports)
                for port_sexpr in region_ports:
                    if ty := self._local_types.get(port_sexpr.name):
                        typexpr = ctx.grm.write(
                            sg.TypeExpr(name=ty.fqn.fullname, args=())
                        )
                        self.insert_typeinfo(port_sexpr.value, typexpr)

            ifelse = ctx.grm.write(
                rg.IfElse(
                    cond=test_expr,
                    body=region_then,
                    orelse=region_else,
                    operands=operands,
                ),
            )
            ctx.update_scope(ifelse, sorted(updated_vars))
            return self.codegen(tail_block)

        else:
            for _, blk in crv:
                last = self.codegen(blk)
            return last

    def codegen(self, block: BasicBlock) -> ase.SExpr | None:
        ctx = self._context
        grm = ctx.grm
        match block:
            case RegionBlock():
                if isinstance(block.subregion, SCFG):
                    if block.kind == "loop":
                        operands = ctx.get_scope_as_operands()
                        operand_names = list(ctx.get_scope_as_parameters())
                        with ctx.new_region(operand_names) as loop_region:
                            self.handle_region(block.subregion)
                            loopcondvar = ctx.loopcond_name

                        updated_vars = ctx.compute_updated_vars(loop_region)
                        loop_end = ctx.close_region(loop_region, updated_vars)

                        # TODO: this should use a rewrite pass
                        #
                        # Redo the loop region so that the incoming ports
                        # matches the outgoing ports
                        new_vars = sorted(updated_vars - {loopcondvar})
                        with ctx.new_region(new_vars) as loop_region:
                            self.handle_region(block.subregion)
                            loopcondvar = ctx.loopcond_name

                        updated_vars = ctx.compute_updated_vars(loop_region)
                        loop_end = ctx.close_region(loop_region, updated_vars)

                        original = dict(zip(operand_names, operands))

                        new_operands = []
                        for k in new_vars:
                            if k in original:
                                new_operands.append(original[k])
                            else:
                                new_operands.append(grm.write(rg.Undef(k)))

                        loop = ctx.grm.write(
                            rg.Loop(
                                body=loop_end, operands=tuple(new_operands)
                            )
                        )

                        ctx.update_scope(
                            loop, sorted(updated_vars - {loopcondvar})
                        )
                        return None
                    else:
                        return self.handle_region(block.subregion)
                else:
                    assert block.kind != "loop"
                    return self.codegen(block.subregion)

            case SpyBasicBlock():
                if not block.body:
                    return None
                assert len(block.body) > 0
                last_expr: ase.SExpr
                for stmt in block.body:
                    last_expr = self.emit_statement(stmt)
                return last_expr

            case SyntheticAssignment():
                for k, v in block.variable_assignment.items():
                    match v:
                        case int(ival):
                            const = grm.write(rg.PyInt(ival))
                        case _:
                            raise ValueError(type(v))
                    ctx.store_local(k, const)
                    return None

            case SyntheticExitingLatch():
                io = ctx.get_io()
                loopcond = ctx.insert_io_node(
                    rg.PyUnaryOp(
                        op="not", io=io, operand=ctx.load_local(block.variable)
                    )
                )

                ctx.store_local(ctx.loopcond_name, loopcond)
                return None

            case SyntheticReturn():
                ctx.load_local("__scfg_return_value__")
                return ctx.get_io()
            case (
                SyntheticTail()
                | SyntheticHead()
                | SyntheticFill()
                | SyntheticExitBranch()
            ):
                # These are empty blocks
                return ctx.get_io()
            case _:
                raise AssertionError(type(block))

        raise AssertionError("unreachable", block)

    def emit_statement(self, stmt: Node) -> ase.SExpr:
        ctx = self._context
        grm = ctx.grm
        match stmt:
            case Node(
                "VarDef",
                kind=None,
                name=Node("StrConst", value=str(name)),
                type=Node(
                    "FQNConst", fqn=Node("literal", value=FQN() as type_fqn)
                ),
            ):
                ctx.scope.vardefs[name] = type_fqn
                return ctx.get_io()
            case Node(
                "AssignLocal",
                target=Node("StrConst", value=str(target)),
                value=rval,
            ):
                expr = self.emit_expression(rval)
                ctx.store_local(target, expr)
                # Debug info
                loc = self.emit_loc(stmt.loc.value)
                unloc = grm.write(rg.unknown_loc())
                md = grm.write(
                    rg.DbgValue(
                        name=target, value=expr, srcloc=loc, interloc=unloc
                    )
                )
                self._metadata.append(md)

                if ty := self._local_types.get(target):
                    typexpr = ctx.grm.write(
                        sg.TypeExpr(name=ty.fqn.fullname, args=())
                    )
                    self.insert_typeinfo(expr, typexpr)

                return expr

            case Node("StmtExpr", value=Node() as value):
                self.emit_expression(value)
                return ctx.get_io()
            case Node("Call"):
                return self.emit_expression(stmt)
            case Node("Return"):
                ret = self.emit_expression(stmt.value)
                ctx.store_local("__scfg_return_value__", ret)
                return ret
            case Node("Pass"):
                return ctx.get_io()
            case _:
                raise NotImplementedError(stmt)

    def emit_expression(self, node: Node) -> ase.SExpr:
        ctx = self._context
        grm = ctx.grm
        vm = self._vm
        match node:
            case Node("NameLocal"):
                return ctx.load_local(node.sym.name)
            case Node(
                "Call",
                func=Node(
                    "FQNConst", fqn=Node("literal", value=FQN() as callee_fqn)
                ),
                args=list(args),
            ):
                w_obj = vm.lookup_global(callee_fqn)
                assert w_obj is not None
                assert isinstance(w_obj, W_Func), type(w_obj)
                functype = w_obj.w_functype
                if "mlir::asm" == w_obj.fqn.namespace.fullname:
                    TODO(
                        "implement custom sexpr conversion so this can be plumbed through"
                    )
                    """
                    tags = vm.irtags[w_obj.fqn]
                    grm.write(sg.MLIR_asm(asm=tags.data['asm'], io))
                    """

                # if callee_fqn.fullname.startswith("mlir_tensor::"):
                #     breakpoint()
                callee = grm.write(
                    rg.PyLoadGlobal(
                        io=ctx.get_io(), name=str(callee_fqn.fullname)
                    )
                )
                self.insert_func_typeinfo(callee, functype)
                res = ctx.insert_io_node(
                    rg.PyCall(
                        io=ctx.get_io(),
                        func=callee,
                        args=tuple(map(self.emit_expression, args)),
                    )
                )
                restype = functype.w_restype
                wrapped_restype = self.emit_type(restype)
                self.insert_typeinfo(res, wrapped_restype)
                return res
            case Node("Constant", value=int(ival)):
                cval = grm.write(rg.PyInt(ival))
                i32_wrapped = grm.write(
                    sg.TypeExpr(name="builtins::i32", args=())
                )
                self.insert_typeinfo(cval, i32_wrapped)
                return cval

            case Node("Constant", value=None):
                return grm.write(rg.PyNone())

            case Node("NameLocal"):
                return ctx.load_local(node.sym.name)

            case Node("StrConst", value=str(text)):
                return grm.write(rg.PyStr(text))
            case _:
                raise NotImplementedError(node)

    def emit_loc(self, loc_node: Loc) -> ase.SExpr:
        return self._context.grm.write(
            rg.Loc(
                filename=loc_node.filename,
                line_first=loc_node.line_start,
                line_last=loc_node.line_end,
                col_first=loc_node.col_start,
                col_last=loc_node.col_end,
            )
        )
