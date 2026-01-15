# mypy: disable-error-code="empty-body"
from __future__ import annotations

import warnings

import egglog
import sealir.eqsat.py_eqsat as py
import sealir.eqsat.rvsdg_eqsat as rvsdg
import sealir.rvsdg.grammar as rg
from egglog import union
from sealir.ase import SExpr

from ..frontend import grammar as sg
from nbcc.developer import TODO

Term = rvsdg.Term
TermList = rvsdg.TermList
_w = rvsdg.wildcard

i64 = egglog.i64


def egraph_optimize(egraph: egglog.EGraph):
    rule_schedule = make_schedule()
    egraph.run(rule_schedule)


def make_schedule() -> egglog.Schedule:
    return (
        ruleset_simplify_builtin_arith
        | ruleset_simplify_builtin_print
        | ruleset_typing
        | ruleset_call_fqn
    ).saturate()


def egraph_convert_metadata(mdlist: list[SExpr], memo) -> egglog.Vec[Metadata]:
    def gen(md):
        from sealir.eqsat.rvsdg_convert import WrapTerm

        match md:
            case rg.DbgValue(name, value, srcloc, interloc):
                TODO("skip DbgValue")
            case sg.TypeInfo(value=value, type_expr=typexpr):
                if value in memo:
                    anchor = memo[value]
                    if isinstance(anchor, WrapTerm):
                        anchor = anchor.term
                    return Metadata.typeinfo(anchor, type_expr=gen(typexpr))
            case sg.TypeExpr(str(name), tuple(args)):
                match name:
                    case ".function":
                        return TypeExpr.function(list(map(gen, args)))
                    case _ if not args:
                        if not args:
                            return TypeExpr.simple(name)

                raise NotImplementedError(md)
            case sg.IRTagData(key=str(key), value=str(value)):
                return IRTagData(key=key, value=value)
            case sg.IRTag(value=value, tag=str(tag), data=tuple(data)):
                if data:
                    data_node = egglog.Vec[IRTagData](*map(gen, data))
                else:
                    data_node = egglog.Vec[IRTagData].empty()
                anchor = memo[value]
                if isinstance(anchor, WrapTerm):
                    anchor = anchor.term
                return Metadata.irtag(value=anchor, tag=tag, data=data_node)
            case _:
                raise NotImplementedError(md)

    return egglog.Vec[Metadata](
        *filter(lambda x: x is not None, map(gen, mdlist))
    )


class TypeExpr(egglog.Expr):
    @classmethod
    def simple(cls, name: egglog.StringLike) -> TypeExpr: ...

    # @classmethod
    # def compound(cls, name: egglog.StringLike, *args: TypeExpr) -> TypeExpr:
    #     ...

    @classmethod
    def function(cls, args: egglog.Vec[TypeExpr]) -> TypeExpr: ...


class IRTagData(egglog.Expr):
    def __init__(self, key: egglog.StringLike, value: egglog.StringLike): ...


class Metadata(egglog.Expr):
    @classmethod
    def typeinfo(cls, value: Term, type_expr: TypeExpr) -> Metadata: ...

    @classmethod
    def irtag(
        cls, value: Term, tag: egglog.StringLike, data: egglog.Vec[IRTagData]
    ) -> Metadata: ...


@egglog.function
def Md_type_info(value: Term, typename: egglog.StringLike) -> Term: ...


@egglog.function
def Op_i32_add(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_sub(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_lt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_gt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_not(operand: Term) -> Term: ...


@egglog.function
def Builtin_print_i32(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_print_str(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_struct__make__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__lift__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__unlift__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__get_field__(struct: Term, pos: egglog.i64) -> Term: ...


class FQN(egglog.Expr):
    @classmethod
    def function(cls, fullname: egglog.StringLike) -> Term: ...


@egglog.function
def CallFQN(fqn: Term, io: Term, args: TermList) -> Term: ...


@egglog.function
def Load_FQN(fqn) -> Term: ...


@egglog.ruleset
def ruleset_simplify_builtin_arith(
    io: Term,
    operand: Term,
    lhs: Term,
    rhs: Term,
    argvec: egglog.Vec[Term],
    call: Term,
):
    BINOPS = {
        "operator::i32_add": Op_i32_add,
        "operator::i32_sub": Op_i32_sub,
        "operator::i32_gt": Op_i32_gt,
        "operator::i32_lt": Op_i32_lt,
    }
    for fname, ctor in BINOPS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == lhs,
            argvec[1] == rhs,
            egglog.eq(argvec.length()).to(i64(2)),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(ctor(lhs, rhs)),
        )

    # Handle PyUnaryOp not
    yield egglog.rule(
        call
        == py.Py_NotIO(
            io=io,
            term=operand,
        ),
    ).then(
        union(call.getPort(0)).with_(io),
        union(call.getPort(1)).with_(Op_i32_not(operand)),
    )


@egglog.ruleset
def ruleset_simplify_builtin_print(
    io: Term, printee: Term, argvec: egglog.Vec[Term], call: Term
):
    KNOWN_PRINTS = {
        "builtins::print_i32": Builtin_print_i32,
        "builtins::print_str": Builtin_print_str,
    }

    for fname, builtin_ctor in KNOWN_PRINTS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == printee,
            egglog.eq(argvec.length()).to(i64(1)),
        ).then(
            union(call.getPort(0)).with_(builtin_ctor(io, printee)),
            union(call.getPort(0)).with_(call.getPort(1)),
        )


def create_ruleset_struct__make__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__make__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__make__(args)),
        )

    ruleset_struct__make__.__name__ += fname

    return egglog.ruleset(ruleset_struct__make__)


def create_ruleset_struct__lift__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__lift__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__lift__(args)),
        )

    ruleset_struct__lift__.__name__ += fname

    return egglog.ruleset(ruleset_struct__lift__)


def create_ruleset_struct__unlift__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__unlift__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__unlift__(args)),
        )

    ruleset_struct__unlift__.__name__ += fname

    return egglog.ruleset(ruleset_struct__unlift__)


def create_ruleset_struct__get_field__(w_obj, field_pos: int):
    fname = w_obj.fqn.fullname

    def ruleset_struct__get_field__(
        io: Term,
        args: TermList,
        call: Term,
        argvec: egglog.Vec[Term],
        struct: Term,
    ):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=TermList(argvec),
            ),
            argvec[0] == struct,
            egglog.eq(argvec.length()).to(i64(1)),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(
                Builtin_struct__get_field__(struct, i64(field_pos))
            ),
        )

    ruleset_struct__get_field__.__name__ += fname
    return egglog.ruleset(ruleset_struct__get_field__)


@egglog.ruleset
def ruleset_typing(x: Term):
    if False:
        yield


@egglog.ruleset
def ruleset_call_fqn(
    io: Term,
    args: TermList,
    call: Term,
    fqn: egglog.String,
    callee: Term,
    functype: TypeExpr,
):
    yield egglog.rule(
        call
        == py.Py_Call(
            io=io,
            func=callee,
            args=args,
        ),
        callee == py.Py_LoadGlobal(io=_w(Term), name=fqn),
        Metadata.typeinfo(callee, functype),
    ).then(
        union(call).with_(CallFQN(fqn=callee, io=io, args=args)),
        union(callee).with_(FQN.function(fqn)),
    )
