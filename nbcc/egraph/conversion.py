from sealir.eqsat.rvsdg_extract_details import EGraphToRVSDG as _EGraphToRVSDG

from ..frontend import grammar as sg


def emit_node(fn):
    def wrapper(self, *args, children, grm, **kwargs):
        return grm.write(fn(**children))

    return wrapper


def match_type(typename: str, op_expect: str):
    def condition(self, key, children, grm, **kwargs):
        nodes = self.gdct["nodes"]
        node = nodes[key]
        eclass = node["eclass"]
        op = node["op"]
        node_type = self._parse_type(self.gdct["class_data"][eclass]["type"])
        return (
            node_type.prefix == "nbcc.egraph.rules"
            and node_type.name == typename
            and op == op_expect
        )

    return condition


class ExtendEGraphToRVSDG(_EGraphToRVSDG):
    # Override base class grammar type expectation - nbcc.frontend.grammar.Grammar
    # is compatible with sealir.rvsdg.grammar.Grammar for dispatch purposes
    grammar: type[sg.Grammar] = sg.Grammar  # type: ignore[assignment]

    @_EGraphToRVSDG._dispatch_term.extend
    @staticmethod
    def _dispatch_term(disp):

        def op_matches(op_expect):
            def f(*args, op, **kwargs):
                return op == op_expect

            return f

        @disp.case(op_matches("Op_i32_add"))
        @emit_node
        def _(lhs, rhs):
            return sg.BuiltinOp(opname="i32_add", args=(lhs, rhs))

        @disp.case(op_matches("Op_i32_sub"))
        @emit_node
        def _(lhs, rhs):
            return sg.BuiltinOp(opname="i32_sub", args=(lhs, rhs))

        @disp.case(op_matches("Op_i32_gt"))
        @emit_node
        def _(lhs, rhs):
            return sg.BuiltinOp(opname="i32_gt", args=(lhs, rhs))

        @disp.case(op_matches("Op_i32_lt"))
        @emit_node
        def _(lhs, rhs):
            return sg.BuiltinOp(opname="i32_lt", args=(lhs, rhs))

        @disp.case(op_matches("Op_i32_not"))
        @emit_node
        def _(operand):
            return sg.BuiltinOp(opname="i32_not", args=(operand,))

        @disp.case(op_matches("Builtin_print_i32"))
        @emit_node
        def _(io, arg):
            return sg.BuiltinOp(opname="print_i32", args=(io, arg))

        @disp.case(op_matches("Builtin_print_str"))
        @emit_node
        def _(io, arg):
            return sg.BuiltinOp(opname="print_str", args=(io, arg))

        @disp.case(op_matches("Builtin_struct__make__"))
        @emit_node
        def _(args):
            return sg.BuiltinOp(opname="struct_make", args=tuple(args))

        @disp.case(op_matches("Builtin_struct__lift__"))
        @emit_node
        def _(args):
            return sg.BuiltinOp(opname="struct_lift", args=tuple(args))

        @disp.case(op_matches("Builtin_struct__unlift__"))
        @emit_node
        def _(args):
            return sg.BuiltinOp(opname="struct_unlift", args=tuple(args))

        @disp.case(op_matches("Builtin_struct__get_field__"))
        @emit_node
        def _(struct, pos):
            return sg.BuiltinOp(opname="struct_get", args=(struct, pos))

        @disp.case(op_matches("CallFQN"))
        @emit_node
        def _(fqn, io, args):
            return sg.CallFQN(fqn=fqn, io=io, args=args)

        @disp.case(op_matches("CalleeFQN"))
        @emit_node
        def _(fullname):
            return sg.FQN(fullname)

        @disp.case(op_matches("FQN.function"))
        @emit_node
        def _(fullname):
            return sg.FQN(fullname)

    @_EGraphToRVSDG._dispatch_function.extend
    @staticmethod
    def _dispatch_function(disp):
        @disp.case(match_type("TypeExpr", "TypeExpr.simple"))
        @emit_node
        def _(name: str):
            return sg.TypeExpr(name=name, args=())

        @disp.case(match_type("TypeExpr", "TypeExpr.function"))
        @emit_node
        def _(args):
            return sg.TypeExpr(name=".function", args=args.children)

        @disp.case(match_type("Metadata", "Metadata.typeinfo"))
        @emit_node
        def _(value, type_expr):
            return sg.TypeInfo(value=value, type_expr=type_expr)

        @disp.case(match_type("Metadata", "Metadata.irtag"))
        @emit_node
        def _(value, tag, data):
            return sg.IRTag(value=value, tag=tag, data=data)

        @disp.case(match_type("IRTagData", "IRTagData"))
        @emit_node
        def _(key: str, value: str):
            return sg.IRTagData(key=key, value=value)
