from __future__ import annotations

from egglog import eq, expr_parts
from sealir import ase
from sealir.eqsat.py_eqsat import Py_Tuple
from sealir.eqsat.rvsdg_eqsat import Term, termlist
from sealir.rvsdg import format_rvsdg
from sealir.rvsdg import grammar as rg

from nbcc.compiler import EGraph, egraph_conversion
from nbcc.egraph.region_port_pruning import define_rule


def make_example(*, then_ports, else_ports):
    def make_ports(g, port_args):
        portvalues = []
        for arg in port_args:
            if isinstance(arg, int):
                portvalues.append(g.write(rg.Unpack(val=rb, idx=arg)))
            else:
                portvalues.append(arg)
        ports = []
        for i, v in enumerate(portvalues):
            ports.append(g.write(rg.Port(name=f"p{i}", value=v)))
        return tuple(ports)

    with ase.Tape() as tape:
        g = rg.Grammar(tape=tape)
        predicate = g.write(rg.PyInt(1))
        attrs = g.write(rg.Attrs(()))

        value_a = g.write(rg.PyInt(1))
        value_b = g.write(rg.PyInt(2))
        value_c = g.write(rg.PyInt(3))
        value_d = g.write(rg.PyInt(4))
        constants = [value_a, value_b, value_c, value_d]
        then_ports = [a if b is None else b
                      for a, b, in zip(constants, then_ports, strict=True)]
        else_ports = [a if b is None else b
                      for a, b, in zip(constants, else_ports, strict=True)]
        ifelse = g.write(
            rg.IfElse(
                cond=predicate,
                body=g.write(
                    rg.RegionEnd(
                        begin=(
                            rb := g.write(
                                rg.RegionBegin(
                                    attrs=attrs, inports=("a", "b", "c")
                                )
                            )
                        ),
                        ports=make_ports(g, then_ports),
                    ),
                ),
                orelse=g.write(
                    rg.RegionEnd(
                        begin=(
                            rb := g.write(
                                rg.RegionBegin(
                                    attrs=attrs, inports=("a", "b", "c")
                                )
                            )
                        ),
                        ports=make_ports(g, else_ports),
                    )
                ),
                operands=(value_a, value_b, value_c),
            )
        )

    outs = [
        g.write(rg.Unpack(val=ifelse, idx=i))
        for i in range(len(ifelse.body.ports))
    ]

    root = g.write(rg.PyTuple(elems=tuple(outs)))
    return root, ifelse


def _extra_rules_to_equate_termlist__getitem__():
    ###### extra
    from egglog import Vec, i64, rewrite, ruleset
    from sealir.eqsat.rvsdg_eqsat import Term, TermList

    @ruleset
    def extras(termlist: TermList, terms: Vec[Term], i: i64):
        yield rewrite(termlist[i]).to(
            terms[i],
            termlist == TermList(terms),
        )

    return extras


def _run_output_port_pruning_test(then_ports, else_ports, expected_terms_builder):
    """Helper function to run output port pruning tests with given parameters.

    Args:
        then_ports: Port configuration for the then branch
        else_ports: Port configuration for the else branch
        expected_terms_builder: Function that takes ifelse_enode and returns expected terms list
    """
    root, ifelse = make_example(
        then_ports=then_ports,
        else_ports=else_ports,
    )
    print(format_rvsdg(root))

    memo = egraph_conversion(root)
    egraph = EGraph()
    root_enode = egraph.let("root", memo[root])

    egraph.run(define_rule() + _extra_rules_to_equate_termlist__getitem__())
    # egraph.display(n_inline_leaves=1)
    print(out := egraph.extract(root_enode))

    ifelse_enode = memo[ifelse].term

    expected_terms = expected_terms_builder(ifelse_enode)
    egraph.check(
        eq(out).to(
            Py_Tuple(
                termlist(*expected_terms)
            )
        )
    )


def test_output_port_pruning_1():
    def build_expected_terms(ifelse_enode):
        return [
            ifelse_enode.getPort(0),
            ifelse_enode.getPort(1),
            Term.LiteralI64(1),
            ifelse_enode.getPort(3),
        ]

    _run_output_port_pruning_test(
        then_ports=[None, None, 0, 1],
        else_ports=[None, None, 0, 2],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_2():
    def build_expected_terms(ifelse_enode):
        return [
            ifelse_enode.getPort(0),
            ifelse_enode.getPort(1),
            Term.LiteralI64(1),
            Term.LiteralI64(2),
        ]

    _run_output_port_pruning_test(
        then_ports=[None, None, 0, 1],
        else_ports=[None, None, 0, 1],
        expected_terms_builder=build_expected_terms
    )



def test_output_port_pruning_3():
    def build_expected_terms(ifelse_enode):
        return [
            Term.LiteralI64(3),
            ifelse_enode.getPort(1),
            Term.LiteralI64(1),
            Term.LiteralI64(2),
        ]

    _run_output_port_pruning_test(
        then_ports=[2, None, 0, 1],
        else_ports=[2, None, 0, 1],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_4():
    """Test with all constant values in both branches."""
    def build_expected_terms(ifelse_enode):
        return [
            ifelse_enode.getPort(0),
            ifelse_enode.getPort(1),
            ifelse_enode.getPort(2),
            ifelse_enode.getPort(3),
        ]

    _run_output_port_pruning_test(
        then_ports=[None, None, None, None],
        else_ports=[None, None, None, None],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_5():
    """Test with mixed constants and port references, different in each branch."""
    def build_expected_terms(ifelse_enode):
        return [
            Term.LiteralI64(3),
            ifelse_enode.getPort(1),
            ifelse_enode.getPort(2),
            Term.LiteralI64(2),
        ]

    _run_output_port_pruning_test(
        then_ports=[2, None, None, 1],
        else_ports=[2, None, None, 1],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_6():
    """Test with all port references, same pattern."""
    def build_expected_terms(ifelse_enode):
        return [
            Term.LiteralI64(1),
            Term.LiteralI64(2),
            Term.LiteralI64(3),
            ifelse_enode.getPort(3),
        ]

    _run_output_port_pruning_test(
        then_ports=[0, 1, 2, None],
        else_ports=[0, 1, 2, None],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_7():
    """Test with alternating constants and port references."""
    def build_expected_terms(ifelse_enode):
        return [
            ifelse_enode.getPort(0),
            Term.LiteralI64(1),
            ifelse_enode.getPort(2),
            Term.LiteralI64(2),
        ]

    _run_output_port_pruning_test(
        then_ports=[None, 0, None, 1],
        else_ports=[None, 0, None, 1],
        expected_terms_builder=build_expected_terms
    )


def test_output_port_pruning_8():
    """Test with different port orderings between branches."""
    def build_expected_terms(ifelse_enode):
        return [
            ifelse_enode.getPort(0),
            ifelse_enode.getPort(1),
            ifelse_enode.getPort(2),
            ifelse_enode.getPort(3),
        ]

    _run_output_port_pruning_test(
        then_ports=[0, 1, 2, 0],
        else_ports=[1, 0, 2, 1],
        expected_terms_builder=build_expected_terms
    )
