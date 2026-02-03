from __future__ import annotations
from sealir import ase
from sealir.rvsdg import grammar as rg, format_rvsdg

from nbcc.compiler import egraph_conversion, EGraph
from nbcc.egraph.region_port_pruning import define_rule


def make_example():
    with ase.Tape() as tape:
        g = rg.Grammar(tape=tape)
        predicate = g.write(rg.PyInt(1))
        attrs = g.write(rg.Attrs(()))

        value_a = g.write(rg.PyInt(1))
        value_b = g.write(rg.PyInt(2))
        value_c = g.write(rg.PyInt(3))
        value_d = g.write(rg.PyInt(4))
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
                        ports=(
                            g.write(rg.Port(name="a", value=value_a)),
                            g.write(rg.Port(name="b", value=value_b)),
                            g.write(
                                rg.Port(
                                    name="c",
                                    value=g.write(rg.Unpack(val=rb, idx=0)),
                                )
                            ),
                            g.write(
                                rg.Port(
                                    name="d",
                                    value=g.write(rg.Unpack(val=rb, idx=2)),
                                )
                            ),
                        ),
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
                        ports=(
                            g.write(rg.Port(name="a", value=value_a)),
                            g.write(rg.Port(name="b", value=value_b)),
                            g.write(
                                rg.Port(
                                    name="c",
                                    value=g.write(rg.Unpack(val=rb, idx=0)),
                                )
                            ),
                            g.write(
                                rg.Port(
                                    name="d",
                                    value=g.write(rg.Unpack(val=rb, idx=1)),
                                )
                            ),
                        ),
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
    from egglog import ruleset, Vec, i64, rewrite
    from sealir.eqsat.rvsdg_eqsat import TermList, Term

    @ruleset
    def extras(termlist: TermList, terms: Vec[Term], i: i64):
        yield rewrite(termlist[i]).to(
            terms[i],
            termlist == TermList(terms),
        )

    return extras


def test_output_port_pruning():
    from egglog import eq, expr_parts
    from sealir.eqsat.rvsdg_eqsat import termlist, Term
    from sealir.eqsat.py_eqsat import Py_Tuple

    root, ifelse = make_example()
    print(format_rvsdg(root))

    memo = egraph_conversion(root)
    egraph = EGraph()
    root_enode = egraph.let("root", memo[root])

    egraph.run(define_rule() + _extra_rules_to_equate_termlist__getitem__())
    # egraph.display(n_inline_leaves=1)
    print(out := egraph.extract(root_enode))

    ifelse_enode = memo[ifelse].term

    egraph.check(
        eq(out).to(
            Py_Tuple(
                termlist(
                    ifelse_enode.getPort(0),
                    ifelse_enode.getPort(1),
                    Term.LiteralI64(1),
                    ifelse_enode.getPort(3),
                )
            )
        )
    )
