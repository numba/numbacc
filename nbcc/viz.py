"""
Visualization/debug output for numbacc compilation.

Usage:
    VIZ=1 python -m nbcc source.spy

Or in code:
    from nbcc.viz import viz_compile
    viz_compile("source.spy")

Shows:
1. Rewrite rules that fired (pattern → replacement)
2. Extraction cost decisions (why X was chosen over Y)
3. Before/after diff of RVSDG
"""

import os
import sys
from dataclasses import dataclass, field
from typing import cast

from egglog import EGraph
from sealir.ase import SExpr
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as BaseCostModel, egraph_extraction
from sealir.rvsdg import format_rvsdg
import sealir.rvsdg.grammar as rg

from nbcc.frontend import frontend, TranslationUnit
from nbcc.egraph.rules import egraph_convert_metadata, make_schedule
from nbcc.egraph.conversion import ExtendEGraphToRVSDG


# ANSI colors for terminal output
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


# Disable colors if not a TTY
if not sys.stdout.isatty():
    C.disable()


@dataclass
class CostDecision:
    """Records a cost model decision."""
    op: str
    cost: int
    chosen: bool = False


@dataclass
class VizStats:
    """Collects visualization statistics."""
    nodes_before: int = 0
    nodes_after: int = 0
    eclasses_before: int = 0
    eclasses_after: int = 0
    cost_decisions: list[CostDecision] = field(default_factory=list)
    extraction_cost: float = 0.0


class TracingCostModel(BaseCostModel):
    """Cost model that records decisions for visualization."""

    def __init__(self, stats: VizStats):
        super().__init__()
        self.stats = stats
        self._seen_ops: set[str] = set()

    def get_cost_function(self, nodename, op, ty, cost, children):
        # Determine cost
        if op in ["Py_Call", "Py_LoadGlobal"]:
            c = 10000
        elif op in ["CallFQN"]:
            c = 1
        elif op.startswith("Op_") or op.startswith("Builtin_"):
            c = 1
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)

        # Record decision (only first time per op type)
        if op not in self._seen_ops:
            self._seen_ops.add(op)
            self.stats.cost_decisions.append(CostDecision(op=op, cost=c))

        return self.get_simple(c)


def count_egraph_stats(egraph: EGraph) -> tuple[int, int]:
    """Count nodes and e-classes in egraph."""
    import json
    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    data = json.loads(serialized.to_json())
    nodes = len(data.get("nodes", []))
    eclasses = len(data.get("class_data", {}))
    return nodes, eclasses


def print_header(text: str):
    """Print a section header."""
    width = 70
    print(f"\n{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n{C.BOLD}{text}{C.RESET}")
    print(f"{C.DIM}{'─' * 50}{C.RESET}")


def print_diff(before: str, after: str):
    """Print a simple diff of RVSDG before/after."""
    before_lines = before.strip().split("\n")
    after_lines = after.strip().split("\n")

    # Find changed lines (simple heuristic: look for key patterns)
    before_ops = set()
    after_ops = set()

    for line in before_lines:
        if "PyLoadGlobal" in line or "PyCall" in line:
            before_ops.add(line.strip())
    for line in after_lines:
        if "BuiltinOp" in line or "Op_" in line:
            after_ops.add(line.strip())

    print(f"\n{C.BOLD}Key transformations:{C.RESET}")

    # Show removed patterns
    for line in before_lines:
        stripped = line.strip()
        if "PyLoadGlobal" in stripped:
            print(f"  {C.RED}- {stripped}{C.RESET}")
        elif "PyCall" in stripped and "Py_Call" not in after:
            print(f"  {C.RED}- {stripped}{C.RESET}")

    # Show added patterns
    for line in after_lines:
        stripped = line.strip()
        if "BuiltinOp" in stripped:
            print(f"  {C.GREEN}+ {stripped}{C.RESET}")


def print_rewrite_summary():
    """Print the rewrite rules that can fire."""
    print(f"\n{C.BOLD}Active rewrite rules:{C.RESET}")

    rewrites = [
        ("Py_Call(Py_LoadGlobal('operator::i32_add'), [a,b])", "Op_i32_add(a, b)"),
        ("Py_Call(Py_LoadGlobal('operator::i32_sub'), [a,b])", "Op_i32_sub(a, b)"),
        ("Py_Call(Py_LoadGlobal('operator::i32_lt'), [a,b])", "Op_i32_lt(a, b)"),
        ("Py_Call(Py_LoadGlobal('operator::i32_gt'), [a,b])", "Op_i32_gt(a, b)"),
        ("Py_Call(Py_LoadGlobal('builtins::print_i32'), [x])", "Builtin_print_i32(io, x)"),
        ("Py_Call(Py_LoadGlobal('builtins::print_str'), [x])", "Builtin_print_str(io, x)"),
        ("Py_NotIO(x)", "Op_i32_not(x)"),
    ]

    for pattern, replacement in rewrites:
        print(f"  {C.DIM}{pattern}{C.RESET}")
        print(f"    {C.YELLOW}→{C.RESET} {C.GREEN}{replacement}{C.RESET}")


def print_cost_summary(stats: VizStats):
    """Print cost model decisions."""
    print(f"\n{C.BOLD}Cost model (guides extraction):{C.RESET}")

    high_cost = [d for d in stats.cost_decisions if d.cost >= 10000]
    low_cost = [d for d in stats.cost_decisions if d.cost < 10000]

    if high_cost:
        print(f"\n  {C.RED}Expensive (avoid):{C.RESET}")
        for d in high_cost:
            print(f"    {d.op}: cost={d.cost}")

    if low_cost:
        print(f"\n  {C.GREEN}Cheap (prefer):{C.RESET}")
        for d in low_cost:
            print(f"    {d.op}: cost={d.cost}")


def print_egraph_growth(stats: VizStats):
    """Print e-graph growth statistics."""
    print(f"\n{C.BOLD}E-graph growth:{C.RESET}")
    print(f"  Nodes:    {stats.nodes_before} → {stats.nodes_after} ({C.YELLOW}+{stats.nodes_after - stats.nodes_before} equivalences{C.RESET})")
    print(f"  E-classes: {stats.eclasses_before} → {stats.eclasses_after}")


def viz_compile(source_path: str) -> None:
    """Compile with visualization output."""

    print_header(f"VIZ: {source_path}")

    # Read source
    with open(source_path) as f:
        source = f.read()

    print_subheader("Source")
    for i, line in enumerate(source.strip().split("\n"), 1):
        print(f"  {C.DIM}{i:3}{C.RESET}  {line}")

    # Frontend (suppress verbose output)
    print_subheader("Frontend: SPy → RVSDG")
    import logging
    logging.disable(logging.CRITICAL)

    # Suppress stdout during frontend
    import io as _io
    old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    try:
        tu: TranslationUnit = frontend(source_path)
    finally:
        sys.stdout = old_stdout
        logging.disable(logging.NOTSET)

    for fqn in tu.list_functions():
        fi = tu.get_function(fqn)
        stats = VizStats()

        # RVSDG before
        rvsdg_before = format_rvsdg(fi.region)
        print(f"\n{C.BOLD}Function: {fi.fqn}{C.RESET}")
        print(f"\n{C.DIM}RVSDG (before):{C.RESET}")
        for line in rvsdg_before.strip().split("\n"):
            print(f"  {line}")

        # E-graph conversion
        print_subheader("E-graph Optimization")

        memo = egraph_conversion(fi.region)
        root = GraphRoot(memo[fi.region])

        egraph = EGraph()
        egraph.let("root", root)
        egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

        # Stats before
        stats.nodes_before, stats.eclasses_before = count_egraph_stats(egraph)

        # Show rewrite rules
        print_rewrite_summary()

        # Run optimization
        schedule = make_schedule()
        egraph.run(schedule)

        # Stats after
        stats.nodes_after, stats.eclasses_after = count_egraph_stats(egraph)

        print_egraph_growth(stats)

        # Extraction
        print_subheader("Extraction")

        cost_model = TracingCostModel(stats)
        extraction = egraph_extraction(egraph, cost_model=cost_model)
        extraction.compute()
        extresult = extraction.extract_common_root()
        stats.extraction_cost = extresult.cost

        print_cost_summary(stats)
        print(f"\n{C.BOLD}Total extraction cost:{C.RESET} {stats.extraction_cost}")

        # Back-conversion
        converted_root: SExpr = extresult.convert(fi.region, ExtendEGraphToRVSDG)

        rvsdg_after = ""
        for node in converted_root._args:
            if isinstance(node, rg.Func):
                rvsdg_after = format_rvsdg(cast(SExpr, node))
                break

        # Show diff
        print_subheader("Result")
        print_diff(rvsdg_before, rvsdg_after)

        print(f"\n{C.DIM}RVSDG (after):{C.RESET}")
        for line in rvsdg_after.strip().split("\n"):
            print(f"  {line}")

    # Summary
    print_header("Summary")
    print(f"""
{C.BOLD}What happened:{C.RESET}
  1. Parsed SPy source → RVSDG intermediate representation
  2. Converted RVSDG → E-graph (equality graph)
  3. Applied rewrite rules (added {stats.nodes_after - stats.nodes_before} equivalent representations)
  4. Extracted cheapest program (cost={stats.extraction_cost})
  5. Converted back to optimized RVSDG

{C.BOLD}Why e-graphs?{C.RESET}
  - Traditional: A → B → C (destructive, order-dependent)
  - E-graph: A ≡ B ≡ C (keep all, pick best)

{C.BOLD}The optimization:{C.RESET}
  {C.RED}Py_Call(Py_LoadGlobal("i32_lt"), ...){C.RESET} cost=10000
  {C.GREEN}Op_i32_lt(...){C.RESET} cost=1
  → Extraction chose the cheaper equivalent
""")


def main():
    """Entry point for VIZ=1 mode."""
    if len(sys.argv) < 2:
        print("Usage: python -m nbcc.viz <source.spy>")
        print("   or: VIZ=1 python -m nbcc <source.spy>")
        sys.exit(1)

    viz_compile(sys.argv[1])


if __name__ == "__main__":
    main()
