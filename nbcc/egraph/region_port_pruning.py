# mypy: disable-error-code="empty-body"
"""EGraph-based optimization for eliminating redundant passthrough ports in
IfElse constructs.

This module implements an optimization that identifies and eliminates
"passthrough ports" in IfElse regions. A passthrough port is an output port
whose value is directly connected to an input operand of the region (no
transformation). When both then/else branches of an IfElse have the same
passthrough ports, the optimization eliminates those ports and replaces their
usage with direct operand references.

The optimization works by: 1. Analyzing both branches to identify passthrough
mappings (output index -> operand index) 2. Finding common passthroughs between
branches via set intersection 3. Creating pruned port lists that exclude the
common passthroughs 4. Redirecting usage of eliminated ports to reference the
original operands directly

This reduces the output arity of IfElse constructs and simplifies the IR when
branches have redundant passthrough outputs.
"""
from __future__ import annotations
from sealir import ase
from sealir.rvsdg import grammar as rg, format_rvsdg

from nbcc.compiler import egraph_conversion, EGraph


from egglog import Expr, function, Set, i64


class PassthroughAnalysisResult(Expr):
    """Marker class for tracking passthrough analysis completion."""

    def __init__(self): ...


class PassthroughMapping(Expr):
    """Represents a mapping from an output port index to its source operand
    index.

    Used to track which output ports are "passthroughs" - ports whose values are
    directly forwarded from input operands without transformation.

    Args:
        out_idx: Index of the output port
        src_idx: Index of the input operand that this port forwards
    """

    def __init__(self, out_idx: i64, src_idx: i64): ...


@function
def record_passthrough_analysis(
    mappings: Set[PassthroughMapping],
) -> PassthroughAnalysisResult:
    """Records the completion of passthrough analysis for debugging purposes."""
    ...


def define_rule():
    from egglog import (
        ruleset,
        rule,
        panic,
        function,
        PyObject,
        Unit,
        Expr,
        rewrite,
        String,
        set_,
        union,
        i64,
        i64Like,
        Vec,
        subsume,
        Set,
        Bool,
        method,
        delete,
    )
    from sealir.eqsat.rvsdg_eqsat import (
        Term,
        TermList,
        Region,
        PortList,
        Port,
        Region,
    )

    class _PruningAction(Expr):
        """Internal action marker for port pruning operations."""

        ...

    @function(merge=lambda x, y: x | y)
    def collect_passthrough_mappings(
        region: Region, ports: Vec[Port], current_index: i64Like
    ) -> Set[PassthroughMapping]:
        """Recursively collects passthrough mappings for a region's output
        ports.

        Analyzes each output port to determine if it directly forwards an input
        operand. If so, creates a PassthroughMapping recording the output-to-
        operand relationship.

        Args:
            region: The region containing the ports
            ports: Vector of output ports to analyze
            current_index: Current position in the ports vector

        Returns:
            Set of PassthroughMapping objects for passthrough ports found
        """
        ...

    @function
    def initiate_passthrough_elimination(
        ifelse: Term,
        then_region: Region,
        then_ports: PortList,
        else_region: Region,
        else_ports: PortList,
        elimination_mappings: Set[PassthroughMapping],
    ) -> _PruningAction:
        """Initiates the elimination of common passthrough ports from an IfElse
        construct.

        Takes the intersection of passthrough mappings from both branches and
        begins the process of pruning those common passthroughs from the output
        port lists.

        Args:
            ifelse: The IfElse term being optimized
            then_region: The then branch region
            then_ports: Output ports of the then branch
            else_region: The else branch region
            else_ports: Output ports of the else branch
            elimination_mappings: Common passthrough mappings to eliminate

        Returns:     Action marker to trigger port pruning rules
        """
        ...

    class EliminationMask(Expr):
        """Bitmap-style data structure tracking which output ports should be
        eliminated.

        Stores a set of PassthroughMapping objects along with the total port
        count, and provides efficient lookup to check if a specific port index
        should be eliminated during pruning.

        Args:
            mappings: Set of passthrough mappings indicating ports to
            eliminate port_count: Total number of output ports in the original
                                  list
        """

        def __init__(
            self, mappings: Set[PassthroughMapping], port_count: i64
        ): ...

        @method(merge=lambda a, b: a | b)
        def should_eliminate(self, port_index: i64Like) -> Bool:
            """Returns True if the port at the given index should be
            eliminated."""
            ...

    @function
    def create_pruned_port_list(
        region: Region, ports: PortList, elimination_mask: EliminationMask
    ) -> PortList:
        """Creates a new port list with eliminated passthrough ports removed.

        Args:
            region: The region containing the ports
            ports: Original list of output ports
            elimination_mask: Mask indicating which ports to eliminate
        Returns:
            New PortList with passthrough ports removed
        """
        ...

    @function
    def _build_pruned_port_list(
        region: Region,
        ports: Vec[Port],
        current_pos: i64,
        elimination_mask: EliminationMask,
        accumulated_ports: Vec[Port],
    ) -> _PruningAction:
        """Helper function that recursively builds the pruned port list.

        Iterates through the original ports vector, including only those ports
        that are not marked for elimination in the mask.

        Args:
            region: The region containing the ports
            ports: Original vector of ports
            current_pos: Current position being processed
            elimination_mask: Mask indicating which ports to skip
            accumulated_ports: Growing list of ports to keep

        Returns:
            Action marker for rule scheduling
        """
        ...

    @function
    def redirect_port_usage_to_operands(
        ifelse_term: Term,
        elimination_mappings: Set[PassthroughMapping],
    ) -> _PruningAction:
        """Redirects usage of eliminated ports to reference the original
        operands.

        For each eliminated passthrough port, replaces references to that port
        with direct references to the corresponding input operand.

        Args:
            ifelse_term: The IfElse term being optimized
            elimination_mappings: Mappings from eliminated port indices to operand
                                  indices
        Returns:
            Action marker to trigger usage redirection rules
        """
        ...

    @function
    def initialize_elimination_mask(
        elimination_mask: EliminationMask, current_pos: i64
    ) -> _PruningAction:
        """Initializes the elimination mask by setting all positions to False
        initially.

        This function recursively processes each position in the mask, marking
        positions as True only if they correspond to ports that should be
        eliminated.

        Args:
            elimination_mask: The mask being initialized
            current_pos: Current position being processed
        Returns:
            Action marker for rule scheduling
        """
        ...

    @ruleset
    def detect_ifelse_passthrough_candidates(
        cond_term: Term,
        then_term: Term,
        else_term: Term,
        operand_terms: TermList,
        then_region: Region,
        then_out_ports: Vec[Port],
        else_region: Region,
        else_out_ports: Vec[Port],
    ):
        """
        Detects IfElse constructs and initiates passthrough analysis for both
        branches.

        This ruleset identifies IfElse terms and triggers the collection of
        passthrough mappings for both the then and else regions by starting the
        recursive analysis at position 0 for each branch's output ports.
        """
        yield rule(
            Term.IfElse(
                cond=cond_term,
                then=then_term,
                orelse=else_term,
                operands=operand_terms,
            ),
            then_term
            == Term.RegionEnd(
                region=then_region, ports=PortList(ports=then_out_ports)
            ),
            else_term
            == Term.RegionEnd(
                region=else_region, ports=PortList(ports=else_out_ports)
            ),
        ).then(
            set_(
                collect_passthrough_mappings(then_region, then_out_ports, 0)
            ).to(Set[PassthroughMapping].empty()),
            set_(
                collect_passthrough_mappings(else_region, else_out_ports, 0)
            ).to(Set[PassthroughMapping].empty()),
        )

    @ruleset
    def analyze_passthrough_mappings(
        i: i64,
        j: i64,
        ports: Vec[Port],
        region: Region,
        wc_name: String,
        mappings: Set[PassthroughMapping],
    ):
        """
        Analyzes individual ports to identify passthrough mappings and continues iteration.
        """
        # 1. Identifies when an output port directly references a region input
        yield rule(
            mappings == collect_passthrough_mappings(region, ports, i),
            ports[i] == Port(name=wc_name, term=region.get(j)),
        ).then(
            set_(collect_passthrough_mappings(region, ports, i)).to(
                {PassthroughMapping(i, j)}
            ),
        )
        # 2. Advances the analysis to the next port position when current
        #    position is within bounds
        yield rule(
            mappings == collect_passthrough_mappings(region, ports, i),
            i < ports.length(),
        ).then(
            set_(collect_passthrough_mappings(region, ports, i + 1)).to(
                mappings
            )
        )

    @ruleset
    def eliminate_common_passthroughs(
        cond_term: Term,
        then_term: Term,
        else_term: Term,
        operand_terms: TermList,
        then_region: Region,
        then_out_ports: Vec[Port],
        else_region: Region,
        else_out_ports: Vec[Port],
        then_mappings: Set[PassthroughMapping],
        else_mappings: Set[PassthroughMapping],
        elimination_mappings: Set[PassthroughMapping],
        ifelse_term: Term,
        then_ports: PortList,
        else_ports: PortList,
        i: i64,
        j: i64,
        nelem: i64,
    ):
        """
        Performs the core elimination of common passthrough ports between branches.

        3. Redirect port usage to point to original operands instead of eliminated ports
        4. Initialize and populate the elimination mask to track which ports should be removed
        """
        # 1. Find the intersection of passthrough mappings from both branches
        #    and initiate elimination
        yield rule(
            ifelse_term
            == Term.IfElse(
                cond=cond_term,
                then=then_term,
                orelse=else_term,
                operands=operand_terms,
            ),
            then_term
            == Term.RegionEnd(
                region=then_region, ports=PortList(ports=then_out_ports)
            ),
            else_term
            == Term.RegionEnd(
                region=else_region, ports=PortList(ports=else_out_ports)
            ),
            then_mappings
            == collect_passthrough_mappings(
                then_region, then_out_ports, then_out_ports.length()
            ),
            else_mappings
            == collect_passthrough_mappings(
                else_region, else_out_ports, else_out_ports.length()
            ),
        ).then(
            # The intersection finds common passthrough ports between branches
            union(PassthroughAnalysisResult()).with_(
                record_passthrough_analysis(then_mappings)
            ),  # TODO: REMOVE ME
            initiate_passthrough_elimination(
                ifelse_term,
                then_region,
                PortList(ports=then_out_ports),
                else_region,
                PortList(ports=else_out_ports),
                then_mappings & else_mappings,  # Intersection
            ),
        )
        # 2. Create pruned port lists for both then and else branches,
        #    removing common passthrough ports
        # 3. Redirect port usage to point to original operands instead of
        #    eliminated ports
        yield rule(
            _del1 := initiate_passthrough_elimination(
                ifelse_term,
                then_region,
                then_ports,
                else_region,
                else_ports,
                elimination_mappings,
            ),
            then_ports == PortList(ports=then_out_ports),
            else_ports == PortList(ports=else_out_ports),
        ).then(
            union(then_ports).with_(
                create_pruned_port_list(
                    then_region,
                    then_ports,
                    EliminationMask(
                        elimination_mappings, then_out_ports.length()
                    ),
                )
            ),
            union(else_ports).with_(
                create_pruned_port_list(
                    else_region,
                    else_ports,
                    EliminationMask(
                        elimination_mappings, else_out_ports.length()
                    ),
                )
            ),
            redirect_port_usage_to_operands(ifelse_term, elimination_mappings),
            delete(_del1),
        )
        # 4. Initialize and populate the elimination mask to track which ports
        #    should be removed
        yield rule(
            EliminationMask(elimination_mappings, nelem),
            elimination_mappings.contains(PassthroughMapping(i, j)),
        ).then(
            set_(
                EliminationMask(elimination_mappings, nelem).should_eliminate(
                    i
                )
            ).to(True)
        )

        yield rule(
            EliminationMask(elimination_mappings, nelem),
        ).then(
            initialize_elimination_mask(
                EliminationMask(elimination_mappings, nelem), 0
            )
        )
        yield rule(
            initialize_elimination_mask(
                EliminationMask(elimination_mappings, nelem), i
            ),
            i + 1 < nelem,
        ).then(
            initialize_elimination_mask(
                EliminationMask(elimination_mappings, nelem), i + 1
            )
        )
        yield rule(
            del1 := initialize_elimination_mask(
                EliminationMask(elimination_mappings, nelem), i
            ),
            i < nelem,
        ).then(
            set_(
                EliminationMask(elimination_mappings, nelem).should_eliminate(
                    i
                )
            ).to(False),
            delete(del1),
        )

    @ruleset
    def construct_pruned_ports(
        ports: Vec[Port],
        new_ports: Vec[Port],
        region: Region,
        i: i64,
        elimination_mask: EliminationMask,
    ):
        """
        Constructs the actual pruned port lists by iterating through original ports.
        """

        # 1. Initiate the pruning process by starting the recursive port list
        #    construction
        yield rule(
            create_pruned_port_list(region, PortList(ports), elimination_mask),
        ).then(
            _build_pruned_port_list(
                region, ports, 0, elimination_mask, Vec[Port].empty()
            )
        )
        # 2. Include ports that should NOT be eliminated (mask returns False)
        yield rule(
            del1 := _build_pruned_port_list(
                region, ports, i, elimination_mask, new_ports
            ),
            ports[i],
            i < ports.length(),
            Bool(False) == (elimination_mask.should_eliminate(i)),
        ).then(
            _build_pruned_port_list(
                region,
                ports,
                i + 1,
                elimination_mask,
                new_ports.push(ports[i]),
            ),
            delete(del1),
        )
        # 3. Skip ports that SHOULD be eliminated (mask returns True)
        yield rule(
            del1 := _build_pruned_port_list(
                region, ports, i, elimination_mask, new_ports
            ),
            ports[i],
            i < ports.length(),
            Bool(True) == (elimination_mask.should_eliminate(i)),
        ).then(
            _build_pruned_port_list(
                region, ports, i + 1, elimination_mask, new_ports
            ),
            delete(del1),
        )
        # 4. Finalize the pruned port list when all positions have been processed
        yield rule(
            del1 := create_pruned_port_list(
                region, PortList(ports), elimination_mask
            ),
            del2 := _build_pruned_port_list(
                region, ports, ports.length(), elimination_mask, new_ports
            ),
        ).then(
            union(del1).with_(PortList(new_ports)), delete(del1), delete(del2)
        )

    @ruleset
    def redirect_eliminated_port_usage(
        ifelse: Term,
        elimination_mappings: Set[PassthroughMapping],
        i: i64,
        j: i64,
        operands: TermList,
        wc_cond: Term,
        wc_then: Term,
        wc_orelse: Term,
    ):
        """
        Redirects references to eliminated ports to point directly to the
        original operands.

        - For each eliminated port (found in elimination_mappings), replaces
          references to that port with direct references to the corresponding
          input operand
        - This completes the optimization by removing the indirection through
          eliminated ports
        """
        yield rule(
            redirect_port_usage_to_operands(ifelse, elimination_mappings),
            ifelse.getPort(i),
            elimination_mappings.contains(PassthroughMapping(i, j)),
            ifelse
            == Term.IfElse(
                cond=wc_cond, then=wc_then, orelse=wc_orelse, operands=operands
            ),
        ).then(union(ifelse.getPort(i)).with_(operands[j]))

    schedule = (
        (
            detect_ifelse_passthrough_candidates | analyze_passthrough_mappings
        ).saturate()
        + eliminate_common_passthroughs.saturate()
        + construct_pruned_ports.saturate()
        + redirect_eliminated_port_usage.saturate()
    )

    return schedule
