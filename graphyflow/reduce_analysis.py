"""
Data structures for holding the analysis results of a ReduceComponent refactoring pass.
These structures describe a plan for how to modify the dataflow graph, rather
than representing the modified graph itself.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
import graphyflow.dataflow_ir as dfir


@dataclass(frozen=True)
class MemoryAccessInfo:
    """Describes a single, unique memory access required by the subgraph."""

    in_idx: int
    base_type: str
    mem_path: Tuple[Union[str, int], ...]

    # The path taken from the initial tuple to get to the base object for memory access.
    # e.g., in `lambda (a, (b, c)): ... c.prop`, the scatter path for `c` is (1, 1)
    scatter_path_to_base: Tuple[int, ...]

    # The data type of the final value after access.
    output_type: dfir.DfirType


@dataclass
class SubgraphAnalysisResult:
    """Holds the analysis results for a single subgraph (key or transform)."""

    refactored_fused_op: dfir.FusedOpComponent
    full_subgraph: dfir.ComponentCollection

    # Maps each input port of the FusedOp to its original data source.
    # The source is either a direct passthrough (described by its scatter path)
    # or a memory access (described by MemoryAccessInfo).
    input_provenance: Dict[dfir.Port, Union[MemoryAccessInfo, Tuple[int, ...]]] = field(default_factory=dict)


@dataclass
class ReduceAnalysisResult:
    """
    The final, consolidated analysis result for a single ReduceComponent.
    This serves as a complete "blueprint" for refactoring.
    """

    key_analysis: SubgraphAnalysisResult
    transform_analysis: SubgraphAnalysisResult
    unit_analysis: SubgraphAnalysisResult

    # A unique list of all memory accesses required by both key and transform subgraphs.
    consolidated_mem_access: List[MemoryAccessInfo]

    # A description of how the data stream entering the Reduce operation
    # should be restructured. It details which original tuple elements are
    # passed through directly and which new data streams are created via memory access.
    restructured_input_plan: Dict[str, Union[Tuple[int, ...], MemoryAccessInfo]]
