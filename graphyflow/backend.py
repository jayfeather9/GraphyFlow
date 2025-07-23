# This is a new backend 'cause the old backend's code is too messy
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


INDENT_UNIT = "    "


class HLSBasicType(Enum):
    UINT = "uint32_t"
    UINT8 = "uint8_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    STRUCT = "struct"
    STREAM = "stream"
    BATCH = "batch"

    def __repr__(self) -> str:
        return self.value

    @property
    def is_simple(self) -> bool:
        return self not in [
            HLSBasicType.STRUCT,
            HLSBasicType.STREAM,
            HLSBasicType.BATCH,
        ]


class HLSType:
    _all_full_names = set()
    _all_names = set()
    _full_to_type = {}
    _id_cnt = 0

    def __init__(
        self,
        basic_type: HLSBasicType,
        sub_types: Optional[List[HLSType]] = None,
        struct_name: Optional[str] = None,
    ) -> None:
        self.type = basic_type
        self.sub_types = sub_types
        self.readable_id = HLSType._id_cnt

        if basic_type.is_simple:
            assert sub_types is None
            self.name = basic_type.value
            self.full_name = self.name
        elif basic_type == HLSBasicType.STREAM:
            assert sub_types and len(sub_types) == 1
            self.name = f"hls::stream<{sub_types[0].name}>"
            self.full_name = f"hls::stream<{sub_types[0].full_name}>"
        elif basic_type == HLSBasicType.BATCH:
            assert sub_types and len(sub_types) == 1
            self.name = f"{sub_types[0].name}[PE_NUM]"
            self.full_name = f"{sub_types[0].full_name}[PE_NUM]"
        elif basic_type == HLSBasicType.STRUCT:
            assert sub_types and len(sub_types) > 0
            self.name = (
                struct_name if struct_name else self._generate_readable_name(sub_types)
            )
            self.full_name = self._generate_canonical_name(sub_types)
        else:
            assert False, f"Basic type {basic_type} not supported"

        if self.full_name in HLSType._all_full_names:
            self = HLSType._full_to_type[self.full_name]
        else:
            HLSType._all_full_names.add(self.full_name)
            assert self.name not in HLSType._all_names
            HLSType._all_names.add(self.name)
            HLSType._full_to_type[self.full_name] = self
            HLSType._id_cnt += 1

    def _generate_canonical_name(self, sub_types: List[HLSType]) -> str:
        name_parts = [t.full_name.replace(" ", "_") for t in sub_types]
        return f"struct_{'_'.join(name_parts)}_t"

    def _generate_readable_name(self, sub_types: List[HLSType]) -> str:
        name_parts = [t.name[:1] for t in sub_types]
        return f"struct_{''.join(name_parts)}_{self.readable_id}_t"

    def get_upper_decl(self, var_name: str):
        """Get decl for upper struct"""
        if self.type == HLSBasicType.BATCH:
            assert self.name[-8:] == "[PE_NUM]"
            return f"{self.name[:-8]} {var_name}[PE_NUM];"
        return f"{self.name} {var_name};"

    def gen_decl(self, member_names: Optional[List[str]] = None) -> str:
        # Generate C++ typedef struct declaration
        assert self.type == HLSBasicType.STRUCT
        if member_names is None:
            member_names = [f"ele_{i}" for i in range(len(self.sub_types))]
        assert len(member_names) == len(self.sub_types)
        decls = [
            st.get_upper_decl(m_name)
            for st, m_name in zip(self.sub_types, member_names)
        ]
        return (
            f"typedef struct {{\n"
            + f"\n".join([INDENT_UNIT + d for d in decls])
            + f"\n}} {self.name};\n"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLSType):
            return NotImplemented
        return self.name == other.name and self.full_name == other.full_name

    def __hash__(self) -> int:
        return hash(self.full_name)


class HLSVar:
    def __init__(self, var_name: str, var_type: HLSType) -> None:
        self.name = var_name
        self.type = var_type


class HLSCodeLine:
    def __init__(self) -> None:
        pass

    def gen_code(self, indent_lvl: int = 0) -> str:
        assert False, "This function shouldn't be called"


class CodeVarDecl(HLSCodeLine):
    def __init__(self, var_name, var_type) -> None:
        super().__init__()
        self.var = HLSVar(var_name, var_type)

    def gen_code(self, indent_lvl: int = 0):
        return indent_lvl * INDENT_UNIT + f"{self.var.type.name} {self.var.name};\n"


class CodeIf(HLSCodeLine):
    def __init__(self, expr: HLSExpr, codes: List[HLSCodeLine]) -> None:
        super().__init__()
        self.expr = expr
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + "if ("
            + self.expr.code
            + ") {\n"
            + "\n".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class CodeFor(HLSCodeLine):
    def __init__(
        self,
        codes: List[HLSCodeLine],
        iter_limit: Union[str, HLSVar],
        iter_cmp="<",
        iter_name="i",
    ) -> None:
        super().__init__()
        self.i_name = iter_name
        self.i_cmp = iter_cmp
        self.i_lim = iter_limit
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + f"for (uint32_t {self.i_name} = 0; {self.i_name} {self.i_cmp} {self.i_lim}; {self.i_name}++) "
            + "{\n"
            + "\n".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class HLSExprType(Enum):
    CONST = "const"
    VAR = "var"
    UOP = "uop"
    BINOP = "binop"
    STREAM_READ = "stream_read"


class HLSExpr:
    def __init__(
        self,
        expr_type: HLSExprType,
        expr_val: Any,
        operands: Optional[List[HLSExpr]] = None,
    ) -> None:
        if expr_type == HLSExprType.CONST:
            assert type(expr_val) in [int, float, bool]
            assert operands is None
        elif expr_type == HLSExprType.VAR:
            assert type(expr_val) == HLSVar
            assert operands is None
        elif expr_type == HLSExprType.STREAM_READ:
            assert type(expr_val) is None
            assert type(operands) == list and len(operands) == 1
            assert operands[0].type == HLSExprType.VAR
            # assert operands[0].val.type.type == HLSBasicType.STREAM # This check will be done at a higher level
        elif expr_type == HLSExprType.UOP:
            assert type(expr_val) == dfir.UnaryOp
            assert type(operands) == list and len(operands) == 1
        elif expr_type == HLSExprType.BINOP:
            assert type(expr_val) == dfir.BinOp
            assert type(operands) == list and len(operands) == 2
        else:
            assert False, f"Type {expr_type} and val {expr_val} not supported"
        self.type = expr_type
        self.val = expr_val
        self.operands = operands

    @property
    def contain_s_read(self) -> bool:
        if self.type in [HLSExprType.CONST, HLSExprType.VAR]:
            return False
        elif self.type == HLSExprType.STREAM_READ:
            return True
        elif self.type in [HLSExprType.UOP, HLSExprType.BINOP]:
            return any(opr.contain_s_read() for opr in self.operands)
        else:
            assert False, f"Type {self.type} not supported"

    @property
    def code(self) -> str:
        if self.type == HLSExprType.CONST:
            if type(self.val) == float:
                return f"(({HLSBasicType.FLOAT.value}){self.val})"
            return str(self.val)
        elif self.type == HLSExprType.VAR:
            return self.val.name
        elif self.type == HLSExprType.STREAM_READ:
            return f"{self.operands[0].name}.read()"
        elif self.type == HLSExprType.UOP:
            trans_dict = {
                dfir.UnaryOp.NOT: "(!operand)",
                dfir.UnaryOp.NEG: "(-operand)",
                dfir.UnaryOp.CAST_BOOL: f"(({HLSBasicType.BOOL.value})(operand))",
                dfir.UnaryOp.CAST_INT: f"(({HLSBasicType.INT.value})(operand))",
                dfir.UnaryOp.CAST_FLOAT: f"(({HLSBasicType.FLOAT.value})(operand))",
                dfir.UnaryOp.SELECT: f"operand.ele_{self.val.val}",
                dfir.UnaryOp.GET_ATTR: f"unary_src.{self.val.val}",
            }
            return trans_dict[self.val].replace("operand", self.operands[0].code)
        elif self.type == HLSExprType.BINOP:
            assert not self.contain_s_read()
            return self.val.gen_repr(self.operands[0].code, self.operands[1].code)
        else:
            assert False, f"Type {self.type} not supported"


class CodeAssign(HLSCodeLine):
    def __init__(self, var: HLSVar, expr: HLSExpr) -> None:
        super().__init__()
        self.var = var
        self.expr = expr

    def gen_code(self, indent_lvl: int = 0) -> str:
        return INDENT_UNIT * indent_lvl + f"{self.var.name} = {self.expr.code};\n"


class CodeCall(HLSCodeLine):
    def __init__(self, func: HLSFunction, params: List[HLSVar]) -> None:
        super().__init__()
        self.func = func
        assert type(params) == list
        self.params = params
        assert len(func.vars) == len(params)
        # for var, call_var in zip(func.vars, params):
        # assert var.type == call_var.type # Type check will be more complex now

    def gen_code(self, indent_lvl: int = 0) -> str:
        return (
            INDENT_UNIT * indent_lvl
            + f"{self.func.name}("
            + ", ".join(var.name for var in self.params)
            + ");\n"
        )


class CodeWriteStream(HLSCodeLine):
    def __init__(self, stream_var: HLSVar, in_var: HLSVar) -> None:
        super().__init__()
        self.stream_var = stream_var
        self.in_var = in_var

    def gen_code(self, indent_lvl: int = 0) -> str:
        return (
            INDENT_UNIT * indent_lvl
            + f"{self.stream_var.name}.write({self.in_var.name});\n"
        )


class HLSFunction:
    def __init__(
        self,
        name: str,
        comp: dfir.Component,
    ) -> None:
        self.name = name
        self.dfir_comp = comp
        self.params: List[HLSVar] = []
        self.codes: List[HLSCodeLine] = []
        self.streamed = True


class BackendManager:
    """Manages the entire HLS code generation process from a ComponentCollection."""

    def __init__(self):
        self.PE_NUM = 8
        # Mappings to store results of type analysis
        self.type_map: Dict[dftype.DfirType, HLSType] = {}
        self.batch_type_map: Dict[HLSType, HLSType] = {}
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        self.unstreamed_funcs: set[str] = set()

    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any
    ) -> Tuple[str, str]:
        """
        Main entry point to generate HLS header and source files.
        :param comp_col: The component collection representing the dataflow graph.
        :param global_graph: An object containing global graph properties like node/edge attributes.
        :return: A tuple containing the (header_code, source_code).
        """
        # Phase 1: Type Analysis
        self._analyze_and_map_types(comp_col, global_graph)

        # --- Placeholder for future phases ---
        header_code = "// Header code to be generated in Phase 4\n"
        source_code = "// Source code to be generated in Phases 2, 3, 4\n"

        # For demonstration, print discovered types
        print("--- Discovered Struct Definitions ---")
        for name, (hls_type, members) in self.struct_definitions.items():
            print(f"Struct: {name}")
            print(hls_type.gen_decl(members))

        print("\n--- Discovered Batch Type Mappings ---")
        for base, batch in self.batch_type_map.items():
            print(f"Base Type: {base.name} -> Batch Type: {batch.name}")

        return header_code, source_code

    def _find_unstreamed_funcs(self, comp_col: dfir.ComponentCollection):
        """
        Identifies all components that are part of a ReduceComponent's sub-graph.
        Their function signatures will use pass-by-reference instead of streams.
        """
        self.unstreamed_funcs.clear()
        q = []
        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q.append(comp.get_port(port_name).connection.parent)

        visited = set()
        while q:
            comp: dfir.Component = q.pop(0)
            if comp.readable_id in visited:
                continue
            visited.add(comp.readable_id)

            if isinstance(comp, dfir.ReduceComponent):
                continue

            self.unstreamed_funcs.add(
                f"{comp.__class__.__name__[:5]}_{comp.readable_id}"
            )
            for port in comp.out_ports:
                if port.connected:
                    q.append(port.connection.parent)

    def _analyze_and_map_types(
        self, comp_col: dfir.ComponentCollection, global_graph: Any
    ):
        """
        Phase 1: Traverse all components and ports to analyze and map DFIR types
        to HLS types, including special batching types for streams.
        """
        self.type_map.clear()
        self.batch_type_map.clear()
        self.struct_definitions.clear()

        # First, identify all functions that are part of a reduce operation
        self._find_unstreamed_funcs(comp_col)

        # Iterate all ports of all components to discover all necessary types
        for comp in comp_col.components:
            for port in comp.ports:
                dfir_type = port.data_type
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_  # Streams operate on the inner type

                if dfir_type:
                    # Get the base HLS type (e.g., a struct without batching wrappers)
                    base_hls_type = self._to_hls_type(dfir_type, global_graph)

                    # If it's a stream port (default case) and not for a reduce sub-function,
                    # create a corresponding batch type.
                    is_stream_port = comp.name not in self.unstreamed_funcs
                    if is_stream_port:
                        self._get_batch_type(base_hls_type)

    def _get_batch_type(self, base_type: HLSType) -> HLSType:
        """
        Creates (or retrieves from cache) a batched version of a base HLSType.
        The batched type is a struct containing an array of the base type and control flags.
        """
        if base_type in self.batch_type_map:
            return self.batch_type_map[base_type]

        # Define the members of the new batch struct
        data_array_type = HLSType(HLSBasicType.BATCH, sub_types=[base_type])
        end_flag_type = HLSType(HLSBasicType.BOOL)
        end_pos_type = HLSType(HLSBasicType.UINT8)

        member_types = [data_array_type, end_flag_type, end_pos_type]
        member_names = ["data", "end_flag", "end_pos"]

        # Create the new HLSType for the batch struct
        batch_type = HLSType(HLSBasicType.STRUCT, sub_types=member_types)

        # Store for reuse and for final declaration generation
        self.batch_type_map[base_type] = batch_type
        if batch_type.name not in self.struct_definitions:
            self.struct_definitions[batch_type.name] = (batch_type, member_names)

        return batch_type

    def _to_hls_type(self, dfir_type: dftype.DfirType, global_graph: Any) -> HLSType:
        """
        Recursively converts a DfirType to a base HLSType, using memoization.
        This handles basic types, tuples, optionals, and special graph types.
        """
        if dfir_type in self.type_map:
            return self.type_map[dfir_type]

        hls_type: HLSType

        # --- BASE CASES ---
        if isinstance(dfir_type, dftype.IntType):
            hls_type = HLSType(HLSBasicType.INT)
        elif isinstance(dfir_type, dftype.FloatType):
            hls_type = HLSType(HLSBasicType.FLOAT)
        elif isinstance(dfir_type, dftype.BoolType):
            hls_type = HLSType(HLSBasicType.BOOL)

        # --- RECURSIVE CASES ---
        elif isinstance(dfir_type, dftype.TupleType):
            sub_types = [self._to_hls_type(t, global_graph) for t in dfir_type.types]
            member_names = [f"ele_{i}" for i in range(len(sub_types))]
            hls_type = HLSType(HLSBasicType.STRUCT, sub_types=sub_types)
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, member_names)

        elif isinstance(dfir_type, dftype.OptionalType):
            data_type = self._to_hls_type(dfir_type.type_, global_graph)
            valid_type = HLSType(HLSBasicType.BOOL)
            hls_type = HLSType(
                HLSBasicType.STRUCT,
                sub_types=[data_type, valid_type],
                struct_name=f"opt_{data_type.name}_t",
            )
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, ["data", "valid"])

        elif isinstance(dfir_type, dftype.SpecialType):
            # Assumes global_graph has node_properties and edge_properties dicts
            props = (
                global_graph.node_properties
                if dfir_type.type_name == "node"
                else global_graph.edge_properties
            )
            prop_names = list(props.keys())
            prop_types = [self._to_hls_type(t, global_graph) for t in props.values()]
            struct_name = f"{dfir_type.type_name}_t"
            hls_type = HLSType(
                HLSBasicType.STRUCT, sub_types=prop_types, struct_name=struct_name
            )
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, prop_names)

        else:
            raise NotImplementedError(
                f"DFIR type conversion not implemented for {type(dfir_type)}"
            )

        # Cache the result before returning
        self.type_map[dfir_type] = hls_type
        return hls_type
