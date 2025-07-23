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


class CodeWhile(HLSCodeLine):
    def __init__(
        self,
        codes: List[HLSCodeLine],
        iter_expr: HLSExpr,
    ) -> None:
        super().__init__()
        self.i_expr = iter_expr
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + f"while ({self.i_expr.code}) "
            + "{\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.codes)
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
            + "".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class CodeBreak(HLSCodeLine):
    def __init__(self) -> None:
        super().__init__()

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + "break;\n"


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
            assert expr_val is None
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
            return any(opr.contain_s_read for opr in self.operands)
        else:
            assert False, f"Type {self.type} not supported"

    @property
    def code(self) -> str:
        if self.type == HLSExprType.CONST:
            if type(self.val) == float:
                return f"(({HLSBasicType.FLOAT.value}){self.val})"
            elif type(self.val) == bool:
                return "true" if self.val else "false"
            return str(self.val)
        elif self.type == HLSExprType.VAR:
            return self.val.name
        elif self.type == HLSExprType.STREAM_READ:
            return f"{self.operands[0].val.name}.read()"
        elif self.type == HLSExprType.UOP:
            trans_dict = {
                dfir.UnaryOp.NOT: "(!operand)",
                dfir.UnaryOp.NEG: "(-operand)",
                dfir.UnaryOp.CAST_BOOL: f"(({HLSBasicType.BOOL.value})(operand))",
                dfir.UnaryOp.CAST_INT: f"(({HLSBasicType.INT.value})(operand))",
                dfir.UnaryOp.CAST_FLOAT: f"(({HLSBasicType.FLOAT.value})(operand))",
                dfir.UnaryOp.SELECT: f"operand.ele_{self.val.val}",
                dfir.UnaryOp.GET_ATTR: f"operand.{self.val.val}",
            }
            return trans_dict[self.val].replace("operand", self.operands[0].code)
        elif self.type == HLSExprType.BINOP:
            assert not self.contain_s_read
            return self.val.gen_repr_forbkd(
                self.operands[0].code, self.operands[1].code
            )
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


class CodePragma(HLSCodeLine):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def gen_code(self, indent_lvl: int = 0) -> str:
        return f"#pragma HLS {self.content}\n"


class CodeBlock(HLSCodeLine):
    """Represents a simple code block enclosed in braces."""

    def __init__(self, codes: List[HLSCodeLine]) -> None:
        super().__init__()
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + "{\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class CodeComment(HLSCodeLine):
    def __init__(self, text: str) -> None:
        super().__init__()
        assert "\n" not in text
        self.text = text

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + "//" + self.text + "\n"


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
        # By default, a function is a standard streaming dataflow block.
        # This will be set to False for reduce sub-functions.
        self.streamed = True


class BackendManager:
    """Manages the entire HLS code generation process from a ComponentCollection."""

    def __init__(self):
        self.PE_NUM = 8
        self.STREAM_DEPTH = 4
        # Mappings to store results of type analysis
        self.type_map: Dict[dftype.DfirType, HLSType] = {}
        self.batch_type_map: Dict[HLSType, HLSType] = {}
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        self.unstreamed_funcs: set[str] = set()

        # State for Phase 2 & 3
        self.hls_functions: Dict[int, HLSFunction] = {}
        # Now stores a tuple of (declaration, pragma)
        self.top_level_stream_decls: List[Tuple[CodeVarDecl, CodePragma]] = []

        self.global_graph_store = None

    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        """
        Main entry point to generate HLS header and source files.
        :param comp_col: The component collection representing the dataflow graph.
        :param global_graph: An object containing global graph properties like node/edge attributes.
        :param top_func_name: The name for the top-level HLS function.
        :return: A tuple containing the (header_code, source_code).
        """
        self.global_graph_store = global_graph

        # Phase 1: Type Analysis
        self._analyze_and_map_types(comp_col)

        # Phase 2: Function Definition and Stream Instantiation
        self._define_functions_and_streams(comp_col, top_func_name)

        # Phase 3: Code Body Generation
        self._translate_functions()

        # --- Placeholder for future phases ---
        header_code = f"// Header code for {top_func_name} to be generated in Phase 4\n"
        source_code = f"// Source code for {top_func_name} to be generated in Phase 4\n"

        return header_code, source_code

    def debug_msgs(self, phases=[1, 2, 3]):
        if 1 in phases:
            # For demonstration, print discovered types
            print("--- Discovered Struct Definitions ---")
            for name, (hls_type, members) in self.struct_definitions.items():
                print(f"Struct: {name}")
                print(hls_type.gen_decl(members))

            print("\n--- Discovered Batch Type Mappings ---")
            for base, batch in self.batch_type_map.items():
                print(f"Base Type: {base.name} -> Batch Type: {batch.name}")
        if 2 in phases:
            # For demonstration, print discovered functions and streams
            print("\n--- Discovered HLS Functions and Signatures ---")
            for func in self.hls_functions.values():
                param_str = ", ".join([f"{p.type.name}& {p.name}" for p in func.params])
                stream_status = "Streamed" if func.streamed else "Unstreamed (by-ref)"
                print(f"Function: {func.name} ({stream_status})")
                print(f"  Signature: void {func.name}({param_str});")

            print("\n--- Intermediate Streams for Top-Level Function ---")
            for decl in self.top_level_stream_decls:
                print(decl.gen_code(indent_lvl=1).strip())
        if 3 in phases:
            print("\n--- Generated HLS Code Bodies (Phase 3) ---")
            for func in self.hls_functions.values():
                print(f"// ======== Code for function: {func.name} ========")
                for code_line in func.codes:
                    # The gen_code method of each HLSCodeLine object produces the C++ string
                    print(code_line.gen_code(indent_lvl=1), end="")
                print(f"// ======== End of function: {func.name} ========\n")

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

    def _analyze_and_map_types(self, comp_col: dfir.ComponentCollection):
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
                is_array_type = False
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                    is_array_type = True

                if dfir_type:
                    # Get the base HLS type (e.g., a struct without batching wrappers)
                    base_hls_type = self._to_hls_type(dfir_type, is_array_type)

                    # If it's a stream port (default case) and not for a reduce sub-function,
                    # create a corresponding batch type.
                    is_stream_port = comp.name not in self.unstreamed_funcs
                    if is_stream_port:
                        self._get_batch_type(base_hls_type)

    def _define_functions_and_streams(
        self, comp_col: dfir.ComponentCollection, top_func_name: str
    ):
        """
        Phase 2: Creates HLSFunction objects, defines their signatures, and
        identifies the intermediate streams needed for the top-level function.
        """
        self.hls_functions.clear()
        self.top_level_stream_decls.clear()

        # --- NEW: Track processed reduce sub-graph components ---
        processed_sub_comp_ids = set()

        # --- NEW: Special handling for ReduceComponent ---
        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                # 1. Create the two main HLS functions for the Reduce component
                pre_process_func = HLSFunction(
                    name=f"{comp.name}_pre_process", comp=comp
                )
                unit_reduce_func = HLSFunction(
                    name=f"{comp.name}_unit_reduce", comp=comp
                )

                # 2. Define their signatures
                # pre_process: stream in, two intermediate streams out
                in_type = self.type_map[comp.get_port("i_0").data_type]
                key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
                transform_type = self.type_map[
                    comp.get_port("i_reduce_transform_out").data_type
                ]

                pre_process_func.params = [
                    HLSVar(
                        "i_0",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[in_type]],
                        ),
                    ),
                    HLSVar(
                        "intermediate_key",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[key_type]],
                        ),
                    ),
                    HLSVar(
                        "intermediate_transform",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[transform_type]],
                        ),
                    ),
                ]

                # unit_reduce: two intermediate streams in, one stream out
                out_type = self.type_map[comp.get_port("o_0").data_type]
                unit_reduce_func.params = [
                    HLSVar(
                        "intermediate_key",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[key_type]],
                        ),
                    ),
                    HLSVar(
                        "intermediate_transform",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[transform_type]],
                        ),
                    ),
                    HLSVar(
                        "o_0",
                        HLSType(
                            HLSBasicType.STREAM,
                            sub_types=[self.batch_type_map[out_type]],
                        ),
                    ),
                ]

                self.hls_functions[comp.readable_id] = pre_process_func
                self.hls_functions[comp.readable_id + 1] = (
                    unit_reduce_func  # Use a unique-ish ID
                )

                # 3. Mark all sub-graph components as processed
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q = [comp.get_port(port_name).connection.parent]
                        visited_sub = set()
                        while q:
                            sub_comp = q.pop(0)
                            if sub_comp.readable_id in visited_sub:
                                continue
                            visited_sub.add(sub_comp.readable_id)
                            processed_sub_comp_ids.add(sub_comp.readable_id)
                            for p in sub_comp.out_ports:
                                if p.connected and not isinstance(
                                    p.connection.parent, dfir.ReduceComponent
                                ):
                                    q.append(p.connection.parent)

        # --- Original logic for non-reduce components ---
        for comp in comp_col.components:
            if comp.readable_id in processed_sub_comp_ids or isinstance(
                comp,
                (
                    dfir.IOComponent,
                    dfir.ConstantComponent,
                    dfir.UnusedEndMarkerComponent,
                    dfir.ReduceComponent,
                ),
            ):
                continue

            hls_func = HLSFunction(name=comp.name, comp=comp)
            # Normal components are always streamed
            for port in comp.ports:
                dfir_type = port.data_type
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                param_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                hls_func.params.append(HLSVar(var_name=port.name, var_type=param_type))
            self.hls_functions[comp.readable_id] = hls_func

        # 2. Identify intermediate streams and add their declarations and pragmas
        visited_ports = set()
        for port in comp_col.all_connected_ports:
            if port.readable_id in visited_ports:
                continue
            conn = port.connection
            is_intermediate = (
                port.parent.readable_id in self.hls_functions
                and conn.parent.readable_id in self.hls_functions
                and self.hls_functions[port.parent.readable_id].streamed
                and self.hls_functions[conn.parent.readable_id].streamed
            )
            if is_intermediate:
                dfir_type = port.data_type
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                stream_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                out_port = port if port.port_type == dfir.PortType.OUT else conn
                stream_name = f"stream_{out_port.unique_name}"

                decl = CodeVarDecl(stream_name, stream_type)
                pragma = CodePragma(
                    f"STREAM variable={stream_name} depth={self.STREAM_DEPTH}"
                )
                self.top_level_stream_decls.append((decl, pragma))

            visited_ports.add(port.readable_id)
            visited_ports.add(conn.readable_id)

    def _translate_functions(self):
        """Phase 3 Entry Point: Populates the .codes for all HLSFunctions."""
        # --- MODIFIED: Handle ReduceComponent first ---
        reduce_comps = [
            f.dfir_comp
            for f in self.hls_functions.values()
            if isinstance(f.dfir_comp, dfir.ReduceComponent)
        ]
        for comp in reduce_comps:
            pre_process_func = next(
                f
                for f in self.hls_functions.values()
                if f.name == f"{comp.name}_pre_process"
            )
            unit_reduce_func = next(
                f
                for f in self.hls_functions.values()
                if f.name == f"{comp.name}_unit_reduce"
            )

            pre_process_func.codes = self._translate_reduce_preprocess(pre_process_func)
            unit_reduce_func.codes = self._translate_reduce_unit_reduce(
                unit_reduce_func
            )

        # --- Translate other functions ---
        for func in self.hls_functions.values():
            if not isinstance(func.dfir_comp, dfir.ReduceComponent):
                if func.streamed:
                    self._translate_streamed_component(func)
                else:  # Should not happen with the new logic
                    assert False
                    self._translate_unstreamed_component(func)

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

    def _to_hls_type(
        self, dfir_type: dftype.DfirType, is_array_type: bool = False
    ) -> HLSType:
        """
        Recursively converts a DfirType to a base HLSType, using memoization.
        This handles basic types, tuples, optionals, and special graph types.
        """
        global_graph = self.global_graph_store
        if dfir_type in self.type_map:
            if is_array_type and dfir.ArrayType(dfir_type) not in self.type_map:
                self.type_map[dfir.ArrayType(dfir_type)] = self.type_map[dfir_type]
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
        if is_array_type:
            self.type_map[dfir.ArrayType(dfir_type)] = hls_type
        return hls_type

    def _translate_streamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component into a standard streamed HLS function body."""
        comp = hls_func.dfir_comp

        # Dispatcher to select the correct translation logic
        if isinstance(comp, dfir.BinOpComponent):
            inner_logic = self._translate_binop_op(comp, "i")
        elif isinstance(comp, dfir.UnaryOpComponent):
            inner_logic = self._translate_unary_op(comp, "i")
        elif isinstance(comp, dfir.CopyComponent):
            inner_logic = self._translate_copy_op(comp, "i")
        elif isinstance(comp, dfir.GatherComponent):
            inner_logic = self._translate_gather_op(comp, "i")
        elif isinstance(comp, dfir.ScatterComponent):
            inner_logic = self._translate_scatter_op(comp, "i")
        elif isinstance(comp, dfir.ConditionalComponent):
            inner_logic = self._translate_conditional_op(comp, "i")
        elif isinstance(comp, dfir.CollectComponent):
            # Collect has a different boilerplate, handle it separately
            hls_func.codes = self._translate_collect_op(hls_func)
            return
        else:
            inner_logic = [
                CodePragma(
                    f"WARNING: Component {type(comp).__name__} translation not implemented."
                )
            ]

        # Wrap the core logic in the standard streaming boilerplate
        hls_func.codes = self._generate_streamed_function_boilerplate(
            hls_func, inner_logic
        )

    def _translate_unstreamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component for an unstreamed (pass-by-reference) function."""
        # This is a placeholder for now, as it's mainly for reduce sub-functions (key, transform, unit)
        hls_func.codes = [
            CodePragma("INLINE"),
            CodePragma(
                f"WARNING: Unstreamed func translation not fully implemented for {hls_func.name}"
            ),
        ]

    def _generate_streamed_function_boilerplate(
        self, hls_func: HLSFunction, inner_loop_logic: List[HLSCodeLine]
    ) -> List[HLSCodeLine]:
        """Creates the standard while/for loop structure for a streamed function."""
        body: List[HLSCodeLine] = []
        in_ports = hls_func.dfir_comp.in_ports
        out_ports = hls_func.dfir_comp.out_ports

        # 1. Declare local batch variables for inputs and outputs
        in_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"in_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [ip.name for ip in in_ports]
        }
        out_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"out_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [op.name for op in out_ports]
        }
        for var in list(in_batch_vars.values()) + list(out_batch_vars.values()):
            body.append(CodeVarDecl(var.name, var.type))
        end_flag_var_decl = CodeVarDecl("end_flag", HLSType(HLSBasicType.BOOL))
        end_flag_var = end_flag_var_decl.var
        body.append(end_flag_var_decl)

        # 2. Create the main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read from all input streams
        for p in hls_func.params:
            if p.name in in_batch_vars:
                read_expr = HLSExpr(
                    HLSExprType.STREAM_READ, None, [HLSExpr(HLSExprType.VAR, p)]
                )
                while_loop_body.append(CodeAssign(in_batch_vars[p.name], read_expr))

        # 4. Create the inner for loop
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL")] + inner_loop_logic,
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(for_loop)

        # 5. Write to all output streams
        for p in hls_func.params:
            if p.name in out_batch_vars:
                while_loop_body.append(CodeWriteStream(p, out_batch_vars[p.name]))

        # 6. Check for end condition and break
        if in_batch_vars:
            # Combine end flags from all inputs
            # For simplicity, we use the first input's end_flag. A real implementation might OR them.
            first_in_batch = list(in_batch_vars.values())[0]
            end_check_expr = HLSExpr(
                HLSExprType.VAR,
                HLSVar(f"{first_in_batch.name}.end_flag", end_flag_var.type),
            )
            assign_end_flag = CodeAssign(end_flag_var, end_check_expr)
            break_if = CodeIf(HLSExpr(HLSExprType.VAR, end_flag_var), [CodeBreak()])
            while_loop_body.extend([assign_end_flag, break_if])

        body.append(
            CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprType.CONST, True))
        )
        return body

    # --- Component-Specific Translators for Inner Loop Logic ---

    def _translate_binop_op(
        self, comp: dfir.BinOpComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a BinOpComponent."""
        # Assume i_0, i_1 are inputs and o_0 is output
        in0_type = (
            self.batch_type_map[self.type_map[comp.input_type]]
            .sub_types[0]
            .sub_types[0]
        )
        in1_type = (
            self.batch_type_map[self.type_map[comp.input_type]]
            .sub_types[0]
            .sub_types[0]
        )
        out_type = (
            self.batch_type_map[self.type_map[comp.output_type]]
            .sub_types[0]
            .sub_types[0]
        )

        # Operands from input batches, indexed by the iterator
        op1 = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in0_type)
        )
        op2 = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_1.data[{iterator}]", in1_type)
        )

        # The binary operation expression
        bin_expr = HLSExpr(HLSExprType.BINOP, comp.op, [op1, op2])

        # The variable to store the result in the output batch
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        return [CodeAssign(target_var, bin_expr)]

    def _translate_unary_op(
        self, comp: dfir.UnaryOpComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a UnaryOpComponent."""
        in_type = (
            self.batch_type_map[self.type_map[comp.input_type]]
            .sub_types[0]
            .sub_types[0]
        )
        out_type = (
            self.batch_type_map[self.type_map[comp.output_type]]
            .sub_types[0]
            .sub_types[0]
        )

        operand = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type)
        )
        unary_expr = HLSExpr(HLSExprType.UOP, comp.op, [operand])
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        return [CodeAssign(target_var, unary_expr)]

    def _translate_copy_op(
        self, comp: dfir.CopyComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a CopyComponent."""
        in_type = self.type_map[comp.get_port("i_0").data_type]

        in_var_expr = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type)
        )

        target_o0 = HLSVar(f"out_batch_o_0.data[{iterator}]", in_type)
        target_o1 = HLSVar(f"out_batch_o_1.data[{iterator}]", in_type)

        return [CodeAssign(target_o0, in_var_expr), CodeAssign(target_o1, in_var_expr)]

    def _translate_gather_op(
        self, comp: dfir.GatherComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a GatherComponent."""
        out_port = comp.get_port("o_0")
        out_type = self.type_map[out_port.data_type]

        assignments = []
        for i, in_port in enumerate(comp.in_ports):
            in_type = self.type_map[in_port.data_type]
            in_var_expr = HLSExpr(
                HLSExprType.VAR,
                HLSVar(f"in_batch_{in_port.name}.data[{iterator}]", in_type),
            )

            # Target is a member of the output struct
            target_member = HLSVar(f"out_batch_o_0.data[{iterator}].ele_{i}", in_type)
            assignments.append(CodeAssign(target_member, in_var_expr))

        return assignments

    def _translate_scatter_op(
        self, comp: dfir.ScatterComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a ScatterComponent."""
        in_port = comp.get_port("i_0")
        in_type = self.type_map[in_port.data_type]

        assignments = []
        for i, out_port in enumerate(comp.out_ports):
            out_type = self.type_map[out_port.data_type]

            get_attr_op = dfir.UnaryOp.GET_ATTR
            get_attr_op.store_val(f"ele_{i}")
            # Source is a member of the input struct
            in_member_expr = HLSExpr(
                HLSExprType.UOP,
                get_attr_op,
                [
                    HLSExpr(
                        HLSExprType.VAR,
                        HLSVar(f"in_batch_i_0.data[{iterator}]", in_type),
                    )
                ],
            )

            target_var = HLSVar(f"out_batch_{out_port.name}.data[{iterator}]", out_type)
            assignments.append(CodeAssign(target_var, in_member_expr))

        return assignments

    def _translate_conditional_op(
        self, comp: dfir.ConditionalComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a ConditionalComponent."""
        data_port = comp.get_port("i_data")
        cond_port = comp.get_port("i_cond")
        out_port = comp.get_port("o_0")

        data_type = self.type_map[data_port.data_type]
        cond_type = self.type_map[cond_port.data_type]
        out_type = self.type_map[out_port.data_type]  # This is an Optional/Struct type

        # Source expressions
        data_expr = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_data.data[{iterator}]", data_type)
        )
        cond_expr = HLSExpr(
            HLSExprType.VAR, HLSVar(f"in_batch_i_cond.data[{iterator}]", cond_type)
        )

        # Target members of the output Optional struct
        target_data_member = HLSVar(f"out_batch_o_0.data[{iterator}].data", data_type)
        target_valid_member = HLSVar(f"out_batch_o_0.data[{iterator}].valid", cond_type)

        return [
            CodeAssign(target_data_member, data_expr),
            CodeAssign(target_valid_member, cond_expr),
        ]

    def _translate_collect_op(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """Generates a custom function body for CollectComponent due to its filtering nature."""
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp
        in_port = comp.get_port("i_0")
        out_port = comp.get_port("o_0")

        # 1. Declare local batch variables
        in_batch_var = HLSVar(
            "in_batch_i_0", self.batch_type_map[self.type_map[in_port.data_type]]
        )
        out_batch_var = HLSVar(
            "out_batch_o_0", self.batch_type_map[self.type_map[out_port.data_type]]
        )
        body.extend(
            [
                CodeVarDecl(in_batch_var.name, in_batch_var.type),
                CodeVarDecl(out_batch_var.name, out_batch_var.type),
            ]
        )

        # 2. Main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read input batch and declare output index
        in_stream_param = hls_func.params[0]  # Assume i_0 is the first param
        read_expr = HLSExpr(
            HLSExprType.STREAM_READ, None, [HLSExpr(HLSExprType.VAR, in_stream_param)]
        )
        while_loop_body.append(CodeAssign(in_batch_var, read_expr))

        out_idx_type = HLSType(HLSBasicType.UINT8)
        out_idx_var = HLSVar("out_idx", out_idx_type)
        while_loop_body.append(CodeAssign(out_idx_var, HLSExpr(HLSExprType.CONST, 0)))

        # 4. Inner for loop for filtering
        in_elem_type = self.type_map[in_port.data_type]  # This is an Optional type
        out_elem_type = self.type_map[out_port.data_type]

        get_attr_op = dfir.UnaryOp.GET_ATTR
        get_attr_op.store_val("valid")
        # Condition: in_batch_i_0.data[i].valid
        cond_expr = HLSExpr(
            HLSExprType.UOP,
            get_attr_op,
            [HLSExpr(HLSExprType.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
        )

        get_attr_op = dfir.UnaryOp.GET_ATTR
        get_attr_op.store_val("data")
        # Assignment if valid: out_batch_o_0.data[out_idx++] = in_batch_i_0.data[i].data
        assign_data = CodeAssign(
            HLSVar(f"out_batch_o_0.data[{out_idx_var.name}]", out_elem_type),
            HLSExpr(
                HLSExprType.UOP,
                get_attr_op,
                [
                    HLSExpr(
                        HLSExprType.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type)
                    )
                ],
            ),
        )
        increment_idx = CodeAssign(
            out_idx_var,
            HLSExpr(
                HLSExprType.BINOP,
                dfir.BinOp.ADD,
                [HLSExpr(HLSExprType.VAR, out_idx_var), HLSExpr(HLSExprType.CONST, 1)],
            ),
        )

        if_block = CodeIf(cond_expr, [assign_data, increment_idx])
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_block], iter_limit="PE_NUM", iter_name="i"
        )
        while_loop_body.append(for_loop)

        # 5. Set output batch metadata and write to stream
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_pos", out_idx_type),
                HLSExpr(HLSExprType.VAR, out_idx_var),
            )
        )
        get_attr_op = dfir.UnaryOp.GET_ATTR
        get_attr_op.store_val("end_flag")
        end_flag_expr = HLSExpr(
            HLSExprType.UOP, get_attr_op, [HLSExpr(HLSExprType.VAR, in_batch_var)]
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )

        out_stream_param = hls_func.params[1]  # Assume o_0 is the second param
        while_loop_body.append(CodeWriteStream(out_stream_param, out_batch_var))

        # 6. Break condition
        while_loop_body.append(CodeIf(end_flag_expr, [CodeBreak()]))

        body.append(
            CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprType.CONST, True))
        )
        return body

    # --- Reduce Component Translation Logic ---

    def _inline_sub_graph_logic(
        self,
        start_ports: List[dfir.Port],
        end_port: dfir.Port,
        io_var_map: Dict[dfir.Port, HLSVar],
    ) -> List[HLSCodeLine]:
        """
        Traverses a sub-graph from start to end ports and generates the inlined logic.
        """
        code_lines: List[HLSCodeLine] = []
        port_to_var_map = io_var_map.copy()

        q = [p.connection.parent for p in start_ports]
        visited_ids = set([c.readable_id for c in q])

        head = 0
        while head < len(q):
            comp = q[head]
            head += 1

            # Check if all inputs for the current component are ready
            inputs_ready = True
            for p in comp.in_ports:
                if p.connection not in port_to_var_map:
                    inputs_ready = False
                    break

            # If not ready, re-queue it and try again later
            if not inputs_ready:
                q.append(comp)
                # Deadlock detection
                if head > len(q) * 2:  # Heuristic to detect non-progress
                    raise RuntimeError(
                        f"Deadlock in sub-graph topological sort, stuck at component {comp.name}"
                    )
                continue

            # --- Inputs are ready, process the component immediately ---

            # 1. Create HLSVars for the output ports of this component
            for out_port in comp.out_ports:
                if out_port.connected and out_port.connection == end_port:
                    # This is the final output port of the sub-graph
                    port_to_var_map[out_port] = port_to_var_map[end_port]
                elif out_port.connected:
                    # This is an intermediate output, create a temporary variable for it
                    temp_var_name = f"temp_{out_port.parent.name}_{out_port.name}"
                    temp_var_type = self.type_map[out_port.data_type]
                    temp_var = HLSVar(temp_var_name, temp_var_type)

                    # Add declaration to code and update the map for successors
                    code_lines.append(CodeVarDecl(temp_var.name, temp_var.type))
                    port_to_var_map[out_port] = temp_var

            # 2. Translate the component's logic into CodeLine objects
            if isinstance(comp, dfir.BinOpComponent):
                op1 = HLSExpr(
                    HLSExprType.VAR, port_to_var_map[comp.get_port("i_0").connection]
                )
                op2 = HLSExpr(
                    HLSExprType.VAR, port_to_var_map[comp.get_port("i_1").connection]
                )
                expr = HLSExpr(HLSExprType.BINOP, comp.op, [op1, op2])
                target_var = port_to_var_map[comp.get_port("o_0")]
                code_lines.append(CodeAssign(target_var, expr))
            # ... You would add translations for other component types here ...
            else:
                # Fallback for any other component types
                code_lines.append(CodeComment(f"Inlined logic for {comp.name}"))

            # 3. Add successors to the queue
            for p in comp.out_ports:
                if p.connected and not isinstance(
                    p.connection.parent, dfir.ReduceComponent
                ):
                    successor_comp = p.connection.parent
                    if successor_comp.readable_id not in visited_ids:
                        q.append(successor_comp)
                        visited_ids.add(successor_comp.readable_id)

        return code_lines

    def _translate_reduce_preprocess(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """Generates the body for the first stage of Reduce (key/transform extraction)."""
        # This function has a custom body, it does not use the standard boilerplate
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # Get params for streams
        in_stream, key_stream, transform_stream = hls_func.params

        # Local single-element variables (these are NOT batches)
        in_elem_var = HLSVar("reduce_src", in_stream.type.sub_types[0])
        key_out_var = HLSVar("reduce_key_out", key_stream.type.sub_types[0])
        transform_out_var = HLSVar(
            "reduce_transform_out", transform_stream.type.sub_types[0]
        )
        body.extend(
            [
                CodeVarDecl(v.name, v.type)
                for v in [in_elem_var, key_out_var, transform_out_var]
            ]
        )
        end_flag_var_decl = CodeVarDecl("end_flag_val", HLSType(HLSBasicType.BOOL))
        end_flag_var = end_flag_var_decl.var
        body.append(end_flag_var_decl)

        # while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]
        while_loop_body.append(
            CodeAssign(
                in_elem_var,
                HLSExpr(
                    HLSExprType.STREAM_READ, None, [HLSExpr(HLSExprType.VAR, in_stream)]
                ),
            )
        )

        # Extract and check end flag
        get_attr_op = dfir.UnaryOp.GET_ATTR
        get_attr_op.store_val("end_flag")
        end_flag_expr = HLSExpr(
            HLSExprType.UOP, get_attr_op, [HLSExpr(HLSExprType.VAR, in_elem_var)]
        )
        while_loop_body.append(CodeAssign(end_flag_var, end_flag_expr))

        # Inline the sub-graph logic
        key_sub_graph_start = comp.get_port("o_reduce_key_in")
        key_sub_graph_end = comp.get_port("i_reduce_key_out")
        key_io_map = {key_sub_graph_start: in_elem_var, key_sub_graph_end: key_out_var}
        while_loop_body.extend(
            self._inline_sub_graph_logic(
                [key_sub_graph_start], key_sub_graph_end, key_io_map
            )
        )

        transform_sub_graph_start = comp.get_port("o_reduce_transform_in")
        transform_sub_graph_end = comp.get_port("i_reduce_transform_out")
        transform_io_map = {
            transform_sub_graph_start: in_elem_var,
            transform_sub_graph_end: transform_out_var,
        }
        while_loop_body.extend(
            self._inline_sub_graph_logic(
                [transform_sub_graph_start], transform_sub_graph_end, transform_io_map
            )
        )

        # Set end_flag on outputs and write to streams
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{key_out_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                HLSExpr(HLSExprType.VAR, end_flag_var),
            )
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(
                    f"{transform_out_var.name}.end_flag", HLSType(HLSBasicType.BOOL)
                ),
                HLSExpr(HLSExprType.VAR, end_flag_var),
            )
        )
        while_loop_body.append(CodeWriteStream(key_stream, key_out_var))
        while_loop_body.append(CodeWriteStream(transform_stream, transform_out_var))

        # Break condition
        while_loop_body.append(
            CodeIf(HLSExpr(HLSExprType.VAR, end_flag_var), [CodeBreak()])
        )

        body.append(
            CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprType.CONST, True))
        )
        return body

    def _translate_reduce_unit_reduce(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """Generates the body for the second stage of Reduce (stateful accumulation)."""
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # Types
        key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
        accum_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]
        # The BRAM stores a struct of {accumulator, valid_flag}
        bram_elem_type = self._to_hls_type(
            dftype.TupleType(
                [comp.get_port("i_reduce_transform_out").data_type, dftype.BoolType()]
            )
        )

        # State declarations
        body.append(CodeVarDecl("key_mem[MAX_NUM]", bram_elem_type))
        body.append(CodePragma("BIND_STORAGE variable=key_mem type=RAM_2P impl=URAM"))
        # ... Add other state declarations and initializations from the C++ example ...

        # Main while loop (processing single elements, not batches)
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]
        # ...
        # Logic to read from intermediate streams, peel off end_flag etc.
        # Logic for buffer management (i_buffer, key_buffer)
        # Logic to read old_ele from key_mem
        # ...

        # Inline the unit sub-graph
        unit_sub_graph_starts = [
            comp.get_port("o_reduce_unit_start_0"),
            comp.get_port("o_reduce_unit_start_1"),
        ]
        unit_sub_graph_end = comp.get_port("i_reduce_unit_end")

        # This part is complex as it involves an if/else block
        # Pseudocode:
        # if (!old_ele.valid) { new_ele.data = real_transform_value; }
        # else {
        #   io_map = {start0: old_ele.data, start1: real_transform_value, end: new_ele.data}
        #   inline_sub_graph_logic(..., io_map)
        # }
        # ...

        # Final output loop after the while loop
        # ...

        # Placeholder until full implementation
        body.append(
            CodePragma(
                "WARNING: Reduce Unit Reduce translation is complex and not yet fully implemented."
            )
        )
        return body
