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
    UINT16 = "uint16_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    AP_FIXED_POD = "int32_t"
    REAL_FLOAT = "float"
    BOOL = "bool"
    STRUCT = "struct"
    STREAM = "stream"
    ARRAY = "array"
    POINTER = "pointer"

    def __repr__(self) -> str:
        return self.value

    @property
    def is_simple(self) -> bool:
        return self not in [
            HLSBasicType.STRUCT,
            HLSBasicType.STREAM,
            HLSBasicType.ARRAY,
            HLSBasicType.POINTER,
        ]


class HLSType:
    _all_full_names = set()
    _all_names = set()
    _full_to_type = {}
    _name_to_full = {}
    _id_cnt = 0

    def __init__(
        self,
        basic_type: HLSBasicType,
        sub_types: Optional[List[HLSType]] = None,
        struct_name: Optional[str] = None,
        struct_prop_names: Optional[List[str]] = None,
        array_dims: Optional[List[Union[str, int]]] = None,
        is_const_ptr: bool = False,
    ) -> None:
        self.type = basic_type
        self.sub_types = sub_types
        self.readable_id = HLSType._id_cnt
        self.struct_prop_names = None
        self.array_dims = array_dims
        self.is_const_ptr = is_const_ptr

        if basic_type.is_simple:
            self.name = basic_type.value
            self.full_name = self.name
        elif basic_type == HLSBasicType.STREAM:
            assert sub_types and len(sub_types) == 1
            self.name = f"hls::stream<{sub_types[0].name}>"
            self.full_name = f"hls::stream<{sub_types[0].full_name}>"
        elif basic_type == HLSBasicType.ARRAY:
            assert sub_types and len(sub_types) == 1 and array_dims and len(array_dims) > 0
            dims_str = "".join(f"[{d}]" for d in self.array_dims)
            self.name = f"{sub_types[0].name}{dims_str}"
            self.full_name = f"{sub_types[0].full_name}{dims_str}"
        elif basic_type == HLSBasicType.POINTER:
            assert sub_types and len(sub_types) == 1
            const_str = "const " if self.is_const_ptr else ""
            self.name = f"{const_str}{sub_types[0].name}*"
            self.full_name = f"{const_str}{sub_types[0].full_name}*"
        elif basic_type == HLSBasicType.STRUCT:
            assert sub_types and len(sub_types) > 0
            self.full_name = self._generate_canonical_name(sub_types, explicit_name=struct_name)

            if self.full_name in HLSType._all_full_names:
                existing_type = HLSType._full_to_type[self.full_name]
                self.__dict__.update(existing_type.__dict__)
                return

            self.name = struct_name if struct_name else self._generate_readable_name(sub_types)

            if struct_prop_names:
                assert len(struct_prop_names) == len(sub_types)
                self.struct_prop_names = struct_prop_names
        else:
            assert False, f"Basic type {basic_type} not supported"

        # Caching for truly new types
        if self.full_name in HLSType._all_full_names:
            existing_type = HLSType._full_to_type[self.full_name]
            self.__dict__.update(existing_type.__dict__)
            return

        HLSType._all_full_names.add(self.full_name)

        # --- *** 关键修正：仅对非简单类型进行名称冲突检查 *** ---
        if not self.type.is_simple:
            if self.name in HLSType._all_names:
                if struct_name is not None:
                    assert False, f"Struct name collision detected: {self.name}"
                else:
                    self.name = f"{self.name}_{self.readable_id}"

        HLSType._all_names.add(self.name)
        HLSType._full_to_type[self.full_name] = self
        HLSType._name_to_full[self.name] = self.full_name
        HLSType._id_cnt += 1

    @classmethod
    def get_type(cls, type_name):
        assert type_name in cls._all_names
        return cls._full_to_type[cls._name_to_full[type_name]]

    def _generate_canonical_name(self, sub_types: List[HLSType], explicit_name: Optional[str] = None) -> str:
        name_parts = [t.full_name.replace(" ", "_").replace("*", "_ptr") for t in sub_types]
        if explicit_name:
            name_parts.insert(0, explicit_name)
        return f"struct_{'_'.join(name_parts)}_t"

    def _generate_readable_name(self, sub_types: List[HLSType]) -> str:
        name_parts = [t.name[:1] for t in sub_types]
        return f"struct_{''.join(name_parts)}_{self.readable_id}_t"

    def get_nth_subname(self, n: int):
        assert self.sub_types
        if not self.struct_prop_names:
            member_names = [f"ele_{i}" for i in range(len(self.sub_types))]
        else:
            member_names = self.struct_prop_names
        assert n < len(member_names)
        return member_names[n]

    def get_upper_decl(self, var_name: str):
        """Get decl for upper struct"""
        if self.type == HLSBasicType.ARRAY:
            match = re.search(r"\[", self.name)
            if match:
                base_type_str = self.name[: match.start()]
                dims_str = self.name[match.start() :]
                return f"{base_type_str} {var_name}{dims_str}"
            else:
                assert False
                # return f"{self.name} {var_name};"
        return f"{self.name} {var_name}"

    def get_upper_param(self, var_name: str, ref: bool):
        if self.type == HLSBasicType.ARRAY:
            match = re.search(r"\[", self.name)
            if match:
                base_type_str = self.name[: match.start()]
                dims_str = self.name[match.start() :]
                if ref:
                    return f"{base_type_str} (&{var_name}){dims_str}"
                return f"{base_type_str} {var_name}{dims_str}"
            else:
                assert False
        if ref:
            return f"{self.name} &{var_name}"
        return f"{self.name} {var_name}"

    def gen_decl(self, member_names: Optional[List[str]] = None) -> str:
        # Generate C++ typedef struct declaration
        assert self.type == HLSBasicType.STRUCT
        if member_names is None:
            if self.struct_prop_names:
                member_names = self.struct_prop_names
            else:
                member_names = [f"ele_{i}" for i in range(len(self.sub_types))]

        assert len(member_names) == len(self.sub_types)
        if self.struct_prop_names:
            assert self.struct_prop_names == member_names
        decls = [st.get_upper_decl(m_name) + ";" for st, m_name in zip(self.sub_types, member_names)]

        return (
            f"struct __attribute__((packed)) {self.name} {{\n"
            + f"\n".join([INDENT_UNIT + d for d in decls])
            + f"\n}};\n"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLSType):
            return NotImplemented
        return self.name == other.name and self.full_name == other.full_name

    def __hash__(self) -> int:
        return hash(self.full_name)

    def __repr__(self) -> str:
        return self.name


class HLSVar:
    def __init__(self, var_name: str, var_type: HLSType) -> None:
        self.name = var_name
        self.type = var_type

    def __repr__(self) -> str:
        return f"HLSVar({self.name}, {self.type})"


class HLSCodeLine:
    def __init__(self) -> None:
        pass

    def gen_code(self, indent_lvl: int = 0) -> str:
        assert False, "This function shouldn't be called"


class CodeVarDecl(HLSCodeLine):
    def __init__(self, var_name, var_type, init_val=None) -> None:
        super().__init__()
        self.var = HLSVar(var_name, var_type)
        self.init_val = init_val

    def gen_code(self, indent_lvl: int = 0):
        init_code = f" = {self.init_val}" if self.init_val is not None else ""
        return indent_lvl * INDENT_UNIT + self.var.type.get_upper_decl(self.var.name) + init_code + ";\n"


class CodeIf(HLSCodeLine):
    def __init__(
        self,
        expr: HLSExpr,
        if_codes: List[HLSCodeLine],
        else_codes: List[HLSCodeLine] = None,
        elifs: List[Tuple[HLSExpr, List[HLSCodeLine]]] = None,
    ) -> None:
        super().__init__()
        if type(expr) == HLSVar:
            expr = HLSExpr(HLSExprT.VAR, expr)
        self.expr = expr
        self.if_codes = if_codes
        self.elifs = elifs if elifs else []
        self.else_codes = else_codes if else_codes else []

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        if_part = (
            oind
            + "if ("
            + self.expr.code
            + ") {\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.if_codes)
            + oind
            + "}"
        )
        elif_part = ""
        for expr, codes in self.elifs:
            elif_part += (
                f" else if ({expr.code}) "
                + "{\n"
                + "".join(c.gen_code(indent_lvl + 1) for c in codes)
                + oind
                + "}"
            )
        else_part = ""
        if self.else_codes:
            else_part = (
                " else {\n" + "".join(c.gen_code(indent_lvl + 1) for c in self.else_codes) + oind + "}"
            )
        return if_part + elif_part + else_part + "\n"


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


class HLSExprT(Enum):
    CONST = "const"
    VAR = "var"
    UOP = "uop"
    BINOP = "binop"
    STREAM_READ = "stream_read"
    STREAM_EMPTY = "stream_empty"


class HLSExpr:
    def __init__(
        self,
        expr_type: HLSExprT,
        expr_val: Any,
        operands: Optional[List[HLSExpr]] = None,
    ) -> None:
        if expr_type == HLSExprT.CONST:
            assert type(expr_val) in [int, float, bool, str]
            assert operands is None
        elif expr_type == HLSExprT.VAR:
            assert type(expr_val) == HLSVar
            assert operands is None
        elif expr_type == HLSExprT.STREAM_READ:
            assert expr_val is None
            assert type(operands) == list and len(operands) == 1
            assert operands[0].type == HLSExprT.VAR
        elif expr_type == HLSExprT.STREAM_EMPTY:
            assert expr_val is None
            assert type(operands) == list and len(operands) == 1
            assert operands[0].type == HLSExprT.VAR
        elif expr_type == HLSExprT.UOP:
            if type(expr_val) == tuple:
                assert type(expr_val[0]) == dfir.UnaryOp
            else:
                assert type(expr_val) == dfir.UnaryOp
            assert type(operands) == list and len(operands) == 1
        elif expr_type == HLSExprT.BINOP:
            assert type(expr_val) == dfir.BinOp
            assert type(operands) == list and len(operands) == 2
        else:
            assert False, f"Type {expr_type} and val {expr_val} not supported"
        self.type = expr_type
        self.val = expr_val
        self.operands = operands

    @classmethod
    def check_const(cls, hls_expr: HLSExpr, port: dfir.Port):
        if port.from_const:
            return HLSExpr(HLSExprT.CONST, port.from_const_val)
        return hls_expr

    @property
    def contain_s_read(self) -> bool:
        if self.type in [HLSExprT.CONST, HLSExprT.VAR]:
            return False
        elif self.type in [HLSExprT.STREAM_READ, HLSExprT.STREAM_EMPTY]:
            return True
        elif self.type in [HLSExprT.UOP, HLSExprT.BINOP]:
            return any(opr.contain_s_read for opr in self.operands)
        else:
            assert False, f"Type {self.type} not supported"

    @property
    def code(self) -> str:
        if self.type == HLSExprT.CONST:
            if type(self.val) == float:
                return f"(({HLSBasicType.FLOAT.value}){self.val})"
            elif type(self.val) == bool:
                return "true" if self.val else "false"
            return str(self.val)
        elif self.type == HLSExprT.VAR:
            return self.val.name
        elif self.type == HLSExprT.STREAM_READ:
            return f"{self.operands[0].val.name}.read()"
        elif self.type == HLSExprT.STREAM_EMPTY:
            return f"{self.operands[0].val.name}.empty()"
        elif self.type == HLSExprT.UOP:
            trans_dict = {
                dfir.UnaryOp.NOT: "(!operand)",
                dfir.UnaryOp.NEG: "(-operand)",
                dfir.UnaryOp.CAST_BOOL: f"(({HLSBasicType.BOOL.value})(operand))",
                dfir.UnaryOp.CAST_INT: f"(({HLSBasicType.INT.value})(operand))",
                dfir.UnaryOp.CAST_FLOAT: f"(({HLSBasicType.FLOAT.value})(operand))",
                dfir.UnaryOp.SELECT: f"operand.ele_{self.val[1] if type(self.val) == tuple else '0'}",
                dfir.UnaryOp.GET_ATTR: f"operand.{self.val[1] if type(self.val) == tuple else 'ele_0'}",
            }
            if type(self.val) == tuple:
                expr_val_val = self.val[0]
            else:
                expr_val_val = self.val
            return trans_dict[expr_val_val].replace("operand", self.operands[0].code)
        elif self.type == HLSExprT.BINOP:
            if self.val in [dfir.BinOp.MAX, dfir.BinOp.MIN]:
                assert not self.contain_s_read
            return self.val.gen_repr_forbkd(self.operands[0].code, self.operands[1].code)
        else:
            assert False, f"Type {self.type} not supported"


class CodeAssign(HLSCodeLine):
    def __init__(self, var: HLSVar, expr: HLSExpr) -> None:
        super().__init__()
        assert type(var) == HLSVar
        if type(expr) == HLSVar:
            expr = HLSExpr(HLSExprT.VAR, expr)
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
        assert len(func.params) == len(params), f"{func.params} != {params}"

    def gen_code(self, indent_lvl: int = 0) -> str:
        def get_name(param):
            if isinstance(param, HLSVar):
                return param.name
            elif isinstance(param, HLSExpr):
                return param.code
            else:
                assert False

        return (
            INDENT_UNIT * indent_lvl
            + f"{self.func.name}("
            + ", ".join(get_name(var) for var in self.params)
            + ");\n"
        )


class CodeWriteStream(HLSCodeLine):
    def __init__(self, stream_var: HLSVar, in_expr: Union[HLSVar, HLSExpr]) -> None:
        super().__init__()
        self.stream_var = stream_var
        if type(in_expr) == HLSVar:
            in_expr = HLSExpr(HLSExprT.VAR, in_expr)
        self.in_expr = in_expr

    def gen_code(self, indent_lvl: int = 0) -> str:
        return INDENT_UNIT * indent_lvl + f"{self.stream_var.name}.write({self.in_expr.code});\n"


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
        return oind + "{\n" + "".join(c.gen_code(indent_lvl + 1) for c in self.codes) + oind + "}\n"


class CodeComment(HLSCodeLine):
    def __init__(self, text: str) -> None:
        super().__init__()
        assert "\n" not in text
        self.text = text

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + "// " + self.text.strip() + "\n"


class CodeOther(HLSCodeLine):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + self.text + "\n"


class HLSFunction:
    _readable_id_cnt = 0

    def __init__(
        self,
        name: str,
        comp: dfir.Component,
    ) -> None:
        self.name = name
        self.readable_id = HLSFunction._readable_id_cnt
        HLSFunction._readable_id_cnt += 1
        self.dfir_comp = comp
        self.params: List[HLSVar] = []
        self.codes: List[HLSCodeLine] = []
        # By default, a function is a standard streaming dataflow block.
        # This will be set to False for reduce sub-functions.
        self.streamed = True

    def __repr__(self) -> str:
        return f"HLSFunction({self.name}, {self.dfir_comp.name}, {self.dfir_comp.in_ports}, {self.params})"
