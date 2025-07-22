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
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    STRUCT = "struct"
    STREAM = "stream"

    def __repr__(self) -> str:
        return self.value

    @property
    def is_simple(self) -> bool:
        return not (self in [HLSBasicType.STRUCT, HLSBasicType.STREAM])


class HLSType:
    def __init__(
        self, basic_type: HLSBasicType, sub_types: Optional[List[HLSType]] = None
    ) -> None:
        self.type = basic_type
        if basic_type.is_simple:
            assert sub_types is None
            self.name = basic_type.value
        else:
            assert type(sub_types) == list and len(sub_types) > 0
            if basic_type == HLSBasicType.STREAM:
                assert len(sub_types) == 1
                self.name = f"stream<{sub_types[0].name}>"
            elif basic_type == HLSBasicType.STRUCT:
                self.name = f"struct_" + "".join(name[0] for name in sub_types) + "_t"
            else:
                assert False, f"Basic type {basic_type} not supported"
        self.sub_types = sub_types

    def gen_decl(self):
        assert self.type == HLSBasicType.STRUCT
        return (
            "typedef struct {\n"
            + INDENT_UNIT
            + f"\n{INDENT_UNIT}".join(
                f"{sub_type} ele_{i}" for i, sub_type in enumerate(self.sub_types)
            )
            + "} "
            + self.name
            + ";\n"
        )


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
            assert operands[0].val.type.type == HLSBasicType.STREAM
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
        if self.type == HLSExprType.CONST or self.type == HLSExprType.VAR:
            return False
        elif self.type == HLSExprType.STREAM_READ:
            return True
        elif self.type == HLSExprType.UOP or self.type == HLSExprType.BINOP:
            for operand in self.operands:
                if operand.contain_s_read():
                    return True
            return False
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
        for var, call_var in zip(func.vars, params):
            assert var.type == call_var.type

    def gen_code(self, indent_lvl: int = 0) -> str:
        return (
            INDENT_UNIT * indent_lvl
            + f"{self.func.name}("
            + "".join(var.name for var in self.params)
            + ");\n"
        )


class CodeWriteStream(HLSCodeLine):
    def __init__(self, stream_var: HLSVar, in_var: HLSVar) -> None:
        super().__init__()
        assert stream_var.type.type == HLSBasicType.STREAM
        assert in_var.type.type != HLSBasicType.STREAM
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
        codes: List[HLSCodeLine],
        params: Optional[List[Tuple[HLSType, str]]] = None,
    ) -> None:
        self.name = name
        self.dfir_comp = comp
        self.params = params
        self.codes = codes
        self.vars: List[HLSVar] = []
        if self.params:
            for p_type, p_name in self.params:
                self.vars.append(HLSVar(p_name, p_type))
