from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir


class HLSDataType(Enum):
    UINT = "uint32_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    EDGE = "edge_t"
    NODE = "node_t"

    def __repr__(self) -> str:
        return self.value

    def to_hls_type(self, dfir_type: dftype.DfirType) -> str:
        translate_map = (
            (dftype.IntType, self.INT),
            (dftype.FloatType, self.FLOAT),
            (dftype.BoolType, self.BOOL),
            (dftype.SpecialType("edge"), self.EDGE),
            (dftype.SpecialType("node"), self.NODE),
        )
        for t, f in translate_map:
            if dfir_type == t:
                return f
        raise ValueError(f"Unsupported type: {dfir_type}")


class HLSFunction:
    def __init__(
        self,
        name: str,
        inputs: Dict[str, dftype.DfirType],
        outputs: Dict[str, dftype.DfirType],
        code_in_loop: List[str],
        comp: dfir.Component,
    ) -> None:
        assert (
            name not in global_hls_config.functions
        ), f"Function {name} already exists"
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.code_in_loop = code_in_loop
        self.comp = comp
        global_hls_config.functions[name] = self

    def __repr__(self) -> str:
        return f"HLSFunction(name={self.name}, inputs={self.inputs}, outputs={self.outputs}, code_in_loop={self.code_in_loop}, comp={self.comp})"


class HLSConfig:
    def __init__(self, header_name: str, source_name: str) -> None:
        self.header_name = header_name
        self.source_name = source_name
        self.includes = []
        self.defines = []
        self.structs = {}
        self.functions = {}

    def __repr__(self) -> str:
        return (
            f"HLSConfig(header_name={self.header_name}, source_name={self.source_name})"
        )


global_hls_config = HLSConfig("graphyflow.h", "graphyflow.cpp")

if __name__ == "__main__":
    hls_func = HLSFunction(
        "test",
        {"a": dftype.IntType(), "b": dftype.IntType()},
        {"c": dftype.IntType()},
        dfir.IOComponent(dfir.IOComponent.IOType.INPUT, dftype.IntType()),
    )
    print(hls_func)
