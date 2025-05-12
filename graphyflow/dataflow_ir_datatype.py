from __future__ import annotations
from typing import List


class DfirType:
    def __init__(self, type_name, is_optional: bool = False) -> None:
        self.type_name = type_name
        self.is_optional = is_optional

    def __eq__(self, other: DfirType) -> bool:
        return self.type_name == other.type_name

    def __hash__(self) -> int:
        return hash(self.type_name)

    def __repr__(self) -> str:
        return self.type_name


class SpecialType(DfirType):
    """Node or Edge"""

    def __init__(self, type_name: str) -> None:
        assert type_name in ["node", "edge"]
        super().__init__(type_name)


class IntType(DfirType):
    def __init__(self) -> None:
        super().__init__("Int")


class FloatType(DfirType):
    def __init__(self) -> None:
        super().__init__("Float")


class BoolType(DfirType):
    def __init__(self) -> None:
        super().__init__("Bool")


class OptionalType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        assert not isinstance(type_, OptionalType)
        super().__init__(f"Optional<{type_.type_name}>", True)
        self.type_ = type_


class TensorType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        assert not isinstance(type_, TensorType)
        super().__init__(f"Tensor<{type_.type_name}>")
        self.type_ = type_


class TupleType(DfirType):
    def __init__(self, types: List[DfirType]) -> None:
        super().__init__(f"Tuple<{', '.join([t.type_name for t in types])}>")
        self.types = types


class ArrayType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        super().__init__(f"Array<{type_.type_name}>")
        self.type_ = type_
