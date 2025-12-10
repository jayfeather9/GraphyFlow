from typing import Any, Tuple, Union, Type, Optional, List
from uuid import UUID
import uuid as uuid_lib
from abc import ABC
import graphyflow.dataflow_ir as dfir


class TransportType(ABC):
    def __init__(self):
        self.data = None


class BasicData(TransportType):
    """最基础的数据单元, 可以是int/float"""

    def __init__(self, data_type: dfir.DfirType):
        self._data_type = data_type

    @property
    def data_type(self):
        return self._data_type

    def to_dfir(self) -> dfir.DfirType:
        assert self.data_type in [dfir.IntType(), dfir.FloatType(), dfir.BoolType()]
        return self.data_type

    def __repr__(self):
        return f"BasicData({self._data_type})"


class BasicNode(TransportType):
    """图节点单元，代表某个节点，也包括若干个 BasicData"""

    def __init__(self, data_types: Optional[List[BasicData]] = None):
        if data_types is None:
            data_types = []
        assert all(isinstance(d, BasicData) for d in data_types)
        self.data_types = data_types

    def __repr__(self):
        return f"BasicNode({self.data_types})"


class BasicEdge(TransportType):
    """图边单元，代表某个边，也包括若干个 BasicData"""

    def __init__(self, data_types: Optional[List[BasicData]] = None):
        if data_types is None:
            data_types = []
        assert all(isinstance(d, BasicData) for d in data_types)
        self.data_types = data_types

    def __repr__(self):
        return f"BasicEdge({self.data_types})"


class DataElement:
    """每条边和每个节点传输的单位是 DataElement 对象, 内部可能包括若干个 BasicData"""

    def __init__(self, element_types: Tuple[Union[Any, Any]]):
        self.element_types = element_types
        assert all([isinstance(e, (BasicData, BasicNode, BasicEdge)) for e in element_types])

    def __repr__(self):
        return f"DataElement({self.element_types})"
