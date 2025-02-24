from typing import Callable
from __future__ import annotations

class DataElement:
    def __init__(self):
        pass

class ChainMethod:
    def __init__(self) -> None:
        pass

class Filter(ChainMethod):
    def __init__(self, filter_func: Callable[[DataElement], DataElement]):
        self.filter_func = filter_func
        super().__init__()

class Map_(ChainMethod):
    def __init__(self, map_func: Callable[[DataElement], DataElement]):
        self.map_func = map_func
        super().__init__()

class ReduceBy(ChainMethod):
    def __init__(
        self,
        reduce_key: Callable[[DataElement], DataElement],
        reduce_method: Callable[[DataElement], DataElement],
    ):
        self.reduce_key = reduce_key
        self.reduce_method = reduce_method
        super().__init__()

class DataflowChain:
    def __init__(self):
        self.chain = []
    
    def filter(self, func: Callable[[DataElement], DataElement]) -> DataflowChain:
        self.chain.append(Filter(func))
        return self
    
    def map_(self, func: Callable[[DataElement], DataElement]) -> DataflowChain:
        self.chain.append(Map_(func))
        return self
    
    def reduce_by(
        self,
        reduce_key: Callable[[DataElement], DataElement],
        reduce_method: Callable[[DataElement], DataElement],
    ) -> DataflowChain:
        self.chain.append(ReduceBy(reduce_key, reduce_method))
        return self
