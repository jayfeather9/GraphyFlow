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

class DataflowChain:
    def __init__(self):
        self.chain = []
    
    def filter(self, func: Callable[[DataElement], DataElement]) -> DataflowChain:
        self.chain.append(Filter(func))
        return self