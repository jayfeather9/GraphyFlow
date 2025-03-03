from typing import Any


class DataElement:
    def __init__(self):
        pass


class DataNode(DataElement):
    def __init__(self, datas: Any = None):
        super().__init__()
        self._datas = datas

    def __repr__(self):
        return f"DataNode({self._datas})"

    @property
    def datas(self):
        return self._datas


class DataEdge(DataElement):
    def __init__(self, src: DataNode, dst: DataNode, datas: Any = None):
        super().__init__()
        self._src = src
        self._dst = dst
        self._datas = datas

    def __repr__(self):
        return f"DataEdge({self._src}, {self._dst}, {self._datas})"

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def datas(self):
        return self._datas
