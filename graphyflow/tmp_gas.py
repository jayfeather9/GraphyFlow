from graphyflow.structures import *

def translate_graph(g: GlobalGraph):
    nodes = g.topo_sort_nodes()
    print(nodes)
    inputs = []
    while isinstance(nodes[0], Inputer):
        inputs.append(nodes[0])
        nodes.pop(0)
    assert isinstance(nodes[0], Map_) and isinstance(nodes[1], ReduceBy) and isinstance(nodes[2], Map_) and isinstance(nodes[3], Updater)
    scatter = nodes[0]
    gather = nodes[1]
    apply = nodes[2]
    update = nodes[3]