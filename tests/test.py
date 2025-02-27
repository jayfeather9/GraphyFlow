from graphyflow.structures import *

g = GlobalGraph()
d = g.pseudo_element()
d.map_(map_func=lambda x: x + 1).filter(filter_func=lambda x: x > 1)
print(g)
