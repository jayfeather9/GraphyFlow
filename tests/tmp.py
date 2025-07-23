edges = g.add_graph_input("edge")
potential_dst_updates = edges.map(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
min_potential_distances = potential_dst_updates.reduce(
    key_func=lambda src_dist, dst, edge_w: dst,
    transform_func=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    reduce_func=lambda x, y: (min(x[0], y[0]), x[1]),
)
update_len = min_potential_distances.filter(
    filter_func=lambda dist, node: dist < node.distance
).length()
end_marker = update_len.map(map_func=lambda length: length == 0)
updated_node_distances = min_potential_distances.map(
    map_func=lambda dist, node: (min(dist, node.distance), node)
)
g.finish_iter(updated_node_distances, {"node": ["distance"]}, end_marker)
