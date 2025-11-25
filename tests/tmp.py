import graphyflow.frontend as gf

graph = gf.Graph(
    node_props={
        "dist": gf.Int(width=16, default=gf.INFINITY),
    }
)

edges = graph.start_iteration([gf.ALL_EDGES])
keys = gf.map([edges], lambda e: e.dst)
updated_dists = gf.map([edges], lambda e: e.src.dist + e.weight)
reduced_dists = gf.reduce(
    keys=keys, values=[updated_dists], method=lambda a, b: gf.min(a, b), output_with_key=True
)
# output [dist, key]
graph.update_nodes(reduced_dists, update_prop="dist", update_method=lambda old, new: gf.min(old, new))

# Print all operations
if __name__ == "__main__":
    print("\n=== FRONTEND REPRESENTATION ===")
    print("\n=== Map Operation 1: Extract Keys ===")
    keys.print_details()

    print("=== Map Operation 2: Compute Updated Distances ===")
    updated_dists.print_details()

    print("=== Reduce Operation: Min Distance by Key ===")
    reduced_dists.print_details()

    print("=== Graph Update Operations ===")
    graph.print_all_operations()

    print("\n" + "=" * 70)
    print("=== MIDDLE-END IR REPRESENTATION ===")
    print("=" * 70)
    ir_graph = graph.to_ir()
    ir_graph.print_graph()
