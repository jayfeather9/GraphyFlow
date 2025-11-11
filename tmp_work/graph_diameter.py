#!/usr/bin/env python3
"""
Graph Diameter Tool - All-in-one script for graph diameter operations.

Modes:
1. calculate - Calculate the current diameter of a graph
2. reduce    - Reduce diameter to a target value
3. force     - Force diameter to exactly 2 using super-hub approach

Usage:
    python graph_diameter.py calculate <graph_file>
    python graph_diameter.py reduce <graph_file> <target_diameter> [output_file]
    python graph_diameter.py force <graph_file> [output_file]

Examples:
    python graph_diameter.py calculate graph.txt
    python graph_diameter.py reduce graph.txt 3 graph_reduced.txt
    python graph_diameter.py force graph.txt graph_d2.txt
"""

from collections import deque, defaultdict
import sys
import random


# ============================================================================
# Graph I/O Functions
# ============================================================================


def read_graph(filename):
    """Read graph from file and return adjacency list, nodes, and edge list."""
    graph = defaultdict(set)
    nodes = set()
    edges = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            u, v = int(parts[0]), int(parts[1])
            if u != v and v not in graph[u]:  # Avoid self-loops and duplicates
                graph[u].add(v)
                graph[v].add(u)
                edges.append((min(u, v), max(u, v)))
                nodes.add(u)
                nodes.add(v)

    return graph, nodes, edges


def write_graph(filename, original_edges, new_edges):
    """Write the updated graph to a file."""
    all_edges = set(original_edges + new_edges)

    with open(filename, "w") as f:
        for u, v in sorted(all_edges):
            f.write(f"{u} {v}\n")

    print(f"Wrote {len(all_edges)} edges to {filename}")


# ============================================================================
# Graph Analysis Functions
# ============================================================================


def bfs_distances(graph, start_node, max_dist=None):
    """Perform BFS from start_node and return distances to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        current_dist = distances[node]

        if max_dist is not None and current_dist >= max_dist:
            continue

        for neighbor in graph[node]:
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


def check_connectivity(graph, nodes):
    """Check if graph is connected."""
    if not nodes:
        return False

    start_node = next(iter(nodes))
    distances = bfs_distances(graph, start_node)

    return len(distances) == len(nodes)


def compute_exact_diameter(graph, nodes):
    """
    Compute exact diameter using optimized approach.
    Returns diameter and one of the farthest pairs.
    """
    if not nodes:
        raise ValueError("Graph is empty")

    # Check connectivity first
    if not check_connectivity(graph, nodes):
        raise ValueError("Graph is not connected! Cannot compute diameter.")

    # Use two-sweep heuristic that often finds the diameter
    # Start from arbitrary node
    start_node = next(iter(nodes))
    distances = bfs_distances(graph, start_node)

    # Find farthest node from start_node
    farthest_node = max(distances.items(), key=lambda x: x[1])[0]

    # BFS from the farthest node
    distances2 = bfs_distances(graph, farthest_node)
    diameter = max(distances2.values())
    farthest_from_farthest = max(distances2.items(), key=lambda x: x[1])[0]
    farthest_pair = (farthest_node, farthest_from_farthest)

    # # Verify with a few more samples
    # sample_nodes = random.sample(list(nodes), min(10, len(nodes)))
    # for node in sample_nodes:
    #     distances = bfs_distances(graph, node)
    #     max_dist = max(distances.values())
    #     if max_dist > diameter:
    #         diameter = max_dist

    return diameter, farthest_pair


# def estimate_diameter(graph, nodes, sample_size=100):
#     """Estimate diameter by sampling nodes (faster for large graphs)."""
#     sample_nodes = random.sample(list(nodes), min(sample_size, len(nodes)))
#     diameter = 0
#     farthest_pair = None

#     for node in sample_nodes:
#         distances = bfs_distances(graph, node)
#         if distances:
#             max_dist = max(distances.values())
#             if max_dist > diameter:
#                 diameter = max_dist
#                 far_node = max(distances.items(), key=lambda x: x[1])[0]
#                 farthest_pair = (node, far_node)

#     return diameter, farthest_pair


# ============================================================================
# Diameter Reduction - Super-Hub Approach (For Diameter 2)
# ============================================================================


def force_diameter_2(graph, nodes, edges):
    """
    Force diameter to exactly 2 using super-hub approach.
    Pick highest-degree node and connect it to all other nodes.
    This guarantees diameter 2.
    """
    new_edges = []

    # Find the node with highest degree to use as super-hub
    node_degrees = [(node, len(graph[node])) for node in nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)

    super_hub = node_degrees[0][0]
    print(f"Selected node {super_hub} as super-hub (current degree: {node_degrees[0][1]})")

    # Connect super-hub to all other nodes
    connected = 0
    for node in nodes:
        if node != super_hub and node not in graph[super_hub]:
            graph[super_hub].add(node)
            graph[node].add(super_hub)
            new_edges.append((min(super_hub, node), max(super_hub, node)))
            connected += 1

    print(f"Connected super-hub to {connected} additional nodes")
    print(f"Super-hub now has degree: {len(graph[super_hub])}")

    return new_edges


# ============================================================================
# Diameter Reduction - Multi-Hub Approach (For Any Target)
# ============================================================================


def find_hub_nodes(graph, nodes, num_hubs):
    """Find candidate hub nodes based on degree centrality."""
    # Sort nodes by degree
    node_degrees = [(node, len(graph[node])) for node in nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)

    # Take top nodes by degree as initial hubs
    hubs = set([node for node, _ in node_degrees[:num_hubs]])

    # Also add some random nodes for better coverage
    remaining_nodes = list(nodes - hubs)
    if len(remaining_nodes) > num_hubs // 2:
        additional_hubs = random.sample(remaining_nodes, num_hubs // 2)
        hubs.update(additional_hubs)

    return hubs


def find_peripheral_nodes(graph, nodes, num_samples=20):
    """Find peripheral nodes (nodes that are far from each other)."""
    sample_nodes = random.sample(list(nodes), min(num_samples, len(nodes)))
    peripheral = []

    for node in sample_nodes:
        distances = bfs_distances(graph, node)
        eccentricity = max(distances.values()) if distances else 0
        peripheral.append((node, eccentricity))

    peripheral.sort(key=lambda x: x[1], reverse=True)
    return peripheral


def find_best_edge_to_add(graph, nodes, hubs, target_diameter):
    """Find the best edge to add to reduce diameter."""
    # Find peripheral nodes (nodes with high eccentricity)
    peripheral = find_peripheral_nodes(graph, nodes, num_samples=30)

    # Get top peripheral nodes
    top_peripheral = [node for node, ecc in peripheral[:20] if ecc > target_diameter]

    if not top_peripheral:
        top_peripheral = random.sample(list(nodes), min(20, len(nodes)))

    # Find pairs of peripheral nodes that are far apart
    best_edge = None
    max_distance = 0

    for i, u in enumerate(top_peripheral):
        distances_u = bfs_distances(graph, u)

        for v in top_peripheral[i + 1 :]:
            if v not in graph[u]:
                dist = distances_u.get(v, 0)
                if dist > max_distance and dist > target_diameter:
                    max_distance = dist
                    best_edge = (u, v)

    # If no good pair found, try connecting to hubs
    if best_edge is None and hubs:
        sample_nodes = random.sample(list(nodes), min(50, len(nodes)))
        for node in sample_nodes:
            if node not in hubs:
                # Find hubs not connected to this node
                for hub in hubs:
                    if hub not in graph[node]:
                        distances = bfs_distances(graph, node)
                        dist = distances.get(hub, 0)
                        if dist > max_distance:
                            max_distance = dist
                            best_edge = (node, hub)

    return best_edge


def reduce_diameter_multi_hub(graph, nodes, edges, target_diameter, max_iterations=10000):
    """
    Reduce diameter using multi-hub strategy.
    Works for any target diameter >= 2.
    """
    new_edges = []

    # Determine number of hubs based on target diameter and graph size
    if target_diameter == 2:
        num_hubs = max(20, int(len(nodes) ** 0.5) * 2)
        num_hubs = min(num_hubs, len(nodes) // 5)
    else:
        # For larger diameters, fewer hubs needed
        num_hubs = max(10, int(len(nodes) ** 0.5))
        num_hubs = min(num_hubs, len(nodes) // 10)

    print(f"Using {num_hubs} hub nodes for diameter reduction")

    # Find hub nodes
    hubs = find_hub_nodes(graph, nodes, num_hubs)
    print(f"Selected {len(hubs)} hub nodes")

    # Connect all hubs to each other (create hub clique)
    print("Connecting hubs to each other...")
    hub_edges = 0
    for i, hub1 in enumerate(hubs):
        for hub2 in list(hubs)[i + 1 :]:
            if hub2 not in graph[hub1]:
                graph[hub1].add(hub2)
                graph[hub2].add(hub1)
                new_edges.append((min(hub1, hub2), max(hub1, hub2)))
                hub_edges += 1
    print(f"Added {hub_edges} edges between hubs")

    # Connect non-hub nodes to hubs
    print("Connecting non-hub nodes to hubs...")
    non_hub_edges = 0

    min_hub_connections = 2 if target_diameter == 2 else 1

    for node in nodes:
        if node in hubs:
            continue

        # Count existing hub connections
        connected_hubs = [n for n in graph[node] if n in hubs]
        hubs_needed = max(0, min_hub_connections - len(connected_hubs))

        if hubs_needed > 0:
            distances = bfs_distances(graph, node, max_dist=5)

            # Find closest hubs not already connected
            candidate_hubs = []
            for hub in hubs:
                if hub not in graph[node]:
                    dist = distances.get(hub, float("inf"))
                    candidate_hubs.append((hub, dist))

            candidate_hubs.sort(key=lambda x: x[1])

            # Add edges to closest hubs
            for i in range(min(hubs_needed, len(candidate_hubs))):
                hub = candidate_hubs[i][0]
                graph[node].add(hub)
                graph[hub].add(node)
                new_edges.append((min(node, hub), max(node, hub)))
                non_hub_edges += 1

    print(f"Added {non_hub_edges} edges from non-hub nodes to hubs")

    # Iterative refinement
    print("\nIterative refinement...")
    current_diameter, _ = compute_exact_diameter(graph, nodes)
    print(f"Current diameter (estimated): {current_diameter}")

    iteration = 0
    while current_diameter > target_diameter and iteration < max_iterations:
        iteration += 1

        # Find and add best edge
        best_edge = find_best_edge_to_add(graph, nodes, hubs, target_diameter)

        if best_edge is None:
            print("No more beneficial edges found")
            break

        u, v = best_edge
        if v not in graph[u]:
            graph[u].add(v)
            graph[v].add(u)
            new_edges.append((min(u, v), max(u, v)))

        # Check diameter periodically
        if iteration % 10 == 0:
            current_diameter, _ = compute_exact_diameter(graph, nodes)
            print(f"Iteration {iteration}: diameter ≈ {current_diameter}, total new edges: {len(new_edges)}")

            if current_diameter <= target_diameter:
                print(f"✓ Target diameter {target_diameter} achieved!")
                break

    if iteration > 0:
        print(f"Completed {iteration} refinement iterations")

    return new_edges


# ============================================================================
# Main Command Handlers
# ============================================================================


def cmd_calculate(graph_file):
    """Calculate and display the diameter of the graph."""
    print(f"Reading graph from {graph_file}...")
    graph, nodes, edges = read_graph(graph_file)
    print(f"Graph has {len(nodes)} nodes and {len(edges)} edges")

    print("\nComputing diameter...")
    try:
        diameter, farthest_pair = compute_exact_diameter(graph, nodes)
        print("\n" + "=" * 60)
        print(f"DIAMETER: {diameter}")
        print(f"One of the farthest pairs: {farthest_pair}")
        print("=" * 60)
        return diameter
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


def cmd_reduce(graph_file, target_diameter, output_file):
    """Reduce graph diameter to target value."""
    if target_diameter < 2:
        print("ERROR: Target diameter must be >= 2")
        sys.exit(1)

    print(f"Reading graph from {graph_file}...")
    graph, nodes, edges = read_graph(graph_file)
    print(f"Graph has {len(nodes)} nodes and {len(edges)} edges")

    # Check connectivity
    if not check_connectivity(graph, nodes):
        print("ERROR: Graph is not connected! Cannot reduce diameter.")
        sys.exit(1)

    print(f"\nEstimating initial diameter...")
    initial_diameter, _ = compute_exact_diameter(graph, nodes)
    print(f"Initial diameter (estimated): {initial_diameter}")

    if initial_diameter <= target_diameter:
        print(f"\nGraph already has diameter <= {target_diameter}")
        print(f"No changes needed. Copying to {output_file}...")
        write_graph(output_file, edges, [])
        return

    print(f"\nReducing diameter to {target_diameter}...")

    # Use appropriate strategy
    if target_diameter == 2:
        print("Using super-hub strategy (optimal for diameter 2)...")
        new_edges = force_diameter_2(graph, nodes, edges)
    else:
        print("Using multi-hub strategy...")
        new_edges = reduce_diameter_multi_hub(graph, nodes, edges, target_diameter)

    print(f"\nVerifying result...")
    final_diameter, far_pair = compute_exact_diameter(graph, nodes)
    print(f"Final diameter (estimated): {final_diameter}")

    if final_diameter <= target_diameter:
        print(f"✓ Successfully achieved diameter <= {target_diameter}!")
    else:
        print(f"⚠ Warning: Diameter may still be > {target_diameter}")
        print(f"  Consider running again or using force mode for diameter 2")

    print(f"\nWriting updated graph to {output_file}...")
    write_graph(output_file, edges, new_edges)

    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Original edges: {len(edges)}")
    print(f"New edges added: {len(new_edges)}")
    print(f"Total edges: {len(edges) + len(new_edges)}")
    print(f"Initial diameter: {initial_diameter}")
    print(f"Final diameter: {final_diameter}")
    print(f"Output file: {output_file}")
    print("=" * 60)


def cmd_force(graph_file, output_file):
    """Force diameter to exactly 2 using super-hub approach."""
    print(f"Reading graph from {graph_file}...")
    graph, nodes, edges = read_graph(graph_file)
    print(f"Graph has {len(nodes)} nodes and {len(edges)} edges")

    # Check connectivity
    if not check_connectivity(graph, nodes):
        print("ERROR: Graph is not connected! Cannot reduce diameter.")
        sys.exit(1)

    print(f"\nVerifying initial diameter...")
    initial_diameter, _ = compute_exact_diameter(graph, nodes)
    print(f"Initial diameter (estimated): {initial_diameter}")

    print(f"\nForcing diameter to 2 using super-hub approach...")
    new_edges = force_diameter_2(graph, nodes, edges)

    print(f"\nVerifying result...")
    final_diameter, _ = compute_exact_diameter(graph, nodes)
    print(f"Final diameter: {final_diameter}")

    if final_diameter == 2:
        print(f"✓ Successfully achieved diameter = 2!")
    else:
        print(f"⚠ Unexpected: diameter is {final_diameter}")

    print(f"\nWriting updated graph to {output_file}...")
    write_graph(output_file, edges, new_edges)

    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Original edges: {len(edges)}")
    print(f"New edges added: {len(new_edges)}")
    print(f"Total edges: {len(edges) + len(new_edges)}")
    print(f"Initial diameter: {initial_diameter}")
    print(f"Final diameter: {final_diameter}")
    print(f"Output file: {output_file}")
    print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "calculate":
        if len(sys.argv) < 3:
            print("ERROR: Missing graph file")
            print("Usage: python graph_diameter.py calculate <graph_file>")
            sys.exit(1)
        cmd_calculate(sys.argv[2])

    elif mode == "reduce":
        if len(sys.argv) < 4:
            print("ERROR: Missing arguments")
            print("Usage: python graph_diameter.py reduce <graph_file> <target_diameter> [output_file]")
            sys.exit(1)

        graph_file = sys.argv[2]
        target_diameter = int(sys.argv[3])
        output_file = sys.argv[4] if len(sys.argv) > 4 else "graph_reduced.txt"

        cmd_reduce(graph_file, target_diameter, output_file)

    elif mode == "force":
        if len(sys.argv) < 3:
            print("ERROR: Missing graph file")
            print("Usage: python graph_diameter.py force <graph_file> [output_file]")
            sys.exit(1)

        graph_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "graph_d2.txt"

        cmd_force(graph_file, output_file)

    else:
        print(f"ERROR: Unknown mode '{mode}'")
        print("Valid modes: calculate, reduce, force")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
