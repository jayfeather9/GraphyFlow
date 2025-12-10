import pytest
import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from graphyflow.simulate import DfirSimulator


@pytest.fixture(scope="module")
def mem_read_test_setup():
    """
    Provides a complex setup for testing MemoryReadComponent simulation.
    Includes a graph, component, and nested property data.
    """
    # --- 1. Define a global graph schema ---
    g = GlobalGraph(
        properties={
            "node": {
                "id": dfir.IntType(),
                "value": dfir.FloatType(),
                "group": dfir.IntType(),
            },
            "edge": {
                "weight": dfir.FloatType(),
                "type": dfir.IntType(),
            },
        }
    )

    # --- 2. Define a complex access pattern ---
    # This pattern reads:
    # - from input 0, base 'edge': the edge's weight
    # - from input 0, base 'edge': the src node's value property
    # - from input 1, base 'node': the node's group property
    access_pattern = [
        (0, "edge", ["weight"]),
        (0, "edge", ["src", "value"]),
        (1, "node", ["group"]),
    ]

    output_types = {
        "o_0_edge_weight": dfir.ArrayType(dfir.FloatType()),
        "o_0_edge_src_value": dfir.ArrayType(dfir.FloatType()),
        "o_1_node_group": dfir.ArrayType(dfir.IntType()),
    }

    # --- 3. Manually build the MemoryReadComponent and its wrapper ---
    mem_read_comp = dfir.MemoryReadComponent(
        access_pattern=access_pattern,
        output_types=output_types,
        parallel=True,
    )

    # The component collection represents the full graph to be simulated
    main_cc = dfir.ComponentCollection(
        components=[mem_read_comp],
        inputs=mem_read_comp.in_ports,
        outputs=mem_read_comp.out_ports,
    )

    # --- 4. Define mock graph data ---
    node_data = {
        10: {"id": 10, "value": 100.1, "group": 1},
        20: {"id": 20, "value": 200.2, "group": 1},
        30: {"id": 30, "value": 300.3, "group": 2},
    }
    edge_data = {
        101: {"src": 10, "dst": 20, "weight": 1.5, "type": 0},
        102: {"src": 20, "dst": 30, "weight": 2.5, "type": 1},
        103: {"src": 30, "dst": 10, "weight": 3.5, "type": 0},
        105: {"src": 10, "dst": 30, "weight": 4.5, "type": 1},
    }

    return {
        "graph": g,
        "collection": main_cc,
        "node_data": node_data,
        "edge_data": edge_data,
    }


def test_simulate_mem_read_parallel(mem_read_test_setup):
    """
    Tests the MemoryReadComponent simulation in parallel (batch) mode.
    """
    # --- 1. Setup simulator ---
    g = mem_read_test_setup["graph"]
    collection = mem_read_test_setup["collection"]
    node_data = mem_read_test_setup["node_data"]
    edge_data = mem_read_test_setup["edge_data"]

    simulator = DfirSimulator(collection, g)
    simulator.add_nodes(nodes=list(node_data.keys()), props=node_data)
    simulator.add_edges(
        edges={i: (d["src"], d["dst"]) for i, d in edge_data.items()},
        props=edge_data,
    )

    # --- 2. Define inputs for the simulation run ---
    # These names must match the input port names of the MemoryReadComponent
    sim_inputs = {
        "i_0_edge_id": [101, 102, 105, 103],
        "i_1_node_id": [10, 30],
    }

    # --- 3. Run simulation ---
    results = simulator.run(sim_inputs)

    # --- 4. Assert correctness of results ---
    assert len(results) == 3  # We expect 3 output ports

    # Expected values derived from mock data and access pattern
    expected_weights = [
        1.5,
        2.5,
        4.5,
        3.5,
    ]  # edge[101].weight, edge[102].weight, edge[105].weight, edge[103].weight
    expected_src_values = [100.1, 200.2, 100.1, 300.3]  # node[edge[101].src].value, node[edge[102].src].value
    expected_groups = [1, 2]  # node[10].group, node[30].group

    assert "o_0_edge_weight" in results
    assert results["o_0_edge_weight"] == expected_weights

    assert "o_0_edge_src_value" in results
    assert results["o_0_edge_src_value"] == expected_src_values

    assert "o_1_node_group" in results
    assert results["o_1_node_group"] == expected_groups
