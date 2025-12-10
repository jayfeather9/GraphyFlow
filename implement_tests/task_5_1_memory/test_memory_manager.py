# implement_tests/task_5_1_memory/test_memory_manager.py

import pytest
from graphyflow.dataflow_ir import (
    ComponentCollection, IOComponent, ReduceComponent, MemoryReadComponent,
    UnusedEndMarkerComponent
)
from graphyflow.dataflow_ir_datatype import (
    ArrayType, SpecialType, IntType, FloatType, TupleType, SpecialIdType
)
from graphyflow.global_graph import GlobalGraph
from graphyflow.backend_mem_manager import MemoryAndGraphManager
from graphyflow.backend_defines import HLSBasicType, HLSType, HLSVar

@pytest.fixture
def mock_global_graph() -> GlobalGraph:
    return GlobalGraph(properties={
        "node": {"prop1": IntType()},
        "edge": {"weight": FloatType()}
    })

@pytest.fixture
def valid_cc(mock_global_graph: GlobalGraph) -> ComponentCollection:
    """
    Creates a structurally valid ComponentCollection for testing.
    All ports are either connected internally or listed as collection I/O.
    """
    # --- Components ---
    io_in = IOComponent(IOComponent.IOType.INPUT, ArrayType(SpecialType('edge')))
    mem_read_pre = MemoryReadComponent(
        access_pattern=[(0, 'edge', ['weight'])],
        output_types={'o_0_edge_weight': ArrayType(FloatType())},
        parallel=True
    )
    reduce_comp = ReduceComponent(
        input_type=ArrayType(FloatType()),
        accumulated_type=ArrayType(TupleType([IntType()])),
        reduce_key_out_type=ArrayType(IntType())
    )
    # The output type of reduce is Array<Tuple<Int>>, but mem_read_post needs Array<node_id>.
    # For this test fixture, we modify mem_read_post to accept the correct input type.
    mem_read_post = MemoryReadComponent(
        access_pattern=[(0, 'node', ['prop1'])],
        output_types={'o_0_node_prop1': ArrayType(IntType())},
        parallel=True
    )
    # Modify the input port type to match the output of the reduce component for a valid connection
    mem_read_post.get_port('i_0_node_id').data_type = reduce_comp.output_type

    # --- Connections ---
    io_in.get_port('o_0').connect(mem_read_pre.get_port('i_0_edge_id'))
    mem_read_pre.get_port('o_0_edge_weight').connect(reduce_comp.get_port('i_0'))
    reduce_comp.get_port('o_0').connect(mem_read_post.get_port('i_0_node_id'))
    
    # --- Handle unconnected internal ReduceComponent ports to make the CC valid ---
    components = [io_in, mem_read_pre, reduce_comp, mem_read_post]
    graph_inputs = []

    # Unconnected internal IN-ports of Reduce must be graph inputs
    for p in reduce_comp.get_port_group("key", "in") + \
               reduce_comp.get_port_group("transform", "in") + \
               reduce_comp.get_port_group("unit", "in"):
        if not p.connected:
            graph_inputs.append(p)
            
    # Unconnected internal OUT-ports of Reduce must be terminated
    for p in reduce_comp.get_port_group("key", "out") + \
               reduce_comp.get_port_group("transform", "out") + \
               reduce_comp.get_port_group("unit", "out"):
        if not p.connected:
            uem = UnusedEndMarkerComponent(p.data_type)
            p.connect(uem.get_port('i_0'))
            components.append(uem)
    
    final_output_port = mem_read_post.get_port('o_0_node_prop1')

    return ComponentCollection(components, graph_inputs, [final_output_port])

# --- Test Cases ---

def test_validation_success(valid_cc: ComponentCollection, mock_global_graph: GlobalGraph):
    try:
        manager = MemoryAndGraphManager(valid_cc, mock_global_graph)
        assert manager.io_comp is not None
        assert manager.reduce_comp is not None
        assert manager.pre_reduce_mem_read is not None
        assert manager.post_reduce_mem_read is not None
    except AssertionError as e:
        pytest.fail(f"Validation failed unexpectedly on a valid graph: {e}")

def test_fail_multiple_reduce(valid_cc: ComponentCollection, mock_global_graph: GlobalGraph):
    extra_reduce = ReduceComponent(ArrayType(IntType()), ArrayType(IntType()), ArrayType(IntType()))
    # To make the CC valid, its ports must be handled. For this test, we can just add it.
    # The validation should fail before the CC constructor is even an issue in the main logic.
    valid_cc.components.append(extra_reduce)
    with pytest.raises(AssertionError, match="Graph must have exactly one ReduceComponent"):
        MemoryAndGraphManager(valid_cc, mock_global_graph)

def test_fail_wrong_input_type(valid_cc: ComponentCollection, mock_global_graph: GlobalGraph):
    """Tests that validation fails if the input IOComponent is not of type 'edge'."""
    # Find and modify the IOComponent
    for comp in valid_cc.components:
        if isinstance(comp, IOComponent):
            comp.output_type = ArrayType(SpecialType('node')) # Change to 'node'
            break
            
    with pytest.raises(AssertionError, match="The single graph input must be of type Array<SpecialType"):
        MemoryAndGraphManager(valid_cc, mock_global_graph)

def test_fail_mixed_memread_source(valid_cc: ComponentCollection, mock_global_graph: GlobalGraph):
    """Tests that validation fails if a MemoryReadComponent reads from both node and edge."""
    for comp in valid_cc.components:
        if isinstance(comp, MemoryReadComponent) and comp.access_pattern[0][1] == 'edge':
            # Add a 'node' access pattern to the 'edge' reader
            comp.access_pattern.append((0, 'node', ['prop1']))
            # Also update output types to avoid unrelated errors
            comp.output_types['o_0_node_prop1'] = ArrayType(IntType())
            break
            
    with pytest.raises(AssertionError, match="must read from a single base type"):
        MemoryAndGraphManager(valid_cc, mock_global_graph)

def test_code_generation_snapshot(valid_cc: ComponentCollection, mock_global_graph: GlobalGraph):
    """Performs a snapshot test for the programmatically generated memory_loader."""
    manager = MemoryAndGraphManager(valid_cc, mock_global_graph)
    memory_loader_func = manager.memory_loader_func
    assert memory_loader_func is not None, "memory_loader_func was not generated"
    
    # Generate code for the function signature and body
    params_str = ", ".join([p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT) for p in memory_loader_func.params])
    signature = f"void memory_loader({params_str})"
    # We must also generate the code for all helper functions that memory_loader calls
    all_funcs_code = ""
    for helper in manager.helper_funcs:
        if helper: # Check if helper is implemented
            h_params = ", ".join([p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT) for p in helper.params])
            h_sig = f"static void {helper.name}({h_params})"
            h_body = "".join([line.gen_code(1) for line in helper.codes])
            all_funcs_code += f"{h_sig} {{\n{h_body}}}\n\n"
            
    body = "".join([line.gen_code(1) for line in memory_loader_func.codes])
    generated_code = f"{all_funcs_code}\n{signature} {{\n{body}}}"

    # Because the full implementation is now very long, we will test for key structural elements
    # instead of a full string match, which would be brittle.
    assert "#pragma HLS DATAFLOW" in generated_code
    assert "hls::stream<node_distance_burst_t> node_distance_burst_stream_0;" in generated_code
    assert "hls::stream<edge_descriptor_batch_t> edge_stream;" in generated_code
    assert "src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);" in generated_code
    assert "node_property_loader(node_distances, node_distance_burst_stream_0, node_distance_burst_stream_1, num_nodes);" in generated_code
    assert "edge_property_loader_and_dispatcher(src_offsets_cache_stream, edge_stream, node_distance_burst_stream_0, num_nodes, response_to_318);" in generated_code
    assert "node_property_responder(node_distance_burst_stream_1, num_nodes, all_node_distances_to_343);" in generated_code

    print("\nSnapshot test passed: Key structural elements found in generated code.")
