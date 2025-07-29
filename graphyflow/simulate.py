import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from typing import Dict, List, Tuple, Any, Callable, Optional, Set
import collections


class UncertainArray:
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"UncertainArray({self.value})"


def _simulate_component(
    comp: dfir.Component,
    inputs: Dict[str, Any],
    node_props: Dict[int, Dict[str, Any]],
    edge_props: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simulates a single DFIR component based on its type and inputs.
    Handles parallel (array) operations where applicable.
    """
    # Remove TODO for Array<> Inputs as we are addressing it now
    support_components = (
        dfir.BinOpComponent,
        dfir.UnaryOpComponent,
        dfir.ConstantComponent,
        dfir.CopyComponent,
        dfir.GatherComponent,
        dfir.ScatterComponent,
        dfir.ConditionalComponent,
        dfir.CollectComponent,
        dfir.PlaceholderComponent,
        dfir.ReduceComponent,  # Add ReduceComponent here as _simulate_component handles it
    )
    # Check if comp is one of the base supported types or ReduceComponent
    assert isinstance(
        comp, support_components
    ), f"Component type {type(comp).__name__} is not directly supported by _simulate_component base logic or ReduceComponent logic."

    # --- Input Port Check (Moved earlier for clarity) ---
    # For ReduceComponent, only 'i_0' might be directly available when called *by* run_flow's loop.
    # Other components expect all their inputs.
    if not isinstance(comp, dfir.ReduceComponent):
        assert all(
            p.name in inputs.keys() for p in comp.in_ports
        ), f"Component {comp.uuid} ({type(comp).__name__}) is missing inputs. Got: {list(inputs.keys())}, Expected: {[p.name for p in comp.in_ports]}"

    # --- Component Simulation Logic ---
    if isinstance(comp, dfir.PlaceholderComponent):
        # Handles scalar or array pass-through
        return {"o_0": inputs["i_0"]}

    if isinstance(comp, dfir.ConstantComponent):
        # Handles scalar or array constant
        return {"o_0": comp.value if not comp.parallel else UncertainArray(comp.value)}

    if isinstance(comp, dfir.CopyComponent):
        # Handles scalar or array copy (by reference for arrays)
        return {"o_0": inputs["i_0"], "o_1": inputs["i_0"]}

    if isinstance(comp, dfir.GatherComponent):
        # Input ports are i_0, i_1, ...
        # Output port is o_0
        input_values = [inputs[p.name] for p in comp.in_ports]

        if comp.parallel:
            # Ensure all input lists have the same length
            first_len = len(input_values[0])
            assert all(
                len(val) == first_len for val in input_values
            ), "Parallel GatherComponent requires all input arrays to have the same length"
            # Zip the lists element-wise: [[a1, b1], [a2, b2], ...] -> [(a1, a2), (b1, b2)]
            output_list = list(zip(*input_values, strict=True))  # Result is Array<Tuple<...>>
            return {"o_0": output_list}
        else:
            # Scalar inputs: gather into a single tuple
            output_tuple = tuple(input_values)  # Result is Tuple<...>
            return {"o_0": output_tuple}

    if isinstance(comp, dfir.ScatterComponent):
        # Input port is i_0
        # Output ports are o_0, o_1, ...
        input_val = inputs["i_0"]

        if comp.parallel:
            # Input is Array<Tuple<T...>>
            # Each element of input_val is a tuple to be scattered
            num_outputs = len(comp.out_ports)
            # Check if tuples have the correct size
            if input_val:  # Check if list is not empty
                assert all(
                    isinstance(item, (list, tuple)) and len(item) == num_outputs for item in input_val
                ), f"Parallel ScatterComponent input elements must be tuples/lists of length {num_outputs}"

            # Unzip the list of tuples: [(a1, b1), (a2, b2), ...] -> ([a1, a2], [b1, b2])
            scattered_outputs = list(zip(*input_val))

            return {p.name: list(scattered_outputs[i]) for i, p in enumerate(comp.out_ports)}
        else:
            # Input is Tuple<T...>
            assert isinstance(
                input_val, (list, tuple)
            ), "Non-parallel ScatterComponent expects tuple/list input"
            assert len(input_val) == len(
                comp.out_ports
            ), f"Non-parallel ScatterComponent input length mismatch: expected {len(comp.out_ports)}, got {len(input_val)}"
            # Scatter elements of the tuple to corresponding output ports
            return {p.name: input_val[i] for i, p in enumerate(comp.out_ports)}

    if isinstance(comp, dfir.BinOpComponent):
        op_func = {  # Keep original scalar functions
            dfir.BinOp.ADD: lambda x, y: x + y,
            dfir.BinOp.SUB: lambda x, y: x - y,
            dfir.BinOp.MUL: lambda x, y: x * y,
            dfir.BinOp.DIV: lambda x, y: x / y,
            dfir.BinOp.AND: lambda x, y: x & y,
            dfir.BinOp.OR: lambda x, y: x | y,
            dfir.BinOp.EQ: lambda x, y: x == y,
            dfir.BinOp.NE: lambda x, y: x != y,
            dfir.BinOp.LT: lambda x, y: x < y,
            dfir.BinOp.GT: lambda x, y: x > y,
            dfir.BinOp.LE: lambda x, y: x <= y,
            dfir.BinOp.GE: lambda x, y: x >= y,
            dfir.BinOp.MIN: lambda x, y: min(x, y),
            dfir.BinOp.MAX: lambda x, y: max(x, y),
        }[comp.op]
        in1 = inputs["i_0"]
        in2 = inputs["i_1"]

        if isinstance(in1, UncertainArray):
            assert not isinstance(in2, UncertainArray)
            in1 = [in1.value] * len(in2)
        if isinstance(in2, UncertainArray):
            assert not isinstance(in1, UncertainArray)
            in2 = [in2.value] * len(in1)

        # Assume comp has a 'parallel' attribute based on dfir definition
        if getattr(comp, "parallel", False):  # Check if parallel attribute exists and is True
            assert isinstance(in1, list) and isinstance(
                in2, list
            ), f"Parallel BinOp requires list inputs, got {in1=}, {in2=}"
            # Handle potential broadcast if lengths differ? For now, assume same length.
            assert len(in1) == len(in2), "Parallel BinOp requires inputs of same length"
            result = [op_func(x, y) for x, y in zip(in1, in2)]
        else:
            # Scalar operation
            result = op_func(in1, in2)
        return {"o_0": result}

    if isinstance(comp, dfir.UnaryOpComponent):
        input_val = inputs["i_0"]
        input_port_type = comp.in_ports[0].data_type

        op_func = {  # Keep original scalar functions
            dfir.UnaryOp.NOT: lambda x: not x,
            dfir.UnaryOp.NEG: lambda x: -x,
            dfir.UnaryOp.CAST_BOOL: lambda x: bool(x),
            dfir.UnaryOp.CAST_INT: lambda x: int(x),
            dfir.UnaryOp.CAST_FLOAT: lambda x: float(x),
            dfir.UnaryOp.SELECT: lambda x: x[comp.select_index],  # Array handling?
            dfir.UnaryOp.GET_LENGTH: lambda x: len(x),  # Array handling?
            dfir.UnaryOp.GET_ATTR: lambda x: (  # Scalar Node/Edge ID expected here
                node_props[x][comp.select_index]
                if isinstance(input_port_type.type_, dfir.SpecialType)
                and input_port_type.type_.type_name == "node"
                else (
                    edge_props[x][comp.select_index]
                    if isinstance(input_port_type.type_, dfir.SpecialType)
                    and input_port_type.type_.type_name == "edge"
                    else (_ for _ in ()).throw(
                        AssertionError(f"GET_ATTR expects input type 'node' or 'edge', got {input_port_type}")
                    )
                )
            ),
        }[comp.op]

        if isinstance(input_val, UncertainArray):
            assert comp.parallel, "UncertainArray input is only supported for parallel UnaryOp"
            result_val = op_func(input_val.value)
            return {"o_0": UncertainArray(result_val)}

        # Assume comp has a 'parallel' attribute
        if getattr(comp, "parallel", False):
            assert isinstance(input_val, list), "Parallel UnaryOp requires list input"
            result = [op_func(item) for item in input_val]
        else:
            result = op_func(input_val)
        return {"o_0": result}

    if isinstance(comp, dfir.ConditionalComponent):
        # This already handles the 'parallel' case correctly. No changes needed.
        data_in = inputs["i_data"]
        cond_in = inputs["i_cond"]
        if comp.parallel:
            assert isinstance(data_in, list), "Parallel ConditionalComponent expects 'i_data' to be a list"
            assert isinstance(cond_in, list), "Parallel ConditionalComponent expects 'i_cond' to be a list"
            assert len(data_in) == len(
                cond_in
            ), "Parallel ConditionalComponent requires 'i_data' and 'i_cond' lists to have the same length"
            output_list = [
                data_item if bool(cond_item) else None for data_item, cond_item in zip(data_in, cond_in)
            ]
            return {"o_0": output_list}
        else:
            return {"o_0": data_in if bool(cond_in) else None}

    if isinstance(comp, dfir.CollectComponent):
        # This operates on Array<Optional<T>> -> Array<T>. Handles arrays correctly. No changes needed.
        optional_data_in = inputs["i_0"]
        assert isinstance(optional_data_in, list), "CollectComponent expects 'i_0' to be a list"
        output_list = [item for item in optional_data_in if item is not None]
        return {"o_0": output_list}

    if isinstance(comp, dfir.ReduceComponent):
        input_array = inputs["i_0"]
        assert isinstance(input_array, list), "ReduceComponent expects 'i_0' to be a list"

        def find_subgraph_entry_port(start_port: dfir.Port) -> dfir.Port:
            assert start_port.port_type == dfir.PortType.OUT
            assert start_port.connected
            return start_port.connection

        def find_subgraph_exit_port(end_port: dfir.Port) -> dfir.Port:
            assert end_port.port_type == dfir.PortType.IN
            assert end_port.connected
            return end_port.connection

        key_entry = find_subgraph_entry_port(comp.get_port("o_reduce_key_in"))
        key_exit = find_subgraph_exit_port(comp.get_port("i_reduce_key_out"))
        transform_entry = find_subgraph_entry_port(comp.get_port("o_reduce_transform_in"))
        transform_exit = find_subgraph_exit_port(comp.get_port("i_reduce_transform_out"))
        accum_entry_0 = find_subgraph_entry_port(comp.get_port("o_reduce_unit_start_0"))
        accum_entry_1 = find_subgraph_entry_port(comp.get_port("o_reduce_unit_start_1"))
        reduce_exit = find_subgraph_exit_port(comp.get_port("i_reduce_unit_end"))
        groups = {}
        for element in input_array:
            key_result_dict = run_flow({key_entry: element}, [key_exit], node_props, edge_props)
            key = key_result_dict[key_exit]
            if key not in groups:
                groups[key] = []
            groups[key].append(element)
        final_results = []
        for key, elements_in_group in groups.items():
            if not elements_in_group:
                continue
            accumulated_value = None
            is_first = True
            for element in elements_in_group:
                if is_first:
                    reduction_result_dict = run_flow(
                        {transform_entry: element},
                        [transform_exit],
                        node_props,
                        edge_props,
                    )
                    accumulated_value = reduction_result_dict[transform_exit]
                    is_first = False
                else:
                    transform_result_dict = run_flow(
                        {transform_entry: element},
                        [transform_exit],
                        node_props,
                        edge_props,
                    )
                    reduction_result_dict = run_flow(
                        {
                            accum_entry_0: accumulated_value,
                            accum_entry_1: transform_result_dict[transform_exit],
                        },
                        [reduce_exit],
                        node_props,
                        edge_props,
                    )
                    accumulated_value = reduction_result_dict[reduce_exit]
            final_results.append(accumulated_value)
        return {"o_0": final_results}
    else:
        # Handle unsupported components
        raise NotImplementedError(
            f"Simulation logic not implemented in _simulate_component for: {type(comp).__name__}"
        )


def run_flow(
    from_ports_values: Dict[dfir.Port, Any],
    to_ports: List[dfir.Port],
    node_props: Dict[int, Dict[str, Any]],
    edge_props: Dict[int, Dict[str, Any]],
    run_no_input_components: List[dfir.Component] = [],
    data_for_io_comp=None,
) -> Dict[dfir.Port, Any]:
    """
    Runs a data flow simulation from a set of starting ports/values
    to a set of target output ports.

    Args:
        from_ports_values: Dict mapping starting input ports to their initial values.
                           Ports must be input ports.
        to_ports: List of target output ports. The simulation stops when all
                  these ports have computed values.
        node_props: Dictionary of node properties.
        edge_props: Dictionary of edge properties.

    Returns:
        A dictionary mapping the target output ports (from to_ports) to their
        computed values.
    """
    computed_values: Dict[dfir.Port, Any] = from_ports_values.copy()
    target_ports_set = set(to_ports)
    computed_target_ports: Set[dfir.Port] = set()
    final_results: Dict[dfir.Port, Any] = {}
    # Check if any target ports are already provided as inputs
    for port in from_ports_values:
        # Input ports shouldn't be target ports (which are outputs)
        assert port not in target_ports_set, f"Target port {port} cannot be an input port."

    ready_queue = collections.deque()
    processed_components = set()
    for start_port, _ in from_ports_values.items():
        assert (
            start_port.port_type == dfir.PortType.IN
        ), f"Port {start_port} in from_ports_values must be an input port."
        component = start_port.parent
        if component.uuid not in processed_components:
            assert all(
                in_port in computed_values for in_port in component.in_ports
            ), f"Component {component.uuid} has inputs not in {computed_values=}"
            ready_queue.append(component)
            processed_components.add(component.uuid)

    for component in list(set(run_no_input_components) - set(ready_queue)):
        assert len(component.in_ports) == 0, f"Component {component.uuid} has input ports"
        ready_queue.append(component)

    # --- Simulation Loop ---
    executed_component_ids = set()  # Track executed components to prevent cycles in simple cases

    while ready_queue and computed_target_ports != target_ports_set:
        component_to_run = ready_queue.popleft()

        # Prevent re-execution in this flow (basic cycle detection)
        assert (
            component_to_run.uuid not in executed_component_ids
        ), f"Component {component_to_run.uuid} already executed"
        executed_component_ids.add(component_to_run.uuid)

        # Gather inputs for this component
        current_inputs = {}
        if isinstance(component_to_run, dfir.ReduceComponent):
            # get the "i_0" port and assert it is in computed_values
            i_0_port = component_to_run.get_port("i_0")
            assert (
                i_0_port in computed_values
            ), f"Component {component_to_run.uuid} has input port 'i_0' not in {computed_values=}"
            current_inputs["i_0"] = computed_values[i_0_port]
        else:
            assert all(
                in_port in computed_values for in_port in component_to_run.in_ports
            ), f"Component {component_to_run.uuid} has inputs not in {computed_values=}"
            current_inputs = {in_port.name: computed_values[in_port] for in_port in component_to_run.in_ports}

        if isinstance(component_to_run, dfir.IOComponent):
            assert (
                data_for_io_comp is not None
            ), f"Data for IO component {component_to_run.uuid} is not provided"
            assert (
                component_to_run.io_type == dfir.IOComponent.IOType.INPUT
            ), f"IO component {component_to_run.uuid} is not an input component"
            outputs = {"o_0": data_for_io_comp[component_to_run.output_type.type_.type_name]}
        else:
            outputs = _simulate_component(component_to_run, current_inputs, node_props, edge_props)
        # print(f"Finished running component {component_to_run}, outputs: {outputs}")

        # --- Process outputs and update state ---
        for out_port_name, value in outputs.items():
            out_port = component_to_run.get_port(out_port_name)
            if out_port in target_ports_set:
                computed_target_ports.add(out_port)
                final_results[out_port] = value
                continue
            connected_in_port = out_port.connection
            assert connected_in_port is not None, f"Output port {out_port} is not connected to any input port"
            computed_values[connected_in_port] = value

            # Find downstream components and check readiness
            downstream_comp = connected_in_port.parent

            # Check if downstream component is now ready
            is_ready = all(in_port in computed_values for in_port in downstream_comp.in_ports)
            if (
                isinstance(downstream_comp, dfir.ReduceComponent)
                and downstream_comp.get_port("i_0") in computed_values
            ):
                is_ready = True
            if (
                is_ready
                and downstream_comp not in ready_queue
                and downstream_comp.uuid not in executed_component_ids
            ):
                ready_queue.append(downstream_comp)

    # --- Final Check and Return ---
    if computed_target_ports != target_ports_set:
        missing = target_ports_set - computed_target_ports
        raise RuntimeError(f"run_flow failed to compute all target ports. Missing: {missing}")

    return final_results


class DfirSimulator:
    def __init__(self, dfirs: dfir.ComponentCollection, g: GlobalGraph) -> None:
        self.dfirs = dfirs
        self.node_properties_schema = g.node_properties
        self.edge_properties_schema = g.edge_properties
        self.node_data = {}
        self.edge_data = {}

    def add_nodes(self, nodes: List[int], props: Dict[int, Dict[str, Any]]):
        assert len(set(nodes)) == len(nodes)
        self.node_data = {i: {} for i in nodes}
        for i in self.node_data.keys():
            assert i in props.keys()
            for prop_name in self.node_properties_schema:
                if prop_name == "id":
                    self.node_data[i][prop_name] = i
                    continue
                assert prop_name in props[i].keys(), f"Node {i} missing property '{prop_name}'"
                self.node_data[i][prop_name] = props[i][prop_name]

    def add_edges(self, edges: Dict[int, Tuple[int, int]], props: Dict[int, Dict[str, Any]]):
        assert len(set(edges.keys())) == len(edges.keys())
        self.edge_data = {i: {"src": v[0], "dst": v[1]} for i, v in edges.items()}
        for i in self.edge_data.keys():
            assert i in props.keys()
            for prop_name in self.edge_properties_schema:
                if prop_name in ["src", "dst"]:
                    continue
                assert prop_name in props[i].keys(), f"Edge {i} missing property '{prop_name}'"
                self.edge_data[i][prop_name] = props[i][prop_name]

    def run(self, initial_graph_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the simulation for the entire dataflow graph.

        Args:
            initial_graph_inputs: A dictionary mapping the names of the graph's
                                  input ports to their initial values.

        Returns:
            A dictionary mapping the names of the graph's output ports to
            their final computed values.
        """
        graph_input_ports: List[dfir.Port] = self.dfirs.inputs
        graph_output_ports: List[dfir.Port] = self.dfirs.outputs

        from_ports_values: Dict[dfir.Port, Any] = {}
        provided_input_names = set(initial_graph_inputs.keys())
        used_input_names = set()

        for port in graph_input_ports:
            assert port.port_type == dfir.PortType.IN, f"Graph input port {port} is not an IN port."
            if port.name in initial_graph_inputs:
                from_ports_values[port] = initial_graph_inputs[port.name]
                used_input_names.add(port.name)
            else:
                raise ValueError(f"Initial value for graph input port '{port.name}' not provided.")

        unused_input_names = provided_input_names - used_input_names
        if unused_input_names:
            raise ValueError(
                f"Provided initial inputs do not correspond to graph input ports: {unused_input_names}"
            )

        to_ports: List[dfir.Port] = graph_output_ports
        for port in to_ports:
            assert (
                port.port_type == dfir.PortType.OUT
            ), f"Graph output port {port} expected to be an OUT port, but got {port.port_type}"

        no_input_components = []
        for component in self.dfirs.components:
            if len(component.in_ports) == 0:
                no_input_components.append(component)

        final_port_values: Dict[dfir.Port, Any] = run_flow(
            from_ports_values=from_ports_values,
            to_ports=to_ports,
            node_props=self.node_data,
            edge_props=self.edge_data,
            run_no_input_components=no_input_components,
            data_for_io_comp={
                "node": [node_id for node_id in self.node_data.keys()],
                "edge": [edge_id for edge_id in self.edge_data.keys()],
            },
        )

        final_results_by_name: Dict[str, Any] = {
            port.name: final_port_values[port] for port in to_ports if port in final_port_values
        }

        if len(final_results_by_name) != len(to_ports):
            computed_port_names = set(final_results_by_name.keys())
            requested_port_names = set(p.name for p in to_ports)
            missing_names = requested_port_names - computed_port_names
            print(
                f"Warning: Simulation finished, but some graph output ports were not computed by run_flow: {missing_names}"
            )

        return final_results_by_name


if __name__ == "__main__":
    print("Running Simulation Test...")

    g = GlobalGraph(properties={"node": {"weight": dfir.IntType()}, "edge": {}})
    nodes = g.add_graph_input("edge")
    src_dst_weight = nodes.map_(map_func=lambda edge: (edge.src.weight, edge.dst, edge))
    test = src_dst_weight.reduce_by(
        reduce_key=lambda data: data[1],
        reduce_transform=lambda data: data,
        reduce_method=lambda data1, data2: (data1[0] + data2[0], data1[1], data1[2]),
    )
    test2 = test.filter(filter_func=lambda data: data[0] > 10)
    # test2 = test.filter(filter_func=lambda data: 10 > data[0])
    # print(g)
    my_dfir = g.to_dfir()[0]
    # print(my_dfir)
    from graphyflow.visualize_ir import visualize_components

    dot = visualize_components(str(my_dfir))
    dot.render("component_graph", view=False, format="png")

    simulator = DfirSimulator(my_dfir, g)

    node_num = 20
    simulator.add_nodes(nodes=list(range(node_num)), props={i: {"weight": i} for i in range(node_num)})
    edges = []
    edges.extend([(i, i + 1) for i in range(node_num - 1)])
    edges.extend([(i, i + 2) for i in range(node_num - 2)])
    simulator.add_edges(
        edges={i * 10: edges[i] for i in range(len(edges))},
        props={i * 10: {} for i in range(len(edges))},
    )

    print(f"Running simulation")
    results = simulator.run({})
    print(f"Simulation results: {results}")
