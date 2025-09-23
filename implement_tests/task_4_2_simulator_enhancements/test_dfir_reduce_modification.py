import pytest
import graphyflow.dataflow_ir as dfir
from graphyflow.dataflow_ir_datatype import IntType, ArrayType, FloatType


@pytest.fixture
def base_reduce_component():
    """Provides a standard ReduceComponent for testing."""
    return dfir.ReduceComponent(
        input_type=ArrayType(IntType()), accumulated_type=FloatType(), reduce_key_out_type=IntType()
    )


def test_reduce_component_initialization(base_reduce_component):
    """
    Tests that the _port_groups dictionary is correctly initialized.
    """
    rc = base_reduce_component
    assert len(rc._port_groups["global"]) == 2  # i_0, o_0
    assert len(rc._port_groups["key"]) == 2  # i_reduce_key_out, o_reduce_key_in
    assert len(rc._port_groups["transform"]) == 2  # i_reduce_transform_out, o_reduce_transform_in
    assert (
        len(rc._port_groups["unit"]) == 3
    )  # i_reduce_unit_end, o_reduce_unit_start_0, o_reduce_unit_start_1

    total_ports_in_groups = sum(len(ports) for ports in rc._port_groups.values())
    assert total_ports_in_groups == len(rc.ports)


def test_add_io_port_pair(base_reduce_component):
    """
    Tests the ability to dynamically add a new I/O port pair to the ReduceComponent.
    """
    rc = base_reduce_component
    initial_port_count = len(rc.ports)
    initial_in_port_count = len(rc.in_ports)
    initial_out_port_count = len(rc.out_ports)
    initial_key_port_count = len(rc._port_groups["key"])

    # Add a new port pair for a passthrough value to the 'key' subgraph
    p_in, p_out = rc._add_io_port_pair(group="key", name_base="passthrough_0", data_type=IntType())

    # Verify port names and types
    assert p_in.name == "i_key_passthrough_0"
    assert p_out.name == "o_key_passthrough_0"
    assert p_in.data_type == IntType()
    assert p_out.data_type == IntType()
    assert p_in.port_type == dfir.PortType.IN
    assert p_out.port_type == dfir.PortType.OUT

    # Verify that the component's internal lists are updated
    assert len(rc.ports) == initial_port_count + 2
    assert len(rc.in_ports) == initial_in_port_count + 1
    assert len(rc.out_ports) == initial_out_port_count + 1
    assert p_in in rc.in_ports
    assert p_out in rc.out_ports

    # Verify that the port group tracking is updated
    assert len(rc._port_groups["key"]) == initial_key_port_count + 2
    assert p_in in rc._port_groups["key"]
    assert p_out in rc._port_groups["key"]


def test_remove_port_by_name(base_reduce_component):
    """
    Tests the ability to dynamically remove a port from the ReduceComponent.
    """
    rc = base_reduce_component
    initial_port_count = len(rc.ports)
    initial_in_port_count = len(rc.in_ports)

    port_i0 = rc.get_port("i_0")
    assert port_i0 in rc._port_groups["global"]

    # Remove the 'i_0' port
    rc._remove_port_by_name("i_0")

    # Verify it was removed from all lists
    assert len(rc.ports) == initial_port_count - 1
    assert len(rc.in_ports) == initial_in_port_count - 1

    with pytest.raises(ValueError):
        rc.get_port("i_0")  # Should raise error as it's gone

    assert port_i0 not in rc._port_groups["global"]


def test_remove_connected_port_fails(base_reduce_component):
    """
    Ensures that removing a connected port raises an assertion error.
    """
    rc = base_reduce_component
    p1 = dfir.Port("o_temp", dfir.CopyComponent(IntType()))
    p2 = rc.get_port("i_0")
    p1.connect(p2)

    with pytest.raises(AssertionError, match="Cannot remove port 'i_0' because it is connected."):
        rc._remove_port_by_name("i_0")
