import pytest
import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from graphyflow.simulate import DfirSimulator, UncertainArray


@pytest.fixture(scope="module")
def fused_op_test_setup():
    """
    Provides a setup for testing FusedOpComponent simulation.
    The internal subgraph uses only allowed component types to calculate:
    (Input_A * Internal_Constant) + (Input_A + Input_B)
    """
    # --- 1. Define data types ---
    int_array = dfir.ArrayType(dfir.IntType())

    # --- 2. Build the internal subgraph using only allowed components ---

    # Input A is used twice, so we need a CopyComponent
    copy_a = dfir.CopyComponent(int_array)

    # Internal constant for multiplication
    const_c = dfir.ConstantComponent(
        dfir.ArrayType(dfir.IntType()), 3
    )  # Note: scalar IntType for broadcasting

    # Operation: A * C
    mul_op = dfir.BinOpComponent(dfir.BinOp.MUL, int_array)

    # Operation: A + B
    add_op_1 = dfir.BinOpComponent(dfir.BinOp.ADD, int_array)

    # Final Operation: (A*C) + (A+B)
    add_op_2 = dfir.BinOpComponent(dfir.BinOp.ADD, int_array)

    # --- 3. Connect the subgraph logic ---
    # Connect inputs of the subgraph to the operators
    # copy_a receives Input A and splits it
    copy_a.get_port("o_0").connect(mul_op.get_port("i_0"))
    copy_a.get_port("o_1").connect(add_op_1.get_port("i_0"))
    # add_op_1 also needs Input B, which will be the second input of the subgraph

    # Connect the internal constant
    const_c.get_port("o_0").connect(mul_op.get_port("i_1"))

    # Connect the outputs of the first-level operations to the final adder
    mul_op.get_port("o_0").connect(add_op_2.get_port("i_0"))
    add_op_1.get_port("o_0").connect(add_op_2.get_port("i_1"))

    subgraph_comps = [copy_a, const_c, mul_op, add_op_1, add_op_2]

    # Define the subgraph collection. It has 2 inputs and 1 output.
    subgraph = dfir.ComponentCollection(
        components=subgraph_comps,
        inputs=[copy_a.get_port("i_0"), add_op_1.get_port("i_1")],  # Input A, Input B
        outputs=[add_op_2.get_port("o_0")],
    )

    # --- 4. Wrap subgraph in a FusedOpComponent ---
    fused_op = dfir.FusedOpComponent(name="my_fused_op", sub_graph=subgraph)

    # --- 5. Build the main graph to drive the test ---
    # Provide constant arrays as inputs to the FusedOp
    # const_in_a = dfir.ConstantComponent(int_array, [1, 6, 3, 8])
    # const_in_b = dfir.ConstantComponent(int_array, [2, 2, 8, 1])

    # const_in_a.get_port("o_0").connect(fused_op.get_port("i_0"))
    # const_in_b.get_port("o_0").connect(fused_op.get_port("i_1"))

    main_cc = dfir.ComponentCollection(
        components=[fused_op],
        inputs=[fused_op.get_port("i_0"), fused_op.get_port("i_1")],
        outputs=[fused_op.get_port("o_0")],
    )

    return {"collection": main_cc, "graph": GlobalGraph()}


def test_simulate_fused_op_parallel_valid_components(fused_op_test_setup):
    """
    Tests the FusedOpComponent simulation using a subgraph that contains
    only valid component types as per the FusedOpComponent's validation rules.
    """
    # --- 1. Setup simulator ---
    g = fused_op_test_setup["graph"]
    collection = fused_op_test_setup["collection"]

    simulator = DfirSimulator(collection, g)

    # --- 2. Run simulation (no inputs needed as they are constants) ---
    results = simulator.run(
        {
            "i_0": [1, 6, 3, 8],  # Input A
            "i_1": [2, 2, 8, 1],  # Input B
        }
    )

    # --- 3. Assert correctness of the final result ---
    # Manual calculation of the expected result:
    # Input A = [1, 6, 3, 8]
    # Input B = [2, 2, 8, 1]
    # Internal Constant C = 3
    #
    # A * C   = [1*3, 6*3, 3*3, 8*3]      = [3, 18, 9, 24]
    # A + B   = [1+2, 6+2, 3+8, 8+1]      = [3, 8, 11, 9]
    # Result  = [3+3, 18+8, 9+11, 24+9]   = [6, 26, 20, 33]
    expected_output = [6, 26, 20, 33]

    # The output port of the FusedOp is 'o_0' by default
    assert "o_0" in results
    assert results["o_0"] == expected_output
