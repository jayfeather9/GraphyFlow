import os
import sys
from pathlib import Path
import re  # Import re at the top level

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.backend_manager import BackendManager
from graphyflow.lambda_func import lambda_min


def test_common_header_generation():
    print("--- Testing Task 1.3: common.h Generation ---")

    # 1. Define a graph to force creation of various types
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
    min_dist = pdu.reduce_by(
        reduce_key=lambda src_dist, dst, edge_w: dst.id,
        reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
        reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
    )

    # 2. Run frontend and passes
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])

    # 3. Instantiate BackendManager and run type analysis
    bkd_mng = BackendManager()
    bkd_mng.comp_col_store = comp_col
    bkd_mng.global_graph_store = g
    bkd_mng._analyze_and_map_types(comp_col)

    # 4. Call the common.h generator
    header_content = bkd_mng.generate_common_header("graphyflow_test")

    # 5. Perform checks on the generated header content
    errors = []

    # --- *** 关键修正：使用更健壮的测试函数 *** ---
    def check_struct(struct_name, members):
        """A more robust function to check for struct definitions."""
        # 1. Find the struct block using a flexible regex for whitespace
        struct_pattern = re.compile(
            f"struct\\s+__attribute__\\s*\\(\\s*\\(packed\\)\\s*\\)\\s+{struct_name}\\s*{{(.*?)}};", re.DOTALL
        )
        match = struct_pattern.search(header_content)
        if not match:
            errors.append(f"FAIL: Struct '{struct_name}' definition not found or is malformed.")
            return

        struct_body = match.group(1)

        # 2. Check for the presence of each member within the body individually
        for member in members:
            # Prepare member string for regex, allowing flexible whitespace
            # e.g., "bool end_flag" becomes "bool\s+end_flag"
            member_pattern_str = r"\s+".join(re.escape(part) for part in member.split())
            # Allow for array brackets with flexible spacing, e.g., data [ PE_NUM ]
            member_pattern_str = member_pattern_str.replace(r"\[", r"\s*\[\s*").replace(r"\]", r"\s*\]\s*")

            member_pattern = re.compile(member_pattern_str + r"\s*;")

            if not member_pattern.search(struct_body):
                errors.append(
                    f"FAIL: Struct '{struct_name}' is missing or has an incorrect member: '{member}'."
                )

    check_struct("node_t", ["ap_fixed<32, 16> distance", "int32_t id"])
    check_struct("edge_t", ["ap_fixed<32, 16> weight", "node_t src", "node_t dst"])
    check_struct("KernelOutputData", ["float distance", "int32_t id"])
    check_struct("KernelOutputBatch", ["KernelOutputData data[PE_NUM]", "bool end_flag", "uint8_t end_pos"])

    # 6. Report results
    if not errors:
        print("common.h Generation Test: SUCCESS")
        return True
    else:
        print("common.h Generation Test: FAILED")
        for error in errors:
            print(f"  - {error}")
        print("\n--- Generated Header for Debugging ---")
        print(header_content)
        return False


if __name__ == "__main__":
    test_dir = project_root / "tests" / "task_1_3"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.3. Test script location: {test_dir}")

    success = test_common_header_generation()

    if success:
        print("\nAll tests for Task 1.3 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.3 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.3 FAILED.")
        sys.exit(1)
