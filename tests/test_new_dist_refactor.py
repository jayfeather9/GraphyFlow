from pathlib import Path
from graphyflow.global_graph import GlobalGraph
import graphyflow.dataflow_ir as dfir
from graphyflow.lambda_func import lambda_min
from graphyflow.passes import delete_placeholder_components_pass, refactor_other_comps
from graphyflow.visualize_ir import visualize_components

# ==================== Config =======================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"


def build_graph() -> GlobalGraph:
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
    pdu = pdu.filter(filter_func=lambda x, y, z: z >= 0.0)
    min_dist = pdu.reduce_by(
        reduce_key=lambda src_dist, dst, edge_w: dst.id,
        reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
        reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
    )
    _updated_nodes = min_dist.map_(map_func=lambda dist, node: (lambda_min(dist, node.distance), node))
    return g


def save_graph_png(comp_col: dfir.ComponentCollection, stem: str) -> None:
    dot = visualize_components(str(comp_col))
    OUT = OUTPUT_DIR / stem
    dot.render(str(OUT), view=False, format="png")


if __name__ == "__main__":
    print("--- Building Graph ---")
    g = build_graph()

    print("\n--- Generating DFIR ---")
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])
    save_graph_png(comp_col, "new_dist_ori")
    print("Saved original DFIR graph to output/new_dist_ori.png")

    print("\n--- Refactoring DFIR (partition + memread/fused islands) ---")
    refactored = refactor_other_comps(comp_col, g)
    save_graph_png(refactored, "new_dist_refactored")
    print("Saved refactored DFIR graph to output/new_dist_refactored.png")

    print("\nDone.")
