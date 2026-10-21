"""
Embed a new STEP file (not in the existing index) with the same BRepNet
pipeline used to build the index, then search for similar parts. Uses a
disposable temp folder; the index and outputs/ are never modified.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_ROOT = PROJECT_ROOT / "external/BRepNet"
SRC_DIR = Path(__file__).resolve().parent

for path in (BREPNET_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embed_step_file import embed_step_file  # noqa: E402
from part_properties import get_part_properties  # noqa: E402
from filtered_search import filtered_search, load_resources  # noqa: E402


def search_new_part(step_path, use_size_filter=False, tolerance=0.2, top_k=5):
    """
    Embed a new STEP file and search the existing index for similar
    parts.

    Returns (query_properties, results).
    """
    step_path = Path(step_path)

    query_vector = embed_step_file(step_path)
    query_properties = get_part_properties(step_path)
    query_dims = [query_properties["dim_1"], query_properties["dim_2"], query_properties["dim_3"]]

    index, manifest, properties = load_resources()
    results = filtered_search(
        query_vector,
        query_dims=query_dims,
        use_size_filter=use_size_filter,
        tolerance=tolerance,
        top_k=top_k,
        index=index,
        manifest=manifest,
        properties=properties,
    )
    return query_properties, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step_file", help="Path to a new STEP file")
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--filter", action="store_true", help="Apply the size filter")
    args = parser.parse_args()

    query_properties, results = search_new_part(
        args.step_file,
        use_size_filter=args.filter,
        tolerance=args.tolerance,
        top_k=args.top_k,
    )

    print(f"Query properties: {query_properties}")
    if not results:
        print("No matches within tolerance")
        return

    print(f"Top {len(results)} matches:")
    for r in results:
        print(f"  {r['stem']}  distance={r['distance']:.4f}")


if __name__ == "__main__":
    main()
