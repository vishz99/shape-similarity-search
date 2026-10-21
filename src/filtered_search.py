"""
Rank the full FAISS index by L2 distance, then optionally keep only
candidates whose sorted bounding-box dimensions are each within a
tolerance of the query's dimensions. Used for both existing-part queries
and new uploaded parts (the caller supplies the query vector and dims).
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from part_properties import load_properties_csv  # noqa: E402

EMBEDDINGS_PATH = PROJECT_ROOT / "outputs/embeddings.npy"
MANIFEST_PATH = PROJECT_ROOT / "outputs/embeddings_manifest.json"
INDEX_PATH = PROJECT_ROOT / "outputs/faiss_index.bin"
PROPERTIES_PATH = PROJECT_ROOT / "outputs/part_properties.csv"


def load_resources():
    index = faiss.read_index(str(INDEX_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text())
    properties = load_properties_csv(PROPERTIES_PATH)
    return index, manifest, properties


def dims_within_tolerance(query_dims, candidate_dims, tolerance):
    for q, c in zip(query_dims, candidate_dims):
        if q <= 0:
            if c > tolerance:
                return False
            continue
        if abs(c - q) / q > tolerance:
            return False
    return True


def filtered_search(
    query_vector,
    query_dims=None,
    use_size_filter=False,
    tolerance=0.2,
    top_k=5,
    exclude_stem=None,
    index=None,
    manifest=None,
    properties=None,
):
    """
    Rank the full index by L2 distance to query_vector, optionally keeping
    only candidates whose sorted bounding-box dims are each within
    tolerance (fraction, e.g. 0.2 = 20%) of query_dims.

    Returns a list of dicts: stem, distance, dim_1, dim_2, dim_3, volume,
    area, num_faces, ordered by increasing distance.
    """
    if index is None or manifest is None or properties is None:
        index, manifest, properties = load_resources()

    if use_size_filter and query_dims is None:
        raise ValueError("query_dims is required when use_size_filter is True")

    n = index.ntotal
    query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, n)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        stem = manifest[idx]
        if stem == exclude_stem:
            continue
        props = properties.get(stem)
        if props is None:
            continue
        candidate_dims = [props["dim_1"], props["dim_2"], props["dim_3"]]

        if use_size_filter and not dims_within_tolerance(query_dims, candidate_dims, tolerance):
            continue

        results.append({
            "stem": stem,
            "distance": float(dist),
            "dim_1": props["dim_1"],
            "dim_2": props["dim_2"],
            "dim_3": props["dim_3"],
            "volume": props["volume"],
            "area": props["area"],
            "num_faces": props["num_faces"],
        })

        if len(results) >= top_k:
            break

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", help="File stem of an existing part")
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_filter", action="store_true", help="Disable the size filter")
    args = parser.parse_args()

    index, manifest, properties = load_resources()
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    if args.stem not in manifest:
        raise ValueError(f"Stem not found in manifest: {args.stem}")
    query_idx = manifest.index(args.stem)
    query_vector = embeddings[query_idx]
    props = properties[args.stem]
    query_dims = [props["dim_1"], props["dim_2"], props["dim_3"]]

    results = filtered_search(
        query_vector,
        query_dims=query_dims,
        use_size_filter=not args.no_filter,
        tolerance=args.tolerance,
        top_k=args.top_k,
        exclude_stem=args.stem,
        index=index,
        manifest=manifest,
        properties=properties,
    )

    if not results:
        print("No matches within tolerance")
        return

    print(f"Top {len(results)} matches for {args.stem} (tolerance={args.tolerance:.0%}):")
    for r in results:
        print(
            f"  {r['stem']}  distance={r['distance']:.4f}  "
            f"dims=({r['dim_1']:.3f}, {r['dim_2']:.3f}, {r['dim_3']:.3f})"
        )


if __name__ == "__main__":
    main()
