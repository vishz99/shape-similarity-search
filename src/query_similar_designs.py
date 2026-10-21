"""
Query the FAISS index by an existing part's file stem. Prints the top 5
most similar other parts with their L2 distances.
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "outputs/embeddings.npy"
MANIFEST_PATH = PROJECT_ROOT / "outputs/embeddings_manifest.json"
INDEX_PATH = PROJECT_ROOT / "outputs/faiss_index.bin"


def load_index_and_manifest():
    index = faiss.read_index(str(INDEX_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text())
    return index, manifest


def query_by_stem(stem, top_k=5, index=None, manifest=None, embeddings=None):
    """
    Find the top_k most similar parts to an existing part, by file stem.

    Returns a list of (stem, distance) tuples, excluding the query part
    itself, ordered by increasing distance.
    """
    if index is None or manifest is None:
        index, manifest = load_index_and_manifest()
    if embeddings is None:
        embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    if stem not in manifest:
        raise ValueError(f"Stem not found in manifest: {stem}")

    query_idx = manifest.index(stem)
    query_vector = embeddings[query_idx:query_idx + 1]

    distances, indices = index.search(query_vector, top_k + 1)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or manifest[idx] == stem:
            continue
        results.append((manifest[idx], float(dist)))

    return results[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", help="File stem of an existing part, e.g. 100243_9fb796fe_0006")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = query_by_stem(args.stem, top_k=args.top_k)
    print(f"Top {len(results)} matches for {args.stem}:")
    for stem, dist in results:
        print(f"  {stem}  distance={dist:.4f}")


if __name__ == "__main__":
    main()
