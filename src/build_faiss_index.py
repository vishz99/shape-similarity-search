"""
Build a FAISS IndexFlatL2 over the BRepNet embeddings and save it to
outputs/faiss_index.bin. Row order matches outputs/embeddings_manifest.json.
"""
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "outputs/embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "outputs/faiss_index.bin"


def build_index(embeddings_path=EMBEDDINGS_PATH):
    embeddings = np.load(embeddings_path).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    index = build_index()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Indexed {index.ntotal} vectors of dimension {index.d}")
    print(f"Saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
