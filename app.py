"""
Streamlit app for CAD design similarity search.

Pick an existing part from the 500-part subset, or upload a new STEP
file, and see the top 5 most similar parts by BRepNet embedding
distance, with an optional bounding-box size filter.

Run with the brepnet conda environment:
    conda activate brepnet
    streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from filtered_search import filtered_search, load_resources  # noqa: E402
from part_properties import get_part_properties, load_solid  # noqa: E402
from search_new_part import search_new_part  # noqa: E402

STEP_DIR = PROJECT_ROOT / "data/processed/fusion360_subset"
EMBEDDINGS_PATH = PROJECT_ROOT / "outputs/embeddings.npy"


@st.cache_resource
def get_resources():
    index, manifest, properties = load_resources()
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    return index, manifest, properties, embeddings


@st.cache_data
def compute_upload_embedding(file_bytes, filename, use_size_filter, tolerance):
    with tempfile.TemporaryDirectory() as tmp_dir:
        step_path = Path(tmp_dir) / filename
        step_path.write_bytes(file_bytes)

        # Validate the file loads with occwl before running BRepNet on it.
        load_solid(step_path)

        query_properties, results = search_new_part(
            step_path,
            use_size_filter=use_size_filter,
            tolerance=tolerance,
            top_k=5,
        )
    return query_properties, results


def thumbnail_path(stem):
    path = STEP_DIR / f"{stem}.png"
    return path if path.exists() else None


def render_properties(props):
    st.write(f"Size: {props['dim_1']:.2f} x {props['dim_2']:.2f} x {props['dim_3']:.2f}")
    st.write(f"Volume: {props['volume']:.2f}")
    st.write(f"Area: {props['area']:.2f}")
    st.write(f"Faces: {props['num_faces']}")


def render_match_card(result):
    with st.container(border=True):
        thumb = thumbnail_path(result["stem"])
        if thumb is not None:
            st.image(str(thumb), width="stretch")
        else:
            st.caption("No preview available")
        st.markdown(f"**{result['stem']}**")
        st.write(f"Distance: {result['distance']:.4f}")
        st.write(f"Size: {result['dim_1']:.2f} x {result['dim_2']:.2f} x {result['dim_3']:.2f}")
        st.write(f"Volume: {result['volume']:.2f}")
        st.write(f"Area: {result['area']:.2f}")
        st.write(f"Faces: {result['num_faces']}")


def main():
    st.set_page_config(page_title="CAD Design Similarity Search", layout="wide")
    st.title("CAD Design Similarity Search")

    index, manifest, properties, embeddings = get_resources()

    st.sidebar.header("Query")
    mode = st.sidebar.radio("Source", ["Existing part", "Upload STEP file"])

    use_size_filter = st.sidebar.checkbox("Filter by size", value=False)
    tolerance_pct = st.sidebar.slider(
        "Tolerance (%)", min_value=5, max_value=50, value=20, disabled=not use_size_filter
    )
    tolerance = tolerance_pct / 100.0

    query_properties = None
    query_thumb = None
    query_label = None
    results = []
    error = None

    if mode == "Existing part":
        stem = st.sidebar.selectbox("Part", sorted(manifest))
        query_label = stem
        query_thumb = thumbnail_path(stem)
        query_properties = properties.get(stem)

        query_idx = manifest.index(stem)
        query_vector = embeddings[query_idx]
        query_dims = None
        if query_properties is not None:
            query_dims = [query_properties["dim_1"], query_properties["dim_2"], query_properties["dim_3"]]

        results = filtered_search(
            query_vector,
            query_dims=query_dims,
            use_size_filter=use_size_filter,
            tolerance=tolerance,
            top_k=5,
            exclude_stem=stem,
            index=index,
            manifest=manifest,
            properties=properties,
        )
    else:
        uploaded = st.sidebar.file_uploader("STEP file", type=["step", "stp"])
        if uploaded is not None:
            query_label = uploaded.name
            try:
                query_properties, results = compute_upload_embedding(
                    uploaded.getvalue(), uploaded.name, use_size_filter, tolerance
                )
            except Exception as e:
                error = str(e)

    st.header("Query")
    if error is not None:
        st.error(f"Could not process the uploaded file: {error}")
    elif query_label is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            if query_thumb is not None:
                st.image(str(query_thumb), width="stretch")
            else:
                st.caption("No preview available")
        with col2:
            st.markdown(f"**{query_label}**")
            if query_properties is not None:
                render_properties(query_properties)
    else:
        st.info("Choose an existing part or upload a STEP file to begin.")

    if query_label is not None and error is None:
        st.header("Top 5 matches")
        if not results:
            st.warning("No matches within tolerance")
        else:
            cols = st.columns(5)
            for col, result in zip(cols, results):
                with col:
                    render_match_card(result)


if __name__ == "__main__":
    main()
