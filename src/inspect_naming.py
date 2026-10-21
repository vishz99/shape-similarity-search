"""
Selects a working subset of the Fusion 360 Gallery reconstruction dataset.

Each design in this dataset has two kinds of STEP files:
  - "3-token" files, e.g. 20203_7e31e92a_0000.step
    -> the FINAL, complete target design. This is what we want.
  - "4-token" files, e.g. 20203_7e31e92a_0000_0001.step
    -> an INTERMEDIATE build step (partway through construction).
       Not useful for similarity search, since comparing a half-built
       sketch to a finished part would give meaningless results.

We filter to 3-token files only, then randomly sample a manageable
subset (500) to work with, copying each part's STEP file and its
preview thumbnail (if available) into data/processed/.
"""

import os
import random
import shutil

data_dir = "../data/raw/fusion360gallery/reconstruction"
output_dir = "../data/processed/fusion360_subset"
os.makedirs(output_dir, exist_ok=True)

step_files = [f for f in os.listdir(data_dir) if f.endswith(".step")]

# Keep only 3-token files: these are the finished target designs
final_designs = [
    f for f in step_files
    if len(f.replace(".step", "").split("_")) == 3
]
print("Final target designs found:", len(final_designs))

# Fix the random seed so re-running this script gives the SAME 500 every
# time -- important while we're iterating, so results stay comparable.
random.seed(42)
sample = random.sample(final_designs, min(500, len(final_designs)))

for f in sample:
    shutil.copy(os.path.join(data_dir, f), os.path.join(output_dir, f))

    # Grab the matching preview thumbnail too, if one exists -- these are
    # free renders we'll reuse later in the Phase 5 demo UI, so no need
    # to generate our own preview images from scratch then.
    png_name = f.replace(".step", ".png")
    png_path = os.path.join(data_dir, png_name)
    if os.path.exists(png_path):
        shutil.copy(png_path, os.path.join(output_dir, png_name))

print("Copied", len(sample), "final designs (+ thumbnails where available) to", output_dir)