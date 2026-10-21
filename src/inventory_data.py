# Selects a working subset of final-design STEP files (3-token filenames)
# from the Fusion 360 Gallery reconstruction dataset. Clears the output
# folder first, so reruns never mix with files from earlier attempts.

import os
import random
import shutil

data_dir = "../data/raw/fusion360gallery/reconstruction"
output_dir = "../data/processed/fusion360_subset"

# Wipe and recreate the output folder so this script is safe to rerun
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

step_files = [f for f in os.listdir(data_dir) if f.endswith(".step")]

# 3-token filenames = finished target designs, 4-token = intermediate build steps
final_designs = [
    f for f in step_files
    if len(f.replace(".step", "").split("_")) == 3
]
print("Final target designs found:", len(final_designs))

random.seed(42)
sample = random.sample(final_designs, min(500, len(final_designs)))

for f in sample:
    shutil.copy(os.path.join(data_dir, f), os.path.join(output_dir, f))
    png_name = f.replace(".step", ".png")
    png_path = os.path.join(data_dir, png_name)
    if os.path.exists(png_path):
        shutil.copy(png_path, os.path.join(output_dir, png_name))

print("Copied", len(sample), "final designs (+ thumbnails where available) to", output_dir)