# This dataset's STEP filenames follow two patterns:
#   - 3-token, e.g. 100243_9fb796fe_0006.step       -> assumed: finished design
#   - 4-token, e.g. 100243_9fb796fe_0006_0021.step  -> assumed: an intermediate
#     build step captured partway through construction
#
# This script tests that assumption directly: for one design with several
# saved stages, we check whether face count increases as the stage number
# increases, with the 3-token file being the most complete. If true, this
# confirms we should keep only 3-token files for the similarity-search subset.

import os
import cadquery as cq
from collections import defaultdict

data_dir = "../data/raw/fusion360gallery/reconstruction"
step_files = os.listdir(data_dir)

# Group files by base design ID
groups = defaultdict(list)
for f in step_files:
    if not f.endswith(".step"):
        continue
    parts = f.replace(".step", "").split("_")
    base_id = "_".join(parts[:2])
    groups[base_id].append(f)

# Pick the first design with 4+ saved stages, for a clear sequence to inspect
target = next(base for base, files in groups.items() if len(files) >= 4)
sequence = sorted(groups[target])

print(f"Inspecting design: {target}\n")
for f in sequence:
    solid = cq.importers.importStep(os.path.join(data_dir, f)).val()
    print(f, "-> faces:", len(solid.Faces()))