# Quick sanity check: confirm our copied subset files are valid STEP files
# that cadquery/OCP can actually read, before building the embedding
# pipeline on top of them in Phase 3.

import os
import cadquery as cq

subset_dir = "../data/processed/fusion360_subset"
step_files = [f for f in os.listdir(subset_dir) if f.endswith(".step")]

# Just check the first file loads and has real geometry
sample_file = os.path.join(subset_dir, step_files[0])
imported = cq.importers.importStep(sample_file)
solid = imported.val()

print("File:", step_files[0])
print("Faces:", len(solid.Faces()))