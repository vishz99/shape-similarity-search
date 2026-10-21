import cadquery as cq

# Build a simple shape
box = cq.Workplane("XY").box(10, 20, 30)

# Write it out as a real STEP file
box.val().exportStep("../data/test_box.step")
print("Exported STEP file.")

# Read it back in
imported = cq.importers.importStep("../data/test_box.step")
solid = imported.val()

print("Read STEP file back in.")
print("Number of faces:", len(solid.Faces()))