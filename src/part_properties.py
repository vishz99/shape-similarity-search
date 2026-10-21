"""
Compute geometric properties for each part with occwl: sorted bounding
box dimensions, volume, surface area, and face count. Writes
outputs/part_properties.csv.
"""
import csv
from pathlib import Path

from occwl.compound import Compound

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_DIR = PROJECT_ROOT / "data/processed/fusion360_subset"
OUT_CSV = PROJECT_ROOT / "outputs/part_properties.csv"

FIELDNAMES = ["stem", "dim_1", "dim_2", "dim_3", "volume", "area", "num_faces"]


def load_solid(step_path):
    """Load the first solid from a STEP file as an occwl Solid."""
    compound = Compound.load_from_step(str(step_path))
    solids = list(compound.solids())
    if not solids:
        raise ValueError(f"No solids found in {step_path}")
    return solids[0]


def get_part_properties(step_path):
    """
    Compute geometric properties for a single STEP file.

    Returns a dict with sorted bounding-box dimensions (dim_1 <= dim_2 <=
    dim_3), volume, surface area, and face count.
    """
    solid = load_solid(step_path)
    box = solid.exact_box()
    dims = sorted([box.x_length(), box.y_length(), box.z_length()])
    return {
        "dim_1": dims[0],
        "dim_2": dims[1],
        "dim_3": dims[2],
        "volume": solid.volume(),
        "area": solid.area(),
        "num_faces": solid.num_faces(),
    }


def load_properties_csv(csv_path=OUT_CSV):
    """Load part_properties.csv into a dict keyed by file stem."""
    properties = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            properties[row["stem"]] = {
                "dim_1": float(row["dim_1"]),
                "dim_2": float(row["dim_2"]),
                "dim_3": float(row["dim_3"]),
                "volume": float(row["volume"]),
                "area": float(row["area"]),
                "num_faces": int(row["num_faces"]),
            }
    return properties


def main():
    step_files = sorted(STEP_DIR.glob("*.step"))
    print(f"Found {len(step_files)} STEP files")

    rows = []
    failed = []
    for step_file in step_files:
        try:
            props = get_part_properties(step_file)
        except Exception as e:
            print(f"WARNING: Skipping {step_file.name}: {e}")
            failed.append(step_file.stem)
            continue
        row = {"stem": step_file.stem}
        row.update(props)
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    if failed:
        print(f"Failed: {len(failed)} -> {failed}")


if __name__ == "__main__":
    main()
