import os
import sys
from pathlib import Path
import json
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = os.path.join(str(PROJECT_ROOT), "outputs/results/all_validation_change_results.json")
INPUT_FILE = DEFAULT_INPUT if os.path.exists(DEFAULT_INPUT) else "all_validation_change_results.json"

# Distance in pixels within which two regions may be merged
MERGE_DISTANCE = 20


def bbox_distance(a, b):
    """
    Minimum distance between two bounding boxes.
    Returns 0 if they overlap/touch.
    """

    dx = max(
        a["x_min"] - b["x_max"],
        b["x_min"] - a["x_max"],
        0
    )

    dy = max(
        a["y_min"] - b["y_max"],
        b["y_min"] - a["y_max"],
        0
    )

    return math.sqrt(dx * dx + dy * dy)


def merge_group(regions):
    """
    Merge a group of connected/nearby regions.
    """

    total_pixels = sum(r["pixels"] for r in regions)
    total_area = sum(r["area_m2"] for r in regions)

    # Weighted centroid
    if total_pixels > 0:
        centroid_x = sum(
            r["centroid_full_image_pixels"]["x"] * r["pixels"]
            for r in regions
        ) / total_pixels

        centroid_y = sum(
            r["centroid_full_image_pixels"]["y"] * r["pixels"]
            for r in regions
        ) / total_pixels
    else:
        centroid_x = 0
        centroid_y = 0

    # Bounding box covering all regions
    bbox = {
        "x_min": min(r["bounding_box_full_image_pixels"]["x_min"] for r in regions),
        "y_min": min(r["bounding_box_full_image_pixels"]["y_min"] for r in regions),
        "x_max": max(r["bounding_box_full_image_pixels"]["x_max"] for r in regions),
        "y_max": max(r["bounding_box_full_image_pixels"]["y_max"] for r in regions)
    }

    # Confidence weighted by region size
    mean_confidence = sum(
        r["mean_confidence"] * r["pixels"]
        for r in regions
    ) / total_pixels

    max_confidence = max(
        r["max_confidence"] for r in regions
    )

    return {
        "regions_merged": len(regions),
        "pixels": total_pixels,
        "area_m2": round(total_area, 2),
        "centroid": {
            "x": round(centroid_x, 2),
            "y": round(centroid_y, 2)
        },
        "bounding_box": bbox,
        "mean_confidence": round(mean_confidence, 4),
        "max_confidence": round(max_confidence, 4)
    }


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

with open(INPUT_FILE, "r") as f:
    data = json.load(f)


total_before = 0
total_after = 0


print("=" * 60)
print("REGION MERGING TEST")
print("=" * 60)
print(f"Merge distance: {MERGE_DISTANCE} pixels")
print("=" * 60)


# --------------------------------------------------
# PROCESS EACH PATCH
# --------------------------------------------------

for patch in data["patch_results"]:

    regions = patch["detected_regions"]

    if not regions:
        continue

    total_before += len(regions)

    # Each region starts in its own group
    groups = [[r] for r in regions]

    changed = True

    # Keep merging until no groups can be merged
    while changed:

        changed = False
        new_groups = []
        used = set()

        for i in range(len(groups)):

            if i in used:
                continue

            current = groups[i]
            used.add(i)

            merged_something = True

            while merged_something:

                merged_something = False

                current_boxes = [
                    r["bounding_box_full_image_pixels"]
                    for r in current
                ]

                for j in range(len(groups)):

                    if j in used:
                        continue

                    candidate = groups[j]

                    candidate_boxes = [
                        r["bounding_box_full_image_pixels"]
                        for r in candidate
                    ]

                    should_merge = False

                    for box_a in current_boxes:

                        for box_b in candidate_boxes:

                            if bbox_distance(box_a, box_b) <= MERGE_DISTANCE:
                                should_merge = True
                                break

                        if should_merge:
                            break

                    if should_merge:

                        current.extend(candidate)
                        used.add(j)
                        merged_something = True
                        changed = True

            new_groups.append(current)

        groups = new_groups

    total_after += len(groups)

    # Print only patches where merging actually happened
    if len(groups) < len(regions):

        print(
            f"\nAOI: {patch['aoi']}"
            f"\nPatch: {patch['patch_index']}"
            f"\nBefore: {len(regions)} regions"
            f"\nAfter:  {len(groups)} regions"
        )

        for idx, group in enumerate(groups, start=1):

            if len(group) > 1:

                merged = merge_group(group)

                print(
                    f"  Merged group {idx}: "
                    f"{len(group)} regions → "
                    f"{merged['pixels']} pixels, "
                    f"{merged['area_m2']} m²"
                )


print("\n" + "=" * 60)
print("FINAL TEST RESULT")
print("=" * 60)

print(f"Regions before merging: {total_before}")
print(f"Regions after merging:  {total_after}")

if total_before > 0:
    reduction = (1 - total_after / total_before) * 100
    print(f"Reduction: {reduction:.1f}%")

print("=" * 60)