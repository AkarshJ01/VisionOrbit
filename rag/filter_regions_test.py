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

# Same merging rule we just tested
MERGE_DISTANCE = 20

# Test these minimum region sizes
MIN_PIXELS = 20

# Test this minimum confidence
MIN_CONFIDENCE = 0.60


def bbox_distance(a, b):
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


def merge_regions(regions):

    groups = [[r] for r in regions]

    changed = True

    while changed:

        changed = False
        new_groups = []
        used = set()

        for i in range(len(groups)):

            if i in used:
                continue

            current = groups[i]
            used.add(i)

            merged = True

            while merged:

                merged = False

                for j in range(len(groups)):

                    if j in used:
                        continue

                    candidate = groups[j]

                    should_merge = False

                    for r1 in current:

                        box_a = r1["bounding_box_full_image_pixels"]

                        for r2 in candidate:

                            box_b = r2["bounding_box_full_image_pixels"]

                            if bbox_distance(box_a, box_b) <= MERGE_DISTANCE:
                                should_merge = True
                                break

                        if should_merge:
                            break

                    if should_merge:

                        current.extend(candidate)
                        used.add(j)
                        merged = True
                        changed = True

            new_groups.append(current)

        groups = new_groups

    return groups


def summarize_group(group):

    total_pixels = sum(r["pixels"] for r in group)

    total_area = sum(r["area_m2"] for r in group)

    mean_confidence = sum(
        r["mean_confidence"] * r["pixels"]
        for r in group
    ) / total_pixels

    max_confidence = max(
        r["max_confidence"]
        for r in group
    )

    return {
        "pixels": total_pixels,
        "area_m2": total_area,
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence
    }


# --------------------------------------------------
# LOAD
# --------------------------------------------------

with open(INPUT_FILE, "r") as f:
    data = json.load(f)


raw_regions = 0
merged_regions = 0
filtered_regions = 0

small_removed = 0
low_confidence_removed = 0


print("=" * 60)
print("REGION FILTERING TEST")
print("=" * 60)

print(f"Merge distance: {MERGE_DISTANCE} pixels")
print(f"Minimum region size: {MIN_PIXELS} pixels")
print(f"Minimum confidence: {MIN_CONFIDENCE}")
print("=" * 60)


# --------------------------------------------------
# PROCESS
# --------------------------------------------------

for patch in data["patch_results"]:

    regions = patch["detected_regions"]

    raw_regions += len(regions)

    if not regions:
        continue

    groups = merge_regions(regions)

    merged_regions += len(groups)

    for group in groups:

        result = summarize_group(group)

        # Size filter
        if result["pixels"] < MIN_PIXELS:

            small_removed += 1
            continue

        # Confidence filter
        if result["mean_confidence"] < MIN_CONFIDENCE:

            low_confidence_removed += 1
            continue

        filtered_regions += 1


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("FILTERING RESULT")
print("=" * 60)

print(f"Raw regions:              {raw_regions}")
print(f"After merging:            {merged_regions}")
print(f"After filtering:          {filtered_regions}")
print()

print(f"Removed by size:          {small_removed}")
print(f"Removed by confidence:    {low_confidence_removed}")

if merged_regions > 0:

    reduction = (
        1 - filtered_regions / merged_regions
    ) * 100

    print(
        f"\nFiltering reduction:      {reduction:.1f}%"
    )

print("=" * 60)
