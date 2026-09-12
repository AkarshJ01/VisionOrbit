import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import rasterio


INPUT_FILE = "all_validation_change_results.json"
OUTPUT_FILE = "filtered_regions_visual.png"

MERGE_DISTANCE = 20
MIN_PIXELS = 20
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

    mean_confidence = sum(
        r["mean_confidence"] * r["pixels"]
        for r in group
    ) / total_pixels

    max_confidence = max(
        r["max_confidence"] for r in group
    )

    return {
        "pixels": total_pixels,
        "area_m2": sum(r["area_m2"] for r in group),
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence
    }


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

with open(INPUT_FILE, "r") as f:
    data = json.load(f)


# --------------------------------------------------
# FIND FIRST VALIDATION PATCH WITH DETECTIONS
# --------------------------------------------------

selected_patch = None

for patch in data["patch_results"]:

    if patch["detected_regions"]:
        selected_patch = patch
        break


if selected_patch is None:
    raise RuntimeError("No detected regions found.")


aoi = selected_patch["aoi"]
patch_index = selected_patch["patch_index"]

print("=" * 60)
print("FILTERED REGION VISUALIZATION")
print("=" * 60)

print(f"AOI: {aoi}")
print(f"Patch: {patch_index}")


# --------------------------------------------------
# MERGE
# --------------------------------------------------

groups = merge_regions(
    selected_patch["detected_regions"]
)

print(
    f"Regions before merging: "
    f"{len(selected_patch['detected_regions'])}"
)

print(f"Regions after merging: {len(groups)}")


# --------------------------------------------------
# FILTER
# --------------------------------------------------

filtered_groups = []

for group in groups:

    result = summarize_group(group)

    if result["pixels"] < MIN_PIXELS:
        continue

    if result["mean_confidence"] < MIN_CONFIDENCE:
        continue

    filtered_groups.append(group)


print(f"Regions after filtering: {len(filtered_groups)}")


# --------------------------------------------------
# LOAD SATELLITE IMAGE
# --------------------------------------------------

image_path = f"data/{aoi}/2019_12.tif"

if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)


with rasterio.open(image_path) as src:

    image = src.read()

    rgb = np.transpose(
        image[:3],
        (1, 2, 0)
    ).astype(float)

    rgb_min = rgb.min()
    rgb_max = rgb.max()

    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)


# --------------------------------------------------
# DRAW
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(rgb)


for idx, group in enumerate(filtered_groups, start=1):

    x_min = min(
        r["bounding_box_full_image_pixels"]["x_min"]
        for r in group
    )

    y_min = min(
        r["bounding_box_full_image_pixels"]["y_min"]
        for r in group
    )

    x_max = max(
        r["bounding_box_full_image_pixels"]["x_max"]
        for r in group
    )

    y_max = max(
        r["bounding_box_full_image_pixels"]["y_max"]
        for r in group
    )

    total_pixels = sum(
        r["pixels"] for r in group
    )

    centroid_x = sum(
        r["centroid_full_image_pixels"]["x"] * r["pixels"]
        for r in group
    ) / total_pixels

    centroid_y = sum(
        r["centroid_full_image_pixels"]["y"] * r["pixels"]
        for r in group
    ) / total_pixels

    confidence = sum(
        r["mean_confidence"] * r["pixels"]
        for r in group
    ) / total_pixels

    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        linewidth=2
    )

    ax.add_patch(rect)

    ax.text(
        centroid_x,
        centroid_y,
        str(idx),
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="circle",
            facecolor="white",
            alpha=0.8
        )
    )


ax.set_title(
    f"Filtered Change Regions\n"
    f"{len(selected_patch['detected_regions'])} raw → "
    f"{len(groups)} merged → "
    f"{len(filtered_groups)} final"
)

ax.set_xlim(0, 256)
ax.set_ylim(256, 0)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print(f"Saved visualization to: {OUTPUT_FILE}")
print("=" * 60)