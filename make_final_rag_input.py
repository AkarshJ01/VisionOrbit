import json
import math

INPUT_FILE = "all_validation_change_results.json"
OUTPUT_FILE = "rag_input_v2.json"

# -----------------------------
# FINAL POST-PROCESSING RULES
# -----------------------------

MERGE_DISTANCE = 20
MIN_PIXELS = 20
MIN_CONFIDENCE = 0.60


def bbox_distance(a, b):
    """
    Minimum distance between two bounding boxes.
    Returns 0 if they overlap or touch.
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


def merge_groups(groups):
    """
    Merge groups whose regions are within MERGE_DISTANCE pixels.
    """

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

                    for region_a in current:

                        box_a = region_a[
                            "bounding_box_full_image_pixels"
                        ]

                        for region_b in candidate:

                            box_b = region_b[
                                "bounding_box_full_image_pixels"
                            ]

                            if bbox_distance(
                                box_a,
                                box_b
                            ) <= MERGE_DISTANCE:

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


def create_event(group, patch_info):

    total_pixels = sum(
        r["pixels"]
        for r in group
    )

    total_area = sum(
        r["area_m2"]
        for r in group
    )

    # Weighted centroid
    centroid_x = sum(
        r["centroid_full_image_pixels"]["x"]
        * r["pixels"]
        for r in group
    ) / total_pixels

    centroid_y = sum(
        r["centroid_full_image_pixels"]["y"]
        * r["pixels"]
        for r in group
    ) / total_pixels

    # Bounding box covering entire event
    bbox = {
        "x_min": min(
            r["bounding_box_full_image_pixels"]["x_min"]
            for r in group
        ),

        "y_min": min(
            r["bounding_box_full_image_pixels"]["y_min"]
            for r in group
        ),

        "x_max": max(
            r["bounding_box_full_image_pixels"]["x_max"]
            for r in group
        ),

        "y_max": max(
            r["bounding_box_full_image_pixels"]["y_max"]
            for r in group
        )
    }

    # Weighted confidence
    mean_confidence = sum(
        r["mean_confidence"] * r["pixels"]
        for r in group
    ) / total_pixels

    max_confidence = max(
        r["max_confidence"]
        for r in group
    )

    # Weighted geographic centroid
    latitude = sum(
        r["centroid_latitude"] * r["pixels"]
        for r in group
    ) / total_pixels

    longitude = sum(
        r["centroid_longitude"] * r["pixels"]
        for r in group
    ) / total_pixels

    # Which patches contributed to this event?
    patch_indices = sorted(
        set(
            r["_patch_index"]
            for r in group
        )
    )

    return {
        "aoi": patch_info["aoi"],

        "time_period": patch_info["time_period"],

        "change_type": patch_info["change_type"],

        "location": {
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6)
        },

        "area_m2": round(total_area, 2),

        "confidence": {
            "mean": round(mean_confidence, 4),
            "maximum": round(max_confidence, 4)
        },

        "bounding_box_full_image_pixels": bbox,

        "source_image_size_pixels":
            patch_info["source_image_size_pixels"],

        "contributing_patches": patch_indices,

        "detected_pixels": total_pixels
    }


# ============================================================
# LOAD
# ============================================================

with open(INPUT_FILE, "r") as f:
    data = json.load(f)


print("=" * 60)
print("CREATING FINAL RAG INPUT")
print("=" * 60)


# ============================================================
# GROUP ALL RAW REGIONS BY AOI
# ============================================================

aoi_data = {}

raw_region_count = 0

for patch in data["patch_results"]:

    aoi = patch["aoi"]

    if aoi not in aoi_data:

        aoi_data[aoi] = {
            "patch_info": patch,
            "regions": []
        }

    for region in patch["detected_regions"]:

        # Copy region so we don't modify original JSON
        region_copy = dict(region)

        # Remember which patch produced this region
        region_copy["_patch_index"] = patch["patch_index"]

        aoi_data[aoi]["regions"].append(
            region_copy
        )

        raw_region_count += 1


print(f"Raw detected regions: {raw_region_count}")
print(f"Validation AOIs: {len(aoi_data)}")


# ============================================================
# PROCESS EACH AOI
# ============================================================

final_events = []

total_merged = 0
total_filtered = 0

for aoi, info in aoi_data.items():

    regions = info["regions"]

    if not regions:
        continue

    # ------------------------------------------
    # CREATE INITIAL GROUPS
    # ------------------------------------------

    groups = [
        [region]
        for region in regions
    ]

    # ------------------------------------------
    # MERGE NEARBY REGIONS
    # ------------------------------------------

    groups = merge_groups(groups)

    total_merged += len(groups)

    # ------------------------------------------
    # FILTER
    # ------------------------------------------

    for group in groups:

        total_pixels = sum(
            r["pixels"]
            for r in group
        )

        mean_confidence = sum(
            r["mean_confidence"] * r["pixels"]
            for r in group
        ) / total_pixels

        # Remove tiny regions
        if total_pixels < MIN_PIXELS:
            total_filtered += 1
            continue

        # Remove low-confidence regions
        if mean_confidence < MIN_CONFIDENCE:
            total_filtered += 1
            continue

        # Create final event
        event = create_event(
            group,
            info["patch_info"]
        )

        final_events.append(event)


# ============================================================
# NUMBER THE EVENTS
# ============================================================

for index, event in enumerate(
    final_events,
    start=1
):

    event["event_id"] = f"change_{index:03d}"


# ============================================================
# SORT BY AOI THEN LOCATION
# ============================================================

final_events.sort(
    key=lambda x: (
        x["aoi"],
        x["location"]["latitude"],
        x["location"]["longitude"]
    )
)


# ============================================================
# CREATE FINAL JSON
# ============================================================

output = {

    "project":
        "SpaceNet 7 long-term urban development change detection",

    "model":
        data["model"],

    "change_type":
        data["change_type"],

    "time_period":
        data["time_period"],

    "processing": {

        "threshold":
            data["threshold"],

        "merge_distance_pixels":
            MERGE_DISTANCE,

        "minimum_region_pixels":
            MIN_PIXELS,

        "minimum_mean_confidence":
            MIN_CONFIDENCE,

        "source_crs":
            data["source_crs"],

        "output_crs":
            data["output_crs"]
    },

    "validation_aois":
        data["validation_aois"],

    "summary": {

        "validation_patches":
            data["validation_patch_count"],

        "raw_detected_regions":
            raw_region_count,

        "merged_regions":
            total_merged,

        "filtered_regions":
            total_filtered,

        "final_change_events":
            len(final_events),

        "predicted_changed_pixels":
            data["summary"]["predicted_changed_pixels"],

        "predicted_changed_area_m2":
            data["summary"]["predicted_changed_area_m2"]
    },

    "change_events":
        final_events
}


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_FILE, "w") as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL RAG INPUT")
print("=" * 60)

print(
    f"Raw regions:        {raw_region_count}"
)

print(
    f"After merging:      {total_merged}"
)

print(
    f"Filtered out:       {total_filtered}"
)

print(
    f"FINAL EVENTS:       {len(final_events)}"
)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print("=" * 60)