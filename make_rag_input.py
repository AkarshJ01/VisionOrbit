import json

INPUT_FILE = "all_validation_change_results.json"
OUTPUT_FILE = "rag_input.json"

# Load CNN results
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rag_output = {
    "project": data["project"],
    "model": data["model"],
    "change_type": data["change_type"],
    "time_period": data["time_period"],
    "validation_aois": data["validation_aois"],
    "changes": []
}

# Extract useful information from every detected region
for patch in data["patch_results"]:

    for region in patch["detected_regions"]:

        change = {
            "aoi": patch["aoi"],
            "time_period": patch["time_period"],

            "location": {
                "latitude": region["centroid_latitude"],
                "longitude": region["centroid_longitude"]
            },

            "area_m2": region["area_m2"],

            "confidence": {
                "mean": region["mean_confidence"],
                "maximum": region["max_confidence"]
            },

            "bounding_box_pixels": region["bounding_box_full_image_pixels"],

            "change_type": patch["change_type"]
        }

        rag_output["changes"].append(change)

# Add summary information
rag_output["summary"] = {
    "validation_patches": data["validation_patch_count"],
    "detected_change_regions": data["summary"]["detected_change_regions"],
    "predicted_changed_area_m2": data["summary"]["predicted_changed_area_m2"],
    "mean_prediction_probability": data["summary"]["mean_prediction_probability"]
}

# Save
with open(OUTPUT_FILE, "w") as f:
    json.dump(rag_output, f, indent=2)

print("=" * 60)
print("RAG INPUT CREATED")
print("=" * 60)
print(f"Total change events: {len(rag_output['changes'])}")
print(f"Output file: {OUTPUT_FILE}")
print("=" * 60)