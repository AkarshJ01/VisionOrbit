from change_detector import detect_change


# ============================================================
# TEST IMAGE PAIR
# ============================================================

before_image = (
    "data/test_public/"
    "L15-0369E-1244N_1479_3214_13/"
    "images_masked/"
    "global_monthly_2018_02_mosaic_"
    "L15-0369E-1244N_1479_3214_13.tif"
)

after_image = (
    "data/test_public/"
    "L15-0369E-1244N_1479_3214_13/"
    "images_masked/"
    "global_monthly_2019_12_mosaic_"
    "L15-0369E-1244N_1479_3214_13.tif"
)


# ============================================================
# RUN CNN
# ============================================================

result = detect_change(
    before_image,
    after_image
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print("\nSummary:")

for key, value in result["summary"].items():
    print(f"{key}: {value}")


# ============================================================
# PRINT DETECTED EVENTS
# ============================================================

print("\nDetected change events:")

for event in result["events"]:

    print("\n------------------------------")

    print(
        f"Event ID: "
        f"{event['event_id']}"
    )

    print(
        f"Change type: "
        f"{event['change_type']}"
    )

    print(
        f"Location: "
        f"{event['latitude']}, "
        f"{event['longitude']}"
    )

    print(
        f"Area: "
        f"{event['area_m2']} m²"
    )

    print(
        f"Mean confidence: "
        f"{event['mean_confidence']}"
    )

    print(
        f"Max confidence: "
        f"{event['max_confidence']}"
    )

print("\n")
print("=" * 70)
print("TEST FINISHED")
print("=" * 70)