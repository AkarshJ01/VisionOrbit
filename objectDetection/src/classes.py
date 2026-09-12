"""
FAIR1M class definitions.

FAIR1M has 37 fine-grained sub-categories under 5 super-categories
(Ship, Vehicle, Airplane, Court, Road). torchvision's RetinaNet uses a
sigmoid/focal-loss classification head, but the `num_classes` argument it
takes still reserves index 0 for "background" -- so we shift all real
class ids up by 1 (1..37) and set NUM_CLASSES = 37 + 1 = 38.

Label id 0 should NEVER appear in your ground-truth targets.
"""

# Order matches the FAIR1M paper / torchgeo's FAIR1M dataset class list.
CLASSES = [
    "Passenger Ship",
    "Motorboat",
    "Fishing Boat",
    "Tugboat",
    "other-ship",
    "Engineering Ship",
    "Liquid Cargo Ship",
    "Dry Cargo Ship",
    "Warship",
    "Small Car",
    "Bus",
    "Cargo Truck",
    "Dump Truck",
    "other-vehicle",
    "Van",
    "Trailer",
    "Tractor",
    "Excavator",
    "Truck Tractor",
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "ARJ21",
    "C919",
    "A220",
    "A321",
    "A330",
    "A350",
    "other-airplane",
    "Baseball Field",
    "Basketball Court",
    "Football Field",
    "Tennis Court",
    "Roundabout",
    "Intersection",
    "Bridge",
]

# name -> label id, reserving 0 for background
CLASS_TO_IDX = {name: idx + 1 for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# +1 for background, required by torchvision's retinanet_resnet50_fpn(num_classes=...)
NUM_CLASSES = len(CLASSES) + 1