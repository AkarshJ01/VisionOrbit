"""
Dataset loader for FAIR1M (blanchon/FAIR1M mirror on Hugging Face).

Expected directory layout under `data/` (matches what you get from
`git clone https://huggingface.co/datasets/blanchon/FAIR1M`):

    data/
        images/
            1002.tif
            1003.tif
            ...
        labelXmls/
            1002.xml
            1003.xml
            ...

Each XML looks like (PASCAL-VOC-like, FAIR1M-specific):

    <annotation>
      <source><filename>1002.tif</filename></source>
      <objects>
        <object>
          <points>
            <point>100.0,200.0</point>
            <point>150.0,200.0</point>
            <point>150.0,250.0</point>
            <point>100.0,250.0</point>
            <point>100.0,200.0</point>   <!-- closing point, repeats first -->
          </points>
          <possibleresult><name>Small Car</name></possibleresult>
        </object>
        ...
      </objects>
    </annotation>

FAIR1M boxes are ORIENTED (4-point polygons). torchvision's RetinaNet only
supports axis-aligned boxes, so we take the min/max x and y of each
polygon to get an axis-aligned [x1, y1, x2, y2] box. This is a standard
simplification -- you lose the rotation angle but keep a valid box that
tightly (often loosely on rotated objects) bounds the object.
"""

import os
import glob
from xml.etree.ElementTree import parse

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from classes import CLASS_TO_IDX

Image.MAX_IMAGE_PIXELS = None  # FAIR1M tiffs can be large


def _parse_annotation(xml_path):
    """Parse one FAIR1M labelXml file into boxes + class names."""
    root = parse(xml_path).getroot()
    boxes, labels = [], []

    objects = root.find("objects")
    if objects is None:
        return boxes, labels

    for obj in objects.findall("object"):
        points_elm = obj.find("points")
        if points_elm is None:
            continue

        pts = []
        for p in points_elm.findall("point"):
            x_str, y_str = p.text.split(",")
            pts.append((float(x_str), float(y_str)))

        if len(pts) < 4:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        possibleresult = obj.find("possibleresult")
        name_elm = possibleresult.find("name") if possibleresult is not None else None
        class_name = name_elm.text.strip() if name_elm is not None and name_elm.text else None

        if class_name not in CLASS_TO_IDX:
            # unknown/typo'd class name in the xml -- skip rather than crash
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_TO_IDX[class_name])

    return boxes, labels


class FAIR1MDataset(Dataset):
    """
    Args:
        root: path to the `data` folder containing `images/` and `labelXmls/`
        transforms: optional callable applied to (image_tensor, target) and
            returning the transformed pair. If None, images are just
            converted to a float tensor in [0, 1] (no augmentation/resize).
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labelXmls")

        image_paths = sorted(
            glob.glob(os.path.join(self.image_dir, "*.tif"))
            + glob.glob(os.path.join(self.image_dir, "*.tiff"))
        )

        # keep only images that actually have a matching annotation file
        self.samples = []
        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(self.label_dir, stem + ".xml")
            if os.path.exists(xml_path):
                self.samples.append((img_path, xml_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No (image, xml) pairs found under {root}. "
                f"Expected {self.image_dir} and {self.label_dir}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        boxes, labels = _parse_annotation(xml_path)

        if len(boxes) == 0:
            # RetinaNet can technically handle empty targets, but an image
            # with zero valid objects is more likely an annotation quirk --
            # give it one dummy background-free entry list instead of crashing.
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        image_tensor = F.to_tensor(image)  # -> float32 [0,1], CxHxW

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def collate_fn(batch):
    """RetinaNet expects a list of images and a list of target dicts, not a stacked batch tensor."""
    return tuple(zip(*batch))


if __name__ == "__main__":
    # FAIR1M dataset is located at ../data
    ds = FAIR1MDataset(root=os.path.join("..", "data"))

    print(f"Found {len(ds)} images with annotations")

    img, target = ds[0]

    print("image shape:", img.shape)
    print("boxes:", target["boxes"].shape)
    print("labels:", target["labels"])