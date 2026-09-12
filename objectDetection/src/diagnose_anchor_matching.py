"""
Confirms (or rules out) the anchor/object-scale mismatch hypothesis.

For each image, transforms it exactly the way RetinaNet does internally
(model.transform), generates the actual anchors used at that resolution,
and computes each GT box's best IoU against all anchors. This tells you,
directly, whether your ground-truth objects can ever become positive
training examples under fg_iou_thresh/bg_iou_thresh -- independent of
learning rate, epochs, or anything else downstream.

Run this BEFORE changing anything else. If most boxes fall in the
"unmatched" bucket, that's the root cause of the zero-gradient / zero-
detection symptom, and changing the anchor generator (see updated
model.py) is the fix to make first.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from dataset import FAIR1MDataset, collate_fn
from model import get_model, get_device


def main():
    device = get_device()
    dataset = FAIR1MDataset("../data")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    model = get_model()
    model.to(device)
    model.eval()

    fg_thresh = model.proposal_matcher.high_threshold
    bg_thresh = model.proposal_matcher.low_threshold
    print(f"fg_iou_thresh={fg_thresh}  bg_iou_thresh={bg_thresh}")
    print(
        "(a GT box needs >= fg_iou_thresh IoU with SOME anchor to ever "
        "become a positive training example)\n"
    )

    n_images = 25
    total_boxes = 0
    unmatched_boxes = 0  # max IoU < bg_thresh -> zero training signal at all
    weak_boxes = 0       # bg_thresh <= max IoU < fg_thresh -> ignored either way
    matched_boxes = 0    # max IoU >= fg_thresh -> actually drives the loss

    it = iter(loader)
    for i in range(n_images):
        images, targets = next(it)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            image_list, transformed_targets = model.transform(images, targets)
            features = model.backbone(image_list.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            anchors = model.anchor_generator(image_list, list(features.values()))

        boxes = transformed_targets[0]["boxes"]
        anch = anchors[0]

        if boxes.numel() == 0:
            continue

        ious = box_iou(boxes, anch)  # [num_gt, num_anchors]
        max_iou_per_box, _ = ious.max(dim=1)

        total_boxes += boxes.shape[0]
        unmatched_boxes += (max_iou_per_box < bg_thresh).sum().item()
        weak_boxes += (
            (max_iou_per_box >= bg_thresh) & (max_iou_per_box < fg_thresh)
        ).sum().item()
        matched_boxes += (max_iou_per_box >= fg_thresh).sum().item()

        rounded = [round(v, 3) for v in max_iou_per_box.tolist()]
        print(f"img {i}: {boxes.shape[0]:3d} boxes | best IoU per box = {rounded}")

    print("\n=== Summary over", n_images, "images ===")
    print("Total GT boxes:                         ", total_boxes)
    print(f"Fully unmatched (<{bg_thresh:.2f} IoU, ZERO signal): {unmatched_boxes}")
    print(f"Weak / ignored ({bg_thresh:.2f}-{fg_thresh:.2f} IoU):        {weak_boxes}")
    print(f"Properly matched (>= {fg_thresh:.2f} IoU):        {matched_boxes}")

    if total_boxes > 0:
        pct_unmatched = 100 * unmatched_boxes / total_boxes
        print(f"\n{pct_unmatched:.1f}% of GT boxes never reach a positive anchor match.")
        if pct_unmatched > 30:
            print(
                "-> This confirms the anchor-scale mismatch hypothesis: the "
                "default RetinaNet anchors are too large for a big chunk of "
                "FAIR1M's small objects. Fix the anchor_generator (see updated "
                "model.py) before spending more time on LR/epochs."
            )


if __name__ == "__main__":
    main()