import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.anchor_utils import AnchorGenerator


NUM_CLASSES = 38  # 37 FAIR1M classes + background


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _small_object_anchor_generator():
    """
    torchvision's default RetinaNet anchors use base sizes
    [32, 64, 128, 256, 512] (x2^(1/3), x2^(2/3) per octave) across
    P3-P7 -- tuned for COCO-scale objects. FAIR1M has many small objects
    ("Small Car", "Motorboat", etc.) at 0.3-0.8 m/pixel GSD that are
    often only ~8-20px across, especially after RetinaNet's internal
    resize (min_size=800/max_size=1333).

    A 32px+ anchor around a ~10-15px object caps out around 5-15% IoU,
    far below the fg_iou_thresh=0.5 needed to ever become a positive
    training example. That anchor/object scale mismatch is what starves
    the classification head of gradient signal (confirmed via
    diagnose_anchor_matching.py), independent of learning rate or epoch
    count.

    Shift the base sizes down (adjust to your data after running the
    diagnostic -- these are a starting point, not a guarantee) while
    keeping the same 3-scales x 3-aspect-ratios = 9 anchors/location
    structure so the head's num_anchors doesn't change.
    """
    base_sizes = [8, 16, 32, 64, 128]
    anchor_sizes = tuple(
        tuple(int(base * 2 ** (octave / 3)) for octave in range(3))
        for base in base_sizes
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def get_model(use_small_anchors=True):

    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT

    model = retinanet_resnet50_fpn(
        weights=weights
    )

    if use_small_anchors:
        model.anchor_generator = _small_object_anchor_generator()

    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES
    )

    return model


def get_param_groups(model, backbone_lr=1e-5, head_lr=1e-3):
    """
    Differential learning rates.

    The backbone/FPN start from COCO-pretrained weights and only need
    gentle fine-tuning. The classification AND regression heads are
    freshly re-initialized (38-class head is brand new; even the
    regression head's weights are re-initialized by torchvision's
    RetinaNetClassificationHead constructor) and need a much larger LR
    to move meaningfully away from their init in a realistic number of
    epochs on a ~1700-image dataset.

    Note: fixing the anchor scale mismatch (see _small_object_anchor_generator)
    matters more than this by itself -- if GT boxes still can't reach
    fg_iou_thresh, no learning rate will fix it. Do both.
    """
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


if __name__ == "__main__":

    device = get_device()

    print("Device:", device)

    model = get_model()

    model.to(device)

    print("RetinaNet loaded successfully")
    print("Number of classes:", NUM_CLASSES)
    print("Anchors per location:", model.anchor_generator.num_anchors_per_location()[0])