from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


IMAGE_DIR = Path("processed/images")
MASK_DIR = Path("processed/masks")


# Get first processed image
image_path = sorted(IMAGE_DIR.glob("*.npy"))[0]

mask_path = MASK_DIR / image_path.name


# Load data
image = np.load(image_path)
mask = np.load(mask_path)


print("Image:", image_path)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("Image min:", image.min())
print("Image max:", image.max())

print("Mask values:", np.unique(mask))


# RGB image
rgb = image[:3]

# Convert CHW -> HWC
rgb = np.transpose(rgb, (1, 2, 0))

# Clip just in case
rgb = np.clip(rgb, 0, 1)


# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("Building Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rgb)
plt.imshow(mask, alpha=0.4)
plt.title("RGB + Building Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig("sample_visualization.png", dpi=150)

plt.show()