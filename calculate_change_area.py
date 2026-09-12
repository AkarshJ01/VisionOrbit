import rasterio

# ============================================================
# SETTINGS
# ============================================================

TIFF_PATH = (
    "data/"
    "L15-0487E-1246N_1950_3207_13/"
    "2019_12.tif"
)


# ============================================================
# READ GEOTIFF METADATA
# ============================================================

with rasterio.open(TIFF_PATH) as src:

    print("=" * 50)
    print("GEOTIFF INFORMATION")
    print("=" * 50)

    print("Width:", src.width)
    print("Height:", src.height)
    print("CRS:", src.crs)

    print("Transform:")
    print(src.transform)

    print()
    print("Bounds:")
    print(src.bounds)

    # Pixel dimensions in CRS units
    pixel_width = abs(src.transform.a)
    pixel_height = abs(src.transform.e)

    pixel_area = pixel_width * pixel_height

    print()
    print("=" * 50)
    print("PIXEL SIZE")
    print("=" * 50)

    print("Pixel width:", pixel_width)
    print("Pixel height:", pixel_height)
    print("Pixel area:", pixel_area, "square metres")

    print()
    print("Approximate GSD:",
          (pixel_width + pixel_height) / 2,
          "metres/pixel")
