import os
from collections import Counter

# -----------------------------
# Function to check dataset sync
# -----------------------------
def check_dataset(img_dir, mask_dir, dataset_name):
    print("\n" + "=" * 60)
    print(f"Checking dataset: {dataset_name}")
    print("=" * 60)

    if not os.path.exists(img_dir):
        print(f"Image folder not found: {img_dir}")
        return

    if not os.path.exists(mask_dir):
        print(f"Mask folder not found: {mask_dir}")
        return

    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    images = [os.path.splitext(f)[0] for f in image_files]
    masks = [os.path.splitext(f)[0] for f in mask_files]

    image_set = set(images)
    mask_set = set(masks)

    # -----------------------------
    # Check images without masks
    # -----------------------------
    missing_masks = []
    for img in images:
        if img not in mask_set:
            missing_masks.append(img)

    print("\nImages without masks:")
    print(missing_masks)
    print("Count:", len(missing_masks))

    # -----------------------------
    # Check masks without images
    # -----------------------------
    extra_masks = []
    for mask in masks:
        if mask not in image_set:
            extra_masks.append(mask)

    print("\nMasks without images:")
    print(extra_masks)
    print("Count:", len(extra_masks))

    # -----------------------------
    # Check duplicate names in images
    # -----------------------------
    img_count = Counter(images)
    duplicate_images = [name for name, count in img_count.items() if count > 1]

    print("\nDuplicate filenames in images folder:")
    print(duplicate_images)
    print("Count:", len(duplicate_images))

    # -----------------------------
    # Check duplicate names in masks
    # -----------------------------
    mask_count = Counter(masks)
    duplicate_masks = [name for name, count in mask_count.items() if count > 1]

    print("\nDuplicate filenames in masks folder:")
    print(duplicate_masks)
    print("Count:", len(duplicate_masks))

    # -----------------------------
    # Summary
    # -----------------------------
    print("\nSummary:")
    print("Total image files:", len(image_files))
    print("Total mask files:", len(mask_files))

    if (
        len(missing_masks) == 0
        and len(extra_masks) == 0
        and len(duplicate_images) == 0
        and len(duplicate_masks) == 0
    ):
        print("Status: Dataset is synced ✅")
    else:
        print("Status: Dataset has issues ⚠️")


# -----------------------------
# Original dataset
# -----------------------------
check_dataset(
    img_dir="mc_dataset/images",
    mask_dir="mc_dataset/masks",
    dataset_name="Original Dataset"
)

# -----------------------------
# Cropped dataset
# -----------------------------
check_dataset(
    img_dir="mc_dataset_cropped/images",
    mask_dir="mc_dataset_cropped/masks",
    dataset_name="Cropped Dataset"
)