import os
import cv2
import shutil
import numpy as np

# =====================================================
# PATHS
# =====================================================
IMG_DIR = "kaggle_dataset/CamouflageData/img"
MASK_DIR = "kaggle_dataset/CamouflageData/gt"

OUT_IMG_DIR = "kaggle_dataset_cropped/images"
OUT_MASK_DIR = "kaggle_dataset_cropped/masks"

# =====================================================
# SETTINGS FOR KAGGLE DATASET: 854 x 480
# =====================================================

# Ignore very tiny masks/noise
MIN_TOTAL_AREA = 300
MIN_COMPONENT_AREA = 150

# Split two soldiers only if they are far apart
SPLIT_DISTANCE = 230

# Smaller padding = soldier becomes more visible / zoomed
PADDING_X_RATIO = 0.25
PADDING_Y_RATIO = 0.20

# Suitable for 854 x 480 images
MIN_CROP_W = 250
MIN_CROP_H = 280

# Keep False to preserve natural rectangular crop
MAKE_SQUARE = False


# =====================================================
# RESET OLD CROPPED KAGGLE DATASET
# =====================================================
if os.path.exists("kaggle_dataset_cropped"):
    shutil.rmtree("kaggle_dataset_cropped")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


# =====================================================
# FUNCTIONS
# =====================================================
def bbox_from_mask(binary_mask):
    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return xs.min(), ys.min(), xs.max(), ys.max()


def adaptive_crop_bbox(
    bbox,
    img_w,
    img_h,
    pad_x_ratio=PADDING_X_RATIO,
    pad_y_ratio=PADDING_Y_RATIO,
    min_crop_w=MIN_CROP_W,
    min_crop_h=MIN_CROP_H,
    make_square=MAKE_SQUARE
):
    x1, y1, x2, y2 = bbox

    mask_w = max(1, x2 - x1)
    mask_h = max(1, y2 - y1)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Crop based on soldier size + padding
    crop_w = int(mask_w * (1 + 2 * pad_x_ratio))
    crop_h = int(mask_h * (1 + 2 * pad_y_ratio))

    crop_w = max(crop_w, min_crop_w)
    crop_h = max(crop_h, min_crop_h)

    if make_square:
        side = max(crop_w, crop_h)
        crop_w = side
        crop_h = side

    nx1 = cx - crop_w // 2
    nx2 = cx + crop_w // 2
    ny1 = cy - crop_h // 2
    ny2 = cy + crop_h // 2

    # Keep crop inside image boundary
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0

    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0

    if nx2 > img_w:
        shift = nx2 - img_w
        nx1 -= shift
        nx2 = img_w

    if ny2 > img_h:
        shift = ny2 - img_h
        ny1 -= shift
        ny2 = img_h

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def get_large_components(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        components.append({
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "center": np.array([x + w / 2, y + h / 2])
        })

    return components


def should_split_components(components):
    """
    Split only when there are at least two clear soldiers
    that are far apart from each other.
    """
    if len(components) < 2:
        return False

    components = sorted(components, key=lambda c: c["area"], reverse=True)

    c1 = components[0]
    c2 = components[1]

    distance = np.linalg.norm(c1["center"] - c2["center"])

    return distance >= SPLIT_DISTANCE


def find_mask_path(base_name):
    """
    Supports PNG, JPG, JPEG masks.
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(MASK_DIR, base_name + ext)

        if os.path.exists(path):
            return path

    return None


def crop_and_save(image, mask, bbox, base_name, crop_id):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = adaptive_crop_bbox(
        bbox=bbox,
        img_w=img_w,
        img_h=img_h
    )

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    if crop_img.size == 0 or crop_mask.size == 0:
        return False

    # Make all masks consistent: background = 0, soldier = 255
    crop_mask = (crop_mask > 0).astype(np.uint8) * 255

    out_base = f"{base_name}_crop_{crop_id}"

    cv2.imwrite(
        os.path.join(OUT_IMG_DIR, out_base + ".jpg"),
        crop_img
    )

    cv2.imwrite(
        os.path.join(OUT_MASK_DIR, out_base + ".png"),
        crop_mask
    )

    print(
        f"Saved: {out_base} | "
        f"Crop size: {crop_img.shape[1]} x {crop_img.shape[0]}"
    )

    return True


# =====================================================
# PROCESS DATASET
# =====================================================
image_files = sorted([
    file_name for file_name in os.listdir(IMG_DIR)
    if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
])

total_saved = 0
total_skipped = 0
total_split_images = 0
total_single_images = 0

for img_name in image_files:
    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = find_mask_path(base)

    if mask_path is None:
        print(f"Missing mask: {img_name}")
        total_skipped += 1
        continue

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Could not read image or mask: {img_name}")
        total_skipped += 1
        continue

    # Ensure mask and image sizes match
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    binary_mask = (mask > 0).astype(np.uint8)
    total_area = int(binary_mask.sum())

    # Skip very tiny / unclear targets
    if total_area < MIN_TOTAL_AREA:
        print(f"Skipped tiny mask: {img_name}")
        total_skipped += 1
        continue

    components = get_large_components(binary_mask)

    # Case 1: Two clearly separate soldiers
    if should_split_components(components):
        components = sorted(
            components,
            key=lambda component: component["area"],
            reverse=True
        )[:2]

        saved_count = 0

        for crop_id, component in enumerate(components, start=1):
            saved = crop_and_save(
                image=image,
                mask=mask,
                bbox=component["bbox"],
                base_name=base,
                crop_id=crop_id
            )

            if saved:
                total_saved += 1
                saved_count += 1

        total_split_images += 1
        print(f"Split into {saved_count} soldier crops: {img_name}")

    # Case 2: One soldier or connected soldier region
    else:
        bbox = bbox_from_mask(binary_mask)

        if bbox is None:
            total_skipped += 1
            continue

        saved = crop_and_save(
            image=image,
            mask=mask,
            bbox=bbox,
            base_name=base,
            crop_id=1
        )

        if saved:
            total_saved += 1
            total_single_images += 1


print("\n========== DONE ==========")
print(f"Saved cropped samples: {total_saved}")
print(f"Single-soldier crops: {total_single_images}")
print(f"Images split into two crops: {total_split_images}")
print(f"Skipped samples: {total_skipped}")