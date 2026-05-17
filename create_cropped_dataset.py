import os
import cv2
import shutil
import numpy as np

# -----------------------
# Paths
# -----------------------
IMG_DIR = "mc_dataset/images"
MASK_DIR = "mc_dataset/masks"

OUT_IMG_DIR = "mc_dataset_cropped/images"
OUT_MASK_DIR = "mc_dataset_cropped/masks"

# -----------------------
# Settings
# -----------------------
MIN_TOTAL_AREA = 800
MIN_COMPONENT_AREA = 300
SPLIT_DISTANCE = 450

# Adaptive crop settings
PADDING_X_RATIO = 0.45
PADDING_Y_RATIO = 0.35

# Minimum crop size so crop is not too tiny
MIN_CROP_W = 450
MIN_CROP_H = 600

# Set True only if you want square crop. For your madam’s idea, keep False.
MAKE_SQUARE = False

# -----------------------
# Reset previous cropped dataset
# -----------------------
if os.path.exists("mc_dataset_cropped"):
    shutil.rmtree("mc_dataset_cropped")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


def bbox_from_mask(binary_mask):
    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return xs.min(), ys.min(), xs.max(), ys.max()


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def adaptive_crop_bbox(
    bbox,
    img_w,
    img_h,
    pad_x_ratio=0.45,
    pad_y_ratio=0.35,
    min_crop_w=450,
    min_crop_h=600,
    make_square=False
):
    x1, y1, x2, y2 = bbox

    mask_w = x2 - x1
    mask_h = y2 - y1

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

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

    # Shift crop back inside image boundary
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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

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
    Split only when there are 2 clear, large, far-apart components.
    Fragmented parts of one soldier should not be split.
    """
    if len(components) < 2:
        return False

    components = sorted(components, key=lambda c: c["area"], reverse=True)
    c1, c2 = components[0], components[1]

    distance = np.linalg.norm(c1["center"] - c2["center"])

    if distance >= SPLIT_DISTANCE:
        return True

    return False


def crop_and_save(image, mask, bbox, base, crop_id):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = adaptive_crop_bbox(
        bbox=bbox,
        img_w=img_w,
        img_h=img_h,
        pad_x_ratio=PADDING_X_RATIO,
        pad_y_ratio=PADDING_Y_RATIO,
        min_crop_w=MIN_CROP_W,
        min_crop_h=MIN_CROP_H,
        make_square=MAKE_SQUARE
    )

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    if crop_img.size == 0 or crop_mask.size == 0:
        return False

    out_base = f"{base}_crop_{crop_id}"

    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_base + ".jpg"), crop_img)
    cv2.imwrite(os.path.join(OUT_MASK_DIR, out_base + ".png"), crop_mask)

    print(f"Saved {out_base}: crop={crop_img.shape[1]}x{crop_img.shape[0]}")

    return True


image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total_saved = 0
total_skipped = 0
total_split = 0
total_single = 0

for img_name in image_files:
    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, base + ".png")

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Missing image/mask:", img_name)
        total_skipped += 1
        continue

    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    binary = (mask > 0).astype(np.uint8)
    total_area = int(binary.sum())

    if total_area < MIN_TOTAL_AREA:
        print("Skipped tiny/unclear mask:", img_name)
        total_skipped += 1
        continue

    components = get_large_components(binary)

    # CASE 1: split only if two large soldiers are very far apart
    if should_split_components(components):
        components = sorted(components, key=lambda c: c["area"], reverse=True)[:2]

        saved_for_image = 0

        for idx, comp in enumerate(components, start=1):
            ok = crop_and_save(
                image=image,
                mask=mask,
                bbox=comp["bbox"],
                base=base,
                crop_id=idx
            )

            if ok:
                total_saved += 1
                saved_for_image += 1

        total_split += 1
        print(f"Split into {saved_for_image} crops: {img_name}")

    # CASE 2: default, one adaptive crop using overall soldier mask bbox
    else:
        bbox = bbox_from_mask(binary)

        if bbox is None:
            total_skipped += 1
            continue

        ok = crop_and_save(
            image=image,
            mask=mask,
            bbox=bbox,
            base=base,
            crop_id=1
        )

        if ok:
            total_saved += 1
            total_single += 1

print("\nDone.")
print("Saved cropped samples:", total_saved)
print("Single-crop images:", total_single)
print("Split images:", total_split)
print("Skipped samples:", total_skipped)