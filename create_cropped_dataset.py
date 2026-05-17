import os
import cv2
import numpy as np

# -----------------------
# Paths
# -----------------------
IMG_DIR = "mc_dataset/images"
MASK_DIR = "mc_dataset/masks"

OUT_IMG_DIR = "mc_dataset_cropped/images"
OUT_MASK_DIR = "mc_dataset_cropped/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# -----------------------
# Settings
# -----------------------
MIN_AREA = 800          # ignore tiny/unclear masks
MERGE_DISTANCE = 80     # if soldiers are close, keep together
PADDING_RATIO = 0.35    # add surrounding context around soldier


def get_bbox(component_mask):
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def merge_close_boxes(boxes, merge_distance=80):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        current = boxes[i]
        used[i] = True
        changed = True

        while changed:
            changed = False
            c_center = bbox_center(current)

            for j in range(len(boxes)):
                if used[j]:
                    continue

                dist = np.linalg.norm(c_center - bbox_center(boxes[j]))

                if dist < merge_distance:
                    x1 = min(current[0], boxes[j][0])
                    y1 = min(current[1], boxes[j][1])
                    x2 = max(current[2], boxes[j][2])
                    y2 = max(current[3], boxes[j][3])
                    current = (x1, y1, x2, y2)

                    used[j] = True
                    changed = True

        merged.append(current)

    return merged


def add_padding_to_bbox(bbox, img_w, img_h, padding_ratio=0.35):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w - 1, x2 + pad_x)
    y2 = min(img_h - 1, y2 + pad_y)

    return x1, y1, x2, y2


image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total_saved = 0
total_skipped = 0

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

    # Match mask size to image size if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []

    for label_id in range(1, num_labels):  # 0 = background
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < MIN_AREA:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        boxes.append((x, y, x + w, y + h))

    if len(boxes) == 0:
        print("Skipped unclear/tiny mask:", img_name)
        total_skipped += 1
        continue

    # Merge close soldiers, but separate far soldiers
    boxes = merge_close_boxes(boxes, MERGE_DISTANCE)

    img_h, img_w = image.shape[:2]

    for crop_id, box in enumerate(boxes):
        x1, y1, x2, y2 = add_padding_to_bbox(
            box,
            img_w,
            img_h,
            padding_ratio=PADDING_RATIO
        )

        crop_img = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        if crop_img.size == 0 or crop_mask.size == 0:
            continue

        out_base = f"{base}_crop_{crop_id+1}"

        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_base + ".jpg"), crop_img)
        cv2.imwrite(os.path.join(OUT_MASK_DIR, out_base + ".png"), crop_mask)

        total_saved += 1

print("Done.")
print("Saved cropped samples:", total_saved)
print("Skipped samples:", total_skipped)