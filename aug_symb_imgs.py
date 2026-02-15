import cv2
import numpy as np
import os
import random

# ================= CONFIG =================
INPUT_DIR = "all_symb"
OUTPUT_DIR = "aug_symb"

AUGS_PER_IMAGE = 1

ROTATE_RANGE = (-30, 30)     # degrees
HSV_H = (-10, 10)
HSV_S = (0.7, 1.3)
HSV_V = (0.7, 1.3)
PERSPECTIVE = 0.12           # distortion strength
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- AUGMENTATION FUNCTIONS ----------

def random_rotate(img):
    h, w = img.shape[:2]
    angle = random.uniform(*ROTATE_RANGE)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[..., 0] += random.uniform(*HSV_H)
    hsv[..., 1] *= random.uniform(*HSV_S)
    hsv[..., 2] *= random.uniform(*HSV_V)

    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_perspective(img):
    h, w = img.shape[:2]
    d = PERSPECTIVE * min(w, h)

    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------- MAIN LOOP ----------

for img_name in os.listdir(INPUT_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(INPUT_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    for i in range(AUGS_PER_IMAGE):
        aug = img.copy()

        aug = random_hsv(aug)
        aug = random_rotate(aug)
        aug = random_perspective(aug)

        out_name = f"{base}_aug{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), aug)

print("Augmentation complete.")
