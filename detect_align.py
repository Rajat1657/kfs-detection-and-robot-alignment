import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ===================== LOAD MODELS =====================
kfs_model = YOLO("final2.pt")
symbol_model = YOLO("symbol.pt")
tf_model = tf.keras.models.load_model("keras_model.h5")

TF_INPUT_SIZE = (224, 224)
CLASS_NAMES = ["Fake", "Real"]

# ===================== ALIGNMENT PARAMS =====================
LOWER_BLUE = np.array([90, 70, 50])
UPPER_BLUE = np.array([130, 255, 255])

MIN_AREA = 70000
MAX_AREA = 80000
MARGIN = 150

# ===================== PERFORMANCE =====================
FRAME_SKIP = 3
frame_count = 0

# ===================== SMOOTHING =====================
prev_face = None
alpha = 0.7  # smoothing factor

# ===================== UTILS =====================
def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def is_valid_rect(rect):
    if rect is None:
        return False

    pts = rect.reshape(4, 2)

    # Check side lengths
    dists = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    if min(dists) < 20:
        return False

    return True


def get_box_face(box_roi):
    gray = cv2.cvtColor(box_roi, cv2.COLOR_BGR2GRAY)

    # Better than Canny
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = 0

    h, w = box_roi.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.2:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        aspect = cw / float(ch)

        if 0.5 < aspect < 1.5:
            rect_area = cw * ch
            fill_ratio = area / rect_area

            score = area * fill_ratio

            if score > best_score:
                best_score = score
                best_rect = approx

    return best_rect


def warp_face(box_roi, face_cnt, size=640):
    if not is_valid_rect(face_cnt):
        return None, None

    pts = order_points(face_cnt)

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(box_roi, M, (size, size))

    # Reject blank warps
    if np.mean(warped) < 10:
        return None, None

    Minv = cv2.getPerspectiveTransform(dst, pts)

    return warped, Minv


def run_tf(img):
    img = cv2.resize(img, TF_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = tf_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx])


# ===================== WINDOWS =====================
cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Symbol Input (TF)", cv2.WINDOW_NORMAL)

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()

last_label = "Unknown"
last_score = 0.0

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    height, width, _ = frame.shape
    centre = width // 2

    # HSV for alignment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Draw alignment lines
    cv2.line(frame, (centre - MARGIN, 0), (centre - MARGIN, height), (255,255,255), 2)
    cv2.line(frame, (centre + MARGIN, 0), (centre + MARGIN, height), (255,255,255), 2)

    # ===================== KFS DETECTION =====================
    kfs_results = kfs_model(frame, verbose=False)

    if kfs_results and len(kfs_results[0].boxes) > 0:

        box = max(kfs_results[0].boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # ===================== ALIGNMENT =====================
        roi_mask = blue_mask[y1:y2, x1:x2]
        blue_ratio = (cv2.countNonZero(roi_mask) / roi_mask.size) if roi_mask.size > 0 else 0

        area = (x2 - x1) * (y2 - y1)

        x_aligned = False
        z_aligned = False

        # Z axis
        areadifference = 0
        if area < MIN_AREA:
            areadifference = MIN_AREA-area
            cv2.putText(frame, "Move Forward", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        elif area > MAX_AREA:
            areadifference = MAX_AREA-area
            cv2.putText(frame, "Move Backward", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            z_aligned = True

        # X axis
        xdifference = 0
        if x1 < (centre - MARGIN):
            xdifference =x1- (centre-MARGIN)
            cv2.putText(frame, "Move Left", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        elif x2 > (centre + MARGIN):
            xdifference=x2 - (centre-MARGIN)
            cv2.putText(frame, "Move Right", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            x_aligned = True

        # Status
        cv2.putText(frame, "Z Align OK" if z_aligned else "Z Not Align",
                    (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if z_aligned else (0,0,255), 2)

        cv2.putText(frame, "X Align OK" if x_aligned else "X Not Align",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if x_aligned else (0,0,255), 2)

        cv2.putText(frame, f"Blue: {blue_ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        print(f"areadiff :{areadifference}, xdiff{xdifference}")
        # ===================== SYMBOL PIPELINE =====================
        if frame_count % FRAME_SKIP == 0:

            box_roi = frame[y1:y2, x1:x2]

            if box_roi.size != 0:

                face = get_box_face(box_roi)

                # smoothing
                if face is not None:
                    if prev_face is not None:
                        face = alpha * face + (1 - alpha) * prev_face
                        face = face.astype(np.int32)
                    prev_face = face
                else:
                    face = prev_face

                warped, Minv = None, None

                if face is not None:
                    warped, Minv = warp_face(box_roi, face)

                if warped is not None:
                    input_img = warped
                    use_warp = True
                    cv2.imshow("Warped Image", warped)
                else:
                    input_img = box_roi
                    use_warp = False
                    cv2.imshow("Warped Image", box_roi)

                input_resized = cv2.resize(input_img, (640, 640))

                sym_results = symbol_model(input_resized, conf=0.15, verbose=False)

                if sym_results and len(sym_results[0].boxes) > 0:

                    sym_box = max(sym_results[0].boxes, key=lambda b: float(b.conf))
                    sx1, sy1, sx2, sy2 = map(int, sym_box.xyxy[0])

                    h_in, w_in = input_resized.shape[:2]
                    h_orig, w_orig = input_img.shape[:2]

                    sx1 = int(sx1 * (w_orig / w_in))
                    sx2 = int(sx2 * (w_orig / w_in))
                    sy1 = int(sy1 * (h_orig / h_in))
                    sy2 = int(sy2 * (h_orig / h_in))

                    symbol_crop = input_img[sy1:sy2, sx1:sx2]

                    if symbol_crop.size != 0:
                        label, score = run_tf(symbol_crop)
                        last_label = label
                        last_score = score

                        disp = symbol_crop.copy()
                        color = (0,255,0) if label=="Real" else (0,0,255)

                        cv2.putText(disp, f"{label} ({score:.2f})",
                                    (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        cv2.imshow("Symbol Input (TF)", disp)

                    color = (0,255,0) if last_label=="Real" else (0,0,255)

                    if use_warp and Minv is not None:
                        pts = np.array([
                            [sx1, sy1],
                            [sx2, sy1],
                            [sx2, sy2],
                            [sx1, sy2]
                        ], dtype="float32").reshape(-1,1,2)

                        mapped = cv2.perspectiveTransform(pts, Minv)
                        mapped[:,:,0] += x1
                        mapped[:,:,1] += y1

                        cv2.polylines(frame, [mapped.astype(int)], True, color, 2)

                    else:
                        cv2.rectangle(frame,
                                      (sx1+x1, sy1+y1),
                                      (sx2+x1, sy2+y1),
                                      color, 2)

        cv2.putText(frame, f"{last_label} ({last_score:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
