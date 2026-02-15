import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time

# ===================== LOAD MODELS =====================
kfs_model = YOLO("final2.pt")
symbol_model = YOLO("symbol.pt")
tf_model = tf.keras.models.load_model("keras_model.h5")

TF_INPUT_SIZE = (224, 224)
CLASS_NAMES = ["Fake", "Real"]

# ===================== PARAMS =====================
MIN_AREA = 70000
MAX_AREA = 80000
MARGIN = 150

SYMBOL_CONF = 0.5
TF_CONF = 0.80

# ===================== TF FUNCTION =====================
def run_tf(img):
    # resize
    img = cv2.resize(img, TF_INPUT_SIZE)

    # convert BGR -> RGB (IMPORTANT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize
    img = img.astype(np.float32) / 255.0

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    pred = tf_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))

    return CLASS_NAMES[idx], float(pred[idx])



def valid_input(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.var(gray) > 25 and np.mean(gray) > 20


# ===================== CAMERA =====================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Camera not found")
    exit()

last_label = "No Symbol"
last_score = 0.0

# ===================== WINDOWS =====================
cv2.namedWindow("Live")
cv2.namedWindow("TF")

# ===================== MAIN LOOP =====================
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    centre = w // 2

    # ===================== KFS DETECTION =====================
    kfs_results = kfs_model(frame, verbose=False)

    if kfs_results and len(kfs_results[0].boxes) > 0:

        # best box
        box = max(kfs_results[0].boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        roi = frame[y1:y2, x1:x2]

        if roi.size != 0:

            # ================= SYMBOL DETECTION =================
            sym_results = symbol_model(roi, conf=0.25, verbose=False)

            crop = None
            symbol_found = False

            if sym_results and len(sym_results[0].boxes) > 0:

                sb = max(sym_results[0].boxes, key=lambda b: float(b.conf))
                conf_sym = float(sb.conf)

                if conf_sym > SYMBOL_CONF:
                    symbol_found = True

                    sx1, sy1, sx2, sy2 = map(int, sb.xyxy[0])

                    # clamp coordinates
                    sx1 = max(0, sx1)
                    sy1 = max(0, sy1)
                    sx2 = min(roi.shape[1], sx2)
                    sy2 = min(roi.shape[0], sy2)

                    crop = roi[sy1:sy2, sx1:sx2]

                    # draw symbol box
                    cv2.rectangle(frame,
                                  (x1 + sx1, y1 + sy1),
                                  (x1 + sx2, y1 + sy2),
                                  (0,255,0), 2)

            # ================= TF CLASSIFICATION =================
            if crop is not None and crop.size > 0 and symbol_found:

                if valid_input(crop):

                    label, score = run_tf(crop)

                    if score > TF_CONF:
                        last_label = label
                        last_score = score
                    else:
                        last_label = "No Symbol"
                        last_score = 0.0

                else:
                    last_label = "No Symbol"
                    last_score = 0.0

            else:
                last_label = "No Symbol"
                last_score = 0.0

            # ================= DISPLAY TF =================
            if crop is not None and crop.size > 0:
                disp = crop.copy()
            else:
                disp = roi.copy()

            color = (0,255,0) if last_label == "Real" else (0,0,255)

            cv2.putText(disp,
                        f"{last_label} ({last_score:.2f})",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            cv2.imshow("TF", disp)

            # ================= ALIGNMENT =================
            if last_label == "Real":

                area = (x2 - x1) * (y2 - y1)

                if area < MIN_AREA:
                    cv2.putText(frame, "Forward", (50,300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                elif area > MAX_AREA:
                    cv2.putText(frame, "Backward", (50,300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                if x1 < (centre - MARGIN):
                    cv2.putText(frame, "Left", (50,250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                elif x2 > (centre + MARGIN):
                    cv2.putText(frame, "Right", (50,250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # show label on main frame
        cv2.putText(frame,
                    f"{last_label} ({last_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2)

    # ================= FPS =================
    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
