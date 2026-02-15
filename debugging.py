import cv2
import numpy as np
import tensorflow as tf

# Load model
tf_model = tf.keras.models.load_model("keras_model.h5")
CLASS_NAMES = ["Fake", "Real"]  # match TM order
TF_INPUT_SIZE = (224, 224)

def run_tf(img):
    img = cv2.resize(img, TF_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = tf_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx])

cap = cv2.VideoCapture(0)  # phone cam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed (press S to snapshot)", frame)
    key = cv2.waitKey(1) & 0xFF

    # -------- SNAPSHOT --------
    if key == ord('s'):
        snapshot = frame.copy()
        cv2.imshow("Snapshot", snapshot)

        # Manual crop selection
        roi = cv2.selectROI(
            "Snapshot",
            snapshot,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, w, h = roi
        if w > 0 and h > 0:
            cropped = snapshot[y:y+h, x:x+w]

            label, conf = run_tf(cropped)
            print(f"Prediction: {label} ({conf:.2f})")

            cv2.imshow("Cropped (TF Input)", cropped)

        cv2.destroyWindow("Snapshot")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
