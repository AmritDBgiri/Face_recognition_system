import json
import cv2
import time
from joblib import load
from utils import extract_face_and_embedding, draw_label

def main():
    clf = load("models/classifier_svm.joblib")
    meta = json.load(open("models/meta.json", "r", encoding="utf-8"))
    labelmap = {int(k): v for k, v in meta["labelmap"].items()}
    thresh = 0.20

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("🎥 Webcam started. Press ESC to exit.")
    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        emb, box = extract_face_and_embedding(frame)
        if emb is not None:
            proba = clf.predict_proba([emb])[0]
            best_id = int(proba.argmax())
            best_p = float(proba[best_id])
            name = labelmap[best_id] if best_p >= thresh else "UNKNOWN"
            draw_label(frame, box, f"{name}")

        frames += 1
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Live FaceNet Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
