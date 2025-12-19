import json
import argparse
import cv2
from joblib import load
from utils import extract_face_and_embedding, draw_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to test image")
    args = ap.parse_args()

    clf = load("models/classifier_svm.joblib")
    meta = json.load(open("models/meta.json", "r", encoding="utf-8"))
    labelmap = {int(k): v for k, v in meta["labelmap"].items()}
    thresh = 0.20

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError("Image not found")

    emb, box = extract_face_and_embedding(img)
    if emb is None:
        print("❌ No face detected.")
        return

    proba = clf.predict_proba([emb])[0]
    best_id = int(proba.argmax())
    best_p = float(proba[best_id])

    name = labelmap[best_id] if best_p >= thresh else "UNKNOWN"
    draw_label(img, box, f"{name} ({best_p:.2f})")

    cv2.imshow("FaceNet Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
