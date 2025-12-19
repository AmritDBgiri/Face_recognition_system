import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from joblib import load
from pathlib import Path

from src.utils import extract_face_and_embedding, draw_label

st.set_page_config(page_title="FaceNet Face Recognition", layout="centered")

@st.cache_resource
def load_model():
    clf = load("models/classifier_svm.joblib")
    meta = json.load(open("models/meta.json", "r", encoding="utf-8"))
    labelmap = {int(k): v for k, v in meta["labelmap"].items()}
    thresh = float(meta["unknown_threshold"])
    return clf, labelmap, thresh

st.title("Face Recognition (FaceNet + SVM)")
st.write("Upload an image → detect face → extract FaceNet embedding → predict identity.")

clf, labelmap, thresh = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    emb, box = extract_face_and_embedding(bgr)
    if emb is None:
        st.error("No face detected.")
    else:
        THRESHOLD = 0.25
        proba = clf.predict_proba([emb])[0]
        best_id = int(proba.argmax())
        best_p = float(proba[best_id])
        name = labelmap[best_id] if best_p >= THRESHOLD else "UNKNOWN"

        draw_label(bgr, box, f"{name} ({best_p:.2f})")
        out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(out_rgb, caption="Prediction", use_container_width=True)
        st.success(f"Predicted: {name}  |  Confidence: {best_p:.2f}")
