import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer

detector = MTCNN()
embedder = FaceNet()
l2 = Normalizer("l2")

def align_face(rgb: np.ndarray, box, keypoints) -> np.ndarray:
    """
    Align face using eye landmarks (simple rotation). If landmarks missing, returns cropped face.
    """
    x, y, w, h = box
    x, y = abs(x), abs(y)

    # Crop first (safe bounds)
    H, W = rgb.shape[:2]
    x2, y2 = min(W, x + w), min(H, y + h)
    x, y = max(0, x), max(0, y)
    face = rgb[y:y2, x:x2]
    if face.size == 0:
        return None

    if not keypoints or "left_eye" not in keypoints or "right_eye" not in keypoints:
        return face

    (lx, ly) = keypoints["left_eye"]
    (rx, ry) = keypoints["right_eye"]

    # Compute rotation angle
    dy = ry - ly
    dx = rx - lx
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate around center of the face crop
    (fh, fw) = face.shape[:2]
    center = (fw // 2, fh // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face, M, (fw, fh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned

def extract_face_and_embedding(bgr: np.ndarray):
    """
    Returns (embedding_128, box) or (None, None).
    Uses MTCNN detection, alignment, then FaceNet embeddings (128-D) with L2 normalization.
    """
    if bgr is None:
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None, None

    # pick highest-confidence face
    detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    d = detections[0]
    box = d["box"]
    keypoints = d.get("keypoints", {})

    face = align_face(rgb, box, keypoints)
    if face is None:
        return None, None

    # FaceNet expects 160x160 RGB
    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA).astype("float32")
    face = np.expand_dims(face, axis=0)  # (1,160,160,3)
    emb = embedder.embeddings(face)[0]   # (128,)
    emb = l2.transform([emb])[0]         # L2 normalize
    return emb.astype("float32"), box

def draw_label(frame, box, text, color=(0, 255, 0)):
    x, y, w, h = box
    x, y = abs(x), abs(y)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
