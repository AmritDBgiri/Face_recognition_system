# Face Recognition System — FaceNet + SVM Version

This version implements a **modern face recognition pipeline** using **FaceNet embeddings** combined with a **Support Vector Machine (SVM)** classifier.  
It is designed to be **robust, scalable, and production-oriented**, unlike classical pixel-based approaches.

---

## 🔍 Overview

**Pipeline Flow:**

1. Face detection from images / webcam
2. Feature extraction using **FaceNet (512-D embeddings)**
3. Classification using **SVM**
4. Probability-based thresholding for `UNKNOWN` identity
5. Support for image, webcam, and web UI inference

---

## 🧠 Why FaceNet + SVM?

| Component | Purpose |
|---------|--------|
| FaceNet | Converts faces into discriminative 512-D embeddings |
| SVM | Classifies embeddings efficiently |
| Thresholding | Rejects low-confidence predictions |

FaceNet embeddings are highly separable, making SVM a strong lightweight classifier.

---

## 📂 Folder Structure

```text
facenet_svm_version/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── classifier_svm.joblib
│   ├── meta.json
│   ├── train_X.npy
│   ├── train_y.npy
│   ├── val_X.npy
│   ├── val_y.npy
│   ├── test_X.npy
│   ├── test_y.npy
│   └── *_labelmap.json
│
├── src/
│   ├── utils.py
│   ├── build_embeddings.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer_image.py
│   └── infer_webcam.py
│
├── extract_umdface.py
├── split_umdface.py
├── requirements.txt
└── README.md
