import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from utils import extract_face_and_embedding

def build_split(split_dir: str, out_prefix: str):
    X, y = [], []
    label2name = {}
    name2label = {}
    label = 0

    persons = sorted([p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))])
    for person in persons:
        if person not in name2label:
            name2label[person] = label
            label2name[label] = person
            label += 1

        person_dir = os.path.join(split_dir, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in tqdm(images, desc=f"[{out_prefix}] {person}", leave=False):
            img_path = os.path.join(person_dir, img_name)
            bgr = cv2.imread(img_path)
            emb, _ = extract_face_and_embedding(bgr)
            if emb is None:
                continue
            X.append(emb)
            y.append(name2label[person])

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    with open(f"{out_prefix}_labelmap.json", "w", encoding="utf-8") as f:
        json.dump(label2name, f, indent=2)

    print(f"✅ Saved {out_prefix}_X.npy ({X.shape}) and {out_prefix}_y.npy ({y.shape})")
    print(f"✅ Saved {out_prefix}_labelmap.json with {len(label2name)} classes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset folder containing train/val/test")
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.dataset, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        build_split(split_dir, out_prefix=os.path.join("models", split))

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
