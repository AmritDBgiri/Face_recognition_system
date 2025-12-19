import os
import mxnet as mx
import cv2
import numpy as np

REC_PATH = "faces_umd/train.rec"
IDX_PATH = "faces_umd/train.idx"
OUT_DIR = "umd_images"

MAX_IMAGES = 6000        
MAX_IMAGES_PER_ID = 70   # avoid imbalance

os.makedirs(OUT_DIR, exist_ok=True)

record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, 'r')
keys = list(record.keys)

saved = 0
id_count = {}

for k in keys:
    if saved >= MAX_IMAGES:
        break

    item = record.read_idx(k)
    header, img = mx.recordio.unpack(item)

    label = int(header.label)
    person_dir = os.path.join(OUT_DIR, f"person_{label:04d}")
    os.makedirs(person_dir, exist_ok=True)

    cnt = id_count.get(label, 0)
    if cnt >= MAX_IMAGES_PER_ID:
        continue

    img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue

    cv2.imwrite(os.path.join(person_dir, f"{cnt}.jpg"), img)

    id_count[label] = cnt + 1
    saved += 1

print("✅ Extraction completed")
print("Total images saved:", saved)
print("Total identities:", len(id_count))
