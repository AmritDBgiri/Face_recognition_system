import os, random, shutil

SRC = "umd_images"
DST = "dataset"

splits = ["train", "val", "test"]
for s in splits:
    os.makedirs(os.path.join(DST, s), exist_ok=True)

for person in os.listdir(SRC):
    imgs = os.listdir(os.path.join(SRC, person))
    if len(imgs) < 10:
        continue

    random.shuffle(imgs)
    n = len(imgs)
    t1 = int(0.7 * n)
    t2 = int(0.85 * n)

    split_map = {
        "train": imgs[:t1],
        "val": imgs[t1:t2],
        "test": imgs[t2:]
    }

    for split, files in split_map.items():
        out_dir = os.path.join(DST, split, person)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(SRC, person, f),
                        os.path.join(out_dir, f))

print("✅ Dataset split completed")
