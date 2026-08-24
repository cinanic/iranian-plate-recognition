"""
Step 1/3 -- Prepare the dataset for YOLO training.

You currently have:
    images/   <- 3000 .png files
    labels/   <- 3000 .txt files (YOLO format), same base filename as the image
    data.yaml <- class names (0-9 digits + letter classes)

Ultralytics YOLO expects a train/ and val/ split, e.g.:
    output/images/train/*.png   output/labels/train/*.txt
    output/images/val/*.png     output/labels/val/*.txt

Your current data.yaml points `train` and `val` at the *same* folder, which
means there's no real validation set (the model would be "validated" on
data it trained on -- accuracy numbers would look great and mean nothing).
This script fixes that: it splits your flat folders into train/val and
writes a corrected data.yaml.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Edit the paths below, then:
    python prepare_dataset.py
"""

import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
SOURCE_IMAGES_DIR = r"images"      # your existing flat folder of 3000 images
SOURCE_LABELS_DIR = r"labels"      # your existing flat folder of 3000 label .txt files
SOURCE_YAML = r"data.yaml"         # your existing class-mapping file

OUTPUT_DIR = r"dataset"            # where the train/val split will be written
VAL_FRACTION = 0.10                # 10% held out for validation (~300 images)
SEED = 42                          # fixed seed so the split is reproducible

# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    src_images = Path(SOURCE_IMAGES_DIR)
    src_labels = Path(SOURCE_LABELS_DIR)
    out = Path(OUTPUT_DIR)

    image_files = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    if not image_files:
        raise SystemExit(f"No images found in {src_images.resolve()}")

    # Match each image to its label file by matching base filename; skip pairs
    # that don't line up instead of silently mis-pairing data.
    pairs = []
    missing = []
    for img_path in image_files:
        label_path = src_labels / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing.append(img_path.name)

    if missing:
        print(f"WARNING: {len(missing)} images have no matching label file and will be skipped.")
        print("First few:", missing[:5])

    if not pairs:
        raise SystemExit("No image/label pairs matched -- check your folder paths.")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / label_path.name)

    # Carry over the class names from the original data.yaml, but point
    # train/val at the new split directories.
    yaml_text = Path(SOURCE_YAML).read_text(encoding="utf-8")
    names_start = yaml_text.index("names:")
    names_block = yaml_text[names_start:]

    new_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"{names_block}"
    )
    (out / "data.yaml").write_text(new_yaml, encoding="utf-8")

    print(f"Train: {len(train_pairs)} images -> {out}/images/train")
    print(f"Val:   {len(val_pairs)} images -> {out}/images/val")
    print(f"Wrote {out}/data.yaml -- use this file (not the original) for training.")


if __name__ == "__main__":
    main()
