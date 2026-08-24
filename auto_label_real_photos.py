"""
Optional step -- auto-label your unannotated real photos.

You have real photos but no label .txt files for them. Rather than drawing
every box by hand, let your already-trained model draft the labels first --
even though it's weak on real photos, it's not useless, and correcting a
draft box/class is much faster than creating one from nothing. This writes
standard YOLO-format .txt files (same format as your synthetic dataset),
so you can open them directly in a labeling tool for correction.

--------------------------------------------------------------------
AFTER RUNNING THIS
--------------------------------------------------------------------
Open the images in a labeling tool that reads/writes YOLO .txt format and
fix the mistakes (wrong class, missed character, wrong box position):
  - LabelImg (simplest, free, desktop): https://github.com/HumanSignal/labelImg
    Point it at your images folder + this labels folder, set format to
    YOLO -- it will load these draft boxes automatically so you're
    correcting, not starting from a blank image.
  - CVAT (web-based, more features): https://www.cvat.ai/
  - makesense.ai (browser, no install): https://www.makesense.ai/

A very low confidence threshold is used here on purpose -- it's fine (even
good) if it over-detects at this stage, since deleting a wrong box in
LabelImg is a single keypress. It's the missed characters that cost you
real time, so we're erring toward "too many draft boxes" over "too few".

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
    python auto_label_real_photos.py
"""

from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = r"runs/detect/iran_plate_chars/weights/best.pt"  # your synthetic-trained model
REAL_IMAGES_DIR = r"real_images"     # folder of your unannotated real photos
OUTPUT_LABELS_DIR = r"real_labels"   # where draft .txt files will be written

CONF_THRESHOLD = 0.05   # deliberately low -- see note above on over- vs under-detecting

# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    images_dir = Path(REAL_IMAGES_DIR)
    labels_dir = Path(OUTPUT_LABELS_DIR)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    if not image_files:
        raise SystemExit(f"No images found in {images_dir.resolve()}")

    model = YOLO(MODEL_WEIGHTS)

    empty_count = 0
    for img_path in image_files:
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]

        lines = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x_center, y_center, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if not lines:
            empty_count += 1

        label_path = labels_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Wrote {len(image_files)} draft label files to {labels_dir.resolve()}")
    if empty_count:
        print(
            f"WARNING: {empty_count} image(s) got zero draft boxes even at "
            f"conf={CONF_THRESHOLD} -- you'll need to label those ones fully "
            f"by hand, the model found nothing to draft there."
        )
    print("\nNext: open these in LabelImg (or similar) pointed at:")
    print(f"  images: {images_dir.resolve()}")
    print(f"  labels: {labels_dir.resolve()}")
    print("and correct the draft boxes/classes.")


if __name__ == "__main__":
    main()
