"""
Step 2/3 -- Train the plate-character detector.

This trains a YOLOv8 object detector that finds AND classifies every
character on a plate in one pass -- it replaces both the old contour-based
segmentation step and Tesseract entirely. Run this after prepare_dataset.py.

--------------------------------------------------------------------
SETUP
--------------------------------------------------------------------
    pip install ultralytics

A GPU makes this much faster but isn't required for a dataset this size
(3000 images) -- CPU training will just take longer.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
    python train_plate_detector.py
"""

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
DATA_YAML = r"dataset/data.yaml"   # the one written by prepare_dataset.py

# Start from a pretrained COCO checkpoint (transfer learning) rather than
# training from scratch -- with only ~2700 training images this matters a
# lot for how well the model generalizes.
#   yolov8n.pt = nano  (fastest, smallest, good starting point)
#   yolov8s.pt = small (a bit more accurate, still light)
BASE_MODEL = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16          # lower this (e.g. 8) if you hit an out-of-memory error
RUN_NAME = "iran_plate_chars"

# Your dataset is synthetically generated (clean rendered plates), but
# you'll be running inference on real phone photos. A model trained purely
# on clean synthetic renders tends to overfit to that style -- it gets
# excellent validation metrics (because val is drawn from the same
# synthetic generator) but sees genuinely unfamiliar input on a real photo
# (camera blur, uneven lighting, embossed-metal shading, JPEG compression,
# perspective from a handheld angle). Pushing augmentation harder is a
# partial fix; mixing in even a small set of labeled real photos (see
# finetune_on_real_photos.py) is the more reliable one.
#
# `pip install albumentations` before running this -- Ultralytics silently
# skips its Blur/MedianBlur/ToGray/CLAHE augmentations if the package isn't
# installed, and those specifically help close the synthetic-vs-real gap.

# ---------------------------------------------------------------------------


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        patience=20,       # stop early if val loss plateaus for 20 epochs

        # Persian characters are direction- and orientation-sensitive:
        # mirroring or flipping a glyph can turn it into a different
        # character (or an invalid one). Disable flip augmentations so the
        # model doesn't learn from corrupted labels.
        fliplr=0.0,
        flipud=0.0,

        # Pushed noticeably higher than a typical starting point, because
        # synthetic renders are too "clean" -- the model needs to see much
        # more geometric and photometric variation during training than
        # your dataset itself contains, to have any chance of handling a
        # real handheld photo at an angle under real lighting.
        degrees=15.0,
        translate=0.15,
        scale=0.3,
        shear=3.0,
        perspective=0.0005,   # simulates the off-axis camera angle a real photo has

        # Plate characters are usually black-on-white/silver with little
        # color signal -- keep color jitter mild so the model doesn't
        # overfit to hue/saturation noise that isn't actually informative.
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,

        # Mosaic (stitching 4 images together) is a strong augmentation
        # that's great for natural-scene detection but can be counter-
        # productive here, since it stitches unrelated plates together in
        # ways that don't resemble real photos of a single plate. Turn it
        # off for the last several epochs so the model fine-tunes on
        # realistic full-plate layouts.
        close_mosaic=10,
    )

    # Run validation once more at the end and print a summary.
    metrics = model.val(data=DATA_YAML)
    print("\nValidation mAP50:", metrics.box.map50)
    print("Validation mAP50-95:", metrics.box.map)
    print(f"\nBest weights saved to: runs/detect/{RUN_NAME}/weights/best.pt")
    print("Use that path as MODEL_WEIGHTS in recognize_plate.py")


if __name__ == "__main__":
    main()
