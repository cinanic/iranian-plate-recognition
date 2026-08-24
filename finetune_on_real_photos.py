"""
Optional step 4 -- fine-tune on real photos.

Your first training run got excellent metrics (mAP50 ~0.995) but is
apparently not generalizing to real handheld photos -- that's expected when
training data is 100% synthetic. The single most reliable fix is not more
synthetic augmentation, it's showing the model a modest number of real,
labeled photos.

You do NOT need anywhere near 3000 real photos for this. Because the model
already knows the character shapes cold from the synthetic data, a small
real set (even 100-300 images) used to *continue* training from your
existing best.pt, at a low learning rate, is usually enough to teach it
"oh, THIS is what real camera input looks like" without needing to relearn
character recognition from zero.

--------------------------------------------------------------------
WHAT YOU NEED TO DO FIRST
--------------------------------------------------------------------
Label a batch of real plate photos in the exact same YOLO format as your
existing dataset (same data.yaml class IDs) -- e.g. using a tool like
LabelImg, Roboflow, or CVAT. Put them in the same images/+labels/ flat
folder layout your original dataset used, then run this AFTER
prepare_dataset.py-style splitting (or point REAL_DATA_YAML at a data.yaml
you've already split into train/val).

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
    python finetune_on_real_photos.py
"""

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
STARTING_WEIGHTS = r"runs/detect/iran_plate_chars/weights/best.pt"  # your synthetic-trained model
REAL_DATA_YAML = r"real_dataset/data.yaml"   # data.yaml for your labeled real-photo set

EPOCHS = 30               # far fewer than the original run -- this is fine-tuning, not training from scratch
IMAGE_SIZE = 640
BATCH_SIZE = 8            # real-photo datasets are usually smaller; a smaller batch is fine
RUN_NAME = "iran_plate_chars_realphoto_finetune"

# ---------------------------------------------------------------------------


def main():
    model = YOLO(STARTING_WEIGHTS)

    model.train(
        data=REAL_DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        patience=10,

        # Much lower learning rate than the original run -- we want to
        # nudge the existing weights toward real-photo characteristics,
        # not overwrite what they already learned about character shapes.
        lr0=0.001,

        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
    )

    metrics = model.val(data=REAL_DATA_YAML)
    print("\nReal-photo validation mAP50:", metrics.box.map50)
    print("Real-photo validation mAP50-95:", metrics.box.map)
    print(f"\nFine-tuned weights: runs/detect/{RUN_NAME}/weights/best.pt")
    print("Point MODEL_WEIGHTS in recognize_plate.py at this file.")


if __name__ == "__main__":
    main()
