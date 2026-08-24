"""
Step 3/3 -- Run the trained detector on a plate photo.

Unlike the old Tesseract pipeline, this does detection + segmentation +
classification in a single model pass, and works directly on full plate
photos (angled, real-world lighting) rather than needing a clean,
pre-cropped, axis-aligned image.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Edit MODEL_WEIGHTS and DEFAULT_IMAGE_PATH below, then:
    python recognize_plate.py
or:
    python recognize_plate.py path/to/plate.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = r"runs/detect/iran_plate_chars/weights/best.pt"
DEFAULT_IMAGE_PATH = r"C:\Users\Niwa-AI01\Downloads\213.jpg"
CONF_THRESHOLD = 0.4       # detections below this confidence are dropped

# If your input images are full car photos (plate is just a small part of
# a bigger scene) rather than already-cropped plates, turn this on. It
# locates the plate first -- using the blank plate template's distinctive
# blue flag box as an anchor -- then runs character detection only on that
# cropped region.
LOCATE_PLATE_FIRST = False
PLATE_TEMPLATE_PATH = r"savari.png"  # your blank plate template image

# ---------------------------------------------------------------------------
# Output uses the class-name strings exactly as they appear in your
# data.yaml ('0'..'9', 'B', 'N', 'EIN', ...) -- no translation into Persian
# letters. results.names comes straight from the model, which was trained
# on your data.yaml, so this is already what you labeled with.


def _flag_box_profile(template_gray_bgr):
    """Measures the blank template's flag-box proportions once, so plate
    localization works for any plate template you point it at (not just
    this one) rather than hardcoding pixel values from a single example."""
    hsv = cv2.cvtColor(template_gray_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    if n <= 1:
        raise ValueError(
            "Couldn't find a blue flag box in the template image -- is "
            "PLATE_TEMPLATE_PATH pointing at a real blank plate template?"
        )
    # largest blue component = the flag box
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = stats[idx]
    th, tw = template_gray_bgr.shape[:2]
    return {
        "flagbox_height_frac_of_plate_height": h / th,
        "plate_aspect_w_over_h": tw / th,
    }


def locate_plate(photo_bgr, template_bgr):
    """Finds the plate in a full photo by locating its flag box (a
    distinctive, mostly-solid blue rectangle) via color + shape filtering,
    then extrapolating the full plate's bounding box from the flag box's
    position using the template's own proportions. Returns the cropped
    plate region, or None if no plausible flag box was found.

    This is deliberately not feature-matching (ORB/SIFT) against the
    template -- a blank template has too little distinctive texture (most
    of it is empty white space) for reliable keypoint matching against a
    cluttered real-world background; a handful of tests here produced
    unstable, sometimes out-of-frame homographies. Color + shape filtering
    on the flag box proved much more reliable in practice.
    """
    profile = _flag_box_profile(template_bgr)

    h_img, w_img = photo_bgr.shape[:2]
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)

    best = None
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 300:
            continue
        fill = area / (w * hh)
        aspect = hh / w if w else 0
        # A real flag box is a solid-filled, tall rectangle. Other blue
        # objects in a photo (sky, painted beams, clothing) are usually
        # either not solid (low fill) or not this elongated -- this filter
        # is what lets it ignore background clutter instead of just
        # grabbing the single largest blue blob in the photo.
        if fill > 0.5 and 1.4 <= aspect <= 4.0:
            if best is None or area > best[0]:
                best = (area, x, y, w, hh)

    if best is None:
        return None

    _, fx, fy, fw, fh = best
    plate_h = int(fh / profile["flagbox_height_frac_of_plate_height"])
    plate_w = int(plate_h * profile["plate_aspect_w_over_h"])

    # Generous padding: better to include a little extra background (the
    # character detector will just ignore it) than to clip part of the
    # plate off.
    pad_x, pad_y = int(0.05 * plate_w), int(0.15 * plate_h)
    x0 = max(0, fx - pad_x)
    y0 = max(0, fy - pad_y)
    x1 = min(w_img, fx + plate_w + pad_x)
    y1 = min(h_img, fy + plate_h + pad_y)
    return photo_bgr[y0:y1, x0:x1]


def recognize(model: YOLO, image_path: str, conf: float = CONF_THRESHOLD):
    results = model.predict(image_path, conf=conf, verbose=False)[0]
    names = results.names  # class id -> class name string, from data.yaml via the model

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        x_center = float(box.xywhn[0][0])  # normalized 0-1, for left-to-right ordering
        confidence = float(box.conf[0])
        detections.append((x_center, cls_name, confidence))

    detections.sort(key=lambda d: d[0])
    return detections


def format_plate(detections) -> str:
    """Groups detections into the main section (2 digits, letter, 3 digits)
    and the province box (last 2 digits), based on position and a large
    x-gap that separates the province box from the main section."""
    if not detections:
        return "(no characters detected)"

    xs = [d[0] for d in detections]
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    if gaps:
        gap_size, split_idx = max(gaps)
    else:
        gap_size, split_idx = 0, len(detections) - 1

    # Only treat the largest gap as the province-box divider if it's
    # meaningfully bigger than a normal inter-character gap.
    if gap_size > 0.06 and split_idx >= len(detections) - 3:
        main_dets = detections[: split_idx + 1]
        province_dets = detections[split_idx + 1:]
    else:
        main_dets = detections
        province_dets = []

    main_str = " ".join(name for _, name, _ in main_dets)
    province_str = " ".join(name for _, name, _ in province_dets)

    if province_str:
        return f"{main_str}   |   {province_str}"
    return main_str


def main():
    args = sys.argv[1:]
    image_path = args[0] if args else (DEFAULT_IMAGE_PATH or input("Path to plate image: ").strip().strip('"'))
    image_path = str(Path(image_path).expanduser())

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Could not read image: {image_path}")

    if LOCATE_PLATE_FIRST:
        template = cv2.imread(PLATE_TEMPLATE_PATH)
        if template is None:
            sys.exit(f"Could not read template image: {PLATE_TEMPLATE_PATH}")
        located = locate_plate(img, template)
        if located is None:
            sys.exit(
                "Could not locate a plate in this photo (no plausible flag "
                "box found). Try a photo where the blue flag box is clearly "
                "visible and not heavily obstructed, or set "
                "LOCATE_PLATE_FIRST = False and pass an already-cropped "
                "plate image instead."
            )
        img = located

    model = YOLO(MODEL_WEIGHTS)
    detections = recognize(model, img)

    if not detections:
        # Zero detections at the normal threshold could mean either "the
        # model isn't confident enough yet" (retrying at a much lower
        # threshold will show *something*) or "something more basic is
        # wrong" (wrong weights file, wrong image, model never learned
        # anything). Auto-diagnose instead of just printing nothing.
        print(f"No detections at conf={CONF_THRESHOLD}. Retrying at conf=0.01 to check...")
        low_conf_detections = recognize(model, img, conf=0.01)
        if low_conf_detections:
            best = max(low_conf_detections, key=lambda d: d[2])
            print(
                f"Found {len(low_conf_detections)} detections at very low "
                f"confidence (best: '{best[1]}' at {best[2]:.3f}).\n"
                f"-> The model IS detecting characters, just with low confidence.\n"
                f"   Lower CONF_THRESHOLD near the top of this file (e.g. to 0.1 or 0.15), "
                f"or the model likely needs more training epochs / more data for "
                f"underrepresented classes."
            )
        else:
            print(
                "Still zero detections even at conf=0.01.\n"
                "-> This points to something more basic than a confidence-threshold issue. Check:\n"
                f"   1) MODEL_WEIGHTS path is correct and points at YOUR trained run, not a "
                f"stray/old one:\n      currently set to: {MODEL_WEIGHTS}\n"
                f"      Look in runs/detect/ -- if you trained more than once, Ultralytics "
                f"auto-numbers new runs (iran_plate_chars2, iran_plate_chars3, ...), so the "
                f"weights this script loads may not be your latest/best training run.\n"
                f"   2) The image actually looks like your training images (same rough "
                f"crop/zoom level) -- a model trained on tightly-cropped plates may fail "
                f"completely on a full uncropped photo, and vice versa.\n"
                f"   3) Your training run's final metrics -- if mAP50 was near 0 when "
                f"training finished, the model didn't actually learn to detect anything, "
                f"and no amount of threshold tuning will fix that; it needs to be retrained "
                f"(more epochs, check for a data/label bug, or confirm loss was actually "
                f"decreasing across epochs)."
            )
        return

    print("Detections (left to right):")
    for x, name, conf in detections:
        print(f"  {name}  (conf={conf:.2f}, x={x:.3f})")

    print("\nPlate number:", format_plate(detections))


if __name__ == "__main__":
    main()
