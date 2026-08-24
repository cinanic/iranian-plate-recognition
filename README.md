# Iranian License Plate Recognition (YOLOv8)

A YOLOv8-based pipeline that detects and reads Iranian license plate
characters directly from photos -- including full car photos, not just
pre-cropped plate images.

Unlike a classic OCR approach (segment characters, then run a generic text
recognizer on each one), this trains a single object detector that finds
*and* classifies every character on the plate in one pass. That turned out
to matter a lot in practice: Iranian plates use a stylized, embossed font
that generic OCR engines (e.g. Tesseract) were never trained on, and
per-character segmentation via classical CV (contours/thresholding) is
fragile under real-world lighting and camera angles. A trained detector
handles both problems at once.

## How it works

```
Full car photo
      |
      v
[optional] locate the plate itself, anchored on the flag box's
           distinctive blue color + shape (see "Locating the plate
           in a full photo" below)
      |
      v
YOLOv8 character detector
   -> finds every digit/letter on the plate AND classifies it
   -> in a single forward pass
      |
      v
Detections sorted left-to-right, split into
main section (2 digits + letter + 3 digits) + province code
      |
      v
Plate number
```

## Repository contents

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Splits a flat `images/`+`labels/` dataset into a proper YOLO train/val structure, and writes a corrected `data.yaml`. |
| `train_plate_detector.py` | Fine-tunes a pretrained YOLOv8 model on the plate-character dataset. |
| `recognize_plate.py` | Runs the trained model on a new image and prints the recognized plate number. Includes optional plate localization for full (uncropped) car photos. |
| `finetune_on_real_photos.py` | Continues training an existing model on a small set of real (non-synthetic) labeled photos, to close the sim-to-real gap (see [Known limitations](#known-limitations)). |
| `auto_label_real_photos.py` | Uses an existing trained model to pre-label unannotated real photos with draft YOLO boxes, so you're correcting rather than labeling from scratch in a tool like [LabelImg](https://github.com/HumanSignal/labelImg). |

## Setup

```bash
pip install -r requirements.txt
```

On Windows, no separate Tesseract installation is required for this
pipeline -- an earlier version of this project used Tesseract OCR, but it
was dropped in favor of the YOLO detector above, precisely because
Tesseract's Farsi model isn't trained on the plate font and struggled with
isolated-character accuracy.

## Usage

### 1. Prepare your dataset

Expects a flat `images/` folder and matching `labels/` folder (YOLO-format
`.txt` files, same base filename as their image) plus a `data.yaml`
listing your classes. Edit the paths at the top of the script, then:

```bash
python prepare_dataset.py
```

This writes a `dataset/` folder with `images/train`, `images/val`,
`labels/train`, `labels/val`, and a corrected `data.yaml` pointing at them.

### 2. Train

Edit `DATA_YAML` at the top to point at `dataset/data.yaml`, then:

```bash
python train_plate_detector.py
```

Uses transfer learning from a pretrained YOLOv8n checkpoint. Flip
augmentation is disabled on purpose -- mirroring a Persian character can
turn it into a different (or invalid) character, so leaving default flip
augmentation on would corrupt training labels.

### 3. Recognize

Edit `MODEL_WEIGHTS` (path to your trained `best.pt`) and either pass an
image path as an argument or set `DEFAULT_IMAGE_PATH`:

```bash
python recognize_plate.py path/to/plate.jpg
```

Output uses the class-name strings exactly as defined in your `data.yaml`
(`'0'`..`'9'`, `'B'`, `'N'`, `'EIN'`, ...) -- see [Class mapping](#class-mapping).

If detections come back empty, the script automatically retries at a much
lower confidence threshold and tells you whether the model found something
weak (needs more training / lower `CONF_THRESHOLD`) or genuinely nothing
(wrong weights path, or the test image doesn't resemble training data).

#### Locating the plate in a full photo

If your input is a full car photo rather than an already-cropped plate,
set `LOCATE_PLATE_FIRST = True` and point `PLATE_TEMPLATE_PATH` at a blank
plate template image (a plate graphic with no characters on it -- just the
flag box, border, and province box).

Plate localization does **not** use feature matching (ORB/SIFT) against
the template -- a blank template has too little distinctive texture for
that to work reliably against a cluttered background; in testing it
produced unstable, sometimes out-of-frame results. Instead, it:

1. Detects the flag box by its distinctive solid blue color, filtering
   candidates by fill-ratio and aspect-ratio (measured automatically from
   your template, so it isn't hardcoded to one specific plate design) to
   reject other blue objects in the scene.
2. Extrapolates the full plate's bounding box from the flag box's position
   using the template's own proportions.

This is a targeted heuristic, not a general object detector -- it assumes
the flag box is reasonably visible and not badly obstructed or overexposed.
If it proves unreliable across your real photos, the more robust (but more
work) fallback is training a second, dedicated YOLO model whose only job
is finding the plate itself, the same way `train_plate_detector.py` trains
one for characters.

### 4. (Optional) Close the sim-to-real gap

A model trained purely on synthetic plate renders tends to score very
highly on validation (because validation is drawn from the same generator)
while performing poorly on real camera photos -- see
[Known limitations](#known-limitations). If that happens:

1. Gather a batch of real plate photos (100-300 is a reasonable starting
   point -- far fewer than the synthetic set, since the model already
   knows character shapes and mainly needs to adapt to real-world
   input).
2. Run `auto_label_real_photos.py` to get draft YOLO labels for them from
   your existing model.
3. Correct the drafts in [LabelImg](https://github.com/HumanSignal/labelImg)
   (or [CVAT](https://www.cvat.ai/) / [makesense.ai](https://www.makesense.ai/)).
4. Run `prepare_dataset.py` again on the corrected real set.
5. Run `finetune_on_real_photos.py`, pointed at your synthetic-trained
   weights and the corrected real dataset. This continues training at a
   low learning rate, nudging the model toward real-world input rather
   than relearning character shapes from scratch.

## Dataset

Training data was generated with
[barzansaeedpour/iranian-license-plate-generator](https://github.com/barzansaeedpour/iranian-license-plate-generator),
an MIT-licensed synthetic Iranian license plate generator that also emits
YOLO-format labels. All credit for the plate rendering and labeling
approach goes to that project.

### Class mapping

| ID | Label | Character | | ID | Label | Character |
|---|---|---|---|---|---|---|
| 0-9 | `'0'`-`'9'` | Digits 0-9 | | 20 | `SAD` | ص |
| 10 | `EIN` | ع (Public) | | 21 | `TA` | ط |
| 11 | `B` | ب | | 22 | `V` | و |
| 12 | `N` | ن | | 23 | `M` | م |
| 13 | `T` | ت (Taxi) | | 24 | `Y` | ی |
| 14 | `H` | هـ | | 25 | `L` | ل |
| 15 | `D` | د | | 26 | `Z` | ز (Blue plate / Ministry of Defense) |
| 16 | `Q` | ق | | 27 | `ZH` | Disabled (wheelchair symbol) |
| 17 | `J` | ج | | 28 | `TH` | ث (Sepah) |
| 18 | `HE` | ح | | 29 | `P` | پ (Police) |
| 19 | `SIN` | س | | 30 | `SH` | ش (Military) |
| | | | | 31 | `A` | الف (Government) |

## Known limitations

- **Sim-to-real gap.** Training exclusively on synthetic renders can
  produce excellent validation metrics (mAP50 ~0.99 in initial testing)
  while performing poorly on real handheld/camera photos, since synthetic
  images lack real camera blur, uneven lighting, embossed-metal shading,
  and compression artifacts. See the fine-tuning workflow above.
- **Plate localization is a heuristic, not a trained detector.** It relies
  on the flag box being a clearly visible, solid blue region. Heavy glare,
  obstruction, or unusual lighting on the flag box can cause it to fail to
  locate the plate.
- **Rare classes.** Some letter classes (police, military, government,
  disabled-parking, etc.) are far less common in casual photos than
  ordinary passenger-plate digits and the `B`/`N`/... letters, so accuracy
  on those classes may lag behind the more common ones unless your
  training data specifically balances for them.

## Acknowledgments

- [barzansaeedpour/iranian-license-plate-generator](https://github.com/barzansaeedpour/iranian-license-plate-generator)
  for the synthetic dataset generator and YOLO class-mapping convention
  used here.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the
  detection framework.

## License

MIT -- see [LICENSE](LICENSE).
