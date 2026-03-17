import os
import cv2
import subprocess
import pandas as pd
import numpy as np

from tqdm import tqdm

# ---------------- CONFIG ----------------

DATASET_ROOT = "dataset_konkani"
TSV_FILE = os.path.join(DATASET_ROOT, "dataset_konkani.tsv")

MISMATCH_TSV = os.path.join(DATASET_ROOT, "mismatches.tsv")

CONF_THRESHOLD = 90
BATCH_SIZE = 1000
AUTOSAVE_INTERVAL = 20000

# ---------------- LOAD DATA ----------------

print("Loading dataset...")

df = pd.read_csv(
    TSV_FILE,
    sep="\t",
    dtype=str
)

print("Total samples:", len(df))

# ---------------- RESUME ----------------

if os.path.exists(MISMATCH_TSV):

    prev = pd.read_csv(MISMATCH_TSV, sep="\t")

    processed = len(prev)

    print("Resuming from:", processed)

    start_idx = processed
    mismatch_rows = prev.to_dict("records")

else:

    start_idx = 0
    mismatch_rows = []

# ---------------- SORT BY SHARD ----------------

df = df.iloc[start_idx:].copy()

df["shard"] = df["image_path"].apply(lambda x: x.split("/")[1])

df = df.sort_values("shard")

records = df.to_dict("records")

# ---------------- CROP VALIDATION ----------------

def bad_crop(image_path):

    full = os.path.join(DATASET_ROOT, image_path)

    img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return True

    _, bin_img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = bin_img.shape

    density = np.sum(bin_img > 0) / (h * w)

    if density < 0.05 or density > 0.55:
        return True

    projection = np.sum(bin_img > 0, axis=0)

    peaks = np.sum(projection > (0.1 * h))

    if peaks < 2:
        return True

    return False

# ---------------- PIPE OCR ----------------

def run_tesseract_pipe(img):

    success, encoded = cv2.imencode(".png", img)

    if not success:
        return "", 0

    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "hin", "--psm", "8", "tsv"],
        input=encoded.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    try:
        df = pd.read_csv(
            pd.io.common.BytesIO(proc.stdout),
            sep="\t",
            engine="python"
        )
    except:
        return "", 0

    if len(df) == 0:
        return "", 0

    row = df.iloc[-1]

    text = str(row.get("text", "")).strip()

    try:
        conf = float(row.get("conf", 0))
    except:
        conf = 0

    return text, conf

# ---------------- PROCESSING ----------------

for batch_start in tqdm(range(0, len(records), BATCH_SIZE)):

    batch = records[batch_start:batch_start + BATCH_SIZE]

    for r in batch:

        gt = r["ground_truth"].strip()
        img_path = r["image_path"]

        if bad_crop(img_path):

            mismatch_rows.append({
                "image_path": img_path,
                "ground_truth": gt
            })

            continue

        full = os.path.join(DATASET_ROOT, img_path)

        img = cv2.imread(full)

        if img is None:

            mismatch_rows.append({
                "image_path": img_path,
                "ground_truth": gt
            })

            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # ---------- ORIGINAL ----------
        pred1, conf1 = run_tesseract_pipe(gray)

        # ---------- FLIPPED ----------
        flip = cv2.flip(gray, 1)
        pred2, conf2 = run_tesseract_pipe(flip)

        # ---------- SMART VALIDATION ----------

        strict_match = (
            pred1 == gt and
            pred2 == gt and
            conf1 >= CONF_THRESHOLD and
            conf2 >= CONF_THRESHOLD
        )

        relaxed_match = (
            pred1 == gt and
            pred1 == pred2 and
            conf1 >= 85
        )

        match = strict_match or relaxed_match

        if not match:

            mismatch_rows.append({
                "image_path": img_path,
                "ground_truth": gt
            })

    # ---------- AUTOSAVE ----------

    if batch_start % AUTOSAVE_INTERVAL == 0 and batch_start != 0:

        print("Autosaving...")

        pd.DataFrame(mismatch_rows).to_csv(
            MISMATCH_TSV,
            sep="\t",
            index=False
        )

# ---------------- FINAL SAVE ----------------

pd.DataFrame(mismatch_rows).to_csv(
    MISMATCH_TSV,
    sep="\t",
    index=False
)

print("\nValidation complete.")
print("Mismatch TSV:", MISMATCH_TSV)

# ---------------- STATS ----------------

total_processed = start_idx + len(records)
total_mismatches = len(mismatch_rows)

ratio = total_mismatches / total_processed if total_processed > 0 else 0

print("\n===== DATASET STATS =====")
print(f"Total processed: {total_processed}")
print(f"Mismatches: {total_mismatches}")
print(f"Mismatch ratio: {ratio:.2%}")
