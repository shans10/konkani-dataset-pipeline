import os
from multiprocessing import Pool, cpu_count

import cv2
import pandas as pd
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_konkani")

CSV_FILES = ["dataset_konkani_1.csv", "dataset_konkani_2.csv", "dataset_konkani_3.csv"]

CHUNK_SIZE = 100000
NUM_WORKERS = max(1, cpu_count() - 2)

# Disable PNG compression (important for speed)
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]


# =========================================================
# IMAGE PROCESSING
# =========================================================


def convert_to_grayscale(rel_path):

    img_path = os.path.join(DATASET_DIR, rel_path)

    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return "corrupt"

        # Already grayscale
        if len(img.shape) == 2:
            return "skip"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(img_path, gray, PNG_PARAMS)

        return "converted"

    except Exception:
        return "error"


# =========================================================
# CSV UTILITIES
# =========================================================


def detect_image_column(columns):

    candidates = ["image_path", "path", "img", "image"]

    for c in candidates:
        if c in columns:
            return c

    raise ValueError("No image column found in CSV")


def count_total_images():

    total = 0

    for csv_file in CSV_FILES:
        csv_path = os.path.join(DATASET_DIR, csv_file)

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, usecols=[0])
        total += len(df)

    return total


# =========================================================
# MAIN PROCESSING
# =========================================================


def main():

    print("\nStarting fast grayscale conversion")
    print("Dataset directory:", DATASET_DIR)
    print("Workers:", NUM_WORKERS)

    total_images = count_total_images()

    print("Total images to process:", total_images)

    converted = 0
    skipped = 0
    corrupt = 0

    with Pool(NUM_WORKERS) as pool:
        with tqdm(
            total=total_images,
            desc="Processing Images",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for csv_file in CSV_FILES:
                csv_path = os.path.join(DATASET_DIR, csv_file)

                if not os.path.exists(csv_path):
                    print(f"Skipping missing CSV: {csv_file}")
                    continue

                reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

                for chunk in reader:
                    img_col = detect_image_column(chunk.columns)

                    paths = chunk[img_col].tolist()

                    for result in pool.imap_unordered(convert_to_grayscale, paths):
                        if result == "converted":
                            converted += 1
                        elif result == "skip":
                            skipped += 1
                        elif result == "corrupt":
                            corrupt += 1

                        pbar.update(1)

    print("\nProcessing Complete")
    print("---------------------")
    print("Converted :", converted)
    print("Skipped   :", skipped)
    print("Corrupt   :", corrupt)


if __name__ == "__main__":
    main()
