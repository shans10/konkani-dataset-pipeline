import os
import cv2
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ---------------- CONFIG ----------------

DATASET_ROOT = "dataset_konkani"

TSV_FILE = os.path.join(DATASET_ROOT, "dataset_konkani.tsv")

OUTPUT_ROOT = "dataset_konkani_otsu"

NUM_WORKERS = cpu_count()

CHUNK_SIZE = 100000

# ----------------------------------------


def process_image(rel_path):

    input_path = os.path.join(DATASET_ROOT, rel_path)
    output_path = os.path.join(OUTPUT_ROOT, rel_path)

    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return

    # Optional: Gaussian blur improves Otsu for document images
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Otsu binarization
    _, bin_img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(output_path, bin_img)


def main():

    total_images = 0

    with Pool(NUM_WORKERS) as pool:

        with pd.read_csv(TSV_FILE, sep="\t", chunksize=CHUNK_SIZE) as reader:

            for chunk in reader:

                paths = chunk["image_path"].tolist()

                total_images += len(paths)

                list(
                    tqdm(
                        pool.imap_unordered(process_image, paths),
                        total=len(paths),
                        desc="Processing chunk"
                    )
                )

    print("Finished processing:", total_images)


if __name__ == "__main__":

    main()
