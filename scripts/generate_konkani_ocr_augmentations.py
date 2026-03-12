import os
import cv2
import numpy as np
import pandas as pd
import random
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_konkani")
AUG_ROOT = os.path.join(SCRIPT_DIR, "dataset_konkani_augmented")

CSV_FILES = [
    "dataset_konkani_1.csv",
    "dataset_konkani_2.csv",
    "dataset_konkani_3.csv"
]

CHUNK_SIZE = 100000
MAX_CSV_ROWS = 500000

NUM_WORKERS = max(1, cpu_count() - 2)

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]

# =========================================================
# AUGMENTATIONS
# =========================================================

def rotate(img):
    h, w = img.shape[:2]
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def gaussian_noise(img):
    noise = random.normalvariate(0, 10)
    gauss = noise * (random.random())
    noisy = img + gauss
    return noisy.clip(0,255).astype("uint8")


def blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)


def dilation(img):

    inv = 255 - img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

    dilated = cv2.dilate(inv, kernel)

    return 255 - dilated


def erosion(img):

    inv = 255 - img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

    eroded = cv2.erode(inv, kernel)

    return 255 - eroded


def perspective(img):
    h, w = img.shape[:2]

    shift = 0.03

    pts1 = [
        [0,0],
        [w,0],
        [0,h],
        [w,h]
    ]

    pts2 = [
        [random.uniform(0,w*shift),random.uniform(0,h*shift)],
        [w-random.uniform(0,w*shift),random.uniform(0,h*shift)],
        [random.uniform(0,w*shift),h-random.uniform(0,h*shift)],
        [w-random.uniform(0,w*shift),h-random.uniform(0,h*shift)]
    ]

    M = cv2.getPerspectiveTransform(
        cv2.UMat(pts1).get().astype("float32"),
        cv2.UMat(pts2).get().astype("float32")
    )

    return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)


def bleed(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    dilated = cv2.dilate(img, kernel)
    blurred = cv2.GaussianBlur(dilated, (3,3), 0)
    result = cv2.addWeighted(img, 0.6, blurred, 0.4, 0)
    return result

def baseline_warp(img):
    """
    Simulate curved baseline often seen in scanned books.
    """

    h, w = img.shape[:2]

    amplitude = random.uniform(1.0, 3.0)
    frequency = random.uniform(1.5, 3.0)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        shift = amplitude * np.sin(2 * np.pi * y / (h * frequency))

        for x in range(w):
            map_x[y, x] = x + shift
            map_y[y, x] = y

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


AUGMENTATIONS = {
    "rotation": rotate,
    "noise": gaussian_noise,
    "blur": blur,
    "dilation": dilation,
    "erosion": erosion,
    "perspective": perspective,
    "bleed": bleed,
    "baseline_warp": baseline_warp
}

# =========================================================
# UTIL
# =========================================================

def count_total_images():

    total = 0

    for csv in CSV_FILES:

        path = os.path.join(DATASET_DIR, csv)

        if os.path.exists(path):
            df = pd.read_csv(path, usecols=[0])
            total += len(df)

    return total


def detect_image_column(columns):

    candidates = ["image_path","path","img","image"]

    for c in candidates:
        if c in columns:
            return c

    raise ValueError("No image column found")


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(args):

    rel_path, text, aug_name = args

    img_path = os.path.join(DATASET_DIR, rel_path)

    try:

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        aug_img = AUGMENTATIONS[aug_name](img)

        name = os.path.splitext(os.path.basename(rel_path))[0]
        new_name = f"{name}_{aug_name}.png"

        image_id = int(name)
        shard = image_id // 10000

        out_dir = os.path.join(
            AUG_ROOT,
            f"aug_{aug_name}",
            "images",
            f"shard_{shard:03d}"
        )

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, new_name)

        if not os.path.exists(out_path):
            cv2.imwrite(out_path, aug_img, PNG_PARAMS)

        rel_out = os.path.join(
            "images",
            f"shard_{shard:03d}",
            new_name
        )

        return (rel_out, text)

    except Exception:
        return None


# =========================================================
# CSV WRITER
# =========================================================

def write_csv(rows, aug_name, index):

    out_dir = os.path.join(AUG_ROOT, f"aug_{aug_name}")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"dataset_aug_{aug_name}_{index}.csv")

    df = pd.DataFrame(rows, columns=["image_path","ground_truth"])

    df.to_csv(csv_path, index=False)


# =========================================================
# MAIN
# =========================================================

def generate_augmentation(aug_name):

    print(f"\nGenerating augmentation: {aug_name}")

    total = count_total_images()

    csv_buffer = []
    csv_index = 1

    with Pool(NUM_WORKERS) as pool:

        with tqdm(total=total, desc=f"{aug_name}", position=0, dynamic_ncols=True) as pbar:

            for csv_file in CSV_FILES:

                csv_path = os.path.join(DATASET_DIR, csv_file)

                if not os.path.exists(csv_path):
                    continue

                reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

                for chunk in reader:

                    img_col = detect_image_column(chunk.columns)

                    tasks = [
                        (row[img_col], row["ground_truth"], aug_name)
                        for _, row in chunk.iterrows()
                    ]

                    for result in pool.imap_unordered(process_image, tasks):

                        if result is not None:
                            if result not in csv_buffer:
                                csv_buffer.append(result)

                        if len(csv_buffer) >= MAX_CSV_ROWS:
                            write_csv(csv_buffer, aug_name, csv_index)
                            csv_buffer = []
                            csv_index += 1

                        pbar.update(1)

    if csv_buffer:
        write_csv(csv_buffer, aug_name, csv_index)


# =========================================================
# ENTRY
# =========================================================

def main():

    print("\nStarting OCR augmentation pipeline")
    print("Workers:", NUM_WORKERS)

    os.makedirs(AUG_ROOT, exist_ok=True)

    args = sys.argv[1:]

    # If augmentation name provided
    if len(args) == 1:

        aug = args[0].strip()

        if aug not in AUGMENTATIONS:
            print(f"Invalid augmentation: {aug}")
            print("Available augmentations:", list(AUGMENTATIONS.keys()))
            return

        generate_augmentation(aug)

    # No argument → run all
    else:

        for aug in AUGMENTATIONS:
            generate_augmentation(aug)

    print("\nAll augmentations completed")

if __name__ == "__main__":
    main()
