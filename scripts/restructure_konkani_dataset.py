# This script restructures the Konkani dataset by moving images into shard directories and creating new CSV files with updated paths.

import os
import shutil

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_konkani")
SRC_IMG_DIR = os.path.join(DATASET_DIR, "konkani")

DEST_IMG_DIR = os.path.join(DATASET_DIR, "images")

SHARD_SIZE = 10000
CSV_LIMIT = 500000


def collect_images():

    paths = []

    for root, _, files in os.walk(SRC_IMG_DIR):
        for f in files:
            if f.endswith(".png"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, DATASET_DIR)
                paths.append(rel)

    return paths


def load_labels():

    dfs = []

    for f in os.listdir(DATASET_DIR):
        if f.startswith("dataset_konkani") and f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(DATASET_DIR, f)))

    df = pd.concat(dfs, ignore_index=True)

    # create mapping: image_path -> ground_truth
    return dict(zip(df["image_path"], df["ground_truth"]))


def main():

    os.makedirs(DEST_IMG_DIR, exist_ok=True)

    label_map = load_labels()

    images = list(label_map.keys())

    csv_rows = []
    csv_index = 1

    for idx, rel in enumerate(tqdm(images)):
        src = os.path.join(DATASET_DIR, rel)

        shard = idx // SHARD_SIZE
        shard_dir = os.path.join(DEST_IMG_DIR, f"shard_{shard:03d}")

        os.makedirs(shard_dir, exist_ok=True)

        new_name = f"{idx:09d}.png"

        dest = os.path.join(shard_dir, new_name)

        shutil.move(src, dest)

        new_rel = os.path.join("images", f"shard_{shard:03d}", new_name)

        csv_rows.append((new_rel, label_map[rel]))

        if len(csv_rows) >= CSV_LIMIT:
            df = pd.DataFrame(csv_rows, columns=["image_path", "ground_truth"])

            df.to_csv(
                os.path.join(DATASET_DIR, f"dataset_konkani_{csv_index}.csv"),
                index=False,
            )

            csv_rows = []
            csv_index += 1

    if csv_rows:
        df = pd.DataFrame(csv_rows, columns=["image_path", "ground_truth"])

        df.to_csv(
            os.path.join(DATASET_DIR, f"dataset_konkani_{csv_index}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
