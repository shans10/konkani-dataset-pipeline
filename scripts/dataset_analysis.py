import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATASET_ROOT = "dataset_konkani"

CSV_FILES = [
    os.path.join(DATASET_ROOT, "dataset_konkani_1.csv"),
    os.path.join(DATASET_ROOT, "dataset_konkani_2.csv"),
    os.path.join(DATASET_ROOT, "dataset_konkani_3.csv"),
]

OUTPUT_JSON = os.path.join(DATASET_ROOT, "dataset_report.json")

CHUNK_SIZE = 100000
# ----------------------------------------


# Devanagari digit unicode range
DEVANAGARI_DIGITS = set("०१२३४५६७८९")


def is_number(token):
    """
    Returns True if token consists only of Devanagari digits
    """
    return all(c in DEVANAGARI_DIGITS for c in token)


def analyze_dataset():

    class_freq = defaultdict(int)

    total_samples = 0
    number_count = 0
    word_count = 0

    unique_numbers = set()
    unique_words = set()

    for csv_file in CSV_FILES:
        print(f"\nProcessing {csv_file}")

        for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
            texts = chunk["ground_truth"].astype(str)

            for token in texts:
                token = token.strip()

                if not token:
                    continue

                total_samples += 1
                class_freq[token] += 1

                if is_number(token):
                    number_count += 1
                    unique_numbers.add(token)

                else:
                    word_count += 1
                    unique_words.add(token)

    report = {
        "dataset_summary": {
            "total_samples": total_samples,
            "total_classes": len(class_freq),
            "total_words": word_count,
            "total_numbers": number_count,
            "unique_words": len(unique_words),
            "unique_numbers": len(unique_numbers),
        },
        "class_frequency_distribution": dict(
            sorted(class_freq.items(), key=lambda x: x[1], reverse=True)
        ),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("\nReport saved to:", OUTPUT_JSON)


if __name__ == "__main__":
    analyze_dataset()
