"""
This script finalizes the dataset by:
1. Loading the validated CSV.
2. Normalizing the extracted text.
3. Classifying entries into English and Konkani.
4. Copying images into a new structured directory.
5. Saving separate CSV files for English and Konkani.
6. Generating a markdown report with dataset statistics and integrity check.
"""

import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from utils.sorting import sort_by_hierarchy

# ==============================
# ROOT CONFIG
# ==============================
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "output"))

VALIDATED_CSV = os.path.join(OUTPUT_ROOT, "labels_validated.csv")
IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")

FINAL_ROOT = os.path.join(OUTPUT_ROOT, "final_dataset")

ENGLISH_ROOT = os.path.join(FINAL_ROOT, "english")
KONKANI_ROOT = os.path.join(FINAL_ROOT, "konkani")

os.makedirs(ENGLISH_ROOT, exist_ok=True)
os.makedirs(KONKANI_ROOT, exist_ok=True)

ENGLISH_CSV = os.path.join(FINAL_ROOT, "dataset_english.csv")
KONKANI_CSV = os.path.join(FINAL_ROOT, "dataset_konkani.csv")
REPORT_MD = os.path.join(FINAL_ROOT, "dataset_report.md")

# ==============================
# LOAD DATA
# ==============================
if not os.path.exists(VALIDATED_CSV):
    print("❌ labels_validated.csv not found.")
    exit()

print("📄 Loading validated CSV...")
df = pd.read_csv(VALIDATED_CSV)

total_rows = len(df)

matched_df = df[df["match"] == "YES"].copy()
unmatched_count = total_rows - len(matched_df)

print(f"Matched rows: {len(matched_df)}")
print(f"Unmatched rows ignored: {unmatched_count}")

# ==============================
# INCREMENTAL FILTER
# ==============================
processed_paths = set()

if os.path.exists(ENGLISH_CSV):
    processed_paths.update(pd.read_csv(ENGLISH_CSV)["image_path"])

if os.path.exists(KONKANI_CSV):
    processed_paths.update(pd.read_csv(KONKANI_CSV)["image_path"])


def convert_to_final_path(rel_path, category):
    parts = Path(rel_path).parts
    return os.path.join(category, *parts[1:])


matched_df = matched_df[
    ~matched_df.apply(
        lambda row: (
            convert_to_final_path(row["image_path"], "english") in processed_paths
            or convert_to_final_path(row["image_path"], "konkani") in processed_paths
        ),
        axis=1,
    )
]

# ==============================
# NORMALIZATION
# ==============================
def normalize(text):
    return unicodedata.normalize("NFC", str(text).strip())


matched_df["extracted_text"] = matched_df["extracted_text"].apply(normalize)

# ==============================
# REMOVE PURE PUNCTUATION / SYMBOL TOKENS
# ==============================
valid_token_pattern = re.compile(r"[A-Za-z0-9\u0900-\u097F]")

matched_df = matched_df[
    matched_df["extracted_text"].apply(lambda w: bool(valid_token_pattern.search(w)))
]

# ==============================
# CLASSIFICATION
# ==============================
def classify(word):

    if re.fullmatch(r"[A-Za-z]+", word):
        return "english"

    if re.fullmatch(r"[0-9]+", word):
        return "english"

    if re.fullmatch(r"[0-9]+(st|nd|rd|th)", word, re.IGNORECASE):
        return "english"

    if re.fullmatch(r"[\u0966-\u096F]+", word):
        return "konkani"

    return "konkani"


matched_df["category"] = matched_df["extracted_text"].apply(classify)

english_df = matched_df[matched_df["category"] == "english"]
konkani_df = matched_df[matched_df["category"] == "konkani"]

# ==============================
# COPY FUNCTION
# ==============================
def copy_dataset(df_subset, target_root):

    updated_paths = []

    for _, row in df_subset.iterrows():
        rel_path = row["image_path"]
        src = os.path.join(OUTPUT_ROOT, rel_path)

        parts = Path(rel_path).parts
        relative_structure = os.path.join(*parts[1:])
        dst = os.path.join(target_root, relative_structure)

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(src):
            shutil.copy2(src, dst)

        new_rel = os.path.relpath(dst, FINAL_ROOT)
        updated_paths.append(new_rel)

    df_subset = df_subset.copy()
    df_subset["image_path"] = updated_paths
    return df_subset


print("📂 Copying images into final_dataset...")

english_df = copy_dataset(english_df, ENGLISH_ROOT)
konkani_df = copy_dataset(konkani_df, KONKANI_ROOT)

# ==============================
# SAVE CSV FILES (APPEND MODE)
# ==============================
def append_or_create(df_subset, path):

    if df_subset.empty:
        return 0

    # ✅ FIX: Rename BEFORE concat to align schema
    df_subset = df_subset[["image_path", "extracted_text"]].copy()
    df_subset.columns = ["image_path", "ground_truth"]

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[["image_path", "ground_truth"]]
        df_subset = pd.concat([existing, df_subset], ignore_index=True)

    df_subset = sort_by_hierarchy(df_subset, path_column="image_path")

    df_subset.to_csv(path, index=False, encoding="utf-8-sig")

    return len(df_subset)


n_english = append_or_create(english_df, ENGLISH_CSV)
n_konkani = append_or_create(konkani_df, KONKANI_CSV)

# ==============================
# UNIQUE WORD STATS
# ==============================
final_english_df = (
    pd.read_csv(ENGLISH_CSV) if os.path.exists(ENGLISH_CSV) else pd.DataFrame()
)
final_konkani_df = (
    pd.read_csv(KONKANI_CSV) if os.path.exists(KONKANI_CSV) else pd.DataFrame()
)

unique_english = (
    len(set(final_english_df["ground_truth"])) if not final_english_df.empty else 0
)
unique_konkani = (
    len(set(final_konkani_df["ground_truth"])) if not final_konkani_df.empty else 0
)

# ==============================
# INTEGRITY CHECK
# ==============================
image_count = 0
for root, _, files in os.walk(FINAL_ROOT):
    image_count += len([f for f in files if f.endswith(".png")])

csv_total = len(final_english_df) + len(final_konkani_df)
integrity_status = "PASS" if image_count == csv_total else "CHECK"

# ==============================
# REPORT
# ==============================
report = f"""
# Final Dataset Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Validation Summary

Total validated entries: {total_rows}
Matched kept: {len(df[df["match"] == "YES"])}
Unmatched ignored: {unmatched_count}

---

## Final Dataset Breakdown

English: {len(final_english_df)}
Konkani: {len(final_konkani_df)}
Total final size: {csv_total}

---

## Vocabulary Statistics

Unique English words: {unique_english}
Unique Konkani words: {unique_konkani}

---

## Integrity Check

Images inside final_dataset: {image_count}
CSV total entries: {csv_total}
Status: {integrity_status}

---

Note: Original images remain untouched in output/images/.
"""

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write(report)

print("\n✅ Final dataset updated incrementally.")
print(f"📁 Location: {FINAL_ROOT}")
print(f"🔒 Integrity: {integrity_status}")
