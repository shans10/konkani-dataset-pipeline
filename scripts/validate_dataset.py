"""
This script validates the extracted words from the dataset against the ground truth text files. It performs the following steps:
1. Loads the extracted words and their corresponding image paths from the labels.csv file.
2. Normalizes the extracted text and tokenizes it to ensure consistent comparison.
3. Extracts the document name from the image path to associate each word with its source document.
4. Loads the ground truth text for each document and creates a vocabulary set for validation.
5. Compares each extracted word against the corresponding document's vocabulary in a vectorized manner for efficiency.
6. Marks each word as a match or mismatch and saves the results in a new CSV file for further analysis and review.
"""

import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils.sorting import sort_by_hierarchy

# ==============================
# ROOT
# ==============================
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "output"))

RAW_CSV = os.path.join(OUTPUT_ROOT, "labels.csv")
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "labels_validated.csv")
TEXT_ROOT = os.path.join(DATA_ROOT, "text_data")


# ==============================
# NORMALIZATION
# ==============================
def normalize(text):
    return unicodedata.normalize("NFC", str(text).strip())


def tokenize(text):
    text = re.sub(r"[^\w\u0900-\u097F\s]", " ", text)
    return [normalize(w) for w in text.split() if w.strip()]


# ==============================
# LOAD RAW CSV
# ==============================
if not os.path.exists(RAW_CSV):
    print(f"❌ labels.csv not found: {RAW_CSV}")
    exit()

print("📄 Loading labels.csv...")
raw_df = pd.read_csv(RAW_CSV)
raw_df["extracted_text"] = raw_df["extracted_text"].apply(normalize)

# ==============================
# INCREMENTAL LOGIC
# ==============================
if os.path.exists(OUTPUT_CSV):
    existing_df = pd.read_csv(OUTPUT_CSV)
    processed_paths = set(existing_df["image_path"])
    df = raw_df[~raw_df["image_path"].isin(processed_paths)].copy()
    print(f"New words to validate: {len(df)}")
else:
    existing_df = None
    df = raw_df.copy()

if df.empty:
    print("No new data to validate.")
    exit()


# ==============================
# Extract document name
# ==============================
def extract_document(path):
    parts = Path(path).parts
    if len(parts) >= 2:
        return parts[1]
    return None


df["document"] = df["image_path"].apply(extract_document)
df["compared_word"] = ""
df["match"] = "NO"

# ==============================
# LOAD GT VOCABS
# ==============================
doc_vocabs = {}

print("📚 Loading ground truth vocabularies...")

for doc in tqdm(df["document"].dropna().unique()):
    gt_path = os.path.join(TEXT_ROOT, f"{doc}.txt")

    if not os.path.exists(gt_path):
        doc_vocabs[doc] = set()
        continue

    with open(gt_path, "r", encoding="utf-8") as f:
        doc_vocabs[doc] = set(tokenize(f.read()))

# ==============================
# VALIDATION
# ==============================
print("🔍 Validating words...")

for doc, group_index in tqdm(df.groupby("document").groups.items()):
    vocab = doc_vocabs.get(doc, set())
    if not vocab:
        continue

    words = df.loc[group_index, "extracted_text"]
    match_mask = words.isin(vocab)

    matched_indices = group_index[match_mask]

    df.loc[matched_indices, "match"] = "YES"
    df.loc[matched_indices, "compared_word"] = df.loc[matched_indices, "extracted_text"]

# ==============================
# MERGE + SORT + SAVE
# ==============================
if existing_df is not None:
    df = pd.concat([existing_df, df], ignore_index=True)

df = sort_by_hierarchy(df, path_column="image_path")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n✅ Validation complete.")
print(f"📄 Saved at: {OUTPUT_CSV}")
