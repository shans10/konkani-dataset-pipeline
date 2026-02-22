"""
This script validates the extracted words from Tesseract OCR against the ground truth text files for each document. It performs the following steps:
1. Load the raw CSV containing the extracted words.
2. Load the ground truth text files for each document.
3. Normalize and tokenize the text.
4. Compare the extracted words with the ground truth vocabularies.
5. Calculate the accuracy of the OCR extraction.
6. Save the validation results to a new CSV file.
"""

import os
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

# ---------- ROOT ----------
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "output"))

RAW_CSV = os.path.join(OUTPUT_ROOT, "raw", "labels.csv")
VALIDATED_ROOT = os.path.join(OUTPUT_ROOT, "validated")
TEXT_ROOT = os.path.join(DATA_ROOT, "text_data")

os.makedirs(VALIDATED_ROOT, exist_ok=True)

OUTPUT_CSV = os.path.join(VALIDATED_ROOT, "labels_validated.csv")


# ---------- NORMALIZATION ----------
def normalize(text):
    return unicodedata.normalize("NFC", str(text).strip())


def tokenize(text):
    text = re.sub(r"[^\w\u0900-\u097F\s]", " ", text)
    return [normalize(w) for w in text.split() if w.strip()]


# ---------- LOAD RAW CSV ----------
if not os.path.exists(RAW_CSV):
    print(f"❌ Raw CSV not found: {RAW_CSV}")
    exit()

df = pd.read_csv(RAW_CSV)
df["extracted_text"] = df["extracted_text"].apply(normalize)

# Extract document name from image_path
df["document"] = df["image_path"].apply(lambda p: p.split(os.sep)[0])

# ---------- LOAD GT VOCABS ----------
doc_vocabs = {}

print("📚 Loading ground truth vocabularies...")

for doc in tqdm(df["document"].unique()):
    gt_path = os.path.join(TEXT_ROOT, f"{doc}.txt")

    if not os.path.exists(gt_path):
        print(f"⚠ Missing GT for {doc}")
        doc_vocabs[doc] = set()
        continue

    with open(gt_path, "r", encoding="utf-8") as f:
        doc_vocabs[doc] = set(tokenize(f.read()))

# ---------- VALIDATE ----------
matches = 0
total = len(df)

print("🔍 Validating words...")

for idx, row in df.iterrows():
    vocab = doc_vocabs.get(row["document"], set())
    word = row["extracted_text"]

    if word in vocab:
        df.at[idx, "compared_word"] = word
        df.at[idx, "match"] = "YES"
        matches += 1
    else:
        df.at[idx, "compared_word"] = ""
        df.at[idx, "match"] = "NO"

accuracy = matches / total if total else 0

print(f"\n📊 Validation Summary")
print(f"Total words: {total}")
print(f"Matched: {matches}")
print(f"Accuracy: {accuracy:.2%}")

# ---------- SAVE ----------
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ Validation complete.")
print(f"📄 Saved at: {OUTPUT_CSV}")
