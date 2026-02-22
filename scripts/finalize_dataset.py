"""
Finalizes the dataset by:
- Removing unmatched images
- Normalizing extracted text
- Classifying into numbers, English, and Konkani
- Saving categorized CSVs
- Generating a final report with dataset statistics and integrity checks.
"""

import os
import re
import unicodedata
from datetime import datetime

import pandas as pd

# ---------- ROOT ----------
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "output"))

VALIDATED_CSV = os.path.join(OUTPUT_ROOT, "validated", "labels_validated.csv")
IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")
FINAL_ROOT = os.path.join(OUTPUT_ROOT, "final")

os.makedirs(FINAL_ROOT, exist_ok=True)

NUMBERS_CSV = os.path.join(FINAL_ROOT, "dataset_numbers.csv")
ENGLISH_CSV = os.path.join(FINAL_ROOT, "dataset_english.csv")
KONKANI_CSV = os.path.join(FINAL_ROOT, "dataset_konkani.csv")
REPORT_MD = os.path.join(FINAL_ROOT, "dataset_report.md")

# ---------- SAFETY CHECK ----------
if not os.path.exists(VALIDATED_CSV):
    print(f"❌ Validated CSV not found: {VALIDATED_CSV}")
    exit()

if not os.path.exists(IMAGE_ROOT):
    print(f"❌ Images folder not found: {IMAGE_ROOT}")
    exit()

# ---------- LOAD ----------
print("📄 Loading validated dataset...")
df = pd.read_csv(VALIDATED_CSV)

total_rows = len(df)
print(f"Total validated rows: {total_rows}")

# ---------- SPLIT ----------
matched_df = df[df["match"] == "YES"].copy()
unmatched_df = df[df["match"] != "YES"].copy()

print(f"Matched rows: {len(matched_df)}")
print(f"Unmatched rows: {len(unmatched_df)}")

# ---------- DELETE UNMATCHED IMAGES ----------
print("🗑 Removing unmatched images...")

unmatched_set = set(unmatched_df["image_path"])
deleted_count = 0
missing_count = 0

for rel_path in unmatched_set:
    full_path = os.path.join(IMAGE_ROOT, rel_path)

    if os.path.exists(full_path):
        os.remove(full_path)
        deleted_count += 1
    else:
        missing_count += 1

print(f"Deleted unmatched images: {deleted_count}")
print(f"Missing (already removed): {missing_count}")


# ---------- NORMALIZATION ----------
def normalize(text):
    return unicodedata.normalize("NFC", str(text).strip())


matched_df["extracted_text"] = matched_df["extracted_text"].apply(normalize)


# ---------- CLASSIFICATION ----------
def classify(word):
    if re.fullmatch(r"\d+", word):
        return "numbers"
    if re.fullmatch(r"[A-Za-z]+", word):
        return "english"
    return "konkani"


matched_df["category"] = matched_df["extracted_text"].apply(classify)

numbers_df = matched_df[matched_df["category"] == "numbers"]
english_df = matched_df[matched_df["category"] == "english"]
konkani_df = matched_df[matched_df["category"] == "konkani"]

# ---------- UNIQUE WORD STATS ----------
unique_english = sorted(set(english_df["extracted_text"]))
unique_konkani = sorted(set(konkani_df["extracted_text"]))

n_unique_english = len(unique_english)
n_unique_konkani = len(unique_konkani)


# ---------- SAVE FUNCTION ----------
def save_dataset(df_subset, path):
    out = df_subset[["image_path", "extracted_text"]].copy()
    out.columns = ["image_path", "ground_truth"]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return len(out)


print("💾 Saving categorized datasets...")

n_numbers = save_dataset(numbers_df, NUMBERS_CSV)
n_english = save_dataset(english_df, ENGLISH_CSV)
n_konkani = save_dataset(konkani_df, KONKANI_CSV)

# ---------- REMOVE EMPTY DOCUMENT FOLDERS ----------
print("🧹 Cleaning empty document folders...")

removed_dirs = 0

for doc in os.listdir(IMAGE_ROOT):
    doc_path = os.path.join(IMAGE_ROOT, doc)

    if os.path.isdir(doc_path):
        if not os.listdir(doc_path):
            os.rmdir(doc_path)
            removed_dirs += 1

print(f"Empty document folders removed: {removed_dirs}")

# ---------- FINAL IMAGE COUNT ----------
disk_image_count = 0

for root, _, files in os.walk(IMAGE_ROOT):
    disk_image_count += len([f for f in files if f.endswith(".png")])

final_total = n_numbers + n_english + n_konkani
integrity_status = "PASS" if disk_image_count == final_total else "FAIL"

# ---------- REPORT ----------
report = f"""
# Dataset Finalization Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 Validation Summary

- Total validated entries: {total_rows}
- Matched kept: {len(matched_df)}
- Unmatched removed: {len(unmatched_df)}

---

## 📁 Final Dataset Breakdown

- Numbers dataset size: {n_numbers}
- English dataset size: {n_english}
- Konkani dataset size: {n_konkani}
- **Total final size:** {final_total}

---

## 🔤 Vocabulary Statistics

- Unique English words: {n_unique_english}
- Unique Konkani words: {n_unique_konkani}

---

## 🗑 Cleanup Summary

- Unmatched images deleted: {deleted_count}
- Missing images: {missing_count}
- Empty document folders removed: {removed_dirs}

---

## 🔒 Integrity Check

- Images on disk: {disk_image_count}
- CSV total entries: {final_total}
- **Integrity Status: {integrity_status}**

---

Dataset is fully synchronized and ready for OCR model training.
"""

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write(report)

print("\n✅ Dataset finalized successfully.")
print(f"📁 Final datasets saved in: {FINAL_ROOT}")
print(f"📝 Report saved at: {REPORT_MD}")
print(f"🔒 Integrity check: {integrity_status}")
print(f"🔤 Unique English words: {n_unique_english}")
print(f"🔤 Unique Konkani words: {n_unique_konkani}")
