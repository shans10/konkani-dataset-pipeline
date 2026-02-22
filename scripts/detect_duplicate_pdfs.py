"""
This script detects duplicate PDF files in the data/pdfs directory.
It computes the SHA256 hash of each PDF and identifies duplicates based on matching hashes.
"""

import hashlib
import os

from tqdm import tqdm

# ---------- ROOT ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
PDF_ROOT = os.path.join(DATA_ROOT, "pdfs")


def file_hash(path, chunk_size=8192):
    """Compute SHA256 hash of a file (memory safe)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ---------- FIND PDF FILES ----------
if not os.path.exists(PDF_ROOT):
    print(f"❌ PDF directory not found: {PDF_ROOT}")
    exit()

pdf_files = sorted([f for f in os.listdir(PDF_ROOT) if f.lower().endswith(".pdf")])

print(f"Found {len(pdf_files)} PDF files.\n")

hash_map = {}
duplicates = {}

# ---------- PROCESS ----------
for pdf in tqdm(pdf_files, desc="Hashing PDFs"):
    pdf_path = os.path.join(PDF_ROOT, pdf)

    try:
        h = file_hash(pdf_path)
    except Exception as e:
        print(f"⚠ Error reading {pdf}: {e}")
        continue

    file_size = os.path.getsize(pdf_path)

    if h in hash_map:
        duplicates.setdefault(h, []).append((pdf, file_size))
    else:
        hash_map[h] = (pdf, file_size)

# ---------- REPORT ----------
print("\n🔍 Duplicate PDFs Report\n")

if not duplicates:
    print("✅ No duplicate PDFs detected.")
else:
    for h, dup_list in duplicates.items():
        original_pdf, original_size = hash_map[h]

        print(f"Original: {original_pdf} ({original_size / 1024:.2f} KB)")

        for dup_pdf, dup_size in dup_list:
            print(f"   Duplicate: {dup_pdf} ({dup_size / 1024:.2f} KB)")

        print("-" * 50)

print("\nDone.")
