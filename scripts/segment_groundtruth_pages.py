"""
This script segments ground truth text files into individual page text files.

It reads all .txt files in the TEXT_ROOT directory, identifies page numbers using a regex pattern,
and splits each file into separate page text files in the PAGE_TEXT_ROOT directory.
"""

import os
import re
from pathlib import Path

# ---------- ROOT ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
TEXT_ROOT = os.path.join(DATA_ROOT, "text")  # full GT txt files
PAGE_TEXT_ROOT = os.path.join(DATA_ROOT, "page_text")  # per-page output

Path(PAGE_TEXT_ROOT).mkdir(parents=True, exist_ok=True)

# ---------- PAGE NUMBER REGEX ----------
PAGE_RE = re.compile(
    r"(?im)^[ \t]*page[ \t\:\-]*?(\d{1,4})[ \t]*$|^[ \t]*(\d{1,4})[ \t]*$",
    re.MULTILINE,
)


def split_pages(text, min_chars_between=200):
    matches = []

    # Find candidate page number lines
    for m in PAGE_RE.finditer(text):
        num = m.group(1) or m.group(2)
        if num:
            matches.append((m.start(), int(num)))

    if not matches:
        return [(-1, text)]

    # Remove OCR-noise close detections
    filtered = []
    last = -999999

    for pos, num in matches:
        if pos - last > min_chars_between:
            filtered.append((pos, num))
            last = pos

    pages = []

    for i, (pos, num) in enumerate(filtered):
        start = text.find("\n", pos)
        start = start + 1 if start != -1 else pos

        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        body = text[start:end].strip()

        pages.append((num, body))

    return pages


# =========================
# PROCESS ALL TEXT FILES
# =========================
txt_files = sorted([f for f in os.listdir(TEXT_ROOT) if f.endswith(".txt")])

print(f"Found {len(txt_files)} ground truth text files.\n")

for file in txt_files:
    filepath = os.path.join(TEXT_ROOT, file)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pages = split_pages(text)

    doc_name = Path(file).stem
    doc_output_dir = os.path.join(PAGE_TEXT_ROOT, doc_name)
    Path(doc_output_dir).mkdir(parents=True, exist_ok=True)

    for num, content in pages:
        if num == -1:
            name = "full_text.txt"
        else:
            name = f"page_{num}.txt"

        outpath = os.path.join(doc_output_dir, name)

        with open(outpath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved: {outpath}")

print("\n✅ Ground truth pages segmented successfully.")
