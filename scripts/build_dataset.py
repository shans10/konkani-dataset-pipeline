"""
Incremental OCR Dataset Builder
Parallelized Per Page (Stable + Smooth Progress)

Features:
- Page extraction if missing
- Word-level cropping
- Parallel per-page processing
- Global per-page tqdm progress
- True incremental CSV append
- No hanging on last PDF
"""

import glob
import os
import re
import multiprocessing as mp
from pathlib import Path

import cv2
import fitz
import pandas as pd
import pytesseract
from tqdm import tqdm
from utils.sorting import sort_by_hierarchy

# ==========================================================
# ROOT CONFIG
# ==========================================================
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

PDF_ROOT = os.path.join(DATA_ROOT, "pdfs")
PAGE_IMAGE_ROOT = os.path.join(DATA_ROOT, "page_images")
IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")

os.makedirs(IMAGE_ROOT, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_ROOT, "labels.csv")

# ==========================================================
# PERFORMANCE SETTINGS
# ==========================================================
MAX_WORKERS = max(1, mp.cpu_count() - 1)
CONF_THRESHOLD = 90
PAD = 3

os.environ["OMP_THREAD_LIMIT"] = "1"
TESS_CONFIG = "--psm 6"


# ==========================================================
# UTILITIES
# ==========================================================
def extract_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def extract_pages_if_missing(pdf_name):
    pdf_path = os.path.join(PDF_ROOT, f"{pdf_name}.pdf")
    page_dir = os.path.join(PAGE_IMAGE_ROOT, pdf_name)

    if os.path.exists(page_dir) and os.listdir(page_dir):
        return

    if not os.path.exists(pdf_path):
        print(f"⚠ PDF missing: {pdf_name}")
        return

    os.makedirs(page_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=300)
        pix.save(os.path.join(page_dir, f"page_{i+1}.png"))

    doc.close()


# ==========================================================
# PAGE WORKER
# ==========================================================
def process_page(args):
    pdf_name, page_path = args

    page_num = extract_number(os.path.basename(page_path))

    pdf_output_dir = os.path.join(IMAGE_ROOT, pdf_name)
    page_output_dir = os.path.join(pdf_output_dir, f"page_{page_num}")
    os.makedirs(page_output_dir, exist_ok=True)

    img = cv2.imread(page_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        lang="hin+eng",
        config=TESS_CONFIG,
        output_type=pytesseract.Output.DATAFRAME,
    )

    data = data.dropna(subset=["text"])
    data = data[data.conf > CONF_THRESHOLD]
    data = data.sort_values(
        by=["block_num", "par_num", "line_num", "word_num"]
    )

    h_img, w_img = img.shape[:2]
    word_id = 1
    rows = []

    for _, row in data.iterrows():
        word = str(row["text"]).strip()
        if not word:
            continue

        x, y, w, h = int(row.left), int(row.top), int(row.width), int(row.height)

        x1 = max(x - PAD, 0)
        y1 = max(y - PAD, 0)
        x2 = min(x + w + PAD, w_img)
        y2 = min(y + h + PAD, h_img)

        crop = img[y1:y2, x1:x2]

        filename = f"word_{word_id}.png"
        save_path = os.path.join(page_output_dir, filename)

        cv2.imwrite(save_path, crop)

        rel_path = os.path.relpath(save_path, OUTPUT_ROOT)
        rows.append([rel_path, word])

        word_id += 1

    return rows


# ==========================================================
# MAIN
# ==========================================================
def main():

    pdf_names = sorted(
        [
            Path(f).stem
            for f in os.listdir(PDF_ROOT)
            if f.lower().endswith(".pdf") and f.startswith("pdf_")
        ],
        key=extract_number,
    )

    print(f"Total PDFs found: {len(pdf_names)}")

    processed_docs = set()

    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        processed_docs = set(
            existing_df["image_path"].apply(
                lambda p: p.split(os.sep)[1] if os.sep in p else None
            )
        )
        print(f"Already processed docs: {len(processed_docs)}")

    new_pdfs = [p for p in pdf_names if p not in processed_docs]

    print(f"New PDFs to process: {len(new_pdfs)}")

    if not new_pdfs:
        print("Nothing to process.")
        return

    # ======================================================
    # PREPARE ALL PAGES
    # ======================================================
    all_pages = []

    for pdf_name in new_pdfs:
        extract_pages_if_missing(pdf_name)

        page_dir = os.path.join(PAGE_IMAGE_ROOT, pdf_name)

        page_images = sorted(
            glob.glob(os.path.join(page_dir, "page_*.png")),
            key=lambda x: extract_number(os.path.basename(x)),
        )

        for page_path in page_images:
            all_pages.append((pdf_name, page_path))

    print(f"Total pages to process: {len(all_pages)}")

    if not all_pages:
        return

    # ======================================================
    # MULTICORE PER PAGE
    # ======================================================
    mp.set_start_method("spawn", force=True)

    all_rows = []

    with mp.Pool(processes=MAX_WORKERS) as pool:
        for rows in tqdm(
            pool.imap_unordered(process_page, all_pages),
            total=len(all_pages),
            desc="Processing Pages",
        ):
            all_rows.extend(rows)

    if not all_rows:
        print("No words extracted.")
        return

    new_df = pd.DataFrame(
        all_rows, columns=["image_path", "extracted_text"]
    )

    new_df = sort_by_hierarchy(new_df, path_column="image_path")

    # ======================================================
    # APPEND CSV (TRUE INCREMENTAL)
    # ======================================================
    if os.path.exists(CSV_PATH):
        new_df.to_csv(
            CSV_PATH,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8-sig",
        )
    else:
        new_df.to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n✅ Incremental build complete.")
    print(f"New words added: {len(new_df)}")


if __name__ == "__main__":
    main()
