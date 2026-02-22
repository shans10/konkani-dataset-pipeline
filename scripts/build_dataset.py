"""
This script processes PDF documents to extract word-level images and their corresponding text labels. It performs the following steps:
1. Extract pages from PDF documents.
2. Process each page to extract words using OCR.
3. Save each word as a separate image.
4. Generate a CSV file containing the image paths and extracted text.
The script is designed to be memory efficient and supports resuming from where it left off in case of interruptions.
"""

import glob
import os
from multiprocessing import Pool, cpu_count

import cv2
import fitz
import pandas as pd
import pytesseract
from tqdm import tqdm

# ---------- ROOT ----------
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(PROJECT_ROOT, "data"),
)

OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    os.path.join(PROJECT_ROOT, "output"),
)

PDF_ROOT = os.path.join(DATA_ROOT, "pdfs")
PAGE_ROOT = os.path.join(DATA_ROOT, "page_images")
IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")
RAW_ROOT = os.path.join(OUTPUT_ROOT, "raw")

os.makedirs(PAGE_ROOT, exist_ok=True)
os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(RAW_ROOT, exist_ok=True)

CSV_PATH = os.path.join(RAW_ROOT, "labels.csv")

# ---------- INIT CSV IF NOT EXISTS ----------
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["image_path", "extracted_text"]).to_csv(
        CSV_PATH, index=False, encoding="utf-8-sig"
    )


# ---------- AUTO PAGE EXTRACTION ----------
def extract_pages(pdf_path, doc_name):
    doc = fitz.open(pdf_path)
    out_dir = os.path.join(PAGE_ROOT, doc_name)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        pix.save(os.path.join(out_dir, f"page_{i + 1:03d}.png"))


# ---------- PAGE PROCESSOR ----------
def process_document(doc_name):

    pdf_path = os.path.join(PDF_ROOT, f"{doc_name}.pdf")
    page_dir = os.path.join(PAGE_ROOT, doc_name)
    doc_output = os.path.join(IMAGE_ROOT, doc_name)

    os.makedirs(doc_output, exist_ok=True)

    # Auto extract pages
    if not os.path.exists(page_dir) or not os.listdir(page_dir):
        if os.path.exists(pdf_path):
            extract_pages(pdf_path, doc_name)
        else:
            return

    image_paths = sorted(
        glob.glob(os.path.join(page_dir, "page_*.png")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )

    for img_path in tqdm(image_paths, desc=doc_name, leave=False):
        page_num = int(os.path.basename(img_path).split("_")[1].split(".")[0])

        # ---------- RESUME PER PAGE ----------
        expected_prefix = f"p{page_num:03d}_"
        existing_files = [
            f for f in os.listdir(doc_output) if f.startswith(expected_prefix)
        ]

        if existing_files:
            continue  # page already processed

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        data = pytesseract.image_to_data(
            thresh, lang="hin+eng", output_type=pytesseract.Output.DATAFRAME
        )

        data = data.dropna(subset=["text"])
        data = data[data.conf > 90]
        data = data.sort_values(by=["block_num", "par_num", "line_num", "word_num"])

        pad = 3
        h_img, w_img = img.shape[:2]
        word_id = 1

        page_rows = []

        for _, row in data.iterrows():
            word = str(row["text"]).strip()
            if not word:
                continue

            x, y, w, h = int(row.left), int(row.top), int(row.width), int(row.height)

            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, w_img)
            y2 = min(y + h + pad, h_img)

            crop = img[y1:y2, x1:x2]

            filename = f"p{page_num:03d}_w{word_id:05d}.png"
            save_path = os.path.join(doc_output, filename)

            cv2.imwrite(save_path, crop)

            rel_path = os.path.relpath(save_path, IMAGE_ROOT)
            page_rows.append([rel_path, word])

            word_id += 1

        # ---------- MEMORY SAFE APPEND ----------
        if page_rows:
            pd.DataFrame(page_rows, columns=["image_path", "extracted_text"]).to_csv(
                CSV_PATH, mode="a", header=False, index=False, encoding="utf-8-sig"
            )


# ---------- FIND DOCUMENTS ----------
pdf_files = sorted([f for f in os.listdir(PDF_ROOT) if f.endswith(".pdf")])
documents = [os.path.splitext(f)[0] for f in pdf_files]

print(f"Found {len(documents)} documents.")

# ---------- MULTIPROCESS ----------
with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap_unordered(process_document, documents),
            total=len(documents),
            desc="Documents",
        )
    )

print("\n✅ Build stage completed.")
