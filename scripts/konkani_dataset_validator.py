import os
import cv2
import glob
import base64
import asyncio
import aiohttp
import unicodedata
import pandas as pd

from tqdm import tqdm
from paddleocr import PaddleOCR


# ---------------- CONFIG ----------------

DATASET_ROOT = "dataset_konkani"

BATCH_SIZE = 256
CONF_THRESHOLD = 0.90
AUTOSAVE_INTERVAL = 20000

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"
PROMPT = "Read the text in this image and output only the exact word."

os.makedirs("validated_csvs", exist_ok=True)
os.makedirs("final_clean_csvs", exist_ok=True)


# ---------------- OCR ----------------

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="hi",
    det=False,
    rec=True,
    use_gpu=True,
    rec_batch_num=BATCH_SIZE,
    use_angle_cls=False,
    show_log=False
)

print("OCR ready")


# ---------------- TEXT NORMALIZATION ----------------

def normalize(text):

    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text).strip()


# ---------------- OCR BATCH ----------------

def recognize_batch(images):

    results = ocr.ocr(images, det=False, cls=False)

    preds = []
    confs = []

    for r in results:

        if r is None or isinstance(r, float):
            preds.append("")
            confs.append(0.0)
            continue

        try:
            text = r[0]
            conf = float(r[1])
        except:
            text = ""
            conf = 0.0

        preds.append(normalize(text))
        confs.append(conf)

    return preds, confs


# ---------------- VLM ----------------

async def run_qwen(session, path):

    try:

        with open(path, "rb") as f:
            img = base64.b64encode(f.read()).decode()

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": PROMPT,
            "images": [img],
            "stream": False
        }

        async with session.post(OLLAMA_URL, json=payload) as resp:

            data = await resp.json()

            return normalize(data.get("response", ""))

    except:
        return ""


async def validate_vlm(rows):

    async with aiohttp.ClientSession() as session:

        tasks = []

        for idx, path in rows:
            tasks.append(run_qwen(session, path))

        results = await asyncio.gather(*tasks)

        return results


# ---------------- MAIN ----------------

csv_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "*.csv")))

print("Found CSVs:", csv_files)


for csv_path in csv_files:

    print("\nProcessing:", csv_path)

    name = os.path.basename(csv_path)
    validated_path = os.path.join("validated_csvs", name)

    # ---------- LOAD CSV OR RESUME ----------

    if os.path.exists(validated_path):

        print("Resuming previous run")

        df = pd.read_csv(validated_path)

        if "Matched" not in df.columns:
            df["Matched"] = ""

        start_idx = df["Matched"].last_valid_index()

        if start_idx is None:
            start_idx = 0
        else:
            start_idx += 1

    else:

        df = pd.read_csv(csv_path)

        df["ocr_text"] = ""
        df["confidence"] = 0.0
        df["Matched"] = ""

        start_idx = 0


    images = []
    idxs = []
    gts = []

    uncertain = []

    for i in tqdm(range(start_idx, len(df))):

        row = df.iloc[i]

        path = os.path.join(DATASET_ROOT, row["image_path"])
        gt = normalize(row["ground_truth"])

        img = cv2.imread(path)

        if img is None:
            continue

        images.append(img)
        idxs.append(i)
        gts.append(gt)

        if len(images) < BATCH_SIZE:
            continue

        preds, confs = recognize_batch(images)

        for j, idx in enumerate(idxs):

            pred = preds[j]
            conf = confs[j]
            gt = gts[j]

            df.at[idx, "ocr_text"] = pred
            df.at[idx, "confidence"] = conf

            if pred == gt and conf >= CONF_THRESHOLD:

                df.at[idx, "Matched"] = "Yes"

            else:

                df.at[idx, "Matched"] = "Pending"

                uncertain.append(
                    (idx, os.path.join(DATASET_ROOT, df.iloc[idx]["image_path"]))
                )

        images.clear()
        idxs.clear()
        gts.clear()

        # ---------- AUTOSAVE ----------

        if i % AUTOSAVE_INTERVAL == 0 and i != 0:

            print(f"\nAutosaving progress at row {i}")

            df.to_csv(validated_path, index=False)


    # ---------- FINAL SAVE AFTER OCR ----------

    print("\nSaving OCR results...")

    df.to_csv(validated_path, index=False)


    # ---------- VLM VALIDATION ----------

    print("Running async VLM validation...")

    vlm_results = asyncio.run(validate_vlm(uncertain))

    for k, (idx, _) in enumerate(uncertain):

        vlm = vlm_results[k]
        gt = normalize(df.iloc[idx]["ground_truth"])

        if vlm == gt:
            df.at[idx, "Matched"] = "Yes"
        else:
            df.at[idx, "Matched"] = "No"


    # ---------- SAVE FINAL RESULTS ----------

    df.to_csv(validated_path, index=False)

    clean_df = df[df["Matched"] == "Yes"][["image_path", "ground_truth"]]

    clean_path = os.path.join("final_clean_csvs", name)

    clean_df.to_csv(clean_path, index=False)

    print("Clean dataset saved:", clean_path)


print("\nValidation finished.")
