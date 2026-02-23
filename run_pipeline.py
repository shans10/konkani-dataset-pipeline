"""
This script orchestrates the entire OCR dataset pipeline, including:
1. Building the dataset by extracting word-level images and labels from PDFs.
2. Validating the extracted dataset through manual review.
3. Finalizing the dataset by cleaning, normalizing, and categorizing the data.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ---------- PROJECT ROOT ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# ---------- DEFAULT PATHS ----------
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser(description="OCR Dataset Pipeline")

parser.add_argument(
    "--full",
    action="store_true",
    help="Full rebuild (delete output and rebuild)",
)

parser.add_argument(
    "--versioned",
    action="store_true",
    help="Create versioned output folder",
)

parser.add_argument(
    "--yes",
    action="store_true",
    help="Skip overwrite confirmation",
)

args = parser.parse_args()

# ---------- VERSION HANDLING ----------
if args.versioned:
    version_name = datetime.now().strftime("dataset_%Y%m%d_%H%M%S")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output_versions", version_name)
    print(f"\n📦 Using versioned output folder: {OUTPUT_ROOT}")

# Ensure output directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------- FULL REBUILD ----------
if args.full:
    if os.path.exists(OUTPUT_ROOT):
        if not args.yes:
            confirm = input(f"\n⚠ This will delete {OUTPUT_ROOT}. Continue? (y/n): ")
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        shutil.rmtree(OUTPUT_ROOT)
        os.makedirs(OUTPUT_ROOT)
        print("🗑 Output folder cleared.")

# ---------- ENVIRONMENT VARIABLES ----------
PIPELINE_ENV = {
    **os.environ,
    "PROJECT_ROOT": PROJECT_ROOT,
    "DATA_ROOT": DATA_ROOT,
    "OUTPUT_ROOT": OUTPUT_ROOT,
}


# ---------- SCRIPT RUNNER ----------
def run_script(script_filename: str) -> None:

    script_path = os.path.join(SCRIPTS_DIR, script_filename)

    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    print(f"\n🚀 Running {script_filename}...\n")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        env=PIPELINE_ENV,
    )

    if result.returncode != 0:
        print(f"\n❌ {script_filename} failed. Stopping pipeline.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n✅ {script_filename} completed in {elapsed:.2f} seconds.")


# ---------- PIPELINE ----------
pipeline_start = time.time()

run_script("build_dataset.py")
run_script("validate_dataset.py")
run_script("finalize_dataset.py")

total_time = time.time() - pipeline_start

print("\n🎉 Pipeline completed successfully.")
print(f"⏱ Total runtime: {total_time / 60:.2f} minutes.")
