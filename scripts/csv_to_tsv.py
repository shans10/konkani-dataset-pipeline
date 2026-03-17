import csv
import os

# Input CSV files
csv_files = [
    "dataset_konkani/dataset_konkani_1.csv",
    "dataset_konkani/dataset_konkani_2.csv",
    "dataset_konkani/dataset_konkani_3.csv"
]

# Output TSV file
output_tsv = "dataset_konkani/dataset_konkani.tsv"

header_written = False

with open(output_tsv, "w", newline="", encoding="utf-8") as tsv_out:
    writer = csv.writer(tsv_out, delimiter="\t")

    for csv_file in csv_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # read header

            # Write header only once
            if not header_written:
                writer.writerow(header)
                header_written = True

            # Write rows
            for row in reader:
                writer.writerow(row)

print(f"TSV file created: {output_tsv}")
