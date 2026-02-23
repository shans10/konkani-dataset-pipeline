import re
from pathlib import Path

import pandas as pd


def sort_by_hierarchy(df, path_column="image_path"):
    """
    Sort dataframe numerically by:
    pdf_number → page_number → word_number

    Expected path format:
    images/pdf_1/page_1/word_1.png
    english/pdf_1/page_1/word_1.png
    """

    def extract_numbers(path):
        parts = Path(path).parts

        # Handle both:
        # images/pdf_1/page_1/word_1.png
        # english/pdf_1/page_1/word_1.png
        # konkani/pdf_1/page_1/word_1.png

        if len(parts) < 4:
            return pd.Series([-1, -1, -1])

        pdf_num = int(re.search(r"\d+", parts[1]).group())
        page_num = int(re.search(r"\d+", parts[2]).group())
        word_num = int(re.search(r"\d+", parts[3]).group())

        return pd.Series([pdf_num, page_num, word_num])

    df[["_pdf", "_page", "_word"]] = df[path_column].apply(extract_numbers)

    df = df.sort_values(by=["_pdf", "_page", "_word"]).reset_index(drop=True)

    return df.drop(columns=["_pdf", "_page", "_word"])
