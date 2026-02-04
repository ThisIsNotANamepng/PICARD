# Prints how many 0s and 1s in the label column there are in the combined human dataset data/combined_human_dataset.csv

import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/combined_human_dataset.csv")

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH.resolve()}")
        exit(1)

    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        print("Column 'label' not found in the dataset.")
        exit(1)

    counts = df["label"].value_counts().sort_index()
    print("Label counts:")
    for val, cnt in counts.items():
        print(f"  {val}: {cnt}")
