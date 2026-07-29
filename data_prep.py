"""
data_prep.py
------------
Data cleaning and label mapping script for 'The Shelf' ML training pipeline.
Processes raw Goodreads book details and maps genre tags to target shelf categories.
"""

import ast
from pathlib import Path
import pandas as pd

# Constants
BASE_DIR = Path(__file__).parent
RAW_CSV_PATH = BASE_DIR / "datasets" / "raw" / "good_reads" / "Book_Details.csv"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
PROCESSED_CSV_PATH = PROCESSED_DIR / "goodreads_labeled.csv"

# Priority-ordered shelf mapping list
SHELF_MAPPING = [
    ("Fantasy", ["Fantasy", "High Fantasy", "Urban Fantasy"]),
    ("Horror", ["Horror", "Supernatural"]),
    ("Romance", ["Romance", "Contemporary Romance", "Paranormal Romance", "Chick Lit", "New Adult"]),
    ("Science Fiction", ["Science Fiction", "Science Fiction Fantasy", "Dystopia"]),
    ("Mystery", ["Mystery", "Mystery Thriller", "Suspense"]),
    ("Thriller", ["Thriller", "Crime"]),
    ("Historical Fiction", ["Historical Fiction", "Historical", "War"]),
    ("Classic Literature", ["Classics", "Literary Fiction", "British Literature", "American"]),
    ("Biography & Memoir", ["Biography", "Memoir"]),
    ("History", ["History"]),
    ("Self-Help & Personal Development", ["Self Help", "Psychology"]),
    ("Religion & Spirituality", ["Religion", "Christian", "Spirituality", "Queer", "LGBT"]),
    ("Philosophy", ["Philosophy"]),
    ("Poetry", ["Poetry"]),
    ("Humor", ["Humor"]),
    ("Graphic Novels", ["Graphic Novels"]),
]


def map_genres_to_shelf(genres_raw) -> str:
    """
    Parses genres string and returns the first matching shelf category based on priority order.
    """
    if pd.isna(genres_raw):
        return "Miscellaneous"

    try:
        if isinstance(genres_raw, list):
            genres_list = genres_raw
        else:
            genres_list = ast.literal_eval(str(genres_raw))
        if not isinstance(genres_list, list):
            genres_list = []
    except Exception:
        genres_list = []

    book_tags_set = set(genres_list)

    for shelf_name, target_tags in SHELF_MAPPING:
        if any(tag in book_tags_set for tag in target_tags):
            return shelf_name

    return "Miscellaneous"


def main():
    print(f"Loading raw dataset from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    total_initial_rows = len(df)

    # 1. Map genres to shelf label
    df["shelf_label"] = df["genres"].apply(map_genres_to_shelf)

    # 2. Build text column by concatenating title and details safely
    title_series = df["book_title"].fillna("").astype(str).str.strip()
    details_series = df["book_details"].fillna("").astype(str).str.strip()

    df["text"] = title_series + " " + details_series
    df["text"] = df["text"].str.strip()

    # 3. Drop empty/null text rows
    df_clean = df[df["text"].str.len() > 0].copy()
    total_final_rows = len(df_clean)

    # 4. Save result
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_df = df_clean[["text", "shelf_label"]]
    output_df.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_CSV_PATH}")
    print()

    # 5. Print summary
    print("--- DATA PROCESSING SUMMARY ---")
    print(f"Total rows before cleaning: {total_initial_rows}")
    print(f"Total rows after cleaning:  {total_final_rows}")
    print()
    print("SHELF DISTRIBUTION (Label: Count):")
    counts = df_clean["shelf_label"].value_counts()
    for label, count in counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
