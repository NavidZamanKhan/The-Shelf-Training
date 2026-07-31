"""
data_prep.py
------------
Data cleaning, label mapping, and dataset merging script for 'The Shelf' ML training pipeline.
Processes raw Goodreads book details, maps genre tags to target shelf categories,
and merges primary labeled data with scraped supplemental data.
"""

import ast
from pathlib import Path
import pandas as pd

# Constants
BASE_DIR = Path(__file__).parent
RAW_CSV_PATH = BASE_DIR / "datasets" / "raw" / "good_reads" / "Book_Details.csv"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
PRIMARY_CSV_PATH = PROCESSED_DIR / "goodreads_labeled.csv"
SUPPLEMENTAL_CSV_PATH = PROCESSED_DIR / "goodreads_scraped_supplemental.csv"
JIKAN_CSV_PATH = PROCESSED_DIR / "jikan_anime_manga.csv"
MERGED_CSV_PATH = PROCESSED_DIR / "goodreads_merged.csv"

# Priority-ordered shelf mapping list
SHELF_MAPPING = [
    ("Fantasy", ["Fantasy", "High Fantasy", "Urban Fantasy"]),
    ("Horror", ["Horror", "Supernatural"]),
    ("Romance", ["Romance", "Contemporary Romance", "Paranormal Romance", "Chick Lit", "New Adult"]),
    ("Science Fiction", ["Science Fiction", "Science Fiction Fantasy", "Dystopia"]),
    ("Mystery", ["Mystery", "Mystery Thriller", "Suspense", "Thriller", "Crime"]),
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
    ("Anime & Manga", ["Manga", "Anime", "Light Novel"]),
]


def map_genres_to_shelf(genres_raw) -> str:
    """Parses genres string and returns the first matching shelf category based on priority order."""
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


def build_primary_dataset() -> pd.DataFrame:
    """Cleans raw Goodreads CSV and creates primary labeled dataset."""
    print(f"Loading raw dataset from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    total_initial_rows = len(df)

    df["shelf_label"] = df["genres"].apply(map_genres_to_shelf)
    title_series = df["book_title"].fillna("").astype(str).str.strip()
    details_series = df["book_details"].fillna("").astype(str).str.strip()

    df["text"] = title_series + " " + details_series
    df["text"] = df["text"].str.strip()

    df_clean = df[df["text"].str.len() > 0].copy()
    output_df = df_clean[["text", "shelf_label"]]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(PRIMARY_CSV_PATH, index=False)
    print(f"Saved primary labeled dataset ({len(output_df)} rows) to {PRIMARY_CSV_PATH}")
    return output_df


def merge_datasets():
    """Merges primary labeled dataset, supplemental scraped dataset, and Jikan anime/manga dataset into goodreads_merged.csv."""
    # Always rebuild primary dataset to apply updated SHELF_MAPPING
    df_primary = build_primary_dataset()
    dataframes_to_merge = [df_primary]

    print(f"\n--- MERGING DATASETS ---")
    print(f"Primary rows:      {len(df_primary)}")

    if SUPPLEMENTAL_CSV_PATH.exists():
        df_supp = pd.read_csv(SUPPLEMENTAL_CSV_PATH)
        df_supp["shelf_label"] = df_supp["shelf_label"].replace({"Thriller": "Mystery"})
        dataframes_to_merge.append(df_supp)
        print(f"Supplemental rows: {len(df_supp)}")

    if JIKAN_CSV_PATH.exists():
        df_jikan = pd.read_csv(JIKAN_CSV_PATH)
        dataframes_to_merge.append(df_jikan)
        print(f"Anime & Manga rows: {len(df_jikan)}")

    df_combined = pd.concat(dataframes_to_merge, ignore_index=True)
    total_combined_rows = len(df_combined)

    # Deduplicate text column
    df_merged = df_combined.drop_duplicates(subset=["text"], keep="first").copy()
    total_merged_rows = len(df_merged)
    duplicates_dropped = total_combined_rows - total_merged_rows

    df_merged.to_csv(MERGED_CSV_PATH, index=False)

    print(f"Duplicates dropped during merge: {duplicates_dropped}")
    print(f"Final merged dataset rows:      {total_merged_rows}")
    print(f"Saved merged dataset to:        {MERGED_CSV_PATH}\n")
    print("MERGED SHELF DISTRIBUTION (Label: Count):")
    counts = df_merged["shelf_label"].value_counts()
    for label, count in counts.items():
        print(f"  - {label:<35}: {count}")


if __name__ == "__main__":
    merge_datasets()
