"""
data_prep.py
------------
Data cleaning, label mapping, and dataset merging script for 'The Shelf' ML training pipeline.
Processes raw Goodreads book details, maps genre tags to target shelf categories,
and merges primary labeled data with scraped supplemental data.
"""

import ast
import re
from pathlib import Path
import pandas as pd

# Constants
BASE_DIR = Path(__file__).parent
RAW_CSV_PATH = BASE_DIR / "datasets" / "raw" / "good_reads" / "Book_Details.csv"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
PRIMARY_CSV_PATH = PROCESSED_DIR / "goodreads_labeled.csv"
SUPPLEMENTAL_CSV_PATH = PROCESSED_DIR / "goodreads_scraped_supplemental.csv"
JIKAN_CSV_PATH = PROCESSED_DIR / "jikan_anime_manga.csv"
ROKOMARI_CSV_PATH = PROCESSED_DIR / "rokomari_bangla.csv"
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

# High-precision text fallback patterns for books with missing or generic Goodreads tags
TEXT_FALLBACK_PATTERNS = [
    (
        "Biography & Memoir",
        re.compile(
            r"\b(biography|autobiography|memoir)s?\b|\b(life and (career|times) of|chronicles the life of|story of the life of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Romance",
        re.compile(
            r"\b(a\s+)?(heartwarming\s+|passionate\s+|sweeping\s+)?romance\b|\b(romance|romantic)\s+(novel|fiction|story|comedy|thriller|relationship)s?\b|\b(love affair|love story|falling in love)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Humor",
        re.compile(
            r"\b(humorous|hilarious|satirical)\b|\b(comedy|humor)\s+(novel|story|fiction|book|memoir)s?\b|\b(witty|laugh-out-loud)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Science Fiction",
        re.compile(
            r"\b(science\s+fiction|sci-fi)(\s+novel|\s+story|\s+adventure)?\b|\b(interplanetary|space\s+vehicle|starship|spacecraft|space\s+opera|alien\s+invasion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Mystery",
        re.compile(
            r"\b(murder\s+mystery|detective\s+(novel|story)|whodunit|crime\s+thriller)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Historical Fiction",
        re.compile(
            r"\b(historical\s+fiction|historical\s+novel)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Fantasy",
        re.compile(
            r"\b(fantasy\s+novel|epic\s+fantasy|urban\s+fantasy|dark\s+fantasy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Horror",
        re.compile(
            r"\b(horror\s+novel|gothic\s+horror|haunted\s+house|supernatural\s+horror)\b",
            re.IGNORECASE,
        ),
    ),
]


def map_genres_to_shelf(genres_raw, text_raw: str = "") -> str:
    """Parses genres string and returns the first matching shelf category based on priority order.
    If tag-based matching returns Miscellaneous, attempts text-based pattern matching on description text.
    """
    genres_list = []
    if not pd.isna(genres_raw):
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

    # Fallback to high-precision text pattern scanning if tag-based mapping yielded Miscellaneous
    if text_raw and isinstance(text_raw, str):
        for shelf_name, pattern in TEXT_FALLBACK_PATTERNS:
            if pattern.search(text_raw):
                return shelf_name

    return "Miscellaneous"


def build_primary_dataset() -> pd.DataFrame:
    """Cleans raw Goodreads CSV and creates primary labeled dataset."""
    print(f"Loading raw dataset from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    total_initial_rows = len(df)

    title_series = df["book_title"].fillna("").astype(str).str.strip()
    details_series = df["book_details"].fillna("").astype(str).str.strip()

    df["text"] = title_series + " " + details_series
    df["text"] = df["text"].str.strip()

    df["shelf_label"] = df.apply(
        lambda row: map_genres_to_shelf(row["genres"], row["text"]), axis=1
    )

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

    if ROKOMARI_CSV_PATH.exists():
        df_rokomari = pd.read_csv(ROKOMARI_CSV_PATH)
        dataframes_to_merge.append(df_rokomari)
        print(f"Rokomari Bangla rows: {len(df_rokomari)}")

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
