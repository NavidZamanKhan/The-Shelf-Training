"""
merge_bangla_scrapes.py
-----------------------
Merges, cleans, and cross-deduplicates scraped Bangla book datasets from Rokomari and Wafilife.

Steps:
  1. Reads `datasets/processed/rokomari_bangla.csv` and `datasets/processed/wafilife_bangla.csv`.
  2. Normalizes titles and removes exact/near duplicates between Rokomari & Wafilife (retaining the record with longer plot synopsis).
  3. Checks overlap with `datasets/processed/goodreads_merged.csv` to ensure cross-dataset integrity.
  4. Saves unified dataset to `datasets/processed/rokomari_wafilife_bangla.csv`.
  5. Outputs category breakdown and dedup metrics.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Set

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"

ROKOMARI_CSV = PROCESSED_DIR / "rokomari_bangla.csv"
WAFILIFE_CSV = PROCESSED_DIR / "wafilife_bangla.csv"
GOODREADS_CSV = PROCESSED_DIR / "goodreads_merged.csv"
OUTPUT_MERGED_CSV = PROCESSED_DIR / "rokomari_wafilife_bangla.csv"


def normalize_title(raw_title: str) -> str:
    """Normalizes Bangla title for cross-source matching."""
    if not raw_title:
        return ""
    text = str(raw_title).lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"১ম সংস্করণ|২য় সংস্করণ|হার্ডকভার|পেপারব্যাক", "", text)
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]", "", text)
    return text.strip()


def merge_and_deduplicate():
    logger.info("Starting Bangla dataset merge and cross-source deduplication...")

    dfs = []
    if ROKOMARI_CSV.exists():
        df_roko = pd.read_csv(ROKOMARI_CSV, encoding="utf-8")
        logger.info(f"Loaded Rokomari dataset: {len(df_roko)} rows.")
        dfs.append(df_roko)
    else:
        logger.warning(f"Rokomari CSV not found at {ROKOMARI_CSV}")

    if WAFILIFE_CSV.exists():
        df_wafi = pd.read_csv(WAFILIFE_CSV, encoding="utf-8")
        logger.info(f"Loaded Wafilife dataset: {len(df_wafi)} rows.")
        dfs.append(df_wafi)
    else:
        logger.warning(f"Wafilife CSV not found at {WAFILIFE_CSV}")

    if not dfs:
        logger.error("No input CSV files found! Cannot merge.")
        return

    df_combined = pd.concat(dfs, ignore_index=True)
    total_raw = len(df_combined)

    # Ensure necessary columns exist
    if "title" not in df_combined.columns:
        df_combined["title"] = ""
    if "text" not in df_combined.columns:
        logger.error("No 'text' column found in combined dataset!")
        return

    # Clean text column
    df_combined = df_combined.dropna(subset=["text"])
    df_combined["title"] = df_combined["title"].fillna("")

    # Deduplicate based on normalized title or text snippet
    seen_map: Dict[str, pd.Series] = {}

    for idx, row in df_combined.iterrows():
        title = str(row.get("title", ""))
        norm_key = normalize_title(title)
        
        if not norm_key:
            norm_key = normalize_title(str(row["text"])[:40])

        if norm_key in seen_map:
            existing_row = seen_map[norm_key]
            if len(str(row["text"])) > len(str(existing_row["text"])):
                seen_map[norm_key] = row
        else:
            seen_map[norm_key] = row

    df_merged = pd.DataFrame(list(seen_map.values()))
    duplicates_removed = total_raw - len(df_merged)

    logger.info(f"Total raw items: {total_raw} | Duplicates removed: {duplicates_removed} | Clean items: {len(df_merged)}")

    # Check Goodreads overlap if present
    goodreads_matches = 0
    if GOODREADS_CSV.exists():
        try:
            df_gr = pd.read_csv(GOODREADS_CSV, encoding="utf-8")
            if "title" in df_gr.columns:
                gr_titles = set(df_gr["title"].dropna().apply(normalize_title))
                for idx, row in df_merged.iterrows():
                    if normalize_title(row.get("title", "")) in gr_titles:
                        goodreads_matches += 1
                logger.info(f"Goodreads title overlap found: {goodreads_matches} titles match existing Goodreads entries.")
        except Exception as e:
            logger.warning(f"Could not perform Goodreads overlap check: {e}")

    # Save final merged output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_MERGED_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("             ROKOMARI + WAFILIFE MERGED BANGLA DATASET SUMMARY")
    print("=" * 80)
    print(f"Total Raw Scraped Rows:             {total_raw}")
    print(f"Duplicates Removed (Cross-Source):   {duplicates_removed}")
    print(f"Goodreads Overlap Matches:          {goodreads_matches}")
    print(f"Final Clean Bangla Rows Achieved:   {len(df_merged)}")
    print(f"Output File Saved To:               {OUTPUT_MERGED_CSV}\n")

    print("CATEGORY DISTRIBUTION IN MERGED DATASET (shelf_label: Count):")
    counts = df_merged["shelf_label"].value_counts()
    for label, count in counts.items():
        print(f"  - {label:<35}: {count}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    merge_and_deduplicate()
