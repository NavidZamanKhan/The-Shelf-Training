"""
fetch_api_data.py
-------------------
API Data Fetcher for 'The Shelf' ML training pipeline.

Fetches manga and anime metadata from public REST APIs (Jikan API v4 / Kitsu API fallback):
  1. Manga / Light Novels (~70% target: ~490 entries)
  2. Anime (~30% target: ~210 entries)

Applies:
  - Cross-dataset title deduplication against existing Goodreads datasets
  - Text cleaning (stripping MAL/Kitsu rewrite notes, whitespace normalization)
  - Rate limiting & automatic failover for high reliability

Outputs directly to `datasets/processed/jikan_anime_manga.csv` with schema (text, shelf_label="Anime & Manga").
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
OUTPUT_CSV_PATH = PROCESSED_DIR / "jikan_anime_manga.csv"

MERGED_CSV_PATH = PROCESSED_DIR / "goodreads_merged.csv"
PRIMARY_CSV_PATH = PROCESSED_DIR / "goodreads_labeled.csv"
SUPPLEMENTAL_CSV_PATH = PROCESSED_DIR / "goodreads_scraped_supplemental.csv"

# API Endpoints
JIKAN_MANGA_URL = "https://api.jikan.moe/v4/manga"
JIKAN_ANIME_URL = "https://api.jikan.moe/v4/anime"

KITSU_MANGA_URL = "https://kitsu.io/api/edge/manga"
KITSU_ANIME_URL = "https://kitsu.io/api/edge/anime"

SHELF_LABEL = "Anime & Manga"


def normalize_string(text: str) -> str:
    """Normalizes string for robust title matching (alphanumeric lowercase)."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_existing_titles() -> Set[str]:
    """Loads normalized titles and text signatures from existing Goodreads datasets to prevent cross-dataset duplication."""
    existing_normalized: Set[str] = set()
    csv_paths = [MERGED_CSV_PATH, PRIMARY_CSV_PATH, SUPPLEMENTAL_CSV_PATH]

    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                logger.info(f"Loading existing dataset for deduplication: {path.name} ({len(df)} rows)")
                for text_val in df["text"].dropna():
                    clean_t = str(text_val).strip()
                    norm = normalize_string(clean_t[:100])
                    if norm:
                        existing_normalized.add(norm)
            except Exception as e:
                logger.warning(f"Could not load {path} for deduplication: {e}")

    logger.info(f"Loaded {len(existing_normalized)} existing normalized text signatures for deduplication.")
    return existing_normalized


def clean_synopsis(synopsis_raw: str) -> str:
    """Cleans synopsis text by removing MAL/Kitsu editorial notes and extra whitespace."""
    if not synopsis_raw:
        return ""

    text = str(synopsis_raw)
    text = re.sub(r"\[Written by MAL Rewrite\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Source: [^\)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Source: [^\]]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_kitsu_data(
    endpoint_url: str,
    target_count: int,
    media_type: str,
    existing_signatures: Set[str]
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """
    Fetches items from Kitsu REST API as primary/fallback source.
    """
    logger.info(f"Fetching {media_type} via Kitsu API (Target: ~{target_count} entries)...")
    results: List[Dict[str, str]] = []
    seen_local_titles: Set[str] = set()

    skipped_cross_duplicates = 0
    skipped_short_synopsis = 0
    retry_count = 0
    offset = 0
    limit = 20

    headers = {"Accept": "application/vnd.api+json"}

    while len(results) < target_count and offset <= 1000:
        params = {
            "page[limit]": limit,
            "page[offset]": offset,
            "sort": "-userCount"
        }

        response = None
        for attempt in range(1, 4):
            try:
                response = requests.get(endpoint_url, params=params, headers=headers, timeout=10)
                if response.status_code in [429, 503, 504]:
                    logger.warning(f"Kitsu HTTP {response.status_code} at offset {offset}. Retrying in 4s...")
                    retry_count += 1
                    time.sleep(4.0)
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Kitsu request error at offset {offset} (attempt {attempt}): {e}")
                retry_count += 1
                time.sleep(4.0)

        if response is None or response.status_code != 200:
            logger.error(f"Failed to fetch Kitsu offset {offset} for {media_type}. Skipping offset.")
            offset += limit
            time.sleep(1.0)
            continue

        data = response.json()
        items = data.get("data", [])
        if not items:
            logger.info(f"No more Kitsu items returned for {media_type} at offset {offset}.")
            break

        for item in items:
            attributes = item.get("attributes", {})
            title_canonical = attributes.get("canonicalTitle", "")
            titles_dict = attributes.get("titles", {}) or {}
            title_english = titles_dict.get("en") or titles_dict.get("en_jp") or title_canonical
            synopsis_raw = attributes.get("synopsis", "") or ""

            cleaned_syn = clean_synopsis(synopsis_raw)

            if len(cleaned_syn) < 50:
                skipped_short_synopsis += 1
                continue

            norm_canon = normalize_string(title_canonical)
            norm_eng = normalize_string(title_english)

            # Check cross-dataset deduplication against existing Goodreads data
            is_cross_dup = False
            for sig in [norm_canon, norm_eng]:
                if sig and any(sig in existing_sig for existing_sig in existing_signatures if len(sig) > 5):
                    is_cross_dup = True
                    break

            if is_cross_dup:
                logger.info(f"Skipping cross-dataset duplicate: '{title_english}'")
                skipped_cross_duplicates += 1
                continue

            if norm_canon in seen_local_titles or norm_eng in seen_local_titles:
                continue

            seen_local_titles.add(norm_canon)
            seen_local_titles.add(norm_eng)

            combined_text = f"{title_english}. {cleaned_syn}"
            results.append({
                "text": combined_text,
                "shelf_label": SHELF_LABEL
            })

            if len(results) >= target_count:
                break

        offset += limit
        time.sleep(0.5)

    logger.info(f"Completed Kitsu {media_type} fetch: {len(results)} items collected.")
    return results, skipped_cross_duplicates, skipped_short_synopsis, retry_count


def fetch_jikan_data(
    endpoint_url: str,
    target_count: int,
    media_type: str,
    existing_signatures: Set[str]
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """
    Fetches items from Jikan API v4.
    """
    logger.info(f"Fetching {media_type} via Jikan API (Target: ~{target_count} entries)...")
    results: List[Dict[str, str]] = []
    seen_local_titles: Set[str] = set()

    skipped_cross_duplicates = 0
    skipped_short_synopsis = 0
    retry_count = 0
    page = 1

    while len(results) < target_count and page <= 30:
        params = {
            "order_by": "popularity",
            "sfw": "true",
            "limit": 25,
            "page": page
        }

        response = None
        for attempt in range(1, 4):
            try:
                logger.info(f"Fetching Jikan {media_type} page {page} (attempt {attempt})...")
                response = requests.get(endpoint_url, params=params, timeout=10)

                if response.status_code in [429, 503, 504]:
                    logger.warning(f"Jikan HTTP {response.status_code} on page {page}. Retrying in 4s...")
                    retry_count += 1
                    time.sleep(4.0)
                    continue

                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Jikan request error on page {page} (attempt {attempt}): {e}")
                retry_count += 1
                time.sleep(4.0)

        if response is None or response.status_code != 200:
            logger.error(f"Jikan page {page} failed with status {getattr(response, 'status_code', None)}.")
            break

        data = response.json()
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            title_default = item.get("title", "")
            title_english = item.get("title_english", "") or title_default
            synopsis_raw = item.get("synopsis", "") or ""

            cleaned_syn = clean_synopsis(synopsis_raw)

            if len(cleaned_syn) < 50:
                skipped_short_synopsis += 1
                continue

            norm_def = normalize_string(title_default)
            norm_eng = normalize_string(title_english)

            is_cross_dup = False
            for sig in [norm_def, norm_eng]:
                if sig and any(sig in existing_sig for existing_sig in existing_signatures if len(sig) > 5):
                    is_cross_dup = True
                    break

            if is_cross_dup:
                logger.info(f"Skipping cross-dataset duplicate: '{title_english}'")
                skipped_cross_duplicates += 1
                continue

            if norm_def in seen_local_titles or norm_eng in seen_local_titles:
                continue

            seen_local_titles.add(norm_def)
            seen_local_titles.add(norm_eng)

            combined_text = f"{title_english}. {cleaned_syn}"
            results.append({
                "text": combined_text,
                "shelf_label": SHELF_LABEL
            })

            if len(results) >= target_count:
                break

        page += 1
        time.sleep(1.0)

    return results, skipped_cross_duplicates, skipped_short_synopsis, retry_count


def main() -> None:
    """Main execution function for Anime & Manga dataset collection."""
    start_time = time.time()
    logger.info("Starting Anime & Manga API data collection...")

    existing_signatures = load_existing_titles()

    # Attempt Jikan API first; if Jikan times out (HTTP 504), automatically use Kitsu API
    manga_items, m_cross_dup, m_short, m_retries = fetch_jikan_data(
        JIKAN_MANGA_URL, target_count=490, media_type="Manga", existing_signatures=existing_signatures
    )

    if len(manga_items) < 100:
        logger.warning("Jikan API unavailable/timing out. Failing over to Kitsu REST API for Manga...")
        manga_items, m_cross_dup, m_short, m_retries = fetch_kitsu_data(
            KITSU_MANGA_URL, target_count=490, media_type="Manga", existing_signatures=existing_signatures
        )

    anime_items, a_cross_dup, a_short, a_retries = fetch_jikan_data(
        JIKAN_ANIME_URL, target_count=210, media_type="Anime", existing_signatures=existing_signatures
    )

    if len(anime_items) < 50:
        logger.warning("Jikan API unavailable/timing out. Failing over to Kitsu REST API for Anime...")
        anime_items, a_cross_dup, a_short, a_retries = fetch_kitsu_data(
            KITSU_ANIME_URL, target_count=210, media_type="Anime", existing_signatures=existing_signatures
        )

    all_items = manga_items + anime_items
    total_cross_dup = m_cross_dup + a_cross_dup
    total_short = m_short + a_short
    total_retries = m_retries + a_retries

    df_output = pd.DataFrame(all_items)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_CSV_PATH, index=False)

    elapsed_seconds = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print("                 ANIME & MANGA API FETCH SUMMARY")
    print("=" * 70)
    print(f"Total Rows Achieved:               {len(df_output)} (Manga: {len(manga_items)}, Anime: {len(anime_items)})")
    print(f"Cross-Dataset Duplicates Skipped:  {total_cross_dup}")
    print(f"Short/Empty Synopses Skipped:      {total_short}")
    print(f"HTTP 429/503/504 Retries/Failovers: {total_retries}")
    print(f"Total Execution Time:              {elapsed_seconds} seconds")
    print(f"Saved Output to:                   {OUTPUT_CSV_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
