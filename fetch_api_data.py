"""
fetch_api_data.py
-------------------
API Data Fetcher for 'The Shelf' ML training pipeline.

This module retrieves book and manga metadata using free public APIs:
  1. Open Library API (https://openlibrary.org/developers/api)
  2. Jikan API / MyAnimeList (https://jikan.moe/)

NOTE:
    - Prefer API fetching over HTML web scraping whenever an official API is available.
    - Web scraping (for non-API sites like Goodreads) is handled in `scrape_data.py`.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = Path(__file__).parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "api_books.json"

OPEN_LIBRARY_SUBJECT_URL = "https://openlibrary.org/subjects/{subject}.json"
JIKAN_MANGA_SEARCH_URL = "https://api.jikan.moe/v4/manga"


def fetch_open_library_data(subjects: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches book metadata for given subject categories from Open Library API.
    
    Args:
        subjects (List[str]): List of subjects/genres (e.g., ['science_fiction', 'history', 'technology']).
        limit (int): Number of works to fetch per subject.
        
    Returns:
        List[Dict[str, Any]]: Processed book records containing title, authors, subject tags, description, etc.
    """
    logger.info("Fetching data from Open Library API...")
    results: List[Dict[str, Any]] = []

    for subject in subjects:
        url = OPEN_LIBRARY_SUBJECT_URL.format(subject=subject.lower())
        params = {"limit": limit}
        
        try:
            logger.info(f"Querying Open Library subject: '{subject}'")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            works = data.get("works", [])
            for work in works:
                # TODO: Enrich description if needed via Open Library Work API endpoint (/works/OL...json)
                title = work.get("title", "")
                authors = [a.get("name", "") for a in work.get("authors", [])]
                subject_tags = work.get("subject", [])
                
                # Combine title and subjects into a rich text snippet for text classification
                combined_text = f"{title}. " + " ".join(subject_tags[:10])
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "description": combined_text,
                    "subject": subject,
                    "shelf_label": subject.capitalize(),
                    "source": "open_library_api"
                })
                
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Open Library data for subject '{subject}': {e}")

        # Respect API rate limits
        time.sleep(1)

    return results


def fetch_jikan_manga_data(query_genres: List[str], limit: int = 25) -> List[Dict[str, Any]]:
    """
    Fetches manga / light novel metadata from Jikan (MyAnimeList unofficial API).
    
    Args:
        query_genres (List[str]): List of genre search queries (e.g., ['Sci-Fi', 'Fantasy', 'Slice of Life']).
        limit (int): Max results per search.
        
    Returns:
        List[Dict[str, Any]]: Processed manga records containing title, synopsis, genres, etc.
    """
    logger.info("Fetching data from Jikan (MyAnimeList) API...")
    results: List[Dict[str, Any]] = []

    for genre in query_genres:
        params = {
            "q": genre,
            "limit": limit,
            "order_by": "popularity"
        }
        
        try:
            logger.info(f"Querying Jikan Manga search for genre/keyword: '{genre}'")
            response = requests.get(JIKAN_MANGA_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            manga_list = data.get("data", [])
            for item in manga_list:
                title = item.get("title", "")
                synopsis = item.get("synopsis", "") or ""
                genres = [g.get("name", "") for g in item.get("genres", [])]
                
                results.append({
                    "title": title,
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "description": f"{title}. {synopsis}",
                    "subject": genre,
                    "shelf_label": "Manga & Light Novels",
                    "source": "jikan_mal_api"
                })
                
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Jikan data for genre '{genre}': {e}")

        # Jikan rate limit rule: max 3 requests per second
        time.sleep(1)

    return results


def main() -> None:
    """
    Main entry point for API data fetching.
    """
    logger.info("Starting API data collection...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch from Open Library
    ol_subjects = ["science_fiction", "history", "computer_science", "business"]
    ol_data = fetch_open_library_data(ol_subjects, limit=20)
    logger.info(f"Retrieved {len(ol_data)} items from Open Library API.")

    # 2. Fetch from Jikan API (MyAnimeList)
    jikan_genres = ["Sci-Fi", "Fantasy", "Mystery"]
    jikan_data = fetch_jikan_manga_data(jikan_genres, limit=20)
    logger.info(f"Retrieved {len(jikan_data)} items from Jikan API.")

    # Combine and save dataset
    combined_dataset = ol_data + jikan_data
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"API data fetch completed. Saved total {len(combined_dataset)} items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
