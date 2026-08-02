"""
scrape_wafilife.py
------------------
Bangla book scraper for Wafilife.com using Playwright.

Scrapes metadata across all 17 target shelf categories:
  - Subject URL category mapping
  - Full pagination discovery
  - Single-pass Category 74 split (Anime & Manga vs Graphic Novels)
  - Stateful checkpointing (`datasets/processed/wafilife_state.json`)
  - Progressive CSV writing (`datasets/processed/wafilife_bangla.csv`)
  - Politeness delays and circuit breaker error recovery
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

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
OUTPUT_CSV_PATH = PROCESSED_DIR / "wafilife_bangla.csv"
STATE_FILE_PATH = PROCESSED_DIR / "wafilife_state.json"

# Wafilife Category Targets (Target Shelf -> Category URL)
CATEGORY_TARGETS = [
    ("Science Fiction", "https://www.wafilife.com/cat/books/subject/science-fiction-novel"),
    ("Horror", "https://www.wafilife.com/cat/books/subject/supernatural-and-horror-novels"),
    ("Mystery", "https://www.wafilife.com/cat/books/subject/mystery-and-detective-novels"),
    ("Romance", "https://www.wafilife.com/cat/books/subject/shahitto-o-uponnash"),
    ("Fantasy", "https://www.wafilife.com/cat/books/subject/parapsychological-novels"),
    ("History", "https://www.wafilife.com/cat/books/subject/history-and-traditions"),
    ("Historical Fiction", "https://www.wafilife.com/cat/books/subject/ঐতিহাসিক-উপন্যাস"),
    ("Biography & Memoir", "https://www.wafilife.com/cat/books/subject/biographies-memories-interviews"),
    ("Poetry", "https://www.wafilife.com/cat/books/subject/chora-kobita-o-abritti"),
    ("Philosophy", "https://www.wafilife.com/cat/books/subject/দর্শন-বিষয়ক-বই"),
    ("Humor", "https://www.wafilife.com/cat/books/subject/rommo-uponyash"),
    ("Religion & Spirituality", "https://www.wafilife.com/cat/books/subject/islamic-books"),
    ("Self-Help & Personal Development", "https://www.wafilife.com/cat/books/subject/self-help-motivational-and-meditation"),
    ("Classic Literature", "https://www.wafilife.com/cat/books/subject/চিরায়ত-উপন্যাস"),
    ("Miscellaneous", "https://www.wafilife.com/cat/books/subject/প্রবন্ধ"),
    ("Sequential Art Category 74", "https://www.wafilife.com/cat/books/subject/comics-noksha-o-chobir-golpo"),
]

# Category 74 Split Terms for Anime & Manga
MANGA_GENERAL_TERMS = [
    "মাঙ্গা", "মানহোয়া", "লাইট নভেল", "অ্যানিমে", "জাপানি", "জাপানিজ", "মানহুয়া", "ওতাকু"
]

MANGA_FRANCHISE_MARKERS = [
    "ডেথ নোট", "নারুতো", "সোলো লেভেলিং", "ডোরেমন", "ওয়ান পিস", "অ্যাটাক অন টাইটান",
    "জুযুতসু কাইসেন", "মাই হিরো একাডেমিয়া", "টোকিও ঘুল", "ড্রাগন বল", "পোকেমন",
    "কিমএতসু নো ইয়াইবা", "ডিমন স্লেয়ার", "ব্লীচ", "বারসার্ক", "হান্টার"
]


def navigate_with_retry(page, url: str, retries: int = 3, timeout: int = 30000) -> bool:
    """Navigates to URL with retries on socket/interruption errors."""
    for attempt in range(1, retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(1500)
            return True
        except Exception as e:
            logger.warning(f"Navigation attempt {attempt}/{retries} failed for Wafilife URL {url}: {e}")
            time.sleep(2.0 * attempt)
    return False


def is_manga_entry(title: str, description: str) -> bool:
    """Evaluates title and description for general manga/anime terms and franchise markers."""
    combined = f"{title} {description}".lower()
    for term in MANGA_GENERAL_TERMS:
        if term in combined:
            return True
    for marker in MANGA_FRANCHISE_MARKERS:
        if marker in combined:
            return True
    return False


def clean_text(raw_text: str) -> str:
    """Cleans scraped Bangla text by collapsing whitespace and stripping UI headers/footers."""
    if not raw_text:
        return ""
    text = str(raw_text)
    text = re.sub(r"Wafilife makes Islamic shopping easy.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"To reach the highest traffic.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Make your online shop easier.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_state() -> Dict[str, Any]:
    """Loads scraping checkpoint state from disk."""
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading state file {STATE_FILE_PATH}: {e}")
    return {
        "completed_categories": [],
        "completed_pages": {},
        "scraped_urls": [],
        "seen_titles": []
    }


def save_state(state: Dict[str, Any]):
    """Saves scraping checkpoint state to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_records_to_csv(records: List[Dict[str, str]]):
    """Appends scraped records incrementally to CSV."""
    if not records:
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(records)
    write_header = not OUTPUT_CSV_PATH.exists()
    df_new.to_csv(OUTPUT_CSV_PATH, mode="a", index=False, header=write_header, encoding="utf-8")


def scrape_wafilife_pipeline(max_pages_per_cat: Optional[int] = None, dry_run: bool = False):
    start_time = time.time()
    logger.info(f"Starting Wafilife Bangla scraping pipeline (max_pages_per_cat={max_pages_per_cat}, dry_run={dry_run})...")

    state = load_state()
    scraped_urls_set: Set[str] = set(state.get("scraped_urls", []))
    seen_titles_set: Set[str] = set(state.get("seen_titles", []))
    completed_pages_dict: Dict[str, int] = state.get("completed_pages", {})

    target_categories = CATEGORY_TARGETS
    if dry_run:
        target_categories = [
            ("Science Fiction", "https://www.wafilife.com/cat/books/subject/science-fiction-novel"),
            ("Philosophy", "https://www.wafilife.com/cat/books/subject/দর্শন-বিষয়ক-বই")
        ]

    consecutive_errors = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for shelf_label, cat_url in target_categories:
            if shelf_label in state.get("completed_categories", []) and not dry_run:
                logger.info(f"Skipping already completed category: '{shelf_label}'")
                continue

            logger.info(f"\n=======================================================")
            logger.info(f"   STARTING WAFILIFE CATEGORY: '{shelf_label}'")
            logger.info(f"=======================================================")

            start_page = completed_pages_dict.get(shelf_label, 0) + 1
            current_page = start_page
            category_new_records = 0

            while True:
                if max_pages_per_cat and current_page > (start_page + max_pages_per_cat - 1):
                    logger.info(f"Reached page limit ({max_pages_per_cat}) for category '{shelf_label}'.")
                    break

                page_url = f"{cat_url}/page/{current_page}" if current_page > 1 else cat_url
                logger.info(f"Navigating to listing page {current_page}: {page_url}")

                page_records: List[Dict[str, str]] = []

                if not navigate_with_retry(page, page_url):
                    logger.error(f"Failed to navigate to Wafilife listing page {page_url} after retries.")
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.error("5 consecutive errors on Wafilife listing page. Pausing for 60 seconds...")
                        time.sleep(60)
                        consecutive_errors = 0
                    current_page += 1
                    continue

                try:
                    soup = BeautifulSoup(page.content(), "html.parser")

                    # Extract detail links (/pd/ URLs)
                    book_links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if "/pd/" in href:
                            full_href = href if href.startswith("http") else f"https://www.wafilife.com{href}"
                            book_links.append(full_href)

                    book_links = list(dict.fromkeys(book_links))
                    logger.info(f"Page {current_page}: Found {len(book_links)} product links.")

                    if not book_links:
                        logger.info(f"No more books found for category '{shelf_label}' at page {current_page}. Reached end.")
                        break

                    consecutive_errors = 0

                    for book_url in book_links:
                        if book_url in scraped_urls_set:
                            continue

                        if not navigate_with_retry(page, book_url):
                            scraped_urls_set.add(book_url)
                            consecutive_errors += 1
                            if consecutive_errors >= 5:
                                logger.error("5 consecutive detail errors on Wafilife. Pausing for 60s...")
                                time.sleep(60)
                                consecutive_errors = 0
                            continue

                        try:
                            detail_soup = BeautifulSoup(page.content(), "html.parser")

                            # Extract Title
                            h1_tag = detail_soup.find("h1")
                            title_text = h1_tag.get_text(strip=True) if h1_tag else ""
                            title_text = re.sub(r"\| Wafilife\.com.*", "", title_text).strip()

                            norm_title = re.sub(r"[^a-z0-9\u0980-\u09ff]", "", title_text.lower())
                            if not norm_title or norm_title in seen_titles_set:
                                scraped_urls_set.add(book_url)
                                continue

                            # Extract Synopsis
                            synopsis_divs = detail_soup.find_all("div", class_=lambda c: c and "text-brand-two" in str(c))
                            synopsis_raw = ""
                            for d in synopsis_divs:
                                txt = d.get_text(strip=True)
                                if len(txt) > len(synopsis_raw):
                                    synopsis_raw = txt

                            if not synopsis_raw:
                                body_text = detail_soup.get_text(strip=True)
                                synopsis_raw = body_text[:500]

                            cleaned_syn = clean_text(synopsis_raw)

                            if len(cleaned_syn) < 30:
                                scraped_urls_set.add(book_url)
                                continue

                            full_text = f"{title_text}. {cleaned_syn}"

                            # Determine Label
                            if shelf_label == "Sequential Art Category 74":
                                assigned_label = "Anime & Manga" if is_manga_entry(title_text, cleaned_syn) else "Graphic Novels"
                            else:
                                assigned_label = shelf_label

                            seen_titles_set.add(norm_title)
                            scraped_urls_set.add(book_url)

                            record = {
                                "title": title_text,
                                "text": full_text,
                                "shelf_label": assigned_label,
                                "source": "wafilife",
                                "url": book_url
                            }
                            page_records.append(record)
                            category_new_records += 1

                            logger.info(f"  + Scraped Wafilife: [{assigned_label}] {title_text[:40]}...")
                            consecutive_errors = 0

                        except Exception as e:
                            logger.warning(f"Error parsing Wafilife detail page {book_url}: {e}")

                        time.sleep(1.0)

                except Exception as e:
                    logger.error(f"Error parsing Wafilife listing page {page_url}: {e}")

                # Append records for this listing page to CSV & save state
                if page_records:
                    append_records_to_csv(page_records)

                completed_pages_dict[shelf_label] = current_page
                state["completed_pages"] = completed_pages_dict
                state["scraped_urls"] = list(scraped_urls_set)
                state["seen_titles"] = list(seen_titles_set)
                save_state(state)

                current_page += 1

            if not dry_run:
                if shelf_label not in state.get("completed_categories", []):
                    state.setdefault("completed_categories", []).append(shelf_label)
                    save_state(state)

            logger.info(f"Wafilife Category '{shelf_label}' complete! Added {category_new_records} new records.")

        browser.close()

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Wafilife scrape run finished in {elapsed}s.")


if __name__ == "__main__":
    scrape_wafilife_pipeline()
