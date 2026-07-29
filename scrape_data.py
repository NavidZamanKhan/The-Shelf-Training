"""
scrape_data.py
----------------
Full two-tier Goodreads scraper for collecting supplemental training examples 
for underrepresented shelf categories ('Graphic Novels', 'Philosophy', 'Poetry', 'Thriller').

Features:
- Tier 1: Paginate shelf listing pages (pages 1-8).
- Tier 2: Visit detail pages to parse full book synopses via `div[data-testid="description"]`.
- Deduplication: Title-based skipping against `goodreads_labeled.csv` and `goodreads_scraped_supplemental.csv`.
- Incremental checkpointing: Appends rows to CSV after each book.
- Politeness delay: 3.0 to 4.5 seconds randomized jitter.
"""

import asyncio
import csv
import logging
import random
import re
import time
from pathlib import Path
from typing import Set

import pandas as pd
from playwright.async_api import async_playwright, BrowserContext
from playwright_stealth import Stealth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants & Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
PRIMARY_CSV_PATH = PROCESSED_DIR / "goodreads_labeled.csv"
SUPPLEMENTAL_CSV_PATH = PROCESSED_DIR / "goodreads_scraped_supplemental.csv"
GOODREADS_BASE_URL = "https://www.goodreads.com"

# Target Categories: (shelf_slug, target_shelf_label, max_pages)
TARGET_SHELVES = [
    ("graphic-novels", "Graphic Novels", 8),
    ("philosophy", "Philosophy", 8),
    ("poetry", "Poetry", 8),
    ("thriller", "Thriller", 8),
]


def normalize_title(title: str) -> str:
    """Normalizes title string by lowercasing, removing format tags, and stripping punctuation."""
    if not title:
        return ""
    # Remove trailing format tags like (Hardcover), (Paperback), etc.
    cleaned = re.sub(r"\s*\([^)]*\)$", "", title)
    cleaned = cleaned.lower().strip()
    return cleaned


def load_existing_titles() -> Set[str]:
    """Loads existing normalized book titles from primary and supplemental CSVs."""
    existing_titles: Set[str] = set()

    # 1. Load primary labeled CSV
    if PRIMARY_CSV_PATH.exists():
        try:
            df_primary = pd.read_csv(PRIMARY_CSV_PATH)
            if "text" in df_primary.columns:
                for text_val in df_primary["text"].dropna():
                    # Extract title prefix before description
                    first_sentence = str(text_val).split(". ")[0]
                    existing_titles.add(normalize_title(first_sentence))
            logger.info(f"Loaded {len(existing_titles)} existing title signatures from primary dataset.")
        except Exception as e:
            logger.warning(f"Error reading primary dataset {PRIMARY_CSV_PATH}: {e}")

    # 2. Load supplemental CSV if it exists
    if SUPPLEMENTAL_CSV_PATH.exists():
        try:
            df_supp = pd.read_csv(SUPPLEMENTAL_CSV_PATH)
            if "text" in df_supp.columns:
                for text_val in df_supp["text"].dropna():
                    first_sentence = str(text_val).split(". ")[0]
                    existing_titles.add(normalize_title(first_sentence))
            logger.info(f"Loaded existing title signatures from supplemental dataset.")
        except Exception as e:
            logger.warning(f"Error reading supplemental dataset: {e}")

    return existing_titles


def init_supplemental_csv():
    """Ensures output directory and supplemental CSV header exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not SUPPLEMENTAL_CSV_PATH.exists():
        with open(SUPPLEMENTAL_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "shelf_label"])
        logger.info(f"Initialized new supplemental CSV at {SUPPLEMENTAL_CSV_PATH}")


def append_scraped_row(text: str, shelf_label: str):
    """Appends a single scraped row to the supplemental CSV file."""
    with open(SUPPLEMENTAL_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([text, shelf_label])


async def setup_stealth_context(playwright_instance) -> BrowserContext:
    browser = await playwright_instance.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ]
    )
    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    )
    return context


async def fetch_detail_description(page, detail_url: str) -> str:
    """Navigates to detail page and extracts full book description."""
    try:
        response = await page.goto(detail_url, wait_until="domcontentloaded", timeout=25000)
        if not response or response.status != 200:
            logger.warning(f"Detail page HTTP {response.status if response else 'N/A'} for {detail_url}")
            return ""

        selectors_to_try = [
            'div[data-testid="description"]',
            'span[data-testid="description"]',
            '.Formatted',
            '#description span[id^="freeText"]',
            '.bookDescription'
        ]

        for sel in selectors_to_try:
            element = await page.query_selector(sel)
            if element:
                text = (await element.inner_text()).strip()
                if len(text) > 20:
                    # Clean up 'Show more' button text if captured
                    if text.endswith("Show more"):
                        text = text[:-9].strip()
                    return text

        return ""
    except Exception as e:
        logger.warning(f"Error fetching detail page {detail_url}: {e}")
        return ""


async def run_scraper():
    start_time = time.time()
    init_supplemental_csv()
    seen_titles = load_existing_titles()

    category_counts = {label: 0 for _, label, _ in TARGET_SHELVES}
    total_scraped = 0

    logger.info("Starting Stage 2 full scraping loop...")

    async with async_playwright() as p:
        context = await setup_stealth_context(p)
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        for slug, shelf_label, max_pages in TARGET_SHELVES:
            logger.info(f"=== Starting Shelf Category: [{shelf_label}] (Slug: {slug}) ===")

            for page_num in range(1, max_pages + 1):
                listing_url = f"{GOODREADS_BASE_URL}/shelf/show/{slug}?page={page_num}"
                logger.info(f"Fetching listing page {page_num}/{max_pages}: {listing_url}")

                # Politeness delay before page fetch
                await asyncio.sleep(random.uniform(3.0, 4.5))

                try:
                    res = await page.goto(listing_url, wait_until="domcontentloaded", timeout=30000)
                    if not res or res.status != 200:
                        logger.error(f"Listing page returned status {res.status if res else 'N/A'}. Stopping scraper.")
                        print(f"SCRAPER_BLOCKED_ERROR: Status {res.status if res else 'N/A'} on {listing_url}")
                        return

                    containers = await page.query_selector_all(".elementList")
                    if not containers:
                        containers = await page.query_selector_all(".bookBox")

                    logger.info(f"Page {page_num}: Found {len(containers)} book item containers.")

                    # Collect titles & detail URLs from listing page
                    items_to_fetch = []
                    for container in containers:
                        title_el = await container.query_selector("a.bookTitle") or await container.query_selector(".bookTitle")
                        if not title_el:
                            continue

                        raw_title = (await title_el.inner_text()).strip()
                        href = await title_el.get_attribute("href")
                        if not href:
                            continue

                        norm_title = normalize_title(raw_title)

                        # Deduplication check
                        if norm_title in seen_titles:
                            continue

                        full_detail_url = GOODREADS_BASE_URL + href if href.startswith("/") else href
                        items_to_fetch.append((raw_title, norm_title, full_detail_url))

                    logger.info(f"Page {page_num}: {len(items_to_fetch)} new candidate books after deduplication.")

                    # Tier 2: Visit individual detail pages
                    for raw_title, norm_title, detail_url in items_to_fetch:
                        # Politeness delay before detail fetch
                        await asyncio.sleep(random.uniform(3.0, 4.5))

                        description = await fetch_detail_description(page, detail_url)
                        if not description:
                            continue

                        # Construct unified text
                        combined_text = f"{raw_title}. {description}"

                        # Append row incrementally to CSV
                        append_scraped_row(combined_text, shelf_label)
                        seen_titles.add(norm_title)
                        category_counts[shelf_label] += 1
                        total_scraped += 1

                        logger.info(f"Saved [{shelf_label}] item #{category_counts[shelf_label]}: {raw_title[:45]}...")

                except Exception as e:
                    logger.error(f"Error scraping listing page {listing_url}: {e}")
                    print(f"SCRAPER_EXCEPTION: {e} on {listing_url}")
                    return

            logger.info(f"Finished category [{shelf_label}]. Total items collected so far: {total_scraped}")

        await context.close()

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("\n" + "=" * 60)
    print("           STAGE 2 SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total Runtime: {elapsed_minutes:.2f} minutes")
    print(f"Total New Items Collected: {total_scraped}")
    print("\nCOLLECTED ITEMS PER CATEGORY:")
    for label, count in category_counts.items():
        print(f"  - {label:<25}: {count} items")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_scraper())
