"""
scrape_rokomari.py
------------------
Bangla book scraper for Rokomari.com using Playwright.

Scrapes metadata (title + plot synopsis) across 17 target shelf categories:
  - Targeted sub-category URL routing (avoiding root generic novels blob)
  - Single-pass Category 74 split (General terms + expanded franchise markers)
  - UTF-8 Bangla text handling
  - Stealth browser automation & politeness delays (1.5 - 2.5s)
  - Progressive CSV checkpointing to `datasets/processed/rokomari_bangla.csv`
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

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
OUTPUT_CSV_PATH = PROCESSED_DIR / "rokomari_bangla.csv"

# Category Configurations (Target Shelf -> (Category URL, Target Count))
CATEGORY_TARGETS = [
    ("Science Fiction", "https://www.rokomari.com/book/category/15/science-fiction", 50),
    ("Horror", "https://www.rokomari.com/book/category/16/horror", 50),
    ("Mystery", "https://www.rokomari.com/book/category/17/detective", 50),
    ("Romance", "https://www.rokomari.com/book/category/2/romantic-novels", 50),
    ("Fantasy", "https://www.rokomari.com/book/category/77/fantasy", 50),
    ("History", "https://www.rokomari.com/book/category/9/history", 50),
    ("Historical Fiction", "https://www.rokomari.com/book/category/79/historical-novel", 50),
    ("Biography & Memoir", "https://www.rokomari.com/book/category/12/biography", 50),
    ("Poetry", "https://www.rokomari.com/book/category/3/poetry", 50),
    ("Philosophy", "https://www.rokomari.com/book/category/11/philosophy", 40),
    ("Humor", "https://www.rokomari.com/book/category/34/humor", 40),
    ("Religion & Spirituality", "https://www.rokomari.com/book/category/6/islamic-books", 50),
    ("Self-Help & Personal Development", "https://www.rokomari.com/book/category/13/self-help-and-personal-development", 50),
    ("Classic Literature", "https://www.rokomari.com/book/category/4/classics", 50),
    ("Miscellaneous", "https://www.rokomari.com/book/category/14/general-knowledge", 50),
    # Category 74: Single-pass fetch for sequential art (Splits into Graphic Novels & Anime & Manga)
    ("Sequential Art Category 74", "https://www.rokomari.com/book/category/74/comics-and-graphic-novels", 90),
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


def is_manga_entry(title: str, description: str) -> bool:
    """Evaluates title and description for general manga/anime terms and franchise markers."""
    combined = f"{title} {description}".lower()

    # Check general medium terms
    for term in MANGA_GENERAL_TERMS:
        if term in combined:
            return True

    # Check franchise markers
    for marker in MANGA_FRANCHISE_MARKERS:
        if marker in combined:
            return True

    return False


def clean_text(raw_text: str) -> str:
    """Cleans scraped Bangla text by collapsing whitespace and stripping promotional headers."""
    if not raw_text:
        return ""
    text = str(raw_text)
    text = re.sub(r"বইটির বিস্তারিত দেখুন", "", text, flags=re.IGNORECASE)
    text = re.sub(r"বইয়ের সংক্ষিপ্ত কথা", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_rokomari_pipeline():
    start_time = time.time()
    logger.info("Starting Rokomari Bangla book scraping pipeline...")

    all_records: List[Dict[str, str]] = []
    seen_titles: Set[str] = set()

    cat_74_manga_count = 0
    cat_74_graphic_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for shelf_label, cat_url, target_count in CATEGORY_TARGETS:
            logger.info(f"\n--- Scraping Category: '{shelf_label}' (Target: {target_count}) ---")
            category_records = 0
            current_page = 1

            while category_records < target_count and current_page <= 5:
                page_url = f"{cat_url}?page={current_page}" if current_page > 1 else cat_url
                logger.info(f"Navigating to listing page {current_page}: {page_url}")

                try:
                    page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(2000)
                    soup = BeautifulSoup(page.content(), "html.parser")

                    # Extract book links from listing page
                    book_links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if "/book/" in href and not any(x in href for x in ["/category/", "/author/", "/publisher/", "/categories"]):
                            full_href = href if href.startswith("http") else f"https://www.rokomari.com{href}"
                            book_links.append(full_href)

                    # Deduplicate links on current page
                    book_links = list(dict.fromkeys(book_links))
                    logger.info(f"Found {len(book_links)} book links on page {current_page}.")

                    if not book_links:
                        logger.info(f"No more books found for '{shelf_label}' at page {current_page}.")
                        break

                    for book_url in book_links:
                        if category_records >= target_count:
                            break

                        try:
                            page.goto(book_url, wait_until="domcontentloaded", timeout=20000)
                            page.wait_for_timeout(1500)

                            detail_soup = BeautifulSoup(page.content(), "html.parser")

                            # Extract Title
                            h1_tag = detail_soup.find("h1") or detail_soup.find("title")
                            title_text = h1_tag.get_text(strip=True) if h1_tag else ""
                            # Remove website suffix
                            title_text = re.sub(r"\| Rokomari\.com", "", title_text).strip()

                            norm_title = re.sub(r"[^a-z0-9\u0980-\u09ff]", "", title_text.lower())
                            if not norm_title or norm_title in seen_titles:
                                continue

                            # Extract Synopsis
                            summary_div = detail_soup.find("div", id="summary") or detail_soup.find("section", id="summary") or detail_soup.find("div", class_=lambda c: c and "summary" in c)
                            if summary_div:
                                synopsis_raw = summary_div.get_text(strip=True)
                            else:
                                synopsis_raw = detail_soup.get_text(strip=True)

                            cleaned_syn = clean_text(synopsis_raw)

                            # Filter out very short descriptions (< 30 chars)
                            if len(cleaned_syn) < 30:
                                continue

                            # Build full combined text
                            full_text = f"{title_text}. {cleaned_syn}"

                            # Determine Label
                            if shelf_label == "Sequential Art Category 74":
                                if is_manga_entry(title_text, cleaned_syn):
                                    assigned_label = "Anime & Manga"
                                    cat_74_manga_count += 1
                                else:
                                    assigned_label = "Graphic Novels"
                                    cat_74_graphic_count += 1
                            else:
                                assigned_label = shelf_label

                            seen_titles.add(norm_title)
                            all_records.append({
                                "text": full_text,
                                "shelf_label": assigned_label
                            })
                            category_records += 1

                            logger.info(f"[{assigned_label}] ({category_records}/{target_count}): {title_text[:40]}...")

                        except Exception as e:
                            logger.warning(f"Error fetching detail page {book_url}: {e}")

                        time.sleep(1.5)

                except Exception as e:
                    logger.error(f"Error fetching listing page {page_url}: {e}")

                current_page += 1

        browser.close()

    # Save to CSV
    df_output = pd.DataFrame(all_records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    elapsed_seconds = round(time.time() - start_time, 2)

    print("\n" + "=" * 80)
    print("                 ROKOMARI BANGLA SCRAPING SUMMARY")
    print("=" * 80)
    print(f"Total Bangla Rows Achieved:          {len(df_output)}")
    print(f"Category 74 Manga Count:            {cat_74_manga_count}")
    print(f"Category 74 Graphic Novels Count:   {cat_74_graphic_count}")
    print(f"Total Execution Time:                {elapsed_seconds} seconds")
    print(f"Saved Output CSV:                    {OUTPUT_CSV_PATH}\n")

    print("BANGLA DATASET DISTRIBUTION (Label: Count):")
    counts = df_output["shelf_label"].value_counts()
    for label, count in counts.items():
        print(f"  - {label:<35}: {count}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    scrape_rokomari_pipeline()
