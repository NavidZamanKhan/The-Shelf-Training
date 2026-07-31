"""
explore_wafilife.py
-------------------
Exploratory script to inspect Wafilife.com category structure, specifically checking for:
  1. Classic Literature (চিরায়ত সাহিত্য / ক্লাসিক)
  2. Anime & Manga (মঙ্গা / লাইট নভেল / অ্যানিমে)
"""

import json
import logging
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WAFILIFE_BASE = "https://www.wafilife.com"
CATEGORIES_URL = "https://www.wafilife.com/cat/books/"


def explore_wafilife():
    logger.info(f"Navigating to Wafilife categories page: {CATEGORIES_URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            page.goto(CATEGORIES_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            soup = BeautifulSoup(page.content(), "html.parser")

            category_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)
                if "/cat/books/" in href or "/cat/" in href:
                    category_links.append((text, href))

            logger.info(f"Found {len(category_links)} category links on Wafilife.")

            # Search specifically for manga, anime, light novel, classic literature keywords
            target_keywords = ["চিরায়ত", "ক্লাসিক", "মঙ্গা", "মানহোয়া", "কমিকস", "নভেল", "উপন্যাস", "ক্লাসিক্যাল"]
            matched_cats = []

            for text, href in category_links:
                if any(k in text for k in target_keywords) or any(k in href for k in ["classic", "manga", "anime", "comics", "novel"]):
                    matched_cats.append((text, href))

            print("\n" + "=" * 80)
            print("                WAFILIFE CATEGORY AUDIT RESULTS")
            print("=" * 80)
            print(f"Matched Category Candidates ({len(matched_cats)}):")
            for t, h in matched_cats:
                print(f"  - {t:<35} -> {h}")

            # Test sample category pages to check book counts
            test_targets = [
                ("চিরায়ত সাহিত্য / অনুবাদ (Classics)", "https://www.wafilife.com/cat/books/translated-books/"),
                ("উপন্যাস (Novels)", "https://www.wafilife.com/cat/books/novel/"),
                ("কমিকস (Comics)", "https://www.wafilife.com/cat/books/comics/"),
            ]

            print("\nTargeted Category Page Inspections:")
            for name, url in test_targets:
                try:
                    logger.info(f"Navigating to: {url}")
                    page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_timeout(2000)
                    cat_soup = BeautifulSoup(page.content(), "html.parser")

                    # Look for book items
                    book_items = cat_soup.find_all("div", class_=lambda c: c and "product" in str(c)) or cat_soup.find_all("li", class_=lambda c: c and "product" in str(c))
                    
                    # Search specifically for Manga titles inside comics/translated categories
                    manga_matches = []
                    for item in book_items:
                        item_text = item.get_text(strip=True)
                        if any(m in item_text.lower() for m in ["manga", "মাঙ্গা", "নারুতো", "ডেথ নোট", "ডোরেমন", "সোলো লেভেলিং", "অনুবাদ"]):
                            manga_matches.append(item_text[:50])

                    print(f"  - {name}: {len(book_items)} books on Page 1 | Manga titles found: {len(manga_matches)}")
                except Exception as e:
                    logger.warning(f"Error checking {url}: {e}")

            print("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Error navigating Wafilife: {e}")
        finally:
            browser.close()


if __name__ == "__main__":
    explore_wafilife()
