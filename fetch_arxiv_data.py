#!/usr/bin/env python3
"""
arXiv API Data Fetcher for School/Reference Shelf Category.
Fetches research paper metadata across 7 academic domains to create a representative
School/Reference training dataset.

Usage:
    python fetch_arxiv_data.py
"""

import os
import re
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd

# arXiv API Base Endpoint (HTTPS)
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Target Academic Domains and Search Queries (Single category per domain for API reliability)
DOMAINS = [
    {
        "domain": "Computer Science (AI)",
        "query": "cat:cs.AI",
        "target": 100,
    },
    {
        "domain": "Mathematics (General)",
        "query": "cat:math.GM",
        "target": 100,
    },
    {
        "domain": "Physics (General)",
        "query": "cat:physics.gen-ph",
        "target": 100,
    },
    {
        "domain": "Quantitative Biology",
        "query": "cat:q-bio.NC",
        "target": 100,
    },
    {
        "domain": "Economics & Econometrics",
        "query": "cat:econ.EM",
        "target": 100,
    },
    {
        "domain": "Statistics & Machine Learning",
        "query": "cat:stat.ML",
        "target": 100,
    },
    {
        "domain": "Systems & Engineering",
        "query": "cat:eess.SP",
        "target": 100,
    },
]

# XML Namespace for Atom Feed
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def clean_text(text: str) -> str:
    """Strips newlines and normalizes whitespace."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_arxiv_papers_for_domain(query: str, total_target: int) -> list:
    """Queries arXiv API for a specific domain using 50-record paginated chunks."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    papers = []
    chunk_size = 50
    pages = (total_target + chunk_size - 1) // chunk_size

    for page in range(pages):
        start = page * chunk_size
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(chunk_size, total_target - len(papers)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            entries = root.findall("atom:entry", NS)

            for entry in entries:
                raw_title = entry.findtext("atom:title", default="", namespaces=NS)
                raw_abstract = entry.findtext("atom:summary", default="", namespaces=NS)

                title = clean_text(raw_title)
                abstract = clean_text(raw_abstract)

                if title and abstract and len(abstract) > 50:
                    combined_text = f"{title} {abstract}"
                    papers.append(
                        {
                            "book_title": title,
                            "book_details": abstract,
                            "text": combined_text,
                            "shelf_label": "School/Reference",
                            "source": "arxiv",
                        }
                    )
        except Exception as e:
            print(f"Error fetching arXiv data for query [{query}] (page {page}): {e}", flush=True)

        if page < pages - 1:
            time.sleep(3.0)

    return papers


def main():
    print("=== FETCHING SCHOOL/REFERENCE DATA VIA ARXIV API ===\n", flush=True)

    os.makedirs("datasets/processed", exist_ok=True)
    all_papers = []

    for idx, domain_config in enumerate(DOMAINS, 1):
        domain = domain_config["domain"]
        query = domain_config["query"]
        target = domain_config["target"]

        print(f"[{idx}/{len(DOMAINS)}] Fetching {target} papers for domain: {domain}...", flush=True)
        papers = fetch_arxiv_papers_for_domain(query, target)
        print(f"    -> Retracted {len(papers)} valid paper metadata records.", flush=True)
        all_papers.extend(papers)

        # Enforce rate limit (3.0 seconds between requests)
        if idx < len(DOMAINS):
            print("    -> Waiting 3.0 seconds (arXiv rate limit policy compliance)...", flush=True)
            time.sleep(3.0)

    df = pd.DataFrame(all_papers)

    # Deduplicate titles
    initial_len = len(df)
    df.drop_duplicates(subset=["book_title"], inplace=True)
    final_len = len(df)
    if initial_len != final_len:
        print(f"\nDeduplicated {initial_len - final_len} duplicate titles.", flush=True)

    output_path = "datasets/processed/arxiv_school_reference.csv"
    df.to_csv(output_path, index=False)

    print(f"\n============================================================", flush=True)
    print(f"SUCCESS: Saved {len(df)} total arXiv records to {output_path}", flush=True)
    print(f"============================================================", flush=True)


if __name__ == "__main__":
    main()

