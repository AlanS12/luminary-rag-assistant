import requests
from bs4 import BeautifulSoup
import time
import os
import re

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL = "https://gds.blog.gov.uk"
OUTPUT_DIR = "luminary-rag-assistant/data/raw/gds_scraped"
MAX_POSTS = 100
DELAY_SECONDS = 1.5
# ───────────────────────────────────────────────────────────────────────────────


def is_post_url(href):
    """
    GDS blog post URLs follow the pattern:
    https://gds.blog.gov.uk/YYYY/MM/DD/post-slug/
    We identify them by checking for a 4-digit year after the domain.
    """
    if not href:
        return False
    if "#" in href:           # exclude comment anchors
        return False
    if "/author/" in href:    # exclude author pages
        return False
    if "/category/" in href:  # exclude category pages
        return False
    # Must be a GDS blog URL with a year in it (post URLs)
    return href.startswith("https://gds.blog.gov.uk/20")


def get_post_urls(max_posts=MAX_POSTS):
    """
    Crawl paginated GDS blog listing pages and collect unique post URLs.
    """
    post_urls = []
    seen = set()
    page = 1

    print(f"Collecting post URLs (target: {max_posts} posts)...")

    while len(post_urls) < max_posts:
        url = BASE_URL if page == 1 else f"{BASE_URL}/page/{page}/"
        print(f"  Scanning listing page {page}: {url}")

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 404:
                print(f"  Reached end of blog at page {page}.")
                break

            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links on the page and filter to post URLs only
            all_links = soup.find_all("a", href=True)
            page_post_urls = []

            for link in all_links:
                href = link["href"]
                if is_post_url(href) and href not in seen:
                    seen.add(href)
                    page_post_urls.append(href)

            if not page_post_urls:
                print(f"  No new post URLs found on page {page} — stopping.")
                break

            post_urls.extend(page_post_urls)
            print(f"  Found {len(post_urls)} unique post URLs so far.")
            page += 1
            time.sleep(DELAY_SECONDS)

        except requests.RequestException as e:
            print(f"  Error fetching listing page {page}: {e}")
            break

    return post_urls[:max_posts]


def scrape_post(url):
    """
    Fetch a single GDS blog post and extract title, date, author, content.
    Returns None if content cannot be extracted.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # ── Title ──────────────────────────────────────────────────────────────
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        # ── Date ───────────────────────────────────────────────────────────────
        date_tag = soup.find("time")
        date = date_tag.get("datetime", "")[:10] if date_tag else "unknown-date"

        # ── Author ─────────────────────────────────────────────────────────────
        author_tag = soup.find("a", rel="author")
        author = author_tag.get_text(strip=True) if author_tag else "GDS Team"

        # ── Main content ───────────────────────────────────────────────────────
        # Try several possible content container class names
        content_tag = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="post-content") or
            soup.find("div", class_="govuk-grid-column-two-thirds") or
            soup.find("article") or
            soup.find("main")
        )

        if not content_tag:
            return None

        # Extract text from paragraphs and headings
        content_lines = []
        for tag in content_tag.find_all(["p", "h2", "h3", "h4", "li"]):
            text = tag.get_text(strip=True)
            if not text:
                continue
            if tag.name in ["h2", "h3", "h4"]:
                level = int(tag.name[1])
                content_lines.append(f"\n{'#' * level} {text}\n")
            else:
                content_lines.append(text)

        content = "\n\n".join(content_lines)

        # Skip posts with very little content
        if len(content) < 200:
            return None

        return {
            "title": title,
            "date": date,
            "author": author,
            "url": url,
            "content": content
        }

    except requests.RequestException as e:
        print(f"    Request error: {e}")
        return None
    except Exception as e:
        print(f"    Parse error: {e}")
        return None


def save_post_as_markdown(post, filename):
    """Save a scraped post as a markdown file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    markdown = f"""# {post['title']}

**Source:** {post['url']}  
**Date:** {post['date']}  
**Author:** {post['author']}  

---

{post['content']}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)


def clean_filename(title, index):
    """Convert a post title to a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", title.lower())
    safe = re.sub(r"[\s-]+", "_", safe).strip("_")[:60]
    return f"{index:03d}_{safe}.md"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Luminary RAG Assistant — GDS Blog Scraper")
    print("=" * 60)

    # Step 1: Collect URLs
    post_urls = get_post_urls(MAX_POSTS)
    print(f"\nCollected {len(post_urls)} post URLs.\n")

    if not post_urls:
        print("No URLs found. Exiting.")
        return

    # Step 2: Scrape each post
    success_count = 0
    fail_count = 0

    for i, url in enumerate(post_urls, start=1):
        print(f"Scraping post {i}/{len(post_urls)}: {url}")

        post = scrape_post(url)

        if post:
            filename = clean_filename(post["title"], i)
            save_post_as_markdown(post, filename)
            print(f"  ✓ Saved: {filename}")
            success_count += 1
        else:
            print(f"  ✗ Skipped (no content extracted)")
            fail_count += 1

        time.sleep(DELAY_SECONDS)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print(f"Scraping complete.")
    print(f"  Successfully saved : {success_count} posts")
    print(f"  Skipped/failed     : {fail_count} posts")
    print(f"  Output folder      : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()