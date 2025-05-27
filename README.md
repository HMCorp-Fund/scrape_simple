# Scrape Simple

A web scraper that uses Tor for anonymity and supports text and media extraction.

## Features

- Tor integration for anonymous web scraping
- Extract text content from web pages
- Extract media files (images, videos) above a specified size
- Optional Russian text simplification using Natasha
- Optional AI-based image description using BLIP

## Installation

```bash
pip install scrape-simple
```

### Optional Dependencies

For Russian text simplification:
```bash
pip install scrape-simple[russian]
```

For AI image descriptions:
```bash
pip install scrape-simple[ai]
```

For all features:
```bash
pip install scrape-simple[russian,ai]
```

## Usage

### Command Line
```bash
# Basic usage
scrape-simple https://example.com

# Advanced usage
scrape-simple https://example.com --depth 3 --min-media-size 20480 --simplify-ru --ai-describe-media
```

### Python API
```python
from scrape_simple import WebScraper, SiteContent

# Create scraper
scraper = WebScraper(
    root_url="https://example.com",
    max_depth=2,
    use_existing_tor=True,
    min_media_size=10240,  # 10KB minimum for media files
    simplify_ru=False,
    ai_describe_media=False
)

# Start scraping
site_content = scraper.start()

# Access results
for page in site_content.TextPages:
    print(f"Page: {page.url}, Content length: {len(page.content)}")

for media in site_content.MediaContentList:
    print(f"Media: {media.url}, Type: {media.media_type}, Description: {media.description}")

# Create scraper with media extraction disabled
scraper = WebScraper(
    root_url="https://example.com",
    max_depth=2,
    use_existing_tor=True,
    skip_media=True  # Disable media extraction
)
```

## Requirements
- Python 3.6+
- Tor (must be installed separately)

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `url` | The URL of the site to scrape |
| `--depth`, `-d` | The depth level for crawling (default: 2) |
| `--use-existing-tor`, `-t` | Use existing Tor instance if available |
| `--output`, `-o` | Output JSON file (default: output.json) |
| `--history-file` | File to store visited URLs for this run (default: .scrape_history) |
| `--simplify-ru` | Simplify Russian text using Natasha |
| `--min-media-size` | Minimum file size for media in bytes (default: 100KB) |
| `--ai-describe-media` | Use AI to generate descriptions for media files |
| `--skip-media` | Disable media extraction completely |
| `--max-retries` | Maximum number of retries for failed requests (default: 3) |

## Anti-Bot Protection Handling

Scrape Simple includes features to bypass common anti-bot protections:

- Browser-like request headers
- Random delays between requests
- Tor IP rotation for 403/429 errors
- Configurable retry mechanism