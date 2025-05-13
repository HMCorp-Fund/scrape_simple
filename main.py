#!/usr/bin/env python3
# filepath: main.py

import argparse
import time
import sys
import json
import os
from src import WebScraper
from src import SiteContent
from src.utils import download_and_cache_models

def scrape_site(url: str, depth: int, simplify: bool, use_existing_tor: bool, preloaded_models=None) -> SiteContent:
    """Main function to scrape a site and return SiteContent."""
    scraper = WebScraper(url, depth, simplify, use_existing_tor, preloaded_models)
    return scraper.start()

def save_to_json(site_content, output_file):
    """Save the site content to a JSON file."""
    data = {
        "html_pages": [page.to_dict() for page in site_content.HTMLPages],
        "text_pages": [page.to_dict() for page in site_content.TextPages],
        "media_content": [media.to_dict() for media in site_content.MediaContentList]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='Web scraper that uses Tor and can simplify text with LLMLingua.')
    parser.add_argument('url', help='The URL of the site to scrape')
    parser.add_argument('--depth', '-d', type=int, default=2, help='The depth level for crawling (default: 2)')
    parser.add_argument('--simplify', '-s', action='store_true', help='Simplify text with LLMLingua')
    parser.add_argument('--use-existing-tor', '-t', action='store_true', help='Use existing Tor instance if available')
    parser.add_argument('--no-preload', action='store_true', help='Skip preloading models (not recommended)')
    parser.add_argument('--force-cpu', action='store_true', help='Force using CPU for models even if GPU is available')
    parser.add_argument('--output', '-o', default='output.json', help='Output JSON file (default: output.json)')
    parser.add_argument('--history-file', default='.scrape_history', 
                        help='File to store visited URLs for this run (default: .scrape_history)')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight models for text simplification')
    
    args = parser.parse_args()
    
    # Preload all required models
    preloaded_models = None
    if not args.no_preload:
        try:
            print("Preloading all required models...")
            preloaded_models = download_and_cache_models(
                force_cpu=args.force_cpu,
                lightweight=args.lightweight
            )
        except Exception as e:
            print(f"Error preloading models: {e}")
            print("Continuing without preloaded models...")
    
    print(f"Starting to scrape {args.url} with depth {args.depth}")
    try:
        # Create WebScraper (don't load history file)
        scraper = WebScraper(args.url, args.depth, args.simplify, args.use_existing_tor, preloaded_models)
        
        # Start scraping
        site_content = scraper.start()
        
        # Save visited URLs to history file
        try:
            with open(args.history_file, 'w') as f:
                for url in scraper.visited_urls:
                    f.write(f"{url}\n")
            print(f"Saved {len(scraper.visited_urls)} visited URLs to {args.history_file}")
        except Exception as e:
            print(f"Error saving URL history: {e}")
        
        # Print summary of results
        print("\n--- Scraping Complete ---")
        print(f"HTML Pages: {len(site_content.HTMLPages)}")
        print(f"Text Pages: {len(site_content.TextPages)}")
        print(f"Media Content: {len(site_content.MediaContentList)}")
        
        # Save results to JSON
        save_to_json(site_content, args.output)
        
        return site_content
    except OSError as e:
        if "Tor executable not found" in str(e):
            print(f"\nERROR: {e}")
            print("\nPlease install Tor and make sure it's in your PATH.")
            print(" - On Ubuntu/Debian: sudo apt install tor")
            print(" - On macOS: brew install tor")
            sys.exit(1)
        else:
            raise

if __name__ == "__main__":
    main()