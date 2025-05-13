#!/usr/bin/env python3
# filepath: main.py

import argparse
import time
import sys
from src import WebScraper
from src import SiteContent
from src.utils import download_and_cache_models

def scrape_site(url: str, depth: int, simplify: bool, use_existing_tor: bool, preloaded_models=None) -> SiteContent:
    """Main function to scrape a site and return SiteContent."""
    scraper = WebScraper(url, depth, simplify, use_existing_tor, preloaded_models)
    return scraper.start()

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='Web scraper that uses Tor and can simplify text with LLMLingua.')
    parser.add_argument('url', help='The URL of the site to scrape')
    parser.add_argument('--depth', '-d', type=int, default=2, help='The depth level for crawling (default: 2)')
    parser.add_argument('--simplify', '-s', action='store_true', help='Simplify text with LLMLingua')
    parser.add_argument('--use-existing-tor', '-t', action='store_true', help='Use existing Tor instance if available')
    parser.add_argument('--no-preload', action='store_true', help='Skip preloading models (not recommended)')
    parser.add_argument('--force-cpu', action='store_true', help='Force using CPU for models even if GPU is available')
    
    args = parser.parse_args()
    
    # Preload all required models
    preloaded_models = None
    if not args.no_preload:
        try:
            print("Preloading all required models...")
            preloaded_models = download_and_cache_models(force_cpu=args.force_cpu)
        except Exception as e:
            print(f"Error preloading models: {e}")
            print("Continuing without preloaded models...")
    
    print(f"Starting to scrape {args.url} with depth {args.depth}")
    try:
        site_content = scrape_site(args.url, args.depth, args.simplify, args.use_existing_tor, preloaded_models)
        
        # Print summary of results
        print("\n--- Scraping Complete ---")
        print(f"HTML Pages: {len(site_content.HTMLPages)}")
        print(f"Text Pages: {len(site_content.TextPages)}")
        print(f"Media Content: {len(site_content.MediaContentList)}")
        
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