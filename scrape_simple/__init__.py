"""
Scrape Simple - A web scraper that uses Tor for anonymity

This package provides tools for scraping websites with Tor anonymity,
extracting text and media content, and optionally applying AI processing.
"""

from .src import WebScraper, SiteContent, HTMLPage, TextPage, MediaContent, TorManager

__version__ = "0.1.0"
__all__ = ["WebScraper", "SiteContent", "HTMLPage", "TextPage", "MediaContent", "TorManager"]