from typing import List

from . import HTMLPage
from . import MediaContent
from . import TextPage


class SiteContent:
    """Container for all scraped content from a site."""
    def __init__(self):
        self.HTMLPages = []
        self.TextPages = []
        self.MediaContentList = []
    
    def add_html_page(self, page):
        self.HTMLPages.append(page)
        print(f"Added HTML page: {page.url} (total: {len(self.HTMLPages)})")
        
    def add_text_page(self, page):
        self.TextPages.append(page)
        print(f"Added text page: {page.url} (total: {len(self.TextPages)})")
        
    def add_media(self, media):
        self.MediaContentList.append(media)
        print(f"Added media content: {media.url} (total: {len(self.MediaContentList)})")
        
    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "html_pages": [page.to_dict() for page in self.HTMLPages],
            "text_pages": [page.to_dict() for page in self.TextPages],
            "media_content": [media.to_dict() for media in self.MediaContentList]
        }